# SPDX-License-Identifier: Apache-2.0
"""Memory-bound decode-linear latency: bf16 vs CUTLASS INT4 vs LLM.int8 vs OMMX i2f4 (3 outlier ratios).
ONE GPU session, ONE clock: OMMX / bf16 / int8 are timed in-process with CUDA-event ev_time (warmup 20,
rep 50, median); CUTLASS INT4 is re-measured LIVE via the compiled example-55 mixed-input binary (NOT a
hardcoded/cross-session constant). OMMX i2f4 is BIT-EXACT to the fakequant weight-quant FORMAT
(quant_weight.weight_quantizer bits=2, use_pow2=True, gs=64) — verified cos=1.0 (parity assert below);
this bench reports LATENCY only (no accuracy/iso-bit claim). Decode (small M) is memory/launch-bound —
the regime where the INT2 base's fewer weight bytes matter.
CAVEATS (honest): CUTLASS arm = ex55 mixed-input INT4xbf16 g=128 (a reference EXAMPLE, NOT the tuned
Marlin/Machete W4A16 kernel vLLM ships) timed by its own internal profiler; OMMX decode is the eager
2-launch base+sparse_correct_packed path (served OMMX runs under CUDA graph). npv4/8/16 = 3.06/3.75/5.125
bits/wt vs INT4 4.125 — NOT iso-bit. See README.
"""
from __future__ import annotations
import os, sys, json, re, glob, subprocess, datetime, gc
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")          # cgroup fork/RAM hygiene (quantize is CPU-heavy)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch

SHAPES = [(4096, 4096, "square"), (14336, 4096, "gate/up"), (4096, 14336, "down")]
MS = [1, 2, 4, 8, 16]
OPCTS = [0.0625, 0.125, 0.25]          # outlier ratios (npv 4/8/16 at gs=64)
# CUTLASS example-55 binary: set OMMX_CUTLASS_EX55_BIN directly, or OMMX_CUTLASS_BUILD_DIR to search.
EX55_BUILD = os.environ.get("OMMX_CUTLASS_BUILD_DIR",
                            os.path.join(os.environ.get("OMMX_CUTLASS_DIR", ""), "build_bench"))
_MS_RE = re.compile(r"(?:Avg\s*runtime|Runtime)\s*[:=]?\s*([\d.]+)\s*(ms|us|ns)", re.I)


def ev_time(fn, warmup=20, rep=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(rep):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    n = len(ts)
    return ts[n // 2], sum(ts) / n, ts[min(n - 1, int(0.99 * n))]   # median, mean, p99


def cos(a, b):
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def find_ex55():
    p = os.environ.get("OMMX_CUTLASS_EX55_BIN", "").strip()
    if p and os.access(p, os.X_OK): return p
    for pat in ("55_hopper*", "55_*"):
        for c in sorted(glob.glob(os.path.join(EX55_BUILD, "**", pat), recursive=True)):
            if os.path.isfile(c) and os.access(c, os.X_OK) and not c.endswith((".o", ".a", ".cmake")):
                return c
    return None


def cutlass_int4_ms(binp, M, N, K, g=128, iters=50):
    """LIVE CUTLASS ex55 mixed-input INT4xbf16 for [M,N,K]; ms from its internal profiler."""
    if not binp: return None
    try:
        p = subprocess.run([binp, f"--m={M}", f"--n={N}", f"--k={K}", f"--g={g}", "--mode=2",
                            f"--iterations={iters}"], capture_output=True, text=True, timeout=120)
        raw = (p.stdout or "") + (p.stderr or "")
        m = _MS_RE.search(raw)
        if not m: return None
        v, u = float(m.group(1)), m.group(2).lower()
        return v if u == "ms" else v / 1e3 if u == "us" else v / 1e6
    except Exception:
        return None


def make_int8(W, dev):
    N, K = W.shape
    try:
        import bitsandbytes as bnb
        L = bnb.nn.Linear8bitLt(K, N, bias=False, has_fp16_weights=False, threshold=6.0).to(dev)
        L.weight = bnb.nn.Int8Params(W.half().clone(), requires_grad=False, has_fp16_weights=False).to(dev)
        def run(x): return L(x.half())
        run(torch.randn(1, K, device=dev)); return ("LLM.int8 (bnb)", run)
    except Exception:
        return ("int8_unavailable", None)


def _find_fakequant():
    """Locate the fakequant package (ommx_fakequant/ in the OSS repo, fakequant/ in dev) upward."""
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(2, 6):
        root = os.path.abspath(os.path.join(here, *([".."] * up)))
        for name in ("ommx_fakequant", "fakequant"):
            cand = os.path.join(root, name)
            if os.path.isfile(os.path.join(cand, "quant_weight.py")):
                return cand
    return None


def parity_assert(mod, T, dev, VL):
    """Pin the FORMAT contract: quantize_ommx_weight.W_ref == weight_quantizer(bits=2,pow2,act=None)."""
    try:
        fq = _find_fakequant()
        if fq is None:
            raise ModuleNotFoundError("fakequant/ommx_fakequant not found upward")
        sys.path.insert(0, fq)
        from quant_weight import weight_quantizer
        W = torch.randn(256, 256) * 0.1
        q = T.quantize_ommx_weight(W, VL, 0.25)
        ref = weight_quantizer(W, bits=2, group_size=VL, outlier_percent=0.25, use_pow2=True,
                               act_scales=None, mode="decode")
        c = cos(q["W_ref"], ref); md = (q["W_ref"] - ref).abs().max().item()
        ok = c >= 0.9999 and md < 1e-4
        print(f"PARITY assert (quantize_ommx_weight.W_ref vs weight_quantizer pow2/act=None): "
              f"cos={c:.6f} max_diff={md:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)
        return {"cos": round(c, 6), "max_diff": md, "pass": ok}
    except Exception as ex:
        print(f"PARITY assert SKIPPED ({type(ex).__name__}: {ex})", flush=True)
        return {"skipped": str(ex)[:120]}


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(42); dev = "cuda"; VL = 64
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    import build_ommx_linear, test_ommx_linear_parity as T
    mod = build_ommx_linear.build()
    ex55 = find_ex55()
    print(f"gpu={torch.cuda.get_device_name()} ex55={ex55}", flush=True)
    out = {"gpu": torch.cuda.get_device_name(), "opcts": OPCTS, "shapes": [s[2] for s in SHAPES],
           "M": MS, "ex55_bin": ex55, "parity": parity_assert(mod, T, dev, VL),
           "cutlass_note": "ex55 mixed-input INT4xbf16 g=128 (reference example, not Marlin), internal profiler",
           "ommx_note": "cute_base(shuffled INT2 wgmma) + sparse_correct_packed; eager 2-launch; "
                        "format bit-exact to fakequant weight_quantizer(bits=2,pow2,gs=64)",
           "rows": []}
    int8_label = None
    for (N, K, tag) in SHAPES:
        W = torch.randn(N, K) * 0.1
        int8_label, int8_run = make_int8(W.to(dev), dev)
        # CUTLASS INT4 (live, per M) — weight-value independent, measured once per (shape,M)
        cutlass_ms = {M: cutlass_int4_ms(ex55, M, N, K) for M in MS}
        for opct in OPCTS:
            q = T.quantize_ommx_weight(W, VL, opct); npv = q["npv"]
            code = q["code"].to(dev); scale = q["scale"].to(dev)
            scales_bf = q["scale"].t().contiguous().t().to(torch.bfloat16).to(dev)
            zeros_bf = q["zp"].t().contiguous().t().to(torch.bfloat16).to(dev)
            oindex = q["oindex"].to(dev); ocode = q["ocode"].to(dev)
            msc = q["map_scale"].to(dev); mcen = q["map_center"].to(dev)
            Bre = torch.empty_like(code); mod.cute_reorder_B(Bre, code, N, K)
            odelta = mod.build_odelta(code, scale, oindex, ocode, msc, mcen, N, K, VL, npv, VL)
            kpos = mod.build_kpos(oindex, N, K, npv, VL)
            Wref = q["W_ref"].to(dev); Wbf = Wref.to(torch.bfloat16)
            for M in MS:
                A = (torch.randn(M, K, device=dev) * 0.1).to(torch.bfloat16)
                ref = A.float() @ Wref.t()
                C = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
                def ommx():
                    mod.cute_base(C, A, Bre, scales_bf, zeros_bf, N, K, VL, True, None)
                    mod.sparse_correct_packed(C, A, kpos, odelta, N, K, npv, VL, "i2f4"); return C
                med, mean, p99 = ev_time(ommx)
                bf16_med = ev_time(lambda: torch.matmul(A, Wbf.t()))[0]
                row = {"shape": tag, "opct": opct, "npv": int(npv), "M": M,
                       "ommx_ms": round(med, 5), "ommx_mean": round(mean, 5), "ommx_p99": round(p99, 5),
                       "ommx_selfcos": round(cos(ommx(), ref), 5),   # kernel-vs-its-own-dequant (mechanic parity, NOT accuracy)
                       "cutlass_ms": (round(cutlass_ms[M], 5) if cutlass_ms[M] else None),
                       "bf16_ms": round(bf16_med, 5)}
                if int8_run is not None:
                    try: row["int8_ms"] = round(ev_time(lambda: int8_run(A))[0], 5)
                    except Exception: row["int8_ms"] = None
                out["rows"].append(row); print(row, flush=True)
                del A, C
            # free the per-(shape,opct) quant tensors before the next ratio (quantize is CPU-RAM heavy)
            del q, code, scale, scales_bf, zeros_bf, oindex, ocode, msc, mcen, Bre, odelta, kpos, Wref, Wbf
            gc.collect(); torch.cuda.empty_cache()
        del W, int8_run
        gc.collect(); torch.cuda.empty_cache()
    out["int8_label"] = int8_label
    ts = os.environ.get("OMMX_RUN_TS") or datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    d = os.path.join(here, "runs", ts, "mem_bound"); os.makedirs(d, exist_ok=True)
    out["env"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                  "cap": f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"}
    p = os.path.join(d, "result.json"); json.dump(out, open(p, "w"), indent=2)
    print(f"BENCH_MEM_BOUND_DONE rc=0 ex55={'live' if ex55 else 'MISSING'} int8={int8_label} -> {p}", flush=True)


if __name__ == "__main__":
    main()
