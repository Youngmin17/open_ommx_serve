# SPDX-License-Identifier: Apache-2.0
"""OMMX linear — fakequant-consistency PARITY gate (law #12 / #4 correctness test).

Authored on a no-GPU host (the kernel math is GPU-deferred); this is the TURNKEY gate the
cluster wave (cap=2) runs FIRST, BEFORE any latency/bench. It:
  1. synthesizes W[N,K] and quantizes EXACTLY per OMMX_LINEAR_CONSISTENCY.md (affine INT2
     base, E8M0 pow2 scale, group-min zp, top-K FP4 weight-space outliers, relidx7 sidecar) —
     this packer IS the contract (mirror of attention/pack.py + codec.py);
  2. computes the fp32 reference dequant W_ref the kernels MUST reproduce;
  3. builds + runs the consolidated kernels and asserts cos>=0.999, max_diff<tol, y_finite,
     firing>0 for: base decode (M=1/8), base prefill (M=32), base+outlier decode (M=1),
     and base+outlier at M>1 through sparse_correct's OTHER (At = A^T, bf16 C) call
     convention — decode M=8 and prefill M=32. The M>1 cases were added after a
     PASSING 5/5 run of this gate failed to catch a caller that sent the M==1 argument
     list at every M: coverage of one branch of `if (M == 1) ... else ...` is not
     coverage of the ABI.

Run (ommx conda env, sm_80 or sm_90a Hopper):
    PYTHONNOUSERSITE=1 python test_ommx_linear_parity.py

A FAIL prints cos first (law #4: cos>=0.999 => kernel correct, suspect tol/accum infra).
These are CORRECTNESS gates only; perf is bench_ommx_linear (separate).
"""
from __future__ import annotations

import os
import sys

import torch

FP4_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]   # idx=(sign<<3)|(exp<<1)|mant


def _fp4_decode(nib: int) -> float:
    return FP4_E2M1[nib & 0xF]


def _fp4_encode_linear(x: float) -> int:
    best, bi = 1e30, 0
    for i, v in enumerate(FP4_E2M1):
        d = abs(x - v)
        if d < best:                       # strict '<' => midpoint-to-lower (codec convention)
            best, bi = d, i
    return bi


def quantize_ommx_weight(W: torch.Tensor, group_size: int, outlier_pct: float):
    """W[N,K] fp32 -> dict(packed code, E8M0 scale_exp, fp32 scale/zp, relidx7 sidecar +
    fp4 nibble + map_scale/center, base-only ref W_base, full ref W_ref). Contract §1."""
    N, K = W.shape
    assert K % group_size == 0 and group_size % 4 == 0
    G = K // group_size
    npv = max(1, int(group_size * outlier_pct)) if outlier_pct > 0 else 0
    Wg = W.view(N, G, group_size)
    omask = torch.zeros_like(Wg, dtype=torch.bool)
    topk = None
    if npv:
        topk = Wg.abs().topk(npv, dim=-1).indices                       # [N,G,npv]
        omask.scatter_(-1, topk, True)

    big = torch.where(omask, torch.full_like(Wg, 1e9), Wg)
    sml = torch.where(omask, torch.full_like(Wg, -1e9), Wg)
    mn = big.min(dim=-1).values; mx = sml.max(dim=-1).values
    mn = torch.where(mn > 1e8, torch.zeros_like(mn), mn)
    mx = torch.where(mx < -1e8, torch.zeros_like(mx), mx)
    scale = (mx - mn).clamp_min(1e-8) / 3.0
    scale = torch.pow(2.0, torch.round(torch.log2(scale.clamp_min(1e-12))))   # E8M0 pow2
    zp = mn
    code = torch.round((Wg - zp.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp(0, 3)  # RNE
    W_base = code * scale.unsqueeze(-1) + zp.unsqueeze(-1)                # base everywhere

    W_ref = W_base.clone()
    nib_g = torch.zeros(N, G, group_size, dtype=torch.uint8)
    o_center = torch.zeros(N, G); map_scale = torch.ones(N, G)
    if npv:
        # IDX-SPACE FP4 map (matches the kernel relidx7_slot_delta + the vLLM weight loader:
        #   delta = (fp4/ms + mc - code)*scale  ->  dq_o = (fp4/ms + mc)*scale + zp).
        # ms = mapping_scale = 12/idx_range, mc = mapping_center over the OUTLIER INDICES
        # idx = (w - zp)/scale. (NOT the KV weight-space form fp4/ms+o_center — that path
        # uses a different exporter; the WEIGHT bundle is idx-space. See CONSISTENCY R2.)
        idx = (Wg - zp.unsqueeze(-1)) / scale.unsqueeze(-1)              # [N,G,gs] index space
        i_big = torch.where(omask, idx, torch.full_like(idx, 1e9))
        i_sml = torch.where(omask, idx, torch.full_like(idx, -1e9))
        imin = i_big.min(dim=-1).values; imax = i_sml.max(dim=-1).values
        o_center = (imax + imin) / 2.0                                   # mapping_center (idx)
        map_scale = 12.0 / (imax - imin).clamp_min(1e-8)                 # mapping_scale (fp_range=12)
        for n in range(N):
            for g in range(G):
                s = float(scale[n, g]); z = float(zp[n, g])
                mc = float(o_center[n, g]); ms = float(map_scale[n, g])
                for j in topk[n, g].tolist():
                    iv = (float(Wg[n, g, j]) - z) / s                    # this outlier's index
                    code4 = _fp4_encode_linear((iv - mc) * ms)
                    nib_g[n, g, j] = code4
                    W_ref[n, g, j] = (_fp4_decode(code4) / ms + mc) * s + z   # == base + kernel delta

    code = code.view(N, K).to(torch.uint8)
    packed = torch.zeros(N, K // 4, dtype=torch.uint8)
    for j in range(4):
        packed |= (code[:, j::4] << (2 * j))
    scale_exp = torch.round(torch.log2(scale.clamp_min(1e-12))).to(torch.int8)

    # relidx7 sidecar: block == group (B=group_size); npv positions ASCENDING per (n,block)
    # as 7-bit LSB-first; nibble stream 2/byte. Matches kernel relidx7_slot_pos/slot_delta.
    n_blk = G
    idx_blk_bytes = (npv * 7 + 7) // 8 if npv else 0
    nib_per_blk = (npv + 1) // 2 if npv else 0
    oindex = torch.zeros(N * n_blk * idx_blk_bytes, dtype=torch.uint8) if npv else torch.zeros(1, dtype=torch.uint8)
    ocode = torch.zeros(N * n_blk * nib_per_blk, dtype=torch.uint8) if npv else torch.zeros(1, dtype=torch.uint8)
    if npv:
        oi = oindex.view(N, n_blk, idx_blk_bytes)
        oc = ocode.view(N, n_blk, nib_per_blk)
        for n in range(N):
            for g in range(G):
                pos = sorted(topk[n, g].tolist())                      # ascending
                for s, p in enumerate(pos):
                    for b in range(7):                                  # 7-bit LSB-first
                        if (p >> b) & 1:
                            bit = s * 7 + b
                            oi[n, g, bit >> 3] |= (1 << (bit & 7))
                    oc[n, g, s >> 1] |= (nib_g[n, g, pos[s]].item() << (4 * (s & 1)))

    return dict(code=packed, scale=scale.view(N, G).to(torch.float32), scale_exp=scale_exp,
                zp=zp.view(N, G).to(torch.float32), map_scale=map_scale.to(torch.float32),
                map_center=o_center.to(torch.float32), oindex=oindex, ocode=ocode,
                W_base=W_base.view(N, K), W_ref=W_ref.view(N, K),
                N=N, K=K, G=G, group_size=group_size, npv=npv, n_blk=n_blk)


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


from ommx_gpu_serve.linear.quantize import (  # noqa: E402
    quantize_ommx_weight as _oq,
)


def _gate(name, y, y_ref, ctol=0.999, mtol=5e-2):
    c = cos(y, y_ref); md = float((y.float() - y_ref.float()).abs().max())
    fin = bool(torch.isfinite(y).all())
    ok = c >= ctol and md < mtol and fin
    print(f"[{name}] cos={c:.5f} max_diff={md:.2e} finite={fin} -> {'PASS' if ok else 'FAIL (cos first!)'}")
    return ok



def _bitmap_arm(mod, dev, N=256, K=1024):
    """relidx7 vs bitmap across the (group_size, npv) grid the shipped arm never covers."""
    RELIDX7, BITMAP = 0, 1
    GRID = [(16, 4), (32, 8), (64, 4), (64, 16), (64, 32), (128, 32), (128, 8)]

    def run(q, x, gs, npv, idx_fmt, M):
        code = q["code"].to(dev)
        scale = torch.pow(2.0, q["scale_exp"].float()).to(dev)
        zp = q["zp"].float().to(dev)
        oi, oc = q["oindex"].to(dev), q["ocode"].to(dev)
        ms, mc = q["map_scale"].float().to(dev), q["map_center"].float().to(dev)
        if M == 1:
            y = mod.decode_base(x, code, scale, zp, N, K, gs, False, "i2f4", 1)
            mod.sparse_correct(y, x, None, code, scale, oi, oc, ms, mc,
                               N, 1, K, gs, npv, gs, "i2f4", idx_fmt)
        else:
            y = mod.prefill_wmma(x, code, scale, zp, N, M, K, gs, False, "i2f4", 1)
            mod.sparse_correct(y, None, x.t().contiguous(), code, scale, oi, oc, ms, mc,
                               N, M, K, gs, npv, gs, "i2f4", idx_fmt)
        return y.float()

    print("\n[bitmap] relidx7 vs flat bitmask (cross gap must not exceed the self gap)")
    ok = True
    for gs, npv in GRID:
        W = torch.randn(N, K) * 0.05
        W[0, ::37] *= 12.0
        kw = dict(npv=npv, outlier_map="idx_range", zp_dtype=torch.bfloat16,
                  reference=True)
        q7 = _oq(W, gs, npv / gs, outlier_repr="relidx7", **kw)
        qb = _oq(W, gs, npv / gs, outlier_repr="bitmap", **kw)
        # if the packer chose different outliers the comparison would be meaningless
        assert torch.equal(q7["W_ref"], qb["W_ref"]), f"gs{gs} npv{npv}: W_ref differs"
        for M in (1, 32):
            x = (torch.randn(M, K, device=dev) * 0.1).to(torch.bfloat16)
            a7, b7 = run(q7, x, gs, npv, RELIDX7, M), run(q7, x, gs, npv, RELIDX7, M)
            ab, bb = run(qb, x, gs, npv, BITMAP, M), run(qb, x, gs, npv, BITMAP, M)
            self7 = float((a7 - b7).abs().max())
            selfb = float((ab - bb).abs().max())
            cross = float((ab - a7).abs().max())
            cell = cross <= max(self7, selfb) + 1e-12
            ok &= cell
            print(f"  gs={gs:<4} npv={npv:<3} M={M:<3} self(r7)={self7:.3e} "
                  f"self(bm)={selfb:.3e} cross={cross:.3e} -> "
                  f"{'PASS' if cell else 'FAIL'}")
    return ok


def main():
    assert torch.cuda.is_available(), "parity gate runs on GPU (sm_80+/sm_90a)"
    torch.manual_seed(42)
    dev = "cuda"
    N, K, gs = 512, 4096, 64
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build_ommx_linear
    mod = build_ommx_linear.build()
    ok = True

    # ---- BASE parity (opct=0): the affine INT2 dequant == fakequant base ----
    q = quantize_ommx_weight(torch.randn(N, K) * 0.1, gs, 0.0)
    Wb = q["W_base"].to(dev); code = q["code"].to(dev); scale = q["scale"].to(dev); zp = q["zp"].to(dev)
    for M in (1, 8):
        x = (torch.randn(M, K, device=dev) * 0.1).to(torch.bfloat16)
        y = mod.decode_base(x, code, scale, zp, N, K, gs, False, "i2f4", 1)
        ok &= _gate(f"decode M={M} base", y, x.float() @ Wb.t(), mtol=1e-2)
    # E8M0 in-kernel scale (lever #4): scale_exp=round(log2(scale)).int8 -> 2^exp is BIT-EXACT
    # to the fp32 scale (it IS a power of two), so the E8M0 path must MATCH the fp32 path exactly.
    scale_exp = q["scale_exp"].to(dev)
    x = (torch.randn(1, K, device=dev) * 0.1).to(torch.bfloat16)
    y_fp32 = mod.decode_base(x, code, scale, zp, N, K, gs, False, "i2f4", 1)
    y_e8m0 = mod.decode_base(x, code, scale, zp, N, K, gs, False, "i2f4", 1, scale_exp)
    ok &= _gate("decode M=1 E8M0 vs fp32 (must be exact)", y_e8m0, y_fp32, ctol=0.99999, mtol=1e-4)
    x = (torch.randn(32, K, device=dev) * 0.1).to(torch.bfloat16)
    y = mod.prefill_wmma(x, code, scale, zp, N, 32, K, gs, False, "i2f4", 1)
    ok &= _gate("prefill M=32 base", y, x.float() @ Wb.t())

    # ---- BASE + FP4 OUTLIER parity (opct=0.25): base decode + decoupled sparse_correct ----
    q = quantize_ommx_weight(torch.randn(N, K) * 0.1, gs, 0.25)
    Wr = q["W_ref"].to(dev); code = q["code"].to(dev); scale = q["scale"].to(dev); zp = q["zp"].to(dev)
    ms = q["map_scale"].to(dev); mcen = q["map_center"].to(dev)
    oindex = q["oindex"].to(dev); ocode = q["ocode"].to(dev)
    x = (torch.randn(1, K, device=dev) * 0.1).to(torch.bfloat16)
    y = mod.decode_base(x, code, scale, zp, N, K, gs, False, "i2f4", 1)   # fp32 [1,N] base
    mod.sparse_correct(y, x, None, code, scale, oindex, ocode, ms, mcen,
                       N, 1, K, gs, q["npv"], gs, "i2f4", 0)               # += outlier (0 = relidx7)
    ok &= _gate("decode M=1 base+outlier", y, x.float() @ Wr.t())

    # ---- BASE + FP4 OUTLIER at M>1: THE OTHER HALF OF sparse_correct's ABI ----
    # Everything above this line calls sparse_correct exactly once, at M=1. That is the
    # reason this gate reported PASS 5/5 while `integration/vllm/linear_method.py` sent
    # the M==1 argument list at every M and killed every prefill step of every recipe
    # with npv>0. The kernel ends with
    #
    #   if (M == 1) { TORCH_CHECK(A_opt.has_value(),  "M==1 correct needs A");  go1(...); }
    #   else        { TORCH_CHECK(At_opt.has_value(), "M>1 correct needs At"); goM(...); }
    #
    # so M>1 is a DIFFERENT CALL, not the same call with a bigger number:
    #   * the activation goes in as At = A^T, a CONTIGUOUS [K, M] bf16 buffer — goM's
    #     kernels index it `At[(size_t)k * M + m]`, and the host wrapper does NOT check
    #     contiguity, so a bare `x.t()` view would be read as dense and give a plausible
    #     wrong answer rather than an error;
    #   * the base output goes in as C, bf16 [M, N] (`out_or_C.data_ptr<at::BFloat16>()`),
    #     NOT the fp32 [1,N] `out` of the M==1 branch.
    # Both M>1 base kernels are covered because they return different dtypes:
    # prefill_wmma gives bf16 (`torch::empty({M,N}, A.options())`) and needs no cast,
    # decode_base gives fp32 (`torch::zeros(..., kFloat32)`) and does.
    for M, base in ((8, "decode"), (32, "prefill")):
        x = (torch.randn(M, K, device=dev) * 0.1).to(torch.bfloat16)
        if base == "prefill":
            C = mod.prefill_wmma(x, code, scale, zp, N, M, K, gs, False, "i2f4", 1)
        else:
            C = mod.decode_base(x, code, scale, zp, N, K, gs, False, "i2f4", 1).to(
                torch.bfloat16)                       # fp32 base -> the bf16 C goM wants
        At = x.t().contiguous()                       # [K, M] — the .contiguous() is the ABI
        assert At.shape == (K, M) and At.is_contiguous() and At.dtype == torch.bfloat16
        mod.sparse_correct(C, None, At, code, scale, oindex, ocode, ms, mcen,
                           N, M, K, gs, q["npv"], gs, "i2f4", 0)
        ok &= _gate(f"{base} M={M} base+outlier (M>1 At convention)", C,
                    x.float() @ Wr.t())

    # ---- BITMAP position reader: same positions, other encoding ----------------
    # The kernel reads two outlier POSITION encodings and they name the same ascending
    # set, so at equal (group_size, npv) they must agree. This arm exists because the
    # rest of this gate is hardcoded to relidx7 at gs=64/npv=16, so the bitmap reader and
    # every group size but 64 were unexercised.
    #
    # THE CRITERION IS NOT BIT-EXACTNESS AT M>1, AND THAT IS NOT A CONCESSION.
    # sparse_correct_batched_relidx7_kernel compacts its slots with
    # `atomicAdd(&s_count, cnt)` (ommx_linear.cu:2007), so the order corrections are
    # summed in is not fixed and ONE encoding does not reproduce ITSELF bit-for-bit
    # across two launches. Measured here: relidx7 against relidx7 differs by 9.766e-04 at
    # gs=128/npv=8/M=32, and bitmap against bitmap by 1.221e-04 at gs=64/npv=32/M=32. So
    # the honest test is that the CROSS-encoding gap never exceeds the gap an encoding
    # already has with itself; a bit-exact assertion would fail on the shipped path.
    ok &= _bitmap_arm(mod, dev)

    fs = dict(mod.fire_stats()); print(f"[fire_stats] {fs}")
    ok &= fs.get("decode_calls", 0) > 0 and fs.get("outlier_calls", 0) > 0
    print("\nPARITY GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
