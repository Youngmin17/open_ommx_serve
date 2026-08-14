"""OMMX canonical-KV-quant decode TPOT (B=1, HF-eager) — same timing path as
baseline/kv/_kivi_tpot_bench.py and baseline/kv/kitty/_kitty_tpot_bench.py.

OMMX attention was vLLM-only; this drives the SAME canonical paged-decode kernel from
an HF-eager full-model forward (LlamaForCausalLM_OMMX), so the OMMX vs bf16 B=1 TPOT
ratio is directly comparable to the KIVI / Kitty benches (identical warmup/measure/
CUDA-event method, B=1, fp16, BENCH_CTXS sweep).

Split (per layer, inside the modeling): PREFILL = flash_attention_2 over bf16 K/V +
pack the prompt into a per-layer CanonicalKVStore; DECODE = append the new bf16 K/V
row, regroup the completed 32-token group, and run ommx_paged_decode_attention_canonical
over the quantized prefix (i2f4 K + i2 V) + the bf16 sink/recent tail. SHADOW-FREE via
OMMX_KV_RING=1 (the store keeps only sink ∪ recent ∪ the live group, not the full bf16
sequence).

Env: ommx_fakequant (torch>=2.4 / transformers>=4.51 / triton). The OMMX Triton decode
kernel JIT-builds on the first real CUDA call (fine on GPU; adds a one-time warmup cost
absorbed by WARMUP=8). PYTHONPATH must include the ommx-serve repo root so
`ommx_gpu_serve` + the modeling import resolve.
Run:
  CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=<ommx-serve-root> \
      BENCH_CTXS=1024,4096,16384,32768 python _ommx_hf_tpot_bench.py
"""
import os, sys, gc, json, statistics

# ── OMMX recipe env (set BEFORE importing the modeling — resolve_serving_config reads
#    these at store-construction time). i2f4 K + i2 V, signed-select relidx7, sink=8/
#    recent=32/group=32, group_channels=32. OMMX_KV_RING=1 = SHADOW-FREE (no full bf16
#    sequence kept). setdefault so an outer caller can override any single knob. ──────
os.environ.setdefault("OMMX_ATTN_K_FORMAT", "i2f4")    # K = INT2 base + FP4 outlier
os.environ.setdefault("OMMX_ATTN_OUTLIERS", "3")       # K outliers per 32-token group
os.environ.setdefault("OMMX_ATTN_OUTLIER_SELECT", "signed")
os.environ.setdefault("OMMX_ATTN_OUTLIER_REPR", "relidx7")
os.environ.setdefault("OMMX_KV_GROUP_TOKENS", "32")    # K token-group vector_length
os.environ.setdefault("OMMX_KV_GROUP_CHANNELS", "32")  # V channel-group vector_length
os.environ.setdefault("OMMX_KV_SINK", "8")             # bf16 sink rows
os.environ.setdefault("OMMX_KV_RECENT", "32")          # bf16 recent rows
os.environ.setdefault("OMMX_KV_RING", "1")             # SHADOW-FREE ring (no bf16 shadow)
os.environ.setdefault("OMMX_KV_GPU_PACK", "1")         # on-device pack (no D2H round-trip)
# OMMX_ATTN_V_BF16 unset -> V quantized (i2). Set =1 for the v10a bf16-V variant.

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import torch
from transformers import LlamaConfig, AutoModelForCausalLM

# import the OMMX HF modeling (sits next to this file in ommx_gpu_serve/hf_eager/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ommx_llama_modeling import LlamaForCausalLM_OMMX

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CTXS = [int(x) for x in os.environ.get("BENCH_CTXS", "1024,4096,16384,32768,65536").split(",") if x]
WARMUP, MEASURE = 8, 40
DTYPE = torch.float16
DEV = "cuda"


def synced_median_step(model, ctx, is_ommx):
    """Prefill ctx tokens (B=1) then time 1-tok/step decode via CUDA events.

    OMMX and bf16 both use the returned past_key_values, so the loop is identical;
    only the model class (and hence the attention/cache representation) differs. For
    the OMMX arm we set OMMX_ATTN_MAXCTX so the per-layer CanonicalKVStore planes are
    sized to THIS ctx + decode steps (not the model's 131072 max_position_embeddings,
    which would allocate huge ≤3-bit planes per layer)."""
    if is_ommx:
        # the OMMX store/workspace are (re)built on this prefill; size them to ctx + steps.
        os.environ["OMMX_ATTN_MAXCTX"] = str(ctx + WARMUP + MEASURE + 8)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    vocab = model.config.vocab_size
    torch.manual_seed(42)
    input_ids = torch.randint(low=10, high=vocab - 10, size=(1, ctx), device=DEV, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    del out
    torch.cuda.synchronize()
    times = []
    se = torch.cuda.Event(enable_timing=True); ee = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for step in range(WARMUP + MEASURE):
            torch.cuda.synchronize(); se.record()
            o = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            ee.record(); torch.cuda.synchronize()
            ms = se.elapsed_time(ee)
            past = o.past_key_values
            next_tok = o.logits[:, -1:, :].argmax(dim=-1)
            del o
            if step >= WARMUP:
                times.append(ms)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    times.sort()
    med = statistics.median(times)
    p99 = times[min(len(times) - 1, int(0.99 * len(times)))]
    return med, p99, peak


def load_ommx():
    # PREFILL attention only (decode uses the OMMX kernel). Use whatever bf16-prefill
    # impl is available: flash_attention_2 if installed (fastest, mem-efficient long-ctx),
    # else sdpa (also mem-efficient, no flash_attn dep) — NOT eager (O(ctx^2) OOM at 128K).
    try:
        import flash_attn  # noqa
        prefill_attn = "flash_attention_2"
    except Exception:
        prefill_attn = "sdpa"
    cfg = LlamaConfig.from_pretrained(MODEL)
    cfg._attn_implementation = prefill_attn
    m = LlamaForCausalLM_OMMX.from_pretrained(
        MODEL, config=cfg, torch_dtype=DTYPE, low_cpu_mem_usage=True,
        attn_implementation=prefill_attn,
    ).to(DEV).eval()
    return m


def load_bf16():
    attn = "sdpa"
    try:
        import flash_attn  # noqa
        attn = "flash_attention_2"
    except Exception:
        attn = "sdpa"
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DTYPE, low_cpu_mem_usage=True, attn_implementation=attn).to(DEV).eval()
    return m, attn


def free(m):
    del m; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def run_arm(name, loader, is_ommx):
    print(f"### loading arm: {name}", flush=True)
    model = loader()
    if isinstance(model, tuple):
        model = model[0]
    out = {}
    for ctx in CTXS:
        try:
            med, p99, peak = synced_median_step(model, ctx, is_ommx)
            out[ctx] = {"tpot_ms": med, "p99_ms": p99, "peak_gb": peak}
            print(f"[{name}] ctx={ctx:6d}  tpot={med:7.3f} ms  p99={p99:7.3f}  peak={peak:6.2f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            out[ctx] = {"tpot_ms": None, "oom": True}
            print(f"[{name}] ctx={ctx:6d}  OOM", flush=True); torch.cuda.empty_cache()
        except Exception as e:
            out[ctx] = {"tpot_ms": None, "error": repr(e)[:300]}
            print(f"[{name}] ctx={ctx:6d}  ERROR {repr(e)[:300]}", flush=True); torch.cuda.empty_cache()
    free(model)
    return out


def main():
    print("torch", torch.__version__, "device", torch.cuda.get_device_name(0), flush=True)
    import transformers
    print("transformers", transformers.__version__, "CUDA_VISIBLE_DEVICES",
          os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("OMMX recipe:",
          "k_fmt=" + os.environ["OMMX_ATTN_K_FORMAT"],
          "npv=" + os.environ["OMMX_ATTN_OUTLIERS"],
          "gt=" + os.environ["OMMX_KV_GROUP_TOKENS"],
          "gc=" + os.environ["OMMX_KV_GROUP_CHANNELS"],
          "sink=" + os.environ["OMMX_KV_SINK"],
          "recent=" + os.environ["OMMX_KV_RECENT"],
          "ring=" + os.environ["OMMX_KV_RING"],
          "v_bf16=" + os.environ.get("OMMX_ATTN_V_BF16", "0"),
          flush=True)
    ommx_res = run_arm("ommx", load_ommx, True)
    bf16_res = run_arm("bf16", load_bf16, False)
    print("\n" + "=" * 80)
    print(f"Llama-3.1-8B-Instruct  B=1 decode TPOT (OMMX {os.environ['OMMX_ATTN_K_FORMAT']} "
          f"K + {'bf16' if os.environ.get('OMMX_ATTN_V_BF16', '0') not in ('0', '', 'false') else 'i2'} V, "
          f"npv={os.environ['OMMX_ATTN_OUTLIERS']} gt={os.environ['OMMX_KV_GROUP_TOKENS']} "
          f"sink={os.environ['OMMX_KV_SINK']} recent={os.environ['OMMX_KV_RECENT']}, "
          f"warmup={WARMUP} measure={MEASURE}, fp16)")
    print("%7s | %12s | %12s | %16s" % ("ctx", "ommx_tpot_ms", "bf16_tpot_ms", "ratio(bf16/ommx)"))
    print("-" * 60)
    for ctx in CTXS:
        kt = ommx_res.get(ctx, {}).get("tpot_ms"); bt = bf16_res.get(ctx, {}).get("tpot_ms")
        ratio = (bt / kt) if (kt and bt) else None
        ks = "OOM" if ommx_res.get(ctx, {}).get("oom") else ("--" if kt is None else f"{kt:.3f}")
        bs = "OOM" if bf16_res.get(ctx, {}).get("oom") else ("--" if bt is None else f"{bt:.3f}")
        rs = f"{ratio:.3f}" if ratio else "--"
        print(f"{ctx:>7} | {ks:>12} | {bs:>12} | {rs:>16}")
    print("=" * 80)
    win = any(
        (ommx_res.get(c, {}).get("tpot_ms") and bf16_res.get(c, {}).get("tpot_ms")
         and bf16_res[c]["tpot_ms"] / ommx_res[c]["tpot_ms"] > 1.0)
        for c in CTXS
    )
    print("OMMX B=1 TPOT win over bf16 at any ctx:", win)
    with open("_ommx_hf_tpot_result.json", "w") as fh:
        json.dump({"ommx": ommx_res, "bf16": bf16_res, "model": MODEL,
                   "k_format": os.environ["OMMX_ATTN_K_FORMAT"],
                   "outliers": os.environ["OMMX_ATTN_OUTLIERS"],
                   "group_tokens": os.environ["OMMX_KV_GROUP_TOKENS"],
                   "sink": os.environ["OMMX_KV_SINK"], "recent": os.environ["OMMX_KV_RECENT"],
                   "ring": os.environ["OMMX_KV_RING"],
                   "v_bf16": os.environ.get("OMMX_ATTN_V_BF16", "0")}, fh, indent=2)
    print("OMMX_HF_TPOT_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
