"""KIVI 2-bit KV decode TPOT (B=1, HF-eager) — same timing path as kitty/_kitty_tpot_bench.py.

KIVI keeps a packed 2-bit K/V cache internally and exposes it as a normal HF
``past_key_values`` tuple, so the decode loop is identical to a stock bf16 run
(prefill once, then 1-token/step with the returned cache). This makes the KIVI vs
bf16 TPOT ratio directly comparable to the Kitty bench (same warmup/measure/CUDA-event
method, B=1, fp16).

Env: kivi_serving (torch 2.4 / older transformers — the kivi package targets it).
PYTHONPATH must include the KIVI vendor repo (for `kivi.models.llama_kivi`).
Run:
  CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=<repo>/baseline \
      python baseline/kivi/_kivi_tpot_bench.py
"""
import os, sys, gc, json, statistics
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import torch
from transformers import LlamaConfig, AutoModelForCausalLM

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CTXS = [int(x) for x in os.environ.get("BENCH_CTXS", "1024,4096,16384,32768,65536").split(",") if x]
WARMUP, MEASURE = 8, 40
DTYPE = torch.float16
DEV = "cuda"
K_BITS, V_BITS, GROUP, RESID = 2, 2, 32, 128  # group must divide head_dim(128) & residual(128)


def synced_median_step(model, ctx, is_kivi):
    """Prefill ctx tokens (B=1) then time 1-tok/step decode via CUDA events.

    KIVI and bf16 both use the returned past_key_values, so the loop is identical;
    only the model class (and hence the cache representation) differs."""
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


def load_kivi():
    # NOTE: llama_kivi.LlamaForCausalLM_KIVI asserts num_key_value_groups==1 (MHA only)
    # -> fails on GQA models (Llama-3.1-8B = 32 q / 8 kv). The _eval variant supports GQA
    # and is what lm-eval/eval_perplexity use for both PPL and generate_until decode.
    from kivi.models.llama_kivi_eval import LlamaForCausalLM_KIVI_eval as LlamaForCausalLM_KIVI
    cfg = LlamaConfig.from_pretrained(MODEL)
    cfg.k_bits = K_BITS; cfg.v_bits = V_BITS
    cfg.group_size = GROUP; cfg.residual_length = RESID
    cfg.use_flash = True
    m = LlamaForCausalLM_KIVI.from_pretrained(
        MODEL, config=cfg, torch_dtype=DTYPE, low_cpu_mem_usage=True).to(DEV).eval()
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


def run_arm(name, loader, is_kivi):
    print(f"### loading arm: {name}", flush=True)
    model = loader() if not is_kivi else loader()
    if isinstance(model, tuple):
        model = model[0]
    out = {}
    for ctx in CTXS:
        try:
            med, p99, peak = synced_median_step(model, ctx, is_kivi)
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
    kivi_res = run_arm("kivi", load_kivi, True)
    bf16_res = run_arm("bf16", load_bf16, False)
    print("\n" + "=" * 80)
    print(f"Llama-3.1-8B-Instruct  B=1 decode TPOT (KIVI k{K_BITS}v{V_BITS} g{GROUP} r{RESID}, "
          f"warmup={WARMUP} measure={MEASURE}, fp16)")
    print("%7s | %12s | %12s | %16s" % ("ctx", "kivi_tpot_ms", "bf16_tpot_ms", "ratio(bf16/kivi)"))
    print("-" * 60)
    for ctx in CTXS:
        kt = kivi_res.get(ctx, {}).get("tpot_ms"); bt = bf16_res.get(ctx, {}).get("tpot_ms")
        ratio = (bt / kt) if (kt and bt) else None
        ks = "OOM" if kivi_res.get(ctx, {}).get("oom") else ("--" if kt is None else f"{kt:.3f}")
        bs = "OOM" if bf16_res.get(ctx, {}).get("oom") else ("--" if bt is None else f"{bt:.3f}")
        rs = f"{ratio:.3f}" if ratio else "--"
        print(f"{ctx:>7} | {ks:>12} | {bs:>12} | {rs:>16}")
    print("=" * 80)
    with open("_kivi_tpot_result.json", "w") as fh:
        json.dump({"kivi": kivi_res, "bf16": bf16_res, "model": MODEL,
                   "k_bits": K_BITS, "v_bits": V_BITS, "group": GROUP, "resid": RESID}, fh, indent=2)
    print("KIVI_TPOT_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
