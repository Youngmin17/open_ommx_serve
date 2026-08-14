import os, sys, gc, json, statistics
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import torch
from transformers import LlamaConfig, AutoModelForCausalLM

# import the ported Kitty Llama modeling (sits next to this file)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kitty_llama_modeling import LlamaForCausalLM_Kitty
from kitty.kvcache import get_kvcache_kitty

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CTXS = [int(x) for x in os.environ.get("BENCH_CTXS", "1024,4096,16384,32768,65536").split(",") if x]
WARMUP, MEASURE = 8, 40
DTYPE = torch.float16
DEV = "cuda"


def synced_median_step_bf16(model, ctx):
    """bf16 baseline: stock HF model, B=1 prefill then manual 1-tok/step decode timing."""
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


def synced_median_step_kitty(model, ctx):
    """Kitty 2-bit KV: build KittyCache, prefill ctx tokens, then manual 1-tok/step decode timing.

    The Kitty decode path runs the Triton qk/sv 2-bit paged kernels + per-page quantize_decode.
    KV memory measured from the packed page tensors (uint8) + metadata (fp16), NOT torch peak,
    because Kitty pre-allocates MAX_PAGE pages up front (static allocation)."""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfg = model.config
    vocab = cfg.vocab_size
    torch.manual_seed(42)
    input_ids = torch.randint(low=10, high=vocab - 10, size=(1, ctx), device=DEV, dtype=torch.long)

    # max_length must cover ctx + warmup+measure decode steps
    max_len = ctx + WARMUP + MEASURE + 8
    kv = get_kvcache_kitty(cfg, max_batch_size=1, max_length=max_len)

    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=kv, use_cache=True)
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    del out
    torch.cuda.synchronize()

    # KV memory actually used by packed pages (bytes used = pages-needed, not MAX allocation)
    # report the *allocated tensor* footprint of the cache (what Kitty holds resident).
    kv_bytes = 0
    for layer in kv.kv_cache:
        for t in (layer.KeyCache, layer.KeyCache_metadata, layer.ValueCache, layer.ValueCache_metadata,
                  layer.Sink_Buffer_K, layer.Sink_Buffer_V, layer.Q_Buffer_K, layer.Q_Buffer_V,
                  layer.Local_Buffer_V, layer.PageTable_K, layer.PageTable_V):
            kv_bytes += t.numel() * t.element_size()
    kv_gb = kv_bytes / (1024 ** 3)

    times = []
    se = torch.cuda.Event(enable_timing=True); ee = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for step in range(WARMUP + MEASURE):
            torch.cuda.synchronize(); se.record()
            o = model(input_ids=next_tok, past_key_values=kv, use_cache=True)
            ee.record(); torch.cuda.synchronize()
            ms = se.elapsed_time(ee)
            next_tok = o.logits[:, -1:, :].argmax(dim=-1)
            del o
            if step >= WARMUP:
                times.append(ms)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    times.sort()
    med = statistics.median(times)
    p99 = times[min(len(times) - 1, int(0.99 * len(times)))]
    del kv; gc.collect(); torch.cuda.empty_cache()
    return med, p99, peak, kv_gb


def load_kitty():
    cfg = LlamaConfig.from_pretrained(MODEL)
    cfg._attn_implementation = "flash_attention_2"  # prefill only; decode uses Kitty kernel
    m = LlamaForCausalLM_Kitty.from_pretrained(
        MODEL, config=cfg, torch_dtype=DTYPE, low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
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
        MODEL, torch_dtype=DTYPE, low_cpu_mem_usage=True, attn_implementation=attn,
    ).to(DEV).eval()
    return m, attn


def free(m):
    del m; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def run_bf16(ctxs):
    print("### loading arm: bf16", flush=True)
    model, attn = load_bf16()
    out = {}
    for ctx in ctxs:
        try:
            med, p99, peak = synced_median_step_bf16(model, ctx)
            out[ctx] = {"tpot_ms": med, "p99_ms": p99, "peak_gb": peak}
            print(f"[bf16] ctx={ctx:6d}  tpot={med:7.3f} ms  p99={p99:7.3f}  peak={peak:6.2f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            out[ctx] = {"tpot_ms": None, "oom": True}
            print(f"[bf16] ctx={ctx:6d}  OOM", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            out[ctx] = {"tpot_ms": None, "error": repr(e)}
            print(f"[bf16] ctx={ctx:6d}  ERROR {repr(e)[:300]}", flush=True)
            torch.cuda.empty_cache()
    free(model)
    return out, attn


def run_kitty(ctxs):
    print("### loading arm: kitty", flush=True)
    model = load_kitty()
    out = {}
    for ctx in ctxs:
        try:
            med, p99, peak, kv_gb = synced_median_step_kitty(model, ctx)
            out[ctx] = {"tpot_ms": med, "p99_ms": p99, "peak_gb": peak, "kv_gb": kv_gb}
            print(f"[kitty] ctx={ctx:6d}  tpot={med:7.3f} ms  p99={p99:7.3f}  peak={peak:6.2f} GB  kv={kv_gb:6.3f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            out[ctx] = {"tpot_ms": None, "oom": True}
            print(f"[kitty] ctx={ctx:6d}  OOM", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            out[ctx] = {"tpot_ms": None, "error": repr(e)}
            print(f"[kitty] ctx={ctx:6d}  ERROR {repr(e)[:300]}", flush=True)
            torch.cuda.empty_cache()
    free(model)
    return out


def main():
    print("torch", torch.__version__, "device", torch.cuda.get_device_name(0), flush=True)
    print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    import transformers
    print("transformers", transformers.__version__, flush=True)

    bf16_res, attn_meta = run_bf16(CTXS)
    kitty_res = run_kitty(CTXS)

    print("\n" + "=" * 88)
    print(f"Llama-3.1-8B-Instruct  B=1 decode TPOT  (warmup={WARMUP} measure={MEASURE}, fp16)")
    print(f"bf16 attn_impl = {attn_meta}")
    print("Kitty config: 2-bit K/V, PAGE_SIZE=128, 25% K channels boosted to INT4 (Kitty-Pro), sink=32")
    print("=" * 88)
    hdr = "%7s | %14s | %13s | %16s | %10s | %11s" % (
        "ctx", "kitty_tpot_ms", "bf16_tpot_ms", "ratio(bf16/kitty)", "kitty_kvGB", "bf16_peakGB")
    print(hdr)
    print("-" * 92)
    for ctx in CTXS:
        k = kitty_res.get(ctx, {}); b = bf16_res.get(ctx, {})
        kt = k.get("tpot_ms"); bt = b.get("tpot_ms")
        kkv = k.get("kv_gb"); bp = b.get("peak_gb")
        ratio = (bt / kt) if (kt and bt) else None

        def f(x, w, p=3):
            if isinstance(x, dict) and x.get("oom"):
                return "OOM".rjust(w)
            return ("--" if x is None else f"{x:.{p}f}").rjust(w)
        kts = f(kt, 14); bts = f(bt, 13)
        rs = (f"{ratio:.3f}".rjust(16)) if ratio else "--".rjust(16)
        kkvs = f(kkv, 10, 3); bps = f(bp, 11, 2)
        print(f"{ctx:>7} | {kts} | {bts} | {rs} | {kkvs} | {bps}")
    print("=" * 88)
    win = any(
        (kitty_res.get(c, {}).get("tpot_ms") and bf16_res.get(c, {}).get("tpot_ms")
         and bf16_res[c]["tpot_ms"] / kitty_res[c]["tpot_ms"] > 1.0)
        for c in CTXS
    )
    print("Kitty B=1 TPOT win over bf16 at any ctx:", win)
    with open("_kitty_tpot_result.json", "w") as fh:
        json.dump({"bf16": bf16_res, "kitty": kitty_res, "attn": attn_meta, "model": MODEL}, fh, indent=2)
    print("KITTY_TPOT_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
