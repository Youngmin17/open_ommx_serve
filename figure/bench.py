"""Unified B=1 decode bench: TPOT + TTFT + quality for one method (HF-eager).

Methods: bf16 | kivi | kitty | ommx  — each in its own venv, its own real low-bit-KV kernel.

Same timing discipline for every method (CUDA events, warmup discard, median +
p99), so the OMMX vs KIVI / Kitty / bf16 numbers are directly comparable:
  TTFT = prefill latency of `ctx` tokens.
  TPOT = median 1-token/step decode latency.
  quality = greedy generation from a real prompt (coherence sanity, not a metric).

Loaders are inlined (proven recipe mirrors baseline/kv/_kivi_tpot_bench.py,
baseline/kv/kitty/_kitty_tpot_bench.py, ommx_gpu_serve/hf_eager/_ommx_hf_tpot_bench.py)
so a single --model flows to every arm. Each method runs in ITS OWN venv (run.sh),
so imports are lazy inside the method branch.

Run:
  CUDA_VISIBLE_DEVICES=<g> python figure/bench.py --method ommx \
      --model meta-llama/Llama-3.1-8B-Instruct --ctxs 1024,4096,16384,32768 \
      --out figure/data/ommx.json
"""
import argparse
import gc
import json
import os
import statistics
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_QUALITY_PROMPT = ("Question: What is the capital of France, and why is it "
                          "historically important?\nAnswer:")


def _pick_prefill_attn():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


# ── per-method loaders (lazy imports) ─────────────────────────────────────────
def load_model(method, model, dtype):
    """Return (model_obj, kind, extra) where kind in {'past','kitty'}."""
    import torch
    from transformers import LlamaConfig, AutoModelForCausalLM

    if method == "bf16":
        attn = _pick_prefill_attn()
        m = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation=attn).cuda().eval()
        return m, "past", {}

    if method == "kivi":
        # vendored kivi package: <repo>/baseline/kivi (package-qualified imports)
        for p in (os.path.join(REPO, "baseline"), os.path.join(REPO, "baseline", "kivi")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from kivi.models.llama_kivi_eval import LlamaForCausalLM_KIVI_eval
        cfg = LlamaConfig.from_pretrained(model)
        cfg.k_bits, cfg.v_bits, cfg.group_size, cfg.residual_length = 2, 2, 32, 128
        cfg.use_flash = True
        m = LlamaForCausalLM_KIVI_eval.from_pretrained(
            model, config=cfg, torch_dtype=dtype, low_cpu_mem_usage=True).cuda().eval()
        return m, "past", {}

    if method == "kitty":
        pkg = os.environ.get("KITTY_PKG_PATH", "")
        for p in (pkg, os.path.join(REPO, "baseline", "kitty")):
            if p and p not in sys.path:
                sys.path.insert(0, p)
        from _kitty_llama_modeling import LlamaForCausalLM_Kitty
        from kitty.kvcache import get_kvcache_kitty
        attn = _pick_prefill_attn()
        cfg = LlamaConfig.from_pretrained(model)
        cfg._attn_implementation = attn
        m = LlamaForCausalLM_Kitty.from_pretrained(
            model, config=cfg, torch_dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation=attn).cuda().eval()
        return m, "kitty", {"get_kv": get_kvcache_kitty, "cfg": cfg}

    if method == "ommx":
        # OMMX best recipe (mirrors fakequant/run.sh --phase kv: INT2 + 12/64 FP4 K-outliers,
        # sink 8, pow2 scales; serving maps 12/64 -> 6 outliers per 32-channel group).
        for k, v in {
            "OMMX_ATTN_K_FORMAT": "i2f4", "OMMX_ATTN_OUTLIERS": "6", "OMMX_ATTN_POW2": "1",
            "OMMX_ATTN_OUTLIER_SELECT": "signed", "OMMX_ATTN_OUTLIER_REPR": "relidx7",
            "OMMX_KV_GROUP_TOKENS": "32", "OMMX_KV_GROUP_CHANNELS": "32",
            "OMMX_KV_SINK": "8", "OMMX_KV_RECENT": "32",
            "OMMX_KV_RING": "1", "OMMX_KV_GPU_PACK": "1",
        }.items():
            os.environ.setdefault(k, v)
        hf = os.path.join(REPO, "ommx_gpu_serve", "hf_eager")
        if hf not in sys.path:
            sys.path.insert(0, hf)
        from _ommx_llama_modeling import LlamaForCausalLM_OMMX
        attn = _pick_prefill_attn()
        cfg = LlamaConfig.from_pretrained(model)
        cfg._attn_implementation = attn
        m = LlamaForCausalLM_OMMX.from_pretrained(
            model, config=cfg, torch_dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation=attn).cuda().eval()
        return m, "past", {}

    raise ValueError(f"unknown method {method}")


# ── timing ────────────────────────────────────────────────────────────────────
def measure(method, model_obj, kind, extra, ctx, warmup, measure_n):
    """Prefill ctx (time TTFT) then 1-tok/step decode (median TPOT). B=1."""
    import torch
    if method == "ommx":  # size the per-layer store to this ctx (not the 131072 max)
        os.environ["OMMX_ATTN_MAXCTX"] = str(ctx + warmup + measure_n + 8)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    vocab = model_obj.config.vocab_size
    torch.manual_seed(42)
    ids = torch.randint(10, vocab - 10, (1, ctx), device="cuda", dtype=torch.long)

    kv = None
    if kind == "kitty":
        kv = extra["get_kv"](extra["cfg"], max_batch_size=1,
                             max_length=ctx + warmup + measure_n + 8)

    se = torch.cuda.Event(enable_timing=True); ee = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); se.record()
    with torch.no_grad():
        out = model_obj(input_ids=ids, past_key_values=kv, use_cache=True) if kv is not None \
            else model_obj(input_ids=ids, use_cache=True)
    ee.record(); torch.cuda.synchronize()
    ttft_ms = se.elapsed_time(ee)

    past = kv if kv is not None else out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(-1)
    del out
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for step in range(warmup + measure_n):
            torch.cuda.synchronize(); se.record()
            o = model_obj(input_ids=next_tok, past_key_values=past, use_cache=True)
            ee.record(); torch.cuda.synchronize()
            ms = se.elapsed_time(ee)
            if kv is None:
                past = o.past_key_values
            next_tok = o.logits[:, -1:, :].argmax(-1)
            del o
            if step >= warmup:
                times.append(ms)
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    times.sort()
    tpot = statistics.median(times)
    p99 = times[min(len(times) - 1, int(0.99 * len(times)))]
    return {"ttft_ms": round(ttft_ms, 4), "tpot_ms": round(tpot, 4),
            "p99_ms": round(p99, 4), "peak_gb": round(peak_gb, 4)}


def quality_gen(method, model_obj, kind, extra, tokenizer, prompt, n=96):
    """Greedy generation from a real prompt -> text (coherence sanity)."""
    import torch
    if method == "ommx":
        os.environ["OMMX_ATTN_MAXCTX"] = str(2048 + n)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    kv = None
    if kind == "kitty":
        kv = extra["get_kv"](extra["cfg"], max_batch_size=1, max_length=ids.shape[1] + n + 8)
    gen = []
    with torch.no_grad():
        out = model_obj(input_ids=ids, past_key_values=kv, use_cache=True) if kv is not None \
            else model_obj(input_ids=ids, use_cache=True)
        past = kv if kv is not None else out.past_key_values
        nt = out.logits[:, -1:, :].argmax(-1)
        for _ in range(n):
            tid = int(nt.item())
            if tid == tokenizer.eos_token_id:
                break
            gen.append(tid)
            o = model_obj(input_ids=nt, past_key_values=past, use_cache=True)
            if kv is None:
                past = o.past_key_values
            nt = o.logits[:, -1:, :].argmax(-1)
    return tokenizer.decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["bf16", "kivi", "kitty", "ommx"])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--ctxs", default=os.environ.get("BENCH_CTXS", "1024,4096,16384,32768,65536"))
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--measure", type=int, default=40)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--quality-prompt", default=DEFAULT_QUALITY_PROMPT)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    dtype = getattr(torch, args.dtype)
    ctxs = [int(x) for x in args.ctxs.split(",") if x]
    out = args.out or os.path.join(REPO, "figure", "data", f"{args.method}.json")

    print(f"torch {torch.__version__}  dev {torch.cuda.get_device_name(0)}  "
          f"method={args.method}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model_obj, kind, extra = load_model(args.method, args.model, dtype)

    per_ctx = {}
    for ctx in ctxs:
        try:
            r = measure(args.method, model_obj, kind, extra, ctx, args.warmup, args.measure)
            per_ctx[str(ctx)] = r
            print(f"[{args.method}] ctx={ctx:6d}  ttft={r['ttft_ms']:8.2f}  "
                  f"tpot={r['tpot_ms']:7.3f}  p99={r['p99_ms']:7.3f}  peak={r['peak_gb']:.2f}GB",
                  flush=True)
        except torch.cuda.OutOfMemoryError:
            per_ctx[str(ctx)] = {"oom": True}
            print(f"[{args.method}] ctx={ctx:6d}  OOM", flush=True); torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            per_ctx[str(ctx)] = {"error": repr(e)[:300]}
            print(f"[{args.method}] ctx={ctx:6d}  ERROR {repr(e)[:300]}", flush=True)
            torch.cuda.empty_cache()

    quality = None
    if not args.no_quality:
        try:
            txt = quality_gen(args.method, model_obj, kind, extra, tok, args.quality_prompt)
            quality = {"prompt": args.quality_prompt, "output": txt}
            print(f"\n[{args.method}] quality:\n{txt}\n", flush=True)
        except Exception as e:  # noqa: BLE001
            quality = {"prompt": args.quality_prompt, "error": repr(e)[:300]}

    # provenance: seed, code + model SHAs, versions, GPU — so a JSON is self-describing/reproducible
    def _git_sha():
        try:
            import subprocess
            return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:  # noqa: BLE001
            return None
    try:
        import triton  # noqa
        triton_v = triton.__version__
    except Exception:  # noqa: BLE001
        triton_v = None
    prov = {"seed": 42, "git_sha": _git_sha(), "torch": torch.__version__, "triton": triton_v,
            "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0]}
    if args.method == "ommx":  # capture the exact resolved recipe env (setdefault may be overridden)
        prov["ommx_recipe"] = {k: os.environ.get(k) for k in (
            "OMMX_ATTN_K_FORMAT", "OMMX_ATTN_OUTLIERS", "OMMX_ATTN_POW2", "OMMX_KV_GROUP_TOKENS",
            "OMMX_KV_GROUP_CHANNELS", "OMMX_KV_SINK", "OMMX_KV_RECENT")}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"method": args.method, "model": args.model, "ctxs": per_ctx,
                   "quality": quality,
                   "meta": {"warmup": args.warmup, "measure": args.measure,
                            "dtype": args.dtype, **prov}}, fh, indent=2)
    del model_obj; gc.collect(); torch.cuda.empty_cache()
    print(f"BENCH_DONE method={args.method} out={out}", flush=True)


if __name__ == "__main__":
    main()
