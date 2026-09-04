"""OMMX canonical-KV-quant decode TPOT (B=1, HF-eager) — same timing path as
baseline/kivi/_kivi_tpot_bench.py and baseline/kitty/_kitty_tpot_bench.py.

OMMX attention was vLLM-only; this drives the SAME canonical paged-decode kernel from
an HF-eager full-model forward (LlamaForCausalLM_OMMX), under the same timing discipline
as those two (identical warmup/measure/CUDA-event method, B=1, fp16, BENCH_CTXS sweep),
so the OMMX ARM's absolute ms/step sits on the same scale as the KIVI / Kitty arms.

The bf16 RATIO does NOT carry across under the default below. Both sibling benches step
their bf16 baseline from ``out.past_key_values``, i.e. a DynamicCache
(``_kivi_tpot_bench.py``'s own note: "KIVI and bf16 both use the returned
past_key_values"; ``_kitty_tpot_bench.py``'s ``synced_median_step_bf16`` does the same),
while this bench defaults to a StaticCache. Comparing this script's bf16/ommx ratio to
theirs therefore compares two different bf16 baselines — set OMMX_BENCH_BF16_CACHE=dynamic
if that cross-bench ratio is what you need, and read the next paragraph first.

THE bf16 ARM'S CACHE IS PART OF THE MEASUREMENT, NOT A DETAIL. With HF's default
DynamicCache the bf16 arm torch.cat's the whole KV of every layer on EVERY decode step
- an O(ctx) per-step cost that none of the quantized arms pay (OMMX's cache object holds
no tensors at all), i.e. it measures the cache scaffold, not bf16 attention. This bench
therefore defaults to a preallocated transformers StaticCache for the bf16 arm
(OMMX_BENCH_BF16_CACHE=static), the same fix and the same helper figure/bench.py uses
for its --bf16-cache flag. OMMX_BENCH_BF16_CACHE=dynamic restores the old behaviour
EXPLICITLY, for reproducing previously published numbers; the mode is printed, written
into the result JSON, and stamped on the verdict line so a number from this script
cannot be quoted without it.

figure/bench.py is the MAINTAINED four-arm bench (bf16 / kivi / kitty / ommx, one flag
set, provenance block, mean/max/interpolated-p99/raw per-step times). This script stays
for the OMMX-vs-bf16 pair alone; prefer figure/bench.py for anything published.

Split (per layer, inside the modeling): PREFILL = the model's configured prefill
attention over bf16 K/V (load_ommx below picks flash_attention_2 when flash-attn is
importable and sdpa otherwise, and the resolved choice is what the arm measured) +
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
  # add OMMX_BENCH_BF16_CACHE=dynamic to reproduce the old DynamicCache bf16 baseline
"""
import os, sys, gc, json, statistics

# ── OMMX recipe env (set BEFORE importing the modeling — resolve_serving_config reads
#    these at store-construction time). i2f4 K + i2 V, signed select, positions in the
#    serving default encoding (DEFAULT_OUTLIER_REPR = bitmap), sink=8/recent=32/group=32,
#    group_channels=32. OMMX_KV_RING=1 = SHADOW-FREE (no full bf16 sequence kept).
#    setdefault so an outer caller can override any single knob.
#
#    These values are the PUBLISHED CANONICAL RECIPE (npv=6 outliers, pow2 -> int8 group
#    scale), identical to figure/bench.py's OMMX_RECIPE_ENV and to
#    eval/lm_eval/models/ommx_hf_model.py. They are ALSO the serving defaults
#    (integration/vllm/config.py DEFAULT_OUTLIERS / DEFAULT_POW2 / DEFAULT_OUTLIER_REPR),
#    so this block is a restatement kept for PROVENANCE: the recipe this bench PRINTS and
#    writes to JSON is readable here, and its numeric knobs stay the ones the published
#    accuracy numbers used even if a library default moves. ─────────────────────────────
from ommx_gpu_serve.integration.vllm.config import DEFAULT_OUTLIER_REPR
os.environ.setdefault("OMMX_ATTN_K_FORMAT", "i2f4")    # K = INT2 base + FP4 outlier
os.environ.setdefault("OMMX_ATTN_OUTLIERS", "6")       # K outliers per 32-token group
os.environ.setdefault("OMMX_ATTN_POW2", "1")           # pow2 group scale -> int8 exponent
os.environ.setdefault("OMMX_ATTN_OUTLIER_SELECT", "signed")
os.environ.setdefault("OMMX_ATTN_OUTLIER_REPR", DEFAULT_OUTLIER_REPR)  # serving default
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
# WARMUP/MEASURE also decide how much of the OMMX regroup cycle the window can see: the
# OMMX arm repacks a completed group every OMMX_KV_GROUP_TOKENS (=32) decode steps, so a
# 40-step measurement contains at most ONE such boundary and the median contains none. A
# 240-step run of this arm on an H100 NVL (fp16, clean GPU) put the boundaries at measured
# indices 23, 55, 87, ... — spacing exactly 32 — costing 86-140 ms against a 34.29-35.44 ms
# steady state, i.e. +17% amortized. max_ms below is what exposes that step; tpot_ms cannot.
WARMUP, MEASURE = 8, 40
DTYPE = torch.float16
DEV = "cuda"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# bf16 cache mode: 'static' (default, fair) or 'dynamic' (the old handicapped baseline).
# An unrecognised value RAISES rather than falling back to either one — silently picking a
# cache is exactly the failure this knob exists to remove.
BF16_CACHE = os.environ.get("OMMX_BENCH_BF16_CACHE", "static").strip().lower()
if BF16_CACHE not in {"static", "dynamic"}:
    raise ValueError(
        f"OMMX_BENCH_BF16_CACHE={BF16_CACHE!r} is not 'static' or 'dynamic'. 'static' "
        "preallocates a transformers StaticCache for the bf16 arm (fair); 'dynamic' is "
        "HF's DynamicCache, which torch.cat's the full KV of every layer on every decode "
        "step and inflates the bf16 baseline.")


def _percentile(asc, q):
    """Linear-interpolated percentile of an ASCENDING sample (numpy's 'linear' method).

    MIRROR of figure/bench.py's ``_percentile``, and for the same reason: the old
    ``times[int(0.99 * len(times))]`` evaluates to index 39 for the shipped n=40, i.e. it
    reported the MAXIMUM as the 99th percentile. max_ms is now reported separately."""
    if not asc:
        raise ValueError("percentile of an empty sample")
    if len(asc) == 1:
        return asc[0]
    pos = (len(asc) - 1) * q
    lo = int(pos // 1)
    hi = min(lo + 1, len(asc) - 1)
    return asc[lo] + (asc[hi] - asc[lo]) * (pos - lo)


def _static_cache_factory():
    """figure/bench.py's ``_make_static_cache``, IMPORTED — never re-implemented here.

    That helper carries the transformers-version skew (max_batch_size <-> batch_size,
    device/dtype added then made optional) plus a read-back check that the constructed
    cache really honoured max_cache_len, and it RAISES instead of degrading to a
    DynamicCache. Copying it into this file would let the two benches drift into two
    different bf16 baselines, which is the defect this whole option exists to close.
    If figure/ is not reachable (this file installed without the repo tree) we raise and
    name both fixes rather than quietly running the handicapped cache."""
    import importlib.util
    path = os.path.join(REPO, "figure", "bench.py")
    if not os.path.isfile(path):
        raise RuntimeError(
            f"OMMX_BENCH_BF16_CACHE=static needs figure/bench.py (looked at {path}), which "
            "owns the StaticCache constructor this bench reuses. Fix: run this script from "
            "an ommx-serve source checkout (its docstring already requires the repo root on "
            "PYTHONPATH), run figure/bench.py --method bf16 instead, or set "
            "OMMX_BENCH_BF16_CACHE=dynamic and accept the DynamicCache handicap explicitly.")
    spec = importlib.util.spec_from_file_location("_ommx_figure_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._make_static_cache


def synced_median_step(model, ctx, is_ommx, static_cache=False):
    """Prefill ctx tokens (B=1) then time 1-tok/step decode via CUDA events.

    The OMMX arm is stepped from its own returned past_key_values (an
    OmmxSeqCounterCache, which stores NO tensors — the quantized planes live in the
    per-layer CanonicalKVStore). The bf16 arm's cache is the fair-comparison axis: with
    ``static_cache=True`` we preallocate a transformers StaticCache to
    ctx+WARMUP+MEASURE+8 and feed it (plus an explicit cache_position, which a fixed-size
    cache needs to know its write offset) every step; with ``static_cache=False`` the arm
    keeps HF's DynamicCache, i.e. a full-KV torch.cat per layer per step. The two are NOT
    the same measurement, so the cache class actually used is returned and published.

    For the OMMX arm we set OMMX_ATTN_MAXCTX so the per-layer CanonicalKVStore planes are
    sized to THIS ctx + decode steps (not the model's 131072 max_position_embeddings,
    which would size every quantized plane to 131072 tokens per layer).

    Returns (median_ms, p99_ms, max_ms, peak_gb, cache_class, static_cache_probe).
    ``static_cache_probe`` is figure/bench.py's read-back note (which StaticCache kwargs
    worked, what max_cache_len the object reports, and whether that size could be
    VERIFIED at all) or None when no static cache was built. It is carried out to the
    JSON for the same reason figure/bench.py carries it: on a transformers whose size
    probes are all deprecated the buffer is unverified, and an unverified cell otherwise
    reads exactly like a verified one."""
    if is_ommx:
        # the OMMX store/workspace are (re)built on this prefill; size them to ctx + steps.
        os.environ["OMMX_ATTN_MAXCTX"] = str(ctx + WARMUP + MEASURE + 8)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    vocab = model.config.vocab_size
    torch.manual_seed(42)
    input_ids = torch.randint(low=10, high=vocab - 10, size=(1, ctx), device=DEV, dtype=torch.long)
    # Built AFTER reset_peak_memory_stats so peak_gb counts the preallocation instead of a
    # DynamicCache's grow-by-cat high-water mark (same accounting as figure/bench.py).
    probe: dict = {}
    kv = (_static_cache_factory()(model, ctx + WARMUP + MEASURE + 8, DTYPE, probe_out=probe)
          if static_cache else None)

    def _cpos(start, n):   # built outside the timed window
        return torch.arange(start, start + n, device=DEV, dtype=torch.long) if kv is not None else None

    kw = {} if kv is None else {"past_key_values": kv, "cache_position": _cpos(0, ctx)}
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, **kw)
    past = kv if kv is not None else out.past_key_values
    cache_class = type(past).__name__      # the object the decode loop actually feeds, by name
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    del out
    torch.cuda.synchronize()
    times = []
    se = torch.cuda.Event(enable_timing=True); ee = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for step in range(WARMUP + MEASURE):
            cp = _cpos(ctx + step, 1)
            kw = {} if cp is None else {"cache_position": cp}
            torch.cuda.synchronize(); se.record()
            o = model(input_ids=next_tok, past_key_values=past, use_cache=True, **kw)
            ee.record(); torch.cuda.synchronize()
            ms = se.elapsed_time(ee)
            if kv is None:
                past = o.past_key_values
            next_tok = o.logits[:, -1:, :].argmax(dim=-1)
            del o
            if step >= WARMUP:
                times.append(ms)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    asc = sorted(times)
    med = statistics.median(asc)
    p99 = _percentile(asc, 0.99)
    return med, p99, asc[-1], peak, cache_class, probe.get("static_cache_probe")


def load_ommx():
    # PREFILL attention only (decode uses the OMMX kernel). Use whatever bf16-prefill
    # impl is available: flash_attention_2 if installed (fastest, mem-efficient long-ctx),
    # else sdpa (also mem-efficient, no flash_attn dep) — NOT eager (O(ctx^2) OOM at 128K).
    # The resolved choice is RETURNED so it can be printed and written to the JSON: this
    # file's own load_bf16 note explains that an sdpa prefill materialises a mask the
    # flash path does not, so peak_gb is not readable without knowing which one ran.
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
    return m, prefill_attn


def load_bf16():
    # A StaticCache is a fixed-size buffer whose unfilled tail slots are real (zeroed)
    # memory that only a masking-capable backend can exclude: sdpa/eager get a
    # [B,1,q,max_cache_len] mask built from cache_position, whereas flash_attention_2 gets
    # NO mask from transformers and would attend to the zero-filled tail — silently wrong
    # output. So the static path pins sdpa (same choice, same reason, as figure/bench.py).
    # Cost of that pin, and the reason the mode is printed: the sdpa prefill materialises an
    # [1,1,S,S+48] bool mask that ONLY the bf16 arm pays, so under 'static' the bf16 peak_gb
    # carries an attention-backend artifact on top of its KV footprint and is not comparable
    # across arms. tpot_ms is what this option fixes.
    if BF16_CACHE == "static":
        attn = "sdpa"
    else:
        try:
            import flash_attn  # noqa
            attn = "flash_attention_2"
        except Exception:
            attn = "sdpa"
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DTYPE, low_cpu_mem_usage=True, attn_implementation=attn).to(DEV).eval()
    print(f"[bf16] cache={BF16_CACHE} attn_implementation={attn}", flush=True)
    return m, attn


def free(m):
    del m; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def run_arm(name, loader, is_ommx, static_cache=False):
    """Run one arm over CTXS. Returns (cells, prefill_attn) — the loader RESOLVES the
    prefill attention implementation at import time (flash_attn present or not), so the
    caller records what actually ran instead of what the docstring hopes ran."""
    print(f"### loading arm: {name}", flush=True)
    model, prefill_attn = loader()
    out = {}
    for ctx in CTXS:
        try:
            med, p99, mx, peak, cache_class, sc_probe = synced_median_step(
                model, ctx, is_ommx, static_cache=static_cache)
            out[ctx] = {"tpot_ms": med, "p99_ms": p99, "max_ms": mx, "peak_gb": peak,
                        "cache_class": cache_class}
            if sc_probe is not None:
                out[ctx]["static_cache_probe"] = sc_probe
            print(f"[{name}] ctx={ctx:6d}  tpot={med:7.3f} ms  p99={p99:7.3f}  max={mx:7.3f}  "
                  f"peak={peak:6.2f} GB  cache={cache_class}", flush=True)
        except torch.cuda.OutOfMemoryError:
            out[ctx] = {"tpot_ms": None, "oom": True}
            print(f"[{name}] ctx={ctx:6d}  OOM", flush=True); torch.cuda.empty_cache()
        except Exception as e:
            out[ctx] = {"tpot_ms": None, "error": repr(e)[:300]}
            print(f"[{name}] ctx={ctx:6d}  ERROR {repr(e)[:300]}", flush=True); torch.cuda.empty_cache()
    free(model)
    return out, prefill_attn


def _v_bf16_on() -> bool:
    """Resolve OMMX_ATTN_V_BF16 EXACTLY as LlamaAttention._ommx_make_store resolves it in
    _ommx_llama_modeling.py.

    MIRROR, do not improve. The header line below labels the number system this run
    measured; the modeling file is what actually picks the V plane. They used to use
    two different falsy sets -- this printed "bf16 V" for OMMX_ATTN_V_BF16 in
    {"off","no","OFF","False","FALSE"," 0 "} while the store served i2 V -- so the
    published header mislabelled the recipe it was reporting (law #5, measurement side).
    """
    return os.environ.get("OMMX_ATTN_V_BF16", "0").strip().lower() not in {
        "0", "false", "off", "no", ""}


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
          "pow2=" + os.environ.get("OMMX_ATTN_POW2", "0"),
          "map=" + os.environ.get("OMMX_KV_OUTLIER_MAP", "1"),
          "ring=" + os.environ["OMMX_KV_RING"],
          "v_bf16=" + os.environ.get("OMMX_ATTN_V_BF16", "0"),
          flush=True)
    print("bf16 arm cache:", BF16_CACHE, flush=True)
    ommx_res, ommx_attn = run_arm("ommx", load_ommx, True)
    bf16_res, bf16_attn = run_arm("bf16", load_bf16, False,
                                  static_cache=(BF16_CACHE == "static"))
    print("\n" + "=" * 80)
    print(f"Llama-3.1-8B-Instruct  B=1 decode TPOT (OMMX {os.environ['OMMX_ATTN_K_FORMAT']} "
          f"K + {'bf16' if _v_bf16_on() else 'i2'} V, "
          f"npv={os.environ['OMMX_ATTN_OUTLIERS']} gt={os.environ['OMMX_KV_GROUP_TOKENS']} "
          f"pow2={os.environ.get('OMMX_ATTN_POW2', '0')} "
          f"sink={os.environ['OMMX_KV_SINK']} recent={os.environ['OMMX_KV_RECENT']}, "
          f"warmup={WARMUP} measure={MEASURE}, fp16, bf16 cache={BF16_CACHE})")
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
    # ── verdict ─────────────────────────────────────────────────────────────────────
    # Kept, but never as a bare boolean: `any(ratio > 1)` is true as soon as ONE context
    # favours OMMX, and the two things that can produce that without any KV-quantization
    # advantage are (a) the bf16 arm's cache object and (b) per-arm Python/launch overhead
    # at short context. Both are printed on the same line as the verdict.
    wins = [c for c in CTXS
            if (ommx_res.get(c, {}).get("tpot_ms") and bf16_res.get(c, {}).get("tpot_ms")
                and bf16_res[c]["tpot_ms"] / ommx_res[c]["tpot_ms"] > 1.0)]
    win = bool(wins)
    bf16_classes = sorted({r["cache_class"] for r in bf16_res.values()
                           if isinstance(r, dict) and "cache_class" in r})
    print(f"OMMX B=1 TPOT win over bf16 at any ctx: {win}  "
          f"(at ctx {wins or '-'}; bf16 cache={BF16_CACHE} {bf16_classes or '-'})")
    if BF16_CACHE == "dynamic":
        print("  ^ NOT a fair-comparison result: the bf16 arm ran HF's DynamicCache, which "
              "torch.cat's the full KV of every layer on every decode step. Rerun with "
              "OMMX_BENCH_BF16_CACHE=static (the default) before quoting this line.")
    print("  ^ The two arms are two DIFFERENT model implementations (stock transformers vs "
          "the OMMX modeling file), each with its own Python/launch overhead. At short "
          "context that overhead dominates the step time, so a short-ctx win is a scaffold "
          "artifact, not a KV result; only the long-context trend is a KV statement.")
    with open("_ommx_hf_tpot_result.json", "w") as fh:
        json.dump({"ommx": ommx_res, "bf16": bf16_res, "model": MODEL,
                   # Measurement conditions, not just the recipe: the header line above
                   # prints dtype/warmup/measure, the JSON did not, and a reader of the
                   # file alone could not tell fp16 from bf16 or see that the window is
                   # shorter than the OMMX regroup period (see "caveat" below).
                   "dtype": str(DTYPE).replace("torch.", ""),
                   "warmup": WARMUP, "measure": MEASURE,
                   # RESOLVED per arm at load time (flash_attn installed or not), NOT a
                   # constant. It selects the prefill and part of peak_gb: only the sdpa
                   # path materialises the [1,1,S,S+48] mask described in load_bf16.
                   "prefill_attn": {"ommx": ommx_attn, "bf16": bf16_attn},
                   "k_format": os.environ["OMMX_ATTN_K_FORMAT"],
                   "outliers": os.environ["OMMX_ATTN_OUTLIERS"],
                   "pow2": os.environ.get("OMMX_ATTN_POW2", "0"),
                   "group_tokens": os.environ["OMMX_KV_GROUP_TOKENS"],
                   "group_channels": os.environ["OMMX_KV_GROUP_CHANNELS"],
                   "sink": os.environ["OMMX_KV_SINK"], "recent": os.environ["OMMX_KV_RECENT"],
                   "ring": os.environ["OMMX_KV_RING"],
                   "v_bf16": os.environ.get("OMMX_ATTN_V_BF16", "0"),
                   "bf16_cache": BF16_CACHE,          # REQUESTED mode
                   # OBSERVED: the cache class each bf16 cell actually fed, collected from
                   # the timed cells themselves. A run whose bf16 cells all OOM'd or raised
                   # (a StaticCache the installed transformers cannot build, say) leaves this
                   # EMPTY, which is what distinguishes it from a run that really ran static.
                   "bf16_cache_class_observed": bf16_classes,
                   "win_ctxs": wins,
                   # the caveat travels with the data: a reader of the JSON never sees this
                   # file's docstring, and both facts below change what the ratio means.
                   "caveat": (
                       "bf16 arm cache = " + BF16_CACHE + " (dynamic = HF DynamicCache, a "
                       "full-KV torch.cat per layer per decode step, which inflates the "
                       "bf16 baseline; static = preallocated transformers StaticCache, "
                       "which pins sdpa and makes bf16 peak_gb carry an sdpa prefill mask). "
                       "The two arms are different model implementations, so short-context "
                       "deltas are scaffold artifacts, not KV results. tpot_ms is a median "
                       "and hides the OMMX regroup step, which recurs every "
                       "OMMX_KV_GROUP_TOKENS decode steps - see max_ms.")},
                  fh, indent=2)
    print("OMMX_HF_TPOT_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
