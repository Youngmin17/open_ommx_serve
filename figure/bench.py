"""Unified B=1 decode bench: TPOT + TTFT + quality for one method (HF-eager).

Methods: bf16 | kivi | kitty | ommx  — each in its own venv, its own real low-bit-KV kernel.

READ THIS BEFORE QUOTING ANY NUMBER FROM THIS BENCH — the four arms are four DIFFERENT
model implementations, NOT one engine with four KV backends:
  bf16  = stock transformers AutoModelForCausalLM             (upstream modeling_llama)
  kivi  = baseline/kivi/models/llama_kivi_eval.py             (~1.3k-line private model)
  kitty = baseline/kitty/_kitty_llama_modeling.py             (~0.5k-line private model)
  ommx  = ommx_gpu_serve/hf_eager/_ommx_llama_modeling.py     (~0.7k-line private model)
Each carries its own Python-side per-layer/per-step overhead, so any cross-arm delta mixes
the KV kernel with the scaffold around it. At SHORT context the scaffold DOMINATES: an 8B
model at the HBM roofline is a few ms/step, yet every arm here measures TENS of ms/step at
ctx=1024 (measured on a clean H100 NVL, B=1: OMMX 35.5 ms/step, and the bf16 arm 24.5 ms with
--bf16-cache static vs 75.6 ms with dynamic). Most of that is Python + kernel-launch overhead
and has nothing to do with the KV kernel, so a short-ctx cross-arm delta is a SCAFFOLD
artifact, not a KV result — and it can point either way, since whose scaffold is cheapest is
not a property of anyone's KV format. Only the trend across long contexts is a KV statement.
Second caveat: the median hides periodic spikes. OMMX regroups every OMMX_KV_GROUP_TOKENS
steps; measured over 240 steps on a clean H100 NVL those boundary steps cost 86-140 ms against
a 34-35 ms steady state (2.5-4.1x) and about +17% amortized over the run, so mean_ms / max_ms /
min_ms / the raw times_ms are reported next to the median and a quoted TPOT that ignores them
overstates the arm. The default --measure 40 spans only ONE boundary (measured at step index
23; the next is at 55), which is why a p99 read off the default window looks modest.

Same timing discipline for every method (CUDA events, warmup discard, median + mean +
interpolated p99 + max + raw per-step times), so the OMMX vs KIVI / Kitty / bf16 numbers are
comparable to the extent the two caveats above allow:
  TTFT = prefill latency of `ctx` tokens.
  TPOT = median 1-token/step decode latency (mean_ms/max_ms/times_ms expose the tail).
  quality = greedy generation from a real prompt (coherence sanity, not a metric).

The bf16 arm defaults to --bf16-cache static: a transformers StaticCache preallocated to
ctx+warmup+measure+8. With the DynamicCache default it instead torch.cat's the whole KV of
every layer on every decode step — an O(ctx) per-step cost that none of the quantized arms
pay (OMMX's cache object stores no tensors at all), i.e. the old bf16 baseline was inflated
by its cache scaffold on top of the Python overhead above. --bf16-cache dynamic restores the
old behaviour exactly, for reproducing the previously published numbers.

Loaders are inlined (proven recipe mirrors baseline/kivi/_kivi_tpot_bench.py,
baseline/kitty/_kitty_tpot_bench.py, ommx_gpu_serve/hf_eager/_ommx_hf_tpot_bench.py)
so a single --model flows to every arm. Each method runs in ITS OWN venv (run.sh),
so imports are lazy inside the method branch.

Run (the --out NAME matters: figure/collect.py keys on it through an exact FILE2KEY map, so
an HF arm must be written as figure/data_<tag>/<method>_hf.json -- which is what run.sh does.
figure/data/ is collect.py's OUTPUT directory, not an input one):
  CUDA_VISIBLE_DEVICES=<g> python figure/bench.py --method ommx \
      --model meta-llama/Llama-3.1-8B-Instruct --ctxs 1024,4096,16384,32768 \
      --out figure/data_h200/ommx_hf.json
"""
import argparse
import gc
import json
import math
import os
import statistics
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_QUALITY_PROMPT = ("Question: What is the capital of France, and why is it "
                          "historically important?\nAnswer:")
DTYPE_SHORT = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
# Emitted to stdout at startup AND stored in every output JSON, so the caveat travels with the
# data instead of living only in this docstring (a plot reads the JSON, not the source).
CAVEAT = (
    "the four arms are four DIFFERENT model implementations (stock transformers / kivi / "
    "kitty / ommx modeling files), not one engine with four KV backends, so each arm carries "
    "its own Python/launch overhead. At short context that overhead dominates the step time "
    "(an 8B model at the HBM roofline is a few ms/step, while every arm here measures tens of "
    "ms/step at ctx=1024), so short-context cross-arm deltas are scaffold artifacts that can "
    "point either way, not KV results; only the long-context trend is a KV statement. tpot_ms "
    "is the median and hides periodic spikes: OMMX regroups every OMMX_KV_GROUP_TOKENS steps, "
    "and on a clean H100 NVL those boundary steps measured 86-140 ms against a 34-35 ms steady "
    "state (2.5-4.1x, about +17% amortized over 240 steps) - see mean_ms / max_ms / min_ms / "
    "times_ms. The default measure=40 window spans only ONE boundary.")
# The OMMX best recipe env, single-sourced so load_model() and the provenance block cannot
# drift. This dict IS the published recipe for the HF-eager arms: it is mirrored verbatim by
# eval/lm_eval/models/ommx_hf_model.py (the accuracy arm, so the CoQA/PPL numbers and the TPOT
# numbers come from the same recipe), and the footprint it implies is pinned as literals by
# ommx_gpu_serve/tests/test_bit_accounting.py (K 6.000 / V 2.750 / 8.750 bit-per-elem total,
# 3.657x vs bf16), which fails loudly if any value below moves. Do not change these without
# moving that test.
OMMX_RECIPE_ENV = {
    "OMMX_ATTN_K_FORMAT": "i2f4", "OMMX_ATTN_OUTLIERS": "6", "OMMX_ATTN_POW2": "1",
    "OMMX_ATTN_OUTLIER_SELECT": "signed", "OMMX_ATTN_OUTLIER_REPR": "relidx7",
    "OMMX_KV_GROUP_TOKENS": "32", "OMMX_KV_GROUP_CHANNELS": "32",
    "OMMX_KV_SINK": "8", "OMMX_KV_RECENT": "32",
    "OMMX_KV_RING": "1", "OMMX_KV_GPU_PACK": "1",
}
# Filled by _install_ommx_recipe_env(); emitted verbatim into the output JSON. Module
# scope because the install happens at load_model() time and the JSON is written at the
# end of main(), and a provenance record reconstructed later from os.environ cannot tell
# WHICH source put a value there.
OMMX_RECIPE_PROVENANCE = {}


def _ommx_provenance():
    """WHICH RECIPE PRODUCED THIS JSON — the ``ommx`` arm's provenance fields.

    A separate function, not four lines inline in ``main``, so it is reachable from a
    test without a GPU and a model: the provenance defect this closes is exactly the
    kind that only shows up in the finished artifact, and a gate that could not execute
    the code that builds the artifact would be decoration.

    ``ommx_recipe`` keeps its old shape (the flat resolved value of every
    ``OMMX_RECIPE_ENV`` key) so existing readers still parse. ``ommx_recipe_provenance``
    adds what the old block could not say:

      * the preset NAME (the old block recorded none, so a paper-kv run's JSON could not
        say which recipe produced it, and its eleven values would have looked exactly
        like a shipped-kv run);
      * the SOURCE of every knob — operator env / preset / this file's default — which
        cannot be recovered after the fact, because ``os.environ`` remembers the value
        and not who wrote it. Hence the capture at resolution time in
        ``_install_ommx_recipe_env``;
      * the knobs a preset sets that ``OMMX_RECIPE_ENV`` does not name at all
        (``OMMX_KV_OUTLIER_MAP``), which the eleven-key record simply could not hold.
    """
    return {"ommx_recipe": {k: os.environ.get(k) for k in OMMX_RECIPE_ENV},
            "ommx_recipe_provenance": dict(OMMX_RECIPE_PROVENANCE)}



def _install_ommx_recipe_env():
    """Resolve the OMMX KV recipe for this run and record where every knob came from.

    PRECEDENCE — this ORDER is the fix, not a tidy-up. It used to be:

        for k, v in OMMX_RECIPE_ENV.items(): os.environ.setdefault(k, v)

    and nothing else, running at model-build time, i.e. BEFORE anything expanded
    OMMX_RECIPE. setdefault means first writer wins, so this dict's eleven keys were
    already in os.environ by the time resolve_serving_config (the only expander) ran,
    and the preset's pending set was empty. `OMMX_RECIPE=paper-kv python3
    figure/bench.py --method ommx` therefore benchmarked shipped-kv (gt=32, k=6,
    4.375 avg bit) while labelling itself paper-kv (2.75) — the bench's own built-in
    defaults outranked the named preset. Correct order, and what is implemented here:

        explicit operator env  >  named preset (OMMX_RECIPE)  >  this dict

    Both later steps are setdefault-shaped (recipes.resolve_env only fills ABSENT
    keys), so each strictly weaker source fills only what the stronger ones left.

    PROVENANCE — the second half of the same defect. The old JSON recorded
    ``{k: os.environ.get(k) for k in OMMX_RECIPE_ENV}``: eleven keys, no preset name,
    and no OMMX_KV_OUTLIER_MAP (which the presets DO set and this dict does not). A
    paper-kv run produced a JSON that could not say which recipe produced it, and whose
    eleven recorded values would have looked exactly like a shipped-kv run. So record
    the NAME, the per-knob SOURCE, and the union of both key sets.
    """
    if REPO not in sys.path:                    # run as `python figure/bench.py`
        sys.path.insert(0, REPO)
    from ommx_gpu_serve.recipes import pending_env, preset_env, resolve_env

    # Snapshot BEFORE anything is materialised: these are the operator's own exports,
    # the only source that outranks everything below.
    keys = sorted(set(OMMX_RECIPE_ENV) | set(preset_env(
        os.environ.get("OMMX_RECIPE", "").strip() or None)))
    before = dict(os.environ)
    operator = {k: before[k] for k in keys
                if before.get(k) is not None and str(before[k]).strip() != ""}

    # 1) named preset. Unknown name RAISES (listing the known ones) and aborts the run
    #    before a single model byte is loaded — a bench that silently fell back to the
    #    defaults would publish a number under a recipe nobody selected. pending_env
    #    against `before` is EXACTLY the key set resolve_env is about to write.
    applied = resolve_env()
    from_preset = sorted(pending_env(applied, before)) if applied else []

    # 2) this file's own defaults, last, filling only what is still absent.
    from_bench = [k for k in OMMX_RECIPE_ENV if os.environ.get(k) is None
                  or str(os.environ.get(k)).strip() == ""]
    for k, v in OMMX_RECIPE_ENV.items():
        os.environ.setdefault(k, v)

    OMMX_RECIPE_PROVENANCE.clear()
    OMMX_RECIPE_PROVENANCE.update({
        # None when OMMX_RECIPE was unset — the honest answer, not an invented name.
        "preset": applied,
        "preset_env_requested": os.environ.get("OMMX_RECIPE"),
        "source": {**{k: "operator-env" for k in operator},
                   **{k: f"preset:{applied}" for k in from_preset},
                   **{k: "bench-default" for k in from_bench}},
        "resolved": {k: os.environ.get(k) for k in keys},
    })
    return OMMX_RECIPE_PROVENANCE
# Kitty ships as an external package (KITTY_PKG_PATH), so its knobs live wherever that package
# put them; record the ones that are present rather than inventing a schema (present-only, so a
# missing knob shows up as absent instead of as a fabricated null).
KITTY_CFG_KEYS = ("k_bits", "v_bits", "group_size", "residual_length", "n_bits", "sink",
                  "recent", "boost_channels", "quant_config", "kitty_config",
                  "_attn_implementation")


class BenchConfigError(RuntimeError):
    """A configuration/API mismatch that must abort the run instead of being recorded as a
    per-ctx {"error": ...} cell: continuing would publish a number produced by a path other
    than the one the flags asked for."""


def _pick_prefill_attn():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _percentile(asc, q):
    """Linear-interpolated percentile of an ASCENDING sample (numpy's default 'linear' method).

    The previous p99 was `times[int(0.99 * len(times))]`, which for the shipped n=40 evaluates
    to index 39 — the maximum — and reported it as the 99th percentile. With 40 samples p99
    genuinely sits between the 39th and 40th order statistic, so interpolate; max_ms is
    reported separately for the actual worst step."""
    if not asc:
        raise ValueError("percentile of an empty sample")
    if len(asc) == 1:
        return asc[0]
    pos = (len(asc) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(asc) - 1)
    return asc[lo] + (asc[hi] - asc[lo]) * (pos - lo)


def _jsonable(v):
    """JSON-safe form of a provenance value of unknown type. The kitty/kivi knobs are read off
    objects owned by external packages, which are free to put a nested config object on the
    config; repr() those instead of letting json.dump raise at the very end and throw away a
    multi-hour run's results."""
    return v if v is None or isinstance(v, (bool, int, float, str)) else repr(v)[:200]


def _scalar_attrs(obj, limit=48):
    """Public scalar attributes of a cache object — enough to describe the recipe it was built
    with in the JSON, without retaining a reference to (or trying to serialise) a GPU tensor."""
    d = getattr(obj, "__dict__", None)
    if not isinstance(d, dict):
        return {"note": f"{type(obj).__name__} exposes no __dict__"}
    out = {"class": type(obj).__name__}
    for k, v in sorted(d.items(), key=lambda kv: kv[0]):
        if k.startswith("_") or not isinstance(v, (bool, int, float, str)):
            continue
        out[k] = v
        if len(out) >= limit:
            break
    return out


def _make_static_cache(model_obj, max_len, dtype, probe_out=None):
    """A transformers StaticCache preallocated to `max_len`, or a loud raise — never a fallback.

    The StaticCache constructor has been renamed across transformers releases (max_batch_size
    <-> batch_size; device/dtype added, then made optional) and this bench runs under three
    different pinned transformers versions (see the pyproject extras), so the kwargs are
    filtered against the live signature and tried newest-first. If every candidate fails we
    RAISE naming the version and each attempt: quietly falling back to a DynamicCache here
    would restore the per-step full-KV torch.cat that this option exists to remove, while the
    JSON still claimed "static" — i.e. it would reintroduce exactly the bug being fixed."""
    import inspect
    import transformers
    ver = getattr(transformers, "__version__", "?")
    try:
        from transformers import StaticCache
    except Exception as e:  # noqa: BLE001  — re-raised immediately, nothing is swallowed
        raise BenchConfigError(
            f"transformers {ver} exposes no importable StaticCache ({e!r}). Fix: use a "
            f"transformers >= 4.38 for the bf16 venv, or rerun this arm with "
            f"--bf16-cache dynamic (explicitly accepting the per-step full-KV torch.cat)") from e

    dev = next(model_obj.parameters()).device
    full = {"config": model_obj.config, "max_cache_len": int(max_len), "max_batch_size": 1,
            "batch_size": 1, "device": dev, "dtype": dtype}
    candidates = [  # newest-first; each entry is a set of kwarg NAMES taken from `full`
        ("config", "max_cache_len", "max_batch_size", "device", "dtype"),
        ("config", "max_cache_len", "batch_size", "device", "dtype"),
        ("config", "max_cache_len", "max_batch_size", "dtype"),
        ("config", "max_cache_len", "dtype"),
        ("config", "max_cache_len"),
    ]
    try:
        params = inspect.signature(StaticCache.__init__).parameters
        var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        names = set(params)
    except (TypeError, ValueError):  # C-implemented / unintrospectable: try the kwargs as-is
        var_kw, names = True, set(full)

    cache, used, tried, seen = None, None, [], set()
    for cand in candidates:
        kw = {k: full[k] for k in cand if var_kw or k in names}
        key = tuple(sorted(kw))
        if "config" not in kw or "max_cache_len" not in kw or key in seen:
            continue
        seen.add(key)
        try:
            cache, used = StaticCache(**kw), key
            break
        except Exception as e:  # noqa: BLE001  — every failure is recorded and re-raised below
            tried.append(f"{key} -> {type(e).__name__}: {e}"[:240])
    if cache is None:
        why = " | ".join(tried) or (f"signature {sorted(names)} accepts no (config, "
                                    f"max_cache_len) pair")
        raise BenchConfigError(
            f"could not construct transformers.StaticCache under transformers {ver}: {why}. "
            f"Fix: pin a transformers whose StaticCache takes (config, max_cache_len), or "
            f"rerun this arm with --bf16-cache dynamic (explicitly accepting the per-step "
            f"full-KV torch.cat that the static cache exists to remove).")

    # A version that silently ignored max_cache_len would overrun mid-decode, so read the size
    # back. get_max_cache_shape() is named for a SHAPE and some versions return the whole
    # [B, H, max_cache_len, D] tuple rather than the int, so normalise to the largest int in it:
    # the check below only ever RAISES on `< max_len`, so taking the max can never fire falsely,
    # while an undersized (1, 8, 128, 128) against max_len=1080 still trips it. (Before this
    # normalisation an `isinstance(got, int)` test skipped the guard entirely for a tuple - the
    # guard was bypassable by exactly the return shape it was written to inspect.)
    got, probed = None, None
    for probe in ("get_max_cache_shape", "get_max_length"):
        fn = getattr(cache, probe, None)
        if callable(fn):
            try:
                got = fn()
            except Exception:  # noqa: BLE001  — deprecated probe; try the next one
                got = None
            if got is not None:
                probed = probe
                break
    seen_len = got
    if isinstance(got, (tuple, list)):
        ints = [x for x in got if isinstance(x, int) and not isinstance(x, bool)]
        # A StaticCache shape is [B, num_kv_heads, max_cache_len, head_dim], so the length lives
        # on axis -2. Taking max(ints) instead is a FALSE-NEGATIVE hole, not a safe over-estimate:
        # head_dim (128 for Llama-3.1) masks any max_cache_len < max_len <= 128, i.e. a short run
        # such as `--ctxs 64` would report size_verified=True on a cache too short to decode into
        # ((1, 8, 64, 128) vs max_len=100 passes under max(), fails under axis -2). Fall back to
        # max() only for a shape we cannot index as [B, H, L, D].
        seen_len = (ints[-2] if len(ints) >= 4 else (max(ints) if ints else None))
    if isinstance(seen_len, int) and not isinstance(seen_len, bool) and seen_len < max_len:
        raise BenchConfigError(
            f"transformers {ver} StaticCache ignored max_cache_len: asked {max_len}, got {got} "
            f"(kwargs {used}); the decode loop would run past the buffer. Fix: pin a different "
            f"transformers, or rerun with --bf16-cache dynamic.")
    # If no probe answered, the preallocated size is UNVERIFIED. We do not raise (a transformers
    # whose probes are all deprecated is not itself evidence of a wrong size) but the fact must
    # reach the JSON, not just stdout: an unverified cell reads exactly like a verified one.
    verified = isinstance(seen_len, int) and not isinstance(seen_len, bool)
    note = {"transformers": ver, "kwargs": list(used), "asked_max_cache_len": int(max_len),
            "probe": probed, "reported": _jsonable(got), "size_verified": bool(verified)}
    if isinstance(probe_out, dict):
        probe_out["static_cache_probe"] = note
    print(f"[bf16] StaticCache(kwargs={used}) max_cache_len={max_len} reports={got} "
          f"size_verified={verified} transformers={ver}"
          + ("" if verified else "  WARNING: size UNVERIFIED (no usable probe on this cache)"),
          flush=True)
    return cache


def _kivi_path_stats():
    """Which KIVI decode path actually ran — recorded, never assumed.

    baseline/kivi/models/llama_kivi_eval.py gates its fused dequant+matmul kernel on
    KIVI_FUSED_GQA and counts firings in _kivi_fused_fires; without it the arm silently takes
    the dequantize-to-fp16-then-dense fallback, whose latency is NOT a KIVI-kernel latency. The
    public accessor is kivi_fused_stats(); if it is missing we still record the private counter
    plus the import error rather than omitting the field (an absent field reads as "fused")."""
    try:
        from kivi.models.llama_kivi_eval import kivi_fused_stats
        return kivi_fused_stats()
    except Exception as e:  # noqa: BLE001  — recorded into the JSON, not swallowed
        rec = {"error": repr(e)[:300]}
        try:
            from kivi.models import llama_kivi_eval as _k
            rec["raw_probe"] = {"fused_fires": int(_k._kivi_fused_fires[0]),
                                "fused_gqa_enabled": bool(_k._fused_gqa_enabled()),
                                "KIVI_FUSED_GQA": os.environ.get("KIVI_FUSED_GQA")}
        except Exception as e2:  # noqa: BLE001
            rec["raw_probe_error"] = repr(e2)[:200]
        return rec


# ── per-method loaders (lazy imports) ─────────────────────────────────────────
def load_model(method, model, dtype, bf16_cache="dynamic"):
    """Return (model_obj, kind, extra) where kind in {'past','static','kitty'}.

    `kind` tells measure() who owns the KV cache:
      'past'   — the model's own cache, re-read from out.past_key_values every step (DynamicCache
                 for the HF arms: a full-KV torch.cat per layer per step),
      'static' — a transformers StaticCache we preallocate (bf16 arm, --bf16-cache static),
      'kitty'  — kitty's own preallocated cache object.
    """
    import torch
    from transformers import LlamaConfig, AutoModelForCausalLM

    if method == "bf16":
        # A StaticCache is a fixed-size buffer whose unfilled tail slots are real (zeroed)
        # memory that only a masking-capable backend can exclude: sdpa/eager get a
        # [B,1,q,max_cache_len] mask built from cache_position (transformers refuses to skip
        # the mask while a compileable/static cache is in use), whereas flash_attention_2 gets
        # NO mask from transformers and would attend to the zero-filled tail — silently wrong
        # output at a wrongly-max_cache_len-sized cost. So the static path pins sdpa; the
        # choice is printed and recorded in the JSON as meta.attn_impl, never swapped quietly.
        # Cost of that pin: at ctx >= 32768 the sdpa prefill materialises an [1,1,S,S+48] mask
        # (~4.3 GB of bool at 64K) and can OOM on a small card — that surfaces as a recorded
        # OOM cell, and --bf16-cache dynamic restores the FA2 + DynamicCache behaviour.
        # Second cost, and the reason meta.attn_impl is recorded: this makes the bf16 arm's
        # TTFT an sdpa prefill while the kivi/kitty/ommx arms still take FA2 where it is
        # installed, so under --bf16-cache static the bf16 TTFT column is NOT comparable across
        # arms. Nor, strictly, is peak_gb: that sdpa prefill mask is a real allocation of
        # ctx * (ctx + warmup + measure + 8) bools that ONLY the bf16 arm pays (~1.0 GiB at
        # ctx=32768, ~4.0 GiB at 65536), so at long ctx the bf16 peak_gb carries an
        # attention-backend artifact on top of its KV footprint. tpot_ms is what this option
        # actually fixes. Compare TTFT/peak_gb only between runs whose meta.attn_impl agree, or
        # rerun the bf16 arm with --bf16-cache dynamic.
        attn = "sdpa" if bf16_cache == "static" else _pick_prefill_attn()
        m = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation=attn).cuda().eval()
        if bf16_cache == "static":
            print(f"[bf16] --bf16-cache static -> attn_implementation={attn} "
                  f"(FA2 cannot mask a preallocated cache's unfilled tail)", flush=True)
            return m, "static", {"dtype": dtype}
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
        # OMMX recipe resolution: explicit operator env > OMMX_RECIPE preset >
        # OMMX_RECIPE_ENV above (INT2 base + 12/64 FP4 K-outliers, sink 8, pow2 scales;
        # serving maps 12/64 -> 6 outliers per 32-channel group, which is what
        # OMMX_ATTN_OUTLIERS=6 with OMMX_KV_GROUP_CHANNELS=32 spells). The middle term
        # is the fix — see _install_ommx_recipe_env for the inversion it replaces.
        _install_ommx_recipe_env()
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
    """Prefill ctx (time TTFT) then 1-tok/step decode (median TPOT). B=1.

    Returns median + mean + interpolated p99 + max + the raw per-step times IN ISSUE ORDER:
    the quantized arms regroup/repack periodically (OMMX every OMMX_KV_GROUP_TOKENS steps), and
    a median by construction reports only the cheap steps of that cycle. times_ms is ~40 floats,
    so shipping it costs nothing and lets a reader recompute any statistic."""
    import torch
    if method == "ommx":  # size the per-layer store to this ctx (not the 131072 max)
        os.environ["OMMX_ATTN_MAXCTX"] = str(ctx + warmup + measure_n + 8)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    vocab = model_obj.config.vocab_size
    torch.manual_seed(42)
    ids = torch.randint(10, vocab - 10, (1, ctx), device="cuda", dtype=torch.long)

    # Caches we own (kitty / bf16-static) are preallocated to the whole run length here, i.e.
    # inside reset_peak_memory_stats(), so peak_gb counts the preallocation rather than the
    # DynamicCache's grow-by-cat high-water mark. Prefill stays a single call for every arm so
    # the [1, ctx, vocab] prefill-logits transient (which dominates peak_gb at long ctx) is
    # identical across arms and peak_gb stays comparable.
    # "dynamic" names a transformers DynamicCache and is only true for the bf16 arm: the ommx arm
    # carries an OmmxSeqCounterCache (a DynamicCache subclass that stores NO tensors) and the kivi
    # arm a legacy quantized tuple, so labelling either "dynamic" would be a fresh mislabel of the
    # same kind this bench is being corrected for. cache_class below records the real class name.
    kv, cache_used = None, ("dynamic" if method == "bf16" else "model-owned")
    if kind == "kitty":
        kv = extra["get_kv"](extra["cfg"], max_batch_size=1,
                             max_length=ctx + warmup + measure_n + 8)
        cache_used = "kitty"
        extra["kitty_cache_probe"] = _scalar_attrs(kv)  # provenance: what kitty actually built
    elif kind == "static":
        kv = _make_static_cache(model_obj, ctx + warmup + measure_n + 8, extra["dtype"],
                                probe_out=extra)  # provenance: was the size actually verified
        cache_used = "static"

    # A StaticCache is fixed-size, so the model cannot infer the write offset from the tensor
    # shape: pass cache_position explicitly. It also fixes the RoPE position_ids and avoids
    # get_seq_length()'s per-step device sync. Built outside the timed window on purpose.
    def _cpos(start, n):
        return torch.arange(start, start + n, device="cuda", dtype=torch.long) \
            if kind == "static" else None

    se = torch.cuda.Event(enable_timing=True); ee = torch.cuda.Event(enable_timing=True)
    kw = {} if kv is None else {"past_key_values": kv}
    cp = _cpos(0, ctx)
    if cp is not None:
        kw["cache_position"] = cp
    torch.cuda.synchronize(); se.record()
    with torch.no_grad():
        out = model_obj(input_ids=ids, use_cache=True, **kw)
    ee.record(); torch.cuda.synchronize()
    ttft_ms = se.elapsed_time(ee)

    past = kv if kv is not None else out.past_key_values
    cache_class = type(past).__name__  # the object the decode loop actually feeds, by name
    next_tok = out.logits[:, -1:, :].argmax(-1)
    del out
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for step in range(warmup + measure_n):
            cp = _cpos(ctx + step, 1)  # outside the timed window
            kw = {} if cp is None else {"cache_position": cp}
            torch.cuda.synchronize(); se.record()
            o = model_obj(input_ids=next_tok, past_key_values=past, use_cache=True, **kw)
            ee.record(); torch.cuda.synchronize()
            ms = se.elapsed_time(ee)
            if kv is None:
                past = o.past_key_values
            next_tok = o.logits[:, -1:, :].argmax(-1)
            del o
            if step >= warmup:
                times.append(ms)
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    if not times:
        raise BenchConfigError(f"--measure {measure_n} produced no timed steps (need >= 1)")
    asc = sorted(times)
    return {"ttft_ms": round(ttft_ms, 4),
            "tpot_ms": round(statistics.median(asc), 4),      # median, as before
            "mean_ms": round(sum(times) / len(times), 4),     # includes the periodic spikes
            "p99_ms": round(_percentile(asc, 0.99), 4),       # interpolated, not just the max
            "max_ms": round(asc[-1], 4),
            # min_ms: the co-tenant tell. On a GPU shared with another process the fast steps
            # stay fast and the contended ones stretch, so min_ms falls toward tpot_ms/2 while
            # mean_ms pulls away from the median; on a clean run both sit close to it. Cheap to
            # record, and repro/README.md's "do not share the GPU" check reads it directly.
            "min_ms": round(asc[0], 4),
            "peak_gb": round(peak_gb, 4),
            "cache": cache_used,
            "cache_class": cache_class,
            "times_ms": [round(t, 4) for t in times]}         # issue order: tail is auditable


def quality_gen(method, model_obj, kind, extra, tokenizer, prompt, n=96):
    """Greedy generation from a real prompt -> text (coherence sanity).

    Untimed, so the bf16 arm runs it on the model's own default cache even under
    --bf16-cache static: this checks that the arm produces sane text, not how fast it does."""
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


def _build_parser():
    ap = argparse.ArgumentParser(
        description="B=1 decode TPOT/TTFT for ONE arm. The arms are four different model "
                    "implementations; at short context the step time is framework overhead, "
                    "not KV work (see the module docstring).")
    ap.add_argument("--method", required=True, choices=["bf16", "kivi", "kitty", "ommx"])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--ctxs", default=os.environ.get("BENCH_CTXS", "1024,4096,16384,32768,65536"))
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--measure", type=int, default=40)
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPE_SHORT),
                    help="compute dtype for the HF arms (default: bfloat16). It used to default "
                         "to float16 while the arm was labelled - and plotted - as 'bf16' next "
                         "to bfloat16 vLLM arms; the resolved dtype is now echoed in the output "
                         "JSON as arm_label so a plot cannot mislabel it.")
    ap.add_argument("--bf16-cache", default="static", choices=["static", "dynamic"],
                    help="KV cache for the bf16 arm (default: static). 'static' = transformers "
                         "StaticCache preallocated to ctx+warmup+measure+8, which pins sdpa (a "
                         "preallocated cache's unfilled tail needs a mask FA2 does not get) and "
                         "at ctx>=32768 makes prefill materialise an [1,1,S,S+48] mask. "
                         "'dynamic' = DynamicCache, i.e. a full-KV torch.cat per layer per "
                         "step that none of the quantized arms pay - the pre-fix behaviour, "
                         "kept for reproducing the previously published bf16 numbers.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--quality-prompt", default=DEFAULT_QUALITY_PROMPT)
    return ap


def main():
    args = _build_parser().parse_args()
    if args.measure < 1 or args.warmup < 0:
        raise BenchConfigError("--measure must be >= 1 and --warmup >= 0 (the statistics need "
                               "at least one timed step)")

    import torch
    # EVERY arm here is a GPU decode measurement (peak_gb, cuda events, the quantized KV
    # kernels), and the first thing main() does with torch is torch.cuda.get_device_name(0)
    # inside a print(). On a CPU-only host that raised a bare
    # "AssertionError: Torch not compiled with CUDA enabled" from deep inside torch, several
    # frames below anything the operator wrote -- and this is run.sh's MAIN bench entry
    # point, so it is the first traceback a new user sees. Say what is wrong instead.
    if not torch.cuda.is_available():
        raise SystemExit(
            f"figure/bench.py needs a CUDA GPU (torch {torch.__version__} reports "
            f"torch.cuda.is_available() == False).\n"
            "  Every method here is a GPU decode measurement -- there is no CPU fallback,\n"
            "  and a CPU number would not be comparable to anything published.\n"
            "  * on a GPU host: check CUDA_VISIBLE_DEVICES is not empty and that this\n"
            "    interpreter has a CUDA build of torch (a CPU wheel reports False even on\n"
            "    a machine with a GPU).\n"
            "  * on a CPU host: nothing in figure/bench.py can run; figure/collect.py and\n"
            "    figure/plot.py DO run on CPU over already-measured figure/data_<tag>/*.json."
        )
    from transformers import AutoTokenizer
    dtype = getattr(torch, args.dtype, None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise BenchConfigError(f"--dtype {args.dtype!r} does not resolve to a floating-point "
                               f"torch dtype (got {dtype!r})")
    # arm_label carries the RESOLVED dtype into the JSON, so the arm a plot draws as "bf16"
    # cannot be an fp16 run (the mislabel this field exists to make impossible).
    dt_short = DTYPE_SHORT.get(args.dtype, args.dtype)
    arm_label = dt_short if args.method == "bf16" else f"{args.method} ({dt_short})"
    # The dtype default moved float16 -> bfloat16 so the arm plotted as "bf16" really is bf16
    # (and matches the bfloat16 vLLM arms in the same figure). That applies to every HF arm, and
    # KIVI's packer hard-casts to fp16 (baseline/kivi/quant/new_pack.py: `data.to(torch.float16)`),
    # so its dequantized K/V come back fp16 whatever --dtype says. Under bfloat16 that surfaces
    # as a loud dtype mismatch inside the arm rather than as a wrong number — but announce it and
    # record it, because the workaround (--dtype float16) changes what the arm IS, and arm_label
    # must then read "kivi (fp16)" wherever it is plotted.
    dtype_note = None
    if args.method == "kivi" and args.dtype != "float16":
        dtype_note = (f"kivi packs/dequantizes KV as float16 (baseline/kivi/quant/new_pack.py); "
                      f"at --dtype {args.dtype} the arm may raise a dtype mismatch. If it does, "
                      f"rerun this arm with --dtype float16 and plot it as arm_label "
                      f"'kivi (fp16)' — do not relabel it bf16.")
        print(f"WARNING: {dtype_note}", flush=True)
    ctxs = [int(x) for x in args.ctxs.split(",") if x]
    out = args.out or os.path.join(REPO, "figure", "data", f"{args.method}.json")

    print(f"torch {torch.__version__}  dev {torch.cuda.get_device_name(0)}  "
          f"method={args.method}  arm_label={arm_label}", flush=True)
    print(f"CAVEAT: {CAVEAT}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model_obj, kind, extra = load_model(args.method, args.model, dtype,
                                        bf16_cache=args.bf16_cache)

    per_ctx = {}
    for ctx in ctxs:
        try:
            r = measure(args.method, model_obj, kind, extra, ctx, args.warmup, args.measure)
            per_ctx[str(ctx)] = r
            print(f"[{args.method}] ctx={ctx:6d}  ttft={r['ttft_ms']:8.2f}  "
                  f"tpot={r['tpot_ms']:7.3f}  mean={r['mean_ms']:7.3f}  p99={r['p99_ms']:7.3f}  "
                  f"max={r['max_ms']:7.3f}  peak={r['peak_gb']:.2f}GB  cache={r['cache']}",
                  flush=True)
        except BenchConfigError:
            # A cache/API mismatch is fatal: recording it as a per-ctx cell (truncated to 300
            # chars) would leave a JSON that looks like a run with one bad context, when in fact
            # no context can be measured with the flags given.
            raise
        except torch.cuda.OutOfMemoryError:
            per_ctx[str(ctx)] = {"oom": True}
            print(f"[{args.method}] ctx={ctx:6d}  OOM", flush=True); torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            per_ctx[str(ctx)] = {"error": repr(e)[:300]}
            print(f"[{args.method}] ctx={ctx:6d}  ERROR {repr(e)[:300]}", flush=True)
            torch.cuda.empty_cache()

    # Which KIVI kernel actually ran. Captured before quality_gen so the untimed generation does
    # not inflate it, but the counters are process-wide and were never reset, so the counts cover
    # every prefill + warmup + measured step of EVERY ctx in this run: read `path` (fused /
    # dequant-fallback / mixed), not the absolute fire count. (llama_kivi_eval also exposes
    # kivi_reset_fused_stats() if a future per-ctx attribution is wanted.) Never omitted: an
    # absent field would read as "the fused path ran".
    kivi_path = _kivi_path_stats() if args.method == "kivi" else None
    if kivi_path is not None:
        print(f"[kivi] path={kivi_path}", flush=True)

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
    import transformers
    cfg_obj = getattr(model_obj, "config", None)
    prov = {"seed": 42, "git_sha": _git_sha(), "torch": torch.__version__, "triton": triton_v,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0],
            "torch_dtype": str(dtype),  # resolved, not the string the caller typed
            "attn_impl": getattr(cfg_obj, "_attn_implementation", None)}
    if dtype_note:
        prov["dtype_note"] = dtype_note
    # Per-arm recipe: an arm's JSON must say which knobs produced it, because the arms do not
    # share a recipe schema (each is a different model implementation).
    if args.method == "bf16":
        prov["bf16_cache_requested"] = args.bf16_cache
        # From the LAST measured ctx (the caches are per-ctx and identical in kind). Present only
        # when a static cache was actually built; size_verified=false there means the buffer size
        # could not be read back, so that run's bf16 numbers are one unchecked assumption deep.
        if "static_cache_probe" in extra:
            prov["static_cache_probe"] = extra["static_cache_probe"]
    elif args.method == "kivi":  # read off the loaded config = what the model actually holds
        prov["kivi_recipe"] = {k: _jsonable(getattr(cfg_obj, k, None)) for k in
                               ("k_bits", "v_bits", "group_size", "residual_length", "use_flash")}
        prov["kivi_env"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith("KIVI_")}
    elif args.method == "kitty":  # external package: record what is present, do not invent keys
        prov["kitty_recipe"] = {
            "cfg": {k: _jsonable(getattr(cfg_obj, k)) for k in KITTY_CFG_KEYS
                    if hasattr(cfg_obj, k)},
            "cache": extra.get("kitty_cache_probe"),
            "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("KITTY_")}}
    elif args.method == "ommx":
        prov.update(_ommx_provenance())

    doc = {"method": args.method, "arm_label": arm_label, "model": args.model,
           "caveat": CAVEAT, "ctxs": per_ctx, "quality": quality,
           "meta": {"warmup": args.warmup, "measure": args.measure,
                    "dtype": args.dtype, **prov}}
    if args.method == "bf16":  # observed, not requested: an OOM-only run claims nothing
        observed = sorted({v["cache"] for v in per_ctx.values()
                           if isinstance(v, dict) and "cache" in v})
        doc["bf16_cache"] = observed[0] if len(observed) == 1 else (observed or None)
    if kivi_path is not None:
        doc["kivi_path"] = kivi_path
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        # default=str is a backstop only: every field above is already coerced, and losing a
        # finished run to a TypeError in the last statement would be the worst possible failure.
        json.dump(doc, fh, indent=2, default=str)
    del model_obj; gc.collect(); torch.cuda.empty_cache()
    print(f"BENCH_DONE method={args.method} out={out}", flush=True)


if __name__ == "__main__":
    main()
