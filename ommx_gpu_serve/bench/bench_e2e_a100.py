# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""A100 (sm_80) B=1 long-context vLLM E2E comparison + ablation, ONE timing source.

WHY: the headline COMPARISON and the ABLATION (attention / outlier-decode / dequant share
of TTFT & TPOT) must use the SAME robust per-step timing so the deltas are physical. We
reuse the LLMEngine.step() CUDA-event-pair method from bench/ablation_matrix.py (NO 2-length
subtraction, NO prefill contamination) + EQUAL-KV-pool pinning.

WHAT THIS RELEASE CAN ACTUALLY RUN. Both OMMX axes now have a serving path: the
paged-decode ATTENTION backend (AttentionBackendEnum.CUSTOM) and the WEIGHT-quant linear
method (quantization "ommx_w"), both registered by integration/vllm/plugin.py.

The ommx_w arms need a WEIGHT BUNDLE, and there is still no fallback if one is absent:
every arm below with quant="ommx_w" (OMMX, abl_linear, abl_full, abl_full_no_outlier,
abl_full_no_dequant) FAILS FAST with a SystemExit naming the offline packer, because an
arm that degrades into "bf16 weights + something" and is reported as an OMMX measurement
is precisely the defect this bench exists to prevent. Produce a bundle first:
    python -m ommx_gpu_serve.linear.w_packer pack --input <hf ckpt> --output <bundle>
then run those arms with --model <bundle> (or OMMX_W_BUNDLE=<bundle>). Drop them with
--only-arms if you only want the KV story.

UNVERIFIED (no GPU this session): the ommx_w arms have never executed. The linear
method's CPU-verified surface is listed in integration/vllm/linear_method.py; nothing
here has run it against a device, and no ommx_w timing exists anywhere in this repo.

  KV HEADLINE (bf16 weights everywhere; the ONLY variable is the attention backend):
    abl_attn = bf16-w + CUSTOM  (OMMX KV-quant attention)   <- the OMMX bar
    TRITON   = bf16-w + TRITON_ATTN                          <- vLLM Triton attention
    FA       = bf16-w + FLASH_ATTN                           <- vLLM DEFAULT attention
  WEIGHT arms (need an OMMX_W_SafeTensor bundle; hard-fail without one):
    OMMX / abl_linear / abl_full / abl_full_no_outlier / abl_full_no_dequant  (quant=ommx_w)

ARMS (each = its own subprocess so the W/attn/toggle env is captured at vLLM build time and
the GPU memory is clean):

  COMPARISON  (B=1, ctx {1024,4096,16384,65536,131072}):
    OMMX    = ommx_w (i2f4) + CUSTOM attention   (needs a bundle -> else SystemExit)
    FA      = bf16-w + FLASH_ATTN                (vLLM default; selectable, not in the
                                                  default order -- see --only-arms)
    FA3     = bf16-w + FLASH_ATTN                (same backend as FA under its historical
                                                  Hopper-flavoured name; kept for old runs)
    TRITON  = bf16-w + TRITON_ATTN
    NOTE: TRITON is the KV-pool probe and the coherence baseline, so it is NOT "the vLLM
    default". Anything published as "bf16 (vLLM)" must say which of TRITON/FA it is.

  ABLATION  (B=1, ctx {4096,65536}, all FULL-graph):
    bf16    = bf16-w + FLASH_ATTN                 (the floor)
    linear  = ommx_w + FLASH_ATTN                 (needs a bundle -> else SystemExit)
    attn    = bf16-w + CUSTOM                      (delta vs bf16 = OMMX ATTENTION cost)
    full    = ommx_w + CUSTOM                      (needs a bundle -> else SystemExit)
    full_no_outlier = full + OMMX_ABL_NO_OUTLIER   (needs a bundle -> else SystemExit)
    full_no_dequant = full + OMMX_ABL_V_NODEQUANT  (needs a bundle -> else SystemExit)

  ATTENTION BREAKDOWN  (B=1, abl_ctxs, all FULL-graph, bf16-w + CUSTOM; PARITY-BREAKING,
  coh=False — each SKIPs one decode cost so its TPOT delta isolates that share):
    attn                = full attention (the reference for these deltas)
    attn_no_outlier     = attn + NO_OUTLIER                 (skip relidx7 outlier splice)
    attn_no_unpack      = attn + NO_UNPACK                  (skip K/V bit-extract ALU)
    attn_no_dequant     = attn + V_NODEQUANT                (skip V scale/zp dequant)
    attn_no_dequant_all = attn + K&V NODEQUANT              (skip K+V scale/zp dequant)
    attn_no_all         = attn + NO_OUTLIER+K&V NODEQUANT+NO_UNPACK (base math+mem only)
    attn_skipwrite      = attn + OMMX_ABL_SKIP_WRITE        (skip per-step KV pack/write)
  DIFFERENTIAL DECOMPOSITION (TPOT(arm); larger arm => more work kept):
    pack/write share  = TPOT(attn)             - TPOT(attn_skipwrite)
    outlier share     = TPOT(attn)             - TPOT(attn_no_outlier)
    bit-extract share = TPOT(attn)             - TPOT(attn_no_unpack)
    V scale/zp share  = TPOT(attn)             - TPOT(attn_no_dequant)
    K+V scale/zp share= TPOT(attn)             - TPOT(attn_no_dequant_all)
    scale/zp-only ALU = TPOT(attn_no_unpack)   - TPOT(attn_no_dequant_all)
    base attn (math+mem; QK/softmax/PV + KV load) = TPOT(attn_no_all)

FAIR-COMPARE (locked for every arm): dtype=bf16 + seed=42 + prefix_caching=False +
chunked_prefill=False + EQUAL geometry. The bf16 arm at the SMALLEST ctx sets
num_gpu_blocks; every other arm is pinned to it (trap C: a quantized arm frees weight bytes
and would otherwise get a bigger KV pool -> fake TTFT/TPOT win). If the KV-pool probe arm
itself fails to build, there is no value to pin to: the sweep then says so in the run log
and in config.fair.equal_kv_pool instead of asserting a pin that did not happen.

NOT locked for every arm, and therefore RECORDED PER ARM instead of asserted globally:
  * enforce_eager -- and with it torch.compile/inductor fusion, which vLLM's
    ``enforce_eager=True`` ("always execute the model in eager mode") turns off together
    with graph capture. Every arm runs CUDA-graph FULL except `abl_attn_bat_eager`, which
    sets eager=True on purpose (the vLLM multi-size capture ladder crashes the batched
    write path). The result JSON used to hardcode enforce_eager=False / cudagraph="FULL"
    into one global provenance block, so that arm shipped a run condition it never met, and
    the line above used to list inductor fusion as locked for every arm on the same basis.
    results["provenance"][<arm>] now carries what each arm actually ran under, cross-checked
    against the "graph"/"eager" `mode` string the worker process itself wrote per cell.
  * cudagraph_capture_sizes. This file does NOT pin it to {1}; the default is vLLM's own
    capture ladder. `--capture-sizes 1` pins it (B=1 study -> one shape -> fast capture) and
    the resolved value is recorded per arm. A malformed, empty or non-positive value is a
    hard error, never a silent revert to the default.
TPOT = steady-state per-step CUDA-event time (p50+p99 over >=measure steps pooled over
reps). TTFT = the prefill step CUDA-event time. OOM is recorded (capacity story).

ROUTE-FIRED EVIDENCE (law #5: no silent fallback). A CUSTOM arm that quietly fell back to
bf16 FlashAttention still produces perfectly plausible TPOT numbers, so a CUSTOM arm is only
reported when it PROVES the OMMX kernel ran. The backend writes one-time per-rank sentinels
to $OMMX_FIRE_FILE (rank 0 = the base path, rank r>0 = "<base>.r<r><ext>"):
  FIRED    DECODE_ROUTE_FIRED | GRAPH_ROUTE_FIRED | BATCHED_ROUTE_FIRED |
           BATCHED_GRAPH_ROUTE_FIRED            (a real OMMX decode route ran)
           PACKED_PREFILL_FIRED                 (PACKED-ONLY varlen prefill; recorded, but
                                                 it is NOT decode proof on its own)
  FAILED   DECODE_ROUTE_NOFIRE | BATCHED_ROUTE_NOFIRE | KV_UPDATE_DEAD
The orchestrator gives each CUSTOM arm its own <workdir>/<arm>.fire.log, deletes stale
copies BEFORE the run, reads them back after, stores the parsed tags under
results["route_evidence"][arm], and forces every cell of that arm to ok=False (reason in
the "err" column) when no *ROUTE_FIRED tag is present, when a NOFIRE/DEAD tag is present,
or when fewer worker processes fired than --tp. Downstream (e2e_to_figure.py, the CSV) reads
ok, so an unproven arm cannot become a published OMMX bar.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" PYTHONNOUSERSITE=1 \
      TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_80" TORCH_CUDA_ARCH_LIST=8.0 \
      VLLM_PLUGINS=ommx_gpu_serve VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      HF_HUB_OFFLINE=1 \
      python -m ommx_gpu_serve.bench.bench_e2e_a100 --model <llama-3.1-8b-path> \
        --only-arms "TRITON,FA,abl_attn,abl_attn_no_all,abl_attn_no_unpack,abl_attn_no_dequant,abl_attn_skipwrite" \
        --csv runs/e2e_a100.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import os
import subprocess
import sys


# ── percentile helpers ───────────────────────────────────────────────────────
def _pctl(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _median(xs):
    return _pctl(xs, 0.5)


def _free_mem_gib():
    import torch
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


def _gpu_util(headroom_gib=8.0, hard_cap=0.92):
    import torch
    free, total = torch.cuda.mem_get_info()
    total_g = total / (1024 ** 3)
    target = max(0.30, (free / (1024 ** 3) - headroom_gib) / total_g)
    return min(hard_cap, target)


def _parse_capture_sizes(spec):
    """'default' -> None (vLLM's own capture ladder); '1,2,4' -> [1, 2, 4]. Else RAISES.

    Law #5 (no silent fallback). This used to be a bare ``except (ValueError, TypeError):
    pass`` inside _make_engine, so ``--capture-sizes '1;2;4'`` -- or any typo'd token --
    threw the operator's fair-comparison control away without a word and the run continued
    on vLLM's default ladder while claiming the pinned one. Zero and negative sizes are
    refused for the same reason: vLLM does not honour them, so accepting them here would
    again mean "recorded one thing, ran another".
    """
    s = (spec or "").strip()
    if s.lower() in ("", "default"):
        return None
    vals = []
    for tok in (t.strip() for t in s.split(",")):
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except ValueError:
            raise ValueError(
                f"--capture-sizes {spec!r}: {tok!r} is not an integer. Use 'default' for "
                f"vLLM's own capture ladder, or a comma list, e.g. '1' or '1,2,4'."
            ) from None
    if not vals:
        raise ValueError(
            f"--capture-sizes {spec!r}: no sizes given. Use 'default' or e.g. '1'.")
    vals = sorted(set(vals))
    nonpos = [v for v in vals if v < 1]
    if nonpos:
        raise ValueError(
            f"--capture-sizes {spec!r}: capture sizes must be >= 1 (got {nonpos}). "
            f"vLLM cannot capture a graph for batch size {nonpos[0]}.")
    return vals


# ── engine construction (quantization-aware; CUDA-graph fair-compare) ─────────
def _make_engine(model, backend, quant, max_len, gpu_mem, seed, enforce_eager,
                 num_gpu_blocks=0, capture_sizes="default", tp=1, kv_dtype="",
                 max_num_seqs=1):
    """LLMEngine via from_engine_args so we can drive .step() directly.

    backend -> VLLM_ATTENTION_BACKEND (env path EngineArgs honors).
    quant   -> 'ommx_w' (+ synthetic hf_override so vLLM v1 reaches the runtime-registered
               OMMXWConfig — integration/vllm/linear_method.py — for a checkpoint whose
               own config.json carries no quantization_config) or '' (stock bf16 linear).
    num_gpu_blocks -> EQUAL-KV-pool pin (fairness).
    max_num_seqs   -> the scheduler concurrency this study actually uses.
    capture_sizes  -> 'default' leaves vLLM's own cudagraph capture ladder alone; a comma
                      list (e.g. '1') pins cudagraph_capture_sizes. Parsed by
                      _parse_capture_sizes, which RAISES on anything it cannot honour.

    DECLARE THE CONCURRENCY. This is a B=1 latency study (one request per step), but
    EngineArgs leaves max_num_seqs unset, so vLLM applies its own large default. Two
    things went wrong with that:
      * the OMMX sidecar pool is sized from the scheduler concurrency, so an unset
        max_num_seqs made it reserve for hundreds of concurrent sequences that this
        bench never creates -- at --model-max-len 131072 that is a ~1.1 TiB projection
        and integration/vllm/preflight.py (correctly) refuses to start;
      * vLLM's own memory profiling also budgets for that phantom concurrency.
    Passing the real value makes the engine, the pool and the preflight agree on what
    the run does. Raise it only together with the batch size the arm actually drives."""
    from vllm import LLMEngine, EngineArgs
    kw = dict(model=model, max_model_len=max_len, gpu_memory_utilization=gpu_mem,
              tensor_parallel_size=int(tp), enforce_eager=enforce_eager,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              max_num_seqs=max(1, int(max_num_seqs)),
              dtype="bfloat16", trust_remote_code=True, seed=seed)
    if backend:
        # 🚨 vLLM 0.21 REMOVED the VLLM_ATTENTION_BACKEND env (it warns "Unknown vLLM
        # environment variable" and silently defaults to FLASH_ATTN). The backend MUST be
        # selected via the EngineArgs.attention_backend field (CLI --attention-backend), or
        # EVERY arm (CUSTOM, TRITON_ATTN, FLASH_ATTN) silently runs FLASH_ATTN -> the OMMX
        # attention never fires and all arms read identical (the bf16 fallback bug).
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        kw["attention_backend"] = backend
    if quant:
        kw["quantization"] = quant
        kw["hf_overrides"] = {"quantization_config": {"quant_method": quant}}
    if kv_dtype:
        # TurboQuant KV-cache-dtype (turboquant_3bit_nc etc.): vLLM auto-selects the turboquant
        # attention backend + Triton decode kernel; do NOT also pass attention_backend.
        kw["kv_cache_dtype"] = kv_dtype
    if num_gpu_blocks and num_gpu_blocks > 0:
        kw["num_gpu_blocks_override"] = int(num_gpu_blocks)
    sizes = _parse_capture_sizes(capture_sizes)   # raises on malformed / non-positive
    if sizes is not None:
        if enforce_eager:
            # An eager arm captures no graphs at all, so the sizes cannot be applied. SAY SO
            # rather than dropping them: the operator asked for a capture pin and this arm is
            # not going to honour it.
            print(f"[arm] enforce_eager=True -> cudagraph_capture_sizes {sizes} NOT applied "
                  f"(this arm captures no graph)", flush=True)
        else:
            kw["compilation_config"] = {"cudagraph_capture_sizes": sizes}
    ea = EngineArgs(**kw)
    return LLMEngine.from_engine_args(ea)


def _kv_blocks(eng):
    for getter in (
        lambda: eng.cache_config.num_gpu_blocks,
        lambda: eng.vllm_config.cache_config.num_gpu_blocks,
        lambda: eng.engine_core.engine_core.scheduler.cache_config.num_gpu_blocks,
        lambda: eng.model_executor.cache_config.num_gpu_blocks,
    ):
        try:
            v = int(getter())
            if v > 0:
                return v
        except Exception:
            continue
    return 0


# ── robust per-step decode timing (LLMEngine.step() loop, NO subtraction) ─────
def _time_cell(eng, ctx, batch, warmup, measure, reps):
    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    total = warmup + measure + 4
    ttft_samples = []
    tpot_samples = []

    def _drain():
        while eng.has_unfinished_requests():
            eng.step()

    for w in range(2):  # warm compile / cudagraph capture / allocator
        sp = SamplingParams(temperature=0.0, max_tokens=total, ignore_eos=True)
        for i in range(batch):
            eng.add_request(f"w{w}_{i}", TokensPrompt(prompt_token_ids=[100] * ctx), sp)
        steps = 0
        while eng.has_unfinished_requests() and steps < total + 2:
            eng.step(); steps += 1
        _drain()

    for rep in range(reps):
        sp = SamplingParams(temperature=0.0, max_tokens=total, ignore_eos=True)
        for i in range(batch):
            eng.add_request(f"r{rep}_{i}", TokensPrompt(prompt_token_ids=[100] * ctx), sp)
        torch.cuda.synchronize()
        prefill_ms = 0.0
        prefill_max = 0.0
        in_decode = False
        decode_idx = 0
        steps = 0
        per_step = []
        while eng.has_unfinished_requests() and steps < total + batch + 4:
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            eng.step()
            e.record(); torch.cuda.synchronize()
            dt = s.elapsed_time(e)
            per_step.append(dt)
            steps += 1
            if not in_decode:
                prefill_max = max(prefill_max, dt)
                if steps >= 1 and dt <= 0.40 * prefill_max and prefill_max > 0:
                    in_decode = True
                    if decode_idx >= warmup and (decode_idx - warmup) < measure:
                        tpot_samples.append(dt)
                    decode_idx += 1
                else:
                    prefill_ms += dt
            else:
                if decode_idx >= warmup and (decode_idx - warmup) < measure:
                    tpot_samples.append(dt)
                decode_idx += 1
        if prefill_ms == 0.0 and per_step:
            prefill_ms = per_step[0]
            for di, dt in enumerate(per_step[1:]):
                if di >= warmup and (di - warmup) < measure:
                    tpot_samples.append(dt)
        ttft_samples.append(prefill_ms)
        _drain()

    if not tpot_samples:
        raise RuntimeError(f"no decode steps captured (ctx={ctx} batch={batch})")
    return {
        "ttft_p50": _median(ttft_samples), "ttft_p99": _pctl(ttft_samples, 0.99),
        "tpot_p50": _median(tpot_samples), "tpot_p99": _pctl(tpot_samples, 0.99),
        "n_steps": len(tpot_samples),
    }


def _coherence(eng, ctx, maxtok=16):
    """Greedy token-id sequence for one length-ctx prompt (law #12 parity check)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(temperature=0.0, max_tokens=maxtok, ignore_eos=True)
    eng.add_request("coh", TokensPrompt(prompt_token_ids=[100] * ctx), sp)
    out_ids = []
    while eng.has_unfinished_requests():
        for o in eng.step():
            if getattr(o, "finished", False) or getattr(o, "outputs", None):
                try:
                    out_ids = list(o.outputs[0].token_ids)
                except Exception:
                    pass
    return out_ids


# ── OMMX route-fired evidence (law #5: prove the kernel ran, never assume) ───
# integration/vllm/backend.py writes ONE-TIME PER-RANK sentinel lines
#   "<TAG> rank=<r> pid=<p> <detail>"
# to $OMMX_FIRE_FILE (rank 0 -> the base path, rank r>0 -> "<base>.r<r><ext>"). A CUSTOM
# arm whose OMMX route never fired falls back to bf16 FlashAttention and still returns a
# perfectly plausible TPOT, so the sentinel is the ONLY thing separating an OMMX
# measurement from a mislabelled bf16 one.
# EXACT MEMBERSHIP HERE IS DELIBERATE -- DO NOT "FIX" IT INTO A SUFFIX RULE. The
# asymmetry with _NOFIRE_SUFFIXES below is the safe direction: a new *_ROUTE_FIRED tag
# added to backend.py and not added here falls into `other_tags`, which does NOT count as
# proof, so the arm reports ok=False. That is fail-CLOSED (a real OMMX arm needs a one-line
# addition here), whereas a suffix match on "_FIRED" would silently accept any future tag
# as decode proof -- including PACKED_PREFILL_FIRED, which is a prefill/capture marker and
# is excluded on exactly that principle.
FIRED_TAGS = ("DECODE_ROUTE_FIRED", "GRAPH_ROUTE_FIRED", "BATCHED_ROUTE_FIRED",
              "BATCHED_GRAPH_ROUTE_FIRED")
# The WEIGHT axis writes its own tags into the SAME sentinel (linear_method._fire): a
# quant='ommx_w' arm that produced no OMMX_W_*_FIRED line executed some other linear path,
# and publishing its TPOT under an "ommx_w" label is the attention bug (MEASURED_FACTS
# section 2) transplanted onto the weight axis. Prefix match, not an explicit list, for the
# same reason the failure tags match by suffix: an unlisted new tag must not read as proof
# of nothing, and OMMX_W_PREFILL_FIRED / OMMX_W_DECODE_FIRED / OMMX_W_OUTLIER_FIRED are all
# equally proof that apply() ran. OMMX_W_KERNEL_BUILT / OMMX_W_WEIGHTS_READY are NOT
# (a built kernel is not an executed one) and are excluded by the _FIRED suffix rule below.
OMMX_W_FIRED_PREFIX = "OMMX_W_"


def _is_ommx_w_fired_tag(tag: str) -> bool:
    """True for an OMMX_W tag that proves ``OMMXWLinearMethod.apply()`` actually ran.

    The _FIRED SUFFIX is the discriminator, not a name list. linear_method also writes
    lifecycle tags into the same sentinel -- OMMX_W_KERNEL_BUILT, OMMX_W_WEIGHTS_READY,
    OMMX_W_UNMAPPED_PASSTHROUGH -- and none of them carries that suffix, deliberately: a
    BUILT kernel and LOADED weights are not an EXECUTED kernel, and
    OMMX_W_UNMAPPED_PASSTHROUGH is the opposite of proof (that Linear served bf16).
    """
    return tag.startswith(OMMX_W_FIRED_PREFIX) and tag.endswith("_FIRED")
# FAILURE tags MATCH BY SUFFIX, NOT BY AN EXPLICIT LIST -- the opposite rule, for the
# opposite reason: an unlisted failure tag must never be ignored. backend.py's
# _ommx_route_failed() writes one tag per failing route -- currently KV_UPDATE_DEAD,
# DECODE_ROUTE_DEAD, GRAPH_ROUTE_DEAD, BATCHED_ROUTE_DEAD, BATCHED_GRAPH_ROUTE_DEAD --
# and an earlier revision of this file listed only KV_UPDATE_DEAD, so the other four fell
# through to `other_tags`, which is purely informational. An arm that fired once during
# cudagraph capture and then died on every real step therefore still reported ok=True:
# exactly the bf16-published-as-OMMX outcome this gate exists to stop. A suffix rule
# cannot drift out of sync when backend.py adds a route.
_NOFIRE_SUFFIXES = ("_NOFIRE", "_DEAD")


def _is_nofire_tag(tag: str) -> bool:
    """True for any route-failure tag (``*_NOFIRE`` / ``*_DEAD``)."""
    return tag.endswith(_NOFIRE_SUFFIXES)


# kept for the failure message / back-compat; the decision uses _is_nofire_tag.
NOFIRE_TAGS = ("DECODE_ROUTE_NOFIRE", "BATCHED_ROUTE_NOFIRE", "KV_UPDATE_DEAD",
               "DECODE_ROUTE_DEAD", "GRAPH_ROUTE_DEAD", "BATCHED_ROUTE_DEAD",
               "BATCHED_GRAPH_ROUTE_DEAD")


def _fire_paths(fire_file):
    """The base sentinel path plus every rank-suffixed sibling (TP ranks > 0)."""
    base, ext = os.path.splitext(fire_file)
    return sorted({fire_file, *glob.glob(f"{base}.r*{ext}")})


def _clear_fire_file(fire_file):
    """Delete stale sentinels BEFORE the arm runs.

    A stale file left by an earlier arm/run would be read back as THIS arm's proof, so an
    undeletable sentinel is fatal (the OSError propagates) rather than ignored."""
    for p in _fire_paths(fire_file):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def _read_fire_evidence(fire_file, tp=1, require_attn=True, require_linear=False):
    """Parse one arm's route sentinels and decide whether it is an OMMX measurement.

    ok=False (with a human reason) when: nothing was written, a REQUIRED firing tag is
    absent, a NOFIRE/DEAD tag is present, or fewer worker PROCESSES proved firing than
    --tp (each TP rank is its own EngineCore process, so the distinct pid count is the
    per-rank proof that survives a rank-lookup fallback).

    The two axes are required INDEPENDENTLY because an arm can select either alone:
    ``require_attn`` for backend='CUSTOM' (a *ROUTE_FIRED decode tag) and
    ``require_linear`` for quant='ommx_w' (an OMMX_W_*_FIRED tag). An ommx_w+FLASH_ATTN
    arm legitimately has no attention tag and must not be failed for it; an
    ommx_w+CUSTOM arm needs BOTH, because a fired OMMX attention route says nothing
    about whether the OMMX linear kernel ran -- e.g. OMMX_W_ALLOW_UNMAPPED=1 serves every
    Linear bf16 while the KV path fires normally."""
    files = [p for p in _fire_paths(fire_file) if os.path.exists(p)]
    lines = []
    for p in files:
        try:
            with open(p) as fh:
                lines += [ln.strip() for ln in fh if ln.strip()]
        except OSError as e:  # unreadable proof is NOT proof
            return dict(ok=False, reason=f"sentinel {p} unreadable ({type(e).__name__}: {e})",
                        files=files, fired=[], linear_fired=[],
                        require_attn=bool(require_attn), require_linear=bool(require_linear),
                        nofire=[], other_tags=[], ranks_fired=[],
                        procs_fired=0, tp=int(tp), lines=lines)
    fired, linear_fired, nofire, other = set(), set(), set(), set()
    ranks, pids = set(), set()
    unrouted_b = set()
    for ln in lines:
        tag = ln.split(" ", 1)[0]
        # DECODE_BF16_UNROUTED is NOT a *_NOFIRE/_DEAD tag, so it lands in `other_tags`
        # (informational) and leaves the arm ok=True -- but it means a decode step of B
        # requests was served by bf16 FlashAttention, i.e. those cells are bf16 timings
        # wearing an OMMX label. It is per-BATCH, not per-arm: the same process routes B=1
        # correctly, so demoting the whole arm would throw away valid cells. Record which
        # batch sizes were unrouted and let the caller demote exactly those.
        if tag == "DECODE_BF16_UNROUTED":
            for tok in ln.split():
                if tok.startswith("B="):
                    try:
                        unrouted_b.add(int(tok[len("B="):].rstrip(":,")))
                    except ValueError:
                        pass
        rank = pid = None
        for tok in ln.split():
            if tok.startswith("rank="):
                rank = tok[len("rank="):]
            elif tok.startswith("pid="):
                pid = tok[len("pid="):]
        if tag in FIRED_TAGS or _is_ommx_w_fired_tag(tag):
            (fired if tag in FIRED_TAGS else linear_fired).add(tag)
            if rank is not None:
                ranks.add(rank)
            if pid is not None:
                pids.add(pid)
        elif _is_nofire_tag(tag):
            nofire.add(tag)
        else:
            # PACKED_PREFILL_FIRED / PACKED_CAPTURE_STUB / PACKED_ONLY_SPEC: recorded for
            # the record, but a prefill/capture marker is NOT proof that decode routed.
            other.add(tag)
    reasons = []
    if not files:
        reasons.append(f"no sentinel written (expected {fire_file}); the OMMX backend "
                       f"recorded no route evidence at all")
    if require_attn and not fired:
        reasons.append(f"no *ROUTE_FIRED tag ({'|'.join(FIRED_TAGS)}); the OMMX decode "
                       f"route never fired, so these timings are not an OMMX measurement")
    if require_linear and not linear_fired:
        reasons.append("no OMMX_W_*_FIRED tag; OMMXWLinearMethod.apply() never ran, so "
                       "these timings are not an ommx_w WEIGHT-quant measurement "
                       "(see linear_method.ommx_w_health())")
    if nofire:
        reasons.append(f"NOFIRE/DEAD tag(s) present: {sorted(nofire)}")
    if int(tp) > 1 and len(pids) < int(tp):
        reasons.append(f"only {len(pids)} worker process(es) proved firing but --tp={tp}; "
                       f"every TP rank must fire (ranks seen: {sorted(ranks)})")
    return dict(ok=not reasons, reason="; ".join(reasons), files=files,
                fired=sorted(fired), linear_fired=sorted(linear_fired),
                require_attn=bool(require_attn), require_linear=bool(require_linear),
                nofire=sorted(nofire), other_tags=sorted(other),
                unrouted_batches=sorted(unrouted_b),
                ranks_fired=sorted(ranks), procs_fired=len(pids), tp=int(tp),
                lines=lines[:64])


# ── ommx_w (weight-quant linear) arms need an OMMX_W_SafeTensor bundle ───────
# The method itself now ships (integration/vllm/linear_method.py registers the ommx_w
# QuantizationConfig), but an arm that asks for it and has no packed WEIGHT BUNDLE must
# still die loudly rather than degrade into a bf16-linear arm reported under an "ommx_w"
# label (H7/M2 audit finding). Same rule, different missing piece.
OMMX_W_ARMS = ("OMMX", "abl_linear", "abl_full", "abl_full_no_outlier",
               "abl_full_no_dequant")
# The arms that need NO weight bundle (bf16 weights; attention backend is the variable).
SHIPPED_ARMS = ("TRITON", "FA", "abl_attn", "abl_attn_no_all", "abl_attn_no_unpack",
                "abl_attn_no_dequant", "abl_attn_skipwrite")


def _ommx_w_unavailable(arm_name: str, cause: str) -> str:
    """The one message every ommx_w refusal prints. Names the packer, not just the fault."""
    return (
        f"[{arm_name or 'arm'}] FATAL: quant='ommx_w' was requested but the OMMX "
        f"WEIGHT-QUANT LINEAR PATH CANNOT SERVE: {cause}.\n"
        f"  An ommx_w arm reads an OMMX_W_SafeTensor bundle — a directory of "
        f"ommx_w-000NN-of-000MM.safetensors shards plus ommx_w_index.json. Build one "
        f"with the offline packer, then point --model at it:\n"
        f"      python -m ommx_gpu_serve.linear.w_packer pack \\\n"
        f"          --input <hf safetensors checkpoint dir> \\\n"
        f"          --output <bundle dir> --group-size 64 --outlier-pct 0.0625\n"
        f"      python -m ommx_gpu_serve.linear.w_packer verify --bundle <bundle dir>\n"
        f"      ... --model <bundle dir>        # or export OMMX_W_BUNDLE=<bundle dir>\n"
        f"  This arm is NOT allowed to fall back to bf16 linear and be reported as an "
        f"ommx_w measurement, so the run stops here.\n"
        f"  ALTERNATIVE: exclude every ommx_w arm via --only-arms. Arms to drop: "
        f"{', '.join(OMMX_W_ARMS)}.\n"
        f"  e.g. --only-arms \"{','.join(SHIPPED_ARMS)}\"")


# ── single-arm worker ────────────────────────────────────────────────────────
def _single_arm(args):
    if args.quant == "ommx_w":
        # FATAL, never a warning (law #5: no silent fallback). A printed warning here would
        # let the arm keep running and be written to the results JSON under its ommx_w name
        # while actually executing some other linear path, i.e. one arm silently degrading
        # into a different arm. Refuse to produce that measurement.
        # Checked before this process imports torch or builds an engine, so a mis-set arm
        # dies while it still costs nothing. (register_ommx_w() does import vLLM's
        # quantization registry — that is unavoidable, it is what registration means —
        # but nothing here initialises CUDA or allocates on a device.)
        try:
            from ommx_gpu_serve.integration.vllm.linear_method import (
                OMMXWError, is_ommx_w_bundle, register_ommx_w, resolve_bundle_dir)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(_ommx_w_unavailable(
                args.single_arm, f"import of integration.vllm.linear_method failed "
                                 f"({type(e).__name__}: {e})"))
        try:
            register_ommx_w()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(_ommx_w_unavailable(
                args.single_arm,
                f"register_ommx_w() raised ({type(e).__name__}: {e})"))
        # THE BUNDLE IS THE OTHER HALF. Registration only teaches vLLM the NAME; without a
        # packed bundle the config cannot be built and the engine would fail somewhere deep
        # inside model construction, long after this process has taken a GPU. Resolve it
        # here, up front, where the message can still name the packer. --model is offered
        # as the explicit candidate only when it IS a bundle, so a plain HF checkpoint path
        # falls through to $OMMX_W_BUNDLE instead of being reported as a broken bundle.
        try:
            bundle = resolve_bundle_dir(args.model if is_ommx_w_bundle(args.model) else None)
        except OMMXWError as e:  # noqa: BLE001
            raise SystemExit(_ommx_w_unavailable(args.single_arm, str(e)))
        print(f"[arm] register_ommx_w() OK; ommx_w bundle = {bundle}", flush=True)
    import torch
    try:
        from ommx_gpu_serve.integration.vllm.plugin import register
        register()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] attn plugin register failed: {e}", flush=True)

    env_extra = json.loads(args.env) if args.env else {}
    for k, v in env_extra.items():
        os.environ[k] = v
    cells = [(int(p.split(":")[0]), int(p.split(":")[1]))
             for p in args.cells.split(",") if p]
    gm = args.gpu_mem if args.gpu_mem > 0 else _gpu_util()
    enforce_eager = args.eager
    mode = "eager" if enforce_eager else "graph"
    print(f"\n===== ARM={args.single_arm} backend={args.backend} quant={args.quant!r} "
          f"mode={mode} capture_sizes={args.capture_sizes} kvpin={args.num_gpu_blocks} "
          f"free={_free_mem_gib():.1f}GiB env={env_extra} =====", flush=True)
    try:
        # cells are (B, ctx) pairs, so the arm's own worst-case batch IS the scheduler
        # concurrency it needs -- tell the engine, instead of letting vLLM's large
        # default size both its memory profile and the OMMX sidecar pool for a
        # concurrency this arm never drives (see _make_engine's docstring).
        arm_max_seqs = max((b for b, _ in cells), default=1)
        eng = _make_engine(args.model, args.backend or None, args.quant or "",
                           args.model_max_len, gm, args.seed, enforce_eager,
                           num_gpu_blocks=args.num_gpu_blocks,
                           capture_sizes=args.capture_sizes, tp=args.tp,
                           kv_dtype=args.kv_dtype, max_num_seqs=arm_max_seqs)
    except Exception as e:  # noqa: BLE001
        print(f"[{args.single_arm}] LOAD FAILED ({type(e).__name__}: {e})", flush=True)
        with open(args.arm_out, "w") as f:
            json.dump({"cells": {}, "kv_blocks": 0,
                       "load_err": f"{type(e).__name__}: {e}"}, f)
        return
    kvb = _kv_blocks(eng)
    print(f"[{args.single_arm}] KV gpu_blocks = {kvb}", flush=True)

    coh = []
    if args.coherence_ctx > 0:
        try:
            coh = _coherence(eng, args.coherence_ctx, args.coherence_tok)
            print(f"[{args.single_arm}] COHERENCE ids[:16]={coh[:16]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{args.single_arm}] coherence skip ({type(e).__name__}: {e})", flush=True)

    out = {}
    for (b, c) in cells:
        key = f"b{b}_ctx{c}"
        try:
            r = _time_cell(eng, c, b, args.warmup, args.measure, args.reps)
            r["ok"] = True; r["mode"] = mode; r["oom"] = False
            out[key] = r
            print(f"[{args.single_arm}] {key:14s} TTFT p50={r['ttft_p50']:9.2f} "
                  f"p99={r['ttft_p99']:9.2f}ms | TPOT p50={r['tpot_p50']:8.4f} "
                  f"p99={r['tpot_p99']:8.4f}ms/step (nstep={r['n_steps']})", flush=True)
        except Exception as e:  # noqa: BLE001
            es = f"{type(e).__name__}: {e}"
            is_oom = ("CUDA out of memory" in es or "OutOfMemory" in es
                      or "no decode steps" in es and False)
            out[key] = dict(ok=False, err=es, mode=mode, oom=bool(is_oom))
            print(f"[{args.single_arm}] {key:14s} SKIP ({es})", flush=True)
            torch.cuda.empty_cache()
    del eng; gc.collect(); torch.cuda.empty_cache()
    with open(args.arm_out, "w") as f:
        json.dump({"cells": out, "kv_blocks": kvb, "coherence": coh}, f)
    print(f"[{args.single_arm}] ARM_DONE kvb={kvb}", flush=True)


# ── arm plan ─────────────────────────────────────────────────────────────────
def _fakequant_kv_recipe_env():
    """The fake-quant sim's KV-phase recipe, with ratio-selectable K outliers.

    NOT a citation of an external script: there is no `fakequant/` directory and no
    `--phase` flag anywhere in this repo (the sim is `ommx_fakequant/`, which ships no
    run.sh), so this dict is the only in-repo definition of the recipe it names. The flags
    it mirrors are `ommx_fakequant/quantizer.py`'s argparse.

    IT IS ALSO NOT THE CANONICAL PUBLISHED SERVING RECIPE. figure/bench.py's
    OMMX_RECIPE_ENV -- mirrored verbatim by eval/lm_eval/models/ommx_hf_model.py and pinned
    as bit-accounting literals by ommx_gpu_serve/tests/test_bit_accounting.py -- uses
    group_tokens/group_channels 32, recent 32 and outlier_select "signed"; this dict uses
    64 / 64 / 8 / "abs". So numbers produced under `--recipe fakequant` (the default here)
    must say so: they are not the published HF-eager arms' recipe.
    """
    return {
        # ommx_fakequant/quantizer.py KV phase: --group_size 64 --attention_sink_num 8
        # --use_pow2 --outlier_method fp4 --outlier_precision_key fp4
        # --no_detect_outliers_value. Serving expresses the back sink as recent=8.
        "OMMX_KV_GROUP_TOKENS": "64",
        "OMMX_KV_GROUP_CHANNELS": "64",
        "OMMX_KV_SINK": "8",
        "OMMX_KV_RECENT": "8",
        "OMMX_ATTN_POW2": "1",
        "OMMX_ATTN_K_FORMAT": "i2f4",
        "OMMX_ATTN_OUTLIER_SELECT": "abs",
        "OMMX_ATTN_OUTLIER_REPR": "relidx7",
        "OMMX_ATTN_COMBINADIC_READ": "0",
        "OMMX_KV_OUTLIER_MAP": "1",
        # This benchmark is Triton-only; CUDA attention route is intentionally excluded.
        "OMMX_ATTN_CUDA_DECODE": "0",
    }


def _arm_provenance(p, cells, capture_sizes):
    """What ONE arm was ACTUALLY measured under — the per-arm half of the fair-compare
    contract, which must not be asserted globally.

    ``enforce_eager`` is a per-arm property (``abl_attn_bat_eager`` sets it True by design,
    see _build_plan), so the single hardcoded ``enforce_eager=False, cudagraph="FULL"``
    provenance block this file used to write described, for that arm, a run that never
    happened. ``observed_modes`` is the "graph"/"eager" string each worker process derived
    from its OWN --eager flag and wrote into every cell it produced, so a plan-vs-worker
    disagreement surfaces in the JSON instead of being asserted away.
    """
    eager = bool(p.get("eager"))
    modes = sorted({c.get("mode") for c in (cells or {}).values()
                    if isinstance(c, dict) and c.get("mode")})
    return dict(
        backend=p.get("backend", ""), quant=p.get("quant", ""), label=p.get("label", ""),
        enforce_eager=eager,
        cudagraph=("NONE (enforce_eager=True)" if eager else "FULL"),
        cudagraph_capture_sizes=(
            "n/a (enforce_eager=True; this arm captures no graph)" if eager
            else ("vLLM default ladder (not pinned by this bench)" if capture_sizes is None
                  else list(capture_sizes))),
        observed_modes=modes,
        env=dict(p.get("env") or {}),
        recipe_resolved=_resolved_recipe(p),
    )


# Recipe knobs that decide WHICH NUMBER SYSTEM an OMMX arm measured. Recording the arm's
# env dict alone is not enough: the arm inherits the parent environment and the plan's env
# is only the OVERLAY, so a run under `--recipe current` records `OUTLIERS=6` and nothing
# else while the group sizes, pow2 and outlier-selection rule come from the shell. Two arms
# with identical `env` blocks can therefore be different recipes, which is exactly the
# ambiguity that made a published speed figure unattributable to a published accuracy
# figure. This resolves the SAME way the backend does (config.resolve_serving_config is
# pure-python, no vLLM) with the overlay applied, so the JSON says what actually ran.
# Field names are the dataclass's own (CanonicalAttentionConfig), verified against
# dataclasses.fields() — a guessed name would silently record None and the provenance would
# claim "unset" for a knob that was set, which is worse than not recording it at all.
_RECIPE_FIELDS = ("k_format", "outliers_per_vector", "use_pow2",
                  "group_tokens", "group_channels", "sink_tokens", "recent_window",
                  "outlier_select", "outlier_repr", "kv_outlier_map", "page_size")


def _resolved_recipe(p):
    """The OMMX recipe an arm actually resolves to (parent env + this arm's overlay).

    Returns None for arms that run no OMMX kernel (bf16 baselines), and a dict with an
    ``error`` key rather than raising if resolution fails — a provenance helper must never
    be able to take a measurement down.
    """
    backend = str(p.get("backend") or "")
    quant = str(p.get("quant") or "")
    if backend != "CUSTOM" and "ommx" not in quant:
        return None
    overlay = dict(p.get("env") or {})
    saved = {k: os.environ.get(k) for k in overlay}
    try:
        os.environ.update({k: str(v) for k, v in overlay.items()})
        from ommx_gpu_serve.integration.vllm.config import resolve_serving_config
        cfg = resolve_serving_config()
        out = {f: _jsonable_scalar(getattr(cfg, f, None)) for f in _RECIPE_FIELDS}
        # The env the resolution actually saw, so a reader can redo it by hand.
        out["ommx_env"] = {k: v for k, v in sorted(os.environ.items())
                           if k.startswith(("OMMX_KV_", "OMMX_ATTN_", "OMMX_RECIPE"))}
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _jsonable_scalar(v):
    return v if v is None or isinstance(v, (bool, int, float, str)) else str(v)


def _vllm_version():
    """vLLM version string, or None. Never raises: this is provenance, not a dependency."""
    try:
        import vllm
        return str(getattr(vllm, "__version__", None) or "") or None
    except Exception:  # noqa: BLE001
        return None


def _build_plan(cmp_ctxs, abl_ctxs, outlier_env, batches=(1,), kv_token_budget=0):
    """arm name -> dict(backend, quant, eager, cells[(b,ctx)], env, label, group).

    batches: decode batch sizes to sweep (B>1 = high-batch memory-bound regime; the
    OMMX CUSTOM arms need OMMX_ATTN_BATCHED_GRAPH=1 in the env so B>1 fires the captured
    batched route). kv_token_budget: drop (b,ctx) cells whose b*ctx exceeds the equal-KV
    pool token capacity (avoids preempt/OOM; 0 = keep all)."""
    plan = {}

    def _cells(ctxs):
        return [(b, c) for b in batches for c in ctxs
                if kv_token_budget <= 0 or b * c <= kv_token_budget]
    cmp_cells = _cells(cmp_ctxs)
    abl_cells = _cells(abl_ctxs)

    # ── headline COMPARISON ──
    plan["OMMX"] = dict(backend="CUSTOM", quant="ommx_w", eager=False, cells=cmp_cells,
                        env={"OMMX_ATTN_GRAPH": "1", **outlier_env},
                        label="ommx_w+CUSTOM", group="cmp")
    plan["FA3"] = dict(backend="FLASH_ATTN", quant="", eager=False, cells=cmp_cells,
                       env={}, label="bf16+FA3", group="cmp")
    plan["TRITON"] = dict(backend="TRITON_ATTN", quant="", eager=False, cells=cmp_cells,
                          env={}, label="bf16+TRITON", group="cmp")
    # vLLM's DEFAULT attention backend under an honest name. "FA3" above is the SAME
    # backend (FLASH_ATTN) named after a kernel version this arm does not verify, and it is
    # what a published "bf16 (vLLM)" bar should be compared against — so give the stock-vLLM
    # baseline a name that claims nothing. Declared exactly like TRITON (bf16 weights, no
    # env, CUDA-graph FULL) so the only difference vs abl_attn is the attention backend.
    # NOT in the default `order` list: adding it there would silently grow every existing
    # sweep by one full arm. Select it explicitly with --only-arms (which reaches arms
    # outside `order`, see main()).
    plan["FA"] = dict(backend="FLASH_ATTN", quant="", eager=False, cells=cmp_cells,
                      env={}, label="bf16+FLASH_ATTN(vLLM default)", group="cmp")
    # TurboQuant KV-cache quant (3-bit K+V + norm-correction), a KV-quant peer to OMMX/KIVI/Kitty.
    # backend="" -> vLLM auto-selects the turboquant attention backend + Triton decode kernel.
    plan["TURBOQUANT"] = dict(backend="", quant="", kv_dtype="turboquant_3bit_nc", eager=False,
                              cells=cmp_cells, env={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
                              label="turboquant_3bit_nc", group="cmp")
    # v10a head-to-head: bf16-w (W NOT quantized) + OMMX CUSTOM attention with K=i2f4 and
    # V=bf16 (OMMX_ATTN_V_BF16). Isolates the attention kernel vs FA3/TRITON (identical
    # bf16 linear); V=bf16 drops the V=i2 dequant ALU (the M=1 decode loss cause) keeping
    # the K i2f4 byte saving -> the win-path test. Coherence vs FA3 proves it fires.
    plan["OMMX_V10A"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=cmp_cells,
        env={"OMMX_ATTN_GRAPH": "1", "OMMX_KV_RING": "1", "OMMX_KV_GPU_PACK": "1",
             "OMMX_ATTN_V_BF16": "1", **outlier_env},
        label="bf16-w+OMMX-v10a(K=i2f4,V=bf16)", group="cmp")

    # ── ABLATION stack (B=1, abl_ctxs) ──
    plan["abl_bf16"] = dict(backend="FLASH_ATTN", quant="", eager=False, cells=abl_cells,
                            env={}, label="bf16-w+FA3", group="abl")
    # ── FAIR-COMPARISON CANON (change ONLY the part under test; OMMX defaults baked in) ──
    # LINEAR comparison: plain-vLLM (abl_bf16: bf16-w + FA) vs OMMX-linear (ommx_w + FA),
    #   identical FA attention -> the TPOT delta is purely the OMMX LINEAR. SCAN-FREE relidx7
    #   correction ON by default (the in-kernel combinadic unrank was the catastrophic prefill).
    plan["abl_linear"] = dict(backend="FLASH_ATTN", quant="ommx_w", eager=False,
                              cells=abl_cells, env={"OMMX_W_CORR_SCANFREE": "1"},
                              label="ommx_w(scanfree)+FA", group="abl")
    # ATTENTION comparison: paged-triton-v1 (TRITON arm: bf16 attn + bf16-w) vs OMMX-attn
    #   (CUSTOM OMMX attn + bf16-w), identical bf16 linear -> TPOT delta is purely the OMMX
    #   ATTENTION. NO-SHADOW RING ON by default: the OMMX quantized planes cost 4.375 bit per
    #   (K,V) element pair for the canonical recipe (3.66x vs bf16 -- MEASURED from the real
    #   MultiSeqKVPool allocation; see packed_only.kv_bits_breakdown and tests/
    #   test_bit_accounting.py, NOT the retracted "<=3-bit / ~4.6x" figure, which belongs to the
    #   group_tokens=64 + OMMX_KV_OUTLIER_MAP=0 recipe), so it uses less HBM than bf16 ONLY once
    #   the bf16 k_hist SHADOW is gone -- that shadow, not the planes, was the A100 OOM cause.
    #   The ring keeps only the sink+recent window. GPU-pack on (boundary-step pack on device,
    #   no host p99).
    plan["abl_attn"] = dict(backend="CUSTOM", quant="", eager=False, cells=abl_cells,
                            env={"OMMX_ATTN_GRAPH": "1", "OMMX_KV_RING": "1",
                                 "OMMX_KV_GPU_PACK": "1", **outlier_env},
                            label="bf16-w+OMMXattn(noshadow)", group="abl", coh=True)
    plan["abl_full"] = dict(backend="CUSTOM", quant="ommx_w", eager=False, cells=abl_cells,
                            env={"OMMX_ATTN_GRAPH": "1", **outlier_env},
                            label="ommx_w+CUSTOM", group="abl")
    plan["abl_full_no_outlier"] = dict(
        backend="CUSTOM", quant="ommx_w", eager=False, cells=abl_cells,
        env={"OMMX_ATTN_GRAPH": "1", "OMMX_ABL_NO_OUTLIER": "1", **outlier_env},
        label="ommx_w+CUSTOM-outlier", group="abl")
    plan["abl_full_no_dequant"] = dict(
        backend="CUSTOM", quant="ommx_w", eager=False, cells=abl_cells,
        env={"OMMX_ATTN_GRAPH": "1", "OMMX_ABL_V_NODEQUANT": "1", **outlier_env},
        label="ommx_w+CUSTOM-Vdequant", group="abl")
    # PURE attention isolation (bf16-w + CUSTOM): outlier-decode and V-dequant share of the
    # ATTENTION cost alone (scenario-2), not confounded by the ommx_w linear path.
    plan["abl_attn_no_outlier"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={"OMMX_ATTN_GRAPH": "1", "OMMX_ABL_NO_OUTLIER": "1", **outlier_env},
        label="bf16-w+CUSTOM-outlier", group="abl")
    plan["abl_attn_no_dequant"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={"OMMX_ATTN_GRAPH": "1", "OMMX_ABL_V_NODEQUANT": "1", **outlier_env},
        label="bf16-w+CUSTOM-Vdequant", group="abl")
    # ── V=i2 DEQUANT LEVERS (bf16-w + CUSTOM = abl_attn + ONE lever env) ──
    # Attack the V=i2 per-element dequant ALU (the measured 12-23ms per-tile tax) to
    # push the OMMX-attn TPOT from parity toward a >=1.10x win vs TRITON at long ctx.
    # All parity-by-construction (bit-exact codes 0..3) -> coh=True match_vs_TRITON gate.
    _abl_attn_env = {"OMMX_ATTN_GRAPH": "1", "OMMX_KV_RING": "1",
                     "OMMX_KV_GPU_PACK": "1", **outlier_env}
    # ── BREAKDOWN-SWEEP ABL arms (bf16-w + CUSTOM = abl_attn + ONE/MORE timing flag) ──
    # PARITY-BREAKING by design (coh=False): each forces the kernel/backend to SKIP one
    # decode cost so its TPOT delta isolates that share (vs abl_attn = full).
    #   abl_attn            : full  (already defined above)
    #   abl_attn_no_outlier : full + NO_OUTLIER (outlier-decode share)
    #   abl_attn_no_dequant : full + V_NODEQUANT only (V-dequant share)  [already above]
    #   abl_attn_no_unpack  : full + NO_UNPACK (bit-extract ALU share)
    #   abl_attn_no_dequant_all : full + K&V NODEQUANT (full scale/zp dequant share)
    #   abl_attn_no_all     : full + NO_OUTLIER+K&V NODEQUANT+NO_UNPACK (base attn math+mem)
    #   abl_attn_skipwrite  : full + SKIP_WRITE (per-step KV pack/write share)
    plan["abl_attn_no_unpack"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={**_abl_attn_env, "OMMX_ABL_NO_UNPACK": "1"},
        label="bf16-w+CUSTOM-unpack", group="abl", coh=False)
    plan["abl_attn_no_dequant_all"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={**_abl_attn_env, "OMMX_ABL_V_NODEQUANT": "1",
             "OMMX_ABL_K_NODEQUANT": "1"},
        label="bf16-w+CUSTOM-KVdequant", group="abl", coh=False)
    plan["abl_attn_no_all"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={**_abl_attn_env, "OMMX_ABL_NO_OUTLIER": "1",
             "OMMX_ABL_V_NODEQUANT": "1", "OMMX_ABL_K_NODEQUANT": "1",
             "OMMX_ABL_NO_UNPACK": "1"},
        label="bf16-w+CUSTOM-base(math+mem)", group="abl", coh=False)
    plan["abl_attn_skipwrite"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={**_abl_attn_env, "OMMX_ABL_SKIP_WRITE": "1"},
        label="bf16-w+CUSTOM-skipwrite", group="abl", coh=False)
    plan["OMMX_VINTPV"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=cmp_cells,
        env={**_abl_attn_env, "OMMX_ATTN_V_INT_PV": "1"},
        label="bf16-w+OMMXattn+V_INT_PV", group="lever", coh=True)
    plan["OMMX_VLUT"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=cmp_cells,
        env={**_abl_attn_env, "OMMX_ATTN_VLUT": "1"},
        label="bf16-w+OMMXattn+VLUT", group="lever", coh=True)
    plan["OMMX_VINTPV_PIPE"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=cmp_cells,
        env={**_abl_attn_env, "OMMX_ATTN_V_INT_PV": "1",
             "OMMX_ATTN_KV_PIPE": "1", "OMMX_V4_NUM_STAGES": "3"},
        label="bf16-w+OMMXattn+V_INT_PV+pipe(s3)", group="lever", coh=True)
    # BATCHED OMMX (B>1) decode. CRITICAL: OMMX_ATTN_GRAPH MUST be ABSENT — the backend
    # gate is `in_batched = _BATCHED_SESSION and not _GRAPH`, so OMMX_ATTN_GRAPH=1 disables
    # the batched route entirely and a B>1 decode silently falls to bf16 FlashAttention
    # (super().forward) with NO route marker. OMMX_ATTN_BATCHED=1 force-latches the pool;
    # OMMX_ATTN_BATCHED_GRAPH=1 captures the batched decode graph. Firing proof =
    # BATCHED_GRAPH_ROUTE_FIRED in $OMMX_FIRE_FILE.
    plan["abl_attn_bat"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=cmp_cells,
        env={"OMMX_ATTN_BATCHED": "1", "OMMX_ATTN_BATCHED_GRAPH": "1",
             "OMMX_KV_RING": "1", "OMMX_KV_GPU_PACK": "1", **outlier_env},
        label="bf16-w+OMMXattn-BATCHED(graph)", group="lever", coh=True)
    # EAGER batched variant: sidesteps the vLLM multi-size graph-capture ladder (the
    # B=1/prefill-shaped capture dummy crashes the batched write path). enforce_eager=True
    # => no capture => the eager batched route (_route_decode_batched, BATCHED_ROUTE_FIRED)
    # runs. NO OMMX_ATTN_BATCHED_GRAPH (eager). Compared vs TRITON(graph) is conservative
    # (TRITON keeps the launch-amortization edge), so an OMMX win here is a real floor.
    plan["abl_attn_bat_eager"] = dict(
        backend="CUSTOM", quant="", eager=True, cells=cmp_cells,
        env={"OMMX_ATTN_BATCHED": "1", "OMMX_KV_RING": "1",
             "OMMX_KV_GPU_PACK": "1", **outlier_env},
        label="bf16-w+OMMXattn-BATCHED(eager)", group="lever", coh=False)
    # LSE-merge wide-grid + tail-dedup A/B: abl_attn + OMMX_MERGE_WIDE=1 (default-OFF lever
    # that fills SMs in the 2nd/merge launch + shares the GQA tail). Parity bit-exact by
    # design; compare vs abl_attn (same pool, same process) to isolate the merge-launch cost.
    plan["abl_attn_mw"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={"OMMX_ATTN_GRAPH": "1", "OMMX_KV_RING": "1", "OMMX_KV_GPU_PACK": "1",
             "OMMX_MERGE_WIDE": "1", **outlier_env},
        label="bf16-w+OMMXattn+merge_wide", group="abl", coh=True)
    plan["OMMX_VINTPV_BAT"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=cmp_cells,
        env={"OMMX_ATTN_BATCHED": "1", "OMMX_ATTN_BATCHED_GRAPH": "1",
             "OMMX_ATTN_V_INT_PV": "1", "OMMX_KV_RING": "1",
             "OMMX_KV_GPU_PACK": "1", **outlier_env},
        label="bf16-w+OMMXattn-BATCHED+V_INT_PV", group="lever", coh=True)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cmp-ctxs", default="1024,4096,16384,65536,131072")
    ap.add_argument("--abl-ctxs", default="4096,65536")
    ap.add_argument("--batches", default="1",
                    help="comma list of decode batch sizes to sweep (B>1 high-batch "
                         "regime; OMMX arms need OMMX_ATTN_BATCHED_GRAPH=1 in env)")
    ap.add_argument("--kv-token-budget", type=int, default=0,
                    help="drop (b,ctx) cells where b*ctx exceeds this (equal-KV pool "
                         "fit; avoids preempt/OOM). 0 = keep all")
    ap.add_argument("--model-max-len", type=int, default=131072)
    ap.add_argument("--gpu-mem", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--capture-sizes", default="default",
                    help="cudagraph_capture_sizes for every graph arm: 'default' = vLLM's "
                         "own capture ladder (the default here -- this bench does NOT pin "
                         "it), or a comma list, e.g. '1' for this B=1 study. Applied to "
                         "graph arms only (an eager arm captures nothing and says so). "
                         "A malformed or non-positive value is a hard error, and the "
                         "resolved value is recorded in results['provenance'][<arm>].")
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--measure", type=int, default=160)  # >=128 steady-state steps
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--outliers", type=int, default=6)  # canonical 12/64 (6 per 32-ch group)
    ap.add_argument("--outlier-pct", default="",
                    help="K outlier fraction per 64-token fakequant recipe group; e.g. 0.05 or 0.10")
    ap.add_argument("--recipe", choices=("fakequant", "current"), default="fakequant",
                    help="fakequant = the ommx_fakequant KV-phase recipe defined in "
                         "_fakequant_kv_recipe_env (group_tokens/channels 64, recent 8, "
                         "outlier_select abs) -- NOT figure/bench.py's published "
                         "OMMX_RECIPE_ENV (32/32/32/signed); current = use only the env "
                         "the caller sets")
    ap.add_argument("--coherence-ctx", type=int, default=128,
                    help="run a short greedy decode per arm for token-id parity (0=off)")
    ap.add_argument("--coherence-tok", type=int, default=16)
    ap.add_argument("--csv", default="runs/e2e_a100.csv")
    ap.add_argument("--json", default="runs/e2e_a100.json")
    ap.add_argument("--workdir", default="runs/e2e_a100_arms")
    ap.add_argument("--arch", default="a100")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor_parallel_size. TP>1 shards KV heads per rank; the OMMX "
                         "sidecar store/pool/kernel are per-rank-sharded already. Firing is "
                         "asserted PER RANK from the rank-suffixed $OMMX_FIRE_FILE: a CUSTOM "
                         "arm where fewer than TP worker processes fired is marked ok=False.")
    ap.add_argument("--only-arms", default="",
                    help="comma list of arm names to run (kvprobe always runs); empty=all. "
                         "Also reaches arms outside the default order (e.g. FA, the "
                         "vLLM-default FLASH_ATTN baseline). Unknown names are a hard error. "
                         "Drop the bundle-dependent ommx_w arms here: "
                         + ", ".join(OMMX_W_ARMS))
    # single-arm worker mode
    ap.add_argument("--single-arm", default="")
    ap.add_argument("--backend", default="")
    ap.add_argument("--quant", default="")
    ap.add_argument("--kv-dtype", default="", dest="kv_dtype")  # e.g. turboquant_3bit_nc
    ap.add_argument("--cells", default="")
    ap.add_argument("--env", default="")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--num-gpu-blocks", type=int, default=0)
    ap.add_argument("--arm-out", default="")
    args = ap.parse_args()

    # Validate the fair-compare knobs BEFORE anything spawns, imports torch or touches a
    # GPU. A malformed --capture-sizes used to be swallowed by a bare `except: pass` inside
    # _make_engine, so the whole sweep ran on vLLM's default ladder while the operator
    # believed the pin was in force.
    try:
        capture_sizes = _parse_capture_sizes(args.capture_sizes)
    except ValueError as e:
        raise SystemExit(f"[e2e] {e}")

    if args.single_arm:
        _single_arm(args)
        return

    cmp_ctxs = [int(x) for x in args.cmp_ctxs.split(",")
                if x and int(x) + args.warmup + args.measure + 8 <= args.model_max_len]
    abl_ctxs = [int(x) for x in args.abl_ctxs.split(",")
                if x and int(x) + args.warmup + args.measure + 8 <= args.model_max_len]
    outlier_env = _fakequant_kv_recipe_env() if args.recipe == "fakequant" else {}
    outlier_pct = (args.outlier_pct or os.environ.get("OMMX_OUTLIER_PERCENT", "")).strip()
    if outlier_pct:
        pct = float(outlier_pct)
        outlier_env["OMMX_OUTLIER_PERCENT"] = f"{pct:g}"
        outlier_desc = f"{pct * 100:g}%"
    else:
        outlier_env["OMMX_ATTN_OUTLIERS"] = str(args.outliers)
        outlier_desc = f"npv={args.outliers}"
    os.makedirs(args.workdir, exist_ok=True)
    batches = tuple(int(b) for b in args.batches.split(",") if b.strip()) or (1,)
    plan = _build_plan(cmp_ctxs, abl_ctxs, outlier_env, batches=batches,
                       kv_token_budget=args.kv_token_budget)
    only_arms = [a.strip() for a in args.only_arms.split(",") if a.strip()]
    unknown = [a for a in only_arms if a not in plan]
    if unknown:
        # Validated BEFORE the KV-pool probe subprocess: a typo'd arm name would otherwise
        # silently shrink the sweep (and waste a probe run) instead of failing.
        raise SystemExit(f"[e2e] --only-arms names unknown arm(s) {unknown}; "
                         f"available: {sorted(plan)}")
    n_cells = sum(len(p["cells"]) for p in plan.values())
    print(f"[e2e] model={args.model}\n[e2e] cmp_ctxs={cmp_ctxs} abl_ctxs={abl_ctxs} "
          f"warmup={args.warmup} measure={args.measure} reps={args.reps} "
          f"recipe={args.recipe} outliers={outlier_desc}\n"
          f"[e2e] {len(plan)} arms, {n_cells} cells", flush=True)

    # EQUAL-KV-POOL: run bf16 arm at SMALLEST ctx first to read num_gpu_blocks, then PIN it
    # for every arm. bf16-w is the largest-weight arm so its KV pool is the binding floor.
    # PROVENANCE: `fair` carries ONLY the conditions that really are identical for every
    # arm. enforce_eager / cudagraph / capture_sizes are NOT among them (abl_attn_bat_eager
    # runs eager on purpose, and an eager arm captures no graph at all), so they are
    # recorded per arm in results["provenance"] from the plan each arm was actually spawned
    # with — never asserted here for arms that did not meet them.
    results = {"model": args.model, "arms": {}, "coherence": {}, "route_evidence": {},
               "provenance": {},
               # The engine version is part of what a serving number means, and it was not
               # recoverable from these JSONs before: a reader had to trust a shell history.
               # Recorded from the interpreter that spawns the arms, i.e. the one whose vLLM
               # they import. None (not a guess) when vLLM is not importable here.
               "vllm_version": _vllm_version(),
               "config": dict(cmp_ctxs=cmp_ctxs, abl_ctxs=abl_ctxs, warmup=args.warmup,
                              measure=args.measure, reps=args.reps, arch=args.arch,
                              recipe=args.recipe, outliers=args.outliers,
                              outlier_pct=outlier_pct,
                              fair=dict(dtype="bfloat16", seed=args.seed,
                                        prefix_caching=False, chunked_prefill=False,
                                        # overwritten below with the value the probe
                                        # actually produced -- see the kv-probe block
                                        equal_kv_pool="pending: kv-pool probe has not run",
                                        cudagraph_capture_sizes_requested=(
                                            "vLLM default ladder (not pinned by this bench)"
                                            if capture_sizes is None else list(capture_sizes)),
                                        per_arm=("enforce_eager / cudagraph / capture "
                                                 "sizes vary by arm: read "
                                                 "results['provenance'][<arm>]")),
                              method="LLMEngine.step() CUDA-event per-step; TPOT p50/p99 "
                                     "over measure steps x reps (NO subtraction); TTFT = "
                                     "prefill-step CUDA-event time")}

    def _spawn(arm_name, p, num_gpu_blocks, coherence_ctx):
        cell_str = ",".join(f"{b}:{c}" for (b, c) in p["cells"])
        arm_out = os.path.join(args.workdir, f"{arm_name}.json")
        cmd = [sys.executable, "-m", "ommx_gpu_serve.bench.bench_e2e_a100",
               "--single-arm", arm_name, "--backend", p["backend"],
               "--quant", p["quant"], "--kv-dtype", p.get("kv_dtype", ""),
               "--model", args.model, "--cells", cell_str,
               "--env", json.dumps(p["env"]),
               "--model-max-len", str(args.model_max_len), "--gpu-mem", str(args.gpu_mem),
               "--seed", str(args.seed), "--warmup", str(args.warmup),
               "--measure", str(args.measure), "--reps", str(args.reps),
               "--num-gpu-blocks", str(num_gpu_blocks),
               "--capture-sizes", args.capture_sizes,
               "--coherence-ctx", str(coherence_ctx),
               "--coherence-tok", str(args.coherence_tok),
               "--tp", str(args.tp),
               "--arm-out", arm_out]
        if p["eager"]:
            cmd.append("--eager")
        # ROUTE-FIRED EVIDENCE (law #5). A CUSTOM arm gets its OWN sentinel path so two
        # arms can never read each other's proof, and any stale copy is deleted BEFORE the
        # child starts (a survivor would be read back as this arm's proof).
        env = dict(os.environ)
        fire_file = ""
        # BOTH axes need the sentinel, not just the attention one. quant='ommx_w' with
        # backend='FLASH_ATTN' (abl_linear) used to get NO sentinel and NO gate at all, so
        # its cells were published under an "ommx_w" label with zero proof the OMMX linear
        # kernel ran -- the same defect the CUSTOM gate exists to prevent, on the other axis.
        want_attn = p["backend"] == "CUSTOM"
        want_linear = p["quant"] == "ommx_w"
        if want_attn or want_linear:
            fire_file = os.path.abspath(os.path.join(args.workdir, f"{arm_name}.fire.log"))
            _clear_fire_file(fire_file)
            env["OMMX_FIRE_FILE"] = fire_file
        print(f"\n########## SPAWN {arm_name} backend={p['backend']} quant={p['quant']!r} "
              f"kvpin={num_gpu_blocks} env={p['env']} ##########", flush=True)
        # PURGE THIS ARM'S OUTPUT FIRST. <workdir>/<arm>.json is read back below purely on
        # EXISTENCE, so a file left by an EARLIER sweep is indistinguishable from one this
        # child wrote: a child that dies (OOM, no CUDA, missing vLLM) then has the previous
        # run's cells published under this run's arch/arm labels while the log says the arm
        # produced no output, and a stale kvprobe.json silently becomes the equal-KV-pool
        # pin every other arm is measured against. Same rule run.sh applies to
        # figure/data_<tag>/ and _clear_fire_file applies to the sentinel: a leg that does
        # not finish must leave NO file. (This is arm_out's only reader; nothing resumes
        # from the workdir, so deleting it up front loses nothing.)
        if os.path.exists(arm_out):
            os.remove(arm_out)
        rc = subprocess.call(cmd, env=env)
        d = {}
        if os.path.exists(arm_out):
            with open(arm_out) as f:
                d = json.load(f)
        else:
            print(f"[orch] arm {arm_name} produced no output (rc={rc})", flush=True)
        if fire_file:
            ev = _read_fire_evidence(fire_file, tp=args.tp, require_attn=want_attn,
                                     require_linear=want_linear)
            ev["fire_file"] = fire_file
            d["route_evidence"] = ev
            print(f"[orch] {arm_name} route_evidence ok={ev['ok']} fired={ev['fired']} "
                  f"linear_fired={ev['linear_fired']} "
                  f"nofire={ev['nofire']} other={ev['other_tags']} "
                  f"procs_fired={ev['procs_fired']} ranks={ev['ranks_fired']}", flush=True)
            cells = d.get("cells") or {}
            for cell in cells.values():
                if isinstance(cell, dict):
                    cell["route_fired"] = bool(ev["ok"])
            if not ev["ok"]:
                # The arm ran, but it cannot prove the OMMX kernel produced these numbers.
                # Mark EVERY cell not-ok: the CSV, the JSON and e2e_to_figure.py all gate on
                # "ok", so an unproven CUSTOM arm can never surface as an OMMX bar.
                print(f"[orch] {arm_name} NOT AN OMMX MEASUREMENT -> forcing "
                      f"{len(cells)} cell(s) ok=False: {ev['reason']}", flush=True)
                for cell in cells.values():
                    if not isinstance(cell, dict):
                        continue
                    prev = cell.get("err", "")
                    cell["ok"] = False
                    cell["err"] = ("OMMX route not proven: " + ev["reason"]
                                   + (f" | prior err: {prev}" if prev else ""))
                # Dump what the backend DID write. The most likely first-run failure is an
                # init/capture-time *_NOFIRE line that precedes a real *_ROUTE_FIRED (the
                # sentinel dedup is per (tag, rank), not per layer/step), and that is only
                # distinguishable by reading the actual lines -- so put them in the run log
                # rather than making the operator open the JSON.
                for ln in ev.get("lines") or ["<sentinel file empty>"]:
                    print(f"[orch]   {arm_name} sentinel: {ln}", flush=True)
            # PER-BATCH GATE. An arm can be ok=True overall (its B=1 decode routed and wrote
            # GRAPH_ROUTE_FIRED) while a B>1 step of the SAME process was served by bf16
            # FlashAttention -- backend.py records that as DECODE_BF16_UNROUTED, which is not
            # a *_NOFIRE/_DEAD tag and so does not move `ok`. Measured on H200: with the
            # published bar's env (OMMX_ATTN_GRAPH=1) the b4 cells came back ok=True and
            # route_fired=True carrying bf16 timings. Demote exactly the batch sizes the
            # backend said it could not route; leave the rest of the arm alone.
            for _b in (ev.get("unrouted_batches") or []):
                for cname, cell in cells.items():
                    if not isinstance(cell, dict) or not cname.startswith(f"b{_b}_"):
                        continue
                    prev = cell.get("err", "")
                    cell["ok"] = False
                    cell["route_fired"] = False
                    cell["err"] = (f"served by bf16 FlashAttention: backend recorded "
                                   f"DECODE_BF16_UNROUTED for B={_b} (no OMMX route accepts "
                                   f"this step)" + (f" | prior err: {prev}" if prev else ""))
                print(f"[orch] {arm_name} B={_b} cells forced ok=False "
                      f"(DECODE_BF16_UNROUTED: bf16 served that batch size)", flush=True)
            # Persist the GATED result over the child's own file: <workdir>/<arm>.json is
            # read by humans (and by anything resuming from the workdir), so leaving the
            # child's un-gated ok=True there would keep an unproven OMMX measurement on
            # disk under an OMMX arm name -- exactly what the gate exists to prevent.
            if arm_out:
                with open(arm_out, "w") as f:
                    json.dump(d, f, indent=2)
        return d

    # ── probe: bf16 TRITON at smallest ctx to fix the KV pool ──
    # FA3 is intentionally not used as probe/baseline for the attention-only win path.
    probe_ctx = min(cmp_ctxs) if cmp_ctxs else min(abl_ctxs)
    probe_plan = dict(backend="TRITON_ATTN", quant="", eager=False,
                      cells=[(1, probe_ctx)], env={}, label="kv-probe-triton")
    print(f"\n[e2e] KV-POOL PROBE: bf16 TRITON ctx={probe_ctx}", flush=True)
    pd = _spawn("kvprobe", probe_plan, 0, 0)
    results["provenance"]["kvprobe"] = _arm_provenance(probe_plan, pd.get("cells", {}),
                                                       capture_sizes)
    pinned_blocks = int(pd.get("kv_blocks", 0))
    print(f"[e2e] pinned num_gpu_blocks = {pinned_blocks}", flush=True)
    # RECORD THE PIN THAT ACTUALLY HAPPENED. _make_engine applies num_gpu_blocks_override
    # only when the value is > 0, so a probe arm that failed to build (load_err -> the
    # worker writes kv_blocks=0) leaves every arm sizing its own KV pool -- while `fair`
    # used to assert "pinned to bf16 TRITON smallest-ctx arm" unconditionally. Same defect
    # class as the hardcoded enforce_eager/cudagraph block: a condition asserted in the
    # result JSON that the run never met.
    if pinned_blocks > 0:
        results["config"]["fair"]["equal_kv_pool"] = (
            f"pinned to the bf16 TRITON smallest-ctx arm: "
            f"num_gpu_blocks_override={pinned_blocks}")
    else:
        results["config"]["fair"]["equal_kv_pool"] = (
            "NOT PINNED: the kv-pool probe (bf16 TRITON at the smallest ctx) reported no "
            "num_gpu_blocks, so every arm sized its own KV pool. Cross-arm TTFT/TPOT are "
            "NOT equal-pool comparable in this run.")
        print(f"[e2e] !! KV-POOL PIN FAILED: probe reported 0 gpu_blocks"
              f"{' (' + str(pd['load_err']) + ')' if pd.get('load_err') else ''} -- every "
              f"arm will size its own KV pool. Do not publish cross-arm deltas from this "
              f"run.", flush=True)

    # ── all arms, pinned KV pool ──
    order = ["OMMX", "FA3", "TRITON", "TURBOQUANT", "OMMX_V10A", "abl_bf16", "abl_linear", "abl_attn", "abl_full",
             "abl_full_no_outlier", "abl_full_no_dequant",
             "abl_attn_no_outlier", "abl_attn_no_dequant",
             "abl_attn_no_unpack", "abl_attn_no_dequant_all", "abl_attn_no_all",
             "abl_attn_skipwrite", "abl_attn_mw",
             "OMMX_VINTPV", "OMMX_VLUT", "OMMX_VINTPV_PIPE",
             "abl_attn_bat", "OMMX_VINTPV_BAT", "abl_attn_bat_eager"]
    if only_arms:
        # Arms that exist in the plan but are deliberately absent from the default `order`
        # (currently "FA", the vLLM-default FLASH_ATTN baseline) must still be selectable —
        # otherwise the filter could never reach them. Appended in the order the user asked.
        # Names were validated against `plan` above, so nothing can silently drop here.
        order = ([a for a in order if a in only_arms]
                 + [a for a in only_arms if a not in order])
        print(f"[e2e] only-arms filter -> {order}", flush=True)
    for arm_name in order:
        if arm_name not in plan:
            continue
        p = plan[arm_name]
        coh_ctx = args.coherence_ctx if p.get("coh", p["group"] == "cmp") else 0
        d = _spawn(arm_name, p, pinned_blocks, coh_ctx)
        results["arms"][arm_name] = d.get("cells", {})
        results["provenance"][arm_name] = _arm_provenance(p, d.get("cells", {}),
                                                          capture_sizes)
        if d.get("route_evidence"):
            results["route_evidence"][arm_name] = d["route_evidence"]
        if d.get("coherence"):
            results["coherence"][arm_name] = d["coherence"]

    # ── CSV (comparison + ablation rows, single contract) ──
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    fields = ["arch", "group", "arm", "label", "backend", "quant", "batch", "ctx",
              "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99", "n_steps",
              "mode", "ok", "oom", "err"]
    rows = []
    for arm_name, cellmap in results["arms"].items():
        p = plan[arm_name]
        for key, r in cellmap.items():
            b = int(key.split("_")[0][1:]); c = int(key.split("ctx")[1])
            row = dict(arch=args.arch, group=p["group"], arm=arm_name, label=p["label"],
                       backend=p["backend"], quant=p["quant"], batch=b, ctx=c,
                       mode=r.get("mode", ""), ok=bool(r.get("ok")),
                       oom=bool(r.get("oom")), err=r.get("err", ""),
                       n_steps=r.get("n_steps", ""))
            if r.get("ok"):
                for f in ("ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99"):
                    row[f] = round(r[f], 5)
            rows.append(row)
    rows.sort(key=lambda x: (x["group"], x["arm"], x["ctx"]))
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(1 for r in rows if r.get("ok"))
    print(f"\n[e2e] wrote {args.csv} ({n_ok}/{len(rows)} cells ok)", flush=True)
    print(f"[e2e] wrote {args.json}", flush=True)
    # CUDA-GRAPH PROVENANCE SUMMARY. Printed rather than left in the JSON because the one
    # thing a reader assumes about a fair-compare sweep is that every arm ran the same way,
    # and here one arm deliberately does not. The mismatch line is a real cross-check: the
    # plan says what the arm was spawned as, `observed_modes` is what the worker process
    # itself reported, and they can only disagree if the --eager plumbing broke.
    eager_arms = sorted(a for a, pv in results["provenance"].items() if pv["enforce_eager"])
    print(f"[e2e] cudagraph provenance: capture_sizes="
          f"{results['config']['fair']['cudagraph_capture_sizes_requested']}; "
          f"enforce_eager arm(s)={eager_arms or 'none'}; all other arms CUDA-graph FULL",
          flush=True)
    for a, pv in results["provenance"].items():
        want = "eager" if pv["enforce_eager"] else "graph"
        odd = [m for m in pv["observed_modes"] if m != want]
        if odd:
            print(f"[e2e] !! provenance MISMATCH {a}: plan says {want}, worker cells "
                  f"report {odd} -- do not cite this arm", flush=True)
    # route-fired summary: every CUSTOM arm either proved the OMMX kernel ran or had all
    # of its cells invalidated above. Printed so the failure is visible in the run log.
    for a, ev in results["route_evidence"].items():
        print(f"[e2e] route_evidence {a}: ok={ev['ok']} fired={ev['fired']} "
              f"nofire={ev['nofire']} procs={ev['procs_fired']}/{ev['tp']}"
              f"{'' if ev['ok'] else ' REASON=' + ev['reason']}", flush=True)
    # coherence summary: TRITON is the only accepted baseline for this attention-only path.
    base = results["coherence"].get("TRITON")
    for a, ids in results["coherence"].items():
        match = (ids[:16] == base[:16]) if base else None
        print(f"[e2e] coherence {a}: match_vs_TRITON={match} ids[:8]={ids[:8]}", flush=True)
    print("BENCH_E2E_A100_DONE", flush=True)


if __name__ == "__main__":
    main()
