# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""vLLM v1 attention backend for the canonical OMMX paged decode (SHADOW mode).

⚠ CLUSTER-VALIDATION: imports vLLM v1 internals at module load (subclasses
``FlashAttentionBackend`` / ``Impl`` / ``MetadataBuilder``). Loaded lazily by the
registry only when ``--attention-backend CUSTOM`` is selected; NOT importable
without vLLM. Validated against vLLM 0.21 (the ``ommx`` cluster env).

SHADOW mode, EAGER single-batch (step1 first slice):
  * vLLM keeps its bf16 paged KV cache (inherited ``get_kv_cache_shape``) and
    serves PREFILL / mixed / non-uniform steps with stock FlashAttention.
  * OMMX keeps a per-layer ``CanonicalKVStore`` sidecar (the canonical i2f4 K /
    i2 V planes + bf16 sink/recent window). ``do_kv_cache_update`` appends the new
    bf16 K/V into the sidecar (and regroups completed 32-token groups); ``forward``
    routes UNIFORM SINGLE-TOKEN DECODE through ``ommx_gpu_serve::paged_decode`` and
    falls back to bf16 FlashAttention otherwise.

SLIDING-WINDOW LAYERS ARE NEVER OMMX-ROUTED, AND SAY SO. The plugin registers CUSTOM
for whatever model is loaded, but every OMMX route here is gated on
``sliding_window == (-1, -1)``: an alternating SWA/full model (Mistral, Gemma-2,
Ministral) gets OMMX on its full-attention layers and stock bf16 FlashAttention, out of
vLLM's own paged cache, on its sliding-window ones. Correct, but PARTIAL — and it used
to be invisible, because the sentinel of such a run looked exactly like a fully routed
Llama run. Both halves of the carve-out now record evidence (``SW_BYPASS_BF16`` on the
write, ``SW_BYPASS_BF16_READ`` on the read, plus
``ommx_route_health()["sliding_window_bypass"]``), so "OMMX served this model" and
"OMMX served part of this model" are distinguishable after the fact.

CUDA-graph: declared ``UNIFORM_SINGLE_TOKEN_DECODE``; the eager slice runs with
``enforce_eager=True`` (the sidecar append uses a python seq index — NOT yet a
slot-mapping scatter, so it is not capture-safe). FULL-graph capture + the paged
multi-batch store is the follow-up (the build-seam scatter in ``metadata.py``).

PACKED-ONLY capacity mode (the KV compression — the headline OMMX benefit over FA3)
is implemented in ``packed_only.py`` and gated by ``OMMX_KV_PACKED_ONLY=1`` (default
OFF; SHADOW stays the correctness-first baseline). It patches
``Attention.get_kv_cache_spec`` so the bf16 paged-cache page budget shrinks by the
OMMX-vs-bf16 byte ratio -> vLLM budgets proportionally more KV blocks.

  MEASURED KV FOOTPRINT — RETRACTS THE "≤3-bit" / "~4.6x" CLAIMS THIS DOCSTRING USED
  TO MAKE. Obtained by SUMMING THE REAL ALLOCATED ``MultiSeqKVPool`` TENSORS (not by
  evaluating a formula), Llama-3.1-8B geometry:
    canonical PUBLISHED recipe — i2f4, 6 outliers/vector, relative-index 7,
      group_tokens=32, group_channels=32, OMMX_KV_OUTLIER_MAP=1, pow2 (int8 scale):
        K+V = 8.750 bit per K/V element PAIR = 4.375 bit/elem -> 32/8.75 = 3.66x
    same recipe with OMMX_KV_OUTLIER_MAP=0:
        K+V = 7.750 -> 3.875 bit/elem -> 4.13x
    group_tokens=64 + group_channels=64 + OMMX_KV_OUTLIER_MAP=0:
        K+V = 5.875 -> 2.938 bit/elem -> 5.45x   (the ONLY "≤3-bit" configuration)
  So the recipe the ACCURACY results were produced with is 4.375 bit/elem = 3.66x.
  Reaching ≤3 bit needs group_tokens=64 + group_channels=64 + OMMX_KV_OUTLIER_MAP=0
  (at group_channels=32 the same knobs give 6.250 bit/pair = 3.125 avg, NOT ≤3),
  i.e. a DIFFERENT number system from the one the accuracy numbers used — the
  compression figure and the accuracy figure must never be quoted from different
  recipes in one sentence.
  (What vLLM applies to the page budget: PACKED-ONLY calls
  ``packed_only.ommx_bits_per_elem(head_size)`` with NO keyword overrides, so the recipe
  comes from the ENV — the same env the pool itself reads. Under the published recipe
  (``OMMX_ATTN_OUTLIERS=6 OMMX_ATTN_POW2=1 OMMX_KV_GROUP_TOKENS=32
  OMMX_KV_GROUP_CHANNELS=32``) that returns K 6.000 / V 2.750 = 8.750 bit = 3.657x,
  IDENTICAL to the measured plane footprint above — verified by calling the function and
  by summing the real ``MultiSeqKVPool`` tensors independently.
  Two ways to get a different number out of it, both of which mean the env was not the
  published recipe: with NOTHING set it falls back to k=3 + bf16 scale (8.250 bit,
  3.88x), and at k=6 but ``use_pow2`` unset it gives 9.250 bit (3.46x) because the scale
  reverts from an int8 pow2 exponent to bf16. Always state which recipe a ratio belongs
  to; ``packed_only.kv_bits_breakdown()`` prints the resolved recipe alongside the
  number for exactly this reason.)

The INT2 sidecar is the real backing store + decode op; the SHRUNK bf16 paged cache
is a BYTE-BUDGET RESERVATION ONLY — never written (``do_kv_cache_update`` skips the
paged write) and never read (a full-prompt prefill runs varlen FlashAttention
directly on the in-batch q/k/v; the KIVI sink/recent residual lives in the store's
own buffers). Because it is a reservation that never stores a token, vLLM's own
"GPU KV cache size: <N> tokens" / "Maximum concurrency <Y>x" log lines merely RESTATE
that reservation — they are NOT a validated OMMX capacity result, and quoting them as
one attributes to OMMX a number vLLM computed from a head_size we asked it to shrink.
The only capacity claim backed by measurement is the pool-tensor byte count above.
Steps neither packed route can serve raise loudly — over the shrunk pages there is no
valid bf16 fallback.

FAILURE POLICY (law #5, NO SILENT FALLBACK) — DEFAULT CHANGED: an OMMX route failure
is now FATAL. Every OMMX route (the sidecar write in ``do_kv_cache_update``, and the
single-batch / batched / batched-graph decode reads) used to answer a failure by
latching ``layer._ommx_dead`` and returning False, which handed the step to bf16
FlashAttention. Measured consequence on vLLM 0.21, whose defaults are
``enable_prefix_caching=True`` (vllm/config/cache.py:91) and
``enable_chunked_prefill=True`` (vllm/config/scheduler.py:84) — BOTH unsupported by
the non-paged sidecar: a Llama-3.1-8B run with ``--attention-backend CUSTOM`` wrote
``KV_UPDATE_DEAD`` to the sentinel and then produced output BYTE-IDENTICAL to the
bf16 reference (228/228 chars), while the same config with prefix caching OFF
produced a genuinely different (OMMX) continuation. A fluent, fast, wrong-backend run
is the worst possible failure mode for a benchmark, so it no longer happens silently:
the evidence is recorded and the exception RE-RAISES. ``OMMX_ALLOW_BF16_FALLBACK=1``
opts back into degrading, and then does so LOUDLY (one-time stderr+logger banner,
``_DEGRADED`` latch, ``ommx_route_health()["degraded"] == True`` for a bench to
assert on). ``OMMX_STRICT=1`` (``cfg.strict``) stays the STRONGER knob: it raises even
when the opt-in is set.
"""
from __future__ import annotations

import sys

import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)

import os

from ommx_gpu_serve.attention.kv_store import CanonicalKVStore
from ommx_gpu_serve.attention.paged_decode import (
    _auto_num_kv_splits,
    ommx_paged_decode_attention_canonical,
)
from .config import resolve_serving_config
from .metadata import (STEP_ROUTE_UNCLASSIFIED, OMMXBatchedStepManager,
                       OMMXStepManager, detect_full_prefill, detect_step_route)
from .packed_only import packed_only_enabled
from .preflight import ommx_preflight_check

OMMX_BACKEND_CLASS_PATH = "ommx_gpu_serve.integration.vllm.backend.OMMXCanonicalBackend"


def _env_on(name: str) -> bool:
    """Route-env truthiness: ``0|false|off|no|<empty>`` = OFF, anything else = ON.

    THE SET IS THE PACKAGE-WIDE ONE and ``"no"`` is in it on purpose. This helper used
    to omit ``"no"`` while ``config._env_bool``, ``metadata._env_on``,
    ``kv_pool``/``kv_store``'s pool flags and ``plugin.py`` all included it, so ONE
    spelling meant opposite things depending on which module read it. What that bought,
    concretely: ``OMMX_ALLOW_BF16_FALLBACK=no`` ENABLED the bf16 degrade (the operator
    had just written the word "no"), and ``OMMX_ABL_SKIP_WRITE=no`` turned ON a
    PARITY-BREAKING ablation whose whole effect is to make the decode read stale KV.
    Both are safety-critical knobs, so the divergence is now closed here rather than
    documented as a quirk.

    ONE MIRROR IS STILL STALE, DELIBERATELY LEFT FOR ITS OWNER:
    ``preflight._backend_env_on`` copies the OLD set verbatim (its docstring names this
    function as the authority) and ``tests/test_preflight_guards.py`` pins that copy by
    parametrizing ``OMMX_ATTN_BATCHED="no"``. Until those two are updated together,
    preflight reads ``"no"`` as ON where this reads it OFF — i.e. preflight assumes a
    B>1 step is reachable when the backend would not batch, so it can only REFUSE a run
    the backend would have served. That direction is fail-closed (a refusal costs a
    re-run; the reverse would publish a FlashAttention number as OMMX), which is why it
    is safe to land this half first.
    """
    return os.environ.get(name, "0").strip().lower() not in {
        "", "0", "false", "off", "no"}


def _env_int(name: str, default: int) -> int:
    """Integer env knob. Unset/blank -> ``default``; MALFORMED -> ``ValueError``.

    Raising is the package law (``packed_only._env_int``: "a malformed value RAISES
    (law: no silent fallback)"; ``config._env_int`` parses unguarded and so raises too).
    This helper used to swallow the ``ValueError`` and substitute ``default``, which on
    its one consumer -- ``OMMX_ATTN_MAX_NUM_SEQS``, a DIRECT HBM MULTIPLIER (the pool is
    ``num_seqs * max_model_len`` bytes per layer, eagerly allocated) -- turned a typo
    into a silent 256-slot pool: the operator asked for a small pool and got the 256-slot
    one instead — which at vLLM's defaults (max_model_len=131072, Llama-3.1-8B geometry)
    is the 35.11 GiB/layer x 32 = 1123.5 GiB projection recorded at
    ``_MAX_NUM_SEQS_DEFAULT`` below, i.e. the first B>1
    step OOMs inside the write path, where a dead-latch used to hand every subsequent
    step to bf16 FlashAttention. The error names the variable and the value so the fix is
    obvious from the traceback.
    """
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[ommx] {name}={raw!r} is not an integer. Unset it to use the default "
            f"({int(default)}), or give it a whole number. Refusing to substitute the "
            "default for a malformed value: this knob sizes an eagerly allocated "
            "per-layer pool, so a silently ignored value is an OOM, not a typo."
        ) from exc


# CUDA-graph-capable path (capture-safe write + host build-seam). Default OFF so the
# validated eager append path stays the baseline; set OMMX_ATTN_GRAPH=1 to develop /
# enable the graph route (cluster-validated eager-first, then enforce_eager=False).
_GRAPH = _env_on("OMMX_ATTN_GRAPH")

# BATCHED (B>1) shared-pool decode is now a B-AWARE CORRECTNESS ROUTE, not an env gate.
# A B>1 uniform decode step ALWAYS uses the per-layer MultiSeqKVPool (read+write); a
# B==1 step ALWAYS uses the single-batch graph/eager path. The single-batch path slices
# query[:1]/output[:1] and the write path's n>1 branch is a PREFILL reset+append, so
# routing a B>1 decode through them corrupts requests 1..B-1 (the confirmed garbage).
# OMMX_ATTN_BATCHED=1 is now only a FORCE-OVERRIDE: route even a B==1 decode through the
# batched pool (eager-only; do not combine with OMMX_ATTN_GRAPH). pool cap =
# _resolved_max_num_seqs().
_BATCHED = _env_on("OMMX_ATTN_BATCHED")
# CUDA-graph-CAPTURABLE batched (B>1) decode. When ON, a B>1 uniform-decode session
# routes the WRITE + READ through the fixed-address device-buffer paths
# (_batched_kv_update_captured / _route_decode_batched_graph) that read ONLY the host-
# refreshed device buffers (no per-step host python -> torch.tensor), so the batched
# attention+GEMM region is CUDA-graph capturable (enforce_eager=False). The regroup
# (CPU pack of completed 32-token groups) stays in build() (HOST, outside capture) —
# the small out-of-graph step. Closes the eager-vs-fullgraph B>1 TPOT gap. Default OFF
# (the validated eager batched path stays the correctness baseline).
_BATCHED_GRAPH = _env_on("OMMX_ATTN_BATCHED_GRAPH")
# ── OMMX pool slot cap (num_seqs) ─────────────────────────────────────────────
# ``MultiSeqKVPool`` is NOT paged: it allocates O(num_seqs * max_model_len) bytes PER
# LAYER, EAGERLY, at the first B>1 step — so this number is a direct HBM multiplier,
# and a hard-coded 256 was the measured cause of a real OOM. At vLLM defaults
# (max_model_len=131072, Llama-3.1-8B geometry) 256 slots project to 35.11 GiB per
# layer x 32 layers = 1123.5 GiB; the observed failure on an H100 NVL was
# "CUDA out of memory. Tried to allocate 514.00 MiB ... 383.81 MiB is free", after
# which the write path latched dead and the engine silently served bf16 (see the
# FAILURE POLICY paragraph in the module docstring).
# The engine ALREADY knows the real concurrency ceiling — it is
# ``vllm_config.scheduler_config.max_num_seqs``, the most requests the scheduler will
# ever run at once. Resolution is a MIN, NOT a precedence chain — write it as the
# arithmetic it is, because the "env > engine > 256" shorthand has already produced two
# self-contradicting operator messages (preflight.py repeats the warning):
#     cap = max(1, min(OMMX_ATTN_MAX_NUM_SEQS or 256, scheduler_config.max_num_seqs))
# So the env can only LOWER the cap: OMMX_ATTN_MAX_NUM_SEQS=512 under --max-num-seqs 128
# resolves to 128, not 512. NEVER exceeding the engine value once it is known: slots the
# scheduler can never fill are pure wasted HBM, and that waste is exactly what OOMs.
# Floored at 1.
# CONSEQUENCE FOR metadata.py's ERROR TEXT: the min() means the env can no longer push
# the cap ABOVE the engine's max_num_seqs, so metadata.py's slot-exhaustion advice
# ("raise OMMX_ATTN_MAX_NUM_SEQS to at least / above the engine's max_num_seqs") now
# tops out at exactly the engine value. That is enough for the exhaustion case (the
# scheduler can never make more than max_num_seqs requests live at once, so cap ==
# max_num_seqs always suffices); the remaining remedy for a dummy/capture batch that
# still cannot find B free slots is to LOWER --max-num-seqs, not to raise the env.
# Every consumer (``_bmanager`` / ``_ommx_ws_batched`` / ``_preflight_once``) MUST read
# ``_resolved_max_num_seqs()``. There is deliberately NO raw module constant holding
# the effective value, so a future consumer cannot re-introduce the 256 by reading a
# stale name.
_MAX_NUM_SEQS_DEFAULT = 256
# The EXPLICIT operator override, or None when the env is unset/blank (then the engine
# value decides). A MALFORMED value now RAISES out of ``_env_int`` at import (see that
# helper): this is a direct HBM multiplier, so substituting the 256 default for a typo
# allocates the pool the operator was trying to shrink.
_MAX_NUM_SEQS_ENV = (
    _env_int("OMMX_ATTN_MAX_NUM_SEQS", _MAX_NUM_SEQS_DEFAULT)
    if str(os.environ.get("OMMX_ATTN_MAX_NUM_SEQS", "")).strip() != "" else None)
# ``scheduler_config.max_num_seqs``, captured ONCE by OMMXCanonicalMetadataBuilder
# (the earliest object holding the whole vllm_config). Stays None if unreadable, and
# then the resolution falls back to the env/default exactly as before.
_ENGINE_MAX_NUM_SEQS = None


def _resolved_max_num_seqs() -> int:
    """The pool slot cap actually used::

        max(1, min(OMMX_ATTN_MAX_NUM_SEQS or 256, scheduler_config.max_num_seqs))

    A MIN, not a precedence chain: the env is an override only in the DOWNWARD
    direction. With the engine at ``--max-num-seqs 128``, ``OMMX_ATTN_MAX_NUM_SEQS=512``
    resolves to 128 — the env does not win. (The "env > engine > 256" prose this
    docstring used to carry is the shorthand that produced two contradicting operator
    messages; ``preflight.py`` keeps a comment block warning against re-introducing it.)
    Capped by the engine's ``max_num_seqs`` whenever that is known (a pool larger than
    the scheduler can ever fill is wasted HBM), floored at 1 (``MultiSeqKVPool``
    clamps <=0 to 1 and would then alias every request onto slot 0).
    """
    v = _MAX_NUM_SEQS_DEFAULT if _MAX_NUM_SEQS_ENV is None else int(_MAX_NUM_SEQS_ENV)
    if _ENGINE_MAX_NUM_SEQS is not None:
        v = min(int(v), int(_ENGINE_MAX_NUM_SEQS))
    return max(1, int(v))

# OMMX_ATTN_V_BF16=1 -> the v10a path: K stays i2f4, V is the FULL bf16 value (no V
# quant/dequant). The canonical kernel auto-selects the StoreBf16 VFmt branch from the
# bf16 v_main dtype. Removes the V=i2 per-element dequant ALU (the M=1 decode loss cause)
# at the cost of 8x the V bytes (== FA3's V bytes) — K-quantized, V-full attention.
_V_BF16 = _env_on("OMMX_ATTN_V_BF16")
# ABLATION-ONLY (differential-timing; PARITY-BREAKING) — OMMX_ABL_SKIP_WRITE=1 makes a
# DECODE step SKIP the new-token KV pack+regroup/write entirely (the attention then reads
# STALE KV -> WRONG output). Isolates the per-step KV pack/write cost:
#   (full_step) - (SKIP_WRITE_step) = the per-step KV pack/write cost.
# Decode-only (n==1); prefill writes are KEPT (else the cache is empty). NOT production;
# default OFF. Mirrors the kernel-side OMMX_ABL_* timing flags.
# IT ANNOUNCES ITSELF WHEN ON (``_announce_parity_breaking_ablations``, called at the
# first write): a flag whose entire purpose is to make the output WRONG must never be
# silent. Before that, the skip was a bare ``return`` — no banner, no sentinel line — so
# the only trace of a knowingly-wrong run was the env of the process that launched it,
# and the sentinel still showed a clean DECODE_ROUTE_FIRED.
_ABL_SKIP_WRITE = _env_on("OMMX_ABL_SKIP_WRITE")
# PACKED-ONLY capacity mode (OMMX_KV_PACKED_ONLY=1, packed_only.py). The spec patch
# shrinks the bf16 paged-cache page budget, so the pages CANNOT hold real headdim-D
# K/V rows — they are a byte-budget reservation. In this mode the backend (a) never
# writes the paged cache, (b) serves a FULL-PROMPT prefill with varlen FlashAttention
# over the in-batch q/k/v, (c) serves uniform decode from the OMMX sidecar, and
# (d) raises on any other step (a bf16 fallback would read the shrunk pages ->
# garbage/OOB, and law #5 forbids doing that silently). OFF -> byte-identical SHADOW.
_PACKED_ONLY = packed_only_enabled()
# vLLM cudagraph CAPTURE latch (set only inside build_for_cudagraph_capture). During
# capture, vLLM drives a SYNTHETIC dummy decode (no real requests / no OMMX pool), so
# the OMMX route can't fire and PACKED has no bf16 fallback (shrunk cache). Attention is
# a vLLM splitting_op -> it runs EAGER (outside the captured pieces) at serve, so the
# captured attention op is a placeholder: a zero-fill stub lets capture proceed while the
# REAL OMMX decode still runs eager at serve. Cleared in finally (law #5: serve steps,
# with the latch False, still raise loud rather than silently zero-serve).
_CAPTURING = [False]
# Flips True on the FIRST OMMX route fire — ANY ``*_FIRED`` tag: the decode routes
# (BATCHED/DECODE/GRAPH) and the PACKED-ONLY prefill (PACKED_PREFILL_FIRED), which the
# older ``"ROUTE_FIRED" in tag`` test did not match. Before that, the process is in vLLM
# INIT (determine_available_memory memory-profiling dummy + cudagraph capture) —
# synthetic steps with no OMMX pool that the packed guard must STUB (not raise), because
# is_current_stream_capturing() is False for the eager memory-profiling dummy. The
# profiling dummy cannot fire a tag (it runs with attn_metadata None, which returns early
# in super()), so widening the latch to ``*_FIRED`` does not shorten the init window; it
# only stops a REAL packed prefill from leaving the process looking un-started.
# After real serving starts, the guard raises on genuine anomalies (law #5).
_REAL_SERVE_STARTED = [False]
# Worker-side batched-route fire counter (mutable module cell; proves the OMMX
# batched decode actually ran in the worker process — see _route_decode_batched).
_BATCHED_FIRES = [0]
# Worker-side packed-prefill fire counter (mirrors _BATCHED_FIRES; proves the
# PACKED-ONLY in-batch varlen prefill ran — see _prefill_packed_varlen).
_PACKED_PREFILL_FIRES = [0]
# Cross-process route-fired evidence: a spawn worker's raw stdout/stderr is NOT
# forwarded to the driver, so write a sentinel FILE (definitive) + log via vLLM's
# logger (forwarded). Path overridable for the bench to read back.
_FIRE_FILE = os.environ.get("OMMX_FIRE_FILE", "/tmp/ommx_route_fired.log")
_FIRE_SEEN: set = set()

# ── DEGRADE POLICY (law #5: NO SILENT FALLBACK) ───────────────────────────────
# OFF by default: an OMMX route failure RE-RAISES and takes the engine down. See the
# FAILURE POLICY paragraph in the module docstring for the run that forced this
# default (a CUSTOM-backend run that emitted output byte-identical to bf16 because a
# KV_UPDATE_DEAD latch had quietly handed every step to FlashAttention).
# OMMX_ALLOW_BF16_FALLBACK=1 opts back into the degrade — LOUDLY, never silently.
_ALLOW_BF16_FALLBACK = _env_on("OMMX_ALLOW_BF16_FALLBACK")
# Latched on the FIRST degrade (reachable only with the opt-in above). Read it through
# ommx_route_health(); a bench MUST assert degraded is False before quoting a number.
_DEGRADED = {"reason": None, "layer": None, "step_hint": None}
# One-time banner latch (per worker process). The banner is the thing a human sees;
# _DEGRADED is the thing a script sees.
_DEGRADE_BANNER_SHOWN = [False]


def _ommx_local_rank() -> int:
    """TP/PP local rank of this worker process (0 if single-GPU / unknown).

    vLLM v1 runs each TP rank in its OWN EngineCore subprocess, so the only
    cross-process proof that EVERY rank fired the OMMX decode (and not a silent
    bf16 FlashAttention fallback on one rank) is a per-rank sentinel. Derive the
    rank from vLLM's distributed state, falling back to the launcher env."""
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank)
        return int(get_tensor_model_parallel_rank())
    except Exception:
        for k in ("LOCAL_RANK", "RANK"):
            v = os.environ.get(k)
            if v is not None and v.strip().lstrip("-").isdigit():
                return int(v)
    return 0


def _ommx_route_evidence(tag: str, detail: str = "") -> None:
    """Record one-time PER-RANK route evidence to the sentinel file (+vLLM log).

    TAG NAMING IS LOAD-BEARING — three consumers key off the SUFFIX:
      ``*_DEAD`` / ``*_NOFIRE``  a route was asked to serve and did not. Both
          ``_sentinel_dead_tags`` and ``bench_e2e_a100.py::_is_nofire_tag`` read these
          as FAILURE (ok=False). Never give an informational tag one of these suffixes.
      ``*_FIRED``  an OMMX route really served a step -> latches ``_REAL_SERVE_STARTED``
          below, i.e. "this process is past vLLM init". A tag that is NOT proof of a
          real serve step must not end in ``_FIRED``.
      anything else  informational; the bench buckets it into ``other_tags`` and it
          changes no verdict (``SW_BYPASS_BF16*``, ``ABL_SKIP_WRITE_ACTIVE``,
          ``PACKED_CAPTURE_STUB``).
    """
    if tag.endswith("_FIRED"):          # an OMMX route really served -> past vLLM init
        # Was ``"ROUTE_FIRED" in tag``, which missed PACKED_PREFILL_FIRED — the only
        # fire evidence a PACKED-ONLY prefill produces. The latch therefore stayed False
        # through a real prefill, and the packed guard below then treated the NEXT real
        # step as a capture dummy and ZERO-FILLED it. Suffix matching cannot drift when
        # a route tag is added (same reasoning as the bench's ``_NOFIRE``/``_DEAD``).
        _REAL_SERVE_STARTED[0] = True
    r = _ommx_local_rank()
    key = f"{tag}.r{r}"  # dedup per (tag, rank) so each TP rank fires once
    if key in _FIRE_SEEN:
        return
    _FIRE_SEEN.add(key)
    line = f"{tag} rank={r} pid={os.getpid()} {detail}".rstrip()
    # Rank 0 keeps the base path (backward-compat with single-GPU firing greps);
    # ranks > 0 write a rank-suffixed file so concurrent TP workers never clobber
    # and each rank's proof is independently checkable.
    path = _FIRE_FILE
    if r > 0:
        base, ext = os.path.splitext(_FIRE_FILE)
        path = f"{base}.r{r}{ext}"
    try:
        with open(path, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass
    try:
        from vllm.logger import init_logger
        init_logger("ommx_gpu_serve").info("[ommx] %s", line)
    except Exception:
        pass


# Sliding-window layers this process served from the bf16 paged cache instead of OMMX:
# id(layer) -> that layer's ``sliding_window``. PER PROCESS, like every other counter in
# this module; the CROSS-process fact is the ``SW_BYPASS_BF16*`` sentinel tag.
_SW_BYPASS_LAYERS: dict = {}
# (tag, id(impl)) pairs already recorded — the per-step early-out (see below).
_SW_BYPASS_SEEN: set = set()


def _sw_bypass_evidence(tag: str, impl, detail: str = "") -> None:
    """Record that a SLIDING-WINDOW layer is being served by bf16, not by OMMX.

    WHY A TAG AT ALL. A mixed SWA/full-attention model (Mistral, Gemma-2, Ministral)
    runs its full-attention layers through OMMX and its sliding-window layers through
    stock FlashAttention over vLLM's bf16 paged cache — correct output, but only PART of
    the model is an OMMX measurement. Without evidence, the sentinel of such a run is
    byte-identical to a fully-OMMX-routed Llama run (``DECODE_ROUTE_FIRED`` and nothing
    else), so partial and total coverage were indistinguishable and the bench reported
    ok=True either way. This tag is the difference.

    WHY IT IS *NOT* A ``*_DEAD`` / ``*_NOFIRE`` TAG. Those suffixes mean "an OMMX route
    was asked to serve a step and failed", which sets ok=False in
    ``bench_e2e_a100.py``/``ommx_route_health``. The sliding-window carve-out is a
    DESIGNED, correct route (the read gates in ``forward`` deliberately exclude these
    layers), so flagging it as a failure would refuse benches that are working exactly
    as specified. It is informational: it tells the reader that OMMX's share of this run
    is partial, and leaves the verdict to them.

    Dedup is per ``(tag, rank)`` in the sentinel, so the FILE carries one line per rank
    per path, not one per layer — the exact per-layer count lives in
    ``ommx_route_health()["sliding_window_bypass"]`` (per process).

    IT IS ON THE PER-STEP DECODE PATH, so the ``(tag, impl)`` early-out below is not
    tidiness: a sliding-window layer calls this on EVERY step, and everything past the
    set lookup (the rank probe, the f-string) would otherwise be per-step TPOT cost in
    the one model family this branch exists for.
    """
    seen_key = (tag, id(impl))
    if seen_key in _SW_BYPASS_SEEN:
        return
    _SW_BYPASS_SEEN.add(seen_key)
    sw = getattr(impl, "sliding_window", None)
    # ``impl`` is the per-layer FlashAttentionImpl (vLLM builds one per attention
    # layer), so its id is a stable per-layer identity for the count below.
    _SW_BYPASS_LAYERS[id(impl)] = sw
    _ommx_route_evidence(
        tag,
        f"sliding_window={sw} impl={type(impl).__name__}@{id(impl):#x} {detail} "
        "(one-time; EVERY sliding-window layer of this model takes the bf16 path, so "
        "this run's OMMX coverage is PARTIAL - see ommx_route_health()"
        "['sliding_window_bypass'])".rstrip())


# B values (>1) whose uniform-decode step reached the bf16 fall-through instead of an
# OMMX route. PER PROCESS; the cross-process fact is the DECODE_BF16_UNROUTED tag.
_UNROUTED_DECODE_B: set = set()


def _decode_unrouted_evidence(B: int, *, graph: bool) -> None:
    """Record that a B>1 uniform decode was served by bf16, not by OMMX.

    Companion to ``_sw_bypass_evidence`` for the OTHER designed bf16 carve-out on the
    read path (see the ``else`` arm in ``forward``). Informational for the same reason:
    the output is correct, only its ATTRIBUTION would be wrong, and a failure suffix
    would fire on vLLM's own graph-capture dummies.

    Deduped on ``B`` (a set lookup) BEFORE anything else, because this sits on the
    per-step decode path — the rank probe and the f-string inside
    ``_ommx_route_evidence`` must not be paid every step.
    """
    if B in _UNROUTED_DECODE_B:
        return
    _UNROUTED_DECODE_B.add(int(B))
    _ommx_route_evidence(
        "DECODE_BF16_UNROUTED",
        f"B={B} graph={bool(graph)}: a uniform single-token decode of {B} requests was "
        "served by bf16 FlashAttention (no OMMX route accepts it: the batched pool is "
        "not active for this step and the single-batch route would answer only request "
        "0). This run's OMMX coverage is PARTIAL - see ommx_route_health()"
        "['unrouted_decode_b'].")


# One-time banner latch for the parity-breaking ablation knobs read in THIS module.
_ABL_BANNER_SHOWN = [False]


def _announce_parity_breaking_ablations() -> None:
    """Announce ONCE, loudly, that a PARITY-BREAKING ablation knob is active.

    ``OMMX_ABL_SKIP_WRITE`` makes the decode read stale KV on purpose, to price the
    per-step KV pack/write by difference. Its output is WRONG by construction. It used
    to take effect through a bare ``return`` with no banner and no sentinel line, so a
    run launched with it looked — in the sentinel, in the logs, and in
    ``ommx_route_health()`` — exactly like a clean one, and only the launching env said
    otherwise. That is the precise failure mode this codebase's law #5 exists to forbid,
    and a knob that deliberately breaks parity is the last place to make an exception.

    The tag is informational by design (see ``_sw_bypass_evidence`` for the suffix
    rules): ``bench_e2e_a100.py`` runs ``abl_attn_skipwrite`` as a declared
    ``coh=False`` arm, and giving it a ``*_DEAD`` suffix would turn that intended
    measurement into a reported route failure.
    """
    if _ABL_BANNER_SHOWN[0] or not _ABL_SKIP_WRITE:
        return
    _ABL_BANNER_SHOWN[0] = True
    _ommx_route_evidence(
        "ABL_SKIP_WRITE_ACTIVE",
        "decode KV pack/write SKIPPED - the decode reads STALE KV, so the OUTPUT OF "
        "THIS RUN IS WRONG BY CONSTRUCTION (timing-only ablation)")
    bar = "=" * 78
    banner = (
        f"\n{bar}\n"
        "[ommx] OMMX_ABL_SKIP_WRITE IS ON - THIS RUN'S OUTPUT IS DELIBERATELY WRONG.\n"
        "  Every decode step skips the new-token KV pack/regroup/write, so the kernel\n"
        "  attends STALE KV. The run stays fluent and fast; its TPOT is meaningful\n"
        "  ONLY as (full step) - (this step) = the per-step KV pack/write cost.\n"
        "  NOTHING from it may be quoted as an accuracy, parity or capacity result.\n"
        f"  route evidence: {_FIRE_FILE} (tag ABL_SKIP_WRITE_ACTIVE)\n"
        "  programmatic check: ommx_route_health()['ablations']['abl_skip_write']\n"
        f"{bar}\n")
    sys.stderr.write(banner)
    sys.stderr.flush()
    try:
        # Best-effort duplicate on the only stream a spawn worker forwards to the
        # driver; stderr above is the authoritative copy (mirrors the degrade banner).
        from vllm.logger import init_logger
        init_logger("ommx_gpu_serve").error("%s", banner)
    except Exception:  # noqa: BLE001
        pass


def _ommx_route_failed(tag: str, exc: BaseException, *, cfg=None, layer=None,
                       step_hint: str = "") -> None:
    """Handle a failure inside an OMMX route: record evidence, then RAISE by default.

    Precedence, strongest knob FIRST (this ordering is deliberate):
      1. ``cfg.strict`` (OMMX_STRICT=1) -> ALWAYS re-raise, even when the bf16 opt-in
         below is set. ``strict`` is the stronger knob, so a run that explicitly asked
         to fail hard cannot be talked out of it by a second, weaker env.
      2. ``OMMX_ALLOW_BF16_FALLBACK`` unset (THE DEFAULT) -> re-raise. Degrading here is
         precisely what let a CUSTOM-backend run produce fluent, fast, byte-identical-
         to-bf16 output that a bench would then have published as an OMMX result.
      3. ``OMMX_ALLOW_BF16_FALLBACK=1`` -> do not raise: latch ``_DEGRADED``, print a
         one-time banner to stderr AND vLLM's logger, and let the caller fall back.
    The route evidence is appended FIRST in every case, so ``$OMMX_FIRE_FILE`` names the
    cause whether the process dies here or limps on. Re-raising ``exc`` (rather than a
    new error) preserves the original traceback and message.

    HISTORICAL GAP IN THE PUBLISHED VERIFIER -- NOW CLOSED (kept as the reason the
    bench's rule is shaped the way it is; do not "simplify" it back). bench/
    bench_e2e_a100.py used to decide "is this an OMMX measurement?" with an EXACT tuple
    membership test over ``("DECODE_ROUTE_NOFIRE", "BATCHED_ROUTE_NOFIRE",
    "KV_UPDATE_DEAD")``. Of the tags THIS function writes, only ``KV_UPDATE_DEAD`` was
    in that tuple, so the four read-path tags -- ``DECODE_ROUTE_DEAD``,
    ``GRAPH_ROUTE_DEAD``, ``BATCHED_ROUTE_DEAD``, ``BATCHED_GRAPH_ROUTE_DEAD`` -- fell
    into the bench's ``other_tags`` bucket and did NOT set ok=False. With
    ``OMMX_ALLOW_BF16_FALLBACK=1`` a run that fired once at capture and then died on
    every real step still reported ``route_fired=True``. The bench now matches by
    SUFFIX (``bench_e2e_a100.py::_is_nofire_tag`` -> ``_NOFIRE_SUFFIXES =
    ("_NOFIRE", "_DEAD")``), which cannot drift out of sync when a new route tag is
    added here. Both driver-side checks are therefore trustworthy now:
    ``ommx_route_health()`` and the bench's ok flag. If you ADD a failure tag, keep the
    ``_DEAD`` / ``_NOFIRE`` suffix or the bench will not see it.
    """
    _ommx_route_evidence(tag, f"{type(exc).__name__}: {exc}")
    if layer is not None:
        # Kept even when we are about to raise: if a caller upstream ever chooses to
        # swallow the exception, the latch still stops this layer from being trusted.
        setattr(layer, "_ommx_dead", True)
    # cfg is None when the caller could not produce one (e.g. the write path passes the
    # CACHED _ommx_serving_cfg, which is unset if cfg construction is what failed). The
    # getattr default then reads as "not strict", and the DEFAULT policy below raises
    # anyway — the only way to reach the fall-through is strict False AND the opt-in set.
    strict = bool(getattr(cfg, "strict", False))
    if strict or not _ALLOW_BF16_FALLBACK:
        raise exc
    if _DEGRADED["reason"] is None:
        _DEGRADED["reason"] = f"{tag}: {type(exc).__name__}: {exc}"
        _DEGRADED["layer"] = (f"{type(layer).__name__}@{id(layer):#x}"
                              if layer is not None else None)
        _DEGRADED["step_hint"] = step_hint or (
            f"B={_STEP.get('B')} batched_read={_STEP.get('batched_read')} "
            f"batched_write={_STEP.get('batched_write')}")
    if not _DEGRADE_BANNER_SHOWN[0]:
        _DEGRADE_BANNER_SHOWN[0] = True
        bar = "=" * 78
        banner = (
            f"\n{bar}\n"
            "[ommx] DEGRADED TO bf16 - THIS RUN IS NO LONGER AN OMMX MEASUREMENT.\n"
            f"  cause: {_DEGRADED['reason']}\n"
            f"  step : {_DEGRADED['step_hint']}\n"
            f"  layer: {_DEGRADED['layer']}\n"
            "  From here the affected layer serves bf16 FlashAttention out of vLLM's\n"
            "  paged cache. The output stays fluent and the latency stays plausible,\n"
            "  so NOTHING downstream will look wrong - and every latency, quality or\n"
            "  compression number taken from this run describes FlashAttention, not\n"
            "  OMMX. Discard it or re-run without the fallback.\n"
            "  You are seeing this instead of a crash ONLY because\n"
            "  OMMX_ALLOW_BF16_FALLBACK is set. Unset it to make route failures fatal\n"
            "  again; set OMMX_STRICT=1 to raise even with the opt-in set.\n"
            f"  route evidence: {_FIRE_FILE}\n"
            f"  programmatic check: ommx_route_health()['degraded'] is now True\n"
            f"{bar}\n")
        sys.stderr.write(banner)
        sys.stderr.flush()
        try:
            # Best-effort DUPLICATE of what stderr already carried: vLLM's logger is the
            # only stream forwarded from a spawn worker to the driver, but it is not
            # importable in every context (CPU unit tests, pre-init). stderr above is the
            # authoritative copy, so a logger failure loses nothing and must not mask the
            # degrade itself. This mirrors _ommx_route_evidence's logger handling.
            from vllm.logger import init_logger
            init_logger("ommx_gpu_serve").error("%s", banner)
        except Exception:  # noqa: BLE001
            pass


def _sentinel_dead_tags() -> dict:
    """Read the route sentinel(s) back and report which routes died — CROSS-PROCESS.

    WHY THIS EXISTS: vLLM v1 runs the engine (and every TP rank) in its OWN spawned
    subprocess. ``_DEGRADED`` / ``_FIRE_SEEN`` / the fire counters are therefore PER
    PROCESS: a bench that imports this module in the DRIVER and asks ommx_route_health()
    would read the state of a process that never served a token, and get a clean answer
    from a run that degraded to FlashAttention in the worker. The sentinel FILE is the
    only channel that crosses that boundary (same reasoning as _ommx_route_evidence's).

    Scans the rank-0 path plus the "<base>.r<rank><ext>" siblings the TP ranks write.
    ``readable`` is False when nothing exists or a read failed, so the caller can tell
    "no evidence" apart from "evidence says clean" — an unreadable proof is not proof.

    THE SENTINEL IS APPEND-ONLY AND IS NOT CLEARED BETWEEN RUNS. A stale ``*_DEAD`` line
    left by an EARLIER run therefore makes this report degraded for a later, healthy one.
    That bias is deliberately toward refusing to certify: a false "degraded" costs a
    re-run, a false "clean" gets a FlashAttention number published as OMMX. Give each arm
    its own ``$OMMX_FIRE_FILE`` (``bench_e2e_a100.py`` already does, per cell) so the
    evidence describes exactly one run.
    """
    base, ext = os.path.splitext(_FIRE_FILE)
    # rank 0 keeps the base name; ranks 1..7 add the suffix (see _ommx_route_evidence).
    paths = [_FIRE_FILE] + [f"{base}.r{r}{ext}" for r in range(1, 8)]
    tags, readable = set(), False
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln:
                        tags.add(ln.split(" ", 1)[0])
            readable = True
        except OSError:
            # Deliberately NOT swallowed into a clean result: readable stays False for
            # this path, which the caller must treat as "no proof", not as "healthy".
            continue
    return {
        "readable": readable,
        "all_tags": sorted(tags),
        # *_DEAD / *_NOFIRE both mean an OMMX route did not serve a step it was asked to.
        "dead_tags": sorted(t for t in tags
                            if t.endswith("_DEAD") or t.endswith("_NOFIRE")),
    }


def ommx_route_health() -> dict:
    """PUBLIC: did this process actually serve OMMX, or did it degrade to bf16?

    A bench MUST consult this (or read ``fire_file``) before attributing any number to
    OMMX. With ``OMMX_ALLOW_BF16_FALLBACK=1`` the engine keeps producing fluent output
    after a route failure — and that output is FlashAttention's. ``degraded is False``
    together with a non-zero fire count / a ``*_ROUTE_FIRED`` tag is the only
    combination that makes a measurement attributable to OMMX.

    Keys:
      degraded             True once any OMMX route failed and the opt-in absorbed it.
      reason               "<tag>: <ExcType>: <msg>" of the FIRST failure, else None.
      layer                identity of the layer that failed first, else None.
      step_hint            the per-step route classification at that moment.
      allow_bf16_fallback  how this process read OMMX_ALLOW_BF16_FALLBACK (at import).
      strict_env           how OMMX_STRICT reads NOW (cfg.strict is seeded from it).
      degraded_in_this_process
                           the RAW per-process latch, separate from ``degraded`` (which
                           also goes True on another worker's *_DEAD sentinel tag).
      sentinel_readable    False when NO sentinel file exists or none could be read.
                           With this False, ``degraded: False`` means "no evidence",
                           NEVER "evidence says OMMX served".
      fires                route-fire counters + the one-time evidence tags recorded.
                           NOTE: under a FULL cudagraph replay the python counters do
                           NOT advance (the replay bypasses python), so the tags are
                           capture-time evidence and a zero counter is not proof of
                           absence — see the GRAPH_ROUTE_FIRED comment.
      fire_file            the sentinel path ($OMMX_FIRE_FILE). TP rank > 0 writes
                           "<base>.r<rank><ext>" instead (see _ommx_route_evidence).
      ablations            the knobs read in THIS module that change what is measured.
                           ``abl_skip_write`` is PARITY-BREAKING: True means the decode
                           read stale KV on purpose and the output is wrong by
                           construction. ``attn_v_bf16`` is not parity-breaking but it
                           swaps the V plane from i2 to full bf16, i.e. it is a
                           DIFFERENT recipe with a different KV footprint, so a
                           compression number from such a run does not describe the
                           published recipe. Neither used to appear here at all.
      unrouted_decode_b    batch sizes (>1) whose uniform decode reached the bf16
                           fall-through instead of an OMMX route (graph mode, or a
                           non-uniform step) — non-empty means PARTIAL OMMX coverage
                           even though ``degraded`` is False. Per process; the cross-
                           process form is the ``DECODE_BF16_UNROUTED`` sentinel tag.
      sliding_window_bypass
                           layers of THIS process that OMMX did not serve because they
                           are sliding-window (Mistral / Gemma-2 / Ministral style):
                           ``{"layers": <n>, "windows": [...]}``. n > 0 means this run's
                           OMMX coverage is PARTIAL — the full-attention layers fired
                           ``DECODE_ROUTE_FIRED`` while these served bf16 FlashAttention
                           out of vLLM's paged cache. Per process; the cross-process
                           form is the ``SW_BYPASS_BF16`` / ``SW_BYPASS_BF16_READ``
                           sentinel tags (visible in ``fires["sentinel_tags"]``).
    """
    sentinel = _sentinel_dead_tags()
    return {
        # TRUE if EITHER this process latched a degrade OR any worker's sentinel names a
        # *_DEAD route. The second term is what makes this key usable from the driver.
        "degraded": (_DEGRADED["reason"] is not None
                     or bool(sentinel["dead_tags"])),
        "reason": _DEGRADED["reason"] or (
            "sentinel names dead route(s): %s" % (sentinel["dead_tags"],)
            if sentinel["dead_tags"] else None),
        # The raw per-process latch, kept separate so a caller can tell "I degraded" from
        # "some other process degraded and told me via the sentinel".
        "degraded_in_this_process": _DEGRADED["reason"] is not None,
        "layer": _DEGRADED["layer"],
        "step_hint": _DEGRADED["step_hint"],
        "allow_bf16_fallback": bool(_ALLOW_BF16_FALLBACK),
        "strict_env": _env_on("OMMX_STRICT"),
        "fires": {
            "batched_decode": int(_BATCHED_FIRES[0]),
            "packed_prefill": int(_PACKED_PREFILL_FIRES[0]),
            "evidence_tags": sorted(_FIRE_SEEN),
            # tags read back OUT of the sentinel files (cross-process; empty in a worker
            # that has not written yet, populated in the driver after the run).
            "sentinel_tags": sentinel["all_tags"],
        },
        # False when NO sentinel file exists or one could not be read. An UNREADABLE
        # sentinel is NOT a clean bill of health: with this False, "degraded": False only
        # means "no evidence", never "evidence says OMMX served". A bench must require
        # sentinel_readable AND not degraded AND a *_ROUTE_FIRED tag.
        "sentinel_readable": sentinel["readable"],
        "fire_file": _FIRE_FILE,
        # Knobs that change WHAT WAS MEASURED, reported next to the route verdict so a
        # bench cannot read "not degraded" as "quotable". abl_skip_write=True means the
        # output is wrong on purpose; attn_v_bf16=True means the KV footprint is not the
        # published recipe's.
        "ablations": {
            "abl_skip_write": bool(_ABL_SKIP_WRITE),
            "attn_v_bf16": bool(_V_BF16),
        },
        # B>1 uniform decodes this process answered with bf16 FlashAttention because no
        # OMMX route accepted them (see _decode_unrouted_evidence). Non-empty = PARTIAL
        # coverage, which "degraded": False alone does not say.
        "unrouted_decode_b": sorted(_UNROUTED_DECODE_B),
        "sliding_window_bypass": {
            "layers": len(_SW_BYPASS_LAYERS),
            "windows": sorted({str(w) for w in _SW_BYPASS_LAYERS.values()}),
        },
    }


_VARLEN_FN = [None]


def _flash_varlen_fn():
    """vLLM's ``flash_attn_varlen_func``, resolved once (version-tolerant import).

    vLLM 0.21 imports it into the v1 flash_attn backend module namespace from
    ``vllm.v1.attention.backends.fa_utils`` (when available); older trees expose it
    via ``vllm.attention.utils.fa_utils`` / ``vllm.vllm_flash_attn``. PACKED-ONLY
    has no substitute prefill path, so failing to resolve is a HARD error (law #5:
    a silent fallback would read the shrunk paged cache).
    """
    if _VARLEN_FN[0] is not None:
        return _VARLEN_FN[0]
    import importlib
    for mod in ("vllm.v1.attention.backends.flash_attn",
                "vllm.v1.attention.backends.fa_utils",
                "vllm.attention.utils.fa_utils",
                "vllm.vllm_flash_attn"):
        try:
            fn = getattr(importlib.import_module(mod), "flash_attn_varlen_func", None)
        except Exception:
            fn = None
        if fn is not None:
            _VARLEN_FN[0] = fn
            return fn
    raise RuntimeError(
        "PACKED-ONLY prefill requires flash_attn_varlen_func; no vLLM module "
        "exposes it (tried v1 flash_attn/fa_utils, attention.utils.fa_utils, "
        "vllm_flash_attn)")


# The REAL serving max sequence length (prompt + generated), captured from vLLM's
# model config by the metadata builder (which holds ``vllm_config``) BEFORE any
# forward / CUDA-graph capture. The OMMX sidecar k_hist + packed planes are sized to
# THIS — not the 4096 config default — or ``write_token``'s device index_copy_ goes
# OUT OF BOUNDS the moment seq exceeds 4096 (the confirmed ctx>=4096 graph crash:
# IndexKernel.cu:193 index_copy_ index-out-of-bounds at num_computed_tokens=4096).
# Under CUDA graph the store is allocated ONCE at this size and reset IN PLACE, so
# the captured tensor addresses stay valid across sequences.
_MAX_MODEL_LEN = None

# Per-worker preflight latch (preflight.py). The metadata builder is constructed ONCE
# PER LAYER, so the unsupported-configuration guards (prefix caching / chunked prefill /
# pool-vs-free-memory) run on the FIRST one and are latched here. A failure is kept and
# RE-RAISED for every later builder — a guard that runs once and is then skipped is the
# silent fallback law #5 forbids.
_PREFLIGHT = {"done": False, "error": None, "report": None}

# One step manager per worker (graph path), lazily created with model geometry.
_MANAGER = None
# One batched step manager per worker (batched path), lazily created.
_BMANAGER = None
# Last CommonAttentionMetadata seen by build() (batched path). build() runs on the
# HOST before do_kv_cache_update/forward each step; the Impl creates the batched
# manager lazily (it has the head geometry the builder lacks) and replays on_build
# against this for the FIRST prefill step (where the manager didn't exist at build).
_LAST_COMMON_MD = None

# Per-step route decision, set by the metadata builder's build() (HOST, runs BEFORE both
# do_kv_cache_update [write] and forward [read] every step) and consumed by BOTH so the
# write and read paths AGREE on the route for the same step. do_kv_cache_update has no
# attn_metadata, so it cannot see max_query_len on its own — this is how it learns the
# step is a B>1 uniform decode (must go to _batched_kv_update) vs a genuine prefill (n>1,
# single seq -> append_block). Keys:
#   "batched_write" (bool): route THIS step's WRITE to _batched_kv_update (the shared pool).
#     True for every step of a batched serving session (prefill AND decode both feed the
#     pool — the batched decode read sources its KV from the pool, so prefills must land
#     there too). The pool's per-request loop does append_block for prefill (T>1) and
#     append for decode (T==1), so one route handles both.
#   "batched_read" (bool): route THIS step's READ to _route_decode_batched. True only on a
#     UNIFORM single-token DECODE step in a batched session (a prefill read still goes to
#     bf16 FlashAttention via super()).
#   "B" (int): number of requests this step.
#   "full_prefill" (bool, PACKED-ONLY): every request's query span covers its WHOLE
#     sequence (q_len == seq_len; chunked prefill + prefix caching disabled), so the
#     in-batch K/V is the complete causal context and the prefill read can run varlen
#     FlashAttention directly on it — never the shrunk paged cache.
# Reset/overwritten every build().
_STEP = {"batched_write": False, "batched_read": False, "B": 0,
         "full_prefill": False}

# Session latch: once ANY step is seen with B>1 (a multi-request batch — a prefill batch
# OR a decode batch), this worker is serving continuous-batching, so EVERY subsequent
# write feeds the shared pool — the batched decode reads its KV from the pool, not the
# single-seq store. Latches True in build() and never clears for the worker's life
# (matching the proven OMMX_ATTN_BATCHED=1 all-writes-to-pool behavior, but enabled
# AUTOMATICALLY by B-awareness instead of by the env). A worker that only ever sees B==1
# steps stays single-batch (fast graph path, no TPOT regression — the gate requirement).
# OMMX_ATTN_GRAPH keeps the single-batch graph path (no pool); OMMX_ATTN_BATCHED forces on.
#   SCOPE NOTE: a request that PREFILLS at B==1 (single-seq store) and only LATER joins a
#   B>1 decode batch would have its prefill in the single-seq store, not the pool. The
#   target serving case (one generate() with N prompts submitted together) schedules a
#   BATCHED prefill (B=N, non-uniform) as the FIRST step -> the latch trips before any
#   write, so every prefill lands in the pool. Staggered-arrival continuous batching that
#   interleaves B==1 prefills with B>1 decodes is a separate follow-up (pre-latch from
#   scheduler max_num_seqs would fix it but regress the B==1 fast path, so it is not done).
_BATCHED_SESSION = [bool(_BATCHED)]


def _manager(cfg, device):
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = OMMXStepManager(cfg, device)
    return _MANAGER


def _bmanager(cfg, device):
    global _BMANAGER
    if _BMANAGER is None:
        _BMANAGER = OMMXBatchedStepManager(cfg, device,
                                           num_seqs=_resolved_max_num_seqs(),
                                           graph=_BATCHED_GRAPH)
    return _BMANAGER


def _hf_geometry(vllm_config) -> dict:
    """PER-TP-RANK KV geometry + layer count from vLLM's HF config (defensive).

    The metadata builder has no head counts of its own (only the Impl sees them, at
    forward time — see ``_ommx_cfg``), so the preflight reads them from
    ``model_config.hf_config``, the same source vLLM seeds the Impl from, and shards
    them the way vLLM does: KV heads are REPLICATED when ``n_kv_heads < TP``, hence
    ``max(1, ·//tp)``. This keeps the projected pool geometry equal to the per-rank
    geometry the pool is actually built with. A value that is absent stays ``None`` —
    the preflight then reports it as 'unknown' and raises instead of guessing (law #5).
    """
    mc = getattr(vllm_config, "model_config", None)
    hf = getattr(mc, "hf_config", None)
    txt = getattr(hf, "text_config", None)   # multimodal wrappers nest the LM config

    def pick(*names):
        for src in (hf, txt):
            if src is None:
                continue
            for n in names:
                v = getattr(src, n, None)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return int(v)
        return None

    tp = getattr(getattr(vllm_config, "parallel_config", None),
                 "tensor_parallel_size", None)
    tp = int(tp) if tp else 1
    nq = pick("num_attention_heads")
    nkv = pick("num_key_value_heads", "num_attention_heads")
    hd = pick("head_dim")
    if hd is None:
        hidden = pick("hidden_size")
        if hidden is not None and nq:
            hd = hidden // nq
    return {
        "head_dim": hd,
        "n_q_heads": (max(1, nq // tp) if nq else None),
        "n_kv_heads": (max(1, nkv // tp) if nkv else None),
        "num_layers": pick("num_hidden_layers", "n_layer", "num_layers"),
        "tensor_parallel_size": tp,
    }


def _preflight_once(vllm_config) -> None:
    """Run ``ommx_preflight_check`` EXACTLY ONCE per worker process; never swallow it."""
    if _PREFLIGHT["error"] is not None:
        raise _PREFLIGHT["error"]
    if _PREFLIGHT["done"]:
        return
    geom = _hf_geometry(vllm_config)
    # cfg carries the RECIPE knobs the pool is built from (group sizes, outlier count /
    # repr, sink+recent window) plus the geometry above.
    #   PLUMBING GAP (metadata.py): OMMXBatchedStepManager.pool() does NOT forward
    #   kv_int8_scale= (nor kv_ring, which has no config field at all), so the POOL
    #   resolves OMMX_KV_INT8_SCALE / OMMX_KV_RING from the env itself. kv_outlier_map
    #   IS forwarded as cfg.kv_outlier_map, but config._env_bool resolves it exactly the
    #   way preflight reads OMMX_KV_OUTLIER_MAP, so the projection matches either way.
    #   preflight.py deliberately reads those ENVS so the projected bytes equal what
    #   actually gets allocated rather than what the config merely declares.
    cfg = resolve_serving_config(
        head_dim=geom["head_dim"], n_q_heads=geom["n_q_heads"],
        n_kv_heads=geom["n_kv_heads"], max_context=_MAX_MODEL_LEN)
    # preflight.py is the CONSUMER of num_seqs: it projects the pool as
    # O(num_seqs * max_model_len) bytes per layer and refuses the config when the
    # projection exceeds OMMX_POOL_BUDGET_FRAC of free memory. Passing the RESOLVED cap
    # (engine-aware, see _resolved_max_num_seqs) makes the projection the one that will
    # really be allocated: at --max-num-seqs 1..8 it drops 32-256x versus the old
    # hard-coded 256, so an honest low-concurrency config (e.g. --max-model-len 131072
    # with a single request in flight) passes on its merits instead of needing an
    # OMMX_UNSAFE_ALLOW_POOL_OVERSUBSCRIBE waiver.
    try:
        _PREFLIGHT["report"] = ommx_preflight_check(
            vllm_config, num_seqs=_resolved_max_num_seqs(), cfg=cfg,
            num_layers=geom["num_layers"], device=None)
    except RuntimeError as exc:
        # remembered so the NEXT layer's builder re-raises instead of running unguarded
        _PREFLIGHT["error"] = exc
        raise
    _PREFLIGHT["done"] = True


class OMMXCanonicalMetadataBuilder(FlashAttentionMetadataBuilder):
    """FA metadata builder. UNIFORM_SINGLE_TOKEN_DECODE CUDA-graph support.

    The eager slice needs no extra host seam (regroup happens in the Impl on each
    write). The build seam is reserved here for the FULL-graph + paged follow-up.
    """

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Capture the real serving max_model_len here (the builder is constructed
        # BEFORE any forward / capture) so the sidecar store is sized to it, not the
        # 4096 default — see _MAX_MODEL_LEN above. Robust to attribute-name /
        # positional variations: try self.vllm_config, the passed args/kwargs, then
        # the process-global current config.
        global _MAX_MODEL_LEN, _ENGINE_MAX_NUM_SEQS
        cfgs = [getattr(self, "vllm_config", None), kwargs.get("vllm_config")]
        cfgs += list(args)
        try:
            from vllm.config import get_current_vllm_config
            cfgs.append(get_current_vllm_config())
        except Exception:
            pass
        chosen = None     # the vllm_config the sizing (and the preflight) came from
        fallback = None   # first candidate that at least exposes a model_config
        for vc in cfgs:
            if fallback is None and getattr(vc, "model_config", None) is not None:
                fallback = vc
            try:
                mml = int(vc.model_config.max_model_len)
                if mml > 0:
                    _MAX_MODEL_LEN = mml
                    chosen = vc
                    break
            except Exception:
                continue
        # Engine concurrency ceiling -> the MIDDLE term of the pool slot cap resolution
        # (see _resolved_max_num_seqs). Read from the SAME candidate list as
        # max_model_len, and BEFORE _preflight_once below, so the preflight projects the
        # pool at the cap that will really be allocated instead of at a hard-coded 256.
        # Unreadable (older/oddly-shaped config) leaves it None -> env/default as before.
        for vc in cfgs:
            try:
                mns = int(vc.scheduler_config.max_num_seqs)
            except Exception:
                continue
            if mns > 0:
                _ENGINE_MAX_NUM_SEQS = mns
                break
        # ── PREFLIGHT (once per worker, BEFORE any forward / cudagraph capture) ────
        # Earliest point holding the whole vllm_config, so it is where the unsupported-
        # configuration guards live: prefix caching (shared first physical block ->
        # shared pool slot -> cross-request KV corruption), chunked prefill (the n>1
        # write is reset+append_block, so a continuation chunk wipes the sequence), and
        # the O(num_seqs * max_model_len) non-paged pool footprint vs free memory.
        # RuntimeError is deliberately NOT caught: an engine that would mis-serve or OOM
        # must fail at startup, not thousands of tokens into a run.
        _preflight_once(chosen if chosen is not None else fallback)

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        # sm_90 (H100/H200) FULL-cudagraph capture of the OMMX uniform single-token decode
        # SKIPS the per-step host metadata seam (build -> on_build): the captured op then
        # reads the FROZEN capture-time write_pos / b_seq_len / tail_idx buffers, so the
        # decode attends ~1 stale position -> incoherent output (a fast "garbage" decode).
        # sm_80 (A100) keeps running this attention PIECEWISE (the seam fires every step)
        # and is correct. So on sm_90 declare NEVER -> vLLM keeps the OMMX attention OUT of
        # the FULL graph (piecewise/eager) so the host seam runs per step and the graph
        # decode reads FRESH buffers. A100/sm_80 keeps the UNIFORM_SINGLE_TOKEN_DECODE
        # declaration (validated coherent under FULL cudagraph). OMMX_ATTN_CG_FORCE overrides
        # (never|uniform) for A/B testing.
        _force = os.environ.get("OMMX_ATTN_CG_FORCE", "").strip().lower()
        _never = getattr(AttentionCGSupport, "NEVER", cls._cudagraph_support)
        if _force == "never":
            return _never
        if _force == "uniform":
            return cls._cudagraph_support
        try:
            import torch as _t
            if _t.cuda.is_available() and _t.cuda.get_device_capability()[0] >= 9:
                return _never
        except Exception:
            pass
        return cls._cudagraph_support

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = super().build(common_prefix_len=common_prefix_len,
                           common_attn_metadata=common_attn_metadata,
                           fast_build=fast_build)
        # ── B-AWARE per-step route decision (HOST seam, runs BEFORE write+read) ──────
        # Classify this step ONCE here so do_kv_cache_update (no attn_metadata) and
        # forward AGREE on the same route. The single-batch path slices [:1] and its n>1
        # write branch is a PREFILL reset+append — both corrupt a B>1 decode. So:
        #   * B>1 anywhere -> latch the worker into a BATCHED session: all writes feed the
        #     shared pool (prefill+decode), uniform-decode reads use the batched op.
        #   * B==1 only (never any B>1) -> the single-batch graph/eager path, UNCHANGED.
        # OMMX_ATTN_BATCHED=1 force-latches batched even for B==1; OMMX_ATTN_GRAPH keeps the
        # single-batch graph path (no batched pool).
        global _LAST_COMMON_MD
        _LAST_COMMON_MD = common_attn_metadata
        B, uniform = detect_step_route(common_attn_metadata)
        _STEP["B"] = int(B)
        if B == STEP_ROUTE_UNCLASSIFIED:
            # NOT a request count — the metadata carried no readable seq_lens. Recorded
            # once per rank so the operator learns it here (at the seam that failed)
            # rather than only from whichever route refuses the step later. Informational
            # tag: on its own it is not proof that a step was mis-served, and the routes
            # that would have had to GUESS B now refuse instead (do_kv_cache_update's
            # unclassified guard and forward's decode consumer).
            _ommx_route_evidence(
                "STEP_ROUTE_UNCLASSIFIED",
                "CommonAttentionMetadata exposed no readable seq_lens; B is unknown "
                "(NOT 1). Check the vLLM version against the validated 0.21.")
        # PACKED-ONLY prefill gate (HOST seam): forward has no host q_len/seq_len,
        # so classify "every request prefills its WHOLE sequence" here.
        _STEP["full_prefill"] = (detect_full_prefill(common_attn_metadata)
                                 if _PACKED_ONLY else False)
        if not _GRAPH and B > 1:
            # Multi-request batch seen -> this worker is continuous-batching for its life.
            _BATCHED_SESSION[0] = True
        in_batched = bool(_BATCHED_SESSION[0]) and not _GRAPH
        # WRITE: every step of a batched session feeds the pool (prefill blocks + decode
        # rows). READ: only a uniform single-token decode uses the batched decode op.
        _STEP["batched_write"] = bool(in_batched)
        _STEP["batched_read"] = bool(in_batched and uniform and B >= 1)
        if _GRAPH and _MANAGER is not None:
            # HOST seam (outside capture): regroup + refresh fixed-address buffers.
            _MANAGER.on_build(common_attn_metadata)
        if in_batched and _BMANAGER is not None:
            # HOST seam (eager, outside capture): per-request seq_lens / query_start_loc
            # -> uniform-decode detection + per-slot regroup. Run whenever the session is
            # batched and the manager exists (created lazily by the Impl on the first
            # batched write; the first step replays via _LAST_COMMON_MD inside the Impl).
            _BMANAGER.on_build(common_attn_metadata)
        return md

    def build_for_cudagraph_capture(self, common_attn_metadata):
        if _GRAPH and _MANAGER is not None:
            _MANAGER.capturing = True
        if _BMANAGER is not None and getattr(_BMANAGER, "graph", False):
            _BMANAGER.capturing = True
            # the capture dummy run drives uniform single-token decode at a fixed B;
            # refresh the device buffers so the captured write/read bind valid indices.
            _BMANAGER.on_build(common_attn_metadata)
        _CAPTURING[0] = True                       # PACKED capture-stub gate (forward())
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            _CAPTURING[0] = False
            if _BMANAGER is not None and getattr(_BMANAGER, "graph", False):
                _BMANAGER.capturing = False
            if _GRAPH and _MANAGER is not None:
                _MANAGER.capturing = False


class OMMXCanonicalImpl(FlashAttentionImpl):
    """FlashAttention impl + OMMX canonical sidecar write + decode routing."""

    # ── sidecar lifecycle ──────────────────────────────────────────────────────

    def _ommx_cfg(self):
        cfg = getattr(self, "_ommx_serving_cfg", None)
        if cfg is None:
            # max_context = the REAL serving max_model_len (sized for k_hist/planes,
            # the manager grid, and the decode workspace). None falls back to the
            # OMMX_ATTN_MAXCTX env / 4096 default inside resolve_serving_config.
            cfg = resolve_serving_config(
                head_dim=self.head_size, n_q_heads=self.num_heads,
                n_kv_heads=self.num_kv_heads, max_context=_MAX_MODEL_LEN)
            # TP sharding guard (law: silent fallback). Under tensor_parallel_size>1
            # the Impl attrs are the PER-RANK sharded head counts, and the sidecar
            # store/pool + the kernel grid are sized from this cfg. If a future
            # caller ever seeds cfg from a GLOBAL (un-sharded) model config, the
            # pool planes and the per-rank query tensor would disagree on the KV-
            # head count and the decode would mis-shard. Fail loud, not silent.
            assert (int(cfg.n_kv_heads) == int(self.num_kv_heads)
                    and int(cfg.head_dim) == int(self.head_size)), (
                "OMMX cfg head geometry desynced from TP-sharded Impl attrs: "
                f"cfg=(Hkv={cfg.n_kv_heads},D={cfg.head_dim}) "
                f"impl=(Hkv={self.num_kv_heads},D={self.head_size})")
            self._ommx_serving_cfg = cfg
        return cfg

    def _ommx_store(self, layer, device, max_ctx) -> CanonicalKVStore:
        st = getattr(layer, "_ommx_store", None)
        if st is None:
            cfg = self._ommx_cfg()
            st = CanonicalKVStore(
                head_dim=self.head_size, n_kv_heads=self.num_kv_heads,
                max_seq_len=int(max_ctx), k_format=cfg.k_format,
                v_format=("bf16" if _V_BF16 else "i2"),
                outliers_per_vector=cfg.outliers_per_vector,
                outlier_select=cfg.outlier_select, outlier_repr=cfg.outlier_repr,
                use_pow2=cfg.use_pow2, window=cfg.window(),
                group_channels=cfg.group_channels, device=device)
            layer._ommx_store = st
        return st

    def _ommx_reset(self, layer, device, max_ctx) -> CanonicalKVStore:
        # New sequence (a prefill arrived). CUDA-graph-SAFE: if a store already
        # exists AND is large enough, reset it IN PLACE (reuse the captured tensor
        # addresses — reallocating would make the captured decode graph touch a
        # stale pointer). Reallocate only when there is no store yet, or the cached
        # one is too small (eager-only — the max_context=max_model_len sizing makes
        # the store big enough for every sequence under graph, so this never trips
        # there).
        st = getattr(layer, "_ommx_store", None)
        if st is not None and int(st.max_seq_len) >= int(max_ctx):
            st.reset_inplace()
            return st
        layer._ommx_store = None
        return self._ommx_store(layer, device, max_ctx)

    # ── BATCHED (B>1) write/read via the per-layer MultiSeqKVPool ────────────────

    def _batched_kv_update(self, layer, key, value) -> None:
        """Split this step's K/V ``[num_tokens, Hkv, D]`` per request into the pool.

        Uses the batched manager's host ``q_start`` (query_start_loc) to slice each
        request's contribution. ``T>1`` => prefill block (``append_block``); ``T==1``
        => decode token (``append``). Both regroup completed 32-token groups on the
        host (outside capture). The pool is created lazily here (the Impl has the head
        geometry the metadata builder lacks).
        """
        cfg = self._ommx_cfg()
        mgr = _bmanager(cfg, key.device)
        # First step: the manager didn't exist when build() ran -> replay it now so
        # seq_lens/q_start reflect THIS step before we split the write.
        if not mgr.seq_lens and _LAST_COMMON_MD is not None:
            mgr.on_build(_LAST_COMMON_MD)
        if mgr.dead:
            raise RuntimeError("batched manager dead")
        n = int(key.shape[0])
        B = len(mgr.seq_lens)
        # vLLM's init PROFILING / _dummy_run drives the model with dummy tokens but
        # NO real requests (B == 0). There is nothing to pack into the OMMX sidecar
        # for a dummy run -> skip (bf16 cache handles it). Only real steps (B >= 1)
        # write the sidecar. This is the difference between init warmup and serving.
        if B == 0 or n == 0:
            return
        pool = mgr.pool(id(layer), key.device)
        q_start = mgr.q_start
        # Derive per-request token boundaries. Prefer q_start (cumulative len B+1);
        # fall back to a uniform 1-token-per-request split when it matches n == B.
        if q_start is not None and len(q_start) == B + 1 and int(q_start[-1]) == n:
            bounds = q_start
        elif n == B:
            bounds = list(range(B + 1))
        else:
            raise RuntimeError(
                f"batched write: cannot map {n} tokens to {B} requests")
        # FAST PATH: a UNIFORM single-token decode step (every request adds exactly 1
        # token, no (re)start this step) -> ONE device-side batched append + GPU regroup
        # instead of the per-request python loop (B tiny scatters + D2H/CPU/H2D pack).
        # The slow per-request loop below still handles prefill / chunked / restart.
        uniform = (n == B) and all(
            int(bounds[i + 1]) - int(bounds[i]) == 1 for i in range(B))
        if uniform:
            restart = any(int(mgr.seq_lens[i]) - 1 <= 0 for i in range(B))
            if not restart:
                Kd = key.view(B, pool.H, pool.D)
                Vd = value.view(B, pool.H, pool.D)
                pool.append_decode_batched(mgr.slots, Kd, Vd)
                return
        for i in range(B):
            slot = mgr.slots[i]
            t0, t1 = int(bounds[i]), int(bounds[i + 1])
            if t1 <= t0:
                continue
            seq_after = int(mgr.seq_lens[i])
            seq_before = seq_after - (t1 - t0)
            if seq_before <= 0:
                # a (re)started request: reset its pool slot history before the block.
                pool.seq_len[slot] = 0
                pool.packed_groups[slot] = 0
                mgr.planner.reset(slot)
            K = key[t0:t1].view(t1 - t0, pool.H, pool.D)
            V = value[t0:t1].view(t1 - t0, pool.H, pool.D)
            if (t1 - t0) > 1:
                pool.append_block(slot, K, V)
            else:
                pool.append(slot, K[0], V[0])

    def _batched_kv_update_captured(self, layer, key, value) -> None:
        """CAPTURE-SAFE B>1 decode write: ONE device scatter at host-refreshed columns.

        Used by the batched-graph path. The host build-seam (``_refresh_graph_buffers``)
        already advanced each pool's ``seq_len``/``packed_groups`` and CPU-regrouped any
        completed group OUTSIDE capture, and filled the manager's fixed-address
        ``b_sel`` (row->slot) + ``b_write_col`` (write column) device buffers. This only
        issues the static-shape ``index_put_`` into ``k_hist``/``v_hist`` (capturable),
        reading ONLY those device buffers — no host ``seq_len`` list, no per-step
        ``torch.tensor``, no regroup. Mirrors ``CanonicalKVStore.write_token`` (B==1).
        """
        mgr = _BMANAGER
        if mgr is None or mgr.dead or not mgr.is_uniform_decode:
            return
        B = int(mgr.cur_B)
        if B == 0 or int(key.shape[0]) == 0:
            return
        pool = mgr.pool(id(layer), key.device)
        rows = mgr.b_sel[:B]
        cols = mgr.b_write_col[:B]
        pool.append_decode_captured(rows, cols, key, value)

    def _ommx_ws_batched(self, layer, B, dev):
        """Per-layer FIXED-capacity batched decode workspace (sized to max_num_seqs).

        Allocated once at the pool's num_seqs capacity; the active step uses the first
        ``B`` rows. Avoids per-step allocation (eager speedup); mirrors ``_ommx_ws``.
        """
        ws = getattr(layer, "_ommx_ws_batched_cache", None)
        if ws is not None and ws["o"].shape[0] >= B:
            return ws
        H, D = self.num_heads, self.head_size
        cfg = self._ommx_cfg()
        cap = max(B, _resolved_max_num_seqs())
        max_seq = max(1, int(cfg.max_context))
        nsplits = max(1, int(_auto_num_kv_splits(max_seq, cap, device=dev,
                                                 kv_heads=self.num_kv_heads)))
        max_groups = max(1, (max_seq + 31) // 32)
        import torch as _t
        ws = {
            "o": _t.empty(cap, H, D, dtype=_t.float32, device=dev),
            "lse": _t.empty(cap, H, dtype=_t.float32, device=dev),
            # D+1 last dim: per split = D o-lanes + 1 lse lane (matches _make_attn_logits).
            # Allocating D makes split lse clobber the next split's o_v -> decode garbage for
            # num_kv_splits>1 (the root cause; see _ommx_llama_modeling._ommx_make_ws).
            "attn_logits": _t.empty(cap, H, nsplits + 1, D + 1, dtype=_t.float32, device=dev),
            "qzp_buf": _t.empty(cap, H, max_groups, dtype=_t.float32, device=dev),
            "num_kv_splits": nsplits,
        }
        layer._ommx_ws_batched_cache = ws
        return ws

    def _route_decode_batched(self, layer, query, output) -> bool:
        """Route a uniform single-token decode batch through the canonical op.

        ``query`` / ``output`` are ``[B, H, D]`` (one query token per request). Pulls
        the per-layer pool's ``decode_inputs_batched(slots)`` bundle (shared planes +
        per-request tables/seq buffers/bf16 tail) and runs the op over all B rows — no
        ``[:1]`` slice. SHADOW: returns False (falls back to bf16 FA) when not a
        uniform decode step or the pool isn't ready.
        """
        cfg = self._ommx_cfg()
        if _BMANAGER is None or _BMANAGER.dead or not _BMANAGER.is_uniform_decode:
            return False
        mgr = _BMANAGER
        pool = mgr.pools.get(id(layer))
        if pool is None:
            return False
        try:
            H, D = self.num_heads, self.head_size
            B = len(mgr.slots)
            q = query.view(-1, H, D)[:B]                    # [B, Hq, D]
            di = pool.decode_inputs_batched(mgr.slots)
            pl = di["planes"]
            ws = self._ommx_ws_batched(layer, B, q.device)
            o = ws["o"][:B]
            lse = ws["lse"][:B]
            attn_logits = ws["attn_logits"][:B]
            qzp_buf = ws["qzp_buf"][:B]
            sm_scale = float(getattr(self, "scale", 1.0 / (D ** 0.5)))
            ommx_paged_decode_attention_canonical(
                q, pl["k_base"], pl["k_scale"], pl["k_zp"],
                pl.get("k_oidx"), pl.get("k_oval"),
                pl["v_main"], pl["v_scale"], pl["v_zp"],
                di["k_tail"], di["v_tail"], di["b_tail_len"],
                o, lse,
                di["req_to_token"], di["req_to_group"], di["b_seq_len"],
                sm_scale=sm_scale, page_size=int(pl["page_size"]),
                k_outliers_per_vector=int(pl["outliers_per_vector"]),
                k_format=str(pl["k_format"]),
                combinadic_read=cfg.combinadic_read,
                k_crank=pl.get("k_crank"),
                k_fp4_mapscale=pl.get("k_fp4_mapscale"),
                k_fp4_mapcenter=pl.get("k_fp4_mapcenter"),
                kv_outlier_map=bool(pl.get("kv_outlier_map", False)),
                kv_int8_scale=bool(pl.get("kv_int8_scale", False)),
                num_kv_splits=ws["num_kv_splits"],
                max_seq_len=di["max_seq_len"], max_tail_len=di["max_tail_len"],
                attn_logits=attn_logits, qzp_buf=qzp_buf,
                packed_start_offset=int(di["packed_start_offset"]))
            output.view(-1, H, D)[:B].copy_(o.to(output.dtype))
            # WORKER-SIDE route-fired evidence (the driver's in-process probe can't
            # see the spawn child): one-time stderr marker proving the OMMX batched
            # decode actually ran in the worker (NOT a silent bf16 fallback, law #5).
            _BATCHED_FIRES[0] += 1
            _ommx_route_evidence("BATCHED_ROUTE_FIRED", f"B={B} fmt={pl['k_format']}")
            return True
        except Exception as e:  # noqa: BLE001
            # FATAL by default (law #5): _ommx_route_failed records the evidence and then
            # re-raises unless OMMX_ALLOW_BF16_FALLBACK is set (cfg.strict still forces
            # the raise). Returning False here means the opt-in was taken, and the caller
            # then hands a B>1 step to bf16 super() — a DEGRADED, non-OMMX result.
            mgr.dead = True
            _ommx_route_failed("BATCHED_ROUTE_DEAD", e, cfg=cfg, layer=layer,
                               step_hint=f"B={_STEP.get('B')} batched uniform-decode read")
            return False

    def _route_decode_batched_graph(self, layer, query, output) -> bool:
        """CUDA-graph-CAPTURABLE B>1 decode read: fixed-address device buffers only.

        Mirrors the B==1 ``_route_decode_graph`` for a batch: all tensor args are
        FIXED shape/address (whole pool planes + the manager's ``b_sel``/``b_seq_len``/
        ``b_tail_len``/``b_tail_idx`` device buffers, refreshed by the HOST build-seam
        OUTSIDE capture, + the pool's persistent ``kt``/``vt`` tail buffers + the
        per-layer fixed batched workspace). No host python loop, no ``.item()``, no
        per-step alloc -> the attention+GEMM region is capturable. ``max_seq_len`` is
        the FIXED worst-case grid int (the device ``b_seq_len`` bounds the live prefix).
        """
        cfg = self._ommx_cfg()
        mgr = _BMANAGER
        if mgr is None or mgr.dead or not mgr.is_uniform_decode:
            return False
        pool = mgr.pools.get(id(layer))
        if pool is None:
            return False
        try:
            H, D = self.num_heads, self.head_size
            B = int(mgr.cur_B)
            if B == 0:
                return False
            q = query.view(-1, H, D)[:B]                    # [B, Hq, D]
            pool._ensure_decode_buffers(B)
            d = pool._dbuf
            kt = d["kt"][:B]
            vt = d["vt"][:B]
            di = pool.decode_inputs_captured(
                mgr.b_sel[:B], mgr.b_seq_len[:B], mgr.b_tail_len[:B],
                mgr.b_tail_idx[:B], kt, vt, mgr.max_seq_len)
            pl = di["planes"]
            ws = self._ommx_ws_batched(layer, B, q.device)
            o = ws["o"][:B]
            lse = ws["lse"][:B]
            attn_logits = ws["attn_logits"][:B]
            qzp_buf = ws["qzp_buf"][:B]
            sm_scale = float(getattr(self, "scale", 1.0 / (D ** 0.5)))
            ommx_paged_decode_attention_canonical(
                q, pl["k_base"], pl["k_scale"], pl["k_zp"],
                pl.get("k_oidx"), pl.get("k_oval"),
                pl["v_main"], pl["v_scale"], pl["v_zp"],
                di["k_tail"], di["v_tail"], di["b_tail_len"],
                o, lse,
                di["req_to_token"], di["req_to_group"], di["b_seq_len"],
                sm_scale=sm_scale, page_size=int(pl["page_size"]),
                k_outliers_per_vector=int(pl["outliers_per_vector"]),
                k_format=str(pl["k_format"]),
                combinadic_read=cfg.combinadic_read,
                k_crank=pl.get("k_crank"),
                k_fp4_mapscale=pl.get("k_fp4_mapscale"),
                k_fp4_mapcenter=pl.get("k_fp4_mapcenter"),
                kv_outlier_map=bool(pl.get("kv_outlier_map", False)),
                kv_int8_scale=bool(pl.get("kv_int8_scale", False)),
                num_kv_splits=ws["num_kv_splits"],
                max_seq_len=di["max_seq_len"], max_tail_len=di["max_tail_len"],
                attn_logits=attn_logits, qzp_buf=qzp_buf,
                packed_start_offset=int(di["packed_start_offset"]))
            output.view(-1, H, D)[:B].copy_(o.to(output.dtype))
            _BATCHED_FIRES[0] += 1
            _ommx_route_evidence("BATCHED_GRAPH_ROUTE_FIRED",
                                 f"B={B} fmt={pl['k_format']}")
            return True
        except Exception as e:  # noqa: BLE001
            # FATAL by default (law #5) — same policy as the eager batched route above.
            mgr.dead = True
            _ommx_route_failed("BATCHED_GRAPH_ROUTE_DEAD", e, cfg=cfg, layer=layer,
                               step_hint=f"B={_STEP.get('B')} batched-graph decode read")
            return False

    # ── write path: keep the bf16 cache (super) + append to the OMMX sidecar ────

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping) -> None:
        # SLIDING-WINDOW layers (Mistral, Gemma-2 alternating) are never OMMX-routed on
        # the read path -- forward() gates every OMMX route on sliding_window==(-1,-1)
        # (backend.py prefill/decode/packed guards), so they serve from the base bf16
        # paged cache via super().forward(). In PACKED-ONLY the spec patch also leaves
        # their cache UNSHRUNK (packed_only.py shrinks FullAttentionSpec only). So they
        # MUST get the bf16 paged write regardless of _PACKED_ONLY (else super().forward()
        # reads an unwritten cache -> silent mis-serve, law #5) and need NO OMMX sidecar
        # (never read). Return right after the base write. Full-attention layers keep the
        # packed behavior below. For Llama (no sliding_window) _sw is always False -> no
        # behavior change.
        #   IT IS TAGGED (law #5). The bypass is correct, but it makes OMMX's coverage of
        #   a mixed SWA/full model PARTIAL, and the sentinel used to say nothing about it:
        #   the full-attention layers wrote DECODE_ROUTE_FIRED and the sliding-window
        #   layers wrote nothing, so a half-OMMX run and a fully-OMMX run left IDENTICAL
        #   evidence. SW_BYPASS_BF16 (write) + SW_BYPASS_BF16_READ (forward) are what make
        #   them distinguishable; both are informational, never *_DEAD (see
        #   _sw_bypass_evidence).
        if getattr(self, "sliding_window", (-1, -1)) != (-1, -1):
            super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
            _sw_bypass_evidence("SW_BYPASS_BF16", self,
                                "write: bf16 paged cache only, NO OMMX sidecar")
            return
        # PACKED-ONLY: the paged cache is SHRUNK (spec head_size reduced to reserve
        # the OMMX byte budget), so a headdim-D reshape_and_cache_flash into it
        # scatters past the page (corruption, not just waste). The sidecar below is
        # the ONLY backing store in packed mode; skip the paged write entirely.
        if not _PACKED_ONLY:
            super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        try:
            # key/value: [num_tokens, n_kv_heads, head_size]. num_tokens > 1 marks a
            # (new) prefill in the single-batch slice -> reset + append the block;
            # num_tokens == 1 is a decode step -> append one row.
            n = int(key.shape[0])
            cfg = self._ommx_cfg()
            max_ctx = cfg.max_context
            # UNCLASSIFIED STEP (law #5). build() could not read this step's per-request
            # seq_lens, so B is UNKNOWN (metadata.STEP_ROUTE_UNCLASSIFIED), not 1. Inside
            # a batched session the write is per-request and never consults B, so it is
            # unaffected; OUTSIDE one, the branches below pick PREFILL-vs-DECODE from
            # ``n`` alone, and a B>1 decode batch arrives as n == B rows — which the n>1
            # branch would take for a prefill and answer with reset + append_block,
            # wiping the sequence. Refuse rather than guess: this raise is caught by the
            # KV_UPDATE_DEAD handler at the bottom, which records the evidence first and
            # then applies the module's failure policy (fatal by default).
            if (int(_STEP.get("B", 0)) == STEP_ROUTE_UNCLASSIFIED
                    and not _STEP["batched_write"]):
                raise RuntimeError(
                    "[ommx] cannot classify this step: CommonAttentionMetadata exposed "
                    "no readable per-request seq_lens, so the number of requests is "
                    f"unknown (n={n} rows in this write). The single-sequence write "
                    "path decides PREFILL vs DECODE from the row count alone, which is "
                    "only sound when B is known to be 1. Refusing to write rather than "
                    "risk resetting a live sequence. This normally means the installed "
                    "vLLM renamed or reshaped seq_lens/seq_lens_cpu — check the vLLM "
                    "version against the validated 0.21 and update "
                    "metadata.detect_step_route's field list.")
            # ABLATION (timing-only, PARITY-BREAKING): skip the per-step new-token KV
            # pack+regroup/write on a DECODE step so the kernel reads STALE KV (wrong
            # output). (full_step) - (SKIP_WRITE_step) = the per-step KV pack/write cost.
            # Prefill (n>1) is KEPT (else the cache is empty). Default OFF.
            # ANNOUNCED, not silent: the first skip writes the ABL_SKIP_WRITE_ACTIVE tag
            # to the sentinel and a one-time banner to stderr + the vLLM log, so a run
            # whose output is wrong on purpose cannot be mistaken for a clean one after
            # the fact. The announcement is placed HERE (at the first real skip) rather
            # than at import, so it fires only when the knob actually changed a step.
            if _ABL_SKIP_WRITE and n == 1:
                _announce_parity_breaking_ablations()
                return
            # B-AWARE WRITE: in a batched session, EVERY write (prefill block + decode
            # row, single or multi request) feeds the shared pool — the batched decode
            # read sources its KV from the pool, so prefills must land there too. This is
            # set by build() (which has the metadata this method lacks) and replaces the
            # OMMX_ATTN_BATCHED env as the correctness gate. A B>1 decode step would
            # otherwise hit the n>1 PREFILL branch below (reset+append_block) and corrupt
            # requests 1..B-1; routing it here (per-request append) is the fix.
            if _STEP["batched_write"]:
                # GRAPH batched decode: a UNIFORM single-token decode write goes through
                # the capture-safe device-indexed scatter (host build-seam already
                # advanced seq_len + regrouped). PREFILL / non-uniform steps (n>1, or the
                # pre-decode prefill) still use the eager per-request path (it does the
                # host seq_len advance + CPU regroup the captured write skips).
                m = _BMANAGER
                if (m is not None and getattr(m, "graph", False)
                        and m.is_uniform_decode and int(key.shape[0]) == int(m.cur_B)
                        and int(m.cur_B) > 0):
                    self._batched_kv_update_captured(layer, key, value)
                else:
                    self._batched_kv_update(layer, key, value)
                return
            if _GRAPH:
                mgr = _manager(cfg, key.device)
                if n > 1:
                    # PREFILL (eager / piecewise — not captured): reset + bulk pack,
                    # register the layer store with the step manager.
                    # ── CHUNKED-PREFILL SEAM ──────────────────────────────────────
                    # Decode-only today (enable_chunked_prefill=False), so every n>1 is
                    # a whole new sequence's prefill -> always reset. To enable chunked
                    # prefill, gate this reset on FIRST-CHUNK only (request identity /
                    # query_start_loc==0) and have continuation chunks call append_block
                    # WITHOUT reset (the store already accumulates via append_block into
                    # seq_len; maybe_regroup packs whatever groups complete). Kernel /
                    # store need NO change — only this reset-vs-append decision does.
                    st = self._ommx_reset(layer, key.device, max(max_ctx, n))
                    st.append_block(key, value)
                    mgr.register(id(layer), st)
                else:
                    # DECODE (captured): capture-safe device-indexed write at the
                    # build-seam-set write_pos. No regroup here (host build() owns it).
                    st = self._ommx_store(layer, key.device, max_ctx)
                    mgr.register(id(layer), st)
                    st.write_token(mgr.write_pos, key[0], value[0])
            elif n > 1:
                st = self._ommx_reset(layer, key.device, max(max_ctx, n))
                st.append_block(key, value)
            else:
                st = self._ommx_store(layer, key.device, max_ctx)
                st.append(key[0], value[0])
                st.maybe_regroup()
        except Exception as e:  # noqa: BLE001
            # THE MEASURED SILENT-bf16 PATH. This write-path pre-latch is the only one
            # that latches _ommx_dead WITHOUT a forward-side sentinel (forward's gate
            # short-circuits once dead), and it is what fired under vLLM 0.21's DEFAULTS:
            # "ring append_block expects a fresh request slot" (chunked prefill) and
            # "CUDA out of memory ... 514.00 MiB" (the 256-slot pool) both landed here,
            # after which the engine served bf16 FlashAttention and produced output
            # byte-identical to the bf16 reference. So it is FATAL by default now:
            # _ommx_route_failed appends KV_UPDATE_DEAD to the sentinel FIRST (a fallback
            # must always name its cause), then re-raises unless OMMX_ALLOW_BF16_FALLBACK
            # is set; cfg.strict raises regardless. cfg is read from the CACHED attribute
            # rather than self._ommx_cfg() so a cfg-construction failure cannot mask the
            # original exception (missing cache -> not strict -> the default raise).
            _ommx_route_failed("KV_UPDATE_DEAD", e,
                               cfg=getattr(self, "_ommx_serving_cfg", None),
                               layer=layer,
                               step_hint=(f"n={int(key.shape[0])} B={_STEP.get('B')} "
                                          f"batched_write={_STEP.get('batched_write')}"))

    # ── read path: route uniform single-token decode through the canonical op ───

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        # SLIDING-WINDOW layers: bf16 FlashAttention out of vLLM's paged cache, tagged.
        # BEHAVIOR-IDENTICAL to the fall-through it replaces — all three gates below
        # already require sliding_window == (-1, -1) (the packed prefill seam, the decode
        # route, and the PACKED no-fallback raise), so such a layer could only ever reach
        # the final super().forward(). Stating the carve-out ONCE and up front (those
        # three keep their own conditions) is what gives the READ path the evidence it
        # had none of: paired with SW_BYPASS_BF16 on the write, the sentinel now
        # distinguishes a fully-OMMX-routed model from one where OMMX served only the
        # full-attention layers. See _sw_bypass_evidence for why it is informational
        # rather than a *_DEAD failure tag.
        if getattr(self, "sliding_window", (-1, -1)) != (-1, -1):
            _sw_bypass_evidence("SW_BYPASS_BF16_READ", self,
                                "read: bf16 FlashAttention over the vLLM paged cache")
            return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                                   output, output_scale=output_scale,
                                   output_block_scale=output_block_scale)
        if (_PACKED_ONLY and attn_metadata is not None
                and output_scale is None and output_block_scale is None
                and int(getattr(attn_metadata, "max_query_len", 0)) > 1
                and getattr(self, "sliding_window", (-1, -1)) == (-1, -1)
                and _STEP.get("full_prefill", False)):
            # PACKED-ONLY prefill seam: super().forward would run FlashAttention
            # over the SHRUNK paged cache and hard-reject (Q/K headdim 128 vs the
            # shrunk cache V headdim — the confirmed first-prefill crash). A
            # full_prefill step's in-batch K/V IS the complete causal context
            # (q_len == seq_len per request, build()-verified), so attend directly
            # on the batch; the paged cache is never touched.
            self._prefill_packed_varlen(query, key, value, attn_metadata, output)
            return output
        if (attn_metadata is not None
                and not getattr(layer, "_ommx_dead", False)
                and output_scale is None and output_block_scale is None
                and int(getattr(attn_metadata, "max_query_len", 0)) == 1
                and getattr(self, "sliding_window", (-1, -1)) == (-1, -1)):
            # B-AWARE READ. This branch is a UNIFORM single-token decode (max_query_len==1,
            # vLLM-confirmed). build() classified the SAME step:
            #   * batched session uniform decode -> _route_decode_batched (reads ALL B rows
            #     from the shared pool — the [B,H,D] op, NO [:1] slice). On failure for B>1
            #     fall back to bf16 super() — NEVER _route_decode (its [:1] slice corrupts
            #     requests 1..B-1, the bug being fixed).
            #   * single-batch (B==1, never batched) -> _route_decode (the cudagraph fast
            #     path), behavior IDENTICAL to before.
            #   * UNCLASSIFIED (B == STEP_ROUTE_UNCLASSIFIED) -> refuse; see below.
            B = int(_STEP.get("B", 0))
            if _STEP["batched_read"]:
                m = _BMANAGER
                # GRAPH batched decode: the capture-safe device-buffer read (capturable).
                # Falls back to the eager batched read if graph mode is off or the graph
                # route declines (then to bf16 super() for B>1 safety).
                if m is not None and getattr(m, "graph", False):
                    if self._route_decode_batched_graph(layer, query, output):
                        return output
                elif self._route_decode_batched(layer, query, output):
                    return output
                # one-time diagnostic: why the batched route did NOT fire (proves
                # whether OMMX is used vs a silent bf16 fallback — the key question).
                m = _BMANAGER
                reason = ("bmanager=None" if m is None else
                          "dead" if m.dead else
                          "not_uniform_decode" if not m.is_uniform_decode else
                          f"pool_missing(layer={id(layer)})"
                          if m.pools.get(id(layer)) is None else "route_returned_False")
                _ommx_route_evidence("BATCHED_ROUTE_NOFIRE", reason)
                # B>1: do NOT fall through to the single-batch _route_decode (it would
                # slice [:1] and corrupt requests 1..B-1). bf16 super() is the safe path.
            elif B == STEP_ROUTE_UNCLASSIFIED:
                # REFUSE — do not guess (law #5). ``B`` is not "small", it is UNKNOWN:
                # build() found no readable per-request seq_lens. This branch used to be
                # ``elif B <= 1``, which swallowed the sentinel, because an unclassified
                # step and a genuine single request both arrived as 0. A real B=4 decode
                # that failed classification then took the single-batch route, whose
                # ``query.view(-1,H,D)[:1]`` / ``output...[:1].copy_()`` compute row 0
                # and leave rows 1..B-1 UNWRITTEN — and it recorded DECODE_ROUTE_FIRED
                # while doing it, so the bench called the run an OMMX measurement.
                # (A genuine B==0 vLLM init/profiling dummy is NOT this case: it still
                # reports B == 0 and still takes the single-batch branch below, where a
                # missing store simply declines. Nothing about init changes here.)
                _ommx_route_failed(
                    "DECODE_ROUTE_UNCLASSIFIED_DEAD",
                    RuntimeError(
                        "[ommx] refusing to route an UNCLASSIFIED uniform decode step: "
                        "CommonAttentionMetadata exposed no readable per-request "
                        "seq_lens, so the batch size is unknown. The single-batch route "
                        "would slice [:1] and answer only request 0, leaving any other "
                        "request in this step unwritten while the sentinel reported a "
                        "clean OMMX decode. Check the vLLM version against the "
                        "validated 0.21 and update metadata.detect_step_route's field "
                        "list; OMMX_ALLOW_BF16_FALLBACK=1 degrades this step to bf16 "
                        "FlashAttention instead (and marks the run non-quotable)."),
                    cfg=getattr(self, "_ommx_serving_cfg", None), layer=layer,
                    step_hint="B=UNCLASSIFIED uniform-decode read")
            elif B <= 1:
                if self._route_decode(layer, query, output):
                    return output
                # one-time diagnostic: why the single-batch (B<=1) route did NOT fire
                # (store unwired vs dead-latch vs kernel decline) — proves OMMX ran vs a
                # silent bf16 fallback for B==1 (mirrors the batched NOFIRE above, law #5).
                _ommx_route_evidence(
                    "DECODE_ROUTE_NOFIRE",
                    "store=None" if getattr(layer, "_ommx_store", None) is None
                    else "dead" if getattr(layer, "_ommx_dead", False)
                    else "route_false")
            else:
                # B > 1 but the step was NOT classified as a batched read: bf16
                # FlashAttention serves it (the fall-through at the bottom), correctly
                # but NOT as OMMX. Two ways to get here, both real:
                #   * OMMX_ATTN_GRAPH=1 -> build() never latches _BATCHED_SESSION (that
                #     line is gated on ``not _GRAPH``), so no B>1 step ever becomes a
                #     batched read;
                #   * ``uniform`` False while ``max_query_len == 1`` (a request
                #     contributing zero query tokens) -> batched_read False for a step
                #     the single-batch route must not touch either.
                # Neither could reach _route_decode (its [:1] slice would answer only
                # request 0), so the bf16 fall-through is the CORRECT answer — but until
                # this tag it was also a SILENT one: the B==1 steps of the same run had
                # already written DECODE_ROUTE_FIRED / GRAPH_ROUTE_FIRED, so a run whose
                # multi-request steps were all served by FlashAttention was indis-
                # tinguishable from a fully OMMX-routed one.
                # INFORMATIONAL, not *_NOFIRE (see _ommx_route_evidence's suffix
                # contract): in graph mode vLLM's own cudagraph CAPTURE drives dummy
                # uniform decodes at every capture size, i.e. at B>1, so a failure
                # suffix here would mark every graph-mode run ok=False for steps that
                # served no request at all. It reports coverage, like SW_BYPASS_BF16.
                _decode_unrouted_evidence(B, graph=_GRAPH)
        if (_PACKED_ONLY and attn_metadata is not None
                and getattr(self, "sliding_window", (-1, -1)) == (-1, -1)
                and str(getattr(self, "attn_type", "decoder")) in
                ("decoder", "AttentionType.DECODER")):
            # PACKED-ONLY: the shrunk paged cache holds no real K/V, so the bf16
            # FlashAttention fallback would read garbage — fail LOUD with the step
            # shape instead of silently mis-serving (law #5). Profiling runs pass
            # (attn_metadata None returns early in super()); sliding-window layers
            # keep an unshrunk cache (the spec patch only touches FullAttentionSpec)
            # and may still fall through.
            # cudagraph CAPTURE/warmup dummy (no real OMMX pool): attention is a vLLM
            # splitting_op (runs eager at serve, NOT replayed from this capture), so a
            # zero-fill placeholder lets capture proceed without reading the shrunk cache.
            # Signals: the build_for_cudagraph_capture latch AND the canonical
            # is_current_stream_capturing() (covers profile_cudagraph_memory, which does
            # NOT go through build_for_cudagraph_capture). Safe here: this runs inside the
            # opaque unified_attention op (not dynamo-traced -> no graph-break). Real serve
            # steps (both signals False) still raise below (law #5).
            _capturing = _CAPTURING[0] or (not _REAL_SERVE_STARTED[0])
            if not _capturing:
                try:
                    _capturing = bool(torch.cuda.is_current_stream_capturing())
                except Exception:  # noqa: BLE001
                    _capturing = False
            if _capturing:
                _ommx_route_evidence("PACKED_CAPTURE_STUB",
                                     f"B={_STEP.get('B')} mql="
                                     f"{getattr(attn_metadata, 'max_query_len', None)}")
                output.zero_()
                return output
            raise RuntimeError(
                "[ommx] PACKED-ONLY has no bf16 fallback for this step: "
                f"max_query_len={getattr(attn_metadata, 'max_query_len', None)} "
                f"B={_STEP.get('B')} full_prefill={_STEP.get('full_prefill')} "
                f"dead={bool(getattr(layer, '_ommx_dead', False))}; supported = "
                "full-prompt prefill (in-batch varlen FA) + uniform single-token "
                "decode (OMMX sidecar). Mixed/chunked-prefill steps are outside "
                "the packed bench scope (enable_chunked_prefill=False).")
        return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                               output, output_scale=output_scale,
                               output_block_scale=output_block_scale)

    # ── PACKED-ONLY prefill: varlen FA on the in-batch q/k/v (no cache read) ────

    def _prefill_packed_varlen(self, query, key, value, attn_metadata,
                               output) -> None:
        """PACKED-ONLY full-prompt prefill DIRECTLY on the batch q/k/v.

        With chunked prefill + prefix caching disabled, a full_prefill step carries
        each request's ENTIRE prompt, so causal varlen FlashAttention over the
        in-batch K/V (``cu_seqlens_k == cu_seqlens_q``) is exact — no paged-cache
        read. The sidecar write already happened (``unified_kv_cache_update`` runs
        ``do_kv_cache_update`` before this forward). Errors PROPAGATE: in packed
        mode there is no valid fallback, so a crash here must stay loud (law #5).
        """
        n = int(getattr(attn_metadata, "num_actual_tokens", query.shape[0]))
        H, D = self.num_heads, self.head_size
        q = query[:n].view(n, H, D)
        k = key[:n].view(n, self.num_kv_heads, D)
        v = value[:n].view(n, self.num_kv_heads, D)
        out = output[:n].view(n, H, D)
        cu = attn_metadata.query_start_loc
        m = int(attn_metadata.max_query_len)
        fn = _flash_varlen_fn()
        base = dict(q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=m, max_seqlen_k=m,
                    softmax_scale=float(getattr(self, "scale", 1.0 / (D ** 0.5))),
                    causal=bool(getattr(attn_metadata, "causal", True)))
        try:
            kw = dict(base, out=out,
                      softcap=float(getattr(self, "logits_soft_cap", 0.0) or 0.0))
            fav = getattr(self, "vllm_flash_attn_version", None)
            if fav is not None:
                kw["fa_version"] = fav
            if getattr(self, "sinks", None) is not None:
                kw["s_aux"] = self.sinks
            fn(**kw)
        except TypeError:
            # older varlen signature (no out=/softcap=/fa_version=): minimal
            # universal kwargs, then copy into the vLLM output buffer.
            res = fn(**base)
            if isinstance(res, tuple):
                res = res[0]
            out.copy_(res.view(n, H, D))
        # WORKER-SIDE route-fired evidence (law #5): proves the packed prefill ran
        # the in-batch varlen path (NOT FlashAttention over the shrunk cache).
        _PACKED_PREFILL_FIRES[0] += 1
        _ommx_route_evidence("PACKED_PREFILL_FIRED",
                             f"tokens={n} B={max(0, int(cu.numel()) - 1)}")

    def _ommx_ws(self, layer, q):
        """Per-layer FIXED-ADDRESS decode workspace (o/lse/attn_logits/qzp_buf).

        Pre-allocated ONCE at worst-case geometry (fixed ``num_kv_splits`` from
        max_context) so the decode path does NO per-step allocation — an eager
        speedup AND the fixed-address foundation for CUDA-graph capture (the op
        reads/writes the same buffers every replay). Returns a dict.
        """
        ws = getattr(layer, "_ommx_ws_cache", None)
        if ws is not None:
            return ws
        H, D = self.num_heads, self.head_size
        cfg = self._ommx_cfg()
        max_seq = max(1, int(cfg.max_context))
        nsplits = max(1, int(_auto_num_kv_splits(max_seq, 1, device=q.device,
                                                 kv_heads=self.num_kv_heads)))
        max_groups = max(1, (max_seq + 31) // 32)
        dev = q.device
        import torch as _t
        ws = {
            "o": _t.empty(1, H, D, dtype=_t.float32, device=dev),
            "lse": _t.empty(1, H, dtype=_t.float32, device=dev),
            # D+1 last dim (D o-lanes + 1 lse lane); D alone clobbers adjacent splits' o_v.
            "attn_logits": _t.empty(1, H, nsplits + 1, D + 1, dtype=_t.float32, device=dev),
            "qzp_buf": _t.empty(1, H, max_groups, dtype=_t.float32, device=dev),
            "num_kv_splits": nsplits,
        }
        layer._ommx_ws_cache = ws
        return ws

    def _route_decode(self, layer, query, output) -> bool:
        st = getattr(layer, "_ommx_store", None)
        if st is None:
            return False
        if _GRAPH and _MANAGER is not None:
            return self._route_decode_graph(layer, query, output, st, _MANAGER)
        try:
            H, D = self.num_heads, self.head_size
            q = query.view(-1, H, D)[:1]                    # [1, Hq, D] (single batch)
            di = st.decode_inputs()
            pl = di["planes"]
            ws = self._ommx_ws(layer, q)
            sm_scale = float(getattr(self, "scale", 1.0 / (D ** 0.5)))
            ommx_paged_decode_attention_canonical(
                q, pl["k_base"], pl["k_scale"], pl["k_zp"],
                pl.get("k_oidx"), pl.get("k_oval"),
                pl["v_main"], pl["v_scale"], pl["v_zp"],
                di["k_tail"], di["v_tail"], di["b_tail_len"],
                ws["o"], ws["lse"],
                pl["req_to_token"], pl["req_to_group"], di["b_seq_len"],
                sm_scale=sm_scale, page_size=int(pl["page_size"]),
                k_outliers_per_vector=int(pl["outliers_per_vector"]),
                k_format=str(pl["k_format"]),
                combinadic_read=self._ommx_cfg().combinadic_read,
                k_crank=pl.get("k_crank"),
                k_fp4_mapscale=pl.get("k_fp4_mapscale"),
                k_fp4_mapcenter=pl.get("k_fp4_mapcenter"),
                kv_outlier_map=bool(pl.get("kv_outlier_map", False)),
                kv_int8_scale=bool(pl.get("kv_int8_scale", False)),
                num_kv_splits=ws["num_kv_splits"],
                max_seq_len=di["max_seq_len"], max_tail_len=di["max_tail_len"],
                attn_logits=ws["attn_logits"], qzp_buf=ws["qzp_buf"],
                packed_start_offset=int(di["packed_start_offset"]))
            output.view(-1, H, D)[:1].copy_(ws["o"].to(output.dtype))
            # WORKER-SIDE route-fired evidence (law #5): proves the single-batch (B==1)
            # OMMX decode actually ran — NOT a silent bf16 fallback — so a coherent B==1
            # result is attributable to the OMMX path on the single-batch route.
            _ommx_route_evidence("DECODE_ROUTE_FIRED",
                                 f"fmt={pl['k_format']}")
            return True
        except Exception as e:  # noqa: BLE001
            # FATAL by default (law #5). B==1 is the arm the repo's own repro publishes,
            # so a silent degrade here is the single most misleading failure available.
            _ommx_route_failed("DECODE_ROUTE_DEAD", e,
                               cfg=getattr(self, "_ommx_serving_cfg", None),
                               layer=layer, step_hint="B=1 single-batch decode read")
            return False

    def _route_decode_graph(self, layer, query, output, st, mgr) -> bool:
        """CUDA-graph-capable decode: FULL-capacity planes + manager fixed buffers.

        All tensor args are FIXED shape/address (full-capacity planes + per-layer
        tail/workspace buffers + manager's b_seq_len/b_tail_len/tail_idx); the
        device b_seq_len bounds the live prefix and the FIXED max_seq_len/max_tail_len
        size the grid. No python-value branching, no `.item()` -> capturable.
        """
        try:
            H, D = self.num_heads, self.head_size
            q = query.view(-1, H, D)[:1]
            pl = st.decode_planes_full()
            # capture-safe gather of the sink∪recent window into the store's fixed
            # tail buffers (the new token, just written by do_kv_cache_update, is in
            # tail_idx because on_build set it before this captured forward).
            st.gather_tail_into(mgr.tail_idx, st.k_tail_buf, st.v_tail_buf,
                                mgr.max_tail_len)
            ws = self._ommx_ws(layer, q)
            sm_scale = float(getattr(self, "scale", 1.0 / (D ** 0.5)))
            ommx_paged_decode_attention_canonical(
                q, pl["k_base"], pl["k_scale"], pl["k_zp"],
                pl.get("k_oidx"), pl.get("k_oval"),
                pl["v_main"], pl["v_scale"], pl["v_zp"],
                st.k_tail_buf, st.v_tail_buf, mgr.b_tail_len,
                ws["o"], ws["lse"],
                pl["req_to_token"], pl["req_to_group"], mgr.b_seq_len,
                sm_scale=sm_scale, page_size=int(pl["page_size"]),
                k_outliers_per_vector=int(pl["outliers_per_vector"]),
                k_format=str(pl["k_format"]),
                combinadic_read=self._ommx_cfg().combinadic_read,
                k_crank=pl.get("k_crank"),
                k_fp4_mapscale=pl.get("k_fp4_mapscale"),
                k_fp4_mapcenter=pl.get("k_fp4_mapcenter"),
                kv_outlier_map=bool(pl.get("kv_outlier_map", False)),
                kv_int8_scale=bool(pl.get("kv_int8_scale", False)),
                num_kv_splits=ws["num_kv_splits"],
                max_seq_len=mgr.max_seq_len, max_tail_len=mgr.max_tail_len,
                attn_logits=ws["attn_logits"], qzp_buf=ws["qzp_buf"],
                packed_start_offset=0)
            output.view(-1, H, D)[:1].copy_(ws["o"].to(output.dtype))
            # WORKER-SIDE route-fired evidence (law #5). The captured replay bypasses
            # python, so this fires ONCE at capture/warmup time — a delta-0 counter on
            # replay is expected; the one-time sentinel proves the OMMX graph decode was
            # the captured op (NOT a silent bf16 fallback via the _ommx_dead latch).
            _ommx_route_evidence("GRAPH_ROUTE_FIRED",
                                 f"fmt={pl['k_format']} ctx={mgr.cur_seq}")
            return True
        except Exception as e:  # noqa: BLE001
            # FATAL by default (law #5). A capture-time failure here is especially worth
            # dying on: the captured graph would then hold the bf16 op for the rest of
            # the run, and the replay bypasses python so nothing would report it again.
            mgr.dead = True
            _ommx_route_failed("GRAPH_ROUTE_DEAD", e,
                               cfg=getattr(self, "_ommx_serving_cfg", None),
                               layer=layer,
                               step_hint=("B=1 graph decode read ctx="
                                          f"{getattr(mgr, 'cur_seq', None)}"))
            return False


class OMMXCanonicalBackend(FlashAttentionBackend):
    """vLLM v1 attention backend (SHADOW mode): bf16 cache + OMMX canonical sidecar."""

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"  # must be an AttentionBackendEnum member name (OOT placeholder)

    @staticmethod
    def get_impl_cls() -> type[OMMXCanonicalImpl]:
        return OMMXCanonicalImpl

    @staticmethod
    def get_builder_cls() -> type[OMMXCanonicalMetadataBuilder]:
        return OMMXCanonicalMetadataBuilder

    # get_kv_cache_shape / get_supported_kernel_block_sizes inherit the bf16
    # FlashAttention layout. SHADOW keeps the full bf16 head_size; PACKED-ONLY
    # (OMMX_KV_PACKED_ONLY=1, packed_only.py) shrinks the SPEC head_size so the
    # inherited get_kv_cache_shape allocates a SMALLER paged cache and vLLM budgets
    # proportionally more KV blocks — no shape-method override needed (head_size
    # flows from the spec into get_kv_cache_shape).
    #   THE RATIO — RETRACTED AND RE-MEASURED. This comment used to say "~4.6x". The
    #   canonical PUBLISHED recipe measures 4.375 bit/elem (K+V = 8.750 bit per K/V
    #   element pair) = 32/8.75 = 3.66x versus bf16, obtained by summing the REAL
    #   allocated MultiSeqKVPool tensors for Llama-3.1-8B rather than by evaluating a
    #   formula. It is also NOT "≤3-bit": 2.938 bit/elem (5.45x) needs group_tokens=64
    #   + group_channels=64 + OMMX_KV_OUTLIER_MAP=0 — a DIFFERENT number system from
    #   the one the accuracy results used, so the two figures must not be quoted
    #   together. Full table in the module docstring.
    #   AND THE SHRUNK CACHE IS A RESERVATION: no token is ever stored in it, so vLLM's
    #   "GPU KV cache size: <N> tokens" / "Maximum concurrency <Y>x" log lines simply
    #   restate the byte budget this shrink reserved. They are NOT a validated OMMX
    #   capacity result. The store that really holds the KV is the non-paged sidecar
    #   pool, sized O(num_seqs * max_model_len) (see _resolved_max_num_seqs and the
    #   projection in preflight.py).


__all__ = [
    "OMMXCanonicalBackend",
    "OMMXCanonicalImpl",
    "OMMXCanonicalMetadataBuilder",
    "OMMX_BACKEND_CLASS_PATH",
    # the public "was this really OMMX?" probe — a bench asserts on it before it is
    # allowed to attribute a number to OMMX (see the FAILURE POLICY docstring).
    "ommx_route_health",
]
