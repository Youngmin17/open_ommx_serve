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

CUDA-graph: declared ``UNIFORM_SINGLE_TOKEN_DECODE``; the eager slice runs with
``enforce_eager=True`` (the sidecar append uses a python seq index — NOT yet a
slot-mapping scatter, so it is not capture-safe). FULL-graph capture + the paged
multi-batch store is the follow-up (the build-seam scatter in ``metadata.py``).

PACKED-ONLY capacity mode (the ~4.6x KV compression — the headline OMMX benefit
over FA3) is implemented in ``packed_only.py`` and gated by ``OMMX_KV_PACKED_ONLY=1``
(default OFF; SHADOW stays the correctness-first baseline). It patches
``Attention.get_kv_cache_spec`` so the bf16 paged-cache page budget shrinks by the
OMMX ≤3-bit ratio -> vLLM admits ~4.6x more KV blocks (the capacity win is visible in
vLLM's own "GPU KV cache size: <N> tokens" / "Maximum concurrency <Y>x" log lines).
The INT2 sidecar is the real backing store + decode op; the SHRUNK bf16 paged cache
is a BYTE-BUDGET RESERVATION ONLY — never written (``do_kv_cache_update`` skips the
paged write) and never read (a full-prompt prefill runs varlen FlashAttention
directly on the in-batch q/k/v; the KIVI sink/recent residual lives in the store's
own buffers). Steps neither packed route can serve raise loudly — over the shrunk
pages there is no valid bf16 fallback.
"""
from __future__ import annotations

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
from .metadata import (OMMXBatchedStepManager, OMMXStepManager,
                       detect_full_prefill, detect_step_route)
from .packed_only import packed_only_enabled

OMMX_BACKEND_CLASS_PATH = "ommx_gpu_serve.integration.vllm.backend.OMMXCanonicalBackend"


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in {"0", "false", "off", ""}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


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
# OMMX_ATTN_MAX_NUM_SEQS.
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
_MAX_NUM_SEQS = _env_int("OMMX_ATTN_MAX_NUM_SEQS", 256)
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
# Flips True on the FIRST real OMMX decode-route fire (BATCHED/DECODE/*ROUTE_FIRED). Before
# that, the process is in vLLM INIT (determine_available_memory memory-profiling dummy +
# cudagraph capture) — synthetic decode steps with no OMMX pool that the packed guard must
# STUB (not raise), because is_current_stream_capturing() is False for the eager memory-
# profiling dummy. After real serving starts, the guard raises on genuine anomalies (law #5).
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
    """Record one-time PER-RANK route evidence to the sentinel file (+vLLM log)."""
    if "ROUTE_FIRED" in tag:            # first real OMMX decode route -> serving started
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
        _BMANAGER = OMMXBatchedStepManager(cfg, device, num_seqs=_MAX_NUM_SEQS,
                                           graph=_BATCHED_GRAPH)
    return _BMANAGER


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
        global _MAX_MODEL_LEN
        cfgs = [getattr(self, "vllm_config", None), kwargs.get("vllm_config")]
        cfgs += list(args)
        try:
            from vllm.config import get_current_vllm_config
            cfgs.append(get_current_vllm_config())
        except Exception:
            pass
        for vc in cfgs:
            try:
                mml = int(vc.model_config.max_model_len)
                if mml > 0:
                    _MAX_MODEL_LEN = mml
                    break
            except Exception:
                continue

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
        cap = max(B, int(_MAX_NUM_SEQS))
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
        except Exception:
            if cfg.strict:
                raise
            setattr(layer, "_ommx_dead", True)
            mgr.dead = True
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
        except Exception:
            if cfg.strict:
                raise
            setattr(layer, "_ommx_dead", True)
            mgr.dead = True
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
        if getattr(self, "sliding_window", (-1, -1)) != (-1, -1):
            super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
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
            # ABLATION (timing-only, PARITY-BREAKING): skip the per-step new-token KV
            # pack+regroup/write on a DECODE step so the kernel reads STALE KV (wrong
            # output). (full_step) - (SKIP_WRITE_step) = the per-step KV pack/write cost.
            # Prefill (n>1) is KEPT (else the cache is empty). Default OFF.
            if _ABL_SKIP_WRITE and n == 1:
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
            if self._ommx_cfg().strict:
                raise
            # write-path pre-latch is the ONLY path that latches _ommx_dead WITHOUT a
            # forward-side sentinel (forward's gate short-circuits once dead) -> an empty
            # /tmp/ommx_route_fired.log. Record it so a silent fallback names its cause.
            _ommx_route_evidence("KV_UPDATE_DEAD", f"{type(e).__name__}: {e}")
            setattr(layer, "_ommx_dead", True)

    # ── read path: route uniform single-token decode through the canonical op ───

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
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
        except Exception:
            if self._ommx_cfg().strict:
                raise
            setattr(layer, "_ommx_dead", True)
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
        except Exception:
            if self._ommx_cfg().strict:
                raise
            setattr(layer, "_ommx_dead", True)
            mgr.dead = True
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
    # inherited get_kv_cache_shape allocates a ~4.6x-smaller paged cache + vLLM
    # budgets ~4.6x more KV blocks (the capacity win) — no shape-method override
    # needed (head_size flows from the spec into get_kv_cache_shape).


__all__ = [
    "OMMXCanonicalBackend",
    "OMMXCanonicalImpl",
    "OMMXCanonicalMetadataBuilder",
    "OMMX_BACKEND_CLASS_PATH",
]
