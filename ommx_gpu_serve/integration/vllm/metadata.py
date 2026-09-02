# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""OMMX per-step host seam for the CUDA-graph-capable vLLM backend (SHADOW mode).

⚠ CLUSTER-VALIDATION-REQUIRED: wires into vLLM v1 `CommonAttentionMetadata` field
names + graph capture/replay ordering. Validated EAGER-first, then graph-enabled on
the serving cluster. Single-batch (step1); batched is a follow-up.

CAPTURE-SAFETY MODEL (UNIFORM_SINGLE_TOKEN_DECODE):
- `build()` runs on the HOST every step, OUTSIDE the captured region. It does ALL the
  dynamic work and writes results into FIXED-ADDRESS device buffers (+ python ints):
    * `advance_to(seq)` on each registered per-layer store -> CPU regroup of any newly
      completed 32-token group (the current token is in the recent TAIL, not a packed
      group, so the regroup never needs it);
    * refresh shared `write_pos` (= seq-1, the about-to-be-written token slot),
      `b_seq_len` (packed tokens), `b_tail_len`, and the `tail_idx` gather indices.
- The captured decode forward reads those fixed-address buffers only:
    * `do_kv_cache_update` -> `store.write_token(write_pos, k, v)` (device index_copy_);
    * `forward` -> `store.gather_tail_into(tail_idx, ...)` + the canonical op with the
      FIXED `max_seq_len`/`max_tail_len` ints (grid sizing) and the device `b_seq_len`
      (actual per-replay loop bound) — no `.item()`, no python-value branching.

Step ordering (vLLM): build() [host] -> forward [captured: write new token, then
attend]. on_build sets write_pos = seq-1 so the captured write lands the new token,
and the tail window (which includes seq-1) is what attention reads.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

import torch

from .config import OMMXServingConfig


class OMMXStepManager:
    """Owns the shared per-step fixed-address buffers + drives every layer store."""

    def __init__(self, cfg: OMMXServingConfig, device: Any) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.window = cfg.window()
        self.stores: dict = {}           # layer-id -> CanonicalKVStore (overwrite-safe)
        self.dead = False
        self.capturing = False
        # FIXED-ADDRESS shared device buffers (single request -> batch 1).
        dev = self.device
        self.write_pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.b_seq_len = torch.zeros(1, dtype=torch.int32, device=dev)
        self.b_tail_len = torch.zeros(1, dtype=torch.int32, device=dev)
        t_max = int(self.window.max_tail_len(cfg.max_context))
        self.tail_idx = torch.zeros(t_max, dtype=torch.long, device=dev)
        self.max_tail_cap = t_max
        # FIXED grid-sizing ints (worst case = max_context) for graph.
        self.max_seq_len_fixed = max(1, int(cfg.max_context))
        self.cur_seq = 0
        # NO-SHADOW ring (OMMX_KV_RING): the per-layer stores keep only a [ring_cap,H,D]
        # bf16 ring instead of the [max_model_len,H,D] shadow. The shared write_pos /
        # tail_idx buffers below still carry ABS token positions; each store's
        # write_token / gather_tail_into remaps abs -> ring slot via capture-safe tensor
        # ops (the single owner of the remap, so eager + graph + the lmcache/sglang bulk
        # callers all stay consistent). Here we only ASSERT the live abs window stays
        # within the ring's collision-free capacity (the off-by-one in-ring check).
        import os as _os
        _rr = _os.environ.get("OMMX_KV_RING")
        self.kv_ring = bool(_rr) and _rr not in {"0", "false", "off", "no"}
        gt = int(self.window.group_tokens)
        self._ring_rec = int(self.window.recent_window) + 2 * gt   # collision-free span
        self._ring_sink = int(self.window.sink_tokens)
        # p50 host-seam elim: gate the per-step 32-layer advance_to loop to group-boundary
        # crossings only (_last_boundary), + a cached device sink-arange so the per-step
        # tail_idx build is device-only (no cpu torch.tensor, no H2D copy).
        self._last_boundary = -1
        self._sink_arange = torch.arange(self.window.sink_tokens, dtype=torch.long,
                                         device=dev)
        self._seq_pin = None             # reused pinned buf for device-only seq_lens D2H

    def register(self, layer_id: int, store) -> None:
        # overwrite-safe: a new sequence's prefill re-registers each layer's fresh
        # store under the same id, evicting the stale one (no accumulation).
        self.stores[int(layer_id)] = store

    def reset(self) -> None:
        """Drop registered stores + clear dead latch (manual full reset)."""
        self.stores = {}
        self.cur_seq = 0
        self.dead = False
        self._last_boundary = -1

    # ── host seam ───────────────────────────────────────────────────────────────

    def on_build(self, common_attn_metadata: Any) -> None:
        if self.dead:
            return
        try:
            seq = self._current_seq(common_attn_metadata)
            self._refresh(seq)
        except Exception:
            if self.cfg.strict:
                raise
            self.dead = True   # backend falls back to bf16 FlashAttention

    def _current_seq(self, common: Any) -> int:
        # Single-batch: the one request's current sequence length. Prefer the host
        # copy (NO device sync). vLLM v1 names vary by version -> try in order.
        for attr in ("seq_lens_cpu", "seq_lens_cpu_upper_bound"):
            v = getattr(common, attr, None)
            if v is not None:
                return int(v[0])                       # host tensor, no sync
        # Only a device seq_lens is exposed -> 1-element D2H into a reused pinned buffer.
        # Cheaper than `.item()` on an arbitrary-stride view (no full-tensor materialize)
        # and the only per-step host coupling left in the seam.
        v = getattr(common, "seq_lens", None)
        if v is not None:
            if self._seq_pin is None:
                self._seq_pin = torch.empty(1, dtype=v.dtype, pin_memory=True)
            self._seq_pin.copy_(v.reshape(-1)[:1], non_blocking=False)
            return int(self._seq_pin[0])
        raise RuntimeError("CommonAttentionMetadata: no seq_lens field found")

    def _refresh(self, seq: int) -> None:
        seq = int(seq)
        self.cur_seq = seq
        sink = self.window.sink_tokens
        boundary = self.window.boundary(seq)
        # regroup completed groups ONLY when a new group boundary is crossed (every
        # group_tokens steps). At non-boundary decode steps advance_to is a no-op, so
        # gating skips the per-step 32-layer python loop — the p50 host-seam (law #1:
        # decode is launch/host-bound, so kill per-step host work).
        if boundary != self._last_boundary:
            for st in self.stores.values():
                st.advance_to(seq)
            self._last_boundary = boundary
        packed_tokens = max(0, boundary - sink)
        n_rec = max(0, seq - boundary)                 # recent window [boundary, seq)
        _valid_sink = min(sink, seq)                   # FIX(seq<sink): rows [seq,sink) don't exist yet
        n_tail = _valid_sink + n_rec                   # tail = sink ∪ [boundary, seq)
        # RING in-ring assertion (host, outside capture): the abs positions the captured
        # forward touches this step span [boundary - gt, seq) (the group being packed at
        # the boundary crossing + the recent tail + the about-to-write token at seq-1).
        # That span must be strictly NARROWER than the ring's recent capacity Rrec, else
        # two live abs positions alias the same ring slot. Off-by-one: width == Rrec
        # already aliases endpoints, so the guard is `< Rrec`. Sink pinned separately.
        if self.kv_ring and seq > 0:
            gt = int(self.window.group_tokens)
            lo = max(sink, boundary - gt)             # oldest non-sink abs row still read
            live_span = max(0, (seq - 1) - lo + 1)    # inclusive of the seq-1 write slot
            if live_span > self._ring_rec:
                raise AssertionError(
                    f"OMMX_KV_RING live span {live_span} > Rrec {self._ring_rec} "
                    f"(seq={seq}, boundary={boundary}, gt={gt}) — ring would alias")
        if n_tail > self.max_tail_cap:
            n_rec = self.max_tail_cap - sink
            n_tail = self.max_tail_cap
        # refresh fixed-address buffers IN PLACE (the captured forward reads these). The
        # buffers carry ABS positions; each store remaps abs->ring slot capture-safely.
        self.write_pos.fill_(max(0, seq - 1))
        self.b_seq_len.fill_(int(packed_tokens))
        self.b_tail_len.fill_(int(n_tail))
        # build tail_idx (sink ∪ [boundary, boundary+n_rec)) DEVICE-SIDE: torch.arange on
        # the device + a single device zero — NO cpu torch.tensor, NO H2D copy (the
        # per-step host-seam stall the bf16 path never pays). boundary/seq are host ints.
        self.tail_idx.zero_()
        if _valid_sink:
            self.tail_idx[:_valid_sink] = self._sink_arange[:_valid_sink]
        if n_rec > 0:
            self.tail_idx[_valid_sink:_valid_sink + n_rec] = torch.arange(
                boundary, boundary + n_rec, device=self.device, dtype=torch.long)

    @property
    def max_seq_len(self) -> int:
        return self.max_seq_len_fixed

    @property
    def max_tail_len(self) -> int:
        return self.max_tail_cap


# ── BATCHED (B>1) host seam ───────────────────────────────────────────────────


def _host_list(v: Any) -> Optional[list]:
    """Coerce a vLLM metadata field (tensor / numpy / list) to a host python list.

    Prefers a no-sync path; tolerates a device tensor by pulling it to CPU (build()
    runs on the HOST, outside any capture, so a one-shot copy is acceptable for the
    eager batched bring-up).
    """
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v.detach().to("cpu").reshape(-1).tolist()
    try:
        return [int(x) for x in v]
    except TypeError:
        return None


# Explicit (UNSAFE) escape hatch for the old row-index request identity. Row indices are
# only stable when the batch order cannot change — a single-request bench `generate()`.
# Under continuous batching vLLM reorders rows across steps, so row-index slots swap KV
# between requests. Opt in with OMMX_UNSAFE_ROW_INDEX_SLOTS=1; it warns LOUDLY once.
_UNSAFE_ROW_INDEX_ENV = "OMMX_UNSAFE_ROW_INDEX_SLOTS"
# vLLM v1 keeps block id 0 as the reserved null block and zero-fills the block-table rows
# of requests that own no KV blocks. So "column 0 == 0 for EVERY row" means the step
# carries no per-request block identity at all (the init memory-profiling / cudagraph-
# capture dummy batches), NOT that B requests share one block. See `_slots`.
_NULL_BLOCK_ID = 0

_WARNED_ONCE: set = set()


def _env_on(name: str) -> bool:
    """Env gate, same convention as backend/config (`0|false|off|no|empty` = OFF)."""
    return os.environ.get(name, "").strip().lower() not in {"", "0", "false", "off", "no"}


def _warn_once(key: str, msg: str) -> None:
    """Emit ONE loud warning per process for ``key`` — stderr AND the vLLM logger.

    Every caller is a path that KEEPS SERVING after something the operator must see
    (an unsafe identity escape hatch, or the pool going dead). Law #5 (no silent
    fallback) is satisfied by evidence, so the evidence must not itself be swallowed:
    stderr goes first (always available), the logger copy is best-effort and reports
    its own failure rather than passing silently.
    """
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    print(f"[OMMX] WARNING: {msg}", file=sys.stderr, flush=True)
    try:
        from vllm.logger import init_logger      # lazy: CPU boxes have no vllm
        init_logger(__name__).warning("[OMMX] %s", msg)
    except Exception as exc:                     # never mask the evidence above
        print(f"[OMMX] (vllm logger unavailable: {type(exc).__name__}: {exc})",
              file=sys.stderr, flush=True)


class OMMXSlotAllocationError(RuntimeError):
    """The static-slot pool cannot serve this step's request set.

    A ``RuntimeError`` subclass so existing ``except RuntimeError`` handlers keep
    working, but distinguishable at the ``on_build`` seam: a slot fault is a
    CORRECTNESS refusal (aliasing two live requests would corrupt KV across
    requests), so it is reported LOUDLY even when the non-strict seam degrades the
    worker to bf16 instead of dying.
    """


def assign_slots(keys: list, key_to_slot: dict, free_slots: list,
                 num_seqs: int) -> list:
    """Map this step's stable request keys -> fixed pool slots (state MUTATED in place).

    Dependency-free (no torch, no vllm, no GPU) so the allocator — the one piece of the
    batched seam that can silently corrupt KV ACROSS requests — is unit-testable on a
    CPU box. ``OMMXBatchedStepManager._slots`` is a thin wrapper over it.

    Arguments:
      * ``keys``: this step's per-row stable request key (first physical block id), in
        batch-row order. MUST be unique — one key is one live request.
      * ``key_to_slot`` / ``free_slots``: the manager's persistent allocator state,
        mutated in place. Keys that no longer appear in ``keys`` (request finished or
        was preempted) release their slot back onto ``free_slots``.
      * ``num_seqs``: the static pool capacity (``OMMX_ATTN_MAX_NUM_SEQS``).

    Returns row -> slot, ``len(out) == len(keys)``.

    Raises ``OMMXSlotAllocationError`` — NEVER aliases. The two ways a batched decode
    used to corrupt KV silently were (a) handing out ``len(key_to_slot) % num_seqs``
    when the free list ran dry (that slot is usually owned by a LIVE request) and
    (b) two rows resolving to the same key (prefix caching shares the first physical
    block between requests with a common prompt prefix). Both now raise.
    """
    num_seqs = int(num_seqs)
    keys = [int(k) for k in keys]
    # (c) duplicate keys — checked BEFORE any mutation so a rejected step leaves the
    # allocator state untouched (the caller may degrade to bf16 and keep serving).
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise OMMXSlotAllocationError(
            f"OMMX batched slot pool: {len(dup)} duplicate request key(s) {dup[:8]} in a "
            f"batch of {len(keys)} rows (keys[:16]={keys[:16]}). Two batch rows resolved "
            "to the SAME stable key, i.e. the same FIRST PHYSICAL BLOCK: vLLM prefix "
            "caching hands requests that share a prompt prefix one shared block, so the "
            "block id stops being a per-request identity. OMMX's static-slot pool cannot "
            "serve that set - both rows land on one slot and overwrite each other's "
            "KV. Fix: run the OMMX backend with prefix caching OFF "
            "(--no-enable-prefix-caching / enable_prefix_caching=False).")
    # release the slots of requests that vanished from the batch (finished/preempted).
    live = set(keys)
    for k in list(key_to_slot):
        if k not in live:
            free_slots.append(int(key_to_slot.pop(k)))
    owned = set(int(v) for v in key_to_slot.values())
    out: list = []
    for k in keys:
        slot = key_to_slot.get(k)
        if slot is None:
            # (a) exhaustion: refuse, do NOT reuse a live slot.
            if not free_slots:
                raise OMMXSlotAllocationError(
                    f"OMMX batched slot pool EXHAUSTED: {len(key_to_slot)} live request "
                    f"key(s) already hold all {num_seqs} slots and request key {k} has "
                    "nowhere to go. The pool is sized ONCE at OMMX_ATTN_MAX_NUM_SEQS "
                    "(default 256) and must be >= the engine's max_num_seqs. Fix: raise "
                    "OMMX_ATTN_MAX_NUM_SEQS to at least the engine's max_num_seqs, or "
                    "lower the engine's max_num_seqs. (Refusing to recycle a live slot: "
                    "two requests on one slot silently corrupt each other's KV.)")
            slot = int(free_slots.pop(0))
            # free-list invariants: a slot must be in range and NOT already owned. A
            # violation means the free list was double-pushed somewhere (a released slot
            # re-appended twice) - that is the aliasing bug in another disguise, so it
            # raises here rather than being handed out.
            if not (0 <= slot < num_seqs):
                raise OMMXSlotAllocationError(
                    f"OMMX batched slot pool: free list yielded out-of-range slot {slot} "
                    f"(pool cap num_seqs={num_seqs}) - allocator state is corrupt.")
            if slot in owned:
                raise OMMXSlotAllocationError(
                    f"OMMX batched slot pool: free list yielded slot {slot}, which is "
                    f"already owned by a live request (owned={sorted(owned)[:8]}...) - "
                    "double-freed slot, refusing to alias two requests onto it.")
            key_to_slot[k] = slot
            owned.add(slot)
        out.append(int(slot))
    return out


class OMMXBatchedStepManager:
    """Batched (B>1) host seam: per-layer ``MultiSeqKVPool`` + ``BatchedStepPlanner``.

    Generalizes ``OMMXStepManager`` from single-batch to a shared-pool batched store.
    EAGER (``enforce_eager=True``) — the build seam reads per-request ``seq_lens`` /
    ``query_start_loc`` from ``CommonAttentionMetadata`` on the HOST and the pool's
    CPU regroup + bf16 history live outside any capture. Capacity-first SHADOW: vLLM
    keeps its bf16 cache; this owns the OMMX quantized sidecar pool, whose MEASURED
    footprint under the PUBLISHED recipe (i2f4 K + i2 V, 6 outliers/vector,
    group_tokens=32, group_channels=32, outlier map ON, pow2/int8 scale) is 8.750 bit
    per K/V element PAIR = 4.375 bit/elem = 3.66x vs bf16 — summed from the really
    allocated pool tensors, not from a formula. "≤3-bit" is a DIFFERENT recipe
    (group_tokens=64 + group_channels=64 + outlier map OFF -> 2.938 bit/elem), which
    is NOT the one the accuracy numbers were produced with; never quote the two in one
    sentence. ``packed_only.kv_bits_breakdown()`` prints the resolved recipe next to
    the number for exactly this reason.

    Per-worker singleton (one manager for all layers). Request-slot mapping keys off the
    request's FIRST PHYSICAL BLOCK ID (``_req_keys`` -> ``assign_slots``), NEVER the batch
    row index: vLLM reorders rows across steps under continuous batching, so a row index
    identifies nothing. A step whose request identity cannot be established, or whose keys
    do not fit the static pool, raises ``OMMXSlotAllocationError`` instead of aliasing two
    live requests onto one slot. The planner tracks per-slot regroup deltas across steps.
    The pool is sized to ``max_num_seqs`` once; layers share the same
    num_seqs/max_seq_len geometry.
    """

    def __init__(self, cfg: OMMXServingConfig, device: Any, num_seqs: int,
                 graph: bool = False) -> None:
        from ommx_gpu_serve.attention.kv_pool import MultiSeqKVPool  # lazy
        from ommx_gpu_serve.integration.common.batched_seam import BatchedStepPlanner

        self._MultiSeqKVPool = MultiSeqKVPool
        self.cfg = cfg
        self.device = torch.device(device)
        self.window = cfg.window()
        self.graph = bool(graph)
        self.num_seqs = max(1, int(num_seqs))
        self.dead = False
        self.pools: dict = {}            # layer-id -> MultiSeqKVPool
        self.planner = BatchedStepPlanner(self.window, cfg.max_context,
                                          device=self.device)
        # per-step host state set by on_build, consumed by the Impl's write/read.
        self.seq_lens: list[int] = []        # per-request CURRENT seq length
        self.q_start: list[int] = []         # query_start_loc (cumulative, len B+1)
        self.slots: list[int] = []           # batch-row -> stable pool slot
        self.is_uniform_decode = False       # all requests contribute exactly 1 token
        self.max_seq_len_fixed = max(1, int(cfg.max_context))
        # ── CUDA-graph-capture fixed-address shared device buffers (B>1 graph path) ──
        # Mirrors OMMXStepManager (B==1) but [cap,*] shaped. The host on_build refreshes
        # the first-B entries OUTSIDE capture; the captured batched write/read read ONLY
        # these (no host list -> torch.tensor each step). Sliced [:B] at capture (B is a
        # python constant for that graph size), so each captured graph binds a fixed view.
        cap = self.num_seqs
        T = int(self.window.max_tail_len(cfg.max_context))
        dev = self.device
        self.cap = cap
        self.max_tail_cap = T
        self.b_sel = torch.zeros(cap, dtype=torch.long, device=dev)       # row -> slot
        self.b_write_col = torch.zeros(cap, dtype=torch.long, device=dev)  # write column
        self.b_seq_len = torch.zeros(cap, dtype=torch.int32, device=dev)   # packed tokens
        self.b_tail_len = torch.zeros(cap, dtype=torch.int32, device=dev)  # bf16 rows
        self.b_tail_idx = torch.zeros(cap, T, dtype=torch.long, device=dev)
        # host staging (filled in on_build, one H2D copy each into the device buffers).
        self._sel_host = torch.zeros(cap, dtype=torch.long)
        self._col_host = torch.zeros(cap, dtype=torch.long)
        self._bseq_host = torch.zeros(cap, dtype=torch.int32)
        self._btail_host = torch.zeros(cap, dtype=torch.int32)
        self._tidx_host = torch.zeros(cap, T, dtype=torch.long)
        # FIXED grid-sizing int (worst case = max_context) for the captured op.
        self.cur_B = 0
        self.capturing = False
        # stable request identity -> pool slot. vLLM reorders batch rows across steps
        # (continuous batching); the FIRST physical block id per request is stable for
        # that request's lifetime, so it keys the slot (reorder/preemption-safe). Frees
        # on the request's slot reset (seq_before <= 0, owned by the Impl's writer).
        self._key_to_slot: dict[int, int] = {}
        self._free_slots: list[int] = list(range(self.num_seqs))
        # True while `_req_keys` is serving ROW INDICES (the OMMX_UNSAFE_ROW_INDEX_SLOTS
        # hatch) instead of block ids. Row index 0 collides numerically with the reserved
        # null block id, so `_slots`'s null-block dummy branch must NOT look at row-index
        # keys — under the hatch, key 0 is a real request.
        self._keys_are_row_index = False

    # ── pool lifecycle ───────────────────────────────────────────────────────

    def pool(self, layer_id: int, device: Any):
        p = self.pools.get(int(layer_id))
        if p is None:
            c = self.cfg
            p = self._MultiSeqKVPool(
                head_dim=c.head_dim, n_kv_heads=c.n_kv_heads,
                num_seqs=self.num_seqs, max_seq_len=int(c.max_context),
                k_format=c.k_format, outliers_per_vector=c.outliers_per_vector,
                outlier_select=c.outlier_select, outlier_repr=c.outlier_repr,
                use_pow2=c.use_pow2, window=self.window,
                # K outlier value codec: pass the RESOLVED config value.
                # MultiSeqKVPool takes Optional[bool] where None = "read
                # OMMX_KV_OUTLIER_MAP yourself", so leaving it out made
                # OMMXServingConfig.kv_outlier_map DEAD config: a caller that built the
                # config in python (tests, harnesses) got the env's recipe instead of
                # its own. resolve_serving_config() already folds the env in, so the
                # config stays the single source of truth for the recipe.
                kv_outlier_map=c.kv_outlier_map,
                group_channels=c.group_channels, device=device)
            self.pools[int(layer_id)] = p
        return p

    def reset(self) -> None:
        self.pools = {}
        self.planner.reset()
        self.seq_lens = []
        self.q_start = []
        self.slots = []
        self.is_uniform_decode = False
        self.dead = False
        self._key_to_slot = {}
        self._free_slots = list(range(self.num_seqs))
        self._keys_are_row_index = False
        self.cur_B = 0

    # ── host seam ─────────────────────────────────────────────────────────────

    def on_build(self, common_attn_metadata: Any) -> None:
        if self.dead:
            return
        try:
            self._refresh(common_attn_metadata)
        except OMMXSlotAllocationError as exc:
            # The allocator refused to alias two live requests. Non-strict mode still
            # degrades this worker to bf16 FlashAttention (vLLM keeps its own bf16 cache
            # in SHADOW mode, so the OUTPUT stays correct) - but never SILENTLY: the
            # cause and the fix go to stderr + the vLLM logger once before the dead
            # latch, so a run that quietly stopped measuring OMMX is visible.
            _warn_once("slot-fault", f"{exc}\n"
                       "  -> the OMMX batched pool is now DEAD for this worker; steps "
                       "route to bf16 FlashAttention (correct output, NOT an OMMX "
                       "measurement). Set OMMX_STRICT=1 to fail hard instead.")
            if self.cfg.strict:
                raise
            self.dead = True
        except Exception:
            if self.cfg.strict:
                raise
            self.dead = True

    def _refresh(self, common: Any) -> None:
        seq_lens = self._seq_lens(common)
        q_start = self._query_start_loc(common)
        B = len(seq_lens)
        self.seq_lens = seq_lens
        self.q_start = q_start
        # OVER-CAPACITY GUARD: vLLM's init _dummy_run drives the model with a transient
        # dummy batch that can EXCEED the serving max_num_seqs (the pool/buffer cap).
        # That step has no real KV to pack; route it to bf16 (super()) instead of dying.
        # cur_B/slots stay empty so the captured read/write skip it. Does NOT latch dead
        # (a later real B<=cap decode must still route through OMMX).
        if B > self.num_seqs:
            self.seq_lens = []   # write path uses len(seq_lens) -> 0 -> skips
            self.q_start = None
            self.slots = []
            self.is_uniform_decode = False
            self.cur_B = 0
            return
        self.slots = self._slots(common, B)  # batch row -> stable pool slot
        # uniform single-token decode: every request contributes exactly one query
        # token (q_start strictly +1 per request). prefill / chunked / mixed -> False.
        if q_start is not None and len(q_start) == B + 1:
            self.is_uniform_decode = all(
                (q_start[i + 1] - q_start[i]) == 1 for i in range(B))
        else:
            self.is_uniform_decode = False
        # plan() advances the planner's per-slot regroup history; the Impl uses the
        # pool's own decode_inputs_batched (which recomputes b_seq/b_tail) for the op,
        # so the plan here is only used to drive per-request regroup at write time.
        if B and self.is_uniform_decode:
            self.planner.plan(seq_lens, self.slots)
        # ── GRAPH path: regroup each pool (HOST, outside capture) + refresh the
        # fixed-address device buffers the captured write/read read. Done for every
        # uniform decode step (the captured forward then writes the new token at
        # b_write_col and attends [0, b_seq_len) + the gathered tail). ───────────────
        if self.graph and B and self.is_uniform_decode:
            self._refresh_graph_buffers(seq_lens, self.slots)

    def _refresh_graph_buffers(self, seq_lens: list[int],
                               slots: list[int]) -> None:
        """HOST seam (outside capture): advance pools + fill fixed device buffers.

        Mirrors ``OMMXStepManager._refresh`` per request. The current decode token at
        ``seq-1`` is NOT yet in ``k_hist`` (the captured ``do_kv_cache_update`` writes
        it at ``b_write_col`` AFTER this); it always lands in the bf16 TAIL window, so
        the CPU regroup here (which packs completed groups strictly below the boundary)
        never needs it. ``b_seq_len`` (packed tokens) / ``b_tail_idx`` (sink∪recent
        positions, INCLUDING ``seq-1`` so the captured tail gather reads the just-
        written token) are sized from ``seq``."""
        B = len(seq_lens)
        win = self.window
        sink = win.sink_tokens
        T = self.max_tail_cap
        self._tidx_host[:B].zero_()
        for i in range(B):
            r = int(slots[i])
            seq = int(seq_lens[i])
            # advance the pool's host seq + regroup any newly completed group (CPU pack).
            # do this on EVERY layer's pool (they share seq geometry) — the regroup reads
            # k_hist rows already written in prior captured steps.
            for pool in self.pools.values():
                pool.seq_len[r] = seq
                target = win.num_groups(seq)
                if target > pool.packed_groups[r]:
                    pool.regroup(r)
            boundary = win.boundary(seq)
            packed_tokens = max(0, boundary - sink)
            tail = win.tail_indices(seq)                # sink ∪ [boundary, seq)
            n_tail = min(len(tail), T)
            self._sel_host[i] = r
            self._col_host[i] = max(0, seq - 1)
            self._bseq_host[i] = int(packed_tokens)
            self._btail_host[i] = int(n_tail)
            if n_tail:
                self._tidx_host[i, :n_tail] = torch.tensor(tail[:n_tail],
                                                           dtype=torch.long)
        self.cur_B = B
        # one H2D per control buffer (non_blocking; the next captured forward depends).
        self.b_sel[:B].copy_(self._sel_host[:B], non_blocking=True)
        self.b_write_col[:B].copy_(self._col_host[:B], non_blocking=True)
        self.b_seq_len[:B].copy_(self._bseq_host[:B], non_blocking=True)
        self.b_tail_len[:B].copy_(self._btail_host[:B], non_blocking=True)
        self.b_tail_idx[:B].copy_(self._tidx_host[:B], non_blocking=True)

    def _seq_lens(self, common: Any) -> list[int]:
        # per-request CURRENT seq length (host copy preferred, version-tolerant).
        for attr in ("seq_lens_cpu", "seq_lens_cpu_upper_bound", "seq_lens"):
            lst = _host_list(getattr(common, attr, None))
            if lst is not None:
                return [int(x) for x in lst]
        raise RuntimeError("CommonAttentionMetadata: no seq_lens field found")

    def _query_start_loc(self, common: Any) -> Optional[list[int]]:
        for attr in ("query_start_loc_cpu", "query_start_loc"):
            lst = _host_list(getattr(common, attr, None))
            if lst is not None:
                return [int(x) for x in lst]
        return None

    def _req_keys(self, common: Any, B: int) -> list[int]:
        """Stable per-request key (first physical block id) for reorder-safe slots.

        ``block_table_tensor`` [B, max_blocks]; column 0 is the request's first physical
        block, stable for that request's lifetime. This is the ONLY sound request
        identity at this seam — vLLM reorders batch ROWS across steps under continuous
        batching, so a row index identifies nothing.

        MANDATORY (law #5, no silent fallback): if neither ``block_table_tensor`` nor
        ``block_table`` yields a column 0, this RAISES. The row-index fallback that used
        to sit here swapped KV between requests the moment vLLM reordered the batch, and
        it did so from a bare ``except Exception: break`` — no evidence at all. It now
        survives only behind the explicit ``OMMX_UNSAFE_ROW_INDEX_SLOTS=1`` escape hatch
        (a single-request bench ``generate()``, where the order cannot change), which
        warns LOUDLY once, to stderr and to the vLLM logger, before it is used.

        ``block_table_tensor`` is still preferred over ``block_table``; every candidate's
        failure reason is collected and reported in the raise.
        """
        causes: list[str] = []
        for attr in ("block_table_tensor", "block_table"):   # tensor preferred
            bt = getattr(common, attr, None)
            if bt is None:
                causes.append(f"{attr}=absent")
                continue
            try:
                col0 = bt[:B, 0] if hasattr(bt, "shape") else bt
                host = _host_list(col0)
                if host is None:
                    raise TypeError(
                        f"{type(bt).__name__} column 0 did not coerce to a host list")
                keys = [int(x) for x in host[:B]]
            except Exception as exc:
                causes.append(f"{attr}={type(exc).__name__}: {exc}")
                continue
            if len(keys) != B:
                causes.append(f"{attr}=got {len(keys)} column-0 entries for B={B}")
                continue
            self._keys_are_row_index = False
            return keys
        detail = "; ".join(causes) if causes else "no candidates"
        if _env_on(_UNSAFE_ROW_INDEX_ENV):
            _warn_once("row-index-slots",
                       f"{_UNSAFE_ROW_INDEX_ENV}=1: no usable block table on "
                       f"CommonAttentionMetadata ({detail}), so OMMX is keying pool "
                       "slots by BATCH ROW INDEX. Only safe while the batch order "
                       "cannot change (a single-request bench generate()). Under "
                       "continuous batching vLLM reorders rows between steps and this "
                       "SWAPS KV between requests. Unset the variable to fail loudly "
                       "instead.")
            self._keys_are_row_index = True
            return list(range(B))
        raise OMMXSlotAllocationError(
            "OMMX batched slot pool: no usable per-request block table on "
            f"CommonAttentionMetadata ({detail}), so there is no stable request identity "
            "to key pool slots by. Refusing to fall back to the batch row index: vLLM "
            "reorders rows across steps under continuous batching, which would swap KV "
            "between requests with no evidence. Fix: use a vLLM build whose "
            "CommonAttentionMetadata exposes block_table_tensor (v1, >=0.21), or - for a "
            f"single-request bench where the order cannot change - set "
            f"{_UNSAFE_ROW_INDEX_ENV}=1 to accept row-index slots explicitly.")

    def _slots(self, common: Any, B: int) -> list[int]:
        """Batch row -> stable pool slot for this step (see ``assign_slots``)."""
        if B <= 0:
            # A request-less step (vLLM init profiling / a dummy build) is NOT evidence
            # that any request finished, so it must not run the release pass - that would
            # hand every live request's slot back to the free list and re-map it (its
            # packed history then belongs to someone else). Nothing to map: return [].
            return []
        keys = self._req_keys(common, B)
        if not self._keys_are_row_index and all(k == _NULL_BLOCK_ID for k in keys):
            # Every row's column-0 entry is the reserved null/unallocated block id: this
            # step's "requests" own no KV blocks, i.e. it is a vLLM INIT dummy batch (the
            # memory-profiling _dummy_run, or the cudagraph-capture dummy - `capturing`).
            # The keys carry no identity, so they are NOT registered: the rows are mapped
            # onto currently-FREE slots only, which cannot alias a live request's slot,
            # and the free list is left untouched for the next real step. (Registering
            # them would trip the duplicate-key guard and kill vLLM init; the pre-fix code
            # mapped every row onto slot 0.) Dummy KV written into a free slot is
            # harmless: a real request resets its slot on its first write (seq_before<=0).
            if len(self._free_slots) < B:
                raise OMMXSlotAllocationError(
                    f"OMMX batched slot pool: dummy/unallocated batch of B={B} rows "
                    f"(every column 0 == null block {_NULL_BLOCK_ID}) but only "
                    f"{len(self._free_slots)} free slot(s) of {self.num_seqs}. Refusing "
                    "to place dummy rows on live requests' slots. Fix: raise "
                    "OMMX_ATTN_MAX_NUM_SEQS above the engine's max_num_seqs.")
            if not self.capturing:
                # expected under the capture latch; anywhere else the operator must see it
                # once, because it means this step carried no request identity.
                _warn_once("null-block-keys",
                           f"OMMX batched slots: every row of a B={B} step exposes the "
                           f"null/unallocated block id {_NULL_BLOCK_ID} in block-table "
                           "column 0, so the step carries NO stable request identity. "
                           "Treating it as a vLLM init/profiling dummy: its rows are "
                           "mapped onto FREE pool slots and are not registered. If this "
                           "appears during real serving, request identity is broken for "
                           "this vLLM build - the batched pool results are not "
                           "trustworthy; report it with the vLLM version.")
            return [int(s) for s in self._free_slots[:B]]
        return assign_slots(keys, self._key_to_slot, self._free_slots, self.num_seqs)

    @property
    def max_seq_len(self) -> int:
        return self.max_seq_len_fixed

    @property
    def max_tail_len(self) -> int:
        return self.max_tail_cap


# Sentinel ``B`` returned by ``detect_step_route`` for a step it could NOT classify at
# all (no readable ``seq_lens`` field, or reading one raised). DELIBERATELY NEGATIVE so
# it can never be read as a request count: a genuinely EMPTY batch is B == 0 (vLLM's
# init memory-profiling / cudagraph-capture dummies legitimately drive those) and a
# single request is B == 1. Before this existed all three collapsed onto 0, and the
# backend's ``elif B <= 1`` consumer then served an UNCLASSIFIED multi-request decode
# through the single-batch route, whose ``query[:1]`` / ``output[:1]`` slices compute
# row 0 only and leave rows 1..B-1 unwritten — while the sentinel still recorded
# DECODE_ROUTE_FIRED, so the bench called that run an OMMX measurement. Consumers MUST
# test for this value explicitly and REFUSE the step (law #5: no silent fallback);
# treating it as "B is small" is exactly the guess that produced the wrong answer.
STEP_ROUTE_UNCLASSIFIED = -1


def detect_step_route(common_attn_metadata: Any) -> tuple[int, bool]:
    """Classify ONE step from ``CommonAttentionMetadata`` -> ``(B, is_uniform_decode)``.

    Standalone (needs no manager instance) so the metadata builder's ``build()`` can set
    the per-step read/write route BEFORE the batched manager exists (the manager is
    created lazily by the Impl on the first batched step). Mirrors
    ``OMMXBatchedStepManager._refresh``'s uniform-decode test:

      * ``B`` = number of requests (len of per-request ``seq_lens``);
      * ``is_uniform_decode`` = every request contributes exactly one query token
        (``query_start_loc`` strictly +1 per request) -> a uniform single-token decode.
        prefill / chunked / mixed steps -> False.

    Returns ``(STEP_ROUTE_UNCLASSIFIED, False)`` — i.e. ``B == -1`` — when the step could
    not be classified (no ``seq_lens`` field found, or an accessor raised). That is NOT
    the same answer as ``(0, False)``, which reports a genuinely EMPTY batch: see
    ``STEP_ROUTE_UNCLASSIFIED`` above for why conflating the two silently corrupted
    requests 1..B-1, and ``backend.OMMXCanonicalImpl.forward`` for the consumer that now
    refuses the unclassified step instead of guessing B==1.
    """
    try:
        seq_lens = None
        for attr in ("seq_lens_cpu", "seq_lens_cpu_upper_bound", "seq_lens"):
            seq_lens = _host_list(getattr(common_attn_metadata, attr, None))
            if seq_lens is not None:
                break
        if seq_lens is None:
            return STEP_ROUTE_UNCLASSIFIED, False
        B = len(seq_lens)
        q_start = None
        for attr in ("query_start_loc_cpu", "query_start_loc"):
            q_start = _host_list(getattr(common_attn_metadata, attr, None))
            if q_start is not None:
                break
        if B >= 1 and q_start is not None and len(q_start) == B + 1:
            uniform = all((q_start[i + 1] - q_start[i]) == 1 for i in range(B))
        else:
            uniform = False  # B==0 dummy/profiling run -> never a decode
        return B, bool(uniform)
    except Exception:
        return STEP_ROUTE_UNCLASSIFIED, False


def detect_full_prefill(common_attn_metadata: Any) -> bool:
    """True iff EVERY request's query span covers its WHOLE sequence (q_len == seq_len).

    The PACKED-ONLY prefill gate: with chunked prefill + prefix caching disabled, a
    True step carries each request's complete causal context in the batch k/v, so the
    prefill read can run varlen FlashAttention directly on it — never the shrunk
    paged cache. Mirrors ``detect_step_route``'s field tolerance; returns False on
    any doubt (the backend then fails LOUD rather than mis-serving a step whose
    context is not fully in-batch). Note ``seq_lens_cpu_upper_bound`` can only
    over-state seq_len, so doubt errs toward False — the safe direction.
    """
    try:
        seq_lens = None
        for attr in ("seq_lens_cpu", "seq_lens_cpu_upper_bound", "seq_lens"):
            seq_lens = _host_list(getattr(common_attn_metadata, attr, None))
            if seq_lens is not None:
                break
        if not seq_lens:
            return False
        B = len(seq_lens)
        q_start = None
        for attr in ("query_start_loc_cpu", "query_start_loc"):
            q_start = _host_list(getattr(common_attn_metadata, attr, None))
            if q_start is not None:
                break
        if q_start is None or len(q_start) != B + 1:
            return False
        return all(int(q_start[i + 1]) - int(q_start[i]) == int(seq_lens[i])
                   for i in range(B))
    except Exception:
        return False


__all__ = ["OMMXStepManager", "OMMXBatchedStepManager", "OMMXSlotAllocationError",
           "assign_slots", "detect_step_route", "detect_full_prefill",
           "STEP_ROUTE_UNCLASSIFIED"]
