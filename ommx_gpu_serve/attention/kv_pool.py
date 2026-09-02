# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared multi-sequence KV plane pool — ONE plane set per layer for ALL requests.

``CanonicalKVStore`` (kv_store.py) is PER-LAYER SINGLE-SEQUENCE: it owns its own
plane tensors and identity ``req_to_token [1,P]`` / ``req_to_group [1,G]`` tables.
A batched decode needs the OPPOSITE shape: ONE pool of the quantized planes
(``k_base`` / ``v_main`` / ``k_scale`` / ``k_zp`` / ``v_scale`` / ``v_zp`` + the K
outlier sidecar) shared by every active request, with per-request
``req_to_token [B,*]`` / ``req_to_group [B,*]`` tables indexing into that pool. This
is the input shape ``ommx_gpu_serve::paged_decode`` already accepts (its ABI is
batch + paged from the start) — so the pool is purely a HOST-side packing/layout
seam, no kernel change.

Layout (paged, but contiguous-per-request for a simple bring-up):

  * pool sized at capacity = ``num_seqs * groups_per_seq`` scale-groups;
  * request slot ``b`` owns a fixed contiguous block of ``groups_per_seq`` group
    slots ``[b*Gps, (b+1)*Gps)`` and the matching ``Pps = Gps*pages_per_group`` page
    slots ``[b*Pps, (b+1)*Pps)``;
  * ``req_to_group[b] = arange(b*Gps, b*Gps + n_groups_b)`` (rest clamped to the
    request's base slot so OOB rows the kernel masks via ``b_seq_len`` stay in-range);
  * ``req_to_token[b]`` likewise over the page slots.

The KV-bf16-in-prefill / OMMX-only-in-decode contract is unchanged: the per-request
bf16 residual history is the SHADOW (prefill stays bf16); ``regroup`` packs completed
32-token groups into the pool at the prefill->decode handoff (and on each decode
step's host seam). ``regroup`` reuses ``ommx_pack_kv_canonical_block`` and mirrors
``CanonicalKVStore._regroup_pack_range`` writing into the request's pool slots.

CPU-verifiable: a request's pool planes (gathered via its ``req_to_token`` /
``req_to_group``) dequant BIT-EXACT (``dequant_kv_canonical``, max_diff == 0) to a
standalone single-seq ``CanonicalKVStore`` packing the same sequence. That gate now
SHIPS as ``ommx_gpu_serve/tests/test_kv_pool_parity.py``, and it asserts something
strictly stronger than this paragraph promised: the packed PLANES THEMSELVES are
byte-identical (``torch.equal`` on the raw uint8/int8/bf16 tensors, so a divergence
cannot hide behind a dequant that happens not to see it), across a ragged multi-slot
batch with ``OMMX_KV_RING`` both OFF and ON. It runs on CPU with no vLLM, no triton and
no GPU; the CUDA repeat of the same parity is ``gpu``-marked and SKIPS (does not pass)
without a device. The complementary end-to-end check is
``ommx_gpu_serve/hf_eager/_ommx_hf_batch_test.py``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

# ── OMMX_RECIPE resolution (see recipes.resolve_env for the model + why) ─────────
#
# Resolution model (a): every reader of a recipe-controlled env var expands
# ``OMMX_RECIPE`` itself, via this idempotent call, BEFORE it reads ``os.environ``.
# Before this, ``OMMX_RECIPE`` was expanded only inside
# ``integration/vllm/config.resolve_serving_config``; the raw ``os.environ.get``
# reads below therefore ignored a named preset in any process where that call had
# not already happened — silently, with a plausible shipped-recipe answer. The knobs
# read raw in this file are recipe-controlled: both registered KV presets set
# ``OMMX_KV_RING=1``, ``OMMX_KV_GPU_PACK=1`` and ``OMMX_KV_OUTLIER_MAP=1``, none of
# which is the bare default (ring in particular defaults OFF = the full bf16 shadow).
# The call is a no-op single dict lookup when OMMX_RECIPE is unset, never overwrites
# an explicitly-set var, and raises (listing the known names) on an unknown one.
#
# The import is LAZY (inside the wrapper, not at module scope) for one measured reason:
# ``ommx_gpu_serve/__init__`` imports this module, so a module-scope
# ``from ..recipes import ...`` puts ``ommx_gpu_serve.recipes`` in sys.modules DURING
# the package __init__, and ``python3 -m ommx_gpu_serve.recipes`` — the documented CLI,
# and what ``run.sh --recipe`` shells out to — then emits
# "RuntimeWarning: 'ommx_gpu_serve.recipes' found in sys.modules after import of
# package 'ommx_gpu_serve' ... this may result in unpredictable behaviour" on every
# invocation. Deferring the import to the first CALL keeps that command clean.


def _resolve_recipe_env():
    from ..recipes import resolve_env
    return resolve_env()


from .kv_window import CANONICAL_GROUP_TOKENS, WindowSpec
from .pack import OUTLIER_REPRS as _OUTLIER_REPRS
from .pack import ommx_pack_kv_canonical_block

_OVAL_BYTES = {4: lambda k: (k + 1) // 2}


class MultiSeqKVPool:
    """Per-layer shared plane pool + per-request tables for batched paged decode."""

    def __init__(
        self,
        head_dim: int,
        n_kv_heads: int,
        num_seqs: int,
        max_seq_len: int,
        *,
        k_format: str = "i2f4",
        outliers_per_vector: int = 3,
        outlier_select: str = "signed",
        outlier_repr: str = "relidx7",
        use_pow2: bool = False,
        kv_outlier_map: Optional[bool] = None,
        kv_int8_scale: Optional[bool] = None,
        window: Optional[WindowSpec] = None,
        group_channels: int = 32,
        scale_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ) -> None:
        if head_dim % 32 != 0:
            raise ValueError(f"head_dim must be a multiple of 32; got {head_dim}")
        kfmt = str(k_format).lower()
        if kfmt not in ("i2f4", "itf4"):
            raise ValueError(f"k_format must be i2f4|itf4; got {k_format!r}")
        self.D = int(head_dim)
        self.H = int(n_kv_heads)
        self.num_seqs = int(num_seqs)
        self.max_seq_len = int(max_seq_len)
        self.k_format = kfmt
        self.k = int(outliers_per_vector)
        self.outlier_select = str(outlier_select).lower()
        self.outlier_repr = str(outlier_repr).lower()
        if self.outlier_repr not in _OUTLIER_REPRS:
            raise ValueError(
                f"outlier_repr must be one of {_OUTLIER_REPRS}; got {outlier_repr!r}")
        self.use_pow2 = bool(use_pow2)
        import os as _os
        # OMMX_RECIPE -> os.environ before the first raw read in this constructor.
        _resolve_recipe_env()
        if kv_outlier_map is None:
            _raw = _os.environ.get("OMMX_KV_OUTLIER_MAP")
            kv_outlier_map = (_raw not in {"0", "false", "off", "no"}) if _raw else True
        if kv_int8_scale is None:
            _raw = _os.environ.get("OMMX_KV_INT8_SCALE")
            kv_int8_scale = ((_raw not in {"0", "false", "off", "no"}) if _raw
                             else bool(use_pow2))
        self.kv_outlier_map = bool(kv_outlier_map)
        self.kv_int8_scale = bool(kv_int8_scale) and bool(use_pow2)
        self.scale_dtype = scale_dtype
        self.device = torch.device(device)
        self.window = window or WindowSpec()
        self.gt = int(self.window.group_tokens) or CANONICAL_GROUP_TOKENS
        self.ps = int(self.window.page_size)
        if self.gt % self.ps != 0:
            raise ValueError(
                f"group_tokens ({self.gt}) must be a multiple of page_size ({self.ps})")
        self.pages_per_group = self.gt // self.ps
        self.gc = int(group_channels)
        if self.gc not in (16, 32, 64, 128):
            raise ValueError(f"group_channels must be in {{16,32,64,128}}; got {self.gc}")
        if self.D % self.gc != 0:
            raise ValueError(f"head_dim ({self.D}) must be a multiple of group_channels ({self.gc})")
        if self.gt not in (16, 32, 64, 128):
            raise ValueError(f"group_tokens must be in {{16,32,64,128}}; got {self.gt}")
        self.k_base_bits = 2
        self.oval_bits = 4
        self.k_base_w = self.D // 4
        self.NGV = self.D // self.gc

        # per-request capacity: one extra group of slack (a just-completed group never
        # OOBs), matching CanonicalKVStore.G_cap. Pool = num_seqs of that block.
        self.G_per_seq = max(1, (self.max_seq_len + self.gt - 1) // self.gt + 1)
        self.P_per_seq = self.G_per_seq * self.pages_per_group
        self.G_cap = self.num_seqs * self.G_per_seq
        self.P_cap = self.num_seqs * self.P_per_seq

        dev, sdt = self.device, self.scale_dtype
        scl_dt = torch.int8 if self.kv_int8_scale else sdt
        self.k_base = torch.zeros(self.P_cap, self.ps, self.H, self.k_base_w,
                                  dtype=torch.uint8, device=dev)
        self.v_main = torch.zeros(self.P_cap, self.ps, self.H, self.D // 4,
                                  dtype=torch.uint8, device=dev)
        self.k_scale = torch.zeros(self.G_cap, self.H, self.D, dtype=scl_dt, device=dev)
        self.k_zp = torch.zeros(self.G_cap, self.H, self.D, dtype=sdt, device=dev)
        self.v_scale = torch.zeros(self.P_cap, self.ps, self.H, self.NGV,
                                   dtype=scl_dt, device=dev)
        self.v_zp = torch.zeros(self.P_cap, self.ps, self.H, self.NGV,
                                dtype=sdt, device=dev)
        if self.k > 0 and self.kv_outlier_map:
            self.k_fp4_mapscale = torch.zeros(self.G_cap, self.H, self.D,
                                              dtype=sdt, device=dev)
            self.k_fp4_mapcenter = torch.zeros(self.G_cap, self.H, self.D,
                                               dtype=sdt, device=dev)
        else:
            self.k_fp4_mapscale = None
            self.k_fp4_mapcenter = None
        # OUTLIER-POSITION plane: exactly ONE of the three membership-equivalent index
        # encodings is allocated (the other two stay None) — mirror of
        # CanonicalKVStore.__init__:
        #   relidx7 (DEFAULT)  k_oidx  ceil(7k/8) B per (group, head, channel) frame
        #   combinadic         k_crank ceil(log2 C(gt,k)/8) B  — the storage floor
        #   bitmap             k_obmp  ceil(gt/8) B  — the paper's flat N-bit-per-group
        #                                              mask; FLAT in k (1.0 bit/element)
        self.k_obmp = None
        if self.k > 0 and self.outlier_repr == "relidx7":
            idx_fb = (7 * self.k + 7) // 8
            self.k_oidx = torch.zeros(self.G_cap, self.H, self.D, idx_fb,
                                      dtype=torch.uint8, device=dev)
            self.k_crank = None
        elif self.k > 0 and self.outlier_repr == "bitmap":
            from .codec import bitmap_index_bytes
            fb = bitmap_index_bytes(self.gt)
            self.k_obmp = torch.zeros(self.G_cap, self.H, self.D, fb,
                                      dtype=torch.uint8, device=dev)
            self.k_oidx = None
            self.k_crank = None
        elif self.k > 0:  # combinadic
            from .codec import combinadic_index_bytes
            fb = combinadic_index_bytes(self.k, self.gt)
            self.k_crank = torch.zeros(self.G_cap, self.H, self.D, fb,
                                       dtype=torch.uint8, device=dev)
            self.k_oidx = None
        else:
            self.k_oidx = None
            self.k_crank = None
        if self.k > 0:
            oval_fb = _OVAL_BYTES[self.oval_bits](self.k)
            self.k_oval = torch.zeros(self.G_cap, self.H, self.D, oval_fb,
                                      dtype=torch.uint8, device=dev)
        else:
            self.k_oval = None

        # per-request bf16 residual history. TWO sizings, gated by OMMX_KV_RING — the
        # EXACT mirror of CanonicalKVStore (kv_store.py:180-214) extended to num_seqs:
        #   * FULL-HIST (default, ring OFF): keep the whole bf16 sequence per request
        #     [num_seqs, max_seq_len, H, D] — the CPU-verifiable reference (test_kv_pool*),
        #     and the only sizing that lets the bulk-pack/chunked-prefill callers read
        #     arbitrary prefix ranges.
        #   * RING (OMMX_KV_RING=1): only the worst-case LIVE set per request (sink ∪
        #     recent ∪ the group being packed ∪ the new accumulating group) is ever read,
        #     so a small ring of `ring_cap` rows PER REQUEST suffices — the [num_seqs,
        #     max_seq_len, ...] bf16 SHADOW (the 64GB B>1 long-ctx OOM) collapses to
        #     num_seqs * ~104 rows. Abs token pos -> ring slot via `_ring_slot_*` (sink
        #     pinned [0:sink), recent positions sliding modulo Rrec). Quantized planes
        #     (the ≤3-bit bulk) are UNCHANGED — only k_hist/v_hist/tail sizing changes.
        import os as _os2
        # OMMX_KV_RING is set by BOTH registered presets and defaults OFF, so this is
        # the read where an unresolved OMMX_RECIPE changed what actually got allocated
        # (full [num_seqs, max_seq_len, H, D] bf16 shadow instead of the ring).
        _resolve_recipe_env()
        _ring_raw = _os2.environ.get("OMMX_KV_RING")
        self.kv_ring = bool(_ring_raw) and _ring_raw not in {"0", "false", "off", "no"}
        if self.kv_ring:
            # ring geometry: identical formula to CanonicalKVStore (kv_store.py:200-202).
            # Rrec must cover, at ANY decode step `seq`, the abs-pos window the ring is
            # read/written over: [boundary - gt, seq). Width < recent + 2*gt, so distinct
            # live positions never alias modulo Rrec.
            self._ring_sink = int(self.window.sink_tokens)
            self._ring_rec = int(self.window.recent_window) + 2 * int(self.gt)
            self.ring_cap = self._ring_sink + self._ring_rec
        else:
            self._ring_sink = 0
            self._ring_rec = 0
            self.ring_cap = self.max_seq_len
        # `S_ring` = the per-request bf16 row stride used by EVERY reader/writer's flat
        # gather (slot*S_ring + ring_slot(token)). ring_cap when ring is ON, max_seq_len OFF.
        self.S_ring = int(self.ring_cap)
        self.k_hist = torch.zeros(self.num_seqs, self.ring_cap, self.H, self.D,
                                  dtype=torch.bfloat16, device=dev)
        self.v_hist = torch.zeros(self.num_seqs, self.ring_cap, self.H, self.D,
                                  dtype=torch.bfloat16, device=dev)
        self.seq_len = [0] * self.num_seqs        # current seq length per request (ABS)
        self.packed_groups = [0] * self.num_seqs  # completed groups packed per request

        # fixed-address batched tables: identity arange within each request's slot
        # block. Rows beyond a request's live region are clamped to its base slot,
        # so the kernel (which masks by b_seq_len) never reads another request's KV.
        rg = torch.arange(self.G_cap, dtype=torch.int32, device=dev)
        self._req_to_group = rg.reshape(self.num_seqs, self.G_per_seq)
        rt = torch.arange(self.P_cap, dtype=torch.int32, device=dev)
        self._req_to_token = rt.reshape(self.num_seqs, self.P_per_seq)
        self.max_tail_len = int(self.window.max_tail_len(self.max_seq_len))

    # ── ring address mapping (OMMX_KV_RING only; identity when OFF) ──────────────
    # EXACT mirror of CanonicalKVStore._ring_slot_host / _ring_slot_dev (kv_store.py
    # :235-261), reused per request: the abs token pos -> ring row map is per-request
    # IDENTICAL (every request owns its own [ring_cap, H, D] slab), so the same map
    # composes with the per-request base offset (slot*S_ring) in the flat gathers.

    def _ring_slot_host(self, pos: int) -> int:
        """Abs token pos -> ring row (python int). Identity when ring is OFF."""
        if not self.kv_ring:
            return int(pos)
        p = int(pos)
        if p < self._ring_sink:
            return p
        return self._ring_sink + (p - self._ring_sink) % self._ring_rec

    def _ring_slot_dev(self, pos: torch.Tensor) -> torch.Tensor:
        """Abs token pos -> ring row (device int tensor, capture-safe). Identity OFF.

        Pure tensor arithmetic (``torch.where`` + ``%``) — no ``.item()``, no host
        sync, fixed shape — so it can live inside a captured decode region. Result
        matches ``pos``'s shape/dtype/device. Mirror of CanonicalKVStore._ring_slot_dev.
        """
        if not self.kv_ring:
            return pos
        sink = self._ring_sink
        rec = self._ring_rec
        rel = (pos - sink) % rec + sink
        return torch.where(pos < sink, pos, rel)

    # ── write path ────────────────────────────────────────────────────────────

    def append_block(self, req: int, K: torch.Tensor, V: torch.Tensor) -> int:
        """Stash request ``req``'s prefill block bf16 K/V ``[T, H, D]`` then regroup.

        Returns the number of groups packed into the pool by this call.
        """
        r = int(req)
        T = int(K.shape[0])
        s = self.seq_len[r]
        if not self.kv_ring:
            if s + T > self.max_seq_len:
                raise RuntimeError(
                    f"MultiSeqKVPool req {r} capacity {self.max_seq_len} exceeded")
            self.k_hist[r, s:s + T] = K.to(torch.bfloat16)
            self.v_hist[r, s:s + T] = V.to(torch.bfloat16)
            self.seq_len[r] = s + T
            return self.regroup(r)
        # RING (no-shadow): mirror CanonicalKVStore.append_block (kv_store.py:352-402).
        # The ring holds only the last <Rrec rows, so a prefill block of T>>Rrec tokens
        # CANNOT be written into the ring then packed from it (the bulk source rows alias
        # before the packer reads them). BUT the input K/V is fully resident for this call
        # -> pack the WHOLE completed prefix [sink, boundary) DIRECTLY FROM THE INPUT in
        # ONE multi-group ommx_pack_kv_canonical_block, then copy only the persistent live
        # tail rows (sink ∪ [boundary-gt, T) — <Rrec rows, distinct ring slots) into the
        # ring. BIT-EXACT to the no-ring path (same packer, same gt decomposition).
        if s != 0:
            raise RuntimeError(
                f"ring append_block expects a fresh request slot (seq_len==0); chunked "
                f"prefill under OMMX_KV_RING is unsupported (the ring cannot retain >Rrec "
                f"unpacked rows). Reset pool slot {r} before each prefill block.")
        Kb = K.to(torch.bfloat16)
        Vb = V.to(torch.bfloat16)
        self.seq_len[r] = T
        target = self.window.num_groups(T)            # completed groups for this prefix
        n_packed = 0
        if target > 0:
            sink = self.window.sink_tokens
            t0 = sink
            t1 = sink + target * self.gt              # == boundary(T)
            # pack the whole completed prefix straight from the INPUT (not the ring),
            # writing into req r's pool slots — ONE call, no per-group loop, no aliasing.
            self._pack_block_into_pool(r, Kb[t0:t1], Vb[t0:t1], 0, target)
            self.packed_groups[r] = target
            n_packed = target
        # persist ONLY the live tail rows into the ring: sink [0,sink) + [boundary-gt, T).
        # These are the only abs positions any later decode step (or its boundary-pack)
        # reads; they number < Rrec so their ring slots are distinct. One device scatter.
        boundary = self.window.boundary(T)
        sink_tokens = self.window.sink_tokens
        lo = max(sink_tokens, boundary - self.gt)
        # WARMUP guard (mirror kv_store.py:393-398): a tiny T (T<sink) -> arange(lo, T)
        # with lo>=sink>T raises; only add the recent span when it is non-empty.
        parts = [torch.arange(0, min(sink_tokens, T), device=Kb.device)]
        if lo < T:
            parts.append(torch.arange(lo, T, device=Kb.device))
        live = torch.cat(parts).to(torch.long)
        slots = self._ring_slot_dev(live)
        self.k_hist[r].index_copy_(0, slots, Kb.index_select(0, live))
        self.v_hist[r].index_copy_(0, slots, Vb.index_select(0, live))
        return n_packed

    def append(self, req: int, k: torch.Tensor, v: torch.Tensor) -> int:
        """Stash one decode step's bf16 K/V row ``[H, D]`` for request ``req``."""
        r = int(req)
        s = self.seq_len[r]
        if not self.kv_ring and s >= self.max_seq_len:
            raise RuntimeError(
                f"MultiSeqKVPool req {r} capacity {self.max_seq_len} exceeded")
        slot = self._ring_slot_host(s)   # identity when ring OFF
        self.k_hist[r, slot] = k.view(self.H, self.D).to(torch.bfloat16)
        self.v_hist[r, slot] = v.view(self.H, self.D).to(torch.bfloat16)
        self.seq_len[r] = s + 1
        return self.regroup(r)

    def append_decode_batched(self, slots: Sequence[int], K: torch.Tensor,
                              V: torch.Tensor) -> int:
        """DEVICE-SIDE batched decode append: B new rows in ONE scatter + GPU regroup.

        ``K`` / ``V`` are ``[B, H, D]`` (one new token per request, already on GPU).
        Replaces the per-request python ``append`` loop (B tiny scatters + B host
        regroup probes + a D2H/CPU/H2D pack on each completing group) with:

          1. ONE ``index_put_`` writing all B rows into ``k_hist``/``v_hist`` at each
             request's current ``seq`` slot — no per-row python scatter;
          2. a GPU regroup (``_regroup_pack_batched_device``) over exactly the requests
             whose seq crossed a 32-token group boundary this step, batched into ONE
             ``ommx_pack_kv_canonical_block`` call (relidx7/i2f4 runs fully on GPU now)
             — no host sync, capture-friendlier.

        BIT-EXACT to the per-request host ``append`` + CPU ``regroup`` (integer
        bit-twiddling is device-invariant; the GPU pack matches the CPU pack for-bit).
        Returns the total number of groups packed across the batch this step.
        """
        slots = [int(s) for s in slots]
        B = len(slots)
        if B == 0:
            return 0
        dev = self.device
        # row index into k_hist[r, ring_slot(seq[r])] for each request (write slot).
        # cols = the ABS write position remapped to its ring slot (identity when ring OFF).
        rows = torch.tensor(slots, dtype=torch.long, device=dev)
        cols = torch.tensor([self._ring_slot_host(self.seq_len[r]) for r in slots],
                            dtype=torch.long, device=dev)
        Kb = K.view(B, self.H, self.D).to(torch.bfloat16)
        Vb = V.view(B, self.H, self.D).to(torch.bfloat16)
        # single coalesced scatter (advanced indexing) — one kernel, not B.
        self.k_hist[rows, cols] = Kb
        self.v_hist[rows, cols] = Vb
        for r in slots:
            self.seq_len[r] += 1
        # collect the requests whose group boundary advanced this step.
        pend = []   # (req, g0, g1)
        total = 0
        for r in slots:
            target = self.window.num_groups(self.seq_len[r])
            if target > self.packed_groups[r]:
                pend.append((r, self.packed_groups[r], target))
                total += target - self.packed_groups[r]
        if pend:
            if self.outlier_repr == "combinadic":
                # combinadic pack has python per-(channel,group) loops (CPU-only); keep
                # the host regroup for that storage-floor mode. relidx7 (default) packs
                # on GPU. (combinadic decode is derived to relidx7 at load anyway.)
                for (r, _g0, _g1) in pend:
                    self.regroup(r)
            else:
                self._regroup_pack_batched_device(pend)
                for r, _g0, g1 in pend:
                    self.packed_groups[r] = g1
        return total

    def append_decode_device(self, K: torch.Tensor, V: torch.Tensor, B: int) -> int:
        """DEVICE decode write: scatter B new rows at the persistent device columns,
        advance device ``seq_dev``, then run the host regroup seam for completing groups.

        The scatter reuses ``append_decode_captured`` with the persistent ``rows`` /
        ``cols`` device buffers (``cols`` = ABS write column = current seq, BEFORE the
        advance) — so the per-step write touches NO per-step ``torch.tensor`` / host int
        (vs ``append_decode_batched`` which builds ``rows``/``cols`` via two
        ``torch.tensor`` allocs + a python ``_ring_slot_host`` list-comp each step).

        After the scatter, ``seq_dev`` is advanced on device (one kernel) and the host
        ``self.seq_len`` / ``packed_groups`` are advanced in lockstep + the GPU regroup
        packs any group that completed this step (the pack itself is unavoidable but
        fires at most once / 32 steps / request, and is O(layers) — NOT the O(B*T)
        per-step python the read loop incurred). ``K``/``V`` are ``[B,H,D]``. Returns
        the number of groups packed this step.
        """
        d = self._dbuf
        # cols currently holds the ABS write column = each request's CURRENT seq (the
        # decode-token position before advance); append_decode_captured remaps to ring.
        # Set cols = seq_dev (pre-advance) so the new row lands at its abs position.
        d["cols"][:B].copy_(d["seq_dev"][:B])
        self.append_decode_captured(d["rows"][:B], d["cols"][:B], K, V)
        # advance device seq (one add_) for the NEXT step's tail/boundary derivation.
        self.advance_seq_device(B)
        # advance host seq + regroup completing groups (host seam; pack is GPU). This is
        # the only remaining host int work — a B-length loop with NO torch.tensor and a
        # pack that fires only on a group boundary (mirror append_decode_batched tail).
        # NOTE: rows == range(B) for the uniform batch (the modeling seeds it so), so the
        # pool slot == batch row; iterate the contiguous batch-row order directly (no
        # per-step .item() sync). VERIFY: holds only for the uniform contiguous batch the
        # HF modeling drives (slots=range(B)); ragged/remapped slots would need rows read.
        pend = []
        total = 0
        for r in range(B):
            self.seq_len[r] += 1
            target = self.window.num_groups(self.seq_len[r])
            if target > self.packed_groups[r]:
                pend.append((r, self.packed_groups[r], target))
                total += target - self.packed_groups[r]
        if pend:
            if self.outlier_repr == "combinadic":
                for (r, _g0, _g1) in pend:
                    self.regroup(r)
            else:
                self._regroup_pack_batched_device(pend)
                for r, _g0, g1 in pend:
                    self.packed_groups[r] = g1
        return total

    def append_decode_captured(self, rows: torch.Tensor, cols: torch.Tensor,
                               K: torch.Tensor, V: torch.Tensor) -> None:
        """CAPTURE-SAFE batched decode write: scatter B new rows by DEVICE indices.

        ``rows`` / ``cols`` are FIXED-ADDRESS device int64 buffers ``[B]`` (the host
        build-seam fills them with each request's pool slot + current write column
        BEFORE this captured forward). ``K`` / ``V`` are ``[B, H, D]``. Unlike
        ``append_decode_batched`` (which builds ``cols`` from the host ``seq_len``
        list via ``torch.tensor`` — a host->device alloc that breaks capture and does
        NOT advance per-step on graph replay), this uses ONLY the pre-built device
        index buffers, so the SAME ``index_put_`` replays with the host-refreshed
        column each step. Does NOT touch ``self.seq_len`` / ``packed_groups`` (the
        host build-seam owns those, outside capture, like ``CanonicalKVStore.advance_to``).

        The advanced-index assignment ``k_hist[rows, cols] = K`` lowers to a single
        ``index_put_`` (static shape, device indices) — capturable.
        """
        B = int(rows.shape[0])
        Kb = K.view(B, self.H, self.D).to(torch.bfloat16)
        Vb = V.view(B, self.H, self.D).to(torch.bfloat16)
        # ``cols`` is the ABS write column (the host build-seam / metadata fills it with
        # seq-1); remap to its ring row here by capture-safe arithmetic (identity when
        # ring OFF) so the pool is the SINGLE ring-remap point — the graph caller
        # (metadata.py) needs no change. Mirror of CanonicalKVStore.write_token.
        cols = self._ring_slot_dev(cols)
        self.k_hist[rows, cols] = Kb
        self.v_hist[rows, cols] = Vb

    def _regroup_pack_batched_device(self, pend: Sequence) -> None:
        """GPU pack of all completing groups this step in ONE packer call (no sync).

        ``pend`` is a list of ``(req, g0, g1)``. Each request contributes its
        ``[g0,g1)`` token block (a multiple of ``gt``) from its GPU ``k_hist``; the
        blocks are CONCATENATED along the token axis into one ``[sum_T, H, D]`` tensor
        and packed by a single ``ommx_pack_kv_canonical_block`` on the GPU device, then
        the per-request group/page slices are scattered back into the pool planes. A
        decode step completes at most one group per request, but several requests can
        complete on the SAME step — batching them amortizes the one packer launch.

        Per-channel K affine stats are computed PER 32-token group inside the packer
        (group axis = the leading token-group), so concatenating independent requests'
        groups does NOT cross-contaminate stats — each ``gt``-token group is packed
        independently. BIT-EXACT to packing each request alone.
        """
        gt = self.gt
        blocks_k = []
        blocks_v = []
        meta = []   # (req, g0, g1, gbase_in_concat)
        gcur = 0
        sink = self.window.sink_tokens
        for (r, g0, g1) in pend:
            t0 = sink + g0 * gt
            t1 = sink + g1 * gt
            if self.kv_ring:
                # gather abs rows [t0,t1) through their ring slots — VECTORIZED on device
                # (mirror kv_store.py:451-457). (t1-t0)==gt < Rrec -> slots distinct. A
                # decode step advances one group/req, so this read-span is exactly gt.
                idx = self._ring_slot_dev(
                    torch.arange(t0, t1, device=self.device)).to(torch.long)
                blocks_k.append(self.k_hist[r].index_select(0, idx))
                blocks_v.append(self.v_hist[r].index_select(0, idx))
            else:
                blocks_k.append(self.k_hist[r, t0:t1])
                blocks_v.append(self.v_hist[r, t0:t1])
            meta.append((r, g0, g1, gcur))
            gcur += (g1 - g0)
        K_blk = torch.cat(blocks_k, dim=0).to(torch.float32)   # GPU, [sum_T,H,D]
        V_blk = torch.cat(blocks_v, dim=0).to(torch.float32)
        planes = ommx_pack_kv_canonical_block(
            K_blk, V_blk,
            outliers_per_vector=self.k,
            scale_dtype=self.scale_dtype,
            page_size=self.ps,
            device=self.device,           # <-- GPU pack: no D2H/H2D round trip
            outlier_repr=self.outlier_repr,
            outlier_select=self.outlier_select,
            k_format=self.k_format,
            use_pow2=self.use_pow2,
            kv_outlier_map=self.kv_outlier_map,
            kv_int8_scale=self.kv_int8_scale,
            group_tokens=self.gt,
            group_channels=self.gc,
        )
        ppg = self.pages_per_group
        for (r, g0, g1, gbase) in meta:
            ng = g1 - g0
            gp = gbase
            pp = gbase * ppg
            gbase_pool = r * self.G_per_seq
            pbase_pool = r * self.P_per_seq
            gp0, gp1 = gbase_pool + g0, gbase_pool + g1
            pp0 = pbase_pool + g0 * ppg
            pp1 = pbase_pool + g1 * ppg
            psl_p = slice(pp, pp + ng * ppg)
            psl_g = slice(gp, gp + ng)
            self.k_base[pp0:pp1] = planes["k_base"][psl_p]
            self.v_main[pp0:pp1] = planes["v_main"][psl_p]
            self.v_scale[pp0:pp1] = planes["v_scale"][psl_p]
            self.v_zp[pp0:pp1] = planes["v_zp"][psl_p]
            self.k_scale[gp0:gp1] = planes["k_scale"][psl_g]
            self.k_zp[gp0:gp1] = planes["k_zp"][psl_g]
            if self.k > 0:
                self.k_oval[gp0:gp1] = planes["k_oval"][psl_g]
                if self.outlier_repr == "relidx7":
                    self.k_oidx[gp0:gp1] = planes["k_oidx"][psl_g]
                elif self.outlier_repr == "bitmap":
                    self.k_obmp[gp0:gp1] = planes["k_obmp"][psl_g]
                else:
                    self.k_crank[gp0:gp1] = planes["k_crank"][psl_g]
                if self.kv_outlier_map:
                    self.k_fp4_mapscale[gp0:gp1] = planes["k_fp4_mapscale"][psl_g]
                    self.k_fp4_mapcenter[gp0:gp1] = planes["k_fp4_mapcenter"][psl_g]

    def regroup(self, req: int) -> int:
        """HOST seam: pack request ``req``'s newly completed groups into the pool.

        Returns the number of groups packed. Mirrors
        ``CanonicalKVStore.maybe_regroup`` but writes into request ``req``'s assigned
        pool slot block. Runs OUTSIDE CUDA-graph capture (CPU pack + the bf16->cpu
        copy it needs).
        """
        r = int(req)
        target = self.window.num_groups(self.seq_len[r])
        if target <= self.packed_groups[r]:
            return 0
        n = target - self.packed_groups[r]
        self._regroup_pack_range(r, self.packed_groups[r], target)
        self.packed_groups[r] = target
        return n

    def _regroup_pack_range(self, req: int, g0: int, g1: int) -> None:
        """Pack request ``req``'s groups ``[g0, g1)`` into its pool slots in ONE call.

        Mirror of ``CanonicalKVStore._regroup_pack_range``: pack the ``(g1-g0)``-group
        block on CPU via ``ommx_pack_kv_canonical_block`` and copy the planes into the
        request's pool slots (group axis offset by ``req*G_per_seq``, page axis by
        ``req*P_per_seq``). One multi-group call (prefill packs hundreds at once).
        """
        r = int(req)
        sink = self.window.sink_tokens
        t0 = sink + g0 * self.gt
        t1 = sink + g1 * self.gt
        if self.kv_ring:
            # gather abs rows [t0,t1) through req r's ring slots (mirror kv_store.py:451).
            idx = self._ring_slot_dev(
                torch.arange(t0, t1, device=self.device)).to(torch.long)
            K_blk = self.k_hist[r].index_select(0, idx)         # bf16 [(g1-g0)*gt,H,D]
            V_blk = self.v_hist[r].index_select(0, idx)
        else:
            K_blk = self.k_hist[r, t0:t1]
            V_blk = self.v_hist[r, t0:t1]
        self._pack_block_into_pool(r, K_blk, V_blk, g0, g1)

    def _pack_block_into_pool(self, req: int, K_blk: torch.Tensor,
                              V_blk: torch.Tensor, g0: int, g1: int) -> None:
        """Pack a bf16 source block of EXACTLY ``(g1-g0)*gt`` tokens into req ``req``'s
        pool group/page slots ``[g0, g1)`` in ONE ``ommx_pack_kv_canonical_block`` call.

        Shared write-pack body for BOTH the ring-prefill path (source = a slice of the
        caller's input that never entered the ring) and the host regroup (source = a
        contiguous shadow slice OR a ring gather). The packer derives per-32-token-group
        affine stats independently, so the source's origin is irrelevant to bit-exactness.
        Mirror of CanonicalKVStore._pack_block_into_planes (kv_store.py:463) for one req.
        """
        import os as _os
        # GPU-PACK default ON for relidx7/i2f4 (bit-exact, device-invariant) — no D2H/H2D
        # round-trip; the flat BITMAP is device-clean too (_pack_bitmap_frames is pure
        # vectorized bit-twiddling); combinadic always falls back to CPU (rank_bytes.cpu).
        _resolve_recipe_env()          # OMMX_RECIPE -> os.environ, before the read
        gpu_pack = (_os.environ.get("OMMX_KV_GPU_PACK", "1") not in {"0", "false", "off", "no"}
                    and self.outlier_repr != "combinadic")
        pdev = self.device if gpu_pack else "cpu"
        planes = ommx_pack_kv_canonical_block(
            K_blk.to(pdev, torch.float32), V_blk.to(pdev, torch.float32),
            outliers_per_vector=self.k,
            scale_dtype=self.scale_dtype,
            page_size=self.ps,
            device=pdev,
            outlier_repr=self.outlier_repr,
            outlier_select=self.outlier_select,
            k_format=self.k_format,
            use_pow2=self.use_pow2,
            kv_outlier_map=self.kv_outlier_map,
            kv_int8_scale=self.kv_int8_scale,
            group_tokens=self.gt,
            group_channels=self.gc,
        )
        r = int(req)
        dev = self.device
        gbase = r * self.G_per_seq
        pbase = r * self.P_per_seq
        gp0, gp1 = gbase + g0, gbase + g1
        pp0 = pbase + g0 * self.pages_per_group
        pp1 = pbase + g1 * self.pages_per_group
        self.k_base[pp0:pp1] = planes["k_base"].to(dev)
        self.v_main[pp0:pp1] = planes["v_main"].to(dev)
        self.v_scale[pp0:pp1] = planes["v_scale"].to(dev)
        self.v_zp[pp0:pp1] = planes["v_zp"].to(dev)
        self.k_scale[gp0:gp1] = planes["k_scale"].to(dev)
        self.k_zp[gp0:gp1] = planes["k_zp"].to(dev)
        if self.k > 0:
            self.k_oval[gp0:gp1] = planes["k_oval"].to(dev)
            if self.outlier_repr == "relidx7":
                self.k_oidx[gp0:gp1] = planes["k_oidx"].to(dev)
            elif self.outlier_repr == "bitmap":
                self.k_obmp[gp0:gp1] = planes["k_obmp"].to(dev)
            else:
                self.k_crank[gp0:gp1] = planes["k_crank"].to(dev)
            if self.kv_outlier_map:
                self.k_fp4_mapscale[gp0:gp1] = planes["k_fp4_mapscale"].to(dev)
                self.k_fp4_mapcenter[gp0:gp1] = planes["k_fp4_mapcenter"].to(dev)

    # ── read path ───────────────────────────────────────────────────────────────

    def pool_planes(self) -> dict:
        """Shared-pool plane dict (whole pool tensors + batched tables) for the op.

        Static-shape (full pool); the live region per request is bounded at run time
        by the device ``b_seq_len`` the kernel reads, so this is CUDA-graph capturable.

        CACHED: every value is a fixed-address pool tensor / immutable scalar whose
        identity never changes after construction, so the ~25-key dict is built ONCE
        and reused — eliminating the per-step/per-layer dict rebuild the host decode
        loop incurred (B=8 x 32 layers x steps). The per-request ``req_to_token`` /
        ``req_to_group`` views (the only step-varying entries) are written OVER the
        cached dict by the callers (``decode_inputs_*``), so the base dict is stable.
        """
        cached = getattr(self, "_planes_cache", None)
        if cached is not None:
            return cached
        d = self._build_pool_planes()
        self._planes_cache = d
        return d

    def _build_pool_planes(self) -> dict:
        return {
            "k_base": self.k_base, "k_scale": self.k_scale, "k_zp": self.k_zp,
            "k_oidx": self.k_oidx, "k_oval": self.k_oval, "k_crank": self.k_crank,
            "k_obmp": self.k_obmp,
            "k_fp4_mapscale": self.k_fp4_mapscale,
            "k_fp4_mapcenter": self.k_fp4_mapcenter,
            "v_main": self.v_main, "v_scale": self.v_scale, "v_zp": self.v_zp,
            "req_to_token": self._req_to_token, "req_to_group": self._req_to_group,
            "page_size": self.ps, "outliers_per_vector": self.k,
            "outlier_repr": self.outlier_repr, "outlier_select": self.outlier_select,
            "k_format": self.k_format, "head_dim": self.D, "n_kv_heads": self.H,
            "group_tokens": self.gt, "group_channels": self.gc,
            "k_base_bits": self.k_base_bits, "oval_bits": self.oval_bits,
            "kv_outlier_map": bool(self.kv_outlier_map and self.k > 0),
            "kv_int8_scale": bool(self.kv_int8_scale),
        }

    def request_planes(self, req: int) -> dict:
        """CONTIGUOUS per-request plane dict (pool slots gathered via the tables).

        Builds the single-request plane dict ``dequant_kv_canonical`` expects: it
        gathers request ``req``'s live page/group slots out of the pool into dense
        ``[pages, ...]`` / ``[G, ...]`` tensors (``dequant_kv_canonical`` reads slot
        ``0..pages``/``0..G`` linearly — it does NOT honor the tables). This is the
        per-request view the bit-exact test dequants.
        """
        r = int(req)
        G = self.packed_groups[r]
        P = G * self.pages_per_group
        gidx = self._req_to_group[r, :G].to(torch.long)
        pidx = self._req_to_token[r, :P].to(torch.long)
        out = {
            "k_base": self.k_base.index_select(0, pidx),
            "k_scale": self.k_scale.index_select(0, gidx),
            "k_zp": self.k_zp.index_select(0, gidx),
            "k_oidx": self.k_oidx.index_select(0, gidx) if self.k_oidx is not None else None,
            "k_oval": self.k_oval.index_select(0, gidx) if self.k_oval is not None else None,
            "k_crank": self.k_crank.index_select(0, gidx) if self.k_crank is not None else None,
            "k_obmp": self.k_obmp.index_select(0, gidx) if self.k_obmp is not None else None,
            "k_fp4_mapscale": (self.k_fp4_mapscale.index_select(0, gidx)
                               if self.k_fp4_mapscale is not None else None),
            "k_fp4_mapcenter": (self.k_fp4_mapcenter.index_select(0, gidx)
                                if self.k_fp4_mapcenter is not None else None),
            "v_main": self.v_main.index_select(0, pidx),
            "v_scale": self.v_scale.index_select(0, pidx),
            "v_zp": self.v_zp.index_select(0, pidx),
            "head_dim": self.D, "n_kv_heads": self.H,
            "num_tokens": G * self.gt, "n_groups": G,
            "page_size": self.ps,
            "outliers_per_vector": self.k,
            "outlier_repr": self.outlier_repr, "outlier_select": self.outlier_select,
            "group_tokens": self.gt, "group_channels": self.gc,
            "k_format": self.k_format,
            "k_base_bits": self.k_base_bits, "oval_bits": self.oval_bits,
            "kv_outlier_map": bool(self.kv_outlier_map and self.k > 0),
            "kv_int8_scale": bool(self.kv_int8_scale),
        }
        return out

    def tail_bf16_batched(self, slots: Sequence[int]) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched bf16 residual tail: ``(k_tail [B,T,H,D], v_tail, b_tail_len [B])``.

        Each request's sink ∪ recent ∪ incomplete-tail rows gathered into a padded
        ``[B, max_tail_len, H, D]`` buffer (rows past ``b_tail_len`` are zero; the
        kernel masks them). Mirrors ``CanonicalKVStore.tail_bf16`` per request.
        """
        slots = [int(s) for s in slots]
        B = len(slots)
        T = self.max_tail_len
        kt = torch.zeros(B, T, self.H, self.D, dtype=torch.bfloat16, device=self.device)
        vt = torch.zeros(B, T, self.H, self.D, dtype=torch.bfloat16, device=self.device)
        b_tail = torch.zeros(B, dtype=torch.int32, device=self.device)
        for b, r in enumerate(slots):
            idx = self.window.tail_indices(self.seq_len[r])
            if self.kv_ring:
                idx = [self._ring_slot_host(p) for p in idx]
            n = min(len(idx), T)
            if n:
                ti = torch.tensor(idx[:n], dtype=torch.long, device=self.device)
                kt[b, :n] = self.k_hist[r].index_select(0, ti)
                vt[b, :n] = self.v_hist[r].index_select(0, ti)
            b_tail[b] = n
        return kt, vt, b_tail

    def _ensure_decode_buffers(self, B: int) -> None:
        """Allocate the persistent (fixed-address) batched-decode arg buffers ONCE.

        Reused every decode step (no per-step realloc): the bf16 tail buffers
        ``[cap,T,H,D]``, ``b_tail_len``/``b_seq_len`` ``[cap]``, the ``[cap,T]`` host +
        device tail-gather index buffers, and the row-select buffer. Sized to the pool
        capacity ``num_seqs`` so any active ``B<=num_seqs`` reuses the same tensors.
        """
        if getattr(self, "_dbuf", None) is not None and self._dbuf["cap"] >= B:
            return
        cap = max(B, self.num_seqs)
        T = self.max_tail_len
        dev = self.device
        self._dbuf = {
            "cap": cap,
            "kt": torch.zeros(cap, T, self.H, self.D, dtype=torch.bfloat16, device=dev),
            "vt": torch.zeros(cap, T, self.H, self.D, dtype=torch.bfloat16, device=dev),
            "b_tail": torch.zeros(cap, dtype=torch.int32, device=dev),
            "b_seq": torch.zeros(cap, dtype=torch.int32, device=dev),
            "sel": torch.zeros(cap, dtype=torch.long, device=dev),
            # host-pinned-ish staging for the per-step tail gather index (one H2D).
            "tidx_host": torch.zeros(cap, T, dtype=torch.long),
            "tidx_dev": torch.zeros(cap, T, dtype=torch.long, device=dev),
            "btail_host": torch.zeros(cap, dtype=torch.int32),
            "bseq_host": torch.zeros(cap, dtype=torch.int32),
            "sel_host": torch.zeros(cap, dtype=torch.long),
            # ── DEVICE decode path (decode_inputs_device): persistent per-step state
            # advanced ON DEVICE (no host loop, no per-step torch.tensor / .item()).
            #   * ``seq_dev`` [cap] int64  — ABS seq length per pool slot, += 1/step;
            #   * ``rows``    [cap] int64  — batch-row -> pool slot (== seq_dev's slot);
            #   * ``cols``    [cap] int64  — ABS write column (= seq_dev - 1) for the
            #                                captured write scatter;
            #   * ``col_arange`` [1, T] int64 — fixed [0..T) tail-column ramp (broadcast).
            # ``seq_dev`` is seeded ONCE from the host ``self.seq_len`` at decode entry
            # (one H2D) and then advanced purely on device; the host ``self.seq_len`` is
            # still advanced in lockstep so the host-side regroup/pack seam stays correct.
            "seq_dev": torch.zeros(cap, dtype=torch.long, device=dev),
            "rows": torch.zeros(cap, dtype=torch.long, device=dev),
            "cols": torch.zeros(cap, dtype=torch.long, device=dev),
            "col_arange": torch.arange(T, dtype=torch.long, device=dev).view(1, T),
        }

    def decode_inputs_captured(self, sel: torch.Tensor, b_seq_len: torch.Tensor,
                               b_tail_len: torch.Tensor, tidx_dev: torch.Tensor,
                               kt: torch.Tensor, vt: torch.Tensor,
                               max_seq_len: int) -> dict:
        """CAPTURE-SAFE ``[B,*]`` arg bundle for ``ommx_paged_decode``.

        All control tensors are FIXED-ADDRESS device buffers the HOST build-seam
        refreshed BEFORE this captured forward (mirrors ``CanonicalKVStore`` /
        ``OMMXStepManager`` for B==1):

          * ``sel`` ``[B]`` int64  — batch-row -> pool slot (stable across the step);
          * ``b_seq_len`` ``[B]`` int32 — packed token count per request;
          * ``b_tail_len`` ``[B]`` int32 — bf16 residual rows per request;
          * ``tidx_dev`` ``[B, T]`` int64 — absolute tail token indices per request;
          * ``kt`` / ``vt`` ``[B, T, H, D]`` bf16 — fixed-address tail OUTPUT buffers.

        The bf16 tail gather is ONE device ``index_select`` of ``B*T`` flat rows out of
        ``k_hist`` (flat row = ``slot*max_seq + token``), written into the fixed ``kt`` /
        ``vt`` buffers — no host loop, no per-step alloc, no ``.item()``. The whole pool
        planes are static; ``req_to_token`` / ``req_to_group`` are gathered by ``sel``
        (a device ``index_select`` — capturable). ``max_seq_len`` is a FIXED grid-size
        int (the manager passes the worst-case capacity, NOT a per-step ``.max()``), so
        the captured grid is constant; the device ``b_seq_len`` bounds the live prefix.
        """
        B = int(sel.shape[0])
        T = int(tidx_dev.shape[1])
        planes = self.pool_planes()
        planes["req_to_token"] = self._req_to_token.index_select(0, sel)
        planes["req_to_group"] = self._req_to_group.index_select(0, sel)
        # ring stride: ring_cap (no shadow) | max_seq_len (full). The flat row is
        # slot*S + ring_slot(token); ``tidx_dev`` carries ABS tail positions (the host
        # build-seam fills them), remapped to ring slots here by capture-safe arithmetic
        # (identity when ring OFF). Padding 0s map to ring slot 0, masked by b_tail_len.
        S = self.S_ring
        rsl = self._ring_slot_dev(tidx_dev)                     # [B, T] ring rows
        kflat = self.k_hist.view(self.num_seqs * S, self.H, self.D)
        vflat = self.v_hist.view(self.num_seqs * S, self.H, self.D)
        flat = (sel.view(B, 1) * S + rsl).reshape(B * T)        # [B*T]
        kt.copy_(kflat.index_select(0, flat).view(B, T, self.H, self.D))
        vt.copy_(vflat.index_select(0, flat).view(B, T, self.H, self.D))
        return {
            "planes": planes,
            "k_tail": kt, "v_tail": vt,
            "b_tail_len": b_tail_len,
            "b_seq_len": b_seq_len,
            "req_to_token": planes["req_to_token"],
            "req_to_group": planes["req_to_group"],
            "packed_start_offset": 0,
            "max_seq_len": int(max_seq_len),
            "max_tail_len": self.max_tail_len,
        }

    def decode_inputs_batched(self, slots: Sequence[int]) -> dict:
        """Full ``[B,*]`` arg bundle for ``ommx_paged_decode`` over ``slots``.

        NO-SYNC fast path: reuses persistent fixed-address buffers (no per-step
        alloc), builds the per-request tail-gather indices on the HOST then issues ONE
        H2D + ONE batched ``index_select``-style gather (vs the old B separate H2D +
        index_selects), and derives ``max_seq_len`` from the HOST ``packed_groups``
        list (no ``b_seq.max().item()`` D2H sync). ``slots`` is the per-row request-id
        list; the whole pool planes are static-shape, the tables route each row.
        """
        slots = [int(s) for s in slots]
        B = len(slots)
        self._ensure_decode_buffers(B)
        d = self._dbuf
        T = self.max_tail_len
        dev = self.device

        # ── row select + per-request seq/tail derived on the HOST (no device sync) ──
        sel_h = d["sel_host"]
        bseq_h = d["bseq_host"]
        btail_h = d["btail_host"]
        tidx_h = d["tidx_host"]
        tidx_h[:B].zero_()
        max_packed_groups = 0
        for b, r in enumerate(slots):
            sel_h[b] = r
            pg = self.packed_groups[r]
            if pg > max_packed_groups:
                max_packed_groups = pg
            bseq_h[b] = pg * self.gt
            idx = self.window.tail_indices(self.seq_len[r])
            if self.kv_ring:
                idx = [self._ring_slot_host(p) for p in idx]   # abs -> ring rows
            n = min(len(idx), T)
            if n:
                tidx_h[b, :n] = torch.tensor(idx[:n], dtype=torch.long)
            btail_h[b] = n
        # ── push host-built control tensors to the fixed device buffers (3 small H2D) ──
        d["sel"][:B].copy_(sel_h[:B], non_blocking=True)
        d["b_seq"][:B].copy_(bseq_h[:B], non_blocking=True)
        d["b_tail"][:B].copy_(btail_h[:B], non_blocking=True)
        d["tidx_dev"][:B].copy_(tidx_h[:B], non_blocking=True)

        sel = d["sel"][:B]
        planes = self.pool_planes()
        planes["req_to_token"] = self._req_to_token.index_select(0, sel)
        planes["req_to_group"] = self._req_to_group.index_select(0, sel)

        # ── batched bf16 tail gather: ONE index_select of B*T rows out of the FLAT
        # k_hist (no [B,max_seq,H,D] materialization). Flat row = slot*max_seq + token.
        kt = d["kt"][:B]
        vt = d["vt"][:B]
        # ring stride: ring_cap (no shadow) | max_seq_len (full). tidx_dev already holds
        # RING ROWS (remapped on the host loop above), so the flat row is slot*S + ring_row.
        S = self.S_ring
        kflat = self.k_hist.view(self.num_seqs * S, self.H, self.D)
        vflat = self.v_hist.view(self.num_seqs * S, self.H, self.D)
        tdev = d["tidx_dev"][:B]                            # [B, T] ring rows
        flat = (sel.view(B, 1) * S + tdev).reshape(B * T)   # [B*T]
        kt.copy_(kflat.index_select(0, flat).view(B, T, self.H, self.D))
        vt.copy_(vflat.index_select(0, flat).view(B, T, self.H, self.D))

        max_seq = max(1, max_packed_groups * self.gt)      # HOST-derived (no sync)
        return {
            "planes": planes,
            "k_tail": kt, "v_tail": vt,
            "b_tail_len": d["b_tail"][:B],
            "b_seq_len": d["b_seq"][:B],
            "req_to_token": planes["req_to_token"],
            "req_to_group": planes["req_to_group"],
            "packed_start_offset": 0,
            "max_seq_len": max_seq,
            "max_tail_len": self.max_tail_len,
        }

    # ── DEVICE decode path (no per-step host loop / alloc / torch.tensor) ────────

    def decode_seed_seq_device(self, slots: Sequence[int]) -> torch.Tensor:
        """Seed the persistent device ``seq_dev`` / ``rows`` buffers from host state.

        Called ONCE at the start of a decode loop (after prefill) — copies each
        request's current host ``seq_len`` into the device ``seq_dev`` buffer (one H2D)
        and writes the row->slot map. From then on ``advance_seq_device`` advances
        ``seq_dev`` purely on device; the host ``self.seq_len`` is advanced in lockstep
        by the host regroup seam (``append_decode_batched``). ``slots`` is the per-row
        request id list (``range(B)`` for the uniform batch).
        """
        slots = [int(s) for s in slots]
        B = len(slots)
        self._ensure_decode_buffers(B)
        d = self._dbuf
        # ONE-TIME H2D of the seed seq lengths + row map (a single small torch.tensor,
        # outside the per-step loop). After this, the per-step path touches NO host int.
        seed = torch.tensor([self.seq_len[r] for r in slots], dtype=torch.long)
        d["seq_dev"][:B].copy_(seed, non_blocking=True)
        d["rows"][:B].copy_(torch.tensor(slots, dtype=torch.long), non_blocking=True)
        d["sel"][:B].copy_(torch.tensor(slots, dtype=torch.long), non_blocking=True)
        return d["seq_dev"][:B]

    def advance_seq_device(self, B: int) -> None:
        """Advance the device ``seq_dev`` (ABS seq length) by ONE on device (one kernel).

        ``cols`` (the ABS write column for the captured scatter) is ``seq_dev - 1`` AFTER
        the advance — i.e. the newly-written decode token's position. No host int, no
        ``.item()``, no per-step ``torch.tensor``.
        """
        d = self._dbuf
        d["seq_dev"][:B].add_(1)
        d["cols"][:B].copy_(d["seq_dev"][:B] - 1)

    def decode_inputs_device(self, B: int, max_seq_len: int) -> dict:
        """Full ``[B,*]`` decode arg bundle built ENTIRELY on device (no host loop).

        Replaces ``decode_inputs_batched``'s per-request HOST loop (``tail_indices`` ->
        python list, the ``_ring_slot_host`` list-comp, the per-request ``torch.tensor``,
        the dict rebuild, 4 H2D copies) with CLOSED-FORM device tensor arithmetic over
        the persistent ``seq_dev`` buffer:

          * ``boundary`` = where(seq-sink-recent<=0, min(sink,seq),
                                  sink + ((seq-sink-recent)//gt)*gt)   — vectorized;
          * ``b_seq_len`` = clamp(boundary - sink, min=0)              — packed tokens;
          * ``b_tail_len`` = min(sink_rows + recent_rows, T);
          * ``tidx`` (ABS) = [0..n_sink) sink rows ++ [boundary, seq) recent rows,
                             0-padded (kernel masks past b_tail_len).
        Proven BIT-EXACT to the host loop (``window.tail_indices`` / ``boundary`` /
        ``num_groups``) for ragged seqs incl <sink / =boundary / +partial / 4K, ring ON
        and OFF (tests/test_kv_pool_device.py). The bf16 tail gather + ring remap +
        table gather then reuse ``decode_inputs_captured`` (which holds ABS in
        ``tidx_dev`` and remaps to ring rows on device). ``max_seq_len`` is the FIXED
        worst-case grid int (the device ``b_seq_len`` bounds the live prefix) — NO
        ``.max().item()`` sync, mirroring the captured manager.
        """
        d = self._dbuf
        T = self.max_tail_len
        dev = self.device
        sink = int(self.window.sink_tokens)
        recent = int(self.window.recent_window)
        gt = int(self.gt)
        seq = d["seq_dev"][:B]                                  # [B] int64, ABS
        # ── closed-form boundary / packed / tail counts (all device, no python) ──
        extra = seq - sink - recent
        bnd_grown = sink + torch.div(extra, gt, rounding_mode="floor") * gt
        bnd_small = torch.minimum(torch.full_like(seq, sink), seq)
        boundary = torch.where(extra <= 0, bnd_small, bnd_grown)   # [B]
        b_seq = torch.clamp(boundary - sink, min=0).to(torch.int32)
        n_sink = torch.minimum(torch.full_like(seq, sink), seq)    # [B]
        n_rec = torch.clamp(seq - boundary, min=0)                 # [B]
        n_total = n_sink + n_rec                                   # [B]
        b_tail = n_total.clamp(max=T).to(torch.int32)
        # ── tidx ABS [B,T]: sink ramp then [boundary, seq) ramp, 0-padded ──
        col = d["col_arange"].expand(B, T)                         # [B,T] fixed ramp
        ns = n_sink.view(B, 1)
        in_sink = col < ns
        valid = col < n_total.view(B, 1)
        tidx_abs = torch.where(in_sink, col, boundary.view(B, 1) + (col - ns))
        tidx_abs = torch.where(valid, tidx_abs, torch.zeros_like(tidx_abs))
        # write into the persistent fixed-address buffers (in-place, no alloc).
        d["b_seq"][:B].copy_(b_seq)
        d["b_tail"][:B].copy_(b_tail)
        d["tidx_dev"][:B].copy_(tidx_abs)                          # ABS (captured remaps)
        kt = d["kt"][:B]
        vt = d["vt"][:B]
        # reuse the proven capture-safe reader: it gathers the bf16 tail (ONE flat
        # index_select), remaps ABS->ring rows on device, and gathers the per-request
        # tables by ``sel`` — all device ops, no host work.
        return self.decode_inputs_captured(
            d["sel"][:B], d["b_seq"][:B], d["b_tail"][:B],
            d["tidx_dev"][:B], kt, vt, int(max_seq_len))


__all__ = ["MultiSeqKVPool"]
