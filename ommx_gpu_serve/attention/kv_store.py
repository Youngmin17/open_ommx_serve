# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Canonical per-layer KV store — incremental pack into the decode-kernel planes.

The fresh, dependency-free analogue of the parent ``OmmxKVTwin`` built ONLY on
``ommx_pack_kv_canonical_block`` (the canonical pack) + ``kv_window`` (the KIVI
residual-window arithmetic). One ``CanonicalKVStore`` per attention layer owns:

  * the QUANTIZED packed planes (``k_base``/``k_scale``/``k_zp``/outlier sidecar +
    ``v_main``/``v_scale``/``v_zp``) for the prefix ``[sink, boundary)``, written
    one 32-token scale-group at a time as groups complete;
  * the bf16 RESIDUAL window (sink ∪ recent + incomplete tail) the decode kernel
    reads directly.

Design split (the serving contract):
  * ``append(k, v)``    — capture-SAFE: stash the new bf16 K/V row (a fixed-shape
    scatter). Does NOT quantize (a single token can't produce a 32-token group's
    affine stats).
  * ``maybe_regroup()`` — HOST seam (run OUTSIDE CUDA-graph capture): pack any
    newly completed group via ``ommx_pack_kv_canonical_block`` and write it into the
    plane slot. This is where the heavy work lives.
  * ``decode_inputs(q)`` — assemble the exact arg bundle for the
    ``ommx_gpu_serve::paged_decode`` op (planes + bf16 tail + seq buffers).

This module is the CPU-verifiable reference for the write-pack path: incremental
per-group packing is BIT-EXACT to a single bulk ``ommx_pack_kv_canonical_block`` of
the same prefix (the ``tests/test_kv_store.py`` gate). The GPU paged vLLM store
(``integration/vllm``) reuses this logic over vLLM's block table.
"""
from __future__ import annotations

from typing import Optional

import torch

from .kv_window import CANONICAL_GROUP_TOKENS, WindowSpec
from .pack import ommx_pack_kv_canonical_block

_OVAL_BYTES = {4: lambda k: (k + 1) // 2}


def _is_capturing() -> bool:
    """True iff inside a CUDA-graph capture. CPU-safe: CPU-only torch builds raise
    from ``is_current_stream_capturing`` (dummy CUDA base class), so guard on
    ``cuda.is_available()`` first — on CPU there is never a capture in flight."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.is_current_stream_capturing()


class CanonicalKVStore:
    """Per-layer incremental KV store producing the canonical decode planes."""

    def __init__(
        self,
        head_dim: int,
        n_kv_heads: int,
        max_seq_len: int,
        *,
        k_format: str = "i2f4",
        v_format: str = "i2",
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
        vfmt = str(v_format).lower()
        if vfmt not in ("i2", "bf16"):
            raise ValueError(f"v_format must be i2|bf16; got {v_format!r}")
        self.D = int(head_dim)
        self.H = int(n_kv_heads)
        self.max_seq_len = int(max_seq_len)
        self.k_format = kfmt
        self.v_format = vfmt
        self.k = int(outliers_per_vector)
        self.outlier_select = str(outlier_select).lower()
        self.outlier_repr = str(outlier_repr).lower()
        self.use_pow2 = bool(use_pow2)
        # KV dedicated FP4 outlier map + int8 pow2-exp scale gating — mirror the
        # pack.py resolution (env OMMX_KV_OUTLIER_MAP / OMMX_KV_INT8_SCALE) so the
        # store's incremental pack is BIT-EXACT to a bulk ommx_pack_kv_canonical_block.
        import os as _os
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
        # V channel-group vector_length (per-token axis): {16,32,64,128}, must divide D.
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

        # capacity: one extra group of slack so a just-completed group never OOBs.
        self.G_cap = max(1, (self.max_seq_len + self.gt - 1) // self.gt + 1)
        self.P_cap = self.G_cap * self.pages_per_group

        dev, sdt = self.device, self.scale_dtype
        # int8 pow2-exp scale (BIT-EXACT when use_pow2): k_scale/v_scale hold the
        # int8 EXPONENT (scale = 2^exp); zp stays bf16. Otherwise bf16 scale.
        scl_dt = torch.int8 if self.kv_int8_scale else sdt
        self.k_base = torch.zeros(self.P_cap, self.ps, self.H, self.k_base_w,
                                  dtype=torch.uint8, device=dev)
        # V=bf16 (v10a): v_main holds the raw bf16 V row (D values); else i2-packed (D//4).
        if self.v_format == "bf16":
            self.v_main = torch.zeros(self.P_cap, self.ps, self.H, self.D,
                                      dtype=torch.bfloat16, device=dev)
        else:
            self.v_main = torch.zeros(self.P_cap, self.ps, self.H, self.D // 4,
                                      dtype=torch.uint8, device=dev)
        self.k_scale = torch.zeros(self.G_cap, self.H, self.D, dtype=scl_dt, device=dev)
        self.k_zp = torch.zeros(self.G_cap, self.H, self.D, dtype=sdt, device=dev)
        self.v_scale = torch.zeros(self.P_cap, self.ps, self.H, self.NGV,
                                   dtype=scl_dt, device=dev)
        self.v_zp = torch.zeros(self.P_cap, self.ps, self.H, self.NGV,
                                dtype=sdt, device=dev)
        # KV dedicated FP4 outlier-map params (per channel-group, bf16). Allocated
        # only when the map is active AND there are outliers.
        if self.k > 0 and self.kv_outlier_map:
            self.k_fp4_mapscale = torch.zeros(self.G_cap, self.H, self.D,
                                              dtype=sdt, device=dev)
            self.k_fp4_mapcenter = torch.zeros(self.G_cap, self.H, self.D,
                                               dtype=sdt, device=dev)
        else:
            self.k_fp4_mapscale = None
            self.k_fp4_mapcenter = None
        if self.k > 0 and self.outlier_repr == "relidx7":
            idx_fb = (7 * self.k + 7) // 8
            self.k_oidx = torch.zeros(self.G_cap, self.H, self.D, idx_fb,
                                      dtype=torch.uint8, device=dev)
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

        # bf16 residual history. TWO sizings, gated by OMMX_KV_RING (default OFF):
        #   * FULL-HIST (default): keep the whole bf16 sequence [max_seq_len,H,D] — the
        #     CPU-verifiable reference; the lmcache/sglang/radix bulk-pack callers read
        #     arbitrary prefix ranges, so they REQUIRE the full shadow.
        #   * RING (OMMX_KV_RING=1, the single-batch vLLM graph decode lever): only the
        #     worst-case LIVE set (sink ∪ recent ∪ the group being packed ∪ the new
        #     accumulating group) is ever read, so a small ring of `ring_cap` rows
        #     suffices — the 17GB@128K bf16 shadow collapses to ~104 rows. Abs token
        #     pos -> ring slot via `_ring_slot` (sink pinned [0:sink), recent positions
        #     sliding modulo Rrec). Capacity-shaped planes are unchanged (≤3-bit, cheap).
        import os as _os2
        _ring_raw = _os2.environ.get("OMMX_KV_RING")
        self.kv_ring = bool(_ring_raw) and _ring_raw not in {"0", "false", "off", "no"}
        self.kv_cap = self.max_seq_len
        if self.kv_ring:
            # ring geometry: sink pinned, then a sliding window over the recent live set.
            # Rrec must cover, at ANY decode step `seq`, the abs-pos window the ring is
            # read/written over: [boundary - gt, seq). Its width is (seq - boundary) + gt,
            # and (seq - boundary) ∈ [recent, recent + gt), so the width < recent + 2*gt.
            # Since width < Rrec, distinct live positions never alias modulo Rrec.
            self._ring_sink = int(self.window.sink_tokens)
            self._ring_rec = int(self.window.recent_window) + 2 * int(self.gt)
            self.ring_cap = self._ring_sink + self._ring_rec
            self.k_hist = torch.zeros(self.ring_cap, self.H, self.D,
                                      dtype=torch.bfloat16, device=dev)
            self.v_hist = torch.zeros(self.ring_cap, self.H, self.D,
                                      dtype=torch.bfloat16, device=dev)
        else:
            self._ring_sink = 0
            self._ring_rec = 0
            self.ring_cap = self.kv_cap
            self.k_hist = torch.zeros(self.kv_cap, self.H, self.D,
                                      dtype=torch.bfloat16, device=dev)
            self.v_hist = torch.zeros(self.kv_cap, self.H, self.D,
                                      dtype=torch.bfloat16, device=dev)
        self.seq_len = 0
        self.packed_groups = 0  # completed groups already written to the planes

        # FIXED-ADDRESS capacity-shaped tables + tail buffers for the CUDA-graph
        # path (the captured op needs static shapes; the device b_seq_len/b_tail_len
        # bound the actual region). Identity page/group tables over full capacity.
        self._req_to_token_full = torch.arange(
            self.P_cap, dtype=torch.int32, device=dev).reshape(1, self.P_cap)
        self._req_to_group_full = torch.arange(
            self.G_cap, dtype=torch.int32, device=dev).reshape(1, self.G_cap)
        t_max = (self.window.sink_tokens + self.window.recent_window
                 + self.window.group_tokens)
        self.tail_cap = int(t_max)
        self.k_tail_buf = torch.zeros(1, self.tail_cap, self.H, self.D,
                                      dtype=torch.bfloat16, device=dev)
        self.v_tail_buf = torch.zeros(1, self.tail_cap, self.H, self.D,
                                      dtype=torch.bfloat16, device=dev)

    # ── ring address mapping (OMMX_KV_RING only) ───────────────────────────────

    def _ring_slot_host(self, pos: int) -> int:
        """Abs token pos -> ring row (python int). Identity when ring is OFF.

        Sink positions ``[0, sink)`` pin to the same ring rows; recent positions
        ``>= sink`` slide modulo ``Rrec`` into ``[sink, sink+Rrec)``. The live abs
        window at any step is narrower than ``Rrec``, so live positions never alias.
        """
        if not self.kv_ring:
            return int(pos)
        p = int(pos)
        if p < self._ring_sink:
            return p
        return self._ring_sink + (p - self._ring_sink) % self._ring_rec

    def _ring_slot_dev(self, pos: torch.Tensor) -> torch.Tensor:
        """Abs token pos -> ring row (device int tensor, capture-safe). Identity OFF.

        Pure tensor arithmetic (``torch.where`` + ``%``) — no ``.item()``, no host
        sync, fixed shape — so it can live inside the captured decode region. ``pos``
        is any int dtype/shape; the result matches ``pos``'s shape/dtype/device.
        """
        if not self.kv_ring:
            return pos
        sink = self._ring_sink
        rec = self._ring_rec
        rel = (pos - sink) % rec + sink
        return torch.where(pos < sink, pos, rel)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset_inplace(self) -> None:
        """CUDA-graph-SAFE new-sequence reset: zero the logical counters but KEEP
        every tensor (no reallocation).

        The captured decode graph holds the ADDRESSES of ``k_hist`` and the packed
        planes (the captured ``write_token`` index_copy_ + ``gather_tail_into`` +
        the kernel's plane reads bind to the tensor pointers AT CAPTURE TIME). A new
        sequence's prefill must therefore reuse the SAME tensors — reallocating
        (the old ``_ommx_reset`` behaviour) makes the captured graph read/write a
        stale address. Only the logical cursors reset; the kernel reads strictly
        ``[0, b_seq_len)`` of the planes and the live tail positions of ``k_hist``,
        so stale slots beyond the new prefix are never observed (the next prefill
        overwrites them as groups pack)."""
        self.seq_len = 0
        self.packed_groups = 0

    # ── write path ────────────────────────────────────────────────────────────

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """EAGER: stash one decode step's bf16 K/V row ``[H, D]`` (python index)."""
        if not self.kv_ring and self.seq_len >= self.kv_cap:
            raise RuntimeError(
                f"CanonicalKVStore capacity {self.kv_cap} exceeded (seq_len)")
        slot = self._ring_slot_host(self.seq_len)
        self.k_hist[slot] = k.to(torch.bfloat16)
        self.v_hist[slot] = v.to(torch.bfloat16)
        self.seq_len += 1

    # ── CUDA-graph capture-safe path (device-indexed write + host regroup) ──────

    def write_token(self, pos: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor) -> None:
        """CAPTURE-SAFE: write one bf16 K/V row at device-index ``pos`` ([1] int64).

        ``index_copy_`` with a DEVICE index is a static-shape op (no python index,
        no host sync) so it can live inside the captured decode region. ``pos`` is a
        fixed-address device buffer the host build-seam updates before each replay.
        Does NOT touch ``self.seq_len`` (the host build-seam owns it via ``advance_to``).

        Under OMMX_KV_RING, ``pos`` (an abs token pos) is remapped to its ring slot by
        capture-safe tensor arithmetic before the scatter. The host build-seam fills
        ``pos`` with the ABS position (``seq-1``); the remap keeps the scatter target a
        valid ring row. Identity when ring is OFF (the index_copy_ shape is unchanged).
        """
        slot = self._ring_slot_dev(pos)
        self.k_hist.index_copy_(0, slot, k.view(1, self.H, self.D).to(torch.bfloat16))
        self.v_hist.index_copy_(0, slot, v.view(1, self.H, self.D).to(torch.bfloat16))

    def advance_to(self, seq_len: int) -> int:
        """HOST seam: set the current seq length and pack any completed groups.

        Returns the number of groups packed this call. Runs OUTSIDE CUDA-graph
        capture (CPU pack + the bf16->cpu copy it needs); the captured forward only
        reads the planes it writes.
        """
        self.seq_len = int(seq_len)
        return self.maybe_regroup()

    def gather_tail_into(self, tail_idx: torch.Tensor, k_tail: torch.Tensor,
                         v_tail: torch.Tensor, n_valid: int) -> None:
        """CAPTURE-SAFE: gather sink∪recent rows into fixed-address tail buffers.

        ``tail_idx`` [T_tail_max] device int64 (host build-seam fills the first
        ``n_valid`` with real ABS positions, the rest clamped to 0); ``k_tail``/``v_tail``
        are fixed ``[1, T_tail_max, H, D]`` bf16 buffers. ``index_select`` is static.

        Under OMMX_KV_RING the abs ``tail_idx`` is remapped to ring slots by capture-safe
        tensor arithmetic before the gather (the padding 0s map to ring slot 0, masked by
        ``b_tail_len`` in the kernel). Identity when ring is OFF.
        """
        slot = self._ring_slot_dev(tail_idx)
        kt = self.k_hist.index_select(0, slot)                # [T_tail_max, H, D]
        vt = self.v_hist.index_select(0, slot)
        k_tail[0].copy_(kt)
        v_tail[0].copy_(vt)

    def append_block(self, K: torch.Tensor, V: torch.Tensor) -> None:
        """Stash a prefill block of bf16 K/V ``[T, H, D]`` each (then regroup)."""
        T = int(K.shape[0])
        if not self.kv_ring:
            if self.seq_len + T > self.kv_cap:
                raise RuntimeError("CanonicalKVStore capacity exceeded (block)")
            self.k_hist[self.seq_len:self.seq_len + T] = K.to(torch.bfloat16)
            self.v_hist[self.seq_len:self.seq_len + T] = V.to(torch.bfloat16)
            self.seq_len += T
            self.maybe_regroup()
            return
        # RING (no-shadow): the ring holds only the last <Rrec rows, so a prefill block of
        # T>>Rrec tokens CANNOT first be written into the ring and packed from there (the
        # bulk source rows alias before they are read by the packer). BUT the prefill source
        # K/V (the input args, [T,H,D]) is already resident in full for the duration of this
        # call — so pack the WHOLE completed prefix [sink, boundary) directly FROM THE INPUT
        # in ONE device ommx_pack_kv_canonical_block (hundreds of groups, no per-token python,
        # GPU pack when OMMX_KV_GPU_PACK=1), then copy only the persistent live tail rows
        # ([boundary-gt, T) ∪ sink — < Rrec rows, distinct ring slots) into the ring. This is
        # BIT-EXACT to the no-ring path (same packer, same gt-group decomposition of the same
        # bf16 source); it just never round-trips the bulk prefix through the small ring.
        if self.seq_len != 0:
            raise RuntimeError(
                "ring append_block expects a fresh sequence (seq_len==0); chunked prefill "
                "under OMMX_KV_RING is not supported (the ring cannot retain >Rrec unpacked "
                "rows across chunks). Reset the store before each prefill block.")
        Kb = K.to(torch.bfloat16)
        Vb = V.to(torch.bfloat16)
        self.seq_len = T
        capturing = _is_capturing()
        target = self.window.num_groups(T)            # completed groups for this prefix
        if target > 0 and not capturing:
            sink = self.window.sink_tokens
            t0 = sink
            t1 = sink + target * self.gt              # == boundary(T)
            # pack the whole completed prefix straight from the INPUT (not the ring) in ONE
            # call — no aliasing (input rows are all live), no per-group loop.
            self._pack_block_into_planes(Kb[t0:t1], Vb[t0:t1], 0, target)
            self.packed_groups = target
        elif capturing:
            # dummy capture only records op/addresses; the real prefill+pack runs eager
            # before replay (mirror maybe_regroup's capture skip). Leave packed_groups at 0.
            target = 0
        # persist ONLY the live tail rows into the ring: sink [0,sink) + [boundary-gt, T).
        # These are the only abs positions any later decode step (or its boundary-pack)
        # reads; they number < Rrec so their ring slots are distinct. One vectorized
        # device scatter — no python per-row loop.
        boundary = self.window.boundary(T)
        sink_tokens = self.window.sink_tokens
        lo = max(sink_tokens, boundary - self.gt)
        # WARMUP guard: vLLM dummy_run uses a tiny T (often T<sink) -> arange(lo, T) with
        # lo>=sink>T raises "upper/lower bound inconsistent with step sign", which the
        # non-strict backend SILENTLY CATCHES -> _ommx_dead -> bf16 FA fallback (the cause
        # of every ring run secretly running FlashAttention, not OMMX). Only add the recent
        # span when it is non-empty.
        parts = [torch.arange(0, min(sink_tokens, T), device=Kb.device)]
        if lo < T:
            parts.append(torch.arange(lo, T, device=Kb.device))
        live = torch.cat(parts).to(torch.long)
        slots = self._ring_slot_dev(live)
        self.k_hist.index_copy_(0, slots, Kb.index_select(0, live))
        self.v_hist.index_copy_(0, slots, Vb.index_select(0, live))

    def maybe_regroup(self) -> int:
        """HOST seam: pack any newly completed 32-token groups. Returns #packed.

        Packs ALL pending groups in ONE ``ommx_pack_kv_canonical_block`` call (a
        multi-group block) rather than group-by-group — critical for prefill, where
        hundreds of groups complete at once (per-group CPU packing was the prefill
        bottleneck). A single decode step completes at most one group (bulk == 1).
        """
        target = self.window.num_groups(self.seq_len)
        if target <= self.packed_groups:
            return 0
        # CUDA-graph capture seam: the warmup dummy_run reaches this prefill/regroup path
        # INSIDE the capture stream, where the host packer's k_hist[...].to("cpu") round-trip
        # is an illegal CPU<->CUDA copy (it latched _ommx_dead -> silent bf16 fallback). The
        # dummy capture only records op/addresses (values are garbage either way); the REAL
        # packed planes are produced by the eager prefill + the host-seam regroup BEFORE each
        # captured replay. So skip the CPU pack while capturing and DO NOT advance
        # packed_groups (so the real run re-packs these groups for real).
        if _is_capturing():
            return 0
        n = target - self.packed_groups
        if self.kv_ring:
            # RING: pack one group at a time so each _regroup_pack_range read-span == gt
            # < Rrec (a multi-group span would alias in the ring). Decode advances by one
            # group per step so this is normally a single iteration; the loop only matters
            # if advance_to ever skips >1 group (then earlier groups' rows must still be
            # live in the ring, which holds only the last <Rrec rows — so the caller MUST
            # advance by ≤ one group per call under ring, the asserted decode contract).
            for g in range(self.packed_groups, target):
                self._regroup_pack_range(g, g + 1)
            self.packed_groups = target
            return n
        self._regroup_pack_range(self.packed_groups, target)
        self.packed_groups = target
        return n

    def _regroup_pack_range(self, g0: int, g1: int) -> None:
        """Pack completed groups ``[g0, g1)`` into their plane slots in ONE call.

        Source rows come from the contiguous shadow ``[t0,t1)`` (no ring) OR a ring
        gather of the same abs span (decode path); the actual pack + plane writes live
        in the shared :meth:`_pack_block_into_planes`. One multi-group call instead of a
        per-group loop (prefill packs hundreds at once).
        """
        sink = self.window.sink_tokens
        t0 = sink + g0 * self.gt
        t1 = sink + g1 * self.gt
        if self.kv_ring:
            # gather abs rows [t0,t1) through their ring slots — VECTORIZED on device
            # (_ring_slot_dev = torch.where+% ); the per-element python _ring_slot_host loop
            # was a host-seam stall (p99 source). (t1-t0)==gt < Rrec -> slots distinct.
            idx = self._ring_slot_dev(torch.arange(t0, t1, device=self.device)).to(torch.long)
            K_blk = self.k_hist.index_select(0, idx)            # bf16 [(g1-g0)*gt,H,D]
            V_blk = self.v_hist.index_select(0, idx)
        else:
            K_blk = self.k_hist[t0:t1]                          # bf16 [(g1-g0)*gt,H,D]
            V_blk = self.v_hist[t0:t1]
        self._pack_block_into_planes(K_blk, V_blk, g0, g1)

    def _pack_block_into_planes(self, K_blk: torch.Tensor, V_blk: torch.Tensor,
                                g0: int, g1: int) -> None:
        """Pack a bf16 source block of EXACTLY ``(g1-g0)*gt`` tokens into plane slots
        ``[g0, g1)`` in ONE ``ommx_pack_kv_canonical_block`` call.

        ``K_blk`` / ``V_blk`` are ``[(g1-g0)*gt, H, D]`` bf16 in ANY layout (a ring
        gather, a contiguous shadow slice, OR — the prefill no-shadow path — a slice of
        the caller's input tensor that never entered the ring). The packer derives the
        per-32-token-group affine stats independently per group, so the source's
        origin is irrelevant to bit-exactness: this is the single shared write-pack
        body for the shadow, ring-decode and ring-prefill paths.
        """
        # GPU-PACKER (OMMX_KV_GPU_PACK): pack on-device, NO D2H/H2D round-trip (the host
        # CPU pack was the p99-spike / slow-TTFT source). relidx7/i2f4 is device-invariant;
        # combinadic still needs CPU (pack.py rank_bytes.cpu) so it falls back. Bit-exact.
        import os as _os
        # default ON for relidx7/i2f4 (bit-exact, device-invariant) — the on-device pack
        # removes the boundary-step CPU pack D2H/H2D round-trip = the p99 spike. Explicit
        # OMMX_KV_GPU_PACK=0 escapes; combinadic always falls back to CPU.
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
            v_format=self.v_format,
            use_pow2=self.use_pow2,
            kv_outlier_map=self.kv_outlier_map,
            kv_int8_scale=self.kv_int8_scale,
            group_tokens=self.gt,
            group_channels=self.gc,
        )
        dev = self.device
        p0 = g0 * self.pages_per_group
        p1 = g1 * self.pages_per_group
        self.k_base[p0:p1] = planes["k_base"].to(dev)
        self.v_main[p0:p1] = planes["v_main"].to(dev)
        self.v_scale[p0:p1] = planes["v_scale"].to(dev)
        self.v_zp[p0:p1] = planes["v_zp"].to(dev)
        self.k_scale[g0:g1] = planes["k_scale"].to(dev)
        self.k_zp[g0:g1] = planes["k_zp"].to(dev)
        if self.k > 0:
            self.k_oval[g0:g1] = planes["k_oval"].to(dev)
            if self.outlier_repr == "relidx7":
                self.k_oidx[g0:g1] = planes["k_oidx"].to(dev)
            else:
                self.k_crank[g0:g1] = planes["k_crank"].to(dev)
            if self.kv_outlier_map:
                self.k_fp4_mapscale[g0:g1] = planes["k_fp4_mapscale"].to(dev)
                self.k_fp4_mapcenter[g0:g1] = planes["k_fp4_mapcenter"].to(dev)

    # ── read path ───────────────────────────────────────────────────────────────

    def decode_planes(self) -> dict:
        """Plane dict (views over the filled slots) for the current packed prefix."""
        G = self.packed_groups
        P = G * self.pages_per_group
        planes = {
            "k_base": self.k_base[:P],
            "k_scale": self.k_scale[:G],
            "k_zp": self.k_zp[:G],
            "k_oidx": self.k_oidx[:G] if self.k_oidx is not None else None,
            "k_oval": self.k_oval[:G] if self.k_oval is not None else None,
            "k_crank": self.k_crank[:G] if self.k_crank is not None else None,
            "k_fp4_mapscale": (self.k_fp4_mapscale[:G]
                               if self.k_fp4_mapscale is not None else None),
            "k_fp4_mapcenter": (self.k_fp4_mapcenter[:G]
                                if self.k_fp4_mapcenter is not None else None),
            "v_main": self.v_main[:P],
            "v_scale": self.v_scale[:P],
            "v_zp": self.v_zp[:P],
            "req_to_token": torch.arange(P, dtype=torch.int32,
                                         device=self.device).reshape(1, P),
            "req_to_group": torch.arange(G, dtype=torch.int32,
                                         device=self.device).reshape(1, G),
            "head_dim": self.D, "n_kv_heads": self.H,
            "num_tokens": G * self.gt, "n_groups": G,
            "page_size": self.ps,
            "outliers_per_vector": self.k,
            "outlier_repr": self.outlier_repr,
            "outlier_select": self.outlier_select,
            "group_tokens": self.gt, "group_channels": self.gc,
            "k_format": self.k_format, "v_format": self.v_format,
            "k_base_bits": self.k_base_bits, "oval_bits": self.oval_bits,
            "kv_outlier_map": bool(self.kv_outlier_map and self.k > 0),
            "kv_int8_scale": bool(self.kv_int8_scale),
        }
        return planes

    def decode_planes_full(self) -> dict:
        """FULL-CAPACITY (fixed-shape, fixed-address) plane dict for CUDA graph.

        Unlike ``decode_planes`` (which slices ``[:P]`` — a shape that grows as
        groups pack and so is NOT graph-capturable), this returns the whole
        pre-allocated plane tensors + identity page/group tables. The actual packed
        region is bounded at run time by the device ``b_seq_len`` the kernel reads,
        so the captured op has static shapes while still attending only the live
        prefix.
        """
        return {
            "k_base": self.k_base, "k_scale": self.k_scale, "k_zp": self.k_zp,
            "k_oidx": self.k_oidx, "k_oval": self.k_oval, "k_crank": self.k_crank,
            "k_fp4_mapscale": self.k_fp4_mapscale,
            "k_fp4_mapcenter": self.k_fp4_mapcenter,
            "v_main": self.v_main, "v_scale": self.v_scale, "v_zp": self.v_zp,
            "req_to_token": self._req_to_token_full,
            "req_to_group": self._req_to_group_full,
            "page_size": self.ps, "outliers_per_vector": self.k,
            "k_format": self.k_format, "v_format": self.v_format,
            "head_dim": self.D, "n_kv_heads": self.H,
            "kv_outlier_map": bool(self.kv_outlier_map and self.k > 0),
            "kv_int8_scale": bool(self.kv_int8_scale),
        }

    def tail_bf16(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """bf16 residual rows (sink ∪ recent tail): ``(k_tail, v_tail, n_rows)``.

        ``k_tail`` / ``v_tail`` are ``[1, n_rows, H, D]`` (single request).
        """
        idx = self.window.tail_indices(self.seq_len)
        if self.kv_ring:
            idx = [self._ring_slot_host(p) for p in idx]
        if idx:
            ti = torch.tensor(idx, dtype=torch.long, device=self.device)
            kt = self.k_hist.index_select(0, ti).unsqueeze(0).contiguous()
            vt = self.v_hist.index_select(0, ti).unsqueeze(0).contiguous()
        else:
            kt = torch.zeros(1, 0, self.H, self.D, dtype=torch.bfloat16,
                             device=self.device)
            vt = kt.clone()
        return kt, vt, len(idx)

    def decode_inputs(self) -> dict:
        """Full arg bundle for ``ommx_paged_decode`` at the current sequence state."""
        planes = self.decode_planes()
        kt, vt, n_tail = self.tail_bf16()
        packed_tokens = self.packed_groups * self.gt
        return {
            "planes": planes,
            "k_tail": kt, "v_tail": vt,
            "b_tail_len": torch.tensor([n_tail], dtype=torch.int32, device=self.device),
            "b_seq_len": torch.tensor([packed_tokens], dtype=torch.int32,
                                      device=self.device),
            "packed_start_offset": 0,  # dedicated packed planes start at rel pos 0
            "max_seq_len": max(1, packed_tokens),
            "max_tail_len": self.window.max_tail_len(self.max_seq_len),
        }


__all__ = ["CanonicalKVStore"]
