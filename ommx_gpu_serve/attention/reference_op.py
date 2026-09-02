# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference custom ops for the canonical OMMX paged-decode attention.

This is the FOUNDATION every framework adapter wraps. It exposes the two canonical
primitives as ``torch.library`` custom ops so they are:

  * traceable / capturable — vLLM v1 (and ``torch.compile``) need a registered op
    with a fake/meta impl to infer shapes during graph capture without launching
    the Triton kernel;
  * graph-safe — the decode op takes ``max_seq_len`` / ``max_tail_len`` as plain
    ints so the underlying kernel never does a device->host ``.item()`` sync;
  * in-place — ``o`` / ``lse`` are caller-preallocated and mutated (declared in
    ``mutates_args``), matching the canonical kernel's out-variant contract.

Two ops:
  * ``ommx_gpu_serve::paged_decode``            — primitive (1): split-KV quantized
    paged decode over the canonical KV planes (writes ``o``/``lse`` in place).
  * ``ommx_gpu_serve::merge_attention_states``  — primitive (2): FlashInfer-style
    cross-segment (o, lse) merge (cascade / radix / context-parallel / chunked).

Plus thin python wrappers (``ommx_paged_decode`` / ``merge_attention_states``) that
preallocate the outputs and accept the pack-dict directly, for PyTorch-only users.

CPU-importable: importing this module REGISTERS the ops (no triton / GPU needed);
the Triton kernel only builds on the first real GPU call (fake-mode tracing and the
parity tests' meta path never touch it).
"""
from __future__ import annotations

from typing import Optional

import torch

from .paged_decode import (
    merge_attention_states as _merge_attention_states_kernel,
)
from .paged_decode import (
    ommx_paged_decode_attention_canonical as _canonical,
)

_OP_NS = "ommx_gpu_serve"


# ════════════════════════════════════════════════════════════════════════════
# primitive (1) — paged decode (in-place out-variant)
# ════════════════════════════════════════════════════════════════════════════

@torch.library.custom_op(
    f"{_OP_NS}::paged_decode", mutates_args=("o", "lse"), device_types="cuda"
)
def paged_decode(
    q: torch.Tensor,
    k_base: torch.Tensor,
    k_scale: torch.Tensor,
    k_zp: torch.Tensor,
    k_oidx: Optional[torch.Tensor],
    k_oval: Optional[torch.Tensor],
    k_crank: Optional[torch.Tensor],
    k_obmp: Optional[torch.Tensor],
    k_fp4_mapscale: Optional[torch.Tensor],
    k_fp4_mapcenter: Optional[torch.Tensor],
    v_main: torch.Tensor,
    v_scale: torch.Tensor,
    v_zp: torch.Tensor,
    k_tail: torch.Tensor,
    v_tail: torch.Tensor,
    b_tail_len: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    req_to_token: torch.Tensor,
    req_to_group: torch.Tensor,
    b_seq_len: torch.Tensor,
    sm_scale: float,
    page_size: int,
    k_outliers_per_vector: int,
    k_format: str,
    combinadic_read: bool,
    bitmap_read: bool,
    kv_outlier_map: bool,
    kv_int8_scale: bool,
    max_seq_len: int,
    max_tail_len: int,
    packed_start_offset: int,
) -> None:
    """Canonical split-KV quantized paged decode. Writes ``o``/``lse`` in place.

    Thin wrapper over ``ommx_paged_decode_attention_canonical``; see that function
    for the full KV-plane ABI. ``max_seq_len`` / ``max_tail_len`` are passed through
    so the kernel never syncs the device seq-len buffers (graph-capture safe).
    ``kv_outlier_map`` enables the dedicated weight-space FP4 outlier map (requires
    ``k_fp4_mapscale``/``k_fp4_mapcenter``); ``kv_int8_scale`` decodes int8 pow2-exp
    K/V scale (also auto-detected from the plane dtype).
    """
    _canonical(
        q,
        k_base, k_scale, k_zp, k_oidx, k_oval,
        v_main, v_scale, v_zp,
        k_tail, v_tail, b_tail_len,
        o, lse,
        req_to_token, req_to_group, b_seq_len,
        sm_scale=float(sm_scale),
        page_size=int(page_size),
        k_outliers_per_vector=int(k_outliers_per_vector),
        k_format=str(k_format),
        combinadic_read=bool(combinadic_read),
        k_crank=k_crank,
        bitmap_read=bool(bitmap_read),
        k_obmp=k_obmp,
        k_fp4_mapscale=k_fp4_mapscale,
        k_fp4_mapcenter=k_fp4_mapcenter,
        kv_outlier_map=bool(kv_outlier_map),
        kv_int8_scale=bool(kv_int8_scale),
        max_seq_len=int(max_seq_len),
        max_tail_len=int(max_tail_len),
        packed_start_offset=int(packed_start_offset),
    )


@paged_decode.register_fake
def _paged_decode_fake(
    q, k_base, k_scale, k_zp, k_oidx, k_oval, k_crank, k_obmp,
    k_fp4_mapscale, k_fp4_mapcenter,
    v_main, v_scale, v_zp, k_tail, v_tail, b_tail_len,
    o, lse, req_to_token, req_to_group, b_seq_len,
    sm_scale, page_size, k_outliers_per_vector, k_format,
    combinadic_read, bitmap_read, kv_outlier_map, kv_int8_scale,
    max_seq_len, max_tail_len, packed_start_offset,
) -> None:
    # Shape contract only (outputs are in-place; nothing is allocated/returned).
    torch._check(q.dim() == 3, lambda: f"q must be [B, Hq, D]; got {tuple(q.shape)}")
    torch._check(k_base.dim() == 4,
                 lambda: f"k_base must be [pages, page_size, n_kv_heads, w]; got {tuple(k_base.shape)}")
    b, hq, d = q.shape[0], q.shape[1], q.shape[-1]
    n_kv = k_base.shape[-2]
    torch._check(d % 32 == 0, lambda: f"head_dim must be a multiple of 32; got {d}")
    torch._check(hq % n_kv == 0,
                 lambda: f"q heads ({hq}) must be a multiple of n_kv_heads ({n_kv}) for GQA")
    torch._check(o.dim() == 3 and o.shape[0] == b and o.shape[1] == hq,
                 lambda: f"o must be [B, Hq, Dv]; got {tuple(o.shape)} vs q {tuple(q.shape)}")
    torch._check(lse.dim() == 2 and lse.shape[0] == b and lse.shape[1] == hq,
                 lambda: f"lse must be [B, Hq]; got {tuple(lse.shape)}")
    return None


# ════════════════════════════════════════════════════════════════════════════
# primitive (1b) — paged decode SEGMENT partial-emit (in-place out-variant)
# ════════════════════════════════════════════════════════════════════════════

@torch.library.custom_op(
    f"{_OP_NS}::paged_decode_segment", mutates_args=("o", "lse"),
    device_types="cuda",
)
def paged_decode_segment(
    q: torch.Tensor,
    k_base: torch.Tensor,
    k_scale: torch.Tensor,
    k_zp: torch.Tensor,
    k_oidx: Optional[torch.Tensor],
    k_oval: Optional[torch.Tensor],
    k_crank: Optional[torch.Tensor],
    k_obmp: Optional[torch.Tensor],
    k_fp4_mapscale: Optional[torch.Tensor],
    k_fp4_mapcenter: Optional[torch.Tensor],
    v_main: torch.Tensor,
    v_scale: torch.Tensor,
    v_zp: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    req_to_token: torch.Tensor,
    req_to_group: torch.Tensor,
    b_seg_len: torch.Tensor,
    sm_scale: float,
    page_size: int,
    k_outliers_per_vector: int,
    k_format: str,
    combinadic_read: bool,
    bitmap_read: bool,
    kv_outlier_map: bool,
    kv_int8_scale: bool,
    max_seg_len: int,
    seg_group_base: int,
    packed_start_offset: int,
) -> None:
    """Decode ONE group-range KV segment; write its UN-MERGED ``(o, lse)`` in place.

    The PARTIAL-EMIT entry: attend only the packed group-range sub-segment
    ``[seg_group_base, seg_group_base + n_seg_groups)`` of each request (``b_seg_len``
    = the SEGMENT packed token count, group-aligned at the base) and store THIS
    segment's softmax state — ``o`` = Σp·v / Σp, ``lse`` = max + log Σp over the
    segment tokens — NOT a merged final. No bf16 residual tail (a segment is packed-
    only; the residual is composed as its own segment). The caller runs this per
    disjoint segment, stacks the ``(o, lse)`` pairs, and combines them with
    ``ommx_gpu_serve::merge_attention_states`` (cascade / context-parallel / radix-
    split). Reuses the canonical split-KV kernel via the ``seg_group_base`` group
    offset — same inner per-split online-softmax math the full decode uses.
    """
    # zero-length bf16 tail: a segment is packed-only. b_tail_len all-zero +
    # max_tail_len=0 makes the kernel's tail path a no-op (the FUSE_TAIL inline
    # loop runs zero iterations -> tail weight 0); reuse a 1-row dummy buffer.
    b = int(q.shape[0])
    zero_tail_len = torch.zeros(b, dtype=torch.int32, device=q.device)
    dummy_kv = q.new_empty((b, 1, k_base.shape[-2], q.shape[-1]))
    _canonical(
        q,
        k_base, k_scale, k_zp, k_oidx, k_oval,
        v_main, v_scale, v_zp,
        dummy_kv, dummy_kv, zero_tail_len,
        o, lse,
        req_to_token, req_to_group, b_seg_len,
        sm_scale=float(sm_scale),
        page_size=int(page_size),
        k_outliers_per_vector=int(k_outliers_per_vector),
        k_format=str(k_format),
        combinadic_read=bool(combinadic_read),
        k_crank=k_crank,
        bitmap_read=bool(bitmap_read),
        k_obmp=k_obmp,
        k_fp4_mapscale=k_fp4_mapscale,
        k_fp4_mapcenter=k_fp4_mapcenter,
        kv_outlier_map=bool(kv_outlier_map),
        kv_int8_scale=bool(kv_int8_scale),
        max_seq_len=int(max_seg_len),
        max_tail_len=0,
        packed_start_offset=int(packed_start_offset),
        seg_group_base=int(seg_group_base),
    )


@paged_decode_segment.register_fake
def _paged_decode_segment_fake(
    q, k_base, k_scale, k_zp, k_oidx, k_oval, k_crank, k_obmp,
    k_fp4_mapscale, k_fp4_mapcenter,
    v_main, v_scale, v_zp,
    o, lse, req_to_token, req_to_group, b_seg_len,
    sm_scale, page_size, k_outliers_per_vector, k_format,
    combinadic_read, bitmap_read, kv_outlier_map, kv_int8_scale,
    max_seg_len, seg_group_base, packed_start_offset,
) -> None:
    # Shape contract only (outputs are in-place; nothing allocated/returned).
    torch._check(q.dim() == 3, lambda: f"q must be [B, Hq, D]; got {tuple(q.shape)}")
    torch._check(k_base.dim() == 4,
                 lambda: f"k_base must be [pages, page_size, n_kv_heads, w]; got {tuple(k_base.shape)}")
    b, hq, d = q.shape[0], q.shape[1], q.shape[-1]
    n_kv = k_base.shape[-2]
    torch._check(d % 32 == 0, lambda: f"head_dim must be a multiple of 32; got {d}")
    torch._check(hq % n_kv == 0,
                 lambda: f"q heads ({hq}) must be a multiple of n_kv_heads ({n_kv}) for GQA")
    torch._check(seg_group_base >= 0,
                 lambda: f"seg_group_base must be >= 0; got {seg_group_base}")
    torch._check(o.dim() == 3 and o.shape[0] == b and o.shape[1] == hq,
                 lambda: f"o must be [B, Hq, Dv]; got {tuple(o.shape)} vs q {tuple(q.shape)}")
    torch._check(lse.dim() == 2 and lse.shape[0] == b and lse.shape[1] == hq,
                 lambda: f"lse must be [B, Hq]; got {tuple(lse.shape)}")
    return None


# ════════════════════════════════════════════════════════════════════════════
# primitive (2) — cross-segment attention-state merge (functional)
# ════════════════════════════════════════════════════════════════════════════

@torch.library.custom_op(
    f"{_OP_NS}::merge_attention_states", mutates_args=(), device_types="cuda"
)
def merge_states(
    o_parts: torch.Tensor,
    lse_parts: torch.Tensor,
    valid_lens: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge ``P`` partial ``(o_parts [P,R,D], lse_parts [P,R])`` states -> ``(o [R,D], lse [R])``.

    The associative FlashInfer online-softmax reduce used to compose disjoint KV
    segments (cascade / radix shared-prefix splits, context-parallel ranks,
    chunked-prefill chunks). ``valid_lens [R]`` (optional) gives the ragged per-row
    valid part count.
    """
    return _merge_attention_states_kernel(
        o_parts, lse_parts, valid_lens=valid_lens
    )


@merge_states.register_fake
def _merge_states_fake(o_parts, lse_parts, valid_lens):
    torch._check(o_parts.dim() == 3, lambda: f"o_parts must be [P, R, D]; got {tuple(o_parts.shape)}")
    _, r, d = o_parts.shape
    o = o_parts.new_empty(r, d)
    lse = o_parts.new_empty(r, dtype=torch.float32)
    return o, lse


# ════════════════════════════════════════════════════════════════════════════
# python ergonomics — preallocate outputs + accept the pack dict directly
# ════════════════════════════════════════════════════════════════════════════


def _resolve_bitmap_read(planes: dict, override: Optional[bool]) -> bool:
    """Should the kernel read the STORED flat bitmask (``k_obmp``) as its index source?

    DEFAULT = follow the pack: a plane dict written with ``outlier_repr="bitmap"``
    carries ``k_obmp`` and NO ``k_oidx``, so reading the bitmap is the only correct
    decode for it — inferring it here is what makes the repr end-to-end selectable
    rather than a knob the caller must remember twice. Every other pack (relidx7, the
    DEFAULT, and combinadic) infers False and takes the byte-identical path it always
    took. An explicit ``bitmap_read=`` overrides the inference.

    LOUD, NOT SILENT: asking for the bitmap read without a ``k_obmp`` plane raises here
    with the fix named, rather than letting the kernel fall through to a relidx7 plane
    that this pack does not have.
    """
    want = (str(planes.get("outlier_repr", "relidx7")).lower() == "bitmap") \
        if override is None else bool(override)
    if want and int(planes.get("outliers_per_vector", 0)) > 0 \
            and planes.get("k_obmp") is None:
        raise ValueError(
            "bitmap_read requires the k_obmp flat-bitmask plane, which this pack does "
            "not carry (outlier_repr="
            f"{str(planes.get('outlier_repr', 'relidx7'))!r}). Fix: pack with "
            "ommx_pack_kv_canonical_block(..., outlier_repr='bitmap') or pass "
            "bitmap_read=False.")
    return want


def ommx_paged_decode(
    q: torch.Tensor,
    planes: dict,
    k_tail: torch.Tensor,
    v_tail: torch.Tensor,
    b_tail_len: torch.Tensor,
    b_seq_len: torch.Tensor,
    *,
    sm_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    lse_out: Optional[torch.Tensor] = None,
    max_seq_len: Optional[int] = None,
    max_tail_len: Optional[int] = None,
    combinadic_read: bool = False,
    bitmap_read: Optional[bool] = None,
    packed_start_offset: int = 0,
    req_to_token: Optional[torch.Tensor] = None,
    req_to_group: Optional[torch.Tensor] = None,
):
    """Ergonomic wrapper: run the canonical decode op over a pack-dict ``planes``.

    ``planes`` is the dict returned by ``ommx_pack_kv_canonical_block`` (or a
    serving KV store). Preallocates ``o`` / ``lse`` (unless given), calls the
    ``ommx_gpu_serve::paged_decode`` op, and returns ``(o, lse)``.

    ``req_to_token`` / ``req_to_group`` default to the pack-dict's identity tables
    (single-request block); serving callers pass their real page/group tables.
    ``max_seq_len`` / ``max_tail_len`` default to ``b_seq_len.max()`` /
    ``b_tail_len.max()`` — pass explicit ints for CUDA-graph capture.

    ``bitmap_read`` defaults to ``None`` = infer from the pack's ``outlier_repr``
    (``"bitmap"`` -> read the stored ``k_obmp`` flat bitmask; anything else -> the
    unchanged relidx7/combinadic path). See :func:`_resolve_bitmap_read`.
    """
    b, hq, dv = int(q.shape[0]), int(q.shape[1]), int(q.shape[-1])
    if out is None:
        out = torch.empty((b, hq, dv), dtype=torch.float32, device=q.device)
    if lse_out is None:
        lse_out = torch.empty((b, hq), dtype=torch.float32, device=q.device)
    if req_to_token is None:
        req_to_token = planes["req_to_token"]
    if req_to_group is None:
        req_to_group = planes["req_to_group"]
    if max_seq_len is None:
        max_seq_len = int(b_seq_len.max().item()) if b_seq_len.numel() else 1
    if max_tail_len is None:
        max_tail_len = int(b_tail_len.max().item()) if b_tail_len.numel() else 0

    torch.ops.ommx_gpu_serve.paged_decode(
        q,
        planes["k_base"], planes["k_scale"], planes["k_zp"],
        planes.get("k_oidx"), planes.get("k_oval"), planes.get("k_crank"),
        planes.get("k_obmp"),
        planes.get("k_fp4_mapscale"), planes.get("k_fp4_mapcenter"),
        planes["v_main"], planes["v_scale"], planes["v_zp"],
        k_tail, v_tail, b_tail_len,
        out, lse_out,
        req_to_token, req_to_group, b_seq_len,
        float(sm_scale), int(planes["page_size"]),
        int(planes["outliers_per_vector"]), str(planes["k_format"]),
        bool(combinadic_read),
        _resolve_bitmap_read(planes, bitmap_read),
        bool(planes.get("kv_outlier_map", False)),
        bool(planes.get("kv_int8_scale", False)),
        int(max_seq_len), int(max_tail_len),
        int(packed_start_offset),
    )
    return out, lse_out


def ommx_paged_decode_segment(
    q: torch.Tensor,
    planes: dict,
    b_seg_len: torch.Tensor,
    seg_group_base: int,
    *,
    sm_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    lse_out: Optional[torch.Tensor] = None,
    max_seg_len: Optional[int] = None,
    combinadic_read: bool = False,
    bitmap_read: Optional[bool] = None,
    packed_start_offset: int = 0,
    req_to_token: Optional[torch.Tensor] = None,
    req_to_group: Optional[torch.Tensor] = None,
):
    """Ergonomic wrapper: decode ONE group-range segment, return its ``(o, lse)``.

    Runs ``ommx_gpu_serve::paged_decode_segment`` over the pack-dict ``planes`` for
    the group-range ``[seg_group_base, seg_group_base + ceil(b_seg_len/32))``; returns
    the segment's UN-MERGED ``(o [B,Hq,Dv], lse [B,Hq])``. Stack each segment's pair
    and combine with ``merge_attention_states`` (see integration.common.cascade).
    ``b_seg_len`` is the segment's packed token count (group-aligned at the base).
    ``bitmap_read`` defaults to ``None`` = infer from the pack's ``outlier_repr``.
    """
    b, hq, dv = int(q.shape[0]), int(q.shape[1]), int(q.shape[-1])
    if out is None:
        out = torch.empty((b, hq, dv), dtype=torch.float32, device=q.device)
    if lse_out is None:
        lse_out = torch.empty((b, hq), dtype=torch.float32, device=q.device)
    if req_to_token is None:
        req_to_token = planes["req_to_token"]
    if req_to_group is None:
        req_to_group = planes["req_to_group"]
    if max_seg_len is None:
        max_seg_len = int(b_seg_len.max().item()) if b_seg_len.numel() else 1

    torch.ops.ommx_gpu_serve.paged_decode_segment(
        q,
        planes["k_base"], planes["k_scale"], planes["k_zp"],
        planes.get("k_oidx"), planes.get("k_oval"), planes.get("k_crank"),
        planes.get("k_obmp"),
        planes.get("k_fp4_mapscale"), planes.get("k_fp4_mapcenter"),
        planes["v_main"], planes["v_scale"], planes["v_zp"],
        out, lse_out,
        req_to_token, req_to_group, b_seg_len,
        float(sm_scale), int(planes["page_size"]),
        int(planes["outliers_per_vector"]), str(planes["k_format"]),
        bool(combinadic_read),
        _resolve_bitmap_read(planes, bitmap_read),
        bool(planes.get("kv_outlier_map", False)),
        bool(planes.get("kv_int8_scale", False)),
        int(max_seg_len), int(seg_group_base),
        int(packed_start_offset),
    )
    return out, lse_out


def merge_attention_states(o_parts, lse_parts, *, valid_lens=None):
    """Functional wrapper for the ``ommx_gpu_serve::merge_attention_states`` op."""
    return torch.ops.ommx_gpu_serve.merge_attention_states(
        o_parts, lse_parts, valid_lens
    )


__all__ = [
    "paged_decode",
    "paged_decode_segment",
    "merge_states",
    "ommx_paged_decode",
    "ommx_paged_decode_segment",
    "merge_attention_states",
]
