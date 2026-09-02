# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Self-contained numeric codecs for the canonical paged-decode attention path.

Inlined (no parent-repo import) from ``ommx.quant``:

  * FP4 E2M1F encode/decode + pow2 scale helper      (from ``ommx.quant._fp4``)
  * relidx7 7-bit LSB-first index codec              (from ``ommx.quant.packer``)
  * combinadic k-subset rank codec + on-chip bitmap  (``packer`` + ``combinadic_bitmap``)
  * sparse-correction CSR delta builder              (from ``ommx.quant.sparse_correction``)

These are the SINGLE source of truth the kernel decode mirrors. Pure-python /
torch-CPU; importable without a GPU and without triton. The encoders/decoders
are bit-for-bit identical to the validated OMMX serving path.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch

# ════════════════════════════════════════════════════════════════════════════
# FP4 E2M1F  (code = (sign<<3) | (exp<<1) | mant)  —  from ommx.quant._fp4
# ════════════════════════════════════════════════════════════════════════════

_FP4_E2M1F_BOUNDARIES: dict = {}


def pow2_scale_exp(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute ``round(log2(x))`` as an int32 exponent (kernel-fed pow2 scale)."""
    x = x.clamp_min(float(eps))
    return torch.round(torch.log2(x)).to(torch.int32)


def _fp4_e2m1f_boundaries(device: torch.device) -> torch.Tensor:
    """Midpoint boundaries for FP4 E2M1F finite magnitudes [0,.5,1,1.5,2,3,4,6]."""
    key = (str(device.type), device.index)
    cached = _FP4_E2M1F_BOUNDARIES.get(key)
    if cached is None:
        cached = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
            device=device, dtype=torch.float32,
        )
        _FP4_E2M1F_BOUNDARIES[key] = cached
    return cached


def encode_fp4_e2m1f(x: torch.Tensor) -> torch.Tensor:
    """Encode a float tensor to FP4 E2M1F finite-only codes (uint8 low 4 bits)."""
    x_f = x.to(torch.float32)
    sign = (x_f < 0).to(torch.uint8)
    ax = x_f.abs()
    idx = torch.bucketize(ax, _fp4_e2m1f_boundaries(x.device), right=False).to(torch.uint8)
    exp = ((idx >= 2).to(torch.uint8) + (idx >= 4).to(torch.uint8) + (idx >= 6).to(torch.uint8))
    mant = idx & 1
    return ((sign << 3) | (exp << 1) | mant).to(torch.uint8)


def decode_fp4_e2m1f(code: torch.Tensor) -> torch.Tensor:
    """Decode FP4 E2M1F codes (uint8 low 4 bits) to float32 magnitudes (signed)."""
    code_i = code.to(torch.int32) & 0xF
    sign = (code_i >> 3) & 1
    exp = (code_i >> 1) & 0x3
    mant = code_i & 1
    mant_f = mant.to(torch.float32)
    exp_i = exp.to(torch.int32)
    v_sub = 0.5 * mant_f
    v_norm = (1.0 + 0.5 * mant_f) * torch.ldexp(
        torch.ones_like(mant_f, dtype=torch.float32), exp_i - 1)
    v = torch.where(exp_i == 0, v_sub, v_norm)
    return torch.where(sign.to(torch.bool), -v, v)


# ════════════════════════════════════════════════════════════════════════════
# relidx7  —  7-bit LSB-first local-column codec  (from ommx.quant.packer)
# ════════════════════════════════════════════════════════════════════════════

RELIDX_BITS = 7
RELIDX_SENTINEL = 0x7F   # 127 — unused-slot marker (only distinguishable by count)


def pack_relidx7(columns) -> bytes:
    """Pack a flat sequence of 7-bit local columns into a LSB-first bitstream.

    Each value occupies ``RELIDX_BITS`` (7) bits, emitted least-significant-first,
    packed into bytes little-endian. Output length ``ceil(len*7/8)`` bytes.
    """
    seq = columns.tolist() if hasattr(columns, "tolist") else list(columns)
    total_bits = len(seq) * RELIDX_BITS
    out = bytearray((total_bits + 7) // 8)
    bitpos = 0
    for v in seq:
        v = int(v) & ((1 << RELIDX_BITS) - 1)
        for b in range(RELIDX_BITS):
            if (v >> b) & 1:
                out[bitpos >> 3] |= 1 << (bitpos & 7)
            bitpos += 1
    return bytes(out)


def unpack_relidx7(blob: bytes, count: int) -> List[int]:
    """Inverse of :func:`pack_relidx7` — recover ``count`` 7-bit values."""
    out = []
    bitpos = 0
    for _ in range(int(count)):
        v = 0
        for b in range(RELIDX_BITS):
            if (blob[bitpos >> 3] >> (bitpos & 7)) & 1:
                v |= 1 << b
            bitpos += 1
        out.append(v)
    return out


# ════════════════════════════════════════════════════════════════════════════
# combinadic  —  k-subset rank codec (pure big-int)  (from ommx.quant.packer)
# ════════════════════════════════════════════════════════════════════════════
# A k-subset of the tile columns is one combinadic rank in [0, C(n,k)).
# ceil(log2 C(n,k)) bits is the information-theoretic floor. The K-outlier
# storage-floor format: it is membership-equivalent to relidx7 at the same k.

def combinadic_index_bits(outliers_per_vector: int, tile_cols: int = 128) -> int:
    """Field width in BITS for one combinadic rank = ceil(log2 C(tile_cols, k))."""
    k = int(outliers_per_vector)
    n = int(tile_cols)
    if k <= 0 or k >= n:
        return 0
    total = math.comb(n, k)
    return max(1, (total - 1).bit_length())


def combinadic_index_bytes(outliers_per_vector: int, tile_cols: int = 128) -> int:
    """Minimal byte count for one combinadic rank field."""
    return (combinadic_index_bits(outliers_per_vector, tile_cols) + 7) // 8


def combinadic_encode(columns, tile_cols: int = 128) -> int:
    """Encode a k-subset of {0..tile_cols-1} → combinadic rank (big-int).

    rank = Σ_{j} C(c_j, k-j) with c descending. Order of ``columns`` irrelevant.
    """
    cols = sorted({int(c) for c in columns}, reverse=True)
    k = len(cols)
    if any(c < 0 or c >= int(tile_cols) for c in cols):
        raise ValueError(f"columns must lie in [0,{tile_cols}); got {cols}")
    rank = 0
    for j, c in enumerate(cols):
        rank += math.comb(c, k - j)
    return rank


def combinadic_decode(rank: int, outliers_per_vector: int, tile_cols: int = 128) -> list:
    """Unrank a combinadic ``rank`` → the k columns, ascending. Greedy inverse."""
    k = int(outliers_per_vector)
    n = int(tile_cols)
    if k <= 0:
        return []
    rem = int(rank)
    out = []
    for i in range(k, 0, -1):
        c = n - 1
        while c >= i - 1 and math.comb(c, i) > rem:
            c -= 1
        if c < i - 1:
            raise ValueError(f"invalid combinadic rank={rank} for k={k}, tile={n}")
        rem -= math.comb(c, i)
        out.append(c)
    if rem != 0:
        raise ValueError(f"residual {rem} after unrank (rank={rank})")
    out.sort()
    return out


# ── combinadic KV-outlier blob (from ommx.quant.combinadic_bitmap) ───────────
FP4_BITS = 4


def _as_int_list(x) -> List[int]:
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [int(v) for v in x]


def kv_storage_bits(k: int, vl: int = 32) -> Tuple[int, int]:
    """(relidx7_bits, combinadic_bits) per vector, INCLUDING the 4·k FP4 values.

    relidx7    = (RELIDX_BITS + FP4_BITS) · k
    combinadic = combinadic_index_bits(k, vl) + FP4_BITS · k
    """
    val_bits = FP4_BITS * int(k)
    relidx = RELIDX_BITS * int(k) + val_bits
    combo = combinadic_index_bits(int(k), int(vl)) + val_bits
    return relidx, combo


def pack_combinadic_kv_outliers(idx_per_vector, val_per_vector, k: int, vl: int = 32):
    """Pack one vector's (positions, FP4 codes) → compact combinadic blob.

    Returns ``{"rank": int, "fp4": [...], "k": k, "vl": vl}`` with ``fp4`` ordered
    by ASCENDING position (aligns with the unranked set).
    """
    k = int(k); vl = int(vl)
    idx = _as_int_list(idx_per_vector)
    val = _as_int_list(val_per_vector)
    if len(idx) != k or len(val) != k:
        raise ValueError(f"idx/val must have length k={k}; got {len(idx)}/{len(val)}")
    if k == 0:
        return {"rank": 0, "fp4": [], "k": 0, "vl": vl}
    if any(p < 0 or p >= vl for p in idx):
        raise ValueError(f"positions must lie in [0,{vl}); got {idx}")
    if len(set(idx)) != k:
        raise ValueError(f"positions must be distinct; got {idx}")
    order = sorted(range(k), key=lambda s: idx[s])
    cols = [idx[s] for s in order]
    fp4 = [val[s] & ((1 << FP4_BITS) - 1) for s in order]
    rank = 0 if k >= vl else combinadic_encode(cols, vl)
    return {"rank": int(rank), "fp4": fp4, "k": k, "vl": vl}


def unpack_combinadic_kv_outliers(blob) -> Tuple[List[int], List[int]]:
    """Inverse of :func:`pack_combinadic_kv_outliers` → (positions, fp4 values)."""
    k = int(blob["k"]); vl = int(blob["vl"])
    if k == 0:
        return [], []
    positions = list(range(vl)) if k >= vl else combinadic_decode(int(blob["rank"]), k, vl)
    fp4 = [int(v) & ((1 << FP4_BITS) - 1) for v in blob["fp4"]]
    return positions, fp4


def combinadic_to_bitmap(rank: int, k: int, vl: int = 32) -> int:
    """Unrank a combinadic ``rank`` → a vl-bit occupancy mask (uint32).

    The EXACT op the GPU kernel performs once per tile: after this, membership is
    a single bit-test, not an O(k) search. ``k<=0`` → 0; ``k>=vl`` → all-ones.
    """
    k = int(k); vl = int(vl)
    if vl > 32:
        raise ValueError(f"combinadic_to_bitmap targets a 32-bit mask; vl={vl}>32")
    if k <= 0:
        return 0
    if k >= vl:
        return (1 << vl) - 1
    mask = 0
    for p in combinadic_decode(int(rank), k, vl):
        mask |= 1 << int(p)
    return mask & 0xFFFFFFFF


def bitmap_to_positions(bitmap: int, vl: int = 32) -> List[int]:
    """Ascending outlier positions encoded in a vl-bit ``bitmap`` (test helper)."""
    return [p for p in range(int(vl)) if (int(bitmap) >> p) & 1]


# ════════════════════════════════════════════════════════════════════════════
# bitmap outlier index  (per-row 1-bit-per-position, O(1) popcount-rank decode)
# ════════════════════════════════════════════════════════════════════════════
# At HIGH outlier density (e.g. W @ 25% -> npv=32 of 128) the relidx7 stream costs
# 7 bit/outlier (= 1.75 bit/weight at 25%) and the in-kernel membership test is an
# O(k) search; the combinadic rank is the storage floor but its in-kernel unrank is
# O(k) dependent LUT loads. A BITMAP — one bit per K position per output row — is
# 1.0 bit/weight FLAT (independent of density) and membership/rank is a SINGLE
# popcount: at 25% density 1.0 < 1.75 (relidx7) so the bitmap wins (the 1/7 ≈ 14.3%
# crossover). Decode: bit set -> the position is an outlier; its FP4-nibble index is
# the popcount of all lower set bits (its ascending ordinal). Bit-exact membership-
# equivalent to relidx7/combinadic; the nibble stream is shared (ascending-position
# order). law #15: ONE wide-load + masked prefix-popcount, no per-position loop.

BITMAP_BITS = 1


def bitmap_index_bytes(n_positions: int) -> int:
    """Bytes for a per-row bitmap over ``n_positions`` columns = ceil(n_positions/8)."""
    return (int(n_positions) + 7) // 8


def pack_bitmap_row(columns, n_positions: int) -> bytes:
    """Pack a set of outlier column indices into a 1-bit-per-position row bitmap.

    ``columns`` are the ABSOLUTE positions (0..n_positions-1) of the outliers in one
    output row; the result is ``ceil(n_positions/8)`` bytes, LSB-first within each byte
    (position p -> byte p>>3, bit p&7). Duplicate/out-of-range columns are ignored.
    O(1) popcount membership: the kernel/loader tests ``(byte>>(p&7))&1`` and the
    nibble ordinal of a set bit p is ``popcount(all set bits < p)``.
    """
    seq = columns.tolist() if hasattr(columns, "tolist") else list(columns)
    out = bytearray(bitmap_index_bytes(n_positions))
    for c in seq:
        c = int(c)
        if 0 <= c < int(n_positions):
            out[c >> 3] |= 1 << (c & 7)
    return bytes(out)


def unpack_bitmap_row(blob: bytes, n_positions: int) -> List[int]:
    """Inverse of :func:`pack_bitmap_row` — ascending outlier positions in one row.

    Returns the positions whose bit is set, ascending (== the FP4-nibble order). The
    s-th returned position has nibble ordinal ``s`` (its popcount-rank).
    """
    out = []
    for p in range(int(n_positions)):
        if (blob[p >> 3] >> (p & 7)) & 1:
            out.append(p)
    return out


def bitmap_rows_to_positions(bitmap_u8, n_positions: int):
    """Vectorized batch unpack: uint8 ``[R, ceil(n/8)]`` -> bool occupancy ``[R, n]``.

    The loader-side O(1) decode: expand the packed bitmap to a per-position boolean
    mask (one wide bit-unpack, NO per-position python loop), so positions = nonzero
    and each set bit's nibble ordinal = its prefix popcount (``cumsum(mask)-1``).
    """
    bm = bitmap_u8.to(torch.int32)                                 # [R, FB]
    R = int(bm.shape[0])
    fb = int(bm.shape[-1])
    bits = (bm.unsqueeze(-1) >> torch.arange(8, dtype=torch.int32, device=bm.device)) & 1
    return bits.reshape(R, fb * 8)[:, : int(n_positions)].to(torch.bool)   # [R, n]


def pack_bitmap_rows(columns, n_positions: int):
    """Vectorized batch pack: positions ``[..., k]`` int -> uint8 bitmap ``[..., ceil(n/8)]``.

    The tensor twin of :func:`pack_bitmap_row` — the form the KV packer needs, where one
    "row" is one (group, head, channel) frame and there are G*H*D of them. Position ``p``
    lands in byte ``p>>3`` at bit ``p&7`` (LSB-first, IDENTICAL byte layout to
    ``pack_bitmap_row``; that equality is a shipped gate in tests/test_bitmap_outlier.py).

    DEVICE-CLEAN: every arange/constant is built on the INPUT tensor's device and the whole
    body is integer bit-twiddling, so a CUDA-resident pack is BIT-IDENTICAL to the CPU one
    (the same property ``pack.py::_pack_relidx7_frames`` relies on for the device-side
    regroup). NO python per-position loop (law #15: one wide op, not a per-slot scan).

    LOUD, NEVER SILENT: the bitmap has no way to express "this slot is empty" and no way
    to express "this position appears twice" — a duplicate would collapse to ONE set bit
    while the shared ``k_oval`` nibble stream still carries k entries, silently shifting
    every popcount-rank after it. So an out-of-range, negative, or duplicated position
    RAISES here rather than producing a frame that decodes to the wrong values. (relidx7
    tolerates a duplicate because its splice is first-match-wins per slot; the bitmap's
    rank decode does not, which is exactly why this check exists.)
    """
    cols = columns.to(torch.int64)
    n = int(n_positions)
    fb = bitmap_index_bytes(n)
    dev = cols.device
    k = int(cols.shape[-1])
    if k == 0:
        return torch.zeros(*cols.shape[:-1], fb, dtype=torch.uint8, device=dev)
    if bool((cols < 0).any()) or bool((cols >= n).any()):
        raise ValueError(
            f"bitmap positions must lie in [0,{n}); got min={int(cols.min())} "
            f"max={int(cols.max())}. A negative entry is the packer's empty-slot "
            "sentinel, which the flat bitmask cannot represent.")
    # occupancy over the FULL byte-padded universe, then fold 8 bits per byte.
    occ = torch.zeros(*cols.shape[:-1], fb * 8, dtype=torch.int64, device=dev)
    occ.scatter_(-1, cols, 1)                       # duplicates collapse (idempotent)
    if bool((occ.sum(-1) != k).any()):
        raise ValueError(
            "bitmap frame lost a position: the k outlier columns of a frame must be "
            "DISTINCT (a duplicate sets one bit but leaves k nibbles in k_oval, which "
            "would shift every popcount-rank after it).")
    byte_w = (1 << torch.arange(8, dtype=torch.int64, device=dev))
    return (occ.reshape(*cols.shape[:-1], fb, 8) * byte_w).sum(-1).to(torch.uint8)


def unpack_bitmap_rows(bitmap_u8, n_positions: int, k: int):
    """Inverse of :func:`pack_bitmap_rows`: uint8 ``[..., ceil(n/8)]`` -> int64 ``[..., k]``.

    Returns the ASCENDING set positions of each row — i.e. exactly the order the shared
    ``k_oval`` FP4 nibble stream is written in, so slot ``s`` of the value stream belongs
    to position ``out[..., s]``. That identity IS the rank rule the kernel exploits
    (a set bit's nibble ordinal == the popcount of all lower set bits).

    A row with fewer than ``k`` set bits RAISES: it means the stored bitmap and the stored
    value stream disagree about how many outliers this frame has, and decoding it would
    quietly read a neighbouring outlier's nibble.
    """
    n = int(n_positions)
    k = int(k)
    bm = bitmap_u8.to(torch.int64)
    fb = int(bm.shape[-1])
    dev = bm.device
    if k == 0:
        return torch.zeros(*bm.shape[:-1], 0, dtype=torch.int64, device=dev)
    bits = (bm.unsqueeze(-1) >> torch.arange(8, dtype=torch.int64, device=dev)) & 1
    occ = bits.reshape(*bm.shape[:-1], fb * 8)[..., :n].to(torch.bool)   # [..., n]
    idx = torch.arange(n, dtype=torch.int64, device=dev).expand_as(occ)
    # ascending set positions first, unset lanes pushed to the sentinel n (same
    # "sort a masked column index" idiom pack.py uses for the signed outlier select).
    key = torch.where(occ, idx, torch.full_like(idx, n)).sort(-1)[0][..., :k]
    if bool((key >= n).any()):
        raise ValueError(
            f"bitmap frame has fewer than k={k} set bits (popcount mismatch): the "
            "stored bitmap and the k_oval nibble stream disagree on the outlier count.")
    return key


# ════════════════════════════════════════════════════════════════════════════
# sparse-correction CSR delta  (from ommx.quant.sparse_correction)
# ════════════════════════════════════════════════════════════════════════════
# The outlier "decouple": run the base INT2 attention with NO outlier overlay and
# apply the FP4 outliers as a separate sparse correction. This helper builds the
# per-column K-bitmap (load-derive) the O(1) decode mirror consumes — the resident
# stream stays the compact CSR; the bitmap is a warmup-derived runtime plane.

def csr_to_correction_bitmap(col_ptr, col_k, K):
    """CSR-by-column ``(col_ptr, col_k)`` → per-column K-BITMAP ``(row_ptr, bitmap)``.

    ``row_ptr`` == ``col_ptr`` unchanged; ``bitmap`` uint32 ``[N, ceil(K/32)]`` with
    bit ``col_k[j]&31`` of word ``col_k[j]>>5`` set per outlier j. ``col_k`` must be
    ascending within each column. BIT-EXACT vs the CSR decode (same delta order).
    """
    cp = col_ptr.detach().to("cpu").to(torch.int32).contiguous()
    ck = col_k.detach().to("cpu").to(torch.int64).contiguous()
    N = int(cp.numel()) - 1
    n_words = (int(K) + 31) // 32
    bitmap = torch.zeros((N, n_words), dtype=torch.int64)
    if int(ck.numel()) > 0:
        counts = (cp[1:] - cp[:-1]).to(torch.int64)
        col_of = torch.repeat_interleave(torch.arange(N, dtype=torch.int64), counts)
        word = ck >> 5
        bit = (ck & 31).to(torch.int64)
        flat = col_of * n_words + word
        bitmap.view(-1).index_add_(0, flat, (torch.ones_like(bit) << bit))
    bitmap = (bitmap & 0xFFFFFFFF).to(torch.int32).contiguous()
    return cp, bitmap


__all__ = [
    # FP4
    "pow2_scale_exp", "encode_fp4_e2m1f", "decode_fp4_e2m1f",
    # relidx7
    "RELIDX_BITS", "RELIDX_SENTINEL", "pack_relidx7", "unpack_relidx7",
    # combinadic
    "FP4_BITS", "combinadic_index_bits", "combinadic_index_bytes",
    "combinadic_encode", "combinadic_decode", "kv_storage_bits",
    "pack_combinadic_kv_outliers", "unpack_combinadic_kv_outliers",
    "combinadic_to_bitmap", "bitmap_to_positions",
    # bitmap outlier index (per-row 1-bit-per-position, popcount-rank)
    "BITMAP_BITS", "bitmap_index_bytes", "pack_bitmap_row", "unpack_bitmap_row",
    "bitmap_rows_to_positions", "pack_bitmap_rows", "unpack_bitmap_rows",
    # sparse correction
    "csr_to_correction_bitmap",
]
