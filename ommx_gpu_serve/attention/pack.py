# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Canonical KV pack + dequant oracle for the paged-decode attention path.

Self-contained extraction of the CANONICAL relidx7/combinadic path from
``ommx.triton.paged_decode_adapter`` — the production KV format the kernel reads.
Imports only from ``.codec`` (FP4 + relidx7 + combinadic) and torch. NO parent
ommx import, no GPU.

QUANTIZATION (the canonical ABI v1):

  * K — per-CHANNEL axis, vector = (channel d, 32-token group): affine RTN over the
    NON-outlier lanes (``scale=(max-min)/3``, ``zp=min``, ``code=round((w-zp)/scale)``),
    top-k |W| (abs) OR signed top+bottom (signed) outlier select per vector, with
    each outlier's FP4 e2m1f code of the normalized value ``(w-zp)/scale`` stored
    in the relidx7 (or combinadic) sidecar. Outlier dequant = ``fp4·scale + zp``.
  * V — per-TOKEN axis, vector = (token t, 32-channel group): plain affine RTN over
    ALL lanes, no outliers (the spec-compliant ≤3-bit i2 capacity path).

OUTLIER-POSITION STORAGE — three membership-EQUIVALENT encodings of one logical
format (the paper's platform-invariant format claim, item B3). Selected by
``outlier_repr``; all three share the SAME ``k_oval`` FP4 nibble stream in ascending
position order, so they dequantize to bit-identical values:

  * ``relidx7`` (DEFAULT, unchanged) — 7-bit LSB-first local token index per outlier
    slot, ``ceil(7k/8)`` bytes/frame in ``k_oidx``.
  * ``combinadic`` — the information-theoretic floor, one ``ceil(log2 C(vl,k))``-bit
    rank per frame in ``k_crank``.
  * ``bitmap`` — the FLAT BITMASK the ICCAD paper describes for the GPU
    implementation ("positions are stored as a flat bitmask, N bits per group"):
    ``ceil(vl/8)`` bytes/frame in ``k_obmp``, one bit per token position, LSB-first.
    Cost is FLAT in k (1.0 bit/element at any vl that is a multiple of 8) instead of
    relidx7's 7k/vl, so it is CHEAPER than relidx7 above the 1/7 = 14.3% density
    crossover — at the canonical recipe (vl=32, k=6) it is 32 bits/frame vs relidx7's
    48. Decode is a single bit-test for membership plus a prefix-popcount for the
    nibble ordinal (law #15), which is why the kernel wants it.

DROPPED vs the source adapter (refuted/experimental): hmask2 / hmask2_tile128
two-level byte-mask, int8 pow2 scale + int8 zp, use_pow2 magnitude path.
"""
from __future__ import annotations

from typing import Any

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


from .codec import (
    bitmap_index_bytes,
    combinadic_decode,
    combinadic_encode,
    combinadic_index_bytes,
    decode_fp4_e2m1f,
    encode_fp4_e2m1f,
    pack_bitmap_rows,
    pack_relidx7,
    unpack_bitmap_rows,
)

OUTLIER_REPRS = ("relidx7", "combinadic", "bitmap")

CANONICAL_GROUP_TOKENS = 32     # K token-group (one scale per channel per 32 tokens)
CANONICAL_GROUP_CHANNELS = 32   # V channel-group (one scale per token per 32 channels)


# ── plane helpers ────────────────────────────────────────────────────────────

def _pack_2bit_lastdim(codes):
    """int codes 0..3 [..., D] -> uint8 [..., D//4] (col c -> byte c//4, bit 2*(c%4))."""
    c = codes.to(torch.int32) & 0x3
    D = int(c.shape[-1])
    out = torch.zeros(*c.shape[:-1], D // 4, dtype=torch.int32, device=c.device)
    for j in range(4):
        out |= c[..., j::4] << (2 * j)
    return out.to(torch.uint8)


def _unpack_2bit_lastdim(plane, head_dim: int):
    """uint8 [..., D//4] -> int32 codes 0..3 [..., D] (inverse of _pack_2bit_lastdim)."""
    p = plane.to(torch.int32)
    out = torch.empty(*p.shape[:-1], head_dim, dtype=torch.int32)
    for j in range(4):
        out[..., j::4] = (p >> (2 * j)) & 0x3
    return out


def _pack_oval_stream(val, K: int, oval_bits: int):
    """Pack K outlier value codes [D, nblk, K] -> uint8 FP4 nibble stream.

    ``oval_bits == 4`` (FP4 e2m1f): nibble stream, even slot = low nibble,
    ceil(K/2) bytes/frame.
    """
    v = val & 0xF
    if K % 2:
        v = torch.cat([v, torch.zeros(*v.shape[:-1], 1, dtype=v.dtype, device=v.device)], dim=-1)
    return (v[..., 0::2] | (v[..., 1::2] << 4)).to(torch.uint8).contiguous()


def _pack_relidx7_frames(oidx, oval, oval_bits: int = 4):
    """(int16 idx, uint8 val) [D, nblk, K] -> packed uint8 streams.

    Per (channel, group) frame: a 7-bit local-token-index LSB-first bitstream
    (``pack_relidx7`` codec, ceil(7K/8) bytes) + an outlier-value stream. The value
    stream is a 4-bit FP4 nibble stream (``oval_bits=4``, ceil(K/2) bytes, the
    ≤3-bit i2f4/itf4 default). Empty slots (idx<0) are replaced by a
    DUPLICATE of the frame's first real slot (value-neutral under the
    first-match-wins splice); an all-empty frame is not representable (7 bits have
    no sentinel) — the canonical top-k always fills every frame.
    """
    D, nblk, K = (int(s) for s in oidx.shape)
    idx = oidx.to(torch.int64)
    val = oval.to(torch.int64)
    empty = idx < 0
    if bool(empty.any()):
        has_real = (~empty).any(-1)
        if not bool(has_real.all()):
            raise ValueError(
                "relidx7_packed cannot encode an all-empty outlier frame "
                "(7-bit idx has no sentinel code).")
        first = (~empty).to(torch.int64).argmax(-1, keepdim=True)
        idx = torch.where(empty, idx.gather(-1, first).expand_as(idx), idx)
        val = torch.where(empty, val.gather(-1, first).expand_as(val), val)
    idx = idx & 0x7F
    dev = idx.device  # device-clean: keep aranges on the input device so the
    # relidx7 pack runs ON GPU when fed GPU tensors (the device-side regroup path),
    # BIT-IDENTICAL to the CPU pack (integer bit-twiddling is device-invariant).
    bits = ((idx.unsqueeze(-1) >> torch.arange(7, device=dev)) & 1).reshape(D, nblk, K * 7)
    idx_fb = (K * 7 + 7) // 8
    pad = idx_fb * 8 - K * 7
    if pad:
        bits = torch.cat([bits, torch.zeros(D, nblk, pad, dtype=bits.dtype, device=dev)], dim=-1)
    byte_w = (1 << torch.arange(8, dtype=torch.int64, device=dev))
    idx_packed = (bits.reshape(D, nblk, idx_fb, 8) * byte_w).sum(-1).to(torch.uint8)
    val_packed = _pack_oval_stream(val, K, oval_bits)
    return idx_packed.contiguous(), val_packed.contiguous()


def _unpack_relidx7_streams(idx_packed, k: int):
    """uint8 7-bit-packed idx streams [..., ceil(7k/8)] -> int64 [..., k]."""
    lead = idx_packed.shape[:-1]
    fb = int(idx_packed.shape[-1])
    bits = ((idx_packed.unsqueeze(-1).to(torch.int64) >> torch.arange(8, dtype=torch.int64)) & 1)
    bits = bits.reshape(*lead, fb * 8)[..., : k * 7].reshape(*lead, k, 7)
    weights = 1 << torch.arange(7, dtype=torch.int64)
    return (bits * weights).sum(-1)


# ── combinadic outlier sidecar (storage-floor, membership-equiv to relidx7) ───

def _pack_combinadic_frames(oidx, oval, vl: int = CANONICAL_GROUP_TOKENS,
                            oval_bits: int = 4):
    """(int16 idx, uint8 val) [D, nblk, K] -> combinadic sidecar.

    Returns ``(rank_bytes, val_packed, field_bytes)``: one little-endian combinadic
    rank per (channel, group) frame (``field_bytes`` wide) packing the k-subset of
    token positions at the information floor, plus the SAME outlier-value stream
    relidx7 uses (ascending-token order, FP4 nibble for ``oval_bits=4``). Storage-floor
    twin of relidx7 — the kernel unranks the rank to a 32-bit occupancy mask once per
    tile (O(1) membership).
    """
    D, nblk, K = (int(s) for s in oidx.shape)
    field_bytes = combinadic_index_bytes(K, vl)
    idx = oidx.to(torch.int64)
    val = oval.to(torch.int64)
    rank_blob = bytearray()
    for d in range(D):
        for b in range(nblk):
            cols = [int(c) for c in idx[d, b].tolist() if c >= 0]
            rank = combinadic_encode(cols, vl) if cols else 0
            rank_blob += int(rank).to_bytes(field_bytes, "little")
    if field_bytes > 0:
        rank_bytes = torch.frombuffer(bytearray(rank_blob), dtype=torch.uint8).reshape(
            D, nblk, field_bytes).contiguous().clone()
    else:
        rank_bytes = torch.zeros(D, nblk, 0, dtype=torch.uint8)
    val_packed = _pack_oval_stream(val, K, oval_bits)
    return rank_bytes, val_packed, field_bytes


def _unpack_combinadic_frames(rank_bytes, k: int, vl: int = CANONICAL_GROUP_TOKENS):
    """Inverse of :func:`_pack_combinadic_frames`'s index half -> int64 [D, nblk, k]."""
    D, nblk, fb = (int(s) for s in rank_bytes.shape)
    out = torch.full((D, nblk, k), -1, dtype=torch.int64)
    rb = rank_bytes.cpu()
    for d in range(D):
        for b in range(nblk):
            rank = int.from_bytes(bytes(rb[d, b].tolist()), "little") if fb > 0 else 0
            cols = combinadic_decode(rank, k, vl) if k > 0 else []
            for s, c in enumerate(cols):
                out[d, b, s] = int(c)
    return out


# ── bitmap outlier sidecar (the paper's flat N-bit-per-group mask) ───────────

def _pack_bitmap_frames(oidx, oval, vl: int = CANONICAL_GROUP_TOKENS,
                        oval_bits: int = 4):
    """(int16 idx, uint8 val) [D, nblk, K] -> flat-bitmask sidecar.

    Returns ``(bmp_bytes, val_packed, field_bytes)``: one ``ceil(vl/8)``-byte FLAT
    BITMASK per (channel, group) frame — bit ``p`` set iff token position ``p`` of that
    frame is an outlier, LSB-first within each byte — plus the SAME outlier-value stream
    relidx7 and combinadic use (ascending-position order, FP4 nibble for
    ``oval_bits=4``). This is the representation the ICCAD paper attributes to the GPU
    implementation ("positions are stored as a flat bitmask (N bits per group)").

    Why the value stream needs no reordering: ``pack_bitmap_rows`` writes bit ``p`` for
    the p-th outlier column, and the ascending scan of set bits is exactly the ascending
    column order the caller already sorted ``oidx`` into — so slot ``s`` of ``k_oval``
    belongs to the s-th set bit, whose ordinal the kernel recovers as the popcount of
    all lower set bits. That is the whole decode; there is no index arithmetic.

    DEVICE-CLEAN (unlike ``_pack_combinadic_frames``, which is a python big-int loop):
    the pack is pure vectorized bit-twiddling, so it runs on GPU when fed GPU tensors
    and is BIT-IDENTICAL to the CPU pack — the on-device regroup path keeps working.

    An empty slot (``idx < 0``) RAISES: relidx7 pads an empty slot with a duplicate of a
    real one (value-neutral under its first-match-wins splice), but a duplicate in a
    bitmap collapses to one bit while ``k_oval`` still carries k nibbles, which would
    shift every popcount-rank after it. The canonical top-k always fills every frame.
    """
    D, nblk, K = (int(s) for s in oidx.shape)
    field_bytes = bitmap_index_bytes(vl)
    idx = oidx.to(torch.int64)
    if K > 0 and bool((idx < 0).any()):
        raise ValueError(
            "outlier_repr='bitmap' cannot encode an empty outlier slot (a flat bitmask "
            "has no sentinel and a duplicated position would desynchronize the k_oval "
            "popcount-rank). The canonical top-k select always fills every frame.")
    bmp = pack_bitmap_rows(idx, vl) if K > 0 else torch.zeros(
        D, nblk, field_bytes, dtype=torch.uint8, device=idx.device)
    val_packed = _pack_oval_stream(oval.to(torch.int64), K, oval_bits)
    return bmp.contiguous(), val_packed.contiguous(), field_bytes


def _unpack_bitmap_frames(bmp_bytes, k: int, vl: int = CANONICAL_GROUP_TOKENS):
    """Inverse of :func:`_pack_bitmap_frames`'s index half -> int64 [D, nblk, k].

    The ascending set positions of each frame; position ``out[..., s]`` owns nibble
    ``s`` of ``k_oval`` (the popcount-rank identity the kernel decode mirrors).
    """
    if k <= 0:
        return torch.zeros(*bmp_bytes.shape[:-1], 0, dtype=torch.int64)
    return unpack_bitmap_rows(bmp_bytes, vl, k)


# ════════════════════════════════════════════════════════════════════════════
# canonical KV pack
# ════════════════════════════════════════════════════════════════════════════

def ommx_pack_kv_canonical_block(
    K, V,
    *,
    outliers_per_vector: int = 3,
    scale_dtype: Any = None,
    page_size: int = 16,
    device: Any = None,
    outlier_repr: str = "relidx7",
    outlier_select: str = "abs",
    k_format: str = "i2f4",
    use_pow2: bool = False,
    kv_outlier_map: Any = None,
    kv_int8_scale: Any = None,
    group_tokens: int = CANONICAL_GROUP_TOKENS,
    group_channels: int = CANONICAL_GROUP_CHANNELS,
    v_format: str = "i2",
):
    """Pack a bf16/fp32 K/V block of EXACTLY 32·n tokens into the canonical planes.

    Inputs ``K`` / ``V``: ``[num_tokens, n_kv_heads, head_dim]`` (num_tokens a
    multiple of 32 AND of ``page_size``; head_dim a multiple of 32).

      * K — per-CHANNEL affine + FP outliers (signed|abs select).
      * V — per-TOKEN affine RTN, no outliers (the ≤3-bit i2 capacity path).

    ``k_format`` selects the K base width + outlier value codec:

      * ``"i2f4"`` / ``"itf4"`` (default) — INT2 base (2-bit, ``D//4`` bytes) + FP4
        e2m1f outlier (4-bit nibble). The spec-compliant ≤3-bit path.

    ``outlier_repr`` ∈ {"relidx7" (default), "combinadic", "bitmap"} chooses the outlier
    INDEX storage plane — ``k_oidx`` (7k bits/frame), ``k_crank`` (the combinadic floor)
    or ``k_obmp`` (the paper's flat ceil(vl/8)-byte bitmask, 1.0 bit/element flat in k).
    All three are membership-equivalent and decode bit-identically; only the index plane
    and its size change, never the values. ``outlier_select``
    ∈ {"abs" (top-|W|), "signed" (top + bottom tail)} chooses the saliency mask.
    Returns the plane dict of the kernel ABI plus identity ``req_to_token`` /
    ``req_to_group`` tables for a single-request block.

    THE LOW-BIT KV LEVER (``group_tokens=64`` + ``group_channels=64`` + ``kv_outlier_map=False``,
    with ``use_pow2=True`` so the scale is an int8 pow2 exponent). It is a DIFFERENT recipe
    from the canonical published one (gt=gc=32, map ON) and has no accuracy number here:
      * gt=64 halves the per-K-weight cost of both per-group params against gt=32:
        int8 scale 8/64 = 0.125 b/wt, bf16 zp 16/64 = 0.25 b/wt.
      * ``kv_outlier_map=False`` (env ``OMMX_KV_OUTLIER_MAP=0``) takes the BASE-SHARED outlier
        path (``norm=(w-zp)/scale``; the FP4 code rides the base scale/zp at dequant, NO
        per-group mapscale/mapcenter), dropping the dedicated map's two bf16 planes
        (2 x 16/64 = 0.5 b/wt off K).
      * Net at the canonical ``outliers_per_vector=6``, relidx7 — the exact per-plane terms
        ``integration.vllm.packed_only.kv_bits_breakdown`` returns for this recipe, which is
        the authority for every bit figure in this repo (head_dim cancels)::

            K = 2 (k_base) + 0.125 (k_scale) + 0.25 (k_zp)
                + 0.75 (k_oidx, relidx7 6 B/group) + 0.375 (k_oval, FP4 3 B/group)
              = 3.500 bit/weight
            V = 2 (v_main) + 0.125 (v_scale) + 0.25 (v_zp)   = 2.375 bit/weight
            K+V = 5.875 bit per (K,V) PAIR -> 2.938 AVERAGE, 5.45x vs bf16's 32

        ONLY THE PAIR AVERAGE IS UNDER 3. K alone is 3.5 bit/weight on this lever and is
        never 2.9 at any outlier count (k=0 -> 2.375, 1 -> 2.625, 2 -> 2.750, 3 -> 3.000,
        6 -> 3.500), so a "≤3 bit/weight" claim about K is wrong — quote the pair average
        or the per-plane table. Conflating the two is what produced the retracted
        "≤3-bit" headline; ``kv_bits_breakdown`` reports both figures for that reason.
      * ``group_channels=64`` is load-bearing for the V term (V amortizes its scale/zp over
        the CHANNEL group). At the canonical gc=32 with the same gt=64: V = 2.750 and
        K+V = 6.250 bit/pair.
      The base-shared dequant is reproduced bit-exactly by :func:`dequant_kv_canonical`
      (``kv_outlier_map=False`` branch: ``dq_o = fp4·scale + zp``) -> the recipe is its OWN
      fakequant oracle (NOT the dedicated-map recipe; the two are different number systems).
    """
    if device is None:
        device = "cpu"
    if scale_dtype is None:
        scale_dtype = torch.bfloat16
    if str(outlier_repr).lower() not in OUTLIER_REPRS:
        raise ValueError(
            f"outlier_repr must be one of {OUTLIER_REPRS}; got {outlier_repr!r}")
    kfmt = str(k_format).lower()
    if kfmt not in ("i2f4", "itf4"):
        raise ValueError(f"k_format must be i2f4|itf4; got {k_format!r}")
    # KV dedicated FP4 outlier map (weight-space center/span, fakequant-EXACT):
    # gated by ``kv_outlier_map`` (kwarg) or env OMMX_KV_OUTLIER_MAP. Default ON for
    # the FP4 outlier formats (i2f4/itf4).
    # OMMX_RECIPE -> os.environ before either raw read below. Both presets name
    # OMMX_KV_OUTLIER_MAP explicitly (the registry refuses to let a published recipe
    # ride on a default holding still), so a preset that reached config.py but not
    # this packer could pack planes the accounting did not price.
    _resolve_recipe_env()
    if kv_outlier_map is None:
        import os as _os
        _raw = _os.environ.get("OMMX_KV_OUTLIER_MAP")
        kv_outlier_map = (_raw not in {"0", "false", "off", "no"}) if _raw else True
    use_outlier_map = bool(kv_outlier_map)
    # int8 pow2-exp K/V scale: UNCONDITIONAL when ``use_pow2`` — the scale is already
    # 2^e so storing the int8 EXPONENT (8b) instead of the bf16 value (16b) is BIT-EXACT
    # (law #12: 2^exp reproduces the pow2 scale for-bit). zp stays bf16 (an arbitrary zp
    # is NOT 2^e — fp8/int8-exp zp would be lossy). Without ``use_pow2`` the scale is an
    # arbitrary bf16 (not 2^e) so int8-exp rounding WOULD change the value -> forced OFF,
    # no env/knob can enable the lossy path. The ``kv_int8_scale`` knob/env only matters
    # in the pow2 case as an opt-OUT (e.g. ABI compat); default ON since it is free.
    if kv_int8_scale is None:
        import os as _os
        _raw = _os.environ.get("OMMX_KV_INT8_SCALE")
        kv_int8_scale = ((_raw not in {"0", "false", "off", "no"}) if _raw else True)
    use_int8_scale = bool(kv_int8_scale) and bool(use_pow2)
    k_base_bits = 2
    k_code_max = (1 << k_base_bits) - 1                 # 3 (i2f4/itf4)
    oval_bits = 4
    T, H, D = (int(s) for s in K.shape)
    if tuple(V.shape) != (T, H, D):
        raise ValueError(f"K/V shape mismatch: {tuple(K.shape)} vs {tuple(V.shape)}")
    gt, gc = int(group_tokens), int(group_channels)
    if gt not in (16, 32, 64, 128):
        raise ValueError(f"group_tokens (K vector_length) must be in {{16,32,64,128}}; got {gt}")
    if gc not in (16, 32, 64, 128):
        raise ValueError(f"group_channels (V vector_length) must be in {{16,32,64,128}}; got {gc}")
    if T % gt != 0 or T % int(page_size) != 0:
        raise ValueError(f"num_tokens={T} must be a multiple of {gt} and page_size")
    if D % gc != 0 or D % 4 != 0:
        raise ValueError(f"head_dim={D} must be a multiple of {gc}")
    G = T // gt
    NGV = D // gc
    k = int(outliers_per_vector)
    if k < 0 or k > gt:
        raise ValueError(f"outliers_per_vector={k} out of range [0, {gt}]")

    Kf = K.to(torch.float32)
    Vf = V.to(torch.float32)
    # device-clean sentinels: live on the INPUT device so the whole pack (incl. the
    # torch.where masks) runs on GPU when fed GPU tensors (device-side regroup).
    _dev = Kf.device
    big = torch.tensor(1e9, dtype=torch.float32, device=_dev)
    small = torch.tensor(-1e9, dtype=torch.float32, device=_dev)

    # ---- K: channel-major [H, D, G, 32] — affine + FP4 outliers
    gw = Kf.permute(1, 2, 0).contiguous().reshape(H, D, G, gt)
    if k > 0:
        # DEVICE-INVARIANT outlier selection: ``torch.topk`` breaks VALUE TIES in an
        # unspecified, device-DEPENDENT order (CPU and CUDA disagree), and bf16 inputs
        # upcast to fp32 produce exact ties (8-bit mantissa) -> the CPU pack and a GPU
        # pack pick DIFFERENT outlier channels (cos ~0.995, byte-mismatch). Select with
        # a STABLE sort instead (ties keep ascending-position order on BOTH devices) so
        # the device-side regroup is BIT-EXACT to the CPU reference pack. (law #4/#10)
        if str(outlier_select).lower() == "signed":
            # protect the SIGNED top + SIGNED bottom tail of each vector.
            kt = (k + 1) // 2
            kb = k - kt
            top_idx = torch.sort(gw, dim=-1, descending=True, stable=True)[1][..., :kt]
            omask = torch.zeros_like(gw, dtype=torch.bool).scatter_(-1, top_idx, True)
            if kb > 0:
                gmin = torch.where(omask, big, gw)
                bot_idx = torch.sort(gmin, dim=-1, descending=False, stable=True)[1][..., :kb]
                omask = omask.scatter_(-1, bot_idx, True)
            cols = torch.arange(gt, device=gw.device).expand_as(gw)
            topk_idx = torch.where(omask, cols, torch.full_like(cols, gt)).sort(-1)[0][..., :k]
        else:
            topk_idx = torch.sort(gw.abs(), dim=-1, descending=True, stable=True)[1][..., :k]
            omask = torch.zeros_like(gw, dtype=torch.bool).scatter_(-1, topk_idx, True)
    else:
        topk_idx = None
        omask = torch.zeros_like(gw, dtype=torch.bool)
    qmask = ~omask
    min_v = torch.where(qmask, gw, big).min(-1, keepdim=True)[0]
    max_v = torch.where(qmask, gw, small).max(-1, keepdim=True)[0]
    min_v = torch.where(min_v > 1e8, torch.zeros_like(min_v), min_v)
    max_v = torch.where(max_v < -1e8, torch.zeros_like(max_v), max_v)
    scale = (max_v - min_v).clamp_min(1e-8) / float(k_code_max)
    if use_pow2:
        # match fakequant quant_function: scale = 2^round(log2(scale)).
        scale = torch.pow(2.0, torch.round(torch.log2(scale.clamp_min(1e-12))))
    scale = scale.to(scale_dtype).to(torch.float32)
    zp = min_v.to(scale_dtype).to(torch.float32)
    code = torch.round((gw - zp) / scale).clamp(0, k_code_max).to(torch.int32)  # [H,D,G,32]

    k_oidx = k_oval = k_crank = k_obmp = None
    k_field_bytes = 0
    k_fp4_mapscale = k_fp4_mapcenter = None
    if use_outlier_map and k > 0:
        # KV DEDICATED FP4 OUTLIER MAP (weight-space center/span) — BIT-IDENTICAL to
        # fakequant ``quant_function._apply_group_quantization_vectorized``, its
        # ``outlier_method == "fp4"`` branch (cited by symbol: the old line range
        # 357-375 pointed at the base-affine scale block, not at this math). The base affine
        # scale/zp protect the NON-outlier lanes (3 levels at i2); the FP4 outlier is
        # mapped over a DEDICATED [-6, 6] window centered on the OUTLIER subset's own
        # min/max so the e2m1f code's 6.0 ceiling spans the real outlier magnitude
        # (the base-shared norm=(w-zp)/scale clamps an 8.1 outlier to ~6.35; the
        # dedicated map recovers ~7.8 = the true fp4-quant value).
        #   o_center = (o_max + o_min)/2 ; o_range = o_max - o_min
        #   map_scale = 12 / o_range ; mapped = (w - o_center) * map_scale  in [-6,6]
        #   dequant   = decode_fp4(level) / map_scale + o_center
        o_min = torch.where(omask, gw, big).min(-1, keepdim=True)[0]
        o_max = torch.where(omask, gw, small).max(-1, keepdim=True)[0]
        o_min = torch.where(o_min > 1e8, torch.zeros_like(o_min), o_min)
        o_max = torch.where(o_max < -1e8, torch.zeros_like(o_max), o_max)
        o_center = (o_max + o_min) / 2.0
        o_range = (o_max - o_min).clamp_min(1e-8)
        map_scale = 12.0 / o_range                                   # [H,D,G,1]
        # match fakequant's o_center/mapping_scale dtype round-trip (it casts both to
        # the working dtype before the encode -> exact match with the bf16 pack).
        o_center = o_center.to(scale_dtype).to(torch.float32)
        map_scale = map_scale.to(scale_dtype).to(torch.float32)
        mapped = (gw - o_center) * map_scale                          # weight space
        fpc = (encode_fp4_e2m1f(mapped).to(torch.int32) & 0xF)
        # store the per-(channel, group) map params (bf16; int8 deferred per spec).
        k_fp4_mapscale = map_scale.squeeze(-1).permute(2, 0, 1).contiguous().to(scale_dtype)
        k_fp4_mapcenter = o_center.squeeze(-1).permute(2, 0, 1).contiguous().to(scale_dtype)
    else:
        # base-SHARED affine splice (legacy): FP4 e2m1f code of the NORMALIZED value,
        # shared by every outlier slot.
        norm = (gw - zp) / scale
        fpc = (encode_fp4_e2m1f(norm).to(torch.int32) & 0xF)
    if k > 0:
        oidx = topk_idx.sort(-1)[0]                                # [H, D, G, k]
        oval = fpc.gather(-1, oidx)                                # [H, D, G, k]
        # slot-major frames [G, H, D, k] (group slot = leading axis, like pages)
        oidx = oidx.permute(2, 0, 1, 3).contiguous().to(torch.int16)
        oval = oval.permute(2, 0, 1, 3).contiguous().to(torch.uint8)
        repr_l = str(outlier_repr).lower()
        if repr_l == "combinadic":
            cr, pv, fb = _pack_combinadic_frames(
                oidx.reshape(G, H * D, k), oval.reshape(G, H * D, k), vl=gt,
                oval_bits=oval_bits)
            k_crank = cr.reshape(G, H, D, -1).contiguous()
            k_oval = pv.reshape(G, H, D, -1).contiguous()
            k_field_bytes = fb
        elif repr_l == "bitmap":
            # FLAT BITMASK (paper item B4 / Fig 6 "Outlier Bitmask"): ceil(gt/8) bytes
            # per (group, head, channel) frame, one bit per token position. Sized by the
            # GROUP (gt), NOT by k — that flatness is the whole point of the encoding.
            bm, pv, fb = _pack_bitmap_frames(
                oidx.reshape(G, H * D, k), oval.reshape(G, H * D, k), vl=gt,
                oval_bits=oval_bits)
            k_obmp = bm.reshape(G, H, D, -1).contiguous()
            k_oval = pv.reshape(G, H, D, -1).contiguous()
            k_field_bytes = fb
        else:
            pi, pv = _pack_relidx7_frames(
                oidx.reshape(G, H * D, k), oval.reshape(G, H * D, k),
                oval_bits=oval_bits)
            k_oidx = pi.reshape(G, H, D, -1).contiguous()
            k_oval = pv.reshape(G, H, D, -1).contiguous()

    pages = T // int(page_size)
    codes_tok = code.reshape(H, D, T).permute(2, 0, 1)             # [T, H, D]
    k_base = _pack_2bit_lastdim(codes_tok).reshape(
        pages, page_size, H, D // 4).contiguous()
    k_scale_f = scale.squeeze(-1).permute(2, 0, 1).contiguous()        # [G,H,D] fp32
    k_zp = zp.squeeze(-1).permute(2, 0, 1).contiguous().to(scale_dtype)
    if use_int8_scale:
        # int8 pow2-exp K scale: scale is already 2^e (use_pow2), so exp =
        # round(log2(scale)) is LOSSLESS; the kernel reconstructs scale = 2^exp.
        # Range check: head_dim affine scale exponents fit comfortably in int8.
        k_scale = torch.round(torch.log2(k_scale_f.clamp_min(1e-12))).to(torch.int8)
    else:
        k_scale = k_scale_f.to(scale_dtype)

    # ---- V: token-major. Two formats:
    #   * "i2" (default): per-(token,channel-group) affine RTN to INT2 (≤3-bit path).
    #   * "bf16" (v10a): the FULL bf16 V value, no quant — K stays i2f4, V full precision.
    #     v_main holds raw bf16 [pages, ps, H, D]; v_scale/v_zp become unused dummies that
    #     keep the kernel ABI tensor-arg shape (the V=bf16 kernel branch ignores them).
    vfmt = str(v_format).lower()
    if vfmt not in ("i2", "bf16"):
        raise ValueError(f"v_format must be i2|bf16; got {v_format!r}")
    if vfmt == "bf16":
        v_main = V.to(torch.bfloat16).reshape(pages, page_size, H, D).contiguous()
        _sdt = torch.int8 if use_int8_scale else scale_dtype
        v_scale = torch.ones((pages, page_size, H, NGV), dtype=_sdt, device=_dev)
        v_zp = torch.zeros((pages, page_size, H, NGV), dtype=scale_dtype, device=_dev)
    else:
        gv = Vf.reshape(T, H, NGV, gc)
        vmin = gv.min(-1, keepdim=True)[0]
        vmax = gv.max(-1, keepdim=True)[0]
        vscale = (vmax - vmin).clamp_min(1e-8) / 3.0
        if use_pow2:
            vscale = torch.pow(2.0, torch.round(torch.log2(vscale.clamp_min(1e-12))))
        vscale = vscale.to(scale_dtype).to(torch.float32)
        vzp = vmin.to(scale_dtype).to(torch.float32)
        vcode = torch.round((gv - vzp) / vscale).clamp(0, 3).to(torch.int32)
        v_main = _pack_2bit_lastdim(vcode.reshape(T, H, D)).reshape(
            pages, page_size, H, D // 4).contiguous()
        v_scale_f = vscale.squeeze(-1).reshape(pages, page_size, H, NGV).contiguous()
        v_zp = vzp.squeeze(-1).reshape(pages, page_size, H, NGV).contiguous().to(scale_dtype)
        if use_int8_scale:
            v_scale = torch.round(torch.log2(v_scale_f.clamp_min(1e-12))).to(torch.int8)
        else:
            v_scale = v_scale_f.to(scale_dtype)

    def _to(t):
        return t.to(device) if t is not None else None

    out = {
        "k_base": _to(k_base),
        "k_scale": _to(k_scale), "k_zp": _to(k_zp),
        "k_oidx": _to(k_oidx), "k_oval": _to(k_oval),
        "k_crank": _to(k_crank),
        "k_obmp": _to(k_obmp),
        "k_fp4_mapscale": _to(k_fp4_mapscale),
        "k_fp4_mapcenter": _to(k_fp4_mapcenter),
        "k_field_bytes": int(k_field_bytes),
        "v_main": _to(v_main),
        "v_scale": _to(v_scale), "v_zp": _to(v_zp),
        "req_to_token": torch.arange(pages, dtype=torch.int32, device=device).reshape(1, pages),
        "req_to_group": torch.arange(G, dtype=torch.int32, device=device).reshape(1, G),
        "head_dim": D, "n_kv_heads": H, "num_tokens": T,
        "n_groups": G, "page_size": int(page_size),
        "outliers_per_vector": k,
        "outlier_repr": str(outlier_repr).lower(),
        "outlier_select": str(outlier_select).lower(),
        "group_tokens": gt, "group_channels": gc,
        "k_format": kfmt,
        "v_format": vfmt,
        "k_base_bits": k_base_bits,
        "oval_bits": oval_bits,
        "use_pow2": bool(use_pow2),
        "kv_outlier_map": bool(use_outlier_map and k > 0),
        "kv_int8_scale": bool(use_int8_scale),
    }
    return out


# ════════════════════════════════════════════════════════════════════════════
# canonical dequant oracle  (the test ground truth)
# ════════════════════════════════════════════════════════════════════════════

def derive_relidx7_from_combinadic(planes: dict) -> dict:
    """Load-time: combinadic-STORED outliers -> relidx7 idx for the fast decode kernel.

    The information-theoretic floor (combinadic rank, ~ceil(log2 C(vl,k)) bits) is the
    right encoding for STORAGE / BANDWIDTH (HBM-resident KV, LMCache DRAM/SSD offload,
    offline pt/safetensors/gguf) — 2.7-4.9x smaller than relidx7 at k=8-32. But the
    in-kernel combinadic READ is 2.3-4.5x SLOWER (measured: the in-register greedy
    unrank adds ~128 dependent LUT loads/slot). So the production design is
    **combinadic on storage, relidx7 at decode**: keep the rank plane resident, and
    derive the relidx7 idx bitstream ONCE at load/restore time (this function).

    Takes a pack dict with ``k_crank`` (outlier_repr=="combinadic") and returns a NEW
    dict with ``k_oidx`` (relidx7) added + ``outlier_repr="relidx7"`` so the kernel
    runs its fast relidx7 path. ``k_oval`` (the FP4/FP8 value stream, ascending-position
    order) is shared by both reprs and passes through unchanged.
    """
    if str(planes.get("outlier_repr")) != "combinadic" or planes.get("k_crank") is None:
        return planes
    k = int(planes["outliers_per_vector"])
    if k <= 0:
        return planes
    gt = int(planes.get("group_tokens", CANONICAL_GROUP_TOKENS))
    crank = planes["k_crank"].cpu()
    G, H, D, fb = (int(s) for s in crank.shape)
    # unrank the rank plane -> ascending outlier positions [G,H,D,k]
    oidx = _unpack_combinadic_frames(
        crank.reshape(G * H * D, 1, fb), k, vl=gt).reshape(G, H, D, k).to(torch.int16)
    # re-pack as the relidx7 7-bit LSB-first idx bitstream (value stream untouched).
    dummy_val = torch.zeros(G, H * D, k, dtype=torch.uint8)
    idx_packed, _ = _pack_relidx7_frames(
        oidx.reshape(G, H * D, k), dummy_val,
        oval_bits=int(planes.get("oval_bits", 4)))
    out = dict(planes)
    out["k_oidx"] = idx_packed.reshape(G, H, D, -1).contiguous().to(
        planes["k_crank"].device)
    out["outlier_repr"] = "relidx7"
    return out


def dequant_kv_canonical(planes):
    """Decode canonical planes -> fp32 ``(K_dq, V_dq)`` [T, H, D] — the test oracle.

    Reconstructs exactly what the kernel computes per element (base
    ``code·scale+zp``; outlier ``fp4_level·scale+zp`` spliced via the relidx7,
    combinadic OR flat-bitmap sidecar), decoding the REAL packed planes (2-bit base +
    sidecar) so the full storage codec is in the loop. All three index reprs land on the
    same ``(position, fp4 level)`` pairs, so the returned tensors are bit-identical
    across them — the membership-equivalence gate in tests/test_bitmap_outlier.py.
    """
    T = int(planes["num_tokens"]); H = int(planes["n_kv_heads"])
    D = int(planes["head_dim"]); G = int(planes["n_groups"])
    gt = int(planes.get("group_tokens", CANONICAL_GROUP_TOKENS))
    k = int(planes["outliers_per_vector"])
    ps = int(planes["page_size"])

    kv_int8_scale = bool(planes.get("kv_int8_scale", False))
    kv_outlier_map = bool(planes.get("kv_outlier_map", False))
    code = _unpack_2bit_lastdim(planes["k_base"].cpu().reshape(T, H, D // 4), D)  # [T,H,D]
    if kv_int8_scale:
        # int8 pow2-exp scale: scale = 2^exp (BIT-EXACT to the use_pow2 bf16 scale).
        scale = torch.exp2(planes["k_scale"].cpu().to(torch.float32))  # [G, H, D]
    else:
        scale = planes["k_scale"].cpu().to(torch.float32)             # [G, H, D]
    zp = planes["k_zp"].cpu().to(torch.float32)
    scale_t = scale.repeat_interleave(gt, dim=0)
    zp_t = zp.repeat_interleave(gt, dim=0)
    K_dq = code.to(torch.float32) * scale_t + zp_t
    if k > 0:
        # outlier value stream: i2f4/itf4 = FP4 nibble (2 codes/byte). Decode the
        # NORMALIZED level then affine-splice.
        oval_p = planes["k_oval"].cpu().to(torch.int64)
        ov = torch.stack(
            [(oval_p[..., s // 2] >> (4 * (s % 2))) & 0xF for s in range(k)], dim=-1)  # [G,H,D,k]
        repr_l = str(planes.get("outlier_repr", "relidx7"))
        if repr_l == "combinadic":
            crank = planes["k_crank"].cpu()
            fb = int(crank.shape[-1])
            oidx = _unpack_combinadic_frames(
                crank.reshape(G * H * D, 1, fb), k, vl=gt).reshape(G, H, D, k)
        elif repr_l == "bitmap":
            # FLAT BITMASK decode: ascending set-bit positions. The s-th ascending
            # position owns nibble s of k_oval — the SAME (position, value) pairing the
            # other two reprs produce, which is why all three dequant bit-identically.
            oidx = _unpack_bitmap_frames(
                planes["k_obmp"].cpu(), k, vl=gt)                   # [G, H, D, k]
        else:
            oidx = _unpack_relidx7_streams(planes["k_oidx"].cpu(), k)  # [G, H, D, k]
        lvl = decode_fp4_e2m1f(ov.to(torch.uint8))                 # [G, H, D, k] fp32
        if kv_outlier_map and planes.get("k_fp4_mapscale") is not None:
            # DEDICATED FP4 map dequant: dq = decode_fp4(level)/map_scale + o_center
            # (weight-space, fakequant-EXACT) — NOT the base-shared level·scale + zp.
            map_scale = planes["k_fp4_mapscale"].cpu().to(torch.float32)   # [G, H, D]
            o_center = planes["k_fp4_mapcenter"].cpu().to(torch.float32)
            dq_o = lvl / map_scale.unsqueeze(-1) + o_center.unsqueeze(-1)  # [G, H, D, k]
        else:
            dq_o = lvl * scale.unsqueeze(-1) + zp.unsqueeze(-1)    # [G, H, D, k]
        Kg = K_dq.reshape(G, gt, H, D).clone()
        g_i, h_i, d_i = torch.meshgrid(
            torch.arange(G), torch.arange(H), torch.arange(D), indexing="ij")
        for s in range(k):
            Kg[g_i, oidx[..., s], h_i, d_i] = dq_o[..., s]
        K_dq = Kg.reshape(T, H, D)

    if str(planes.get("v_format", "i2")).lower() == "bf16":
        # V=bf16 (v10a): v_main is the raw bf16 V — the oracle V IS the stored value
        # (no quant), so the kernel's PV reads exactly this (parity is K-only error).
        V_dq = planes["v_main"].cpu().to(torch.float32).reshape(T, H, D)
    else:
        vcode = _unpack_2bit_lastdim(planes["v_main"].cpu().reshape(T, H, D // 4), D)
        NGV = D // int(planes.get("group_channels", CANONICAL_GROUP_CHANNELS))
        if kv_int8_scale:
            vs = torch.exp2(planes["v_scale"].cpu().to(torch.float32)).reshape(T, H, NGV)
        else:
            vs = planes["v_scale"].cpu().to(torch.float32).reshape(T, H, NGV)
        vz = planes["v_zp"].cpu().to(torch.float32).reshape(T, H, NGV)
        gc = D // NGV
        V_dq = (vcode.to(torch.float32).reshape(T, H, NGV, gc)
                * vs.unsqueeze(-1) + vz.unsqueeze(-1)).reshape(T, H, D)
    assert ps * planes["k_base"].shape[0] == T
    return K_dq, V_dq


__all__ = [
    "CANONICAL_GROUP_TOKENS", "CANONICAL_GROUP_CHANNELS", "OUTLIER_REPRS",
    "ommx_pack_kv_canonical_block", "dequant_kv_canonical",
]
