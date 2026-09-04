# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""OMMX weight quantizer — the offline half of ``OMMX_Linear`` (paper §3.3 / Fig 6).

WHY THIS MODULE EXISTS
----------------------
Until now the ONLY implementation of the OMMX weight recipe lived inside a *test
script*: ``ommx_gpu_serve/csrc/linear/test_ommx_linear_parity.py::quantize_ommx_weight``.
That file is not importable from the serving tree (it is not a package, it has a
``main()`` that asserts ``torch.cuda.is_available()``), so nothing under
``ommx_gpu_serve/`` could actually pack a checkpoint — which is exactly why the
paper's ``OMMX_W_SafeTensor`` / "OMMX W-Packer" boxes (Fig 6, claims C3/D3/D4) had
no code behind them in this release.

This module lifts the SAME algorithm into a real module. It is deliberately a
LIFT, not a rewrite: the parity script is a shipped, passing correctness gate
(PARITY GATE: PASS 5/5 on sm_90a — cos 1.00000 / max_diff 5.96e-07 base,
0.99999 / 1.26e-02 prefill), so its numerics ARE the format contract. Every
element of every returned plane is asserted bit-identical to it by
``ommx_gpu_serve/tests/test_w_packer.py::test_bit_exact_vs_parity_packer`` over a
sweep of (shape, group_size, outlier_pct, seed). That comparison is the whole
point of this file; do not "clean up" an expression here without re-running it.

WHAT IS VERIFIED, WHAT IS NOT
-----------------------------
* VERIFIED on CPU, this session: bit-exactness vs the parity packer, the
  relidx7/bitmap position streams, and that ``dequantize_ommx_weight`` reproduces
  ``W_ref`` exactly from the stored planes alone.
* UNVERIFIED (no GPU this session): that the CUDA kernels in
  ``csrc/linear/ommx_linear.cu`` consume these planes correctly. That is what the
  parity gate does, and it needs a device. Nothing here was run against a kernel.

THE FOUR AXES (unchanged from the parity packer — changing any one changes accuracy)
------------------------------------------------------------------------------------
 1. group scale  = (range EXCLUDING outliers) / 3, then snapped to E8M0 pow2;
 2. zero-point   = group min (ASYMMETRIC affine, NOT symmetric);
 3. outlier count npv = ``max(1, int(group_size * outlier_pct))``, block == group;
 4. outliers     = FP4 E2M1 in INDEX space through a per-group range map with
                   ``fp_range = 12``  (``ms = 12/(imax-imin)``, ``mc = (imax+imin)/2``).

Dequant, mirrored by the kernel (``ommx_linear.cu`` header comment):
    base      w = code * 2^e + z
    outlier   w = (fp4(nibble) / ms + mc) * 2^e + z        (OVERWRITES the base lane)

Pure torch, CPU-capable. No CUDA, no triton, no vLLM import anywhere in this file.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

# ════════════════════════════════════════════════════════════════════════════
# Fig 6 pipeline names — the packer logs these so the code/figure map is legible
# ════════════════════════════════════════════════════════════════════════════
# The paper's Fig 6 "OMMX Safetensor Weight Format Offline Packing" box draws four
# steps. They are annotated below with `--- Fig 6 step (n) ---` banners.
#
# HONEST DEVIATION, stated once: Fig 6 draws (2) Min/Max Scaling BEFORE (3) Top-K
# Permutation, but the shipped recipe cannot run in that order — the scale is the
# range *excluding* the outliers, so the top-K mask must exist before the min/max
# is taken. The two steps are mutually dependent and the parity packer resolves the
# cycle by selecting top-K first. Reordering them changes every number, so the
# banners below label the steps by their Fig 6 name in the order the recipe runs.
FIG6_STEPS = (
    "(1) Vector Grouping",
    "(2) Min/Max Scaling",
    "(3) Top-K Permutation",
    "(4) FP4 Outlier Encoding",
)

# FP4 E2M1 decode table, idx = (sign<<3) | (exp<<1) | mant. Verbatim from the parity
# packer, INCLUDING the ``-0.0`` at index 8: the encoder below is a first-wins
# nearest-value scan over this list in THIS order, so the table order is load-bearing
# for the stored nibble bits (not for the decoded value).
FP4_E2M1: Sequence[float] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

#: FP4 index-space map range. ``ms = FP4_RANGE / (imax - imin)`` maps the outlier
#: index spread onto the E2M1 representable range [-6, 6] with headroom.
FP4_RANGE = 12.0

#: Outlier position representations this packer can emit.
#:   ``relidx7`` — 7 bit per outlier slot, LSB-first, byte padded per group. The
#:                 SHIPPED representation: it is what the parity gate and the CUDA
#:                 ``relidx7_slot_pos`` decoder consume.
#:   ``bitmap``  — one flat bit per group element (paper §3.1 claim B4: "on our GPU
#:                 implementation, positions are stored as a flat bitmask, N bits per
#:                 group"). Costs more metadata at small npv, decodes in O(1) by
#:                 popcount-rank. Bit layout is identical to
#:                 ``attention/codec.py::pack_bitmap_row`` (position p -> byte p>>3,
#:                 bit p&7), which the tests cross-check against.
#: UNVERIFIED (no GPU this session): no CUDA kernel in this repo reads a stored
#: bitmap plane today — ``ommx_linear.cu`` decodes relidx7. ``bitmap`` is a format
#: option and a bit-budget statement, not a wired-up execution path.
OUTLIER_REPRS = ("relidx7", "bitmap")

#: Outlier value maps.
#:   ``idx_range`` — the shipped per-group range map (needs the ``map_scale`` and
#:                   ``map_center`` planes). This is the parity-gated encoding.
#:   ``none``      — degenerate instance ms=1, mc=0: the FP4 code IS the index-space
#:                   residual. Stores no map planes at all, which is what the paper's
#:                   bundle definition (§3.1 B6: dense payload + SC + ZP + position
#:                   metadata + extension codes) actually lists. Lower fidelity per
#:                   outlier, 2 fewer per-group planes. NOT covered by the parity gate.
OUTLIER_MAPS = ("idx_range", "none")

#: Storage dtypes accepted for the shared zero-point plane. BF16 is the paper's
#: choice (§3.1 Eq (2) claim B9: "z is a BF16 shared zero-point") and is what the
#: bits/weight accounting in ``csrc/linear/README.md`` assumes (16/gs bits). F32 is
#: kept because the parity packer's reference math is F32 — see ``zp_dtype`` below.
ZP_DTYPES = (torch.bfloat16, torch.float32)


class OMMXQuantizeError(ValueError):
    """Raised when a weight cannot be packed by THIS recipe.

    Project law: no silent fallback. Every raise below names the tensor property
    that is wrong and the value it would have to have instead — a packer that
    quietly rounded a shape or clamped an exponent would emit a bundle that
    dequantizes to garbage with no visible failure.
    """


def _fp4_encode_first_wins(x: torch.Tensor) -> torch.Tensor:
    """Nearest-FP4-value encode with FIRST-WINS ties — exact mirror of the reference.

    The parity packer's ``_fp4_encode_linear`` is a linear scan with a STRICT ``<``
    update and ``best`` seeded at ``1e30``. Both details are observable:

      * strict ``<`` sends every exact midpoint to the LOWER table index, and sends
        the ``-0.0`` tie (index 8) back to ``+0.0`` (index 0). Decoded value is the
        same; the STORED NIBBLE is not, and the nibble is what we write to disk.
      * seeding at ``1e30`` rather than ``inf`` means an input whose distance to
        every table entry is >= 1e30 encodes as index 0 rather than as the nearest
        finite entry. Unreachable for in-range inputs, reproduced anyway: this
        function's contract is bit-equality, not plausibility.

    ``x`` must be float64 — see the note on precision in ``quantize_ommx_weight``.
    """
    best = torch.full_like(x, 1e30)
    out = torch.zeros(x.shape, dtype=torch.uint8, device=x.device)
    for i, v in enumerate(FP4_E2M1):
        d = (x - v).abs()
        upd = d < best                     # strict '<' => first (lowest) index wins
        best = torch.where(upd, d, best)
        out = torch.where(upd, torch.full_like(out, i), out)
    return out


def _fp4_decode(nib: torch.Tensor) -> torch.Tensor:
    """Decode FP4 nibbles (uint8, low 4 bits) to float64 via the same table."""
    table = torch.tensor(FP4_E2M1, dtype=torch.float64, device=nib.device)
    return table[nib.to(torch.long) & 0xF]


def _pack_bits_lsb_first(bits: torch.Tensor, n_bytes: int) -> torch.Tensor:
    """Pack a trailing 0/1 axis into bytes, LSB-first (bit i -> byte i>>3, bit i&7).

    ``bits`` is ``[..., nbits]``; the result is ``[..., n_bytes]`` uint8, zero-padded
    when ``nbits < 8*n_bytes``. This is the vectorized form of the reference's
    ``out[bitpos >> 3] |= 1 << (bitpos & 7)`` loop and of
    ``attention/codec.py::pack_relidx7`` / ``pack_bitmap_row``; the tests assert
    equality against those python primitives rather than trusting the rewrite.
    """
    nbits = int(bits.shape[-1])
    pad = n_bytes * 8 - nbits
    if pad < 0:
        raise OMMXQuantizeError(
            f"bit stream of {nbits} bits does not fit in {n_bytes} bytes")
    if pad:
        bits = torch.nn.functional.pad(bits, (0, pad))
    weights = (1 << torch.arange(8, dtype=torch.int32, device=bits.device))
    return (bits.to(torch.int32).reshape(*bits.shape[:-1], n_bytes, 8) * weights) \
        .sum(dim=-1).to(torch.uint8)


def relidx7_index_bytes(npv: int) -> int:
    """Bytes of relidx7 position stream per group: ``ceil(npv*7/8)`` (byte padded)."""
    return (int(npv) * 7 + 7) // 8 if npv else 0


def bitmap_index_bytes(group_size: int) -> int:
    """Bytes of flat bitmask per group: ``ceil(group_size/8)`` — one bit per element."""
    return (int(group_size) + 7) // 8


def outlier_index_bytes(npv: int, group_size: int, outlier_repr: str) -> int:
    """Bytes of position metadata per group for the chosen representation."""
    if not npv:
        return 0
    if outlier_repr == "relidx7":
        return relidx7_index_bytes(npv)
    if outlier_repr == "bitmap":
        return bitmap_index_bytes(group_size)
    raise OMMXQuantizeError(
        f"unknown outlier_repr {outlier_repr!r}; expected one of {OUTLIER_REPRS}")


def combinadic_index_bits(npv: int, group_size: int) -> int:
    """Position cost in BITS per group under the NPU's codec: ``ceil(log2 C(gs, npv))``.

    This is NOT a storage option of this packer -- no plane is ever written in this
    encoding and no GPU kernel reads one. It exists because the paper specifies the
    NPU's position metadata this way (§3.1: "compressed via a combinatorial number
    system codec into ceil(log2 C(N,K)) bits per group"), and because §3.1 also says
    the two platforms differ in position metadata ALONE ("All other bundle fields are
    platform-invariant, requiring only position-metadata regeneration per target").
    Those two sentences together define a second, fully determined bit budget for any
    recipe, and :meth:`w_format.Recipe.bits_breakdown` reports it as the ``npu`` basis
    so a paper AvgBits figure can be checked against the recipe it is claimed for
    instead of being asserted.

    Deliberately duplicated from ``attention/codec.py::combinadic_index_bits`` rather
    than imported: ``linear/`` imports nothing from ``attention/`` today and one
    reporting figure is not worth coupling the two. ``tests/test_npu_bit_basis.py::
    test_combinadic_bits_have_not_drifted_from_the_kv_codec`` asserts the two agree
    over the whole grid, so the copy cannot silently diverge.

    ``npv == group_size`` costs 0 bits (one subset), matching the KV codec.
    """
    k, n = int(npv), int(group_size)
    if k <= 0 or k >= n:
        return 0
    return max(1, (math.comb(n, k) - 1).bit_length())


def outlier_code_bytes(npv: int) -> int:
    """Bytes of FP4 nibble stream per group: ``ceil(npv/2)`` (2 nibbles per byte)."""
    return (int(npv) + 1) // 2 if npv else 0


def derive_npv(group_size: int, outlier_pct: float) -> int:
    """Outliers per group. Verbatim rule from the parity packer (format axis 3)."""
    if outlier_pct <= 0:
        return 0
    return max(1, int(group_size * float(outlier_pct)))


def _decided_plane(decided: dict, key: str, N: int, G: int) -> torch.Tensor:
    """One [N, G] plane from a caller-supplied decision, shape-checked.

    A wrong-shaped plane that happened to broadcast would encode against parameters the
    decider never chose, and the only symptom would be a slightly wrong model -- so the
    shape is a refusal, not a reshape.
    """
    if key not in decided:
        raise OMMXQuantizeError(
            f"decided=... is missing {key!r}. The encode-only path decides nothing, so "
            f"every plane it needs must be supplied: scale, zp"
            + (", map_scale, map_center" if key.startswith("map") else ""))
    t = torch.as_tensor(decided[key])
    if tuple(t.shape) != (N, G):
        raise OMMXQuantizeError(
            f"decided[{key!r}] has shape {tuple(t.shape)}, expected {(N, G)} "
            f"(one value per output row per group)")
    if not bool(torch.isfinite(t).all()):
        raise OMMXQuantizeError(f"decided[{key!r}] contains NaN/Inf")
    return t


def _decided_omask(decided: dict, N: int, G: int, group_size: int, npv: int
                   ) -> torch.Tensor:
    """The outlier mask from a caller-supplied decision, validated to exactly npv/group.

    relidx7 stores a FIXED number of slots per group, so a group with the wrong count
    would either overrun the stream or leave a slot holding position 0 -- silently
    "correcting" element 0 of that group. Checked here rather than trusted.
    """
    if npv == 0:
        return torch.zeros(N, G, group_size, dtype=torch.bool)
    if "omask" not in decided:
        raise OMMXQuantizeError(
            "decided=... is missing 'omask' ([N, G, group_size] bool) but npv>0")
    m = torch.as_tensor(decided["omask"]).to(torch.bool)
    if tuple(m.shape) != (N, G, group_size):
        raise OMMXQuantizeError(
            f"decided['omask'] has shape {tuple(m.shape)}, expected "
            f"{(N, G, group_size)}")
    per_group = m.sum(dim=-1)
    if not bool((per_group == npv).all()):
        bad = int((per_group != npv).sum())
        raise OMMXQuantizeError(
            f"decided['omask'] must mark EXACTLY npv={npv} lanes in every group; "
            f"{bad} group(s) differ (min {int(per_group.min())}, max "
            f"{int(per_group.max())}). The outlier stream is fixed-width per group.")
    return m


def _require_e8m0(scale: torch.Tensor) -> None:
    """Refuse a scale the E8M0 plane cannot store.

    The format keeps ONE BYTE of exponent per group: the stored scale is always a power
    of two. A calibrated solver run WITHOUT its pow2 option produces full-precision
    scales, and packing those would round every one of them -- discarding exactly the
    thing the calibration chose. That must fail loudly at pack time, not show up later
    as an unexplained accuracy drop.
    """
    exp = torch.log2(scale.clamp_min(1e-38))
    off = (exp - torch.round(exp)).abs().max()
    if float(off) > 1e-6:
        raise OMMXQuantizeError(
            "decided['scale'] is not a power of two, but the E8M0 scale plane stores "
            f"only an exponent byte (max deviation {float(off):.3e} in log2). Re-run the "
            "calibration with its power-of-two scale option (wq_eval: --pow2) so the "
            "scales it chooses are the scales this format can store.")


def quantize_ommx_weight(
    W: torch.Tensor,
    group_size: int,
    outlier_pct: float,
    *,
    npv: Optional[int] = None,
    outlier_repr: str = "relidx7",
    outlier_map: str = "idx_range",
    zp_dtype: torch.dtype = torch.float32,
    reference: bool = True,
    decided: Optional[dict] = None,
) -> dict:
    """Quantize ``W[N, K]`` to the OMMX i2f4 weight bundle.

    Returns EXACTLY the key set of
    ``csrc/linear/test_ommx_linear_parity.py::quantize_ommx_weight`` — ``code``,
    ``scale``, ``scale_exp``, ``zp``, ``map_scale``, ``map_center``, ``oindex``,
    ``ocode``, ``W_base``, ``W_ref``, ``N``, ``K``, ``G``, ``group_size``, ``npv``,
    ``n_blk`` — and with the DEFAULT arguments every one of them is bit-identical
    to it. The optional keyword arguments are additive format axes; each is a
    no-op at its default:

    ``npv``
        Override the derived outlier count. ``None`` derives it from
        ``outlier_pct`` exactly as the reference does.
    ``outlier_repr``
        ``relidx7`` (default, the shipped/gated stream) or ``bitmap`` (paper B4).
        Only the ``oindex`` plane changes; ``ocode`` and every numeric plane are
        untouched, so the DEQUANTIZED WEIGHTS ARE IDENTICAL either way — this axis
        trades metadata bytes for decode complexity, nothing else.
    ``outlier_map``
        ``idx_range`` (default, gated) or ``none`` (ms=1, mc=0, no map planes).
    ``zp_dtype``
        Storage dtype of the zero-point. ``float32`` reproduces the reference
        exactly. ``bfloat16`` is the paper's format (claim B9) and the packer's
        default; the rounding is applied BEFORE the INT2 codes are chosen, so the
        bundle stays self-consistent — quantizing against an f32 zero-point and
        then rounding it on the way to disk would leave every code fractionally
        mismatched to the zero-point the kernel actually reads back.
    ``reference``
        ``False`` skips ``W_base`` / ``W_ref`` (the keys are still present and hold
        ``None``). Those two are full-precision copies of ``W``; the offline packer
        does not need them and a 4096x14336 f32 weight is 235 MiB each.

    Precision note (WHY float64 shows up below): the reference encodes outliers in
    a per-element PYTHON loop, so ``(w - z)/s`` and ``(iv - mc)*ms`` are evaluated
    in python float == float64, while the base path is evaluated as float32 tensor
    ops. The vectorized form here must therefore use float32 for the base and
    float64 for the outlier stage, or a value sitting on an FP4 decision boundary
    encodes to a different nibble. This is not defensive rounding; it is the
    difference between passing and failing the bit-exactness gate.
    """
    # ── input refusals (each names what is wrong, per the no-silent-fallback law) ──
    if W.dim() != 2:
        raise OMMXQuantizeError(f"expected a 2-D weight [N, K]; got shape {tuple(W.shape)}")
    if W.dtype != torch.float32:
        raise OMMXQuantizeError(
            f"expected a float32 weight; got {W.dtype}. Cast first with "
            f"`W.to(torch.float32)` — bf16/f16 -> f32 is exact and is what the "
            f"packer does, but doing it implicitly here would hide a f64 input "
            f"whose extra mantissa bits change every code.")
    group_size = int(group_size)
    N, K = int(W.shape[0]), int(W.shape[1])
    if group_size <= 0 or K % group_size != 0:
        raise OMMXQuantizeError(
            f"group_size={group_size} must divide K={K} (weight shape {(N, K)}); "
            f"K/group_size = {K / group_size if group_size else float('nan')}")
    if group_size % 4 != 0:
        raise OMMXQuantizeError(
            f"group_size={group_size} must be a multiple of 4: the INT2 payload packs "
            f"4 codes per byte along K.")
    if outlier_repr not in OUTLIER_REPRS:
        raise OMMXQuantizeError(
            f"unknown outlier_repr {outlier_repr!r}; expected one of {OUTLIER_REPRS}")
    if outlier_map not in OUTLIER_MAPS:
        raise OMMXQuantizeError(
            f"unknown outlier_map {outlier_map!r}; expected one of {OUTLIER_MAPS}")
    if zp_dtype not in ZP_DTYPES:
        raise OMMXQuantizeError(
            f"unsupported zp_dtype {zp_dtype}; expected one of "
            f"{[str(d) for d in ZP_DTYPES]}")
    if not bool(torch.isfinite(W).all()):
        raise OMMXQuantizeError(
            "weight contains NaN/Inf. The reference encoder maps a non-finite value "
            "to FP4 code 0 without complaining, which would write a plausible-looking "
            "bundle for a broken checkpoint.")

    npv = derive_npv(group_size, outlier_pct) if npv is None else int(npv)
    if npv < 0 or npv > group_size:
        raise OMMXQuantizeError(f"npv={npv} must be in [0, group_size={group_size}]")
    if npv and outlier_repr == "relidx7" and group_size > 128:
        raise OMMXQuantizeError(
            f"relidx7 stores a position in 7 bits (max 127) but group_size="
            f"{group_size}; use outlier_repr='bitmap' or group_size <= 128.")

    G = K // group_size

    # ── Fig 6 step (1) Vector Grouping ─────────────────────────────────────────
    # The "vector" is one contiguous run of `group_size` elements along K within one
    # output row. Grouping is a VIEW, never a copy: [N, K] -> [N, G, group_size].
    Wg = W.view(N, G, group_size)

    # ── Fig 6 step (3) Top-K Permutation (runs first; see FIG6_STEPS note) ─────
    # Top-K by |w| within the group. The parity packer's exact call, so the
    # device-dependent tie-break in torch.topk resolves identically here.
    omask = torch.zeros_like(Wg, dtype=torch.bool)
    topk = None
    if decided is not None:
        # ENCODE-ONLY PATH. Every DECISION (which lanes are outliers, the group scale,
        # the zero-point, the FP4 range map) is supplied; this function then runs the
        # SAME encoding lines as the deciding path, so a calibrated quantizer cannot
        # drift from the on-disk convention by reimplementing it.
        #
        # WHY IT EXISTS: wq_eval's error-feedback solver returns Q, the DEQUANTIZED
        # weight, and re-quantizing Q with the min/max path is NOT idempotent
        # (measured: max|W_ref2 - W_ref1| = 5.2e-2, codes and scales both differ), so
        # packing a calibrated weight through the normal path silently throws the
        # calibration away. The solver's own per-group params are exactly this format's
        # params, so handing them over is lossless -- and the caller can assert it,
        # because dequantize_ommx_weight(planes) must then reproduce Q exactly.
        omask = _decided_omask(decided, N, G, group_size, npv)
        scale = _decided_plane(decided, "scale", N, G).to(torch.float32)
        _require_e8m0(scale)
        zp = _decided_plane(decided, "zp", N, G).to(torch.float32)
        if zp_dtype != torch.float32:
            zp = zp.to(zp_dtype).to(torch.float32)
        if npv:
            # relidx7 packing needs ASCENDING positions; derive them from the mask so
            # the caller does not have to know the packing order.
            topk = omask.nonzero(as_tuple=False)[:, 2].view(N, G, npv)
    elif npv:
        topk = Wg.abs().topk(npv, dim=-1).indices                       # [N, G, npv]
        omask.scatter_(-1, topk, True)

    # ── Fig 6 step (2) Min/Max Scaling ─────────────────────────────────────────
    # Range EXCLUDING the outliers (that is the whole point of the dual-resolution
    # format: the INT2 grid does not have to span the tail). The +-1e9 sentinels and
    # the >1e8 / <-1e8 rescue are the reference's handling of an all-outlier group.
    if decided is None:
        big = torch.where(omask, torch.full_like(Wg, 1e9), Wg)
        sml = torch.where(omask, torch.full_like(Wg, -1e9), Wg)
        mn = big.min(dim=-1).values
        mx = sml.max(dim=-1).values
        mn = torch.where(mn > 1e8, torch.zeros_like(mn), mn)
        mx = torch.where(mx < -1e8, torch.zeros_like(mx), mx)
        scale = (mx - mn).clamp_min(1e-8) / 3.0                         # 3 INT2 steps
        scale = torch.pow(2.0, torch.round(torch.log2(scale.clamp_min(1e-12))))  # E8M0 pow2

        # Zero-point = group min. Rounded to its STORAGE dtype here, before the codes
        # are chosen (see the ``zp_dtype`` note in the docstring). float32 is a no-op,
        # which is what keeps the default path bit-identical to the reference.
        zp = mn if zp_dtype == torch.float32 else mn.to(zp_dtype).to(torch.float32)

    code_f = torch.round((Wg - zp.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp(0, 3)  # RNE
    W_base = code_f * scale.unsqueeze(-1) + zp.unsqueeze(-1) if reference else None

    # ── Fig 6 step (4) FP4 Outlier Encoding ────────────────────────────────────
    W_ref = W_base.clone() if reference else None
    nib_g = torch.zeros(N, G, group_size, dtype=torch.uint8)
    o_center = torch.zeros(N, G)
    map_scale = torch.ones(N, G)
    if npv:
        # IDX-SPACE map (ommx_linear.cu header: delta = (fp4/ms + mc - code)*scale).
        # imin/imax are taken over the OUTLIER indices only; mc centres them and ms
        # stretches them onto the E2M1 range. Computed in float32 exactly as the
        # reference does, because these two planes are themselves stored.
        idx = (Wg - zp.unsqueeze(-1)) / scale.unsqueeze(-1)
        i_big = torch.where(omask, idx, torch.full_like(idx, 1e9))
        i_sml = torch.where(omask, idx, torch.full_like(idx, -1e9))
        imin = i_big.min(dim=-1).values
        imax = i_sml.max(dim=-1).values
        if decided is not None:
            # The caller decided the range map too (its FP4 nibbles were chosen against
            # THESE planes, so recomputing them here would re-encode against a different
            # map and the reconstruction would not match what the solver produced).
            if outlier_map == "idx_range":
                o_center = _decided_plane(decided, "map_center", N, G).to(torch.float32)
                map_scale = _decided_plane(decided, "map_scale", N, G).to(torch.float32)
        elif outlier_map == "idx_range":
            o_center = (imax + imin) / 2.0
            map_scale = FP4_RANGE / (imax - imin).clamp_min(1e-8)
        # else: ``none`` keeps mc=0, ms=1 -> the nibble is the raw index residual.

        # float64 from here: mirrors the reference's per-element python arithmetic.
        s64 = scale.to(torch.float64).unsqueeze(-1)
        z64 = zp.to(torch.float64).unsqueeze(-1)
        mc64 = o_center.to(torch.float64).unsqueeze(-1)
        ms64 = map_scale.to(torch.float64).unsqueeze(-1)
        iv = (Wg.to(torch.float64) - z64) / s64                         # index space
        nib_g = _fp4_encode_first_wins((iv - mc64) * ms64)
        nib_g = torch.where(omask, nib_g, torch.zeros_like(nib_g))      # base lanes = 0
        if reference:
            rec = (_fp4_decode(nib_g) / ms64 + mc64) * s64 + z64
            W_ref = torch.where(omask, rec.to(torch.float32), W_ref)

    # ── INT2 dense payload: 4 codes per byte, code j at bit 2*(j%4) ─────────────
    code_u8 = code_f.view(N, K).to(torch.uint8)
    packed = torch.zeros(N, K // 4, dtype=torch.uint8)
    for j in range(4):
        packed |= (code_u8[:, j::4] << (2 * j))

    scale_exp_f = torch.round(torch.log2(scale.clamp_min(1e-12)))
    if bool(((scale_exp_f < -128) | (scale_exp_f > 127)).any()):
        raise OMMXQuantizeError(
            "E8M0 scale exponent out of int8 range for at least one group "
            f"(min {float(scale_exp_f.min())}, max {float(scale_exp_f.max())}). The "
            "weight has a group whose dynamic range cannot be represented by a "
            "power-of-two byte exponent; rescale the checkpoint before packing.")
    scale_exp = scale_exp_f.to(torch.int8)

    # ── outlier sidecar: positions + FP4 nibbles, block == group ───────────────
    idx_blk_bytes = outlier_index_bytes(npv, group_size, outlier_repr)
    nib_per_blk = outlier_code_bytes(npv)
    if npv:
        pos_sorted, _ = torch.sort(topk, dim=-1)                        # ASCENDING
        nib_sorted = torch.gather(nib_g, -1, pos_sorted)                # nibble order
        if outlier_repr == "relidx7":
            # 7 bits per slot, LSB-first, slots concatenated then byte padded.
            bits = (pos_sorted.unsqueeze(-1)
                    >> torch.arange(7, dtype=pos_sorted.dtype)) & 1     # [N,G,npv,7]
            oindex = _pack_bits_lsb_first(bits.reshape(N, G, npv * 7), idx_blk_bytes)
        else:
            # flat bitmask: one bit per group element, position p -> byte p>>3 bit p&7.
            oindex = _pack_bits_lsb_first(omask.to(torch.int32), idx_blk_bytes)
        # nibble stream: slot s -> byte s>>1, low nibble for even s.
        pad = nib_per_blk * 2 - npv
        nib_pad = torch.nn.functional.pad(nib_sorted, (0, pad)) if pad else nib_sorted
        pairs = nib_pad.reshape(N, G, nib_per_blk, 2).to(torch.int32)
        ocode = (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)
        oindex = oindex.reshape(-1)
        ocode = ocode.reshape(-1)
    else:
        # Reference emits a 1-byte placeholder rather than an empty tensor so the
        # pybind signature always has something to bind. Kept identical.
        oindex = torch.zeros(1, dtype=torch.uint8)
        ocode = torch.zeros(1, dtype=torch.uint8)

    return dict(
        code=packed,
        scale=scale.view(N, G).to(torch.float32),
        scale_exp=scale_exp,
        zp=zp.view(N, G).to(torch.float32),
        map_scale=map_scale.to(torch.float32),
        map_center=o_center.to(torch.float32),
        oindex=oindex,
        ocode=ocode,
        W_base=W_base.view(N, K) if reference else None,
        W_ref=W_ref.view(N, K) if reference else None,
        N=N, K=K, G=G, group_size=group_size, npv=npv, n_blk=G,
    )


def unpack_int2_codes(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Inverse of the INT2 payload packing -> uint8 codes ``[N, K]`` in {0,1,2,3}."""
    N = int(packed.shape[0])
    if int(packed.shape[1]) != K // 4:
        raise OMMXQuantizeError(
            f"INT2 payload has {int(packed.shape[1])} bytes per row but K={K} needs "
            f"{K // 4}")
    code = torch.zeros(N, K, dtype=torch.uint8)
    for j in range(4):
        code[:, j::4] = (packed >> (2 * j)) & 0x3
    return code


def outlier_positions(
    oindex: torch.Tensor, N: int, G: int, npv: int, group_size: int, outlier_repr: str
) -> torch.Tensor:
    """Decode the position stream -> ascending positions ``[N, G, npv]`` (int64).

    Both representations are decoded WITHOUT a python per-element loop:
      * relidx7 — 7-bit LSB-first gather over the unpacked bit plane;
      * bitmap  — the O(1) popcount-rank decode ``attention/codec.py`` documents:
                  the s-th set bit is the s-th nibble, so ``cumsum(mask) - 1`` is the
                  nibble ordinal and the positions are the sorted set bits.
    """
    if npv == 0:
        return torch.zeros(N, G, 0, dtype=torch.long)
    nbytes = outlier_index_bytes(npv, group_size, outlier_repr)
    blob = oindex.reshape(N, G, nbytes).to(torch.int32)
    bits = (blob.unsqueeze(-1) >> torch.arange(8, dtype=torch.int32)) & 1
    bits = bits.reshape(N, G, nbytes * 8)
    if outlier_repr == "relidx7":
        sl = bits[..., : npv * 7].reshape(N, G, npv, 7)
        return (sl * (1 << torch.arange(7, dtype=torch.int32))).sum(-1).to(torch.long)
    mask = bits[..., :group_size].to(torch.bool)
    count = int(mask.sum(-1).min()), int(mask.sum(-1).max())
    if count != (npv, npv):
        raise OMMXQuantizeError(
            f"bitmap position plane has {count[0]}..{count[1]} set bits per group but "
            f"the manifest declares npv={npv}")
    # Ascending set-bit positions per group: sort pushes the npv True lanes first.
    order = torch.argsort((~mask).to(torch.int32), dim=-1, stable=True)
    return order[..., :npv].sort(dim=-1).values.to(torch.long)


def dequantize_ommx_weight(
    *,
    code: torch.Tensor,
    scale_exp: torch.Tensor,
    zp: torch.Tensor,
    N: int,
    K: int,
    group_size: int,
    npv: int,
    oindex: Optional[torch.Tensor] = None,
    ocode: Optional[torch.Tensor] = None,
    map_scale: Optional[torch.Tensor] = None,
    map_center: Optional[torch.Tensor] = None,
    outlier_repr: str = "relidx7",
) -> torch.Tensor:
    """Reconstruct ``W_ref[N, K]`` float32 from the STORED planes alone.

    This is the read side of the format contract and the reason the round-trip gate
    can be exact: it consumes only what a bundle actually holds (E8M0 exponent byte,
    stored zero-point, packed codes, position stream, nibbles, map planes) and
    reproduces ``quantize_ommx_weight(...)["W_ref"]`` element for element.

    ``2 ** scale_exp`` is bit-exact to the float32 ``scale`` the quantizer computed —
    it IS a power of two — which is the same property the GPU parity gate asserts
    ("decode M=1 E8M0 vs fp32 (must be exact)", max_diff 0.00e+00 on sm_90a).
    """
    G = K // group_size
    scale = torch.pow(2.0, scale_exp.to(torch.float32)).reshape(N, G)
    zp = zp.to(torch.float32).reshape(N, G)
    code_f = unpack_int2_codes(code, K).to(torch.float32).view(N, G, group_size)
    W = code_f * scale.unsqueeze(-1) + zp.unsqueeze(-1)
    if npv:
        if oindex is None or ocode is None:
            raise OMMXQuantizeError(
                f"npv={npv} but the outlier planes are missing (oindex="
                f"{oindex is not None}, ocode={ocode is not None})")
        pos = outlier_positions(oindex, N, G, npv, group_size, outlier_repr)
        nb = outlier_code_bytes(npv)
        raw = ocode.reshape(N, G, nb).to(torch.int32)
        nib = torch.stack([raw & 0xF, (raw >> 4) & 0xF], dim=-1).reshape(N, G, nb * 2)
        nib = nib[..., :npv].to(torch.uint8)
        ms = (torch.ones(N, G) if map_scale is None else map_scale.to(torch.float32)
              ).reshape(N, G).to(torch.float64).unsqueeze(-1)
        mc = (torch.zeros(N, G) if map_center is None else map_center.to(torch.float32)
              ).reshape(N, G).to(torch.float64).unsqueeze(-1)
        s64 = scale.to(torch.float64).unsqueeze(-1)
        z64 = zp.to(torch.float64).unsqueeze(-1)
        val = ((_fp4_decode(nib) / ms + mc) * s64 + z64).to(torch.float32)
        W.scatter_(-1, pos, val)
    return W.reshape(N, K)
