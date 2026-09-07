# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The weight kernel's native bitmap position reader.

WHY IT EXISTS. The packer could always emit flat-bitmask outlier positions, but
``ommx_linear.cu`` decoded a relidx7 slot stream unconditionally, so a bitmap bundle was
either refused at load or re-encoded to relidx7 by an opt-in load-time transcode. The
transcode is lossless but it spends the saving it was meant to deliver: a bundle stored at
3.6250 bits/weight streamed at 4.1250. ``nc::idx_slot_pos`` now dispatches on an
``idx_fmt`` argument threaded from the recipe, so both encodings serve natively.

WHICH ONE IS CHEAPER IS NOT FIXED. relidx7 spends ceil(7*npv/8) bytes per group and the
mask ceil(group_size/8), so the mask wins above roughly 12.5-14% coverage and loses below
it. At the shipped ``optimized-weight`` recipe (25% coverage) the mask saves 0.75
bits/weight; at ``paper-weight`` (6.25%) it costs 0.50 more. The paper's own rationale --
that the mask is chosen "at the cost of higher metadata overhead" -- is therefore backwards
for the high-coverage regime the format is built around.

WHAT THIS FILE CAN AND CANNOT SHOW. The CUDA compiles (nvcc -arch=sm_90a, clean) but has
never RUN: no GPU was available. These gates mirror the kernel's byte -> word -> popcount
-> rank -> position arithmetic in Python over frames the real packer produced, which is
the only way to catch a wrong formula without a device. Execution parity is
``csrc/linear/test_ommx_linear_parity.py``, and it is GPU-only.
"""
import pytest

torch = pytest.importorskip("torch")

from ommx_gpu_serve.linear.quantize import (  # noqa: E402
    bitmap_index_bytes,
    outlier_index_bytes,
    quantize_ommx_weight,
    relidx7_index_bytes,
)
from ommx_gpu_serve.linear.w_format import (  # noqa: E402
    KERNEL_OUTLIER_REPRS,
    Recipe,
    plan_transcode,
)

GS, NPV = 64, 16
QK = dict(npv=NPV, outlier_map="idx_range", zp_dtype=torch.bfloat16)


def _mirror_slot_pos(blk: bytes, n_bytes: int, r: int) -> int:
    """Python mirror of ``nc::bitmap_slot_pos_bytes``.

    Deliberately transcribed step for step -- little-endian 4-byte word assembly, running
    popcount, then clearing the low set bits of the word that contains the r-th one --
    rather than written the short way with a list comprehension. A shorter mirror would
    agree with the packer while disagreeing with the kernel, which is the failure this is
    supposed to catch.
    """
    acc = 0
    for base in range(0, n_bytes, 4):
        nb = min(4, n_bytes - base)
        w = 0
        for t in range(nb):
            w |= blk[base + t] << (8 * t)
        pc = bin(w).count("1")
        if acc + pc > r:
            for _ in range(acc, r):
                w &= w - 1                      # drop lower set bits
            return (base << 3) + ((w & -w).bit_length() - 1)   # __ffs(w) - 1
        acc += pc
    return -1                                    # padded slot: fewer than r+1 outliers


def _mirror_walker(blk: bytes, n_bytes: int, fmt: str, npv: int) -> list:
    """Python mirror of ``nc::SlotWalker`` (the sequential position reader the M<=16
    decode kernels use). Transcribed step for step from the CUDA struct -- the relidx7
    shift register refilled a byte at a time, the bitmap word assembled little-endian then
    drained by lowest-set-bit -- so that a wrong refill bound or bit order in the .cu is
    caught here rather than by a parity run. Returns the ``npv`` emitted positions
    (-1 once a bitmap block's population is exhausted)."""
    out = []
    if fmt == "bitmap":
        pos_byte, buf, wbase = 0, 0, 0

        def load_word():
            nonlocal pos_byte, buf, wbase
            nb = min(4, n_bytes - pos_byte)
            w = 0
            for t in range(nb):
                w |= blk[pos_byte + t] << (8 * t)
            buf, wbase = w, pos_byte << 3
            pos_byte += nb

        load_word()
        for _ in range(npv):
            exhausted = False
            while buf == 0:
                if pos_byte >= n_bytes:
                    exhausted = True
                    break
                load_word()
            if exhausted:
                out.append(-1)
                continue
            out.append(wbase + ((buf & -buf).bit_length() - 1))   # __ffs(buf) - 1
            buf &= buf - 1
        return out
    buf, have, pos_byte = 0, 0, 0
    for _ in range(npv):
        while have < 7:
            assert pos_byte < n_bytes, "relidx7 refill read past the block"
            buf |= blk[pos_byte] << have
            pos_byte += 1
            have += 8
        out.append(buf & 0x7F)
        buf >>= 7
        have -= 7
    return out


@pytest.fixture(scope="module")
def packed():
    torch.manual_seed(0)
    W = torch.randn(64, 512) * 0.05
    W[0, ::37] *= 12.0
    bm = quantize_ommx_weight(W, GS, NPV / GS, outlier_repr="bitmap", **QK)
    r7 = quantize_ommx_weight(W, GS, NPV / GS, outlier_repr="relidx7", **QK)
    return W, bm, r7


# ── the sequential walker: same positions as the random-access readers ────────────

def test_the_walker_emits_exactly_the_random_access_positions(packed):
    """``SlotWalker::next()`` on its r-th call must equal ``idx_slot_pos(fmt, blk, r, B)``
    for BOTH encodings: that identity is what makes the walker-based kernels produce the
    same (k, delta) per slot -- and the same fp32 FMA order -- as the slot-indexed ones."""
    W, bm, r7 = packed
    N, K = W.shape
    G = K // GS
    from ommx_gpu_serve.attention.codec import unpack_relidx7

    ib = bitmap_index_bytes(GS)
    r7b = relidx7_index_bytes(NPV)
    oi_bm = bm["oindex"].reshape(N, G, ib)
    oi_r7 = r7["oindex"].reshape(N, G, r7b)
    for n in range(0, N, 5):
        for g in range(G):
            blk_bm = bytes(oi_bm[n, g].tolist())
            blk_r7 = bytes(oi_r7[n, g].tolist())
            want = [_mirror_slot_pos(blk_bm, ib, s) for s in range(NPV)]
            assert _mirror_walker(blk_bm, ib, "bitmap", NPV) == want, f"bitmap n={n} g={g}"
            assert _mirror_walker(blk_r7, r7b, "relidx7", NPV) == \
                unpack_relidx7(blk_r7, NPV), f"relidx7 n={n} g={g}"
            assert want == unpack_relidx7(blk_r7, NPV)   # the two encodings agree


@pytest.mark.parametrize("gs,npv", [(16, 4), (32, 8), (64, 4), (64, 16), (64, 32),
                                    (128, 8), (128, 32)])
def test_the_walker_handles_padding_and_every_word_boundary(gs, npv):
    """Blocks with FEWER than npv set bits (the bitmap padding case) and populations that
    straddle 32-bit words; the relidx7 register must never read past ceil(7*npv/8)."""
    ib = bitmap_index_bytes(gs)
    for pop in (0, 1, npv // 2, npv):
        positions = sorted(set(int(p) for p in torch.randperm(gs)[:pop].tolist()))
        raw = bytearray(ib)
        for p in positions:
            raw[p >> 3] |= 1 << (p & 7)
        got = _mirror_walker(bytes(raw), ib, "bitmap", npv)
        want = [_mirror_slot_pos(bytes(raw), ib, r) for r in range(npv)]
        assert got == want == positions + [-1] * (npv - len(positions))
    # relidx7: npv arbitrary 7-bit values, LSB-first, byte padded
    vals = [int(v) for v in torch.randint(0, gs, (npv,)).tolist()]
    bits = 0
    for s, v in enumerate(vals):
        bits |= v << (7 * s)
    r7b = relidx7_index_bytes(npv)
    blk = bytes((bits >> (8 * i)) & 0xFF for i in range(r7b))
    assert _mirror_walker(blk, r7b, "relidx7", npv) == vals


@pytest.mark.parametrize("vl", [4, 8, 12, 16, 32, 48, 64, 96, 128])
def test_group_index_shift_matches_the_division_for_every_group_size(vl):
    """``nc::group_of(k, VL, group_shift(VL))`` == ``k // VL``: one shift for the power-of-
    two sizes, the exact division kept for the others (the format allows any multiple of 4)."""
    sh = (vl.bit_length() - 1) if (vl & (vl - 1)) == 0 else -1   # 31 - __clz(VL), or -1
    for k in range(0, 4 * vl * 3 + 1):
        got = (k >> sh) if sh >= 0 else (k // vl)
        assert got == k // vl


# ── the arithmetic ─────────────────────────────────────────────────────────────────

def test_the_mirror_recovers_exactly_the_positions_the_packer_chose(packed):
    """The whole reader rests on one identity: the r-th SET BIT of the mask is the r-th
    outlier in ascending position order, which is the order the FP4 nibbles are written
    in. If that fails, every nibble is paired with the wrong weight."""
    W, bm, r7 = packed
    N, K = W.shape
    G = K // GS
    ib = bitmap_index_bytes(GS)
    oi = bm["oindex"].reshape(N, G, ib)
    ref = r7["oindex"]                       # relidx7 stream of the SAME position set
    from ommx_gpu_serve.attention.codec import unpack_relidx7

    r7b = relidx7_index_bytes(NPV)
    ref = ref.reshape(N, G, r7b)
    for n in range(0, N, 7):                 # stride: 64x8 frames is enough signal
        for g in range(0, G, 3):
            blk = bytes(oi[n, g].tolist())
            want = unpack_relidx7(bytes(ref[n, g].tolist()), NPV)
            got = [_mirror_slot_pos(blk, ib, s) for s in range(NPV)]
            assert got == want, f"n={n} g={g}: mirror {got} != packed {want}"


def test_a_slot_past_the_masks_population_returns_the_sentinel():
    """A block with fewer real outliers than npv is what the -1 exists for. Without it,
    the kernel's ``k = bi*B + local`` would be one short of the block base -- in range,
    and decoding a nibble that belongs to another slot."""
    blk = bytes([0b0000_0101, 0, 0, 0, 0, 0, 0, 0])   # two set bits at 0 and 2
    assert _mirror_slot_pos(blk, 8, 0) == 0
    assert _mirror_slot_pos(blk, 8, 1) == 2
    assert _mirror_slot_pos(blk, 8, 2) == -1
    assert _mirror_slot_pos(blk, 8, 9) == -1


@pytest.mark.parametrize("gs", [16, 32, 64, 128])
def test_the_mirror_spans_word_boundaries(gs):
    """Groups of 16 and 32 fit one word; 64 and 128 need two and four. The multi-word arm
    carries the running popcount across words and is the half that a single-word test
    would never reach."""
    ib = bitmap_index_bytes(gs)
    positions = list(range(0, gs, max(1, gs // 9)))[:9]
    raw = bytearray(ib)
    for p in positions:
        raw[p >> 3] |= 1 << (p & 7)
    got = [_mirror_slot_pos(bytes(raw), ib, r) for r in range(len(positions))]
    assert got == positions
    assert _mirror_slot_pos(bytes(raw), ib, len(positions)) == -1


# ── the byte budget, which is the reason to do this at all ─────────────────────────

@pytest.mark.parametrize("gs,npv,cheaper", [
    (64,  16, "bitmap"),    # optimized-weight, 25%  -- 8 B against relidx7's 14
    (64,   4, "relidx7"),   # paper-weight,   6.25%  -- 4 B against the mask's 8
    (16,   4, "bitmap"),    # 25% again, at a fine group
    (128, 32, "bitmap"),    # 25% again, at the widest group
    (128,  8, "relidx7"),   # 6.25% at a wide group
])
def test_which_encoding_is_cheaper_where(gs, npv, cheaper):
    r7 = outlier_index_bytes(npv, gs, "relidx7")
    bm = outlier_index_bytes(npv, gs, "bitmap")
    assert ("bitmap" if bm < r7 else "relidx7") == cheaper, f"relidx7 {r7} B, mask {bm} B"


def test_the_shipped_recipe_gets_cheaper_by_three_quarters_of_a_bit():
    """The concrete win: optimized-weight is 25% coverage at group 64, where the mask is
    8 B/group against relidx7's 14."""
    r7 = Recipe(GS, NPV, NPV / GS, "relidx7", "idx_range", torch.bfloat16)
    bm = Recipe(GS, NPV, NPV / GS, "bitmap", "idx_range", torch.bfloat16)
    assert r7.bits_breakdown()["bits_per_weight"] == 6.1250
    assert bm.bits_breakdown()["bits_per_weight"] == 5.3750
    # and the NPU basis is unmoved, because it re-derives the position set either way
    assert (r7.bits_breakdown()["bits_per_weight_npu"]
            == bm.bits_breakdown()["bits_per_weight_npu"])


# ── what the loader now does with each bundle ──────────────────────────────────────

def test_both_encodings_serve_without_a_transcode():
    assert set(KERNEL_OUTLIER_REPRS) == {"relidx7", "bitmap"}
    for repr_ in ("relidx7", "bitmap"):
        r = Recipe(GS, NPV, NPV / GS, repr_, "idx_range", torch.bfloat16)
        assert plan_transcode(r) is None, f"{repr_} should need no transcode"


def test_a_cheaper_layout_is_never_by_itself_a_reason_to_rewrite():
    """relidx7 at 25% coverage is the dearer encoding, but the kernel reads it, so the
    bundle must still load untouched. Turning an optimisation into a load-time refusal
    would be a regression wearing an improvement's clothes."""
    dear = Recipe(GS, NPV, NPV / GS, "relidx7", "idx_range", torch.bfloat16)
    assert plan_transcode(dear) is None


def test_a_rewrite_that_must_happen_anyway_takes_the_cheaper_encoding():
    """paper-weight needs its range map materialised regardless. Since the planes are
    being rewritten, the positions go to the cheaper encoding at the same time -- which
    at 6.25% coverage is relidx7, so the resident figure stays 4.1250 rather than the
    4.6250 that keeping the mask would have cost."""
    paper = Recipe(GS, 4, 0.0625, "bitmap", "none", torch.bfloat16)
    plan = plan_transcode(paper)
    assert plan is not None
    assert plan.on_disk_bits_per_weight == 3.6250
    assert plan.resident_bits_per_weight == 4.1250
    assert any("oindex" in s for s in plan.steps)
    assert any("map_scale" in s for s in plan.steps)
