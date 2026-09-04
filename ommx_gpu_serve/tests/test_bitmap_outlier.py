# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Gate the FLAT-BITMASK outlier-position storage (``outlier_repr="bitmap"``).

WHAT THIS EXISTS TO PROVE. The ICCAD paper says of the GPU implementation: *"positions
are stored as a flat bitmask (N bits per group), enabling simple decoding at the cost of
higher metadata overhead"*, and Fig 6 draws an "Outlier Bitmask" inside both the K-Cache
and V-Cache blocks. The release used to STORE relidx7 (7 bits per outlier slot) and merely
DERIVE a bitmap in kernel registers, so the described storage did not exist. ``k_obmp``
is that storage. These gates check the two things that make it a real format rather than
a renaming:

  1. MEMBERSHIP EQUIVALENCE (paper item B3, the platform-invariant-format claim). relidx7,
     combinadic and bitmap are three ENCODINGS of one logical format. Same input, same k
     => same outlier SET and byte-identical dequantized K and V. If this fails, they are
     not encodings of one format and the paper's B3 claim does not hold for this repo.
  2. THE RANK RULE the kernel decode depends on: a set bit's ordinal in the shared
     ``k_oval`` FP4 nibble stream equals the POPCOUNT OF ALL LOWER SET BITS. The whole
     O(1) decode (one bit-test for membership, one masked prefix-popcount for the value
     index) is that identity; if the packer ever wrote values in another order the kernel
     would read a neighbouring outlier's nibble and nothing else would notice.

Plus the guardrails: the plane is exactly ``ceil(VL/8)`` bytes with popcount == k, the
DEFAULT (relidx7) planes are unchanged, the bit accounting matches an independent
longhand recomputation, and the encodings that cannot be represented raise loudly.

SCOPE — CPU vs GPU. Everything above is pure storage/codec arithmetic and runs on CPU
with no vLLM, no triton and no GPU. The KERNEL branch that consumes ``k_obmp``
(``paged_decode.py``'s ``BITMAP_READ`` arm) is UNVERIFIED: it has never executed on a
device. Its equivalence gate lives here too, ``gpu``-marked, so it SKIPS — visibly, never
passes — on a host without CUDA (``conftest.py``).
"""
from __future__ import annotations

import math

import pytest
import torch

from ommx_gpu_serve.attention import codec
from ommx_gpu_serve.attention.kv_pool import MultiSeqKVPool
from ommx_gpu_serve.attention.kv_store import CanonicalKVStore
from ommx_gpu_serve.attention.kv_window import WindowSpec
from ommx_gpu_serve.attention.pack import (
    dequant_kv_canonical,
    ommx_pack_kv_canonical_block,
)
from ommx_gpu_serve.integration.vllm.packed_only import kv_bits_breakdown

# The canonical published recipe's shape knobs (MEASURED_FACTS section 5 /
# test_bit_accounting.py): i2f4, k=6, gt=gc=32, pow2 -> int8 scale, dedicated FP4 map ON.
RECIPE = dict(
    outliers_per_vector=6,
    outlier_select="signed",
    k_format="i2f4",
    use_pow2=True,
    kv_outlier_map=True,
    kv_int8_scale=True,
    group_tokens=32,
    group_channels=32,
    page_size=16,
)
REPRS = ("relidx7", "combinadic", "bitmap")


def _kv(T: int = 64, H: int = 2, D: int = 64, seed: int = 20260826):
    """A deterministic bf16 K/V block with a few deliberate heavy-tailed lanes.

    The outlier select must have something to select: a pure-gaussian block makes the
    top-k an arbitrary tie-break, which would make "the three reprs agree" a much weaker
    statement than it looks. The spikes give every (channel, group) vector unambiguous
    winners, so an encoding that lost a position would change the SET, not just the order.
    """
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(T, H, D, generator=g)
    V = torch.randn(T, H, D, generator=g)
    spike = torch.randn(T, H, D, generator=g).abs() * 12.0
    take = torch.rand(T, H, D, generator=g) < 0.10
    K = torch.where(take, spike * torch.sign(K), K)
    return K.to(torch.bfloat16), V.to(torch.bfloat16)


def _pack(repr_: str, K=None, V=None, **over):
    if K is None:
        K, V = _kv()
    kw = dict(RECIPE)
    kw.update(over)
    return ommx_pack_kv_canonical_block(K, V, outlier_repr=repr_, **kw)


def _index_plane(planes: dict):
    """(name, tensor) of the ONE outlier-position plane this pack allocated."""
    live = [(n, planes.get(n)) for n in ("k_oidx", "k_crank", "k_obmp")
            if planes.get(n) is not None]
    assert len(live) == 1, (
        f"expected exactly ONE outlier-index plane, got {[n for n, _ in live]}")
    return live[0]


def _positions_from_planes(planes: dict) -> torch.Tensor:
    """Ascending outlier positions [G,H,D,k], decoded from whichever plane is stored.

    Deliberately routed through the SHIPPED per-repr decoders (the same ones
    ``dequant_kv_canonical`` uses), because the claim under test is that those decoders
    agree — not that a test-local reimplementation of one of them does.
    """
    from ommx_gpu_serve.attention import pack as _p

    k = int(planes["outliers_per_vector"])
    gt = int(planes["group_tokens"])
    G, H, D = (int(s) for s in (planes["n_groups"], planes["n_kv_heads"],
                                planes["head_dim"]))
    repr_ = planes["outlier_repr"]
    if repr_ == "relidx7":
        return _p._unpack_relidx7_streams(planes["k_oidx"], k)
    if repr_ == "combinadic":
        fb = int(planes["k_crank"].shape[-1])
        return _p._unpack_combinadic_frames(
            planes["k_crank"].reshape(G * H * D, 1, fb), k, vl=gt).reshape(G, H, D, k)
    return _p._unpack_bitmap_frames(planes["k_obmp"], k, vl=gt)


# ════════════════════════════════════════════════════════════════════════════
# 1. membership equivalence — the whole correctness claim (paper item B3)
# ════════════════════════════════════════════════════════════════════════════

def test_three_reprs_select_the_same_outlier_set() -> None:
    """Same input, same k => the same outlier POSITIONS, whatever the encoding."""
    K, V = _kv()
    sets = {r: _positions_from_planes(_pack(r, K, V)) for r in REPRS}
    base = sets["relidx7"]
    for r in ("combinadic", "bitmap"):
        # positions are stored ascending by every packer, so equality is elementwise;
        # compare as SETS too so a hypothetical ordering change reports as ordering.
        assert torch.equal(torch.sort(sets[r], dim=-1)[0], torch.sort(base, dim=-1)[0]), (
            f"{r} selects a different outlier SET than relidx7 at the same k")
        assert torch.equal(sets[r], base), (
            f"{r} stores the same set as relidx7 but in a different order; the shared "
            "k_oval nibble stream is ascending-position order, so this would "
            "desynchronize the values")


@pytest.mark.parametrize("k,gt", [(6, 32), (1, 32), (3, 16), (8, 64), (13, 128)])
def test_three_reprs_dequantize_bit_identically(k: int, gt: int) -> None:
    """The decoded K and V are BYTE-identical across the three encodings.

    Bit-exact, not "close": these are three ways of writing down the same integers, so
    any difference at all is a bug in one of the codecs, never rounding. Swept across the
    density regimes that matter — k/gt from 3.1% (below the 1/7 relidx7 crossover, where
    the bitmap is the more expensive choice) to 18.8% (the canonical recipe) to 20%.
    """
    K, V = _kv(T=2 * gt, D=64)
    out = {r: dequant_kv_canonical(
        _pack(r, K, V, outliers_per_vector=k, group_tokens=gt)) for r in REPRS}
    kb, vb = out["relidx7"]
    for r in ("combinadic", "bitmap"):
        kr, vr = out[r]
        assert torch.equal(kr, kb), (
            f"k={k} gt={gt}: {r} dequant differs from relidx7 "
            f"(max |diff| = {float((kr - kb).abs().max()):.3e})")
        assert torch.equal(vr, vb), f"k={k} gt={gt}: {r} V differs from relidx7"
    # ... and the outlier lanes are actually EXERCISED: a splice that never fired would
    # make all three agree trivially. The dequantized K must differ from the base-only
    # reconstruction somewhere.
    base_only = dequant_kv_canonical(
        _pack("relidx7", K, V, outliers_per_vector=0, group_tokens=gt))[0]
    assert not torch.equal(kb, base_only), (
        f"k={k} gt={gt}: the outlier splice changed nothing, so this comparison "
        "proves nothing about outlier handling")


# ════════════════════════════════════════════════════════════════════════════
# 2. the plane: size, popcount, and the popcount-rank identity
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("k,gt", [(6, 32), (1, 32), (3, 16), (8, 64), (13, 128)])
def test_bitmap_plane_is_ceil_vl_over_8_bytes_with_popcount_k(k: int, gt: int) -> None:
    """``ceil(VL/8)`` bytes per frame — sized by the GROUP, not by k — popcount == k."""
    K, V = _kv(T=2 * gt, D=64)
    planes = _pack("bitmap", K, V, outliers_per_vector=k, group_tokens=gt)
    name, plane = _index_plane(planes)
    assert name == "k_obmp"
    G, H, D = (int(planes[x]) for x in ("n_groups", "n_kv_heads", "head_dim"))
    assert plane.dtype == torch.uint8
    assert tuple(plane.shape) == (G, H, D, (gt + 7) // 8), (
        f"k={k} gt={gt}: bitmap frame must be ceil({gt}/8) = {(gt + 7) // 8} bytes; "
        f"got shape {tuple(plane.shape)}")
    # FLATNESS: the plane size must not depend on k at all. That is the defining
    # property of this encoding and the reason its bit rate is constant.
    other_k = 1 if k != 1 else 2
    alt = _pack("bitmap", K, V, outliers_per_vector=other_k, group_tokens=gt)["k_obmp"]
    assert alt.shape == plane.shape, (
        f"gt={gt}: bitmap frame size moved when k went {k} -> {other_k}; a flat bitmask "
        "must be ceil(VL/8) bytes at every k")
    # popcount of every frame == k (one bit per selected position, no duplicates).
    bits = ((plane.to(torch.int64).unsqueeze(-1) >> torch.arange(8)) & 1)
    pc = bits.reshape(G, H, D, -1).sum(-1)
    assert int(pc.min()) == k and int(pc.max()) == k, (
        f"k={k} gt={gt}: frame popcounts span [{int(pc.min())}, {int(pc.max())}], "
        f"expected exactly {k} set bits in every frame")
    # and no bit is set past the group's last position (the padding bits of the last byte)
    tail = bits.reshape(G, H, D, -1)[..., gt:]
    assert int(tail.sum()) == 0, (
        f"gt={gt}: bits are set beyond position {gt - 1} (byte padding must stay zero, "
        "or the kernel's masked prefix-popcount over-counts)")


@pytest.mark.parametrize("k,gt", [(6, 32), (8, 64), (13, 128)])
def test_nibble_ordinal_equals_prefix_popcount(k: int, gt: int) -> None:
    """THE RANK RULE the kernel relies on, checked against the actual stored nibbles.

    For every set bit p of every frame: the FP4 code stored for position p sits at nibble
    ``popcount(all set bits < p)`` of ``k_oval``. This is checked by re-deriving the rank
    from the raw bitmap bytes with an independent prefix-popcount (NOT by calling the
    unpacker under test) and confirming the nibble it points at is the same code the
    relidx7 pack — a completely different index encoding — stored for that position.
    """
    K, V = _kv(T=2 * gt, D=64)
    bm_planes = _pack("bitmap", K, V, outliers_per_vector=k, group_tokens=gt)
    rl_planes = _pack("relidx7", K, V, outliers_per_vector=k, group_tokens=gt)
    G, H, D = (int(bm_planes[x]) for x in ("n_groups", "n_kv_heads", "head_dim"))
    plane = bm_planes["k_obmp"].reshape(-1, (gt + 7) // 8)
    oval = bm_planes["k_oval"].reshape(-1, (k + 1) // 2)
    assert torch.equal(bm_planes["k_oval"], rl_planes["k_oval"]), (
        "the FP4 value stream must be SHARED verbatim by the two reprs; only the index "
        "plane may differ")
    # relidx7 positions, decoded independently, give the (position -> nibble slot) truth.
    from ommx_gpu_serve.attention import pack as _p
    rl_pos = _p._unpack_relidx7_streams(rl_planes["k_oidx"], k).reshape(-1, k)

    rows = [0, 1, 7, plane.shape[0] // 2, plane.shape[0] - 1]      # a sample of frames
    for row in rows:
        blob = bytes(plane[row].tolist())
        set_bits = codec.unpack_bitmap_row(blob, gt)               # ascending, pure python
        assert len(set_bits) == k
        for p in set_bits:
            # independent prefix popcount over the raw bytes
            rank = sum(1 for q in range(p) if (blob[q >> 3] >> (q & 7)) & 1)
            nib = (int(oval[row, rank // 2]) >> (4 * (rank % 2))) & 0xF
            # the same position under relidx7 sits at slot `slot`; its nibble must match
            slot = int((rl_pos[row] == p).nonzero()[0])
            nib_rl = (int(oval[row, slot // 2]) >> (4 * (slot % 2))) & 0xF
            assert rank == slot, (
                f"row {row} position {p}: prefix-popcount rank {rank} != relidx7 slot "
                f"{slot}; the kernel would read the wrong nibble")
            assert nib == nib_rl


def test_pack_bitmap_rows_matches_the_pure_python_row_codec() -> None:
    """The vectorized frame packer and the scalar ``pack_bitmap_row`` agree byte for byte.

    ``pack_bitmap_rows`` is what the KV packer calls (it must be vectorized and
    device-clean); ``pack_bitmap_row`` is the plain-python definition of the layout. They
    are two implementations of one spec, so they get cross-checked rather than trusted.
    """
    g = torch.Generator().manual_seed(7)
    for n in (16, 32, 64, 128):
        for k in (1, 3, 6):
            cols = torch.stack([torch.randperm(n, generator=g)[:k].sort()[0]
                                for _ in range(11)])
            got = codec.pack_bitmap_rows(cols, n)
            assert got.shape == (11, (n + 7) // 8)
            for r in range(11):
                want = codec.pack_bitmap_row(cols[r].tolist(), n)
                assert bytes(got[r].tolist()) == want, f"n={n} k={k} row={r}"
            # round-trip
            back = codec.unpack_bitmap_rows(got, n, k)
            assert torch.equal(back, cols.to(torch.int64))


def test_bitmap_refuses_what_it_cannot_represent() -> None:
    """No silent fallback: an empty slot or a duplicate position RAISES.

    A flat bitmask has no empty-slot sentinel (relidx7 pads with a duplicate of a real
    slot, which is harmless under its first-match-wins splice) and a duplicate collapses
    to ONE bit while ``k_oval`` still carries k nibbles — which would silently shift every
    popcount-rank after it. Both must be loud.
    """
    with pytest.raises(ValueError, match="must lie in"):
        codec.pack_bitmap_rows(torch.tensor([[0, 5, -1]]), 32)
    with pytest.raises(ValueError, match="lost a position|DISTINCT"):
        codec.pack_bitmap_rows(torch.tensor([[0, 5, 5]]), 32)
    with pytest.raises(ValueError, match="must lie in"):
        codec.pack_bitmap_rows(torch.tensor([[0, 5, 32]]), 32)
    # a frame whose stored popcount is short of k must not decode to garbage
    short = codec.pack_bitmap_rows(torch.tensor([[0, 5, 9]]), 32)
    with pytest.raises(ValueError, match="fewer than k"):
        codec.unpack_bitmap_rows(short, 32, 4)
    # and an unknown repr name never resolves to a default
    with pytest.raises(ValueError, match="outlier_repr"):
        _pack("bitmaps")


# ════════════════════════════════════════════════════════════════════════════
# 3. the DEFAULT is untouched
# ════════════════════════════════════════════════════════════════════════════

def test_default_recipe_planes_are_byte_identical_to_explicit_relidx7() -> None:
    """Not asking for bitmap must produce exactly the planes it always produced.

    Two independent checks, because there is no pre-change tree in this checkout to diff
    against:

      A. the DEFAULT ``outlier_repr`` is still ``relidx7`` and its planes are
         byte-identical to an explicit ``outlier_repr="relidx7"`` pack, with NO bitmap
         plane allocated anywhere;
      B. the relidx7 index stream is reconstructed LONGHAND here — outlier positions
         re-derived from the pack's own decoded set, re-encoded with the untouched
         pure-python ``codec.pack_relidx7`` — and must equal the stored bytes. That is a
         golden the bitmap work cannot have moved, since it shares no code with it.
    """
    K, V = _kv()
    kw = dict(RECIPE)
    default = ommx_pack_kv_canonical_block(K, V, **kw)             # no outlier_repr=
    explicit = _pack("relidx7", K, V)
    assert default["outlier_repr"] == "relidx7", "the default repr must not have moved"
    assert default["k_obmp"] is None and default["k_crank"] is None
    for name, t in default.items():
        if not isinstance(t, torch.Tensor):
            continue
        assert torch.equal(t, explicit[name]), f"default plane {name} is not relidx7's"

    # (B) longhand relidx7 reconstruction from the decoded positions.
    k = int(default["outliers_per_vector"])
    pos = _positions_from_planes(default).reshape(-1, k)
    stored = default["k_oidx"].reshape(-1, (7 * k + 7) // 8)
    for row in (0, 3, stored.shape[0] // 2, stored.shape[0] - 1):
        want = codec.pack_relidx7(pos[row].tolist())
        assert bytes(stored[row].tolist()) == want, (
            f"row {row}: the stored relidx7 bitstream is not what codec.pack_relidx7 "
            "produces for the same positions")


def test_stores_default_to_relidx7_and_allocate_only_on_request() -> None:
    """``CanonicalKVStore`` / ``MultiSeqKVPool`` allocate exactly one index plane."""
    win = WindowSpec(sink_tokens=8, recent_window=32, group_tokens=32, page_size=16)
    common = dict(head_dim=64, n_kv_heads=2, outliers_per_vector=6,
                  outlier_select="signed", use_pow2=True, kv_outlier_map=True,
                  window=win, group_channels=32, device="cpu")
    for repr_, want, others in (
            ("relidx7", "k_oidx", ("k_crank", "k_obmp")),
            ("combinadic", "k_crank", ("k_oidx", "k_obmp")),
            ("bitmap", "k_obmp", ("k_oidx", "k_crank"))):
        for obj in (CanonicalKVStore(max_seq_len=256, outlier_repr=repr_, **common),
                    MultiSeqKVPool(num_seqs=2, max_seq_len=256, outlier_repr=repr_,
                                   **common)):
            tag = f"{type(obj).__name__}/{repr_}"
            assert getattr(obj, want) is not None, f"{tag}: {want} not allocated"
            for o in others:
                assert getattr(obj, o) is None, f"{tag}: {o} allocated too"
            if repr_ == "bitmap":
                assert obj.k_obmp.shape[-1] == 4, f"{tag}: ceil(32/8) = 4 bytes"
    # default (no outlier_repr= at all)
    st = CanonicalKVStore(max_seq_len=256, **common)
    assert st.outlier_repr == "relidx7" and st.k_obmp is None
    with pytest.raises(ValueError, match="outlier_repr"):
        CanonicalKVStore(max_seq_len=256, outlier_repr="flat", **common)


def test_store_round_trip_under_bitmap_matches_the_bulk_pack() -> None:
    """The incremental store path carries the bitmap plane correctly.

    A store fed a prefill block must land the same bytes a single bulk
    ``ommx_pack_kv_canonical_block`` of the same prefix produces — the same property
    ``tests/test_kv_pool_parity.py`` pins for relidx7, re-checked for the new plane so a
    write-path that forgot ``k_obmp`` (silently leaving it zero) cannot pass.
    """
    win = WindowSpec(sink_tokens=8, recent_window=32, group_tokens=32, page_size=16)
    # sink(8) + 3 packable groups + the recent(32) window that stays bf16.
    T = 8 + 32 * 3 + 32
    n_groups = win.num_groups(T)
    assert n_groups == 3, f"window geometry changed: {n_groups} packable groups, not 3"
    K, V = _kv(T=T, H=2, D=64)
    st = CanonicalKVStore(head_dim=64, n_kv_heads=2, max_seq_len=512,
                          outliers_per_vector=6, outlier_select="signed",
                          outlier_repr="bitmap", use_pow2=True, kv_outlier_map=True,
                          window=win, group_channels=32, device="cpu")
    st.append_block(K, V)
    planes = st.decode_planes()
    assert planes["outlier_repr"] == "bitmap"
    assert int(planes["n_groups"]) == 3
    assert planes["k_obmp"] is not None and int(planes["k_obmp"].abs().sum()) > 0, (
        "the store wrote an all-zero bitmap plane (popcount 0 == 'no outliers'), which "
        "is exactly the miss/zero aliasing a shape-only check would not catch")
    bulk = _pack("bitmap", K[8:8 + 32 * n_groups], V[8:8 + 32 * n_groups])
    assert torch.equal(planes["k_obmp"], bulk["k_obmp"])
    assert torch.equal(planes["k_oval"], bulk["k_oval"])
    # and it decodes to the same K/V the relidx7 store would produce
    st_r = CanonicalKVStore(head_dim=64, n_kv_heads=2, max_seq_len=512,
                            outliers_per_vector=6, outlier_select="signed",
                            outlier_repr="relidx7", use_pow2=True, kv_outlier_map=True,
                            window=win, group_channels=32, device="cpu")
    st_r.append_block(K, V)
    kb, vb = dequant_kv_canonical(st_r.decode_planes())
    kx, vx = dequant_kv_canonical(planes)
    assert torch.equal(kx, kb) and torch.equal(vx, vb)


def _pool_common():
    """The CPU pool geometry both pool gates below share (canonical shape knobs)."""
    win = WindowSpec(sink_tokens=8, recent_window=32, group_tokens=32, page_size=16)
    return win, dict(head_dim=64, n_kv_heads=2, outliers_per_vector=6,
                     outlier_select="signed", use_pow2=True, kv_outlier_map=True,
                     window=win, group_channels=32, device="cpu")


def _assert_pool_bitmap_plane_is_live(planes: dict, k: int, gt: int) -> None:
    """The pool's k_obmp really carries k bits per frame — not zeros, not a partial write.

    ADDED BY THE RE-CHECK. ``!= 0`` alone is too weak: a write that lands for SOME groups
    (the batched device regroup writes a slot RANGE) leaves the untouched frames at
    popcount 0, and popcount 0 is indistinguishable from "this frame has no outliers" to
    every shape-only check. So assert the popcount of EVERY frame, which is the
    miss/zero-aliasing trap stated positively.
    """
    plane = planes["k_obmp"]
    assert plane is not None, "the pool allocated no k_obmp plane"
    bits = ((plane.to(torch.int64).unsqueeze(-1) >> torch.arange(8)) & 1)
    pc = bits.reshape(*plane.shape[:-1], -1)
    assert int(pc[..., gt:].sum()) == 0, "bits set past the group's last position"
    pc = pc.sum(-1)
    assert int(pc.min()) == k and int(pc.max()) == k, (
        f"pool frame popcounts span [{int(pc.min())}, {int(pc.max())}], expected exactly "
        f"{k}; a frame at 0 is a write the pool never made (miss/zero aliasing)")


def test_pool_prefill_under_bitmap_matches_the_bulk_pack() -> None:
    """``MultiSeqKVPool`` prefill carries k_obmp — the BATCHED path the serving path uses.

    ADDED BY THE RE-CHECK. ``test_store_round_trip_under_bitmap_matches_the_bulk_pack``
    covers ``CanonicalKVStore`` only, but the vLLM serving path builds a
    ``MultiSeqKVPool`` (``integration/vllm/metadata.py`` passes ``outlier_repr`` to it),
    and the pool has its OWN two bitmap write sites. Deleting BOTH of them left the whole
    suite green before this gate existed.
    """
    win, common = _pool_common()
    T = 8 + 32 * 3 + 32
    K, V = _kv(T=T, H=2, D=64)
    pool = MultiSeqKVPool(num_seqs=2, max_seq_len=512, outlier_repr="bitmap", **common)
    n = pool.append_block(1, K, V)          # request 1, not 0: exercises the slot offset
    assert n == 3, f"window geometry changed: {n} packable groups, not 3"
    planes = pool.request_planes(1)
    assert planes["outlier_repr"] == "bitmap"
    _assert_pool_bitmap_plane_is_live(planes, k=6, gt=32)
    bulk = _pack("bitmap", K[8:8 + 32 * n], V[8:8 + 32 * n])
    assert torch.equal(planes["k_obmp"], bulk["k_obmp"])
    assert torch.equal(planes["k_oval"], bulk["k_oval"])
    # ... and it decodes to exactly what the relidx7 pool decodes.
    ref = MultiSeqKVPool(num_seqs=2, max_seq_len=512, outlier_repr="relidx7", **common)
    ref.append_block(1, K, V)
    kb, vb = dequant_kv_canonical(ref.request_planes(1))
    kx, vx = dequant_kv_canonical(planes)
    assert torch.equal(kx, kb) and torch.equal(vx, vb)


def test_pool_batched_decode_regroup_writes_the_bitmap() -> None:
    """The DEVICE batched regroup (``_regroup_pack_batched_device``) writes k_obmp too.

    ADDED BY THE RE-CHECK. This is a SECOND, separate write site from the prefill one
    above (``append_decode_batched`` -> ``_regroup_pack_batched_device``), and it is the
    one that fires every ``group_tokens`` decode steps for every request in a batch — the
    steady-state serving path. Stepping past a group boundary and then requiring the new
    group's frames to have popcount k is what makes a forgotten write fail here.
    """
    win, common = _pool_common()
    T = 8 + 32 * 2 + 32                      # 2 packed groups after prefill
    K, V = _kv(T=T, H=2, D=64, seed=4242)
    step_k, step_v = _kv(T=80, H=2, D=64, seed=99)   # 40 steps x B=2 rows
    out = {}
    for repr_ in ("relidx7", "bitmap"):
        pool = MultiSeqKVPool(num_seqs=2, max_seq_len=512, outlier_repr=repr_, **common)
        pool.append_block(0, K, V)
        pool.append_block(1, K, V)
        g0 = pool.packed_groups[0]
        packed = 0
        for t in range(40):                  # 40 steps > gt=32 -> at least one boundary
            packed += pool.append_decode_batched(
                [0, 1], step_k[2 * t:2 * t + 2], step_v[2 * t:2 * t + 2])
        assert packed > 0, "no group boundary crossed; this test proves nothing"
        assert pool.packed_groups[0] > g0
        out[repr_] = pool.request_planes(0)
    _assert_pool_bitmap_plane_is_live(out["bitmap"], k=6, gt=32)
    assert out["bitmap"]["k_oidx"] is None and out["relidx7"]["k_obmp"] is None
    kb, vb = dequant_kv_canonical(out["relidx7"])
    kx, vx = dequant_kv_canonical(out["bitmap"])
    assert torch.equal(kx, kb) and torch.equal(vx, vb), (
        "the batched device regroup produced a different decode under bitmap storage")


# ════════════════════════════════════════════════════════════════════════════
# 4. bit accounting
# ════════════════════════════════════════════════════════════════════════════

def _longhand_k_bits(*, gt: int, gc: int, k: int, repr_: str, omap: bool,
                     scale_bytes: int) -> float:
    """The K bit rate written out by hand from the plane list — no shared code with
    ``kv_bits_breakdown``, so agreeing with it means something."""
    bits = 2.0                                   # k_base, INT2
    bits += scale_bytes * 8.0 / gt               # k_scale
    bits += 2 * 8.0 / gt                         # k_zp (always bf16)
    if k > 0 and omap:
        bits += 2 * (2 * 8.0 / gt)               # k_fp4_mapscale + k_fp4_mapcenter
    if k > 0:
        if repr_ == "relidx7":
            idx_bytes = math.ceil(7 * k / 8)
        elif repr_ == "bitmap":
            idx_bytes = math.ceil(gt / 8)        # FLAT in k — the defining property
        else:
            idx_bytes = math.ceil(max(1, (math.comb(gt, k) - 1).bit_length()) / 8)
        bits += idx_bytes * 8.0 / gt
        bits += math.ceil(k / 2) * 8.0 / gt      # k_oval, FP4 nibbles
    return bits


@pytest.mark.parametrize("repr_", REPRS)
@pytest.mark.parametrize("k,gt", [(6, 32), (1, 32), (3, 16), (8, 64), (13, 128)])
def test_bit_accounting_matches_longhand(repr_: str, k: int, gt: int) -> None:
    got = kv_bits_breakdown(
        128, k_format="i2f4", group_tokens=gt, group_channels=32,
        outliers_per_vector=k, outlier_repr=repr_, kv_outlier_map=True,
        use_pow2=True, kv_int8_scale=True)
    want = _longhand_k_bits(gt=gt, gc=32, k=k, repr_=repr_, omap=True, scale_bytes=1)
    assert got["k_bits_per_elem"] == pytest.approx(want, rel=0.0, abs=1e-12), (
        f"repr={repr_} k={k} gt={gt}: {got['k_planes']}")
    # the recipe must name the plane the index bits were charged to
    assert got["recipe"]["outlier_index_plane"] == {
        "relidx7": "k_oidx", "combinadic": "k_crank", "bitmap": "k_obmp"}[repr_]


def test_bitmap_is_flat_in_k_and_relidx7_is_not() -> None:
    """1.0 bit/element at every k — the claim that makes the encoding worth having."""
    for gt in (16, 32, 64, 128):
        for k in (1, 2, 5, gt // 4):
            b = kv_bits_breakdown(
                128, k_format="i2f4", group_tokens=gt, group_channels=32,
                outliers_per_vector=k, outlier_repr="bitmap", kv_outlier_map=True,
                use_pow2=True, kv_int8_scale=True)
            assert b["k_planes"]["k_obmp"] == pytest.approx(1.0, abs=1e-12), (
                f"gt={gt} k={k}: the flat bitmask must cost exactly 1.0 bit/element")
            r = kv_bits_breakdown(
                128, k_format="i2f4", group_tokens=gt, group_channels=32,
                outliers_per_vector=k, outlier_repr="relidx7", kv_outlier_map=True,
                use_pow2=True, kv_int8_scale=True)
            # relidx7 = ceil(7k/8)*8/gt, which tracks k. Below the 1/7 density crossover
            # relidx7 is the cheaper plane; above it the bitmap is.
            assert r["k_planes"]["k_oidx"] == pytest.approx(
                math.ceil(7 * k / 8) * 8.0 / gt, abs=1e-12)
            cheaper = "bitmap" if b["k_bits_per_elem"] < r["k_bits_per_elem"] else "relidx7"
            expect = "bitmap" if math.ceil(gt / 8) < math.ceil(7 * k / 8) else "relidx7"
            if math.ceil(gt / 8) != math.ceil(7 * k / 8):
                assert cheaper == expect, f"gt={gt} k={k}: crossover misprediction"


def test_canonical_recipe_delta_is_pinned() -> None:
    """The published number, and exactly what the bitmask does to it.

    relidx7 (the shipped default) is the row MEASURED_FACTS section 5 pins:
    K 6.000 / V 2.750 / avg 4.375 / 3.657x. Swapping ONLY the outlier-position encoding
    to the paper's flat bitmask takes K to 5.500 and the average to 4.125 (3.879x) —
    i.e. the bitmask is CHEAPER here, by 0.5 bit per K element, because k/gt = 6/32 =
    18.8% is above the 1/7 = 14.3% crossover. Values are unchanged (the equivalence gates
    above), so this is a pure storage delta.
    """
    kw = dict(head_dim=128, k_format="i2f4", group_tokens=32, group_channels=32,
              outliers_per_vector=6, kv_outlier_map=True, use_pow2=True,
              kv_int8_scale=True)
    rl = kv_bits_breakdown(outlier_repr="relidx7", **kw)
    bm = kv_bits_breakdown(outlier_repr="bitmap", **kw)
    assert (rl["k_bits_per_elem"], rl["v_bits_per_elem"], rl["avg_bits_per_elem"]) == (
        6.0, 2.75, 4.375)
    assert rl["compression_ratio"] == pytest.approx(3.657, abs=5e-4)
    assert (bm["k_bits_per_elem"], bm["v_bits_per_elem"], bm["avg_bits_per_elem"]) == (
        5.5, 2.75, 4.125)
    assert bm["compression_ratio"] == pytest.approx(3.879, abs=5e-4)
    assert rl["k_bits_per_elem"] - bm["k_bits_per_elem"] == pytest.approx(0.5, abs=1e-12)
    assert rl["v_planes"] == bm["v_planes"], "V is untouched by the K index encoding"


# ════════════════════════════════════════════════════════════════════════════
# 5. the op wiring (CPU) + the kernel equivalence (GPU-only, SKIPS here)
# ════════════════════════════════════════════════════════════════════════════

def test_reference_op_infers_the_index_source_from_the_pack() -> None:
    """``bitmap_read`` defaults to "follow the pack", and never guesses silently."""
    from ommx_gpu_serve.attention.reference_op import _resolve_bitmap_read

    bm = _pack("bitmap")
    rl = _pack("relidx7")
    assert _resolve_bitmap_read(bm, None) is True
    assert _resolve_bitmap_read(rl, None) is False
    assert _resolve_bitmap_read(rl, False) is False
    # asking for a bitmap read of a relidx7 pack must raise, not fall back
    with pytest.raises(ValueError, match="k_obmp"):
        _resolve_bitmap_read(rl, True)


def _kernel_bitmap_arithmetic(blob_bytes: torch.Tensor, vword: int, vl: int,
                              tokens: torch.Tensor):
    """CPU mirror of the Triton ``BITMAP_READ`` arm's register arithmetic.

    Line-for-line the same integer operations ``_bitmap_splice_from_bytes`` performs, in
    torch INT32 so the signed-shift behaviour matches Triton's int32 (a VL=32 group can
    set bit 31, which makes the occupancy word negative — ``(w >> t) & 1`` still selects
    the right bit because the mask discards the sign extension, and
    ``(1 << 31) - 1 == 0x7FFFFFFF`` is still the correct prefix mask).

    This CANNOT verify that the Triton kernel compiles, schedules or addresses memory
    correctly — only a GPU can. What it does verify is the part that is easy to get
    silently wrong and impossible to see in a benchmark: the multi-word byte->occupancy
    extraction, the membership bit-test, and the two-popcount rank.

    Returns ``(membership [T] bool, rank [T] int, value [T] int)``.
    """
    vl_words = (vl + 31) // 32
    fbb = (vl + 7) // 8
    b = blob_bytes.to(torch.int32)                                  # [FBB]
    offs = torch.arange(fbb, dtype=torch.int32)
    t_bit = tokens.to(torch.int32) & 0x1F
    prefix_mask = (torch.ones_like(t_bit) << t_bit) - 1
    if vl_words == 1:
        w_acc0 = int((b.to(torch.int64) << (8 * offs.to(torch.int64))).sum())
        own = torch.full_like(t_bit, torch.tensor(w_acc0, dtype=torch.int64).to(
            torch.int32).item())
        prefix_below = torch.zeros_like(t_bit)
    else:
        t_word = tokens.to(torch.int32) >> 5
        own = torch.zeros_like(t_bit)
        prefix_below = torch.zeros_like(t_bit)
        for w in range(vl_words):
            in_w = (offs >= 4 * w) & (offs < 4 * w + 4)
            sh = torch.where(in_w, (offs - 4 * w) * 8, torch.zeros_like(offs))
            acc = int(torch.where(in_w, b.to(torch.int64) << sh.to(torch.int64),
                                  torch.zeros_like(b, dtype=torch.int64)).sum())
            acc32 = torch.tensor(acc, dtype=torch.int64).to(torch.int32).item()
            pc = bin(acc & 0xFFFFFFFF).count("1")
            own = own + torch.where(t_word == w, torch.full_like(own, acc32),
                                    torch.zeros_like(own))
            prefix_below = prefix_below + torch.where(
                t_word > w, torch.full_like(own, pc), torch.zeros_like(own))
    masked = (own & prefix_mask).to(torch.int64) & 0xFFFFFFFF
    rank = prefix_below.to(torch.int64) + torch.tensor(
        [bin(int(x)).count("1") for x in masked], dtype=torch.int64)
    mem = (((own >> t_bit) & 1) != 0)
    val = torch.tensor([(vword >> (4 * int(r))) & 0xF for r in rank], dtype=torch.int64)
    return mem, rank, val


@pytest.mark.parametrize("gt", [16, 32, 64, 128])
def test_kernel_bitmap_arithmetic_mirror_matches_the_packer(gt: int) -> None:
    """The kernel's byte->occupancy->membership->rank chain, checked on CPU.

    Runs the mirror above over REAL packed frames and asserts it recovers exactly the
    packer's membership set and the packer's FP4 code for every outlier position — the
    same thing ``dequant_kv_canonical`` recovers via a completely different route (a
    python position list). A mismatch here is a bug in the kernel's arithmetic that no
    other CPU gate would catch, since nothing else executes that formula.

    STILL NOT A GPU RESULT: see the module docstring.
    """
    k = 6 if gt >= 32 else 2
    K, V = _kv(T=2 * gt, D=64)
    planes = _pack("bitmap", K, V, outliers_per_vector=k, group_tokens=gt)
    bm = planes["k_obmp"].reshape(-1, (gt + 7) // 8)
    oval = planes["k_oval"].reshape(-1, (k + 1) // 2)
    truth_pos = _positions_from_planes(planes).reshape(-1, k)
    tokens = torch.arange(gt)
    for row in (0, 1, bm.shape[0] // 3, bm.shape[0] - 1):
        vword = int(sum(int(oval[row, i]) << (8 * i) for i in range(oval.shape[1])))
        mem, rank, val = _kernel_bitmap_arithmetic(bm[row], vword, gt, tokens)
        got = set(int(t) for t in tokens[mem])
        want = set(int(p) for p in truth_pos[row])
        assert got == want, f"gt={gt} row={row}: membership {sorted(got)} != {sorted(want)}"
        for slot, p in enumerate(sorted(want)):
            assert int(rank[p]) == slot, (
                f"gt={gt} row={row} pos={p}: prefix-popcount rank {int(rank[p])} != "
                f"ascending slot {slot}")
            nib = (int(oval[row, slot // 2]) >> (4 * (slot % 2))) & 0xF
            assert int(val[p]) == nib, (
                f"gt={gt} row={row} pos={p}: the rank-indexed nibble is {int(val[p])}, "
                f"the packer stored {nib}")


@pytest.mark.gpu
def test_kernel_bitmap_read_matches_relidx7_on_gpu() -> None:
    """UNVERIFIED PATH. The Triton ``BITMAP_READ`` arm reads the STORED ``k_obmp``
    instead of rebuilding an occupancy word from relidx7. Everything else — the FP4
    dequant, the online softmax, the LSE merge — is shared, so the two decodes must
    produce IDENTICAL logits for the same sequence.

    This gate has NEVER RUN: no CUDA device was available when the bitmap storage was
    written, so the kernel branch is unverified. It is ``gpu``-marked precisely so that
    fact stays visible — on a CPU host it SKIPS, it does not pass.
    """
    dev = "cuda"
    K, V = _kv(T=128, H=2, D=64)
    K, V = K.to(dev), V.to(dev)
    q = torch.randn(1, 8, 64, generator=torch.Generator().manual_seed(3)).to(
        dev, torch.bfloat16)
    outs = {}
    for repr_ in ("relidx7", "bitmap"):
        planes = _pack(repr_, K, V, device=dev)
        from ommx_gpu_serve.attention.reference_op import ommx_paged_decode
        n_tok = int(planes["num_tokens"])
        b_seq = torch.tensor([n_tok], dtype=torch.int32, device=dev)
        b_tail = torch.zeros(1, dtype=torch.int32, device=dev)
        tail = torch.zeros(1, 1, 2, 64, dtype=torch.bfloat16, device=dev)
        outs[repr_] = ommx_paged_decode(
            q, planes, tail, tail, b_tail, b_seq, sm_scale=0.125,
            max_seq_len=n_tok, max_tail_len=0)
    (o_r, l_r), (o_b, l_b) = outs["relidx7"], outs["bitmap"]
    assert torch.equal(o_r, o_b), (
        f"stored-bitmap decode differs from relidx7: max |diff| = "
        f"{float((o_r - o_b).abs().max()):.3e}")
    assert torch.equal(l_r, l_b)
