# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The gate ``kv_pool.py`` promises in its own docstring.

``kv_pool.py`` ends with:

    CPU-verifiable: a request's pool planes (gathered via its ``req_to_token`` /
    ``req_to_group``) dequant BIT-EXACT ... to a standalone single-seq
    ``CanonicalKVStore`` packing the same sequence (dev-tree gate
    ``tests/test_kv_pool.py``, not shipped here ...)

This file IS that gate, shipped. It asserts something strictly stronger than the
docstring's "dequant bit-exact": the packed PLANES THEMSELVES are byte-identical
(``torch.equal`` on the raw uint8 / int8 / bf16 tensors), so a divergence cannot be
masked by a dequant that happens to be insensitive to it.

Why this is a CPU test and not a GPU one: ``ommx_pack_kv_canonical_block`` is pure
torch (integer bit-twiddling + affine RTN), so the whole write-pack path runs on
``device="cpu"``. The same parity is re-run on CUDA by the ``gpu``-marked test at
the bottom, which is SKIPPED (not passed) with no device.

What each test is designed to catch:

  * ``test_pool_planes_match_store_prefill`` — a pool/store packer divergence at
    the prefill->decode handoff (the single-slot base case).
  * ``test_pool_planes_match_store_ragged_batch`` — SLOT ALIASING. B=3 with
    DIFFERENT sequence lengths (200 / 71 / 333 + 40 decode steps each) means every
    request sits at a different group count, so an off-by-one in
    ``req * G_per_seq`` / ``req * P_per_seq`` writes one request's groups over
    another's and the byte compare fails for at least one slot.
  * ``test_pool_planes_match_store_ragged_batch_ring`` — the same under
    ``OMMX_KV_RING=1`` (the no-shadow ring), where abs->ring remapping is a second
    chance to alias.
  * ``test_writes_to_one_slot_leave_neighbour_slots_byte_identical`` — the direct
    non-interference claim, checked on the RAW pool buffers (not through the
    tables), plus a non-vacuity assertion that the written slot really did change.
  * ``test_batched_decode_write_matches_the_single_seq_store`` — the SERVED B>1
    decode write seam (``append_decode_batched`` ->
    ``_regroup_pack_batched_device``, what ``backend.py`` actually calls), which
    holds a second independent copy of the ``req*G_per_seq`` / ``req*P_per_seq``
    arithmetic and which the per-request ``append`` tests above never reach.
  * ``test_batched_decode_write_follows_the_slot_list_not_the_row_index`` — the
    same batched write seam driven with PERMUTED / SUBSET slot lists (what
    ``assign_slots`` actually hands ``backend.py:954``), which the identity-list
    test above cannot distinguish from a write keyed on the batch row index.
  * ``test_slot_table_arithmetic_structural`` — the CPU-only structural check of
    the table arithmetic (``G_per_seq`` / ``P_per_seq`` / per-slot index ranges /
    pairwise non-overlap / exact cover / in-bounds), over several geometries. This
    is the check that still runs even if the pack path were ever GPU-gated.
  * ``test_decode_inputs_batched_matches_single_seq_store`` — the read side:
    ``b_seq_len`` / ``b_tail_len`` / the gathered bf16 residual rows per request.
  * ``test_decode_inputs_batched_follows_the_slot_list_not_the_row_index`` — the
    same read side driven with a PERMUTED / SUBSET slot list (what ``assign_slots``
    actually hands it), which the identity call above cannot distinguish from a
    gather keyed on the row index.
"""
from __future__ import annotations

import pytest
import torch

from ommx_gpu_serve.attention.kv_pool import MultiSeqKVPool
from ommx_gpu_serve.attention.kv_store import CanonicalKVStore
from ommx_gpu_serve.attention.kv_window import WindowSpec

# ── the canonical published serving recipe (README / run.sh) ─────────────────────
#   OMMX_ATTN_K_FORMAT=i2f4 OMMX_ATTN_OUTLIERS=6 OMMX_ATTN_POW2=1
#   OMMX_KV_GROUP_TOKENS=32 OMMX_KV_GROUP_CHANNELS=32
#   OMMX_KV_SINK=8 OMMX_KV_RECENT=32
# Passed as explicit kwargs (never via env) so the gate pins the recipe rather than
# inheriting whatever the developer's shell exports; conftest scrubs OMMX_* anyway.
HEAD_DIM = 128
N_KV_HEADS = 2          # Llama-3.1-8B uses 8; 2 keeps the CPU pack fast and is
                        # sufficient — the head axis is a pure batch axis in the packer.
GROUP_TOKENS = 32
GROUP_CHANNELS = 32
SINK = 8
RECENT = 32
PAGE_SIZE = 16
OUTLIERS = 6

# Planes addressed by the PAGE table (``req_to_token``) vs the GROUP table
# (``req_to_group``) — the two independent index spaces the pool must keep separate.
PAGE_PLANES = ("k_base", "v_main", "v_scale", "v_zp")
GROUP_PLANES = ("k_scale", "k_zp", "k_oidx", "k_oval", "k_crank",
                "k_fp4_mapscale", "k_fp4_mapcenter")


def _window() -> WindowSpec:
    return WindowSpec(sink_tokens=SINK, recent_window=RECENT,
                      group_tokens=GROUP_TOKENS, page_size=PAGE_SIZE)


def _recipe(device: str) -> dict:
    """Recipe kwargs common to CanonicalKVStore and MultiSeqKVPool.

    ``kv_outlier_map`` is passed explicitly (its env default is ON) and
    ``kv_int8_scale`` is left to derive from ``use_pow2`` — the tests assert the
    derivation landed on int8 so the recipe under test is unambiguous.
    """
    return dict(
        k_format="i2f4",
        outliers_per_vector=OUTLIERS,
        outlier_select="signed",
        outlier_repr="relidx7",
        use_pow2=True,
        kv_outlier_map=True,
        window=_window(),
        group_channels=GROUP_CHANNELS,
        device=device,
    )


def _make_store(max_seq_len: int, device: str) -> CanonicalKVStore:
    # v_format="i2" is the pool's hard-wired V path (MultiSeqKVPool has no
    # v_format knob), so the single-seq reference must be built the same way.
    return CanonicalKVStore(head_dim=HEAD_DIM, n_kv_heads=N_KV_HEADS,
                            max_seq_len=max_seq_len, v_format="i2",
                            **_recipe(device))


def _make_pool(num_seqs: int, max_seq_len: int, device: str) -> MultiSeqKVPool:
    return MultiSeqKVPool(head_dim=HEAD_DIM, n_kv_heads=N_KV_HEADS,
                          num_seqs=num_seqs, max_seq_len=max_seq_len,
                          **_recipe(device))


def _rand_kv(n_tokens: int, gen: torch.Generator, device: str):
    """Synthetic bf16 K/V ``[T, H, D]``. Generated on CPU with a seeded generator and
    moved, so the CUDA run packs bit-identical INPUT to the CPU run."""
    k = torch.randn(n_tokens, N_KV_HEADS, HEAD_DIM, generator=gen).to(torch.bfloat16)
    v = torch.randn(n_tokens, N_KV_HEADS, HEAD_DIM, generator=gen).to(torch.bfloat16)
    return k.to(device), v.to(device)


def _decode_step(store: CanonicalKVStore, pool: MultiSeqKVPool, req: int,
                 k: torch.Tensor, v: torch.Tensor) -> None:
    """One decode token into both stores, driving each through ITS OWN documented seam.

    ``MultiSeqKVPool.append`` regroups internally; ``CanonicalKVStore.append`` is the
    capture-SAFE stash only and leaves the pack to the host seam
    (``maybe_regroup``) — see kv_store.py's "Design split (the serving contract)".
    Forgetting the store's host seam silently leaves it one group behind the pool.
    """
    store.append(k, v)
    store.maybe_regroup()
    pool.append(req, k, v)


def _full_tables(pool: MultiSeqKVPool):
    """The pool's FULL ``[num_seqs, *]`` routing tables, read from ``pool_planes()``.

    ``pool_planes()`` memoises one dict and ``decode_inputs_batched`` /
    ``decode_inputs_captured`` OVERWRITE its ``req_to_token`` / ``req_to_group``
    entries in place with a ``[B, *]`` gather. So this must be called BEFORE any
    ``decode_inputs_*`` call on the same pool; the shape assertion below makes a
    violation a loud failure instead of a silently wrong gather.
    """
    planes = pool.pool_planes()
    rt = planes["req_to_token"]
    rg = planes["req_to_group"]
    assert tuple(rt.shape) == (pool.num_seqs, pool.P_per_seq), (
        f"req_to_token is {tuple(rt.shape)}, expected the full "
        f"{(pool.num_seqs, pool.P_per_seq)} table — pool_planes() was already "
        f"overwritten by a decode_inputs_* call")
    assert tuple(rg.shape) == (pool.num_seqs, pool.G_per_seq), (
        f"req_to_group is {tuple(rg.shape)}, expected the full "
        f"{(pool.num_seqs, pool.G_per_seq)} table")
    return rt, rg


def _gather_via_tables(pool: MultiSeqKVPool, req: int, n_groups: int,
                       rt: torch.Tensor, rg: torch.Tensor) -> dict:
    """Gather request ``req``'s live planes out of the shared pool THROUGH the tables.

    Deliberately does NOT call ``pool.request_planes`` — that helper is part of the
    code under test, so using it would let a table bug and a gather bug cancel. The
    gather here reads only ``req_to_token`` / ``req_to_group`` (the exact indices the
    kernel is handed) and the raw pool plane tensors.
    """
    n_pages = n_groups * pool.pages_per_group
    gidx = rg[req, :n_groups].to(torch.long)
    pidx = rt[req, :n_pages].to(torch.long)
    out = {}
    for name in PAGE_PLANES:
        t = getattr(pool, name)
        out[name] = None if t is None else t.index_select(0, pidx)
    for name in GROUP_PLANES:
        t = getattr(pool, name)
        out[name] = None if t is None else t.index_select(0, gidx)
    return out


def _assert_planes_identical(ref: dict, got: dict, *, what: str) -> None:
    """Byte-identity over every plane, with a non-vacuity guard.

    ``compared`` counts the planes that actually existed AND carried a non-zero
    byte; a recipe/geometry mistake that produced empty planes on both sides would
    otherwise make this assertion trivially true.
    """
    nonzero = 0
    for name in PAGE_PLANES + GROUP_PLANES:
        a = ref.get(name)
        b = got.get(name)
        if a is None and b is None:
            continue
        assert a is not None and b is not None, (
            f"{what}: plane {name!r} present on one side only "
            f"(store={type(a).__name__}, pool={type(b).__name__})")
        assert a.shape == b.shape, f"{what}: plane {name!r} shape {a.shape} != {b.shape}"
        assert a.dtype == b.dtype, f"{what}: plane {name!r} dtype {a.dtype} != {b.dtype}"
        assert torch.equal(a, b), (
            f"{what}: plane {name!r} is NOT byte-identical "
            f"({int((a != b).sum())} differing elements of {a.numel()})")
        if bool((a != 0).any()):
            nonzero += 1
    assert nonzero >= 6, (
        f"{what}: only {nonzero} planes carried a non-zero byte — the comparison "
        f"is vacuous, the sequence probably packed zero groups")


def _run_parity(device: str, lens, n_decode: int, seed: int) -> None:
    """Pack the SAME synthetic sequences into (a) one standalone single-seq
    ``CanonicalKVStore`` each and (b) the matching slots of ONE ``MultiSeqKVPool``,
    then compare the pool's table-gathered planes to the stores' planes byte-for-byte.
    """
    num_seqs = len(lens)
    max_seq_len = max(lens) + n_decode + GROUP_TOKENS
    gen = torch.Generator().manual_seed(seed)

    pool = _make_pool(num_seqs, max_seq_len, device)
    assert pool.kv_int8_scale is True, (
        "recipe drift: use_pow2=True must resolve kv_int8_scale=True (int8 pow2 "
        "exponent scale); got False")
    rt, rg = _full_tables(pool)     # BEFORE any decode_inputs_* call (see _full_tables)

    stores = []
    for req, seq_len in enumerate(lens):
        k, v = _rand_kv(seq_len, gen, device)
        store = _make_store(max_seq_len, device)
        store.append_block(k, v)
        stores.append(store)
        pool.append_block(req, k, v)

    # Interleave the decode steps across requests (round-robin, exactly how the
    # served batch advances) so a slot that leaked would be overwritten by a
    # NEIGHBOUR's step, not by its own.
    for _ in range(n_decode):
        for req in range(num_seqs):
            k, v = _rand_kv(1, gen, device)
            _decode_step(stores[req], pool, req, k[0], v[0])

    for req in range(num_seqs):
        store = stores[req]
        assert store.seq_len == pool.seq_len[req], (
            f"req {req}: seq_len drift store={store.seq_len} pool={pool.seq_len[req]}")
        assert store.packed_groups == pool.packed_groups[req], (
            f"req {req}: packed_groups drift store={store.packed_groups} "
            f"pool={pool.packed_groups[req]}")
        n_groups = store.packed_groups
        assert n_groups >= 2, (
            f"req {req}: only {n_groups} packed groups — lengthen the fixture, the "
            f"parity check needs several groups to be meaningful")
        _assert_planes_identical(
            store.decode_planes(),
            _gather_via_tables(pool, req, n_groups, rt, rg),
            what=f"req {req} (seq_len={store.seq_len}, groups={n_groups}, {device})")


# ── the parity gate ─────────────────────────────────────────────────────────────

def test_pool_planes_match_store_prefill() -> None:
    """Single slot, prefill only: the base case of the pool == store contract."""
    _run_parity("cpu", lens=[200], n_decode=0, seed=1234)


def test_pool_planes_match_store_ragged_batch() -> None:
    """B=3 with DIFFERENT lengths + interleaved decode: the slot-aliasing gate.

    200 / 71 / 333 prefill tokens land the three requests on different group counts
    (6 / 2 / 10 after 40 decode steps each), so any ``req*G_per_seq`` /
    ``req*P_per_seq`` off-by-one puts one request's groups inside another's block.
    """
    _run_parity("cpu", lens=[200, 71, 333], n_decode=40, seed=7)


def test_pool_planes_match_store_ragged_batch_ring(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Same, under ``OMMX_KV_RING=1`` (the no-shadow bf16 ring).

    The ring adds a second aliasing surface: abs token position -> ring row is
    ``sink + (pos-sink) % Rrec`` per request, and the batched gathers compose it
    with a ``slot*S_ring`` base. Read at construction time, hence setenv first.
    """
    monkeypatch.setenv("OMMX_KV_RING", "1")
    _run_parity("cpu", lens=[200, 71, 333], n_decode=40, seed=99)


def test_writes_to_one_slot_leave_neighbour_slots_byte_identical() -> None:
    """Slot ``b``'s pool bytes are untouched by writes to slot ``b+1``.

    Checked on the RAW pool buffers (before/after clone compare), not through the
    tables, so it also catches a write that lands out of the request's block but
    inside a range the tables happen not to address. The final assertion is the
    non-vacuity guard: slot ``b+1``'s own block MUST have changed, otherwise the
    "nothing else changed" claim is meaningless.
    """
    lens = [180, 96, 260]
    num_seqs = len(lens)
    max_seq_len = max(lens) + 64
    gen = torch.Generator().manual_seed(2026)
    pool = _make_pool(num_seqs, max_seq_len, "cpu")

    # Fill slots 0 and 1 first; slot 2 stays empty so its later prefill is the
    # single write event under test.
    for req in (0, 1):
        k, v = _rand_kv(lens[req], gen, "cpu")
        pool.append_block(req, k, v)

    plane_names = [n for n in PAGE_PLANES + GROUP_PLANES
                   if getattr(pool, n) is not None]
    before = {n: getattr(pool, n).clone() for n in plane_names}

    # The write event: prefill slot 2, then decode-step it 40 more times.
    k, v = _rand_kv(lens[2], gen, "cpu")
    pool.append_block(2, k, v)
    for _ in range(40):
        k1, v1 = _rand_kv(1, gen, "cpu")
        pool.append(2, k1[0], v1[0])

    changed_in_own_block = 0
    for name in plane_names:
        after = getattr(pool, name)
        is_page = name in PAGE_PLANES
        per_seq = pool.P_per_seq if is_page else pool.G_per_seq
        for req in range(num_seqs):
            lo, hi = req * per_seq, (req + 1) * per_seq
            a, b = before[name][lo:hi], after[lo:hi]
            if req == 2:
                if not torch.equal(a, b):
                    changed_in_own_block += 1
                continue
            assert torch.equal(a, b), (
                f"writing pool slot 2 mutated slot {req}'s block of plane {name!r} "
                f"(rows [{lo}, {hi}) — {int((a != b).sum())} bytes changed)")
    assert changed_in_own_block >= 6, (
        f"only {changed_in_own_block} planes changed inside slot 2's own block — "
        f"the isolation assertion above is vacuous (nothing was written at all)")


def test_batched_decode_write_matches_the_single_seq_store() -> None:
    """The B>1 DECODE WRITE path (``append_decode_batched``) == the store reference.

    Every other test in this file drives the pool through the per-request
    ``append`` / ``append_block`` seam. The served vLLM batched decode does NOT use
    that seam: ``backend.py`` calls ``pool.append_decode_batched(mgr.slots, K, V)``,
    which scatters all B rows in one ``index_put_`` and then regroups the completing
    requests through ``_regroup_pack_batched_device`` -- a SECOND, independent copy
    of the ``req * G_per_seq`` / ``req * P_per_seq`` slot arithmetic. An off-by-one
    there is the same cross-request KV corruption
    ``test_pool_planes_match_store_ragged_batch`` exists to catch, on the path that
    actually serves, and no other test in this file reaches it.

    Ragged lengths (200 / 71 / 333) put the three requests on different group counts,
    so the requests cross their 32-token group boundaries on DIFFERENT steps -- which
    is what makes the batched regroup's per-request scatter non-trivial.
    """
    lens = [200, 71, 333]
    n_decode = 40
    num_seqs = len(lens)
    max_seq_len = max(lens) + n_decode + GROUP_TOKENS
    gen = torch.Generator().manual_seed(4242)

    pool = _make_pool(num_seqs, max_seq_len, "cpu")
    rt, rg = _full_tables(pool)    # BEFORE any decode_inputs_* call (see _full_tables)

    stores = []
    for req, seq_len in enumerate(lens):
        k, v = _rand_kv(seq_len, gen, "cpu")
        store = _make_store(max_seq_len, "cpu")
        store.append_block(k, v)
        stores.append(store)
        pool.append_block(req, k, v)

    for _ in range(n_decode):
        k, v = _rand_kv(num_seqs, gen, "cpu")     # [B, H, D], one new token each
        for req in range(num_seqs):
            stores[req].append(k[req], v[req])
            stores[req].maybe_regroup()
        pool.append_decode_batched(list(range(num_seqs)), k, v)

    for req in range(num_seqs):
        store = stores[req]
        assert store.seq_len == pool.seq_len[req], (
            f"req {req}: seq_len drift store={store.seq_len} pool={pool.seq_len[req]}")
        assert store.packed_groups == pool.packed_groups[req], (
            f"req {req}: packed_groups drift store={store.packed_groups} "
            f"pool={pool.packed_groups[req]}")
        n_groups = store.packed_groups
        assert n_groups >= 2, (
            f"req {req}: only {n_groups} packed groups — lengthen the fixture")
        _assert_planes_identical(
            store.decode_planes(),
            _gather_via_tables(pool, req, n_groups, rt, rg),
            what=f"req {req} via append_decode_batched (seq_len={store.seq_len}, "
                 f"groups={n_groups})")


def test_batched_decode_write_follows_the_slot_list_not_the_row_index() -> None:
    """The B>1 decode WRITE must key on the SLOT it is handed, not on the batch row.

    ``test_batched_decode_write_matches_the_single_seq_store`` above drives
    ``append_decode_batched`` with the IDENTITY list ``range(num_seqs)``, which — for
    exactly the reason
    ``test_decode_inputs_batched_follows_the_slot_list_not_the_row_index`` gives for
    the READ side — cannot distinguish "writes row i into slot ``slots[i]``" from
    "writes row i into slot ``i``". The served writer never gets the identity list:
    ``backend.py:954`` calls ``pool.append_decode_batched(mgr.slots, Kd, Vd)`` and
    ``mgr.slots`` is ``metadata._slots`` -> ``assign_slots``, i.e. an arbitrary
    PERMUTATION of a SUBSET of the live slots (vLLM reorders batch rows freely and a
    step only carries the requests that are actually decoding).

    Mutation-tested: keying the scatter rows, the ring columns or the ``seq_len``
    advance on ``range(B)`` instead of ``slots`` survives every other test in this
    file, and each is silent cross-request KV corruption.
    """
    lens = [200, 71, 333]
    num_seqs = len(lens)
    # 56 rounds so the SHORTEST request (71 tokens, and absent from one schedule
    # entry) still crosses two 32-token group boundaries: 71 + 48 = 119 -> 2 groups.
    n_rounds = 56
    max_seq_len = max(lens) + n_rounds + GROUP_TOKENS
    gen = torch.Generator().manual_seed(5150)

    pool = _make_pool(num_seqs, max_seq_len, "cpu")
    rt, rg = _full_tables(pool)    # BEFORE any decode_inputs_* call (see _full_tables)

    stores = []
    for req, seq_len in enumerate(lens):
        k, v = _rand_kv(seq_len, gen, "cpu")
        store = _make_store(max_seq_len, "cpu")
        store.append_block(k, v)
        stores.append(store)
        pool.append_block(req, k, v)

    # Permutations and proper subsets, exactly the shapes assign_slots produces.
    schedule = [[2, 0, 1], [1, 2, 0], [2, 1], [0, 2, 1], [1], [1, 0, 2], [2, 0]]
    assert any(sl != sorted(sl) for sl in schedule), "no non-identity permutation"
    assert any(len(sl) < num_seqs for sl in schedule), "no proper subset"
    for rnd in range(n_rounds):
        slots = list(schedule[rnd % len(schedule)])
        k, v = _rand_kv(len(slots), gen, "cpu")   # [B, H, D], row i -> slot slots[i]
        for row, slot in enumerate(slots):
            stores[slot].append(k[row], v[row])
            stores[slot].maybe_regroup()
        pool.append_decode_batched(slots, k, v)

    # the slots must have ended at DIFFERENT lengths, else a row/slot mix-up is
    # unobservable in seq_len and the byte compare is the only witness left.
    assert len({int(s.seq_len) for s in stores}) == num_seqs, (
        f"fixture is vacuous: store lengths {[int(s.seq_len) for s in stores]}")

    for req in range(num_seqs):
        store = stores[req]
        assert store.seq_len == pool.seq_len[req], (
            f"req {req}: seq_len drift store={store.seq_len} pool={pool.seq_len[req]} "
            f"— the batched write advanced the wrong slot")
        assert store.packed_groups == pool.packed_groups[req], (
            f"req {req}: packed_groups drift store={store.packed_groups} "
            f"pool={pool.packed_groups[req]}")
        n_groups = store.packed_groups
        assert n_groups >= 2, (
            f"req {req}: only {n_groups} packed groups — lengthen the fixture")
        _assert_planes_identical(
            store.decode_planes(),
            _gather_via_tables(pool, req, n_groups, rt, rg),
            what=f"req {req} via permuted/subset append_decode_batched "
                 f"(seq_len={store.seq_len}, groups={n_groups})")


def test_slot_table_arithmetic_structural(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU-only structural check of the pool's table arithmetic.

    Independent of the packer entirely: it re-derives ``G_per_seq`` / ``P_per_seq``
    from ``max_seq_len`` / ``group_tokens`` / ``page_size`` and then proves the
    per-slot index sets are in-bounds, pairwise DISJOINT, and an EXACT cover of the
    pool. This is the check that still holds the line if the pack path ever becomes
    GPU-only.

    ``OMMX_KV_RING=1`` is forced here purely to keep this gate CHEAP on a developer
    box: the ring only resizes the bf16 shadow (``k_hist``/``v_hist``, sized
    ``ring_cap`` instead of ``max_seq_len`` — kv_pool.py:~195), which this test never
    reads. ``G_per_seq`` / ``P_per_seq`` / ``G_cap`` / ``P_cap`` / the tables are all
    computed BEFORE the ring block from ``max_seq_len`` alone, so the arithmetic under
    test is bit-identical either way — and the assertions below re-derive it from the
    formula, so a regression that made it ring-dependent would fail here. Without
    this, the 128K-context geometry alone allocates 2 x [num_seqs, 131072, H, D] bf16
    of shadow just to look at an ``arange`` (~1.0 GiB at num_seqs=8).
    """
    monkeypatch.setenv("OMMX_KV_RING", "1")
    geometries = [
        # (num_seqs, max_seq_len, group_tokens, page_size)
        (1, 4096, 32, 16),
        (3, 333, 32, 16),
        (8, 8192, 32, 16),      # many slots, ordinary context
        (2, 131072, 32, 16),    # vLLM's default max_model_len (few slots so the
                                # quantized planes stay ~72 MiB, not ~290 MiB)
        (4, 1024, 64, 16),
        (2, 1000, 64, 32),
        (5, 17, 32, 16),        # max_seq_len < group_tokens: the G_per_seq floor
    ]
    for num_seqs, max_seq_len, gt, ps in geometries:
        tag = f"num_seqs={num_seqs} max_seq_len={max_seq_len} gt={gt} ps={ps}"
        win = WindowSpec(sink_tokens=SINK, recent_window=RECENT,
                         group_tokens=gt, page_size=ps)
        pool = MultiSeqKVPool(
            head_dim=HEAD_DIM, n_kv_heads=N_KV_HEADS, num_seqs=num_seqs,
            max_seq_len=max_seq_len, k_format="i2f4", outliers_per_vector=OUTLIERS,
            outlier_select="signed", outlier_repr="relidx7", use_pow2=True,
            kv_outlier_map=True, window=win, group_channels=GROUP_CHANNELS,
            device="cpu")

        # kv_pool.py: "one extra group of slack (a just-completed group never OOBs)".
        exp_g = max(1, (max_seq_len + gt - 1) // gt + 1)
        exp_p = exp_g * (gt // ps)
        assert pool.G_per_seq == exp_g, f"{tag}: G_per_seq {pool.G_per_seq} != {exp_g}"
        assert pool.P_per_seq == exp_p, f"{tag}: P_per_seq {pool.P_per_seq} != {exp_p}"
        assert pool.G_cap == num_seqs * exp_g, f"{tag}: G_cap"
        assert pool.P_cap == num_seqs * exp_p, f"{tag}: P_cap"
        assert pool.pages_per_group == gt // ps, f"{tag}: pages_per_group"

        rt, rg = _full_tables(pool)
        # The tables must address exactly the plane rows that exist.
        assert pool.k_scale.shape[0] == pool.G_cap, f"{tag}: k_scale rows != G_cap"
        assert pool.k_base.shape[0] == pool.P_cap, f"{tag}: k_base rows != P_cap"

        seen_g: set[int] = set()
        seen_p: set[int] = set()
        for req in range(num_seqs):
            g = rg[req].tolist()
            p = rt[req].tolist()
            assert g == list(range(req * exp_g, (req + 1) * exp_g)), (
                f"{tag}: req {req} group table is {g[:4]}..., expected a contiguous "
                f"arange over [{req * exp_g}, {(req + 1) * exp_g})")
            assert p == list(range(req * exp_p, (req + 1) * exp_p)), (
                f"{tag}: req {req} page table is not the contiguous block "
                f"[{req * exp_p}, {(req + 1) * exp_p})")
            assert min(g) >= 0 and max(g) < pool.G_cap, f"{tag}: req {req} group OOB"
            assert min(p) >= 0 and max(p) < pool.P_cap, f"{tag}: req {req} page OOB"
            assert not (seen_g & set(g)), (
                f"{tag}: req {req} group slots overlap an earlier request's")
            assert not (seen_p & set(p)), (
                f"{tag}: req {req} page slots overlap an earlier request's")
            seen_g |= set(g)
            seen_p |= set(p)
        assert seen_g == set(range(pool.G_cap)), f"{tag}: group slots are not an exact cover"
        assert seen_p == set(range(pool.P_cap)), f"{tag}: page slots are not an exact cover"


def test_decode_inputs_batched_matches_single_seq_store() -> None:
    """Read side: the pool's batched decode bundle == each store's single-seq bundle.

    Compares ``b_seq_len`` (packed token count), ``b_tail_len`` (bf16 residual row
    count) and the gathered bf16 residual rows themselves. Rows past ``b_tail_len``
    are deliberately NOT compared: the pool gathers a padded ``[B, max_tail_len]``
    block whose tail rows are masked by ``b_tail_len`` in the kernel.
    """
    lens = [200, 71, 333]
    num_seqs = len(lens)
    max_seq_len = max(lens) + 64
    gen = torch.Generator().manual_seed(31337)
    pool = _make_pool(num_seqs, max_seq_len, "cpu")
    stores = []
    for req, seq_len in enumerate(lens):
        k, v = _rand_kv(seq_len, gen, "cpu")
        store = _make_store(max_seq_len, "cpu")
        store.append_block(k, v)
        stores.append(store)
        pool.append_block(req, k, v)
    for _ in range(37):
        for req in range(num_seqs):
            k, v = _rand_kv(1, gen, "cpu")
            _decode_step(stores[req], pool, req, k[0], v[0])

    batched = pool.decode_inputs_batched(list(range(num_seqs)))
    for req in range(num_seqs):
        single = stores[req].decode_inputs()
        exp_seq = int(single["b_seq_len"][0])
        exp_tail = int(single["b_tail_len"][0])
        got_seq = int(batched["b_seq_len"][req])
        got_tail = int(batched["b_tail_len"][req])
        assert got_seq == exp_seq, (
            f"req {req}: b_seq_len pool={got_seq} store={exp_seq}")
        assert got_tail == exp_tail, (
            f"req {req}: b_tail_len pool={got_tail} store={exp_tail}")
        assert exp_tail > 0, f"req {req}: empty bf16 residual tail — fixture is vacuous"
        assert torch.equal(batched["k_tail"][req, :got_tail],
                           single["k_tail"][0, :exp_tail]), (
            f"req {req}: gathered bf16 K residual rows differ")
        assert torch.equal(batched["v_tail"][req, :got_tail],
                           single["v_tail"][0, :exp_tail]), (
            f"req {req}: gathered bf16 V residual rows differ")


def test_decode_inputs_batched_follows_the_slot_list_not_the_row_index() -> None:
    """The batched gather must key on the SLOT it is handed, not on the row index.

    The identity call ``decode_inputs_batched(range(num_seqs))`` cannot distinguish
    "gathers slot ``slots[i]``" from "gathers slot ``i``" — and the served seam never
    passes the identity list: ``assign_slots`` hands vLLM's reordered batch rows an
    arbitrary PERMUTATION of a SUBSET of the live slots (metadata.py). So both shapes
    are exercised here against the same single-seq stores.
    """
    lens = [200, 71, 333]
    num_seqs = len(lens)
    max_seq_len = max(lens) + 64
    gen = torch.Generator().manual_seed(4242)
    pool = _make_pool(num_seqs, max_seq_len, "cpu")
    stores = []
    for req, seq_len in enumerate(lens):
        k, v = _rand_kv(seq_len, gen, "cpu")
        store = _make_store(max_seq_len, "cpu")
        store.append_block(k, v)
        stores.append(store)
        pool.append_block(req, k, v)
    for _ in range(23):
        for req in range(num_seqs):
            k, v = _rand_kv(1, gen, "cpu")
            _decode_step(stores[req], pool, req, k[0], v[0])

    def _check(slots) -> None:
        batched = pool.decode_inputs_batched(list(slots))
        assert int(batched["b_seq_len"].shape[0]) == len(slots), (
            f"slots={list(slots)}: bundle has {int(batched['b_seq_len'].shape[0])} "
            f"rows, expected {len(slots)}")
        for row, slot in enumerate(slots):
            single = stores[slot].decode_inputs()
            exp_seq = int(single["b_seq_len"][0])
            exp_tail = int(single["b_tail_len"][0])
            assert int(batched["b_seq_len"][row]) == exp_seq, (
                f"slots={list(slots)} row {row}: b_seq_len "
                f"{int(batched['b_seq_len'][row])} != slot {slot}'s {exp_seq} — the "
                f"gather followed the ROW INDEX, not the slot")
            assert int(batched["b_tail_len"][row]) == exp_tail, (
                f"slots={list(slots)} row {row}: b_tail_len "
                f"{int(batched['b_tail_len'][row])} != slot {slot}'s {exp_tail}")
            assert torch.equal(batched["k_tail"][row, :exp_tail],
                               single["k_tail"][0, :exp_tail]), (
                f"slots={list(slots)} row {row}: bf16 K residual rows are not slot "
                f"{slot}'s")
            assert torch.equal(batched["v_tail"][row, :exp_tail],
                               single["v_tail"][0, :exp_tail]), (
                f"slots={list(slots)} row {row}: bf16 V residual rows are not slot "
                f"{slot}'s")
            # the routing tables handed to the kernel must point at the SLOT's block.
            rt_row = batched["req_to_token"][row].tolist()
            assert rt_row[0] == slot * pool.P_per_seq, (
                f"slots={list(slots)} row {row}: req_to_token starts at {rt_row[0]}, "
                f"expected slot {slot}'s page base {slot * pool.P_per_seq}")

    # the three lens are pairwise different, so a row/slot mix-up moves b_seq_len.
    assert len({int(s.seq_len) for s in stores}) == num_seqs, (
        "fixture is vacuous: the slots must have DIFFERENT lengths for a row/slot "
        "mix-up to be observable")
    _check([2, 0, 1])          # permutation
    _check([1])                # single-row subset (a decode step with one live req)
    _check([2, 1])             # subset, out of order


# ── the same parity on a real device ────────────────────────────────────────────

@pytest.mark.gpu
def test_pool_planes_match_store_ragged_batch_cuda() -> None:
    """The ragged-batch parity re-run on CUDA (the device the serving path uses).

    Marked ``gpu``: with no CUDA device this is SKIPPED by ``conftest.py``, which
    reports as ``s`` — it can never be mistaken for a pass. On a GPU host the pool
    and store both take the on-device pack (``OMMX_KV_GPU_PACK`` default ON), so
    this also gates the GPU packer against the same synthetic sequences.
    """
    _run_parity("cuda", lens=[200, 71, 333], n_decode=40, seed=7)
