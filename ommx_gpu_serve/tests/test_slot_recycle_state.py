# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""A recycled pool slot must not inherit its previous occupant's packed KV.

WHY IT EXISTS. ``test_slot_allocator.py`` pins that two LIVE requests never share a slot,
and ``test_page_grid_alignment.py`` pins that a table row never addresses another request's
storage. Both are about MEMORY ownership, and both hold. Neither can express the failure
below, which is about CONTENT: one request reading bytes that a *previous* occupant of the
same slot wrote.

THE SEQUENCE. ``assign_slots`` returns a finished request's slot to the free list and
clears only its own bookkeeping (``metadata.assign_slots``); the pool's ``seq_len`` and
``packed_groups`` for that slot are untouched. The single place that resets them is the
EAGER batched write, gated on a restart (``backend.py``'s ``if seq_before <= 0``). The
capture-safe write path documents that it deliberately touches neither, because the host
build seam is supposed to have done it -- and the build seam only advances them:

    pool.seq_len[r] = seq
    target = win.num_groups(seq)
    if target > pool.packed_groups[r]:      # stale-high -> never fires
        pool.regroup(r)

``regroup`` is itself monotone (``if target <= self.packed_groups[r]: return 0``), so a
stale-high count is permanent for that slot. Meanwhile the kernel's ``b_seq_len`` is
derived from the window boundary over vLLM's own sequence length, NOT from
``packed_groups`` -- so it says "there are packed tokens here" while nothing was packed for
this occupant, and the decode reads the previous one's planes.

WHAT THESE TESTS DO AND DO NOT COVER. They are CPU-only and exercise the pool and the
allocator directly, which is where the state lives. They do not run vLLM, so they do not
prove the graph path reaches this state in a real engine -- the entry conditions there
(``OMMX_ATTN_BATCHED_GRAPH=1``, a single-token prompt landing on a recycled slot) are
stated in the module docstring and remain unverified on hardware.
"""
import pytest

torch = pytest.importorskip("torch")

from ommx_gpu_serve.attention.kv_pool import MultiSeqKVPool  # noqa: E402
from ommx_gpu_serve.attention.kv_window import WindowSpec  # noqa: E402
from ommx_gpu_serve.integration.vllm.metadata import assign_slots  # noqa: E402

GT, SINK, RECENT = 32, 8, 32


def _pool(num_seqs=4, max_seq_len=512):
    return MultiSeqKVPool(
        head_dim=64, n_kv_heads=2, num_seqs=num_seqs, max_seq_len=max_seq_len,
        window=WindowSpec(sink_tokens=SINK, recent_window=RECENT, group_tokens=GT,
                          page_size=GT),
        outliers_per_vector=4, device="cpu")


def _fill(pool, slot, n_tokens, value):
    """Write ``n_tokens`` of a constant-valued K/V into ``slot`` and pack what completes."""
    K = torch.full((n_tokens, pool.H, pool.D), float(value), dtype=torch.bfloat16)
    V = torch.full((n_tokens, pool.H, pool.D), float(value), dtype=torch.bfloat16)
    pool.append_block(slot, K, V)
    pool.regroup(slot)


# ── the allocator's half ───────────────────────────────────────────────────────────

def test_releasing_a_slot_does_not_clear_the_pool_state_behind_it():
    """Keys are vLLM block ids (ints), not names.

    ``assign_slots`` is deliberately dependency-free -- it cannot reach the pool. So
    releasing a slot is bookkeeping only, and whoever calls it owns the reset."""
    pool = _pool()
    key_to_slot, free_slots = {}, list(range(pool.num_seqs))
    slots, fresh = assign_slots([101], key_to_slot, free_slots, pool.num_seqs)
    a = slots[0]
    assert fresh == [a], "a first assignment is a fresh one"
    _fill(pool, a, 200, 1.0)
    packed_before = pool.packed_groups[a]
    assert packed_before > 0, "the fixture must actually pack something"

    assign_slots([202], key_to_slot, free_slots, pool.num_seqs)   # A vanishes
    assert a in free_slots, "the slot should have been released"
    assert pool.packed_groups[a] == packed_before, (
        "if this ever becomes 0 the allocator grew a pool reference; update this test "
        "rather than deleting it")


# ── the consequence, which is the actual defect ────────────────────────────────────

def _recycle(reset: bool):
    """Run A (400 tokens of 1.0) -> release -> C (100 tokens of 2.0) on ONE slot.

    ``append_block`` regroups as it writes, so the packing happens there rather than in a
    separate call; what decides the outcome is only whether the slot's counters were
    cleared when it was handed to C.
    """
    pool = _pool(num_seqs=1)
    key_to_slot, free_slots = {}, list(range(pool.num_seqs))
    a = assign_slots([101], key_to_slot, free_slots, pool.num_seqs)[0][0]
    _fill(pool, a, 400, 1.0)
    stale = pool.packed_groups[a]

    _slots, fresh = assign_slots([303], key_to_slot, free_slots, pool.num_seqs)
    assert key_to_slot[303] == a, "the fixture needs C to land on A's slot"
    assert fresh == [a], "recycling a released slot is a fresh assignment"
    pool.seq_len[a] = 0
    if reset:
        # what OMMXBatchedStepManager._slots now does for every freshly assigned slot
        pool.packed_groups[a] = 0
    _fill(pool, a, 100, 2.0)
    zp = pool.request_planes(a)["k_zp"].float().abs().max().item()
    return stale, pool.packed_groups[a], zp


def test_a_recycled_slot_without_the_reset_serves_the_previous_occupants_kv():
    """The defect, isolated. ``regroup`` is monotone, so a stale-high count from the
    previous occupant is permanent: C's groups are never packed and its planes still
    dequantize to A's constant."""
    stale, packed, zp = _recycle(reset=False)
    assert packed == stale, "the stale count survives, so nothing was repacked"
    assert abs(zp - 1.0) < 0.25, (
        f"group 0's zero-point is {zp:.3f}; 1.0 is request A's constant. If this is now "
        "2.0 the reset reached this path and the test should be re-derived, not deleted")


def test_the_reset_at_the_assignment_site_closes_it():
    """Same sequence with the reset the caller performs: C's own sequence is packed and
    its planes carry its own values."""
    stale, packed, zp = _recycle(reset=True)
    assert packed < stale, f"expected C's shorter sequence, got {packed} vs A's {stale}"
    assert abs(zp - 2.0) < 0.5, (
        f"group 0's zero-point is {zp:.3f}, not request C's 2.0 -- the recycled slot is "
        "still serving the previous occupant's packed KV")


def test_assign_slots_reports_which_assignments_are_fresh():
    """The signal the fix rests on. A surviving key must NOT be reported, or the caller
    would clear a live request's packed history every step."""
    pool = _pool(num_seqs=2)
    k2s, free = {}, list(range(2))
    _s, fresh = assign_slots([101, 202], k2s, free, 2)
    assert sorted(fresh) == sorted(_s), "both are new"
    _s2, fresh2 = assign_slots([101, 202], k2s, free, 2)
    assert fresh2 == [], "neither key moved, so nothing may be cleared"
    _s3, fresh3 = assign_slots([101], k2s, free, 2)
    assert fresh3 == [], "202 departing does not make 101 fresh"


def test_the_memory_ownership_gate_cannot_see_this():
    """Stated explicitly so the two gates are not confused for each other: every table row
    still points inside its own slice throughout the sequence above. Memory ownership
    holds; content ownership is what broke."""
    pool = _pool()
    for b in range(pool.num_seqs):
        lo, hi = b * pool.P_per_seq, (b + 1) * pool.P_per_seq
        row = pool._req_to_token[b]
        assert int(row.min()) >= lo and int(row.max()) < hi
