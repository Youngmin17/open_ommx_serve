# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the batched-decode slot allocator (``metadata.assign_slots``).

The allocator is the one piece of the B>1 seam that can corrupt KV ACROSS requests
without raising anything: two live requests handed the SAME ``MultiSeqKVPool`` slot
write over each other's planes, and the only symptom is wrong tokens. It is
dependency-free python (no torch, no vLLM, no GPU), so it is unit-testable here.

Contract under test (``ommx_gpu_serve/integration/vllm/metadata.py``)::

    assign_slots(keys, key_to_slot, free_slots, num_seqs) -> (list[int], list[int])

  * ``keys``          this step's per-row stable request key, in batch-row order.
  * ``key_to_slot``   persistent ``{key: slot}`` allocator state, MUTATED IN PLACE.
  * ``free_slots``    persistent list of unassigned slots, MUTATED IN PLACE.
  * ``num_seqs``      static pool slot capacity.
  * returns           row -> slot, ``len(out) == len(keys)``.

The four behaviours the finding requires, plus two invariants an adversarial
reviewer would reach for:

  1. a key that survives across steps keeps its slot (and keeps it under batch-row
     REORDERING, which vLLM does freely);
  2. a departed key's slot is released and becomes reusable;
  3. exhaustion RAISES instead of recycling a live slot;
  4. DUPLICATE keys RAISE (the prefix-caching aliasing case: two requests that share
     a prompt prefix share their first physical block, so the block id stops being a
     per-request identity);
  5. all ``num_seqs`` slots are usable (no off-by-one that wastes or over-hands one);
  6. across a long randomized arrive/depart churn, the live rows of every single step
     hold PAIRWISE DISTINCT in-range slots.

``pytest.mark.xfail`` on ImportError (never ``skip``): if ``assign_slots`` is not
exported, these report as expected-failures naming the missing symbol rather than
quietly reporting nothing.
"""
from __future__ import annotations

import random

import pytest

_IMPORT_ERR = None
try:
    from ommx_gpu_serve.integration.vllm.metadata import (
        OMMXSlotAllocationError,
        assign_slots,
    )
except ImportError as exc:  # pragma: no cover - exercised only before the fix lands
    _IMPORT_ERR = f"{type(exc).__name__}: {exc}"

_XFAIL = pytest.mark.xfail(
    _IMPORT_ERR is not None,
    reason=("ommx_gpu_serve.integration.vllm.metadata.assign_slots / "
            "OMMXSlotAllocationError are not importable "
            f"({_IMPORT_ERR}) — the metadata.py slot-allocator fix has not landed. "
            "This is an XFAIL, not a skip: the gate is present and failing, not absent."),
    strict=True,
)

pytestmark = _XFAIL


def _fresh(num_seqs: int):
    """A brand-new allocator state pair, exactly as ``OMMXBatchedStepManager`` seeds it."""
    return {}, list(range(num_seqs))


def _assert_step_invariants(out, keys, num_seqs: int) -> None:
    """Every step's result must be same-length, in-range and pairwise DISTINCT.

    Distinctness is the whole point: two rows on one slot is the silent cross-request
    KV corruption this allocator exists to prevent.
    """
    assert len(out) == len(keys), f"len(out)={len(out)} != len(keys)={len(keys)}"
    assert all(isinstance(s, int) for s in out), f"non-int slot in {out}"
    assert all(0 <= s < num_seqs for s in out), (
        f"slot out of range [0, {num_seqs}): {out}")
    assert len(set(out)) == len(out), (
        f"ALIASING: {len(keys)} rows share {len(set(out))} slots ({out}) — two live "
        f"requests would write the same pool slot")


# ── 1. stability ────────────────────────────────────────────────────────────────

def test_surviving_key_keeps_its_slot_across_steps() -> None:
    k2s, free = _fresh(4)
    first = assign_slots([100, 200, 300], k2s, free, 4)[0]
    _assert_step_invariants(first, [100, 200, 300], 4)
    second = assign_slots([100, 200, 300], k2s, free, 4)[0]
    assert second == first, (
        f"slots moved for an unchanged batch: {first} -> {second}; a request's KV "
        f"lives in its slot, so the slot must be stable for its lifetime")


def test_slots_follow_the_key_not_the_batch_row_order() -> None:
    """vLLM reorders batch rows between steps; the slot must follow the KEY.

    If the allocator keyed on the row index instead, the reordered step would return
    ``[0, 1, 2]`` again and requests would silently swap KV planes.
    """
    k2s, free = _fresh(4)
    first = assign_slots([100, 200, 300], k2s, free, 4)[0]
    reordered = assign_slots([300, 100, 200], k2s, free, 4)[0]
    # the slots must be the SAME slots, permuted the same way the rows were. A
    # row-index-keyed allocator would return `first` unchanged here, which is exactly
    # the failure mode: request 300's KV would be read out of request 100's slot.
    assert reordered == [first[2], first[0], first[1]], (
        f"batch [300,100,200] -> {reordered}; expected the permutation "
        f"{[first[2], first[0], first[1]]} of the first step's {first}")
    _assert_step_invariants(reordered, [300, 100, 200], 4)


# ── 2. release ──────────────────────────────────────────────────────────────────

def test_departed_key_frees_its_slot_for_a_new_request() -> None:
    """A finished/preempted request's slot must come back and be REUSABLE.

    The pool is exactly ``num_seqs`` slots wide, so a leak here turns into a false
    exhaustion after ``num_seqs`` requests have come and gone. The step
    ``[200, 300]`` with ``num_seqs=2`` only succeeds if key 100's slot was released
    BEFORE the new key was served — i.e. it also pins the release-then-assign order.
    """
    k2s, free = _fresh(2)
    first = assign_slots([100, 200], k2s, free, 2)[0]
    freed_slot = first[0]
    second = assign_slots([200, 300], k2s, free, 2)[0]
    _assert_step_invariants(second, [200, 300], 2)
    assert 100 not in k2s, "departed key 100 is still holding a slot"
    assert second[0] == first[1], "surviving key 200 lost its slot"
    assert second[1] == freed_slot, (
        f"new key 300 got slot {second[1]}, expected the slot {freed_slot} released "
        f"by the departed key 100")


def test_all_slots_are_usable_then_all_are_reclaimed() -> None:
    """No off-by-one: ``num_seqs`` distinct keys fill exactly ``range(num_seqs)``,
    and emptying the batch returns every one of them."""
    n = 6
    k2s, free = _fresh(n)
    keys = list(range(1000, 1000 + n))
    out = assign_slots(keys, k2s, free, n)[0]
    _assert_step_invariants(out, keys, n)
    assert set(out) == set(range(n)), (
        f"only {sorted(set(out))} of the {n} pool slots were handed out")
    assert assign_slots([], k2s, free, n)[0] == []
    assert k2s == {}, f"slots still held after an empty batch: {k2s}"
    assert sorted(free) == list(range(n)), (
        f"free list is {sorted(free)}, expected every slot back")


# ── 3. exhaustion ───────────────────────────────────────────────────────────────

def test_exhaustion_raises_instead_of_recycling_a_live_slot() -> None:
    """More live requests than slots must RAISE.

    The pre-fix behaviour handed out ``len(key_to_slot) % num_seqs`` — a slot that is
    almost always owned by a LIVE request, i.e. silent cross-request corruption.
    """
    k2s, free = _fresh(2)
    assign_slots([1, 2], k2s, free, 2)[0]
    with pytest.raises(OMMXSlotAllocationError) as ei:
        assign_slots([1, 2, 3], k2s, free, 2)[0]
    msg = str(ei.value)
    assert msg.strip(), "the exhaustion error carries no message"
    assert "OMMX_ATTN_MAX_NUM_SEQS" in msg, (
        f"the exhaustion error must name the knob that fixes it; got: {msg}")


def test_exhaustion_raises_on_a_cold_allocator_too() -> None:
    """The first step alone can overflow; it must raise, not wrap onto slot 0."""
    k2s, free = _fresh(3)
    with pytest.raises(OMMXSlotAllocationError):
        assign_slots([10, 11, 12, 13], k2s, free, 3)[0]


# ── 4. duplicate keys (the prefix-caching aliasing case) ────────────────────────

def test_duplicate_keys_raise_and_leave_the_allocator_untouched() -> None:
    """Two rows resolving to one key is prefix caching sharing a first physical block.

    Both rows would land on one slot and overwrite each other's KV, so the step must
    be refused — and refused BEFORE any mutation, so the caller can degrade to bf16
    and keep serving on a consistent allocator.
    """
    k2s, free = _fresh(4)
    assign_slots([7, 8], k2s, free, 4)[0]
    before_map = dict(k2s)
    before_free = list(free)
    with pytest.raises(OMMXSlotAllocationError) as ei:
        assign_slots([7, 8, 9, 9], k2s, free, 4)[0]
    assert "prefix caching" in str(ei.value).lower(), (
        f"the duplicate-key error must name prefix caching as the cause; "
        f"got: {ei.value}")
    assert k2s == before_map, f"allocator state mutated by a rejected step: {k2s}"
    assert free == before_free, f"free list mutated by a rejected step: {free}"


def test_duplicate_key_raises_even_when_capacity_is_ample() -> None:
    """Not a disguised capacity error: 2 rows, 64 slots, still refused."""
    k2s, free = _fresh(64)
    with pytest.raises(OMMXSlotAllocationError):
        assign_slots([5, 5], k2s, free, 64)[0]


# ── 5/6. invariants under churn + corrupt free list ─────────────────────────────

def test_double_freed_slot_is_refused_not_handed_out() -> None:
    """A slot that is already owned must never come back out of the free list.

    Simulates the free-list double-push (a released slot appended twice) — the same
    aliasing bug wearing a different hat, so it must raise rather than be served.
    """
    k2s, free = _fresh(4)
    out = assign_slots([1, 2], k2s, free, 4)[0]
    # corrupt: the slot owned by key 1 is pushed back onto the FRONT of the free list,
    # so it is the next one popped (appending it to the tail would only surface after
    # the genuinely-free slots were drained — same bug, later step).
    free.insert(0, out[0])
    with pytest.raises(OMMXSlotAllocationError) as ei:
        assign_slots([1, 2, 3], k2s, free, 4)[0]
    assert "already owned" in str(ei.value) or "double-freed" in str(ei.value), (
        f"expected a double-free diagnosis; got: {ei.value}")


def test_randomized_churn_never_aliases_two_live_requests() -> None:
    """200 steps of arrivals/departures under a fixed capacity: no step may alias.

    This is the property the whole file exists for, exercised over shapes no
    hand-written case enumerates. Seeded, so a failure is reproducible.
    """
    rng = random.Random(20260824)
    num_seqs = 8
    k2s, free = _fresh(num_seqs)
    live: list[int] = []
    next_key = 1
    for step in range(200):
        # depart a random subset
        live = [k for k in live if rng.random() > 0.25]
        # admit new requests up to the pool cap (the caller's own admission control;
        # exceeding it is tested separately as an explicit raise)
        while len(live) < num_seqs and rng.random() < 0.5:
            live.append(next_key)
            next_key += 1
        rng.shuffle(live)
        out, _newly = assign_slots(list(live), k2s, free, num_seqs)
        _assert_step_invariants(out, live, num_seqs)
        for key, slot in zip(live, out):
            assert k2s[key] == slot, f"step {step}: key {key} -> {slot} not recorded"
        assert set(k2s) == set(live), (
            f"step {step}: allocator holds {sorted(set(k2s) - set(live))} that are no "
            f"longer live")
        assert sorted(list(free) + out) == list(range(num_seqs)), (
            f"step {step}: free list + live slots is not the whole pool "
            f"(free={sorted(free)}, live={sorted(out)}) — a slot leaked or duplicated")
