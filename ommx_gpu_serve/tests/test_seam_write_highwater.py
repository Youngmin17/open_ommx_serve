# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The build seam may not claim a pool holds tokens nothing wrote.

WHY IT EXISTS. ``MultiSeqKVPool.seq_len[r]`` is the pool's own write high-water mark --
every write path increments it. The batched-graph build seam assigns vLLM's sequence
length over it and then regroup-packs up to the new count. When the two agree that is an
anticipation of the one token the captured write is about to store; when they do not, the
pack reads rows nothing ever wrote.

WHAT THAT COSTS, measured rather than reasoned about. On an H200 the pool held ONE token
against a 1357-token prompt and the seam regrouped to 41 groups anyway. The pack then read
32768 of 32768 non-finite values, which collapses the signed outlier selector -- its
``1e9`` mask loses to ``+inf``, so the bottom-k re-picks the top-k, half the mask goes
unfilled, and the selector's ``gt`` sentinel lands in the outlier index. ``gather`` then
indexes 32 in a 32-wide axis and ATen aborts the CUDA context with a message naming no
tensor, no line and no value. It reproduced on 4 of 6 trials.

THE CAUSE was upstream of the seam: prefills arrive at ``B == 1``, before the batched
session latch trips, so the prompt was written to the per-request store and the shared pool
never saw it. That is fixed in ``backend.py`` (graph mode pre-latches). These tests pin the
seam's half -- the part that turned a missing write into an unattributable device abort --
so a future write path that misses the pool degrades to bf16 instead.

SCOPE. CPU-only, exercising the manager's state directly. They do not run vLLM, so they do
not prove the engine reaches these states; the H200 numbers above are what establishes
that.
"""
import pytest

torch = pytest.importorskip("torch")

from ommx_gpu_serve.attention.kv_pool import MultiSeqKVPool  # noqa: E402
from ommx_gpu_serve.attention.kv_window import WindowSpec  # noqa: E402
from ommx_gpu_serve.integration.vllm.metadata import (  # noqa: E402
    OMMXBatchedStepManager,
)

GT, SINK, RECENT = 32, 8, 32


def _pool(num_seqs=2, max_seq_len=2048):
    return MultiSeqKVPool(
        head_dim=64, n_kv_heads=2, num_seqs=num_seqs, max_seq_len=max_seq_len,
        window=WindowSpec(sink_tokens=SINK, recent_window=RECENT, group_tokens=GT,
                          page_size=GT),
        outliers_per_vector=4, device="cpu")


def _mgr(pool):
    """A manager holding one pool, with the graph buffers the seam fills."""
    m = OMMXBatchedStepManager.__new__(OMMXBatchedStepManager)
    m.pools = {0: pool}
    m.window = pool.window
    m.dead = False
    m.dead_reason = None
    m.max_tail_cap = SINK + RECENT + GT
    m.cur_B = 0
    n = pool.num_seqs
    T = m.max_tail_cap
    m._sel_host = torch.zeros(n, dtype=torch.long)
    m._col_host = torch.zeros(n, dtype=torch.long)
    m._bseq_host = torch.zeros(n, dtype=torch.long)
    m._btail_host = torch.zeros(n, dtype=torch.long)
    m._tidx_host = torch.zeros(n, T, dtype=torch.long)
    m.b_sel = torch.zeros(n, dtype=torch.long)
    m.b_write_col = torch.zeros(n, dtype=torch.long)
    m.b_seq_len = torch.zeros(n, dtype=torch.long)
    m.b_tail_len = torch.zeros(n, dtype=torch.long)
    m.b_tail_idx = torch.zeros(n, T, dtype=torch.long)
    return m


def _write(pool, slot, n_tokens):
    K = torch.randn(n_tokens, pool.H, pool.D, dtype=torch.float32).to(torch.bfloat16)
    V = torch.randn(n_tokens, pool.H, pool.D, dtype=torch.float32).to(torch.bfloat16)
    pool.append_block(slot, K, V)


# ── the healthy case ───────────────────────────────────────────────────────────────

def test_the_one_token_anticipation_is_allowed():
    """The seam runs BEFORE the captured write, so ``seq`` is legitimately one ahead."""
    pool = _pool()
    _write(pool, 0, 200)
    m = _mgr(pool)
    m._refresh_graph_buffers([201], [0])
    assert not m.dead, f"a one-token gap was refused: {m.dead_reason}"
    assert int(pool.seq_len[0]) == 201


def test_an_exactly_current_sequence_is_allowed():
    """``seq == written`` (a step where the write already landed) must also pass."""
    pool = _pool()
    _write(pool, 0, 200)
    m = _mgr(pool)
    m._refresh_graph_buffers([200], [0])
    assert not m.dead, f"an exact match was refused: {m.dead_reason}"


# ── the defect ─────────────────────────────────────────────────────────────────────

def test_a_pool_that_never_saw_the_prompt_degrades_instead_of_packing():
    """The measured shape: one written token, a 1357-token sequence.

    Before this check the seam assigned 1357 and regrouped 41 groups of uninitialised
    memory. It must now refuse and let the backend fall back to bf16.
    """
    pool = _pool()
    _write(pool, 0, 1)
    m = _mgr(pool)
    m._refresh_graph_buffers([1357], [0])
    assert m.dead, "the seam packed a pool that holds one token for a 1357-token sequence"
    assert int(pool.seq_len[0]) == 1, (
        "the pool's write high-water mark was overwritten before the refusal")
    assert pool.packed_groups[0] == 0, "uninitialised rows were packed anyway"


def test_the_refusal_names_the_numbers_that_identify_it():
    """An operator reading only this line must be able to tell it from a capacity or
    allocator problem: which slot, how many tokens are really there, what vLLM claimed."""
    pool = _pool()
    _write(pool, 0, 1)
    m = _mgr(pool)
    m._refresh_graph_buffers([1357], [0])
    msg = m.dead_reason or ""
    for token in ("slot 0", "1 written tokens", "seq=1357", "1355"):
        assert token in msg, f"the refusal does not report {token!r}: {msg}"


def test_one_bad_slot_does_not_let_the_others_pack():
    """The seam returns on the first violation rather than continuing down the batch: a
    manager that is dead for one request is dead for the step."""
    pool = _pool()
    _write(pool, 0, 1)
    _write(pool, 1, 200)
    m = _mgr(pool)
    m._refresh_graph_buffers([1357, 201], [0, 1])
    assert m.dead
    assert pool.packed_groups[1] == 0 or int(pool.seq_len[1]) <= 200, (
        "the second slot was advanced by a step that had already failed")


@pytest.mark.parametrize("written,seq,ok", [
    (0, 0, True),
    (0, 1, True),
    (0, 2, False),
    (100, 101, True),
    (100, 102, False),
    (1, 1357, False),
])
def test_the_boundary_is_exactly_one_token(written, seq, ok):
    """Pinned as a boundary, not a magnitude: the check must not become a tolerance."""
    pool = _pool()
    if written:
        _write(pool, 0, written)
    m = _mgr(pool)
    m._refresh_graph_buffers([seq], [0])
    assert (not m.dead) == ok, (
        f"written={written} seq={seq}: dead={m.dead} (expected ok={ok})")
