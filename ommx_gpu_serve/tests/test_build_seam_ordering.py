# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The host build seam packs BEFORE the captured forward writes the current token.

WHY IT EXISTS. ``OMMXBatchedStepManager._refresh_graph_buffers`` runs at metadata-build
time and regroup-packs completed groups. The captured ``do_kv_cache_update`` writes the
step's own token into ``k_hist`` AFTER that. So for one step there is a token that the
window arithmetic can see (it is counted in ``seq``) but whose K/V bytes are not in the
cache yet -- reading it would pack uninitialised memory into a quantized group, silently,
with no index ever going out of range.

The write path's docstring asserts this cannot happen, because the current token "always
lands in the bf16 TAIL window". That is an arithmetic claim about ``packed_boundary`` and
it was carrying the correctness of the whole capture-safe path on prose alone. It is two
lines to check, so it is checked here.

THE CLAIM, precisely: for every sequence length, ``boundary(seq) <= seq - recent``, hence
the packed region ``[sink, boundary)`` never contains token ``seq-1`` as long as
``recent >= 1``. The margin is not marginal -- it is at least ``recent_window`` tokens.

WHAT THIS DOES NOT COVER. It pins the arithmetic, not the plumbing: that
``_refresh_graph_buffers`` is in fact the only packer on that path, and that nothing else
advances ``seq`` early, are separate properties. It also says nothing about the
batched-graph abort under ``OMMX_ATTN_BATCHED_GRAPH=1``, which was measured to be a
synchronization defect elsewhere -- this rules the window arithmetic OUT as its cause.
"""
import pytest

from ommx_gpu_serve.attention.kv_window import packed_boundary

# the shipped recipe, plus shapes that stress the corners of the floor division
WINDOWS = [
    (8, 32, 32),      # SHIPPED
    (0, 32, 32),      # no sink
    (8, 1, 32),       # the tightest legal recent window
    (128, 128, 128),
    (8, 32, 1),       # group_tokens=1: boundary advances every token
    (4, 7, 3),        # nothing divides anything
]


@pytest.mark.parametrize("sink,recent,gt", WINDOWS)
def test_the_current_token_is_never_inside_the_packed_region(sink, recent, gt):
    """Token ``seq-1`` is never in the packed region ``[sink, boundary)``.

    Stated over the REGION, not over ``boundary`` alone. The first draft asserted
    ``seq-1 >= boundary`` unconditionally and failed on the shipped window at ``seq < sink``:
    ``packed_boundary`` returns ``min(sink, seq)`` there, so boundary equals seq while the
    packed region ``[sink, boundary)`` is EMPTY. The bound on boundary only holds on the
    branch that packs anything -- which is the branch the ordering argument is about.
    """
    for seq in range(1, 600):
        b = packed_boundary(seq, sink, recent, gt)
        cur = seq - 1
        assert not (sink <= cur < b), (
            f"seq={seq} window(sink={sink},recent={recent},gt={gt}): packed region "
            f"[{sink},{b}) contains token {cur}, which the captured write has NOT "
            f"stored yet")
        if b > sink:                      # the packing branch: the bound must be tight
            assert b <= seq - recent, (
                f"seq={seq} window(sink={sink},recent={recent},gt={gt}): boundary={b} "
                f"exceeds seq-recent={seq - recent}")


@pytest.mark.parametrize("sink,recent,gt", WINDOWS)
def test_the_margin_is_at_least_the_recent_window(sink, recent, gt):
    """Stronger, and the reason the claim is robust rather than an off-by-one that
    happens to hold: the unpacked tail is never shorter than ``recent``."""
    for seq in range(1, 600):
        b = packed_boundary(seq, sink, recent, gt)
        if b > sink:                       # only once a group has actually been packed
            assert seq - b >= recent, (
                f"seq={seq} window(sink={sink},recent={recent},gt={gt}): tail is "
                f"{seq - b} tokens, shorter than the {recent}-token recent window")


def test_a_zero_recent_window_would_break_it():
    """The claim depends on ``recent >= 1``; recorded so that a future recipe that drops
    the recent window cannot quietly invalidate the capture-safe write path.

    With recent=0 and gt=1 the boundary reaches ``seq`` itself, i.e. the build seam would
    pack the very token the captured write has not stored.
    """
    seq, sink = 100, 8
    assert packed_boundary(seq, sink, 0, 1) == seq, (
        "recent=0/gt=1 no longer reaches seq; re-derive the ordering argument in "
        "metadata._refresh_graph_buffers before relaxing the recent window")


def test_the_write_path_still_documents_the_ordering_it_relies_on():
    """Source-level: the arithmetic above is only half the argument. If the docstring's
    ordering claim is ever deleted, the reader loses the reason these tests exist."""
    import pathlib

    import ommx_gpu_serve

    src = (pathlib.Path(ommx_gpu_serve.__file__).parent
           / "integration" / "vllm" / "metadata.py").read_text()
    assert "is NOT yet in ``k_hist``" in src, (
        "the capture-safe write path no longer documents that the current token is "
        "written after the build seam -- check whether the ordering itself changed")
