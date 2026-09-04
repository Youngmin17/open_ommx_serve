# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The OMMX page grid against vLLM's block grid.

WHY IT EXISTS. The OMMX sidecar addresses its own storage, so for a long time nothing
forced its page size to match the engine's block size -- and the default happened to be
16, which is also vLLM's default, so the two agreed by coincidence. They diverged the
moment anyone passed ``--block-size``.

WHY IT MATTERS BEYOND TIDINESS. A K group's scale, zero-point and outlier set are a pure
function of its ``group_tokens`` tokens (``pack.py``'s per-group statistics), so a group is
*content-addressable* and could in principle be shared between requests. It can only be
shared at BLOCK granularity if it sits inside one block: vLLM's allocator gives no
co-location guarantee for the two blocks of a straddling group. So ``pages_per_group == 1``
is the precondition for any future paged or prefix-sharing design, and it is reachable --
``--block-size 32`` is legal (FlashAttention advertises ``MultipleOf(16)`` for a
non-hybrid model) and makes group_tokens=32 exactly one block.

WHAT THIS DOES NOT CLAIM. Aligning the grids does not make the sidecar paged. It still
allocates its own planes, still indexes them with an identity table, and prefix caching
and chunked prefill are still refused. This is A precondition, not the feature -- and by
itself not even a sufficient one, which is the point of
``test_pages_per_group_one_is_not_the_same_as_block_aligned`` below.
"""
import pytest

torch = pytest.importorskip("torch")

from ommx_gpu_serve.attention.kv_pool import MultiSeqKVPool  # noqa: E402
from ommx_gpu_serve.attention.kv_window import WindowSpec  # noqa: E402
from ommx_gpu_serve.integration.vllm.config import (  # noqa: E402
    resolve_serving_config,
)


def _cfg(page_size=None, **kw):
    return resolve_serving_config(head_dim=64, n_q_heads=4, n_kv_heads=2,
                                  max_context=512, page_size=page_size, **kw)


# ── the grid the engine asks for is the grid OMMX uses ─────────────────────────────

@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
def test_page_size_follows_the_engine_block_size(block_size):
    assert _cfg(page_size=block_size).page_size == block_size


def test_an_absent_block_size_keeps_the_shipped_default():
    """A caller with no engine (tests, the HF-eager path) must still resolve."""
    assert _cfg(page_size=None).page_size == 16


def test_the_backend_sources_the_page_size_from_cache_config():
    """Source-level: backend.py imports vllm at module scope, so this suite cannot
    import it. The failure being pinned is a resolve call that silently keeps the
    hardcoded default while the engine runs a different block size."""
    import pathlib

    import ommx_gpu_serve

    src = (pathlib.Path(ommx_gpu_serve.__file__).parent
           / "integration" / "vllm" / "backend.py").read_text()
    assert 'getattr(getattr(vllm_config, "cache_config", None), "block_size", None)' in src
    assert "page_size=int(_bs) if isinstance(_bs, int) and _bs > 0 else None" in src


# ── what alignment buys, stated as the pool's own arithmetic ───────────────────────

@pytest.mark.parametrize("gt,ps,expect", [
    (32,  16, 2),      # vLLM's default: a group straddles two blocks
    (32,  32, 1),      # --block-size 32: one group == one block
    (64,  64, 1),
    (128, 32, 4),
    (64,  16, 4),
])
def test_pages_per_group_is_the_precondition(gt, ps, expect):
    """``pages_per_group == 1`` is the only case in which a group is expressible as a
    block. Anything else needs a second, group-granular allocator that vLLM has no hook
    for -- which is exactly why this number is worth pinning."""
    pool = MultiSeqKVPool(
        head_dim=64, n_kv_heads=2, num_seqs=2, max_seq_len=256,
        window=WindowSpec(sink_tokens=8, recent_window=32, group_tokens=gt,
                          page_size=ps),
        outliers_per_vector=4, device="cpu")
    assert pool.pages_per_group == expect


def test_a_group_smaller_than_a_block_is_refused():
    """``gt % ps == 0`` is enforced by the store, not merely assumed. A fractional
    pages_per_group would make the page and group tables describe different geometries
    while both indexing the same planes."""
    with pytest.raises(ValueError, match="multiple of page_size"):
        MultiSeqKVPool(
            head_dim=64, n_kv_heads=2, num_seqs=1, max_seq_len=128,
            window=WindowSpec(sink_tokens=8, recent_window=32, group_tokens=16,
                              page_size=32),
            outliers_per_vector=4, device="cpu")


# ── the ownership invariant that any paged design must preserve ────────────────────

def test_no_table_row_can_address_another_requests_storage():
    """The safety property the current identity partition provides for free, written
    down because a block-table-sourced design would have to reproduce it deliberately:
    vLLM pads a short block-table row with block id 0, which is a REAL block that may
    belong to someone else. Today every entry of row b lies inside request b's own
    slice, so an out-of-range read is a stale read of the request's own storage rather
    than a cross-request leak."""
    pool = MultiSeqKVPool(
        head_dim=64, n_kv_heads=2, num_seqs=4, max_seq_len=256,
        window=WindowSpec(sink_tokens=8, recent_window=32, group_tokens=32,
                          page_size=32),
        outliers_per_vector=4, device="cpu")
    for b in range(pool.num_seqs):
        lo, hi = b * pool.P_per_seq, (b + 1) * pool.P_per_seq
        row = pool._req_to_token[b]
        assert int(row.min()) >= lo and int(row.max()) < hi, (
            f"req_to_token row {b} leaves its own page slice [{lo},{hi})")
        glo, ghi = b * pool.G_per_seq, (b + 1) * pool.G_per_seq
        grow = pool._req_to_group[b]
        assert int(grow.min()) >= glo and int(grow.max()) < ghi, (
            f"req_to_group row {b} leaves its own group slice [{glo},{ghi})")


# ── the check itself, not just the arithmetic ─────────────────────────────────────

@pytest.fixture
def fake_vllm(monkeypatch):
    """The version check runs first and reads ``vllm.__version__``; this suite has no
    vLLM. Same stub the shipped preflight tests use."""
    import sys
    import types

    mod = types.ModuleType("vllm")
    mod.__version__ = "0.21.0"
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return mod


def _stub_vllm_config(block_size):
    import types
    return types.SimpleNamespace(
        cache_config=types.SimpleNamespace(block_size=block_size,
                                           enable_prefix_caching=False),
        scheduler_config=types.SimpleNamespace(chunked_prefill_enabled=False,
                                               max_num_seqs=1),
        model_config=types.SimpleNamespace(max_model_len=4096))


@pytest.mark.parametrize("block_size,pages_per_group", [
    (16, 2),      # vLLM's default against group_tokens=32
    (32, 1),      # aligned: the precondition for a block-granular design
    (64, None),   # a group narrower than a block -- not a multiple, so undefined
    (24, None),   # not a multiple either
])
def test_preflight_reports_the_grid_relationship(fake_vllm, block_size,
                                                pages_per_group):
    """Pins that the check RUNS and populates the report, not merely that the arithmetic
    is right elsewhere.

    It caught nothing when first written because ``_attr`` returns ``(value, found)`` and
    the check compared the PAIR against ``int``, so the guard was permanently false and
    every field silently stayed "unknown". A serving run showed no warning where one was
    due, which is how it surfaced -- an assertion here would have been cheaper.
    """
    from ommx_gpu_serve.integration.vllm import preflight

    cfg = _cfg()
    report = preflight.ommx_preflight_check(
        _stub_vllm_config(block_size), num_seqs=1, cfg=cfg, num_layers=32, device=None)
    assert report["vllm_block_size"] == block_size
    assert report["ommx_group_tokens"] == cfg.group_tokens
    assert report["pages_per_group"] == pages_per_group


def test_an_engine_that_reports_no_block_size_is_unknown_not_guessed(fake_vllm):
    import types

    from ommx_gpu_serve.integration.vllm import preflight

    vc = _stub_vllm_config(32)
    vc.cache_config = types.SimpleNamespace(enable_prefix_caching=False)   # no block_size
    report = preflight.ommx_preflight_check(
        vc, num_seqs=1, cfg=_cfg(), num_layers=32, device=None)
    assert report["vllm_block_size"] == "unknown"


# ── the precondition is necessary, not sufficient ─────────────────────────────────

@pytest.mark.parametrize("sink,page_size,aligned", [
    (8,  32, False),   # the SHIPPED default: group 0 straddles two blocks
    (8,  16, False),
    (0,  32, True),    # only with no sink at all
    (32, 32, True),    # or a sink that is itself a whole number of blocks
    (16, 16, True),    # gt=32 over page 16 is two pages wide -- see gt below
])
def test_pages_per_group_one_is_not_the_same_as_block_aligned(sink, page_size, aligned):
    """``pages_per_group == 1`` makes a group one page WIDE. It does not make it start on
    a page boundary.

    The quantized region begins at ``sink``, not at 0 (``kv_window``: [0, sink) is bf16
    residual, [sink, boundary) is packed), so group g covers absolute tokens
    ``[sink + g*gt, sink + (g+1)*gt)``. At the shipped ``sink=8`` with ``--block-size 32``
    that is [8, 40) -- two blocks -- so a block-granular design still could not address a
    group as one block. Writing this down because the opposite was concluded once from
    ``pages_per_group == 1`` alone, and the arithmetic is a two-line check.
    """
    # hold pages_per_group == 1 so the parametrisation isolates ALIGNMENT: a group two
    # pages wide would straddle for a reason that has nothing to do with where it starts.
    gt = page_size
    lo, hi = sink, sink + gt
    pages = set(range(lo // page_size, (hi - 1) // page_size + 1))
    assert (len(pages) == 1) == aligned, (
        f"sink={sink} page={page_size}: group 0 = tokens [{lo},{hi}) spans pages "
        f"{sorted(pages)}")


def test_the_group_table_read_is_clamped_not_masked():
    """The kernel's two indirections are protected differently, and only one of them by a
    mask. The page-table load carries ``mask=token_mask``; the group-table load carries no
    mask at all and is bounded by an explicit ``tl.minimum``.

    That asymmetry decides what a block-table-sourced design may assume: "the kernel masks
    by b_seq_len" is true of the page table and false of the group table, so an
    out-of-range group column would be READ, not skipped. Pinned at source level because
    the alternative is a GPU.
    """
    import pathlib

    import ommx_gpu_serve

    src = (pathlib.Path(ommx_gpu_serve.__file__).parent
           / "attention" / "paged_decode.py").read_text()
    # two "g_next = tl.load(" sites: the loop preamble (a scalar at split_kv_start, in
    # range by construction) and the per-tile advance, which is the one that could run
    # past the live groups. Take the second.
    i = src.index("g_next = tl.load(", src.index("g_next = tl.load(") + 1)
    window = src[i:i + 400]
    assert "tl.minimum(" in window, "the group-table clamp is gone; check what replaced it"
    call = window[:window.index("\n\n")] if "\n\n" in window else window[:200]
    assert "mask=" not in call, (
        "the group-table load grew a mask -- if that is deliberate the clamp may now be "
        f"removable, but the paged design notes assume it is clamp-only:\n{call}")


def test_preflight_names_the_package_copy_it_is_running(fake_vllm):
    """An editable install and a PYTHONPATH can resolve to different checkouts, and the
    loser leaves no trace: every subsequent result describes code nobody is reading. This
    was not hypothetical -- a stale tree served this project's own runs until the resolved
    path was checked by hand. Reporting it costs one line and makes the failure visible."""
    import ommx_gpu_serve

    from ommx_gpu_serve.integration.vllm import preflight

    report = preflight.ommx_preflight_check(
        _stub_vllm_config(32), num_seqs=1, cfg=_cfg(), num_layers=32, device=None)
    assert report["package_path"] == ommx_gpu_serve.__file__
