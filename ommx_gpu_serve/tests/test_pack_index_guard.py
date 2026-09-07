# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The outlier-gather index guard checks BOTH bounds, because ATen does.

WHY IT EXISTS. ``ommx_pack_kv_canonical_block`` gathers the outlier values with
``fpc.gather(-1, oidx)``. ATen's kernel asserts ``idx_dim >= 0 && idx_dim < index_size``
and, on failure, aborts the whole CUDA context with a message that names no tensor, no
line and no value -- and does it asynchronously, so the traceback points at whatever ran
next (in a real engine: the embedding lookup, several layers away). The guard exists to
turn that into the group's own numbers before the launch.

WHAT WAS WRONG. It tested ``oidx.max() >= gt`` only. A NEGATIVE index passed the check
and then aborted the device anyway -- the guard would have been silent through exactly the
failure it is there to explain. Both bounds are now checked, and the message reports the
range rather than one end of it.

REACHABILITY, measured rather than assumed. The obvious way in -- asking for more
outliers than a group has tokens, so the selector's ``gt`` sentinel survives the sort --
never reaches the guard: ``outliers_per_vector`` is range-checked against ``group_tokens``
at entry and raises ``outliers_per_vector=36 out of range [0, 32]`` first. Nor does any
current selector emit a negative. So the guard is DEFENSIVE on both ends: it is the net
under a future selector or a caller that reaches the internals, not a check the public API
can trip. It is pinned at the predicate, which is what can honestly be pinned; the entry
validator that actually does the refusing is pinned separately below.

This distinction is the point of the file. An earlier draft of these tests asserted that
``k > gt`` "is named by the guard", passed, and proved nothing -- the exception it caught
came from the entry validator several hundred lines earlier.
"""
import pytest

torch = pytest.importorskip("torch")

from ommx_gpu_serve.attention.pack import (  # noqa: E402
    ommx_pack_kv_canonical_block,
)

GT, H, D = 32, 2, 64


def _kv(T=128, seed=7):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(T, H, D, generator=g).to(torch.bfloat16)
    V = torch.randn(T, H, D, generator=g).to(torch.bfloat16)
    return K, V


def _pack(**over):
    K, V = _kv()
    kw = dict(outliers_per_vector=6, group_tokens=GT, group_channels=32,
              page_size=16, k_format="i2f4")
    kw.update(over)
    return ommx_pack_kv_canonical_block(K, V, **kw)


def test_a_sane_recipe_packs():
    """The control: without it, a guard that rejected everything would look correct."""
    planes = _pack()
    assert planes, "the baseline pack produced nothing"


@pytest.mark.parametrize("select", ["signed", "abs"])
def test_the_entry_validator_refuses_k_over_group_tokens_before_the_guard(select):
    """The refusal that actually fires, attributed to the code that fires it.

    This is the ENTRY range check, not the gather guard -- it reports the offending
    number and the legal interval, and it is why the guard's upper bound is unreachable
    from the public API. Asserted on the message so that a future refactor which drops
    this check (and lets the sentinel reach the gather) fails here rather than on a GPU.
    """
    with pytest.raises(ValueError, match=r"outliers_per_vector=36 out of range"):
        _pack(outliers_per_vector=GT + 4, outlier_select=select)


def test_the_guard_tests_the_lower_bound_too():
    """Pinned at the predicate: ``oidx.min()`` must participate.

    The first version compared only ``oidx.max()`` against ``gt``, so a negative index --
    which ATen rejects just as hard -- passed the check and aborted the device anyway.
    """
    import inspect

    import ommx_gpu_serve.attention.pack as pack_mod

    src = inspect.getsource(pack_mod)
    i = src.index("_PACK_INDEX_CHECK:")
    block = src[i:i + 900]
    assert "oidx.min()" in block, (
        "the outlier-index guard no longer looks at the lower bound; ATen asserts "
        "idx_dim >= 0 as well as idx_dim < index_size")
    assert "_omin < 0" in block, "the lower bound is computed but not tested"


def test_the_clamp_diagnostic_is_off_by_default():
    """``OMMX_KV_PACK_CLAMP`` silently corrupts packed values in exchange for surviving an
    out-of-range index. It is a bisection tool for device-side asserts; if it were ever on
    by default, a real indexing bug would serve wrong numbers instead of raising."""
    import ommx_gpu_serve.attention.pack as pack_mod

    assert pack_mod._PACK_INDEX_CLAMP is False
    assert pack_mod._PACK_INDEX_CHECK is True, (
        "the guard must stay on by default; it costs one .item() per pack")
