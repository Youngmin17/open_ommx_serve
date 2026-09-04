# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The ``npu`` bit-accounting basis: what a recipe costs on Nucleus, not on a GPU.

WHY IT EXISTS. A published AvgBits figure is meaningless without saying which encoding
it counts, and OMMX has two that differ by more than the figure itself. The GPU writes
positions as relidx7 (7 bits per outlier, byte padded per group) or a flat bitmap
(one bit per element); the paper specifies the NPU as storing the same positions in
``ceil(log2 C(N,K))`` bits per group via an on-chip combinatorial codec, and says the
rest of the bundle is platform-invariant. Those two sentences pin a second budget
exactly, so it can be COMPUTED and compared rather than asserted -- which is the whole
point, because the two differ by 1.0 b/wt at gs=64/npv=16 and 1.3 b/wt at gs=16/npv=4.

WHAT IT IS NOT. No plane is ever written in this encoding and no kernel here reads one.
``npu`` is a reporting column. ``test_npu_basis_is_not_a_storage_option`` pins that, so
nobody later reads the number as something this packer can emit.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from ommx_gpu_serve.linear.quantize import (  # noqa: E402
    OUTLIER_MAPS,
    OUTLIER_REPRS,
    combinadic_index_bits,
)
from ommx_gpu_serve.linear.w_format import Recipe  # noqa: E402

GRID = [(gs, npv) for gs in (16, 32, 64, 128) for npv in (1, 2, 4, 8, 16, 32)
        if npv <= gs]


# ── the formula ────────────────────────────────────────────────────────────────────

def test_combinadic_bits_have_not_drifted_from_the_kv_codec():
    """``linear/`` keeps its own copy rather than importing ``attention/``. That is only
    safe while the two agree, so the agreement is asserted rather than assumed."""
    from ommx_gpu_serve.attention.codec import combinadic_index_bits as kv
    for n in (4, 8, 16, 32, 64, 128, 256):
        for k in range(0, n + 1):
            assert combinadic_index_bits(k, n) == kv(k, n), f"drift at n={n} k={k}"


@pytest.mark.parametrize("gs,npv", GRID)
def test_combinadic_bits_are_exactly_ceil_log2_binomial(gs, npv):
    got = combinadic_index_bits(npv, gs)
    if npv >= gs:                       # one subset: nothing to encode
        assert got == 0
        return
    total = math.comb(gs, npv)
    assert got == math.ceil(math.log2(total))
    assert 2 ** got >= total > 2 ** (got - 1), "not the MINIMAL field width"


def test_a_full_group_of_outliers_costs_no_position_bits():
    """C(n,n) == 1. A basis that charged for it would make npv==gs look worse than
    npv==gs-1, which is backwards."""
    assert combinadic_index_bits(64, 64) == 0      # npv == gs
    assert combinadic_index_bits(63, 64) > 0       # npv == gs - 1


# ── the basis ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gs,npv", GRID)
def test_npu_basis_differs_from_stored_only_in_the_position_term(gs, npv):
    """The paper's justification for a second budget is 'position-metadata regeneration
    per target', so every OTHER plane must be identical between the two bases. If this
    ever fails, the npu column has quietly become a different recipe rather than a
    different encoding of the same one."""
    b = Recipe(gs, npv, npv / gs, "relidx7", "idx_range").bits_breakdown()
    for plane in ("base_int2", "scale_e8m0", "zero_point", "outlier_fp4",
                  "fp4_range_map"):
        assert b["npu"][plane] == b["stored"][plane], plane
    assert b["npu"]["outlier_index"] == combinadic_index_bits(npv, gs) / gs
    assert abs(sum(v for k, v in b["npu"].items() if k != "total")
               - b["npu"]["total"]) < 1e-12


@pytest.mark.parametrize("gs,npv", GRID)
def test_npu_basis_is_invariant_to_the_gpu_position_encoding(gs, npv):
    """relidx7 and bitmap are two GPU spellings of one position SET; the NPU re-derives
    the set. A recipe's npu figure must therefore not move when repr does -- otherwise
    quoting it would require also quoting a GPU encoding, which defeats its purpose."""
    a = Recipe(gs, npv, npv / gs, "relidx7", "idx_range").bits_breakdown()
    c = Recipe(gs, npv, npv / gs, "bitmap", "idx_range").bits_breakdown()
    assert a["bits_per_weight_npu"] == c["bits_per_weight_npu"]


@pytest.mark.parametrize("gs,npv", GRID)
@pytest.mark.parametrize("mp", OUTLIER_MAPS)
def test_npu_basis_never_exceeds_either_gpu_encoding(gs, npv, mp):
    """The combinadic field is the minimum width that can name a k-subset, so it cannot
    cost more than relidx7 (7 bits/slot, padded) or a bitmap (gs bits). A violation
    means the formula, not the encoding, is wrong."""
    n = Recipe(gs, npv, npv / gs, "relidx7", mp).bits_breakdown()["bits_per_weight_npu"]
    for rp in OUTLIER_REPRS:
        stored = Recipe(gs, npv, npv / gs, rp, mp).bits_breakdown()["bits_per_weight"]
        assert n <= stored + 1e-12, f"npu {n} > {rp} stored {stored}"


def test_a_recipe_without_outliers_has_no_position_cost_on_either_basis():
    b = Recipe(64, 0, 0.0, "relidx7", "none").bits_breakdown()
    assert b["npu"]["outlier_index"] == 0.0
    assert b["bits_per_weight_npu"] == b["bits_per_weight"] == b["stored"]["total"]


def test_npu_basis_is_not_a_storage_option():
    """'combinadic' is a KV *storage* representation but NOT a weight one. If it ever
    became one, this basis stops being a cross-platform statement and becomes just
    another column of ``stored`` -- and the docstring above would be wrong."""
    assert "combinadic" not in OUTLIER_REPRS
    with pytest.raises(Exception):
        Recipe(64, 4, 0.0625, "combinadic", "idx_range").validate()


# ── the numbers this project actually quotes ───────────────────────────────────────

@pytest.mark.parametrize("gs,npv,stored,npu", [
    # The measured accuracy grid (Llama-3.1-8B, wikitext, calibrated, all-INT2), so a
    # bit figure and the perplexity it belongs to cannot drift apart silently.
    (16,   4, 10.5000, 9.1875),   # sim 7.7082  served 7.8950
    (32,   8,  7.5000, 6.5000),   # sim 8.2845
    (64,  16,  6.1250, 5.1406),   # sim 8.7400  served 9.0098  <- optimized-weight
    (128, 32,  5.4375, 4.4766),   # sim 9.2593  served 132.65  <- diverges, see below
    (64,   8,  4.7500, 4.3906),   # sim 11.4614
    (128, 16,  4.0625, 3.7109),   # sim 12.2442 -- sub-4 on the NPU basis, unserved
    (128,  8,  3.3750, 3.2578),   # sim 18.8577
    (64,   4,  4.1250, 3.9375),   # sim 5072.97 -- destroyed; the cheap end is not usable
])
def test_quoted_budgets_reproduce(gs, npv, stored, npu):
    b = Recipe(gs, npv, npv / gs, "relidx7", "idx_range").bits_breakdown()
    assert round(b["bits_per_weight"], 4) == stored
    assert round(b["bits_per_weight_npu"], 4) == npu


def test_optimized_weight_is_the_served_point_not_the_cheapest_simulated_one():
    """gs=128/npv=32 prices better AND scores better on the simulator (4.4766 bits, PPL
    9.2593 against 5.1406 / 8.7400), so a future edit that picks a recipe by those two
    columns alone would land on it. Served, its per-window perplexity is
    [9.84, 14.68, 8.61, 248947.69] -- the model diverges inside a window and never
    recovers -- while gs=64/npv=16 gives [9.65, 13.11, 7.32, 7.12]. The kernel is exact
    for both (real-bundle parity, all 224 modules, median cos 0.99999774 vs 0.99999768),
    so nothing downstream will catch the swap. This test is the thing that would."""
    from ommx_gpu_serve.recipes import RECIPES
    args = RECIPES["optimized-weight"].packer_args
    assert (args["group_size"], args["outlier_pct"]) == (64, 0.25), (
        "optimized-weight moved off the SERVED operating point; a recipe here must have "
        "been measured through the CUDA kernel end to end, not only on the simulator")
