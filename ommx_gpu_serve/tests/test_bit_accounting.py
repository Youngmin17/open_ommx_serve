# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pin the corrected KV bit accounting (``packed_only.kv_bits_breakdown``).

The retracted "~4.6x / <=3-bit" KV claim was an ACCOUNTING error, not a kernel error:
the formula omitted the two dedicated FP4 outlier-map planes and hardcoded a bf16
scale width. A published compression ratio is a load-bearing number, so it gets a
gate rather than a comment.

METHOD — the expected values are NEVER produced by the function under test. Two
independent derivations are computed here and cross-checked against each other
BEFORE either is compared to ``kv_bits_breakdown``:

  A. ``_bits_longhand`` — the arithmetic written out by hand in this file from the
     plane list in ``kv_pool.MultiSeqKVPool.__init__``.
  B. ``_bits_from_allocated_pool`` — measured off a REAL ``MultiSeqKVPool``
     constructed on CPU: each plane's own ``shape[-1]`` and ``element_size()``,
     divided by the elements one scale-group covers. This is the derivation that
     cannot drift, because it reads the allocation itself.

Only then is ``kv_bits_breakdown`` asserted equal. If a future recipe change moves
the real footprint, (B) moves with it and the pinned literals fail loudly — which is
the intended behaviour, not a nuisance.

CANONICAL PUBLISHED RECIPE (README / run.sh): ``OMMX_ATTN_K_FORMAT=i2f4
OMMX_ATTN_OUTLIERS=6 OMMX_ATTN_POW2=1 OMMX_KV_GROUP_TOKENS=32
OMMX_KV_GROUP_CHANNELS=32 OMMX_KV_SINK=8 OMMX_KV_RECENT=32 OMMX_KV_RING=1``, D=128,
relidx7 indices, dedicated FP4 outlier map ON (the default) ->

    K = 6.000  V = 2.750  avg = 4.375  total = 8.750 bit/elem  (3.657x vs bf16)

All these are exact binary fractions, so the comparisons are exact (``abs=1e-12``).
"""
from __future__ import annotations

import pytest
import torch

from ommx_gpu_serve.attention.kv_pool import MultiSeqKVPool
from ommx_gpu_serve.attention.kv_window import WindowSpec

_IMPORT_ERR = None
try:
    from ommx_gpu_serve.integration.vllm.packed_only import (
        kv_bits_breakdown,
        ommx_bits_per_elem,
        packed_compression_ratio,
    )
except ImportError as exc:  # pragma: no cover - exercised only before the fix lands
    _IMPORT_ERR = f"{type(exc).__name__}: {exc}"

_XFAIL_MISSING = pytest.mark.xfail(
    _IMPORT_ERR is not None,
    reason=("ommx_gpu_serve.integration.vllm.packed_only.kv_bits_breakdown is not "
            f"importable ({_IMPORT_ERR}) — the corrected KV bit accounting has not "
            "landed. XFAIL, not skip: the gate is present and failing, not absent."),
    strict=True,
)

# The pinned canonical numbers. Exact binary fractions; see the module docstring.
CANON_K_BITS = 6.0
CANON_V_BITS = 2.75
CANON_AVG_BITS = 4.375
CANON_TOTAL_BITS = 8.75
EXACT = dict(rel=0.0, abs=1e-12)

CANONICAL = dict(
    head_dim=128,
    group_tokens=32,
    group_channels=32,
    outliers_per_vector=6,
    outlier_repr="relidx7",
    kv_outlier_map=True,
    use_pow2=True,
)

# A second, DIFFERENT number system (pack.py's "<=3-bit KV lever"): gt=gc=64 with the
# dedicated outlier map OFF. Included so the arithmetic cannot be a hardcode of the
# canonical answer.
LEVER = dict(
    head_dim=128,
    group_tokens=64,
    group_channels=64,
    outliers_per_vector=3,
    outlier_repr="relidx7",
    kv_outlier_map=False,
    use_pow2=True,
)

# A THIRD recipe, added because the two above are DEGENERATE in two ways and let real
# accounting bugs through (mutation-tested: both mutants below survived without it):
#   * both have group_tokens == group_channels, so swapping the V-plane denominator
#     (``/ gc`` -> ``/ gt``) changes nothing;
#   * both have k in {3, 6}, where relidx7's ceil(7k/8) index bytes happen to equal a
#     plain 8-bit ceil(8k/8) (3 and 6 respectively). At k=8 they differ (7 vs 8).
# gt=32 != gc=64 and k=8 make both mistakes visible.
ASYMMETRIC = dict(
    head_dim=128,
    group_tokens=32,
    group_channels=64,
    outliers_per_vector=8,
    outlier_repr="relidx7",
    kv_outlier_map=True,
    use_pow2=True,
)


def _bits_longhand(*, head_dim: int, group_tokens: int, group_channels: int,
                   outliers_per_vector: int, outlier_repr: str,
                   kv_outlier_map: bool, use_pow2: bool):
    """Derivation A: the arithmetic written out here, from kv_pool.py's plane list.

    Units: bits per (head, token, head-dim channel). ``head_dim`` cancels out of every
    term (the group planes are ``[..., D]`` and the V planes are ``[..., D // gc]``),
    so it is accepted and validated but never used — asserting that is itself a test.
    """
    gt = int(group_tokens)
    gc = int(group_channels)
    k = int(outliers_per_vector)
    assert head_dim % gc == 0, "head_dim must be a multiple of group_channels"
    # kv_pool.py: scl_dt = int8 iff (env-resolved kv_int8_scale AND use_pow2); the
    # zero point, the FP4 map planes and every other bf16 plane stay 2 bytes.
    scale_bits = 8 if use_pow2 else 16
    zp_bits = 16

    # K: k_base is D//4 bytes per (page-slot, token, head) covering D channels -> 2 b/elem.
    k_bits = 2.0
    per_group_bits = scale_bits + zp_bits                 # k_scale + k_zp
    if k > 0:
        assert outlier_repr == "relidx7", "only relidx7 is derived longhand here"
        per_group_bits += 8 * ((7 * k + 7) // 8)          # k_oidx: 7-bit indices
        per_group_bits += 8 * ((k + 1) // 2)              # k_oval: FP4 nibbles
        if kv_outlier_map:
            per_group_bits += 16 + 16                     # k_fp4_mapscale + mapcenter
    k_bits += per_group_bits / gt

    # V: v_main is D//4 bytes -> 2 b/elem; v_scale/v_zp are one value per gc channels.
    v_bits = 2.0 + (scale_bits + zp_bits) / gc
    return k_bits, v_bits


def _bits_from_allocated_pool(*, head_dim: int, group_tokens: int,
                              group_channels: int, outliers_per_vector: int,
                              outlier_repr: str, kv_outlier_map: bool,
                              use_pow2: bool):
    """Derivation B: measured off a real ``MultiSeqKVPool``'s allocated tensors.

    Sums the bytes each plane devotes to ONE scale-group (``pages_per_group`` pages,
    or one group row) and divides by ``group_tokens * n_kv_heads * head_dim`` — the
    elements that group covers. Reads only ``shape``/``dtype`` off the pool, so it
    tracks the allocator automatically.
    """
    win = WindowSpec(sink_tokens=8, recent_window=32,
                     group_tokens=group_tokens, page_size=16)
    pool = MultiSeqKVPool(
        head_dim=head_dim, n_kv_heads=2, num_seqs=1, max_seq_len=4 * group_tokens,
        k_format="i2f4", outliers_per_vector=outliers_per_vector,
        outlier_select="signed", outlier_repr=outlier_repr, use_pow2=use_pow2,
        kv_outlier_map=kv_outlier_map, window=win, group_channels=group_channels,
        device="cpu")
    H, D = pool.H, pool.D
    tokens_per_group = pool.pages_per_group * pool.ps   # == group_tokens
    assert tokens_per_group == pool.gt
    elems = pool.gt * H * D

    def page_bytes(t):
        """Bytes one group's worth of a PAGE-indexed plane ``[P, ps, H, w]`` occupies."""
        return pool.pages_per_group * pool.ps * H * int(t.shape[-1]) * t.element_size()

    def group_bytes(t):
        """Bytes one group row of a GROUP-indexed plane ``[G, H, D(, fb)]`` occupies."""
        trailing = int(t.shape[-1]) if t.dim() == 4 else 1
        return H * D * trailing * t.element_size()

    k_bytes = page_bytes(pool.k_base) + group_bytes(pool.k_scale) + group_bytes(pool.k_zp)
    for plane in (pool.k_oidx, pool.k_crank, pool.k_oval,
                  pool.k_fp4_mapscale, pool.k_fp4_mapcenter):
        if plane is not None:
            k_bytes += group_bytes(plane)
    v_bytes = (page_bytes(pool.v_main) + page_bytes(pool.v_scale)
               + page_bytes(pool.v_zp))
    return k_bytes * 8.0 / elems, v_bytes * 8.0 / elems


# ── the two independent derivations must agree, before anything is pinned ───────

@pytest.mark.parametrize("recipe,name", [(CANONICAL, "canonical"), (LEVER, "lever"),
                                         (ASYMMETRIC, "asymmetric")])
def test_longhand_arithmetic_matches_the_real_allocation(recipe: dict, name: str) -> None:
    """Derivation A == derivation B. Neither involves ``packed_only``.

    A failure here means the hand arithmetic in this file has drifted from what
    ``MultiSeqKVPool`` actually allocates — fix the test, not the accounting.
    """
    a_k, a_v = _bits_longhand(**recipe)
    b_k, b_v = _bits_from_allocated_pool(**recipe)
    assert a_k == pytest.approx(b_k, **EXACT), (
        f"{name}: longhand K={a_k} != allocated K={b_k}")
    assert a_v == pytest.approx(b_v, **EXACT), (
        f"{name}: longhand V={a_v} != allocated V={b_v}")


def test_canonical_recipe_is_6_and_2_75_bits() -> None:
    """The published numbers, from the allocation itself.

    This is the assertion the retracted claim would have failed: the old accounting
    gave K=5.25/V=3.00 (map planes omitted, bf16 scale hardcoded).
    """
    k_bits, v_bits = _bits_from_allocated_pool(**CANONICAL)
    assert k_bits == pytest.approx(CANON_K_BITS, **EXACT), f"K bits/elem = {k_bits}"
    assert v_bits == pytest.approx(CANON_V_BITS, **EXACT), f"V bits/elem = {v_bits}"
    assert (k_bits + v_bits) / 2.0 == pytest.approx(CANON_AVG_BITS, **EXACT)
    assert k_bits + v_bits == pytest.approx(CANON_TOTAL_BITS, **EXACT)


def test_bits_per_element_are_independent_of_head_dim() -> None:
    """D cancels out of every per-element term; if it stops cancelling, something
    is being amortized over the wrong denominator."""
    base = _bits_from_allocated_pool(**CANONICAL)
    for head_dim in (64, 128, 256):
        alt = _bits_from_allocated_pool(**{**CANONICAL, "head_dim": head_dim})
        assert alt[0] == pytest.approx(base[0], **EXACT), f"K moved at D={head_dim}"
        assert alt[1] == pytest.approx(base[1], **EXACT), f"V moved at D={head_dim}"


def test_scale_dtype_and_map_planes_are_really_what_the_recipe_says() -> None:
    """Non-vacuity: the two corrections the audit found are actually present.

    Without this, ``6.0`` could be reached by an unrelated combination of planes.
    """
    win = WindowSpec(sink_tokens=8, recent_window=32, group_tokens=32, page_size=16)
    pool = MultiSeqKVPool(head_dim=128, n_kv_heads=2, num_seqs=1, max_seq_len=128,
                          k_format="i2f4", outliers_per_vector=6,
                          outlier_select="signed", outlier_repr="relidx7",
                          use_pow2=True, kv_outlier_map=True, window=win,
                          group_channels=32, device="cpu")
    assert pool.k_scale.dtype is torch.int8, (
        "OMMX_ATTN_POW2=1 must store the K scale as an int8 pow2 exponent")
    assert pool.v_scale.dtype is torch.int8, "V scale should be the int8 pow2 exponent"
    assert pool.k_zp.dtype is torch.bfloat16, "the zero point is never 2^e -> stays bf16"
    assert pool.k_fp4_mapscale is not None and pool.k_fp4_mapcenter is not None, (
        "the dedicated FP4 outlier-map planes are allocated by default and were the "
        "planes the retracted ~4.6x accounting omitted")
    assert pool.k_fp4_mapscale.dtype is torch.bfloat16
    assert int(pool.k_oidx.shape[-1]) == 6, "relidx7 at k=6 is ceil(7*6/8) = 6 bytes"
    assert int(pool.k_oval.shape[-1]) == 3, "FP4 nibbles at k=6 is ceil(6/2) = 3 bytes"


# ── the function under test, checked against the derivations above ──────────────

@_XFAIL_MISSING
def test_kv_bits_breakdown_matches_the_independent_derivation() -> None:
    """``kv_bits_breakdown`` (explicit kwargs) == the allocation-derived numbers."""
    for recipe, name in ((CANONICAL, "canonical"), (LEVER, "lever"),
                         (ASYMMETRIC, "asymmetric")):
        exp_k, exp_v = _bits_from_allocated_pool(**recipe)
        kw = {k: v for k, v in recipe.items() if k != "head_dim"}
        got = kv_bits_breakdown(recipe["head_dim"], **kw)
        assert got["k_bits_per_elem"] == pytest.approx(exp_k, **EXACT), (
            f"{name}: k_bits_per_elem={got['k_bits_per_elem']} != allocated {exp_k} "
            f"(planes={got['k_planes']})")
        assert got["v_bits_per_elem"] == pytest.approx(exp_v, **EXACT), (
            f"{name}: v_bits_per_elem={got['v_bits_per_elem']} != allocated {exp_v} "
            f"(planes={got['v_planes']})")
        assert got["total_bits_per_elem"] == pytest.approx(exp_k + exp_v, **EXACT), (
            f"{name}: total={got['total_bits_per_elem']} != {exp_k + exp_v}")
        assert got["avg_bits_per_elem"] == pytest.approx(
            (exp_k + exp_v) / 2.0, **EXACT), (
            f"{name}: avg={got['avg_bits_per_elem']} != {(exp_k + exp_v) / 2.0}")
        # the per-plane dicts must actually sum to the reported totals (no stray term)
        assert sum(got["k_planes"].values()) == pytest.approx(exp_k, **EXACT), (
            f"{name}: k_planes {got['k_planes']} do not sum to {exp_k}")
        assert sum(got["v_planes"].values()) == pytest.approx(exp_v, **EXACT), (
            f"{name}: v_planes {got['v_planes']} do not sum to {exp_v}")


@_XFAIL_MISSING
def test_kv_bits_breakdown_pins_the_canonical_recipe_from_the_environment() -> None:
    """The env spelling of the published recipe resolves to the same 6.0 / 2.75.

    The recipe ships as environment variables, so the env RESOLUTION path — not just
    the kwargs path — is what a served run actually exercises.
    """
    import os

    env = {
        "OMMX_ATTN_K_FORMAT": "i2f4",
        "OMMX_ATTN_OUTLIERS": "6",
        "OMMX_ATTN_POW2": "1",
        "OMMX_KV_GROUP_TOKENS": "32",
        "OMMX_KV_GROUP_CHANNELS": "32",
        "OMMX_KV_SINK": "8",
        "OMMX_KV_RECENT": "32",
        "OMMX_KV_RING": "1",
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        got = kv_bits_breakdown(128)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    resolved = got["recipe"]
    assert resolved["outliers_per_vector"] == 6, f"resolved recipe: {resolved}"
    assert resolved["kv_int8_scale"] is True, (
        f"OMMX_ATTN_POW2=1 did not resolve the int8 pow2 scale: {resolved}")
    assert resolved["kv_outlier_map"] is True, (
        f"the dedicated FP4 outlier map is on by default: {resolved}")
    assert got["k_bits_per_elem"] == pytest.approx(CANON_K_BITS, **EXACT), (
        f"K={got['k_bits_per_elem']} != {CANON_K_BITS} (planes={got['k_planes']})")
    assert got["v_bits_per_elem"] == pytest.approx(CANON_V_BITS, **EXACT), (
        f"V={got['v_bits_per_elem']} != {CANON_V_BITS} (planes={got['v_planes']})")
    assert got["avg_bits_per_elem"] == pytest.approx(CANON_AVG_BITS, **EXACT), (
        f"avg={got['avg_bits_per_elem']} != {CANON_AVG_BITS}")
    assert got["total_bits_per_elem"] == pytest.approx(CANON_TOTAL_BITS, **EXACT), (
        f"total={got['total_bits_per_elem']} != {CANON_TOTAL_BITS}")
    assert got["compression_ratio"] == pytest.approx(
        32.0 / CANON_TOTAL_BITS, rel=1e-9), (
        f"compression_ratio={got['compression_ratio']}")
    # the retracted claims must not be reachable from the published recipe.
    assert got["compression_ratio"] < 4.0, (
        f"compression_ratio={got['compression_ratio']} — the retracted ~4.6x claim is "
        f"back; the canonical recipe compresses {32.0 / CANON_TOTAL_BITS:.3f}x")
    assert got["k_bits_per_elem"] > 3.0, (
        "K <= 3 bit is a DIFFERENT recipe (gt=64 + OMMX_KV_OUTLIER_MAP=0), not this one")


@_XFAIL_MISSING
def test_public_helpers_agree_with_the_breakdown() -> None:
    """``ommx_bits_per_elem`` / ``packed_compression_ratio`` are the same number."""
    kw = {k: v for k, v in CANONICAL.items() if k != "head_dim"}
    assert ommx_bits_per_elem(128, **kw) == pytest.approx(CANON_TOTAL_BITS, **EXACT)
    assert packed_compression_ratio(128, **kw) == pytest.approx(
        32.0 / CANON_TOTAL_BITS, rel=1e-9)


@_XFAIL_MISSING
def test_bad_recipe_raises_instead_of_silently_defaulting() -> None:
    """Project law (no silent fallback): an unbuildable recipe must not price."""
    with pytest.raises(ValueError):
        kv_bits_breakdown(128, group_tokens=48)          # not in {16,32,64,128}
    with pytest.raises(ValueError):
        kv_bits_breakdown(128, group_channels=48)
    with pytest.raises(ValueError):
        kv_bits_breakdown(100)                            # head_dim not a multiple of 32
    with pytest.raises(ValueError):
        kv_bits_breakdown(128, k_format="i4f8")           # not a KV format
    with pytest.raises(ValueError):
        kv_bits_breakdown(128, group_tokens=32, outliers_per_vector=33)
