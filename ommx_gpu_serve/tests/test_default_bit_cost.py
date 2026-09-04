# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The published bit figure must price the encoding the default actually serves.

WHY IT EXISTS. `test_bit_accounting.py` derives the canonical bundle longhand and pins
`CANON_AVG_BITS = 4.375` -- but every case in it sets `outlier_repr="relidx7"`, and it says so
("only relidx7 is derived longhand here"). That was correct while relidx7 was the default. It
stopped being correct the moment the default became `bitmap`, and nothing failed: the README
went on advertising **4.375** average KV bits beside a recipe it described, correctly, as
using flat-bitmask positions. The two sentences named different formats.

The gap is 0.25 bits per element and it is in the DIRECTION THAT FLATTERS US -- a bitmap
position costs 1.000 b/elem against relidx7's 1.500, so the shipped configuration is cheaper
than the number published for it (`packed_only.py`'s own table: relidx7 K 6.000 / avg 4.375,
bitmap K 5.500 / avg 4.125). An overstated bit cost is still a wrong bit cost, and on the axis
this paper argues about it is the one number a reader checks first.

This test pins the RELATIONSHIP rather than a literal: whatever `resolve_serving_config()`
resolves to with no environment, that encoding's price is what the accounting must report.
"""
import pytest

torch = pytest.importorskip("torch")

from ommx_gpu_serve.integration.vllm.config import resolve_serving_config  # noqa: E402

#: packed_only.py's own table for the canonical recipe (gt=32, k=6, map ON, pow2).
#: avg = (K + V) / 2 with V fixed at 2.750.
PRICE = {
    "relidx7":   {"index_bits": 1.500, "k": 6.000, "avg": 4.375},
    "bitmap":    {"index_bits": 1.000, "k": 5.500, "avg": 4.125},
    "combinadic": {"index_bits": 0.750, "k": 5.250, "avg": 4.000},
}


@pytest.fixture
def default_cfg(monkeypatch):
    import os

    for k in [k for k in os.environ if k.startswith("OMMX_")]:
        monkeypatch.delenv(k, raising=False)
    return resolve_serving_config()


def test_the_default_encoding_has_a_published_price(default_cfg):
    """A fourth encoding must arrive with its bit cost, not just its kernel."""
    assert default_cfg.outlier_repr in PRICE, (
        f"default outlier_repr={default_cfg.outlier_repr!r} has no entry in this table; add "
        f"its index-bits cost here and to packed_only.py's docstring table")


def test_the_pool_accounting_defaults_to_the_serving_encoding(default_cfg):
    """packed_only prices the pool the engine allocates. Its env fallback for the encoding
    was a second literal (relidx7) after the serving default moved to bitmap, so a bare
    ``kv_bits_breakdown()`` / ``shrunk_head_size()`` priced 8.750 bit while the engine
    allocated 8.250 (+6%). Both must resolve through config.py's one constant."""
    from ommx_gpu_serve.integration.vllm import packed_only as po

    bare = po.kv_bits_breakdown(128)
    assert bare["recipe"]["outlier_repr"] == default_cfg.outlier_repr
    assert bare["recipe"]["outlier_index_plane"] == "k_obmp"
    assert po.ommx_bits_per_elem(128) == po.ommx_bits_per_elem(
        128, outlier_repr=default_cfg.outlier_repr)
    want = max(8, int(round(128 * po.ommx_bits_per_elem(
        128, outlier_repr=default_cfg.outlier_repr) / 32.0 / 8.0)) * 8)
    assert po.shrunk_head_size(128) == want


def test_the_index_cost_matches_the_encoding_arithmetic(default_cfg):
    """Derived, not restated: relidx7 packs 7 bits per outlier byte-padded, the bitmask is one
    bit per position in the group, and combinadic is the rank of the chosen set."""
    import math

    gt = default_cfg.group_tokens
    k = default_cfg.outliers_per_vector
    repr_ = default_cfg.outlier_repr
    if repr_ == "relidx7":
        want_bytes = (7 * k + 7) // 8
    elif repr_ == "bitmap":
        want_bytes = (gt + 7) // 8
    else:
        want_bytes = (math.comb(gt, k).bit_length() + 7) // 8
    got = want_bytes * 8 / gt
    assert abs(got - PRICE[repr_]["index_bits"]) < 1e-9, (
        f"{repr_} at gt={gt} k={k} costs {got} index bits/elem, but the table says "
        f"{PRICE[repr_]['index_bits']}")


def test_the_figure_quotes_the_bit_width_on_the_npu_basis():
    """The bit claim moved from the README onto the figure canvas, and its BASIS moved with it.

    The three outlier-position encodings are membership-equivalent -- identical decoded values,
    different storage -- so "OMMX stores N bits" means nothing until it names which encoding it
    prices. The figures now quote the NPU basis (combinadic), the platform-invariant one: 4.00
    average KV bits at the canonical recipe, against the GPU's 4.125 (bitmap) and 4.375
    (relidx7). Pinned at the plot source, because that is where the claim now lives.
    """
    import pathlib

    import ommx_gpu_serve

    plot = pathlib.Path(ommx_gpu_serve.__file__).parent.parent / "figure" / "plot.py"
    if not plot.exists():
        pytest.skip("figure/plot.py not present")
    src = plot.read_text()
    assert "NPU basis" in src, (
        "the TPOT figure no longer names the basis of its bit figure; an unbased number is "
        "ambiguous across three membership-equivalent encodings")
    want = f"{PRICE['combinadic']['avg']:.2f} avg bits"
    assert want in src, (
        f"the figure must quote the NPU-basis average ({want}), not a GPU encoding's price")
    for r, v in PRICE.items():
        if r == "combinadic":
            continue
        assert f"{v['avg']:.2f} avg bits" not in src, (
            f"the figure quotes {v['avg']:.2f} avg bits, which prices {r}, not the NPU basis")


def test_the_readme_makes_no_unbased_bit_claim():
    """The README carried "4.375 average KV bits" beside a recipe it described, correctly, as
    flat-bitmask -- two formats in adjacent sentences. The claim has been removed; this gate is
    that it cannot come back WITHOUT naming a basis."""
    import pathlib

    import ommx_gpu_serve

    readme = pathlib.Path(ommx_gpu_serve.__file__).parent.parent / "README.md"
    if not readme.exists():
        pytest.skip("README.md not present")
    flat = " ".join(readme.read_text().split())
    i = flat.find("average KV bits")
    if i < 0:
        return                                    # no claim at all: correct
    s0 = flat.rfind(". ", 0, i) + 2
    s1 = flat.find(". ", i)
    sentence = flat[s0:s1 if s1 > 0 else len(flat)]
    assert "NPU" in sentence or "basis" in sentence, (
        f"the README states an average-KV-bits figure without naming its basis, which is "
        f"ambiguous across the three encodings: {sentence[:170]}")


# ── the encoding default ───────────────────────────────────────────────────────

def test_the_default_encoding_is_bitmap_and_is_decodable(default_cfg):
    """bitmap is the main path. It must also RESOLVE to a readable one: the store allocates
    k_obmp and no k_oidx, so a default that did not also resolve bitmap_read would raise on
    the first quantized group."""
    assert default_cfg.outlier_repr == "bitmap"
    assert default_cfg.resolved_bitmap_read() is True
    assert default_cfg.resolved_combinadic_read() is False


def test_the_default_outlier_count_is_inside_the_bitmap_kernel_gate(default_cfg):
    """The bitmap decode stages its value stream in one int32, i.e. ceil(k/2) <= 4 bytes.
    A default of k > 8 with a bitmap default would be a configuration that cannot run."""
    assert default_cfg.outliers_per_vector <= 8, (
        f"default k={default_cfg.outliers_per_vector} exceeds the bitmap kernel's k<=8 gate "
        f"while the default encoding is bitmap; one of the two must change")
