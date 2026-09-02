# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pin the two properties that make ``figure/plot_fig7.py`` citable.

Both guard against a specific defect the published Fig. 7 actually had, so neither is a
style check:

  1. **A series that was not measured must not become a bar.** Three of the four series in
     the published figure were literals digitized from an earlier picture; nothing in the
     tree could reproduce or contradict them. The replacement draws only what the data file
     carries, reports the rest by name, and says so in the title -- so an incomplete render
     cannot pass as a complete one.
  2. **The ratio annotation must divide by the series it names.** The published caption said
     the annotations were GPU-OMMX "relative to BF16 vLLM" while the code computed
     ``KIVI / GPU-OMMX``. Here the denominator is a parameter and the label is derived from
     the same value, so the two cannot drift apart.

The ratio is checked on the returned numbers rather than by reading pixels: ``build`` returns
the ratios it drew, in draw order.
"""
import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load():
    path = os.path.join(REPO, "figure", "plot_fig7.py")
    spec = importlib.util.spec_from_file_location("plot_fig7", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)      # import-safe: every statement is inside a function
    return mod


# Two contexts is enough for every property under test and keeps the fixture readable.
def _data(**over):
    d = {"gpu": "TEST", "ctxs": [1024, 4096],
         "methods": {
             "ommx_vllm": {"tpot": [10.0, 20.0],
                           "breakdown": {"base": [5.0, 5.0], "outlier": [2.0, 8.0],
                                         "unpack": [1.0, 2.0], "scale": [1.0, 3.0],
                                         "pack": [1.0, 1.0]}},
             "kivi_hf": {"tpot": [30.0, 80.0]},
             "vllm_bf16": {"tpot": [8.0, 9.0]},
         }}
    d["methods"].update(over)
    return d


def test_unmeasured_series_is_not_drawn_and_is_named():
    m = _load()
    drawn, missing = m.resolve(_data()["methods"])
    keys = [s["key"] for s in drawn]
    # lmdeploy was never measured in this fixture -> no bar, and it is reported by name.
    assert "lmdeploy" not in keys
    assert "lmdeploy" in missing
    # A series present but carrying only nulls is ALSO not a bar: an OOM-only arm claims nothing.
    d = _data(kivi_hf={"tpot": [None, None]})
    drawn2, missing2 = m.resolve(d["methods"])
    assert "kivi_hf" not in [s["key"] for s in drawn2]
    assert "kivi_hf" in missing2


def test_flash_attn_series_falls_back_and_relabels():
    """The stock-vLLM bar may only be available as the Triton-attn arm; it must not keep the
    stock-vLLM label when it does, because they are different attention kernels."""
    m = _load()
    drawn, _ = m.resolve(_data()["methods"])            # fixture has vllm_bf16, not FA
    bf = next(s for s in drawn if s["key"] == "vllm_bf16")
    assert "Triton" in bf["label"] and "FlashAttn" not in bf["label"]
    with_fa = _data(vllm_flash_attn={"tpot": [7.0, 8.0]})
    drawn2, _ = m.resolve(with_fa["methods"])
    fa = next(s for s in drawn2 if s["key"] == "vllm_flash_attn")
    assert "FlashAttn" in fa["label"]


@pytest.mark.parametrize("ref,expected", [
    ("kivi_hf", [3.0, 4.0]),        # 30/10, 80/20
    ("vllm_bf16", [0.8, 0.45]),     # 8/10, 9/20  -- the caption's claimed reference
])
def test_ratio_divides_by_the_named_reference(ref, expected, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    m = _load()
    _, _, _, ratios, _, _ = m.build(_data(), ref, str(tmp_path))
    assert [round(r, 6) for r in ratios] == expected


def test_render_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    m = _load()
    path, drawn, missing, _, ctxs, _ = m.build(_data(), "kivi_hf", str(tmp_path))
    assert os.path.exists(path) and os.path.getsize(path) > 0
    assert ctxs == [1024, 4096] and missing == ["lmdeploy"]
    assert len(drawn) == 3


def test_empty_data_refuses_instead_of_writing_a_blank_figure(tmp_path):
    """No series -> SystemExit, not a PNG. A blank panel saved at rc=0 is the failure mode
    that lets an operator publish a figure that measured nothing."""
    pytest.importorskip("matplotlib")
    m = _load()
    with pytest.raises(SystemExit):
        m.build({"gpu": "TEST", "ctxs": [1024], "methods": {}}, "kivi_hf", str(tmp_path))
    assert not os.path.exists(os.path.join(str(tmp_path), "fig7_tpot_vs_ctx.png"))


def test_collect_map_carries_lmdeploy():
    """collect.py must map lmdeploy.json, or the measured LMDeploy bar is silently dropped
    between the bench and the figure."""
    spec = importlib.util.spec_from_file_location(
        "collect", os.path.join(REPO, "figure", "collect.py"))
    col = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(col)
    assert col.FILE2KEY.get("lmdeploy.json") == "lmdeploy"


def test_lmdeploy_bench_builds_an_exact_length_prompt():
    """The x-axis is a token count, so the prompt has to land on it exactly; a decode/encode
    round trip through a BPE tokenizer is not length-preserving in general."""
    spec = importlib.util.spec_from_file_location(
        "bench_lmdeploy", os.path.join(REPO, "figure", "bench_lmdeploy.py"))
    bl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bl)

    class Tok:  # " the" is one token; decode/encode round-trips
        def __call__(self, s, add_special_tokens=True):
            return type("O", (), {"input_ids": [1234] * s.count(bl.FILLER)})()

        def decode(self, ids):
            return bl.FILLER * len(ids)

    for n in (1024, 8192, 65536):
        _, built = bl.build_prompt(Tok(), n)
        assert built == n


def test_lmdeploy_bench_divides_coalesced_chunks_by_their_token_count():
    """TurboMind may deliver several tokens in one streamed chunk. Counting such a chunk as
    one step would report a multi-token gap as a single token's cost."""
    spec = importlib.util.spec_from_file_location(
        "bench_lmdeploy", os.path.join(REPO, "figure", "bench_lmdeploy.py"))
    bl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bl)

    class R:
        def __init__(self, n):
            self.generate_token_len, self.input_token_len, self.finish_reason = n, 128, None

    class Pipe:
        def __init__(self, sizes):
            self.sizes = sizes

        def stream_infer(self, prompt, **kw):
            cum = 0
            for k in self.sizes:
                cum += k
                yield R(cum)

    cell = bl.run_ctx(Pipe([1, 1, 2, 1, 1, 1]), lambda **kw: None, "p", 0, 8, 42)
    assert cell["chunk_tokens_max"] == 2
    # 6 chunks -> 5 gaps delivering 1+2+1+1+1 = 6 tokens, so 6 per-token entries.
    assert len(cell["times_ms"]) == 6
    assert cell["peak_gb"] is None       # never invented for an engine that cannot report it


def test_fig7_data_file_roundtrip(tmp_path):
    """A collect.py-shaped file on disk renders; this is the path repro/fig7_sweep.sh uses."""
    pytest.importorskip("matplotlib")
    m = _load()
    p = tmp_path / "d.json"
    p.write_text(json.dumps(_data()))
    with open(p) as fh:
        out, _, _, _, _, _ = m.build(json.load(fh), "kivi_hf", str(tmp_path))
    assert os.path.exists(out)
