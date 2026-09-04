# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU gates for ``bench/bench_decode_kernel.py``.

The bench's numbers are only as good as what it does before and around the kernel: how
it parses the cell it is asked to measure, how it isolates one arm's environment from the
next, how it prices the bytes it hands the kernel, and what it refuses to print for an arm
that is not a result. All of that is pure CPU and is pinned here.

BYTE ACCOUNTING. The bench sums the ACTUAL plane tensors from a real
``ommx_pack_kv_canonical_block`` call; ``packed_only.kv_bits_breakdown`` derives the same
figure from the recipe alone. The two must agree over the same plane set, map planes
included (1.0 of 8.25 bit/elem at the canonical recipe).

The kernel itself is Triton; the end-to-end smoke is ``gpu``-marked.
"""
from __future__ import annotations

import argparse
import json
import os

import pytest

torch = pytest.importorskip("torch")

from ommx_gpu_serve.attention.kv_window import WindowSpec  # noqa: E402
from ommx_gpu_serve.attention.oracle import synth_kv  # noqa: E402
from ommx_gpu_serve.attention.pack import ommx_pack_kv_canonical_block  # noqa: E402
from ommx_gpu_serve.bench import bench_decode_kernel as bench  # noqa: E402
from ommx_gpu_serve.integration.vllm.packed_only import kv_bits_breakdown  # noqa: E402


def test_defaults_match_the_spec():
    a = bench.parse_args([])
    assert a.seq == [4096, 16384, 65536]
    assert a.batch == 1 and a.repr == "bitmap" and a.outliers == 6
    assert a.arm == [("base", {})]
    assert a.out == "results.json"
    assert (a.q_heads, a.kv_heads, a.head_dim) == (32, 8, 128)
    assert (a.sink, a.recent, a.group_tokens, a.group_channels, a.page) == (8, 32, 32, 32, 32)


def test_seq_and_arm_are_repeatable():
    a = bench.parse_args([
        "--seq", "4096", "--seq", "16384", "--repr", "relidx7", "--batch", "4",
        "--arm", 'base={}', "--arm", 'splits8={"OMMX_ATTN_NUM_KV_SPLITS": 8}',
        "--arm", 'stages4={"OMMX_V4_NUM_STAGES": 4, "OMMX_ABL_NO_OUTLIER": true}',
        "--out", "x.json"])
    assert a.seq == [4096, 16384]
    assert a.repr == "relidx7" and a.batch == 4 and a.out == "x.json"
    assert a.arm == [("base", {}), ("splits8", {"OMMX_ATTN_NUM_KV_SPLITS": "8"}),
                     ("stages4", {"OMMX_V4_NUM_STAGES": "4", "OMMX_ABL_NO_OUTLIER": "True"})]


@pytest.mark.parametrize("bad", [
    "noequals", "=", 'x=[1]', 'x={"NOT_A_KNOB": 1}', "x={bad",
    'x={"OMMX_KV_RING": 1}',   # an OMMX_ variable the launcher never reads
])
def test_arm_rejects_malformed_specs(bad):
    with pytest.raises(SystemExit):
        bench.parse_args(["--arm", bad])


def test_combinadic_is_gated_to_four_outliers():
    with pytest.raises(SystemExit):
        bench.parse_args(["--repr", "combinadic"])           # default --outliers 6
    assert bench.parse_args(["--repr", "combinadic", "--outliers", "4"]).repr == "combinadic"


def test_arm_env_scrubs_every_launcher_knob_then_restores(monkeypatch):
    monkeypatch.setenv("OMMX_ATTN_NUM_KV_SPLITS", "3")
    monkeypatch.setenv("OMMX_V4_NUM_STAGES", "4")
    monkeypatch.setenv("OMMX_KV_RING", "1")                  # not a launcher knob: kept
    with bench.arm_env({"OMMX_ATTN_BLOCK_H": "16"}):
        assert os.environ.get("OMMX_ATTN_BLOCK_H") == "16"
        assert "OMMX_ATTN_NUM_KV_SPLITS" not in os.environ
        assert "OMMX_V4_NUM_STAGES" not in os.environ
        assert os.environ["OMMX_KV_RING"] == "1"
    assert os.environ.get("OMMX_ATTN_NUM_KV_SPLITS") == "3"
    assert os.environ.get("OMMX_V4_NUM_STAGES") == "4"
    assert "OMMX_ATTN_BLOCK_H" not in os.environ


def test_dram_peak_table_never_guesses():
    assert bench.dram_peak_gbps("NVIDIA H200") == 4800.0
    assert bench.dram_peak_gbps("NVIDIA H100 80GB HBM3") == 3350.0
    assert bench.dram_peak_gbps("NVIDIA A100-SXM4-80GB") == 2039.0
    assert bench.dram_peak_gbps("NVIDIA H100 PCIe") is None
    assert bench.dram_peak_gbps("Tesla V100-SXM2-32GB") is None


def test_median_p99_protocol_and_its_honest_header():
    med, _mean, p99 = bench.median_mean_p99(list(range(1, 21)))
    assert med == 10.5 and p99 == 20
    assert bench.MEASURE == 20 and bench.P99_LABEL == "max(ms)"


_REC = dict(arm="a", seq=4096, batch=1, bytes_read=2.0e6, bits_per_elem=4.125)


def test_a_failed_or_errored_arm_prints_no_latency_and_fails_the_verdict():
    failed = dict(_REC, finite=True, cos=0.5, max_abs=1.0, parity_ok=False)
    row = bench.fmt_row(failed, peak=4800.0)
    assert "FAIL" in row and row.count("invalid") == 4 and "0.500000" in row
    errored = dict(_REC, parity_ok=False, error="ValueError: boom")
    assert "ERROR ValueError: boom" in bench.fmt_row(errored, peak=None)
    nonfinite = dict(_REC, finite=False, cos=None, max_abs=None, parity_ok=False)
    assert " nan " in bench.fmt_row(nonfinite, peak=None)
    passed = dict(_REC, finite=True, cos=0.9999, max_abs=1e-3, parity_ok=True,
                  timing="cuda_graph", med_ms=0.05, p99_ms=0.06, gbps=40.0, pct_peak=0.8)
    row = bench.fmt_row(passed, peak=4800.0)
    assert " ok " in row and "cuda_graph" in row and "0.0500" in row and "invalid" not in row
    assert bench.all_valid([passed])
    assert not bench.all_valid([passed, failed])
    assert not bench.all_valid([passed, dict(passed, error="x")])
    assert "max(ms)" in bench.fmt_header() and "p99" not in bench.fmt_header()


_H_KV, _D, _GT, _GC, _PAGE, _K = 2, 64, 32, 32, 32, 6


@pytest.mark.parametrize("repr_", ["bitmap", "relidx7"])
def test_byte_accounting_agrees_with_kv_bits_breakdown(repr_):
    """Bytes summed off the real planes == the recipe-derived rate, per element, to 0.01 bit.

    Both sides price the same plane set (the kernel's inputs); ``packed`` is a multiple of
    both group_tokens and page_size here so no padding slot can leak into either side.
    """
    seq = 1024
    w = WindowSpec(sink_tokens=8, recent_window=32, group_tokens=_GT, page_size=_PAGE)
    boundary = w.boundary(seq)
    packed = boundary - 8
    assert packed % _GT == 0 and packed % _PAGE == 0
    K, V = synth_kv(seq, _H_KV, _D, bench.SEED)
    planes = ommx_pack_kv_canonical_block(
        K[8:boundary], V[8:boundary], outliers_per_vector=_K, group_tokens=_GT,
        group_channels=_GC, page_size=_PAGE, k_format="i2f4", outlier_repr=repr_,
        use_pow2=True, outlier_select="signed")
    per_plane = bench.plane_bytes(planes)
    assert "k_fp4_mapscale" in per_plane and "k_fp4_mapcenter" in per_plane
    assert ("k_obmp" in per_plane) == (repr_ == "bitmap")
    assert ("k_oidx" in per_plane) == (repr_ == "relidx7")
    got = bench.bits_per_elem(sum(per_plane.values()), packed, _H_KV, _D)

    ref = kv_bits_breakdown(
        _D, k_format="i2f4", group_tokens=_GT, group_channels=_GC,
        outliers_per_vector=_K, outlier_repr=repr_, kv_outlier_map=True,
        kv_int8_scale=True, use_pow2=True)
    assert set(per_plane) == set(ref["k_planes"]) | set(ref["v_planes"])
    assert got == pytest.approx(ref["avg_bits_per_elem"], abs=0.01)


def test_map_planes_are_a_material_share_of_the_bytes():
    """1.0 of 8.25 bit/elem at the canonical recipe: a count that omits them is not noise."""
    ref = kv_bits_breakdown(
        128, k_format="i2f4", group_tokens=32, group_channels=32, outliers_per_vector=6,
        outlier_repr="bitmap", kv_outlier_map=True, kv_int8_scale=True, use_pow2=True)
    maps = ref["k_planes"]["k_fp4_mapscale"] + ref["k_planes"]["k_fp4_mapcenter"]
    assert maps == pytest.approx(1.0) and ref["total_bits_per_elem"] == pytest.approx(8.25)


@pytest.mark.gpu
def test_end_to_end_smoke_on_a_gpu(tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = bench.main([
        "--seq", "512", "--batch", "2", "--q-heads", "4", "--kv-heads", "2",
        "--head-dim", "64", "--arm", "base={}", "--out", str(out)])
    assert rc == 0
    assert f"{bench.DONE_MARKER} ok=true" in capsys.readouterr().out
    data = json.loads(out.read_text())
    assert data["env"]["device"] and data["env"]["torch"]
    assert isinstance(data["env"]["ommx_env"], dict)
    (rec,) = data["results"]
    assert rec["parity_ok"] and rec["batch"] == 2, rec
    assert rec["timing"] in ("cuda_graph", "eager") and rec["replay_cos"] >= bench.COS_MIN
    assert rec["med_ms"] > 0 and rec["gbps"] > 0


def test_env_block_builds_without_a_gpu(monkeypatch):
    """The committed bench once died in env_block() on a name deleted from the module
    (PARITY_COS) -- after every arm had run, before the JSON was written. The CPU suite
    never reached that line because it needs a device name; fake one."""
    import torch
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "FAKE-GPU", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    env = bench.env_block(bench.parse_args([]), "cuda")
    proto = env["protocol"]
    assert proto["parity_cos"] == bench.COS_MIN and proto["parity_rel_l2"] == bench.REL_L2_MAX
    assert "device" in env and "torch" in env
