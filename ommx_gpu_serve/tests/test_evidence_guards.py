# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The guards that keep an unearned number out of a figure, each exercised end to end.

Every one of these was added after an audit found the hole it closes: an ablation flag
inherited from the shell, a stale fp16-era input collected into the H200 figure, a breakdown
built from an ablation JSON measured under the wrong flags. A guard with no test is a
docstring, so each has one here, on CPU, through the code path a real run takes.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

import ommx_gpu_serve

ROOT = pathlib.Path(ommx_gpu_serve.__file__).parent
REPO = ROOT.parent


def _load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hf_json(tpot=20.0, meta=None, superseded=False):
    d = {"method": "ommx", "arm_label": "ommx (bf16)", "model": "m",
         "ctxs": {str(c): {"tpot_ms": tpot, "ttft_ms": 1.0, "peak_gb": 1.0} for c in (1024, 4096)},
         "meta": {"dtype": "bfloat16", **(meta or {})}}
    if superseded:
        d = {"SUPERSEDED": {"by": "x", "why": "y", "do_not": "z"}, **d}
    return d


def _run_collect(src, out):
    return subprocess.run([sys.executable, str(REPO / "figure" / "collect.py"), "--src", str(src),
                           "--gpu", "T", "--ctxs", "1024,4096", "--out", str(out)],
                          capture_output=True, text=True)


# ── collect.py ────────────────────────────────────────────────────────────────

def test_collect_refuses_an_ablated_ommx_hf(tmp_path):
    src = tmp_path / "d"; src.mkdir()
    (src / "ommx_hf.json").write_text(json.dumps(_hf_json(meta={"ommx_abl_active": ["OMMX_ABL_NO_OUTLIER"]})))
    r = _run_collect(src, tmp_path / "out.json")
    assert r.returncode != 0 and "ablation flags" in (r.stdout + r.stderr)
    assert not (tmp_path / "out.json").exists(), "refusal must happen before the write"


def test_collect_refuses_a_superseded_input(tmp_path):
    src = tmp_path / "d"; src.mkdir()
    (src / "bf16_hf.json").write_text(json.dumps(_hf_json(superseded=True)))
    r = _run_collect(src, tmp_path / "out.json")
    assert r.returncode != 0 and "SUPERSEDED" in (r.stdout + r.stderr)
    assert not (tmp_path / "out.json").exists()


def test_collect_accepts_a_clean_ommx_hf_and_records_the_pre_field_case(tmp_path):
    src = tmp_path / "d"; src.mkdir()
    (src / "ommx_hf.json").write_text(json.dumps(_hf_json(meta={"ommx_abl_active": []})))
    (src / "bf16_hf.json").write_text(json.dumps(_hf_json()))          # no field: allowed, said
    r = _run_collect(src, tmp_path / "out.json")
    assert r.returncode == 0, r.stdout + r.stderr
    got = json.load(open(tmp_path / "out.json"))["methods"]
    assert set(got) == {"ommx_hf", "bf16_hf"}


# ── hf_abl_to_breakdown.py ────────────────────────────────────────────────────

def test_breakdown_refuses_an_ablation_json_measured_under_the_wrong_flags(tmp_path):
    hab = _load("figure/hf_abl_to_breakdown.py")
    (tmp_path / "full.json").write_text(json.dumps(_hf_json(meta={"ommx_abl_active": ["OMMX_ABL_NO_OUTLIER"]})))
    with pytest.raises(SystemExit):
        hab._check_flags(str(tmp_path))
    (tmp_path / "full.json").write_text(json.dumps(_hf_json(meta={"ommx_abl_active": []})))
    (tmp_path / "no_outlier.json").write_text(json.dumps(_hf_json(meta={"ommx_abl_active": ["OMMX_ABL_NO_OUTLIER"]})))
    hab._check_flags(str(tmp_path))                                   # exactly its own flag: fine


# ── the E2E orchestrator's verdict (source-level: the module imports vLLM at top) ─

def test_e2e_orchestrator_fails_a_plain_arm_that_carried_an_ablation_flag():
    src = (ROOT / "bench" / "bench_e2e_a100.py").read_text()
    i = src.index('abl_env = d.get("abl_env_active")')
    window = src[i:i + 1400]
    assert 'ev["ok"] = False' in window, "an inherited OMMX_ABL_* no longer fails the arm"
    assert '"abl_env_active"' in src[:i], "the worker no longer records the active flags"
    assert '"OMMX_ABL_", "OMMX_ALLOW_"' in src, "the recorded env prefixes dropped the ablation keys"


# ── the kernel side: mask, names, sentinel ────────────────────────────────────

def test_abl_flag_names_follow_the_kernel_bit_order():
    from ommx_gpu_serve.attention import paged_decode as pd

    assert pd.abl_flag_names(0) == []
    assert pd.abl_flag_names(1) == ["OMMX_ABL_NO_OUTLIER"]
    assert pd.abl_flag_names(2 | 8) == ["OMMX_ABL_V_NODEQUANT", "OMMX_ABL_NO_UNPACK"]
    # the mask the launcher ORs into _LAUNCHES uses the same bit assignment as the kernel
    src = (ROOT / "attention" / "paged_decode.py").read_text()
    i = src.index("abl_flags = ((1 if _env_bool(\"OMMX_ABL_NO_OUTLIER\"")
    assert '_LAUNCHES["abl"] |= int(abl_flags)' in src[i:i + 900]


def test_vllm_backend_announces_active_ablation_next_to_both_route_sentinels():
    src = (ROOT / "integration" / "vllm" / "backend.py").read_text()
    for tag in ("DECODE_ROUTE_FIRED", "GRAPH_ROUTE_FIRED"):
        i = src.index(f'_ommx_route_evidence("{tag}"')
        assert '_ommx_route_evidence("ABL_ATTN_ACTIVE"' in src[i:i + 600], f"{tag} lost its ABL announcement"
    for bad in ("_FIRED", "_DEAD", "_NOFIRE"):
        assert not "ABL_ATTN_ACTIVE".endswith(bad)     # informational: latches nothing, fails no arm


# ── the stale H200 collection declares itself ─────────────────────────────────

def test_the_superseded_h200_collection_declares_itself():
    p = REPO / "figure" / "data" / "h200.json"
    if not p.exists():
        pytest.skip("figure/data/h200.json is not present")
    d = json.load(open(p))
    assert "SUPERSEDED" in d and (REPO / d["SUPERSEDED"]["by"]).exists()


# ── the shared recipe-altering scan and the arm rule ─────────────────────────

def test_recipe_altering_scan_uses_the_kernels_truthiness_and_covers_every_axis():
    from ommx_gpu_serve.attention.paged_decode import active_recipe_altering_env

    env = {"OMMX_ABL_NO_OUTLIER": "false", "OMMX_ABL_NO_UNPACK": "off", "OMMX_W_ABL_NO_CORR": "1",
           "OMMX_ATTN_V_BF16": "1", "OMMX_ATTN_FP8_QK": "0", "OMMX_ATTN_OUTLIERS": "6"}
    assert active_recipe_altering_env(env) == ["OMMX_ATTN_V_BF16", "OMMX_W_ABL_NO_CORR"]
    assert active_recipe_altering_env({}) == []


def test_e2e_ablation_arm_rule_does_not_skip_the_full_arms():
    src = (ROOT / "bench" / "bench_e2e_a100.py").read_text()
    i = src.index("is_abl_arm = ")
    rule = src[i:src.index("\n", i)]
    for full in ("abl_attn", "abl_full", "ommx_vllm", "OMMX"):
        assert not any(t in full for t in ("_no_", "skipwrite", "nocorr")), full
    assert '"_no_"' in rule and '"abl",' not in rule, rule
    assert "ABL_" in src[i:i + 600] and "_ACTIVE" in src[i:i + 600], "sentinel tags are not folded"


def test_graph_route_refuses_a_dead_store_and_health_reports_it():
    src = (ROOT / "integration" / "vllm" / "backend.py").read_text()
    i = src.index("def _route_decode(self, layer, query, output)")
    window = src[i:i + 1600]
    assert "_MANAGER.dead" in window and '"GRAPH_SEAM_DEAD"' in window
    assert window.index("_MANAGER.dead") < window.index("_route_decode_graph(layer, query, output, st, _MANAGER)")
    assert '"single_store_dead"' in src
    meta = (ROOT / "integration" / "vllm" / "metadata.py").read_text()
    j = meta.index("def register(self, layer_id: int, store)")
    assert "self._last_boundary = -1" in meta[j:j + 500]
