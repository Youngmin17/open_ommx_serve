# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Three ways a run could publish a number it did not earn, closed and pinned.

Each of these was found by audit and each has the same shape: a safeguard EXISTS and
something downstream declines to use it. That is worse than no safeguard, because the
docstrings read as though the case is handled.

  1. ``plugin.register()`` re-raises a foreign name collision -- another package owning
     ``"ommx_w"`` -- because the engine would then start and serve SOMEBODY ELSE'S kernel
     under the name the operator selected. ``bench_e2e_a100`` wrapped the call in a bare
     ``except Exception`` and printed a warning, so the one case the plugin refuses to
     swallow was swallowed by its only caller.
  2. ``ommx_route_health()`` answers "did THIS process degrade to bf16 after the route
     fired". It had no caller in the repository. It cannot have one in the orchestrator --
     that runs each arm as a subprocess and only sees the sentinel file -- so the arm worker
     must ask it and carry the answer out.
  3. ``figure/data/a100.json`` is fp16 runs under bf16 labels with no provenance. It is kept
     as the evidence of that mislabelling, which means it is also still citable.
  4. ``PAGE_GRID``: preflight computes the page-grid alignment in the EngineCore worker
     and left it only in that process's ``_PREFLIGHT["report"]``. A driver reading it
     back got ``None`` for every field -- indistinguishable from "the check ran and
     found nothing", and measured exactly that way on H200 (block_size 16 AND 32 both
     reported null while the route demonstrably fired). Same shape as item 2.
  5. ``SW_BYPASS_BF16*`` records that sliding-window layers ran bf16 while the arm still
     passes -- a deliberate carve-out, correctly NOT a failure. But the tag stopped at the
     orchestrator's stdout, so a Mistral or Gemma-2 bar whose SWA layers were bf16 produced a
     figure JSON identical to a fully-OMMX one.

These are CPU-only: they read source and JSON, not GPUs.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import ommx_gpu_serve

ROOT = pathlib.Path(ommx_gpu_serve.__file__).parent
REPO = ROOT.parent


# ── 1. the bench may not swallow a foreign name collision ──────────────────────

def test_a_foreign_name_collision_is_classified_not_benign():
    """The classifier the bench now consults. A collision raises ``OMMXWError`` *from
    ValueError*; the benign paths raise it *from ImportError*."""
    from ommx_gpu_serve.integration.vllm.plugin import is_benign_registration_failure

    def exc(cause):
        e = RuntimeError("registration failed")
        if cause is not None:
            e.__cause__ = cause
        return e

    assert is_benign_registration_failure(exc(ImportError("no Linear bases"))) is True
    assert is_benign_registration_failure(exc(ValueError("ommx_w already registered"))) is False
    # An unclassifiable failure is NOT benign: refusing a run costs a rerun, publishing
    # someone else's kernel as OMMX costs the result.
    assert is_benign_registration_failure(exc(None)) is False


def test_the_bench_refuses_rather_than_warning_on_a_collision():
    """Source-level: bench_e2e_a100 imports vLLM at module scope, so this suite cannot import
    it. What is pinned is that the bare downgrade is gone and the classifier is consulted."""
    src = (ROOT / "bench" / "bench_e2e_a100.py").read_text()
    i = src.index("from ommx_gpu_serve.integration.vllm.plugin import register")
    window = src[i:i + 1600]
    assert "is_benign_registration_failure" in window, (
        "the plugin registration call site no longer consults the classifier; a foreign "
        "name collision would be downgraded to a warning again")
    assert "raise SystemExit" in window, (
        "a non-benign registration failure must stop the bench, not warn")


# ── 2. route health must be asked by the process that can answer ───────────────

def test_the_arm_worker_asks_route_health_and_carries_it_out():
    src = (ROOT / "bench" / "bench_e2e_a100.py").read_text()
    assert "ommx_route_health()" in src, (
        "ommx_route_health() has no caller again; an in-process degrade check that nothing "
        "calls is documentation, not a gate")
    assert '"route_health": health' in src, (
        "the arm worker must carry the health verdict into its JSON -- the orchestrator runs "
        "arms as subprocesses and cannot ask in-process state itself")


def test_the_orchestrator_folds_route_health_into_the_verdict():
    """Recording it is not gating it. The parent must set ok=False, on the same path the
    sentinel failure takes, or a degraded arm still publishes."""
    src = (ROOT / "bench" / "bench_e2e_a100.py").read_text()
    i = src.index('hz = d.get("route_health")')
    window = src[i:i + 900]
    assert 'ev["ok"] = False' in window, (
        "route_health is read but never folded into the arm verdict; a process that degraded "
        "after firing would still publish its bf16 timings as OMMX")


# ── 3. the superseded A100 file must say so ───────────────────────────────────

def test_the_superseded_a100_file_declares_itself():
    p = REPO / "figure" / "data" / "a100.json"
    if not p.exists():
        pytest.skip("figure/data/a100.json is not present")
    d = json.load(open(p))
    assert "SUPERSEDED" in d, (
        "figure/data/a100.json is fp16 runs under bf16 labels and is kept as the evidence of "
        "that; it must carry a SUPERSEDED block so it cannot be cited by accident")
    s = d["SUPERSEDED"]
    for k in ("by", "why", "do_not"):
        assert s.get(k), f"SUPERSEDED block is missing {k!r}"
    assert pathlib.Path(REPO / s["by"]).exists(), (
        f"SUPERSEDED.by points at {s['by']}, which does not exist")
    for name, m in (d.get("methods") or {}).items():
        prov = (m or {}).get("provenance") or {}
        assert prov.get("superseded") is True, (
            f"method {name} does not carry the superseded marker, so a reader that only "
            f"inspects one arm would not see it")


def test_the_replacement_records_the_dtype_the_stale_one_omitted():
    """The point of the replacement: every arm's dtype is recorded, and the two fp16 arms are
    labelled fp16 rather than left blank."""
    p = REPO / "figure" / "data" / "a100_20260903_bitmap.json"
    if not p.exists():
        pytest.skip("the replacement A100 collection is not present")
    d = json.load(open(p))
    got = {k: ((v.get("provenance") or {}).get("dtype"))
           for k, v in (d.get("methods") or {}).items()}
    for arm in ("kivi_hf", "kitty_hf"):
        if arm in got:
            assert got[arm] == "fp16", (
                f"{arm} must be labelled fp16 -- its kernel casts to tl.float16 before every "
                f"tl.dot, and a bf16 label on it is the defect this replaces")
    for arm in ("ommx_hf", "bf16_hf", "ommx_vllm", "vllm_bf16"):
        if arm in got:
            assert got[arm] == "bf16", f"{arm} should record bf16; got {got[arm]!r}"


# ── 4. partial bf16 coverage must travel with the bar ─────────────────────────

def test_partial_coverage_is_carried_into_the_figure_json():
    """The SWA carve-out is by design and must NOT fail the arm. What it must do is be
    visible in the artifact a reader cites, not only in a log line nobody keeps."""
    src = (ROOT / "bench" / "e2e_to_figure.py").read_text()
    assert '"partial_bf16_coverage"' in src, (
        "the ommx_vllm figure JSON no longer carries partial_bf16_coverage; a bar whose "
        "sliding-window layers ran bf16 would look identical to a fully-OMMX one")
    i = src.index('"partial_bf16_coverage"')
    window = src[max(0, i - 1200):i]
    assert "SW_BYPASS_BF16" in window, (
        "partial_bf16_coverage is present but no longer sourced from the SW_BYPASS tags")


def test_a_full_attention_model_reports_no_partial_coverage():
    """The negative control: on Llama-3.1-8B every layer is full attention, so the field must
    be EMPTY rather than absent. An absent field and an empty one read the same to a human and
    differently to a script."""
    p = REPO / "figure" / "data_bm" / "ommx_vllm.json"
    if not p.exists():
        pytest.skip("no committed ommx_vllm arm to check")
    d = json.load(open(p))
    if "partial_bf16_coverage" not in d:
        pytest.skip("this arm predates the field; regenerate with e2e_to_figure to populate it")
    assert d["partial_bf16_coverage"] == [], (
        f"Llama-3.1-8B is full-attention throughout, so no SWA bypass is possible; got "
        f"{d['partial_bf16_coverage']}")


# ── 5. the page grid must cross the process boundary ──────────────────────────

def test_the_page_grid_is_written_to_the_cross_process_sentinel():
    """vLLM v1 runs EngineCore as a separate process. Any preflight result read through
    module state in the driver is None there, whatever the worker computed."""
    src = (ROOT / "integration" / "vllm" / "backend.py").read_text()
    i = src.index('_PREFLIGHT["done"] = True')
    window = src[i:i + 1400]
    assert '"PAGE_GRID"' in window, (
        "the page grid no longer reaches the sentinel; a driver asking _PREFLIGHT for it "
        "gets None from its own untouched module and cannot tell that from a real null")
    for field in ("vllm_block_size", "ommx_group_tokens", "pages_per_group"):
        assert field in window, f"the PAGE_GRID sentinel line omits {field!r}"


def test_the_page_grid_tag_is_informational_by_its_suffix():
    """The tag contract in ``_ommx_route_evidence``: ``_FIRED`` latches "past init",
    ``_DEAD``/``_NOFIRE`` fail the arm. A preflight report is neither -- it is emitted
    BEFORE any step runs, so a ``_FIRED`` suffix would latch the serve flag during init
    and a ``_DEAD`` one would fail every arm that reports its grid."""
    for bad in ("_FIRED", "_DEAD", "_NOFIRE"):
        assert not "PAGE_GRID".endswith(bad)


# ── 6. the linear ablation must be loud, never a silent skip ──────────────────

def test_linear_no_corr_ablation_announces_itself():
    """``OMMX_W_ABL_NO_CORR`` skips the outlier correction so the stage can be attributed.
    A skip that leaves no trace is indistinguishable from a broken route; the flag must
    fire an informational sentinel tag (no ``_FIRED``/``_DEAD`` suffix) and the bench must
    only ever set it on the dedicated arm."""
    src = (ROOT / "integration" / "vllm" / "linear_method.py").read_text()
    i = src.index('_env_on("OMMX_W_ABL_NO_CORR")')
    window = src[i:i + 900]
    assert '_fire("ABL_W_NO_CORR_ACTIVE"' in window, "the skip is silent"
    for bad in ("_FIRED", "_DEAD", "_NOFIRE"):
        assert not "ABL_W_NO_CORR_ACTIVE".endswith(bad)
    bench = (ROOT / "bench" / "bench_e2e_a100.py").read_text()
    assert bench.count('"OMMX_W_ABL_NO_CORR": "1"') == 1, (
        "the ablation flag must appear on exactly one arm (abl_linear_nocorr)")
