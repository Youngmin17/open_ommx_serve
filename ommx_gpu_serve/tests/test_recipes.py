# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Gates for the named-recipe registry (``ommx_gpu_serve/recipes.py``).

WHAT THESE GATES ARE FOR. A registry of "the recipe that reproduces the paper's
number" is worth exactly as much as its coupling to the code. Four failure modes, one
class each below:

  1. REGISTRY DRIFT — the recorded bits stop matching what the accounting/packer
     actually returns (``test_*_measured_bits_match_the_code``). This is the gate that
     catches someone changing a plane in ``kv_pool.py`` / ``w_format.py`` and leaving a
     stale 2.7500 in the registry.
  2. DEFAULT DRIFT — ``shipped-*`` stops being what the code ships, which would
     silently invalidate every published number by making the "published recipe" a
     historical curiosity (``test_shipped_*_reproduces_current_defaults``).
  3. SILENT SELECTION — an unknown/typo'd preset resolving to something instead of
     raising (``test_unknown_preset_*``).
  4. LOST OVERRIDE — a preset overwriting a knob the operator set explicitly
     (``test_explicit_env_overrides_preset``, ``test_explicit_flag_overrides_preset``).

Plus additivity: with no preset named, both paths must resolve byte-identically to what
they resolved before this feature existed (``test_no_preset_*``).

EVERY test here was mutation-proven: the code under test was broken, the test was
observed to FAIL, and the break was reverted. See the ``MUTATION:`` note on each.

No GPU, no vLLM, no triton. ``kv_bits_breakdown`` and ``Recipe.bits_breakdown`` are the
measuring instruments and both are CPU-only.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shlex
import subprocess
import sys

import pytest

from ommx_gpu_serve import recipes as R
from ommx_gpu_serve.integration.vllm.config import resolve_serving_config
from ommx_gpu_serve.integration.vllm.packed_only import kv_bits_breakdown
from ommx_gpu_serve.linear import w_packer

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ════════════════════════════════════════════════════════════════════════════════
# 1. registry drift — every recorded number is recomputed from the live code
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", R.names("kv"))
def test_kv_measured_bits_match_the_code(name: str) -> None:
    """The registry's KV bits ARE what ``kv_bits_breakdown`` returns right now.

    MUTATION: set ``paper-kv``'s recorded ``avg_bits_per_elem`` to 2.76 -> FAILS with
    "avg_bits_per_elem: registry 2.76 vs code 2.75". Reverted -> passes.
    """
    rec = R.get(name)
    got = R.measure_kv(rec)
    assert set(got) == set(rec.measured), (
        f"{name}: measured key set drifted; registry {sorted(rec.measured)} vs code "
        f"{sorted(got)}")
    for key, want in rec.measured.items():
        # Exact for the bit figures (all dyadic rationals: halves down to 1/16 of a
        # bit, so binary-exact in float). ``compression_ratio`` is 32/total, which is
        # not dyadic, hence the relative tolerance on that one only.
        if "ratio" in key:
            assert got[key] == pytest.approx(want, rel=1e-12), (
                f"{name}: {key}: registry {want} vs code {got[key]}")
        else:
            assert got[key] == want, (
                f"{name}: {key}: registry {want} vs code {got[key]}")


@pytest.mark.parametrize("name", R.names("weight"))
def test_weight_measured_bits_match_the_packer_budget(name: str) -> None:
    """The registry's weight bits ARE what the packer's own budget path returns.

    MUTATION: set ``paper-weight``'s recorded ``bits_per_weight`` to 3.63 (the paper's
    ROUNDED figure) -> FAILS with "registry 3.63 vs code 3.625". That is the point of
    the gate: 3.6250 is the measurement, 3.63 is the paper's rounding of it, and the
    registry must carry the former. Reverted -> passes.
    """
    rec = R.get(name)
    got = R.measure_weight(rec)
    assert set(got) == set(rec.measured)
    for key, want in rec.measured.items():
        assert got[key] == want, f"{name}: {key}: registry {want} vs code {got[key]}"


def test_paper_kv_hits_the_papers_2_75_exactly() -> None:
    """Claim E1 is 2.75 AvgBits and ``paper-kv`` must MEASURE exactly that.

    Not "close to". The audit's own shortlist stopped at 2.688 / 2.781 and this recipe
    exists because an exact hit is reachable; if a future plane change moves it off
    2.7500 the preset is no longer the paper recipe and must be renamed or re-chosen,
    not quietly re-labelled.

    MUTATION: change ``OMMX_ATTN_OUTLIERS`` in ``_PAPER_KV_ENV`` from 10 to 11 -> the
    relidx7 index plane grows from 9 to 10 bytes/group, avg becomes 2.78125 -> FAILS.
    Reverted -> passes.
    """
    b = R.measure_kv(R.get("paper-kv"))
    assert b["avg_bits_per_elem"] == 2.75
    assert b["total_bits_per_elem"] == 5.5


def test_paper_weight_hits_the_papers_3_625_exactly() -> None:
    """Claim E2 is 3.63 AvgBits; the layout's exact value is 3.6250.

    Also pins the DECOMPOSITION, because "3.625" can be reached by more than one plane
    set and the paper's own claims B4 (flat bitmask) + B6 (no range-map parameters)
    say which one: 2 base + 8/64 E8M0 + 16/64 BF16 ZP + 64/64 bitmask + 4*4/64 FP4.

    MUTATION: switch ``_PAPER_WEIGHT_ARGS["outlier_repr"]`` to "relidx7" -> the index
    plane drops to 0.5 and the total to 3.125 -> FAILS on both the total and the
    ``outlier_index`` term. Reverted -> passes.
    """
    rec = R.get("paper-weight")
    r = w_packer.build_recipe(**dict(rec.packer_args))
    bb = r.bits_breakdown()
    assert bb["bits_per_weight"] == 3.625
    stored = bb["stored"]
    assert stored["base_int2"] == 2.0
    assert stored["scale_e8m0"] == 8.0 / 64      # INT8 E8M0 exponent, claim B9
    assert stored["zero_point"] == 16.0 / 64     # BF16 zero-point, claim B9
    assert stored["outlier_index"] == 1.0        # FLAT bitmask, claim B4
    assert stored["outlier_fp4"] == 4 * 4.0 / 64
    assert stored["fp4_range_map"] == 0.0        # no range-map params, claim B6


def test_paper_kv_is_the_best_exact_hit_on_the_legal_grid() -> None:
    """Re-derive the CHOICE, not just the number: over the whole legal grid, no recipe
    that also hits 2.7500 exactly under the same paper-mandated constraints carries
    MORE outlier coverage than the one registered.

    The constraints are the ones documented in ``recipes.py``: relidx7 positions (the
    only encoding the vLLM decode path reads), pow2/INT8 exponent (claim B9), and at
    least one outlier (claims B1/B2/B7). Coverage is the tie-break because the paper's
    thesis is outlier coverage — among equal-bit recipes, more outliers is strictly the
    one to publish.

    MUTATION: change ``_PAPER_KV_ENV`` to the gt=128/gc=32/k=3 exact hit (coverage
    2.34%) -> FAILS: "registered coverage 0.0234 but (128,128,10) reaches 0.0781 at the
    same 2.7500". Reverted -> passes.
    """
    best = None
    for gt in (16, 32, 64, 128):
        for gc in (16, 32, 64, 128):
            for k in range(1, gt + 1):
                for omap in (True, False):
                    b = kv_bits_breakdown(
                        128, k_format="i2f4", group_tokens=gt, group_channels=gc,
                        outliers_per_vector=k, outlier_repr="relidx7",
                        kv_outlier_map=omap, use_pow2=True)
                    if b["avg_bits_per_elem"] != 2.75:
                        continue
                    cov = k / gt
                    if best is None or cov > best[0]:
                        best = (cov, gt, gc, k, omap)
    assert best is not None, "the legal grid no longer contains an exact 2.7500 recipe"
    env = R.get("paper-kv").env
    gt_r = int(env["OMMX_KV_GROUP_TOKENS"])
    cov_r = int(env["OMMX_ATTN_OUTLIERS"]) / gt_r
    assert cov_r == best[0], (
        f"registered coverage {cov_r:.4f} (gt={gt_r}, k={env['OMMX_ATTN_OUTLIERS']}) but "
        f"(gt={best[1]}, gc={best[2]}, k={best[3]}, map={best[4]}) reaches {best[0]:.4f} "
        "at the same measured 2.7500 avg bits")
    assert (int(env["OMMX_KV_GROUP_CHANNELS"]), env["OMMX_KV_OUTLIER_MAP"] == "1") \
        == (best[2], best[4])


# ════════════════════════════════════════════════════════════════════════════════
# 2. default drift — shipped-* must stay equal to what the code ships
# ════════════════════════════════════════════════════════════════════════════════

def _resolved_fields(cfg) -> dict:
    """The recipe-bearing fields of an OMMXServingConfig (model geometry excluded — it
    comes from the vLLM model config, not from the recipe)."""
    return {f: getattr(cfg, f) for f in (
        "k_format", "outliers_per_vector", "outlier_select", "outlier_repr",
        "use_pow2", "combinadic_read", "bitmap_read", "sink_tokens", "recent_window",
        "group_tokens", "group_channels", "kv_outlier_map", "strict")}


#: The published canonical KV recipe, RESTATED here independently of the registry.
#: Deliberately a literal and not ``R.get("shipped-kv").env``: comparing the registry
#: against itself is a tautology that passes no matter how far the registry drifts, and
#: the point of this gate is that ``shipped-kv`` still IS the recipe every published
#: number came from. Keep it in step with the two dicts in figure/bench.py and
#: eval/lm_eval/models/ommx_hf_model.py (which the ast gate below reads directly).
_PUBLISHED_KV_ENV = {
    "OMMX_ATTN_K_FORMAT": "i2f4", "OMMX_ATTN_OUTLIERS": "6", "OMMX_ATTN_POW2": "1",
    "OMMX_ATTN_OUTLIER_SELECT": "signed", "OMMX_ATTN_OUTLIER_REPR": "relidx7",
    "OMMX_KV_GROUP_TOKENS": "32", "OMMX_KV_GROUP_CHANNELS": "32",
    "OMMX_KV_SINK": "8", "OMMX_KV_RECENT": "32",
    "OMMX_KV_RING": "1", "OMMX_KV_GPU_PACK": "1",
    "OMMX_KV_OUTLIER_MAP": "1",
}


def test_shipped_kv_reproduces_the_published_recipe_path(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolving via the ``shipped-kv`` PRESET == resolving with the published env dict
    set by hand. Two paths, one answer — and the "by hand" side is the independent
    literal above, so the comparison has something to catch.

    MUTATION: change ``_SHIPPED_KV_ENV["OMMX_ATTN_OUTLIERS"]`` from "6" to "5" -> FAILS
    ("shipped-kv env drifted from the published recipe"). Reverted -> passes.
    """
    assert dict(R.get("shipped-kv").env) == _PUBLISHED_KV_ENV, \
        "shipped-kv env drifted from the published recipe"

    for k, v in _PUBLISHED_KV_ENV.items():
        monkeypatch.setenv(k, v)
    by_hand = _resolved_fields(resolve_serving_config())

    for k in _PUBLISHED_KV_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(R.RECIPE_ENV_VAR, "shipped-kv")
    by_preset = _resolved_fields(resolve_serving_config())

    assert by_preset == by_hand
    # …and that answer is the published recipe, not merely self-consistent.
    assert by_preset["outliers_per_vector"] == 6
    assert by_preset["use_pow2"] is True
    assert (by_preset["group_tokens"], by_preset["group_channels"]) == (32, 32)


def test_shipped_kv_is_the_recipe_the_benches_and_evals_actually_set() -> None:
    """``shipped-kv`` == the canonical dict duplicated in ``figure/bench.py`` and
    ``eval/lm_eval/models/ommx_hf_model.py``.

    Those two dicts ARE the published recipe (every OMMX TPOT / CoQA number came from
    them), so if one of them moves and the registry does not, the preset silently stops
    being "the published recipe". Parsed with ``ast`` rather than imported: importing
    figure/bench.py drags in transformers, and the value is a literal anyway.

    MUTATION: change ``OMMX_KV_GROUP_TOKENS`` to "64" in figure/bench.py's
    ``OMMX_RECIPE_ENV`` -> FAILS ("figure/bench.py sets OMMX_KV_GROUP_TOKENS=64, the
    shipped-kv preset says 32"). Reverted -> passes.
    """
    shipped = dict(R.get("shipped-kv").env)
    for rel in ("figure/bench.py", "eval/lm_eval/models/ommx_hf_model.py"):
        path = os.path.join(_REPO, rel)
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        found = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict) or not node.keys:
                continue
            try:
                d = {ast.literal_eval(k): ast.literal_eval(v)
                     for k, v in zip(node.keys, node.values)}
            except (ValueError, SyntaxError):
                continue
            if "OMMX_ATTN_K_FORMAT" in d:
                found = d
                break
        assert found, f"{rel} no longer contains an OMMX recipe dict to compare against"
        for key, val in found.items():
            assert key in shipped, (
                f"{rel} sets {key}={val}, which the shipped-kv preset does not carry — "
                "the published recipe grew a knob the registry does not name")
            assert shipped[key] == val, (
                f"{rel} sets {key}={val}, the shipped-kv preset says {shipped[key]}")


def test_shipped_kv_outlier_map_is_behaviourally_the_current_default(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """``shipped-kv`` names ``OMMX_KV_OUTLIER_MAP=1`` while the two published dicts
    leave it implicit. That is only legitimate while the DEFAULT is in fact on — pin it,
    because the two ``k_fp4_map*`` planes are a full bit of K at gt=32.

    MUTATION: flip ``config.py``'s ``_env_bool("OMMX_KV_OUTLIER_MAP", True)`` to False
    -> FAILS ("kv_outlier_map default is now False"). Reverted -> passes.
    """
    monkeypatch.delenv("OMMX_KV_OUTLIER_MAP", raising=False)
    assert resolve_serving_config().kv_outlier_map is True, \
        "kv_outlier_map default is now False; shipped-kv's explicit '1' is no longer a " \
        "restatement of the default but a behaviour change"


def test_shipped_weight_reproduces_the_packer_defaults() -> None:
    """The ``shipped-weight`` preset resolves to the SAME ``Recipe`` as ``pack`` with
    no flags at all — so a future default change cannot leave the "shipped" preset
    describing a recipe nothing ships.

    MUTATION: change ``_SHIPPED_WEIGHT_ARGS["group_size"]`` from 64 to 32 -> FAILS
    ("group_size 32 != 64"). MUTATION 2: change ``pack``'s ``--group-size`` argparse
    default from 64 to 32 -> FAILS the same way (from the other side). Both reverted.
    """
    ap_default = w_packer.build_recipe(
        group_size=64, outlier_pct=None, npv=None,
        outlier_repr="relidx7", outlier_map="idx_range", zp_dtype="bf16")
    preset = w_packer.build_recipe(**dict(R.get("shipped-weight").packer_args))
    for field in ("group_size", "npv", "outlier_pct", "outlier_repr", "outlier_map",
                  "zp_dtype"):
        assert getattr(preset, field) == getattr(ap_default, field), (
            f"shipped-weight.{field}={getattr(preset, field)} but the packer default "
            f"resolves to {getattr(ap_default, field)}")


def test_shipped_weight_matches_the_live_argparse_defaults() -> None:
    """The other half of the previous gate: the literals above ARE ``pack``'s current
    argparse defaults, read off the parser rather than restated.

    MUTATION: change ``pk.add_argument("--group-size", type=int, default=64)`` to
    ``default=128`` -> FAILS ("pack's --group-size default is 128, shipped-weight says
    64"). Reverted -> passes.
    """
    # The knobs an un-presetted `pack` resolves to ARE the live argparse defaults —
    # resolve_preset_knobs fills every un-typed dest from pk.get_default(). Driving the
    # real CLI is therefore a stronger read than poking at the parser object, because it
    # also proves the preset plumbing does not disturb the no-preset path.
    live = _parse_pack(["pack", "--input", "x", "--output", "y"])
    want = dict(R.get("shipped-weight").packer_args)
    assert live["group_size"] == want["group_size"], (
        f"pack's --group-size default is {live['group_size']}, shipped-weight says "
        f"{want['group_size']}")
    assert live["outlier_repr"] == want["outlier_repr"]
    assert live["outlier_map"] == want["outlier_map"]
    assert live["zp_dtype"] == want["zp_dtype"]
    assert live["npv"] == want["npv"] is None
    # outlier_pct is the one knob the preset spells out where argparse defaults to None
    # (build_recipe substitutes 0.0625). Compare through build_recipe, where the
    # substitution lives.
    assert w_packer.build_recipe(
        live["group_size"], live["outlier_pct"], live["npv"], live["outlier_repr"],
        live["outlier_map"], live["zp_dtype"]).outlier_pct == want["outlier_pct"]


def _parse_pack(argv) -> dict:
    """Resolve a ``pack`` argv through the REAL CLI and return the knob namespace.

    Uses ``w_packer.main`` indirection-free by rebuilding the same parser main() builds
    — but main() owns it, so we drive main() and intercept at ``build_recipe``.
    """
    seen: dict = {}
    real = w_packer.build_recipe

    def spy(group_size, outlier_pct, npv, outlier_repr, outlier_map, zp_dtype):
        seen.update(group_size=group_size, outlier_pct=outlier_pct, npv=npv,
                    outlier_repr=outlier_repr, outlier_map=outlier_map,
                    zp_dtype=zp_dtype)
        return real(group_size, outlier_pct, npv, outlier_repr, outlier_map, zp_dtype)

    w_packer.build_recipe = spy
    try:
        # --dry-run against a nonexistent input: the recipe is built BEFORE the
        # checkpoint is opened, so the spy fires and the run then fails harmlessly.
        w_packer.main(argv + ["--dry-run"])
    except Exception:
        pass
    finally:
        w_packer.build_recipe = real
    assert seen, f"build_recipe was never reached for argv={argv}"
    return seen


# ════════════════════════════════════════════════════════════════════════════════
# 3. silent selection — an unknown name must raise AND list the alternatives
# ════════════════════════════════════════════════════════════════════════════════

def test_unknown_preset_raises_and_names_the_alternatives() -> None:
    """MUTATION: make ``recipes.get`` ``return RECIPES.get(key, RECIPES['shipped-kv'])``
    -> FAILS (no exception raised). Reverted -> passes."""
    with pytest.raises(R.UnknownRecipeError) as exc:
        R.get("paper_kv")
    msg = str(exc.value)
    assert "paper_kv" in msg
    for known in R.RECIPE_NAMES:
        assert known in msg, f"{known!r} missing from the error message: {msg}"


def test_unknown_preset_raises_from_the_serving_path(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in ``OMMX_RECIPE`` must stop the engine, not fall back to the defaults.

    MUTATION: wrap ``apply_recipe_env()`` in ``config.resolve_serving_config`` in
    ``try/except Exception: pass`` -> FAILS (no exception). Reverted -> passes.
    """
    monkeypatch.setenv(R.RECIPE_ENV_VAR, "nope-kv")
    with pytest.raises(R.UnknownRecipeError) as exc:
        resolve_serving_config()
    assert "shipped-kv" in str(exc.value)


def test_weight_axis_refuses_a_kv_preset() -> None:
    """A KV name on the weight axis would set nothing and pack the defaults under a
    name the operator believes they selected.

    MUTATION: drop the ``axis`` check from ``recipes.get`` -> FAILS (no exception).
    Reverted -> passes.
    """
    with pytest.raises(R.UnknownRecipeError) as exc:
        R.get("paper-kv", axis="weight")
    assert "paper-weight" in str(exc.value)


def test_packer_cli_exits_2_on_an_unknown_preset(capsys) -> None:
    """MUTATION: remove the ``except UnknownRecipeError`` clause from ``w_packer.main``
    -> the KeyError escapes and the test FAILS with an error rather than rc=2.
    Reverted -> passes."""
    rc = w_packer.main(["pack", "--preset", "nope", "--input", "x", "--output", "y",
                        "--dry-run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "paper-weight" in err and "nope" in err


# ════════════════════════════════════════════════════════════════════════════════
# 4. lost override — an explicit setting must beat the preset
# ════════════════════════════════════════════════════════════════════════════════

def test_explicit_env_overrides_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """One knob overridden, the rest of the preset intact.

    MUTATION: make ``apply_recipe_env`` assign unconditionally
    (``env[key] = val`` for every key, dropping the ``pending_env`` filter) -> FAILS
    ("outliers_per_vector 10 != 7"). Reverted -> passes.
    """
    monkeypatch.setenv(R.RECIPE_ENV_VAR, "paper-kv")
    monkeypatch.setenv("OMMX_ATTN_OUTLIERS", "7")
    cfg = resolve_serving_config()
    assert cfg.outliers_per_vector == 7          # the operator's value survived
    assert cfg.group_tokens == 128               # …and the rest of the preset applied
    assert cfg.group_channels == 128
    assert cfg.use_pow2 is True


def test_a_partly_applied_preset_records_what_it_did_not_supply(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """``cfg.extra["ommx_recipe"]`` is EVIDENCE, so it must not claim a recipe that only
    partly applied.

    A preset fills only knobs the environment does not already carry — correct, but it
    means anything that reaches ``os.environ`` FIRST takes the knob, and that is not
    always the operator.

    ``figure/bench.py`` used to be the example: it applied the shipped canonical dict
    with ``os.environ.setdefault()`` at model-build time, before anything expanded
    ``OMMX_RECIPE``, so ``OMMX_RECIPE=paper-kv python3 figure/bench.py --method ommx``
    RAN shipped-kv (gt=32, k=6, 4.375 avg bit) while recording only ``ommx_recipe:
    paper-kv`` (2.75). That file now resolves the preset first, so it is no longer the
    example — ``test_bench_named_preset_beats_the_benchs_own_defaults`` pins the fix.

    ``eval/lm_eval/models/ommx_hf_model.py`` — the ACCURACY arm — still is: the same
    eleven-key ``setdefault`` loop runs at module import with no preset resolution at
    all. Measured by replaying that loop verbatim: under ``OMMX_RECIPE=paper-kv`` the
    accounting reports gt=32/gc=32/k=6, avg 4.375. So this marker still has a live case
    to be evidence for, and case (b) below reproduces exactly that ordering.

    MUTATION: delete the ``conflicting``/``ommx_recipe_overridden`` block from
    ``config.resolve_serving_config`` -> FAILS (extra has one key and the mislabelled
    run is indistinguishable from a faithful one). Reverted -> passes.
    """
    marker = R.RECIPE_ENV_VAR.lower()
    # (a) faithful run: the preset supplied everything -> extra is exactly as before.
    monkeypatch.setenv(R.RECIPE_ENV_VAR, "paper-kv")
    cfg = resolve_serving_config()
    assert cfg.extra == {marker: "paper-kv"}
    assert (cfg.group_tokens, cfg.outliers_per_vector) == (128, 10)

    # (b) exactly figure/bench.py's ordering: its canonical dict lands first.
    for k, v in _PUBLISHED_KV_ENV.items():
        monkeypatch.setenv(k, v)          # os.environ.setdefault(), same effect here
    cfg = resolve_serving_config()
    assert (cfg.group_tokens, cfg.outliers_per_vector) == (32, 6), \
        "sanity: this really is the shipped recipe running under the paper-kv name"
    over = cfg.extra.get(marker + "_overridden")
    assert over, (
        "resolve_serving_config recorded ommx_recipe=paper-kv for a run that resolved to "
        f"gt={cfg.group_tokens}/k={cfg.outliers_per_vector} and said nothing about it; "
        f"extra={cfg.extra}")
    assert over["OMMX_KV_GROUP_TOKENS"] == "32" and over["OMMX_ATTN_OUTLIERS"] == "6"


def test_explicit_alias_overrides_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OMMX_ATTN_* back-compat spelling must win too.

    Without alias suppression the preset materialises the CANONICAL name, which every
    reader checks FIRST — so the operator's explicit alias would be silently ignored
    while looking like it had been honoured. That is precisely the silent-recipe bug
    class this registry exists to remove.

    MUTATION: empty ``recipes._ENV_ALIASES`` to ``{}`` -> FAILS ("group_tokens 128 !=
    64"). Reverted -> passes.
    """
    monkeypatch.setenv(R.RECIPE_ENV_VAR, "paper-kv")
    monkeypatch.setenv("OMMX_ATTN_GROUP_TOKENS", "64")
    cfg = resolve_serving_config()
    assert cfg.group_tokens == 64
    assert cfg.group_channels == 128             # untouched knob still from the preset


def test_explicit_flag_overrides_preset() -> None:
    """``--preset paper-weight --outlier-map idx_range`` keeps the bitmap positions and
    takes the operator's map back.

    MUTATION: in ``resolve_preset_knobs``, drop the ``if dest in explicit: continue``
    guard -> FAILS ("outlier_map none != idx_range"). Reverted -> passes.
    """
    got = _parse_pack(["pack", "--preset", "paper-weight", "--input", "x",
                       "--output", "y", "--outlier-map", "idx_range"])
    assert got["outlier_repr"] == "bitmap"       # from the preset
    assert got["outlier_map"] == "idx_range"     # from the operator
    assert got["group_size"] == 64


def test_explicit_npv_flag_displaces_the_presets_outlier_pct() -> None:
    """``--npv`` and ``--outlier-pct`` are mutually exclusive in ``build_recipe``, so an
    explicit ``--npv`` must displace a preset's ``outlier_pct`` instead of colliding
    with it.

    MUTATION: remove the ``overlay.pop("outlier_pct")`` branch -> ``build_recipe``
    raises PackError "both set", the spy records nothing and the helper's assertion
    FAILS. Reverted -> passes.
    """
    got = _parse_pack(["pack", "--preset", "shipped-weight", "--input", "x",
                       "--output", "y", "--npv", "8"])
    assert got["npv"] == 8 and got["outlier_pct"] is None


# ════════════════════════════════════════════════════════════════════════════════
# 5. additivity — with no preset named, nothing changes
# ════════════════════════════════════════════════════════════════════════════════

def test_no_preset_leaves_the_serving_defaults_untouched(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The paper presets are NOT the defaults, and naming none of them must resolve to
    the historical default recipe (k=3, pow2 off, gt=gc=32, relidx7, map on).

    MUTATION: default ``OMMX_RECIPE`` to "paper-kv" inside ``apply_recipe_env``
    (``want = name or env.get(RECIPE_ENV_VAR, "paper-kv")``) -> FAILS
    ("group_tokens 128 != 32"). Reverted -> passes.
    """
    monkeypatch.delenv(R.RECIPE_ENV_VAR, raising=False)
    cfg = resolve_serving_config()
    assert _resolved_fields(cfg) == {
        "k_format": "i2f4", "outliers_per_vector": 3, "outlier_select": "signed",
        "outlier_repr": "relidx7", "use_pow2": False, "combinadic_read": False,
        "bitmap_read": None, "sink_tokens": 8, "recent_window": 32,
        "group_tokens": 32, "group_channels": 32, "kv_outlier_map": True,
        "strict": False}
    assert cfg.extra == {}, "an un-presetted config must carry no recipe marker"


def test_no_preset_touches_no_environment_variable(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """``apply_recipe_env`` is a genuine no-op without ``OMMX_RECIPE``.

    MUTATION: make ``apply_recipe_env`` fall back to "shipped-kv" when OMMX_RECIPE is
    unset -> FAILS (12 OMMX_* vars appear). Reverted -> passes.
    """
    monkeypatch.delenv(R.RECIPE_ENV_VAR, raising=False)
    before = dict(os.environ)
    assert R.apply_recipe_env() is None
    assert dict(os.environ) == before


def test_no_preset_leaves_the_packer_arguments_untouched() -> None:
    """``pack`` with no ``--preset`` builds exactly the pre-existing Recipe.

    MUTATION: make ``resolve_preset_knobs`` default ``preset`` to "paper-weight" ->
    FAILS ("outlier_repr bitmap != relidx7"). Reverted -> passes.
    """
    got = _parse_pack(["pack", "--input", "x", "--output", "y"])
    assert got == {"group_size": 64, "outlier_pct": None, "npv": None,
                   "outlier_repr": "relidx7", "outlier_map": "idx_range",
                   "zp_dtype": "bf16"}


def test_paper_presets_are_not_the_defaults() -> None:
    """Stated as its own gate because it is the additivity contract in one line: the
    two paper recipes must DIFFER from the two shipped ones, or 'additive' is vacuous.

    MUTATION: copy ``_SHIPPED_KV_ENV`` into ``_PAPER_KV_ENV`` -> FAILS. Reverted.
    """
    assert R.get("paper-kv").env != R.get("shipped-kv").env
    assert R.get("paper-kv").measured["avg_bits_per_elem"] \
        != R.get("shipped-kv").measured["avg_bits_per_elem"]
    assert R.get("paper-weight").packer_args != R.get("shipped-weight").packer_args
    assert R.get("paper-weight").measured["bits_per_weight"] \
        != R.get("shipped-weight").measured["bits_per_weight"]


# ════════════════════════════════════════════════════════════════════════════════
# 6. the honesty fields — the reason this registry exists at all
# ════════════════════════════════════════════════════════════════════════════════

def test_paper_presets_declare_no_accuracy_measurement() -> None:
    """A reader must not be able to assume Table 1's accuracy transfers to a
    paper-budget recipe. The field is load-bearing documentation, so it is gated.

    MUTATION: change ``paper-kv``'s accuracy_status to "MEASURED — same as shipped-kv"
    -> FAILS (does not start with NONE). Reverted -> passes.
    """
    for name in ("paper-kv", "paper-weight"):
        status = R.get(name).accuracy_status
        assert status.startswith("NONE"), (
            f"{name}.accuracy_status must open by stating that this repo has NO "
            f"accuracy measurement for it; got {status[:60]!r}")
    assert R.get("shipped-kv").accuracy_status.startswith("MEASURED")


def test_every_recipe_declares_provenance_and_serving_status() -> None:
    """Each entry must say where its number came from and whether the shipped code can
    actually serve it — ``paper-weight`` in particular PACKS but does not SERVE
    (``linear_method.require_kernel_readable`` refuses a bitmap/no-map bundle).

    MUTATION: blank any ``measured_by`` -> FAILS. Reverted -> passes.
    """
    for rec in R.RECIPES.values():
        assert rec.measured_by.strip(), f"{rec.name}: empty measured_by"
        assert rec.serving_status.strip(), f"{rec.name}: empty serving_status"
        assert rec.axis in ("kv", "weight")
        assert rec.measured, f"{rec.name}: no measured numbers"
    assert R.get("paper-weight").serving_status.startswith("PACKS; SERVES ONLY VIA")


def test_paper_weight_is_actually_refused_by_the_weight_kernel_gate() -> None:
    """The ``serving_status`` claim is verified against the code that enforces it, not
    merely asserted in prose: by DEFAULT a paper-weight bundle is refused, and a
    shipped-weight bundle is not.

    Deliberately gates only the DEFAULT refusal. ``linear_method.py`` belongs to another
    change and is moving (it grew an opt-in transcode path mid-flight); the durable
    claim — and the one the registry makes — is that serving the paper's encoding is
    never something that just happens.

    MUTATION: ``lm.plan_transcode = lambda r: None`` (the planner is what decides now)
    -> ``require_kernel_readable`` returns None, nothing raises -> FAILS. Reverted ->
    passes. (An earlier draft mutated ``KERNEL_OUTLIER_REPRS``/``KERNEL_OUTLIER_MAPS``
    instead; that mutation SURVIVED, because those tuples no longer decide the refusal —
    which is exactly why the mutation step is not optional.)
    """
    # NOT importorskip: linear_method.py is first-party and the package's import
    # contract says it imports with no vllm / no triton / no GPU (backend.py is the one
    # exception). importorskip would convert a real breakage into a silent skip — proven:
    # renaming ``w_format.plan_transcode`` made this gate SKIP instead of FAIL.
    from ommx_gpu_serve.integration.vllm import linear_method as lm
    recipe = w_packer.build_recipe(**dict(R.get("paper-weight").packer_args))
    with pytest.raises(Exception) as exc:
        lm.require_kernel_readable(recipe)
    assert "bitmap" in str(exc.value)
    # …and the shipped one is accepted, so the gate is discriminative, not a blanket no.
    lm.require_kernel_readable(
        w_packer.build_recipe(**dict(R.get("shipped-weight").packer_args)))


def test_paper_weight_resident_cost_matches_the_registrys_prose() -> None:
    """The registry says a SERVED paper-weight bundle costs 4.1250 bit/weight resident
    even though it is 3.6250 on disk. That is the single most misquotable fact about
    this preset — the paper's E2 budget is a STORAGE figure on this implementation — so
    it is measured here rather than left in prose.

    Skips (never silently passes) if the transcode planner disappears: it lives in
    another change's files and this gate must not become a blocker on someone else's
    work, but it must also never report success without having checked.

    MUTATION: change the registry's "measured RESIDENT 4.1250" to "4.0000" -> FAILS
    ("registry prose says 4.0000, plan_transcode measures 4.125"). Reverted -> passes.
    """
    # Same reasoning as above: import for real, so an import-time breakage FAILS. The
    # skip below stays, because "the planner was removed / now returns None" is an API
    # question about someone else's change, not a breakage of this gate.
    from ommx_gpu_serve.linear import w_format as wf
    plan_transcode = getattr(wf, "plan_transcode", None)
    if plan_transcode is None:
        # The planner disappearing is only a LEGITIMATE skip if the claim it backs
        # disappeared with it. While the registry still quotes a measured resident cost,
        # an unmeasurable claim must FAIL — otherwise renaming the planner leaves
        # "measured RESIDENT 4.1250" standing with nothing checking it, and a skip in a
        # 500-test run is invisible. (Proven: this branch used to swallow exactly that.)
        assert "measured RESIDENT" not in R.get("paper-weight").serving_status, (
            "w_format.plan_transcode is gone but paper-weight.serving_status still "
            "quotes a measured resident bit budget; re-derive it or drop the claim")
        pytest.skip("w_format.plan_transcode is gone AND the registry no longer claims "
                    "a measured resident cost, so there is nothing left to check")
    rec = R.get("paper-weight")
    plan = plan_transcode(w_packer.build_recipe(**dict(rec.packer_args)))
    if plan is None:
        pytest.skip("the shipped kernel now reads bitmap/none directly; paper-weight's "
                    "serving_status is stale and needs rewriting")
    assert plan.on_disk_bits_per_weight == rec.measured["bits_per_weight"] == 3.625
    resident = plan.resident_bits_per_weight
    assert resident == R.get("shipped-weight").measured["bits_per_weight"] == 4.125
    prose = rec.serving_status
    assert f"measured RESIDENT {resident:.4f}" in prose, (
        f"paper-weight.serving_status must quote the measured resident cost "
        f"{resident:.4f}; it says: {prose[-160:]!r}")
    assert "measured on-disk 3.6250" in prose


def test_the_honest_cost_note_quotes_only_measured_ratios() -> None:
    """Every compression ratio written in prose in ``recipes.py`` must BE a measured one.

    The registry's ``measured`` dict is gated by the drift tests above, but the "HONEST
    COST" narrative next to it is free text and was not. It said shipped-kv's effective
    4K ratio was ``3.353x``; the measurement — the registry's own
    ``effective_compression_ratio_at_4k``, and 32/(8.7500+0.8125) summed from the real
    allocated MultiSeqKVPool planes — is ``3.346x``. A reader quoting the comment quotes
    a number nothing produced.

    So: every ``N.NNNx`` in that block must be one of the four measured ratios, and all
    four must appear. The second half is what catches a STALE number left behind, which
    is how 3.353x survived.

    MUTATION: put ``3.353x`` back -> FAILS ("unmeasured ratio(s) quoted: {'3.353x'}").
    MUTATION 2: delete the ``4.096x`` mention -> FAILS (missing measured ratio).
    Both reverted -> pass.
    """
    src = open(R.__file__, encoding="utf-8").read()
    start = src.index("# HONEST COST")
    block = src[start:src.index("_SHIPPED_WEIGHT_ARGS", start)]
    pk, sk = R.measure_kv(R.get("paper-kv")), R.measure_kv(R.get("shipped-kv"))
    measured = {pk["compression_ratio"], pk["effective_compression_ratio_at_4k"],
                sk["compression_ratio"], sk["effective_compression_ratio_at_4k"]}
    allowed = {f"{v:.3f}x" for v in measured}
    quoted = set(re.findall(r"\b\d+\.\d{3}x", block))
    assert not (quoted - allowed), (
        f"unmeasured ratio(s) quoted in recipes.py's HONEST COST note: "
        f"{sorted(quoted - allowed)}; measured are {sorted(allowed)}")
    assert allowed <= quoted, (
        f"HONEST COST no longer quotes every measured ratio; missing "
        f"{sorted(allowed - quoted)}")


def test_kv_presets_use_only_legal_group_sizes() -> None:
    """``kv_window``/``kv_pool`` restrict group_tokens and group_channels to
    {16,32,64,128}; a preset outside that is unbuildable.

    MUTATION: set ``_PAPER_KV_ENV["OMMX_KV_GROUP_TOKENS"]`` to "96" -> FAILS (and
    ``measure_kv`` would raise too). Reverted -> passes.
    """
    for name in R.names("kv"):
        env = R.get(name).env
        assert int(env["OMMX_KV_GROUP_TOKENS"]) in (16, 32, 64, 128)
        assert int(env["OMMX_KV_GROUP_CHANNELS"]) in (16, 32, 64, 128)
        assert 0 < int(env["OMMX_ATTN_OUTLIERS"]) <= int(env["OMMX_KV_GROUP_TOKENS"])


# ════════════════════════════════════════════════════════════════════════════════
# 7. the CLI surfaces stay wired
# ════════════════════════════════════════════════════════════════════════════════

def test_recipes_cli_verify_passes(capsys) -> None:
    """``python3 -m ommx_gpu_serve.recipes verify`` is the operator-facing form of the
    drift gate; it must agree with pytest.

    MUTATION: set ``shipped-weight``'s ``bits_per_weight`` to 4.0 -> rc becomes 1 and
    the output carries "DRIFT" -> FAILS. Reverted -> passes.
    """
    assert R.main(["verify"]) == 0
    out = capsys.readouterr().out
    assert "VERIFY: PASS" in out and "DRIFT" not in out


def test_recipes_cli_export_omits_already_set_vars(
        monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """The ``--export`` form run.sh evals must not clobber the caller's own exports.

    MUTATION: make ``_cmd_env`` print ``rec.env`` instead of ``pending_env(...)`` under
    --export -> FAILS (an OMMX_KV_GROUP_TOKENS export line reappears). Reverted -> passes.
    """
    monkeypatch.setenv("OMMX_KV_GROUP_TOKENS", "64")
    assert R.main(["env", "paper-kv", "--export"]) == 0
    out = capsys.readouterr().out
    assert "export OMMX_KV_GROUP_CHANNELS='128'" in out or \
           "export OMMX_KV_GROUP_CHANNELS=128" in out
    assert not re.search(r"^export OMMX_KV_GROUP_TOKENS=", out, re.M), out


def _run_sh_source() -> str:
    return open(os.path.join(_REPO, "run.sh"), encoding="utf-8").read()


def _extract_shell_function(src: str, name: str) -> str:
    """The literal bytes of one shell function, from ``name() {`` to the column-0 ``}``.

    Extracted rather than restated so the gates below execute the SHIPPED function, not
    a paraphrase of it. Fails loudly if the function is gone: a gate that silently found
    nothing to test is the failure mode this file exists to prevent.
    """
    m = re.search(r"^%s\(\)[^\n]*\n(?:.*\n)*?^\}\n" % re.escape(name), src, re.M)
    assert m, f"run.sh no longer defines a {name}() function"
    return m.group(0)


def test_run_sh_exposes_the_recipe_flag() -> None:
    """``run.sh --recipe`` must exist and route to the registry.

    STRUCTURAL ONLY — deliberately paired with the two BEHAVIOURAL gates below, because
    on its own a string search is close to vacuous: the first draft of this gate
    (``assert "apply_recipe" in src``) survived deleting the CALL SITE, deleting the
    ``eval`` that applies the exports, and replacing the unknown-name abort with a
    warn-and-continue. All three are now killed by the gates below.

    MUTATION: delete the ``--recipe)`` case from run.sh's option loop -> FAILS.
    Reverted -> passes.
    """
    src = _run_sh_source()
    assert "--recipe)" in src, "run.sh no longer parses --recipe"
    assert "ommx_gpu_serve.recipes env" in src, \
        "run.sh's --recipe no longer resolves through the registry"
    # A CALL, not merely the definition: `apply_recipe() {` also contains the substring,
    # so the old check passed with the call site deleted and --recipe silently ignored.
    assert re.search(r"^\s+apply_recipe\b(?!\()", src, re.M), \
        "run.sh defines apply_recipe but never CALLS it — --recipe would be accepted " \
        "and then ignored"


def test_run_sh_apply_recipe_really_exports_the_preset() -> None:
    """Execute run.sh's OWN ``apply_recipe`` body and read the environment back.

    The point is that the preset must actually reach the child processes, i.e. the
    resolved assignments must be EVAL'd, not merely printed. ``venv_python`` is stubbed
    (it is not the code under test); everything else is the shipped bytes.

    MUTATION: replace ``eval "$LINES"`` with ``:`` -> run.sh still prints the "recipe
    paper-kv applied" block, so the old string gate passed, but no OMMX_* var is exported
    and this FAILS ("apply_recipe exported nothing"). Reverted -> passes.
    """
    body = _extract_shell_function(_run_sh_source(), "apply_recipe")
    script = (
        "set -u\n"
        f"REPO={shlex.quote(_REPO)}\n"
        f"venv_python() {{ echo {shlex.quote(sys.executable)}; }}\n"
        + body +
        "RECIPE=paper-kv\napply_recipe >/dev/null\nenv | grep '^OMMX_' | sort\n")
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          env={"PATH": os.environ.get("PATH", "")})
    assert proc.returncode == 0, f"apply_recipe failed: {proc.stderr[-500:]}"
    got = dict(l.split("=", 1) for l in proc.stdout.splitlines() if "=" in l)
    assert got, ("apply_recipe exported nothing — the resolved recipe is printed but "
                 "never applied, so every arm would run the shipped defaults under the "
                 f"preset's name. stdout={proc.stdout!r}")
    want = dict(R.get("paper-kv").env)
    want[R.RECIPE_ENV_VAR] = "paper-kv"
    assert got == want, f"apply_recipe exported {got}, expected {want}"


def test_run_sh_aborts_the_whole_run_on_an_unknown_recipe() -> None:
    """``./run.sh bench --recipe <typo>`` must exit non-zero having run NOTHING.

    A bench under the wrong recipe is worse than no bench, so this is the one run.sh
    behaviour that cannot be left to a string search. Driven as a real subprocess; it
    touches no GPU because ``apply_recipe`` is the first statement of ``do_bench``.

    MUTATION: replace the ``exit 2`` in apply_recipe's failure branch with
    ``echo "continuing with defaults"; return 0`` -> the run proceeds at the shipped
    defaults and this FAILS (rc=0). Reverted -> passes.
    """
    env = dict(os.environ)
    env["GPU"] = "0"          # skip resolve_gpu's interactive refusal paths
    env.pop("RECIPE", None)
    proc = subprocess.run([os.path.join(_REPO, "run.sh"), "bench",
                           "--recipe", "paper_kv", "--methods", "bogus"],
                          cwd=_REPO, capture_output=True, text=True, env=env)
    assert proc.returncode != 0, (
        "run.sh accepted an unknown --recipe and kept going — the sweep would be "
        f"measured at the shipped defaults and labelled paper_kv. stdout={proc.stdout[-400:]!r}")
    blob = proc.stdout + proc.stderr
    assert "NOTHING was run" in blob, blob[-600:]
    # …and it never got as far as an arm.
    assert "unknown method: bogus" not in blob, (
        "run.sh reached the method loop after refusing the recipe")


def test_packer_budget_preset_reports_the_registry_number(capsys) -> None:
    """``w_packer budget --preset paper-weight`` prints the recipe AND a live budget row
    carrying the same 3.6250.

    MUTATION: make ``preset_budget_line`` build a fixed relidx7/idx_range recipe ->
    FAILS (the row reads "relidx7 idx_range 4.1250"). Reverted -> passes.

    The assertion targets the LAST line — the recomputed budget ROW — and not merely
    "3.6250 appears somewhere in the output": the registry blurb printed above the row
    already contains that string and every other token, so a whole-output search would
    pass no matter what the row said. (It did: that was this gate's first draft, and the
    mutation survived it.)
    """
    assert w_packer.main(["budget", "--preset", "paper-weight"]) == 0
    out = capsys.readouterr().out
    row = out.rstrip("\n").splitlines()[-1].split()
    #  gs   npv  %out    repr     map      stored   unpadded
    assert row == ["64", "4", "6.2%", "bitmap", "none", "3.6250", "3.6250"], (
        f"live budget row for paper-weight is {row}")


# ════════════════════════════════════════════════════════════════════════════════
# 8. RESOLUTION REACH — the preset must reach EVERY reader, not just one
# ════════════════════════════════════════════════════════════════════════════════
#
# THE DEFECT THESE GATES CLOSE, reproduced verbatim before the fix. ``OMMX_RECIPE``
# was expanded lazily inside ``config.resolve_serving_config``, which was
# ``recipes.apply_recipe_env``'s only caller in the repo. Every other reader of the
# recipe-controlled env vars went to ``os.environ`` raw, so in any process where
# ``resolve_serving_config`` had not already run the preset did NOTHING — silently.
# One fresh process, ``OMMX_RECIPE=paper-kv``:
#
#     kv_bits_breakdown(128)  ->  gt=32 gc=32 k=3  avg=4.125    (SHIPPED, preset gone)
#     resolve_serving_config()
#     kv_bits_breakdown(128)  ->  gt=128 gc=128 k=10 avg=2.75   (preset honoured)
#
# An operator who selected paper-kv and read a bit budget, packed a bundle or sized a
# pool got the shipped recipe's answer with no warning.
#
# THE MODEL (recipes.resolve_env, stated once, applied everywhere): every reader calls
# the idempotent ``recipes.resolve_env()`` before it reads. These gates run each reader
# in a FRESH SUBPROCESS that never imports ``integration.vllm.config`` — a leaked
# ``os.environ`` key or a module-level latch from an earlier in-process test cannot
# make any of them pass, which is how the previous rounds of this work shipped vacuous
# gates. Every probe asserts the resolved NUMBERS, never a substring of a banner.

_PAPER_KV_NUMBERS = {"group_tokens": 128, "group_channels": 128,
                     "outliers_per_vector": 10, "k": 3.3125, "v": 2.1875,
                     "avg": 2.75, "total": 5.5}
_BARE_DEFAULT_NUMBERS = {"group_tokens": 32, "group_channels": 32,
                         "outliers_per_vector": 3, "k": 5.25, "v": 3.0,
                         "avg": 4.125, "total": 8.25}


def _probe(code: str, expect_rc: int = 0, **envvars: str) -> "subprocess.CompletedProcess":
    """Run ``code`` in a FRESH interpreter whose environment carries NO ``OMMX_*`` key
    except the ones named here.

    Subprocess rather than ``monkeypatch`` on purpose, and it is the whole point of
    this section: the claim under test is "a process that never called
    ``resolve_serving_config`` still honours the preset". In-process that claim is
    unfalsifiable — pytest imports ``config`` at module scope in THIS file, and
    ``resolve_env`` writes into the real ``os.environ``, so an earlier test's
    resolution could satisfy a later one. A clean interpreter has neither.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMMX_")}
    # LEAK GUARD, and it bit during development: importing ``figure/bench.py`` runs
    # ``os.environ.setdefault("PYTHONNOUSERSITE", "1")`` at MODULE scope, which
    # monkeypatch cannot undo because it is a raw write. Once one test imported bench
    # in-process, every LATER probe inherited PYTHONNOUSERSITE and its interpreter
    # skipped the user site-packages where torch lives — so probes started failing with
    # "No module named 'torch'" for a reason that had nothing to do with recipes. A
    # probe must have the same import capability as the parent interpreter.
    env.pop("PYTHONNOUSERSITE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [_REPO] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update({k: v for k, v in envvars.items() if v is not None})
    proc = subprocess.run([sys.executable, "-c", code], cwd=_REPO, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == expect_rc, (
        f"probe exited {proc.returncode}, wanted {expect_rc}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return proc


_BREAKDOWN_PROBE = """
import sys
# THE PREMISE OF THIS WHOLE SECTION, enforced rather than assumed: this process must
# never CALL resolve_serving_config, the function that used to own the expansion.
# (Asserting it is merely absent from sys.modules does NOT work and is not the claim:
# ``integration/vllm/__init__`` imports config, so it is always loaded. Importing it
# expands nothing; only calling it did.) Replace it with a tripwire BEFORE the reader
# under test is imported, so a future refactor that quietly re-routes through the
# serving path fails this gate instead of passing it for the old reason.
import ommx_gpu_serve.integration.vllm.config as _cfg
def _tripwire(*a, **k):
    raise AssertionError("resolve_serving_config was called: this probe no longer "
                         "tests that OTHER readers expand OMMX_RECIPE themselves")
_cfg.resolve_serving_config = _tripwire
from ommx_gpu_serve.integration.vllm.packed_only import kv_bits_breakdown
b = kv_bits_breakdown(128)
print(b["recipe"]["group_tokens"], b["recipe"]["group_channels"],
      b["recipe"]["outliers_per_vector"], b["k_bits_per_elem"], b["v_bits_per_elem"],
      b["avg_bits_per_elem"], b["total_bits_per_elem"])
"""


def _parse_breakdown(out: str) -> dict:
    gt, gc, k, kb, vb, avg, tot = out.split()
    return {"group_tokens": int(gt), "group_channels": int(gc),
            "outliers_per_vector": int(k), "k": float(kb), "v": float(vb),
            "avg": float(avg), "total": float(tot)}


def test_preset_reaches_kv_bits_breakdown_with_no_serving_path() -> None:
    """THE REPRODUCTION, as a gate. ``OMMX_RECIPE=paper-kv`` + ``kv_bits_breakdown`` in
    a process that never touches ``resolve_serving_config`` must return PAPER-KV's
    measured numbers (K 3.3125 / V 2.1875 / avg 2.7500 / total 5.5000 at gt=gc=128,
    k=10) — the pre-fix answer was the shipped 32/32/3 -> 4.125.

    MUTATION: delete the ``_resolve_recipe_env()`` call from the top of
    ``packed_only.kv_bits_breakdown`` and from ``packed_only._env`` / ``_env_int`` /
    ``_env_float`` / ``_os_present`` / ``_env_flag_config`` / ``_env_flag_outlier_map``
    -> FAILS with the pre-fix numbers {'group_tokens': 32, ..., 'avg': 4.125}.
    Reverted -> passes.
    """
    got = _parse_breakdown(_probe(_BREAKDOWN_PROBE, OMMX_RECIPE="paper-kv").stdout)
    assert got == _PAPER_KV_NUMBERS


def test_no_preset_leaves_kv_bits_breakdown_byte_identical() -> None:
    """The control for the gate above: unset ``OMMX_RECIPE`` must still resolve the bare
    default recipe (k=3, pow2 off, gt=gc=32 -> K 5.250 / V 3.000 / avg 4.125), i.e. the
    fix is additive and did not smuggle a recipe in.

    MUTATION: make ``recipes.resolve_env`` fall back to ``"shipped-kv"`` when
    ``OMMX_RECIPE`` is unset -> FAILS (avg 4.375, k=6). Reverted -> passes.
    """
    got = _parse_breakdown(_probe(_BREAKDOWN_PROBE).stdout)
    assert got == _BARE_DEFAULT_NUMBERS


# The residual block of ``kv_bits_breakdown`` reads ``OMMX_KV_RING`` RAW — not through
# one of the ``_env*`` accessors — and it is reached BEFORE any accessor fires when the
# caller supplies every keyword. Only the explicit ``_resolve_recipe_env()`` at the top
# of the function covers that read; the seven accessor-level calls do not. Deleting just
# that one line left the whole 592-test suite green, so the probe below exists to close
# it. Verified consequence of the deletion, at OMMX_RECIPE=paper-kv:
#     residual {'kv_ring': True,  'bf16_rows_per_seq': 296,  'bits_per_elem': 2.3125}
#  -> residual {'kv_ring': False, 'bf16_rows_per_seq': None, 'bits_per_elem': 32.0}
# i.e. the bf16 shadow is priced at the FULL per-token rate instead of the ring's
# O(1)/request rate — a 13.8x overstatement of the residual footprint, silently.
_KWARGD_BREAKDOWN_PROBE = """
import ommx_gpu_serve.integration.vllm.config as _cfg
def _tripwire(*a, **k):
    raise AssertionError("resolve_serving_config was called: this probe no longer "
                         "tests that packed_only expands OMMX_RECIPE itself")
_cfg.resolve_serving_config = _tripwire
from ommx_gpu_serve.integration.vllm.packed_only import kv_bits_breakdown
# EVERY keyword supplied, so no _env* accessor runs before the residual block's raw
# os.environ.get("OMMX_KV_RING"). This is the call shape recipes.measure_kv is
# documented as making, and the one the top-of-function resolve exists for.
b = kv_bits_breakdown(128, k_format='i2f4', group_tokens=128, group_channels=128,
                      outliers_per_vector=10, outlier_repr='relidx7',
                      kv_outlier_map=True, kv_int8_scale=True, use_pow2=True,
                      scale_bytes=1, seq_len=4096)
r = b["residual"]
print(int(r["kv_ring"]), r["bf16_rows_per_seq"], r["bits_per_elem"])
"""


def test_preset_reaches_the_residual_block_of_a_fully_kwargd_breakdown() -> None:
    """The one read in ``kv_bits_breakdown`` the accessor-level resolution cannot reach.

    MUTATION: delete ONLY the ``_resolve_recipe_env()`` at the top of
    ``packed_only.kv_bits_breakdown`` (leaving all seven ``_env*`` accessor calls in
    place) -> FAILS ("['0', 'None', '32.0']"). Reverted -> passes. Before this gate
    existed that same deletion passed the entire suite: 592 passed, 4 skipped.
    """
    on = _probe(_KWARGD_BREAKDOWN_PROBE, OMMX_RECIPE="paper-kv").stdout.split()
    assert on == ["1", "296", "2.3125"], on
    # Additivity control: with no preset the ring is OFF and the residual is priced at
    # the full bf16 rate, exactly as it was before this feature existed.
    off = _probe(_KWARGD_BREAKDOWN_PROBE).stdout.split()
    assert off == ["0", "None", "32.0"], off


_POOL_PROBE = """
import torch
from ommx_gpu_serve.attention.kv_pool import MultiSeqKVPool
from ommx_gpu_serve.attention.kv_window import WindowSpec
p = MultiSeqKVPool(num_seqs=2, max_seq_len=512, n_kv_heads=2, head_dim=64,
                   outliers_per_vector=6, use_pow2=True, window=WindowSpec(),
                   device="cpu")
print(int(p.kv_ring), p.k_hist.shape[1], p.v_hist.shape[1], p.ring_cap,
      int(p.kv_outlier_map))
"""

_STORE_PROBE = """
import torch
from ommx_gpu_serve.attention.kv_store import CanonicalKVStore
from ommx_gpu_serve.attention.kv_window import WindowSpec
s = CanonicalKVStore(max_seq_len=512, n_kv_heads=2, head_dim=64,
                     outliers_per_vector=6, use_pow2=True, window=WindowSpec(),
                     device="cpu")
print(int(s.kv_ring), s.k_hist.shape[0], s.kv_cap, int(s.kv_outlier_map))
"""


def test_preset_reaches_a_real_multiseq_kv_pool_allocation() -> None:
    """A REAL allocation, not an accounting call. Both registered presets set
    ``OMMX_KV_RING=1`` while the bare default is OFF, so an unresolved preset made the
    pool allocate the full ``[num_seqs, max_seq_len, H, D]`` bf16 shadow (512 rows per
    request here) instead of the ``sink + recent + 2*gt = 8 + 32 + 64 = 104``-row ring.
    That is a 4.9x difference in the biggest tensor the pool owns, decided silently by
    whether some other module had run first.

    MUTATION: delete ``_resolve_recipe_env()`` from the ring-sizing block of
    ``kv_pool.MultiSeqKVPool.__init__`` -> FAILS ("0 512 512 512 1", i.e. ring OFF and
    the full shadow allocated). Reverted -> passes.
    """
    on = _probe(_POOL_PROBE, OMMX_RECIPE="paper-kv").stdout.split()
    assert on == ["1", "104", "104", "104", "1"], on
    off = _probe(_POOL_PROBE).stdout.split()
    assert off == ["0", "512", "512", "512", "1"], off


def test_preset_reaches_a_real_canonical_kv_store_allocation() -> None:
    """Same claim for the single-sequence container (``kv_store.py``), which has its own
    copy of the ring-sizing block and its own raw env read.

    MUTATION: delete ``_resolve_recipe_env()`` from the ring-sizing block of
    ``kv_store.CanonicalKVStore.__init__`` -> FAILS ("0 512 512 1"). Reverted -> passes.
    """
    # kv_cap (the QUANTIZED-plane capacity) stays max_seq_len either way; it is
    # k_hist — the bf16 residual shadow — that collapses 512 -> 104 under the ring.
    on = _probe(_STORE_PROBE, OMMX_RECIPE="paper-kv").stdout.split()
    assert on == ["1", "104", "512", "1"], on
    off = _probe(_STORE_PROBE).stdout.split()
    assert off == ["0", "512", "512", "1"], off


_PACKER_PROBE = """
import os, torch
from ommx_gpu_serve.attention.pack import ommx_pack_kv_canonical_block
K = torch.randn(32, 2, 64); V = torch.randn(32, 2, 64)
ommx_pack_kv_canonical_block(K, V, outliers_per_vector=6, use_pow2=True,
                             outlier_select="signed", device="cpu")
# What the pack call left behind: the preset, materialised for THIS process, which is
# what makes every reader after it (including the accounting) agree with the planes
# that were just written.
print(os.environ.get("OMMX_KV_GROUP_TOKENS"), os.environ.get("OMMX_KV_OUTLIER_MAP"),
      os.environ.get("OMMX_ATTN_OUTLIERS"), os.environ.get("OMMX_KV_RING"))
"""


def test_preset_reaches_the_kv_packer() -> None:
    """``pack.ommx_pack_kv_canonical_block`` reads ``OMMX_KV_OUTLIER_MAP`` (which both
    presets name explicitly, precisely so a published recipe does not ride on a default
    holding still) directly out of ``os.environ``.

    HONEST SCOPE, so this gate is not read as more than it is: neither REGISTERED preset
    moves ``OMMX_KV_OUTLIER_MAP`` off its default, so today no preset changes which
    planes this function writes. What IS observable, and what is asserted, is that the
    packer expands the preset ITSELF — a third preset with ``OMMX_KV_OUTLIER_MAP=0``
    would otherwise pack base-shared outliers or a dedicated FP4 map depending purely on
    whether some unrelated module had run first. The unknown-name gate below covers the
    same call from the other side.

    MUTATION: delete ``_resolve_recipe_env()`` from
    ``pack.ommx_pack_kv_canonical_block`` -> FAILS ("None None None None"). Reverted ->
    passes.
    """
    got = _probe(_PACKER_PROBE, OMMX_RECIPE="paper-kv").stdout.split()
    assert got == ["128", "1", "10", "1"], got
    # Additivity: with no preset the packer must leave the environment alone.
    assert _probe(_PACKER_PROBE).stdout.split() == ["None", "None", "None", "None"]


# ── the weight axis: the packer's budget path ────────────────────────────────────

def test_preset_reaches_the_packer_budget_path(capsys) -> None:
    """``w_packer budget`` is the weight axis's budget path and it had no env
    resolution at all, so ``OMMX_RECIPE=paper-weight`` printed the generic sweep table
    and the operator had to notice their preset had quietly done nothing.

    MUTATION: change ``budget_preset = env_preset(args.preset)`` back to
    ``args.preset`` in ``w_packer.main`` -> FAILS (the paper-weight detail block and its
    3.6250 row are absent). Reverted -> passes.
    """
    os.environ[R.RECIPE_ENV_VAR] = "paper-weight"      # scrubbed by the autouse fixture
    try:
        assert w_packer.main(["budget"]) == 0
    finally:
        os.environ.pop(R.RECIPE_ENV_VAR, None)
    out = capsys.readouterr().out
    assert "recipe        paper-weight" in out
    # the NUMBER, not just the name: bitmap positions + no range map at gs=64, npv=4.
    assert re.search(r"^\s*64\s+4\s+6\.2%\s+bitmap\s+none\s+3\.6250\s+3\.6250\s*$",
                     out, re.M), out


def test_explicit_packer_preset_flag_beats_the_env(capsys) -> None:
    """Precedence on the weight axis mirrors the KV side: EXPLICIT beats preset.

    MUTATION: make ``w_packer.env_preset`` ignore its ``default`` argument (always
    return the env name) -> FAILS (the paper-weight 3.6250 row is printed instead of
    shipped-weight's 4.1250). Reverted -> passes.
    """
    os.environ[R.RECIPE_ENV_VAR] = "paper-weight"
    try:
        assert w_packer.main(["budget", "--preset", "shipped-weight"]) == 0
    finally:
        os.environ.pop(R.RECIPE_ENV_VAR, None)
    out = capsys.readouterr().out
    assert "recipe        shipped-weight" in out
    assert re.search(r"^\s*64\s+4\s+6\.2%\s+relidx7\s+idx_range\s+4\.1250\s+4\.0625\s*$",
                     out, re.M), out


def test_a_kv_preset_on_the_weight_axis_is_reported_not_silent(capsys) -> None:
    """A KV ``OMMX_RECIPE`` genuinely carries no weight knobs, so the packer keeps its
    defaults — but it says so on STDERR rather than letting the operator assume the
    preset applied. Stderr, not stdout, because ``budget``'s stdout is a table.

    MUTATION: make ``recipes.env_preset_name`` return ``None`` for a cross-axis name
    WITHOUT calling ``note`` -> FAILS (nothing on stderr). Reverted -> passes.
    """
    os.environ[R.RECIPE_ENV_VAR] = "paper-kv"
    try:
        assert w_packer.main(["budget", "--group-sizes", "64", "--npvs", "4"]) == 0
    finally:
        os.environ.pop(R.RECIPE_ENV_VAR, None)
    cap = capsys.readouterr()
    assert "paper-kv" in cap.err and "KV recipe" in cap.err
    assert "shipped-weight" in cap.err and "paper-weight" in cap.err
    assert "paper-kv" not in cap.out, "the note must not pollute the parseable table"
    # and the defaults really did survive
    assert re.search(r"^\s*64\s+4\s+6\.2%\s+relidx7\s+idx_range\s+4\.1250", cap.out, re.M)


# ── idempotence, precedence, and the absence of a cache ─────────────────────────

def test_resolving_twice_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Idempotence, including under a partial override: the second call must not
    "finish the job" by overwriting the knob the operator pinned.

    MUTATION: make ``resolve_env`` assign unconditionally
    (``for k, v in preset_env(rec.name).items(): env[k] = v``) -> FAILS
    ("OMMX_KV_GROUP_TOKENS 128 != 64"). Reverted -> passes.
    """
    monkeypatch.setenv(R.RECIPE_ENV_VAR, "paper-kv")
    monkeypatch.setenv("OMMX_KV_GROUP_TOKENS", "64")     # operator pins one knob
    assert R.resolve_env() == "paper-kv"
    first = dict(os.environ)
    assert R.resolve_env() == "paper-kv"
    assert dict(os.environ) == first
    assert os.environ["OMMX_KV_GROUP_TOKENS"] == "64"    # explicit still wins
    assert os.environ["OMMX_KV_GROUP_CHANNELS"] == "128"  # rest of the preset applied


def test_resolution_is_not_cached_and_cannot_stick(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """PROOF THAT NO CACHE CAN MAKE A LATER TEST PASS FOR THE WRONG REASON.

    ``resolve_env`` deliberately has no memo and no module-level latch: it re-reads the
    environment every call. Resolve paper-kv, wipe every OMMX_* key, then resolve
    shipped-kv IN THE SAME PROCESS — a cached or latched implementation would either
    return the first answer or write nothing at all.

    MUTATION: add ``global _DONE`` / ``if _DONE: return _DONE`` memoisation to
    ``resolve_env`` -> FAILS ("paper-kv" returned the second time, and gt is None
    because nothing was written). Reverted -> passes.
    """
    monkeypatch.setenv(R.RECIPE_ENV_VAR, "paper-kv")
    assert R.resolve_env() == "paper-kv"
    assert os.environ["OMMX_KV_GROUP_TOKENS"] == "128"
    for key in [k for k in os.environ if k.startswith("OMMX_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(R.RECIPE_ENV_VAR, "shipped-kv")
    assert R.resolve_env() == "shipped-kv"
    assert os.environ["OMMX_KV_GROUP_TOKENS"] == "32"
    assert os.environ["OMMX_ATTN_OUTLIERS"] == "6"


def test_resolving_with_no_preset_mutates_nothing(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset (and blank) ``OMMX_RECIPE`` must leave the environment byte-identical.

    MUTATION: drop the empty-string check in ``resolve_env``
    (``if want is None:`` only) -> FAILS on the blank case with UnknownRecipeError.
    Reverted -> passes.
    """
    monkeypatch.delenv(R.RECIPE_ENV_VAR, raising=False)
    before = dict(os.environ)
    assert R.resolve_env() is None
    assert dict(os.environ) == before
    monkeypatch.setenv(R.RECIPE_ENV_VAR, "   ")
    before = dict(os.environ)
    assert R.resolve_env() is None
    assert dict(os.environ) == before


# ── unknown name: EVERY entry point, not just one ───────────────────────────────

# (probe source, module the reader lives in). Each one is a DIFFERENT reader with its
# own env access; a fix applied to some of them is the same defect with a smaller blast
# radius, so they are enumerated rather than sampled.
_ENTRY_POINTS = {
    "recipes.resolve_env":
        "from ommx_gpu_serve.recipes import resolve_env; resolve_env()",
    "config.resolve_serving_config":
        "from ommx_gpu_serve.integration.vllm.config import resolve_serving_config;"
        " resolve_serving_config()",
    "packed_only.kv_bits_breakdown":
        "from ommx_gpu_serve.integration.vllm.packed_only import kv_bits_breakdown;"
        " kv_bits_breakdown(128)",
    "kv_pool.MultiSeqKVPool":
        "import torch;"
        " from ommx_gpu_serve.attention.kv_pool import MultiSeqKVPool;"
        " MultiSeqKVPool(num_seqs=1, max_seq_len=64, n_kv_heads=2, head_dim=64,"
        " device='cpu')",
    "kv_store.CanonicalKVStore":
        "import torch;"
        " from ommx_gpu_serve.attention.kv_store import CanonicalKVStore;"
        " CanonicalKVStore(max_seq_len=64, n_kv_heads=2, head_dim=64, device='cpu')",
    "pack.ommx_pack_kv_canonical_block":
        "import torch;"
        " from ommx_gpu_serve.attention.pack import ommx_pack_kv_canonical_block;"
        " ommx_pack_kv_canonical_block(torch.randn(32, 2, 64), torch.randn(32, 2, 64),"
        " device='cpu')",
    "w_packer.env_preset":
        "from ommx_gpu_serve.linear.w_packer import env_preset; env_preset(None)",
    "figure/bench.py::_install_ommx_recipe_env":
        "import sys; sys.path.insert(0, 'figure');"
        " import bench; bench._install_ommx_recipe_env()",
}


@pytest.mark.parametrize("entry", sorted(_ENTRY_POINTS))
def test_unknown_preset_raises_from_every_entry_point(entry: str) -> None:
    """A typo'd ``OMMX_RECIPE`` must stop EVERY reader with the known names listed —
    not fall through to the defaults and produce a plausible number under a recipe
    nobody selected. Run in a fresh interpreter so no earlier resolution can mask it.

    MUTATION (per entry): delete that reader's ``resolve_env`` /
    ``_resolve_recipe_env`` / ``env_preset_name`` call -> that parameter FAILS (rc 0,
    no traceback). Reverted -> passes. Verified for all eight, one at a time.
    """
    proc = _probe(_ENTRY_POINTS[entry], expect_rc=1, OMMX_RECIPE="papper-kv")
    assert "UnknownRecipeError" in proc.stderr, proc.stderr
    assert "papper-kv" in proc.stderr
    for known in R.RECIPE_NAMES:
        assert known in proc.stderr, f"{known!r} missing from:\n{proc.stderr}"


#: ``figure/bench.py`` is deliberately EXCLUDED from the no-op gate below, and the
#: exclusion is named rather than left as a quiet gap: that file's whole job is to
#: install its own published recipe (``OMMX_RECIPE_ENV``, 11 keys) when no preset is
#: named, so "left the environment untouched" is the wrong contract for it. Its
#: additivity is gated instead by
#: ``test_bench_with_no_preset_is_byte_identical_to_today``, which asserts the resolved
#: environment equals ``OMMX_RECIPE_ENV`` EXACTLY and the footprint is the shipped
#: 4.375 — a stricter statement than "nothing leaked", not a weaker one.
_NO_OP_ENTRY_POINTS = sorted(set(_ENTRY_POINTS)
                             - {"figure/bench.py::_install_ommx_recipe_env"})


#: Per-entry extra assertion for a reader whose additivity the "nothing leaked" check
#: cannot express. ``w_packer.env_preset`` is PURE — it never writes to the
#: environment — so "no OMMX_* leaked" is true of it no matter what it returns, and a
#: gate that only checked leakage would survive a mutation that made it invent a preset
#: out of thin air. Assert the RESOLUTION instead. (Verified: with only the leak check,
#: the mutation "``env_preset_name`` reads ``env.get(RECIPE_ENV_VAR, 'shipped-weight')``"
#: SURVIVED; with this line it dies.)
_NO_OP_EXTRA = {"w_packer.env_preset": "assert env_preset(None) is None, env_preset(None)"}


@pytest.mark.parametrize("entry", _NO_OP_ENTRY_POINTS)
def test_every_entry_point_is_a_no_op_without_a_preset(entry: str) -> None:
    """The additivity half of the gate above, entry point by entry point: with
    ``OMMX_RECIPE`` unset each reader must run clean and leave the environment with no
    OMMX knob in it that it did not find there.

    MUTATION: make ``resolve_env`` treat an unset ``OMMX_RECIPE`` as ``"paper-kv"``
    -> FAILS for six of the seven entries (each prints the leaked keys). Reverted ->
    passes. The seventh, ``w_packer.env_preset``, is killed by the separate mutation
    named on ``_NO_OP_EXTRA``.
    """
    code = (_ENTRY_POINTS[entry] + "\n" + _NO_OP_EXTRA.get(entry, "") + "\nimport os\n"
            "leaked = sorted(k for k in os.environ if k.startswith('OMMX_'))\n"
            "assert not leaked, leaked\nprint('clean')")
    assert _probe(code).stdout.strip().endswith("clean")


# ── figure/bench.py: the precedence inversion and the provenance record ─────────

_BENCH_PROBE = """
import json, os, sys
sys.path.insert(0, 'figure')
import bench
prov = bench._install_ommx_recipe_env()
from ommx_gpu_serve.integration.vllm.packed_only import kv_bits_breakdown
prov = dict(prov)
prov['avg_bits_per_elem'] = kv_bits_breakdown(128)['avg_bits_per_elem']
print(json.dumps(prov))
"""


def _bench_probe(**env: str) -> dict:
    return json.loads(_probe(_BENCH_PROBE, **env).stdout)


def test_bench_named_preset_beats_the_benchs_own_defaults() -> None:
    """THE PRECEDENCE INVERSION. ``figure/bench.py`` installed its own 11-key
    ``OMMX_RECIPE_ENV`` with ``os.environ.setdefault`` at model-build time, i.e. BEFORE
    anything expanded ``OMMX_RECIPE``. setdefault = first writer wins, so the bench's
    built-in defaults outranked the named preset and ``OMMX_RECIPE=paper-kv
    figure/bench.py --method ommx`` could not move the bench: it measured shipped-kv
    (gt=32, k=6, 4.375 avg bit) while calling itself paper-kv.

    MUTATION: in ``_install_ommx_recipe_env``, move the ``os.environ.setdefault`` loop
    ABOVE the ``resolve_env()`` call (the pre-fix order) -> FAILS
    (avg_bits_per_elem 4.375, gt 32). Reverted -> passes.
    """
    got = _bench_probe(OMMX_RECIPE="paper-kv")
    assert got["preset"] == "paper-kv"
    assert got["resolved"]["OMMX_KV_GROUP_TOKENS"] == "128"
    assert got["resolved"]["OMMX_KV_GROUP_CHANNELS"] == "128"
    assert got["resolved"]["OMMX_ATTN_OUTLIERS"] == "10"
    assert got["avg_bits_per_elem"] == 2.75


def test_bench_explicit_env_still_beats_the_named_preset() -> None:
    """The other half of the ordering: operator env > preset > bench defaults. Moving
    the preset above the bench's dict must NOT move it above the operator's own export.

    MUTATION: make ``_install_ommx_recipe_env`` call
    ``resolve_env(name='paper-kv')``-style unconditional assignment (or drop the
    ``pending_env`` filter inside ``resolve_env``) -> FAILS
    ("OMMX_KV_GROUP_TOKENS 128 != 64"). Reverted -> passes.
    """
    got = _bench_probe(OMMX_RECIPE="paper-kv", OMMX_KV_GROUP_TOKENS="64")
    assert got["resolved"]["OMMX_KV_GROUP_TOKENS"] == "64"
    assert got["source"]["OMMX_KV_GROUP_TOKENS"] == "operator-env"
    assert got["source"]["OMMX_KV_GROUP_CHANNELS"] == "preset:paper-kv"
    # 128-channel V + 64-token K groups, k=10 -> measured, not assumed:
    assert got["avg_bits_per_elem"] == 3.40625


def test_bench_with_no_preset_is_byte_identical_to_today() -> None:
    """Additivity for the bench: with ``OMMX_RECIPE`` unset the resolved environment
    must be exactly ``OMMX_RECIPE_ENV`` and the footprint the shipped 4.375.

    MUTATION: give ``_install_ommx_recipe_env`` a default preset
    (``resolve_env(name='paper-kv')``) -> FAILS (avg 2.75, preset 'paper-kv').
    Reverted -> passes.
    """
    got = _bench_probe()
    assert got["preset"] is None
    assert got["preset_env_requested"] is None
    assert got["resolved"] == dict(_bench_module().OMMX_RECIPE_ENV)
    assert set(got["source"].values()) == {"bench-default"}
    assert got["avg_bits_per_elem"] == 4.375


_PROVENANCE_PROBE = """
import json, sys
sys.path.insert(0, 'figure')
import bench
bench._install_ommx_recipe_env()
print(json.dumps(bench._ommx_provenance()))
"""


def test_bench_provenance_names_the_preset_and_every_knobs_source() -> None:
    """THE PROVENANCE DEFECT, exercised through the function that BUILDS the JSON
    fields (``bench._ommx_provenance``), not through the resolver alone. The old block
    recorded ``{k: os.environ.get(k) for k in OMMX_RECIPE_ENV}`` — eleven keys, no
    preset name, and no ``OMMX_KV_OUTLIER_MAP`` (which the presets DO set and that dict
    does not). A paper-kv run therefore produced a JSON that could not say which recipe
    made it, and whose eleven values would have looked like an ordinary shipped run.

    MUTATION: delete the ``"ommx_recipe_provenance"`` key from
    ``bench._ommx_provenance`` (leaving exactly the old 11-key ``ommx_recipe`` dict)
    -> FAILS (KeyError 'ommx_recipe_provenance' in the probe, rc 1). Reverted ->
    passes.
    """
    doc = json.loads(_probe(_PROVENANCE_PROBE, OMMX_RECIPE="paper-kv").stdout)
    # the old field survives unchanged, so existing readers still parse
    assert set(doc["ommx_recipe"]) == set(_bench_module().OMMX_RECIPE_ENV)
    got = doc["ommx_recipe_provenance"]
    assert got["preset"] == "paper-kv"
    # the knob the bench's own dict does not carry at all
    assert got["resolved"]["OMMX_KV_OUTLIER_MAP"] == "1"
    assert got["source"]["OMMX_KV_OUTLIER_MAP"] == "preset:paper-kv"
    # every recorded knob is attributed to exactly one source
    assert set(got["source"]) == set(got["resolved"])
    assert set(got["source"].values()) <= {"operator-env", "preset:paper-kv",
                                           "bench-default"}


def test_bench_main_actually_records_the_provenance_it_builds() -> None:
    """The wiring inch the behavioural gate above cannot reach without a GPU and an 8B
    checkpoint: ``main`` must actually CALL ``_ommx_provenance`` on the ommx arm. An
    AST check of the call site, the same technique ``test_paged_decode_env.py`` uses for
    its call-site half — it is a complement to the behavioural gate, never a substitute.

    MUTATION: revert ``main``'s ommx branch to the old inline
    ``prov["ommx_recipe"] = {k: os.environ.get(k) for k in OMMX_RECIPE_ENV}``
    -> FAILS ("_ommx_provenance is never called from main"). Reverted -> passes.
    """
    src = open(os.path.join(_REPO, "figure", "bench.py"), encoding="utf-8").read()
    main_fn = next(n for n in ast.walk(ast.parse(src))
                   if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {n.func.id for n in ast.walk(main_fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_ommx_provenance" in called, (
        "_ommx_provenance is never called from main(): the provenance block is built "
        "but never reaches the output JSON")


def test_bench_load_model_actually_installs_the_recipe_on_the_ommx_arm() -> None:
    """The OTHER wiring inch, and the one that was missing: ``load_model``'s ``ommx``
    branch must CALL ``_install_ommx_recipe_env``. Every behavioural gate above calls
    that function directly, so all of them stay green with the production call site
    deleted — measured: deleting it left the suite at 592 passed, 4 skipped.

    What the deletion actually costs, measured in-process:
      * ``OMMX_RECIPE`` unset  -> the arm silently drops from the PUBLISHED recipe
        (k=6, pow2 on, 4.375 avg bit — the recipe every OMMX number in this release was
        produced under) to the BARE default (k=3, pow2 OFF, 4.125), because the
        ``OMMX_RECIPE_ENV`` setdefault loop lives inside that function too;
      * the output JSON's ``ommx_recipe`` becomes eleven ``null``s and
        ``ommx_recipe_provenance`` becomes ``{}``.

    AST, for the same reason as the gate above: reaching ``load_model`` needs a GPU and
    an 8B checkpoint. It is scoped to the ``method == "ommx"`` branch specifically, so
    moving the call into some other arm does not satisfy it.

    MUTATION: delete the ``_install_ommx_recipe_env()`` line from ``load_model`` ->
    FAILS. Reverted -> passes.
    """
    src = open(os.path.join(_REPO, "figure", "bench.py"), encoding="utf-8").read()
    load_fn = next(n for n in ast.walk(ast.parse(src))
                   if isinstance(n, ast.FunctionDef) and n.name == "load_model")
    ommx_branches = [
        n for n in ast.walk(load_fn)
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name) and n.test.left.id == "method"
        and any(isinstance(c, ast.Constant) and c.value == "ommx"
                for c in n.test.comparators)]
    assert ommx_branches, (
        "figure/bench.py::load_model no longer has a `method == \"ommx\"` branch; this "
        "gate is pinned to that branch and must be rewritten, not deleted")
    called = {n.func.id for br in ommx_branches for stmt in br.body
              for n in ast.walk(stmt)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_install_ommx_recipe_env" in called, (
        "load_model's ommx branch never calls _install_ommx_recipe_env(): neither the "
        "named preset NOR the bench's own published OMMX_RECIPE_ENV dict is installed, "
        "so the arm measures the bare env defaults (k=3, pow2 off) under whatever name "
        "the run was given")


def _bench_module():
    """Import ``figure/bench.py`` once, for its module constants only.

    NOTE the side effect this pays for and why it is confined here: importing that file
    runs ``os.environ.setdefault("PYTHONNOUSERSITE", "1")`` at module scope, a raw write
    monkeypatch cannot undo. ``_probe`` strips that key from every subprocess
    environment precisely because of this.
    """
    sys.path.insert(0, os.path.join(_REPO, "figure"))
    import bench                                        # noqa: PLC0415
    return bench
