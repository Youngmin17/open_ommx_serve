# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Gate for ``attention/paged_decode.py::_env_int`` and EVERY knob that reads it.

WHY THIS FILE EXISTS. ``_env_int`` used to end in ``return max(1, int(raw))``. That
clamp is the law-#11 trap in its pure form: a knob whose ``0`` means something
("use the ladder", "no threshold") was silently rewritten to ``1``, so the operator
got the OPPOSITE of what was typed with nothing in the output to say so — and a
NEGATIVE value, which is always a typo, became a working geometry. The current body
raises instead, and grew an ``allow_zero=`` kwarg for the three knobs where 0 IS a
value.

That change shipped with no coverage, and an adversarial mutation proved the gap:
reverting BOTH halves at once — stripping ``allow_zero=True`` from the two
``OMMX_OVERSPLIT_*`` call sites AND undoing the ``_auto_num_kv_splits`` override
routing — left the suite green. Two distinct properties therefore have to be pinned
SEPARATELY, because each mutation defeats a test that only pins the other:

  (a) the HELPER contract  — unset/blank/malformed/negative/zero, per ``allow_zero``;
  (b) the CALL SITES       — which knobs pass ``allow_zero=True`` and which do not,
                             and that the value each call site computes actually
                             comes back out of ``_env_int``.

(b) is the half the mutation walked through. Where the call site is reachable on a
CPU (the two ``OMMX_OVERSPLIT_*`` thresholds, read inside ``_auto_num_kv_splits``),
it is pinned DYNAMICALLY and DISCRIMINATIVELY — the assertions compare split counts
that MOVE with the knob, so a wrong-but-not-raising value fails too. Where the call
site lives inside the Triton launcher and cannot run without a device
(``OMMX_ATTN_BLOCK_H``, ``OMMX_ATTN_OUTLIER_WARPS``, ``OMMX_MERGE_WIDE_*``), it is
pinned STATICALLY by parsing the module's own AST: stripping ``allow_zero=True`` from
``OMMX_ATTN_OUTLIER_WARPS`` — a mutation with NO CPU-observable behaviour — still
fails :func:`test_allow_zero_is_set_exactly_where_zero_is_meaningful`. A gate that a
deliberate bug can walk through is not a gate; an AST gate that the bug cannot walk
through is one, even though it reads source rather than behaviour.

NO GPU AND NO TRITON WERE USED. ``paged_decode`` imports its Triton symbols lazily
inside ``_require_triton`` and defines the ``@triton.jit`` kernels behind that guard,
so the module imports with neither Triton nor CUDA present — VERIFIED by
:func:`test_module_imports_with_no_triton_present`, which runs that import in a FRESH
INTERPRETER and reads back ``('triton' in sys.modules, _KERNELS_BUILT)``. The check is
deliberately NOT made in-process: both of those are process-global and a
``gpu``-marked test in an earlier file legitimately sets both on a CUDA host, which
made the in-process form report a foreign test's work as this module's regression. No
stub loader is needed; every function exercised below is pure host-side Python
arithmetic.

DIVERGENCE FROM THE BRIEF, RECORDED RATHER THAN PAPERED OVER. The brief lists a
whitespace-only value (``" "``) among the malformed values that must raise. The
shipped code deliberately treats blank as UNSET (``if raw is None or
str(raw).strip() == "": return int(default)``), and the launcher documents that
choice at length (the ``OMMX_V4_NUM_STAGES`` branch comments: a whitespace-only value
must mean "unset -> default" on BOTH sides of that branch, or the per-arch cp.async
depth cap is silently dropped). Editing ``paged_decode.py`` is out of scope for this
file, so the SHIPPED behaviour is what is pinned here — see
:func:`test_blank_is_unset_not_malformed`, which states the divergence in its own
assertion message.
"""
from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys

import pytest

from ommx_gpu_serve.attention import paged_decode as pd

# ── the nine knobs, split by whether 0 is a value for them ──────────────────────
#
# This split is the SPECIFICATION, restated here independently of the module under
# test so that the module cannot move it. ``_env_int``'s docstring gives the same
# three-vs-six division and the reason for each entry.

#: 0 is MEANINGFUL: the two thresholds mean "no threshold", and OUTLIER_WARPS=0 is
#: the DOCUMENTED default meaning "use the ctx/batch ladder", not "one warp".
ZERO_OK = ("OMMX_OVERSPLIT_BATCH", "OMMX_OVERSPLIT_CTX", "OMMX_ATTN_OUTLIER_WARPS")

#: 0 is NOT a value: each selects a launch geometry, and a zero-sized grid / tile /
#: stage count / warp count is not a launch.
ZERO_REFUSED = (
    "OMMX_ATTN_NUM_KV_SPLITS",
    "OMMX_V4_NUM_STAGES",
    "OMMX_V4_NUM_WARPS",
    "OMMX_ATTN_BLOCK_H",
    "OMMX_MERGE_WIDE_BLOCK_DV",
    "OMMX_MERGE_WIDE_NUM_WARPS",
)

ALL_KNOBS = ZERO_OK + ZERO_REFUSED

#: Call sites whose default is a LITERAL (not a computed expression). Pinning these
#: is what makes "unset -> the documented default" checkable for the knobs whose only
#: reader is the Triton launcher: the helper-level test proves ``_env_int`` returns
#: the default it is handed, and this proves WHICH default it is handed.
LITERAL_DEFAULTS = {
    "OMMX_OVERSPLIT_BATCH": {8},
    "OMMX_OVERSPLIT_CTX": {1024},
    "OMMX_ATTN_OUTLIER_WARPS": {0},
    "OMMX_MERGE_WIDE_NUM_WARPS": {2},
    "OMMX_V4_NUM_STAGES": {2, 3, 4},        # the seq/batch ladder's rungs
    "OMMX_V4_NUM_WARPS": {4, 8},            # the num_warps ladder's two rungs
}


def _allow_zero(knob: str) -> bool:
    return knob in ZERO_OK


# ══════════════════════════════════════════════════════════════════════════════
# 0. the environment this file claims to run in
# ══════════════════════════════════════════════════════════════════════════════

def test_module_imports_with_no_triton_present() -> None:
    """The premise of every other test here: importing ``paged_decode`` does not pull
    Triton in and does not build kernels.

    CHECKED IN A FRESH INTERPRETER, ON PURPOSE. ``sys.modules["triton"]`` and
    ``paged_decode._KERNELS_BUILT`` are PROCESS-GLOBAL, and they are properties of the
    whole pytest session rather than of the import under test. Asserting them in-process
    made this test fail on exactly the host the GPU gates are meant to run on: on a CUDA
    box ``test_bitmap_outlier.py::test_kernel_bitmap_read_matches_relidx7_on_gpu``
    executes a decode, which calls ``_build_kernels()`` -> ``import triton`` and sets
    ``_KERNELS_BUILT = True``; that file sorts BEFORE this one, so this test failed with
    "triton got imported by importing paged_decode" while paged_decode was blameless.
    A gate that reports someone else's work as this module's regression is worse than no
    gate — it trains the reader to ignore it.

    The subprocess measures the thing actually claimed: a clean interpreter that imports
    ``ommx_gpu_serve.attention.paged_decode`` and nothing else ends with Triton absent
    and no kernels built. It kills the same mutations the in-process version did (a
    module-scope ``import triton``; ``_KERNELS_BUILT`` initialised True) without
    depending on what ran earlier in this session.
    """
    assert "ommx_gpu_serve.attention.paged_decode" in sys.modules

    # <repo>/ommx_gpu_serve/tests/this_file.py -> <repo>  (same walk as conftest.py)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    # Only PYTHONPATH is forced: the probe must import THIS checkout, but it must keep
    # the rest of the interpreter's search path (torch lives in site-packages, and
    # scrubbing it turns "no triton" into "no torch either" — a green-looking probe that
    # never reached the module under test).
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "import ommx_gpu_serve.attention.paged_decode as pd\n"
         "print('triton' in sys.modules, pd._KERNELS_BUILT)\n"],
        cwd=repo_root, env=env, capture_output=True, text=True)
    assert probe.returncode == 0, (
        "a clean interpreter could not import paged_decode at all:\n" + probe.stderr)
    assert probe.stdout.strip() == "False False", (
        f"clean-interpreter probe said (triton_imported, kernels_built) = "
        f"{probe.stdout.strip()!r}, want 'False False'. The lazy _require_triton / "
        "_build_kernels guard has regressed and this file's no-GPU premise no longer "
        f"holds.\nstderr:\n{probe.stderr}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. the helper contract — unset / blank / malformed / negative / zero
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("knob", ALL_KNOBS)
def test_unset_returns_the_default_verbatim(knob, monkeypatch) -> None:
    """Unset -> the default, UNCLAMPED.

    The sentinel is deliberately 4242 (not 1): under the old ``max(1, int(raw))``
    body the default path returned ``int(default)`` too, so a small default cannot
    tell the two bodies apart. What this pins is that no clamp, no rounding and no
    substitution happens on the way through.
    """
    monkeypatch.delenv(knob, raising=False)
    assert pd._env_int(knob, 4242, allow_zero=_allow_zero(knob)) == 4242


@pytest.mark.parametrize("knob", ALL_KNOBS)
def test_blank_is_unset_not_malformed(knob, monkeypatch) -> None:
    """A whitespace-only value reads as UNSET -> the default. SHIPPED BEHAVIOUR.

    The task brief lists ``" "`` among the malformed values that must raise; the
    shipped ``_env_int`` strips first and returns the default, and the launcher's
    ``OMMX_V4_NUM_STAGES`` branch depends on exactly that (a bare truthiness test
    there once made ``"  "`` count as an explicit pin and silently dropped the
    per-arch cp.async stage cap). Changing it is out of this file's scope, so the
    divergence is PINNED AND NAMED instead of being left uncovered.
    """
    for blank in ("", " ", "\t", "  \n "):
        monkeypatch.setenv(knob, blank)
        assert pd._env_int(knob, 7, allow_zero=_allow_zero(knob)) == 7, (
            f"{knob}={blank!r} no longer reads as unset. If that is intentional, the "
            "OMMX_V4_NUM_STAGES explicit-pin branch in "
            "ommx_paged_decode_attention_canonical must change in the same commit — "
            "it uses .strip() precisely so blank means the same thing on both sides.")


@pytest.mark.parametrize("knob", ALL_KNOBS)
@pytest.mark.parametrize("bad", ["auto", "3s", "1.5", "0x10", "None", "8,16", "+-1"])
def test_malformed_raises_naming_the_knob_and_a_fix(knob, bad, monkeypatch) -> None:
    """Malformed -> ValueError whose message names the VARIABLE and a fix.

    Naming matters operationally: this is read out of a vLLM worker's stderr, where a
    bare ``invalid literal for int()`` says nothing about which of nine knobs was
    typo'd. ``"1.5"`` is in the list on purpose — ``int("1.5")`` raises, so a knob
    silently truncating to 1 would be a regression, not a convenience.
    """
    monkeypatch.setenv(knob, bad)
    with pytest.raises(ValueError) as ei:
        pd._env_int(knob, 4, allow_zero=_allow_zero(knob))
    msg = str(ei.value)
    assert knob in msg, f"error message does not name the knob: {msg!r}"
    assert repr(bad) in msg or bad in msg, f"message does not quote the value: {msg!r}"
    assert "Fix:" in msg, f"message gives the operator no action: {msg!r}"


@pytest.mark.parametrize("knob", ALL_KNOBS)
@pytest.mark.parametrize("neg", ["-1", "-4", "-128"])
def test_negative_raises_for_every_knob(knob, neg, monkeypatch) -> None:
    """Negative -> ValueError, INCLUDING the ``allow_zero`` knobs.

    ``allow_zero`` lowers the floor from 1 to 0; it does not open the door to -1. A
    negative threshold is a typo, not an intent, and under the old clamp
    ``OMMX_ATTN_BLOCK_H=-4`` silently became 1.
    """
    monkeypatch.setenv(knob, neg)
    with pytest.raises(ValueError) as ei:
        pd._env_int(knob, 4, allow_zero=_allow_zero(knob))
    msg = str(ei.value)
    assert knob in msg and "minimum" in msg, msg


@pytest.mark.parametrize("knob", ZERO_REFUSED)
def test_zero_is_refused_for_launch_geometry_knobs(knob, monkeypatch) -> None:
    """0 -> ValueError for the six knobs that select a launch geometry.

    MEASURED against the released body (see ``_env_int``'s docstring):
    ``OMMX_ATTN_NUM_KV_SPLITS=0`` used to produce a ONE-split launch. The operator
    typed one geometry and silently got another; a crash they can read beats a tuned
    point nobody asked for.
    """
    monkeypatch.setenv(knob, "0")
    with pytest.raises(ValueError) as ei:
        pd._env_int(knob, 4)
    assert "not a value" in str(ei.value), str(ei.value)


@pytest.mark.parametrize("knob", ZERO_OK)
def test_zero_is_accepted_where_zero_is_meaningful(knob, monkeypatch) -> None:
    """0 -> 0 for the three knobs where 0 IS a value. NOT rewritten to 1.

    This is the half of the contract that ``allow_zero=`` exists for: under the clamp,
    typing the DOCUMENTED default ``OMMX_ATTN_OUTLIER_WARPS=0`` pinned ``num_warps=1``
    and produced a wrong MEASUREMENT rather than an error.
    """
    monkeypatch.setenv(knob, "0")
    assert pd._env_int(knob, 4, allow_zero=True) == 0


@pytest.mark.parametrize("knob", ZERO_OK)
def test_zero_without_allow_zero_still_refused(knob, monkeypatch) -> None:
    """The kwarg — not the knob NAME — is what permits 0.

    Pins that ``_env_int`` has no per-name special case hidden inside it: the same
    three knobs are refused when the caller does not pass ``allow_zero=True``. That
    is what makes :func:`test_allow_zero_is_set_exactly_where_zero_is_meaningful`
    load-bearing rather than decorative.
    """
    monkeypatch.setenv(knob, "0")
    with pytest.raises(ValueError):
        pd._env_int(knob, 4)


@pytest.mark.parametrize("knob", ALL_KNOBS)
def test_wellformed_values_pass_through_unclamped(knob, monkeypatch) -> None:
    """A valid value is returned verbatim — no ceiling, no rounding, no snapping."""
    monkeypatch.setenv(knob, " 37 ")     # surrounding space is stripped, not rejected
    assert pd._env_int(knob, 4, allow_zero=_allow_zero(knob)) == 37


# ══════════════════════════════════════════════════════════════════════════════
# 2. the CALL SITES — the half MUTATION B walked through
# ══════════════════════════════════════════════════════════════════════════════

def _env_int_call_sites():
    """Every ``_env_int("LITERAL", default, ...)`` call in ``paged_decode``.

    Parsed from the module's own source. Returns ``{knob: [(lineno, allow_zero,
    default_node), ...]}``. A call whose first argument is not a string literal would
    make this analysis blind, so it is reported rather than skipped.
    """
    src = inspect.getsource(pd)
    tree = ast.parse(src)
    sites: dict = {}
    dynamic = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_env_int"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            dynamic.append(node.lineno)
            continue
        allow_zero = None
        for kw in node.keywords:
            if kw.arg == "allow_zero":
                allow_zero = ast.literal_eval(kw.value)
        default = node.args[1] if len(node.args) > 1 else None
        sites.setdefault(node.args[0].value, []).append(
            (node.lineno, allow_zero, default))
    assert not dynamic, (
        "_env_int is called with a non-literal knob name at line(s) "
        f"{dynamic}; this file's call-site analysis cannot see that knob's "
        "allow_zero, so the gate would be blind exactly where it matters")
    return sites


def test_call_site_inventory_is_exactly_the_nine_documented_knobs() -> None:
    """A NEW knob must be classified here on purpose, not slip past every assertion.

    Same discipline as ``test_preflight_guards._tags``: an unrecognised entry fails
    loudly. Without this, adding a tenth knob with a wrong ``allow_zero`` would be
    invisible to the test below, which only iterates over the nine it knows.
    """
    assert set(_env_int_call_sites()) == set(ALL_KNOBS)


def test_allow_zero_is_set_exactly_where_zero_is_meaningful() -> None:
    """THE MUTATION-B GATE. ``allow_zero=True`` at the three call sites, nowhere else.

    Stripping ``allow_zero=True`` from ``OMMX_ATTN_OUTLIER_WARPS`` (line ~3075, inside
    the Triton launcher) has NO behaviour this host can observe — there is no CPU path
    that reaches it. Stripping it from the two ``OMMX_OVERSPLIT_*`` call sites does,
    and is caught dynamically below as well. This static check covers all three
    uniformly, so no call site's classification depends on whether a GPU was available
    to the test runner.

    Adding ``allow_zero=True`` to a launch-geometry knob is caught in the same pass:
    that direction re-opens the law-#11 trap (a 0-sized grid accepted silently).
    """
    for knob, sites in sorted(_env_int_call_sites().items()):
        want = knob in ZERO_OK
        for lineno, allow_zero, _default in sites:
            got = bool(allow_zero)
            assert got == want, (
                f"paged_decode.py:{lineno}: _env_int({knob!r}, ...) has "
                f"allow_zero={allow_zero!r}; expected {want}. "
                + ("0 is meaningful for this knob (a threshold, or the documented "
                   "'use the ladder' default) and must not be rewritten to 1."
                   if want else
                   "This knob selects a launch geometry; 0 is not a grid and must "
                   "raise, not be accepted."))


@pytest.mark.parametrize("knob,allowed", sorted(LITERAL_DEFAULTS.items()))
def test_literal_call_site_defaults_are_the_documented_ones(knob, allowed) -> None:
    """"Unset -> the DOCUMENTED default", for the call sites whose default is literal.

    The helper test above proves ``_env_int`` hands back the default it was given;
    this proves which default that is. It is the only reachable way to pin
    ``OMMX_ATTN_OUTLIER_WARPS``'s documented default of 0 and
    ``OMMX_MERGE_WIDE_NUM_WARPS``'s 2 — both are read only inside the launcher.

    Note ``OMMX_ATTN_OUTLIER_WARPS``'s default is 0 AND its call site passes
    ``allow_zero=True``: the default and the accepted-value set have to agree, or the
    documented default would be a value the knob refuses when typed explicitly.
    """
    got = set()
    for lineno, _az, default in _env_int_call_sites()[knob]:
        assert default is not None, f"paged_decode.py:{lineno}: {knob} has no default"
        if isinstance(default, ast.Constant) and isinstance(default.value, int):
            got.add(default.value)
        else:
            pytest.fail(f"paged_decode.py:{lineno}: {knob}'s default became a computed "
                        f"expression ({ast.dump(default)[:60]}...). Move it into "
                        "LITERAL_DEFAULTS' companion check or restore the literal.")
    assert got == allowed, f"{knob} literal defaults {sorted(got)} != {sorted(allowed)}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. OMMX_ATTN_NUM_KV_SPLITS — the two readers of one variable
# ══════════════════════════════════════════════════════════════════════════════
#
# THE ADJUDICATION FINDING. This variable is read at TWO places: the ladder (via
# ``_env_kv_splits``) and the explicit override in ``_auto_num_kv_splits``. The
# override kept its OWN ``max(1, int(env_override))`` copy of the clamp — so
# ``OMMX_ATTN_NUM_KV_SPLITS=0`` raised on one path and silently became 1 on the
# other, and the silent one WINS whenever the variable is set. The strict path was
# unreachable from the knob's primary entry point. Both directions are pinned below.

@pytest.mark.parametrize("bad", ["0", "-1", "auto"])
def test_num_kv_splits_override_refuses_bad_values(bad, monkeypatch) -> None:
    """``_auto_num_kv_splits`` — the entry point the launcher calls — must RAISE.

    This is the branch that WINS whenever the variable is set. Under the reverted
    body every one of these returns 1 (``max(1, int("0"))``, ``max(1, int("-1"))``)
    or raises an unnamed ``invalid literal`` — a one-split launch the operator never
    asked for, silently replacing the geometry they typed.
    """
    monkeypatch.setenv("OMMX_ATTN_NUM_KV_SPLITS", bad)
    with pytest.raises(ValueError) as ei:
        pd._auto_num_kv_splits(32768, 1)
    assert "OMMX_ATTN_NUM_KV_SPLITS" in str(ei.value)


@pytest.mark.parametrize("bad", ["0", "-1", "auto"])
def test_num_kv_splits_ladder_refuses_bad_values(bad, monkeypatch) -> None:
    """The OTHER reader (``_env_kv_splits`` inside the static ladder) refuses too.

    Kept separate from the override test on purpose: the whole defect was that these
    two readers of ONE variable name disagreed, so a single test covering whichever
    one happens to run first would restate the bug rather than catch it.
    """
    monkeypatch.setenv("OMMX_ATTN_NUM_KV_SPLITS", bad)
    with pytest.raises(ValueError) as ei:
        pd._auto_num_kv_splits_static(32768, 1)
    assert "OMMX_ATTN_NUM_KV_SPLITS" in str(ei.value)


def test_num_kv_splits_override_routes_through_env_int(monkeypatch) -> None:
    """The override's RETURN VALUE comes out of ``_env_int`` — not a private clamp.

    Proven by substitution, which is the only way to tell "calls ``_env_int`` and
    returns its result" apart from "computes the same number some other way": the
    stub returns a sentinel no clamp could produce. Under the reverted
    ``max(1, int(env_override))`` the call returns 7 (the env value) and this fails.
    """
    calls = []
    real = pd._env_int

    def _recording(name, default, *, allow_zero=False):
        calls.append((name, default, allow_zero))
        if name == "OMMX_ATTN_NUM_KV_SPLITS":
            return 4242
        return real(name, default, allow_zero=allow_zero)

    monkeypatch.setattr(pd, "_env_int", _recording)
    monkeypatch.setenv("OMMX_ATTN_NUM_KV_SPLITS", "7")
    assert pd._auto_num_kv_splits(32768, 1) == 4242
    assert calls and calls[-1][0] == "OMMX_ATTN_NUM_KV_SPLITS", (
        "the override branch did not end in an _env_int('OMMX_ATTN_NUM_KV_SPLITS', ...) "
        f"call; recorded: {calls}")
    assert calls[-1][2] is False, (
        "the override must NOT pass allow_zero=True: 0 splits is not a grid")


def test_num_kv_splits_override_wins_over_the_ladder(monkeypatch) -> None:
    """A valid pin is honoured verbatim, and the ladder does not clamp it.

    Documents WHY the two readers had to be reconciled rather than one deleted: the
    override returns before the ladder, so a value outside the ladder's window (128
    at a bucket whose window tops out at 16) survives.
    """
    monkeypatch.setenv("OMMX_ATTN_NUM_KV_SPLITS", "128")
    assert pd._auto_num_kv_splits(1024, 1) == 128
    assert pd._auto_num_kv_splits_static(1024, 1) == 16     # ladder DOES clamp


def test_num_kv_splits_unset_gives_the_documented_ladder() -> None:
    """Unset -> the sweep-tuned ladder, untouched by any of this."""
    assert pd._auto_num_kv_splits_static(1024, 1) == 1
    assert pd._auto_num_kv_splits_static(8192, 1) == 4
    assert pd._auto_num_kv_splits_static(32768, 1) == 16
    assert pd._auto_num_kv_splits_static(4096, 8) == 16
    assert pd._auto_num_kv_splits_static(65536, 8) == 32


# ══════════════════════════════════════════════════════════════════════════════
# 4. OMMX_OVERSPLIT_BATCH / OMMX_OVERSPLIT_CTX — allow_zero, proven by behaviour
# ══════════════════════════════════════════════════════════════════════════════
#
# These two are the ONLY allow_zero call sites reachable without a device, so they
# carry the dynamic half of the MUTATION-B gate.
#
# Reaching them needs ``occ > 0``, which needs a non-zero SM count. ``_sm_count_for_
# device`` is a pure best-effort probe returning 0 for any non-CUDA device — exactly
# the seam that makes the surrounding arithmetic testable on a CPU. It is replaced by
# a constant (108 = A100), NOT by a fake torch device; nothing CUDA is touched.
#
# The cell is chosen so the down-merge branch is LIVE and the knobs are
# DISCRIMINATIVE: at seq=4096, batch=8, sm=108 the static ladder says 16 while the
# occupancy estimate says 3, and the branch requires ``batch >= OVERSPLIT_BATCH`` and
# ``seq >= OVERSPLIT_CTX``. Merged -> 8 (the window floor); not merged -> 16.

_CELL = dict(seq_len=4096, batch_size=8)
_MERGED, _NOT_MERGED = 8, 16


@pytest.fixture
def a100_sm(monkeypatch):
    """Pin the SM count so the occupancy branch is reachable with no CUDA device."""
    monkeypatch.setattr(pd, "_sm_count_for_device", lambda device=None: 108)


def test_oversplit_defaults_take_the_down_merge(a100_sm) -> None:
    """Unset -> defaults 8 / 1024 -> this cell merges. The baseline for §4."""
    assert pd._auto_num_kv_splits(**_CELL) == _MERGED


@pytest.mark.parametrize("knob", ["OMMX_OVERSPLIT_BATCH", "OMMX_OVERSPLIT_CTX"])
def test_oversplit_zero_is_accepted_and_means_no_threshold(knob, a100_sm,
                                                           monkeypatch) -> None:
    """0 -> "no threshold". MUST NOT RAISE, and must not be rewritten to 1.

    THE DYNAMIC MUTATION-B GATE. Strip ``allow_zero=True`` from this call site and
    ``_env_int`` raises here instead of returning 0, so this errors out. The two are
    thresholds compared with ``>=``, and every real launch has batch >= 1 and
    seq >= 1, so 0 and 1 select the same set — which is exactly why the old clamp was
    never WRONG here and why refusing 0 would break a working A/B config for no
    correctness gain.
    """
    monkeypatch.setenv(knob, "0")
    assert pd._auto_num_kv_splits(**_CELL) == _MERGED


@pytest.mark.parametrize("knob,value", [("OMMX_OVERSPLIT_BATCH", "9"),
                                        ("OMMX_OVERSPLIT_CTX", "8192")])
def test_oversplit_thresholds_actually_gate_the_branch(knob, value, a100_sm,
                                                       monkeypatch) -> None:
    """The knobs MOVE the result — so §4 cannot pass on a knob that is read and ignored.

    Raising either threshold just past this cell (batch 8 < 9; ctx 4096 < 8192) turns
    the down-merge off and the static ladder's 16 comes back. Without this, a mutation
    that hard-coded the thresholds would leave every other assertion here green.
    """
    monkeypatch.setenv(knob, value)
    assert pd._auto_num_kv_splits(**_CELL) == _NOT_MERGED


@pytest.mark.parametrize("knob", ["OMMX_OVERSPLIT_BATCH", "OMMX_OVERSPLIT_CTX"])
def test_oversplit_negative_raises_through_the_call_site(knob, a100_sm,
                                                         monkeypatch) -> None:
    """``allow_zero`` lowers the floor to 0, not below — checked where it is READ."""
    monkeypatch.setenv(knob, "-1")
    with pytest.raises(ValueError) as ei:
        pd._auto_num_kv_splits(**_CELL)
    assert knob in str(ei.value) and "0 IS meaningful" in str(ei.value)


@pytest.mark.parametrize("knob", ["OMMX_OVERSPLIT_BATCH", "OMMX_OVERSPLIT_CTX"])
def test_oversplit_malformed_raises_through_the_call_site(knob, a100_sm,
                                                          monkeypatch) -> None:
    monkeypatch.setenv(knob, "auto")
    with pytest.raises(ValueError) as ei:
        pd._auto_num_kv_splits(**_CELL)
    assert knob in str(ei.value)


# ══════════════════════════════════════════════════════════════════════════════
# 5. OMMX_V4_NUM_STAGES / OMMX_V4_NUM_WARPS — read by CPU-reachable ladders
# ══════════════════════════════════════════════════════════════════════════════

def test_stages_and_warps_defaults() -> None:
    """Unset -> the documented ladders (s=2 plateau at seq>=4096; 4-warp winner)."""
    assert pd._auto_num_stages_decode(8192, 1) == 2
    assert pd._auto_num_stages(512, 4) == 4
    assert pd._auto_num_stages(512, 3) == 2
    assert pd._auto_num_stages(2048, 8) == 3
    assert pd._auto_num_warps_decode(256, 1) == 8
    assert pd._auto_num_warps_decode(4096, 1) == 4


@pytest.mark.parametrize("value,expect", [("1", 1), ("5", 5)])
def test_stages_pin_is_honoured_verbatim(value, expect, monkeypatch) -> None:
    """An explicit pin replaces the ladder on EVERY rung — no clamp, no ceiling."""
    monkeypatch.setenv("OMMX_V4_NUM_STAGES", value)
    assert pd._auto_num_stages_decode(8192, 1) == expect
    assert pd._auto_num_stages(512, 4) == expect


@pytest.mark.parametrize("bad", ["0", "-2", "auto"])
def test_stages_bad_values_raise_at_every_rung(bad, monkeypatch) -> None:
    """0 stages is not a pipeline. Checked on two different ladder rungs, because
    each rung is a SEPARATE ``_env_int`` call site and a partial revert would leave
    the others green."""
    monkeypatch.setenv("OMMX_V4_NUM_STAGES", bad)
    for fn, args in ((pd._auto_num_stages_decode, (8192, 1)),
                     (pd._auto_num_stages, (512, 4)),
                     (pd._auto_num_stages, (2048, 8))):
        with pytest.raises(ValueError) as ei:
            fn(*args)
        assert "OMMX_V4_NUM_STAGES" in str(ei.value)


@pytest.mark.parametrize("bad", ["0", "-2", "auto"])
def test_warps_bad_values_raise_at_every_rung(bad, monkeypatch) -> None:
    """0 warps is not a block. Both rungs of the warp ladder."""
    monkeypatch.setenv("OMMX_V4_NUM_WARPS", bad)
    for args in ((256, 1), (4096, 1)):
        with pytest.raises(ValueError) as ei:
            pd._auto_num_warps_decode(*args)
        assert "OMMX_V4_NUM_WARPS" in str(ei.value)


def test_warps_pin_is_honoured_verbatim(monkeypatch) -> None:
    monkeypatch.setenv("OMMX_V4_NUM_WARPS", "16")
    assert pd._auto_num_warps_decode(256, 1) == 16
    assert pd._auto_num_warps_decode(4096, 1) == 16


# ══════════════════════════════════════════════════════════════════════════════
# 6. the launcher-only knobs — helper contract + call-site shape
# ══════════════════════════════════════════════════════════════════════════════
#
# OMMX_ATTN_BLOCK_H, OMMX_ATTN_OUTLIER_WARPS, OMMX_MERGE_WIDE_BLOCK_DV and
# OMMX_MERGE_WIDE_NUM_WARPS are read ONLY inside
# ``ommx_paged_decode_attention_canonical``, which needs real CUDA tensors and a
# Triton JIT. There is no honest way to execute those reads here, and a gpu-marked
# test would SKIP on this host — i.e. prove nothing. They are therefore covered by
# the union of §1 (the helper contract, parametrized over all nine knobs) and §2
# (the AST call-site gate, which is what a stripped ``allow_zero=True`` fails).
#
# The one thing left to pin is that they are still read through ``_env_int`` at all —
# a call site rewritten to ``int(os.environ[...])`` would satisfy both of the above
# while re-opening every trap they exist to close.

@pytest.mark.parametrize("knob", ["OMMX_ATTN_BLOCK_H", "OMMX_ATTN_OUTLIER_WARPS",
                                  "OMMX_MERGE_WIDE_BLOCK_DV",
                                  "OMMX_MERGE_WIDE_NUM_WARPS"])
def test_launcher_knobs_are_read_only_through_env_int(knob) -> None:
    """Each launcher-only knob appears in the source EXACTLY as an ``_env_int`` arg.

    Counts raw ``os.environ`` mentions of the knob and requires zero: the strict read
    cannot be bypassed the way ``_auto_num_kv_splits``'s override once bypassed it.
    """
    src = inspect.getsource(pd)
    tree = ast.parse(src)
    env_int_lines = {ln for ln, _az, _d in _env_int_call_sites()[knob]}
    mentions = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == knob
    ]
    stray = sorted(set(mentions) - env_int_lines)
    assert not stray, (
        f"{knob} is referenced outside an _env_int() call at paged_decode.py line(s) "
        f"{stray}. A second reader of one variable name is the exact defect that made "
        "OMMX_ATTN_NUM_KV_SPLITS mean two different things.")
