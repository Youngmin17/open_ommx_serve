# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Gate for ``integration/vllm/packed_only.py::install_packed_only_spec``'s guard.

THE GUARD IS ONE LINE AND BOTH HALVES ARE LOAD-BEARING::

    if _PATCHED or not packed_only_enabled():
        return None

``not packed_only_enabled()`` is what keeps SHADOW mode — THE DEFAULT — from ever
touching a vLLM symbol. That matters because the code just past the guard was
deliberately changed to RAISE on a missing symbol (law #5: PACKED-ONLY makes the
sidecar the only backing store, so a swallowed install failure leaves a
self-contradictory engine — an unshrunk bf16 paged cache that ``do_kv_cache_update``
never writes and ``forward()`` refuses to read). SHADOW has a valid bf16 fallback for
every step and must never be taken down by that raise.

MUTATION C — deleting ``not packed_only_enabled()`` so every default plugin load
becomes a vLLM-symbol import — SURVIVED the previous suite. It is killed here by
:func:`test_shadow_returns_none_without_touching_any_vllm_symbol`, which does not
merely assert the return value is None: it installs ``sys.modules`` entries whose
every non-dunder attribute access appends to a tripwire list and then raises, and
asserts the tripwire stayed EMPTY. "Returned None" alone cannot distinguish the two
guards; "never looked at vLLM" can.

LATCH DISCIPLINE (the other way a gate goes vacuous). ``_PATCHED`` is a module-level
latch that lives for the worker's lifetime, and ``_EVIDENCE_DONE`` /
``_SKIP_EVIDENCE_DONE`` are one-time evidence latches. Leaked across tests, a
``_PATCHED`` left True by an earlier test makes the SHADOW test above return at the
FIRST half of the guard and pass no matter what the second half says — passing for
exactly the wrong reason. :func:`_reset_packed_only_latches` clears all three around
every test (the same discipline as ``test_preflight_guards._reset_warn_latch``), and
the SHADOW test additionally ASSERTS the latch is clear before it starts, so a future
fixture regression fails loudly instead of quietly disarming the gate.

NO vLLM, NO TRITON, NO GPU. Every vLLM symbol below is a stub installed with
``monkeypatch.setitem(sys.modules, ...)``; the stub package's ``__path__`` is empty,
so a submodule that is not explicitly installed is genuinely unimportable — which is
how the "vLLM symbols missing" direction is produced deterministically rather than by
relying on this host happening to have no vLLM.
"""
from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from ommx_gpu_serve.integration.vllm import packed_only as po

# The two symbols the patch imports, as dotted module + attribute names. Kept as data
# so the stubs and the tripwire cannot drift apart from each other.
_ATTENTION_MOD = "vllm.model_executor.layers.attention.attention"
_SPEC_MOD = "vllm.v1.kv_cache_interface"
_PKG_CHAIN = (
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.attention",
    _ATTENTION_MOD,
    "vllm.v1",
    _SPEC_MOD,
)

#: Every spelling ``packed_only_enabled`` treats as OFF, plus "not set at all".
#: ``_env`` strips and lowercases, so the mixed-case and padded forms belong here too.
SHADOW_SPELLINGS = [None, "", "   ", "0", "false", "FALSE", "off", "no", " No "]

#: ...and the ones that mean ON. Anything not in the off-set enables PACKED-ONLY.
PACKED_SPELLINGS = ["1", "true", "yes", "on", "2"]


# ══════════════════════════════════════════════════════════════════════════════
# fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _reset_packed_only_latches():
    """Clear the three module-level latches around EVERY test in this file.

    ``_PATCHED`` survives for the life of the worker process by design (it is what
    makes ``install_packed_only_spec`` idempotent across vLLM's per-worker plugin
    loads). Inside one pytest process that same longevity is a cross-test leak, and a
    leaked True disarms the guard's second half — the exact thing under test. The
    evidence latches are cleared for the same reason: a one-time line already emitted
    by an earlier test would make a later assertion about the fire file read as "no
    evidence" when the real answer is "evidence, but you missed it".
    """
    def _clear():
        po._PATCHED = False
        po._EVIDENCE_DONE[0] = False
        po._SKIP_EVIDENCE_DONE[0] = False
    _clear()
    yield
    _clear()


@pytest.fixture
def fire_file(tmp_path, monkeypatch):
    """Point the route-evidence file at tmp_path.

    Without this the evidence lands in ``/tmp/ommx_route_fired.log`` — a real
    operator artefact shared with every other process on the box. Returns a reader so
    an assertion can look at what was actually written.
    """
    path = tmp_path / "ommx_route_fired.log"
    monkeypatch.setenv("OMMX_FIRE_FILE", str(path))
    return lambda: path.read_text() if path.exists() else ""


@pytest.fixture
def canonical_recipe(monkeypatch):
    """Pin the CANONICAL PUBLISHED RECIPE so the shrink figure is a fixed number.

    ``conftest`` scrubs every ``OMMX_*`` var before each test, so without this the
    accounting would run on the module defaults (k=3, pow2 off) — a different recipe
    from the one every published measurement used. These four knobs are the recipe
    named in ``packed_only``'s module docstring; they give 8.750 bit per (K,V)
    element pair, which is the figure the head_size assertions below are derived
    from.
    """
    monkeypatch.setenv("OMMX_ATTN_OUTLIERS", "6")
    monkeypatch.setenv("OMMX_ATTN_POW2", "1")
    monkeypatch.setenv("OMMX_KV_GROUP_TOKENS", "32")
    monkeypatch.setenv("OMMX_KV_GROUP_CHANNELS", "32")


# ── stub vLLM trees ─────────────────────────────────────────────────────────────

class _TripwireModule(types.ModuleType):
    """A module whose every non-dunder attribute access is RECORDED and then EXPLODES.

    This is the instrument that makes "did not touch a vLLM symbol" checkable rather
    than assumed. Dunder lookups are allowed to fail normally (``AttributeError``)
    because the import machinery itself probes ``__path__`` / ``__spec__`` / ... and
    those probes are not the module under test looking anything up.
    """

    def __init__(self, name: str, tripped: list) -> None:
        super().__init__(name)
        self.__path__ = []          # make it a package so submodule resolution works
        self._tripped = tripped

    def __getattr__(self, item: str):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        self._tripped.append(f"{self.__name__}.{item}")
        raise AssertionError(
            f"SHADOW mode reached into the vLLM symbol {self.__name__}.{item}. The "
            "`not packed_only_enabled()` half of install_packed_only_spec's guard is "
            "what prevents that; without it a vLLM tree missing this symbol takes down "
            "the DEFAULT serving mode, which has a valid bf16 fallback for every step.")


@pytest.fixture
def tripwire_vllm(monkeypatch):
    """Install a vLLM tree that explodes on ANY symbol access. Returns the record list."""
    tripped: list = []
    for name in _PKG_CHAIN:
        monkeypatch.setitem(sys.modules, name, _TripwireModule(name, tripped))
    return tripped


@pytest.fixture
def empty_vllm(monkeypatch):
    """Install a bare ``vllm`` package with NO submodules.

    ``__path__ = []`` means ``vllm.model_executor...`` is genuinely not importable, so
    the "this vLLM does not expose the v1 attention symbols" case is produced by
    construction. Deliberately not "let the host's missing vLLM do it": that would make
    the test pass for a reason that disappears the day a GPU host runs the suite.
    """
    mod = types.ModuleType("vllm")
    mod.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", mod)
    # Any previously-imported submodule must go too, or the stale entry satisfies the
    # import and this fixture silently becomes a no-op.
    for name in list(sys.modules):
        if name.startswith("vllm."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return mod


@dataclasses.dataclass
class _StubFullAttentionSpec:
    """Stands in for ``vllm.v1.kv_cache_interface.FullAttentionSpec``.

    A real dataclass because the patch rewrites the spec with ``dataclasses.replace``;
    only the three fields the patch reads/writes are modelled.
    """
    dtype: object
    head_size: int
    head_size_v: int


@dataclasses.dataclass
class _StubOtherSpec:
    """A NON-full-attention spec (sliding-window / MLA / TQ stand-in).

    IT IS SHRINKABLE ON PURPOSE. It carries a real ``torch.bfloat16`` dtype AND a
    ``head_size_v`` field, so it satisfies the dtype gate and survives
    ``dataclasses.replace(spec, head_size=..., head_size_v=...)``. That leaves the
    ``isinstance(spec, FullAttentionSpec)`` check as the ONLY thing standing between
    this spec and a shrink — which is what makes
    :func:`test_patched_spec_leaves_foreign_specs_alone` a gate on that check.

    ADVERSARIAL NOTE (why the fields are specified this precisely): with
    ``dtype="bf16"`` (a plain string) the dtype gate rejected it first, so deleting the
    isinstance check entirely left the whole suite GREEN — a mutation that would shrink
    a sliding-window cache vLLM really does read, i.e. the one direction of this patch
    that is NOT fail-safe, walked straight through.
    """
    dtype: object
    head_size: int
    head_size_v: int


@pytest.fixture
def working_vllm(monkeypatch):
    """A vLLM tree that HAS both symbols, so the success path can be exercised.

    Returns the stub ``Attention`` class. Its ``get_kv_cache_spec`` returns whatever
    was put in ``Attention.next_spec``, which is how the tests below feed the patched
    function a FullAttentionSpec, a foreign spec, or an exploding one.
    """
    for name in _PKG_CHAIN:
        mod = types.ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)

    class Attention:
        next_spec: object = None

        def get_kv_cache_spec(self, vllm_config):
            spec = Attention.next_spec
            if isinstance(spec, BaseException):
                raise spec
            return spec

    sys.modules[_ATTENTION_MOD].Attention = Attention
    sys.modules[_SPEC_MOD].FullAttentionSpec = _StubFullAttentionSpec

    # vllm.logger.init_logger is best-effort inside the evidence helpers; give it a
    # real (recording) implementation so the log branch is executed rather than
    # swallowed by its `except Exception: pass`.
    logger_mod = types.ModuleType("vllm.logger")
    records: list = []

    class _Rec:
        def info(self, fmt, *a):
            records.append(("info", fmt % a if a else fmt))

        def warning(self, fmt, *a):
            records.append(("warning", fmt % a if a else fmt))

    logger_mod.init_logger = lambda name: _Rec()
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)
    Attention.log_records = records
    return Attention


# ══════════════════════════════════════════════════════════════════════════════
# 1. SHADOW (the default) — returns None AND never touches vLLM
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("spelling", SHADOW_SPELLINGS)
def test_shadow_returns_none_without_touching_any_vllm_symbol(
        spelling, tripwire_vllm, monkeypatch) -> None:
    """THE MUTATION-C GATE.

    Delete ``not packed_only_enabled()`` from the guard and this test does not merely
    fail an equality — it raises out of the tripwire, because the default plugin load
    now performs a vLLM-symbol import. Every off-ish spelling is covered so the guard
    cannot be narrowed to a literal ``"0"`` check either.
    """
    assert po._PATCHED is False, (
        "the _PATCHED latch leaked in from an earlier test. It would short-circuit "
        "this test at the FIRST half of the guard, making it pass without ever "
        "exercising the half under test — see _reset_packed_only_latches.")
    if spelling is None:
        monkeypatch.delenv("OMMX_KV_PACKED_ONLY", raising=False)
    else:
        monkeypatch.setenv("OMMX_KV_PACKED_ONLY", spelling)

    assert po.install_packed_only_spec() is None
    assert tripwire_vllm == [], (
        f"SHADOW mode touched vLLM symbol(s): {tripwire_vllm}")
    assert po._PATCHED is False, "SHADOW must not set the patched latch"


@pytest.mark.parametrize("spelling", SHADOW_SPELLINGS)
def test_shadow_predicate_agrees_with_the_plugin_side_predicate(spelling,
                                                                monkeypatch) -> None:
    """``packed_only_enabled`` and ``plugin._packed_only_requested`` must agree.

    They are two hand-kept copies of one env read (``plugin.py`` deliberately avoids
    importing ``packed_only`` so it can still report on a failure to import it). If
    they ever disagree, ``plugin.register`` would re-raise a module-import failure for
    an operator whom ``install_packed_only_spec`` then treats as SHADOW — or, worse,
    the reverse.
    """
    from ommx_gpu_serve.integration.vllm import plugin as pl
    if spelling is None:
        monkeypatch.delenv("OMMX_KV_PACKED_ONLY", raising=False)
    else:
        monkeypatch.setenv("OMMX_KV_PACKED_ONLY", spelling)
    assert po.packed_only_enabled() is False
    assert pl._packed_only_requested() is False


@pytest.mark.parametrize("spelling", PACKED_SPELLINGS)
def test_packed_predicate_agrees_with_the_plugin_side_predicate(spelling,
                                                               monkeypatch) -> None:
    """The ON direction of the same two-copy agreement."""
    from ommx_gpu_serve.integration.vllm import plugin as pl
    monkeypatch.setenv("OMMX_KV_PACKED_ONLY", spelling)
    assert po.packed_only_enabled() is True
    assert pl._packed_only_requested() is True


# ══════════════════════════════════════════════════════════════════════════════
# 2. PACKED-ONLY with the vLLM symbols missing -> RAISES, naming the cause
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("spelling", PACKED_SPELLINGS)
def test_packed_only_raises_when_the_vllm_module_is_missing(
        spelling, empty_vllm, monkeypatch) -> None:
    """An operator who ASKED for PACKED-ONLY is stopped, loudly, at plugin load.

    Returning None here is what the previous code did, and it left
    ``backend._PACKED_ONLY`` True (resolved from the env, independently of this patch)
    against an UNSHRUNK bf16 paged cache that is never written and never read.
    """
    monkeypatch.setenv("OMMX_KV_PACKED_ONLY", spelling)
    with pytest.raises(RuntimeError) as ei:
        po.install_packed_only_spec()
    msg = str(ei.value)
    # the knob, the cause, the two symbols, and an action — all four or the operator
    # cannot act on it from a worker's stderr.
    assert "OMMX_KV_PACKED_ONLY" in msg
    assert "ModuleNotFoundError" in msg or "ImportError" in msg, msg
    assert _ATTENTION_MOD in msg and _SPEC_MOD in msg, msg
    assert "FIX:" in msg, msg
    assert po._PATCHED is False, "a failed install must not claim to have patched"


def test_packed_only_raises_when_the_symbol_is_absent_from_a_present_module(
        monkeypatch) -> None:
    """The realistic shape of the failure: the modules exist, the SYMBOL does not.

    A vLLM tree that moved ``FullAttentionSpec`` is not a missing package — it is a
    present package with a missing attribute — and that is the case the module
    docstring promises to catch ("A vLLM tree missing the v1 symbols therefore still
    runs SHADOW; only an operator who asked for PACKED-ONLY is stopped").
    """
    for name in _PKG_CHAIN:
        mod = types.ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
    # Attention exists; FullAttentionSpec deliberately does not.
    sys.modules[_ATTENTION_MOD].Attention = type("Attention", (), {})
    monkeypatch.setenv("OMMX_KV_PACKED_ONLY", "1")
    with pytest.raises(RuntimeError) as ei:
        po.install_packed_only_spec()
    assert "FullAttentionSpec" in str(ei.value)
    assert po._PATCHED is False


def test_the_raise_is_not_unconditional(empty_vllm, monkeypatch) -> None:
    """Same broken vLLM tree, env unset -> None. The asymmetry IS the design.

    Pins the two directions against ONE fixture, so a change that made the raise
    unconditional (or the return unconditional) cannot satisfy both halves of this
    file by satisfying two differently-configured worlds.
    """
    monkeypatch.setenv("OMMX_KV_PACKED_ONLY", "1")
    with pytest.raises(RuntimeError):
        po.install_packed_only_spec()
    po._PATCHED = False                      # the failed attempt left it False anyway
    monkeypatch.delenv("OMMX_KV_PACKED_ONLY", raising=False)
    assert po.install_packed_only_spec() is None


# ══════════════════════════════════════════════════════════════════════════════
# 3. the success path + idempotence
# ══════════════════════════════════════════════════════════════════════════════

def test_install_patches_get_kv_cache_spec(working_vllm, monkeypatch) -> None:
    """PACKED-ONLY with the symbols present actually replaces the method.

    Without this, the idempotence test below would be satisfied by a function that
    never patches at all — "called twice, still the original" is trivially true when
    the first call did nothing.
    """
    monkeypatch.setenv("OMMX_KV_PACKED_ONLY", "1")
    original = working_vllm.get_kv_cache_spec
    assert po.install_packed_only_spec() is None
    assert working_vllm.get_kv_cache_spec is not original
    assert po._PATCHED is True


def test_install_is_idempotent_and_does_not_re_patch(working_vllm,
                                                     monkeypatch) -> None:
    """A second call with the env STILL SET must not wrap the wrapper.

    vLLM loads general_plugins once per worker, and a re-entrant load must not stack
    another ``_packed_only_spec`` on top: the second wrapper's ``_orig`` would be the
    first wrapper, so the shrink would be applied to an already-shrunk head_size.
    Identity — not behaviour — is asserted, because a double-wrap that happens to be
    idempotent numerically is still a latch regression.
    """
    monkeypatch.setenv("OMMX_KV_PACKED_ONLY", "1")
    assert po.install_packed_only_spec() is None
    after_first = working_vllm.get_kv_cache_spec
    assert po.install_packed_only_spec() is None
    assert working_vllm.get_kv_cache_spec is after_first, (
        "install_packed_only_spec re-patched on the second call; the `_PATCHED` half "
        "of the guard has regressed and the shrink would be applied twice")
    assert po.install_packed_only_spec() is None      # and a third time
    assert working_vllm.get_kv_cache_spec is after_first


def test_patched_spec_shrinks_the_bf16_page_budget(working_vllm, canonical_recipe,
                                                   fire_file, monkeypatch) -> None:
    """The patch does the thing it exists for, at the CANONICAL PUBLISHED RECIPE.

    head_size 128 -> 32: the canonical recipe is 8.750 bit per (K,V) element pair, so
    the exact byte-equivalent is 128 * 8.750/32 = 35.0, rounded to the nearest
    multiple of 8 -> 32. vLLM therefore budgets 128/32 = 4.00x while the planes only
    compress 3.66x — the budget is 8.6% OPTIMISTIC, and BOTH numbers are asserted to
    be present in the evidence line so the log can never be mistaken for a measured
    compression result.

    THIS IS A BYTE BUDGET, NOT A MEASURED CAPACITY. The shrunk pages are never written
    and never read (module docstring, reasons 1-4); nothing here claims otherwise.
    """
    import torch
    monkeypatch.setenv("OMMX_KV_PACKED_ONLY", "1")
    po.install_packed_only_spec()

    working_vllm.next_spec = _StubFullAttentionSpec(
        dtype=torch.bfloat16, head_size=128, head_size_v=128)
    out = working_vllm().get_kv_cache_spec(object())
    assert (out.head_size, out.head_size_v) == (32, 32)
    assert working_vllm.next_spec.head_size == 128, "the original spec was mutated"

    line = fire_file()
    assert "PACKED_ONLY_SPEC head_size 128 -> 32" in line, line
    assert "ommx_bits/elem=8.750" in line, line
    assert "planes=3.66x" in line, line
    assert any(kind == "info" and "PACKED_ONLY_SPEC" in text
               for kind, text in working_vllm.log_records), working_vllm.log_records


def test_patched_spec_leaves_foreign_specs_alone(working_vllm, canonical_recipe,
                                                 fire_file, monkeypatch) -> None:
    """Sliding-window / MLA / fp8 specs are NOT the OMMX path and pass through.

    Shrinking a spec the OMMX sidecar does not back would under-reserve a cache that
    IS read — the one direction of this patch that is not fail-safe.
    """
    monkeypatch.setenv("OMMX_KV_PACKED_ONLY", "1")
    po.install_packed_only_spec()
    import torch
    for spec in (
            # bf16 + head_size_v present: EVERYTHING except the type says "shrink me",
            # so only the isinstance(spec, FullAttentionSpec) check can save it.
            _StubOtherSpec(dtype=torch.bfloat16, head_size=128, head_size_v=128),
            # the other exclusion axis: right type, wrong dtype (quantized cache).
            _StubFullAttentionSpec(dtype="fp8", head_size=128, head_size_v=128)):
        working_vllm.next_spec = spec
        got = working_vllm().get_kv_cache_spec(object())
        assert got is spec, (
            f"{type(spec).__name__} was rewritten (head_size {spec.head_size} -> "
            f"{getattr(got, 'head_size', None)}). Shrinking a spec the OMMX sidecar "
            "does not back UNDER-reserves a cache vLLM really reads — the one "
            "direction of this patch that is not fail-safe.")
        assert spec.head_size == 128, "the original spec was mutated in place"
    assert fire_file() == "", "an untouched spec must not emit shrink evidence"


def test_patched_spec_records_a_skip_instead_of_shrinking_blind(
        working_vllm, fire_file, monkeypatch) -> None:
    """An unbuildable recipe leaves the spec at FULL bf16 and says so, once.

    This is the ONE fail-safe direction (over-reserved, never under), so it must not
    raise — but it must not be silent either, or the operator sees PACKED-ONLY with no
    ``PACKED_ONLY_SPEC`` line and no stated reason. The same geometry still raises
    loudly at pool construction, so no silently-wrong path exists.
    """
    import torch
    monkeypatch.setenv("OMMX_KV_PACKED_ONLY", "1")
    monkeypatch.setenv("OMMX_KV_GROUP_TOKENS", "31")     # not in {16,32,64,128}
    po.install_packed_only_spec()
    spec = _StubFullAttentionSpec(dtype=torch.bfloat16, head_size=128, head_size_v=128)
    working_vllm.next_spec = spec
    assert working_vllm().get_kv_cache_spec(object()) is spec
    line = fire_file()
    assert "PACKED_ONLY_SPEC_SKIPPED" in line and "UNSHRUNK" in line, line
    assert "OMMX_KV_GROUP_TOKENS" in line, line


def test_a_failing_original_spec_call_is_not_swallowed(working_vllm, canonical_recipe,
                                                       monkeypatch) -> None:
    """The patch must not convert vLLM's OWN spec failure into a skip note.

    ``_orig(self, vllm_config)`` runs OUTSIDE the patch's ``try``. If it were inside,
    a real vLLM error during KV-cache sizing would be recorded as "left UNSHRUNK" and
    the engine would carry on with a spec that was never built.
    """
    monkeypatch.setenv("OMMX_KV_PACKED_ONLY", "1")
    po.install_packed_only_spec()
    working_vllm.next_spec = KeyError("vllm exploded while building the spec")
    with pytest.raises(KeyError):
        working_vllm().get_kv_cache_spec(object())


# ══════════════════════════════════════════════════════════════════════════════
# 4. plugin._register_ommx_w_quant — note vs RE-RAISE
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY THIS SECTION LIVES IN THIS FILE. It is the third gate of the same audit
# finding — a blanket ``except Exception`` in the vLLM plugin-load path that turned a
# real, unresolvable conflict into a stderr note — and it shares this file's entire
# apparatus: stub ``sys.modules`` vLLM trees, and module-level registration latches
# that leak between tests unless they are cleared. Splitting it into a third file
# would duplicate both.
#
# THE DECISION UNDER TEST (see ``plugin._register_ommx_w_quant``'s docstring):
#
#   benign   -> note, do not raise. ``ommx_w`` stays UNREGISTERED, so vLLM rejects
#               ``--quantization ommx_w`` by name at argument-parse time. This
#               function runs in every worker of every run, including KV-only runs
#               that never touch a weight bundle, so a blanket raise is wrong.
#   collision-> RE-RAISE. A FOREIGN quantization config already owns ``"ommx_w"``.
#               The by-name rejection does NOT fire (the name resolves), so the
#               engine would start and serve somebody else's implementation under the
#               name the operator selected — a conflict settled by import order.

_QUANT_MOD = "vllm.model_executor.layers.quantization"


@pytest.fixture(autouse=True)
def _reset_linear_method_latches():
    """Clear ``linear_method``'s registration latches and class cache around each test.

    ``register_ommx_w`` sets ``_REGISTERED`` on EVERY outcome including the no-registry
    no-op, and returns early forever after. Leaked, that turns every later test in this
    section into "the latch was already set", i.e. a pass that never ran the code.
    ``_CLASSES`` is cleared for the same reason: a class built against one test's stub
    vLLM must not be handed to the next test's different stub.
    """
    from ommx_gpu_serve.integration.vllm import linear_method as lm

    def _clear():
        lm._REGISTERED = False
        lm._REGISTERED_AS = None
        lm._CLASSES.clear()
    _clear()
    yield
    _clear()


class _ForeignConfig:
    """Somebody else's quantization config, squatting on the name ``ommx_w``."""


@pytest.fixture
def quant_registry(monkeypatch):
    """A stub vLLM quantization registry with vLLM 0.21's own semantics.

    ``register_quantization_config(name)`` returns a decorator that raises
    ``ValueError`` when the name is taken — that is the behaviour
    ``register_ommx_w``'s collision branch is written against — and
    ``get_quantization_config(name)`` raises for an unknown name, which is what makes
    the plugin's registry probe return "undeterminable" rather than "no collision" in
    the benign cases.

    Returns the backing dict so a test can plant a foreign incumbent.
    """
    table: dict = {}
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers",
                 _QUANT_MOD):
        mod = types.ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)

    def register_quantization_config(name):
        def _deco(cls):
            if name in table:
                raise ValueError(f"quantization method {name} is already registered")
            table[name] = cls
            return cls
        return _deco

    def get_quantization_config(name):
        if name not in table:
            raise ValueError(f"invalid quantization method: {name}")
        return table[name]

    quant = sys.modules[_QUANT_MOD]
    quant.register_quantization_config = register_quantization_config
    quant.get_quantization_config = get_quantization_config
    return table


@pytest.fixture
def our_config_class(monkeypatch):
    """Seed ``linear_method._CLASSES`` so ``ommx_w_config_class()`` needs no vLLM bases.

    ``_build_classes`` imports vLLM's Linear/QuantizationConfig bases and defines a
    real subclass against them. Reproducing that whole surface in a stub would test
    the stub, not the classification under test, so the module's own per-process class
    CACHE is used as the seam instead. Returns the class that plays "ours".
    """
    from ommx_gpu_serve.integration.vllm import linear_method as lm

    class _OursConfig:
        pass

    monkeypatch.setitem(lm._CLASSES, "config", _OursConfig)
    monkeypatch.setitem(lm._CLASSES, "method", object)
    return _OursConfig


def test_no_quantization_registry_is_silent_and_does_not_raise(empty_vllm,
                                                               capsys) -> None:
    """"This vLLM has no quantization registry" -> not even a note.

    ``register_ommx_w`` catches that ImportError itself and returns None, so nothing
    reaches the plugin's except at all. Pinned because it is the case that must stay
    cheap: it is what happens in every CPU-side process that imports the plugin.
    """
    from ommx_gpu_serve.integration.vllm import plugin as pl
    pl._register_ommx_w_quant()                      # must not raise
    assert capsys.readouterr().err == ""


def test_missing_vllm_linear_bases_is_a_note_not_a_raise(quant_registry,
                                                         capsys) -> None:
    """A registry but no Linear bases -> note. THE KV-ONLY-RUN PROTECTION.

    ``_import_vllm_bases`` raises ``OMMXWError`` here, with an ``ImportError`` cause.
    Registration did not happen, so ``ommx_w`` stays unknown and vLLM refuses it by
    name later. Raising instead would take down every KV-only run in every worker —
    which is why a blanket re-raise is the wrong fix for the audit finding.
    """
    from ommx_gpu_serve.integration.vllm import linear_method as lm
    from ommx_gpu_serve.integration.vllm import plugin as pl
    # The quantization registry exists (fixture) but vllm.model_executor.layers.linear
    # does not, so _import_vllm_bases fails inside ommx_w_config_class().
    pl._register_ommx_w_quant()                      # must not raise
    err = capsys.readouterr().err
    assert "[ommx] note:" in err and "ommx_w" in err, err
    assert "OMMXWError" in err, err
    assert "Linear/quantization base classes" in err, (
        "the note did not come from _import_vllm_bases, so this test is exercising "
        f"some other failure than the one it names: {err!r}")
    assert "unknown method" in err, err
    assert lm.OMMX_W_METHOD_NAME not in quant_registry, (
        "a failed registration must not have claimed the name")


def test_foreign_claim_on_the_name_is_RE_RAISED(quant_registry, our_config_class,
                                                capsys) -> None:
    """THE DECISION GATE. A foreign config owns ``ommx_w`` -> the refusal propagates.

    Under the old blanket ``except Exception`` this printed a note and the engine
    carried on, so ``--quantization ommx_w`` resolved to ``_ForeignConfig`` and the run
    measured a method nobody selected. The by-name rejection the note promises cannot
    fire here, because the name DOES resolve — that asymmetry is the whole reason this
    one case re-raises while every other failure notes.
    """
    from ommx_gpu_serve.integration.vllm import linear_method as lm
    from ommx_gpu_serve.integration.vllm import plugin as pl
    quant_registry[lm.OMMX_W_METHOD_NAME] = _ForeignConfig     # somebody got there first

    with pytest.raises(lm.OMMXWError) as ei:
        pl._register_ommx_w_quant()
    msg = str(ei.value)
    assert "ommx_w" in msg and "_ForeignConfig" in msg, msg
    assert "already registered" in msg, msg
    assert capsys.readouterr().err == "", (
        "a collision must RAISE, not raise AND print the benign note")


def test_same_class_reregistration_is_not_a_collision(quant_registry,
                                                      our_config_class,
                                                      capsys) -> None:
    """OUR class already under the name (a second worker import) -> silent success.

    ``register_ommx_w`` tolerates this itself, so the plugin never sees an exception.
    Pinned because it is the case a naive "ValueError means collision" rule would get
    wrong — and getting it wrong would make every re-entrant plugin load fatal.
    """
    from ommx_gpu_serve.integration.vllm import linear_method as lm
    from ommx_gpu_serve.integration.vllm import plugin as pl
    quant_registry[lm.OMMX_W_METHOD_NAME] = our_config_class

    pl._register_ommx_w_quant()                      # must not raise
    assert capsys.readouterr().err == ""


def test_clean_registration_claims_the_name(quant_registry, our_config_class,
                                            capsys) -> None:
    """The happy path: nobody owns the name, we take it, nothing is printed.

    Without this the three tests above would all be satisfied by a function that never
    registers anything.
    """
    from ommx_gpu_serve.integration.vllm import linear_method as lm
    from ommx_gpu_serve.integration.vllm import plugin as pl
    pl._register_ommx_w_quant()
    assert quant_registry[lm.OMMX_W_METHOD_NAME] is our_config_class
    assert capsys.readouterr().err == ""


def test_foreign_claim_is_caught_even_when_the_registry_cannot_be_probed(
        quant_registry, our_config_class, monkeypatch, capsys) -> None:
    """The fallback discriminator: an unanswerable probe must not downgrade to a note.

    ``_ommx_w_name_claimed_by_a_foreign_config`` returns None whenever it cannot ask
    the registry (here: ``get_quantization_config`` itself is broken). The plugin then
    falls back to the cause type — the collision branch is the ONLY place
    ``register_ommx_w`` raises ``OMMXWError`` from a ``ValueError`` — so the conflict
    is still re-raised. Assuming "benign" on an unanswerable probe is exactly how a
    real conflict gets settled by import order.
    """
    from ommx_gpu_serve.integration.vllm import linear_method as lm
    from ommx_gpu_serve.integration.vllm import plugin as pl
    quant_registry[lm.OMMX_W_METHOD_NAME] = _ForeignConfig

    def _broken(name):
        raise RuntimeError("this vLLM's registry lookup is not usable")
    monkeypatch.setattr(sys.modules[_QUANT_MOD], "get_quantization_config", _broken)

    assert pl._ommx_w_name_claimed_by_a_foreign_config() is None
    with pytest.raises(lm.OMMXWError):
        pl._register_ommx_w_quant()
    assert capsys.readouterr().err == ""


def test_only_the_collision_branch_can_raise_from_a_ValueError() -> None:
    """THE FALLBACK DISCRIMINATOR'S PREMISE, pinned in the file that depends on it.

    ``_register_ommx_w_quant`` falls back to ``isinstance(exc.__cause__, ValueError)``
    whenever the registry cannot answer. That is only sound while the collision branch
    is the ONLY place a ``register_ommx_w()`` call can produce an ``OMMXWError`` whose
    ``__cause__`` is a ``ValueError``. Nothing in ``linear_method.py`` enforces that,
    and the failure mode is silent in BOTH directions:

      * a new ``raise ... from exc`` under an ``except ValueError``/``except Exception``
        anywhere on the ``register_ommx_w -> ommx_w_config_class -> _build_classes ->
        _import_vllm_bases`` path makes a BENIGN failure look like a collision, and the
        plugin then re-raises it — taking down every KV-only worker, which is exactly
        the outcome the note exists to prevent;
      * deleting the collision branch's ``from exc`` makes a REAL collision look benign.

    Neither is observable from the plugin's own behaviour on a healthy tree, so it is
    checked structurally. ``except Exception`` counts as "can carry a ValueError" —
    ``_ensure_kernel`` already has one of those (a build failure is re-raised ``from
    exc``); it is off this path today, and this test is what notices if it moves onto it.
    """
    import ast as _ast
    import inspect as _inspect
    from ommx_gpu_serve.integration.vllm import linear_method as lm

    #: the call path ``plugin._register_ommx_w_quant`` actually executes.
    on_path = {"register_ommx_w", "ommx_w_config_class", "_build_classes",
               "_import_vllm_bases"}
    #: handler types that can bind a ValueError to ``exc``.
    catches_value_error = {"ValueError", "Exception", "BaseException"}

    def _handler_names(handler):
        t = handler.type
        if t is None:
            return {"BaseException"}                    # bare except
        if isinstance(t, _ast.Name):
            return {t.id}
        if isinstance(t, _ast.Tuple):
            return {e.id for e in t.elts if isinstance(e, _ast.Name)}
        return {"<computed>"}                           # e.g. `except (a.b):` — be loud

    tree = _ast.parse(_inspect.getsource(lm))
    found = []
    for top in _ast.walk(tree):
        if not (isinstance(top, _ast.FunctionDef) and top.name in on_path):
            continue
        stack = []

        def walk(node):
            if isinstance(node, _ast.ExceptHandler):
                stack.append(_handler_names(node))
                for child in _ast.iter_child_nodes(node):
                    walk(child)
                stack.pop()
                return
            if (isinstance(node, _ast.Raise) and node.cause is not None
                    and stack and (stack[-1] & catches_value_error)):
                found.append((top.name, node.lineno, sorted(stack[-1])))
            for child in _ast.iter_child_nodes(node):
                walk(child)

        walk(top)

    assert len(found) == 1, (
        "the ValueError-cause fallback in plugin._register_ommx_w_quant is only sound "
        "while exactly ONE raise-from-a-ValueError-capable-handler exists on the "
        f"register_ommx_w call path; found {found}. Either classify the new one in "
        "plugin.py or stop using __cause__ as the fallback discriminator.")
    fn, lineno, handler = found[0]
    assert fn == "register_ommx_w" and handler == ["ValueError"], (
        f"linear_method.py:{lineno}: the one raise-from-ValueError moved into {fn}() "
        f"under `except {handler}`; the fallback discriminator names the collision "
        "branch of register_ommx_w specifically.")


def test_module_import_failure_is_a_note_not_a_raise(monkeypatch, capsys) -> None:
    """``linear_method`` itself unimportable -> note. Nothing was registered.

    The outer try in ``_register_ommx_w_quant`` covers the MODULE import; keeping it a
    note is the same by-name-rejection argument, and keeping it SEPARATE from the
    inner try is what lets the inner one classify.
    """
    from ommx_gpu_serve.integration.vllm import plugin as pl
    monkeypatch.setitem(
        sys.modules, "ommx_gpu_serve.integration.vllm.linear_method", None)
    pl._register_ommx_w_quant()                      # must not raise
    err = capsys.readouterr().err
    assert "[ommx] note:" in err and "ommx_w" in err, err
