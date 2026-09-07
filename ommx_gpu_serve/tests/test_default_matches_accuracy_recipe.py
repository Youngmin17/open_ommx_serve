# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""An engine started with no environment must serve the format the accuracy table describes.

WHY IT EXISTS. Every published KV accuracy number in this repository is produced by
``eval/lm_eval/models/ommx_hf_model.py``, which ``setdefault``s an eleven-key env block before
loading the model. The serving defaults in ``integration/vllm/config.py`` are a SEPARATE set
of literals. They disagreed: the harness pinned six FP4 outliers per group with power-of-two
scales, the serving default was three outliers with pow2 OFF.

Nothing detected that, because nothing compares them -- the harness's ``setdefault`` silently
corrected the serving default inside the accuracy process, and every serving benchmark set
the knobs explicitly. So the only configuration that ever ran the mismatched format was the
one a reader of the README would get: `vllm serve` with no OMMX environment at all. The
accuracy table and that engine described different formats.

This test is the comparison neither file performs. It reads the harness's own dict rather
than restating it, so the two cannot drift apart again without failing here.

SCOPE: KV only. The weight axis has its own presets and is not covered.
"""
import pytest

from ommx_gpu_serve.integration.vllm.config import (
    DEFAULT_OUTLIER_REPR,
    resolve_serving_config,
)

#: The knobs that change what the format IS. ``outlier_repr`` is deliberately absent: bitmap
#: and relidx7 encode the same positions and values and produce bit-identical token ids, so
#: they are interchangeable for accuracy and the default may differ from the harness's.
NUMERIC_KNOBS = {
    "OMMX_ATTN_K_FORMAT": ("k_format", str),
    "OMMX_ATTN_OUTLIERS": ("outliers_per_vector", int),
    "OMMX_ATTN_POW2": ("use_pow2", lambda v: v in ("1", "true", "on", "yes")),
    "OMMX_ATTN_OUTLIER_SELECT": ("outlier_select", str),
    "OMMX_KV_GROUP_TOKENS": ("group_tokens", int),
    "OMMX_KV_GROUP_CHANNELS": ("group_channels", int),
    "OMMX_KV_SINK": ("sink_tokens", int),
    "OMMX_KV_RECENT": ("recent_window", int),
}


def _harness_env():
    """The accuracy harness's own env block, read from source -- not restated here.

    Parsed textually because importing the module pulls in lm_eval and transformers.
    """
    import pathlib
    import re

    import ommx_gpu_serve

    p = (pathlib.Path(ommx_gpu_serve.__file__).parent.parent
         / "eval" / "lm_eval" / "models" / "ommx_hf_model.py")
    if not p.exists():
        pytest.skip(f"accuracy harness not present at {p}")
    src = p.read_text()
    env = dict(re.findall(r'"(OMMX_[A-Z0-9_]+)"\s*:\s*"([^"]*)"', src))
    if not env:
        pytest.fail(f"no OMMX env block found in {p}; the parse in this test is stale")
    return env


@pytest.fixture
def clean_env(monkeypatch):
    """No OMMX_* anywhere: the configuration a reader of the README actually gets."""
    import os

    for k in [k for k in os.environ if k.startswith("OMMX_")]:
        monkeypatch.delenv(k, raising=False)
    return resolve_serving_config()


@pytest.mark.parametrize("env_key", sorted(NUMERIC_KNOBS))
def test_each_numeric_knob_matches_the_accuracy_harness(clean_env, env_key):
    harness = _harness_env()
    if env_key not in harness:
        pytest.skip(f"{env_key} is not pinned by the accuracy harness")
    field, conv = NUMERIC_KNOBS[env_key]
    want = conv(harness[env_key])
    got = getattr(clean_env, field)
    assert got == want, (
        f"serving default {field}={got!r} but the accuracy harness pins {env_key}="
        f"{harness[env_key]!r} ({want!r}). An engine started with no environment would serve "
        f"a different format from the one every accuracy number describes")


def test_the_harness_block_is_now_a_restatement_not_a_correction(clean_env):
    """Stronger, and the property that actually matters: applying the harness's env on top of
    the serving defaults must change NOTHING numeric. If it changes something, the harness is
    silently correcting the defaults again and the two have drifted."""
    harness = _harness_env()
    drift = []
    for env_key, (field, conv) in NUMERIC_KNOBS.items():
        if env_key not in harness:
            continue
        if getattr(clean_env, field) != conv(harness[env_key]):
            drift.append(f"{field}: default={getattr(clean_env, field)!r} "
                         f"harness={harness[env_key]!r}")
    assert not drift, "the accuracy harness still corrects the serving defaults: " + "; ".join(drift)


# ── the encoding default ───────────────────────────────────────────────────────

def test_the_default_encoding_is_bitmap_and_is_decodable(clean_env):
    """bitmap is the main path. It must also RESOLVE to a readable one: the store allocates
    k_obmp and no k_oidx, so a default that did not also resolve bitmap_read would raise on
    the first quantized group."""
    assert clean_env.outlier_repr == "bitmap"
    assert clean_env.resolved_bitmap_read() is True
    assert clean_env.resolved_combinadic_read() is False


def test_the_default_outlier_count_is_inside_the_bitmap_kernel_gate(clean_env):
    """The bitmap decode stages its value stream in one int32, i.e. ceil(k/2) <= 4 bytes.
    A default of k > 8 with a bitmap default would be a configuration that cannot run."""
    assert clean_env.outliers_per_vector <= 8, (
        f"default k={clean_env.outliers_per_vector} exceeds the bitmap kernel's k<=8 gate "
        f"while the default encoding is bitmap; one of the two must change")


# ── the encoding default, per harness ──────────────────────────────────────────

#: Every harness that pins OMMX_ATTN_OUTLIER_REPR, relative to the repo root. NOT in
#: NUMERIC_KNOBS on purpose: accuracy is repr-invariant (bitmap and relidx7 carry the same
#: positions and values and produce bit-identical token ids) but PERFORMANCE and the pool
#: footprint are not -- bitmap decodes with one wide load plus a masked popcount and is
#: 0.58-0.67x the relidx7 TPOT at >=16K, and prices 8.250 vs 8.750 bit per (K,V) pair. A
#: harness that kept relidx7 would publish TPOT / footprint numbers for an encoding the
#: engine no longer serves by default.
HARNESS_FILES = (
    "figure/bench.py",
    "eval/lm_eval/models/ommx_hf_model.py",
    "ommx_gpu_serve/hf_eager/_ommx_hf_tpot_bench.py",
    "ommx_gpu_serve/hf_eager/_ommx_hf_batch_test.py",
    "ommx_gpu_serve/bench/bench_e2e_a100.py",
)


_REPR_KEY = "OMMX_ATTN_OUTLIER_REPR"
_CONFIG_MODULE = "ommx_gpu_serve.integration.vllm.config"


def _harness_repr_bindings(path):
    """EVERY value a harness binds to OMMX_ATTN_OUTLIER_REPR, each resolved to a repr name.

    Walked as an AST (not a regex) for the same reason ``_harness_env`` is not imported:
    the modules pull in transformers / vLLM. Three binding shapes are collected --
    a dict entry (``{"OMMX_ATTN_OUTLIER_REPR": ...}``, also ``dict(...=)`` keywords), a
    ``setdefault("OMMX_ATTN_OUTLIER_REPR", ...)`` call, and a subscript store
    (``os.environ["OMMX_ATTN_OUTLIER_REPR"] = ...``) -- and ALL of them are returned, not
    the first: a later ``os.environ[...] = "relidx7"`` would override the pin this test
    approved, and a docstring that mentions the key is not a binding and is never seen.

    A value must be a string literal, the ``DEFAULT_OUTLIER_REPR`` name imported from
    integration/vllm/config.py (checked against the file's own import statements, any
    form -- a multi-line ``import (...)`` included -- so a stale local alias cannot pass),
    ``config.DEFAULT_OUTLIER_REPR`` on that module, or ``<one of those> if <cond> else
    "<known repr>"``: the shape bench_e2e_a100 uses to drop to relidx7 above the bitmap
    kernel's k<=8 gate. Its fallback must name a known encoding so it is a deliberate,
    readable choice; the pin it is judged by is the default branch. Anything else fails.
    """
    import ast

    from ommx_gpu_serve.integration.vllm.config import OUTLIER_REPRS

    tree = ast.parse(path.read_text(), filename=str(path))

    const_names, module_names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == _CONFIG_MODULE:
                const_names.update(a.asname or a.name for a in node.names
                                   if a.name == "DEFAULT_OUTLIER_REPR")
            elif node.module == _CONFIG_MODULE.rsplit(".", 1)[0]:
                module_names.update(a.asname or a.name for a in node.names
                                    if a.name == "config")
        elif isinstance(node, ast.Import):
            module_names.update(a.asname or a.name for a in node.names
                                if a.name == _CONFIG_MODULE)

    def _is_imported_default(v):
        if isinstance(v, ast.Name):
            return v.id in const_names
        if isinstance(v, ast.Attribute) and v.attr == "DEFAULT_OUTLIER_REPR":
            return ast.unparse(v.value) in module_names
        return False

    def _resolve(v):
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return v.value
        if _is_imported_default(v):
            return DEFAULT_OUTLIER_REPR
        if (isinstance(v, ast.IfExp) and isinstance(v.orelse, ast.Constant)
                and v.orelse.value in OUTLIER_REPRS
                and (_is_imported_default(v.body)
                     or (isinstance(v.body, ast.Constant) and isinstance(v.body.value, str)))):
            return _resolve(v.body)
        pytest.fail(
            f"{path}:{v.lineno} binds {_REPR_KEY} to `{ast.unparse(v)}`, which is neither a "
            f"string literal nor DEFAULT_OUTLIER_REPR imported from {_CONFIG_MODULE} (nor a "
            f"conditional falling back from one of those to a known encoding)")

    bound = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            bound += [v for k, v in zip(node.keys, node.values)
                      if isinstance(k, ast.Constant) and k.value == _REPR_KEY]
        elif isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr == "setdefault" and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == _REPR_KEY):
                bound.append(node.args[1])
            bound += [kw.value for kw in node.keywords if kw.arg == _REPR_KEY]
        elif isinstance(node, ast.Assign):
            bound += [node.value for t in node.targets
                      if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                      and t.slice.value == _REPR_KEY]
    if not bound:
        pytest.fail(f"no {_REPR_KEY} binding found in {path}; the parse here is stale")
    return [(v.lineno, _resolve(v)) for v in bound]


@pytest.mark.parametrize("rel", HARNESS_FILES)
def test_each_harness_pins_the_serving_encoding_default(clean_env, rel):
    import pathlib

    import ommx_gpu_serve

    p = pathlib.Path(ommx_gpu_serve.__file__).parent.parent / rel
    if not p.exists():
        pytest.skip(f"harness not present at {p}")
    bad = [(line, got) for line, got in _harness_repr_bindings(p)
           if got != clean_env.outlier_repr]
    assert not bad, (
        f"{rel} binds OMMX_ATTN_OUTLIER_REPR to {bad} (line, value) but the engine serves "
        f"{clean_env.outlier_repr!r} by default; its TPOT / footprint numbers would describe "
        f"an encoding the bare engine does not run")
