# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Gate for the ``ommx_w`` vLLM weight-quant linear method
(``integration/vllm/linear_method.py``).

WHAT THIS FILE CAN AND CANNOT PROVE. There is no GPU and no vLLM on the host these
gates were written for, so the split is stated up front rather than implied:

  * PROVEN HERE (CPU, no vLLM, no CUDA) —
      - importing ``linear_method`` does not import vLLM (checked in a SUBPROCESS, so a
        stub installed by another test in this session cannot make it look true);
      - ``register_ommx_w()`` is importable, callable and idempotent with no vLLM, and
        no-ops returning ``None`` — the shape ``plugin.register`` uses for a missing
        registry;
      - with vLLM stubbed, registration happens exactly once, and a name collision with
        a FOREIGN config raises instead of being won by import order;
      - ``OMMXWConfig.from_config`` parses a REAL bundle — built here by the packer's
        ``make-synthetic`` + ``pack``, not by a fixture that re-implements the format —
        and rejects a foreign ``quant_method``, a path that is not a bundle, and the two
        recipes the CUDA kernel cannot read (``outlier_repr='bitmap'``,
        ``outlier_map='none'``);
      - every refusal message names its fix (the packer CLI / the two measured build
        environment requirements);
      - the plane shapes ``create_weights`` allocates equal the shapes the bundle's own
        MANIFEST declares — the check that would catch a layout drift between the packer
        and the serving path;
      - the opt-in LOAD-TIME TRANSCODE that makes the paper's own weight format
        (``bitmap`` positions + no range map, the 3.6250 b/wt recipe) servable: that the
        re-encoded position plane is BYTE-IDENTICAL to packing the same weight with
        ``outlier_repr="relidx7"`` directly, that the dequantized weight does not move,
        that BOTH footprints (3.6250 on disk / 4.1250 resident) are computed and reported,
        that the refusal is unchanged without the opt-in, that the shipped path still hands
        the kernel the parameter tensors THEMSELVES (asserted by storage address), and that
        the recipes no transcode can reach are still refused WITH the opt-in.
  * NOT PROVEN HERE — that the CUDA kernel builds, that it consumes these planes
    correctly, that vLLM's weight loader fills the parameters (see the KNOWN GAP in
    ``linear_method``'s docstring), or any latency. The one test that would show it is
    ``gpu``-MARKED, so on this host it SKIPS. A skip is visibly not a pass.

Stubbing follows ``test_preflight_guards.py``: ``sys.modules`` gets purpose-built module
objects, monkeypatched so pytest removes them again. Two directions are needed here —
``_no_vllm`` maps ``sys.modules['vllm']`` to ``None`` (which makes ``import vllm`` raise
even on a host that HAS vLLM, so the no-op path is tested for real), and ``fake_vllm``
installs the five modules the lazily-built classes import from.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import pytest
import torch

from ommx_gpu_serve.integration.vllm import linear_method as lm
from ommx_gpu_serve.linear.w_format import INDEX_FILENAME, Recipe, plane_name
from ommx_gpu_serve.linear.w_packer import (
    build_recipe,
    make_synthetic_checkpoint,
    pack_checkpoint,
)

# The synthetic checkpoint's geometry. hidden=128 with group_size=64 gives G=2 groups per
# row, and down_proj's K=256 gives G=4 — so the fixtures cover more than one group count
# without packing anything big enough to slow the suite down.
_HIDDEN = 128
_INTER = 256
_GROUP_SIZE = 64
_NPV = 4

_Q_PROJ = "model.layers.0.self_attn.q_proj.weight"
_K_PROJ = "model.layers.0.self_attn.k_proj.weight"
_LM_HEAD = "lm_head.weight"


# ══════════════════════════════════════════════════════════════════════════════
# bundles — built by the REAL packer, once per session
# ══════════════════════════════════════════════════════════════════════════════

def _pack(src: str, out: str, **recipe_kw) -> str:
    kw = dict(group_size=_GROUP_SIZE, outlier_pct=None, npv=_NPV,
              outlier_repr="relidx7", outlier_map="idx_range", zp_dtype="bf16")
    kw.update(recipe_kw)
    pack_checkpoint(src, out, build_recipe(**kw), source_model="synthetic/test-model",
                    verbose=False)
    return out


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory) -> str:
    """A tiny Llama-shaped HF checkpoint. The packer's own answer to "no model, no net"."""
    d = str(tmp_path_factory.mktemp("hf_ckpt"))
    make_synthetic_checkpoint(d, layers=1, hidden=_HIDDEN, inter=_INTER, vocab=256,
                              shards=2)
    return d


@pytest.fixture(scope="session")
def bundle(checkpoint, tmp_path_factory) -> str:
    """The canonical KERNEL-READABLE bundle: relidx7 positions + idx_range map."""
    return _pack(checkpoint, str(tmp_path_factory.mktemp("bundle_ok")))


@pytest.fixture(scope="session")
def bundle_bitmap(checkpoint, tmp_path_factory) -> str:
    """A VALID bundle the weight kernel cannot read — bitmap outlier positions."""
    return _pack(checkpoint, str(tmp_path_factory.mktemp("bundle_bitmap")),
                 outlier_repr="bitmap")


@pytest.fixture(scope="session")
def bundle_nomap(checkpoint, tmp_path_factory) -> str:
    """A VALID bundle the weight kernel cannot read — no FP4 range-map planes."""
    return _pack(checkpoint, str(tmp_path_factory.mktemp("bundle_nomap")),
                 outlier_map="none")


@pytest.fixture(scope="session")
def bundle_paper(checkpoint, tmp_path_factory) -> str:
    """The PAPER's own weight format — and the reason the transcode exists.

    ``outlier_repr="bitmap"`` (claim B4: "positions are stored as a flat bitmask, N bits
    per group") with no range-map planes (claim B6's bundle listing has none). At the
    fixture geometry (gs=64, npv=4) that is exactly Table 1's AvgBits 3.63 — 3.6250
    stored — and it is a recipe the CUDA kernel cannot read a single byte of.
    """
    return _pack(checkpoint, str(tmp_path_factory.mktemp("bundle_paper")),
                 outlier_repr="bitmap", outlier_map="none")


@pytest.fixture(scope="session")
def bundle_manifests(bundle) -> dict:
    """``{tensor name: manifest entry}`` read straight off the shards.

    Read INDEPENDENTLY of ``OMMXWBundle`` so the shape assertions below compare the
    serving path against the FILES, not against another call into the same object.
    """
    from ommx_gpu_serve.linear.w_format import read_manifest
    with open(os.path.join(bundle, INDEX_FILENAME), encoding="utf-8") as fh:
        index = json.load(fh)
    out = {}
    for shard in sorted(set(index["weight_map"].values())):
        out.update(read_manifest(os.path.join(bundle, shard))["tensors"])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# module-state + vLLM stubs
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _clean_module_state():
    """Reset every process latch around each test.

    ``linear_method`` deliberately latches registration, the kernel handle and the fire
    counters for the lifetime of a worker process. That is correct in production and
    poisonous across tests, so it is cleared explicitly here — before AND after, so a
    test that fails mid-way cannot leak into the next one.
    """
    lm._reset_state_for_tests()
    yield
    lm._reset_state_for_tests()


@pytest.fixture(autouse=True)
def _transcode_opt_in_is_off(monkeypatch: pytest.MonkeyPatch):
    """No test inherits the transcode opt-in from the caller's shell.

    Every "the refusal still stands" gate below would pass vacuously if
    ``OMMX_W_TRANSCODE`` happened to be exported in the environment pytest was launched
    from, so it is cleared for every test and re-set explicitly by the ones that want it.
    """
    monkeypatch.delenv(lm.TRANSCODE_ENV, raising=False)


@pytest.fixture
def no_vllm(monkeypatch: pytest.MonkeyPatch):
    """Make ``import vllm`` fail, even on a host that has vLLM installed.

    A ``None`` value in ``sys.modules`` is the documented way to force an ImportError, so
    this exercises the real no-op path rather than merely relying on the test host
    happening to lack vLLM.
    """
    for name in [k for k in list(sys.modules) if k == "vllm" or k.startswith("vllm.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "vllm", None)


class _StubQuantizationConfig:
    """Stand-in for ``vllm...quantization.base_config.QuantizationConfig``."""

    def __init__(self) -> None:
        pass


class _StubLinearMethodBase:
    """Stand-in for ``vllm.model_executor.layers.linear.LinearMethodBase``."""


class _StubLinearBase(torch.nn.Module):
    """Stand-in for ``vllm...linear.LinearBase`` — a real Module, so parameters register."""


class _StubUnquantizedLinearMethod:
    """Stand-in for ``vllm...linear.UnquantizedLinearMethod``."""


def _stub_set_weight_attrs(weight, attrs):
    for key, value in attrs.items():
        setattr(weight, key, value)


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch):
    """Install the five vLLM modules ``_build_classes`` / ``register_ommx_w`` import.

    Returns the registry dict so a test can inspect what was registered. No real vLLM is
    involved and none is required; what is under test is OUR wiring, not vLLM's.
    """
    registry: dict = {}

    def register_quantization_config(name: str):
        def _deco(cls):
            if name in registry:
                raise ValueError(f"quantization method {name!r} is already registered")
            registry[name] = cls
            return cls
        return _deco

    def get_quantization_config(name: str):
        return registry.get(name)

    mods = {}
    vllm = types.ModuleType("vllm")
    vllm.__version__ = "0.21.0"
    mods["vllm"] = vllm
    me = types.ModuleType("vllm.model_executor")
    layers = types.ModuleType("vllm.model_executor.layers")
    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.LinearBase = _StubLinearBase
    linear.LinearMethodBase = _StubLinearMethodBase
    linear.UnquantizedLinearMethod = _StubUnquantizedLinearMethod
    quant = types.ModuleType("vllm.model_executor.layers.quantization")
    quant.register_quantization_config = register_quantization_config
    quant.get_quantization_config = get_quantization_config
    base_config = types.ModuleType(
        "vllm.model_executor.layers.quantization.base_config")
    base_config.QuantizationConfig = _StubQuantizationConfig
    utils = types.ModuleType("vllm.model_executor.utils")
    utils.set_weight_attrs = _stub_set_weight_attrs
    mods.update({
        "vllm.model_executor": me,
        "vllm.model_executor.layers": layers,
        "vllm.model_executor.layers.linear": linear,
        "vllm.model_executor.layers.quantization": quant,
        "vllm.model_executor.layers.quantization.base_config": base_config,
        "vllm.model_executor.utils": utils,
    })
    # NOTE: no ``vllm.config`` module. That is deliberate — it makes
    # ``_current_vllm_model_path()`` take its ImportError branch, so bundle resolution in
    # these tests can only come from the sources a test sets explicitly.
    vllm.model_executor = me
    me.layers = layers
    me.utils = utils
    layers.linear = linear
    layers.quantization = quant
    quant.base_config = base_config
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return registry


# ══════════════════════════════════════════════════════════════════════════════
# import hygiene
# ══════════════════════════════════════════════════════════════════════════════

def test_importing_linear_method_does_not_import_vllm():
    """The package's import contract, checked in a FRESH interpreter.

    ``backend.py`` is the single module in ``ommx_gpu_serve`` allowed to need vLLM at
    import time; every other module — this one included — must import on a bare CPU box.
    Asserted in a SUBPROCESS because by the time this file runs, another test in the
    session may already have stubbed ``vllm`` into ``sys.modules``, which would make an
    in-process check pass for the wrong reason.
    """
    code = (
        "import sys;"
        "import ommx_gpu_serve.integration.vllm.linear_method as m;"
        "leaked=[k for k in sys.modules if k=='vllm' or k.startswith('vllm.')];"
        "assert not leaked, leaked;"
        "assert m.register_ommx_w.__name__=='register_ommx_w';"
        "print('OK')"
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    env = dict(os.environ, PYTHONPATH=repo_root)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, cwd=repo_root)
    assert out.returncode == 0, f"stdout={out.stdout!r} stderr={out.stderr!r}"
    assert "OK" in out.stdout


def test_module_exposes_the_contracted_entry_point():
    """``bench_e2e_a100.py`` imports this exact path + name. It is a contract."""
    from ommx_gpu_serve.integration.vllm.linear_method import register_ommx_w  # noqa: F401
    assert lm.OMMX_W_METHOD_NAME == "ommx_w"


# ══════════════════════════════════════════════════════════════════════════════
# registration
# ══════════════════════════════════════════════════════════════════════════════

def test_register_is_a_noop_and_idempotent_without_vllm(no_vllm):
    """No vLLM -> return None, twice, without raising.

    Mirrors ``plugin.register()``'s ``None`` for a missing v1 attention registry: the
    entry point has to be callable on a packer host or a test runner. Returning None is
    not permission to serve — nothing can select ``--quantization ommx_w`` in a process
    with no vLLM to select it in.
    """
    assert lm.register_ommx_w() is None
    assert lm.register_ommx_w() is None
    assert lm._REGISTERED is True
    assert lm._REGISTERED_AS is None


def test_register_with_vllm_registers_exactly_once(fake_vllm):
    assert lm.register_ommx_w() == "ommx_w"
    assert set(fake_vllm) == {"ommx_w"}
    cls = fake_vllm["ommx_w"]
    assert cls.get_name() == "ommx_w"
    assert cls.get_min_capability() == 80          # build_ommx_linear asserts sm >= 80
    assert cls.get_supported_act_dtypes() == [torch.bfloat16]
    # Idempotent: the second call must NOT hit the registry again (which would raise
    # ValueError) and must report the same name.
    assert lm.register_ommx_w() == "ommx_w"
    assert fake_vllm["ommx_w"] is cls


def test_register_refuses_to_share_the_name_with_a_foreign_config(fake_vllm):
    """Two implementations under one ``--quantization`` name is not a tie to break."""
    class _SomeoneElsesConfig:
        pass

    fake_vllm["ommx_w"] = _SomeoneElsesConfig
    with pytest.raises(lm.OMMXWError) as exc:
        lm.register_ommx_w()
    msg = str(exc.value)
    assert "already registered" in msg
    assert "_SomeoneElsesConfig" in msg


def test_plugin_register_also_registers_ommx_w_without_the_attention_registry(
        fake_vllm, monkeypatch):
    """``plugin.register()`` must register ``ommx_w`` even with NO v1 attention registry.

    The two registries are independent, and ``register()`` returns early (returning
    ``None``) when ``vllm.v1.attention.backends.registry`` cannot be imported — which is
    precisely why the ``ommx_w`` call sits OUTSIDE that early return. Without this gate
    the placement is unasserted: deleting ``_register_ommx_w_quant()`` from
    ``register()`` altogether leaves the whole suite green (verified by mutation), and
    plugin-side registration is the ONLY thing that makes ``--quantization ommx_w``
    reachable in a real ``vllm serve`` run.

    ``vllm.v1...registry`` is force-failed rather than merely left out of the stub, so
    this asserts the same thing on a host that HAS vLLM installed.
    """
    from ommx_gpu_serve.integration.vllm import plugin

    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.registry", None)
    monkeypatch.setattr(plugin, "_REGISTERED", False, raising=False)

    assert plugin.register() is None                 # no v1 attention registry
    assert set(fake_vllm) == {"ommx_w"}              # ...but the quant method landed
    assert fake_vllm["ommx_w"] is lm.ommx_w_config_class()
    assert lm._REGISTERED_AS == "ommx_w"


def test_class_access_without_vllm_raises_a_named_error(no_vllm):
    """``linear_method.OMMXWConfig`` on a CPU box must say WHAT is missing."""
    with pytest.raises(lm.OMMXWError) as exc:
        lm.ommx_w_config_class()
    assert "vllm>=0.21" in str(exc.value)
    with pytest.raises(lm.OMMXWError):
        _ = lm.OMMXWConfig                          # PEP 562 __getattr__ path


def test_unknown_module_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = lm.definitely_not_a_symbol


# ══════════════════════════════════════════════════════════════════════════════
# bundle discovery + manifest parsing
# ══════════════════════════════════════════════════════════════════════════════

def test_load_bundle_parses_a_real_manifest(bundle, bundle_manifests):
    b = lm.load_bundle(bundle)
    assert b.recipe.group_size == _GROUP_SIZE
    assert b.recipe.npv == _NPV
    assert b.recipe.outlier_repr == "relidx7"
    assert b.recipe.bundle_format == "i2f4"
    assert b.source_model() == "synthetic/test-model"
    # Every tensor the shards describe is reachable, with its packer-recorded kind.
    assert set(b.tensors) == set(bundle_manifests)
    assert b.entry(_Q_PROJ)["kind"] == "quantized"
    assert b.entry(_LM_HEAD)["kind"] == "passthrough"
    assert "never quantized" in b.entry(_LM_HEAD)["reason"]


def test_layer_kind_resolves_fused_vllm_modules(bundle):
    """vLLM fuses q/k/v and gate/up; the bundle names them separately, as the ckpt does."""
    b = lm.load_bundle(bundle)
    kind, names = b.layer_kind("model.layers.0.self_attn.qkv_proj")
    assert kind == "quantized"
    assert names == [_Q_PROJ, _K_PROJ, "model.layers.0.self_attn.v_proj.weight"]
    assert b.layer_kind("model.layers.0.mlp.gate_up_proj")[0] == "quantized"
    assert b.layer_kind("model.layers.0.self_attn.o_proj")[0] == "quantized"
    assert b.layer_kind("lm_head")[0] == "passthrough"
    assert b.layer_kind("model.layers.0.self_attn.not_a_real_proj")[0] == "unmapped"


def test_resolve_bundle_dir_precedence(bundle, monkeypatch, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv("OMMX_W_BUNDLE", bundle)
    # explicit wins over the env...
    assert lm.resolve_bundle_dir(bundle) == os.path.abspath(bundle)
    # ...the quantization_config key wins over the env...
    assert lm.resolve_bundle_dir(None, {"bundle": bundle}) == os.path.abspath(bundle)
    # ...and the env is the fallback.
    assert lm.resolve_bundle_dir() == os.path.abspath(bundle)
    # A candidate that exists but is not a bundle is skipped, not accepted.
    assert lm.resolve_bundle_dir(str(other)) == os.path.abspath(bundle)


def test_no_bundle_configured_names_the_packer(no_vllm, monkeypatch):
    monkeypatch.delenv("OMMX_W_BUNDLE", raising=False)
    with pytest.raises(lm.OMMXWError) as exc:
        lm.resolve_bundle_dir()
    msg = str(exc.value)
    assert "python -m ommx_gpu_serve.linear.w_packer pack" in msg
    assert "make-synthetic" in msg
    assert INDEX_FILENAME in msg
    assert "OMMX_W_BUNDLE" in msg


def test_a_plain_hf_checkpoint_is_not_a_bundle(checkpoint, no_vllm):
    """The most likely operator mistake: pointing ommx_w at the UNPACKED model."""
    assert not lm.is_ommx_w_bundle(checkpoint)
    with pytest.raises(lm.OMMXWError) as exc:
        lm.resolve_bundle_dir(checkpoint)
    msg = str(exc.value)
    assert "none of the candidate paths is an OMMX_W_SafeTensor bundle" in msg
    assert "w_packer pack" in msg
    assert checkpoint in msg                        # says WHICH path was rejected


def test_a_broken_bundle_is_reported_as_such(bundle, tmp_path):
    """An index that references a shard which is not there must not read as "no bundle"."""
    broken = tmp_path / "broken"
    broken.mkdir()
    with open(os.path.join(bundle, INDEX_FILENAME), encoding="utf-8") as fh:
        index = json.load(fh)
    with open(broken / INDEX_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(index, fh)                        # index only; no shards copied
    with pytest.raises(lm.OMMXWError) as exc:
        lm.load_bundle(str(broken))
    msg = str(exc.value)
    assert "is not a loadable OMMX_W_SafeTensor bundle" in msg
    assert "w_packer pack" in msg


# ══════════════════════════════════════════════════════════════════════════════
# recipes the CUDA kernel cannot read
# ══════════════════════════════════════════════════════════════════════════════

def test_bitmap_outlier_positions_are_refused(bundle_bitmap):
    """The packer can emit bitmap positions; the WEIGHT kernel has no bitmap reader.

    Serving it would not fail — ``sparse_correct`` would read the bitmap bytes as a
    relidx7 slot stream and reconstruct the wrong weights. Refusing at load is the only
    place that failure is visible.
    """
    with pytest.raises(lm.OMMXWError) as exc:
        lm.load_bundle(bundle_bitmap)
    msg = str(exc.value)
    assert "outlier_repr='bitmap'" in msg
    assert "relidx7" in msg
    assert "--outlier-repr relidx7" in msg
    assert "NO bitmap reader" in msg


def test_missing_range_map_planes_are_refused(bundle_nomap):
    with pytest.raises(lm.OMMXWError) as exc:
        lm.load_bundle(bundle_nomap)
    msg = str(exc.value)
    assert "outlier_map='none'" in msg
    assert "--outlier-map idx_range" in msg


def test_require_kernel_readable_ignores_an_outlier_free_recipe():
    """npv=0 has no position stream at all, so no representation question arises."""
    lm.require_kernel_readable(Recipe(group_size=64, npv=0, outlier_pct=0.0,
                                      outlier_repr="bitmap", outlier_map="none"))


# ══════════════════════════════════════════════════════════════════════════════
# OMMXWConfig
# ══════════════════════════════════════════════════════════════════════════════

def test_config_from_config_parses_a_real_bundle(fake_vllm, bundle):
    cfg = lm.ommx_w_config_class().from_config(
        {"quant_method": "ommx_w", "bundle": bundle})
    assert cfg.recipe.group_size == _GROUP_SIZE
    assert cfg.bundle.bundle_dir == os.path.abspath(bundle)
    assert cfg.packed_modules_mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]


def test_config_rejects_a_foreign_quant_method(fake_vllm, bundle):
    with pytest.raises(lm.OMMXWError) as exc:
        lm.ommx_w_config_class().from_config({"quant_method": "awq", "bundle": bundle})
    msg = str(exc.value)
    assert "'awq'" in msg
    assert "--quantization awq" in msg


def test_config_rejects_an_unsupported_recipe(fake_vllm, bundle_bitmap):
    """The refusal survives the whole from_config path, not just load_bundle."""
    with pytest.raises(lm.OMMXWError) as exc:
        lm.ommx_w_config_class().from_config(
            {"quant_method": "ommx_w", "bundle": bundle_bitmap})
    assert "--outlier-repr relidx7" in str(exc.value)


def test_get_quant_method_dispatches_from_the_manifest(fake_vllm, bundle):
    cfg = lm.ommx_w_config_class().from_config({"quant_method": "ommx_w",
                                                "bundle": bundle})
    method_cls = lm.ommx_w_linear_method_class()
    layer = _StubLinearBase()
    got = cfg.get_quant_method(layer, "model.layers.0.self_attn.qkv_proj")
    assert isinstance(got, method_cls)
    # A manifest 'passthrough' (lm_head) is a DECISION the packer recorded with a reason,
    # not a fallback: vLLM's unquantized method is the right answer for it.
    assert isinstance(cfg.get_quant_method(layer, "lm_head"),
                      _StubUnquantizedLinearMethod)
    # Not a Linear at all -> None, so vLLM keeps whatever that layer's own path is.
    assert cfg.get_quant_method(object(), "model.embed_tokens") is None


def test_get_quant_method_refuses_an_unmapped_linear(fake_vllm, bundle):
    """A Linear the bundle never described must not quietly serve bf16."""
    cfg = lm.ommx_w_config_class().from_config({"quant_method": "ommx_w",
                                                "bundle": bundle})
    with pytest.raises(lm.OMMXWError) as exc:
        cfg.get_quant_method(_StubLinearBase(), "model.layers.9.self_attn.qkv_proj")
    msg = str(exc.value)
    assert "model.layers.9.self_attn.qkv_proj" in msg
    assert "synthetic/test-model" in msg            # names the bundle's source model
    assert "OMMX_W_ALLOW_UNMAPPED" in msg
    assert "w_packer pack" in msg


def test_unmapped_linear_can_be_opted_into_loudly(fake_vllm, bundle, monkeypatch,
                                                  capsys):
    monkeypatch.setenv("OMMX_W_ALLOW_UNMAPPED", "1")
    monkeypatch.setenv("OMMX_FIRE_FILE", os.path.join(bundle, "fire.log"))
    cfg = lm.ommx_w_config_class().from_config({"quant_method": "ommx_w",
                                                "bundle": bundle})
    got = cfg.get_quant_method(_StubLinearBase(), "model.layers.9.self_attn.qkv_proj")
    assert isinstance(got, _StubUnquantizedLinearMethod)
    err = capsys.readouterr().err
    assert "WARNING" in err and "bits/weight figure for this model is wrong" in err


# ══════════════════════════════════════════════════════════════════════════════
# create_weights — the shapes must equal the MANIFEST's
# ══════════════════════════════════════════════════════════════════════════════

def _make_layer_and_method(bundle: str):
    cfg = lm.ommx_w_config_class().from_config({"quant_method": "ommx_w",
                                                "bundle": bundle})
    method = cfg.get_quant_method(_StubLinearBase(), "model.layers.0.self_attn.o_proj")
    return cfg, method


def test_create_weights_allocates_exactly_the_manifest_planes(fake_vllm, bundle,
                                                              bundle_manifests):
    """Plane-by-plane equality between what is ALLOCATED and what is ON DISK.

    This is the check that catches a drift between the packer's layout and the serving
    path's: both derive from ``Recipe.plane_layout``, but only this test proves that the
    parameters a layer actually ends up holding match the manifest of the file that will
    fill them.
    """
    cfg, method = _make_layer_and_method(bundle)
    layer = _StubLinearBase()
    N, K = bundle_manifests[_Q_PROJ]["shape"]       # [128, 128] for q_proj
    method.create_weights(layer, K, [N], K, N, torch.bfloat16,
                          weight_loader=lambda *a, **k: None)

    declared = bundle_manifests[_Q_PROJ]["planes"]
    params = dict(layer.named_parameters())
    assert set(params) == {lm.param_name_for_plane(p) for p in declared}
    st_to_torch = {"U8": torch.uint8, "I8": torch.int8, "BF16": torch.bfloat16,
                   "F32": torch.float32}
    for plane, desc in declared.items():
        param = params[lm.param_name_for_plane(plane)]
        assert list(param.shape) == list(desc["shape"]), plane
        assert param.dtype is st_to_torch[desc["dtype"]], plane
        assert param.requires_grad is False
        # The loader attributes vLLM needs to narrow a TP shard. UNVERIFIED against a
        # real multi-GPU run; asserted here so they cannot silently disappear.
        assert param.output_dim == 0
        assert param.input_dim == 1
        assert param.ommx_plane == plane
    assert params[lm.param_name_for_plane("code")].packed_factor == 4
    assert params[lm.param_name_for_plane("zp")].packed_factor == _GROUP_SIZE
    assert layer.ommx_w_N == N and layer.ommx_w_K == K
    assert layer.ommx_w_group_size == _GROUP_SIZE and layer.ommx_w_npv == _NPV
    assert layer.ommx_w_ready is False
    assert lm.ommx_w_fire_stats()["layers_created"] == 1


def test_create_weights_sums_the_output_partitions_of_a_fused_linear(fake_vllm, bundle,
                                                                     bundle_manifests):
    """A fused qkv Linear owns q+k+v rows; the planes must be sized for all of them."""
    cfg, method = _make_layer_and_method(bundle)
    layer = _StubLinearBase()
    parts = [bundle_manifests[n]["shape"][0]
             for n in (_Q_PROJ, _K_PROJ, "model.layers.0.self_attn.v_proj.weight")]
    K = bundle_manifests[_Q_PROJ]["shape"][1]
    method.create_weights(layer, K, parts, K, sum(parts), torch.bfloat16)
    code = dict(layer.named_parameters())[lm.param_name_for_plane("code")]
    assert list(code.shape) == [sum(parts), K // 4]


def test_create_weights_refuses_a_k_the_group_size_does_not_divide(fake_vllm, bundle):
    """Under TP the PER-PARTITION K must stay a multiple of the group size."""
    cfg, method = _make_layer_and_method(bundle)
    with pytest.raises(lm.OMMXWError) as exc:
        method.create_weights(_StubLinearBase(), _GROUP_SIZE + 4, [64],
                              _GROUP_SIZE + 4, 64, torch.bfloat16)
    msg = str(exc.value)
    assert "group_size=64" in msg
    assert "cannot be split across ranks" in msg


# ══════════════════════════════════════════════════════════════════════════════
# JIT build failure + apply() guards
# ══════════════════════════════════════════════════════════════════════════════

def test_build_failure_names_both_measured_environment_requirements(monkeypatch):
    """MEASURED_FACTS §10: `ninja` on PATH, and LIBRARY_PATH on cu13 conda envs.

    Both were real blockers on the cluster before the parity gate could run at all, and
    neither is discoverable from the raw torch error, so the message carries them.
    """
    calls = []

    def _boom():
        calls.append(1)
        raise RuntimeError("Ninja is required to load C++ extensions")

    monkeypatch.setattr(lm, "_load_build_module",
                        lambda: types.SimpleNamespace(build=_boom))
    with pytest.raises(lm.OMMXWError) as exc:
        lm.build_kernel()
    msg = str(exc.value)
    assert "FAILED TO BUILD" in msg
    assert "ninja" in msg and "PATH" in msg
    assert "LIBRARY_PATH" in msg and "-lcudart" in msg
    assert "will NOT fall back to a bf16 Linear" in msg
    assert "test_ommx_linear_parity.py" in msg
    # LATCHED: a second layer gets the same message, not a second confusing traceback
    # from a half-initialised torch extension directory.
    with pytest.raises(lm.OMMXWError) as exc2:
        lm.build_kernel()
    assert str(exc2.value) == msg
    assert len(calls) == 1
    assert lm.ommx_w_health()["ok"] is False
    assert "FAILED TO BUILD" in lm.ommx_w_health()["reason"]


def test_process_weights_after_loading_propagates_the_build_failure(fake_vllm, bundle,
                                                                    bundle_manifests,
                                                                    monkeypatch):
    cfg, method = _make_layer_and_method(bundle)
    layer = _StubLinearBase()
    N, K = bundle_manifests[_Q_PROJ]["shape"]
    method.create_weights(layer, K, [N], K, N, torch.bfloat16)
    monkeypatch.setattr(lm, "_load_build_module", lambda: (_ for _ in ()).throw(
        RuntimeError("/usr/bin/ld: cannot find -lcudart")))
    with pytest.raises(lm.OMMXWError) as exc:
        method.process_weights_after_loading(layer)
    assert "LIBRARY_PATH" in str(exc.value)
    assert layer.ommx_w_ready is False              # never claims readiness on failure


def test_missing_kernel_sources_are_reported_by_path(monkeypatch, tmp_path):
    """An installed wheel without csrc/ must say so, not fail as a mystery ImportError."""
    monkeypatch.setattr(lm.os.path, "isfile", lambda p: False)
    with pytest.raises(lm.OMMXWError) as exc:
        lm._load_build_module()
    msg = str(exc.value)
    assert "build_ommx_linear.py" in msg
    assert "source checkout" in msg


def test_apply_refuses_before_process_weights(fake_vllm, bundle, bundle_manifests):
    cfg, method = _make_layer_and_method(bundle)
    layer = _StubLinearBase()
    N, K = bundle_manifests[_Q_PROJ]["shape"]
    method.create_weights(layer, K, [N], K, N, torch.bfloat16)
    with pytest.raises(lm.OMMXWError) as exc:
        method.apply(layer, torch.zeros(1, K, dtype=torch.bfloat16))
    assert "before process_weights_after_loading()" in str(exc.value)


def test_apply_refuses_a_non_bf16_activation(fake_vllm, bundle, bundle_manifests):
    """Checked BEFORE the kernel is touched, so it is reachable with no GPU."""
    cfg, method = _make_layer_and_method(bundle)
    layer = _StubLinearBase()
    N, K = bundle_manifests[_Q_PROJ]["shape"]
    method.create_weights(layer, K, [N], K, N, torch.bfloat16)
    layer.ommx_w_ready = True                       # skip the build; test the dtype gate
    with pytest.raises(lm.OMMXWError) as exc:
        method.apply(layer, torch.zeros(1, K, dtype=torch.float16))
    msg = str(exc.value)
    assert "bfloat16" in msg
    assert "--dtype bfloat16" in msg


# ══════════════════════════════════════════════════════════════════════════════
# name remap (the KNOWN GAP) + firing evidence
# ══════════════════════════════════════════════════════════════════════════════

def test_bundle_plane_names_remap_to_parameter_names(bundle):
    """The bundle plane name IS the parameter name — the remap is now a no-op.

    OMMX_W_SafeTensor v2 dropped the ``.weight`` infix (``w_format.plane_name``), so
    ``<w>.weight.ommx_code`` no longer exists on disk and ``bundle_to_param_name`` has
    nothing left to rewrite. This gate is kept, and tightened, rather than deleted:
    it now pins the invariant that actually matters — a plane read out of a bundle
    binds to the parameter ``create_weights`` registered, with no translation step —
    and it fails if either side drifts back.
    """
    on_disk = plane_name(_Q_PROJ, "code")
    assert on_disk == "model.layers.0.self_attn.q_proj.ommx_code"
    assert ".weight.ommx_" not in on_disk
    # module path + the attribute create_weights registers, joined by a single dot.
    assert on_disk == "model.layers.0.self_attn.q_proj." + lm.param_name_for_plane("code")
    assert lm.bundle_to_param_name(on_disk) == on_disk
    # Non-plane tensors (passthrough weights) are untouched.
    assert lm.bundle_to_param_name(_LM_HEAD) == _LM_HEAD
    assert lm.bundle_to_param_name("model.norm.weight") == "model.norm.weight"
    remapped = dict(lm.remap_bundle_weight_names(
        [(plane_name(_Q_PROJ, p), p) for p in ("code", "scale_exp", "zp")]))
    assert set(remapped) == {"model.layers.0.self_attn.q_proj.ommx_code",
                             "model.layers.0.self_attn.q_proj.ommx_scale_exp",
                             "model.layers.0.self_attn.q_proj.ommx_zp"}


def test_health_starts_unproven_and_says_why():
    """A bench must be able to tell "OMMX linear ran" from "something produced logits"."""
    health = lm.ommx_w_health()
    assert health["ok"] is False
    assert "apply() never ran" in health["reason"]
    stats = health["stats"]
    assert stats["apply_calls"] == 0 and stats["outlier_calls"] == 0
    assert stats["kernel_built"] is False


def test_fire_sentinel_is_written_where_the_attention_path_writes(tmp_path,
                                                                  monkeypatch):
    """Same ``$OMMX_FIRE_FILE`` as backend.py, so one file describes a combined arm."""
    path = tmp_path / "fire.log"
    monkeypatch.setenv("OMMX_FIRE_FILE", str(path))
    lm._fire("OMMX_W_TEST_FIRED", "detail=1")
    lm._fire("OMMX_W_TEST_FIRED", "detail=1")       # deduped per (tag, pid)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("OMMX_W_TEST_FIRED rank=")
    # The suffix rule bench_e2e_a100.py matches on: never _DEAD / _NOFIRE for a
    # non-failure, and these informational tags are NOT in its FIRED_TAGS whitelist,
    # so they cannot change the attention verdict.
    from ommx_gpu_serve.bench import bench_e2e_a100 as bench
    tag = lines[0].split(" ", 1)[0]
    assert not bench._is_nofire_tag(tag)
    assert tag not in bench.FIRED_TAGS


def test_env_int_refuses_a_typo(monkeypatch):
    """A malformed knob must raise, not silently become the default (law #11's lesson)."""
    monkeypatch.setenv("OMMX_W_SPLIT", "two")
    with pytest.raises(lm.OMMXWError):
        lm._env_int("OMMX_W_SPLIT", 1)
    monkeypatch.delenv("OMMX_W_SPLIT")
    assert lm._env_int("OMMX_W_SPLIT", 1) == 1


# ══════════════════════════════════════════════════════════════════════════════
# the bench verdict must gate the WEIGHT axis too (pure file parsing — CPU-testable)
# ══════════════════════════════════════════════════════════════════════════════

def _ev(tmp_path, tags, **kw):
    """Run bench._read_fire_evidence over a hand-written sentinel."""
    from ommx_gpu_serve.bench import bench_e2e_a100 as bench
    f = tmp_path / "arm.fire.log"
    f.write_text("".join(f"{t} rank=0 pid=1234\n" for t in tags))
    return bench._read_fire_evidence(str(f), **kw)


def test_an_ommx_w_arm_needs_an_ommx_w_fired_tag(tmp_path):
    """quant='ommx_w' + FLASH_ATTN (the abl_linear arm) has NO attention tag by design.

    Before this gate that arm got no sentinel and no verdict at all, so its TPOT was
    published under an "ommx_w" label with nothing showing the OMMX linear kernel ran.
    """
    ok = _ev(tmp_path, ["OMMX_W_DECODE_FIRED"], require_attn=False, require_linear=True)
    assert ok["ok"] is True
    assert ok["linear_fired"] == ["OMMX_W_DECODE_FIRED"]

    bad = _ev(tmp_path, ["OMMX_W_KERNEL_BUILT", "OMMX_W_WEIGHTS_READY"],
              require_attn=False, require_linear=True)
    assert bad["ok"] is False
    assert "no OMMX_W_*_FIRED tag" in bad["reason"]
    # A BUILT kernel and READY weights are not an EXECUTED kernel.
    assert bad["linear_fired"] == []


def test_a_fired_attention_route_is_not_proof_the_ommx_linear_ran(tmp_path):
    """ommx_w + CUSTOM needs BOTH tags: the axes are independent."""
    only_attn = _ev(tmp_path, ["DECODE_ROUTE_FIRED"], require_attn=True,
                    require_linear=True)
    assert only_attn["ok"] is False
    assert "OMMXWLinearMethod.apply() never ran" in only_attn["reason"]

    both = _ev(tmp_path, ["DECODE_ROUTE_FIRED", "OMMX_W_PREFILL_FIRED"],
               require_attn=True, require_linear=True)
    assert both["ok"] is True


def test_the_attention_only_verdict_is_unchanged(tmp_path):
    """No regression on the KV arms: they still need exactly what they needed."""
    assert _ev(tmp_path, ["DECODE_ROUTE_FIRED"])["ok"] is True
    dead = _ev(tmp_path, ["DECODE_ROUTE_FIRED", "KV_UPDATE_DEAD"])
    assert dead["ok"] is False and dead["nofire"] == ["KV_UPDATE_DEAD"]
    # An ommx_w tag alone never satisfies an ATTENTION arm.
    assert _ev(tmp_path, ["OMMX_W_DECODE_FIRED"])["ok"] is False


# ══════════════════════════════════════════════════════════════════════════════
# the load-time TRANSCODE — making the paper's own weight format servable
#
# The gates are ordered as the argument runs: (1) the premise the whole transcode
# rests on, checked against the packer's real output; (2) the refusal is unchanged
# without the opt-in; (3) the re-encoding is byte-identical to packing relidx7
# directly; (4) the numbers do not move; (5) both footprints are reported and correct;
# (6) the hard refusals survive the opt-in; (7) the opt-in cannot be mistaken for
# proof that anything was served.
# ══════════════════════════════════════════════════════════════════════════════

_PAPER_ON_DISK_BPW = 3.6250      # gs=64, npv=4, bitmap + no map  — Table 1's AvgBits 3.63
_PAPER_RESIDENT_BPW = 4.1250     # gs=64, npv=4, relidx7 + idx_range — what the kernel reads


def _planes_of(bundle_dir: str, weight: str = _Q_PROJ):
    """``(planes, recipe, N, K)`` for one weight, read straight out of a bundle."""
    from ommx_gpu_serve.linear.w_format import load_planes, read_manifest
    with open(os.path.join(bundle_dir, INDEX_FILENAME), encoding="utf-8") as fh:
        index = json.load(fh)
    shard = os.path.join(bundle_dir, index["weight_map"][weight])
    return load_planes(shard, weight, read_manifest(shard))


def test_bitmap_and_relidx7_differ_in_exactly_one_plane(bundle_paper, bundle_nomap):
    """THE PREMISE, checked against the packer — not assumed from the docstrings.

    The transcode is only legitimate because a bitmap row and a relidx7 slot stream are
    two encodings of the SAME ascending position set, and because the FP4 nibble stream
    is SHARED — ``quantize_ommx_weight`` builds ``ocode`` from
    ``gather(nib_g, pos_sorted)`` OUTSIDE the ``outlier_repr`` branch, so slot ``s`` is
    the s-th outlier in ascending position order either way. That is exactly what the
    kernel's ``relidx7_slot_delta`` assumes (``ocode_blk[s >> 1]`` paired with
    ``relidx7_slot_pos(oindex_blk, s)``).

    If that were ever to stop being true for the WEIGHT path, the transcode would splice
    the right positions with the wrong values and produce a plausible, wrong model. So it
    is pinned here, on two REAL bundles packed from the same checkpoint by the real
    packer, plane by plane.
    """
    paper, r_paper, N, K = _planes_of(bundle_paper)
    relidx, r_rel, N2, K2 = _planes_of(bundle_nomap)
    assert (N, K) == (N2, K2)
    assert r_paper.outlier_repr == "bitmap" and r_rel.outlier_repr == "relidx7"
    assert r_paper.outlier_map == "none" and r_rel.outlier_map == "none"
    assert set(paper) == set(relidx)
    differ = [p for p in paper if not torch.equal(paper[p], relidx[p])]
    assert differ == ["oindex"], (
        f"only the position plane may differ between the two encodings; got {differ}")
    # And the shapes differ exactly as the two encodings predict: ceil(gs/8) vs
    # ceil(npv*7/8) bytes per group.
    assert list(paper["oindex"].shape)[-1] == (_GROUP_SIZE + 7) // 8      # 8
    assert list(relidx["oindex"].shape)[-1] == (_NPV * 7 + 7) // 8        # 4


def test_the_paper_bundle_is_still_refused_without_the_opt_in(bundle_paper):
    """Nothing changed for an operator who did not ask for anything.

    The refusal is the same refusal; it merely learned to name the opt-in, both bit
    budgets, and the native fix. The load must still raise.
    """
    with pytest.raises(lm.OMMXWError) as exc:
        lm.load_bundle(bundle_paper)
    msg = str(exc.value)
    assert "NO bitmap reader" in msg                       # the original reason, intact
    assert "--outlier-repr relidx7" in msg                 # the original fix, intact
    assert lm.TRANSCODE_ENV in msg                         # the new alternative
    assert f"{_PAPER_ON_DISK_BPW:.4f} bits/weight ON DISK" in msg
    assert f"{_PAPER_RESIDENT_BPW:.4f}" in msg
    assert "NOT serving at" in msg                         # the caveat, in the refusal
    assert "BITMAP_READER_SPEC.md" in msg                  # the native fix
    # And the recipe object itself still reports a plan only when asked.
    from ommx_gpu_serve.linear.w_format import Recipe as _R
    paper = _R(_GROUP_SIZE, _NPV, 0.0625, outlier_repr="bitmap", outlier_map="none",
               zp_dtype=torch.bfloat16)
    assert lm.require_kernel_readable(paper, True) is not None


def test_transcode_opt_in_has_exactly_two_explicit_sources(monkeypatch, bundle_paper):
    """Env or quantization_config. No heuristic, and off unless someone said so."""
    assert lm.transcode_requested() is False
    assert lm.transcode_requested({}) is False
    monkeypatch.setenv(lm.TRANSCODE_ENV, "1")
    assert lm.transcode_requested() is True
    monkeypatch.setenv(lm.TRANSCODE_ENV, "0")
    assert lm.transcode_requested() is False
    # A config key wins over an unset env, and its string spellings parse the same way.
    assert lm.transcode_requested({"ommx_w_transcode": True}) is True
    assert lm.transcode_requested({"transcode": "yes"}) is True
    assert lm.transcode_requested({"ommx_w_transcode": "off"}) is False
    # The env route makes the bundle load; that is the whole user-visible effect.
    monkeypatch.setenv(lm.TRANSCODE_ENV, "1")
    b = lm.load_bundle(bundle_paper)
    assert b.transcode is not None
    assert b.transcode.source.outlier_repr == "bitmap"
    assert b.transcode.resident.outlier_repr == "relidx7"


def _paper_layer(bundle_dir, monkeypatch, *, transcode=True):
    """Build a layer, fill it with the bundle's REAL planes, run process_weights.

    ``build_kernel`` is stubbed out: it is the one step that needs a CUDA device, and
    every plane decision under test happens on either side of it. Nothing else is faked —
    the planes come off the packer's own shards.
    """
    cfg = lm.ommx_w_config_class().from_config(
        {"quant_method": "ommx_w", "bundle": bundle_dir, "ommx_w_transcode": transcode})
    method = cfg.get_quant_method(_StubLinearBase(), "model.layers.0.self_attn.o_proj")
    planes, recipe, N, K = _planes_of(bundle_dir)
    layer = _StubLinearBase()
    method.create_weights(layer, K, [N], K, N, torch.bfloat16)
    for plane, tensor in planes.items():
        getattr(layer, lm.param_name_for_plane(plane)).data.copy_(tensor)
    monkeypatch.setattr(lm, "build_kernel", lambda *a, **k: None)
    method.process_weights_after_loading(layer)
    return cfg, method, layer, planes, N, K


def test_transcode_produces_the_relidx7_stream_byte_for_byte(bundle_paper, bundle_nomap,
                                                             bundle, fake_vllm,
                                                             monkeypatch, capsys):
    """The gate the whole feature stands on: re-encode == pack relidx7 directly.

    Not "decodes to the same positions" — BYTE-IDENTICAL to the plane the packer would
    have written for the same weight with ``--outlier-repr relidx7``. Anything weaker
    would leave room for a padding or bit-order difference that only shows up as wrong
    numbers inside a kernel nobody can run here.
    """
    _cfg, _m, layer, planes, N, K = _paper_layer(bundle_paper, monkeypatch)
    got = layer.ommx_w_oindex_k
    for other in (bundle_nomap, bundle):
        want = _planes_of(other)[0]["oindex"]
        assert got.dtype == want.dtype and got.shape == want.shape
        assert torch.equal(got, want), f"transcoded stream differs from {other}"
    # The FP4 nibble plane was NOT touched — it is shared by construction.
    assert torch.equal(getattr(layer, lm.param_name_for_plane("ocode")).data,
                       planes["ocode"])
    # The source bitmap plane's storage is released, so the resident figure below is not
    # an understatement of what is actually held.
    assert getattr(layer, lm.param_name_for_plane("oindex")).data.numel() == 0
    # The synthesised map is the degenerate instance 'none' already means, in f32 because
    # sparse_correct dereferences both with data_ptr<float>().
    assert layer.ommx_w_map_scale_k.dtype is torch.float32
    assert torch.equal(layer.ommx_w_map_scale_k, torch.ones(N, K // _GROUP_SIZE))
    assert torch.equal(layer.ommx_w_map_center_k, torch.zeros(N, K // _GROUP_SIZE))
    # LOUD: the operator cannot miss that the format changed, or what it cost.
    err = capsys.readouterr().err
    assert "TRANSCODE ACTIVE" in err
    assert f"{_PAPER_ON_DISK_BPW:.4f} b/wt" in err and f"{_PAPER_RESIDENT_BPW:.4f} b/wt" in err
    assert "Do NOT report this run at the bundle's on-disk bits/weight" in err


def test_transcode_leaves_the_dequantized_weight_identical(bundle_paper, bundle_nomap,
                                                           fake_vllm, monkeypatch):
    """Lossless means BIT-EQUAL weights, checked through the CPU oracle both ways.

    The oracle (``dequantize_ommx_weight``) is fed (a) the bundle's own bitmap planes and
    (b) the transcoded relidx7 planes the kernel would read, and the two reconstructions
    must be identical to each other AND to the independently-packed relidx7 bundle.
    """
    from ommx_gpu_serve.linear.quantize import dequantize_ommx_weight
    from ommx_gpu_serve.linear.w_format import load_weight

    _cfg, _m, layer, planes, N, K = _paper_layer(bundle_paper, monkeypatch)

    from_bitmap = dequantize_ommx_weight(
        code=planes["code"], scale_exp=planes["scale_exp"],
        zp=planes["zp"].to(torch.float32), N=N, K=K, group_size=_GROUP_SIZE, npv=_NPV,
        oindex=planes["oindex"], ocode=planes["ocode"], outlier_repr="bitmap")
    from_transcoded = dequantize_ommx_weight(
        code=getattr(layer, lm.param_name_for_plane("code")).data,
        scale_exp=getattr(layer, lm.param_name_for_plane("scale_exp")).data,
        zp=layer.ommx_w_zp_f32, N=N, K=K, group_size=_GROUP_SIZE, npv=_NPV,
        oindex=layer.ommx_w_oindex_k, ocode=planes["ocode"],
        map_scale=layer.ommx_w_map_scale_k, map_center=layer.ommx_w_map_center_k,
        outlier_repr="relidx7")
    assert torch.equal(from_bitmap, from_transcoded)

    with open(os.path.join(bundle_nomap, INDEX_FILENAME), encoding="utf-8") as fh:
        idx = json.load(fh)
    ref = load_weight(os.path.join(bundle_nomap, idx["weight_map"][_Q_PROJ]), _Q_PROJ)
    assert torch.equal(from_transcoded, ref)


def test_transcode_reports_both_footprints_and_they_are_correct(bundle_paper, fake_vllm,
                                                                monkeypatch):
    """On-disk AND resident, both computed, both checked against the real tensors.

    The bits/weight pair is the paper's 3.6250 vs the relidx7 4.1250. The BYTE totals are
    checked the hard way — against the actual tensors the kernel will be handed — because
    a plan that agreed with itself and disagreed with reality is precisely the accounting
    error this repo has already shipped once on the KV side.
    """
    _cfg, _m, layer, planes, N, K = _paper_layer(bundle_paper, monkeypatch)
    stats = lm.ommx_w_fire_stats()["transcode"]
    assert stats["layers"] == 1 and lm.ommx_w_fire_stats()["transcoded_layers"] == 1
    plan = stats["plan"]
    assert plan["on_disk_bits_per_weight"] == _PAPER_ON_DISK_BPW
    assert plan["resident_bits_per_weight"] == _PAPER_RESIDENT_BPW
    assert plan["delta_bits_per_weight"] == pytest.approx(0.5)
    assert len(plan["steps"]) == 2                       # positions AND the missing map

    def _nbytes(t):
        return 0 if t is None else t.numel() * t.element_size()

    # ON DISK: exactly the bytes the bundle holds for this weight.
    assert stats["on_disk_bytes"] == sum(_nbytes(t) for t in planes.values())
    assert stats["on_disk_bytes"] * 8 / (N * K) == _PAPER_ON_DISK_BPW
    # RESIDENT: exactly the bytes of the planes the kernel is handed. code / scale_exp /
    # zp are the parameters as loaded; the position stream and the map are the transcode's
    # output. The fp32 scale/zp twins are outside both figures by definition (they exist
    # identically with and without a transcode) — see TranscodePlan's docstring.
    resident = sum(_nbytes(getattr(layer, lm.param_name_for_plane(p)).data)
                   for p in ("code", "scale_exp", "zp", "ocode"))
    resident += _nbytes(layer.ommx_w_oindex_k)
    resident += _nbytes(layer.ommx_w_map_scale_k) + _nbytes(layer.ommx_w_map_center_k)
    assert stats["resident_bytes"] == resident
    assert stats["resident_bytes"] * 8 / (N * K) == _PAPER_RESIDENT_BPW
    # The bitmap plane really was released, by its full size.
    assert stats["freed_bytes"] == _nbytes(planes["oindex"])
    assert stats["resident_bytes"] > stats["on_disk_bytes"]     # the transcode COSTS HBM


def test_without_the_opt_in_the_kernel_planes_are_the_parameters_themselves(
        bundle, fake_vllm, monkeypatch):
    """ADDITIVE: with no transcode the resolution step is a rebind, not a change.

    Asserted by STORAGE ADDRESS (``data_ptr``), which is stronger than byte equality and
    is the only form that rules out a copy, a cast or a reallocation slipping into the
    shipped path. (``Parameter.data`` hands back a fresh view object on every access, so
    ``is`` would compare wrappers rather than memory and could never hold.) The transcode
    accounting must also stay at zero.
    """
    _cfg, _m, layer, planes, N, K = _paper_layer(bundle, monkeypatch, transcode=False)
    assert layer.ommx_w_transcode is None
    for attr, plane in (("ommx_w_oindex_k", "oindex"),
                        ("ommx_w_map_scale_k", "map_scale"),
                        ("ommx_w_map_center_k", "map_center")):
        got = getattr(layer, attr)
        want = getattr(layer, lm.param_name_for_plane(plane)).data
        assert got.data_ptr() == want.data_ptr(), plane      # same memory, not a copy
        assert got.dtype is want.dtype and got.shape == want.shape, plane
    assert getattr(layer, lm.param_name_for_plane("oindex")).data.numel() > 0
    stats = lm.ommx_w_fire_stats()["transcode"]
    assert stats == {"plan": None, "layers": 0, "on_disk_bytes": 0,
                     "resident_bytes": 0, "freed_bytes": 0, "twin_bytes": 0}
    assert lm.ommx_w_fire_stats()["transcoded_layers"] == 0
    # ... and even ASKING for a transcode on an already-readable bundle is a no-op.
    lm._reset_state_for_tests()
    assert lm.load_bundle(bundle, allow_transcode=True).transcode is None


def test_recipes_the_transcode_cannot_reach_are_refused_even_with_the_opt_in():
    """The opt-in buys a re-encoding, not a suspension of the kernel's limits.

    Two hard refusals, both read off ``ommx_linear.cu`` rather than chosen:
      * ``relidx7_slot_pos`` returns ``v & 0x7F`` -> 7 bits -> no group wider than 128;
      * ``sparse_correct`` asserts ``npv > 0 && npv <= 32`` (MAX_NPV) host-side.
    A transcode that ignored either would move a loud kernel abort to the first served
    token, or (worse, for the width case) truncate every position silently.
    """
    from ommx_gpu_serve.linear.w_format import Recipe as _R
    wide = _R(256, 4, 0.0, outlier_repr="bitmap", outlier_map="none")
    with pytest.raises(lm.OMMXWError) as exc:
        lm.require_kernel_readable(wide, True)
    assert "7 bits of position" in str(exc.value)
    assert "at most 128" in str(exc.value)

    many = _R(128, 64, 0.5, outlier_repr="bitmap", outlier_map="none")
    with pytest.raises(lm.OMMXWError) as exc2:
        lm.require_kernel_readable(many, True)
    assert "MAX_NPV=32" in str(exc2.value)
    assert "npv=64" in str(exc2.value)


def test_the_resident_figure_is_a_delta_not_an_hbm_total(bundle_paper, fake_vllm,
                                                        monkeypatch, capsys):
    """The two published bit budgets count BUNDLE PLANES ONLY — so neither is HBM.

    Regression gate for a wording defect found by audit. ``process_weights_after_loading``
    materialises fp32 ``scale``/``zp`` twins and ``apply()`` hands the KERNEL those twins,
    not the int8 ``scale_exp`` / bf16 ``zp`` planes that the 4.1250 figure counts. So at
    ``gs=64, npv=4`` the real numbers are:

        bundle planes (the reported "resident")            4.1250 b/wt
        + fp32 scale/zp twins (32+32 bits per group)       1.0000 b/wt
        = actually in HBM                                  5.1250 b/wt
        of which the kernel actually streams               4.7500 b/wt
        (scale_exp 0.125 + bf16 zp 0.25 are held but never read on the default path)

    Excluding the twins from the PAIR is right — they are identical with and without a
    transcode, so counting them would inflate the delta without changing it. Printing
    "the kernel streams the resident figure above" was not: that sentence made 4.1250
    readable as a memory-traffic number, which is the precise class of claim this feature
    exists to prevent. The twins are therefore measured off the allocated tensors, carried
    in ``ommx_w_fire_stats()["transcode"]["twin_bytes"]``, and named in the banner.
    """
    _cfg, _m, layer, planes, N, K = _paper_layer(bundle_paper, monkeypatch)
    stats = lm.ommx_w_fire_stats()["transcode"]

    def _nbytes(t):
        return t.numel() * t.element_size()

    twins = _nbytes(layer.ommx_w_scale_f32) + _nbytes(layer.ommx_w_zp_f32)
    G = K // _GROUP_SIZE
    assert twins == G * N * 4 * 2                      # two f32 planes, longhand
    assert stats["twin_bytes"] == twins
    assert twins * 8 / (N * K) == 1.0                  # +1.0000 b/wt at gs=64
    # …and the twins are NOT inside the reported pair, which is what makes it a delta.
    assert stats["resident_bytes"] == sum(
        _nbytes(getattr(layer, lm.param_name_for_plane(p)).data)
        for p in ("code", "scale_exp", "zp", "ocode")) + _nbytes(layer.ommx_w_oindex_k) \
        + _nbytes(layer.ommx_w_map_scale_k) + _nbytes(layer.ommx_w_map_center_k)
    assert (stats["resident_bytes"] + twins) * 8 / (N * K) == 5.125

    err = capsys.readouterr().err
    assert "NOT an HBM total" in err
    assert "+1.0000 b/wt" in err                       # the twins, signed
    assert "real HBM for this layer is 5.1250 b/wt" in err
    # The sentence that made 4.1250 readable as memory traffic must not come back.
    assert "the kernel streams the resident figure above" not in err


def test_npv_above_the_kernel_bound_is_refused_for_the_ENCODING_THE_KERNEL_READS_too():
    """The kernel's MAX_NPV is a bound on the KERNEL, so it cannot be a transcode-only
    check. Regression gate for a hole found by audit.

    ``linear_method.apply()`` calls ``sparse_correct`` for EVERY recipe that has
    outliers, and ``csrc/linear/ommx_linear.cu:2346`` asserts
    ``TORCH_CHECK(npv > 0 && npv <= 32, "npv must be in (0,32] (MAX_NPV)")``. Meanwhile
    ``Recipe.validate`` only requires ``npv <= group_size``, so the packer emits
    ``(gs=64, npv=40, relidx7, idx_range)`` without complaint — a bundle that needs NO
    transcode at all. While the MAX_NPV check sat AFTER ``plan_transcode``'s
    ``if not steps: return None``, that bundle loaded clean and then aborted on its
    FIRST SERVED TOKEN, while the byte-for-byte equivalent bitmap bundle was refused at
    load with an actionable message. Same kernel, opposite verdicts.

    The gate is deliberately DISCRIMINATIVE — npv=32 (the boundary the kernel itself
    allows) must still be accepted, so this cannot degrade into a blanket refusal.
    """
    from ommx_gpu_serve.linear import w_packer

    # 1. the hole is REACHABLE: the packer really will build this recipe.
    over = w_packer.build_recipe(group_size=64, outlier_pct=None, npv=40,
                                 outlier_repr="relidx7", outlier_map="idx_range",
                                 zp_dtype="bf16")
    assert over.outlier_repr == "relidx7" and over.outlier_map == "idx_range"

    # 2. it is refused at LOAD, with or without the opt-in, and the message names the
    #    kernel bound and the offending value rather than describing a transcode.
    for allow in (False, True):
        with pytest.raises(lm.OMMXWError) as exc:
            lm.require_kernel_readable(over, allow)
        msg = str(exc.value)
        assert "MAX_NPV=32" in msg and "npv=40" in msg, msg
        assert "--npv" in msg                       # names the fix

    # 3. …and the bound is the kernel's own, not a stricter invention: npv=32 passes,
    #    and passes as "nothing to do" rather than as a transcode.
    at_bound = w_packer.build_recipe(group_size=64, outlier_pct=None, npv=32,
                                     outlier_repr="relidx7", outlier_map="idx_range",
                                     zp_dtype="bf16")
    assert lm.require_kernel_readable(at_bound, False) is None
    # 4. the shipped canonical recipe is untouched by all of this.
    shipped = w_packer.build_recipe(group_size=_GROUP_SIZE, outlier_pct=None, npv=_NPV,
                                    outlier_repr="relidx7", outlier_map="idx_range",
                                    zp_dtype="bf16")
    assert lm.require_kernel_readable(shipped, False) is None


def test_the_transcode_tag_is_not_evidence_that_the_linear_kernel_ran(bundle_paper,
                                                                      fake_vllm,
                                                                      tmp_path,
                                                                      monkeypatch):
    """A load-time transcode proves a format was accepted, not that a token was served.

    ``bench_e2e_a100`` counts ANY ``OMMX_W_*_FIRED`` tag as proof ``apply()`` ran, so the
    transcode tag deliberately ends in ``_APPLIED``. If it were ever renamed to
    ``..._FIRED`` an ``ommx_w`` arm that loaded a bundle and then served NOTHING would
    pass its own verdict, which is the exact defect class this release removed.

    Driven through a REAL transcode (not a hand-written sentinel), so the tag under test
    is the one the serving path actually writes.
    """
    from ommx_gpu_serve.bench import bench_e2e_a100 as bench
    path = tmp_path / "fire.log"
    monkeypatch.setenv("OMMX_FIRE_FILE", str(path))
    _paper_layer(bundle_paper, monkeypatch)

    tags = [line.split(" ", 1)[0] for line in path.read_text().strip().splitlines()]
    assert "OMMX_W_TRANSCODE_APPLIED" in tags, tags
    for tag in tags:
        assert not bench._is_ommx_w_fired_tag(tag), (
            f"{tag} was written at LOAD time but reads as proof that apply() ran")
        assert not bench._is_nofire_tag(tag)     # nor is any of it a route FAILURE
    ev = bench._read_fire_evidence(str(path), require_attn=False, require_linear=True)
    assert ev["ok"] is False
    assert "no OMMX_W_*_FIRED tag" in ev["reason"]


# ══════════════════════════════════════════════════════════════════════════════
# sparse_correct's M>1 call convention — the defect the M=1-only parity gate hid
# ══════════════════════════════════════════════════════════════════════════════

#: What the spy's ``sparse_correct`` adds into ``out_or_C``, standing in for the real
#: correction (which needs a device). Its ONLY job is to be observable: the M>1 kernels'
#: whole effect is the IN-PLACE ``C[idx] = __float2bfloat16(__bfloat162float(C[idx]) +
#: acc)``, so a caller that writes the correction into a tensor it then discards is
#: indistinguishable from a correct one unless the spy actually mutates the buffer it
#: was handed. Exactly representable in fp32 and bf16, and non-zero so it cannot alias
#: the ``torch.zeros`` base ``decode_base`` returns.
_CORRECTION_SENTINEL = 8.0


class _KernelABISpy:
    """A transcription of what ``csrc/linear/ommx_linear.cu`` DEREFERENCES, in Python.

    This is not a mock that accepts anything and records it. Every check below is a
    line of the .cu, quoted in its own comment, and the failure text is the kernel's
    own so a broken call site reads here exactly as it would read on a device:

      * ``decode_base``  — ``check_contig_cuda(A/code/scale)``; returns
        ``torch::zeros({M, N}, A.options().dtype(torch::kFloat32))`` -> **fp32**.
      * ``prefill_wmma`` — ``TORCH_CHECK(M >= 16, ...)``, ``code`` uint8, ``scale``
        fp32; returns ``torch::empty({M, N}, A.options())`` -> **bf16**.
      * ``sparse_correct`` — ``npv in (0, 32]``, ``fmt in {i2f4, i2}``, and then the
        M-dependent split that this whole section exists for.

    What it deliberately does NOT do is compute the answer: the numbers are the parity
    gate's job (``csrc/linear/test_ommx_linear_parity.py``, which now has M>1 cases).
    This gate is about the ABI, and an ABI is checkable without a GPU.
    """

    def __init__(self):
        self.calls = []

    # ommx_linear.cu:2075 decode_base(A, code, scale, zp_opt, N, K, vector_length,
    #                                 symmetric, fmt, num_k_splits, scale_exp_opt)
    def decode_base(self, A, code, scale, zp, N, K, vector_length, symmetric, fmt,
                    num_k_splits, scale_exp=None):
        assert A.dtype is torch.bfloat16 and A.is_contiguous()   # data_ptr<at::BFloat16>
        assert code.dtype is torch.uint8 and code.is_contiguous()
        assert scale.dtype is torch.float32 and scale.is_contiguous()
        if zp is not None:
            assert zp.dtype is torch.float32                     # data_ptr<float>()
        if scale_exp is not None:
            assert scale_exp.dtype is torch.int8                 # data_ptr<int8_t>()
        assert fmt in ("i2f4", "i2")                             # check_fmt
        M = int(A.shape[0])
        # decode_gemv_batched_kernel is instantiated at M_MAX <= 16 and its epilogue
        # guards `m < M_MAX`, so M > 16 would leave rows 16.. as the zeros this
        # torch::zeros put there. apply() must never route such an M here.
        assert M <= 16, f"decode_base handed M={M}; batched kernel tops out at M_MAX=16"
        out = torch.zeros((M, N), dtype=torch.float32)           # kFloat32, NOT A's dtype
        self.calls.append(dict(fn="decode_base", A=A, M=M, out=out, scale_exp=scale_exp))
        return out

    # ommx_linear.cu:2398 prefill_wmma(A, code, scale, zp_opt, N, M, K, vector_length,
    #                                  symmetric, fmt, splitk, oindex_opt, odelta_opt,
    #                                  npv, B_blk)
    def prefill_wmma(self, A, code, scale, zp, N, M, K, vector_length, symmetric, fmt,
                     splitk, oindex=None, odelta=None, npv=0, B_blk=0):
        assert A.dtype is torch.bfloat16 and A.is_contiguous()
        assert code.dtype is torch.uint8                         # TORCH_CHECK code uint8
        assert scale.dtype is torch.float32                      # TORCH_CHECK scale fp32
        assert fmt in ("i2f4", "i2")
        assert M >= 16, "prefill_wmma needs M>=16 (m16n8k16 16-row tile)"
        assert int(A.shape[0]) == M, "M argument must be A.size(0)"
        # A.options() -> bf16. torch::empty in the .cu, but ZEROED here on purpose: the
        # real kernel writes every element of that allocation, whereas torch.empty on
        # this host hands back reused memory that can contain NaN — and a NaN in the
        # base makes any value assertion about the correction non-deterministic
        # (torch.equal is False for NaN even against itself). The dtype, shape and
        # object identity — the things the ABI is about — are unaffected.
        out = torch.zeros((M, N), dtype=A.dtype)
        self.calls.append(dict(fn="prefill_wmma", A=A, M=M, out=out))
        return out

    # ommx_linear.cu:2340 sparse_correct(out_or_C, A_opt, At_opt, code, scale, oindex,
    #                                    ocode, map_scale, map_center, N, M, K,
    #                                    vector_length, npv, B, fmt)
    def sparse_correct(self, out_or_C, A_opt, At_opt, code, scale, oindex, ocode,
                       map_scale, map_center, N, M, K, vector_length, npv, B, fmt):
        assert 0 < npv <= 32, f"npv must be in (0,32] (MAX_NPV); got {npv}"
        assert fmt in ("i2f4", "i2")
        assert code.dtype is torch.uint8
        for t in (scale, map_scale, map_center):
            assert t.dtype is torch.float32                      # data_ptr<float>()
        for t in (oindex, ocode):
            assert t.dtype is torch.uint8                        # data_ptr<uint8_t>()
        if M == 1:
            # `if (M == 1) { TORCH_CHECK(A_opt.has_value(), "M==1 correct needs A"); ... }`
            assert A_opt is not None, "M==1 correct needs A"
            # go1: `A_opt->data_ptr<at::BFloat16>()` and `out_or_C.data_ptr<float>()`
            assert A_opt.dtype is torch.bfloat16
            assert out_or_C.dtype is torch.float32, (
                "expected scalar type Float but found " + str(out_or_C.dtype))
            assert A_opt.numel() >= K, "kernel reads A[k] for k < K"
        else:
            # `else { TORCH_CHECK(At_opt.has_value(), "M>1 correct needs At"); ... }`
            assert At_opt is not None, "M>1 correct needs At"
            # goM: `At_opt->data_ptr<at::BFloat16>()` + `out_or_C.data_ptr<at::BFloat16>()`
            assert At_opt.dtype is torch.bfloat16
            assert out_or_C.dtype is torch.bfloat16, (
                "expected scalar type BFloat16 but found " + str(out_or_C.dtype))
            # `At[(size_t)k_sh[s] * M + m]` — a flat k*M+m walk over the WHOLE buffer,
            # which is a [K, M] contiguous tensor and nothing else. The host wrapper does
            # NOT check this (there is no check_contig_cuda(At)), so a strided transpose
            # view would be read as if it were dense and silently give wrong numbers.
            assert tuple(At_opt.shape) == (K, M), (
                f"At must be [K={K}, M={M}], got {tuple(At_opt.shape)}")
            assert At_opt.is_contiguous() and At_opt.stride() == (M, 1), (
                f"At must be contiguous [K,M]; got stride {At_opt.stride()}")
            assert tuple(out_or_C.shape) == (M, N), "C is indexed C[m*N+n]"
            # MAX_STAGE_DENSE = 5120, host-checked before the parallel launch.
            n_blk = (K + B - 1) // B
            assert n_blk * npv <= 5120, "exceeds smem stage bound 5120"
        # The kernel's entire effect is IN PLACE on out_or_C — go1 does `out[n] += corr`
        # and goM does `C[idx] = __float2bfloat16(__bfloat162float(C[idx]) + acc)`. The
        # spy cannot compute `corr` without a device, but it CAN be in-place, and that is
        # what lets a caller which corrects a temporary and returns the uncorrected base
        # be caught on CPU (it otherwise satisfies every dtype/shape/stride assert above).
        out_or_C.add_(_CORRECTION_SENTINEL)
        self.calls.append(dict(fn="sparse_correct", C=out_or_C, A_opt=A_opt,
                               At_opt=At_opt, M=M, N=N, K=K, npv=npv, B=B))

    def fire_stats(self):
        return {}

    def of(self, fn):
        got = [c for c in self.calls if c["fn"] == fn]
        assert len(got) == 1, f"expected exactly one {fn} call, got {len(got)}"
        return got[0]


def _spied_layer(bundle_dir, monkeypatch):
    """A load-complete layer holding the bundle's REAL planes, wired to the ABI spy.

    Only ``build_kernel`` is replaced — every plane, shape and dtype ``apply()`` hands
    the kernel is the one the packer wrote and ``process_weights_after_loading`` derived.
    """
    spy = _KernelABISpy()
    cfg, method = _make_layer_and_method(bundle_dir)
    planes, _recipe, N, K = _planes_of(bundle_dir)
    layer = _StubLinearBase()
    method.create_weights(layer, K, [N], K, N, torch.bfloat16)
    for plane, tensor in planes.items():
        getattr(layer, lm.param_name_for_plane(plane)).data.copy_(tensor)
    monkeypatch.setattr(lm, "build_kernel", lambda *a, **k: spy)
    method.process_weights_after_loading(layer)
    return method, layer, spy, N, K


def _kernel_index_walk(At, M):
    """Rebuild ``A[m, k]`` from ``At`` using the kernel's OWN arithmetic, ``At[k*M+m]``.

    Written as an explicit flat gather rather than ``At.t()`` so the test is checking the
    offsets the CUDA code computes, not torch's opinion of what a transpose means.
    """
    K = At.shape[0]
    flat = At.reshape(-1)
    k = torch.arange(K).view(K, 1).expand(K, M)
    m = torch.arange(M).view(1, M).expand(K, M)
    return flat[(k * M + m).reshape(-1)].view(K, M).t().contiguous()


@pytest.mark.parametrize("M", [17, 32])
def test_apply_at_m_gt_1_uses_the_kernels_m_gt_1_call_convention(bundle, fake_vllm,
                                                                 monkeypatch, M):
    """THE ROOT-CAUSE GATE: prefill with an outlier recipe must not abort.

    Until this release ``apply()`` sent ``sparse_correct`` the M==1 argument list at
    every M, so ``TORCH_CHECK(At_opt.has_value(), "M>1 correct needs At")`` killed every
    prefill step of every recipe with ``npv > 0``. The shipped parity gate passed 5/5
    throughout because its one and only ``sparse_correct`` call site is at M=1 — which is
    why the fix lands here AND as new M>1 cases in that gate.

    Checked, all without a device: ``A_opt`` is NULL, ``At_opt`` is a contiguous
    ``[K, M]`` bf16 buffer that reproduces the activation under the kernel's own
    ``At[k*M+m]`` indexing, and ``C`` is the bf16 tensor ``prefill_wmma`` returned (so
    the in-place ``+=`` lands in the tensor ``apply()`` gives back).
    """
    method, layer, spy, N, K = _spied_layer(bundle, monkeypatch)
    assert method.recipe.has_outliers          # the branch under test is reachable
    x = (torch.randn(M, K) * 0.1).to(torch.bfloat16)
    out = method.apply(layer, x)

    pre, corr = spy.of("prefill_wmma"), spy.of("sparse_correct")
    assert corr["M"] == M and corr["A_opt"] is None
    at = corr["At_opt"]
    assert at.dtype is torch.bfloat16
    assert tuple(at.shape) == (K, M) and at.is_contiguous()
    assert torch.equal(_kernel_index_walk(at, M), pre["A"])      # At IS A^T, densely
    assert corr["C"] is pre["out"] and corr["C"].dtype is torch.bfloat16
    # ... and the in-place `C[m*N+n] += acc` is observable in what apply() hands back.
    # `is` cannot be used: apply() ends with `y.to(x.dtype).reshape(out_shape)` and
    # reshape returns a new Tensor OBJECT over the same storage.
    assert torch.equal(out, corr["C"])
    # prefill_wmma's base is zeroed in the spy, so a correction that reached the caller
    # shows up as exactly the sentinel. Same claim as the batched-decode gate, made here
    # too because the two M>1 regimes differ in whether C is a fresh tensor.
    assert torch.all(out.float() == _CORRECTION_SENTINEL)
    assert corr["npv"] == _NPV and corr["B"] == _GROUP_SIZE
    assert out.shape == (M, N) and out.dtype is torch.bfloat16
    assert lm.ommx_w_fire_stats()["outlier_calls"] == 1


def test_apply_at_batched_decode_gives_sparse_correct_the_bf16_C_it_dereferences(
        bundle, fake_vllm, monkeypatch):
    """1 < M < 16 is the SECOND half of the same defect, and it is a dtype, not an arg.

    ``decode_base`` returns ``torch::zeros({M, N}, ... kFloat32)`` while ``goM`` reads
    ``out_or_C.data_ptr<at::BFloat16>()``. Handing the fp32 base straight through would
    abort inside ``data_ptr<T>()`` even once ``At`` was supplied — so batched decode with
    outliers would still have been dead after a fix that only added the transpose.
    """
    method, layer, spy, N, K = _spied_layer(bundle, monkeypatch)
    x = (torch.randn(8, K) * 0.1).to(torch.bfloat16)
    out = method.apply(layer, x)

    dec, corr = spy.of("decode_base"), spy.of("sparse_correct")
    assert dec["out"].dtype is torch.float32               # what the kernel really returns
    assert corr["C"].dtype is torch.bfloat16               # what the kernel really reads
    assert corr["At_opt"] is not None and corr["A_opt"] is None
    assert torch.equal(_kernel_index_walk(corr["At_opt"], 8), dec["A"])
    # The correction lands in the tensor that is RETURNED, not in a discarded copy.
    # This is the load-bearing half of the batched-decode fix and it needs a value
    # assertion, not a shape one: at 1 < M < 16 the bf16 `C` is necessarily a NEW tensor
    # (decode_base returned fp32), so `y` must be REBOUND to it. Write the correction
    # into a local and return the fp32 base instead and every dtype/shape/stride check
    # above still passes while every FP4 outlier is silently dropped — base-only INT2
    # served under an i2f4 label, with OMMX_W_OUTLIER_FIRED still in the log.
    # decode_base's base is torch.zeros, so the corrected result is exactly the sentinel.
    assert out.shape == (8, N)
    assert torch.equal(out, corr["C"]), (
        "apply() returned a tensor the correction did not land in")
    assert torch.all(out.float() == _CORRECTION_SENTINEL), (
        "the FP4 correction was written to a buffer that was then thrown away")


def test_apply_at_m_1_still_makes_the_exact_call_the_parity_gate_measured(
        bundle, fake_vllm, monkeypatch):
    """ADDITIVE, proven rather than asserted: M==1 is untouched by the M>1 fix.

    The shipped, gate-measured call is ``sparse_correct(y, x, None, ...)`` with ``y`` the
    fp32 tensor ``decode_base`` returned. Both are checked by IDENTITY (``is``), which is
    the only form that rules out a copy, a cast or a re-materialisation having slipped
    into the M==1 path while the M>1 branch was added beside it.
    """
    method, layer, spy, N, K = _spied_layer(bundle, monkeypatch)
    x = (torch.randn(1, K) * 0.1).to(torch.bfloat16)
    method.apply(layer, x)

    dec, corr = spy.of("decode_base"), spy.of("sparse_correct")
    assert corr["M"] == 1
    assert corr["A_opt"] is dec["A"]          # the activation itself, not a transpose
    assert corr["At_opt"] is None
    assert corr["C"] is dec["out"] and corr["C"].dtype is torch.float32


def test_the_m_gt_1_operands_helper_refuses_what_the_kernel_cannot_read():
    """The helper's own contract, reachable with no bundle and no kernel."""
    x = (torch.randn(4, 6) * 0.1).to(torch.bfloat16)
    C, At = lm.sparse_correct_m_gt_1_operands(torch.zeros(4, 3), x)
    assert C.dtype is torch.bfloat16 and At.shape == (6, 4) and At.is_contiguous()
    # bf16 in -> bf16 out is an IDENTITY: the prefill path must not pay for a copy.
    y_bf16 = torch.zeros(4, 3, dtype=torch.bfloat16)
    assert lm.sparse_correct_m_gt_1_operands(y_bf16, x)[0] is y_bf16
    with pytest.raises(lm.OMMXWError) as e1:
        lm.sparse_correct_m_gt_1_operands(torch.zeros(4, 3), x.float())
    assert "data_ptr<at::BFloat16>()" in str(e1.value)
    with pytest.raises(lm.OMMXWError) as e2:
        lm.sparse_correct_m_gt_1_operands(torch.zeros(5, 3), x)
    assert "At[k*M+m]" in str(e2.value)


def test_the_routing_threshold_is_bounded_by_both_kernel_limits(bundle, fake_vllm,
                                                                monkeypatch):
    """``OMMX_W_PREFILL_MIN_M`` is bracketed by a LOUD limit and a SILENT one.

    Below 16: ``prefill_wmma``'s ``TORCH_CHECK(M >= 16, ...)`` aborts — bad, but visible.
    Above 17: ``decode_base`` gets M > 16, its batched kernel is instantiated at
    ``M_MAX=16``, and the epilogue's ``if (lane == 0 && m < M)`` inside
    ``for (int m = 0; m < M_MAX; ++m)`` never writes rows 16..M-1 of a tensor that came
    from ``torch::zeros``. Those tokens would be served an all-zero activation with no
    error anywhere — the same class of silent fallback as the defect this release fixes,
    one entry point over. Refused in Python instead.

    17 must stay ALLOWED: it routes M==16 to decode_base (all 16 rows written) and
    M>=17 to prefill_wmma, so it is a legitimate setting, and a blanket "must equal 16"
    would be a stricter rule than the kernel's.
    """
    assert lm.prefill_min_m_env() == 16                       # unset -> shipped default
    for ok in ("16", "17"):
        monkeypatch.setenv("OMMX_W_PREFILL_MIN_M", ok)
        assert lm.prefill_min_m_env() == int(ok)
    for bad in ("8", "15", "18", "64"):
        monkeypatch.setenv("OMMX_W_PREFILL_MIN_M", bad)
        with pytest.raises(lm.OMMXWError) as exc:
            lm.prefill_min_m_env()
        assert "OMMX_W_PREFILL_MIN_M" in str(exc.value)
        assert "M_MAX=16" in str(exc.value) or "TORCH_CHECK(M >= 16)" in str(exc.value)
    monkeypatch.delenv("OMMX_W_PREFILL_MIN_M")

    # WIRING. Everything above tests the helper in isolation, which is not the same
    # claim: replace `prefill_min_m = prefill_min_m_env()` in apply() with a hardcoded
    # 16 and every assertion above stays green while the refusal never protects a
    # forward pass and the knob is silently ignored. Both halves are checked here —
    # that a legal value actually MOVES the routing, and that an illegal one aborts
    # apply() BEFORE any kernel is dispatched.
    method, layer, spy, N, K = _spied_layer(bundle, monkeypatch)
    x16 = (torch.randn(16, K) * 0.1).to(torch.bfloat16)
    monkeypatch.setenv("OMMX_W_PREFILL_MIN_M", "17")
    method.apply(layer, x16)
    assert [c["fn"] for c in spy.calls][0] == "decode_base", (
        "OMMX_W_PREFILL_MIN_M=17 must route M=16 to decode_base; apply() ignored it")
    spy.calls.clear()
    monkeypatch.setenv("OMMX_W_PREFILL_MIN_M", "18")
    with pytest.raises(lm.OMMXWError) as wired:
        method.apply(layer, x16)
    assert "OMMX_W_PREFILL_MIN_M" in str(wired.value)
    assert spy.calls == [], "refused only AFTER dispatching a kernel"


def test_an_outlier_free_recipe_never_reaches_sparse_correct_at_any_m(
        checkpoint, tmp_path_factory, fake_vllm, monkeypatch):
    """``npv=0`` must still take the base-only path at M>1 — the fix added a branch,
    not a call. Uses a REAL outlier-free bundle so the recipe, not a flag, decides."""
    nb = _pack(checkpoint, str(tmp_path_factory.mktemp("bundle_noout")), npv=0)
    method, layer, spy, N, K = _spied_layer(nb, monkeypatch)
    assert not method.recipe.has_outliers
    for M in (1, 8, 32):
        spy.calls.clear()
        method.apply(layer, (torch.randn(M, K) * 0.1).to(torch.bfloat16))
        assert [c["fn"] for c in spy.calls] == [
            "prefill_wmma" if M >= 16 else "decode_base"]


# ══════════════════════════════════════════════════════════════════════════════
# the one gate that needs a device — SKIPS here, never passes vacuously
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
def test_gpu_end_to_end_linear_matches_the_cpu_dequant_oracle(fake_vllm, bundle,
                                                              bundle_manifests):
    """UNVERIFIED on the authoring host (no GPU): the ONLY test of the executed path.

    Loads the real planes out of the bundle into a layer, JIT-builds the kernel, runs
    ``apply()``, and compares against ``x @ dequantize_ommx_weight(...)^T`` — the CPU
    oracle in ``w_format.load_weight``. cos first (law #4): cos >= 0.999 means the kernel
    is right and any max_diff complaint is a tolerance question, not a correctness one.
    """
    from ommx_gpu_serve.linear.w_format import load_planes, load_weight

    b = lm.load_bundle(bundle)
    shard = os.path.join(bundle, b.tensors[_Q_PROJ]["shard"])
    planes, recipe, N, K = load_planes(shard, _Q_PROJ)
    w_ref = load_weight(shard, _Q_PROJ).cuda()

    cfg, method = _make_layer_and_method(bundle)
    layer = _StubLinearBase().cuda()
    method.create_weights(layer, K, [N], K, N, torch.bfloat16)
    for plane, tensor in planes.items():
        getattr(layer, lm.param_name_for_plane(plane)).data.copy_(tensor.cuda())
    method.process_weights_after_loading(layer)
    assert layer.ommx_w_ready is True

    x = (torch.randn(1, K, device="cuda") * 0.1).to(torch.bfloat16)
    y = method.apply(layer, x)
    y_ref = x.float() @ w_ref.t()
    assert torch.isfinite(y).all()
    a, c = y.float().flatten(), y_ref.float().flatten()
    cos = float((a @ c) / (a.norm() * c.norm() + 1e-12))
    assert cos >= 0.999, f"cos={cos:.5f} (law #4: cos first)"
    stats = lm.ommx_w_fire_stats()
    assert stats["apply_calls"] == 1 and stats["decode_calls"] == 1
    assert stats["outlier_calls"] == 1              # npv>0 -> the FP4 splice MUST fire
    assert lm.ommx_w_health()["ok"] is True


@pytest.mark.gpu
def test_gpu_apply_at_m_gt_1_with_outliers_matches_the_cpu_dequant_oracle(
        fake_vllm, bundle, bundle_manifests):
    """UNVERIFIED here (no GPU): the ONLY execution of the M>1 outlier path.

    The CPU gates above prove the ARGUMENTS satisfy the contract read out of
    ``ommx_linear.cu``. Nothing on this host can prove the kernel then computes the right
    NUMBER — that needs a device, so this test skips rather than passing vacuously.

    It runs BOTH M>1 regimes, because they reach different base kernels and therefore
    different ``C`` dtypes:
      * M = 8  -> decode_base (fp32 base, cast to bf16 for the correction)
      * M = 32 -> prefill_wmma (bf16 base, no cast)
    and compares against ``x @ dequantize_ommx_weight(...)^T``, the same CPU oracle the
    M=1 gate uses. cos first (law #4).

    WHAT WOULD FALSIFY THE FIX: a cos below 0.999 at either M while the M=1 case in the
    same process stays at 1.00000. That would mean ``At`` is being read in an order this
    reading of ``At[(size_t)k_sh[s] * M + m]`` got wrong — the most likely alternative
    being an ``[M, K]`` buffer (i.e. no transpose at all, kernel stride assumption
    different from the one derived here).
    """
    from ommx_gpu_serve.linear.w_format import load_planes, load_weight

    b = lm.load_bundle(bundle)
    shard = os.path.join(bundle, b.tensors[_Q_PROJ]["shard"])
    planes, recipe, N, K = load_planes(shard, _Q_PROJ)
    assert recipe.has_outliers, "this gate is meaningless on an outlier-free bundle"
    w_ref = load_weight(shard, _Q_PROJ).cuda()

    cfg, method = _make_layer_and_method(bundle)
    layer = _StubLinearBase().cuda()
    method.create_weights(layer, K, [N], K, N, torch.bfloat16)
    for plane, tensor in planes.items():
        getattr(layer, lm.param_name_for_plane(plane)).data.copy_(tensor.cuda())
    method.process_weights_after_loading(layer)

    for M in (8, 32):
        x = (torch.randn(M, K, device="cuda") * 0.1).to(torch.bfloat16)
        y = method.apply(layer, x)
        assert y.shape == (M, N)
        assert torch.isfinite(y).all(), f"M={M} produced a non-finite output"
        a = y.float().flatten()
        c = (x.float() @ w_ref.t()).flatten()
        cos = float((a @ c) / (a.norm() * c.norm() + 1e-12))
        assert cos >= 0.999, f"M={M}: cos={cos:.5f} (law #4: cos first)"
    stats = lm.ommx_w_fire_stats()
    assert stats["outlier_calls"] == 2      # the M>1 correction FIRED, both times
    assert stats["prefill_calls"] == 1 and stats["decode_calls"] == 1


# ════════════════════════════════════════════════════════════════════════════
# deferred bundle resolution — the hop that makes `vllm serve <bundle>` work
# ════════════════════════════════════════════════════════════════════════════
#
# vLLM builds the quantization config inside ``VllmConfig._get_quantization_config``,
# which runs while the VllmConfig is still being constructed. ``get_current_vllm_config()``
# therefore has nothing to return, and ``resolve_bundle_dir``'s last rule — "the model path
# vLLM is building for" — cannot fire. That rule is what makes the documented invocation
# ``vllm serve <bundle> --quantization ommx_w`` work, so an eager resolve left the bundle
# findable only through ``$OMMX_W_BUNDLE`` or an absolute path written into the checkpoint.
# Deferring moves the lookup to first ``get_quant_method``, when the context IS active.
#
# The deferral is deliberately NARROW: only "nothing named a bundle" is deferrable. A path
# that is not a bundle, an undecodable recipe and a foreign quant_method are terminal and
# must still surface from from_config, or a precise refusal becomes a confusing one later.

def test_from_config_defers_when_no_bundle_is_configured_yet(fake_vllm, monkeypatch):
    monkeypatch.delenv("OMMX_W_BUNDLE", raising=False)
    cfg = lm.ommx_w_config_class().from_config({"quant_method": "ommx_w"})
    assert cfg is not None, "must not refuse: the model path is simply not in scope yet"


def test_a_deferred_config_resolves_once_the_path_appears(fake_vllm, bundle, monkeypatch):
    monkeypatch.delenv("OMMX_W_BUNDLE", raising=False)
    cfg = lm.ommx_w_config_class().from_config({"quant_method": "ommx_w"})
    monkeypatch.setenv("OMMX_W_BUNDLE", bundle)          # stands in for the model path
    assert cfg.bundle.bundle_dir == os.path.abspath(bundle)
    assert cfg.recipe.group_size == _GROUP_SIZE


def test_a_deferred_config_still_raises_if_nothing_ever_names_a_bundle(
        fake_vllm, monkeypatch):
    """Deferring changes WHEN the operator is told, not WHETHER."""
    monkeypatch.delenv("OMMX_W_BUNDLE", raising=False)
    cfg = lm.ommx_w_config_class().from_config({"quant_method": "ommx_w"})
    with pytest.raises(lm.OMMXWBundleUnconfigured) as exc:
        _ = cfg.bundle
    assert "no OMMX_W_SafeTensor bundle was configured" in str(exc.value)
    assert "w_packer pack" in str(exc.value), "the packer hint must survive the deferral"


def test_a_configured_but_wrong_path_is_NOT_deferred(fake_vllm, tmp_path, monkeypatch):
    """Something named a bundle and it was not one -- that is terminal, not 'not yet'."""
    monkeypatch.delenv("OMMX_W_BUNDLE", raising=False)
    empty = str(tmp_path / "not_a_bundle")
    os.makedirs(empty, exist_ok=True)
    with pytest.raises(lm.OMMXWError) as exc:
        lm.ommx_w_config_class().from_config({"quant_method": "ommx_w", "bundle": empty})
    assert not isinstance(exc.value, lm.OMMXWBundleUnconfigured)
    assert "none of the candidate paths" in str(exc.value)


def test_unconfigured_is_a_subclass_so_existing_handlers_still_catch_it():
    assert issubclass(lm.OMMXWBundleUnconfigured, lm.OMMXWError)
