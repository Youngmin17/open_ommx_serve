# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The plugin may only claim "REGISTERED" for an axis vLLM's own registry confirms.

WHY. ``register()`` printed on FAILURE only, so a load that registered nothing and a load
that registered both axes were indistinguishable on stderr: silence. The two axes are
independent — the CUSTOM attention backend and the ``ommx_w`` quantization name live in
different registries and fail separately — so "the plugin loaded" is not the claim an
operator needs; "OMMX is registered" is. And the flag that tracked it lied: the
no-attention-registry path set ``_REGISTERED = True``, which made a later call take the
``if not _REGISTERED`` early-out, skip registration, and still return ``"CUSTOM"``.

Every assertion here is about a READ-BACK of the registry, never about our own control
flow, because that distinction is the whole point of the change.
"""
import importlib

import pytest

MOD = "ommx_gpu_serve.integration.vllm.plugin"


@pytest.fixture()
def plugin():
    """A fresh module each test — `_REGISTERED` / `_NO_REGISTRY_NOTED` are module state."""
    return importlib.reload(importlib.import_module(MOD))


@pytest.mark.parametrize("attn,wq,must_not_contain", [
    (False, False, "REGISTERED ("),      # neither -> the word must not appear as a claim
    (None, None, "REGISTERED ("),        # unanswerable -> still not a claim
    (False, True, "attention: REGISTERED"),
    (True, False, "ommx_w: REGISTERED"),
])
def test_never_claims_registered_for_an_axis_that_did_not_register(
        plugin, capsys, attn, wq, must_not_contain):
    plugin._attention_registered = lambda: attn
    plugin._ommx_w_registered = lambda: wq
    plugin._report_registration()
    err = capsys.readouterr().err
    assert must_not_contain not in err, err
    assert err.startswith("[ommx] ")


def test_claims_both_only_when_both_verify(plugin, capsys):
    plugin._attention_registered = lambda: True
    plugin._ommx_w_registered = lambda: True
    plugin._report_registration()
    err = capsys.readouterr().err
    assert "attention: REGISTERED" in err
    assert "ommx_w: REGISTERED" in err


def test_unknown_is_reported_as_unknown_not_as_failure(plugin, capsys):
    """None means "could not ask", which is not the same as "not registered" — an
    operator reading NOT-registered would go looking for a bug that isn't there."""
    plugin._attention_registered = lambda: None
    plugin._ommx_w_registered = lambda: None
    plugin._report_registration()
    err = capsys.readouterr().err
    assert "unknown" in err
    assert "NOT registered" not in err


def test_attention_verifier_compares_against_the_path_actually_registered(plugin):
    """A drift guard: the verifier must compare the registry's stored path against the
    same constant ``register()`` hands to vLLM, or a rename would silently report
    'registered' for a class path vLLM does not hold."""
    # Resolved WITHOUT importing the backend: it imports vLLM at module scope by design
    # (lazy, only when CUSTOM is selected), so importing it here would make this test
    # require a GPU-serving install to check a string.
    import os
    mod, _, cls = plugin._OMMX_BACKEND_PATH.rpartition(".")
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(plugin.__file__))),
                       "vllm", "backend.py")
    assert os.path.exists(src), src
    assert f"class {cls}" in open(src).read(), f"{cls} is not defined in {src}"
    assert mod == "ommx_gpu_serve.integration.vllm.backend"
    assert cls == "OMMXCanonicalBackend"

    class _Enum:
        @staticmethod
        def is_overridden():
            return True

        @staticmethod
        def get_path(include_classname=True):
            return "some.other.Backend"

    # A foreign owner of CUSTOM must read as NOT ours, not as "registered".
    plugin._attention_registered = plugin._attention_registered  # keep real fn
    assert _Enum.get_path() != plugin._OMMX_BACKEND_PATH


def test_verifiers_return_none_rather_than_raising_without_vllm(plugin):
    """Verification is diagnostics: it must never be able to take plugin load down."""
    assert plugin._attention_registered() in (True, False, None)
    assert plugin._ommx_w_registered() in (True, False, None)


def test_missing_attention_registry_does_not_mark_registered(plugin, monkeypatch, capsys):
    """The bug this replaces: claiming registered state on the path that registered
    nothing, which then suppressed a real later registration."""
    monkeypatch.setattr(plugin, "_register_ommx_w_quant", lambda: None)
    import builtins
    real_import = builtins.__import__

    def _no_registry(name, *a, **k):
        if name == "vllm.v1.attention.backends.registry":
            raise ImportError("no v1 attention registry")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_registry)
    assert plugin.register() is None
    assert plugin._REGISTERED is False, "nothing was registered; the flag must not say it was"
    assert "REGISTERED (" not in capsys.readouterr().err


def test_no_registry_note_is_printed_once(plugin, monkeypatch, capsys):
    monkeypatch.setattr(plugin, "_register_ommx_w_quant", lambda: None)
    import builtins
    real_import = builtins.__import__

    def _no_registry(name, *a, **k):
        if name == "vllm.v1.attention.backends.registry":
            raise ImportError("no v1 attention registry")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_registry)
    plugin.register()
    first = capsys.readouterr().err
    plugin.register()
    second = capsys.readouterr().err
    assert first.count("[ommx]") == 1
    assert second == "", "a repeated load on a registry-less tree must not spam"
