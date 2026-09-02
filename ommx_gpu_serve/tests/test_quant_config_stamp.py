# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""A packed bundle must be DISCOVERABLE by vLLM, not merely well-formed.

``OMMXWConfig`` registers correctly under the name ``ommx_w`` (the plugin verifies that
against vLLM's own registry) and ``from_config`` / ``resolve_bundle_dir`` have always been
ready to receive the checkpoint's ``quantization_config`` dict. But vLLM only CALLS
``from_config`` when the model's HF config carries that key, and the packer copied
``config.json`` through byte-for-byte and never wrote one — so
``vllm serve <bundle> --quantization ommx_w``, the invocation the README documents, could
never construct the config and the whole weight-quant path was unreachable.

These gates pin the key, its content, the opt-out, the refusal to overwrite a foreign
method, and the two things that would make the stamp a lie: an index digest that no longer
describes the file on disk, and a method name that has drifted from the serving side's.
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")

from ommx_gpu_serve.linear import w_packer  # noqa: E402
from ommx_gpu_serve.linear.w_format import validate_bundle  # noqa: E402


@pytest.fixture()
def packed(tmp_path):
    """A real synthetic checkpoint packed into a real bundle."""
    src, out = str(tmp_path / "src"), str(tmp_path / "bundle")
    w_packer.make_synthetic_checkpoint(src)
    rep = w_packer.pack_checkpoint(src, out, _default_recipe(), verbose=False)
    return src, out, rep


def _default_recipe():
    """The packer's own default recipe, built through its own factory so this test cannot
    drift from what the CLI actually packs."""
    return w_packer.build_recipe(group_size=64, outlier_pct=None, npv=None,
                                 outlier_repr="relidx7", outlier_map="idx_range",
                                 zp_dtype="bf16")


def _quant_cfg(bundle_dir):
    with open(os.path.join(bundle_dir, "config.json")) as fh:
        return json.load(fh).get(w_packer.HF_QUANT_CONFIG_KEY)


def test_pack_stamps_the_discovery_key(packed):
    _, out, _ = packed
    q = _quant_cfg(out)
    assert q is not None, "no quantization_config -> vLLM never calls from_config"
    assert q["quant_method"] == "ommx_w"


def test_stamped_config_is_accepted_by_the_serving_side(packed, monkeypatch):
    """The round trip that was broken: bundle -> config.json -> resolve_bundle_dir."""
    _, out, _ = packed
    from ommx_gpu_serve.integration.vllm import linear_method as lm
    monkeypatch.setenv("OMMX_W_BUNDLE", out)
    q = _quant_cfg(out)
    assert os.path.realpath(lm.resolve_bundle_dir(quant_config=dict(q))) == \
        os.path.realpath(out)
    assert lm.load_bundle(quant_config=dict(q)) is not None


def test_no_quant_config_opts_out(tmp_path):
    src, out = str(tmp_path / "src"), str(tmp_path / "bundle")
    w_packer.make_synthetic_checkpoint(src)
    w_packer.pack_checkpoint(src, out, _default_recipe(), verbose=False,
                             stamp_quant_config=False)
    assert _quant_cfg(out) is None
    # ...and the bundle is still a valid bundle; it is only undiscoverable.
    validate_bundle(out)


def test_a_foreign_quantization_config_is_refused(tmp_path):
    """Stamping ommx_w over an AWQ checkpoint would claim planes the bundle lacks."""
    src, out = str(tmp_path / "src"), str(tmp_path / "bundle")
    w_packer.make_synthetic_checkpoint(src)
    cfg_path = os.path.join(src, "config.json")
    cfg = json.load(open(cfg_path))
    cfg[w_packer.HF_QUANT_CONFIG_KEY] = {"quant_method": "awq", "bits": 4}
    json.dump(cfg, open(cfg_path, "w"))
    with pytest.raises(w_packer.PackError) as e:
        w_packer.pack_checkpoint(src, out, _default_recipe(), verbose=False)
    assert "awq" in str(e.value)
    assert "--no-quant-config" in str(e.value), "the message must name the escape hatch"


def test_repacking_an_ommx_w_bundle_is_allowed(tmp_path):
    """Our own key is not foreign -- re-packing must not be refused by its own output."""
    src, out = str(tmp_path / "src"), str(tmp_path / "bundle")
    w_packer.make_synthetic_checkpoint(src)
    cfg_path = os.path.join(src, "config.json")
    cfg = json.load(open(cfg_path))
    cfg[w_packer.HF_QUANT_CONFIG_KEY] = {"quant_method": "ommx_w"}
    json.dump(cfg, open(cfg_path, "w"))
    w_packer.pack_checkpoint(src, out, _default_recipe(), verbose=False)
    assert _quant_cfg(out)["quant_method"] == "ommx_w"


def test_index_digest_describes_the_stamped_file_not_the_copied_one(packed):
    """The stamp edits config.json AFTER the copy. If the recorded digest were left at
    the pre-edit value the index would describe a file that no longer exists on disk."""
    _, out, _ = packed
    idx = json.load(open(os.path.join(out, "ommx_w_index.json")))
    ent = next(e for e in idx["aux_files"]["copied"] if e["name"] == "config.json")
    from ommx_gpu_serve.linear.w_format import sha256_file
    path = os.path.join(out, "config.json")
    assert ent["bytes"] == os.path.getsize(path)
    assert ent["sha256"] == sha256_file(path)


def test_stamped_bundle_still_validates(packed):
    _, out, _ = packed
    validate_bundle(out)


def test_method_name_has_not_drifted_from_the_serving_side():
    """The packer duplicates the name rather than importing the integration package. If
    the two ever diverge, bundles advertise a method the serving side does not answer to
    -- the exact undiscoverable-bundle failure this stamping exists to fix."""
    from ommx_gpu_serve.integration.vllm.linear_method import OMMX_W_METHOD_NAME
    assert w_packer.OMMX_W_METHOD_NAME == OMMX_W_METHOD_NAME


def test_recipe_echo_is_informational_only(packed):
    """from_config reads ONLY quant_method; the echoed recipe must never be the thing the
    serving side trusts (the index is authoritative), so a mangled echo must not break
    discovery."""
    _, out, _ = packed
    from ommx_gpu_serve.integration.vllm import linear_method as lm
    q = dict(_quant_cfg(out))
    q["recipe"] = {"group_size": "nonsense"}
    assert lm.resolve_bundle_dir(explicit=out, quant_config=q) == out
