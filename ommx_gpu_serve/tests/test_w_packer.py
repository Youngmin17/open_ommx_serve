# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU gates for the OMMX offline weight path (``ommx_gpu_serve.linear``).

These are the gates for paper claims C3 (``OMMX_W_SafeTensor``), D3 (HF safetensors
-> OMMX W-Packer) and D4 (the four offline packing steps). Everything here runs on a
laptop with no GPU, no triton, no vLLM and no ``safetensors`` package, and nothing
here is ``gpu``-marked because nothing here needs a device.

What is deliberately NOT covered (and cannot be, without hardware): that
``csrc/linear/ommx_linear.cu`` decodes these planes to the same numbers. That is the
job of ``csrc/linear/test_ommx_linear_parity.py``, which asserts
``torch.cuda.is_available()`` on entry. The bridge between the two is
``test_bit_exact_vs_parity_packer`` below: it pins this package's quantizer to the
EXACT bytes of the packer that gate already validates on a GPU, so a passing parity
gate transfers to this package without re-running it.

Two of the gates here exist because an adversarial verifier found the ``ommx_w``
bundle unloadable by a real engine for two independent reasons, both now fixed in the
FORMAT rather than worked around downstream:

  * plane names dropped the ``.weight`` infix (format v1 -> v2) so a plane binds to
    the parameter a vLLM Linear registers — section 4b;
  * ``pack`` copies the source checkpoint's non-weight files through, so the output
    directory is a model directory a loader can resolve — section 4c.

Kept small and fast on purpose (largest tensor here is 24x256 f32): these run in the
default ``-m 'not gpu'`` set on every developer's machine.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from typing import Dict, List, Tuple

import pytest
import torch

from ommx_gpu_serve.attention import codec
from ommx_gpu_serve.linear import w_format as F
from ommx_gpu_serve.linear import w_packer as P
from ommx_gpu_serve.linear.quantize import (
    OMMXQuantizeError,
    dequantize_ommx_weight,
    outlier_positions,
    quantize_ommx_weight,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CSRC_LINEAR_DIR = os.path.join(_REPO_ROOT, "ommx_gpu_serve", "csrc", "linear")
_PARITY_PATH = os.path.join(_CSRC_LINEAR_DIR, "test_ommx_linear_parity.py")


def _load_parity_module():
    """Import ``csrc/linear/test_ommx_linear_parity.py`` BY FILE PATH.

    That directory carries no ``__init__.py`` and is excluded from the installed
    package (see ``ommx_gpu_serve/pyproject.toml``: ``csrc/linear`` is deliberately
    not a subpackage), so there is no import path to it. Loading it by path is the
    only way to compare against it, and comparing against it is the entire point of
    this file — the packer under test must be byte-identical to the implementation
    that the GPU parity gate blesses. Its module body is import-safe: the CUDA
    assertions live in ``main()`` behind ``if __name__ == "__main__"``.

    TWO DIFFERENT ABSENCES, kept apart on purpose (this file IS shipped in the wheel,
    ``csrc/`` is NOT):

      * ``csrc/linear/`` missing entirely -> we are running against an INSTALLED
        package, where the reference physically cannot be present. SKIP, naming the
        cause, so ``pytest --pyargs ommx_gpu_serve.tests`` reports a skip instead of
        a bare ``FileNotFoundError`` from ``exec_module``.
      * ``csrc/linear/`` present but the reference file gone -> somebody deleted or
        renamed the contract in a SOURCE checkout. Hard FAIL: that is exactly the
        regression this gate exists to catch, and skipping it would be the silent
        fallback the project forbids.
    """
    if not os.path.isdir(_CSRC_LINEAR_DIR):
        pytest.skip(
            "csrc/linear/ is not present, so this is an installed ommx-gpu-serve "
            "rather than a source checkout (pyproject.toml ships "
            "ommx_gpu_serve.tests but deliberately not csrc/). The bit-exactness "
            "gate against test_ommx_linear_parity.py can only run from the repo.")
    if not os.path.isfile(_PARITY_PATH):
        raise AssertionError(
            f"{_PARITY_PATH} is missing but csrc/linear/ exists. That file IS the "
            f"OMMX weight-format contract (the packer under test must be byte-"
            f"identical to it); it must not be deleted or renamed without moving "
            f"this gate to whatever replaces it.")
    spec = importlib.util.spec_from_file_location("_ommx_parity_ref", _PARITY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ════════════════════════════════════════════════════════════════════════════
# 1. quantize.py is BIT-EXACT to the parity gate's packer
# ════════════════════════════════════════════════════════════════════════════

# (N, K) x group_size x outlier_pct x seed. Shapes are tiny because the reference is
# a per-outlier python loop; the sweep still covers every branch that matters:
# npv == 0 (no sidecar at all), npv odd (nibble stream padded), npv*7 not a multiple
# of 8 (index stream padded), and more than one group per row.
_SWEEP_SHAPES = ((4, 64), (8, 128), (3, 192))
_SWEEP_GS = (16, 32, 64)
_SWEEP_PCT = (0.0, 0.0625, 0.125, 0.25)
_SWEEP_SEEDS = (0, 1)


@pytest.mark.parametrize("shape", _SWEEP_SHAPES)
@pytest.mark.parametrize("group_size", _SWEEP_GS)
@pytest.mark.parametrize("outlier_pct", _SWEEP_PCT)
@pytest.mark.parametrize("seed", _SWEEP_SEEDS)
def test_bit_exact_vs_parity_packer(shape, group_size, outlier_pct, seed):
    """Every plane, elementwise identical to the reference. No tolerance anywhere.

    ``torch.equal`` (not ``allclose``) and an explicit dtype/shape check: the planes
    are BYTES on disk, so "close" is meaningless — a nibble that rounds the other way
    is a different weight, and an int8 exponent that became an int32 is a different
    file layout.
    """
    N, K = shape
    if K % group_size:
        pytest.skip(f"group_size {group_size} does not divide K={K}")
    ref = _load_parity_module()
    torch.manual_seed(seed)
    W = torch.randn(N, K) * 0.1

    expect = ref.quantize_ommx_weight(W.clone(), group_size, outlier_pct)
    got = quantize_ommx_weight(W.clone(), group_size, outlier_pct)

    assert set(got) == set(expect), (
        f"key set drifted: only-in-new={sorted(set(got) - set(expect))}, "
        f"only-in-ref={sorted(set(expect) - set(got))}")
    for key, want in expect.items():
        have = got[key]
        if torch.is_tensor(want):
            assert torch.is_tensor(have), f"{key}: expected a tensor"
            assert have.dtype == want.dtype, f"{key}: dtype {have.dtype} != {want.dtype}"
            assert have.shape == want.shape, f"{key}: shape {have.shape} != {want.shape}"
            assert torch.equal(have, want), (
                f"{key}: {int((have != want).sum())} of {want.numel()} elements differ")
        else:
            assert have == want, f"{key}: {have} != {want}"


def test_bit_exact_sweep_covers_the_padding_branches():
    """Guard the sweep itself: it must actually exercise padded and unpadded streams."""
    seen = set()
    for gs in _SWEEP_GS:
        for pct in _SWEEP_PCT:
            npv = int(gs * pct) if pct > 0 else 0
            npv = max(1, npv) if pct > 0 else 0
            seen.add(("npv0" if npv == 0 else
                      "padded" if (npv * 7) % 8 else "aligned"))
            if npv % 2:
                seen.add("odd_nibbles")
    assert {"npv0", "padded", "aligned", "odd_nibbles"} <= seen, seen


# ════════════════════════════════════════════════════════════════════════════
# 2. position streams match the shipped codec primitives
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("outlier_repr", ["relidx7", "bitmap"])
def test_position_stream_matches_codec_primitives(outlier_repr):
    """The vectorized packing equals ``attention/codec.py``'s python primitives.

    ``codec.pack_relidx7`` / ``pack_bitmap_row`` are the shipped, kernel-mirroring
    definitions of these two bit layouts. The quantizer rewrites them as a batched
    tensor op for speed; this asserts the rewrite did not change a single bit.
    """
    torch.manual_seed(7)
    N, K, gs, pct = 5, 128, 32, 0.125
    W = torch.randn(N, K) * 0.1
    q = quantize_ommx_weight(W, gs, pct, outlier_repr=outlier_repr)
    G, npv = q["G"], q["npv"]
    # positions come from the relidx7 stream, which the bit-exactness test above pins
    ref = quantize_ommx_weight(W, gs, pct)
    pos = outlier_positions(ref["oindex"], N, G, npv, gs, "relidx7")

    nb = F.Recipe(gs, npv, pct, outlier_repr).index_bytes
    blob = q["oindex"].reshape(N, G, nb)
    for n in range(N):
        for g in range(G):
            cols = pos[n, g].tolist()
            want = (codec.pack_relidx7(cols) if outlier_repr == "relidx7"
                    else codec.pack_bitmap_row(cols, gs))
            assert bytes(blob[n, g].tolist()) == want, (n, g, outlier_repr)


def test_outlier_repr_does_not_change_the_weights():
    """relidx7 vs bitmap is a METADATA choice: the dequantized weights are identical."""
    torch.manual_seed(11)
    W = torch.randn(6, 128) * 0.1
    a = quantize_ommx_weight(W, 64, 0.0625, outlier_repr="relidx7")
    b = quantize_ommx_weight(W, 64, 0.0625, outlier_repr="bitmap")
    assert torch.equal(a["W_ref"], b["W_ref"])
    assert torch.equal(a["ocode"], b["ocode"])
    assert not torch.equal(a["oindex"].reshape(-1)[:1], b["oindex"].reshape(-1)[:1]) \
        or a["oindex"].numel() != b["oindex"].numel()


# ════════════════════════════════════════════════════════════════════════════
# 3. round trip: quantize -> write -> read -> dequant reproduces W_ref exactly
# ════════════════════════════════════════════════════════════════════════════

_RT_RECIPES = [
    (64, 4, "relidx7", "idx_range", torch.bfloat16),
    (64, 4, "bitmap", "idx_range", torch.bfloat16),
    (64, 4, "bitmap", "none", torch.bfloat16),      # the 3.63 bit/weight recipe
    (32, 3, "relidx7", "idx_range", torch.float32),  # odd npv -> padded nibble stream
    (64, 0, "relidx7", "idx_range", torch.bfloat16),  # no outliers at all (i2)
]


@pytest.mark.parametrize("gs,npv,repr_,map_,zp_dtype", _RT_RECIPES)
def test_round_trip_bit_exact(tmp_path, gs, npv, repr_, map_, zp_dtype):
    """A bundle read back off disk dequantizes to EXACTLY the quantizer's W_ref.

    Exactness (not closeness) is achievable because the format stores everything the
    dequant needs and stores it losslessly: the E8M0 exponent reconstructs the float32
    scale bit-for-bit (2^e IS a power of two), and the zero-point is rounded to its
    STORAGE dtype before the INT2 codes are chosen, so no plane is ever re-rounded on
    the way to disk. If this test ever needs a tolerance, the format has a lossy step
    in it that nobody declared.
    """
    torch.manual_seed(5)
    N, K = 8, 256
    W = torch.randn(N, K) * 0.1
    recipe = F.Recipe(gs, npv, npv / gs, repr_, map_, zp_dtype)
    q = quantize_ommx_weight(W, gs, npv / gs, npv=npv, outlier_repr=repr_,
                             outlier_map=map_, zp_dtype=zp_dtype)

    path = str(tmp_path / "shard.safetensors")
    name = "model.layers.0.mlp.down_proj.weight"
    planes = P._quantize_one(W, name, recipe)
    writer = F.SafeTensorsWriter(path)
    pdesc, nbytes = {}, 0
    for plane in recipe.planes():
        pn = F.plane_name(name, plane)
        nbytes += writer.add(pn, planes[plane])
        pdesc[plane] = {"name": pn, "shape": list(planes[plane].shape),
                        "dtype": F.st_dtype_name(planes[plane].dtype)}
    manifest = {
        "format": F.OMMX_W_FORMAT, "version": F.OMMX_W_FORMAT_VERSION,
        "producer": "test", "source_model": "synthetic",
        "shard": {"index": 1, "count": 1, "file": os.path.basename(path)},
        # Mandatory since format v2: a manifest that does not declare its plane-naming
        # convention is refused, so this hand-built one has to declare it too.
        "naming": F.naming_document(),
        "recipe": recipe.to_json(),
        "tensors": {name: {"kind": "quantized", "reason": "test", "dtype": "F32",
                           "shape": [N, K], "n_groups": K // gs, "planes": pdesc,
                           "packed_bytes": nbytes,
                           "bits_per_weight": F.bits_per_weight(recipe, N, K, nbytes)}},
    }
    writer.close({"format": F.OMMX_W_FORMAT, "version": str(F.OMMX_W_FORMAT_VERSION),
                  "manifest": json.dumps(manifest)})

    F.validate_shard(path)
    back = F.load_weight(path, name)
    assert back.dtype == torch.float32
    assert torch.equal(back, q["W_ref"]), (
        f"round trip differs at {int((back != q['W_ref']).sum())} of {N * K} elements; "
        f"max |delta| = {float((back - q['W_ref']).abs().max()):.3e}")


def test_dequant_needs_no_reference_tensors():
    """``reference=False`` drops W_base/W_ref but keeps the key set and the planes.

    The packer runs with ``reference=False`` — those two tensors are full-precision
    copies of the weight and would double peak RAM on a real checkpoint — so the
    planes must be complete without them.
    """
    torch.manual_seed(2)
    W = torch.randn(4, 128) * 0.1
    full = quantize_ommx_weight(W, 64, 0.0625)
    lean = quantize_ommx_weight(W, 64, 0.0625, reference=False)
    assert set(lean) == set(full)
    assert lean["W_base"] is None and lean["W_ref"] is None
    for key in ("code", "scale_exp", "zp", "oindex", "ocode", "map_scale", "map_center"):
        assert torch.equal(lean[key], full[key]), key
    back = dequantize_ommx_weight(
        code=lean["code"], scale_exp=lean["scale_exp"], zp=lean["zp"], N=4, K=128,
        group_size=64, npv=lean["npv"], oindex=lean["oindex"], ocode=lean["ocode"],
        map_scale=lean["map_scale"], map_center=lean["map_center"])
    assert torch.equal(back, full["W_ref"])


# ════════════════════════════════════════════════════════════════════════════
# 4. end-to-end packer on a SYNTHETIC checkpoint
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """A tiny Llama-shaped safetensors checkpoint (no model, no network on this host)."""
    d = tmp_path_factory.mktemp("ckpt")
    P.make_synthetic_checkpoint(str(d), layers=2, hidden=64, inter=128, vocab=96,
                                shards=2, dtype=torch.bfloat16)
    return str(d)


def _pack(src, dst, **kw) -> dict:
    recipe = F.Recipe(kw.pop("gs", 64), kw.pop("npv", 4), kw.pop("pct", 0.0625),
                      kw.pop("repr_", "relidx7"), kw.pop("map_", "idx_range"),
                      kw.pop("zp_dtype", torch.bfloat16))
    return P.pack_checkpoint(src, dst, recipe, source_model="synthetic",
                             verbose=False, **kw)


def test_pack_end_to_end(synthetic, tmp_path):
    """Pack -> validate -> read one weight back and confirm it matches a fresh quantize."""
    out = str(tmp_path / "bundle")
    report = _pack(synthetic, out)
    index = F.validate_bundle(out)
    assert index["weight_map"], "empty bundle"
    assert report["totals"]["n_quantized"] == 14      # 7 projections x 2 layers
    assert report["totals"]["n_passthrough"] == 9     # embed + lm_head + 5 norms + 2 bias

    # Every tensor of the input checkpoint is present in the bundle exactly once.
    src_names = set()
    for f in P.discover_shards(synthetic):
        src_names |= set(F.SafeTensorsFile(f).names)
    assert set(index["weight_map"]) == src_names

    name = "model.layers.1.self_attn.o_proj.weight"
    shard = os.path.join(out, index["weight_map"][name])
    src = [f for f in P.discover_shards(synthetic)
           if name in F.SafeTensorsFile(f).names][0]
    W = F.SafeTensorsFile(src).tensor(name).to(torch.float32)
    recipe = F.Recipe.from_json(index["recipe"])
    q = quantize_ommx_weight(W, recipe.group_size, recipe.outlier_pct, npv=recipe.npv,
                             outlier_repr=recipe.outlier_repr,
                             outlier_map=recipe.outlier_map, zp_dtype=recipe.zp_dtype)
    assert torch.equal(F.load_weight(shard, name), q["W_ref"])


def test_passthrough_tensors_survive_unquantized(synthetic, tmp_path):
    """lm_head / embeddings / norms / biases are copied BYTE-FOR-BYTE, and recorded.

    The failure this guards against is silent: a quantized ``lm_head`` still loads,
    still generates text, and only shows up as a slightly worse eval score.
    """
    out = str(tmp_path / "bundle")
    _pack(synthetic, out)
    index = F.validate_bundle(out)

    must_pass_through = ["lm_head.weight", "model.embed_tokens.weight",
                         "model.norm.weight",
                         "model.layers.0.input_layernorm.weight",
                         "model.layers.0.post_attention_layernorm.weight",
                         "model.layers.0.self_attn.q_proj.bias"]
    src_by_name: Dict[str, str] = {}
    for f in P.discover_shards(synthetic):
        for n in F.SafeTensorsFile(f).names:
            src_by_name[n] = f
    for name in must_pass_through:
        shard = os.path.join(out, index["weight_map"][name])
        man = F.read_manifest(shard)
        ent = man["tensors"][name]
        assert ent["kind"] == "passthrough", f"{name} was QUANTIZED: {ent}"
        assert ent["reason"], f"{name} passthrough with no recorded reason"
        got = F.SafeTensorsFile(shard).tensor(name)
        want = F.SafeTensorsFile(src_by_name[name]).tensor(name)
        assert got.dtype == want.dtype and torch.equal(got, want), name
        # ...and no ommx_* plane was emitted for it
        assert not any(n.startswith(name + ".ommx_")
                       for n in F.SafeTensorsFile(shard).names), name

    quantized = [n for n in index["weight_map"]
                 if F.read_manifest(os.path.join(out, index["weight_map"][n]))
                 ["tensors"][n]["kind"] == "quantized"]
    assert quantized, "nothing was quantized at all"
    assert all(n.rsplit(".", 2)[-2] in P.DEFAULT_QUANT_MODULES for n in quantized), \
        f"a non-projection was quantized: {quantized}"


def test_manifest_describes_every_written_tensor(synthetic, tmp_path):
    """No tensor in a shard is undeclared, and no declaration lacks a tensor."""
    out = str(tmp_path / "bundle")
    _pack(synthetic, out)
    index = F.validate_bundle(out)
    for shard_file in index["shards"]:
        path = os.path.join(out, shard_file)
        st = F.SafeTensorsFile(path)
        man = F.read_manifest(path, st)
        declared = set()
        for name, ent in man["tensors"].items():
            if ent["kind"] == "passthrough":
                declared.add(name)
            else:
                declared |= {p["name"] for p in ent["planes"].values()}
        assert declared == set(st.names), (
            f"{shard_file}: undeclared={sorted(set(st.names) - declared)}, "
            f"declared-but-absent={sorted(declared - set(st.names))}")


# ════════════════════════════════════════════════════════════════════════════
# 4b. plane NAMES bind to a vLLM parameter (format v2 dropped the .weight infix)
# ════════════════════════════════════════════════════════════════════════════
# Format v1 stored a plane as ``<module>.weight.ommx_code``. vLLM and transformers
# match a checkpoint tensor name against ``dict(model.named_parameters())``, and a
# parameter registered on a Linear module is reachable as ``<module>.<attr>`` — which
# is why AWQ ships ``<module>.qweight`` and not ``<module>.weight.qweight``. A v1
# bundle therefore cannot be loaded by an engine at all. These gates pin the fix at
# the FORMAT, where it belongs, rather than at a loader-side remap.

def test_plane_names_have_no_weight_infix_and_round_trip():
    """``...q_proj.weight`` -> ``...q_proj.ommx_code``, and back again."""
    w = "model.layers.3.self_attn.q_proj.weight"
    for plane in F.ALL_PLANES:
        pn = F.plane_name(w, plane)
        assert pn == f"model.layers.3.self_attn.q_proj.ommx_{plane}"
        assert ".weight.ommx_" not in pn, pn
        # The module path a loader will look under is exactly the tensor name minus
        # ``.weight`` — no extra component, no dropped one.
        assert pn.rsplit(".", 1)[0] == w[: -len(".weight")]
        assert F.split_plane_name(pn) == (w, plane)
    # Non-planes are not claimed by the parser.
    assert F.split_plane_name("model.norm.weight") is None
    assert F.split_plane_name("model.layers.0.self_attn.q_proj.ommx_nonsense") is None
    assert F.split_plane_name("ommx_code") is None          # no owning module


def test_plane_name_refuses_a_tensor_that_is_not_a_dot_weight():
    """No silent fallback to "append the suffix" for a name that has no module.

    Appending would produce a name no loader looks for, and would do it quietly —
    exactly the failure the version bump exists to remove.
    """
    for bad in ("model.layers.0.self_attn.q_proj.bias", "lm_head", ".weight"):
        with pytest.raises(F.OMMXWFormatError) as exc:
            F.plane_name(bad, "code")
        assert "'.weight'" in str(exc.value) or "no owning module" in str(exc.value)


def test_packed_bundle_contains_no_weight_infix_plane(synthetic, tmp_path):
    """The gate on the ARTEFACT, not on the helper: read the shards back off disk.

    ``plane_name`` could be correct while the packer wrote something else (it did not
    always route through the helper), so this asserts on the bytes that shipped.
    """
    out = str(tmp_path / "bundle")
    _pack(synthetic, out)
    index = F.validate_bundle(out)
    planes_seen = 0
    for shard_file in index["shards"]:
        st = F.SafeTensorsFile(os.path.join(out, shard_file))
        for name in st.names:
            assert ".weight.ommx_" not in name, f"v1-style plane name in {shard_file}: {name}"
            split = F.split_plane_name(name)
            if split is not None:
                planes_seen += 1
                weight_name, _ = split
                # ...and the name a loader would derive really is a weight the
                # manifest describes as quantized.
                man = F.read_manifest(os.path.join(out, shard_file))
                assert man["tensors"][weight_name]["kind"] == "quantized", name
    assert planes_seen == 14 * 7, planes_seen      # 7 projections x 2 layers x 7 planes


def test_every_manifest_and_the_index_declare_the_naming_convention(synthetic, tmp_path):
    """Self-describing: a reader never has to infer the convention from a tensor list."""
    out = str(tmp_path / "bundle")
    _pack(synthetic, out)
    index = F.validate_bundle(out)
    want = F.naming_document()
    assert index["naming"] == want
    assert "{module}" in want["plane_name"] and ".weight" not in want["plane_name"]
    for shard_file in index["shards"]:
        assert F.read_manifest(os.path.join(out, shard_file))["naming"] == want


# ════════════════════════════════════════════════════════════════════════════
# 4c. the bundle is a SELF-CONTAINED MODEL DIRECTORY
# ════════════════════════════════════════════════════════════════════════════
# ``pack`` used to write only the ommx_w shards and ommx_w_index.json, so the
# documented recipe — pack, then point vLLM at the bundle directory — could not
# resolve the model at all: no config.json means no architecture to build.

def test_bundle_carries_the_model_files_byte_for_byte(synthetic, tmp_path):
    out = str(tmp_path / "bundle")
    _pack(synthetic, out)
    index = F.validate_bundle(out)
    aux = index["aux_files"]
    assert aux["policy"] == "copy" and aux["self_contained"] is True

    copied = {e["name"]: e for e in aux["copied"]}
    assert set(copied) == set(P.SYNTHETIC_AUX_FILES), sorted(copied)
    for name, ent in copied.items():
        src = os.path.join(synthetic, name)
        dst = os.path.join(out, name)
        assert os.path.isfile(dst), f"{name} is recorded as copied but is not in the bundle"
        if name != "config.json":
            with open(src, "rb") as a, open(dst, "rb") as b:
                assert a.read() == b.read(), f"{name} was altered in transit"
        # The recorded provenance describes the file that is actually there -- including
        # config.json, whose digest is re-taken AFTER the stamp below.
        assert ent["bytes"] == os.path.getsize(dst)
        assert ent["sha256"] == F.sha256_file(dst)

    # config.json is the ONE file the packer deliberately edits, and the one that makes or
    # breaks a load; state it separately so a failure here reads as "the bundle is
    # unloadable", not "a file list changed".
    #
    # It is NOT byte-for-byte: the packer stamps `quantization_config` into it, because
    # vLLM only calls OMMXWConfig.from_config when that key is present -- without it
    # `--quantization ommx_w` is rejected by name and the bundle is undiscoverable. The
    # contract is therefore "every OTHER key survives unchanged, plus exactly that one".
    src_cfg = json.load(open(os.path.join(synthetic, "config.json")))
    dst_cfg = json.load(open(os.path.join(out, "config.json")))
    assert dst_cfg["architectures"]
    assert dst_cfg.get(P.HF_QUANT_CONFIG_KEY, {}).get("quant_method") == "ommx_w"
    assert {k: v for k, v in dst_cfg.items() if k != P.HF_QUANT_CONFIG_KEY} == src_cfg, \
        "the stamp must add quantization_config and change nothing else"


def test_weight_bearing_source_files_are_never_copied(tmp_path):
    """A leftover unquantized weight file in the bundle is a SILENT bf16 fallback.

    A loader that finds ``pytorch_model.bin`` (or the source's own safetensors shards)
    next to the ommx planes can serve the ORIGINAL weights while the arm is labelled
    ``ommx_w`` — the same defect MEASURED_FACTS §2 records on the attention side,
    where a CUSTOM run was byte-identical to bf16 and timed under the OMMX label.
    """
    src = str(tmp_path / "ck")
    P.make_synthetic_checkpoint(src, layers=1, hidden=64, inter=128, vocab=64, shards=1)
    # A checkpoint that carries BOTH formats, which is common on the Hub.
    open(os.path.join(src, "pytorch_model.bin"), "wb").write(b"\x00" * 32)
    with open(os.path.join(src, "pytorch_model.bin.index.json"), "w") as fh:
        json.dump({"weight_map": {}}, fh)

    out = str(tmp_path / "bundle")
    _pack(src, out)
    index = F.validate_bundle(out)

    present = set(os.listdir(out))
    for banned in ("pytorch_model.bin", "pytorch_model.bin.index.json",
                   "model.safetensors.index.json", "model-00001-of-00001.safetensors"):
        assert banned not in present, f"{banned} was copied into the bundle"
    # The ONLY safetensors files in a bundle are its own shards.
    assert all(n.startswith("ommx_w-") for n in present if n.endswith(".safetensors"))
    # ...and every exclusion is written down with a reason, not silently dropped.
    skipped = {e["name"]: e["reason"] for e in index["aux_files"]["skipped"]}
    assert "pytorch_model.bin" in skipped
    assert "carries weights" in skipped["pytorch_model.bin"]
    assert "model.safetensors.index.json" in skipped


def test_unrecognised_non_weight_files_are_copied_not_dropped(tmp_path):
    """The copy policy is EXCLUSION, not a whitelist.

    The set of files HF needs keeps growing (chat templates, preprocessor configs,
    per-model extras). A packer that copied only files it recognised would drop the
    next one silently, and the bundle would misbehave in a way that reads as
    quantization damage.
    """
    src = str(tmp_path / "ck")
    P.make_synthetic_checkpoint(src, layers=1, hidden=64, inter=128, vocab=64, shards=1)
    exotic = ("preprocessor_config.json", "chat_template.json",
              "a_file_this_packer_has_never_heard_of.yaml", "README.md")
    for name in exotic:
        with open(os.path.join(src, name), "w") as fh:
            fh.write(name)

    out = str(tmp_path / "bundle")
    _pack(src, out)
    copied = {e["name"] for e in F.validate_bundle(out)["aux_files"]["copied"]}
    for name in exotic:
        assert name in copied, f"{name} was dropped"
        assert open(os.path.join(out, name)).read() == name


def test_pack_refuses_a_source_with_no_config_json(tmp_path, capsys):
    """FAIL, not warn: a bundle with no config.json cannot be identified by any loader.

    Warning instead would move the diagnosis an hour downstream into a transformers
    traceback about a missing ``architectures`` key, on a directory the packer had
    already reported as successfully written.
    """
    src = str(tmp_path / "ck")
    P.make_synthetic_checkpoint(src, layers=1, hidden=64, inter=128, vocab=64,
                                shards=1, aux=False)
    assert not os.path.exists(os.path.join(src, "config.json"))
    out = str(tmp_path / "bundle")
    with pytest.raises(P.PackError) as exc:
        _pack(src, out)
    msg = str(exc.value)
    assert "config.json" in msg
    assert "--no-copy-aux" in msg           # the escape hatch is named, not hidden
    # Nothing was written: a refused pack must not leave a half-bundle behind.
    assert not os.path.exists(out) or not os.listdir(out)
    # ...and it refuses BEFORE quantizing anything, so --dry-run reports it in a second.
    with pytest.raises(P.PackError):
        _pack(src, str(tmp_path / "b2"), dry_run=True)


def test_no_copy_aux_produces_a_bundle_that_admits_it_is_not_loadable(tmp_path, capsys):
    """The escape hatch cannot produce a bundle that LOOKS self-contained."""
    src = str(tmp_path / "ck")
    P.make_synthetic_checkpoint(src, layers=1, hidden=64, inter=128, vocab=64, shards=1)
    out = str(tmp_path / "bundle")
    _pack(src, out, copy_aux=False)
    index = F.validate_bundle(out)          # a valid bundle, just not a model directory
    aux = index["aux_files"]
    assert aux["policy"] == "no-copy-aux"
    assert aux["self_contained"] is False
    assert aux["copied"] == []
    assert not os.path.exists(os.path.join(out, "config.json"))
    # The files it deliberately did not take are still enumerated.
    assert "config.json" in {e["name"] for e in aux["missing"]}
    assert "NOT a loadable model directory" in capsys.readouterr().err


def test_pack_warns_when_a_behaviour_changing_aux_file_is_absent(tmp_path, capsys):
    """Absent generation_config.json is legal, silent, and changes when generation stops."""
    src = str(tmp_path / "ck")
    P.make_synthetic_checkpoint(src, layers=1, hidden=64, inter=128, vocab=64, shards=1)
    os.unlink(os.path.join(src, "generation_config.json"))
    _pack(src, str(tmp_path / "bundle"))
    err = capsys.readouterr().err
    assert "generation_config.json" in err and "WARNING" in err


def test_validator_rejects_a_bundle_whose_model_files_vanished(synthetic, tmp_path):
    """A bundle that lost its config.json is not the model directory it claims to be."""
    out = str(tmp_path / "bundle")
    _pack(synthetic, out)
    F.validate_bundle(out)
    os.unlink(os.path.join(out, "config.json"))
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    msg = str(exc.value)
    assert "config.json" in msg and "self-contained" in msg


def test_validator_rejects_an_empty_required_model_file(synthetic, tmp_path):
    """0 bytes (a full disk mid-copy) is present-but-unloadable, so it is refused."""
    out = str(tmp_path / "bundle")
    _pack(synthetic, out)
    open(os.path.join(out, "config.json"), "wb").close()
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    assert "EMPTY" in str(exc.value)


def test_validator_rejects_a_false_self_contained_claim(synthetic, tmp_path):
    """``self_contained`` is checked against the facts, not taken as a free-text claim."""
    out = str(tmp_path / "bundle")
    _pack(synthetic, out)
    idx_path = os.path.join(out, F.INDEX_FILENAME)
    doc = json.load(open(idx_path, encoding="utf-8"))
    doc["aux_files"]["policy"] = "no-copy-aux"      # ...while still listing copies
    json.dump(doc, open(idx_path, "w", encoding="utf-8"))
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    assert "self_contained" in str(exc.value)

    doc["aux_files"]["policy"] = "copy"
    doc["aux_files"]["copied"] = [e for e in doc["aux_files"]["copied"]
                                  if e["name"] != "config.json"]
    json.dump(doc, open(idx_path, "w", encoding="utf-8"))
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    assert "required file(s)" in str(exc.value) and "config.json" in str(exc.value)


def test_validator_rejects_an_aux_entry_that_escapes_the_bundle(synthetic, tmp_path):
    """An index must not be able to point a reader outside the bundle directory."""
    out = str(tmp_path / "bundle")
    _pack(synthetic, out)
    idx_path = os.path.join(out, F.INDEX_FILENAME)
    doc = json.load(open(idx_path, encoding="utf-8"))
    doc["aux_files"]["copied"].append({"name": "../config.json", "bytes": 1,
                                       "sha256": "0" * 64})
    json.dump(doc, open(idx_path, "w", encoding="utf-8"))
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    assert "plain file name" in str(exc.value)


def test_validator_rejects_a_bundle_with_no_aux_record(synthetic, tmp_path):
    """A v2 bundle that does not say what model files it carries is malformed."""
    out = str(tmp_path / "bundle")
    _pack(synthetic, out)
    idx_path = os.path.join(out, F.INDEX_FILENAME)
    doc = json.load(open(idx_path, encoding="utf-8"))
    del doc["aux_files"]
    json.dump(doc, open(idx_path, "w", encoding="utf-8"))
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    assert "aux_files" in str(exc.value)


def test_repacking_after_a_reshard_refuses_to_leave_stale_ommx_shards(tmp_path):
    """DEFECT THIS CLOSES (found by the adversarial pass, not reported by the author).

    The shard file name encodes the shard COUNT. Re-packing the same model with
    ``--overwrite`` after the source checkpoint was re-sharded writes a NEW set of
    names and leaves the OLD set behind. The index references only the new ones, so
    ``verify`` said OK while the directory held two complete, disagreeing copies of
    every plane — reachable in one command, and the exact hazard the ``*.safetensors``
    copy exclusion exists to prevent, arriving through the packer's own output.

    Reproduced here end to end: pack 2 shards, re-shard the source to 1, re-pack with
    ``--overwrite``. The first assertion is the bug (stale files must not survive); the
    second is that the refusal happens BEFORE anything is written, so a refused re-pack
    leaves the previous bundle exactly as it was rather than half-replaced.
    """
    src = str(tmp_path / "ck")
    out = str(tmp_path / "bundle")
    P.make_synthetic_checkpoint(src, layers=2, hidden=64, inter=128, vocab=64, shards=2)
    _pack(src, out)
    first = sorted(f for f in os.listdir(out) if f.endswith(".safetensors"))
    assert len(first) == 2, first
    before = {f: os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)}

    # Same model, same recipe, same --source-model: re-sharded to one file.
    for f in os.listdir(src):
        if f.endswith(".safetensors"):
            os.unlink(os.path.join(src, f))
    P.make_synthetic_checkpoint(src, layers=2, hidden=64, inter=128, vocab=64, shards=1)

    with pytest.raises(P.PackError) as exc:
        _pack(src, out, overwrite=True)
    msg = str(exc.value)
    assert "ommx_w-00001-of-00002.safetensors" in msg
    assert "ommx_w-00001-of-00001.safetensors" in msg    # what it WOULD have written
    # Nothing was written, so the bundle that was already there is still the one there.
    assert {f: os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)} == before
    F.validate_bundle(out)

    # The escape hatch the message names really works: a fresh directory packs fine.
    fresh = str(tmp_path / "fresh")
    _pack(src, fresh)
    assert sorted(f for f in os.listdir(fresh) if f.endswith(".safetensors")) == [
        "ommx_w-00001-of-00001.safetensors"]


def test_pack_refuses_to_write_into_its_own_source(tmp_path):
    """--input == --output would read a checkpoint while overwriting it."""
    src = str(tmp_path / "ck")
    P.make_synthetic_checkpoint(src, layers=1, hidden=64, inter=128, vocab=64, shards=1)
    with pytest.raises(P.PackError) as exc:
        _pack(src, src)
    assert "same directory" in str(exc.value)


# ════════════════════════════════════════════════════════════════════════════
# 5. bits/weight — independent longhand recomputation
# ════════════════════════════════════════════════════════════════════════════

def _longhand_bits_per_weight(gs: int, npv: int, repr_: str, map_: str,
                              zp_bytes: int, map_bytes: int, N: int, K: int) -> float:
    """Bits/weight written out by hand from the SPEC, touching no packer code.

    Deliberately duplicated arithmetic: if this and ``Recipe.bits_breakdown()`` were
    the same function, agreeing would prove nothing.
    """
    G = K // gs
    code = N * (K // 4)                                   # 4 INT2 codes per byte
    scale = N * G * 1                                     # int8 E8M0 exponent
    zp = N * G * zp_bytes
    if npv:
        idx_b = math.ceil(npv * 7 / 8) if repr_ == "relidx7" else math.ceil(gs / 8)
        oindex = N * G * idx_b
        ocode = N * G * math.ceil(npv / 2)                # 2 FP4 nibbles per byte
    else:
        oindex = ocode = 0
    mp = 2 * N * G * map_bytes if (npv and map_ == "idx_range") else 0
    return 8.0 * (code + scale + zp + oindex + ocode + mp) / (N * K)


@pytest.mark.parametrize("gs,npv,repr_,map_,zp_dtype,zp_bytes", [
    (64, 4, "relidx7", "idx_range", torch.bfloat16, 2),
    (64, 4, "relidx7", "none", torch.bfloat16, 2),
    (64, 4, "bitmap", "none", torch.bfloat16, 2),
    (64, 8, "relidx7", "none", torch.bfloat16, 2),
    (64, 16, "relidx7", "none", torch.bfloat16, 2),
    (32, 3, "relidx7", "idx_range", torch.float32, 4),
    (128, 8, "bitmap", "idx_range", torch.bfloat16, 2),
])
def test_reported_bits_per_weight_matches_longhand(tmp_path, gs, npv, repr_, map_,
                                                   zp_dtype, zp_bytes):
    """The packer's reported bits/weight == longhand == bytes actually on disk."""
    N, K = 8, 256
    want = _longhand_bits_per_weight(gs, npv, repr_, map_, zp_bytes, 4, N, K)
    recipe = F.Recipe(gs, npv, npv / gs, repr_, map_, zp_dtype)
    assert recipe.bits_breakdown()["bits_per_weight"] == pytest.approx(want, abs=1e-12)
    assert recipe.packed_bytes(N, K) * 8.0 / (N * K) == pytest.approx(want, abs=1e-12)

    # ...and the same number comes out of a real file rather than a formula.
    name = "model.layers.0.mlp.up_proj.weight"
    torch.manual_seed(1)
    planes = P._quantize_one(torch.randn(N, K) * 0.1, name, recipe)
    path = str(tmp_path / "one.safetensors")
    w = F.SafeTensorsWriter(path)
    total = sum(w.add(F.plane_name(name, p), planes[p]) for p in recipe.planes())
    w.close({"format": F.OMMX_W_FORMAT, "version": "1", "manifest": "{}"})
    assert total == recipe.packed_bytes(N, K)
    assert F.bits_per_weight(recipe, N, K, total) == pytest.approx(want, abs=1e-12)


def test_readme_published_bit_budgets_reproduce():
    """The figures ``csrc/linear/README.md`` PUBLISHES: 4.125 / 4.750 / 6.125 b/wt.

    Those are the STORED budget of the shipped recipe — gs=64, ``relidx7`` positions
    plus the ``idx_range`` FP4 range map, which is what ``Recipe`` defaults to — at
    npv 4 / 8 / 16.

    This gate (and its name) used to say the README published 3.06 / 3.75 / 5.125.
    It no longer does, and says so at length: those figures omitted the ``map_scale``
    / ``map_center`` planes the timed kernel actually takes as arguments (+1.0 b/wt
    exactly at gs=64), and at npv=4 they quoted the unpadded column while the layout
    pads a 28-bit position stream to 4 bytes per group. Both corrections are pinned
    below, so the arithmetic and the prose cannot drift apart again in either
    direction: the published stored figures, the no-map column they were mistaken
    for, and the exact size of each of the two errors.
    """
    published_stored = {4: 4.125, 8: 4.750, 16: 6.125}
    for npv, published in published_stored.items():
        r = F.Recipe(64, npv, npv / 64, "relidx7", "idx_range")     # the SHIPPED recipe
        assert r.outlier_repr == "relidx7" and r.outlier_map == "idx_range"
        bb = r.bits_breakdown()
        assert bb["bits_per_weight"] == pytest.approx(published, abs=1e-9), npv
        # error 1: the range map is exactly +1.0 b/wt at gs=64 (two F32 per 64 weights).
        no_map = F.Recipe(64, npv, npv / 64, "relidx7", "none").bits_breakdown()
        assert bb["stored"]["fp4_range_map"] == pytest.approx(1.0, abs=1e-9)
        assert (bb["bits_per_weight"] - no_map["bits_per_weight"]
                == pytest.approx(1.0, abs=1e-9))

    # error 2: stored vs unpadded differ ONLY at npv=4, and by exactly 1/16 b/wt
    # (28 index bits padded to 32 per group of 64).
    for npv, unpadded in ((4, 3.0625), (8, 3.75), (16, 5.125)):
        bb = F.Recipe(64, npv, npv / 64, "relidx7", "none").bits_breakdown()
        assert bb["bits_per_weight_unpadded"] == pytest.approx(unpadded, abs=1e-9), npv
        delta = bb["bits_per_weight"] - bb["bits_per_weight_unpadded"]
        assert delta == pytest.approx(0.0625 if npv == 4 else 0.0, abs=1e-9), npv
    assert F.Recipe(64, 4, 0.0625, "relidx7", "none").bits_breakdown()[
        "bits_per_weight"] == pytest.approx(3.125, abs=1e-9)


def test_paper_3_63_avgbits_is_the_bitmap_recipe():
    """Paper Table 1 (claim E2) reports weight AvgBits 3.63. Locate the solution SET.

    3.625 = INT2 base (2) + flat bitmask (1 bit/element, paper claim B4) + INT8 E8M0
    scale (8/64) + BF16 zero-point (16/64) + 4 FP4 outliers (4*4/64) at group 64.

    This test deliberately pins the WHOLE solution set, not one member of it. An
    earlier version of this gate asserted only that gs=64/npv=4/bitmap/none hits
    3.625 while its docstring claimed that recipe was UNIQUE and that relidx7 could
    never reach 3.625. Both of those claims are false, and the assertion could not
    have caught either, because it never enumerated anything. The exhaustive scan
    below is the fix: it fails if the solution set changes at all, so the prose can
    no longer drift away from the arithmetic.
    """
    hit = F.Recipe(64, 4, 0.0625, "bitmap", "none").bits_breakdown()
    assert hit["bits_per_weight"] == pytest.approx(3.625, abs=1e-9)
    # 3.625 prints as 3.63 under round-half-UP, which is what a results table shows.
    # (python's round()/format() are round-half-to-EVEN and would print 3.62 — worth
    # writing down, because it is the difference between "matches the paper" and
    # "off by one in the last digit".)
    assert math.floor(hit["bits_per_weight"] * 100 + 0.5) / 100 == 3.63

    # The shipped relidx7 ladder at gs=64 with no map STEPS OVER 3.625 (3.125 -> 3.75),
    # which is the true, narrow version of "relidx7 does not get you there".
    ladder = {npv: F.Recipe(64, npv, npv / 64, "relidx7", "none")
              .bits_breakdown()["bits_per_weight"] for npv in (4, 8)}
    assert ladder[4] < 3.625 < ladder[8], ladder

    # ...but relidx7 DOES reach 3.625 elsewhere in the grid. Exhaustive scan over the
    # packer's whole recipe space at the paper's BF16 zero-point (claim B9).
    found = []
    for gs in (16, 32, 64, 128):
        for npv in range(0, gs + 1):
            for repr_ in F.OUTLIER_REPRS:
                for map_ in ("idx_range", "none"):
                    try:
                        r = F.Recipe(gs, npv, npv / gs, repr_, map_, torch.bfloat16)
                    except F.OMMXWFormatError:
                        continue            # recipe the format itself refuses
                    if abs(r.bits_breakdown()["bits_per_weight"] - 3.625) < 1e-9:
                        found.append((gs, npv, repr_, map_))
    assert found == [
        (64, 1, "relidx7", "idx_range"),
        (64, 3, "bitmap", "none"),
        (64, 4, "bitmap", "none"),
        (128, 13, "bitmap", "none"),
        (128, 14, "bitmap", "none"),
    ], found
    # Four of the five are bitmap + no range map, which is the family claims B4/B6
    # describe -> "3.63 is the bitmap recipe FAMILY" is supportable. "3.63 identifies
    # a unique recipe" is not: gs=128/npv=13 and 14 are equally consistent and match
    # the paper's own N=128 worked example (claim B10).
    assert sum(1 for _, _, rp, mp in found if (rp, mp) == ("bitmap", "none")) == 4


# ════════════════════════════════════════════════════════════════════════════
# 6. the validator refuses tampered bundles
# ════════════════════════════════════════════════════════════════════════════

def _rewrite_shard(path: str, *, drop: str = None, metadata_edit=None,
                   manifest_edit=None) -> None:
    """Rewrite one shard with a deliberate defect (tests only)."""
    st = F.SafeTensorsFile(path)
    md = dict(st.metadata)
    tensors: List[Tuple[str, torch.Tensor]] = [
        (n, st.tensor(n)) for n in st.names if n != drop]
    if manifest_edit is not None:
        man = json.loads(md["manifest"])
        manifest_edit(man)
        md["manifest"] = json.dumps(man)
    if metadata_edit is not None:
        metadata_edit(md)
    w = F.SafeTensorsWriter(path)
    for n, t in tensors:
        w.add(n, t)
    w.close(md)


@pytest.fixture()
def bundle(synthetic, tmp_path):
    out = str(tmp_path / "bundle")
    _pack(synthetic, out)
    index = F.validate_bundle(out)
    return out, index


def test_validator_rejects_dropped_plane(bundle):
    out, index = bundle
    name = "model.layers.0.self_attn.q_proj.weight"
    shard = os.path.join(out, index["weight_map"][name])
    _rewrite_shard(shard, drop=F.plane_name(name, "ocode"))
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    assert "ocode" in str(exc.value)


def test_validator_rejects_corrupted_shape(bundle):
    out, index = bundle
    name = "model.layers.0.mlp.down_proj.weight"
    shard = os.path.join(out, index["weight_map"][name])

    def edit(man):
        man["tensors"][name]["planes"]["scale_exp"]["shape"] = [1, 1]
    _rewrite_shard(shard, manifest_edit=edit)
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    msg = str(exc.value)
    assert "scale_exp" in msg and "recipe requires" in msg


def test_validator_rejects_unknown_version(bundle):
    out, index = bundle
    shard = os.path.join(out, index["shards"][0])

    def edit(md):
        md["version"] = "99"
    _rewrite_shard(shard, metadata_edit=edit)
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    msg = str(exc.value)
    assert "unknown OMMX_W_SafeTensor version 99" in msg
    # An unknown version must still hand the operator the command that fixes it.
    assert "w_packer pack" in msg


def test_validator_refuses_version_1_by_name_with_a_repack_command(bundle):
    """A v1 bundle is refused BY NAME, with the reason and the re-pack command.

    v1 named planes ``<module>.weight.ommx_<plane>``; nothing in a vLLM/transformers
    model is called that, so a v1 bundle fails at load deep inside a weight loader
    with "parameter not found". Accepting v1 "for compatibility" would mean guessing
    which convention a name follows, so it is refused — but the refusal has to be
    legible enough that nobody goes looking for a compat flag.
    """
    out, index = bundle
    shard = os.path.join(out, index["shards"][0])

    def edit(md):
        md["version"] = "1"
    _rewrite_shard(shard, metadata_edit=edit)
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    msg = str(exc.value)
    assert "VERSION 1" in msg
    assert ".weight" in msg and "ommx_" in msg          # the reason, spelled out
    assert "python -m ommx_gpu_serve.linear.w_packer pack" in msg
    assert "no in-place migration" in msg

    # ...and the same refusal comes from the INDEX, not only from a shard: an operator
    # who validates a directory must not have to reach a shard to get the message.
    idx_path = os.path.join(out, F.INDEX_FILENAME)
    with open(idx_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["version"] = 1
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    assert "VERSION 1" in str(exc.value) and F.INDEX_FILENAME in str(exc.value)


def test_validator_rejects_a_v1_named_plane_inside_a_v2_manifest(bundle):
    """The manifest does not get to CHOOSE a plane's tensor name.

    A shard can be perfectly self-consistent — manifest says ``x.weight.ommx_code``,
    file contains ``x.weight.ommx_code`` — and still bind to nothing in a real engine.
    The name is a function of (weight, plane), so the validator recomputes it rather
    than trusting the ``name`` field. Without this check a packer that bypassed
    ``plane_name`` would round-trip through its own validator unnoticed.
    """
    out, index = bundle
    name = "model.layers.0.self_attn.q_proj.weight"
    shard = os.path.join(out, index["weight_map"][name])
    v1_name = name + ".ommx_code"                       # the format-v1 spelling

    st = F.SafeTensorsFile(shard)
    md = dict(st.metadata)
    man = json.loads(md["manifest"])
    man["tensors"][name]["planes"]["code"]["name"] = v1_name
    md["manifest"] = json.dumps(man)
    tensors = [(v1_name if n == F.plane_name(name, "code") else n, st.tensor(n))
               for n in st.names]
    w = F.SafeTensorsWriter(shard)
    for n, t in tensors:
        w.add(n, t)
    w.close(md)

    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    msg = str(exc.value)
    assert v1_name in msg and "w_packer pack" in msg


def test_validator_rejects_a_foreign_naming_convention(bundle):
    """The self-describing ``naming`` block is CHECKED, not merely carried.

    A bundle that declares the v1 convention while claiming v2 would load its planes
    through the manifest's explicit ``name`` fields and bind none of them.
    """
    out, index = bundle
    shard = os.path.join(out, index["shards"][0])

    def edit(man):
        man["naming"]["plane_name"] = "{module}.weight.ommx_{plane}"
    _rewrite_shard(shard, manifest_edit=edit)
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    msg = str(exc.value)
    assert "plane-naming convention" in msg and "weight.ommx_" in msg

# ── gates added by the adversarial verifier ────────────────────────────────────
# Each one closes a guard that a deliberate mutation walked straight through: the
# code was already right, but nothing failed when it was broken. The mutation that
# each of these kills is named in its docstring so a future edit can re-run it.

def test_validator_rejects_a_foreign_naming_convention_in_the_INDEX(bundle):
    """The index's ``naming`` block is checked too, not only each shard's.

    ``validate_bundle`` calls ``check_naming(INDEX_FILENAME, index["naming"])``, but
    every existing gate edits a SHARD manifest, so deleting that index-side call left
    the suite green. The index is the file a loader reads FIRST (it is what maps a
    tensor name to a shard), so a bundle whose index declares the v1 convention while
    its shards use v2 is exactly the mixed-convention artefact the block exists to
    refuse.  MUTATION: delete the two ``naming`` lines from ``validate_bundle``.
    """
    out, _index = bundle
    idx_path = os.path.join(out, F.INDEX_FILENAME)
    doc = json.load(open(idx_path, encoding="utf-8"))
    doc["naming"]["plane_name"] = "{module}.weight.ommx_{plane}"      # the v1 spelling
    json.dump(doc, open(idx_path, "w", encoding="utf-8"))
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    msg = str(exc.value)
    assert F.INDEX_FILENAME in msg and "plane-naming convention" in msg

    # ...and an index with NO naming block at all is refused by name, rather than
    # reaching check_naming and dying on a KeyError an operator cannot act on.
    doc["naming"] = F.naming_document()
    del doc["naming"]["rationale"]           # a partial block is still a foreign one
    json.dump(doc, open(idx_path, "w", encoding="utf-8"))
    with pytest.raises(F.OMMXWFormatError):
        F.validate_bundle(out)
    doc.pop("naming")
    json.dump(doc, open(idx_path, "w", encoding="utf-8"))
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    assert "naming" in str(exc.value)


def test_the_read_path_derives_plane_names_too_not_only_the_validator(bundle):
    """``load_planes`` refuses a v1-named plane, exactly as ``validate_shard`` does.

    DEFECT THIS CLOSES (found by mutation, not reported by the author): the validator
    was hardened to RECOMPUTE a plane's tensor name from ``(weight, plane)``, but the
    module's own read path still preferred the manifest's ``name`` field. The result
    was a shard that ``validate_bundle`` refuses and ``load_weight`` reads anyway —
    so the CPU oracle could produce clean-looking numbers for a bundle no engine can
    bind, which is the one piece of evidence that must never be obtainable.

    Both directions are asserted here, because a fix that only made the loader raise
    on a name it does not like would be satisfied by a loader that raised on the
    CORRECT name too.
    """
    out, index = bundle
    name = "model.layers.0.self_attn.q_proj.weight"
    shard = os.path.join(out, index["weight_map"][name])

    # The untampered bundle still loads (the derivation is the right derivation).
    assert F.load_weight(shard, name).shape == (64, 64)

    v1_name = name + ".ommx_code"                   # the format-v1 spelling
    st = F.SafeTensorsFile(shard)
    md = dict(st.metadata)
    man = json.loads(md["manifest"])
    man["tensors"][name]["planes"]["code"]["name"] = v1_name
    md["manifest"] = json.dumps(man)
    tensors = [(v1_name if n == F.plane_name(name, "code") else n, st.tensor(n))
               for n in st.names]
    w = F.SafeTensorsWriter(shard)
    for n, t in tensors:
        w.add(n, t)
    w.close(md)

    with pytest.raises(F.OMMXWFormatError) as exc:
        F.load_weight(shard, name)
    msg = str(exc.value)
    assert v1_name in msg and "w_packer pack" in msg
    with pytest.raises(F.OMMXWFormatError):
        F.load_planes(shard, name)
    # ...and the validator has not been loosened to compensate.
    with pytest.raises(F.OMMXWFormatError):
        F.validate_bundle(out)


def test_validator_rejects_a_shard_manifest_with_no_naming_block(bundle):
    """A missing ``naming`` key is an OMMXWFormatError, not a raw KeyError.

    ``read_manifest`` requires the key before it dereferences it. Dropping ``naming``
    from that required-key tuple left every test green (the packer always writes the
    key), and turned a legible refusal into ``KeyError: 'naming'`` from inside the
    validator.  MUTATION: remove ``"naming"`` from ``read_manifest``'s key tuple.
    """
    out, index = bundle
    shard = os.path.join(out, index["shards"][0])
    _rewrite_shard(shard, manifest_edit=lambda man: man.pop("naming"))
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    assert "naming" in str(exc.value)


def test_validator_rejects_an_unknown_aux_policy(bundle):
    """``aux_files.policy`` must be one of the two policies, not free text.

    Nothing exercised the membership check: the existing self_contained gate swaps
    ``copy`` for the OTHER legal policy. An unrecognised policy would otherwise fall
    through to ``want_self_contained = (policy == "copy" and ...)`` = False and
    validate cleanly, so a bundle could describe its own model files in a vocabulary
    this build does not implement.  MUTATION: ``if policy not in AUX_POLICIES`` -> ``if False``.
    """
    out, _index = bundle
    idx_path = os.path.join(out, F.INDEX_FILENAME)
    doc = json.load(open(idx_path, encoding="utf-8"))
    doc["aux_files"]["policy"] = "copy-but-only-the-nice-ones"
    doc["aux_files"]["self_contained"] = False      # self-consistent, still refused
    json.dump(doc, open(idx_path, "w", encoding="utf-8"))
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    msg = str(exc.value)
    assert "aux_files.policy" in msg
    assert str(list(F.AUX_POLICIES)) in msg         # the legal set is named


def test_validator_rejects_a_plane_stored_with_the_wrong_dtype(bundle):
    """The dtype ON DISK is compared, not only the one the manifest declares.

    A plane rewritten as F32 keeps its ``[N, G]`` shape, so every shape check —
    manifest-vs-recipe and manifest-vs-file — still agrees; only ``_check_tensor``'s
    dtype comparison stands between that file and a dequantize that reinterprets 4
    bytes as 2 bf16 values. The existing tamper gate edits a SHAPE, so dropping the
    dtype half of that comparison left the suite green.
    MUTATION: ``if got_shape != want or got_dtype != dtype`` -> ``if got_shape != want``.
    """
    out, index = bundle
    name = "model.layers.0.self_attn.q_proj.weight"
    shard = os.path.join(out, index["weight_map"][name])
    zp = F.plane_name(name, "zp")

    st = F.SafeTensorsFile(shard)
    md = dict(st.metadata)
    # Same shape, wider dtype: the manifest is left untouched and still says BF16.
    tensors = [(n, st.tensor(n).to(torch.float32) if n == zp else st.tensor(n))
               for n in st.names]
    w = F.SafeTensorsWriter(shard)
    for n, t in tensors:
        w.add(n, t)
    w.close(md)

    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    msg = str(exc.value)
    assert zp in msg and "F32" in msg and "BF16" in msg


def test_pack_validates_the_bundle_it_just_wrote(synthetic, tmp_path, monkeypatch):
    """``pack_checkpoint`` re-validates its own output before reporting success.

    Nothing pinned that final ``validate_bundle(out_dir)``: every other gate calls the
    validator itself, so deleting the call from the packer changed no test outcome. It
    is the guard that makes "pack returned a report" mean "this directory validates",
    which is the whole basis for pointing an engine at the result.

    Broken here through the aux copy — the copy is made to REPORT a ``config.json`` it
    did not leave behind, which is the real failure (a full disk, a killed process)
    that the self-validation is there to turn into a refusal instead of a bundle that
    only fails hours later inside a loader.
    MUTATION: delete ``validate_bundle(out_dir)`` from ``pack_checkpoint``.
    """
    real_copy = P.copy_aux_files

    def lying_copy(src_dir, out_dir, plan):
        out = real_copy(src_dir, out_dir, plan)
        os.unlink(os.path.join(out_dir, "config.json"))   # reported copied; not there
        return out

    monkeypatch.setattr(P, "copy_aux_files", lying_copy)
    dst = str(tmp_path / "bundle")
    with pytest.raises(F.OMMXWFormatError) as exc:
        _pack(synthetic, dst)
    assert "config.json" in str(exc.value)


def test_a_source_subdirectory_is_recorded_and_never_flattened_in(tmp_path):
    """A nested source layout is SKIPPED with a reason naming it as a directory.

    ``classify_aux_file`` excludes directories first and says so, because copying a
    subdirectory into a flat bundle would rename its files. Nothing had a subdirectory
    in the source, so removing that branch changed nothing observable — the entry then
    falls through to the generic "not a regular file" reason and an operator with a
    nested checkpoint is told the wrong thing about why their files are absent.
    MUTATION: ``if os.path.isdir(path)`` -> ``if False``.
    """
    src = str(tmp_path / "ck")
    P.make_synthetic_checkpoint(src, layers=1, hidden=64, inter=128, vocab=64, shards=1)
    os.makedirs(os.path.join(src, "nested"))
    with open(os.path.join(src, "nested", "extra.json"), "w") as fh:
        fh.write("{}")

    out = str(tmp_path / "bundle")
    report = _pack(src, out)                       # must not crash on the directory
    index = F.validate_bundle(out)
    skipped = {e["name"]: e["reason"] for e in index["aux_files"]["skipped"]}
    assert "nested" in skipped, sorted(skipped)
    assert "directory" in skipped["nested"], skipped["nested"]
    assert "nested" not in {e["name"] for e in index["aux_files"]["copied"]}
    assert not os.path.exists(os.path.join(out, "nested"))
    assert not os.path.exists(os.path.join(out, "extra.json"))   # not flattened in
    assert report["aux_files"]["self_contained"] is True


def test_a_stale_ommx_w_index_in_the_source_is_not_copied(tmp_path):
    """The packer's own output filenames are never taken in as "aux files".

    Re-packing a directory that already holds a bundle (or a source that was packed in
    place by an older build) would otherwise copy a STALE ``ommx_w_index.json`` in and
    record it in the fresh index's ``copied`` list with the stale file's digest — an
    index that describes itself with the wrong hash, and a ``verify`` that reports
    permanent drift.  MUTATION: ``if name in GENERATED_FILENAMES`` -> ``if False``.
    """
    src = str(tmp_path / "ck")
    P.make_synthetic_checkpoint(src, layers=1, hidden=64, inter=128, vocab=64, shards=1)
    with open(os.path.join(src, F.INDEX_FILENAME), "w", encoding="utf-8") as fh:
        json.dump({"format": "ommx_w", "version": 1, "stale": True}, fh)

    out = str(tmp_path / "bundle")
    _pack(src, out)
    index = F.validate_bundle(out)
    assert F.INDEX_FILENAME not in {e["name"] for e in index["aux_files"]["copied"]}
    skipped = {e["name"]: e["reason"] for e in index["aux_files"]["skipped"]}
    assert F.INDEX_FILENAME in skipped
    assert "packer itself" in skipped[F.INDEX_FILENAME]
    # The index in the bundle is the freshly written one, not the stale document.
    assert "stale" not in index and index["version"] == F.OMMX_W_FORMAT_VERSION

def test_validator_rejects_undeclared_tensor(bundle):
    """A tensor smuggled into a shard without a manifest entry is a hard error."""
    out, index = bundle
    shard = os.path.join(out, index["shards"][0])
    st = F.SafeTensorsFile(shard)
    md = dict(st.metadata)
    tensors = [(n, st.tensor(n)) for n in st.names]
    w = F.SafeTensorsWriter(shard)
    for n, t in tensors:
        w.add(n, t)
    w.add("smuggled.weight", torch.zeros(2, 2))
    w.close(md)
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    assert "smuggled.weight" in str(exc.value)


def test_validator_rejects_recipe_disagreement_between_shards(bundle):
    """Two shards packed with different recipes must not load as one model.

    Edited via ``outlier_pct``, which does NOT change any plane shape — so the shard
    passes its own internal consistency checks and only the CROSS-SHARD comparison
    can catch it. That is the path being tested; a group_size edit would be caught
    one step earlier by the plane-shape check.
    """
    out, index = bundle
    shard = os.path.join(out, index["shards"][1])

    def edit(man):
        man["recipe"]["outlier_pct"] = 0.5
    _rewrite_shard(shard, manifest_edit=edit)
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    assert "disagrees with the bundle recipe" in str(exc.value)


def test_validator_rejects_group_size_edit(bundle):
    """A group_size edit is caught by the plane-shape check, one step earlier."""
    out, index = bundle
    shard = os.path.join(out, index["shards"][1])

    def edit(man):
        man["recipe"]["group_size"] = 32
    _rewrite_shard(shard, manifest_edit=edit)
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    assert "the recipe requires" in str(exc.value)


def test_validator_rejects_wrong_bits_accounting(bundle):
    """A manifest that lies about its own byte/bit budget is refused."""
    out, index = bundle
    name = "model.layers.0.mlp.gate_proj.weight"
    shard = os.path.join(out, index["weight_map"][name])

    def edit(man):
        man["tensors"][name]["bits_per_weight"] = 2.0
    _rewrite_shard(shard, manifest_edit=edit)
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.validate_bundle(out)
    assert "bits_per_weight" in str(exc.value)


# ════════════════════════════════════════════════════════════════════════════
# 7. the packer refuses silently-wrong input
# ════════════════════════════════════════════════════════════════════════════

def test_refuses_group_size_that_does_not_divide_k(synthetic, tmp_path):
    with pytest.raises(P.PackError) as exc:
        _pack(synthetic, str(tmp_path / "b"), gs=48, npv=3, pct=3 / 48)
    assert "not divisible by group_size=48" in str(exc.value)


def test_refuses_unhandleable_dtype(tmp_path):
    src = tmp_path / "ck"
    src.mkdir()
    w = F.SafeTensorsWriter(str(src / "model.safetensors"))
    w.add("model.layers.0.self_attn.q_proj.weight", torch.zeros(64, 64, dtype=torch.int8))
    w.close({"format": "pt"})
    # The packer checks the source is a MODEL DIRECTORY before it quantizes anything,
    # so this hand-built checkpoint needs a config.json to reach the dtype gate at all.
    (src / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
    with pytest.raises(P.PackError) as exc:
        _pack(str(src), str(tmp_path / "b"))
    assert "dtype I8 cannot be quantized" in str(exc.value)


def test_refuses_existing_bundle(synthetic, tmp_path):
    out = str(tmp_path / "bundle")
    _pack(synthetic, out)
    # identical manifest -> needs an explicit --overwrite
    with pytest.raises(P.PackError) as exc:
        _pack(synthetic, out)
    assert "--overwrite" in str(exc.value)
    _pack(synthetic, out, overwrite=True)          # ...which then works
    # different recipe -> refused outright, overwrite or not
    with pytest.raises(P.PackError) as exc:
        _pack(synthetic, out, npv=8, pct=0.125, overwrite=True)
    assert "DIFFERENT manifest" in str(exc.value)


def test_refuses_nonfinite_weight():
    W = torch.randn(4, 64)
    W[0, 0] = float("nan")
    with pytest.raises(OMMXQuantizeError) as exc:
        quantize_ommx_weight(W, 64, 0.0625)
    assert "NaN/Inf" in str(exc.value)


def test_refuses_relidx7_beyond_seven_bits():
    """A 7-bit position field cannot address a group of 256; say so, do not truncate."""
    with pytest.raises(OMMXQuantizeError) as exc:
        quantize_ommx_weight(torch.randn(2, 256), 256, 0.0625)
    assert "bitmap" in str(exc.value)
    # bitmap has no such limit
    q = quantize_ommx_weight(torch.randn(2, 256), 256, 0.0625, outlier_repr="bitmap")
    assert q["npv"] == 16 and q["oindex"].numel() == 2 * 1 * 32


def test_refuses_non_float32_input():
    with pytest.raises(OMMXQuantizeError) as exc:
        quantize_ommx_weight(torch.randn(4, 64).to(torch.bfloat16), 64, 0.0625)
    assert "float32" in str(exc.value)


def test_dry_run_writes_nothing(synthetic, tmp_path):
    out = tmp_path / "bundle"
    report = _pack(synthetic, str(out), dry_run=True)
    assert report["dry_run"] is True
    assert not out.exists() or not any(out.iterdir())
    # ...and its byte budget matches what a real pack produces
    real = _pack(synthetic, str(tmp_path / "real"))
    assert (report["totals"]["quantized_packed_bytes"]
            == real["totals"]["quantized_packed_bytes"])
    assert report["totals"]["bits_per_weight"] == real["totals"]["bits_per_weight"]


# ════════════════════════════════════════════════════════════════════════════
# 8. container + CLI
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16,
                                   torch.int8, torch.uint8, torch.int32, torch.int64,
                                   torch.bool])
def test_safetensors_container_round_trip(tmp_path, dtype):
    """The bundled safetensors reader/writer round-trips every dtype the packer emits."""
    torch.manual_seed(0)
    t = (torch.randn(3, 5) * 4).to(dtype)
    path = str(tmp_path / "t.safetensors")
    w = F.SafeTensorsWriter(path)
    w.add("x", t)
    w.close({"note": "hello"})
    st = F.SafeTensorsFile(path)
    assert st.metadata["note"] == "hello"
    assert torch.equal(st.tensor("x"), t)
    # header layout per spec: u64 LE length, then that many JSON bytes, 8-byte aligned
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        assert n % 8 == 0
        assert json.loads(fh.read(n).decode("utf-8"))["x"]["dtype"] \
            == F.st_dtype_name(dtype)


def test_reader_rejects_truncated_file(tmp_path):
    path = str(tmp_path / "t.safetensors")
    w = F.SafeTensorsWriter(path)
    w.add("x", torch.zeros(64))
    w.close({})
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 16)
    with pytest.raises(F.OMMXWFormatError) as exc:
        F.SafeTensorsFile(path).tensor("x")
    assert "truncated" in str(exc.value)


def test_cli_pack_verify_budget(tmp_path, capsys):
    ck = str(tmp_path / "ck")
    bd = str(tmp_path / "bd")
    assert P.main(["make-synthetic", "--output", ck, "--layers", "1", "--hidden", "64",
                   "--inter", "128", "--vocab", "64", "--shards", "1"]) == 0
    assert P.main(["pack", "--input", ck, "--output", bd, "--dry-run"]) == 0
    assert not os.path.exists(os.path.join(bd, F.INDEX_FILENAME))
    assert P.main(["pack", "--input", ck, "--output", bd]) == 0
    assert P.main(["verify", "--bundle", bd]) == 0
    assert P.main(["budget", "--group-sizes", "64", "--npvs", "4"]) == 0
    out = capsys.readouterr().out
    assert "3.6250" in out          # the bitmap/no-map recipe appears in the budget table
    # `verify` must answer "can I point an engine at this directory?" without the
    # operator opening the index by hand.
    assert "self_contained=True" in out
    assert "config.json" in out

    # ...and the CLI's own opt-out round-trips: a weights-only bundle validates and
    # says, in both the pack report and `verify`, that it is not loadable alone.
    bd2 = str(tmp_path / "bd2")
    assert P.main(["pack", "--input", ck, "--output", bd2, "--no-copy-aux"]) == 0
    assert P.main(["verify", "--bundle", bd2]) == 0
    out2 = capsys.readouterr().out
    assert "self_contained=False" in out2
    assert "not a loadable model directory" in out2
    assert not os.path.exists(os.path.join(bd2, "config.json"))
    # a bad recipe exits non-zero rather than raising
    assert P.main(["pack", "--input", ck, "--output", bd, "--npv", "4",
                   "--outlier-pct", "0.25"]) == 2


def test_modules_import_without_gpu_or_triton():
    """Import hygiene: the offline path must never pull in a GPU/serving dependency.

    RUN IN A FRESH INTERPRETER, because the claim is about what importing
    ``ommx_gpu_serve.linear`` DOES, and ``sys.modules`` is process-global: any earlier
    test in the session that legitimately imports the serving integration (the linear
    method round-trip, the plugin registration gates) would otherwise make this fail —
    and on a machine with no vLLM installed it would pass without testing anything.
    Both of those are the assertion answering a question it was not asked.
    """
    import subprocess
    import sys
    probe = (
        "import sys;"
        "import ommx_gpu_serve.linear as L;"
        "assert L.OMMX_W_FORMAT_VERSION == 2, L.OMMX_W_FORMAT_VERSION;"
        "mods = (L.quantize_ommx_weight.__module__,"
        " 'ommx_gpu_serve.linear.w_format', 'ommx_gpu_serve.linear.w_packer');"
        "assert all(m.startswith('ommx_gpu_serve.linear') for m in mods), mods;"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in {'vllm', 'triton'});"
        "assert not leaked, leaked;"
        "print('CLEAN')"
    )
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Explicit PYTHONPATH rather than inheriting: an earlier test in the session may have
    # changed cwd or the environment, and this child must see the repo the way a plain
    # `python -c "import ommx_gpu_serve.linear"` would.
    env = dict(os.environ, PYTHONPATH=repo)
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                       cwd=repo, env=env)
    if r.returncode != 0 and "No module named 'torch'" in r.stderr:
        # The offline path legitimately needs torch. An interpreter without it cannot
        # answer "does importing this pull in vllm/triton" either way, so skipping is the
        # honest outcome -- asserting here would report an environment gap as a hygiene
        # regression.
        pytest.skip(f"{sys.executable} has no torch; cannot probe import hygiene")
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr[-2000:]!r}"
    assert "CLEAN" in r.stdout
