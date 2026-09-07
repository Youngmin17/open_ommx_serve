# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The packer's calibration path: pack a bundle from a solver's DECISIONS, losslessly.

WHY IT EXISTS. The packer chooses its INT2 codes by plain min/max round-to-nearest -- no
calibration, no activation-aware saliency. Measured on Llama-3.1-8B at the repo's own
``shipped-weight`` recipe, that destroys the model: the served output degenerates to a
repeated token fragment, and a pure-numeric simulation (no kernel, no vLLM) reproduces it
exactly, so it is the quantization and not the serving path.
``ommx_fakequant/wq_eval.py``'s error-feedback solver is what makes 2-bit weights work
("strongest exactly at 2-bit", its own docstring) and there was no way to get its result
into a bundle.

WHY DECISIONS RATHER THAN THE CALIBRATED WEIGHT. The solver returns the DEQUANTIZED weight,
and re-quantizing that through the min/max path is NOT idempotent -- measured
``max|W_ref2 - W_ref1| = 5.2e-2``, with different codes AND different scales -- so packing
it the ordinary way would silently discard the calibration and still produce a valid-looking
bundle. ``test_requantizing_an_on_grid_weight_is_not_idempotent`` pins that, because it is
the entire justification for the design.
"""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from ommx_gpu_serve.linear import w_packer as P  # noqa: E402
from ommx_gpu_serve.linear.quantize import (  # noqa: E402
    OMMXQuantizeError,
    quantize_ommx_weight,
)

GS, NPV, PCT = 64, 4, 0.0625
QKW = dict(npv=NPV, outlier_repr="relidx7", outlier_map="idx_range",
           zp_dtype=torch.bfloat16)


def _decide(W):
    """The decisions the DECIDING path makes, in the shape the encode-only path takes."""
    N, K = W.shape
    G = K // GS
    a = quantize_ommx_weight(W, GS, PCT, **QKW)
    om = torch.zeros(N, G, GS, dtype=torch.bool)
    om.scatter_(-1, W.view(N, G, GS).abs().topk(NPV, dim=-1).indices, True)
    return dict(omask=om, scale=a["scale"], zp=a["zp"],
                map_scale=a["map_scale"], map_center=a["map_center"]), a


def _recipe():
    return P.build_recipe(group_size=GS, outlier_pct=None, npv=None,
                          outlier_repr="relidx7", outlier_map="idx_range",
                          zp_dtype="bf16")


# ── the premise ────────────────────────────────────────────────────────────────────

def test_requantizing_an_on_grid_weight_is_not_idempotent():
    """If this ever became idempotent the whole decisions-carrying design would be
    unnecessary -- so it is pinned rather than assumed."""
    torch.manual_seed(0)
    W = torch.randn(128, 512) * 0.05
    W[0, ::37] *= 12.0
    r1 = quantize_ommx_weight(W, GS, PCT, reference=True, **QKW)["W_ref"]
    r2 = quantize_ommx_weight(r1.float(), GS, PCT, reference=True, **QKW)["W_ref"]
    assert float((r2 - r1).abs().max()) > 1e-3, \
        "re-quantization became lossless; revisit whether decisions still need carrying"


# ── the encode-only path ───────────────────────────────────────────────────────────

def test_encode_only_is_bit_identical_to_the_deciding_path():
    """Given the SAME decisions, encoding-only must reproduce every plane exactly --
    otherwise a calibrated bundle would differ from an RTN bundle for reasons that have
    nothing to do with the calibration."""
    torch.manual_seed(0)
    W = torch.randn(256, 512) * 0.05
    W[0, ::37] *= 12.0
    dec, a = _decide(W)
    b = quantize_ommx_weight(W, GS, PCT, decided=dec, reference=True, **QKW)
    for k in ("code", "scale_exp", "zp", "oindex", "ocode", "map_scale", "map_center"):
        assert torch.equal(a[k], b[k]), f"plane {k} differs"
    a_ref = quantize_ommx_weight(W, GS, PCT, reference=True, **QKW)["W_ref"]
    assert float((a_ref - b["W_ref"]).abs().max()) == 0.0


def test_a_non_power_of_two_scale_is_refused():
    """The E8M0 plane stores an exponent BYTE. A calibration run without its pow2 option
    produces full-precision scales, and packing those would round away exactly what the
    calibration chose -- silently."""
    torch.manual_seed(0)
    W = torch.randn(64, 256) * 0.05
    dec, _ = _decide(W)
    dec["scale"] = dec["scale"] * 1.3
    with pytest.raises(OMMXQuantizeError) as e:
        quantize_ommx_weight(W, GS, PCT, decided=dec, **QKW)
    assert "power of two" in str(e.value)
    assert "--pow2" in str(e.value), "the message must name the fix"


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d.__setitem__("omask", torch.zeros_like(d["omask"])), "EXACTLY npv"),
    (lambda d: d.__setitem__("scale", d["scale"][:, :1]), "expected"),
    (lambda d: d.pop("zp"), "missing 'zp'"),
    (lambda d: d.__setitem__("zp", d["zp"] * float("nan")), "NaN/Inf"),
])
def test_malformed_decisions_are_refused(mutate, expect):
    torch.manual_seed(0)
    W = torch.randn(64, 256) * 0.05
    dec, _ = _decide(W)
    mutate(dec)
    with pytest.raises(OMMXQuantizeError) as e:
        quantize_ommx_weight(W, GS, PCT, decided=dec, **QKW)
    assert expect in str(e.value)


# ── the packer path ────────────────────────────────────────────────────────────────

def _export(src, calib, perturb=0.0):
    from safetensors import safe_open
    os.makedirs(calib, exist_ok=True)
    n = 0
    for sh in sorted(f for f in os.listdir(src) if f.endswith(".safetensors")):
        with safe_open(os.path.join(src, sh), framework="pt") as f:
            for name in f.keys():
                if not name.endswith(".weight"):
                    continue
                W = f.get_tensor(name)
                if W.ndim != 2:
                    continue
                kind, _ = P.classify_tensor(name, list(W.shape),
                                            P.st_dtype_name(W.dtype),
                                            P.DEFAULT_QUANT_MODULES)
                if kind != "quantized":
                    continue
                W = W.float()
                dec, _ = _decide(W)
                b = quantize_ommx_weight(W, GS, PCT, decided=dec, reference=True, **QKW)
                save_file({**dec, "W": b["W_ref"] + perturb},
                          os.path.join(calib, name[: -len(".weight")] + ".safetensors"))
                n += 1
    return n


@pytest.fixture()
def synthetic(tmp_path):
    src = str(tmp_path / "src")
    P.make_synthetic_checkpoint(src)
    return src


def test_pack_from_calibration_records_it_in_the_index(synthetic, tmp_path):
    calib = str(tmp_path / "calib")
    assert _export(synthetic, calib) > 0
    out = str(tmp_path / "bundle")
    rep = P.pack_checkpoint(synthetic, out, _recipe(), verbose=False, calibrated=calib)
    assert rep["calibration"]["modules_used"] > 0
    assert rep["calibration"]["unused_count"] == 0
    import json
    idx = json.load(open(os.path.join(out, "ommx_w_index.json")))
    # A calibrated bundle and an RTN bundle are byte-compatible, so the index MUST say
    # which one this is -- otherwise the difference between a model that works and one
    # that does not is undiscoverable after packing.
    assert idx["calibration"] is not None
    P.validate_bundle(out)


def test_an_rtn_pack_records_no_calibration(synthetic, tmp_path):
    out = str(tmp_path / "bundle")
    P.pack_checkpoint(synthetic, out, _recipe(), verbose=False)
    import json
    assert json.load(open(os.path.join(out, "ommx_w_index.json")))["calibration"] is None


def test_a_weight_that_disagrees_with_its_decisions_is_refused(synthetic, tmp_path):
    """The gate that makes the path trustworthy: the packed planes must reconstruct the
    calibrated weight to float32 round-off, or the bundle is not the model the solver
    measured. The 0.01 perturbation below is orders of magnitude above that tolerance."""
    calib = str(tmp_path / "calib")
    _export(synthetic, calib, perturb=0.01)
    with pytest.raises(P.PackError) as e:
        P.pack_checkpoint(synthetic, str(tmp_path / "b"), _recipe(), verbose=False,
                          calibrated=calib)
    assert "do NOT reconstruct" in str(e.value)
    assert "round-off" in str(e.value), "the message must say what the tolerance means"


def test_float32_roundoff_between_the_two_evaluation_orders_is_tolerated(tmp_path):
    """The solver computes code*scale + zp in one order, the dequantizer in another, so
    identical decisions still differ by an ulp. A gate at exactly 0 would refuse every
    real calibration -- measured 1.5e-8 on ~0.05-magnitude weights."""
    torch.manual_seed(0)
    W = torch.randn(64, 256) * 0.05
    dec, _ = _decide(W)
    b = quantize_ommx_weight(W, GS, PCT, decided=dec, reference=True, **QKW)
    src = P.CalibrationSource.__new__(P.CalibrationSource)   # verify() needs no state
    planes = {k: b[k] for k in ("code", "scale_exp", "zp", "oindex", "ocode",
                                "map_scale", "map_center")}
    Wc = b["W_ref"] + 1e-8          # an ulp-sized disagreement must pass
    src.verify("t", planes, Wc, _recipe())


def test_a_partial_calibration_directory_is_refused(synthetic, tmp_path):
    """Half-calibrated is neither recipe, and the mixture is invisible once packed."""
    calib = str(tmp_path / "calib")
    _export(synthetic, calib)
    for f in sorted(os.listdir(calib))[2:]:
        os.remove(os.path.join(calib, f))
    with pytest.raises(P.PackError) as e:
        P.pack_checkpoint(synthetic, str(tmp_path / "b"), _recipe(), verbose=False,
                          calibrated=calib)
    assert "does not exist" in str(e.value)


def test_an_empty_or_missing_calibration_dir_fails_before_writing(synthetic, tmp_path):
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    out = str(tmp_path / "b")
    with pytest.raises(P.PackError):
        P.pack_checkpoint(synthetic, out, _recipe(), verbose=False, calibrated=empty)
    assert not os.path.exists(os.path.join(out, "ommx_w_index.json")), \
        "the pack must fail before it writes anything"
    with pytest.raises(P.PackError):
        P.pack_checkpoint(synthetic, out, _recipe(), verbose=False,
                          calibrated=str(tmp_path / "nope"))
