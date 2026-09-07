# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The KIVI arm's GQA handling against KIVI's own.

Upstream KIVI (jy-yuan/KIVI, 2025-01-18) supports the Llama-3 family through its CUDA kernel,
which groups query heads onto KV heads itself (``nh / nh_kv``) on the UNEXPANDED packed cache;
its Mistral model instead expands the cache head-wise (``repeat_kv_quant``) before the same
kernel, and its Triton kernel (``qbvm``) is a single-row GEMV. The Llama arm in
``baseline/kivi/models/llama_kivi_eval.py`` exposes those as ``KIVI_GQA_MODE=official``,
``expand``, ``triton``, plus this repository's M-tiled Triton variant ``contig``. This gate
pins all of them to KIVI's own dequantize-and-matmul result on the same packed inputs, and
the official path to the expand path bit-exactly, so "KIVI's official kernel, GQA handled as
upstream does" is a measured statement.

REQUIRES A GPU and the vendored ``kivi`` package (baseline/ on sys.path); the CUDA modes need
``kivi_gemv`` built for the GPU in use.
"""
from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="KIVI's kernels are GPU-only")

_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "baseline")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
kivi_pack = pytest.importorskip("kivi.quant.new_pack")
kivi_mm = pytest.importorskip("kivi.quant.matmul")
kv = pytest.importorskip("kivi.models.llama_kivi_eval")

B, KH, N_REP, D, T, GS, BITS = 1, 8, 4, 128, 512, 32, 2
H = KH * N_REP


def _key_planes(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    K = torch.randn(B, KH, T, D, generator=g, device="cuda", dtype=torch.float16)
    # keys are packed TRANSPOSED: [B, KH, D, T] -> code [B, KH, D, T//feat], scale/mn [B, KH, D, T//GS]
    return kivi_pack.triton_quantize_and_pack_along_last_dim(K.transpose(2, 3).contiguous(), GS, BITS)


def _value_planes(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    V = torch.randn(B, KH, T, D, generator=g, device="cuda", dtype=torch.float16)
    return kivi_pack.triton_quantize_and_pack_along_last_dim(V.contiguous(), GS, BITS)


def _dequant_reference(fA, planes, transposed):
    """KIVI's own unpack (the function its dequantize path calls) + a dense matmul: the
    number every mode must reproduce. Keys are packed transposed ([B,KH,D,T]), values not."""
    code, sc, mn = planes
    dense = kivi_pack.unpack_and_dequant_triton_packed(code, sc, mn, GS, BITS)   # [B,KH,D,T] | [B,KH,T,D]
    dense = dense[:, :, None].expand(B, KH, N_REP, *dense.shape[2:]).reshape(B, H, *dense.shape[2:])
    return torch.matmul(fA, dense.to(fA.dtype))


def _sm():
    return torch.cuda.get_device_capability()[0]


def _run(mode, fA, planes):
    code, sc, mn = planes
    os.environ["KIVI_GQA_MODE"] = mode
    try:
        return kv._fused_qb_matmul(GS, fA, code, sc, mn, BITS, N_REP, mode=mode)
    finally:
        os.environ.pop("KIVI_GQA_MODE", None)


def _cuda_ext():
    return getattr(kivi_mm, "kivi_gemv", None) is not None


def _inputs(side, M, seed):
    planes = _key_planes(seed) if side == "keys" else _value_planes(seed + 100)
    fA = (torch.randn(B, H, M, D, device="cuda", dtype=torch.float16) if side == "keys"
          else torch.softmax(torch.randn(B, H, M, T, device="cuda", dtype=torch.float32), -1).to(torch.float16))
    return fA, planes


TOL = dict(rtol=1e-2, atol=5e-2)   # fp16 outputs from different accumulation orders: <= 2 ulps at |x| ~ 32


@pytest.mark.parametrize("side", ["keys", "values"])
def test_official_native_gqa_matches_kivis_dequantized_matmul(side):
    if not _cuda_ext():
        pytest.skip("kivi_gemv not built for this GPU")
    fA, planes = _inputs(side, 1, 1)
    ref = _dequant_reference(fA, planes, transposed=(side == "keys"))
    got = _run("official", fA, planes)
    assert torch.allclose(got.float(), ref.float(), **TOL), f"official {side}: {(got-ref).abs().max()}"


@pytest.mark.parametrize("side", ["keys", "values"])
def test_official_equals_upstreams_expand_recipe(side):
    """Same CUDA kernel fed nh_kv=KH (native) and nh_kv=H (expanded): bit-exact."""
    if not _cuda_ext():
        pytest.skip("kivi_gemv not built for this GPU")
    fA, planes = _inputs(side, 1, 2)
    a = _run("official", fA, planes); b = _run("expand", fA, planes)
    assert torch.equal(a, b), f"{side}: native vs expanded differ (max {(a-b).abs().max()})"


@pytest.mark.parametrize("side", ["keys", "values"])
def test_upstream_triton_gemv_matches_the_reference(side):
    """Upstream's Triton GEMV asserts group_size % 64 == 0, so it is packed at 64 here (the
    published KIVI recipe packs at 32, which only the CUDA kernel accepts)."""
    GS64 = 64
    g = torch.Generator(device="cuda").manual_seed(3)
    X = torch.randn(B, KH, T, D, generator=g, device="cuda", dtype=torch.float16)
    if side == "keys":
        planes = kivi_pack.triton_quantize_and_pack_along_last_dim(X.transpose(2, 3).contiguous(), GS64, BITS)
        fA = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16)
    else:
        planes = kivi_pack.triton_quantize_and_pack_along_last_dim(X.contiguous(), GS64, BITS)
        fA = torch.softmax(torch.randn(B, H, 1, T, device="cuda", dtype=torch.float32), -1).to(torch.float16)
    code, sc, mn = planes
    dense = kivi_pack.unpack_and_dequant_triton_packed(code, sc, mn, GS64, BITS)
    dense = dense[:, :, None].expand(B, KH, N_REP, *dense.shape[2:]).reshape(B, H, *dense.shape[2:])
    ref = torch.matmul(fA, dense.to(fA.dtype))
    os.environ["KIVI_GQA_MODE"] = "triton"
    try:
        got = kv._fused_qb_matmul(GS64, fA, code, sc, mn, BITS, N_REP, mode="triton")
    finally:
        os.environ.pop("KIVI_GQA_MODE", None)
    assert torch.allclose(got.float(), ref.float(), **TOL), f"triton {side}: {(got-ref).abs().max()}"


@pytest.mark.parametrize("M", [1, 32])
@pytest.mark.parametrize("side", ["keys", "values"])
def test_this_repos_tiled_variant_matches_the_reference(M, side):
    fA, planes = _inputs(side, M, 4)
    ref = _dequant_reference(fA, planes, transposed=(side == "keys"))
    got = _run("contig", fA, planes)
    assert torch.allclose(got.float(), ref.float(), **TOL), f"contig {side} M={M}: {(got-ref).abs().max()}"
