# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Algorithm parity: serving packers vs the vendored fakequant reference, RATIO EXCLUDED.

The accuracy numbers in this repo come from ``ommx_fakequant/`` (pure-torch fake
quantization); the served numbers come from ``attention/pack.py`` (KV) and
``linear/quantize.py`` (weights). These gates prove that, given the SAME tensor and the
SAME outlier POSITIONS, both sides compute the same quantized numbers. The outlier
SELECTION rule (fakequant: percent thresholds / top-k; packer: ``outliers_per_vector`` +
``outlier_select``) is out of scope: every test fixes the positions first and asserts the
packer stored exactly that set before any value is compared.

VERDICT (CPU, fp32 on both sides):
  * INT2 affine base — K vector = (channel, ``group_tokens`` tokens), V vector = (token,
    ``group_channels`` channels); fakequant groups keys along the TOKEN axis and values
    along the CHANNEL axis, so its ``group_size`` is ``group_tokens`` for K and
    ``group_channels`` for V: BIT-EXACT.
  * pow2 scale ``2^round(log2(range/3))`` and the stored int8 E8M0 exponent: BIT-EXACT,
    exact log2 ties round half-to-even on both sides.
  * dedicated FP4 outlier map (``o_center``, ``map_scale = 12/range``) and the base
    lanes with the outliers excluded from the range: BIT-EXACT.
  * FP4 E2M1 rounding rule: ``codec.encode_fp4_e2m1f`` rounds in log magnitude exactly as
    ``fp16_to_fp4_e2m1`` does (it used to be linear-nearest; ~3% of outliers differed).
  * weights: ``quantize_ommx_weight`` vs ``weight_quantizer`` at the microbench
    criterion (cos >= 0.9999, max_abs < 1e-4); base lanes bit-exact, outlier lanes
    within fp32 rounding (both are linear-nearest FP4 in index space; the packer
    evaluates the map in fp64, fakequant in fp32).

WHY fp32 on both sides: fakequant computes in the INPUT dtype, so a bf16 input makes it
do bf16 arithmetic for ``(max-min)/3``, ``(x-zp)/scale`` and the outlier map, while the
packer upcasts to fp32 and rounds only the STORED zp / map planes to ``scale_dtype``.
That is an intermediate-precision difference, not an algorithm difference (bf16 inputs:
~1.7% of lanes differ, up to one scale flip), so it is isolated by giving both sides
fp32 and ``scale_dtype=torch.float32``.

NOT COMPARED: the base-SHARED outlier recipe (``kv_outlier_map=False``, dequant
``fp4((w-zp)/scale)*scale + zp``) has no fakequant counterpart — pack.py names its own
``dequant_kv_canonical`` as that recipe's oracle; fakequant's prefill whole-group FP4
(``_fp4_group_quant_core``) has no packer counterpart.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

from ommx_gpu_serve.attention import codec
from ommx_gpu_serve.attention.pack import (
    dequant_kv_canonical,
    ommx_pack_kv_canonical_block,
)
from ommx_gpu_serve.linear.quantize import outlier_positions, quantize_ommx_weight

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FAKEQUANT_DIR = os.path.join(_REPO_ROOT, "ommx_fakequant")


def _load_fakequant(name: str):
    """Import ``ommx_fakequant/<name>.py`` by path (the directory is not a package).

    Directory absent -> SKIP (installed wheel: the reference cannot be present).
    Directory present but the file gone -> FAIL (a source checkout lost the reference).
    ``conftest.py`` scrubs ``OMMX_*`` first, which matters here: the reference reads
    ``OMMX_KV_SYMMETRIC`` / ``OMMX_W_SYMMETRIC`` / ``OMMX_N_LEVELS`` from the environment.
    """
    if not os.path.isdir(_FAKEQUANT_DIR):
        pytest.skip(f"vendored reference {_FAKEQUANT_DIR} not present (installed package?)")
    path = os.path.join(_FAKEQUANT_DIR, f"{name}.py")
    if not os.path.isfile(path):
        pytest.fail(f"{path} is missing from a source checkout")
    modname = f"_ommx_fakequant_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def qf():
    return _load_fakequant("quant_function")


@pytest.fixture(scope="module")
def qw():
    return _load_fakequant("quant_weight")


# ── shared helpers ──────────────────────────────────────────────────────────

def _block(T: int, H: int, D: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, H, D, generator=g), torch.randn(T, H, D, generator=g)


def _pack(K, V, *, k: int, gt: int = 32, gc: int = 32, select: str = "signed",
          int8_scale: bool = True):
    """The canonical i2f4 recipe (pow2 scale, dedicated FP4 map) with fp32 stored planes."""
    return ommx_pack_kv_canonical_block(
        K, V, outliers_per_vector=k, outlier_select=select, k_format="i2f4",
        use_pow2=True, kv_outlier_map=True, kv_int8_scale=int8_scale,
        group_tokens=gt, group_channels=gc, scale_dtype=torch.float32, page_size=16)


def _fq_group_quant(qf, x_thd, *, is_key: bool, group: int, top: float = 0.0,
                    bottom: float = 0.0):
    """fakequant decode-path group quant of a [T,H,D] block -> [T,H,D].

    ``bits=2, use_pow2=True, outlier_method="fp4"`` is the INT2 + dedicated-map FP4
    recipe pack.py mirrors. fakequant takes [B,H,S,D] and, for keys, permutes to
    [B,H,D,S] before grouping along the last axis -> ``group`` tokens per channel;
    values group along D -> ``group`` channels per token. No sinks, no prefill path.
    """
    inp = x_thd.permute(1, 0, 2).unsqueeze(0).contiguous()
    out = qf._apply_group_quantization_vectorized(
        inp, group_size=group, bits=2, use_pow2=True, attention_sink_num=0,
        outlier_percent_topk=top, outlier_percent_bottomk=bottom, is_key=is_key,
        debug_mode=False, outlier_method="fp4", outliers_per_group=0, is_prefill=False)
    return out[0].permute(1, 0, 2).contiguous()


def _k_vectors(x_thd, gt: int):
    """[T,H,D] -> [H,D,G,gt]: the packer's K vector layout (channel-major, token groups)."""
    T, H, D = (int(s) for s in x_thd.shape)
    return x_thd.permute(1, 2, 0).reshape(H, D, T // gt, gt)


def _from_k_vectors(v, T: int):
    H, D = int(v.shape[0]), int(v.shape[1])
    return v.reshape(H, D, T).permute(2, 0, 1).contiguous()


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
def _fakequant_pow2_rule(raw):
    """quant_function.py: ``2 ** round(log2(clamp(scale, 1e-12)))`` on the raw range/(qmax-qmin)."""
    return torch.pow(2.0, torch.round(torch.log2(torch.clamp(raw, min=1e-12))))


def _packer_fp4_roundtrip(x):
    return codec.decode_fp4_e2m1f(codec.encode_fp4_e2m1f(x))


# ── 1. FP4 E2M1 codec ───────────────────────────────────────────────────────

def test_fp4_codec_is_a_fixed_point_on_the_e2m1_level_set(qf):
    lv = torch.tensor(_E2M1, dtype=torch.float32)
    x = torch.cat([lv, -lv])
    pk = _packer_fp4_roundtrip(x)
    fq = qf.fp16_to_fp4_e2m1(x)
    assert torch.equal(pk, x), f"packer round-trip moved a level: {pk.tolist()}"
    assert torch.equal(fq, x), f"fakequant round-trip moved a level: {fq.tolist()}"



def test_pow2_scale_and_int8_exponent_match_fakequant_rule():
    T, H, D = 128, 2, 64
    K, V = _block(T, H, D, seed=20260904)
    p8 = _pack(K, V, k=0)
    pf = _pack(K, V, k=0, int8_scale=False)
    assert p8["kv_int8_scale"] and p8["k_scale"].dtype == torch.int8 \
        and p8["v_scale"].dtype == torch.int8
    assert not pf["kv_int8_scale"] and pf["k_scale"].dtype == torch.float32
    # the E8M0 byte is a LOSSLESS encoding of the fp32 pow2 scale
    assert torch.equal(torch.exp2(p8["k_scale"].float()), pf["k_scale"])
    assert torch.equal(torch.exp2(p8["v_scale"].float()), pf["v_scale"])
    # ... and that fp32 scale IS fakequant's rule on fakequant's raw scale range/(2^2-1)
    gw = _k_vectors(K, 32)                                        # [H,D,G,32]
    raw_k = (gw.amax(-1) - gw.amin(-1)) / 3                       # [H,D,G]
    gv = V.reshape(T, H, D // 32, 32)
    raw_v = (gv.amax(-1) - gv.amin(-1)) / 3                       # [T,H,NGV]
    assert bool((raw_k > 1e-8).all()) and bool((raw_v > 1e-8).all()), \
        "degenerate group: fakequant (scale=1.0) and the packer (1e-8/3) differ there"
    want_k = _fakequant_pow2_rule(raw_k).permute(2, 0, 1)         # [G,H,D]
    want_v = _fakequant_pow2_rule(raw_v)
    assert torch.equal(pf["k_scale"], want_k), \
        f"K scale != 2^round(log2(range/3)) on {int((pf['k_scale'] != want_k).sum())} vectors"
    assert torch.equal(pf["v_scale"].reshape(T, H, D // 32), want_v), \
        f"V scale != 2^round(log2(range/3)) on {int((pf['v_scale'].reshape(T, H, -1) != want_v).sum())} vectors"
    assert torch.equal(p8["k_scale"].float(), torch.log2(want_k)), "int8 exponent != log2(scale)"
    assert torch.equal(p8["v_scale"].float().reshape(T, H, D // 32), torch.log2(want_v))


def _exact_log2_tie_maxima():
    """(n, m) pairs: a vector with min 0 and max m has raw scale m/3 whose fp32 log2 is
    EXACTLY n + 0.5, found by ulp-stepping around 3*2^(n+0.5) with the packer's/fakequant's
    own expression ``(max-min).clamp_min(1e-8)/3``."""
    hits = []
    for n in range(-12, 6):
        t = torch.tensor([3.0 * 2.0 ** (n + 0.5)], dtype=torch.float32)
        cand = (t.view(torch.int32) + torch.arange(-8192, 8193, dtype=torch.int32)).view(torch.float32)
        raw = cand.clamp_min(1e-8) / 3.0
        hit = torch.log2(raw.clamp_min(1e-12)) == (n + 0.5)
        if bool(hit.any()):
            hits.append((n, float(cand[hit][0])))
    return hits


def test_pow2_scale_ties_round_half_to_even_like_fakequant(qf):
    """Exact log2 ties (n+0.5) snap to the EVEN exponent on both sides (torch.round RNE).

    Vectors are crafted so ``(max-min)/3`` has fp32 log2 == n+0.5 exactly; the packer's
    stored exponent must equal round-half-even(n+0.5), and fakequant's own dequant of the
    same vectors must be bit-identical (it rounds the same tie the same way). A packer
    switched to floor/ceil/half-up would fail on the odd or the even n.
    """
    hits = _exact_log2_tie_maxima()[:32]
    if not any(n % 2 == 0 for n, _ in hits) or not any(n % 2 for n, _ in hits):
        pytest.skip(f"this platform's fp32 log2 produced no exact n+0.5 ties: {hits}")
    T = 32
    g = torch.Generator().manual_seed(7)
    K = torch.rand(T, 1, 32, generator=g) + 0.5
    V = torch.rand(T, 1, 32, generator=g) + 0.5
    ramp = torch.arange(32, dtype=torch.float32) / 31.0             # lane0 = 0, lane31 = 1
    for j, (_, m) in enumerate(hits):
        K[:, 0, j] = m * ramp                                       # channel j: min 0, max m
        V[j, 0, :] = m * ramp                                       # token j: min 0, max m
    p = _pack(K, V, k=0)
    exp_k = p["k_scale"][0, 0, :len(hits)].tolist()                 # [G=1,H=1,D]
    exp_v = p["v_scale"].reshape(T, 1, 1)[:len(hits), 0, 0].tolist()
    want = [n if n % 2 == 0 else n + 1 for n, _ in hits]            # round half to even
    assert exp_k == want, f"K exponents {exp_k} != RNE {want} for ties n+0.5, n={[n for n, _ in hits]}"
    assert exp_v == want, f"V exponents {exp_v} != RNE {want}"
    K_dq, V_dq = dequant_kv_canonical(p)
    assert torch.equal(K_dq, _fq_group_quant(qf, K, is_key=True, group=32)), \
        "fakequant resolves the same log2 tie differently for K"
    assert torch.equal(V_dq, _fq_group_quant(qf, V, is_key=False, group=32)), \
        "fakequant resolves the same log2 tie differently for V"


# ── 3 / 6. INT2 affine base, no outliers ────────────────────────────────────

@pytest.mark.parametrize("group", [32, 64])
def test_k_int2_affine_base_no_outliers_bit_exact(qf, group):
    """K: packer vector (channel, ``group`` tokens) == fakequant key group_size=``group``."""
    K, V = _block(128, 2, 64, seed=20260904)
    planes = _pack(K, V, k=0, gt=group)
    assert planes["outliers_per_vector"] == 0 and planes["k_oidx"] is None
    K_dq, _ = dequant_kv_canonical(planes)
    ref = _fq_group_quant(qf, K, is_key=True, group=group)
    assert torch.equal(K_dq, ref), (
        f"K INT2 base differs from fakequant on {int((K_dq != ref).sum())} of {ref.numel()} "
        f"lanes, max|d|={(K_dq - ref).abs().max().item():.3e}")


@pytest.mark.parametrize("group", [32, 64])
def test_v_int2_affine_no_outliers_bit_exact(qf, group):
    """V: packer vector (token, ``group`` channels) == fakequant value group_size=``group``."""
    K, V = _block(128, 2, 64, seed=20260905)
    planes = _pack(K, V, k=0, gc=group)
    _, V_dq = dequant_kv_canonical(planes)
    ref = _fq_group_quant(qf, V, is_key=False, group=group)
    assert torch.equal(V_dq, ref), (
        f"V INT2 differs from fakequant on {int((V_dq != ref).sum())} of {ref.numel()} "
        f"lanes, max|d|={(V_dq - ref).abs().max().item():.3e}")


# ── 4. K outliers at FIXED positions, dedicated FP4 map ─────────────────────

def _fixed_outlier_case(k: int, seed: int = 20260906):
    """A K block plus the FIXED set: top-kt and bottom-kb by VALUE per 32-token vector.

    Lanes are strictly distinct inside every vector (asserted), so fakequant's percent
    thresholds (``>= kt-th largest`` | ``<= kb-th smallest``) name exactly this set and
    so does the packer's ``outlier_select="signed"``.
    """
    T, H, D, gt = 64, 2, 64, 32
    K, V = _block(T, H, D, seed=seed)
    gw = _k_vectors(K, gt)
    assert bool((gw.sort(-1).values.diff(dim=-1) > 0).all()), "seed produced a tie"
    kt, kb = (k + 1) // 2, k // 2
    fixed = torch.zeros_like(gw, dtype=torch.bool)
    fixed.scatter_(-1, gw.topk(kt, dim=-1).indices, True)
    if kb:
        fixed.scatter_(-1, (-gw).topk(kb, dim=-1).indices, True)
    assert bool((fixed.sum(-1) == k).all())
    return K, V, gw, fixed, kt, kb


def _stored_positions(planes, k: int):
    """relidx7 sidecar -> bool [H,D,G,gt] via the public codec (no packer internals)."""
    G, H, D = (int(s) for s in planes["k_oidx"].shape[:3])
    gt = int(planes["group_tokens"])
    out = torch.zeros(G, H, D, gt, dtype=torch.bool)
    for g_ in range(G):
        for h in range(H):
            for d in range(D):
                pos = codec.unpack_relidx7(bytes(planes["k_oidx"][g_, h, d].tolist()), k)
                out[g_, h, d, pos] = True
    return out.permute(1, 2, 0, 3)


@pytest.mark.parametrize("k", [2, 6, 10])
def test_k_outliers_fixed_positions_dedicated_fp4_map(qf, k):
    """Same positions -> same base, same map params, same dequant arithmetic; the ONLY
    divergence is the FP4 rounding rule, and it is exactly where the level oracles say.
    """
    K, V, gw, fixed, kt, kb = _fixed_outlier_case(k)
    T, gt = int(K.shape[0]), 32
    planes = _pack(K, V, k=k, select="signed")
    assert planes["kv_outlier_map"] and planes["k_fp4_mapscale"] is not None
    # (a) the packer stored EXACTLY the fixed set
    assert torch.equal(_stored_positions(planes, k), fixed), "packer positions != fixed set"
    K_dq, _ = dequant_kv_canonical(planes)
    ref = _fq_group_quant(qf, K, is_key=True, group=gt, top=kt / gt, bottom=kb / gt)
    fixed_thd = _from_k_vectors(fixed, T)
    # (b) base lanes, range taken EXCLUDING the fixed outliers: bit-exact
    nb = int((~fixed_thd).sum())
    assert torch.equal(K_dq[~fixed_thd], ref[~fixed_thd]), (
        f"base lanes differ on {int((K_dq[~fixed_thd] != ref[~fixed_thd]).sum())} of {nb}")
    # (c) outlier lanes: both sides == fakequant's fp16_to_fp4_e2m1 level / map_scale +
    #     o_center with the SAME fp32 map (o_center=(max+min)/2, map_scale=12/(max-min)
    #     over the fixed set) -> bit-identical
    big = torch.tensor(1e9)
    o_min = torch.where(fixed, gw, big).amin(-1, keepdim=True)
    o_max = torch.where(fixed, gw, -big).amax(-1, keepdim=True)
    c = (o_max + o_min) / 2.0
    s = 12.0 / (o_max - o_min).clamp_min(1e-8)
    mapped = ((gw - c) * s)[fixed]
    c_o, s_o = c.expand_as(gw)[fixed], s.expand_as(gw)[fixed]
    pk_o = _k_vectors(K_dq, gt)[fixed]
    fq_o = _k_vectors(ref, gt)[fixed]
    lg = qf.fp16_to_fp4_e2m1(mapped)
    assert torch.equal(fq_o, lg / s_o + c_o), (
        f"fakequant outlier dequant != its level / map_scale + o_center on "
        f"{int((fq_o != lg / s_o + c_o).sum())} of {fq_o.numel()} outliers")
    assert torch.equal(pk_o, fq_o), (
        f"packer outlier dequant != fakequant on {int((pk_o != fq_o).sum())} of "
        f"{pk_o.numel()} outliers -- the FP4 rounding rule drifted from fp16_to_fp4_e2m1")


def test_k_outlier_values_bit_identical_to_fakequant_dedicated_map(qf):
    k = 6
    K, V, gw, fixed, kt, kb = _fixed_outlier_case(k)
    T, gt = int(K.shape[0]), 32
    planes = _pack(K, V, k=k, select="signed")
    assert torch.equal(_stored_positions(planes, k), fixed)
    K_dq, _ = dequant_kv_canonical(planes)
    ref = _fq_group_quant(qf, K, is_key=True, group=gt, top=kt / gt, bottom=kb / gt)
    fixed_thd = _from_k_vectors(fixed, T)
    assert torch.equal(K_dq[fixed_thd], ref[fixed_thd]), (
        f"{int((K_dq[fixed_thd] != ref[fixed_thd]).sum())} of {int(fixed_thd.sum())} "
        f"outlier values differ from fakequant (both should round log-nearest FP4)")


# ── 5. weights ──────────────────────────────────────────────────────────────

def test_weight_quantizer_parity_pinned_microbench(qw):
    """csrc/linear/bench_linear_memory_bound.py::parity_assert, pinned: seed 42,
    W = randn(256,256)*0.1, group 64, 25% outliers, cos >= 0.9999 and max_abs < 1e-4.

    Beyond the microbench criterion: base lanes are bit-exact (same fp32 ops), and the
    outlier lanes agree to fp32 rounding — both sides are linear-nearest FP4 in index
    space with ``ms = 12/(imax-imin)``, ``mc = (imax+imin)/2``; the packer evaluates the
    map in fp64 and fakequant in fp32, hence the sub-ulp residual. The seed is pinned
    because a value within ~1e-7 of an FP4 boundary would flip a level under that
    precision difference (a full level step, ~1e-1) -- other seeds are not promised.
    """
    torch.manual_seed(42)
    W = torch.randn(256, 256) * 0.1
    q = quantize_ommx_weight(W, 64, 0.25)
    ref = qw.weight_quantizer(W, bits=2, group_size=64, outlier_percent=0.25,
                              use_pow2=True, act_scales=None, mode="decode")
    assert ref.dtype == torch.float32 and q["W_ref"].shape == ref.shape
    cos = F.cosine_similarity(q["W_ref"].flatten(), ref.flatten(), dim=0).item()
    md = (q["W_ref"] - ref).abs().max().item()
    assert cos >= 0.9999 and md < 1e-4, f"microbench parity: cos={cos:.6f} max_abs={md:.3e}"
    N, G, npv = q["N"], q["G"], q["npv"]
    assert npv == 16
    # positions from the FORMAT (relidx7 sidecar), not from a re-run of the selection
    pos = outlier_positions(q["oindex"], N, G, npv, 64, "relidx7")
    omask = torch.zeros(N, G, 64, dtype=torch.bool).scatter_(-1, pos, True).view(N, -1)
    assert int(omask.sum()) == N * G * npv
    assert torch.equal(q["W_ref"][~omask], ref[~omask]), (
        f"base lanes differ on {int((q['W_ref'][~omask] != ref[~omask]).sum())} lanes")
    tol = 4 * torch.finfo(torch.float32).eps * ref.abs().max().item()   # 4 fp32 ulp of max|W|
    od = (q["W_ref"][omask] - ref[omask]).abs().max().item()
    assert od <= tol, f"outlier lanes: max|d|={od:.3e} > {tol:.3e} (fp64-vs-fp32 map residual)"
