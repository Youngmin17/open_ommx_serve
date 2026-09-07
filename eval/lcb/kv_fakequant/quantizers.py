# SPDX-License-Identifier: Apache-2.0
"""KV-cache fake-quantization methods for the LiveCodeBench comparison.

Each method is a per-block quantizer over a committed KV block of shape [B, Hkv, L, D]
(L = tokens, D = head_dim) returning a quantize->dequantized tensor of the same shape and dtype.
All methods share the primitives below, so grouping convention, scale dtype and rounding are
identical and the comparison is apples-to-apples.

Block selection -- which tokens are quantized at all -- belongs to ``cache.py``. That split matters:
a method's fp16 region (KIVI's residual window, Kitty's front sink) is part of its published recipe,
and an earlier revision advertised a residual that the cache never enforced. Two rules keep that
from recurring:

  * ``PUBLISHED_RECIPE`` below carries each method's high-precision region, and ``build_quantizer``
    applies it unless the caller overrides it, so running an arm without its recipe now takes a
    deliberate act rather than an omission.
  * ``describe()`` here reports ONLY what this file decides. The cache appends the sink/residual it
    actually enforced, so a recorded description cannot claim a region that was not applied.
"""

import torch

# Each method's high-precision region, as its paper specifies it. `sink` = leading tokens kept
# fp16; `residual_length` = trailing tokens kept fp16.
#   KIVI        fixed 128-token fp16 residual on the recent tokens
#   Kitty       32-token attention sink, on top of its per-channel precision boost
#   TurboQuant  rotation-only; no high-precision region is specified for it. Left at 0 and flagged
#               here rather than guessed -- if the paper does specify one, set it explicitly.
PUBLISHED_RECIPE = {
    "kivi":       {"sink": 0,  "residual_length": 128},
    "kitty":      {"sink": 32, "residual_length": 0},
    "turboquant": {"sink": 0,  "residual_length": 0},
}


# --------------------------------------------------------------------------- #
# Shared primitives
# --------------------------------------------------------------------------- #
def uniform_quant_dequant(x, bits, axis, symmetric=False):
    """Uniform min-max (asymmetric) or abs-max (symmetric) quant along ``axis``."""
    qmax = (1 << bits) - 1
    xf = x.float()
    if symmetric:
        amax = xf.abs().amax(dim=axis, keepdim=True).clamp_min(1e-8)
        scale = amax / (qmax // 2 if qmax > 1 else 1)
        q = torch.round(xf / scale).clamp(-(qmax // 2 + 1), qmax // 2)
        return (q * scale).to(x.dtype)
    xmin = xf.amin(dim=axis, keepdim=True)
    xmax = xf.amax(dim=axis, keepdim=True)
    scale = (xmax - xmin).clamp_min(1e-8) / qmax
    q = torch.round((xf - xmin) / scale).clamp(0, qmax)
    return (q * scale + xmin).to(x.dtype)


def per_channel_token_group(x, bits, group_size, symmetric=False):
    """KIVI-style KEY quant: per channel (D), grouped along the token axis."""
    B, H, L, D = x.shape
    if L == 0:
        return x
    ng = (L + group_size - 1) // group_size
    pad = ng * group_size - L
    xp = torch.cat([x, x[:, :, -1:, :].expand(B, H, pad, D)], dim=2) if pad else x
    xg = xp.view(B, H, ng, group_size, D)
    qg = uniform_quant_dequant(xg, bits, axis=3, symmetric=symmetric)
    return qg.view(B, H, ng * group_size, D)[:, :, :L, :].to(x.dtype)


def per_token_channel_group(x, bits, symmetric=False):
    """KIVI-style VALUE quant: per token, grouped along the channel axis (D)."""
    if x.shape[2] == 0:
        return x
    return uniform_quant_dequant(x, bits, axis=3, symmetric=symmetric)


def hadamard_matrix(n, device, dtype=torch.float32):
    """Normalized Walsh-Hadamard matrix (n must be a power of two)."""
    assert (n & (n - 1)) == 0, "Hadamard dim %d is not a power of two" % n
    H = torch.ones(1, 1, device=device, dtype=dtype)
    i = 1
    while i < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        i *= 2
    return H / (n ** 0.5)


def apply_hadamard(x, Hm):
    return torch.matmul(x.float(), Hm).to(x.dtype)


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
class KVQuantizer:
    """Base: subclasses implement quant_k / quant_v on a [B,Hkv,L,D] block."""

    name = "base"

    def __init__(self, k_bits=2, v_bits=2, group_size=128, residual_length=0, sink=0,
                 outlier_bits=8, outlier_frac=0.01, **kw):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        # Carried for the cache to enforce; deliberately absent from describe() so this object
        # can never advertise a region it does not control.
        self.residual_length = residual_length
        self.sink = sink
        self.outlier_bits = outlier_bits
        self.outlier_frac = outlier_frac
        self._cfg = kw

    def quant_k(self, k):
        raise NotImplementedError

    def quant_v(self, v):
        raise NotImplementedError

    def describe(self):
        return "%s(k=%db,v=%db,g=%d)" % (self.name, self.k_bits, self.v_bits, self.group_size)


class KIVIQuantizer(KVQuantizer):
    """KIVI: per-channel KEY (grouped along tokens), per-token VALUE, fp16 residual window."""
    name = "kivi"

    def quant_k(self, k):
        return per_channel_token_group(k, self.k_bits, self.group_size, symmetric=False)

    def quant_v(self, v):
        return per_token_channel_group(v, self.v_bits, symmetric=False)


class KittyQuantizer(KVQuantizer):
    """Kitty: INT2 base + per-channel precision boost, on top of a front attention sink.

    The highest-variance channels (variance across tokens) are kept fp16; the rest is INT2. The
    sink is enforced by the cache, not here, so both parts of the recipe are visible in one place.
    """
    name = "kitty"

    def _boost_mask(self, x):
        imp = x.float().var(dim=2).mean(dim=(0, 1))          # [D]
        k = max(1, int(round(self.outlier_frac * imp.numel())))
        return imp >= torch.topk(imp, k).values.min()

    def quant_k(self, k):
        m = self._boost_mask(k)
        q = per_channel_token_group(k, self.k_bits, self.group_size, symmetric=False)
        q[..., m] = k[..., m]
        return q

    def quant_v(self, v):
        m = self._boost_mask(v)
        q = per_token_channel_group(v, self.v_bits, symmetric=False)
        q[..., m] = v[..., m]
        return q

    def describe(self):
        return "%s(k=%db,v=%db,g=%d,boost=%.3f)" % (
            self.name, self.k_bits, self.v_bits, self.group_size, self.outlier_frac)


class TurboQuantQuantizer(KVQuantizer):
    """TurboQuant: Walsh-Hadamard rotation along head_dim, uniform quant in the rotated domain,
    then the inverse rotation (the WHT is orthonormal, so H^T = H^-1)."""
    name = "turboquant"

    def quant_k(self, k):
        Hm = hadamard_matrix(k.shape[-1], k.device)
        return apply_hadamard(
            per_channel_token_group(apply_hadamard(k, Hm), self.k_bits, self.group_size), Hm.t())

    def quant_v(self, v):
        Hm = hadamard_matrix(v.shape[-1], v.device)
        return apply_hadamard(
            per_token_channel_group(apply_hadamard(v, Hm), self.v_bits), Hm.t())


_METHODS = {"kivi": KIVIQuantizer, "kitty": KittyQuantizer, "turboquant": TurboQuantQuantizer}


def available_methods():
    return sorted(_METHODS)


def build_quantizer(method, apply_published_recipe=True, **kw):
    """Build a quantizer, applying the method's published high-precision region by default.

    Pass ``sink=`` / ``residual_length=`` explicitly to override, or
    ``apply_published_recipe=False`` to run a method stripped of its fp16 region -- which is a
    real experiment, but one you now have to ask for.
    """
    if method in (None, "none", "bf16"):
        return None
    if method not in _METHODS:
        raise ValueError("unknown KV-quant method %r (have: %s)"
                         % (method, ", ".join(available_methods())))
    if apply_published_recipe:
        for key, val in PUBLISHED_RECIPE.get(method, {}).items():
            kw.setdefault(key, val)
    return _METHODS[method](**kw)
