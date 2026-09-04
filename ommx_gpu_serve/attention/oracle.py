# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The independent attention oracle the decode kernel is gated against.

Shared by ``tests/test_decode_kernel_parity.py`` (the gate) and
``bench/bench_decode_kernel.py`` (which refuses to time an arm that fails it), so the two
cannot drift apart. It shares nothing with the kernel: the caller dequantizes the packed
planes with ``pack.dequant_kv_canonical`` and hands the oracle plain fp32 K/V rows (packed
prefix first, then the bf16 sink/recent tail -- the rows the kernel reads, in any order).
``attention/reference_op.py`` is NOT an oracle; it wraps the kernel.
"""
from __future__ import annotations

import torch


def synth_kv(num_tokens: int, kv_heads: int, head_dim: int, seed: int):
    """bf16 ``(K, V)``, each ``[num_tokens, kv_heads, head_dim]``.

    K carries genuine outliers -- 12% of its lanes spiked to ~9x -- so a broken outlier
    splice cannot pass; a flat Gaussian K would let it. The draw order (K, V, spike,
    mask) is fixed, so one seed gives the same tensors to every caller.
    """
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(num_tokens, kv_heads, head_dim, generator=g)
    V = torch.randn(num_tokens, kv_heads, head_dim, generator=g)
    spike = torch.randn(num_tokens, kv_heads, head_dim, generator=g).abs() * 9.0
    K = torch.where(torch.rand(num_tokens, kv_heads, head_dim, generator=g) < 0.12,
                    spike * torch.sign(K), K)
    return K.to(torch.bfloat16), V.to(torch.bfloat16)


def reference_output(q, K_full, V_full, q_heads: int, kv_heads: int, sm_scale: float):
    """Textbook fp32 GQA attention of ONE query row.

    ``q`` is ``[q_heads, head_dim]``; ``K_full``/``V_full`` are ``[T, kv_heads, head_dim]``
    over every token the kernel sees. Returns fp32 ``[q_heads, head_dim]``.
    """
    per = q_heads // kv_heads
    out = torch.empty(q_heads, K_full.shape[-1], dtype=torch.float32)
    for h in range(q_heads):
        kh = h // per
        k = K_full[:, kh, :].float()
        v = V_full[:, kh, :].float()
        s = (k @ q[h].float()) * sm_scale
        s = s - s.max()
        w = torch.softmax(s, dim=0)
        out[h] = (w.unsqueeze(-1) * v).sum(0)
    return out


def compare(got, want):
    """Kernel-vs-oracle agreement as (cos, max_abs, rel_l2).

    cosine alone passes a wrongly SCALED output (2x the reference is cos 1.0), so a gate
    must also bound the relative L2 error. bf16 accumulation over 64K tokens sits well
    under 1e-2 on both.
    """
    g = got.float().flatten(); w = want.float().flatten()
    cos = torch.nn.functional.cosine_similarity(g[None], w[None]).item()
    max_abs = (g - w).abs().max().item()
    rel_l2 = ((g - w).norm() / w.norm().clamp_min(1e-12)).item()
    return cos, max_abs, rel_l2


COS_MIN = 0.999
REL_L2_MAX = 0.02
