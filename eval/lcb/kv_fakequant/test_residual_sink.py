#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove the cache enforces each method's fp16 region. CPU only, no model, no GPU.

This test exists because its absence is what let the bug live: the previous cache stored
``residual_length`` and printed ``res=128`` while keeping only ``T mod group_size`` tokens in fp16
-- an average of ~63.5 and exactly 0 whenever T was a multiple of 128 -- and nothing checked.
A sentinel quantizer makes the quantized region unmistakable, so the fp16 region can be asserted
positionally instead of inferred from a description string.

    python eval/lcb/kv_fakequant/test_residual_sink.py
"""
import sys

import torch

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kv_fakequant import FakeQuantKVCache, build_quantizer, PUBLISHED_RECIPE  # noqa: E402
from kv_fakequant.quantizers import KVQuantizer  # noqa: E402

B, H, D = 1, 2, 128
SENTINEL = -12345.0


class SentinelQuantizer(KVQuantizer):
    """Stamps every quantized element, so 'was this token quantized?' is directly observable."""
    name = "sentinel"

    def quant_k(self, k):
        return torch.full_like(k, SENTINEL)

    def quant_v(self, v):
        return torch.full_like(v, SENTINEL)


def _cache(**kw):
    q = SentinelQuantizer(group_size=kw.pop("group_size", 128), **kw)
    try:
        return FakeQuantKVCache(q)
    except TypeError:                       # older/newer transformers want an explicit config
        return FakeQuantKVCache(q, config=None)


def fp16_region(cache, T, step=1):
    """Feed T tokens `step` at a time; return the mask of positions still in full precision."""
    cache.reset()
    read_k = None
    for t in range(0, T, step):
        n = min(step, T - t)
        pos = torch.arange(t, t + n, dtype=torch.float32).view(1, 1, n, 1)
        k = pos.expand(B, H, n, D).contiguous()
        read_k, _ = cache.update(k.clone(), k.clone(), 0)
    return (read_k[0, 0, :, 0] != SENTINEL)


def check(name, cond, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if cond else "FAIL", "  " + detail if detail else ""))
    return bool(cond)


def main():
    ok = True
    print("residual window is a real fixed window, not `T mod group_size`")
    for T in (300, 384, 512, 640, 777):     # 384/512/640 are multiples of 128: the old bug's zeros
        c = _cache(residual_length=128, sink=0, group_size=128)
        m = fp16_region(c, T)
        n_fp16 = int(m.sum())
        ok &= check("T=%-4d fp16 tail >= 128" % T, n_fp16 >= 128, "got %d" % n_fp16)
        ok &= check("T=%-4d fp16 tail is contiguous at the END" % T,
                    bool(m[-n_fp16:].all()) and not bool(m[:T - n_fp16].any()))

    print("\nfront sink is kept in full precision")
    for T in (300, 512):
        c = _cache(residual_length=0, sink=32, group_size=128)
        m = fp16_region(c, T)
        ok &= check("T=%-4d first 32 tokens fp16" % T, bool(m[:32].all()))
        ok &= check("T=%-4d token 32 is quantized" % T, not bool(m[32]))

    print("\nsink and residual compose")
    # Commits happen in whole group blocks, so the residual is the MINIMUM, not the exact size:
    # commit = sink + floor((T - res - sink)/g)*g, and everything past it stays fp16.
    for T, S, R, G in ((512, 32, 128, 128), (777, 32, 128, 128), (300, 8, 64, 64)):
        commit = S + max(0, (T - R - S)) // G * G
        if T - R - S < G:
            commit = S
        m = fp16_region(_cache(residual_length=R, sink=S, group_size=G), T)
        ok &= check("T=%-4d S=%-2d R=%-3d front sink fp16" % (T, S, R), bool(m[:S].all()))
        ok &= check("T=%-4d S=%-2d R=%-3d quantized region is exactly [%d,%d)"
                    % (T, S, R, S, commit), not bool(m[S:commit].any()))
        ok &= check("T=%-4d S=%-2d R=%-3d fp16 tail is exactly [%d,%d)  (>= R)"
                    % (T, S, R, commit, T),
                    bool(m[commit:].all()) and (T - commit) >= R, "tail=%d" % (T - commit))

    print("\nincremental decode agrees with a single prefill of the same length")
    a = fp16_region(_cache(residual_length=128, sink=32, group_size=128), 512, step=1)
    b = fp16_region(_cache(residual_length=128, sink=32, group_size=128), 512, step=512)
    ok &= check("step=1 vs step=512 give the same fp16 region", bool((a == b).all()))

    print("\ndescription reports the region the cache actually enforced")
    c = _cache(residual_length=128, sink=32, group_size=128)
    d = c.describe()
    ok &= check("describe() names sink and residual", "sink=32" in d and "res=128" in d, d)

    print("\npublished recipes are applied unless explicitly overridden")
    q = build_quantizer("kivi", k_bits=2, v_bits=2, group_size=128)
    ok &= check("kivi gets residual_length=128", q.residual_length == 128)
    q = build_quantizer("kitty", k_bits=2, v_bits=2, group_size=128)
    ok &= check("kitty gets sink=32", q.sink == 32)
    q = build_quantizer("kivi", k_bits=2, v_bits=2, group_size=128, residual_length=0)
    ok &= check("an explicit override still wins", q.residual_length == 0)
    ok &= check("every method has a declared recipe",
                set(PUBLISHED_RECIPE) >= {"kivi", "kitty", "turboquant"})

    print("\n%s" % ("ALL CHECKS PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
