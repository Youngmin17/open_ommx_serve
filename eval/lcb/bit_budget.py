#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Average KV bits per element for the exact arm configs the LCB comparison runs.

This exists because the comparison is NOT iso-bit and it is easy to misread the accuracy table
without that fact in view: TurboQuant here runs 3-bit K/V, roughly 37% more payload than OMMX's
INT2-plus-FP4-outlier recipe. A published per-method table is not a substitute -- it describes
each method's DEFAULT config (TurboQuant at 2 bits, Kitty at outlier_frac=0.01), while this
comparison overrides both.

What is counted, and what is not:

  counted      the quantized payload: base bits, plus the promotion cost of whichever elements a
               recipe keeps at higher precision.
  NOT counted  scale/zero-point (asymmetric min/max costs 2x16 bits per group -- +0.25 bit/elem
               at group 128, +0.5 at group 64), and OMMX's outlier-MEMBERSHIP encoding, i.e. which
               12 of each 64 elements are the FP4 outliers. A 64-bit bitmap would add a full
               1.0 bit/elem.

So these numbers order the payload, not the wire format. A metadata-inclusive budget has not been
established for the OMMX cache path, and until it is, "method A uses fewer bits than method B"
is not a claim this file supports.

Usage: python eval/lcb/bit_budget.py [--head-dim 128]
"""
import argparse


def kitty_bits(outlier_frac, head_dim, base=2, high=16):
    """Kitty keeps the top-variance channels at bf16: k = max(1, round(frac * D)) of D."""
    k = max(1, int(round(outlier_frac * head_dim)))
    return k, base + (k / head_dim) * (high - base)


def ommx_k_bits(outliers_per_group, group=64, base=2, outlier=4):
    """OMMX promotes `outliers_per_group` of each group from INT2 to FP4 (K only; V stays INT2)."""
    return base + (outliers_per_group / group) * (outlier - base)


def main(a):
    D = a.head_dim
    k_kitty, b_kitty = kitty_bits(0.02, D)
    b_ommx_k = ommx_k_bits(12)
    rows = [
        ("bf16 (none)", "no quantization", 16.0, 16.0, None),
        ("KIVI", "plain INT2, group 128", 2.0, 2.0, 128),
        ("Kitty", "INT2 + %d/%d bf16 channels, group 128" % (k_kitty, D), b_kitty, b_kitty, 128),
        ("TurboQuant", "Walsh-Hadamard rotate + INT3, group 128", 3.0, 3.0, 128),
        ("OMMX", "INT2 + 12/64 FP4 outliers on K only, group 64", b_ommx_k, 2.0, 64),
    ]
    print("KV payload bits per element -- head_dim=%d, LCB comparison configs\n" % D)
    print("%-12s %-46s %8s %8s %9s" % ("arm", "scheme", "K", "V", "avg"))
    print("-" * 88)
    for name, scheme, kb, vb, _ in rows:
        print("%-12s %-46s %8.3f %8.3f %9.3f" % (name, scheme, kb, vb, (kb + vb) / 2))
    print("\nexcluded from every number above:")
    for g in (64, 128):
        print("  scale+zero-point, asymmetric, group %3d ....... +%.3f bit/elem" % (g, 32.0 / g))
    print("  OMMX outlier membership (which 12 of 64) ...... +1.000 bit/elem if a 64-bit bitmap")
    print("\nKitty at its DEFAULT outlier_frac=0.01 would be %.3f (k=%d), which is the value a"
          % (kitty_bits(0.01, D)[1], kitty_bits(0.01, D)[0]))
    print("per-method reference table reports -- this comparison runs 0.02.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--head-dim", type=int, default=128)
    main(p.parse_args())
