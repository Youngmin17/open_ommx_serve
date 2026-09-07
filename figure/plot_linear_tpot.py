#!/usr/bin/env python3
# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Decode-linear latency per token, OMMX against its weight-quantization peers.

WHAT THIS IS. The KV figures answer "what does a decode STEP cost as context grows". This one
answers the other half: what one decode step's LINEAR layers cost as the batch grows, for
OMMX i2f4 weights against bf16, CUTLASS mixed-input INT4 and LLM.int8. Input is
`bench_linear_memory_bound.py`'s result.json (`figure/data_<gpu>/linear_mem_bound.json`).

WHAT IT IS NOT, and the caption says so on the canvas:
  * NOT iso-bit. OMMX i2f4 at 6.25 / 12.5 / 25 % outliers is not the same bit budget as INT4
    or INT8, and no accuracy is measured here at all -- this is latency only.
  * NOT a full TPOT. A decode step is attention + linear; this is the linear half. Reading it
    as end-to-end TPOT would double-count what the KV figures already show.
  * CUTLASS is example 55 (`55_hopper_mixed_dtype_gemm`), a reference kernel, not a tuned
    production INT4 path. It and the CuTe OMMX arm are Hopper-only, so the A100 panel carries
    bf16, LLM.int8 and the two-launch OMMX pair.

    python figure/plot_linear_tpot.py figure/data_h200/linear_mem_bound.json figure/fig_linear_tpot_h200.png
"""
from __future__ import annotations

import json
import sys

C = {"bf16": "#8a8a86", "cutlass": "#2a78d6", "int8": "#eb6834"}
OMMX = {0.0625: "#5cc3a3", 0.125: "#1baf7a", 0.25: "#0c6b4a"}
SHAPE_TITLE = {"square": "square (4096x4096)", "gate/up": "gate / up projection",
               "down": "down projection"}


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[-1].strip())
    res = json.load(open(sys.argv[1]))
    rows = res.get("rows") or []
    if not rows:
        sys.exit(f"{sys.argv[1]} has no rows")
    gpu = res.get("gpu", "?")
    Ms = sorted({r["M"] for r in rows})
    shapes = [s for s in ("square", "gate/up", "down") if any(r["shape"] == s for r in rows)]
    opcts = sorted({r["opct"] for r in rows})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(shapes), figsize=(4.6 * len(shapes), 4.2), dpi=150,
                             sharey=True)
    if len(shapes) == 1:
        axes = [axes]

    def series(shape, key, opct=None):
        out = []
        for m in Ms:
            cand = [r for r in rows if r["shape"] == shape and r["M"] == m
                    and (opct is None or r["opct"] == opct)]
            vals = [r[key] for r in cand if isinstance(r.get(key), (int, float))]
            out.append(min(vals) if vals else None)
        return out

    for ax, shape in zip(axes, shapes):
        x = list(range(len(Ms)))
        # baselines: one curve each, taken at the batch size (they do not vary with opct)
        for key, lbl in (("bf16_ms", "bf16"), ("cutlass_ms", "CUTLASS INT4 (ex55)"),
                         ("int8_ms", "LLM.int8")):
            y = series(shape, key)
            if all(v is None for v in y):
                continue
            ax.plot(x, y, marker="o", ms=4, lw=1.6,
                    color=C[key.split("_")[0]], label=lbl, zorder=2)
        # OMMX: one curve per outlier ratio, light -> dark. CuTe (Hopper, solid) where it
        # was measured; the shipped two-launch pair dashed (the only OMMX arm on sm_80).
        for opct in opcts:
            for key, style, lbl in (("ommx_ms", "-", "CuTe"),
                                    ("ommx_classic_ms", "--", "two-launch")):
                y = series(shape, key, opct)
                if all(v is None for v in y):
                    continue
                ax.plot(x, y, marker="s", ms=4, lw=1.8, ls=style, color=OMMX.get(opct, "#0c6b4a"),
                        label=f"OMMX i2f4 {opct*100:g}% ({lbl})", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in Ms])
        ax.set_xlabel("batch (decode tokens per step)")
        ax.set_title(SHAPE_TITLE.get(shape, shape), fontsize=11)
        ax.set_yscale("log")
        ax.grid(axis="y", which="both", alpha=0.3, ls="--")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("linear latency (ms / decode step)")

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(3, len(l)), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.22))
    fig.suptitle(f"Decode-linear latency — {gpu}", fontsize=13, y=1.0)
    fig.text(0.5, -0.30,
             "LATENCY ONLY, and NOT iso-bit: on the NPU basis (combinadic positions) OMMX i2f4 "
             "at gs=64 is 3.94 / 4.39 / 5.14 bits per weight for 6.25 / 12.5 / 25 % outliers,\n"
             "a different budget from INT4 or INT8, and no accuracy is measured here. "
             "This is the LINEAR half of a decode step, not end-to-end TPOT. "
             "CUTLASS is example 55, a Hopper reference kernel, not a tuned production path. "
             "OMMX solid = CuTe wgmma base + packed correction (Hopper); dashed = the shipped two-launch pair.\n"
             "The A100 host is shared: the same bf16 matmul measured 0.041-0.104 ms across three runs, so on that panel compare "
             "arms WITHIN one batch point (timed back to back), not levels across panels.",
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout()
    fig.savefig(sys.argv[2], bbox_inches="tight")
    print(f"LINEAR_TPOT_FIG_DONE {sys.argv[2]}")

    # the numbers, so the figure is checkable without opening it
    print(f"[linear] {gpu}  batch  " + "  ".join(f"{m:>7}" for m in Ms))
    for shape in shapes:
        for key, lbl in (("bf16_ms", "bf16"), ("cutlass_ms", "cutlass"), ("int8_ms", "int8")):
            y = series(shape, key)
            if all(v is None for v in y):
                continue
            print(f"[linear] {shape:>9} {lbl:>8}  " +
                  "  ".join(f"{v:7.4f}" if v is not None else "      -" for v in y))
        for opct in opcts:
            for key, lbl in (("ommx_ms", "cute"), ("ommx_classic_ms", "classic")):
                y = series(shape, key, opct)
                if all(v is None for v in y):
                    continue
                print(f"[linear] {shape:>9} {f'{lbl}{opct*100:g}%':>12}  " +
                      "  ".join(f"{v:7.4f}" if v is not None else "      -" for v in y))


if __name__ == "__main__":
    main()
