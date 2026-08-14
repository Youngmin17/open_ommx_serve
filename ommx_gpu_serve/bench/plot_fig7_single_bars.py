#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# fig7: SINGLE grouped bar chart. x = context length, y = B=1 decode TPOT (ms).
# Bars per ctx group: vLLM paged-triton-attn v1 / OMMX / KIVI / Kitty.
# OMMX bar ONLY is stacked with internal component ratios (base/outlier/unpack/scale/pack)
# + % labels. Others are solid totals. Llama-3.1-8B, H200.
# HONEST: triton-v1 + OMMX = vLLM FULL CUDA-graph wall-clock (comparable); KIVI/Kitty
# (hatched //) = HF eager loop (incl ~10ms/step py+launch overhead) -> trend not abs height.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.environ.get("FIG_OUT", os.path.join(os.path.dirname(__file__), "figs"))
CTXS = [1024, 4096, 16384, 32768, 65536]
CTXL = ["1K", "4K", "16K", "32K", "64K"]

TRITON = [5.72, 6.24, 8.31, 11.03, 16.49]
OMMX_FULL = [7.37, 8.46, 12.95, 18.76, 30.67]
KIVI = [21.59, 21.54, 22.88, 32.60, 51.79]
KITTY = [11.67, 16.99, 40.13, 71.04, 132.68]
OMMX_COMP = [  # base, outlier, unpack, scale, pack (ms, ABL diff timing)
    (6.94, 0.62, 0.08, 0.02, 0.40),
    (7.04, 1.04, 0.44, 0.10, 0.41),
    (7.19, 2.78, 1.87, 0.26, 0.44),
    (7.37, 6.80, 3.72, 0.50, 0.39),
    (7.69, 13.90, 7.72, 0.99, 0.41),
]
CLABEL = ["OMMX base (attn math+KV load)", "OMMX outlier-membership",
          "OMMX unpack (bit-extract)", "OMMX scale/zp dequant", "OMMX pack/write"]
CCOL = ["#888", "#c00", "#e80", "#06c", "#093"]

fig, ax = plt.subplots(figsize=(13, 6.5))
g = np.arange(len(CTXS)); w = 0.2
x_tr = g - 1.5 * w; x_om = g - 0.5 * w; x_ki = g + 0.5 * w; x_kt = g + 1.5 * w

ax.bar(x_tr, TRITON, w, color="#bbb", edgecolor="k", label="vLLM triton-v1 (bf16)")
# OMMX stacked with internal % (normalized to measured OMMX_FULL height)
for i in range(len(CTXS)):
    comp = OMMX_COMP[i]; s = sum(comp); full = OMMX_FULL[i]; bottom = 0.0
    for j, (v, col) in enumerate(zip(comp, CCOL)):
        seg = v / s * full
        ax.bar(x_om[i], seg, w, bottom=bottom, color=col, edgecolor="white", lw=0.4,
               label=CLABEL[j] if i == len(CTXS) - 1 else None)
        if v / s > 0.06:
            ax.text(x_om[i], bottom + seg / 2, f"{100*v/s:.0f}", ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold")
        bottom += seg
ax.bar(x_ki, KIVI, w, color="#39c", edgecolor="k", hatch="//", label="KIVI 2-bit (HF eager)")
ax.bar(x_kt, KITTY, w, color="#e6a", edgecolor="k", hatch="//", label="Kitty 2-bit (HF eager)")
# total labels
for xs, ys in [(x_tr, TRITON), (x_om, OMMX_FULL), (x_ki, KIVI), (x_kt, KITTY)]:
    for xi, yi in zip(xs, ys):
        ax.text(xi, yi + 1.5, f"{yi:.0f}", ha="center", va="bottom", fontsize=6.5)

ax.set_xticks(g); ax.set_xticklabels(CTXL)
ax.set_xlabel("context length (B=1)"); ax.set_ylabel("decode TPOT (ms/step)")
ax.set_title("B=1 decode TPOT vs context — vLLM triton-v1 / OMMX (internal % shown) / KIVI / Kitty\n"
             "Llama-3.1-8B, H200. OMMX segmented: outlier+unpack(dequant) overwhelms bf16 as ctx grows. "
             "// = HF-eager path (abs height not vs-vLLM comparable)")
ax.legend(fontsize=7.5, ncol=2, loc="upper left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_single_4method_bars.png", dpi=145)
print("FIG7_DONE", f"{OUT}/fig7_single_4method_bars.png")
