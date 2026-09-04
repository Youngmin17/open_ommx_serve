#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""fig7: SINGLE grouped bar chart. x = context length, y = B=1 decode TPOT (ms).

Bars per ctx group: vLLM paged-triton-attn v1 / OMMX / KIVI / Kitty. The OMMX bar ONLY is
stacked with internal component ratios (base/outlier/unpack/scale/pack) + % labels; the
others are solid totals. Llama-3.1-8B, H200.

HONEST: triton-v1 + OMMX = vLLM FULL CUDA-graph wall-clock (comparable); KIVI/Kitty
(hatched //) = HF eager loop (incl ~10ms/step py+launch overhead) -> trend not abs height.

SUPERSEDED — use `figure/plot_fig7.py`. That module renders the same panel from
`figure/collect.py` output, i.e. from measurements, and `repro/fig7_sweep.sh` produces those
measurements for all four bars. This file is kept only as the layout prototype it always was.

PROVENANCE — DO NOT CITE THIS FIGURE. The five series below are hardcoded literals. No
script in this repo produces them, nothing in run.sh or repro/ regenerates them, and they
are not the shipped H200 series either. Against `figure/data/h200.json` (the collected
series the README's own table is drawn from):
  * OMMX  7.37/8.46/12.95/18.76/30.67  vs  ommx_vllm  9.47/8.45/18.04/30.06/51.28 ms
    — up to 1.67x apart, and the internal component split below matches neither;
  * KIVI  21.59/21.54/22.88/32.60/51.79  vs  kivi_hf  29.44/29.22/52.70/89.59/163.52 ms
    — 1.36x to 3.16x apart;
  * the triton-v1 and Kitty rows land within ~3% and ~1% of vllm_bf16 / kitty_hf.
So this is an earlier, superseded sweep that nothing in the tree can reproduce or check —
and even the series it is closest to is declared uncitable by README.md ("The shipped
`figure/data_*/` JSONs predate the current bench and must be regenerated before being
cited"). Treat this module as a LAYOUT PROTOTYPE for the fig7 panel, not as data; the same
warning is stamped onto the rendered PNG so a screenshot cannot escape it.

IMPORT SAFETY: every statement below lives inside a function. This module ships inside the
declared `ommx_gpu_serve.bench` package, and it used to build and save the figure at module
scope — so `import ommx_gpu_serve.bench.plot_fig7_single_bars`, and even `--help`, raised
`FileNotFoundError` from the PIL save because `bench/figs/` does not exist. matplotlib and
numpy are imported lazily for the same reason: neither is a dependency of this package.

    python -m ommx_gpu_serve.bench.plot_fig7_single_bars --out <dir>
"""
import argparse
import os

CTXS = [1024, 4096, 16384, 32768, 65536]
CTXL = ["1K", "4K", "16K", "32K", "64K"]

# ── UNSOURCED HARDCODED SERIES — see PROVENANCE above before using any of these ──────────
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

# Stamped on the PNG itself. A rendered figure travels without its source file, so the
# provenance caveat has to travel with the pixels.
STAMP = ("PROTOTYPE LAYOUT — NOT CITABLE: these series are hardcoded in "
         "plot_fig7_single_bars.py, are produced by no script in this repo, and disagree "
         "with the shipped figure/data/h200.json (OMMX by up to 1.67x, KIVI by "
         "1.36-3.16x).")


def build_figure(out_dir, dpi=145):
    """Render fig7 into out_dir, creating it if needed. Returns the written path."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # headless: write a PNG, never open a display
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)   # the old module-scope savefig died here
    fig, ax = plt.subplots(figsize=(13, 6.5))
    g = np.arange(len(CTXS)); w = 0.2
    x_tr = g - 1.5 * w; x_om = g - 0.5 * w; x_ki = g + 0.5 * w; x_kt = g + 1.5 * w

    ax.bar(x_tr, TRITON, w, color="#bbb", edgecolor="k", label="vLLM triton-v1 (bf16)")
    # OMMX stacked with internal % (normalized to the OMMX_FULL height)
    for i in range(len(CTXS)):
        comp = OMMX_COMP[i]; s = sum(comp); full = OMMX_FULL[i]; bottom = 0.0
        for j, (v, col) in enumerate(zip(comp, CCOL)):
            seg = v / s * full
            ax.bar(x_om[i], seg, w, bottom=bottom, color=col, edgecolor="white", lw=0.4,
                   label=CLABEL[j] if i == len(CTXS) - 1 else None)
            if v / s > 0.06:
                ax.text(x_om[i], bottom + seg / 2, f"{100*v/s:.0f}", ha="center",
                        va="center", fontsize=6.5, color="white", fontweight="bold")
            bottom += seg
    ax.bar(x_ki, KIVI, w, color="#39c", edgecolor="k", hatch="//",
           label="KIVI 2-bit (HF eager)")
    ax.bar(x_kt, KITTY, w, color="#e6a", edgecolor="k", hatch="//",
           label="Kitty 2-bit (HF eager)")
    # total labels
    for xs, ys in [(x_tr, TRITON), (x_om, OMMX_FULL), (x_ki, KIVI), (x_kt, KITTY)]:
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + 1.5, f"{yi:.0f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(g); ax.set_xticklabels(CTXL)
    ax.set_xlabel("context length (B=1)"); ax.set_ylabel("decode TPOT (ms/step)")
    # Title wrapped to three lines: as one 130-char line the second row was clipped off the
    # right edge of the canvas, taking the "not vs-vLLM comparable" caveat with it.
    ax.set_title("B=1 decode TPOT vs context — vLLM triton-v1 / OMMX (internal % shown) / "
                 "KIVI / Kitty\n"
                 "Llama-3.1-8B, H200. OMMX segmented: outlier+unpack(dequant) overwhelms "
                 "bf16 as ctx grows.\n"
                 "// = HF-eager path (abs height not vs-vLLM comparable)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.5, 0.008, STAMP, ha="center", va="bottom", fontsize=6.5, color="#a00",
             wrap=True)
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    path = os.path.join(out_dir, "fig7_single_4method_bars.png")
    plt.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(
        description=("Render the fig7 prototype panel. The series are hardcoded and "
                     "unsourced -- see the module docstring's PROVENANCE block; the "
                     "caveat is stamped on the PNG."))
    ap.add_argument("--out", default=os.environ.get(
        "FIG_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")),
        help="output directory (created if absent; default $FIG_OUT or bench/figs/)")
    ap.add_argument("--dpi", type=int, default=145)
    args = ap.parse_args()
    print("FIG7_UNSOURCED " + STAMP)
    print("FIG7_DONE", build_figure(args.out, dpi=args.dpi))


if __name__ == "__main__":
    main()
