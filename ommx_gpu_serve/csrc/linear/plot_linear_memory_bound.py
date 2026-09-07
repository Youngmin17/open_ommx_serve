# SPDX-License-Identifier: Apache-2.0
"""Render the memory-bound decode-linear latency figure from bench_linear_memory_bound result.json.
bf16 vs CUTLASS INT4 (live ex55) vs LLM.int8 vs OMMX i2f4 (3 outlier ratios). dataviz: gray=bf16 (upper
byte bound), blue=CUTLASS INT4, orange=LLM.int8, aqua sequential ramp=OMMX i2f4 (light->dark = more
outliers). LATENCY only — honest caption states no accuracy/iso-bit claim + CUTLASS=ex55 reference.
Usage: python plot_linear_memory_bound.py result.json out.png
"""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#dcdbd6"
C_BF16    = "#8a8a86"   # gray  (upper byte bound)
C_CUTLASS = "#2a78d6"   # blue
C_INT8    = "#eb6834"   # orange
OMMX_RAMP = {0: "#5cc3a3", 1: "#1baf7a", 2: "#0c6b4a"}   # aqua light->dark by outlier ratio


def main():
    res = json.load(open(sys.argv[1]))
    out = sys.argv[2] if len(sys.argv) > 2 else "fig_linear_tpot.png"
    rows, shapes, Ms, opcts = res["rows"], res["shapes"], res["M"], res["opcts"]
    int8_label = res.get("int8_label", "LLM.int8")

    def ser(shape, key, opct=None):
        rs = sorted([r for r in rows if r["shape"] == shape and (opct is None or r["opct"] == opct)],
                    key=lambda r: r["M"])
        return [r["M"] for r in rs], [r.get(key) for r in rs]

    def first(shape, key, opct=None):
        for r in rows:
            if r["shape"] == shape and (opct is None or r["opct"] == opct) and r.get(key) is not None:
                return r[key]
        return None

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK,
                         "axes.labelcolor": INK2, "axes.edgecolor": GRID, "xtick.color": INK2,
                         "ytick.color": INK2, "font.size": 11})
    fig, axes = plt.subplots(1, len(shapes), figsize=(4.7 * len(shapes), 4.8), sharey=True)
    if len(shapes) == 1: axes = [axes]

    for ax, shape in zip(axes, shapes):
        for i, opct in enumerate(opcts):
            x, y = ser(shape, "ommx_ms", opct)
            npv = first(shape, "npv", opct)
            ax.plot(x, y, marker="^", ms=6, lw=2.0, color=OMMX_RAMP[i], zorder=5,
                    label=f"OMMX i2f4  npv={npv} ({opct*100:.1f}% outliers)")
        xc, yc = ser(shape, "cutlass_ms")
        if any(v is not None for v in yc):
            ax.plot(xc, [v for v in yc], marker="s", ms=6, lw=2.4, color=C_CUTLASS, zorder=6,
                    label="CUTLASS INT4 (ex55, live)")
        xi, yi = ser(shape, "int8_ms")
        if any(v is not None for v in yi):
            ax.plot(xi, yi, marker="o", ms=5.5, lw=1.8, ls="--", color=C_INT8, zorder=3, label=int8_label)
        xb, yb = ser(shape, "bf16_ms")
        ax.plot(xb, yb, marker="D", ms=5, lw=1.8, ls=":", color=C_BF16, zorder=2, label="bf16 (cuBLAS)")
        ax.set_yscale("log")
        ax.set_title(shape, color=INK, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("batch M (decode)")
        ax.set_xticks(Ms)
        ax.grid(True, which="both", color=GRID, lw=0.7, alpha=0.7, zorder=0)
        ax.tick_params(length=0)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.set_yticks([0.02, 0.03, 0.05, 0.1, 0.2, 0.5])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].set_ylabel("decode-linear latency (ms, log)")

    handles, labels = axes[0].get_legend_handles_labels()
    order = [3, 4, 5, 0, 1, 2]  # CUTLASS, int8, bf16, OMMX x3
    order = [o for o in order if o < len(handles)]
    fig.legend([handles[o] for o in order], [labels[o] for o in order], loc="lower center", ncol=3,
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.02), labelcolor=INK2)
    fig.suptitle("Memory-bound decode-linear latency: OMMX i2f4 vs CUTLASS INT4 vs LLM.int8 vs bf16  —  H200",
                 color=INK, fontsize=14, fontweight="bold", y=1.0)
    fig.text(0.5, 0.935, "lower = faster (log y). Decode = memory/launch-bound: the INT2-base regime where "
             "OMMX's fewer weight bytes win at small M.", ha="center", color=INK2, fontsize=10)
    p = res.get("parity", {})
    par = (f"format bit-exact to fakequant (cos={p.get('cos')})" if p.get("cos") else "format = fakequant i2f4")
    fig.text(0.5, -0.055, "LATENCY only (no accuracy/iso-bit claim). OMMX i2f4 " + par +
             "; CUTLASS = ex55 mixed-input reference (not the tuned Marlin/Machete kernel); OMMX eager 2-launch.",
             ha="center", color=INK2, fontsize=8.5, style="italic")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    print(f"PLOT_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
