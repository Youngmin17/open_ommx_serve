#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paper Fig. 7 — B=1 decode TPOT vs context — rendered from MEASURED data only.

WHAT THIS REPLACES. The figure in the paper was produced by a script whose own header says
the three baseline bars were "CITED from the original ICCAD figure (digitized), NOT
re-measured" -- only the GPU-OMMX bar was a measurement. Three of four series therefore had
no reproducer anywhere in this tree, and one of them (KIVI) was non-monotonic in context
length, which is not something a decode-TPOT curve does. This module draws the same layout
from `figure/data/<gpu>.json`, i.e. from `figure/collect.py` output, and refuses to invent a
series it was not given: a missing arm is a missing bar plus a line on stdout, never a
literal.

THE RATIO ANNOTATION NAMES ITS REFERENCE. The published caption reads "the speedup and
slowdown ... of GPU-OMMX relative to BF16 vLLM", but the arithmetic in the script that drew
it was `KIVI[i] / GPU_OMMX[i]` -- the annotation was OMMX-vs-KIVI. Both are defensible
numbers; captioning one as the other is not. `--ratio-ref` picks the denominator series and
the chosen reference is printed *on the axes*, so the figure cannot drift from its caption
again.

THE SERIES ARE NOT ONE ENGINE, AND THE FIGURE SAYS SO. OMMX and bf16 come from the vLLM
engine (CUDA-event timed), KIVI from an HF-eager model file, LMDeploy from TurboMind
(wall-clock timed, and the only *weight*-quantized arm). Those asymmetries are stamped under
the axes rather than left to a caption, because a rendered PNG travels without its caption.

    python figure/plot_fig7.py --data figure/data/h200.json --outdir figure
"""
import argparse
import json
import os

# Bar order inside a ctx group, and where each series comes from. `key` is a collect.py
# method key; `stacked` marks the one series drawn as its internal component breakdown.
SERIES = [
    {"key": "ommx_vllm", "label": "GPU-OMMX (vLLM)", "color": None, "stacked": True},
    {"key": "lmdeploy", "label": "LMDeploy (W4A16KV4)", "color": "#c4b7e6", "stacked": False},
    {"key": "vllm_flash_attn", "label": "vLLM (bf16, FlashAttn)", "color": "#5b9bc4",
     "stacked": False, "alt": "vllm_bf16", "alt_label": "vLLM (bf16, Triton attn)"},
    {"key": "kivi_hf", "label": "KIVI (HF-eager)", "color": "#15546e", "stacked": False},
]
SEG_ORDER = ["base", "outlier", "unpack", "scale", "pack"]
SEG_LABEL = {"base": "base (attn math+KV load)", "outlier": "outlier-membership",
             "unpack": "unpack (bit-extract)", "scale": "scale/zp dequant",
             "pack": "pack/write"}
SEG_COL = {"base": "#9a9a9a", "outlier": "#cc1f1f", "unpack": "#f08a00",
           "scale": "#1f6fd0", "pack": "#0a9b3c"}
CTX_LBL = {1024: "1K", 2048: "2K", 4096: "4K", 8192: "8K", 16384: "16K",
           32768: "32K", 65536: "64K", 131072: "128K"}
# Engine / timing asymmetries, stamped onto the canvas. Keyed by the series actually drawn.
NOTE = {
    "ommx_vllm": "vLLM engine, CUDA-event timed",
    "vllm_flash_attn": "vLLM engine, CUDA-event timed",
    "vllm_bf16": "vLLM engine (Triton attn, NOT stock), CUDA-event timed",
    "kivi_hf": "HF-eager model file, CUDA-event timed",
    "lmdeploy": "TurboMind engine, WALL-CLOCK timed, AWQ-INT4 WEIGHTS (all others bf16 w)",
}


def _vals(methods, key):
    m = methods.get(key) or {}
    return m.get("tpot")


def _present(methods, key):
    v = _vals(methods, key)
    return bool(v) and any(x is not None for x in v)


def resolve(methods):
    """Series to draw, in order, each with the key that actually carried data.

    A series with an `alt` falls back to it (stock-vLLM FlashAttn -> the Triton-attn arm) and
    RELABELS itself when it does, because the two are different attention kernels and only one
    of them is what vLLM runs out of the box.
    """
    out, missing = [], []
    for s in SERIES:
        if _present(methods, s["key"]):
            out.append(dict(s))
        elif s.get("alt") and _present(methods, s["alt"]):
            out.append({**s, "key": s["alt"], "label": s["alt_label"]})
        else:
            missing.append(s["key"])
    return out, missing


def build(data, ratio_ref, outdir, dpi=170, title_gpu=None):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = data.get("methods") or {}
    ctxs = data.get("ctxs") or []
    gpu = title_gpu or data.get("gpu") or "?"
    drawn, missing = resolve(methods)
    if not drawn:
        raise SystemExit("plot_fig7: no series in this data file carry TPOT values. Nothing "
                         "was drawn and no PNG was written.")

    x = np.arange(len(ctxs), dtype=float)
    n = len(drawn)
    w = min(0.8 / n, 0.24)
    off = (np.arange(n) - (n - 1) / 2.0) * w

    fig, ax = plt.subplots(figsize=(3.0 + 1.9 * len(ctxs), 6.4))
    seg_handles = {}
    heights = {}

    for j, s in enumerate(drawn):
        key = s["key"]
        tp = np.array([v if v is not None else np.nan for v in _vals(methods, key)], float)
        heights[key] = tp
        xb = x + off[j]
        if s["stacked"] and (methods.get(key) or {}).get("breakdown"):
            bd = methods[key]["breakdown"]
            # Segments are differential ablation timings, so they need not sum to the measured
            # total; normalise to the measured bar height and let the % labels describe the
            # SHARE. A segment set that sums to zero (all arms OOMed) leaves the bar solid.
            raw = np.vstack([
                np.array([(v if v is not None else 0.0) for v in (bd.get(sg) or [])]
                         or [0.0] * len(ctxs), float)
                for sg in SEG_ORDER])
            tot = raw.sum(axis=0)
            scale = np.where(tot > 0, np.nan_to_num(tp, nan=0.0) / np.where(tot > 0, tot, 1.0), 0.0)
            seg = raw * scale
            bottom = np.zeros(len(ctxs))
            for i_sg, sg in enumerate(SEG_ORDER):
                h = ax.bar(xb, seg[i_sg], w, bottom=bottom, color=SEG_COL[sg],
                           edgecolor="white", linewidth=0.4)
                seg_handles[sg] = h
                for i in range(len(ctxs)):
                    if tp[i] and seg[i_sg][i] / tp[i] > 0.10:
                        ax.text(xb[i], bottom[i] + seg[i_sg][i] / 2,
                                f"{100 * seg[i_sg][i] / tp[i]:.0f}%", ha="center", va="center",
                                fontsize=7, color="white", fontweight="bold")
                bottom = bottom + seg[i_sg]
            ax.bar(xb, np.nan_to_num(tp, nan=0.0), w, fill=False, edgecolor="k", linewidth=1.5,
                   zorder=6, label=s["label"])
        else:
            ax.bar(xb, tp, w, color=s["color"], edgecolor="k", linewidth=0.6, label=s["label"])

    # ── ratio annotation over the OMMX bars, with its denominator NAMED on the axes ──
    ref = ratio_ref if _present(methods, ratio_ref) else None
    ratios = []
    if ref and "ommx_vllm" in heights:
        j = [s["key"] for s in drawn].index("ommx_vllm")
        om, rf = heights["ommx_vllm"], np.array(
            [v if v is not None else np.nan for v in _vals(methods, ref)], float)
        for i in range(len(ctxs)):
            if not (np.isfinite(om[i]) and np.isfinite(rf[i]) and om[i] > 0):
                ratios.append(None)
                continue
            r = rf[i] / om[i]
            ratios.append(r)
            ax.text(x[i] + off[j], om[i] * 1.06, f"×{r:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color=("#0a7d10" if r >= 1 else "#b01010"))
        ref_lbl = next((s["label"] for s in drawn if s["key"] == ref), ref)
        # Left, just under the component legend: the top strip is the legend's and the bottom
        # is the shortest bars'. Overlapping either hid the half of this box that says WHICH
        # series the ratio is against -- the exact ambiguity this annotation exists to remove.
        ax.text(0.012, 0.88, f"×N  =  {ref_lbl} TPOT / GPU-OMMX TPOT\n"
                             f"green ≥ 1 (OMMX faster)   red < 1 (OMMX slower)",
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="#fbfbf2", ec="#888", alpha=0.95))

    ax.set_yscale("log")
    # A log axis autoscaled to the data starts at the shortest bar, which visually truncates
    # every bar to its excess over the fastest arm. Anchor the floor below the data and leave
    # headroom above for the xN labels and the component legend.
    finite = [v for h in heights.values() for v in h if v == v and v > 0]
    if finite:
        ax.set_ylim(max(0.5, min(finite) / 2.5), max(finite) * 8.0)
    ax.set_xticks(x)
    ax.set_xticklabels([CTX_LBL.get(c, str(c)) for c in ctxs], fontsize=11)
    ax.set_xlabel("Context Length", fontsize=12)
    ax.set_ylabel("TPOT (ms)", fontsize=12)
    # "all bars measured" is a claim about THIS render, so it is only made when every series
    # the paper layout asks for actually carried data. With an arm missing the title says so
    # instead, because the incompleteness is the first thing a reader has to know.
    subtitle = ("all bars measured" if not missing
                else "MISSING: " + ", ".join(missing))
    ax.set_title(f"B=1 decode TPOT vs context — Llama-3.1-8B, {gpu} ({subtitle})", fontsize=13)
    ax.grid(axis="y", which="both", alpha=0.3, ls="--")

    handles, labels = ax.get_legend_handles_labels()
    if seg_handles:
        sl = [sg for sg in SEG_ORDER if sg in seg_handles]
        comp = ax.legend([seg_handles[sg] for sg in sl], [SEG_LABEL[sg] for sg in sl],
                         title="GPU-OMMX internal TPOT", fontsize=9, title_fontsize=9,
                         loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, framealpha=0.95)
        ax.add_artist(comp)
    ax.legend(handles, labels, fontsize=10, ncol=max(2, len(drawn)), loc="upper center",
              bbox_to_anchor=(0.5, -0.09))

    stamp = "  |  ".join(f"{s['label']}: {NOTE.get(s['key'], '?')}" for s in drawn)
    if missing:
        stamp += "  |  NOT MEASURED (no bar drawn): " + ", ".join(missing)
    fig.text(0.5, 0.005, stamp, ha="center", va="bottom", fontsize=6.6, color="#444", wrap=True)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "fig7_tpot_vs_ctx.png")
    plt.tight_layout(rect=(0, 0.055, 1, 1))
    plt.savefig(path, dpi=dpi)
    plt.close(fig)
    return path, drawn, missing, ratios, ctxs, heights


def main():
    ap = argparse.ArgumentParser(
        description="Render paper Fig.7 from collect.py output. Every bar is measured; a "
                    "series absent from the data file gets no bar and is named on stdout.")
    ap.add_argument("--data", required=True, help="figure/data/<gpu>.json from collect.py")
    ap.add_argument("--outdir", default="figure")
    ap.add_argument("--ratio-ref", default="kivi_hf",
                    help="denominator of the xN annotation. Default kivi_hf = the arithmetic "
                         "the published figure actually used (its caption said BF16 vLLM); "
                         "pass vllm_flash_attn to annotate what that caption claimed.")
    ap.add_argument("--gpu", default=None, help="override the GPU name in the title")
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()

    with open(args.data) as fh:
        data = json.load(fh)
    path, drawn, missing, ratios, ctxs, heights = build(
        data, args.ratio_ref, args.outdir, dpi=args.dpi, title_gpu=args.gpu)

    print("FIG7_SERIES " + ", ".join(f"{s['label']}[{s['key']}]" for s in drawn))
    if missing:
        print("FIG7_MISSING " + ", ".join(missing) +
              "  <- no bar drawn for these; the figure is incomplete against the paper layout")
    hdr = f"{'ctx':>7} " + " ".join(f"{s['key'][:14]:>14}" for s in drawn)
    print(hdr + ("  ratio_vs_" + args.ratio_ref if ratios else ""))
    for i, c in enumerate(ctxs):
        row = f"{CTX_LBL.get(c, c):>7} " + " ".join(
            (f"{heights[s['key']][i]:>14.2f}"
             if heights[s["key"]][i] == heights[s["key"]][i] else f"{'-':>14}")
            for s in drawn)
        if ratios:
            r = ratios[i] if i < len(ratios) else None
            row += f"  {('x%.2f' % r) if r else '-':>8}"
        print(row)
    print("FIG7_DONE " + path)


if __name__ == "__main__":
    main()
