#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the measured figures for one GPU from figure/data/<gpu>.json (collect.py output).

  fig_tpot_<gpu>.png     B=1 decode TPOT vs context, 6 methods (bf16/OMMX in HF-eager+vLLM, KIVI, Kitty). OMMX(vLLM) is a stacked
                         bar split into its internal kernel work (base / outlier-membership
                         / unpack / scale-zp dequant / pack-write), each a distinct color.
  fig_peakmem_<gpu>.png  peak GPU memory vs context (the low-bit KV footprint story).

Measured-only: methods/ctxs absent from the data are skipped (no cited fallback).

Usage: python figure/plot.py --data figure/data/h200.json --outdir figure
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# OMMX(vLLM) internal-TPOT segments (bottom -> top) + colors
SEG_ORDER = ["base", "outlier", "unpack", "scale", "pack"]
SEG_LABEL = {"base": "base (attn math + KV load)", "outlier": "outlier-membership",
             "unpack": "unpack (bit-extract)", "scale": "scale/zp dequant",
             "pack": "pack/write"}
SEG_COL = {"base": "#9a9a9a", "outlier": "#cc1f1f", "unpack": "#f08a00",
           "scale": "#1f6fd0", "pack": "#0a9b3c"}

# method draw order (left->right in each ctx group), label, single-bar color
METHODS = [
    ("ommx_vllm", "OMMX (vLLM)", None),                 # stacked breakdown (special)
    ("turboquant_vllm", "TurboQuant-3bit (vLLM)", "#e08a1e"),  # KV-quant peer, vLLM engine
    ("ommx_hf",   "OMMX (HF-eager)", "#2a8c6a"),
    ("bf16_hf",   "bf16 (HF-eager)", "#5b9bc4"),
    ("vllm_bf16", "bf16 (vLLM)", "#a9cfe8"),
    ("kivi_hf",   "KIVI (HF-eager)", "#15546e"),
    ("kitty_hf",  "Kitty (HF-eager)", "#d94f9a"),
]
CTX_LBL = {1024: "1K", 2048: "2K", 4096: "4K", 8192: "8K", 16384: "16K",
           32768: "32K", 65536: "64K", 131072: "128K"}


def _present(methods, key):
    m = methods.get(key)
    return m and any(v is not None for v in m.get("tpot") or [])


def plot_tpot(data, out):
    ctxs = data["ctxs"]; M = data["methods"]; gpu = data["gpu"]
    keys = [(k, lbl, col) for k, lbl, col in METHODS if _present(M, k)]
    n = len(keys); x = np.arange(len(ctxs)); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(15.5, 6.4))

    seg_handles = {}
    for j, (key, lbl, col) in enumerate(keys):
        xb = x - 0.4 + w / 2 + j * w
        tpot = np.array([v if v is not None else np.nan for v in M[key]["tpot"]], float)
        bd = M[key].get("breakdown")
        if key == "ommx_vllm" and bd:
            bottom = np.zeros(len(ctxs))
            for s in SEG_ORDER:
                seg = np.array([v if v is not None else 0.0 for v in bd.get(s, [])], float)
                if seg.size != len(ctxs):
                    seg = np.zeros(len(ctxs))
                b = ax.bar(xb, seg, w, bottom=bottom, color=SEG_COL[s],
                           edgecolor="white", linewidth=0.4)
                seg_handles[s] = b
                for i in range(len(ctxs)):
                    if tpot[i] > 0 and seg[i] / tpot[i] > 0.11:
                        ax.text(xb[i], bottom[i] + seg[i] / 2, f"{100*seg[i]/tpot[i]:.0f}%",
                                ha="center", va="center", fontsize=6.5, color="white",
                                fontweight="bold")
                bottom += seg
            ax.bar(xb, tpot, w, fill=False, edgecolor="k", linewidth=1.5, zorder=6,
                   label=lbl)
        else:
            ax.bar(xb, np.nan_to_num(tpot), w, color=col, edgecolor="k", linewidth=0.7,
                   label=lbl)

    # speedup ×label over KIVI — SAME-ENGINE (both HF-eager) so the ratio is the KV-kernel
    # effect only, not the vLLM CUDA-graph engine (KIVI/Kitty have no vLLM integration; using
    # ommx_vllm here would fold the engine speedup into the "OMMX vs KIVI" number).
    ref_key = "ommx_hf"
    ommx = np.array([v if v is not None else np.nan for v in M[ref_key]["tpot"]], float)
    if _present(M, "kivi_hf"):
        kivi = np.array([v if v is not None else np.nan for v in M["kivi_hf"]["tpot"]], float)
        jo = [k for k, _, _ in keys].index(ref_key)
        xo = x - 0.4 + w / 2 + jo * w
        for i in range(len(ctxs)):
            if ommx[i] > 0 and not np.isnan(kivi[i]):
                s = kivi[i] / ommx[i]
                ax.text(xo[i], ommx[i] * 1.06, f"×{s:.1f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold", color="#0a7d10")

    ax.set_yscale("log"); ax.set_ylim(1, max(320, np.nanmax(
        [np.nanmax([v for v in M[k]["tpot"] if v]) for k, _, _ in keys]) * 1.3))
    ax.set_yticks([1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300])
    ax.set_yticklabels([1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300])
    ax.set_xticks(x); ax.set_xticklabels([CTX_LBL.get(c, str(c)) for c in ctxs], fontsize=11)
    ax.set_xlabel("Context length (B=1)", fontsize=12)
    ax.set_ylabel("Decode TPOT (ms/step)", fontsize=12)
    ax.set_title(f"B=1 decode TPOT — Llama-3.1-8B, {gpu}  (×N = KIVI/OMMX decode, both HF-eager)",
                 fontsize=13)
    ax.grid(axis="y", which="both", alpha=0.3, ls="--")

    h, l = ax.get_legend_handles_labels()
    leg1 = ax.legend(h, l, fontsize=10, ncol=len(keys), loc="upper center",
                     bbox_to_anchor=(0.5, -0.09))
    ax.add_artist(leg1)
    if seg_handles:
        sh = [seg_handles[s] for s in SEG_ORDER if s in seg_handles]
        sl = [SEG_LABEL[s] for s in SEG_ORDER if s in seg_handles]
        ax.legend(sh, sl, title="OMMX(vLLM) internal kernel TPOT", fontsize=9,
                  title_fontsize=9.5, loc="upper left", ncol=1, framealpha=0.95)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out, dpi=150); plt.close()
    print("FIG_DONE", out)


def plot_peakmem(data, out):
    ctxs = data["ctxs"]; M = data["methods"]; gpu = data["gpu"]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    marker = "o"
    for key, lbl, col in METHODS:
        m = M.get(key)
        if not m or not any(v is not None for v in m.get("peak") or []):
            continue
        peak = [v if v is not None else np.nan for v in m["peak"]]
        c = SEG_COL["base"] if key == "ommx_vllm" else col
        ls = "--" if key == "ommx_vllm" else "-"
        ax.plot(range(len(ctxs)), peak, marker=marker, ls=ls, lw=2, color=c, label=lbl)
    ax.set_xticks(range(len(ctxs))); ax.set_xticklabels([CTX_LBL.get(c, str(c)) for c in ctxs])
    ax.set_xlabel("Context length (B=1)", fontsize=12)
    ax.set_ylabel("Peak GPU memory (GB)", fontsize=12)
    ax.set_title(f"Peak process memory — Llama-3.1-8B, {gpu}  "
                 f"(weights + prefill activations + KV; NOT KV storage alone)", fontsize=12)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=9.5, ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()
    print("FIG_DONE", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    with open(args.data) as fh:
        data = json.load(fh)
    tag = data["gpu"].lower()
    plot_tpot(data, os.path.join(args.outdir, f"fig_tpot_{tag}.png"))
    plot_peakmem(data, os.path.join(args.outdir, f"fig_peakmem_{tag}.png"))


if __name__ == "__main__":
    main()
