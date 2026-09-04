#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the LiveCodeBench-v6 KV-quant accuracy figure from analyze_lcb.py's data file.

  fig_lcb_<tag>.png   three panels, all measured, no cited fallback:
    (a) pass@1 on every scored pair
    (b) pass@1 restricted to pairs where NEITHER side ran out of tokens
    (c) token-cap hit rate, the mechanism that separates (a) from (b)

Panels (a) and (b) share a form on purpose. A deficit that survives from (a) into (b) is the
method getting answers wrong; a deficit that disappears is the method being verbose enough to run
out of tokens before it emits a code block.

Every arm is drawn on the COMMON SUBSET -- the problems every arm generated -- so all bars, and
the bf16 reference tick, describe the same problem set. That equal footing is what makes the
left-to-right ordering meaningful; an arm stopped early would otherwise sit on an easier or harder
slice of the seeded subset. The tick is drawn per bar rather than as one global line because the
data file also carries a per-arm-coverage view where the baselines legitimately differ.

The data file's "_provenance" string is RENDERED ONTO THE FIGURE, not just carried in the JSON.
The only LCB data shipped in this repo (figure/data/lcb_qwen3_30b_stripped_recipes.json) records
there that it predates the fp16-region fix -- KIVI ran without its 128-token fp16 residual and
Kitty without its 32-token sink, so those two columns bound their methods FROM BELOW -- and a PNG
is what gets pasted into a slide, where a caveat that stayed in the JSON is a caveat nobody reads.
A data file with no "_provenance" draws no such band, but says so on stdout (LCB_PROVENANCE ...
NONE RECORDED) -- unstated provenance is unknown provenance, not a clean bill of health.
For the same reason the output filename is derived from the DATA filename by default: rendering a
superseded file and its corrected replacement under one fig_lcb_qwen3_30b.png silently overwrote
one with the other, leaving no way to tell which one is on disk.

Usage: python figure/plot_lcb.py --data figure/data/lcb_qwen3_30b.json --outdir figure
"""
import argparse
import json
import os
import textwrap

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as _exc:                      # noqa: BLE001 - re-raised below
    # Same contract as figure/plot.py: the plotting extras are NOT needed to measure, so
    # an env that produced the data can legitimately lack them. Name the missing dep and
    # say what does NOT have to be re-run, instead of a bare traceback.
    raise SystemExit(
        f"figure/plot_lcb.py needs the plotting extras ({type(_exc).__name__}: {_exc}).\n"
        "  install: pip install -e '.[ommx]'   (numpy + matplotlib are declared there)\n"
        "        or: pip install numpy matplotlib\n"
        "  Only the PNG rendering is blocked; the scored LiveCodeBench JSON this script\n"
        "  reads is already on disk, so install the extras and re-run this script alone."
    ) from _exc

# Categorical hues, fixed per arm and never cycled; validated for CVD separation, lightness band
# and chroma floor against a light surface. The worst adjacent CVD pair sits in the 6-8 band,
# which is legal only with secondary encoding -- hence a value label on every bar, always.
ARM_COLOR = {"ommx": "#1f9d63", "turboquant": "#e59216", "kitty": "#d64c96", "kivi": "#4a72d8"}
REF = "#6b6b6b"        # bf16 reference: ink, not a series colour -- it is a baseline, not an arm
INK, MUTED, GRID, SURFACE = "#1a1a1a", "#5c5c5c", "#dcdcdc", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)


def _bars(ax, arms, vals, refs, title, ylabel, ylim, fmt="%.3f", shared_ref=None):
    """refs=None + shared_ref: one baseline line (all arms on the same problems).
    refs=list: a tick per bar (the cap-free subset differs per arm, so bf16 does too)."""
    x = np.arange(len(arms))
    if shared_ref is not None:
        ax.axhline(shared_ref, color=REF, lw=2, zorder=4)
        # left margin: a shared line can pass through every bar, so the label must clear them
        ax.text(-0.49, shared_ref + ylim[1] * 0.012, "bf16 " + fmt % shared_ref,
                ha="left", va="bottom", fontsize=9.5, color=REF, zorder=6)
    for i, a in enumerate(arms):
        if vals[i] is None:
            continue
        ax.bar(x[i], vals[i], width=0.54, color=ARM_COLOR[a["arm"]], zorder=3, linewidth=0)
        if refs is not None and refs[i] is not None:
            # the tick sits just above the bar, so the value goes inside it and the two never collide
            ax.text(x[i], vals[i] - ylim[1] * 0.028, fmt % vals[i], ha="center", va="top",
                    fontsize=11, color="white", fontweight="bold", zorder=6)
            ax.plot([x[i] - 0.31, x[i] + 0.31], [refs[i]] * 2, color=REF, lw=2,
                    solid_capstyle="butt", zorder=5)
            ax.text(x[i], refs[i] + ylim[1] * 0.012, fmt % refs[i], ha="center", va="bottom",
                    fontsize=9, color=REF)
        else:
            ax.text(x[i], vals[i] + ylim[1] * 0.018, fmt % vals[i], ha="center", va="bottom",
                    fontsize=11, color=INK, fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels([a["label"] for a in arms], fontsize=10.5, color=INK)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=10.5, color=MUTED)
    ax.set_title(title, fontsize=12, color=INK, pad=12, loc="left")
    _style(ax)


def _tag_from_data(path):
    """Default output tag = the data file's own name, minus the lcb_ prefix.

    figure/data/lcb_qwen3_30b.json -> qwen3_30b (the documented output name is unchanged), and
    lcb_qwen3_30b_stripped_recipes.json -> qwen3_30b_stripped_recipes. Before this, BOTH
    rendered to fig_lcb_qwen3_30b.png, so the superseded data and its corrected replacement
    overwrote each other and the PNG on disk could not be attributed to either. --tag still
    overrides."""
    base = os.path.splitext(os.path.basename(path))[0]
    return base[4:] if base.startswith("lcb_") else base


def main(a):
    if not os.path.exists(a.data):
        # An explicit refusal, not a traceback: the default path is the file the eval README's
        # step 3 tells you to produce, and it is not in the repo -- only the superseded one is.
        raise SystemExit(
            "plot_lcb.py: no data file at %r.\n"
            "  Produce it with `python eval/lcb/analyze_lcb.py --runs <dir> --ids <ids.json>\n"
            "  --json %s` (eval/lcb/README.md, Reproduce step 3), or\n"
            "  point --data at a file you already have. The only LCB data shipped in this repo\n"
            "  is figure/data/lcb_qwen3_30b_stripped_recipes.json, and it is SUPERSEDED -- read\n"
            "  its _provenance before citing anything drawn from it."
            % (a.data, a.data))
    d = json.load(open(a.data))
    arms = [r for r in d["arms"] if r["arm"] != "none"]
    arms.sort(key=lambda r: -r["pass1"])

    # PROVENANCE IS DRAWN, NOT JUST CARRIED. The data file records how it was produced (and,
    # for the shipped file, that two of its arms ran without the fp16 region their papers
    # specify, so those columns bound their methods from below). A figure is what gets pasted
    # into a slide; a caveat that stays in the JSON is a caveat the reader never sees. Absent
    # is reported on stdout rather than assumed benign -- an unstated provenance is unknown
    # provenance, not a clean bill of health.
    prov = d.get("_provenance")
    if prov is not None and not isinstance(prov, str):
        prov = json.dumps(prov, sort_keys=True)   # a structured block is still drawn, verbatim
    prov_lines = textwrap.wrap("PROVENANCE — " + prov, 168) if prov else []
    if prov:
        print("LCB_PROVENANCE %s: %s" % (a.data, prov))
    else:
        print("LCB_PROVENANCE %s: NONE RECORDED (the file carries no _provenance key, so the "
              "figure cannot state how it was produced)" % a.data)
    regions = d.get("_recipe_regions_enforced")
    if regions is not None:
        print("LCB_RECIPE_REGIONS_ENFORCED %s" % json.dumps(regions, sort_keys=True))

    # The caption band is sized in INCHES from the top so the panels keep their proportions
    # however many provenance lines there are; _y() converts to the figure fraction Matplotlib
    # wants. The no-provenance case reproduces the previous layout exactly.
    band = 0.30 + 0.155 * len(prov_lines) if prov_lines else 0.0
    fig_h = 5.9 + band
    def _y(inches_from_top):
        return 1.0 - inches_from_top / fig_h

    fig, axes = plt.subplots(1, 3, figsize=(16.5, fig_h))
    fig.patch.set_facecolor(SURFACE)

    bf = next(r for r in d["arms"] if r["arm"] == "none")
    # (a) and (c): one shared baseline -- every arm is scored on the same problems.
    # (b): the cap-free subset is arm-specific, so bf16 is too; that needs a tick per bar.
    _bars(axes[0], arms, [r["pass1"] for r in arms], None,
          "(a) pass@1 — all scored pairs", "pass@1", (0, 1.06), shared_ref=bf["pass1"])
    _bars(axes[1], arms, [r["pass1_free"] for r in arms], [r["bf16_free"] for r in arms],
          "(b) pass@1 — neither side hit the token cap", "pass@1", (0, 1.06))
    _bars(axes[2], arms, [100 * r["cap_hit"] for r in arms], None,
          "(c) generations that ran out of tokens", "cap-hit (%)", (0, 45), fmt="%.1f",
          shared_ref=100 * bf["cap_hit"])

    # Panel (c) reads as a mechanism only with the length inflation beside it.
    for i, r in enumerate(arms):
        axes[2].text(i, 2.0, "%+.0f%% len" % (100 * r["length_vs_bf16"]), ha="center",
                     va="bottom", fontsize=9.5, color="white", fontweight="bold", zorder=6)

    # Identity is never colour-alone: a legend is present, and every bar is directly labelled.
    handles = [plt.Rectangle((0, 0), 1, 1, color=ARM_COLOR[r["arm"]]) for r in arms]
    labels = ["%s — %.2f KV bits" % (r["label"], r["bits"]) for r in arms]
    handles.append(plt.Line2D([0], [0], color=REF, lw=2))
    labels.append("bf16 baseline (per-bar in (b): its cap-free subset differs by arm)")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.005), labelcolor=INK)

    n = "all arms on the same n=%d problems" % d.get("common_n", arms[0]["n"])
    fig.suptitle("KV-cache quantization on LiveCodeBench-v6 — %s" % d["model"],
                 fontsize=14.5, color=INK, x=0.008, ha="left", y=_y(0.0885))
    fig.text(0.008, _y(0.4425),
             "seeded subset (seed %d), %s · pass@1 @ 1 sample, T=%.1f top_p=%.2f top_k=%d "
             "seed=%d · batch_size=%d · max_new_tokens=%d · cache-level fake-quant · "
             "unofficial harness"
             % (d["subset_seed"], n, d["temperature"], d["top_p"], d["top_k"],
                d["seed"], d["batch_size"], d["cap_tokens"]),
             fontsize=9.2, color=MUTED, ha="left")
    fig.text(0.008, _y(0.6195),
             "Each arm runs its own recipe — NOT an iso-bit comparison. KV bits are payload "
             "only (scale/zero-point and OMMX outlier-membership encoding excluded).",
             fontsize=9.2, color=MUTED, ha="left")
    # Ink, not muted grey: this is the line that decides whether the bars may be cited at all.
    for i, line in enumerate(prov_lines):
        fig.text(0.008, _y(0.90 + 0.155 * i), line,
                 fontsize=8.8, color=INK, ha="left",
                 fontweight="bold" if i == 0 else "normal")

    fig.tight_layout(rect=[0, 0.055, 1, _y(0.7375 + band)])
    # Create --outdir rather than letting savefig raise FileNotFoundError from inside PIL after
    # every panel has already been rendered; figure/collect.py makes its output dir the same way.
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "fig_lcb_%s.png" % (a.tag or _tag_from_data(a.data)))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print("wrote %s" % out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="figure/data/lcb_qwen3_30b.json")
    p.add_argument("--outdir", default="figure")
    p.add_argument("--tag", default=None,
                   help="output name: fig_lcb_<tag>.png (default: derived from --data, so two "
                        "different data files cannot overwrite each other's figure)")
    main(p.parse_args())
