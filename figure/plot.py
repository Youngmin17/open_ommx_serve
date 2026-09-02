#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the measured figures for one GPU from figure/data/<gpu>.json (collect.py output).

  fig_tpot_<gpu>.png     B=1 decode TPOT vs context, one bar per method (bf16/OMMX in
                         HF-eager, bf16/OMMX under the vLLM engine, KIVI, Kitty). OMMX(vLLM)
                         is a stacked bar split into its internal kernel work (base /
                         outlier-membership / unpack / scale-zp dequant / pack-write), each a
                         distinct color.
  fig_peakmem_<gpu>.png  peak GPU memory vs context (the low-bit KV footprint story).

Measured-only: methods/ctxs absent from the data are skipped (no cited fallback).

LABELS NAME THE BACKEND, NOT THE FILE. "vllm_bf16" is the TRITON_ATTN arm of the vLLM engine
(that is what e2e_to_figure.py feeds into vllm_bf16.json -- its payload says
method="vllm_triton"), so it is plotted as "bf16 (vLLM, Triton attn)". vLLM's DEFAULT
attention backend is FLASH_ATTN, which arrives as the separate "vllm_flash_attn" series and
is plotted as "bf16 (vLLM, FlashAttn)". Calling the Triton arm plain "bf16 (vLLM)" read as
"stock vLLM" and it is not.

DTYPE IS PART OF THE LABEL when the figure is not single-dtype: collect.py carries each
series' recorded compute dtype into methods[key]["provenance"]["dtype"], and _dtype_suffix()
appends it to every label as soon as two series disagree or a series records something other
than bf16. KIVI's packer is fp16-only (baseline/kivi/quant/new_pack.py hard-casts to
float16), so a run whose other arms are bf16 is genuinely mixed-dtype -- a fairness caveat
that has to be readable off the figure, not only out of repro/README.md.

Usage: python figure/plot.py --data figure/data/h200.json --outdir figure
"""
import argparse
import json
import os

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as _exc:                      # noqa: BLE001 - re-raised below
    # Plotting deps are NOT part of the serving/bench path, so an env that runs
    # `run.sh bench` perfectly can still be missing them and would otherwise fail here
    # with a bare traceback AFTER a multi-hour sweep. Say what is missing and what to do.
    raise SystemExit(
        f"figure/plot.py needs the plotting extras ({type(_exc).__name__}: {_exc}).\n"
        "  install: pip install -e '.[ommx]'   (numpy + matplotlib are declared there)\n"
        "        or: pip install numpy matplotlib\n"
        "  NOTE the measured data is already safe: `run.sh bench` has written\n"
        "       figure/data_<tag>/*.json and `figure/collect.py` (which needs NEITHER\n"
        "       numpy nor matplotlib) can still normalise them into\n"
        "       figure/data/<tag>.json. Only the PNG rendering is blocked, so install\n"
        "       the extras and re-run `run.sh figure --tag <tag>` -- do NOT re-run the\n"
        "       sweep."
    ) from _exc

# OMMX(vLLM) internal-TPOT segments (bottom -> top) + colors
SEG_ORDER = ["base", "outlier", "unpack", "scale", "pack"]
SEG_LABEL = {"base": "base (attn math + KV load)", "outlier": "outlier-membership",
             "unpack": "unpack (bit-extract)", "scale": "scale/zp dequant",
             "pack": "pack/write"}
SEG_COL = {"base": "#9a9a9a", "outlier": "#cc1f1f", "unpack": "#f08a00",
           "scale": "#1f6fd0", "pack": "#0a9b3c"}

# method draw order (left->right in each ctx group), label, single-bar color.
# The two vLLM bf16 series are the SAME engine with DIFFERENT attention backends, so each is
# named after its backend; "vllm_bf16" keeps its data key (old collected JSONs still plot)
# while its label stops implying stock vLLM.
METHODS = [
    ("ommx_vllm", "OMMX (vLLM)", None),                 # stacked breakdown (special)
    ("turboquant_vllm", "TurboQuant-3bit (vLLM)", "#e08a1e"),  # KV-quant peer, vLLM engine
    ("ommx_hf",   "OMMX (HF-eager)", "#2a8c6a"),
    ("bf16_hf",   "bf16 (HF-eager)", "#5b9bc4"),
    ("vllm_bf16", "bf16 (vLLM, Triton attn)", "#a9cfe8"),      # TRITON_ATTN arm
    ("vllm_flash_attn", "bf16 (vLLM, FlashAttn)", "#7a5ea8"),  # FLASH_ATTN = vLLM default
    ("kivi_hf",   "KIVI (HF-eager)", "#15546e"),
    ("kitty_hf",  "Kitty (HF-eager)", "#d94f9a"),
    # The ONLY weight-quantized series in this figure, and the label says so: every other bar
    # runs bf16 weights, so a plain "LMDeploy" would read as a like-for-like KV comparison.
    # It is also the only wall-clock-timed one (TurboMind exposes no per-step call to wrap) --
    # figure/bench_lmdeploy.py's docstring carries the full caveat.
    ("lmdeploy",  "LMDeploy W4A16KV4 (AWQ-INT4 w)", "#c4b7e6"),
]
CTX_LBL = {1024: "1K", 2048: "2K", 4096: "4K", 8192: "8K", 16384: "16K",
           32768: "32K", 65536: "64K", 131072: "128K"}


def _present(methods, key):
    m = methods.get(key)
    return m and any(v is not None for v in m.get("tpot") or [])


def _dtype_suffix(methods):
    """key -> " [dtype]" suffix, or {} when the whole figure is one dtype.

    Silent while every series that recorded a dtype recorded bf16 (the intended, comparable
    case). As soon as two series disagree, or a series records something else (fp16 -- what
    the KIVI arm must run at, and what every JSON shipped in figure/data_*/ actually is), the
    dtype is spelled out on EVERY label, including "?" for a series whose raw JSON recorded
    no dtype at all. Mislabelling an fp16 arm "bf16" is exactly the failure this prevents.

    An UNRECORDED dtype counts as a disagreement, not as agreement. "known == {bf16} -> stay
    silent" alone was still a way to draw an unmarked bar next to verified-bf16 ones: the
    vLLM JSONs shipped in figure/data_*/ predate the identity fields and record no dtype, so
    regenerating only the HF arms at bfloat16 and keeping those files left the legacy series
    -- an fp16-era measurement -- labelled plain "bf16 (vLLM, Triton attn)" while collect.py's
    COLLECT_DONE line was honestly printing it as "?". Only a figure where EVERY drawn series
    proved bf16 gets the silent treatment.
    """
    drawn = [k for k in methods if _present(methods, k)
             or any(v is not None for v in (methods.get(k) or {}).get("peak") or [])]
    seen = {k: ((methods.get(k) or {}).get("provenance") or {}).get("dtype")
            for k in methods}
    vals = {seen.get(k) for k in drawn}          # None included on purpose
    if not vals or vals == {"bf16"}:
        return {}
    return {k: f" [{seen.get(k) or '?'}]" for k in methods}


def plot_tpot(data, out):
    ctxs = data["ctxs"]; M = data["methods"]; gpu = data["gpu"]
    sfx = _dtype_suffix(M)
    keys = [(k, lbl + sfx.get(k, ""), col) for k, lbl, col in METHODS if _present(M, k)]
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
    # Both series must be present: the label is a RATIO, and M[ref_key] is not guaranteed to
    # exist at all (a vLLM-only collect has no HF-eager arms and used to KeyError here).
    ref_key = "ommx_hf"
    if _present(M, ref_key) and _present(M, "kivi_hf"):
        ommx = np.array([v if v is not None else np.nan for v in M[ref_key]["tpot"]], float)
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
    sfx = _dtype_suffix(M)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    marker = "o"
    for key, lbl, col in METHODS:
        lbl = lbl + sfx.get(key, "")
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
    # Only the HF-eager arms record peak_gb (the vLLM adapter writes peak_gb=null), so a
    # vLLM-only collect draws no lines at all -- ask for a legend then and matplotlib warns
    # about an empty one, which reads like a bug instead of "this data has no peak column".
    if ax.get_legend_handles_labels()[0]:
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
    # NOTHING TO DRAW is a normal outcome here, not a bug: run.sh deletes each leg's
    # figure/data_<tag>/*.json BEFORE running that leg, so a sweep in which every leg failed
    # leaves the dir empty, collect.py writes methods={}, and `run.sh all` still calls this
    # script. Without this guard plot_tpot() divided 0.8 by len(keys)==0 and the operator got
    # a bare ZeroDivisionError traceback -- which reads like a plotting bug instead of "the
    # bench produced no measurement". Same treatment as the missing-matplotlib guard above.
    M = data.get("methods") or {}
    if not any(_present(M, k) for k, _, _ in METHODS):
        raise SystemExit(
            f"figure/plot.py: {args.data} has no drawable series -- every method's 'tpot' is "
            f"missing or all-null (methods in file: {sorted(M) or 'none'}).\n"
            "  That is what an empty or fully-failed bench looks like: run.sh purges each\n"
            "  leg's figure/data_<tag>/*.json before running it, so a leg that did not finish\n"
            "  leaves NO file and figure/collect.py collects nothing.\n"
            "  Read the `run.sh bench` log for the leg(s) that failed and re-run those legs.\n"
            "  Nothing is plotted and no PNG is written -- an empty figure would be worse."
        )
    tag = data["gpu"].lower()
    plot_tpot(data, os.path.join(args.outdir, f"fig_tpot_{tag}.png"))
    plot_peakmem(data, os.path.join(args.outdir, f"fig_peakmem_{tag}.png"))


if __name__ == "__main__":
    main()
