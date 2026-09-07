# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot the canonical run.sh results.csv — stdlib csv + matplotlib(Agg), NO torch.

Three figures (whichever the CSV supports):
  * tpot_vs_ctx.png    : TPOT (p50, p99 shaded) vs context length, one line per arm,
                         faceted nothing (a fixed batch, default the smallest present).
  * ttft_vs_batch.png  : TTFT (p50) vs batch, one line per arm (at a fixed ctx).
  * ablation_bar.png   : stacked/grouped bar of TPOT p50 per arm at one (batch, ctx)
                         cell — the outlier/act-lane decomposition.

The CSV schema (run.sh canonical, one row per cell):
  run_id,utc,git_sha,model_sha,arch,gpu,suite,arm,backend,batch,ctx,
  ttft_p50,ttft_p99,tpot_p50,tpot_p99

Runs on a CPU-only box (no GPU, no torch). matplotlib uses the Agg backend so it
writes PNGs without a display, and is imported lazily (_pyplot) so that importing this
module -- it ships inside the declared `ommx_gpu_serve.bench` package -- never requires a
plotting stack. Asking it to actually plot without one still fails loudly.

    python -m ommx_gpu_serve.bench.plot_results --csv runs/<id>/results.csv --out runs/<id>
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


def _pyplot():
    """Import pyplot lazily, on the headless Agg backend.

    `import matplotlib` at module scope made this file un-importable wherever matplotlib is
    absent -- and `ommx_gpu_serve.bench` is a DECLARED, SHIPPED package while matplotlib is
    in none of its extras, so `import ommx_gpu_serve.bench.plot_results` raised
    ModuleNotFoundError even for `--help`. Importing here keeps the module importable and
    still fails with the same loud ModuleNotFoundError at the moment a plot is requested;
    main() forces that moment BEFORE the per-facet try/except, so a missing matplotlib can
    never be reported as three skipped figures and a rc=0 `PLOT_DONE n=0`.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: write PNGs, never open a display
    import matplotlib.pyplot as plt
    return plt


def _read(csv_path):
    """Read the CSV -> list of dict rows with numeric fields coerced (NaN-safe)."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("batch", "ctx"):
                try:
                    r[k] = int(r[k])
                except (KeyError, ValueError, TypeError):
                    r[k] = None
            for k in ("ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99"):
                try:
                    r[k] = float(r[k])
                except (KeyError, ValueError, TypeError):
                    r[k] = float("nan")
            # DROP CELLS THE SWEEP ITSELF MARKED NOT-OK. bench_e2e_a100 writes an `ok`
            # column and forces ok=False on every cell of an arm that OOMed, errored, or
            # could not prove the OMMX decode route fired (law #5: no silent fallback);
            # e2e_to_figure.py already gates on exactly that field. Without the same gate
            # here, `plot_results.py --csv <sweep>.csv` drew an "OMMX route not proven"
            # row as a published OMMX line -- the failure the gate exists to prevent, one
            # tool downstream. A CSV with no `ok` column is unaffected.
            if "ok" in r and str(r["ok"]).strip().lower() not in ("true", "1", "yes", ""):
                continue
            rows.append(r)
    return rows


def _finite(v):
    return v is not None and v == v  # NaN != NaN


def plot_tpot_vs_ctx(rows, out_dir, suite_filter=None):
    """TPOT-vs-ctx, one line per arm, at the smallest batch present. p99 shaded band."""
    data = [r for r in rows if (suite_filter is None or r.get("suite") in suite_filter)]
    if not data:
        return None
    batches = sorted({r["batch"] for r in data if r["batch"] is not None})
    if not batches:
        return None
    fixed_b = batches[0]
    sub = [r for r in data if r["batch"] == fixed_b]
    by_arm = defaultdict(list)
    for r in sub:
        if r["ctx"] is not None and _finite(r["tpot_p50"]):
            by_arm[r["arm"]].append(r)
    if not by_arm:
        return None
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in sorted(by_arm):
        pts = sorted(by_arm[arm], key=lambda r: r["ctx"])
        xs = [r["ctx"] for r in pts]
        p50 = [r["tpot_p50"] for r in pts]
        p99 = [r["tpot_p99"] if _finite(r["tpot_p99"]) else r["tpot_p50"] for r in pts]
        line, = ax.plot(xs, p50, marker="o", label=arm)
        ax.fill_between(xs, p50, p99, alpha=0.15, color=line.get_color())
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens, log2)")
    ax.set_ylabel("TPOT (ms / batch-agg decode step)")
    ax.set_title(f"TPOT vs context  (batch={fixed_b}, shaded=p50..p99)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(title="arm/backend")
    p = os.path.join(out_dir, "tpot_vs_ctx.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_ttft_vs_batch(rows, out_dir, suite_filter=None):
    """TTFT-vs-batch, one line per arm, at the smallest ctx present."""
    data = [r for r in rows if (suite_filter is None or r.get("suite") in suite_filter)]
    ctxs = sorted({r["ctx"] for r in data if r["ctx"] is not None})
    if not ctxs:
        return None
    fixed_c = ctxs[0]
    sub = [r for r in data if r["ctx"] == fixed_c]
    by_arm = defaultdict(list)
    for r in sub:
        if r["batch"] is not None and _finite(r["ttft_p50"]):
            by_arm[r["arm"]].append(r)
    if not by_arm:
        return None
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in sorted(by_arm):
        pts = sorted(by_arm[arm], key=lambda r: r["batch"])
        xs = [r["batch"] for r in pts]
        p50 = [r["ttft_p50"] for r in pts]
        p99 = [r["ttft_p99"] if _finite(r["ttft_p99"]) else r["ttft_p50"] for r in pts]
        line, = ax.plot(xs, p50, marker="s", label=arm)
        ax.fill_between(xs, p50, p99, alpha=0.15, color=line.get_color())
    ax.set_xlabel("batch size")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(f"TTFT vs batch  (ctx={fixed_c}, shaded=p50..p99)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(title="arm/backend")
    p = os.path.join(out_dir, "ttft_vs_batch.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_ablation_bar(rows, out_dir):
    """Grouped bar of TPOT p50 per arm at one representative (batch, ctx) cell.

    Picks the largest batch present (amortization regime — where arm differences show)
    at the largest ctx present, among the 'linear'/'ablation' suite rows."""
    data = [r for r in rows if r.get("suite") in {"linear", "ablation"}]
    if not data:
        return None
    cells = sorted({(r["batch"], r["ctx"]) for r in data
                    if r["batch"] is not None and r["ctx"] is not None})
    if not cells:
        return None
    b_sel = max(c[0] for c in cells)
    c_sel = max(c[1] for c in cells if c[0] == b_sel)
    sub = [r for r in data if r["batch"] == b_sel and r["ctx"] == c_sel and _finite(r["tpot_p50"])]
    if not sub:
        return None
    arms = sorted({r["arm"] for r in sub})
    tpot = {r["arm"]: r["tpot_p50"] for r in sub}
    p99 = {r["arm"]: (r["tpot_p99"] if _finite(r["tpot_p99"]) else r["tpot_p50"]) for r in sub}
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = range(len(arms))
    vals = [tpot[a] for a in arms]
    errs = [max(0.0, p99[a] - tpot[a]) for a in arms]
    bars = ax.bar(xs, vals, yerr=errs, capsize=4,
                  color=plt.cm.viridis([i / max(1, len(arms) - 1) for i in xs]))
    for x, a in zip(xs, arms):
        ax.text(x, tpot[a], f"{tpot[a]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(xs)); ax.set_xticklabels(arms, rotation=20, ha="right")
    ax.set_ylabel("TPOT (ms / batch-agg decode step)")
    ax.set_title(f"Ablation: TPOT p50 by arm  (batch={b_sel}, ctx={c_sel}, err=p99)")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    p = os.path.join(out_dir, "ablation_bar.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to run.sh results.csv")
    ap.add_argument("--out", default="", help="output dir for PNGs (default = CSV's dir)")
    args = ap.parse_args()

    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv)) or "."
    os.makedirs(out_dir, exist_ok=True)
    rows = _read(args.csv)
    if not rows:
        # "no rows" also covers "every row was gated out by ok=False" -- say so, or the
        # operator reads it as an empty file and goes looking for the wrong bug.
        raise SystemExit(
            f"[plot] nothing to plot from {args.csv}: it has no data rows, or every row "
            f"carries ok=False (a cell that OOMed, errored, or could not prove the OMMX "
            f"route fired). Read the `err` column of that CSV; not-ok cells are never "
            f"plotted."
        )

    # Import the plotting stack HERE, outside the per-facet try/except below: a missing
    # matplotlib is a broken environment, not "a figure that could not be drawn", and must
    # not be swallowed three times and reported as PLOT_DONE n=0 with rc=0.
    _pyplot()

    # Comparison line plots use the attention-backend rows (OMMX vs FA3 vs triton);
    # fall back to no filter if a CSV has no 'attn' suite (e.g. a linear-only run).
    suites = {r.get("suite") for r in rows}
    cmp_filter = {"attn"} if "attn" in suites else None
    made = []
    for fn in (
        lambda: plot_tpot_vs_ctx(rows, out_dir, suite_filter=cmp_filter),
        lambda: plot_ttft_vs_batch(rows, out_dir, suite_filter=cmp_filter),
        lambda: plot_ablation_bar(rows, out_dir),
    ):
        try:
            p = fn()
            if p:
                made.append(p)
        except Exception as e:  # noqa: BLE001 — one bad facet must not kill the rest
            print(f"[plot] skipped a figure: {type(e).__name__}: {e}")
    for p in made:
        print(f"[plot] wrote {p}")
    print(f"PLOT_DONE n={len(made)}")


if __name__ == "__main__":
    main()
