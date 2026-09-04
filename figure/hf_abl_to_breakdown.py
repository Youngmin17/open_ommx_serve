#!/usr/bin/env python3
# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Turn the HF-eager ablation sweep into the same breakdown schema the vLLM arm carries.

The four ``OMMX_ABL_*`` flags are read inside the decode kernel, so they apply to ANY caller
-- not just ``bench_e2e_a100``'s ablation arms. Running ``figure/bench.py --method ommx`` five
times under different flags therefore gives the HF-eager path the same differential
attribution the vLLM path gets, by the same formulas
(``ommx_gpu_serve/bench/e2e_to_figure.py``)::

    outlier = TPOT(full) - TPOT(no_outlier)
    unpack  = TPOT(full) - TPOT(no_unpack)
    scale   = TPOT(full) - TPOT(no_dequant)
    pack    = 0                      # no HF equivalent of the skipwrite arm
    base    = TPOT(no_all) - pack

WHAT THE NUMBERS ARE WORTH. The segments are differences of separately-timed runs, so they do
not sum to the total; the vLLM path handles this by clamping negatives and renormalizing, and
so does this script. That is an attribution, not a measurement of each stage in isolation --
cite ``breakdown_raw``.

The HF arm is also far noisier than the vLLM one at short context (p99 up to 30x the median
at 1K-16K, against ~1.4x at 64K), because the per-step Python overhead that dominates there
is not what the flags remove. Treat the 32K/64K columns as the informative ones; the script
records the observed p99/median ratio per cell so a reader can see which is which.

    python figure/hf_abl_to_breakdown.py --src figure/data_hfabl --into figure/data_bm/ommx_hf.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ARMS = ("full", "no_outlier", "no_unpack", "no_dequant", "no_all")
SEGMENTS = ("base", "outlier", "unpack", "scale", "pack")


def _load(src: str) -> dict:
    out = {}
    for a in ARMS:
        p = os.path.join(src, f"{a}.json")
        if not os.path.exists(p):
            sys.exit(f"missing ablation arm {a}: {p}")
        cells = json.load(open(p)).get("ctxs") or {}
        out[a] = {int(k): v for k, v in cells.items()}
    return out


EXPECT_FLAGS = {"full": set(), "no_outlier": {"OMMX_ABL_NO_OUTLIER"},
                "no_unpack": {"OMMX_ABL_NO_UNPACK"},
                "no_dequant": {"OMMX_ABL_K_NODEQUANT", "OMMX_ABL_V_NODEQUANT"},
                "no_all": {"OMMX_ABL_NO_OUTLIER", "OMMX_ABL_NO_UNPACK", "OMMX_ABL_K_NODEQUANT",
                           "OMMX_ABL_V_NODEQUANT"}}


def _check_flags(src) -> None:
    """The ablation JSONs are only distinguishable by the flags they were measured under.
    bench.py records them as meta.ommx_abl_active; when the field is present, each arm must
    carry exactly its own set (a leaked flag in `full` would silently shrink every segment)."""
    for arm, want in EXPECT_FLAGS.items():
        p = os.path.join(src, f"{arm}.json")
        if not os.path.exists(p):
            continue
        meta = (json.load(open(p)).get("meta") or {})
        if "ommx_abl_active" not in meta:
            print(f"[hf_abl] {arm}.json: meta.ommx_abl_active absent (pre-field JSON; unverifiable)")
            continue
        got = set(meta["ommx_abl_active"])
        if got != want:
            sys.exit(f"[hf_abl] {arm}.json was measured with {sorted(got)} but the arm means "
                     f"{sorted(want)}; the breakdown would be wrong")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with full/no_*.json from the sweep")
    ap.add_argument("--into", required=True, help="ommx_hf.json to write the breakdown into")
    args = ap.parse_args()

    T = _load(args.src)
    _check_flags(args.src)
    ctxs = sorted(set.intersection(*(set(T[a]) for a in ARMS)))
    if not ctxs:
        sys.exit("no context measured by all five arms")

    # Normalize onto the TARGET file's tpot, not the ablation "full" arm's. They are separate
    # measurements of the same configuration and differ by 1-4%, so normalizing onto the
    # ablation total while the bar is drawn at the target's height produced segments summing
    # to MORE than the bar -- printed as "base 102%", which is what made the A100 HF-eager bar
    # look wrong. The differentials still come from the ablation arms; only the scale factor
    # changes, and it now matches what the reader sees.
    target = json.load(open(args.into))
    tgt_tpot = {int(k): v.get("tpot_ms") for k, v in (target.get("ctxs") or {}).items()}
    raw, norm, noise = {s: {} for s in SEGMENTS}, {s: {} for s in SEGMENTS}, {}
    for c in ctxs:
        full = T["full"][c]["tpot_ms"]
        drawn = tgt_tpot.get(c) or full
        seg = {
            "outlier": full - T["no_outlier"][c]["tpot_ms"],
            "unpack": full - T["no_unpack"][c]["tpot_ms"],
            "scale": full - T["no_dequant"][c]["tpot_ms"],
            "pack": 0.0,
        }
        seg["base"] = T["no_all"][c]["tpot_ms"] - seg["pack"]
        for s in SEGMENTS:
            raw[s][str(c)] = round(seg[s], 4)
        clamped = {s: max(0.0, seg[s]) for s in SEGMENTS}
        tot = sum(clamped.values())
        for s in SEGMENTS:                       # renormalize onto the bar the reader sees
            norm[s][str(c)] = round((clamped[s] / tot * drawn) if tot > 0 else 0.0, 4)
        cell = T["full"][c]
        p99, med = cell.get("p99_ms"), cell.get("tpot_ms")
        noise[str(c)] = round(p99 / med, 2) if p99 and med else None

    d = target
    d["breakdown"] = norm
    d["breakdown_raw"] = raw
    d["breakdown_meta"] = {
        "method": "differential-ablation",
        "clamped_negative": True,
        "renormalized_to_total": True,
        "segments": list(SEGMENTS),
        "full_arm": "full",
        "base_arm": "no_all",
        "diff_arms": {"outlier": "no_outlier", "unpack": "no_unpack", "scale": "no_dequant"},
        "pack_segment": "absent on this path -- there is no HF equivalent of the vLLM "
                        "skipwrite arm, so pack is 0 and its cost sits inside base",
        "p99_over_median": noise,
        "normalized_to": "the target file's own tpot_ms, so the segments sum to the drawn bar",
        "ablation_full_tpot": {str(c): round(T["full"][c]["tpot_ms"], 4) for c in ctxs},
        "caveat": "segments are differences of separately-timed runs; they do not sum to the "
                  "total before renormalization. Short contexts are dominated by per-step "
                  "Python overhead the flags do not remove -- see p99_over_median.",
    }
    json.dump(d, open(args.into, "w"), indent=1)
    print(f"HF_BREAKDOWN_DONE {args.into}")
    for c in ctxs:
        print(f"  ctx={c:>6} " + "  ".join(
            f"{s}={norm[s][str(c)]:7.3f}" for s in SEGMENTS)
            + f"   p99/med={noise[str(c)]}")


if __name__ == "__main__":
    main()
