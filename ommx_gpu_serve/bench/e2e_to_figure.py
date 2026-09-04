#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Adapt bench_e2e_a100.py output JSON -> figure/collect.py inputs:
  vllm_bf16.json   (the TRITON_ATTN arm: bf16 weights + vLLM's *Triton* attention)
  ommx_vllm.json   (OMMX arm total + internal-kernel breakdown, differential ATTRIBUTION)
  <--out-turboquant>  (optional: the TURBOQUANT KV-quant peer, total bar only)
  <--flash-arm>       (optional: vLLM's DEFAULT FlashAttention arm — the stock baseline)

NAMING (H7): the file written to --out-vllm-bf16 keeps that FILENAME (figure/collect.py
keys off the filename), but its payload now says what it actually is:
  "method": "vllm_triton", "attention_backend": "TRITON_ATTN", "method_alias": "vllm_bf16".
It is NOT vLLM's default attention path. vLLM's default is FLASH_ATTN; run the bench with
the FA arm and pass --flash-arm to emit that baseline as "vllm_flash_attn".

BREAKDOWN HONESTY CONTRACT (H4) — read before citing the stacked OMMX bar. The five
segments are a differential *attribution*, not five measured stage timers. Each segment is
a TPOT difference between two separately-built engines (one arm per subprocess), so it
carries the full run-to-run noise of both arms and can come out NEGATIVE. The plot needs
segments that are non-negative and sum to the measured total, so "breakdown" is produced by
(1) substituting 0.0 for any missing arm, (2) clamping negatives to 0.0, and (3) rescaling
all five by total/sum(clamped). Steps (1)-(3) destroy information, therefore this adapter
ALSO emits:
  "breakdown_raw"  : the same five values RAW — unclamped, unnormalized, and null (not 0.0)
                     when the differential arm is missing. Negative values are kept.
  "breakdown_meta" : {"method": "differential-ablation", "clamped_negative": true,
                      "renormalized_to_total": true, "residual_bucket": "outlier",
                      "renorm_factor": {<ctx>: total/sum(clamped)}, plus the per-ctx
                      clamped-segment list, the clamped sum, the missing arms, and the raw
                      per-arm TPOT the whole decomposition was derived from}
so a reader can recompute every plotted number and see exactly what was clamped/rescaled.

Differential recipe (larger arm keeps more work; all arms bf16-w + CUSTOM, B=1):
  unpack = TPOT(abl_attn) - TPOT(abl_attn_no_unpack)          (bit-extract ALU)
  scale  = TPOT(abl_attn) - TPOT(abl_attn_no_dequant)         (scale/zp dequant)
  pack   = TPOT(abl_attn) - TPOT(abl_attn_skipwrite)          (per-step KV pack/write)
  base   = TPOT(abl_attn_no_all) - pack                       (attn math + KV load)
  outlier= TPOT(abl_attn) - base_arm - unpack - scale         (RESIDUAL bucket: it absorbs
                                                               every error in the four above)
"outlier" is a residual, so it is the least trustworthy segment by construction: whatever
the other four fail to explain lands there. In the shipped H200 data it is 0.0 at 1K/4K and
37.4 of 51.3 ms at 64K, i.e. it is dominated by the residual, not by a measured outlier
kernel. Cite "base/unpack/scale/pack" deltas from breakdown_raw, never the residual alone.

INPUT CONTRACT: the bench JSON must carry `config.cmp_ctxs` and a non-empty `arms` map.
Either one missing is a HARD ERROR, not an empty output: every file written here is keyed
off those two, so an adaptation of zero cells used to exit 0 after writing four
well-formed-looking JSONs whose "ctxs" were all `{}`.

Usage: python e2e_to_figure.py --in runs/e2e.json --out-vllm-bf16 vllm_bf16.json \
         --out-ommx-vllm ommx_vllm.json [--flash-arm vllm_flash_attn.json]
"""
import argparse
import json
import os
import re

# plot segment order (bottom -> top); figure/collect.py SEGS must stay in sync
SEGMENTS = ("base", "outlier", "unpack", "scale", "pack")
# arm roles in the differential decomposition
FULL_ARM = "abl_attn"            # every delta is measured against this arm
BASE_ARM = "abl_attn_no_all"     # base attn math+mem (all OMMX decode costs skipped)
DIFF_ARMS = {                    # segment -> the arm that SKIPS that cost
    "unpack": "abl_attn_no_unpack",
    "scale": "abl_attn_no_dequant",
    "pack": "abl_attn_skipwrite",
}
# candidate arm names for vLLM's DEFAULT FlashAttention baseline, first match wins.
# "FA" is the honestly-named arm added to bench_e2e_a100._build_plan; "FA3" is the same
# FLASH_ATTN backend under its historical (Hopper-flavoured) name.
FLASH_ARM_NAMES = ("FA", "FA3")



_INPUT_PATH = [None]   # set from --in so _provenance can find fire logs beside it


def _provenance(res, arm):
    """What a figure reader needs to know a bar's geometry and recipe without the raw run.

    ``recipe_resolved`` is the arm's own echo of ``resolve_serving_config`` (None for the
    bf16 baselines). ``block_size`` comes from the run config when the bench recorded it;
    older runs only prove it through the ``PAGE_GRID`` sentinel line, which is read from
    the route-evidence files when they are reachable, else reported as unknown rather
    than guessed.
    """
    pv = (res.get("provenance") or {}).get(arm) or {}
    fair = ((res.get("config") or {}).get("fair") or {})
    if isinstance(fair, str):
        fair = {}
    block = fair.get("block_size")
    grid = None
    for f in ((res.get("route_evidence") or {}).get(arm) or {}).get("files") or []:
        # the run's absolute path first, then a copy placed next to the JSON (a run
        # pulled from another machine keeps its fire logs beside its _e2e.json)
        local = os.path.join(os.path.dirname(os.path.abspath(_INPUT_PATH[0] or ".")), os.path.basename(f)) if _INPUT_PATH[0] else None
        try:
            src = f if os.path.exists(f) else (local if local and os.path.exists(local) else None)
            if src is None:
                continue
            for line in open(src):
                if line.startswith("PAGE_GRID"):
                    grid = line.strip()
        except OSError:
            continue
    if block is None and grid:
        m = re.search(r"vllm_block_size=(\d+)", grid)
        block = int(m.group(1)) if m else None
    if isinstance(block, str) and block.isdigit():
        block = int(block)
    return {"recipe_resolved": pv.get("recipe_resolved"),
            "block_size": block if block is not None else "unknown",
            "page_grid": grid,
            "route_tags": ((res.get("route_evidence") or {}).get(arm) or {}).get("other_tags")}


def _cell(arms, arm, ctx, field="tpot_p50"):
    c = (arms.get(arm) or {}).get(f"b1_ctx{ctx}") or {}
    return c.get(field) if c.get("ok") else None


def _r(v, nd=6):
    """round() that passes null through (a missing measurement stays missing, never 0.0)."""
    return None if v is None else round(v, nd)


def _total_bar(arms, ctxs, arm):
    """{"<ctx>": {tpot_ms, ttft_ms, peak_gb}} for a single-total-bar arm."""
    out = {}
    for c in ctxs:
        t = _cell(arms, arm, c)
        if t is not None:
            out[str(c)] = {"tpot_ms": round(t, 4),
                           "ttft_ms": _cell(arms, arm, c, "ttft_p50"),
                           "peak_gb": None}
    return out


def _resolve_flash_arm(arms, names):
    """First arm in `names` that exists in the bench JSON with at least one ok cell."""
    for n in names:
        cells = arms.get(n) or {}
        if any(v.get("ok") for v in cells.values() if isinstance(v, dict)):
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-vllm-bf16", required=True,
                    help="output path for the TRITON_ATTN arm (payload method=vllm_triton; "
                         "the filename is kept for figure/collect.py's FILE2KEY)")
    ap.add_argument("--out-ommx-vllm", required=True)
    ap.add_argument("--out-turboquant", default="", help="TURBOQUANT arm -> turboquant_vllm.json")
    ap.add_argument("--flash-arm", default="",
                    help="output path for vLLM's DEFAULT FlashAttention baseline arm "
                         "(method=vllm_flash_attn). Empty = do not emit. Requires the bench "
                         "to have run the FA (or FA3) arm; missing arm = hard error, never a "
                         "silently skipped file.")
    ap.add_argument("--flash-arm-name", default=",".join(FLASH_ARM_NAMES),
                    help="comma list of candidate arm names for --flash-arm (first match wins)")
    args = ap.parse_args()
    _INPUT_PATH[0] = args.inp
    with open(args.inp) as fh:
        res = json.load(fh)
    arms = res.get("arms", {})
    cfg = res.get("config") or {}
    ctxs = cfg.get("cmp_ctxs") or []

    # REFUSE A ZERO-CELL ADAPTATION (law: no silent fallback). `ctxs` drives EVERY output
    # file, so a bench JSON with no config.cmp_ctxs used to produce four well-formed-looking
    # JSONs carrying `"ctxs": {}`, print `E2E_ADAPT_DONE ...=0 ...` and exit 0 -- a
    # successful-looking adaptation of nothing. The FLASH-arm hard error below inspects only
    # `arms`, so it never caught this; collect.py surfaces the consequence one stage later,
    # by which point the adapter has already reported success.
    if not ctxs:
        raise SystemExit(
            f"{args.inp} has no config.cmp_ctxs, so there is nothing to adapt: every output "
            f"would be written with an empty 'ctxs' map and this run would report success "
            f"for zero cells.\n"
            f"  config keys present: {sorted(cfg)}\n"
            f"  arms present: {sorted(arms)}\n"
            f"FIX: re-run bench_e2e_a100 (its main() writes config.cmp_ctxs from --cmp-ctxs); "
            f"a hand-built or truncated bench JSON must carry that key.")
    if not arms:
        raise SystemExit(
            f"{args.inp} has no 'arms', so no cell can be adapted at any of ctxs={ctxs}.\n"
            f"FIX: re-run bench_e2e_a100 -- a sweep in which every arm died still writes an "
            f"'arms' map, so an absent one means this file is not a bench result.")

    # Resolve the optional FLASH arm BEFORE writing anything, so an unsatisfiable request
    # fails cleanly instead of half-way through the output set (law: no silent fallback).
    flash_arm = None
    if args.flash_arm:
        cand = [n.strip() for n in args.flash_arm_name.split(",") if n.strip()]
        flash_arm = _resolve_flash_arm(arms, cand)
        if flash_arm is None:
            raise SystemExit(
                f"--flash-arm {args.flash_arm!r} requested but {args.inp} has no FLASH_ATTN "
                f"arm with an ok cell (looked for {cand}; arms present: {sorted(arms)}).\n"
                f"FIX: re-run bench_e2e_a100 with the FA arm selected, e.g.\n"
                f"  --only-arms 'TRITON,FA,abl_attn,abl_attn_no_all,abl_attn_no_unpack,"
                f"abl_attn_no_dequant,abl_attn_skipwrite'\n"
                f"or drop --flash-arm to emit only the Triton baseline.")

    # ── bf16-weights baseline = the TRITON_ATTN arm (NOT vLLM's default FlashAttention) ──
    vb = {"ctxs": _total_bar(arms, ctxs, "TRITON")}
    with open(args.out_vllm_bf16, "w") as fh:
        json.dump({"method": "vllm_triton", "provenance": _provenance(res, "TRITON"),
                   # figure/collect.py + figure/plot.py still key this file as "vllm_bf16";
                   # keep the alias so the legacy key resolves while the payload stays honest.
                   "method_alias": "vllm_bf16",
                   "arm": "TRITON",
                   "attention_backend": "TRITON_ATTN",
                   "weights": "bf16",
                   "note": "vLLM engine, bf16 weights, TRITON_ATTN backend. vLLM's DEFAULT "
                           "backend is FLASH_ATTN -- see --flash-arm for that baseline.",
                   **vb}, fh, indent=2)

    # ── OMMX vLLM total + breakdown ──
    # The OMMX serving-kernel bar is the OMMX paged-decode ATTENTION path (abl_attn: bf16
    # weights + OMMX KV) -- directly comparable to ommx_hf and the exact thing the segments
    # decompose (they sum to it). Fall back to the full OMMX arm only if abl_attn is missing.
    meta = {
        # the four keys the honesty contract mandates
        "method": "differential-ablation",
        "clamped_negative": True,
        "renormalized_to_total": True,
        "residual_bucket": "outlier",
        # how to read the rest
        "segments": list(SEGMENTS),
        "full_arm": FULL_ARM,
        "base_arm": BASE_ARM,
        "diff_arms": dict(DIFF_ARMS),
        "note": ("breakdown = plotted segments (missing arm -> 0.0, negative -> 0.0, then "
                 "x renorm_factor so the five sum to tpot_ms). breakdown_raw = the same "
                 "differentials RAW: unclamped, unnormalized, null when the arm is missing. "
                 "base/outlier in `breakdown` are derived from the CLAMPED diffs (historical "
                 "recipe, kept so the plot is unchanged); in `breakdown_raw` they are derived "
                 "from the RAW diffs, so the two can disagree whenever a diff was clamped."),
        "renorm_factor": {},       # per ctx: tpot_ms / sum(clamped segments)
        "clamped_sum_ms": {},      # per ctx: sum(clamped segments) BEFORE rescaling
        "clamped_segments": {},    # per ctx: segments whose plotted value != the raw value
        "missing_arms": {},        # per ctx: differential arms with no ok cell
        "arm_tpot_ms": {},         # per ctx: every raw TPOT the decomposition was built from
        "source_arm": {},          # per ctx: which arm supplied tpot_ms/ttft_ms
        "no_breakdown": {},        # per ctx: why the segments are absent (plain total bar)
    }
    ov = {"ctxs": {},
          "breakdown": {k: {} for k in SEGMENTS},
          "breakdown_raw": {k: {} for k in SEGMENTS}}
    for c in ctxs:
        full = _cell(arms, FULL_ARM, c)
        src = FULL_ARM if full is not None else "OMMX"
        total = full if full is not None else _cell(arms, "OMMX", c)
        if total is None:
            continue
        ov["ctxs"][str(c)] = {"tpot_ms": round(total, 4),
                              "ttft_ms": _cell(arms, src, c, "ttft_p50"),
                              "peak_gb": None}
        meta["source_arm"][str(c)] = src
        base_arm = _cell(arms, BASE_ARM, c)
        if full is None or base_arm is None:
            # no breakdown for this ctx -> plot draws a plain total bar
            meta["no_breakdown"][str(c)] = (
                f"{FULL_ARM} missing" if full is None else f"{BASE_ARM} missing")
            continue

        # ── RAW differentials: no clamp, no renorm, missing arm = null (NOT 0.0) ──
        raw = {}
        missing = []
        for seg, arm in DIFF_ARMS.items():
            v = _cell(arms, arm, c)
            if v is None:
                missing.append(arm)
                raw[seg] = None
            else:
                raw[seg] = full - v          # may be NEGATIVE (cross-process run noise)
        raw["base"] = None if raw["pack"] is None else base_arm - raw["pack"]
        raw["outlier"] = (None if (raw["unpack"] is None or raw["scale"] is None)
                          else full - base_arm - raw["unpack"] - raw["scale"])

        # ── PLOT segments: the historical recipe, byte-for-byte unchanged ──
        # (missing -> 0.0, clamp each diff at 0.0, base/outlier built from the CLAMPED diffs)
        cd = {seg: (0.0 if raw[seg] is None else max(0.0, raw[seg])) for seg in DIFF_ARMS}
        seg = {
            "unpack": cd["unpack"], "scale": cd["scale"], "pack": cd["pack"],
            "outlier": max(0.0, full - base_arm - cd["unpack"] - cd["scale"]),
            "base": max(0.0, base_arm - cd["pack"]),
        }
        s = sum(seg.values())
        # s == 0 reproduces the old `sum(...) or 1.0` guard: every segment plots as 0.0.
        factor = (total / s) if s > 0 else 0.0
        for k in SEGMENTS:
            ov["breakdown"][k][str(c)] = round(seg[k] * factor, 4)
            ov["breakdown_raw"][k][str(c)] = _r(raw[k])
        meta["renorm_factor"][str(c)] = round(factor, 6)
        meta["clamped_sum_ms"][str(c)] = round(s, 6)
        # a segment is flagged when the pre-renorm plotted value differs from the raw one:
        # covers substituted-missing, clamped-negative, and clamp-propagated base/outlier.
        meta["clamped_segments"][str(c)] = [
            k for k in SEGMENTS
            if raw[k] is None or abs(seg[k] - raw[k]) > 1e-9]
        if missing:
            meta["missing_arms"][str(c)] = missing
        meta["arm_tpot_ms"][str(c)] = {
            FULL_ARM: _r(full), BASE_ARM: _r(base_arm),
            **{arm: _r(_cell(arms, arm, c)) for arm in DIFF_ARMS.values()}}
    # PARTIAL COVERAGE MUST TRAVEL WITH THE BAR. Some tags are informational by contract --
    # they record that part of the model ran bf16 while the arm still passes. SW_BYPASS_BF16*
    # is the live one: sliding-window layers are never OMMX-routed, so a Mistral or Gemma-2
    # run under CUSTOM publishes an OMMX bar whose SWA layers were bf16 FlashAttention. That
    # is a deliberate carve-out, not a failure, and forcing ok=False would be wrong -- but
    # until now the fact stopped at the orchestrator's stdout and the figure JSON looked
    # identical to a fully-OMMX one. It does not any more.
    _ev = (res.get("route_evidence") or {}).get(FULL_ARM) or {}
    _other = list(_ev.get("other_tags") or [])
    _partial = sorted(t for t in _other if t.startswith("SW_BYPASS_BF16")
                      or t.startswith("DECODE_BF16_UNROUTED"))
    with open(args.out_ommx_vllm, "w") as fh:
        json.dump({"method": "ommx_vllm", "arm": FULL_ARM, "provenance": _provenance(res, FULL_ARM),
                   "attention_backend": "CUSTOM", "weights": "bf16",
                   "partial_bf16_coverage": _partial,
                   "informational_tags": _other,
                   **ov, "breakdown_meta": meta}, fh, indent=2)
    if _partial:
        print(f"E2E_ADAPT_PARTIAL_COVERAGE {FULL_ARM}: {_partial} -- part of the model ran "
              f"bf16 while this arm passed; the bar is NOT fully OMMX and its JSON now says so")

    # ── TurboQuant (3-bit KV) arm — a single total-TPOT bar, KV-quant peer of OMMX(vLLM) ──
    ntq = 0
    if args.out_turboquant:
        tq = {"ctxs": _total_bar(arms, ctxs, "TURBOQUANT")}
        ntq = len(tq["ctxs"])
        with open(args.out_turboquant, "w") as fh:
            json.dump({"method": "turboquant_vllm", "arm": "TURBOQUANT", "weights": "bf16", "provenance": _provenance(res, "TURBOQUANT"),
                       "kv_cache_dtype": "turboquant_3bit_nc", **tq}, fh, indent=2)

    # ── vLLM DEFAULT FlashAttention baseline (opt-in; resolved above) ──
    nfa = 0
    if flash_arm is not None:
        fa = {"ctxs": _total_bar(arms, ctxs, flash_arm)}
        nfa = len(fa["ctxs"])
        with open(args.flash_arm, "w") as fh:
            json.dump({"method": "vllm_flash_attn", "arm": flash_arm, "provenance": _provenance(res, flash_arm),
                       "attention_backend": "FLASH_ATTN", "weights": "bf16",
                       "note": "vLLM's DEFAULT attention backend -- the stock-vLLM baseline.",
                       **fa}, fh, indent=2)
    print(f"E2E_ADAPT_DONE vllm_triton(alias vllm_bf16)={len(vb['ctxs'])} "
          f"ommx_vllm={len(ov['ctxs'])} turboquant={ntq} "
          f"vllm_flash_attn={nfa}{'' if flash_arm is None else f' (arm={flash_arm})'} ctxs")
    # A MANDATORY OUTPUT WITH ZERO CELLS IS NOT AN ERROR, BUT IT MUST BE NAMED. Zero cells
    # here is a legitimate, expected outcome: bench_e2e_a100 forces ok=False on every cell of
    # a CUSTOM arm that could not prove the OMMX route fired, and run.sh depends on
    # ommx_vllm.json EXISTING with 0 ctx cells to assert that the gate held (it deletes the
    # file and fails the leg if cells survived instead). So the file is still written -- but
    # the operator is told which bar will be absent from the figure, by name, rather than
    # having to notice a `=0` inside the DONE line.
    # The two OPT-IN outputs are named on the same terms. An opt-in path that was asked for
    # and produced nothing is exactly as invisible as a mandatory one: --flash-arm resolves
    # on "this arm has an ok cell SOMEWHERE", while the bars are built only from the
    # `b1_ctx<c>` cells at ctxs, so an arm whose ok cells sit at another batch or another
    # ctx resolves, writes a file with `"ctxs": {}`, and reports nfa=0 with no other trace.
    empty_checks = [("vllm_triton (arm TRITON -> vllm_bf16.json)",
                     len(vb["ctxs"]), args.out_vllm_bf16),
                    (f"ommx_vllm (arm {FULL_ARM})", len(ov["ctxs"]), args.out_ommx_vllm)]
    if args.out_turboquant:
        empty_checks.append(("turboquant_vllm (arm TURBOQUANT)", ntq, args.out_turboquant))
    if flash_arm is not None:
        empty_checks.append((f"vllm_flash_attn (arm {flash_arm})", nfa, args.flash_arm))
    for label, n, path in empty_checks:
        if not n:
            print(f"E2E_ADAPT_EMPTY {label}: 0 of {len(ctxs)} ctx(s) had an ok cell -> "
                  f"{path} carries no bar. Either the arm was not in --only-arms, every "
                  f"cell failed/OOMed, or the route-fired gate invalidated them all; read "
                  f"route_evidence in {args.inp}.")


if __name__ == "__main__":
    main()
