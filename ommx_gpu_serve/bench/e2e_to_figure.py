#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Adapt bench_e2e_a100.py output JSON -> figure/collect.py inputs:
  vllm_bf16.json   (TRITON arm = bf16 vLLM baseline)
  ommx_vllm.json   (OMMX arm total + internal-kernel breakdown, differential decomposition)

Breakdown segments (bottom->top), normalized to the OMMX total bar height — the exact
decomposition behind the paper's stacked OMMX bar:
  unpack = TPOT(abl_attn) - TPOT(abl_attn_no_unpack)          (bit-extract ALU)
  scale  = TPOT(abl_attn) - TPOT(abl_attn_no_dequant)         (scale/zp dequant)
  pack   = TPOT(abl_attn) - TPOT(abl_attn_skipwrite)          (per-step KV pack/write)
  base   = TPOT(abl_attn_no_all) - pack                       (attn math + KV load)
  outlier= TPOT(abl_attn) - base_arm - unpack - scale         (outlier-membership residual)

Usage: python e2e_to_figure.py --in runs/e2e.json --out-vllm-bf16 vllm_bf16.json \
         --out-ommx-vllm ommx_vllm.json
"""
import argparse
import json


def _cell(arms, arm, ctx, field="tpot_p50"):
    c = (arms.get(arm) or {}).get(f"b1_ctx{ctx}") or {}
    return c.get(field) if c.get("ok") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-vllm-bf16", required=True)
    ap.add_argument("--out-ommx-vllm", required=True)
    ap.add_argument("--out-turboquant", default="", help="TURBOQUANT arm -> turboquant_vllm.json")
    args = ap.parse_args()
    with open(args.inp) as fh:
        res = json.load(fh)
    arms = res.get("arms", {})
    ctxs = res.get("config", {}).get("cmp_ctxs") or []

    # vllm bf16 baseline (TRITON attention arm)
    vb = {"ctxs": {}}
    for c in ctxs:
        t = _cell(arms, "TRITON", c)
        if t is not None:
            vb["ctxs"][str(c)] = {"tpot_ms": round(t, 4), "ttft_ms": _cell(arms, "TRITON", c, "ttft_p50"),
                                  "peak_gb": None}
    with open(args.out_vllm_bf16, "w") as fh:
        json.dump({"method": "vllm_bf16", **vb}, fh, indent=2)

    # OMMX vLLM total + breakdown. The full "OMMX" arm (ommx-weight + custom attn) can be
    # absent; the OMMX serving-kernel bar is the OMMX paged-decode ATTENTION path (abl_attn:
    # bf16 weights + OMMX KV), directly comparable to ommx_hf and the exact thing the segments
    # decompose (they sum to it). Fall back to the full OMMX arm only if abl_attn is missing.
    ov = {"ctxs": {}, "breakdown": {k: {} for k in ("base", "outlier", "unpack", "scale", "pack")}}
    for c in ctxs:
        full = _cell(arms, "abl_attn", c)
        total = full if full is not None else _cell(arms, "OMMX", c)
        if total is None:
            continue
        ov["ctxs"][str(c)] = {"tpot_ms": round(total, 4), "ttft_ms": _cell(arms, "abl_attn", c, "ttft_p50"),
                              "peak_gb": None}
        base_arm = _cell(arms, "abl_attn_no_all", c)
        if full is None or base_arm is None:
            continue  # no breakdown for this ctx -> plot draws a plain total bar
        def d(arm):
            v = _cell(arms, arm, c)
            return max(0.0, full - v) if v is not None else 0.0
        unpk = d("abl_attn_no_unpack")
        scal = d("abl_attn_no_dequant")
        pack = d("abl_attn_skipwrite")
        outl = max(0.0, full - base_arm - unpk - scal)
        base = max(0.0, base_arm - pack)
        raw = {"base": base, "outlier": outl, "unpack": unpk, "scale": scal, "pack": pack}
        s = sum(raw.values()) or 1.0
        for k, v in raw.items():
            ov["breakdown"][k][str(c)] = round(v * total / s, 4)
    with open(args.out_ommx_vllm, "w") as fh:
        json.dump({"method": "ommx_vllm", **ov}, fh, indent=2)

    # TurboQuant (3-bit KV) arm — a single total-TPOT bar, KV-quant peer of OMMX(vLLM)
    ntq = 0
    if args.out_turboquant:
        tq = {"ctxs": {}}
        for c in ctxs:
            t = _cell(arms, "TURBOQUANT", c)
            if t is not None:
                tq["ctxs"][str(c)] = {"tpot_ms": round(t, 4),
                                      "ttft_ms": _cell(arms, "TURBOQUANT", c, "ttft_p50"),
                                      "peak_gb": None}
        ntq = len(tq["ctxs"])
        with open(args.out_turboquant, "w") as fh:
            json.dump({"method": "turboquant_vllm", **tq}, fh, indent=2)
    print(f"E2E_ADAPT_DONE vllm_bf16={len(vb['ctxs'])} ommx_vllm={len(ov['ctxs'])} "
          f"turboquant={ntq} ctxs")


if __name__ == "__main__":
    main()
