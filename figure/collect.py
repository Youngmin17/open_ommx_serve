#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Normalize raw per-method bench JSONs into one figure/data/<gpu>.json the plots read.

Raw inputs (per GPU dir, e.g. figure/data_h200/):
  bf16_hf.json  ommx_hf.json  kivi_hf.json  kitty_hf.json   -> figure/bench.py schema
  vllm_bf16.json                                            -> vLLM-engine bf16 (optional)
  ommx_vllm.json                                            -> bench_e2e breakdown (optional)

All share {"ctxs": {"<ctx>": {"tpot_ms","ttft_ms","peak_gb"}}}. ommx_vllm additionally
carries {"breakdown": {"base|outlier|unpack|scale|pack": {"<ctx>": ms}}}.

Output figure/data/<gpu>.json:
  {"gpu","ctxs":[...],"methods":{<key>:{"tpot":[...],"peak":[...],"breakdown":{seg:[...]}?}}}
Missing method/ctx cells are null so the plot can skip them (no cited fallback).

Usage: python figure/collect.py --src figure/data_h200 --gpu H200 --out figure/data/h200.json
"""
import argparse
import json
import os

# raw filename -> normalized method key
FILE2KEY = {
    "ommx_vllm.json": "ommx_vllm", "ommx_hf.json": "ommx_hf",
    "bf16_hf.json": "bf16_hf", "vllm_bf16.json": "vllm_bf16",
    "kivi_hf.json": "kivi_hf", "kitty_hf.json": "kitty_hf",
    "turboquant_vllm.json": "turboquant_vllm",
}
SEGS = ["base", "outlier", "unpack", "scale", "pack"]


def _cell(ctxs, table, field):
    return [(table.get(str(c)) or {}).get(field) for c in ctxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with raw per-method JSONs")
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--ctxs", default="1024,4096,16384,32768,65536")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ctxs = [int(x) for x in args.ctxs.split(",") if x]

    methods = {}
    for fn, key in FILE2KEY.items():
        p = os.path.join(args.src, fn)
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        table = d.get("ctxs") or {}
        entry = {"tpot": _cell(ctxs, table, "tpot_ms"),
                 "peak": _cell(ctxs, table, "peak_gb")}
        bd = d.get("breakdown")
        if bd:
            entry["breakdown"] = {s: [(bd.get(s) or {}).get(str(c)) for c in ctxs] for s in SEGS}
        methods[key] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"gpu": args.gpu, "ctxs": ctxs, "methods": methods}, fh, indent=2)
    print(f"COLLECT_DONE {args.out}  methods={sorted(methods)}")


if __name__ == "__main__":
    main()
