#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Normalize raw per-method bench JSONs into one figure/data/<gpu>.json the plots read.

Raw inputs (per GPU dir, e.g. figure/data_h200/):
  bf16_hf.json  ommx_hf.json  kivi_hf.json  kitty_hf.json   -> figure/bench.py schema
  vllm_bf16.json                                            -> vLLM engine, TRITON_ATTN arm
  vllm_flash_attn.json                                      -> vLLM engine, FLASH_ATTN arm
                                                               (vLLM's DEFAULT backend)
  ommx_vllm.json                                            -> bench_e2e breakdown (optional)

All share {"ctxs": {"<ctx>": {"tpot_ms","ttft_ms","peak_gb"}}}. ommx_vllm additionally
carries {"breakdown": {"base|outlier|unpack|scale|pack": {"<ctx>": ms}}} plus the
"breakdown_raw" / "breakdown_meta" honesty pair that e2e_to_figure.py writes next to it.

Output figure/data/<gpu>.json:
  {"gpu","ctxs":[...],"methods":{<key>:{"tpot":[...],"peak":[...],"breakdown":{seg:[...]}?,
                                        "breakdown_raw":{seg:[...]}?,"breakdown_meta":{...}?,
                                        "provenance":{...}}}}
Missing method/ctx cells are null so the plot can skip them (no cited fallback).

The stacked ommx_vllm bar is a differential ATTRIBUTION, not a measurement: negative
differentials are clamped to 0.0, "outlier" is the residual bucket, and the five segments
are rescaled to sum to the measured total. e2e_to_figure.py records exactly that in
"breakdown_meta" (clamped_negative / renormalized_to_total / residual_bucket /
renorm_factor / clamped_segments) and keeps the unclamped, null-preserving differentials in
"breakdown_raw". Both are carried through HERE rather than dropped: figure/data/<gpu>.json
is the only file that survives to the figure, so a disclosure that stops at the raw JSON is
a disclosure nobody plotting or citing the bar will ever see.

"provenance" carries forward the identity fields the raw JSONs record (resolved compute
dtype, arm_label, the payload's own method name, the vLLM arm + attention backend) so the
plot can label a series by what it MEASURED instead of by its filename. The arms do not all
run at the same dtype (KIVI's packer is fp16-only), and a mixed-dtype figure is a fairness
problem that has to be visible in the legend, not only in a README.

Usage: python figure/collect.py --src figure/data_h200 --gpu H200 --out figure/data/h200.json
"""
import argparse
import json
import os

# raw filename -> normalized method key.
# NOTE on the two vLLM bf16 baselines: they are DIFFERENT attention backends of the same
# engine and must stay separate series. "vllm_bf16.json" is written from the TRITON arm
# (e2e_to_figure.py emits it with method="vllm_triton", method_alias="vllm_bf16" -- the
# filename is legacy, the payload is honest); "vllm_flash_attn.json" is the FA arm, i.e.
# vLLM's DEFAULT backend. Without the second entry that file was emitted by the adapter and
# then silently dropped here, so the stock-vLLM baseline never reached a figure.
FILE2KEY = {
    "ommx_vllm.json": "ommx_vllm", "ommx_hf.json": "ommx_hf",
    "bf16_hf.json": "bf16_hf", "vllm_bf16.json": "vllm_bf16",
    "vllm_flash_attn.json": "vllm_flash_attn",
    "kivi_hf.json": "kivi_hf", "kitty_hf.json": "kitty_hf",
    "turboquant_vllm.json": "turboquant_vllm",
    # LMDeploy W4A16KV4 (figure/bench_lmdeploy.py). Its key carries no "_hf"/"_vllm" suffix
    # because it is neither: TurboMind is a third engine, wall-clock timed, with AWQ-INT4
    # WEIGHTS where every other series here is bf16-weight. plot.py spells that out in the
    # label; collect.py only has to not drop it.
    "lmdeploy.json": "lmdeploy",
}
SEGS = ["base", "outlier", "unpack", "scale", "pack"]
# dtype spellings the raw JSONs use -> the short tag the plot prints. figure/bench.py writes
# meta.dtype (the string the caller typed) AND meta.torch_dtype (the resolved torch dtype,
# e.g. "torch.float16"); both are accepted so a file written by either generation resolves.
DTYPE_SHORT = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
               "torch.bfloat16": "bf16", "torch.float16": "fp16",
               "torch.float32": "fp32", "bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}


def _cell(ctxs, table, field):
    return [(table.get(str(c)) or {}).get(field) for c in ctxs]


def _provenance(d):
    """Identity fields of one raw JSON: what this series actually is, not what it is called.

    Every value is read back from the file, never assumed. A field the file does not carry
    stays None -- the pre-provenance JSONs shipped in figure/data_*/ record only
    {"warmup","measure","dtype"}, so for them everything except dtype is None, and the plot
    prints "?" rather than filling in a plausible default.

    dtype: figure/bench.py records the RESOLVED compute dtype under meta.dtype/torch_dtype.
    The vLLM adapter files carry no dtype (the engine is locked to bf16 by bench_e2e_a100's
    fair-compare contract) but do record "weights", so that is used for them.
    """
    meta = d.get("meta") or {}
    dt = meta.get("dtype") or meta.get("torch_dtype") or d.get("weights")
    return {"dtype": DTYPE_SHORT.get(dt, dt),
            "arm_label": d.get("arm_label"),          # bench.py: e.g. "kivi (fp16)"
            "payload_method": d.get("method"),        # what the file calls itself
            "arm": d.get("arm"),                      # bench_e2e arm name (vLLM files)
            "attention_backend": d.get("attention_backend")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with raw per-method JSONs")
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--ctxs", default="1024,4096,16384,32768,65536")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ctxs = [int(x) for x in args.ctxs.split(",") if x]

    # A MISSING --src IS A REFUSAL, NOT AN EMPTY RESULT. --src is derived from run.sh's --tag,
    # so a typo'd tag (or a tag whose bench never ran) used to walk the FILE2KEY map over a
    # directory that does not exist, find nothing, and still write a well-formed
    # {"methods": {}} to --out at rc=0.
    if not os.path.isdir(args.src):
        raise SystemExit("collect.py: --src %r is not a directory. Nothing was collected and "
                         "%r was NOT written or overwritten. Check --tag / --src."
                         % (args.src, args.out))

    methods = {}
    for fn, key in FILE2KEY.items():
        p = os.path.join(args.src, fn)
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        table = d.get("ctxs") or {}
        entry = {"tpot": _cell(ctxs, table, "tpot_ms"),
                 "peak": _cell(ctxs, table, "peak_gb"),
                 "provenance": {"source_file": fn, **_provenance(d)}}
        bd = d.get("breakdown")
        if bd:
            entry["breakdown"] = {s: [(bd.get(s) or {}).get(str(c)) for c in ctxs] for s in SEGS}
            # carry the attribution disclosure with the numbers it describes (see the
            # module docstring): raw aligned to ctxs exactly like `breakdown`, meta verbatim.
            raw = d.get("breakdown_raw")
            if raw:
                entry["breakdown_raw"] = {
                    s: [(raw.get(s) or {}).get(str(c)) for c in ctxs] for s in SEGS}
            meta = d.get("breakdown_meta")
            if meta:
                entry["breakdown_meta"] = meta
        methods[key] = entry

    # ZERO COLLECTABLE FILES IS ALSO A REFUSAL, AND IT MUST HAPPEN BEFORE THE WRITE. --out is a
    # PUBLISHED, git-tracked artifact (figure/data/h200.json, figure/data/a100.json), so writing
    # an empty result over it destroys the shipped normalized data of a run that did happen --
    # the same hazard run.sh's stash_stale exists to prevent, one stage later. It also hands
    # figure/plot.py an empty series set, which is where `./run.sh all` with every leg failed
    # ended up raising a raw traceback instead of saying what went wrong. Refuse instead.
    if not methods:
        raise SystemExit(
            "collect.py: no collectable JSON in %r, so %r was NOT written or overwritten "
            "(an existing one is left intact). Expected one or more of: %s. An empty bench "
            "directory means every leg failed or none was run -- read the bench log."
            % (args.src, args.out, ", ".join(sorted(FILE2KEY))))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"gpu": args.gpu, "ctxs": ctxs, "methods": methods}, fh, indent=2)
    # Print the per-method dtype next to the method list: a mixed-dtype set (the KIVI arm is
    # fp16-only) is a fairness caveat the operator must see at collect time, not discover in
    # the figure. "?" = the raw JSON recorded no dtype at all.
    # A FILE THAT EXISTS IS NOT A MEASUREMENT. e2e_to_figure.py writes vllm_bf16.json /
    # ommx_vllm.json / vllm_flash_attn.json whenever it is invoked, even when every cell of
    # the arm was ok=False (all-OOM, or the route-gate forced them not-ok), in which case
    # "ctxs" is {} and every value here is None. Listing those keys on the COLLECT_DONE line
    # made the log say the arm was collected while the figure silently had no bar for it and
    # run.sh still exited 0. Name them separately instead; they stay IN the JSON (with their
    # provenance) because plot.py already skips all-null series and dropping them would lose
    # the record that the file was read.
    def _has_data(v):
        return any(x is not None for x in v["tpot"]) or any(x is not None for x in v["peak"])
    drawn = sorted(k for k, v in methods.items() if _has_data(v))
    empty = sorted(k for k, v in methods.items() if not _has_data(v))
    dts = {k: (methods[k]["provenance"]["dtype"] or "?") for k in drawn}
    print(f"COLLECT_DONE {args.out}  methods={drawn}  dtypes={dts}")
    if empty:
        print(f"COLLECT_EMPTY {len(empty)} file(s) were read but contain NO measurement "
              f"(every ctx null) and will draw no bar: "
              + ", ".join(f"{k} <- {methods[k]['provenance']['source_file']}" for k in empty)
              + ". That is what an arm whose cells all failed, OOMed, or were forced ok=False "
                "by the route gate looks like -- check the bench log for that arm.")


if __name__ == "__main__":
    main()
