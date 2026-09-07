#!/usr/bin/env python3
# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""kivi_hf.json <- per context, the faster of KIVI's two decode paths.

    python figure/merge_kivi_paths.py figure/data_h200_final

Inputs, both written by figure/bench.py --method kivi and left in place as evidence:
  kivi_hf_dequant.json   the dequantize-and-dense path (KIVI_FUSED_GQA unset)
  kivi_hf_fused.json     the official CUDA kernel (--kivi-gqa-mode official)

The merged file takes every top-level field (arm_label, kivi_path, meta.kivi_env, ...) from the
path that won at EVERY context, so its self-description matches its numbers; if the winner
differs by context the file says "mixed-per-ctx" and figure/collect.py refuses it. The choice
per context is kept in kivi_path_per_ctx.
"""
import json
import os
import sys


def merge(d):
    deq = json.load(open(os.path.join(d, "kivi_hf_dequant.json")))
    fus = json.load(open(os.path.join(d, "kivi_hf_fused.json")))
    per_ctx, cells = {}, {}
    for c, cell in deq["ctxs"].items():
        f = (fus.get("ctxs") or {}).get(c)
        f_ok = bool(f) and isinstance(f.get("tpot_ms"), (int, float)) and not f.get("error")
        d_ok = isinstance(cell.get("tpot_ms"), (int, float)) and not cell.get("error")
        if f_ok and (not d_ok or f["tpot_ms"] < cell["tpot_ms"]):
            cells[c] = f
            per_ctx[c] = {"path": "fused", "dequant_ms": cell.get("tpot_ms") if d_ok else None,
                          "fused_ms": f["tpot_ms"]}
        else:
            cells[c] = cell
            per_ctx[c] = {"path": "dequant", "dequant_ms": cell.get("tpot_ms") if d_ok else None,
                          "fused_ms": f["tpot_ms"] if f_ok else None}
    winners = {v["path"] for v in per_ctx.values()}
    base = json.loads(json.dumps(fus if winners == {"fused"} else deq))
    base["ctxs"] = cells
    if winners == {"fused"}:
        kp = base.get("kivi_path") or {}
        gs = ((base.get("meta") or {}).get("kivi_recipe") or {}).get("group_size")
        base["arm_label"] = f"kivi (fp16, {kp.get('gqa_mode', 'fused')}, gs{gs})"
    if len(winners) > 1:
        base["kivi_path"] = {"path": "mixed-per-ctx", "detail": per_ctx}
        base["arm_label"] = "kivi (fp16, mixed paths per ctx)"
    base["kivi_path_per_ctx"] = per_ctx
    base["kivi_path_policy"] = ("per context, the faster of KIVI's dequantize-and-dense path and its official "
                                "CUDA kernel; originals: kivi_hf_dequant.json, kivi_hf_fused.json")
    json.dump(base, open(os.path.join(d, "kivi_hf.json"), "w"), indent=2)
    return per_ctx, base.get("arm_label"), (base.get("kivi_path") or {}).get("path")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        per_ctx, label, path = merge(d)
        print(d, label, path, {c: v["path"] for c, v in per_ctx.items()})
