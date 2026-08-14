# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""K-outlier budget redistribution for the fakequant KV path — leaf module (torch + stdlib only).

Env gate: ``OMMX_KV_OUTLIER_PROFILE=<path.json>``. Unset => every helper returns None and
quant_attention's OFF path passes the exact same kwargs as before (byte-identical).

Profile JSON schema (written by fakesparse/build_outlier_profile.py):

    {
      "allocation":     {"<layer>": {"K_opg": int}},   # per-layer K outliers_per_group
      "K_opg_per_head": {"<layer>": [h0, h1, ...]},    # optional per-KV-head override (len == n_kv_heads)
      "default_K_opg":  int                            # optional fallback for unlisted layers
    }

Semantics: when the env is set, K quantization is routed through the DECODE-branch recipe of
_apply_group_quantization_vectorized (INT2 base + per-128-token-pair FP4 outliers) with the
layer's (or head's) opg — in BOTH prefill and decode forwards. Rationale: the stock prefill
branch is pure FP4 min/max group quant and IGNORES outliers_per_group entirely (empirically
verified: opg=0/1/2 prefill outputs are bit-identical), so a wikitext-PPL path (prefill-only
forwards) would be flat across any outlier-budget arm without this routing. V is never touched
(quantization-axis isolation, CLAUDE.md law 13).

opg=0 is a VALID value (no outliers). Ints are parsed directly — never clamped with max(1, .)
(law 11: a clamp would silently turn OFF(0) into ON(1)).
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

import torch

ENV_PROFILE = "OMMX_KV_OUTLIER_PROFILE"

# path -> (mtime, parsed) — one JSON read per process even with 32 layer instances resolving.
_PROFILE_CACHE: Dict[str, tuple] = {}


def _parse(raw: dict) -> dict:
    """Validate + normalize the profile JSON. Loud on malformed input (no silent fallback)."""
    alloc = {}
    for k, v in (raw.get("allocation") or {}).items():
        opg = int(v["K_opg"])
        if opg < 0:
            raise ValueError(f"{ENV_PROFILE}: allocation[{k}].K_opg={opg} < 0")
        alloc[int(k)] = opg
    per_head = {}
    for k, v in (raw.get("K_opg_per_head") or {}).items():
        hs = [int(x) for x in v]
        if any(h < 0 for h in hs):
            raise ValueError(f"{ENV_PROFILE}: K_opg_per_head[{k}] has a negative opg: {hs}")
        per_head[int(k)] = hs
    default = raw.get("default_K_opg")
    if default is not None:
        default = int(default)
        if default < 0:
            raise ValueError(f"{ENV_PROFILE}: default_K_opg={default} < 0")
    return {"alloc": alloc, "per_head": per_head, "default": default}


def load_profile() -> Optional[dict]:
    """Parsed profile dict, or None when the env var is unset/empty (the OFF path).

    A set-but-missing/corrupt file raises (silent-fallback rule): an experiment arm must never
    quietly run as the baseline because its profile path was wrong.
    """
    path = os.environ.get(ENV_PROFILE, "")
    if not path:
        return None
    mtime = os.path.getmtime(path)  # FileNotFoundError on a bad path — intentional
    hit = _PROFILE_CACHE.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    with open(path) as f:
        prof = _parse(json.load(f))
    _PROFILE_CACHE[path] = (mtime, prof)
    return prof


def resolve_layer(profile: Optional[dict], layer_idx: Optional[int], fallback_opg: int) -> Optional[dict]:
    """Per-layer plan {'opg': int, 'per_head': Optional[List[int]]}, or None when profile is None.

    Precedence: K_opg_per_head[layer] (kept alongside the scalar) > allocation[layer].K_opg >
    default_K_opg > fallback_opg (the module's uniform self.outliers_per_group). Note: when the
    profile env IS set, even unlisted layers get a plan (with the fallback opg) so every layer's
    K runs through the same decode recipe — only the budget varies across arms.
    """
    if profile is None:
        return None
    li = int(layer_idx) if layer_idx is not None else -1
    opg = profile["alloc"].get(li)
    if opg is None:
        opg = profile["default"] if profile["default"] is not None else int(fallback_opg)
    return {"opg": int(opg), "per_head": profile["per_head"].get(li)}


def plan_max_opg(plan: Optional[dict]) -> int:
    """Largest opg this layer can use — drives the 128-wide processing-unit/cache boundary."""
    if plan is None:
        return 0
    ph = plan.get("per_head")
    return max(ph) if ph else int(plan["opg"])


def quantize_k_with_plan(k: torch.Tensor, plan: dict,
                         quant_call: Callable[[torch.Tensor, int], torch.Tensor]) -> torch.Tensor:
    """Quantize K [B, Hkv, T, D] under a per-layer/per-head opg plan.

    ``quant_call(tensor, opg)`` is the decode-recipe attention_quantizer closure. Per-head mode
    slices KV heads (dim=1) into equal-opg groups: key grouping runs along the token axis per
    (b, h, d) lane and the sink split is along dim=2, so head slicing is bit-identical to a
    full-tensor call at that opg (verified by test_kv_outlier_per_layer_and_per_head_quant).
    """
    per_head = plan.get("per_head")
    if per_head is None:
        return quant_call(k, int(plan["opg"]))
    n_heads = k.shape[1]
    if len(per_head) != n_heads:
        raise ValueError(
            f"{ENV_PROFILE}: K_opg_per_head has {len(per_head)} entries but K has {n_heads} kv heads")
    out = torch.empty_like(k)
    for opg in sorted(set(per_head)):
        idx = [h for h, o in enumerate(per_head) if o == opg]
        out[:, idx] = quant_call(k[:, idx].contiguous(), int(opg))
    return out
