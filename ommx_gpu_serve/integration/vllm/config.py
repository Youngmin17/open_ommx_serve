# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Serving-recipe resolution (env -> canonical KV recipe). Pure-python, no vLLM.

Resolves the OMMX KV recipe knobs from environment variables into a
``CanonicalAttentionConfig`` + ``WindowSpec`` for the vLLM backend. Kept separate
from the backend so it imports + tests on CPU without vLLM / triton.

The defaults encode the validated step1 recipe (K=i2f4 signed 3-of-32 outliers
relidx7, V=i2, sink=8, recent=32, group=32) — the closest canonical match to the
fakequant ``run.sh --phase kv`` reference. Knobs exist to drive the M2 fakequant
gap-table reconciliation (sink/recent alignment; pow2 lives in the packer).

Env vars (all optional). The MANDATE-named knobs (OMMX_KV_*) are the canonical
spellings; the OMMX_ATTN_* names are kept as back-compat aliases (the OMMX_KV_*
form wins when both are set):

  OMMX_KV_GROUP_TOKENS      K token-group vector_length {16,32,64,128}  (default 32)
                            (alias: OMMX_ATTN_GROUP_TOKENS). Set 64 to amortize the
                            bf16-zp 16b over 64 tokens (0.25b -> 0.0625b/weight) so K hits
                            <=3 bit/weight (the ≤3-bit KV lever; combine with OMMX_KV_OUTLIER_MAP=0).
  OMMX_KV_GROUP_CHANNELS    V channel-group vector_length {16,32,64,128} (default 32)
                            (alias: OMMX_ATTN_GROUP_CHANNELS; must divide head_dim). 64 -> V
                            ~2.4 bit/weight.
  OMMX_KV_OUTLIER_MAP       1 (dedicated FP4 range-map, default) | 0 (BASE-SHARED outlier: the
                            outlier rides the base scale/zp, dropping the per-group FP4 map's
                            16b mapscale+16b mapcenter -> K outlier 0.375b not the mapped cost,
                            the second half of the ≤3-bit KV lever). pack.py reads this too;
                            the config carries it so the recipe is explicit + gate-able.
  OMMX_KV_SINK              sink tokens kept bf16          (default 8)
                            (alias: OMMX_ATTN_SINK)
  OMMX_KV_RECENT            recent window kept bf16        (default 32)
                            (alias: OMMX_ATTN_RECENT)
  OMMX_OUTLIER_PERCENT      K outlier fraction per group -> npv = round(gt*pct)
                            (overrides OMMX_ATTN_OUTLIERS when set; default unset)
  OMMX_ATTN_OUTLIERS        K outliers per group (absolute npv)  (default 3)
  OMMX_ATTN_K_FORMAT        i2f4 | itf4                    (default i2f4)
  OMMX_ATTN_OUTLIER_SELECT  signed | abs                   (default signed)
  OMMX_ATTN_OUTLIER_REPR    relidx7 | combinadic           (default relidx7)
  OMMX_ATTN_COMBINADIC_READ 0 | 1 (in-kernel rank unrank)  (default 0)
  OMMX_STRICT               0 | 1 (re-raise instead of bf16 fallback) (default 0)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from ommx_gpu_serve.attention.config import (
    CanonicalAttentionConfig,
    TensorQuantSpec,
)
from ommx_gpu_serve.attention.kv_window import WindowSpec

GROUP_TOKENS = 32


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return str(default)
    return str(raw).strip().lower()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "off", "no"}


def _env_present(name: str) -> bool:
    raw = os.environ.get(name)
    return raw is not None and str(raw).strip() != ""


def _env_int_alias(canonical: str, alias: str, default: int) -> int:
    """Read ``canonical`` (the mandate spelling) first, then ``alias`` (back-compat),
    then ``default``. The canonical name wins when both are set."""
    if _env_present(canonical):
        return _env_int(canonical, default)
    return _env_int(alias, default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(raw)


@dataclass
class OMMXServingConfig:
    """Resolved serving recipe: format/window knobs + per-model geometry."""

    k_format: str = "i2f4"
    outliers_per_vector: int = 3
    outlier_select: str = "signed"
    outlier_repr: str = "relidx7"
    use_pow2: bool = False
    combinadic_read: bool = False
    sink_tokens: int = 8
    recent_window: int = 32
    group_tokens: int = GROUP_TOKENS         # K token-group vector_length {16,32,64,128}
    group_channels: int = GROUP_TOKENS       # V channel-group vector_length {16,32,64,128}
    # K outlier value codec: True = dedicated FP4 range-map (per-group mapscale/mapcenter, current
    # default); False = BASE-SHARED outlier (outlier rides the base scale/zp -> drops the 2x bf16
    # map params, the ≤3-bit KV lever). pack.py honours kv_outlier_map=; we mirror the env here.
    kv_outlier_map: bool = True
    strict: bool = False
    # per-model geometry (filled from the vLLM model config at backend init).
    head_dim: int = 128
    n_q_heads: int = 32
    n_kv_heads: int = 8
    page_size: int = 16
    max_context: int = 4096
    sm_arch: int = 90
    extra: dict = field(default_factory=dict)

    def window(self) -> WindowSpec:
        return WindowSpec(sink_tokens=self.sink_tokens,
                          recent_window=self.recent_window,
                          group_tokens=self.group_tokens,
                          page_size=self.page_size)

    def attention_config(self) -> CanonicalAttentionConfig:
        """Build the validated ``CanonicalAttentionConfig`` for this recipe."""
        has_outlier = self.k_format != "i2"
        k = TensorQuantSpec(
            format=self.k_format, vector_length=self.group_tokens,
            vector_axis="channel",
            outliers_per_vector=self.outliers_per_vector if has_outlier else 0,
            outlier_select=self.outlier_select, outlier_repr=self.outlier_repr)
        v = TensorQuantSpec(
            format="i2", vector_length=self.group_channels, vector_axis="token",
            outliers_per_vector=0, outlier_select="abs",
            outlier_repr=self.outlier_repr)
        return CanonicalAttentionConfig(
            k=k, v=v, head_dim=self.head_dim, n_q_heads=self.n_q_heads,
            n_kv_heads=self.n_kv_heads, sm_arch=self.sm_arch,
            max_context=self.max_context, batch_size=1,
            page_size=self.page_size, sink_window=self.sink_tokens,
            recent_window=self.recent_window)


def resolve_serving_config(
    *,
    head_dim: Optional[int] = None,
    n_q_heads: Optional[int] = None,
    n_kv_heads: Optional[int] = None,
    page_size: Optional[int] = None,
    max_context: Optional[int] = None,
    sm_arch: Optional[int] = None,
) -> OMMXServingConfig:
    """Read the env knobs (+ model geometry overrides) into an OMMXServingConfig."""
    group_tokens = _env_int_alias("OMMX_KV_GROUP_TOKENS", "OMMX_ATTN_GROUP_TOKENS",
                                  GROUP_TOKENS)
    group_channels = _env_int_alias("OMMX_KV_GROUP_CHANNELS", "OMMX_ATTN_GROUP_CHANNELS",
                                    GROUP_TOKENS)
    # outlier count: OMMX_OUTLIER_PERCENT (fraction of the K token-group) wins when set,
    # else the absolute OMMX_ATTN_OUTLIERS npv. round to nearest, floor 1 for pct>0.
    if _env_present("OMMX_OUTLIER_PERCENT"):
        pct = _env_float("OMMX_OUTLIER_PERCENT", 0.0)
        npv = max(1, int(round(group_tokens * pct))) if pct > 0 else 0
    else:
        npv = _env_int("OMMX_ATTN_OUTLIERS", 3)
    cfg = OMMXServingConfig(
        k_format=_env_str("OMMX_ATTN_K_FORMAT", "i2f4"),
        outliers_per_vector=npv,
        outlier_select=_env_str("OMMX_ATTN_OUTLIER_SELECT", "signed"),
        outlier_repr=_env_str("OMMX_ATTN_OUTLIER_REPR", "relidx7"),
        use_pow2=_env_bool("OMMX_ATTN_POW2", False),
        combinadic_read=_env_bool("OMMX_ATTN_COMBINADIC_READ", False),
        sink_tokens=_env_int_alias("OMMX_KV_SINK", "OMMX_ATTN_SINK", 8),
        recent_window=_env_int_alias("OMMX_KV_RECENT", "OMMX_ATTN_RECENT", 32),
        group_tokens=group_tokens,
        group_channels=group_channels,
        # default ON (dedicated FP4 map = current recipe); OMMX_KV_OUTLIER_MAP=0 -> base-shared
        # outlier (the ≤3-bit KV lever). pack.py reads the same env; carrying it on the config
        # makes the recipe explicit so the backend can pass kv_outlier_map=cfg.kv_outlier_map.
        kv_outlier_map=_env_bool("OMMX_KV_OUTLIER_MAP", True),
        strict=_env_bool("OMMX_STRICT", False),
        max_context=_env_int("OMMX_ATTN_MAXCTX", 4096),
    )
    if head_dim is not None:
        cfg.head_dim = int(head_dim)
    if n_q_heads is not None:
        cfg.n_q_heads = int(n_q_heads)
    if n_kv_heads is not None:
        cfg.n_kv_heads = int(n_kv_heads)
    if page_size is not None:
        cfg.page_size = int(page_size)
    if max_context is not None:
        cfg.max_context = int(max_context)
    if sm_arch is not None:
        cfg.sm_arch = int(sm_arch)
    if cfg.k_format not in ("i2f4", "itf4"):
        raise ValueError(f"OMMX_ATTN_K_FORMAT must be i2f4|itf4; got {cfg.k_format!r}")
    return cfg


__all__ = ["OMMXServingConfig", "resolve_serving_config", "GROUP_TOKENS"]
