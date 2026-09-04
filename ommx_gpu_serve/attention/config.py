# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Clean config API for the canonical paged-decode attention path.

A single validated dataclass that exposes, per the OMMX KV spec:

  * per-K and per-V FORMAT independently       {i2, i2f4, itf4}
  * per-K and per-V vector_length + vector_axis (K per-channel/seq, V per-token/head_dim)
  * per-vector outlier count + select mode      (outliers_per_vector, signed|abs)
  * model / arch / GQA / context / batch fields that drive the kernel choice
    (split-KV count, sink/recent window, GQA head-pack)

Pure-python, torch/triton-free. The defaults encode the validated production
recipe (vl=32, K=i2f4 relidx7 signed-select, V=i2, sink=8/recent=32/group=32).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── format / axis / select enums (plain str tokens, validated) ───────────────
KV_FORMATS = ("i2", "i2f4", "itf4")
VECTOR_AXES = ("channel", "token")        # "channel" = per-seq, "token" = per-head_dim
OUTLIER_SELECTS = ("abs", "signed")
# "bitmap" is the paper's flat N-bit-per-group mask. It is a real storage encoding
# here -- the pool allocates k_obmp, the packer emits it, and the decode kernel reads
# it -- so TensorQuantSpec must be able to describe it. It was omitted while the
# backend did not forward the plane to the kernel; that wiring now exists.
OUTLIER_REPRS = ("relidx7", "combinadic", "bitmap")

# bits-per-base-code by format (the INT base; FP4 outlier value rides the sidecar)
#   i2 / i2f4 / itf4 — INT2 base (the spec-compliant ≤3-bit CORE path)
_FORMAT_BASE_BITS = {"i2": 2, "i2f4": 2, "itf4": 2}
# whether the format carries an FP outlier sidecar at all
_FORMAT_HAS_OUTLIER = {"i2": False, "i2f4": True, "itf4": True}
# outlier value-code WIDTH (bits) on the sidecar by format:
#   i2f4 / itf4 — FP4 e2m1f (4-bit nibble)
_FORMAT_OVAL_BITS = {"i2": 0, "i2f4": 4, "itf4": 4}


@dataclass
class TensorQuantSpec:
    """Per-tensor (K or V) quantization spec."""
    format: str = "i2f4"                  # {i2, i2f4, itf4}
    vector_length: int = 32               # quant group size along vector_axis
    vector_axis: str = "channel"          # "channel" (per-seq) | "token" (per-head_dim)
    outliers_per_vector: int = 3          # FP outliers per vector (0 for i2)
    outlier_select: str = "abs"           # "abs" (top-|W|) | "signed" (top+bottom)
    outlier_repr: str = "relidx7"         # "relidx7" | "combinadic" (membership-equiv)

    def validate(self) -> None:
        if self.format not in KV_FORMATS:
            raise ValueError(f"format must be one of {KV_FORMATS}; got {self.format!r}")
        if self.vector_axis not in VECTOR_AXES:
            raise ValueError(f"vector_axis must be one of {VECTOR_AXES}; got {self.vector_axis!r}")
        if self.outlier_select not in OUTLIER_SELECTS:
            raise ValueError(f"outlier_select must be one of {OUTLIER_SELECTS}; got {self.outlier_select!r}")
        if self.outlier_repr not in OUTLIER_REPRS:
            raise ValueError(f"outlier_repr must be one of {OUTLIER_REPRS}; got {self.outlier_repr!r}")
        if self.vector_length <= 0 or self.vector_length % 8 != 0:
            raise ValueError(f"vector_length must be a positive multiple of 8; got {self.vector_length}")
        k = int(self.outliers_per_vector)
        if k < 0 or k > self.vector_length:
            raise ValueError(f"outliers_per_vector={k} out of range [0, {self.vector_length}]")
        if not _FORMAT_HAS_OUTLIER[self.format] and k != 0:
            raise ValueError(
                f"format {self.format!r} carries no outlier sidecar; outliers_per_vector must be 0")

    @property
    def base_bits(self) -> int:
        return _FORMAT_BASE_BITS[self.format]

    @property
    def has_outlier(self) -> bool:
        return _FORMAT_HAS_OUTLIER[self.format]

    @property
    def oval_bits(self) -> int:
        """Outlier value-code width: FP4 nibble (4) / none (0)."""
        return _FORMAT_OVAL_BITS[self.format]


@dataclass
class CanonicalAttentionConfig:
    """Full config for the canonical paged-decode attention kernel.

    K defaults to i2f4 (per-channel/seq vector, signed-select FP4 outliers), V to
    i2 (per-token/head_dim vector, no outliers) — the spec-compliant ≤3-bit
    capacity path. The model/arch/GQA/context/batch fields drive the kernel's
    launch heuristics (split-KV count, sink/recent window, GQA head-pack).
    """
    # ── per-tensor quant specs (independent K / V) ──────────────────────────
    k: TensorQuantSpec = field(default_factory=lambda: TensorQuantSpec(
        format="i2f4", vector_length=32, vector_axis="channel",
        outliers_per_vector=3, outlier_select="signed", outlier_repr="relidx7"))
    v: TensorQuantSpec = field(default_factory=lambda: TensorQuantSpec(
        format="i2", vector_length=32, vector_axis="token",
        outliers_per_vector=0, outlier_select="abs", outlier_repr="relidx7"))

    # ── model / arch / GQA fields ───────────────────────────────────────────
    head_dim: int = 128                   # must be a multiple of 32 (64/128 validated)
    n_q_heads: int = 32
    n_kv_heads: int = 8                   # GQA: n_q_heads % n_kv_heads == 0
    sm_arch: int = 90                     # 80 (A100) | 89 (Ada) | 90 (H100/H200)

    # ── context / batch fields (drive split-KV + window) ────────────────────
    max_context: int = 4096
    batch_size: int = 1
    page_size: int = 16                   # paged-KV page (group_tokens=32 must be a multiple is NOT required; page may be < 32)
    sink_window: int = 8                  # leading bf16 sink rows (TAIL buffer)
    recent_window: int = 32               # trailing bf16 recent rows (TAIL buffer)

    # ── kernel launch (None = auto from context/batch) ──────────────────────
    num_kv_splits: Optional[int] = None
    block_h: Optional[int] = None         # GQA head-pack tile (None = auto; >=16 -> tensor-core)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.k.validate()
        self.v.validate()
        if self.head_dim % 32 != 0:
            raise ValueError(f"head_dim must be a multiple of 32; got {self.head_dim}")
        if self.n_kv_heads < 1 or self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_q_heads ({self.n_q_heads}) must be an integer multiple of "
                f"n_kv_heads ({self.n_kv_heads}) for GQA.")
        if self.sm_arch < 80:
            raise ValueError(
                f"sm_arch {self.sm_arch} < 80: this Triton decode needs ldmatrix/cp.async "
                "(sm_80+). sm_70 is fakequant-emulation only.")
        if self.k.vector_axis != "channel":
            raise ValueError("canonical K vector_axis must be 'channel' (per-seq).")
        if self.v.vector_axis != "token":
            raise ValueError("canonical V vector_axis must be 'token' (per-head_dim).")
        if self.sink_window < 0 or self.recent_window < 0:
            raise ValueError("sink_window / recent_window must be non-negative.")

    @property
    def gqa_group_size(self) -> int:
        """Query heads per KV head (the GQA head-pack BLOCK_H lower bound)."""
        return self.n_q_heads // self.n_kv_heads

    @property
    def group_tokens(self) -> int:
        """K token-group scale vector_length (== k.vector_length), in {16,32,64,128}.

        Larger = fewer K scale/zp groups = lower K bit/elem; VL=128 reaches the
        ≤3-bit K target. The kernel reads this group size; pack.py + the stores
        thread it end-to-end."""
        return int(self.k.vector_length)

    @property
    def group_channels(self) -> int:
        """V channel-group scale vector_length (== v.vector_length), in {16,32,64,128}.

        Per-token axis (must divide head_dim)."""
        return int(self.v.vector_length)

    def auto_num_kv_splits(self) -> int:
        """A coarse split-KV count when ``num_kv_splits`` is unset (kernel refines)."""
        if self.num_kv_splits is not None:
            return max(1, int(self.num_kv_splits))
        # more splits for long context, fewer when batch already saturates SMs.
        groups = max(1, self.max_context // 32)
        target = min(128, max(1, groups // max(1, self.batch_size)))
        return target

    def auto_block_h(self) -> int:
        """GQA head-pack tile: >=16 routes Q·K / P·V through tensor-core tl.dot."""
        if self.block_h is not None:
            return int(self.block_h)
        g = self.gqa_group_size
        base = max(8, 1 << (max(1, g - 1)).bit_length())
        tc = (self.batch_size >= 2 and self.batch_size * g >= 4) or (
            self.batch_size == 1 and g >= 4 and self.max_context >= 4096)
        return 16 if (base < 16 and tc) else base


def default_config(**overrides) -> CanonicalAttentionConfig:
    """The validated production config (vl=32, K=i2f4 signed-select, V=i2)."""
    cfg = CanonicalAttentionConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"CanonicalAttentionConfig has no field {key!r}")
        setattr(cfg, key, value)
    cfg.validate()
    return cfg


__all__ = [
    "KV_FORMATS", "VECTOR_AXES", "OUTLIER_SELECTS", "OUTLIER_REPRS",
    "TensorQuantSpec", "CanonicalAttentionConfig", "default_config",
]
