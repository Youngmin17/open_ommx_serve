# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Canonical OMMX paged-decode attention (self-contained).

Public surface:
  * ``ommx_paged_decode_attention_canonical`` — the Triton decode kernel entry.
  * ``merge_attention_states`` — FlashInfer-style cross-segment (o, lse) merge.
  * ``ommx_pack_kv_canonical_block`` / ``dequant_kv_canonical`` — KV pack + oracle.
  * ``CanonicalAttentionConfig`` / ``default_config`` — the clean config API.

CPU-importable without triton (the kernels build lazily when triton is present).
"""
from __future__ import annotations

from .config import (
    CanonicalAttentionConfig,
    TensorQuantSpec,
    default_config,
)
from .pack import (
    CANONICAL_GROUP_CHANNELS,
    CANONICAL_GROUP_TOKENS,
    dequant_kv_canonical,
    ommx_pack_kv_canonical_block,
)
from .paged_decode import (
    is_available,
    merge_attention_states,
    ommx_paged_decode_attention_canonical,
)
# Importing reference_op REGISTERS the torch.library custom ops
# (ommx_gpu_serve::paged_decode / ::merge_attention_states). No triton/GPU needed
# to register; the kernel builds lazily on the first real CUDA call.
from .reference_op import ommx_paged_decode

__all__ = [
    "ommx_paged_decode_attention_canonical",
    "ommx_paged_decode",
    "merge_attention_states",
    "is_available",
    "ommx_pack_kv_canonical_block",
    "dequant_kv_canonical",
    "CANONICAL_GROUP_TOKENS",
    "CANONICAL_GROUP_CHANNELS",
    "CanonicalAttentionConfig",
    "TensorQuantSpec",
    "default_config",
]
