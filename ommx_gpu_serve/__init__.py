# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""OMMX GPU serving — standalone low-bit attention kernels + framework adapters.

This top-level package re-exports the canonical attention surface so a PyTorch-only
user can ``from ommx_gpu_serve import ommx_pack_kv_canonical_block, ...`` without
reaching into the ``attention`` subpackage. The framework adapters live under
``ommx_gpu_serve.integration.*`` and are imported lazily by their entry points
(vLLM plugin, SGLang factory) — they are NOT imported here so that importing this
package never requires vLLM / SGLang / triton to be installed.

Scope: ATTENTION only. The CUTLASS linear/fusion kernels under ``csrc/`` + ``serve/``
(built by ``build_serve.py``) are a separate, off-limits concern and are not part
of this package surface.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Pure-python / lazy-triton surface (safe to import without a GPU or triton).
from .attention import (
    CANONICAL_GROUP_CHANNELS,
    CANONICAL_GROUP_TOKENS,
    CanonicalAttentionConfig,
    TensorQuantSpec,
    default_config,
    dequant_kv_canonical,
    is_available,
    merge_attention_states,
    ommx_paged_decode,
    ommx_pack_kv_canonical_block,
    ommx_paged_decode_attention_canonical,
)

__all__ = [
    "__version__",
    "CanonicalAttentionConfig",
    "TensorQuantSpec",
    "default_config",
    "ommx_pack_kv_canonical_block",
    "dequant_kv_canonical",
    "ommx_paged_decode",
    "ommx_paged_decode_attention_canonical",
    "merge_attention_states",
    "is_available",
    "CANONICAL_GROUP_TOKENS",
    "CANONICAL_GROUP_CHANNELS",
]
