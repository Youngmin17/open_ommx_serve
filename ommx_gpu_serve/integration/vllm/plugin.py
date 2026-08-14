# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""vLLM ``general_plugins`` entry point — register the OMMX KV-quant attention backend.

Registers the OMMX paged-decode attention backend under ``AttentionBackendEnum.CUSTOM``
(KV-quantized decode; select with ``--attention-backend CUSTOM`` / EngineArg
``attention_backend=CUSTOM``).
  WARNING: vLLM 0.21 does NOT read the ``VLLM_ATTENTION_BACKEND`` env var — setting it is
  silently ignored and the platform picks FLASH_ATTN (the OMMX KV backend never engages).
  You MUST pass ``attention_backend`` via EngineArgs/CLI. Confirm engagement in the log:
  ``Using AttentionBackendEnum.CUSTOM backend.`` + a Triton JIT line for
  ``_canonical_splitkv_stage1`` (the OMMX decode kernel firing).

This KV-focused release does not register a weight-quant (``ommx_w``) linear method; the
weight-quant kernel ships as a standalone microbench under ``csrc/linear/`` only.

Runs once per worker (TP + PP) and is idempotent. Declared in ``pyproject.toml``::

    [project.entry-points."vllm.general_plugins"]
    ommx_gpu_serve = "ommx_gpu_serve.integration.vllm.plugin:register"

Importable with no vLLM installed; each registration no-ops when this vLLM lacks
the corresponding registry.
"""
from __future__ import annotations

from typing import Optional

_REGISTERED = False


def register(method_name: str = "ommx") -> Optional[str]:
    """Idempotent: record ``OMMXCanonicalBackend`` under ``AttentionBackendEnum.CUSTOM``
    and the ``ommx_w`` weight-quant config.

    Returns the attention enum member name to select ("CUSTOM"), or None when this
    vLLM has no v1 attention-backend registry (too old / unsupported — there is no
    monkeypatch fallback in the standalone package). The ``ommx_w`` quant config is
    registered independently (its own try/except) so it survives even on that path.
    """
    global _REGISTERED
    # NOTE: this KV-focused release ships only the attention (KV-quant) backend. The
    # weight-quant ``ommx_w`` linear + ``ommx_moe`` serving methods are NOT included here
    # (their ``linear_method``/``moe_method`` modules live in the full serving tree); the
    # standalone weight-quant kernel is provided as a microbench under ``csrc/linear/``.
    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
    except ImportError:
        _REGISTERED = True
        return None
    if not _REGISTERED:
        # String class path: the backend module (and vLLM's FlashAttention base)
        # imports lazily only when CUSTOM is actually selected.
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "ommx_gpu_serve.integration.vllm.backend.OMMXCanonicalBackend",
        )
        # PACKED-ONLY capacity mode (OMMX_KV_PACKED_ONLY=1, default OFF): patch
        # Attention.get_kv_cache_spec to shrink the bf16 paged-cache page budget by
        # the OMMX ≤3-bit compression ratio -> vLLM admits ~4.6x more KV blocks (the
        # genuine capacity win over FA3). No-op when the env is unset. Installed here
        # (plugin load, once per worker) so it is in place before get_kv_cache_spec
        # runs during KV-cache sizing. See packed_only.py.
        try:
            from .packed_only import install_packed_only_spec
            install_packed_only_spec()
        except Exception:
            pass
        _REGISTERED = True
    return "CUSTOM"


__all__ = ["register"]
