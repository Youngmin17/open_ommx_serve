# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""vLLM v1 attention-backend adapter for the canonical OMMX paged decode.

Public surface:
  * ``register`` — the ``vllm.general_plugins`` entry point (idempotent).
  * ``OMMXServingConfig`` / ``resolve_serving_config`` — env -> recipe resolution
    (pure-python, importable + testable without vLLM).

The backend / metadata-builder classes (``backend``, ``metadata``) import vLLM
internals lazily and are only loaded when ``--attention-backend CUSTOM`` is
selected, so importing this package never requires vLLM.
"""
from __future__ import annotations

from .config import OMMXServingConfig, resolve_serving_config

__all__ = ["register", "OMMXServingConfig", "resolve_serving_config"]


def register(method_name: str = "ommx") -> str | None:
    """vLLM general-plugins entry point. Delegates to ``plugin.register``."""
    from .plugin import register as _register

    return _register(method_name)
