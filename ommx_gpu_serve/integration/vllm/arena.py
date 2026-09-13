# SPDX-License-Identifier: Apache-2.0
"""Exact single-store KV arena layout; no vLLM import or device allocation at import.

The engine byte tensor is the backing storage, not an unused reservation alongside
an independently allocated store. Layouts include packed planes, BF16 residuals,
identity tables and tail buffers. Binding/initialization and B1 scheduler guards
belong to the backend; this module only plans bytes and exposes storage views.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from typing import Any

import torch

from ...attention.kv_store import CanonicalKVStore
from .config import OMMXServingConfig, resolve_serving_config

_ALIGNMENT = 256
_SPEC_CLASS = None


def _align(value: int) -> int:
    return (value + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


@dataclass(frozen=True)
class ArenaTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    nbytes: int


@dataclass(frozen=True)
class KVStoreArenaLayout:
    max_context: int
    block_size: int
    head_dim: int
    n_kv_heads: int
    recipe: tuple[tuple[str, Any], ...]
    tensors: tuple[ArenaTensorSpec, ...]
    payload_bytes: int
    total_store_bytes: int
    required_blocks: int
    page_size_bytes: int

    @property
    def allocation_bytes(self) -> int:
        return self.required_blocks * self.page_size_bytes


def plan_kv_arena(cfg: OMMXServingConfig, block_size: int) -> KVStoreArenaLayout:
    """Inventory an actual meta store, including slack/tables/ring/tails exactly.

    No CUDA tensor or CUDA API is used. The ring must already be enabled through
    the resolved recipe/environment; silently planning a full BF16 history would
    defeat the no-shadow contract. A changed recipe produces a different layout.
    """
    block_size = int(block_size)
    if block_size <= 0 or block_size % 16:
        raise ValueError("KV arena block_size must be a positive multiple of 16")
    if cfg.max_context <= 0 or cfg.n_kv_heads <= 0 or cfg.head_dim <= 0:
        raise ValueError("KV arena context and head geometry must be positive")
    store = CanonicalKVStore(
        cfg.head_dim, cfg.n_kv_heads, cfg.max_context, k_format=cfg.k_format,
        v_format="i2", outliers_per_vector=cfg.outliers_per_vector,
        outlier_select=cfg.outlier_select, outlier_repr=cfg.outlier_repr,
        use_pow2=cfg.use_pow2, kv_outlier_map=cfg.kv_outlier_map,
        window=cfg.window(), group_channels=cfg.group_channels, device="meta")
    if not store.kv_ring:
        raise ValueError("KV arena requires OMMX_KV_RING=1; full BF16 history is not supported")
    tensors = []
    end = payload = 0
    for name, value in vars(store).items():
        if not isinstance(value, torch.Tensor):
            continue
        if not value.is_meta or not value.is_contiguous():
            raise ValueError(f"KV arena requires a contiguous meta inventory: {name}")
        size = value.numel() * value.element_size()
        offset = _align(end)
        tensors.append(ArenaTensorSpec(name, tuple(value.shape), value.dtype, offset, size))
        end = offset + size
        payload += size
    if not tensors:
        raise ValueError("KV arena inventory is empty")
    total = _align(end)
    blocks = (int(cfg.max_context) + block_size - 1) // block_size
    page_bytes = _align((total + blocks - 1) // blocks)
    # Include numerical recipe fields too: equal shape is not an equal codec.
    recipe = tuple((name, getattr(store, name)) for name in (
        "k_format", "v_format", "k", "outlier_select", "outlier_repr", "use_pow2",
        "kv_outlier_map", "kv_int8_scale", "scale_dtype", "gt", "ps", "gc",
        "_ring_sink", "_ring_rec", "tail_cap"))
    return KVStoreArenaLayout(int(cfg.max_context), block_size, int(cfg.head_dim),
                             int(cfg.n_kv_heads), recipe, tuple(tensors), payload,
                             total, blocks, page_bytes)


def resolve_kv_arena_layout(*, head_dim: int, n_kv_heads: int, max_context: int,
                            block_size: int, n_q_heads: int | None = None) -> KVStoreArenaLayout:
    """Resolve the current recipe using per-rank engine head geometry."""
    cfg = resolve_serving_config(head_dim=head_dim, n_kv_heads=n_kv_heads,
                                 n_q_heads=n_q_heads, max_context=max_context)
    return plan_kv_arena(cfg, block_size)


def arena_tensor_views(arena: torch.Tensor,
                       layout: KVStoreArenaLayout) -> dict[str, torch.Tensor]:
    """Return uninitialized aliases, never copies; caller initializes identity tables.

    ``arena`` may include extra scheduler blocks, but may not under-reserve the
    declared maximum context. No shape/dtype conversion is allowed to allocate.
    """
    if arena.dtype not in (torch.uint8, torch.int8) or not arena.is_contiguous():
        raise ValueError("KV arena must be a contiguous byte tensor")
    if arena.storage_offset() % _ALIGNMENT:
        raise ValueError("KV arena storage offset must be 256-byte aligned")
    if arena.numel() < layout.allocation_bytes:
        raise ValueError(f"KV arena needs {layout.allocation_bytes} bytes for max_context="
                         f"{layout.max_context}, got {arena.numel()}")
    if arena.numel() % layout.page_size_bytes:
        raise ValueError("KV arena byte count must be a whole number of declared pages")
    raw = arena.view(torch.uint8).view(-1)
    return {entry.name: raw.narrow(0, entry.offset, entry.nbytes).view(entry.dtype).view(entry.shape)
            for entry in layout.tensors}


def _register_spec_manager(spec_class) -> None:
    # vLLM 0.21 dispatches by exact spec type, not isinstance. Registration must
    # also happen when the scheduler process recreates this class during unpickle.
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager, spec_manager_map)

    if spec_manager_map.get(FullAttentionSpec) is not FullAttentionManager:
        raise RuntimeError("KV arena requires vLLM's standard FullAttentionManager mapping")
    if spec_class in spec_manager_map and spec_manager_map[spec_class] is not FullAttentionManager:
        raise RuntimeError("KV arena scheduler manager registration conflicts with an existing mapping")
    spec_manager_map[spec_class] = FullAttentionManager


def _arena_spec_class():
    global _SPEC_CLASS
    if _SPEC_CLASS is not None:
        _register_spec_manager(_SPEC_CLASS)
        return _SPEC_CLASS
    # Lazy import keeps CPU/meta layout tests independent of vLLM availability.
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    @dataclass(frozen=True, kw_only=True)
    class OMMXArenaSpec(FullAttentionSpec):
        arena_layout: KVStoreArenaLayout

        @property
        def real_page_size_bytes(self) -> int:
            return self.arena_layout.page_size_bytes

        @property
        def page_size_bytes(self) -> int:
            return self.arena_layout.page_size_bytes

        @classmethod
        def merge(cls, specs):
            if not specs or any(type(spec) is not cls or spec != specs[0] for spec in specs):
                raise ValueError("KV arena cannot merge heterogeneous geometry/recipes/layouts")
            return copy.deepcopy(specs[0])

        def copy_with_new_block_size(self, block_size):
            if int(block_size) != self.block_size:
                raise ValueError("KV arena does not support changing scheduler block_size")
            return copy.deepcopy(self)

        def max_memory_usage_bytes(self, vllm_config):
            if int(vllm_config.model_config.max_model_len) != self.arena_layout.max_context:
                raise ValueError("KV arena max_context changed after layout planning")
            parallel = getattr(vllm_config, "parallel_config", None)
            if (getattr(parallel, "decode_context_parallel_size", 1) != 1
                    or getattr(parallel, "prefill_context_parallel_size", 1) != 1):
                raise ValueError("KV arena does not support context parallelism")
            return self.arena_layout.allocation_bytes

    # vLLM serializes specs across workers. Give the lazily created class a stable
    # module name; module __getattr__ recreates it when unpickled in a new process.
    OMMXArenaSpec.__qualname__ = "OMMXArenaSpec"
    OMMXArenaSpec.__module__ = __name__
    _register_spec_manager(OMMXArenaSpec)
    _SPEC_CLASS = OMMXArenaSpec
    globals()["OMMXArenaSpec"] = OMMXArenaSpec
    return OMMXArenaSpec


def __getattr__(name):
    if name == "OMMXArenaSpec":
        return _arena_spec_class()
    raise AttributeError(name)


def make_kv_arena_spec(base_spec, *, max_context: int, n_q_heads: int | None = None):
    """Replace a full BF16/FP16 spec with an exact, engine-owned byte arena spec.

    Real attention head dimensions are retained. The backend must return a 2-D
    (num_blocks, page_size_bytes) byte-cache shape with identity stride order and
    must never pass this storage to the BF16 FlashAttention cache writer/reader.
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    if not isinstance(base_spec, FullAttentionSpec):
        raise ValueError("KV arena requires FullAttentionSpec")
    cls = _arena_spec_class()
    if isinstance(base_spec, cls):
        if base_spec.arena_layout.max_context != int(max_context):
            raise ValueError("KV arena max_context changed after layout planning")
        return base_spec
    if (base_spec.dtype not in (torch.bfloat16, torch.float16)
            or int(base_spec.kv_quant_mode) != 0
            or base_spec.sliding_window is not None
            or base_spec.attention_chunk_size is not None
            or base_spec.head_size_v != base_spec.head_size):
        raise ValueError("KV arena supports unquantized full attention with equal K/V head sizes only")
    layout = resolve_kv_arena_layout(head_dim=base_spec.head_size,
                                    n_kv_heads=base_spec.num_kv_heads,
                                    max_context=max_context, block_size=base_spec.block_size,
                                    n_q_heads=n_q_heads)
    kwargs = {field.name: getattr(base_spec, field.name) for field in fields(FullAttentionSpec)}
    kwargs.update(dtype=torch.uint8, page_size_padded=None, arena_layout=layout)
    return cls(**kwargs)
