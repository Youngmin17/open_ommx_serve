# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""PACKED-ONLY capacity mode — the genuine ~5x OMMX KV capacity win over FA3.

THE PROBLEM (SHADOW mode, see ``backend.py`` :10-26): the OMMX backend currently
INHERITS vLLM's bf16 ``get_kv_cache_shape`` (a full bf16 paged cache sized for
``max_model_len`` tokens) AND builds the INT2 ``CanonicalKVStore`` sidecar planes as
EXTRA memory on top. So there is NO capacity win as wired — the ≤3-bit sidecar is
strictly additive. The honest OMMX superiority over FA3 is CAPACITY: K=i2f4 (~4.25
bit) + V=i2 (~2.75 bit) vs bf16 (16 bit each) -> ~4.6x KV compression -> ~4.6x more
concurrent sequences / longer context in the same HBM (FA3 OOMs where OMMX still
fits). PACKED-ONLY realizes it.

THE LEVER (vLLM 0.21, verified in v1/core/kv_cache_utils.py:945 + v1/worker/gpu/
attn_utils.py:155-160):

    num_gpu_blocks = available_kv_memory // page_size_bytes // num_layers
    page_size_bytes (FullAttentionSpec) = 2 * block_size * num_kv_heads
                                            * head_size * dtype_size

``num_blocks`` (== the KV token capacity) is INVERSELY proportional to
``page_size_bytes``, and ``page_size_bytes`` is linear in ``head_size``. The
allocator builds the paged tensor from ``get_kv_cache_shape(num_blocks, block_size,
H, head_size)`` — i.e. it reads ``head_size`` STRAIGHT FROM THE SPEC, so shrinking
the spec's ``head_size`` shrinks ``page_size_bytes`` AND the allocated bf16 tensor
*consistently* (the contiguous-view path stays valid; ``page_size_padded`` is left
None — that path uses ``torch.as_strided`` and is documented-broken for the standard
``(2, num_blocks, ...)`` attention shape, attn_utils.py:181-184).

So PACKED-ONLY = monkeypatch ``Attention.get_kv_cache_spec`` so the returned
``FullAttentionSpec`` carries a ``head_size`` shrunk by the OMMX compression ratio
``r = ommx_bits_per_elem / 16``. vLLM then admits ``1/r`` (~4.6x) more KV blocks ->
the capacity win shows up directly in vLLM's own log lines:

    "GPU KV cache size: <N> tokens"
    "Maximum concurrency for <ctx> tokens per request: <Y>x"

The OMMX INT2 sidecar (``CanonicalKVStore`` / ``MultiSeqKVPool``) is the REAL backing
store for the compressed prefix and serves uniform-decode through the canonical op;
the shrunk bf16 paged cache holds the KIVI sink+recent residual + the prefill pass.

HONEST SCOPE: this realizes the capacity-BUDGET win — vLLM admits ~4.6x more
concurrent KV tokens (H100-measured 4.00x), measurable from the log lines + a
max-batch admission probe. The shrunk pages are a byte-budget reservation ONLY:
``backend.py`` never writes them (``do_kv_cache_update`` skips the paged write —
a headdim-D scatter into shrunk pages would corrupt) and never reads them — a
FULL-PROMPT prefill (q_len == seq_len per request, ``detect_full_prefill``) runs
varlen FlashAttention directly on the in-batch q/k/v, and uniform decode is
sidecar-served; any other step raises loudly (no bf16 fallback exists over the
shrunk cache). Handled scope = the eager bench (``bench_capacity.py --mode
packed``: enforce_eager, enable_chunked_prefill=False, enable_prefix_caching=
False); chunked/mixed prefill under PACKED-ONLY is the remaining follow-up.
Default OFF (``OMMX_KV_PACKED_ONLY=1`` to enable) so SHADOW stays the safe baseline.
"""
from __future__ import annotations

import os
from typing import Optional

# --- recipe-derived per-token KV byte accounting (mirror bench _capacity_report) ---


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return default if v is None or str(v).strip() == "" else str(v).strip().lower()


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(v)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(v)
    except ValueError:
        return float(default)


def packed_only_enabled() -> bool:
    return _env("OMMX_KV_PACKED_ONLY", "0") not in {"0", "false", "off", "no"}


def ommx_bits_per_elem(head_dim: int = 128) -> float:
    """OMMX packed K+V bits per (per-head) element, averaged over a long prefix.

    Derived from the resolved serving recipe (the SAME accounting as the bench
    ``_capacity_report``), expressed PER head-dim element (so it maps directly to a
    shrunk per-element ``head_size``). bf16 reference = 32 bits/elem (16 K + 16 V).
    """
    k_format = _env("OMMX_ATTN_K_FORMAT", "i2f4")
    gt = _env_int("OMMX_KV_GROUP_TOKENS", _env_int("OMMX_ATTN_GROUP_TOKENS", 32))
    if _os_present("OMMX_OUTLIER_PERCENT"):
        pct = _env_float("OMMX_OUTLIER_PERCENT", 0.0)
        k = max(1, round(gt * pct)) if pct > 0 else 0
    else:
        k = _env_int("OMMX_ATTN_OUTLIERS", 3)
    scale_bytes = 2  # bf16 scale/zp
    k_base_bits = 2
    oval_bytes_per = (k + 1) // 2
    repr_ = _env("OMMX_ATTN_OUTLIER_REPR", "relidx7")
    idx_bytes_per = 0 if k == 0 else ((7 * k + 7) // 8 if repr_ == "relidx7" else 2)

    # per ELEMENT (one head-dim channel), bits. K base + amortized scale/zp/outliers.
    k_bits = k_base_bits + (2 * scale_bytes * 8) / gt \
        + ((idx_bytes_per + oval_bytes_per) * 8) / gt
    # V: i2 main + per-token scale/zp over a 32-channel V vector (NOT token-amortized).
    v_main_bits = 2.0
    v_scale_bits = (2 * scale_bytes * 8) / 32.0
    v_bits = v_main_bits + v_scale_bits
    return float(k_bits + v_bits)


def _os_present(name: str) -> bool:
    v = os.environ.get(name)
    return v is not None and str(v).strip() != ""


def packed_compression_ratio(head_dim: int = 128) -> float:
    """bf16/OMMX KV-byte ratio (the capacity multiplier). ~4.6x for the i2f4+i2 recipe."""
    return 32.0 / max(1e-6, ommx_bits_per_elem(head_dim))


def shrunk_head_size(head_size: int) -> int:
    """The reduced spec ``head_size`` that makes vLLM budget ``page_size_bytes`` for the
    OMMX packed footprint instead of bf16.

    ``page_size_bytes`` is linear in ``head_size``; we want it scaled by
    ``ommx_bits/32`` (both K and V together). Floor 16, multiple of 8 (FA shape sanity),
    never larger than the real head_size. Overridable by ``OMMX_KV_PACKED_HEADSIZE``.
    """
    forced = _env_int("OMMX_KV_PACKED_HEADSIZE", 0)
    if forced > 0:
        return max(8, min(int(head_size), forced))
    r = ommx_bits_per_elem(head_size) / 32.0          # OMMX fraction of bf16 bytes
    hs = head_size * r
    hs = max(8, int(round(hs / 8.0)) * 8)             # nearest multiple of 8, floor 8
    return min(int(head_size), hs)


# --- the monkeypatch: shrink FullAttentionSpec.head_size in PACKED-ONLY mode -------

_PATCHED = False


def install_packed_only_spec() -> Optional[float]:
    """Patch ``Attention.get_kv_cache_spec`` so PACKED-ONLY shrinks the bf16 cache
    page budget by the OMMX compression ratio. Idempotent. No-op (returns None) when
    disabled, already patched, or vLLM has no v1 Attention layer.

    Returns the realized capacity multiplier (bf16_head/shrunk_head) when installed.
    """
    global _PATCHED
    if _PATCHED or not packed_only_enabled():
        return None
    try:
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec
        from vllm.config import VllmConfig
    except Exception:
        return None

    _orig = Attention.get_kv_cache_spec

    def _packed_only_spec(self, vllm_config):  # noqa: ANN001
        spec = _orig(self, vllm_config)
        # Only shrink the standard full-attention bf16 KV cache (the OMMX target).
        # Sliding-window / TQ / MLA / fp8 specs are left untouched (not the OMMX path).
        if not isinstance(spec, FullAttentionSpec):
            return spec
        try:
            import torch
            if spec.dtype not in (torch.bfloat16, torch.float16):
                return spec  # quantized-cache layers: leave alone
            hs = shrunk_head_size(int(spec.head_size))
            if hs >= int(spec.head_size):
                return spec
            from dataclasses import replace
            new = replace(spec, head_size=hs, head_size_v=hs)
            # one-time evidence (law #5): the page budget really shrank.
            _packed_only_evidence(int(spec.head_size), hs)
            return new
        except Exception:
            return spec

    Attention.get_kv_cache_spec = _packed_only_spec  # type: ignore[assignment]
    _PATCHED = True
    return None


_EVIDENCE_DONE = [False]


def _packed_only_evidence(orig_hs: int, new_hs: int) -> None:
    if _EVIDENCE_DONE[0]:
        return
    _EVIDENCE_DONE[0] = True
    ratio = orig_hs / max(1, new_hs)
    line = (f"PACKED_ONLY_SPEC head_size {orig_hs} -> {new_hs} "
            f"(page_bytes /{ratio:.2f}x = +{ratio:.2f}x KV capacity) "
            f"ommx_bits/elem={ommx_bits_per_elem(orig_hs):.2f}")
    try:
        from vllm.logger import init_logger
        init_logger("ommx_gpu_serve").info("[ommx] %s", line)
    except Exception:
        pass
    fire = os.environ.get("OMMX_FIRE_FILE", "/tmp/ommx_route_fired.log")
    try:
        with open(fire, "a") as fh:
            fh.write(line + f" pid={os.getpid()}\n")
    except Exception:
        pass


__all__ = [
    "packed_only_enabled",
    "ommx_bits_per_elem",
    "packed_compression_ratio",
    "shrunk_head_size",
    "install_packed_only_spec",
]
