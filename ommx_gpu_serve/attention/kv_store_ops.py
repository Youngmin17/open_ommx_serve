# Copyright (c) 2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Allocation-free Triton copies for the fixed-address decode KV ring.

These kernels only move BF16 residual rows. Quantization and packed-plane
semantics remain in the canonical packer and attention kernels.
"""
from __future__ import annotations

_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl
    globals().update(triton=triton, tl=tl)

    @triton.jit
    def _kv_write_token(K, V, KH, VH, Pos, H: tl.constexpr, D: tl.constexpr,
                        KS0: tl.constexpr, KS1: tl.constexpr,
                        VS0: tl.constexpr, VS1: tl.constexpr,
                        CAP: tl.constexpr, SINK: tl.constexpr,
                        RECENT: tl.constexpr, BLOCK: tl.constexpr):
        pos = tl.load(Pos)
        if RECENT > 0:
            slot = tl.where(pos < SINK, pos, SINK + (pos - SINK) % RECENT)
        else:
            slot = pos
        valid = (slot >= 0) & (slot < CAP)
        tl.device_assert(valid, "OMMX KV write outside residual capacity")
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        h, c = x // D, x % D
        k = tl.load(K + h * KS0 + c * KS1, x < H * D, 0)
        v = tl.load(V + h * VS0 + c * VS1, x < H * D, 0)
        tl.store(KH + slot * H * D + x, k, valid & (x < H * D))
        tl.store(VH + slot * H * D + x, v, valid & (x < H * D))

    @triton.jit
    def _kv_gather_tail(KH, VH, Idx, KT, VT, HD: tl.constexpr,
                        CAP: tl.constexpr, SINK: tl.constexpr,
                        RECENT: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        pos = tl.load(Idx + row)
        if RECENT > 0:
            slot = tl.where(pos < SINK, pos, SINK + (pos - SINK) % RECENT)
        else:
            slot = pos
        valid = (slot >= 0) & (slot < CAP)
        tl.device_assert(valid, "OMMX KV gather outside residual capacity")
        x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        k = tl.load(KH + slot * HD + x, valid & (x < HD), 0)
        v = tl.load(VH + slot * HD + x, valid & (x < HD), 0)
        tl.store(KT + row * HD + x, k, x < HD)
        tl.store(VT + row * HD + x, v, x < HD)

    _KERNELS = triton, _kv_write_token, _kv_gather_tail
    return _KERNELS


def write_token(k_hist, v_hist, pos, k, v, sink: int, recent: int):
    triton, write, _ = _kernels()
    h, d = k_hist.shape[1:]
    write[(triton.cdiv(h * d, 256),)](
        k, v, k_hist, v_hist, pos, h, d, *k.stride(), *v.stride(),
        k_hist.shape[0], sink, recent, 256, num_warps=4, debug=True)


def gather_tail(k_hist, v_hist, indices, k_tail, v_tail, sink: int, recent: int):
    triton, _, gather = _kernels()
    hd = k_hist.shape[1] * k_hist.shape[2]
    gather[(indices.numel(), triton.cdiv(hd, 1024))](
        k_hist, v_hist, indices, k_tail, v_tail, hd, k_hist.shape[0],
        sink, recent, 1024, num_warps=4, debug=True)
