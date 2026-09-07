# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""KIVI sink/recent residual-window arithmetic for the canonical KV store.

Pure-python, stateless, dependency-free (no torch, no env globals — every knob is
an explicit argument). The serving KV store splits each request's KV into three
regions along the token axis:

    [0, sink)            attention SINK   -> bf16 residual (k_tail/v_tail)
    [sink, boundary)     QUANTIZED PREFIX -> canonical packed planes (i2f4/i2)
    [boundary, seq_len)  RECENT window +  -> bf16 residual (k_tail/v_tail)
                         incomplete tail group

``boundary`` is a STATELESS function of ``seq_len`` (so the per-request state can
never drift across preemption / batch-slot reordering — the KIVI design rule), and
is always ``sink + (a multiple of group_tokens)``, i.e. the quantized prefix is an
exact number of 32-token scale-groups, which is exactly what
``ommx_pack_kv_canonical_block`` requires.

Invariant (once ``seq >= sink + recent``):
    recent <= seq - boundary < recent + group_tokens
A full ``group_tokens`` accumulating beyond ``recent`` triggers one regroup-pack
(``seq - boundary >= recent + group_tokens`` advances boundary by one group).

This mirrors the parent ``ommx/integration/vllm/kv_twin_core.py`` boundary logic
(``packed_boundary`` / ``sink_*_for_page_size``) re-expressed as free functions.
"""
from __future__ import annotations

from dataclasses import dataclass

CANONICAL_GROUP_TOKENS = 32


def packed_boundary(seq_len: int, sink_tokens: int, recent_window: int,
                    group_tokens: int = CANONICAL_GROUP_TOKENS) -> int:
    """Absolute end token index of the quantized prefix for ``seq_len``.

    Tokens ``[sink_tokens, boundary)`` are served from the packed planes; the sink
    ``[0, sink_tokens)`` and the tail ``[boundary, seq_len)`` from the bf16 cache.
    Returns ``min(sink_tokens, seq_len)`` (empty quantized prefix) until the
    sequence grows past ``sink + recent``.
    """
    seq_len = int(seq_len); sink_tokens = int(sink_tokens)
    recent_window = int(recent_window); group_tokens = int(group_tokens)
    if group_tokens <= 0:
        raise ValueError(f"group_tokens must be > 0; got {group_tokens}")
    extra = seq_len - sink_tokens - recent_window
    if extra <= 0:
        return min(sink_tokens, seq_len)
    return sink_tokens + (extra // group_tokens) * group_tokens


def packed_len(seq_len: int, sink_tokens: int, recent_window: int,
               group_tokens: int = CANONICAL_GROUP_TOKENS) -> int:
    """Number of quantized tokens = ``boundary - sink`` (a multiple of group_tokens)."""
    return max(0, packed_boundary(seq_len, sink_tokens, recent_window, group_tokens)
               - int(sink_tokens))


def num_packed_groups(seq_len: int, sink_tokens: int, recent_window: int,
                      group_tokens: int = CANONICAL_GROUP_TOKENS) -> int:
    """Number of completed 32-token scale-groups in the quantized prefix."""
    return packed_len(seq_len, sink_tokens, recent_window, group_tokens) // int(group_tokens)


def newly_completed_groups(prev_seq_len: int, new_seq_len: int, sink_tokens: int,
                           recent_window: int,
                           group_tokens: int = CANONICAL_GROUP_TOKENS) -> int:
    """Groups whose boundary was crossed between ``prev_seq_len`` and ``new_seq_len``.

    The regroup-pack trigger: ``> 0`` means that many fresh groups must be packed
    from the bf16 window into the canonical planes at this step's build seam.
    """
    g0 = num_packed_groups(prev_seq_len, sink_tokens, recent_window, group_tokens)
    g1 = num_packed_groups(new_seq_len, sink_tokens, recent_window, group_tokens)
    return max(0, g1 - g0)


def tail_token_indices(seq_len: int, sink_tokens: int, recent_window: int,
                       group_tokens: int = CANONICAL_GROUP_TOKENS) -> list[int]:
    """Absolute token indices of the bf16 residual rows = sink ∪ (recent + tail).

    The contiguous ``k_tail``/``v_tail`` rows the decode kernel reads (any order;
    here sink-first then tail). ``len(...)`` is ``b_tail_len`` for the request.
    """
    seq_len = int(seq_len); sink_tokens = int(sink_tokens)
    boundary = packed_boundary(seq_len, sink_tokens, recent_window, group_tokens)
    sink = list(range(0, min(sink_tokens, seq_len)))
    tail = list(range(boundary, seq_len))
    return sink + tail


def sink_page_shift(sink_tokens: int, page_size: int) -> int:
    """Leading whole pages the packed page-table skips (sink // page_size)."""
    return int(sink_tokens) // max(1, int(page_size))


def sink_token_offset(sink_tokens: int, page_size: int) -> int:
    """In-page token offset where packed attention starts (``packed_start_offset``)."""
    return int(sink_tokens) % max(1, int(page_size))


@dataclass(frozen=True)
class WindowSpec:
    """Bundled residual-window geometry (the per-config constants)."""

    sink_tokens: int = 8
    recent_window: int = 32
    group_tokens: int = CANONICAL_GROUP_TOKENS
    page_size: int = 16

    def boundary(self, seq_len: int) -> int:
        return packed_boundary(seq_len, self.sink_tokens, self.recent_window,
                               self.group_tokens)

    def packed_len(self, seq_len: int) -> int:
        return packed_len(seq_len, self.sink_tokens, self.recent_window,
                          self.group_tokens)

    def num_groups(self, seq_len: int) -> int:
        return num_packed_groups(seq_len, self.sink_tokens, self.recent_window,
                                 self.group_tokens)

    def newly_completed_groups(self, prev_seq_len: int, new_seq_len: int) -> int:
        return newly_completed_groups(prev_seq_len, new_seq_len, self.sink_tokens,
                                      self.recent_window, self.group_tokens)

    def tail_indices(self, seq_len: int) -> list[int]:
        return tail_token_indices(seq_len, self.sink_tokens, self.recent_window,
                                  self.group_tokens)

    def packed_start_offset(self) -> int:
        return sink_token_offset(self.sink_tokens, self.page_size)

    def sink_page_shift(self) -> int:
        return sink_page_shift(self.sink_tokens, self.page_size)

    def max_tail_len(self, max_seq_len: int) -> int:
        """Worst-case bf16 residual rows = sink + (recent + a full incomplete group)."""
        return int(self.sink_tokens) + int(self.recent_window) + int(self.group_tokens)


__all__ = [
    "CANONICAL_GROUP_TOKENS",
    "WindowSpec",
    "packed_boundary",
    "packed_len",
    "num_packed_groups",
    "newly_completed_groups",
    "tail_token_indices",
    "sink_page_shift",
    "sink_token_offset",
]
