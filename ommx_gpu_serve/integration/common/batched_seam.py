# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Framework-agnostic batched decode seam — metadata -> paged_decode args.

The single piece of host-side logic every multi-request OMMX decode adapter needs,
factored out so vLLM (generalizing its single-batch ``OMMXStepManager``) and SGLang
(its new ``AttentionBackend``) share ONE implementation instead of two copies (both
framework research passes flagged the SGLang seam as "mirror the vLLM manager").

Given per-request ``seq_lens`` and a ``WindowSpec``, it computes — purely on the host,
OUTSIDE any CUDA-graph capture — the per-request quantities the
``ommx_gpu_serve::paged_decode`` op reads:

  * ``b_seq_len[B]``  — packed (quantized-prefix) token count per request
                        (= ``num_packed_groups * group_tokens``);
  * ``b_tail_len[B]`` — bf16 residual row count (sink ∪ recent ∪ incomplete tail);
  * ``tail_pos[B, max_tail_len]`` — absolute token indices to gather the bf16 tail
                        (padded with 0, masked by ``b_tail_len`` in the kernel);
  * ``regroup[B]``    — (prev_groups, new_groups) so the caller packs the newly
                        completed groups on each request's store at the build seam;
  * ``max_seq_len`` / ``max_tail_len`` — fixed grid-sizing ints for graph capture.

Pure-python + torch tensors; no vLLM / SGLang / GPU import. CPU-tested against the
canonical ``kv_window`` arithmetic (which is itself the KIVI residual-window source).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from ommx_gpu_serve.attention.kv_window import WindowSpec


@dataclass
class BatchedDecodePlan:
    """Host-computed per-request decode geometry (device buffers + plan)."""

    b_seq_len: torch.Tensor      # [B] int32  packed token count
    b_tail_len: torch.Tensor     # [B] int32  bf16 residual rows
    tail_pos: torch.Tensor       # [B, max_tail_len] int64  absolute gather indices
    regroup: list[tuple[int, int]]   # per request (prev_packed_groups, new_packed_groups)
    max_seq_len: int             # fixed grid size (worst-case packed tokens)
    max_tail_len: int            # fixed grid size (worst-case residual rows)
    seq_lens: list[int]          # echoed per-request seq lengths


class BatchedStepPlanner:
    """Stateless-per-call planner; tracks prior packed-group counts for regrouping.

    ``window`` is the shared residual-window spec. ``max_context`` sizes the fixed
    ``max_tail_len`` (graph-capture grid). One planner per worker; ``plan(seq_lens)``
    each decode step. State held: the previous packed-group count per request slot,
    so the caller knows which groups newly completed (and must be packed).
    """

    def __init__(self, window: WindowSpec, max_context: int,
                 device: str | torch.device = "cpu") -> None:
        self.window = window
        self.max_context = int(max_context)
        self.device = torch.device(device)
        # worst-case residual rows = sink + recent + one full incomplete group.
        self.max_tail_len = int(window.max_tail_len(self.max_context))
        self._prev_groups: dict[int, int] = {}   # request-slot -> last packed groups

    def reset(self, slot: Optional[int] = None) -> None:
        if slot is None:
            self._prev_groups.clear()
        else:
            self._prev_groups.pop(int(slot), None)

    def plan(self, seq_lens: Sequence[int],
             slots: Optional[Sequence[int]] = None) -> BatchedDecodePlan:
        """Build the per-request decode geometry for this step's batch.

        ``seq_lens[i]`` is request ``i``'s current sequence length; ``slots[i]`` is its
        stable store/request slot id (defaults to the row index ``i``) used to track
        regroup deltas across steps. Returns a ``BatchedDecodePlan``.
        """
        seq_lens = [int(s) for s in seq_lens]
        B = len(seq_lens)
        slots = [int(s) for s in (slots if slots is not None else range(B))]
        win = self.window

        b_seq = torch.zeros(B, dtype=torch.int32, device=self.device)
        b_tail = torch.zeros(B, dtype=torch.int32, device=self.device)
        tail_pos = torch.zeros(B, self.max_tail_len, dtype=torch.int64,
                               device=self.device)
        regroup: list[tuple[int, int]] = []

        for i, (seq, slot) in enumerate(zip(seq_lens, slots)):
            groups = win.num_groups(seq)
            packed_tokens = groups * win.group_tokens
            tail = win.tail_indices(seq)
            n_tail = len(tail)
            if n_tail > self.max_tail_len:   # clamp (kernel masks by b_tail_len)
                tail = tail[:self.max_tail_len]
                n_tail = self.max_tail_len
            b_seq[i] = packed_tokens
            b_tail[i] = n_tail
            if n_tail:
                tail_pos[i, :n_tail] = torch.tensor(tail, dtype=torch.int64)
            prev = self._prev_groups.get(slot, 0)
            if seq <= 0:                     # a (re)started slot resets its history
                prev = 0
            regroup.append((prev, groups))
            self._prev_groups[slot] = groups

        return BatchedDecodePlan(
            b_seq_len=b_seq, b_tail_len=b_tail, tail_pos=tail_pos,
            regroup=regroup,
            max_seq_len=max(1, int(b_seq.max().item()) if B else 1),
            max_tail_len=self.max_tail_len, seq_lens=seq_lens)


__all__ = ["BatchedDecodePlan", "BatchedStepPlanner"]
