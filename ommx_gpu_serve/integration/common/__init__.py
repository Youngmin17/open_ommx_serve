# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Framework-agnostic OMMX serving glue shared by the vLLM / SGLang adapters.

``batched_seam`` — ``BatchedStepPlanner`` turning per-request ``seq_lens`` into the
``ommx_gpu_serve::paged_decode`` host args (``b_seq_len`` / ``b_tail_len`` /
``tail_pos`` + the regroup plan). One implementation, two adapters.
"""
from .batched_seam import BatchedDecodePlan, BatchedStepPlanner

__all__ = ["BatchedDecodePlan", "BatchedStepPlanner"]
