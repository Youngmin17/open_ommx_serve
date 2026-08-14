# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""OMMX serving-framework adapters (attention only).

Each subpackage wires the canonical OMMX paged-decode attention (the two
``ommx_gpu_serve.attention.reference_op`` custom ops) into one serving framework,
delegating paging / scheduling / CUDA-graph orchestration / TP / radix to the
framework and providing only the decode kernel + KV pack/store + the minimal
metadata glue.

Subpackages:
  * ``vllm``    — the validated step1 vLLM v1 backend (single-batch long-context).
  * ``sglang``  — SGLang ``AttentionBackend`` adapter (engine + cluster-validated socket).
  * ``lmcache`` — OMMX-packed KV offload/restore (capacity/bandwidth tier).
  * ``common``  — framework-agnostic glue shared by the adapters (``BatchedStepPlanner``
    for the metadata->paged_decode seam; ``cascade`` for the P2 cross-segment merge).

See ``SCENARIO_MATRIX.md`` for the per-framework, per-scenario verdicts (which
scenarios the current two primitives handle vs. which need a kernel extension).
"""
