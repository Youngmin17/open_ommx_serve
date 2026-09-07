# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""OMMX serving-framework adapters (attention only).

Each subpackage wires the canonical OMMX paged-decode attention (the two
``ommx_gpu_serve.attention.reference_op`` custom ops) into one serving framework,
delegating paging / scheduling / CUDA-graph orchestration / TP / radix to the
framework and providing only the decode kernel + KV pack/store + the minimal
metadata glue.

Subpackages — this list is the real tree of this release, not a roadmap:
  * ``vllm``   — the validated vLLM v1 backend (single-batch long-context, plus
    the batched pool). Modules: ``plugin`` (the ``vllm.general_plugins`` entry
    point), ``backend`` + ``metadata`` (the v1 attention backend and its metadata
    builder), ``config`` (env -> recipe resolution), ``preflight`` (the startup
    refusal guards) and ``packed_only`` (KV bit accounting).
  * ``common`` — framework-agnostic glue: ``batched_seam.BatchedStepPlanner``,
    the metadata -> ``paged_decode`` seam.

vLLM is the ONLY framework adapter shipped here, and ``batched_seam`` the only
module under ``common``. SGLang and LMCache adapters, a cross-segment ``cascade``
merge and a ``SCENARIO_MATRIX.md`` per-scenario verdict table are discussed in
this repo's prose and in the sibling docstrings; none of them exist as code in
this release. Import failures for ``ommx_gpu_serve.integration.sglang``,
``.lmcache`` or ``.common.cascade`` are expected, not a broken install.
"""
