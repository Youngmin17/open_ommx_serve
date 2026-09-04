# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shipped correctness gates for the served OMMX KV path.

Two disjoint sets, separated by the ``gpu`` pytest marker (registered in
``conftest.py``):

  * CPU-only (the default set, ``-m 'not gpu'``): everything the write-pack /
    table / accounting path can prove WITHOUT a GPU, without vLLM and without
    triton. The canonical pack (``ommx_pack_kv_canonical_block``) is pure torch,
    so the real bit-exactness gate between ``CanonicalKVStore`` and
    ``MultiSeqKVPool`` runs here on ``device="cpu"``.
  * ``gpu``-marked: the same parity re-run on ``device="cuda"``. SKIPPED (never
    silently "passed") when ``torch.cuda.is_available()`` is False.

See ``README.md`` in this directory for the exact commands. That file is
shipped with the package, not just with the checkout: it is declared in
``ommx_gpu_serve/pyproject.toml`` under ``[tool.setuptools.package-data]``,
since a wheel carries only ``.py`` files otherwise.
"""
