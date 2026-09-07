# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""pytest wiring for the shipped OMMX serving gates.

Three jobs, all of them about making the gates HONEST rather than convenient:

  1. Register the ``gpu`` marker so ``-m 'not gpu'`` (the CPU-only default set)
     selects cleanly and ``--strict-markers`` never trips on it.
  2. Turn ``gpu``-marked tests into an explicit SKIP (reported as ``s``, with a
     reason naming the missing device) when there is no CUDA device — a skip is
     visibly not a pass, which is the point: a GPU gate must never be able to
     report success on a machine that could not have run it.
  3. Scrub every ``OMMX_*`` environment variable before each test. The KV recipe
     (``OMMX_KV_RING`` / ``OMMX_KV_OUTLIER_MAP`` / ``OMMX_KV_INT8_SCALE`` /
     ``OMMX_KV_GPU_PACK`` / ``OMMX_ATTN_*``) is read from ``os.environ`` inside
     ``CanonicalKVStore.__init__`` / ``MultiSeqKVPool.__init__`` /
     ``packed_only.py``, so a developer who exported a recipe in their shell would
     otherwise silently run a DIFFERENT test than CI does. Each test sets the exact
     knobs it means to pin via ``monkeypatch.setenv``.
"""
from __future__ import annotations

import os
import sys

import pytest

# Repo root on sys.path so the gates run from a source checkout without an
# `pip install -e ommx_gpu_serve` first. pytest's rootdir insertion normally does
# this already (this package has __init__.py all the way up); the explicit guard
# keeps `python -m pytest ommx_gpu_serve/tests` working from any cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _cuda_available() -> bool:
    """True iff a real CUDA device is usable. CPU-only torch builds must not raise."""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: requires a real CUDA device. Skipped (NOT passed) when "
        "torch.cuda.is_available() is False; select with -m gpu, exclude with "
        "-m 'not gpu'.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    """Skip — never silently pass — ``gpu`` tests on a host with no CUDA device."""
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(
        reason="no CUDA device (torch.cuda.is_available() is False); run this file "
               "on a GPU host with: python -m pytest ommx_gpu_serve/tests -m gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _hermetic_ommx_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every ambient ``OMMX_*`` env var for the duration of one test.

    The recipe knobs are read at construction time, so an exported
    ``OMMX_KV_RING=1`` (or ``OMMX_KV_OUTLIER_MAP=0``, ...) would otherwise change
    what these gates actually verify without changing what they claim to verify.
    """
    for name in [k for k in os.environ if k.startswith("OMMX_")]:
        monkeypatch.delenv(name, raising=False)
