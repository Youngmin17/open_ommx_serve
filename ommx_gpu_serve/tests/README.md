# `ommx_gpu_serve` correctness gates

These are the shipped tests for the **served** OMMX KV path — the write-pack seam
(`attention/kv_pool.py`, `attention/kv_store.py`), the batched-decode slot allocator
(`integration/vllm/metadata.py`), the KV bit accounting
(`integration/vllm/packed_only.py`) and the vLLM startup guards
(`integration/vllm/preflight.py`).

**CPU-only is the default set.** Everything except one test runs on plain CPU torch
with **no GPU, no vLLM and no triton** installed: the canonical packer
(`ommx_pack_kv_canonical_block`) is pure torch, so the real bit-exactness gate between
a single-sequence `CanonicalKVStore` and a `MultiSeqKVPool` slot runs for real on
`device="cpu"`, and the preflight tests drive the guards with plain stub objects and a
stubbed `sys.modules["vllm"]` rather than a vLLM install. **GPU-only** is the single
`gpu`-marked test (`test_pool_planes_match_store_ragged_batch_cuda`), which re-runs the
ragged-batch plane parity on `device="cuda"`; with no CUDA device `conftest.py` turns it
into an explicit `SKIP` (reported as `s`, with the reason) so it can never be mistaken
for a pass. The whole CPU set finishes in well under a second and peaks around
120 MiB of resident memory above the bare `import torch` baseline (measured with
`/usr/bin/time -l`): the structural geometries run with `OMMX_KV_RING=1` so the
bf16 shadow never sizes to `max_model_len`, and the one 128K-context geometry uses
few slots. Keep it that way — this gate is meant to run on a laptop.

Run the CPU set (what CI and a release check should run) from the repo root:

```
python -m pytest ommx_gpu_serve/tests -m 'not gpu' -q
```

Run the GPU set on a CUDA host (same env, plus a device):

```
python -m pytest ommx_gpu_serve/tests -m gpu -q
```

Everything, on a GPU host: `python -m pytest ommx_gpu_serve/tests -q`.

The same gates run against an **installed** `ommx-gpu-serve` rather than this
checkout — `ommx_gpu_serve.tests` is a shipped package and this file travels with it
(`[tool.setuptools.package-data]` in `ommx_gpu_serve/pyproject.toml`), so the paths
above are not needed:

```
python -m pytest --pyargs ommx_gpu_serve.tests -m 'not gpu' -q
```

Tests that depend on a symbol which may not be present yet are marked
`xfail(strict=True)` on `ImportError` — never `skip` — so a missing gate reports as an
expected failure naming the symbol instead of quietly reporting nothing.
`conftest.py` also scrubs every `OMMX_*` environment variable before each test, since
the KV recipe is read from `os.environ` at construction time; an exported recipe would
otherwise silently change what is being verified.
