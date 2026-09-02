# High-Coverage Outlier Representation in Low-Bit MX Formats for Efficient LLM Inference (ICCAD 2026)

Outliers are the main reason low-bit LLM quantization loses resolution, and with it model
robustness. **OMMX (Outlier-Managed-MX)** is a **dual-resolution MX format** — MXINT2 for the dense
majority, MXFP4 for a high-coverage outlier region under a shared power-of-two scale — that widens
outlier coverage while keeping a compact bundle-packed layout a CUDA/Triton kernel can execute
directly.

This repository is the **GPU half** of the paper's artifact: the accuracy evaluation of OMMX against
other quantization methods, and the serving kernels integrated into vLLM and HF-eager. The Nucleus
NPU (§3.2, §4.3 of the paper) is a separate codebase and is **not** included here.

---

## What is in this artifact

| axis | what runs | status |
|---|---|---|
| **KV cache** | Triton paged-decode kernel + vLLM backend + HF-eager path | measured (accuracy and latency) |
| **Weights** (`ommx_w`) | offline packer → `OMMX_W_SafeTensor` bundle → vLLM `LinearMethod` → CUDA GEMM | measured (accuracy + kernel parity); **no serving TPOT** |
| **Accuracy simulator** | `ommx_fakequant/` fake-quant for `lm_eval` / RULER | measured |
| **Baselines** | KIVI, Kitty, TurboQuant, LMDeploy W4A16KV4, vLLM bf16 | measured |
| Nucleus NPU (RTL, cycle model, area/energy) | — | **not in this repository** |

---

## Results

**Accuracy / bit tradeoff** (paper, `lm_eval`, 5 model families):

| ![Table 1](figure/paper_table1.png) |
|:--:|
| **Table 1** — KV-cache (left) and weight (right) quantization vs KIVI / KVQ / Oaken / AWQ / GPTQ / MXFP4. |

| ![Fig 6](figure/paper_fig6.png) |
|:--:|
| **Fig. 6** — RULER-32K vs outlier ratio (weight / KV / joint WKV): accuracy recovers as outlier coverage grows. |

**Decode TPOT, B=1** (Llama-3.1-8B, greedy, seed 42). Two figures below are drawn from
`figure/data_*/`, the **already-published** measurement set produced by an earlier `figure/bench.py`;
[`docs/MEASUREMENT_NOTES.md` §4](docs/MEASUREMENT_NOTES.md) prices exactly what that costs and must
be read before any row is cited.

| ![H200 TPOT](figure/fig_tpot_h200.png) | ![A100 TPOT](figure/fig_tpot_a100.png) |
|:--:|:--:|
| **H200** | **A100-SXM4-80GB** |

`OMMX (vLLM)` is the OMMX KV/attention kernel under vLLM — **bf16 weights with INT2/FP4 KV**, i.e. a
KV-quant comparison, not weight-quantized OMMX. Its bar is split into the kernel's internal stages.
On sm_90 the OMMX decode runs piecewise (a full CUDA graph would freeze the per-step metadata), so
vLLM-native TurboQuant is faster single-stream there; on sm_80 the OMMX decode is captured and the
ranking flips. **OMMX's claim is accuracy at low bits, not sm_90 latency.**

Three things about the comparison, so no row is over-read:

- **OMMX is the highest-bit method here, not the lowest.** Its canonical recipe measures **4.375**
  average KV bits (K 6.00 / V 2.75) against ~2.5 for KIVI and Kitty. The speed and memory numbers
  are **not** an equal-bit comparison.
- **"Same engine" means the same framework, not the same code.** The HF-eager arms are four separate
  model implementations with different per-step Python overhead, which dominates below 16K. Treat the
  ≥16K rows as the meaningful comparison.
- **OMMX is slower than 16-bit bf16 at long context.** Its decode is dequant-bound. The win is over
  the other *quantized* methods, and in accuracy.

Full derivations, the KIVI fused-GQA measurement, and the per-arm dtype record are in
[`docs/MEASUREMENT_NOTES.md`](docs/MEASUREMENT_NOTES.md).

---

## Setup

KIVI, Kitty and vLLM pin mutually incompatible `transformers`/engine versions, so each method gets
its own venv ([`uv`](https://github.com/astral-sh/uv) recommended). `./run.sh setup` prints these:

```bash
uv venv .venv-ommx  --python 3.10 && uv pip install -e '.[ommx]'  && uv pip install -e ommx_gpu_serve && uv pip install -e eval
uv venv .venv-kivi  --python 3.10 && uv pip install -e '.[kivi]'  && uv pip install -e eval
uv venv .venv-kitty --python 3.10 && uv pip install -e '.[kitty]' && uv pip install -e eval   # + KITTY_PKG_PATH
```

Not vendored: KIVI's optional CUDA gemv (the Triton fallback works) and the
[Kitty](https://github.com/Summer-Summer/Kitty) package.

The weight kernel additionally needs `uv pip install -e '.[linear]'`, because it is JIT-compiled at
weight-load time and that shells out to the **`ninja` executable** — installing the wheel is not
enough, `<env>/bin` must be on `PATH`. On conda environments that keep `libcudart.so` in `lib/`,
also export `LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64`.

---

## Reproduce

### KV axis — decode TPOT, peak memory, quality

```bash
./run.sh all --gpu 0 --tag h200 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --ctxs 1024,4096,16384,32768,65536 \
  --methods "bf16 ommx kivi kitty vllm_bf16 ommx_vllm"
```

`run.sh` writes `figure/data_<tag>/*.json`; `figure/collect.py` normalizes them into
`figure/data/<tag>.json`; `figure/plot.py` renders `fig_tpot_<tag>.png` and `fig_peakmem_<tag>.png`.
RULER-32K accuracy runs through `eval/` (task `ruler` @ 32768).

### Paper Fig. 7 — every bar measured

The four-series panel needs a baseline `run.sh` does not carry (LMDeploy W4A16KV4) and an 8K point
the default `--ctxs` omits, so it has its own reproducer. The stages are separate because they need
three incompatible Python environments.

```bash
export OMMXPY=… KIVIPY=… LMDPY=…      # one interpreter per stage's environment
repro/fig7_sweep.sh vllm              # OMMX (CUSTOM) + bf16 (FlashAttn) + stage breakdown
repro/fig7_sweep.sh kivi              # KIVI, HF-eager, fp16
repro/fig7_sweep.sh lmdeploy          # LMDeploy W4A16KV4 (needs an AWQ-INT4 checkpoint)
repro/fig7_sweep.sh figure            # -> figure/fig7_tpot_vs_ctx.png
```

A method absent from the data gets **no bar** and is named in the title, and the `×N` label states
its own denominator on the axes (`--ratio-ref`, default `kivi_hf`).

### Weight axis (`ommx_w`) — pack, then serve

```bash
# 1. offline pack (CPU only)
python -m ommx_gpu_serve.linear.w_packer pack --preset optimized-weight \
    --input /models/Llama-3.1-8B-Instruct --output /models/llama31-8b-ommx_w
python -m ommx_gpu_serve.linear.w_packer verify --bundle /models/llama31-8b-ommx_w

# 2. serve — the bundle stamps its own quantization_config, so no env var is needed
vllm serve /models/llama31-8b-ommx_w --quantization ommx_w --dtype bfloat16
```

Add `--attention-backend CUSTOM` to put the OMMX KV cache on the same run. The two axes are
independent; either can be selected without the other.

**Calibration.** The packer's default path chooses INT2 codes by min/max round-to-nearest, which is
not sufficient at 2 bits. To carry an error-feedback solver's decisions into a bundle:

```bash
python ommx_fakequant/wq_eval.py --model <hf-checkpoint> --gpu 0 \
    --bits 2 --group-size 64 --outlier-percent 0.25 --pow2 \
    --special-first-k 0 --special-last-k 0 --special-projs "" \
    --export-decisions /tmp/decisions
python -m ommx_gpu_serve.linear.w_packer pack --preset optimized-weight \
    --input <hf-checkpoint> --output <bundle> --calibrated /tmp/decisions
```

The pack gate requires the packed planes to reconstruct the weight the solver measured, so a
mismatch is refused rather than shipped. Re-quantizing an already-on-grid weight is not idempotent,
which is why the decisions travel rather than the calibrated weight.

### Correctness gates

```bash
python -m pytest ommx_gpu_serve/tests -m 'not gpu' -q   # 768 passed, 2 skipped, 4 deselected
python -m pytest ommx_gpu_serve/tests -m gpu -q         # 5 CUDA cases; they SKIP with no device
```

| gate | what it pins |
|---|---|
| `test_kv_pool_parity.py` | the shared `MultiSeqKVPool` dequants bit-identically to a standalone single-sequence store, and a write to one slot leaves its neighbours byte-identical |
| `test_slot_allocator.py` | the batched allocator never aliases two live requests; exhaustion and duplicate keys raise instead of recycling a live slot |
| `test_bit_accounting.py` | the KV bit accounting against an independent longhand derivation from the pool's real allocation shapes |
| `test_npu_bit_basis.py` | the NPU bit basis against the KV codec, and that the shipped recipe is the one measured through the kernel rather than the cheapest on the simulator |
| `test_calibration_path.py` | that encoding from decisions is bit-identical to deciding, and that a weight disagreeing with its own decisions is refused |
| `test_recipes.py` | every recorded bit figure recomputed from the live code; an unknown preset raises; an explicit setting beats the preset |
| `test_preflight_guards.py` | the byte projection matches a real allocated pool plane-for-plane; prefix caching, chunked prefill and an oversubscribed pool each raise |
| `test_linear_method.py` | the `ommx_w` vLLM wiring: idempotent registration, a bundle parsed from one the packer built in-test, and every refusal path named |

GPU-only: `hf_eager/_ommx_hf_batch_test.py` (OMMX-KV greedy parity vs bf16) and
`csrc/linear/test_ommx_linear_parity.py` (weight-kernel format parity).

---

## Named recipes

A published bit budget is a flag, not a guess. `ommx_gpu_serve/recipes.py` is a registry of named,
measured, tested recipes; `python3 -m ommx_gpu_serve.recipes list` prints it and `… recipes verify`
recomputes every number from the live code.

| recipe | axis | bits | NPU basis | accuracy here |
|---|---|--:|--:|---|
| `shipped-kv` | KV | 4.3750 avg/elem | — | **yes** — every published OMMX KV number |
| `paper-kv` | KV | 2.7500 avg/elem | — | **none** |
| `shipped-weight` | weight | 4.1250 /weight | 3.9375 | none |
| `paper-weight` | weight | 3.6250 /weight *on disk* | 2.9375 | none |
| **`optimized-weight`** | weight | **6.1250** /weight | **5.1406** | **yes** — served, wikitext PPL 9.0098 vs bf16 6.2433 |

```bash
OMMX_RECIPE=paper-kv vllm serve <model> --attention-backend CUSTOM ...
./run.sh bench --recipe paper-kv --tag a100
python -m ommx_gpu_serve.linear.w_packer pack --preset optimized-weight --input … --output …
```

An explicit environment variable or CLI flag always beats the preset; an unknown name raises and
lists the known ones; with no preset named the shipped defaults resolve unchanged. Use
`./run.sh --recipe NAME`, which exports the assignments before any child process starts —
[`docs/MEASUREMENT_NOTES.md` §2](docs/MEASUREMENT_NOTES.md) explains where `OMMX_RECIPE` alone is
not enough, and how each recipe was selected.

The `bits` column is a **byte** claim. `paper-kv` and `paper-weight` reproduce a published budget
and have **no accuracy measurement at that recipe** in this repository; Table 1's accuracy figures
were produced with `shipped-kv` and do not transfer. The **NPU basis** re-prices the same bundle
with positions stored as `ceil(log2 C(group, npv))` per group, which is how the paper specifies the
Nucleus codec; `w_packer budget` prints both columns.

---

## Known limitations

- **vLLM defaults do not work with the OMMX KV backend.** The sidecar is not paged, so
  `enable_prefix_caching` and `enable_chunked_prefill` (both `True` on vLLM 0.21) must be off.
  `integration/vllm/preflight.py` refuses to start rather than degrading silently, and a route
  failure is fatal by default. Earlier revisions fell back to bf16 FlashAttention while producing
  fluent output — a bench run that way measured FlashAttention.
- **No `ommx_w` serving TPOT number exists in this repository.** The weight path's accuracy is
  measured end to end through the CUDA kernel; its serving latency is not. A standalone
  decode-linear microbench does exist, with its own caveats (not iso-bit, not group-size matched,
  eager two-launch path) — [`docs/MEASUREMENT_NOTES.md` §5](docs/MEASUREMENT_NOTES.md).
- **The weight packer's default path is RTN.** At 2 bits it is not sufficient — use `--calibrated`.
  Coarse groups sit near a stability edge: a configuration can score well on the simulator and still
  diverge under vLLM prefill, so a recipe is only shipped here after being served.
- **The shipped `figure/data_*/` JSONs predate the current bench** and must be regenerated before
  being cited. See [`docs/MEASUREMENT_NOTES.md` §4](docs/MEASUREMENT_NOTES.md).
- **The Nucleus NPU results are not reproducible from this repository.**

---

## Layout

```
open_ommx_serve/
├── ommx_fakequant/   # fake-quant accuracy simulator (weight + KV) for lm_eval / RULER
├── ommx_gpu_serve/   # Triton paged-decode KV kernel + vLLM integration + HF-eager path
│   ├── recipes.py                         #   named recipes + measured bit budgets
│   ├── linear/                            #   offline weight path: quantizer, format, packer CLI
│   ├── csrc/linear/                       #   i2f4 weight GEMM kernel + parity gate + microbench
│   ├── integration/vllm/preflight.py      #   startup guards: refuse what the sidecar cannot serve
│   └── tests/                             #   correctness gates (CPU set + GPU-marked cases)
├── baseline/         # KIVI / Kitty comparison drivers
├── eval/             # trimmed lm-eval (RULER + adapters); eval/lcb/ = LiveCodeBench KV study
├── figure/           # bench + collect + plot + measured data + figures
├── repro/            # per-GPU orchestrators, Fig. 7 sweep, provenance audit
├── docs/             # measurement notes (provenance, recipe selection, vLLM constraints)
└── run.sh            # single entry: setup / bench / figure / all
```

---

## Citation

```bibtex
@inproceedings{cho2026high,
  author    = {Cho, Youngmin and Lee, Yoontae and Park, Rojin and Lee, Sunjae and Park, Jonghyeok and Lee, Jimin and Oh, Young H. and Park, Jeongwoo},
  title     = {High-Coverage Outlier Representation in Low-Bit {MX} Formats for Efficient {LLM} Inference},
  booktitle = {IEEE/ACM International Conference on Computer Aided Design (ICCAD)},
  year      = {2026}
}
```

## License

Apache-2.0
