# High-Coverage Outlier Representation in Low-Bit MX Formats for Efficient LLM Inference (ICCAD 2026)

LLM 양자화에서 이상치(outlier)는 해상도를 낮추어 모델 견고성을 해치는 주요한 요인입니다.
**OMMX(Outlier-Managed-MX)** 는 Dense는 MXINT, Outlier는 MXFP를 사용하는 **이중 해상도 포맷**으로,
압축된 데이터 레이아웃을 유지하면서 이상치 처리 범위를 향상시킵니다. 우리는 이를 **HW-SW Co-Design**
을 통해 효과적으로 LLM 서빙에 통합시킵니다.

이 저장소는 OMMX 및 타 베이스라인의 **추론 정확도 평가**와, **vLLM 및 HF-eager 에 통합된 GPU 서빙
커널**을 공개합니다.

> **OMMX** casts ultra-low-bit quantization as a dual-resolution MX format — **MXINT2** for the
> dense majority, **MXFP4** for a high-coverage outlier region under a shared power-of-two scale —
> keeping a bundle-packed layout that a custom CUDA/Triton kernel executes inside vLLM. This repo
> releases the accuracy evaluation (OMMX vs baselines) and the GPU serving kernels (vLLM + HF-eager).

---

## Results at a glance

**Accuracy / bit tradeoff** (from the paper — `lm_eval`, 5 model families):

| ![Table 1](figure/paper_table1.png) |
|:--:|
| **Table 1** — KV-cache (left) + weight (right) quantization vs KIVI / KVQ / Oaken / AWQ / GPTQ / MXFP4. |

| ![Fig 6](figure/paper_fig6.png) |
|:--:|
| **Fig. 6** — RULER-32K vs outlier ratio (Weight / KV / joint WKV): accuracy recovers as outlier coverage grows. |

**Serving — B=1 decode TPOT** (Llama-3.1-8B, fp16, greedy, CUDA-event timed, seed=42):

| ![H200 TPOT](figure/fig_tpot_h200.png) |
|:--:|
| **H200** — OMMX (vLLM / HF-eager), bf16 (vLLM / HF-eager), KIVI, Kitty, TurboQuant. `OMMX (vLLM)` is the OMMX paged-decode **KV/attention** kernel under vLLM (bf16 weights + INT2/FP4 KV — a KV-quant comparison, *not* weight-quantized OMMX); its bar is split into the kernel's internal stages (base attention / outlier-membership / unpack / scale-zp dequant / pack-write). On sm_90 the OMMX decode runs **piecewise** (a full CUDA-graph would freeze the per-step metadata), so vLLM-native `TurboQuant` (3-bit KV via `--kv-cache-dtype`) is faster single-stream here — OMMX's edge is **accuracy at low bits**, not sm_90 latency. |

| ![A100 TPOT](figure/fig_tpot_a100.png) |
|:--:|
| **A100** — the same methods on A100-SXM4-80GB. Same-engine (HF-eager), OMMX beats KIVI **×1.4→×2.4** across 1K→64K and passes Kitty by 64K (×1.5; Kitty leads at 1K). Low-bit-KV dequant makes OMMX (HF-eager) trail 16-bit bf16 at every context (×1.8 at 1K → ×3.3 at 64K) — OMMX's win is over the other **quant** methods and in memory, not over bf16 speed. Against **TurboQuant**, `OMMX (vLLM)` is at parity single-stream and faster at long context (14.5→76.3 vs 27.5→116.6 ms) — TurboQuant's Walsh–Hadamard rotation is much slower on sm_80, flipping the sm_90 ranking. |

| ![Peak memory](figure/fig_peakmem_h200.png) |
|:--:|
| **Peak process memory** (weights + prefill activations + KV) — *not* KV storage. OMMX / Kitty stay low; the KIVI spike is its HF-eager `_eval` decode path (full-KV fp16 dequant + dense matmul each step), not a larger *stored* KV cache. |

**Demo — live inference.** An `asciinema` recording of the **actual OMMX paged-decode kernel**
generating on H200 (Llama-3.1-8B, 32k context), streaming each token with a running tok/s
(`figure/demo_gen.py --live`, rasterized by `figure/demo_cast_to_gif.py`):

![live demo](assets/demo_ommx_live.gif)

**Demo — 4-method comparison.** Four real generations (H200, best recipe) replayed side by side at
their real relative speed: bf16 (29 tok/s) and **OMMX (23)** finish well ahead of KIVI / Kitty (~10).
The timing is real (steady-state tok/s, one-time Triton JIT excluded); the side-by-side layout is a replay.

![race demo](assets/demo_race.gif)

## What reproduces (measured)

All OMMX numbers use the **canonical best recipe** (INT2 base + **12-of-64 FP4 K-outliers**, sink 8,
pow2 scales), selected by `OMMX_ATTN_OUTLIERS=6 OMMX_ATTN_POW2=1 OMMX_ATTN_K_FORMAT=i2f4
OMMX_KV_GROUP_TOKENS=32 OMMX_KV_SINK=8 OMMX_KV_RECENT=32` — set identically in `figure/bench.py` and
`eval/lm_eval/models/ommx_hf_model.py`.

Each method uses **its own best low-bit-KV recipe** (not iso-bit):

| method | KV recipe | avg KV bits | bf16 tail |
|---|---|--:|---|
| **OMMX** | INT2 base + 12/64 FP4 K-outliers, pow2 scales | **~4.1** | sink 8 + recent 32 |
| KIVI | 2-bit K/V, group 32 | ~2.55 | residual 128 |
| Kitty | 2-bit K/V + 25% INT4-channel boost | ~2.5 | sink 32 |

**OMMX is the highest-bit method here, not the lowest.** Its ~4.1 average KV bits (K 5.25, V 3.0) buy
the paper's high-coverage accuracy — so the speed and memory numbers are **not** an equal-bit comparison.
OMMX trades ~1.6× more KV bits than KIVI / Kitty for accuracy, and still decodes faster than them while
fitting far more context than bf16. (KIVI is timed on its HF-eager `_eval` decode path — Triton pack +
dequant + `torch.matmul` GEMV; its fused `kivi_gemv` is MHA-only and slower for this GQA model at M=1.)

**H200, decode TPOT (ms/step)** — HF-eager (OMMX also shown under vLLM):

| ctx | bf16 | **OMMX (hf)** | OMMX (vLLM) | KIVI | Kitty |
|----:|-----:|--------------:|------------:|-----:|------:|
| 1K  | 31.4 | **17.7** | 9.5  | 29.4 | 11.6 |
| 4K  | 31.8 | **22.1** | 8.4  | 29.2 | 17.0 |
| 16K | 32.7 | **25.3** | 18.0 | 52.7 | 39.8 |
| 32K | 34.3 | **41.7** | 30.1 | 89.6 | 70.7 |
| 64K | 37.4 | **69.8** | 51.3 | 163.5 | 132.1 |

- **Fastest low-bit-KV decode (same-engine).** All HF-eager, OMMX beats **KIVI ×1.7→×2.3 (H200) /
  ×1.4→×2.4 (A100)** across 1K→64K, and passes Kitty by 64K (Kitty leads at short context). The `×N`
  figure label uses same-engine HF-eager on purpose — KIVI / Kitty have no vLLM path. Accuracy is a
  *separate* axis (Table 1 / Fig. 6).
- **Slower than 16-bit bf16 — the honest tradeoff.** OMMX's decode is dequant-bound (outlier-membership
  + unpack + scale/zp), so at long context it trails bf16, which does no dequant but pays ~6× the KV
  memory. The memory win is **shared** — OMMX, KIVI and Kitty all keep the KV quantized in HBM; OMMX's
  edge over them is decode speed + accuracy, not a smaller cache.
- **vLLM CUDA-graph ≈ 1.9× faster than HF-eager** for OMMX (9.5 vs 17.7 ms @1K).

**KV recipe ↔ fakequant parity.** The serving kernel and the `ommx_fakequant/` accuracy model share the
canonical recipe: CoQA f1 matches within 0.01, and WikiText-2 4K-chunked PPL is bit-identical between the
open extract and upstream. The served path is `ommx_gpu_serve` (`kv_store.py` + `pack.py` + `codec.py`);
`ommx_fakequant/` is the separate accuracy simulator.

## Weight-quant linear kernel (standalone microbench)

Separate from the KV story, the repo also ships OMMX's **i2f4 weight-quant linear** GEMM kernel (INT2
base + sparse FP4 **weight** outliers, sm80/sm90a) as a standalone latency + correctness microbench. It
is *not* wired into the serving path; it is the **weight**-quant axis, orthogonal to the KV story. Code
+ scope: [`ommx_gpu_serve/csrc/linear/`](ommx_gpu_serve/csrc/linear/).

![weight-quant linear microbench](figure/fig_linear_tpot_h200.png)

Decode-linear latency (batch `M` = 1–16) of one projection GEMM `x[M,K] @ W[N,K]ᵀ` on H200, vs bf16
(cuBLAS), LLM.int8 (bnb), and CUTLASS INT4 (the `ex55` mixed-input **reference** example, timed by its
own profiler). **Latency only — no accuracy or iso-bit claim.**

- **Where OMMX wins:** only at **npv = 4** (6.25% outliers, 3.06 b/wt), small **M**, on the **square** and
  **down** projections — square M=1 **0.037 vs 0.040 ms** (×1.10 over ex55), down M=1 **0.101 vs 0.125 ms**
  (×1.24), for M ≤ 4. It **loses** at npv = 8/16, on gate/up, and at M ≥ 8 (slower than bf16 too).
- **Not iso-bit:** npv4 (3.06 b/wt) is ~26% fewer bits than INT4 (4.125 b/wt). `ex55` is a reference GEMM,
  **not** the tuned Marlin/Machete W4A16 kernel vLLM ships, and OMMX here is the **eager 2-launch** path.
- **Correctness:** the kernel's i2f4 dequant is **bit-exact** to the in-repo reference packer, gated by
  [`test_ommx_linear_parity.py`](ommx_gpu_serve/csrc/linear/test_ommx_linear_parity.py) — base and
  base+FP4-outlier decode + prefill, cos ≥ 0.999; PASS on sm80/sm90a.

## Layout

```
open_ommx_serve/
├── ommx_fakequant/   # OMMX fake-quant (weight + KV) — accuracy sim for lm_eval / RULER
├── ommx_gpu_serve/   # OMMX Triton paged-decode KV kernel + vLLM integration + HF-eager path
│   └── csrc/linear/  #   standalone i2f4 weight-quant linear microbench (not wired into serving)
├── baseline/         # KIVI / Kitty comparison drivers
├── eval/             # trimmed lm-eval (RULER + kivi/kitty/ommx_hf adapters)
├── figure/           # bench.py + collect.py + plot.py + measured data + figures
├── assets/           # demo recordings
└── run.sh            # single entry: setup / bench / figure / all
```

## Setup

KIVI, Kitty and vLLM pin mutually-incompatible `transformers`/engine versions, so each method gets its
own venv ([`uv`](https://github.com/astral-sh/uv) recommended). `./run.sh setup` prints the exact commands:

```bash
uv venv .venv-ommx     --python 3.10 && uv pip install -e '.[ommx]'  && uv pip install -e ommx_gpu_serve && uv pip install -e eval
uv venv .venv-kivi     --python 3.10 && uv pip install -e '.[kivi]'  && uv pip install -e eval
uv venv .venv-kitty    --python 3.10 && uv pip install -e '.[kitty]' && uv pip install -e eval   # + external kitty pkg -> KITTY_PKG_PATH
```

External pieces (not vendored): KIVI's CUDA gemv (optional; Triton fallback works) and the
[Kitty](https://github.com/Summer-Summer/Kitty) package (`KITTY_PKG_PATH`). The `OMMX (vLLM)` breakdown
needs **no** weight bundle — it is the bf16-weight + OMMX-KV attention path.

## Reproduce

```bash
# B=1 decode TPOT/TTFT + peak-mem + quality for every method, then build the figures:
./run.sh all --gpu 0 --tag h200 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --ctxs 1024,4096,16384,32768,65536 \
  --methods "bf16 ommx kivi kitty vllm_bf16 ommx_vllm"

# add the vLLM engine arms (bf16 vLLM + OMMX stacked breakdown) — no weight bundle needed:
./run.sh bench --gpu 0 --tag h200 --methods "vllm_bf16 ommx_vllm"
./run.sh figure --tag h200
```

`run.sh` writes `figure/data_<tag>/<method>_hf.json` (+ `vllm_bf16.json` / `ommx_vllm.json`), then
`figure/collect.py` normalizes them into `figure/data/<tag>.json` and `figure/plot.py` renders
`fig_tpot_<tag>.png` + `fig_peakmem_<tag>.png`. RULER-32K accuracy runs through `eval/` (task `ruler`
@ 32768).

**Runnable correctness checks:** `ommx_gpu_serve/hf_eager/_ommx_hf_batch_test.py` (OMMX-KV greedy parity
vs bf16, needs a GPU + gated weights) and
`ommx_gpu_serve/csrc/linear/test_ommx_linear_parity.py` (weight-kernel format parity, cos ≥ 0.999).

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
