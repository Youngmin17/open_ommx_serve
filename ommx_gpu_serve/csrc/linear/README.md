# OMMX i2f4 linear — sm90 wgmma kernel + memory-bound decode-latency comparison

This directory adds OMMX's **i2f4 weight-quant linear** GEMM kernel (INT2 base + sparse FP4 weight
outliers) for Hopper (sm90) and a **memory-bound decode-latency** comparison against CUTLASS INT4,
LLM.int8, and bf16. It complements the KV-cache attention story elsewhere in this repo (this is the
**weight**-quant linear axis, not KV).

> **Scope — standalone microbench.** This weight-quant kernel is **not wired into the vLLM serving
> path** in this release (the KV-quant `CUSTOM` attention backend is; there is no `ommx_w`
> `LinearMethodBase`). It is a self-contained latency + correctness microbench. Correctness is
> runnable and passing: `test_ommx_linear_parity.py` (base decode/prefill + FP4-outlier decode,
> cos ≥ 0.999, firing > 0; PASS on sm80/sm90a) and the format cross-check against the real
> `ommx_fakequant.weight_quantizer` (cos = 1.0, max_abs ≈ 6e-8 for npv 4/8/16).

## What is measured

Decode-linear latency (batch `M` = 1..16, the memory/launch-bound regime) of one projection GEMM
`x[M,K] @ W[N,K]^T` on H200, for four method families:

| arm | weight format | bits/weight (gs=64) | how timed |
|-----|---------------|---------------------|-----------|
| **bf16** | dense bf16 | 16 | cuBLAS, in-process CUDA-event median |
| **LLM.int8** | INT8 (bnb Linear8bitLt) | 8 | in-process |
| **CUTLASS INT4** | INT4×bf16 mixed-input | 4.125 | **live** ex55 binary, its internal profiler |
| **OMMX i2f4** | INT2 base + FP4 outliers (npv 4/8/16) | 3.06 / 3.75 / 5.125 | in-process |

`npv` = FP4 outliers per group of 64 = 6.25% / 12.5% / 25%.

## Correctness — format parity with the accuracy-eval fakequant (the only correctness claim)

The kernel's i2f4 dequant is **bit-exact** to the real fakequant weight-quant used for accuracy eval:
`quantize_ommx_weight(W, gs, opct).W_ref` == `fakequant.quant_weight.weight_quantizer(W, bits=2,
group_size=gs, outlier_percent=opct, use_pow2=True, act_scales=None, mode="decode")` →
**cos = 1.000000, max_abs_diff ≈ 3e-8** (the INT2 base alone is bit-identical). The four format axes
match: (1) group scale = range-excluding-outliers / 3 then E8M0 pow2; (2) asymmetric zero-point =
group-min; (3) outlier count `npv = max(1, int(gs·pct))` over block==group; (4) FP4 = idx-space E2M1
range-map, fp_range = 12. `bench_linear_memory_bound.py` runs this parity assert at startup.

**Two config axes must be held to keep parity:** the accuracy numbers this kernel serves must come from
a `weight_quantizer(use_pow2=True)` run at the **same group_size**. The eval's default `--pow2` is OFF
(fp16-relaxed scale, cos 0.97 vs the E8M0 kernel) and its default `group_size` is 16; a fair accuracy
number must be regenerated at `--pow2`, gs=64. Also: the served accuracy path selects outliers by
`|W|·act_scales` (calibration saliency) while this bench uses pure `|W|` — the **outlier count (npv) is
identical** so latency is unaffected, but the two do not select the *same* weights as outliers, so this
bench's `W_ref` is not the bit-identical served weight under calibration.

## Honest scope of the latency comparison (read before citing a number)

This is a **latency-only** comparison. It makes **no accuracy or iso-bit claim**. In particular:

- **Not iso-bit.** OMMX npv4 (3.06 b/wt) uses ~26% fewer bits than CUTLASS INT4 (4.125 b/wt). Any OMMX
  latency win at npv4 is partly a bit-budget effect. At the near-iso-bit point (npv8 ≈ 3.75 b) OMMX does
  not win; at npv16 (5.125 b) OMMX is slower than bf16 at larger M.
- **CUTLASS = ex55 reference example**, not the tuned Marlin/Machete W4A16 kernel vLLM actually ships.
  ex55 runs a fixed GEMM tile (flat latency across M) and is timed by its own internal profiler (not the
  same CUDA-event timing as OMMX). Against Marlin the OMMX small-M win would likely shrink.
- **OMMX here is the eager 2-launch path** (`cute_base` INT2 wgmma + `sparse_correct_packed` correction).
  Served OMMX runs under a full CUDA graph; the eager launch overhead here is not the served TPOT.
- The accuracy axis is untestable on the synthetic Gaussian weights used here (they have no heavy tails,
  so every npv scores cos≈1.0 vs the self-quantized reference — that is mechanic parity, not accuracy).

**Defensible headline (measured, H200 NVL, live ex55):** *in the memory-bound regime (small M),
OMMX i2f4 at 6.25% outliers (npv4) has lower median latency than the CUTLASS ex55 INT4 example on the
square and down projections — square M=1: **0.0367 vs 0.0404 ms** (1.10×), M=2: 0.0374 vs 0.0404;
down M=1: **0.1009 vs 0.1248 ms** (1.24×), M≤4. This is a memory-bound, non-iso-bit result (npv4 =
3.06 b/wt < INT4 4.125 b/wt); the kernel format is bit-exact to the fakequant i2f4 (cos=1.0 at
npv 4/8/16).* OMMX loses at npv8/16, on gate/up, and at M≥8 (there it is slower than bf16). Do **not**
state "OMMX beats CUTLASS" unconditionally, "iso-accuracy", or a bf16 win.

## Reproduce

```bash
# needs a CUTLASS 4.4.2 tree (flashinfer bundle) + the compiled ex55 example for the CUTLASS arm.
export OMMX_CUTLASS_DIR=/path/to/cutlass         # 4.4.2 with tools/util mixed_dtype_utils.hpp
export OMMX_CUTLASS_EX55_BIN=/path/to/55_hopper_mixed_dtype_gemm   # optional; else searched
bash run_linear_bench.sh --gpu 0 --tag h200
# -> figure/fig_linear_tpot_<tag>.png + figure/data_<tag>/linear_mem_bound.json
```

The kernel builds via `build_ommx_linear.py` (torch JIT, sm_90a, `OMMX_ENABLE_SM90_CUTLASS=1`). The
sm90 **fused** wgmma-mainloop i2f4 kernel (`ommx_sm90_mixed_input_fused.hpp`, opt-in `OMMX_CUTE_FUSED=1`)
is included: it fuses the FP4-outlier correction into the wgmma mainloop (cos=1.0, 3 implementations)
but is **not** the fastest decode path — see the fusion notes; the decoupled `base + sparse_correct_packed`
is used for the latency numbers above.
