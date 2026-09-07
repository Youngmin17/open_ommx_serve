# OMMX i2f4 linear — sm90 wgmma kernel + memory-bound decode-latency comparison

This directory adds OMMX's **i2f4 weight-quant linear** GEMM kernel (INT2 base + sparse FP4 weight
outliers) for Hopper (sm90) and a **memory-bound decode-latency** comparison against CUTLASS INT4,
LLM.int8, and bf16. It complements the KV-cache attention story elsewhere in this repo (this is the
**weight**-quant linear axis, not KV).

> **Scope — standalone microbench. STALE CLAIM CORRECTED.** This note used to read "not wired into
> the vLLM serving path ... there is no `ommx_w` `LinearMethodBase`". That is no longer true of the
> tree: `integration/vllm/linear_method.py` now defines an `ommx_w` `LinearMethodBase` and
> `plugin.py::_register_ommx_w_quant` registers it on every plugin load. Two defects that made the
> weight bundle *unloadable by any engine* regardless of that wiring have also been fixed in
> `ommx_gpu_serve/linear/` — plane names dropped the `.weight` infix so they bind to a vLLM Linear
> parameter (`OMMX_W_SafeTensor` version 1 -> 2), and `w_packer pack` now copies the source
> checkpoint's `config.json` / tokenizer / template files through, so the bundle directory is a
> model directory. Both are **CPU-verified at the artefact level** (names read back off a packed
> shard; files compared byte-for-byte), and both are **GPU-UNVERIFIED as load events**: no vLLM
> engine has ever opened one. What IS still true, and is
> what matters for every number below: **that serving path has never run on a device** (its own
> header marks the whole execution side `UNVERIFIED (no GPU this session)`), and **not one latency
> figure in this file came through it** — they are all from the standalone
> `bench_linear_memory_bound.py` two-launch microbench. Treat the numbers here as microbench
> numbers, never as served ones. Correctness is
> runnable and passing: `test_ommx_linear_parity.py` (base decode/prefill + FP4-outlier decode,
> cos ≥ 0.999, firing > 0; PASS re-verified on sm90a / H100 NVL — sm80 is supported by the
> build but has no recorded transcript) and the format cross-check against the real
> `ommx_fakequant.weight_quantizer` (cos = 1.0, max_abs ≈ 6e-8 for npv 4/8/16).

## What is measured

Decode-linear latency (batch `M` = 1..16, the memory/launch-bound regime) of one projection GEMM
`x[M,K] @ W[N,K]^T` on H200, for four method families:

| arm | weight format | bits/weight | how timed |
|-----|---------------|-------------|-----------|
| **bf16** | dense bf16 | 16 | cuBLAS, in-process CUDA-event median |
| **LLM.int8** | INT8 (bnb Linear8bitLt) | 8 | in-process |
| **CUTLASS INT4** | INT4×bf16 mixed-input | 4.125 **at g=128** | **live** ex55 binary, its internal profiler |
| **OMMX i2f4** | INT2 base + FP4 outliers (npv 4/8/16) | 4.125 / 4.750 / 6.125 **at gs=64** | in-process |

`npv` = FP4 outliers per group of 64 = 6.25% / 12.5% / 25%.

### Where the OMMX bits/weight come from (corrected)

The three OMMX figures are the **stored** budget of the shipped recipe, recomputed with the
packer's own budget path rather than by hand:

```
$ python3 -m ommx_gpu_serve.linear.w_packer budget --group-sizes 64 --npvs 4 8 16
  gs  npv   %out  repr     map          stored  unpadded
  64    4   6.2%  relidx7  idx_range    4.1250    4.0625
  64    4   6.2%  relidx7  none         3.1250    3.0625
  64    4   6.2%  bitmap   idx_range    4.6250    4.6250
  64    4   6.2%  bitmap   none         3.6250    3.6250
  64    8  12.5%  relidx7  idx_range    4.7500    4.7500
  64    8  12.5%  relidx7  none         3.7500    3.7500
  64    8  12.5%  bitmap   idx_range    4.8750    4.8750
  64    8  12.5%  bitmap   none         3.8750    3.8750
  64   16  25.0%  relidx7  idx_range    6.1250    6.1250
  64   16  25.0%  relidx7  none         5.1250    5.1250
  64   16  25.0%  bitmap   idx_range    5.3750    5.3750
  64   16  25.0%  bitmap   none         4.3750    4.3750
```

The shipped recipe is the **first row of each npv block** — `relidx7` positions plus the
`idx_range` FP4 range map, which is what `w_format.Recipe` defaults to
(`outlier_repr="relidx7"`, `outlier_map="idx_range"`) — so **4.125 / 4.750 / 6.125 b/wt**.
Per-plane, at npv=4: `base_int2 2.0 + scale_e8m0 0.125 + zero_point 0.25 + outlier_index 0.5 +
outlier_fp4 0.25 + fp4_range_map 1.0 = 4.125` (the keys are `Recipe.bits_breakdown()["stored"]`).

**This file used to publish 3.06 / 3.75 / 5.125, which was wrong twice over:**

* **It omitted the range map.** Those are the `map = none` rows. The shipped `idx_range`
  encoding stores two more F32 planes per group, `map_scale` and `map_center` (`[N, G]` each,
  `w_format.PLANES_MAP`), and the timed kernel takes both as arguments — `ommx_linear.cu`
  declares `const float* __restrict__ map_scale, map_center` on every correction entry point,
  and `bench_linear_memory_bound.py` feeds them (`msc = q["map_scale"]`, `mcen = q["map_center"]`).
  Two F32 per 64 weights is **+1.0 bit/weight exactly** at gs=64 — the `fp4_range_map` term.
* **At npv=4 it quoted the unpadded column.** Four `relidx7` positions are 4×7 = 28 bits, and the
  layout pads the position stream to a whole byte per group (4 bytes) because the kernel's
  `relidx7_slot_pos` indexes it from a byte boundary. Stored is **3.125, not 3.0625** with no map,
  and **4.125, not 4.0625** with it. At npv=8/16 the stream is already byte-aligned (56 and 112
  bits) so `stored == unpadded` there — which is why only the npv=4 figure moves for this reason.

`stored` is what the packer writes to disk; `unpadded` is the information-theoretic figure
(`2 + 8/gs + 16/gs + npv*11/gs` at relidx7, no map). **Only `stored` is a memory-traffic
statement**, so `stored` is what this file publishes. This is the same class of accounting error
the KV side already corrected: `integration/vllm/packed_only.py`'s pre-fix formula also omitted
the FP4 map planes and reported 4.125 avg bit/elem where the real allocation, summed from the
allocated pool tensors, is 4.375.

The paper's weight AvgBits 3.63 (claim E2) is **not** this arm's budget. 3.6250 is the
`bitmap` + `map = none` family (the `64 / 4 / bitmap / none` row above), and it is not uniquely
reachable: an exhaustive scan of every `(gs, npv, repr, map)` over `{16,32,64,128} × [0..gs]`
with the paper's BF16 zero-point returns exactly five recipes at 3.6250 stored — `(64,1,relidx7,
idx_range)`, `(64,3,bitmap,none)`, `(64,4,bitmap,none)`, `(128,13,bitmap,none)`,
`(128,14,bitmap,none)`. `linear/README.md` §4 carries that scan and
`test_paper_3_63_avgbits_is_the_bitmap_recipe` pins the whole solution set. This directory ships
`relidx7 + idx_range` and lands on 4.125, so nothing here should be read as a 3.63 result.

The 4.125 / 4.750 / 6.125 figures published above are pinned by
`ommx_gpu_serve/tests/test_w_packer.py::test_readme_published_bit_budgets_reproduce`, which also
pins the exact size of both corrections (the +1.0 b/wt range map, and the 1/16 b/wt npv=4
padding), so this file and the packer's accounting cannot drift apart again in either direction.

### What the timed path actually streams (a third, larger number)

The 4.125 / 4.750 / 6.125 above are the **on-disk bundle**. The microbench does not hand the
bundle to the timed kernel: it pre-materializes the correction at load time, so the bytes the
timed launches stream per weight element are different again. From the shapes in
`bench_linear_memory_bound.py` and `build_kpos` / `build_odelta` in `ommx_linear.cu` —
`kpos` is `int32 [N, G*npv]` (absolute k, decode-free) and `odelta` is `bf16 [N, G*npv]`, and the
base GEMM is fed `scales_bf` / `zeros_bf`, i.e. the E8M0 scale is uploaded as **bf16**, not as the
int8 plane:

| npv | base | scale | zp | kpos | odelta | streamed total | on-disk stored |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2.0000 | 0.2500 | 0.2500 | 2.0000 | 1.0000 | **5.5000** | 4.1250 |
| 8 | 2.0000 | 0.2500 | 0.2500 | 4.0000 | 2.0000 | **8.5000** | 4.7500 |
| 16 | 2.0000 | 0.2500 | 0.2500 | 8.0000 | 4.0000 | **14.5000** | 6.1250 |

That is arithmetic from the tensor layout, **not** a measured DRAM counter — UNVERIFIED (no GPU
this session); nothing here re-times anything. It is stated because it is the accounting that
matches the latency numbers: the decode-free correction trades bits for launches, and at npv=16
the timed path streams 14.5 b/wt against bf16's 16, which is the same direction as the measured
npv16 loss to bf16 at larger M.

**The group size is NOT matched across arms.** The CUTLASS arm is `ex55` as it actually ran, which
is g=128 (`cutlass_note` in `figure/data_<tag>/linear_mem_bound.json`: *"ex55 mixed-input INT4xbf16
g=128"*), so its 4.125 = 4 + 16/128; the same INT4 weight re-grouped to gs=64 would be 4.25 b/wt.
Every bits/weight comparison below is against the 4.125 the timed binary actually used.

## Correctness — format parity with the accuracy-eval fakequant (the only correctness claim)

The kernel's i2f4 dequant is **bit-exact** to the real fakequant weight-quant used for accuracy eval:
`quantize_ommx_weight(W, gs, opct).W_ref` == `fakequant.quant_weight.weight_quantizer(W, bits=2,
group_size=gs, outlier_percent=opct, use_pow2=True, act_scales=None, mode="decode")` →
**cos = 1.000000, max_abs_diff = 5.96e-08** (the INT2 base alone is bit-identical) — the value
recorded under `parity` in `figure/data_<tag>/linear_mem_bound.json`, i.e. the same ≈6e-8 quoted
in the scope note at the top of this file. The four format axes match: (1) group scale =
range-excluding-outliers / 3 then E8M0 pow2; (2) asymmetric zero-point = group-min; (3) outlier
count `npv = max(1, int(gs·pct))` over block==group; (4) FP4 = idx-space E2M1 range-map,
fp_range = 12. `bench_linear_memory_bound.py` runs this parity assert at startup.

**Two config axes must be held to keep parity:** the accuracy numbers this kernel serves must come from
a `weight_quantizer(use_pow2=True)` run at the **same group_size**. The eval's default `--pow2` is OFF
(fp16-relaxed scale, cos 0.97 vs the E8M0 kernel) and its default `group_size` is 16; a fair accuracy
number must be regenerated at `--pow2`, gs=64. Also: the served accuracy path selects outliers by
`|W|·act_scales` (calibration saliency) while this bench uses pure `|W|` — the **outlier count (npv) is
identical** so latency is unaffected, but the two do not select the *same* weights as outliers, so this
bench's `W_ref` is not the bit-identical served weight under calibration.

## Honest scope of the latency comparison (read before citing a number)

This is a **latency-only** comparison. It makes **no accuracy or iso-bit claim**. In particular:

- **The bit budgets, corrected — the npv4 win is NOT a bit-budget effect.** This bullet used to read
  "OMMX npv4 (3.06 b/wt) uses ~26% fewer bits than CUTLASS INT4 (4.125 b/wt)", which came from the
  wrong 3.06 (see *Where the OMMX bits/weight come from*). On the stored budget the two are the
  **same number**: OMMX npv4 = 4.1250 b/wt (gs=64, `relidx7` + `idx_range`) and the ex55 INT4 arm =
  4.125 b/wt (g=128, 4 + 16/128). On what the timed kernels actually stream OMMX is *more* expensive
  (5.5 b/wt at npv4). So the npv4 latency result cannot be explained away as OMMX spending fewer
  bits — but equal bits is **not** equal anything else: the group sizes differ (64 vs 128), the
  formats differ, and **no accuracy or iso-accuracy claim is made** (see the last bullet). npv8
  (4.750 b/wt) and npv16 (6.125 b/wt) both cost MORE than the INT4 arm; OMMX does not win at npv8,
  and at npv16 it is slower than bf16 at larger M.
- **CUTLASS = ex55 reference example**, not the tuned Marlin/Machete W4A16 kernel vLLM actually ships.
  ex55 runs a fixed GEMM tile (flat latency across M) and is timed by its own internal profiler (not the
  same CUDA-event timing as OMMX). Against Marlin the OMMX small-M win would likely shrink.
- **OMMX here is the eager 2-launch path** (`cute_base` INT2 wgmma + `sparse_correct_packed` correction),
  so the two launch overheads are inside every number above. Do **not** subtract them by assuming a
  served run amortizes them away: the `ommx_w` serving path that *would* amortize them
  (`integration/vllm/linear_method.py`) has **never executed on a device** — see the corrected scope
  note at the top — so there is no measurement in which they are amortized. The OMMX component that
  *is* served with a device transcript — the KV-quant `CUSTOM`
  attention — is deliberately kept **out** of the full CUDA graph on sm_90, the arch every published
  serving transcript in this release was measured on (`integration/vllm/backend.py`
  `OMMXCanonicalMetadataBuilder.get_cudagraph_support` returns `AttentionCGSupport.NEVER` for
  compute capability ≥ 9, because a FULL capture would freeze the per-step host metadata). There is
  therefore no measured served TPOT for this kernel to be compared against, on any arch.
- The accuracy axis is untestable on the synthetic Gaussian weights used here (they have no heavy tails,
  so every npv scores cos≈1.0 vs the self-quantized reference — that is mechanic parity, not accuracy).

**Defensible headline (measured, H200 NVL, live ex55):** *in the memory-bound regime (small M),
OMMX i2f4 at 6.25% outliers (npv4) has lower median latency than the CUTLASS ex55 INT4 example on the
square and down projections — square M=1: **0.0367 vs 0.0404 ms** (1.10×), M=2: 0.0374 vs 0.0404;
down M=1: **0.1009 vs 0.1248 ms** (1.24×), M≤4. This is a memory-bound result at an EQUAL stored bit budget
(npv4 = 4.125 b/wt, the same figure as the ex55 INT4 arm's 4.125 — the older "3.06 b/wt, ~26%
fewer bits" reading was an accounting error, corrected above), at unmatched group sizes (64 vs
128) and with no accuracy claim; the kernel format is bit-exact to the fakequant i2f4 (cos=1.0 at
npv 4/8/16).* OMMX loses at npv8/16 and on gate/up. **Against bf16 the loss starts far earlier than
M≥8**, which an earlier draft of this bullet understated: read straight off
`figure/data_h200/linear_mem_bound.json` at npv4, on `down` OMMX is slower than bf16 at *every* M —
0.10086 / 0.10237 / 0.10582 / 0.12477 / 0.17155 ms against bf16's 0.08602 / 0.08547 / 0.08502 /
0.08490 / 0.08541 at M = 1 / 2 / 4 / 8 / 16, i.e. 0.85× / 0.83× / 0.80× / 0.68× / 0.50×. `gate/up`
crosses at M=16 (0.76×) and `square` is still 1.06× there. So the bf16 comparison is shape-dependent
and `down` never wins it at any M. Do **not**
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
