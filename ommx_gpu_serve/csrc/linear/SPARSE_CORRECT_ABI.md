# `sparse_correct` — the two call conventions, and the one that had no caller

**Status: DERIVED FROM SOURCE on a host with no GPU, no CUDA toolkit and no network.**
Every claim about the kernel below carries a `ommx_linear.cu:<line>`. Nothing here is a
measurement. The M>1 convention it documents is **UNVERIFIED on hardware** — see §6 for
exactly what would falsify it.

This note exists because a *passing* gate hid a *fatal* bug for the entire life of the
weight-quant serving path.

---

## 1. The defect, and why 5/5 PASS did not catch it

`ommx_linear.cu:2392-2393`, the last two lines of `sparse_correct`:

```c++
if (M == 1) { TORCH_CHECK(A_opt.has_value(),  "M==1 correct needs A");  go1(nc::StoreI2F4{}); }
else        { TORCH_CHECK(At_opt.has_value(), "M>1 correct needs At"); goM(nc::StoreI2F4{}); }
```

`sparse_correct` is two functions behind one name. The argument list is selected by `M`.

`integration/vllm/linear_method.py` sent the `M == 1` list unconditionally:

```python
mod.sparse_correct(y, x2d, None, code, scale, oindex, ocode, map_scale, map_center, ...)
#                     ^A_opt ^At_opt = None
```

Positionally that is `A_opt = x2d`, `At_opt = None`. Correct at M=1. At every M>1 it hit
`TORCH_CHECK(At_opt.has_value())` and **aborted** — so **every prefill step of every
recipe with `npv > 0` died**, i.e. every outlier recipe this repo ships.

`test_ommx_linear_parity.py` reported `PARITY GATE: PASS`, 5/5, throughout. Its cause is
structural and worth stating plainly: the gate contained **one** `sparse_correct` call
site and it was at `M=1`. It exercised one arm of an `if (M == 1) ... else ...` and
reported on "sparse_correct". Coverage of a branch is not coverage of an ABI.

The fix is in `linear_method.apply()`; the gate now carries M>1 cases
(`decode M=8` and `prefill M=32`, both `base+outlier`) so the *file that is the repo's
worked example of this ABI* stops teaching only half of it.

---

## 2. The contract, from the source

```c++
// ommx_linear.cu:2340
void sparse_correct(
    torch::Tensor out_or_C, c10::optional<torch::Tensor> A_opt,
    c10::optional<torch::Tensor> At_opt,
    torch::Tensor code, torch::Tensor scale, torch::Tensor oindex, torch::Tensor ocode,
    torch::Tensor map_scale, torch::Tensor map_center,
    int64_t N, int64_t M, int64_t K, int64_t vector_length, int64_t npv, int64_t B,
    const std::string& fmt);
```

| | `M == 1` (`go1`, `:2355`) | `M > 1` (`goM`, `:2363`) |
|---|---|---|
| activation argument | `A_opt` | `At_opt` |
| activation layout | `[1, K]` row-major | **`[K, M]`, i.e. `A` TRANSPOSED** |
| activation dtype | bf16 (`A_opt->data_ptr<at::BFloat16>()`, `:2357`) | bf16 (`At_opt->data_ptr<at::BFloat16>()`, `:2364`) |
| contiguity | required, **unchecked** | required, **unchecked** |
| base/result buffer | `out_or_C` **fp32** `[1, N]` (`out_or_C.data_ptr<float>()`, `:2362`) | `out_or_C` **bf16** `[M, N]` (`out_or_C.data_ptr<at::BFloat16>()`, `:2365`) |
| accumulation | `out[n] += corr` (`:1896`, fp32 in place) | `C[idx] = __float2bfloat16(__bfloat162float(C[idx]) + acc)` |
| kernel | `sparse_correct_gemv_relidx7_kernel` (`:1857`) | `sparse_correct_batched_relidx7_parallel_kernel` (`:1984`, default) or `..._kernel` (`:1905`, `OMMX_W_CORR_PARALLEL=0`) |

### 2.1 Why `At` is `[K, M]` contiguous and nothing else

Both M>1 kernels read the activation exactly once, here:

```c++
// ommx_linear.cu:1955 (D2)   and   :2024 (D3, the default)
acc = fmaf(__bfloat162float(At[(size_t)k_sh[s] * M + m]), d_sh[s], acc);
```

`k_sh[s]` is a column index in `[0, K)` (it is the decoded outlier position) and `m` is a
row of the batch in `[0, M)`. The element the kernel wants is `A[m, k]`, and it reaches it
at flat offset `k*M + m`. That is the layout of a **dense, row-major `[K, M]`** buffer, and
only that. Both declarations say so in a comment as well — `const __nv_bfloat16*
__restrict__ At,  // [K, M]  (A^T, coalesced col reads)` (`:1906`, `:1985`).

**This is not ambiguous, but it is also not enforced.** `sparse_correct` calls
`check_contig_cuda` on nothing (contrast `decode_base`, `:2081`, which checks A, code and
scale, and `sparse_correct_ws`, `:2179`). So:

* `x2d.t()` alone is a **stride-(1, K) view** whose `data_ptr()` is still A's original
  row-major storage. The kernel would apply a `k*M + m` index to `A`'s `m*K + k` layout,
  read the wrong elements, and return a finite, plausible, **wrong** number with no error
  anywhere. `.contiguous()` is load-bearing.
* A buffer of the right *shape* but the wrong *content* (e.g. `x2d.reshape(K, M)`) is
  equally invisible to the kernel. The CPU stand-in run of the new parity cases scores
  `cos = 0.80–0.82` for that mistake — comfortably caught by `cos >= 0.999`, comfortably
  missed by "it ran and the numbers look like numbers".

One thing the kernel genuinely *cannot* see: `At_opt`'s `shape`. It only does pointer
arithmetic, so any buffer of `>= K*M` bf16 elements in `k*M + m` order would work.
`linear_method` and the gates assert the exact `[K, M]` anyway — stricter than the kernel,
deliberately, because the shape is the only cheap place to catch a caller that transposed
the wrong axis.

Likewise `M` is a plain `int64_t` argument and is **never checked against a tensor**. A
caller that passes an `M` disagreeing with `At`'s rows gets out-of-bounds reads.

### 2.2 Why `C` is bf16 where `out` was fp32

`go1` takes the fp32 accumulator `decode_base` returns
(`torch::zeros({M, N}, A.options().dtype(torch::kFloat32))`, `:2083`). `goM` does not —
it dereferences `out_or_C.data_ptr<at::BFloat16>()`, which *throws* on an fp32 tensor.
So the base kernel's return dtype decides whether a cast is needed:

| base kernel | returns | line | cast before `sparse_correct`? |
|---|---|---|---|
| `prefill_wmma` (M>=16) | **bf16** `torch::empty({M, N}, A.options())` | `:2410` | no |
| `decode_base` (M<=16) | **fp32** `torch::zeros(..., kFloat32)` | `:2083` | **yes**, for 1 < M < 16 |

Rounding the base to bf16 *before* adding the correction rather than after is a real
numeric difference from the M==1 path. It is the M>1 kernel's own contract (its update
reads, adds in fp32 and stores back through bf16), the prefill path has always had it, and
`apply()` casts to the activation dtype on the way out regardless.

### 2.3 The other host-side bound

Only the **default** (`OMMX_W_CORR_PARALLEL` unset) branch is guarded:

```c++
// ommx_linear.cu:2369  — inside `if (corr_parallel)`
TORCH_CHECK(n_blk * npv <= MAX_STAGE_DENSE, "sparse_correct(parallel): n_blk*npv=", ...);
```

`OMMX_W_CORR_PARALLEL=0` selects the legacy D2 kernel, which has **no such host check** and
instead clamps in device code — `for (int i = 0; i < cnt && base + i < MAX_STAGE; ++i)`
(`:1945`) and `const int cnt = s_count < MAX_STAGE ? s_count : MAX_STAGE;` (`:1951`). Past
`n_blk*npv = 5120` that path **silently drops corrections**. Nothing in the serving path
reaches it today (the shipped geometries are far below the bound, and the env var is a
GPU-A/B knob), but anyone who sets that variable is one geometry away from a silent
accuracy loss with a green gate.

---

## 3. The correct call, both halves

```python
# M == 1 — unchanged, and the only form the gate measured before this release
y = mod.decode_base(x, code, scale, zp, N, K, gs, False, "i2f4", 1)      # fp32 [1,N]
mod.sparse_correct(y, x, None, code, scale, oindex, ocode, ms, mc,
                   N, 1, K, gs, npv, gs, "i2f4")

# M > 1
C  = mod.prefill_wmma(x, code, scale, zp, N, M, K, gs, False, "i2f4", 1)  # bf16 [M,N]
#  … or, for 1 < M < 16:
C  = mod.decode_base(x, code, scale, zp, N, K, gs, False, "i2f4", 1).to(torch.bfloat16)
At = x.t().contiguous()                                                  # bf16 [K,M]
mod.sparse_correct(C, None, At, code, scale, oindex, ocode, ms, mc,
                   N, M, K, gs, npv, gs, "i2f4")
```

`B` (the second `gs`) is the outlier BLOCK size, and `block == group` on the weight path:
the packer writes one relidx7 slot stream per group (`../../linear/w_packer.py`).

---

## 4. The cost this convention imposes, stated rather than hidden

`x.t().contiguous()` is a **full transpose plus copy of the activation, every forward that
carries outliers**: `M*K*2` bytes read and `M*K*2` written, before any correction work
begins. At Llama-3.1-8B prefill (M=4096 tokens, K=4096) that is 32 MiB moved twice per
linear, seven linears per layer, thirty-two layers.

It is correct, it is what the kernel demands, and it is a performance defect. The repo
already contains the entry points that would remove it — they take `A` row-major and need
no transpose at all:

| entry point | line | M range | takes |
|---|---|---|---|
| `prefill_wmma` with its `oindex_opt` / `odelta_opt` overlay | `:2398` | `>= 16` | `A` `[M, K]`, single-pass fused i2f4 |
| `sparse_correct_packed` | `:2246` | `[1, 16]` | `A` `[M, K]` + precomputed `kpos`/`odelta` |

Switching to any of them is a **measured-perf decision on a device**, not a reading
decision, so it is named here and left undone rather than taken blind. Whoever takes it
should note that `sparse_correct_packed` is a **third** activation convention (row-major
`A`, like `go1`, unlike `goM`) — this family has no single answer and each entry point
must be read.

---

## 5. Gates

**CPU, no device** — `tests/test_linear_method.py`:

| test | pins |
|---|---|
| `test_apply_at_m_gt_1_uses_the_kernels_m_gt_1_call_convention` (M=17, 32) | `A_opt` NULL, `At_opt` contiguous `[K, M]` bf16 reproducing the activation under the kernel's own `At[k*M+m]` walk, `C` is `prefill_wmma`'s bf16 tensor |
| `test_apply_at_batched_decode_gives_sparse_correct_the_bf16_C_it_dereferences` | the fp32→bf16 `C` cast on the 1 < M < 16 path |
| `test_apply_at_m_1_still_makes_the_exact_call_the_parity_gate_measured` | M==1 is untouched, by tensor IDENTITY |
| `test_the_m_gt_1_operands_helper_refuses_what_the_kernel_cannot_read` | the helper's own refusals |
| `test_the_routing_threshold_is_bounded_by_both_kernel_limits` | `OMMX_W_PREFILL_MIN_M` in `[16, 17]` (§7) |
| `test_an_outlier_free_recipe_never_reaches_sparse_correct_at_any_m` | `npv=0` still takes the base-only path at every M |

These run against `_KernelABISpy`, a transcription of what the `.cu` dereferences. Each was
mutation-proven: reverting the fix, dropping `.contiguous()`, substituting a
same-shape/wrong-content buffer, dropping the bf16 cast, and making `C` a copy each make
them FAIL.

**GPU** — the numbers, which no CPU can supply:

* `csrc/linear/test_ommx_linear_parity.py`, cases 6 and 7:
  `decode M=8 base+outlier` and `prefill M=32 base+outlier`, `cos >= 0.999` against the
  same fp32 reference `x @ W_ref^T` the first five use.
* `tests/test_linear_method.py::test_gpu_apply_at_m_gt_1_with_outliers_matches_the_cpu_dequant_oracle`
  — `apply()` end to end at M=8 and M=32 against `w_format.load_weight`'s CPU oracle.
  `gpu`-marked, so it SKIPS with a reason and cannot pass vacuously.

---

## 6. What would falsify this

The M>1 convention is a **reading**, not a measurement. It is wrong if, on first contact
with a device:

* either new parity case reports `cos < 0.999` **while `decode M=1 base+outlier` in the
  same process still reports `1.00000`**. That isolates the fault to the M>1 argument
  construction rather than to the packer, the map or the base kernel. The most likely
  alternative reading is that `At` should be `[M, K]` (no transpose) — try
  `At = x.contiguous()` and see whether cos recovers; if it does, §2.1 is wrong and this
  document, `linear_method.sparse_correct_m_gt_1_operands` and both gates must change
  together.
* the M>1 cases abort inside `data_ptr` — then §2.2's dtype table is wrong.
* `cos` is fine at `M=32` but not at `M=8`, or vice versa — then the fault is the bf16
  `C` cast (the only difference between those two cases), not the transpose.

`cos` first, in every case (law #4): `cos >= 0.999` with a `max_diff` complaint is a
tolerance question, not a correctness one.

---

## 7. Neighbouring finding: the routing threshold has a silent side

`decode_base` (`:2075`) has **no upper `TORCH_CHECK` on M**. Its batched launcher tops out
at `decode_gemv_batched_kernel<SF, 16>` (`:2106`) and that kernel's epilogue is

```c++
// ommx_linear.cu:439-445, decode_gemv_batched_kernel's epilogue
for (int m = 0; m < M_MAX; ++m) { ...
    if (lane == 0 && m < M) out[(size_t)m * N + n] = sum;
}
```

At `M > 16`, rows `16 .. M-1` are never written — and `out` came from `torch::zeros`, so
those tokens are served an **all-zero activation with no error**. `prefill_wmma` bounds the
other end loudly (`TORCH_CHECK(M >= 16, ...)`, `:2409`).

`OMMX_W_PREFILL_MIN_M` is therefore bracketed in Python at `[16, 17]`
(`linear_method.PREFILL_MIN_M_BOUNDS`) — 17 stays legal because it routes `M==16` to
`decode_base` (all 16 rows written) and `M>=17` to `prefill_wmma`. Refused, not clamped.

---

## 8. Related

* `BITMAP_READER_SPEC.md` — a native bitmap position reader. Any such reader lives inside
  `relidx7_slot_delta`'s callers and therefore inherits **both** conventions above
  unchanged; §5.3 of that spec lists the entry points it must touch.
* `test_ommx_linear_parity.py`'s `quantize_ommx_weight` — the packer/dequant contract
  these kernels reproduce, and the reference `W_ref` both new M>1 cases are scored against.
  (`ommx_linear.cu:23` names an `OMMX_LINEAR_CONSISTENCY.md`; that file is **not** in this
  checkout, so the parity gate's packer is the contract of record here.)
