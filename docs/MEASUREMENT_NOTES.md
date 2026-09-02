# Measurement notes

Everything here is provenance: what a number in this repository covers, how it was produced, and
what it does not license a reader to conclude. It is separated from the README because the README
should be readable in one sitting, not because any of it is optional. Each section is the record as
it was written when the measurement was made.

---

## 1. KV recipes and bit accounting

Each method uses **its own best low-bit-KV recipe** (not iso-bit):

| method | KV recipe | avg KV bits | bf16 tail |
|---|---|--:|---|
| **OMMX** | INT2 base + 12/64 FP4 K-outliers, pow2 scales | **4.375** | sink 8 + recent 32 |
| KIVI | 2-bit K/V, group 32 | ~2.55 | residual 128 |
| Kitty | 2-bit K/V + 25% INT4-channel boost | ~2.5 | sink 32 |

**OMMX is the highest-bit method here, not the lowest.** Its **4.375** average KV bits
(**K 6.00, V 2.75**) buy the paper's high-coverage accuracy — so the speed and memory numbers are
**not** an equal-bit comparison. OMMX trades ~1.7× more KV bits than KIVI / Kitty for accuracy.

> **The 4.375 figure is measured, not derived.** It is the sum of the tensors
> `MultiSeqKVPool` actually allocates for the canonical recipe (`k_base`, `v_main`, `k_scale`,
> `k_zp`, `v_scale`, `v_zp`, the two dedicated FP4 range-map planes, `k_oidx`, `k_oval`), divided
> by the stored KV elements: **8.750 bit per (K,V) element pair → 4.375 average, 3.66× vs bf16.**
> An earlier revision of this table said ~4.1 (3.88×) because the accounting helper omitted the
> two `k_fp4_mapscale` / `k_fp4_mapcenter` planes, which `OMMX_KV_OUTLIER_MAP` allocates **by
> default**. `ommx_gpu_serve/tests/test_bit_accounting.py` now pins the corrected numbers against
> an independent recomputation.
>
> **A "≤3-bit" OMMX exists but is a different recipe.** `group_tokens=64` + `group_channels=64` +
> `OMMX_KV_OUTLIER_MAP=0` measures 5.875 bit/pair (2.938 average, 5.45×). `attention/pack.py`
> calls it "its OWN fakequant oracle … a different number system" from the dedicated-map recipe.
> **The accuracy results in Table 1 / Fig. 6 were produced with the 4.375-bit recipe**, so the
> ≤3-bit configuration has no accuracy number in this repo and must not be quoted alongside them.

**KIVI's decode path — measured, not assumed.** The published KIVI numbers use its HF-eager `_eval`
decode path (Triton pack + dequant + `torch.matmul` GEMV). This repo also carries a purpose-built
GQA extension for KIVI's *fused* dequant+matmul kernel, behind `KIVI_FUSED_GQA=1`. Both were timed
on one H100 NVL, both paths fully pre-warmed, each cell run in both orders, with KIVI's own
`_kivi_fused_fires` counter proving which kernel executed (token output identical, 48/48):

| ctx | dequant fallback (default) | fused GQA | fallback / fused (>1 = fused faster) |
|----:|---------------------------:|----------:|-----------------:|
| 1K  | 67.98 / 70.88 ms | 60.24 / 59.27 ms | **1.16×** |
| 4K  | 66.63 / 68.33 ms | 106.24 / 107.55 ms | 0.63× |
| 16K | 144.10 / 130.03 ms | 360.51 / 338.76 ms | **0.39×** |

The default (fallback) is therefore the **faster** KIVI at every context ≥ 4K — running it is not a
handicap. `KIVI_FUSED_GQA=1` is available and `kivi_fused_stats()` records which path a run took.

**H200, decode TPOT (ms/step)** — HF-eager (OMMX also shown under vLLM):

| ctx | bf16 | **OMMX (hf)** | OMMX (vLLM) | KIVI | Kitty |
|----:|-----:|--------------:|------------:|-----:|------:|
| 1K  | 31.4 | **17.7** | 9.5  | 29.4 | 11.6 |
| 4K  | 31.8 | **22.1** | 8.4  | 29.2 | 17.0 |
| 16K | 32.7 | **25.3** | 18.0 | 52.7 | 39.8 |
| 32K | 34.3 | **41.7** | 30.1 | 89.6 | 70.7 |
| 64K | 37.4 | **69.8** | 51.3 | 163.5 | 132.1 |

> **How these numbers were produced — read before citing them.** The table is
> `figure/data/h200.json` verbatim: the already-published set, produced by an earlier
> `figure/bench.py`, so re-running the [Reproduce](#reproduce) commands does **not** reproduce
> it. Three things have to be said before any row is quoted.
>
> - **The bf16 column carries a `DynamicCache` handicap.** That arm used stock `transformers`
>   `DynamicCache`, which re-`torch.cat`s the whole KV every decode step — something no
>   quantized arm does. A same-process A/B on an **H100 NVL** (not this H200), the cache object
>   the only difference, measured `StaticCache` **3.09× / 2.59× / 3.17×** faster at 1K / 4K /
>   16K. An unhandicapped bf16 arm therefore lands *below* the OMMX (hf) column, i.e. **the
>   OMMX-beats-bf16 ordering in the rows above inverts at 1K–16K.** `figure/bench.py` now
>   defaults to `--bf16-cache static`.
> - **Every HF arm here ran in `float16`**, including the one labelled bf16 (`meta.dtype` in each
>   `figure/data_h200/*_hf.json`). The current bench runs bf16 / ommx / kitty in bfloat16 and
>   records the resolved dtype per arm.
> - **The OMMX column is a median and misses most of the regroup tail.** OMMX repacks every
>   `OMMX_KV_GROUP_TOKENS` (=32) decode steps. Measured over 240 steps on an H100 NVL the
>   boundaries are perfectly periodic (23, 55, 87, 119, 151, 183, 215) and cost 86–140 ms against
>   a 34–40 ms steady-state step — **+17% amortized** — and the shipped `--measure 40` window
>   reaches only the first of them. `mean_ms` is the honest per-token cost, the median is the
>   steady-state one; quote both.
>
> `repro/README.md` holds the full provenance audit of these files (including their `p99`, which
> was really the maximum). The same warning is repeated at the end of [Reproduce](#reproduce).

- **Fastest low-bit-KV decode among the quantized methods from 16K up — not at short context.**
  All four arms run under HF-eager. OMMX beats **KIVI ×1.7→×2.3 (H200) / ×1.4→×2.4 (A100)**
  across 1K→64K, but passes **Kitty** only from 16K on (H200 ×1.6→×1.9, A100 ×1.2→×1.5 over
  16K→64K): **Kitty is faster at 1K and 4K on both GPUs** (H200 11.6 vs 17.7 and 17.0 vs 22.1 ms).
  KIVI / Kitty have no vLLM path, so the `×N` label is HF-eager. Accuracy is a *separate* axis
  (Table 1 / Fig. 6).
  > **"Same engine" means same framework, NOT same code.** The four arms are four DIFFERENT model
  > implementations: bf16 is stock `transformers` `AutoModelForCausalLM`, while ommx
  > (`ommx_gpu_serve/hf_eager/_ommx_llama_modeling.py`), kivi
  > (`baseline/kivi/models/llama_kivi_eval.py`) and kitty
  > (`baseline/kitty/_kitty_llama_modeling.py`) are each a separate hand-written model file with
  > its own cache object and its own per-step Python overhead. **At short context that overhead,
  > not the KV kernel, dominates**: an 8B model on H200 is ~3 ms/step at the memory roofline, yet
  > every arm here measures 12–31 ms/step at 1K. This is why OMMX (hf) can appear *faster than
  > bf16* at 1K — it is a scaffold difference, not a KV-quantization result. Treat the ≥16K rows,
  > where KV traffic actually dominates, as the meaningful comparison.
  > `--bf16-cache static` (now the default in `figure/bench.py`) removes the largest single
  > scaffold asymmetry: stock `DynamicCache` re-`torch.cat`s the whole KV every decode step, which
  > none of the quantized arms do.
- **Mixed dtype, by necessity and on the record.** In the **current** bench (`run.sh` →
  `figure/bench.py`) the bf16 / ommx / kitty arms run **bfloat16** and the KIVI arm runs
  **float16**, because KIVI's packer hard-casts every dequantized K/V to
  `torch.float16` (`baseline/kivi/quant/new_pack.py`) and its kernel allocates an fp16 output, so
  a bfloat16 KIVI run raises rather than producing a number. `run.sh` passes the dtype explicitly
  per arm and `figure/bench.py` records the resolved dtype in the JSON so the legend cannot
  mislabel it. **The table above is not such a run**: it predates the per-arm dtype and ran
  **every** HF arm in float16 while labelling one of them "bf16".
- **Slower than 16-bit bf16 — the honest tradeoff.** OMMX's decode is dequant-bound (outlier-membership
  + unpack + scale/zp), so at long context it trails bf16, which does no dequant but stores **3.66×
  the KV bytes OMMX does** (16 bit/element against the measured 4.375 of the published recipe — the
  accounting above; *not* the ≤3-bit lever, and not a comparison against KIVI / Kitty). The memory
  win is **shared** — OMMX, KIVI and Kitty all keep the KV quantized in HBM; OMMX's
  edge over them is decode speed + accuracy, not a smaller cache.
- **The same OMMX kernel is ≈ 1.9× faster under vLLM than under HF-eager** (9.5 vs 17.7 ms @1K).
  The win is the surrounding engine — vLLM's per-step path against a hand-written HF-eager model
  file — **not** CUDA-graph capture of the OMMX decode: on sm_90 `get_cudagraph_support` in
  `integration/vllm/backend.py` declares `NEVER`, so vLLM deliberately keeps the OMMX attention
  **out** of the graph (piecewise), as the H200 caption above says. A100/sm_80 is the arch where
  the OMMX decode is captured.

**KV recipe ↔ fakequant parity.** The serving kernel and the `ommx_fakequant/` accuracy model share the
canonical recipe: CoQA f1 matches within 0.01, and WikiText-2 4K-chunked PPL is bit-identical between the
open extract and upstream. The served path is `ommx_gpu_serve` (`kv_store.py` + `pack.py` + `codec.py`);
`ommx_fakequant/` is the separate accuracy simulator.


---

## 2. How the named recipes were chosen

**How `paper-kv` was chosen.** The legal grid — `group_tokens` × `group_channels` in
{16,32,64,128}², outlier count, range-map on/off, pow2 on/off, and the three position
encodings — is 11712 recipes (3904 per encoding); each was
priced with `packed_only.kv_bits_breakdown`. Eight hit **2.7500 exactly** while keeping the
INT8 pow2 exponent (claim **B9**), at least one outlier (**B1/B2/B7**) and the `relidx7`
positions the vLLM decode path actually reads. The registered one has the **highest outlier
coverage** of the eight (10/128 = 7.81 %) and uses the paper's own worked-example group size
N = 128 (**B10**). Full shortlist in `recipes.py`. Honest cost, also measured: `gt=128`
quadruples the bf16 residual ring, so at 4 K context the *effective* ratio is **4.096×**, not
5.818× (`shipped-kv`: 3.346×).

**`paper-weight` is 3.6250 on disk and 4.1250 resident.** The 3.6250 decomposition is
2 (INT2 base) + 8/64 (INT8 E8M0) + 16/64 (BF16 ZP) + 64/64 (the **flat bitmask** of claim
**B4**) + 4·4/64 (FP4 codes) + **0** range map (claim **B6** lists none) — self-consistent
with the paper's own format choices, and exactly what
`w_packer budget --preset paper-weight` prints. But the shipped CUDA weight kernel has no
bitmap reader and takes the range-map planes as required arguments, so such a bundle is
**refused at load** unless `OMMX_W_TRANSCODE` re-encodes it — and that transcode restores
relidx7 positions and synthesises the map planes, measuring **4.1250 bit/weight resident**.
Quote 3.6250 as a **storage** figure; the HBM traffic of a served `paper-weight` bundle is
`shipped-weight`'s.

Claim ids (`E1`, `E2`, `B4`, `B6`, `B9`, `B10`) index the paper's GPU claims; every number
above is recomputed by `ommx_gpu_serve/tests/test_recipes.py`, which fails if the registry
drifts from the code, if `shipped-*` stops matching the shipped defaults, or if a preset
silently overwrites an explicit setting.


---

## 3. Running OMMX under vLLM

**vLLM's defaults do not work with the OMMX KV backend, and the old failure mode was silent.**
The OMMX sidecar (`attention/kv_pool.py`) is not paged: request slot `b` owns a *static contiguous*
page block and `req_to_token` is an identity `arange`, so vLLM's `slot_mapping` never reaches it.
Two vLLM v1 defaults break that model — `enable_prefix_caching` (`vllm/config/cache.py:91`) and
`enable_chunked_prefill` (`vllm/config/scheduler.py:84`), **both `True` on vLLM 0.21**.

Measured on an H100 NVL, Llama-3.1-8B, two prompts sharing a prefix, `--attention-backend CUSTOM`,
reading the backend's own route sentinel (`$OMMX_FIRE_FILE`) back after each run:

| configuration | route evidence | generated text |
|---|---|---|
| `enable_prefix_caching=False` (supported) | `BATCHED_ROUTE_FIRED` | diverges from bf16 — **really OMMX** |
| `enable_prefix_caching=True` (**vLLM default**) | `KV_UPDATE_DEAD` | one output **byte-identical** to bf16 |
| `OMMX_ATTN_MAX_NUM_SEQS` left at its old 256 default | `KV_UPDATE_DEAD` | **both outputs byte-identical to bf16** |

The sidecar write raised (`ring append_block expects a fresh request slot`, and a CUDA OOM from a
1.1 TiB pool projection), the backend latched `_ommx_dead`, and every later step fell back to bf16
FlashAttention **while still producing fluent, correct-looking output**. Anyone benchmarking "OMMX
under vLLM" that way measured FlashAttention.

**What the current code does instead:**

- `integration/vllm/preflight.py` runs at engine construction and **refuses to start** on prefix
  caching, chunked prefill, an unreadable/oversubscribed pool budget, or `vllm < 0.21` — each with
  the arithmetic and the fix in the message, and each overridable only by an explicit
  `OMMX_UNSAFE_ALLOW_*` env that warns loudly.
- **A route failure is now fatal by default.** `OMMX_ALLOW_BF16_FALLBACK=1` opts back into the
  degrade, and even then it latches `ommx_route_health()["degraded"] = True` and shouts. A bench
  must assert that flag before quoting a number; `bench/bench_e2e_a100.py` now marks a CUSTOM arm
  `ok=False` when the sentinel carries no `*ROUTE_FIRED` tag.
- The pool slot cap is `max(1, min(OMMX_ATTN_MAX_NUM_SEQS or 256, scheduler_config.max_num_seqs))`
  (`_resolved_max_num_seqs` in `integration/vllm/backend.py`) instead of a fixed 256, so the static
  pool is no larger than what the scheduler can actually fill. It is a **MIN**, not a precedence
  chain: the engine can only *lower* the cap. An engine above 256 concurrent sequences still caps
  at 256 and is then refused by the slot-cap check below — it is never silently aliased.

Supported configuration: `enable_prefix_caching=False`, `enable_chunked_prefill=False`, one
KV-cache group, no sliding-window layers on the OMMX route — **and a pool that fits**. Those four
are necessary, not sufficient. The sidecar is not paged: its footprint is
O(`max_num_seqs` × `max_model_len`) per layer, allocated eagerly at the first B>1 step, so
preflight also refuses to start when `max_num_seqs` exceeds the slot cap above, or when the
projection exceeds `OMMX_POOL_BUDGET_FRAC` (default 0.5) of **free** HBM. At vLLM's default
`max_num_seqs=256` with `max_model_len=131072` that projection is at least 35.11 GiB per layer ×
32 layers ≈ 1.1 TiB — more when the pool also keeps a bf16 history (`OMMX_KV_RING` off, the
default) — so the engine is refused on any card. The error carries the arithmetic and names the
three knobs that shrink the projection: `OMMX_ATTN_MAX_NUM_SEQS` (it prints the largest value that
fits), `--max-model-len`, and `--max-num-seqs 1`, which makes a B>1 step unreachable so the pool is
never built. Raising `OMMX_POOL_BUDGET_FRAC` widens the budget instead of shrinking the pool. A
run that can never take a B>1 step (`--max-num-seqs 1`, this repo's own single-request bench)
allocates none of it and is reported as informational instead of refused.

**Served-path accuracy gate.** Long-context needle retrieval (3 needles at early/middle/late depth,
greedy, H100 NVL), with the sentinel confirming `DECODE_ROUTE_FIRED fmt=i2f4` for the OMMX arm:

| arm | ctx 4K | ctx 16K |
|---|---|---|
| OMMX `CUSTOM` (INT2+FP4 KV) | **3/3** | **3/3** |
| bf16 `FLASH_ATTN` (reference ceiling) | 3/3 | 3/3 |

OMMX matches the bf16 ceiling. This is a **regression gate, not a discriminative accuracy
measurement** — both arms score full marks, so it proves "no retrieval loss at 4K/16K", not a
ranking. RULER-32K through `eval/` remains the real accuracy test.


---

## 4. Provenance of the shipped `figure/data_*` measurements

**The shipped `figure/data_*/` JSONs predate the current bench and must be regenerated before
being cited** — the H200 table near the top of this README is drawn from them, and *How these
numbers were produced* beside it prices the consequences. They carry no provenance block (no git
SHA, GPU, torch or triton version); they record `dtype: float16` under arms labelled bf16; they
gave the bf16 arm a stock `DynamicCache`, which a same-process A/B measures at 2.59×–3.17× slower
than the `StaticCache` that is now the default (`--bf16-cache static`); and their `p99` was
computed as `times[int(0.99*n)]`, which for n=40 is simply the maximum. `figure/bench.py` now
records provenance, the resolved per-arm dtype, `mean_ms`/`max_ms`/a real percentile, and the raw
per-step `times_ms`.

---

## 5. Weight-quant linear microbench

![weight-quant linear microbench](figure/fig_linear_tpot_h200.png)

The numbers below are the **standalone microbench**, not a serving measurement.
Decode-linear latency (batch `M` = 1–16) of one projection GEMM `x[M,K] @ W[N,K]ᵀ` on H200, vs bf16
(cuBLAS), LLM.int8 (bnb), and CUTLASS INT4 (the `ex55` mixed-input **reference** example, timed by its
own profiler). **Latency only — no accuracy or iso-bit claim.**

- **Where OMMX wins:** only at **npv = 4** (6.25% outliers, 3.06 b/wt), small **M**, on the **square** and
  **down** projections — square M=1 **0.037 vs 0.040 ms** (×1.10 over ex55), down M=1 **0.101 vs 0.125 ms**
  (×1.24), for M ≤ 4. It **loses** at npv = 8/16, on gate/up, and at M ≥ 8 (slower than bf16 too).
- **Not iso-bit, and not group-size matched:** npv4 (3.06 b/wt at **gs=64**) is ~26% fewer bits than
  the 4.125 b/wt the timed `ex55` binary actually used — INT4 at **g=128**; the same INT4 weight
  re-grouped to gs=64 is 4.25 b/wt (see [`csrc/linear/README.md`](ommx_gpu_serve/csrc/linear/)).
  `ex55` is a reference GEMM, **not** the tuned Marlin/Machete W4A16 kernel vLLM ships, and OMMX
  here is the **eager 2-launch** path.
- **Correctness:** the kernel's i2f4 dequant is **bit-exact** to the in-repo reference packer, gated by
  [`test_ommx_linear_parity.py`](ommx_gpu_serve/csrc/linear/test_ommx_linear_parity.py) — base and
  base+FP4-outlier decode + prefill, cos ≥ 0.999. Re-verified on an **H100 NVL (`-arch=sm_90a`)**:

  ```
  [decode M=1 base]                          cos=1.00000 max_diff=5.96e-07 -> PASS
  [decode M=8 base]                          cos=1.00000 max_diff=7.75e-07 -> PASS
  [decode M=1 E8M0 vs fp32 (must be exact)]  cos=1.00000 max_diff=0.00e+00 -> PASS
  [prefill M=32 base]                        cos=0.99999 max_diff=1.26e-02 -> PASS
  [decode M=1 base+outlier]                  cos=1.00000 max_diff=4.92e-07 -> PASS
  PARITY GATE: PASS
  ```
- **Two environment requirements, or the gate cannot run at all.** `build_ommx_linear.py` compiles
  through `torch.utils.cpp_extension.load()`, which (a) shells out to the **`ninja` executable** —
  installing the wheel is not enough, `<env>/bin` must be on `PATH`, else
  `RuntimeError: Ninja is required to load C++ extensions`; and (b) links with
  `-L$CUDA_HOME/lib64`, which fails as `/usr/bin/ld: cannot find -lcudart` on conda envs that keep
  `libcudart.so` in `lib/` — export `LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64`. `ninja`
  is declared in the root `pyproject.toml` `[linear]` extra, and nothing else installs it:
  `uv pip install -e '.[linear]'` into `.venv-ommx` before building this kernel. `run.sh setup`
  covers only the three benchmark venvs.
