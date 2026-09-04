# Repro provenance — per-GPU orchestration scripts

The concrete per-GPU scripts behind the published serving numbers: `figure/fig_tpot_<tag>.png`
and `figure/fig_peakmem_<tag>.png`, their normalized input `figure/data/<tag>.json`, and the
decode-TPOT / peak-memory tables in the top-level `README.md`. (There is no `figure/RESULTS.md`
in this release; an earlier draft of this line named one.)
Each runs every method in its own env on ONE assigned GPU (`CUDA_VISIBLE_DEVICES`).

- `orchestrate_h200.sh` — NVIDIA H200 NVL (fig1). Runs bf16 / OMMX / KIVI / Kitty, each in its own conda env or venv. Override `REPO`, `CONDASH`, the `ENV_*`
  conda env names, `KITTY_SRC`, `KVENV`, `HF_HOME` for your layout (see the script header).
- `orchestrate_a100.sh` — NVIDIA A100-SXM4-80GB (fig2). Installable subset (bf16 / OMMX /
  KIVI) over per-method venvs. Override `REPO`, `OMMXPY`, `KIVIPY`.

Both assume model / Kitty artifacts are already present locally (`HF_HUB_OFFLINE=1`).
The portable single-venv entry point for a fresh host is `../run.sh`; these scripts are the
concrete multi-env instances behind the two published figures.

## Which arm runs in which dtype, and why

`figure/bench.py --dtype` defaults to **bfloat16**, so an arm plotted as "bf16" really is
bf16 and is comparable to the vLLM engine arms. That default does not fit every arm, so the
scripts pass `--dtype` **explicitly for every arm** rather than inheriting a default that has
already flipped once (float16 → bfloat16) and broke a baseline when it did:

| arm | knob (env or flag) | dtype | why |
|---|---|---|---|
| `bf16`, `ommx` | `DTYPE` / `--dtype` | `bfloat16` | matches the vLLM engine arms, which `bench_e2e_a100.py` locks to bf16 for its fair-compare contract |
| `kitty` | `DTYPE_KITTY` / `--dtype-kitty` (default `$DTYPE`) | `bfloat16` | the vendored `baseline/kitty/_kitty_llama_modeling.py` contains **no** fp16 hard-cast — it upcasts to fp32 for RMSNorm/softmax and casts back to `query.dtype`. Its **external** kernel package (`KITTY_PKG_PATH`) is not vendored here and was not inspected; if it turns out to be fp16-only the arm will raise a loud dtype error, and the fix is `DTYPE_KITTY=float16` plus an fp16 label — never a bf16 label on an fp16 run |
| `kivi` | `DTYPE_KIVI` / `--dtype-kivi` | `float16` | KIVI is **fp16-only**. `baseline/kivi/quant/new_pack.py` hard-casts every dequantized K/V with `data = data.to(torch.float16)` — once in each of `unpack_and_dequant_kcache`, `unpack_and_dequant_vcache` and `unpack_and_dequant_triton_packed` — and `baseline/kivi/quant/matmul.py` allocates the fused kernel's output as `dtype=torch.float16`; `_kivi_check_fp16` in `baseline/kivi/models/llama_kivi_eval.py` rejects a bfloat16 model at the boundary instead of casting it. A bfloat16 KIVI run therefore produces **no number at all** |
| vLLM arms (`TRITON`, `FA`, `abl_attn*`) | — | `bfloat16` | fixed by `bench_e2e_a100.py`'s fair-compare contract |

**This makes the figure mixed-dtype, which is a fairness caveat, so it is printed rather than
buried here.** `figure/bench.py` records the resolved dtype in `meta.dtype` /
`meta.torch_dtype` and in `arm_label` (e.g. `kivi (fp16)`); `figure/collect.py` carries it
into `figure/data/<tag>.json` under `methods[<key>].provenance.dtype` and prints a
`dtypes={...}` map on its `COLLECT_DONE` line; `figure/plot.py` appends the dtype to **every**
legend label as soon as two drawn series disagree, any series is not bf16, or any drawn series
recorded **no** dtype at all (which prints as `[?]` — an unrecorded dtype is treated as a
disagreement, never as agreement, so a legacy fp16-era JSON cannot sit unmarked next to
verified-bf16 bars). A KIVI bar next to a
bf16 bar is "each method at the dtype it supports", not a same-dtype comparison — say so
wherever the figure is cited.

`KIVI_FUSED_GQA` is **off** by default, i.e. the published KIVI bar uses the dequant fallback
path. That is not a handicap. Measured on an H100 NVL (torch 2.11.0+cu130, triton 3.6.0) with
**both** paths pre-warmed at every context and each cell run in both arm orders
(`fused_fires` read back to prove which path ran):

| ctx | fallback TPOT (both orders) | fused TPOT (both orders) | **fallback / fused** (>1 = fused faster) |
|---|---|---|---|
| 1024 | 67.98 / 70.88 ms | 60.24 / 59.27 ms | **1.16×** (fused faster) |
| 4096 | 66.63 / 68.33 ms | 106.24 / 107.55 ms | 0.63× (fused slower) |
| 16384 | 144.10 / 130.03 ms | 360.51 / 338.76 ms | 0.39× (fused slower) |

(Ratio of the two-order means, e.g. 1024: (67.98+70.88)/2 ÷ (60.24+59.27)/2 = 69.43/59.76 =
1.16. The column is fallback÷fused so that ">1" reads as "fused wins"; inverted it is
1.58×/2.55× at 4K/16K, which is where the "1.6-2.6× slower" below comes from.)

The fused path wins only at 1K and is 1.6–2.6× slower from 4K up, so the default is the
better KIVI at every published context ≥ 4096. An earlier un-prewarmed run reported 1.44× at
ctx=1024; that cell was contaminated by a 57.7-second first-arm Triton JIT and is withdrawn.

## The two vLLM bf16 baselines are different kernels

`../run.sh bench` runs both, because only one of them is what vLLM does out of the box:

- `vllm_bf16.json` ← the **`TRITON`** arm (`TRITON_ATTN`). The filename is legacy; the payload
  says `method: "vllm_triton"`. Plotted as **"bf16 (vLLM, Triton attn)"**.
- `vllm_flash_attn.json` ← the **`FA`** arm (`FLASH_ATTN`), vLLM's **default** attention
  backend. Plotted as **"bf16 (vLLM, FlashAttn)"**. `FA` is deliberately absent from
  `bench_e2e_a100.py`'s default arm order, so it runs only because `run.sh` names it in
  `--only-arms`; `e2e_to_figure.py` emits the file only when `--flash-arm` is passed.

If the `FA` arm produced no ok cell, `run.sh` says so loudly and the figure simply has no
stock-vLLM bar. Do not caption the Triton bar as stock vLLM.

## Stale outputs: moved aside, never deleted, and never silently re-collected

`figure/data_<tag>/` is keyed by **filename**, so a file from an *earlier* run is
indistinguishable from one this run produced — `collect.py` picks it up and `plot.py` draws it.
That is how "vLLM's DEFAULT baseline will be MISSING from this figure" could be printed while a
previous sweep's `vllm_flash_attn.json` was still being plotted as "bf16 (vLLM, FlashAttn)".

`run.sh` **and both orchestrators** therefore rename each leg's own output to `<file>.prev`
*before* running that leg (`<m>_hf.json` for an HF arm; `vllm_bf16.json` /
`vllm_flash_attn.json` / `ommx_vllm.json` for the vLLM sweep). A leg that does not finish
leaves **no** bar.

Renamed, **not deleted**, because those same filenames are the repo's shipped, git-tracked
measurements: `figure/data_h200/` and `figure/data_a100/` are in git, and the first documented
Reproduce command writes into `figure/data_h200/`. Deleting up front destroyed published
artifacts before the leg had done anything, and kept them destroyed when the leg failed — a
git checkout brought them back, a tarball did not. `collect.py`'s `FILE2KEY` is an **exact**
filename map, so a `.prev` file is never collected and never plotted; rename it back if you
want it published again (but read the section below first). Only one generation is kept — a
second run overwrites the `.prev`.

`figure/collect.py` closes the last gap on the same rule: `figure/data/<tag>.json` is itself a
published, git-tracked artifact (`figure/data/h200.json`, `figure/data/a100.json`), and a typo'd
`--tag` or a bench directory where every leg failed used to be collected as a well-formed
`{"methods": {}}` written straight over it at rc=0. Both cases now **refuse before the write**
(missing `--src` directory; no collectable JSON in it), leaving any existing `--out` intact and
returning nonzero, so `run.sh figure` reports the real problem instead of publishing an empty
figure input.

The other half of the same problem is a file **no** leg of this run owns: an arm left out of
`--methods`, an arm skipped because `KITTY_SRC` is unset, the vLLM arms when you run the HF
sweep alone, or a shipped historical JSON. Those are still collected — often deliberately, e.g.
the documented HF-then-vLLM split run — so they are not touched, but each one is named on a
`[collect] NOTE ... was NOT produced by this run` line before the figure step. Combining two
sweeps is legitimate; not knowing that you did is not.

Every `CUSTOM` (OMMX) arm is additionally gated on route-fired evidence: `bench_e2e_a100.py`
gives each one its own `$OMMX_FIRE_FILE` sentinel under `--workdir` and forces every cell to
`ok=False` unless a `*_ROUTE_FIRED` tag proves the OMMX decode kernel ran. `run.sh` re-reads
that evidence, prints it, and — if any arm is unproven — asserts that no OMMX cell survived
into `ommx_vllm.json` (deleting the file and failing if one did). A CUSTOM arm that silently
fell back to bf16 FlashAttention still returns a plausible TPOT, so this sentinel is the only
thing separating an OMMX measurement from a FlashAttention measurement wearing its name.

## What the published `figure/data_*/` JSONs do NOT contain

The JSONs currently shipped in `figure/data_h200/` and `figure/data_a100/` were produced by an
**older** `figure/bench.py` and are kept only as the record of what the published figures were
drawn from. They are missing things the current bench emits, and they carry at least one
label that is now known to be wrong:

- **No provenance block.** Their `meta` is exactly `{"warmup": 8, "measure": 40, "dtype":
  "float16"}` — no `git_sha`, no `gpu`, no `torch` / `triton` / `transformers` version, no
  `python`, no `attn_impl`, no per-arm recipe (`kivi_recipe`, `ommx_recipe`, `bf16_cache`).
  Nothing in the file identifies the commit or the machine that produced it.
- **They are fp16 runs under bf16 labels.** `bf16_hf.json` — the arm the figure draws as
  "bf16" — records `meta.dtype: "float16"`, as do `ommx_hf.json`, `kivi_hf.json` and
  `kitty_hf.json`. They also predate `arm_label`, so the files cannot even state what they
  are; `arm_label` reads `None`.
- **Their `p99_ms` is the maximum, not a 99th percentile.** The old bench computed
  `times[int(0.99 * len(times))]`; at the shipped `measure = 40` that is index 39 of 40
  ascending samples, i.e. `max`. The current `figure/bench.py` uses a linearly interpolated
  percentile (`_percentile`) and also ships `mean_ms`, `max_ms`, `min_ms` and the raw
  `times_ms`.
- **The vLLM-derived files predate the honesty fields.** `vllm_bf16.json`, `ommx_vllm.json`
  and `turboquant_vllm.json` in those directories carry only `{"method", "ctxs"}` (plus
  `breakdown` for the OMMX one) — no `arm`, no `attention_backend`, no `breakdown_raw`, no
  `breakdown_meta`. The Triton-vs-FlashAttn identity of the "vllm_bf16" series and the
  clamping/renormalisation applied to the stacked OMMX bar cannot be recovered from the files;
  both are only knowable from the adapter source.

**Consequence: regenerate them with the current `figure/bench.py` +
`ommx_gpu_serve/bench/e2e_to_figure.py` before citing any number from them.** They are left in
place deliberately (deleting them would erase the provenance of the already-published
figures), but they are a historical record, not a current measurement.

## Runbook — reproducing a full figure on a fresh GPU

Validated end to end on an H100 NVL (2026-08-24): `run.sh bench` -> `figure/bench.py` ->
`figure/data_<tag>/<m>_hf.json` -> `figure/collect.py` -> `figure/data/<tag>.json` ->
`figure/plot.py` -> `fig_tpot_<tag>.png` + `fig_peakmem_<tag>.png`.

```bash
# one leg per GPU allocation; --tag names the output set
./run.sh bench  --tag h200 --ctxs 1024,4096,16384,32768,65536 --methods "bf16 ommx kivi kitty"
./run.sh bench  --tag h200 --ctxs 1024,4096,16384             --methods "ommx_vllm"
./run.sh figure --tag h200
```

Five things that will otherwise cost you a re-run:

1. **Free HBM, not total HBM, is the constraint.** These arms preallocate: bf16 with
   `--bf16-cache static` reserves the whole KV up front, and the OMMX sidecar is sized from
   `max_model_len`. On a card already holding another tenant, the long-context cells OOM —
   they are recorded as `{"oom": true}` and the bar is simply absent from the figure, so a
   half-empty plot means "not enough free memory", not "the method failed". Check free
   memory before starting, not after.
2. **Do not share the GPU if you care about the numbers.** A co-tenant shows up as `min_ms`
   roughly **half** of `tpot_ms` in the per-step distribution — the steps that get the GPU to
   themselves keep the true speed while the median stretches — and as `mean_ms` pulling away
   from the median; on a clean run both sit within a few percent of it. `min_ms`, `mean_ms`,
   `max_ms` and the raw `times_ms` are all written by the current `figure/bench.py`; look at
   them before quoting anything. The shipped `figure/data_*/` JSONs carry only
   `ttft_ms / tpot_ms / p99_ms / peak_gb` (see the section above), so for those this check
   cannot be made at all — one more reason they are not citable. Under a GPU allocator,
   `run.sh` inherits a **single** masked `CUDA_VISIBLE_DEVICES` rather than forcing device 0,
   and **refuses** a multi-GPU mask (asking for `--gpu N`) rather than resolving it to device
   0, which is not even a member of the mask — so the run stays on the device you were
   actually given.
3. **`--measure 40` (the default) reaches only ONE OMMX regroup boundary.** OMMX repacks
   every `OMMX_KV_GROUP_TOKENS` (=32) decode steps, so the measured window 0..39 contains
   the boundary at index 23 and misses the next at 55. Measured over 240 steps the
   boundaries are perfectly periodic (23, 55, 87, 119, 151, 183, 215) and cost 86-140 ms
   against a ~34 ms steady-state step: about +17% amortized, i.e. `mean_ms` is the honest
   per-token cost and `tpot_ms` (median) is the steady-state one. Quote both, or raise
   `--measure` past a few boundaries.
4. **The plotting extras are not needed to bench.** `figure/plot.py` needs numpy +
   matplotlib; `run.sh bench` and `figure/collect.py` do not. If the render fails, the
   measured JSONs are already on disk — install the extras and re-run `run.sh figure`
   only. Never re-run the sweep for a plotting dependency.
5. **kitty needs an external package.** `baseline/kitty/_kitty_llama_modeling.py` is
   vendored but `import kitty.kvcache` is not; point `KITTY_PKG_PATH` at a checkout of
   https://github.com/Summer-Summer/Kitty or drop `kitty` from `--methods`.
