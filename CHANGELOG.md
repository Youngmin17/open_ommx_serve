# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `demo/`: a terminal demo that streams one request per arm -- four HF-eager arms (KIVI,
  Kitty, bf16, OMMX) and three vLLM arms (bf16 FlashAttention, TurboQuant 3-bit, OMMX).
  The default prompt is `demo/ommx_brief.md`, a ~3K-token brief on OMMX written from this
  repository's own numbers, with a question that asks the model to explain the format and
  quote three facts (`--prompt needles` builds a long filler document with three buried
  facts instead, for the long-context variant). Each arm prints TTFT, per-token latency,
  tokens/s, the fact check and the evidence that the arm's own kernel ran (the vLLM route sentinel, the decode
  launcher's counters, the backend the engine selected). Plus a pty recorder
  (`record.py`, asciinema v2 casts) and a cast-to-GIF renderer (`render_cast.py`,
  pyte + Pillow). `figure/demo_h200.gif` is the recording the README opens with.
- `attention.paged_decode.launch_stats()`: per-process counters of canonical decode
  launches and bitmap reads, so the HF-eager arm and the demo can prove the OMMX kernel
  ran without the vLLM sentinel file. The vLLM `DECODE_ROUTE_FIRED` sentinel now carries
  the outlier representation (`repr=bitmap`) next to the K format.
- The OMMX weight path (`ommx_w`) is now an opaque custom op, `torch.ops.ommx_w.linear`:
  `apply()` only reshapes and calls it, and the M-dependent routing, the env lanes, the
  outlier correction and the sentinel bookkeeping all live inside the op. vLLM 0.21 runs
  its Dynamo pipeline even under `enforce_eager`, and a raw pybind kernel call inside a
  LinearMethod was a graph break there -- every `ommx_w` arm died before its first
  step, so no weight-quant TPOT had ever been measured through the engine.
- `csrc/linear/bench_linear_memory_bound.py` runs on sm_80 as well as sm_90 and carries
  LLM.int8 (bitsandbytes 0.50, cu130) again next to bf16, CUTLASS ex55 INT4 (Hopper) and
  the two OMMX arms: the CuTe wgmma base + packed correction (Hopper) and the shipped
  two-launch pair. `fig_linear_tpot_{h200,a100}.png` draw CuTe solid and two-launch dashed.
- `bench_e2e_a100 --compile-mode none` (CompilationMode.NONE) and a `abl_linear_nocorr`
  arm (`OMMX_W_ABL_NO_CORR=1`, announced by the `ABL_W_NO_CORR_ACTIVE` sentinel tag),
  so the linear stage split is measured the same differential way as the attention one.
- `THIRD_PARTY.md` and `scripts/check_release_hygiene.sh` — the root Apache-2.0
  `LICENSE` sat above 31k lines of vendored code it does not cover (`eval/lm_eval`
  22311 lines, `baseline/kivi` 7043, `eval/lcb` 1085, `baseline/kitty` 690) plus a
  BSD-3 CUTLASS fork inside our own tree, none of them carrying an upstream LICENSE.
  The manifest records what each file states, measured rather than assumed, and
  `./run.sh check` enforces it instead of trusting it. It is red on `baseline/kitty`,
  which has no copyright line, no SPDX tag and no header at all: that is a decision
  about whether to publish, not a task, so the gate holds it open.

- `bench/bench_decode_kernel.py`: the KV decode-attention call timed on its own -- parity
  against the independent oracle first, then CUDA-graph replay latency, then the bytes it
  read -- per context, batch and arm, so a kernel retune is measured where it acts rather
  than inferred from a TPOT delta.
- `bench_e2e_a100 --block-size`: pins vLLM's `block_size` for every arm (0 = engine
  default) and records it in the run's `fair` block; the recipe echo prices the same
  page grid. 32 makes one OMMX group one vLLM block (`PAGE_GRID pages_per_group=1`),
  which is the geometry the paged verification runs at -- but on an H200 it costs
  +26% decode TPOT at 4K (6.85 -> 8.62 ms, neutral from 16K up), so the published
  figure keeps the engine default of 16.
- Decode kernel warp ladder (`_auto_num_warps_outlier`) set from the measured table on
  both GPUs (2/4/8 warps, 1K-64K, batch 1 and 8, CUDA graph, 3 seeds within 1 %): H200
  batch 1 takes 2 warps up to 4K (1K 0.470 vs 0.521 ms, 4K 0.149 vs 0.191), 4 at 8K-16K
  (8K 0.105 vs 0.139) and 2 from 24K (64K TPOT 22.6 vs 26.1 ms); H200 batch >= 8 takes 4
  warps below 24K (1K 0.080 vs 0.110, 16K 1.12 vs 1.17) and 2 beyond (64K 4.43 vs 4.64);
  A100 takes 4 warps everywhere (batch 1 8K 0.151 vs 0.235, 64K TPOT 38.3 vs 51.8). The
  vLLM path passes `max_model_len`, so a 128K engine sits on the long-context rung. 8 warps
  never wins on either GPU; `OMMX_ATTN_OUTLIER_WARPS` still overrides for A/B.
- Recipe defaults are single-sourced: `DEFAULT_OUTLIER_REPR`, `DEFAULT_OUTLIERS` and
  `DEFAULT_POW2` in `integration/vllm/config.py`, consumed by `packed_only`, `preflight`
  and the HF-eager / lm-eval / bench harnesses instead of each restating a literal. A
  bare run is the published recipe with bitmap positions (K 5.500 / V 2.750 / avg 4.125,
  3.88x vs bf16); `shipped-kv` differs from it only by relidx7 (avg 4.375, 3.66x).
- The committed data directories are now self-describing: eight held
  measured JSON that no document named, so the only way to identify one was to read
  the script that wrote it.
- `figure/plot_linear_tpot.py` and `fig_linear_tpot_h200.png`: decode-linear latency vs
  batch for OMMX i2f4 against bf16, CUTLASS INT4 and LLM.int8, with the not-iso-bit and
  not-end-to-end caveats on the canvas. H200 only: the CUTLASS arm is a Hopper example.
- `tests/test_decode_kernel_parity.py`: the KV decode kernel against an independent
  attention reference (dequantized planes, textbook fp32 attention). The axis had no
  numeric gate on the kernel's arithmetic -- plane parity, encoding cross-checks and
  the torch.library 'reference' (which calls the kernel) all leave it ungated.
- `kitty_hf` measured on H200, completing baseline parity with A100 (all seven arms,
  one day). The external Kitty package is pinned in the notes by commit SHA, with the
  transformers window and fp16 requirement that make it run.
- Both OMMX series carry their own KIVI speedup annotation on the TPOT figure, coloured
  and named apart (HF-eager = KV kernel only, vLLM = kernel plus engine).
- `turboquant_vllm` measured on H200, bringing its baseline set to six of A100's seven.
- Same-day H200 and A100 measurements of the two outlier-position encodings on the vLLM
  serving arm, plus a single-date H200 figure set (`figure/data/h200_20260903*.json`,
  `fig7_tpot_vs_ctx.png`). The encoding-A/B figure was later removed once bitmap became
  the default; the numbers it drew are in `figure/data/repr_ab_{h200,a100}_20260903.json`.
- A design note for sourcing the KV sidecar's page and
  group tables from vLLM's block table (per-physical-block planes, writer through the
  table, masked group load, owner check), with the four code-level blockers it removes
  and the accuracy gate the geometry change (`sink` 8 -> 32) must pass first.
- H200 re-measurement of the TPOT figures, with `bf16` carried as a control. Peak
  memory reproduces to the fourth decimal in all ten cells and OMMX's TPOT to within
  1.7-19%, but the control moves -57%/+52%, so absolute TPOT cannot be compared
  against the 2026-08-14 reference. The tail (p99/max) is not measurable on this node.
- `ommx_w` weight-quantization path end to end: offline packer, on-disk format model,
  vLLM `QuantizationConfig`/`LinearMethodBase` binding, and a CUDA kernel entry point.
- Calibration transport for the weight axis. `wq_eval.py --export-decisions` writes the
  solver's per-module decisions (outlier mask, group scale, zero-point, FP4 range map) and
  `w_packer pack --calibrated` encodes them without re-deciding, gated on
  `dequant(planes) == the calibrated weight`. Re-quantizing an already-on-grid weight is
  not idempotent, so carrying the weight alone would silently discard the calibration.
- Bundle self-description: the packer stamps `quantization_config` into `config.json`, so
  `vllm serve <bundle> --quantization ommx_w` resolves with no environment variable.
- `npu` bit-accounting basis on `Recipe.bits_breakdown()`, reporting the same bundle with
  positions re-encoded as `ceil(log2 C(group_size, npv))` per group. Surfaced as a third
  column of `w_packer budget`.
- `optimized-weight` recipe: group 64, 16 FP4 outliers per group, relidx7 positions,
  idx_range map. 6.1250 bits/weight stored, 5.1406 on the NPU basis.
- Named-recipe registry entries now record their NPU-basis budget alongside the stored
  one.

### Fixed
- The attention ablation flags (`OMMX_ABL_NO_OUTLIER` / `V_NODEQUANT` / `K_NODEQUANT` /
  `NO_UNPACK`) were read by every kernel launch and recorded nowhere: a flag inherited from
  the shell would have published an outlier-free kernel under a plain OMMX label with no
  trace in any JSON or sentinel. The launcher now keeps the resolved mask in
  `launch_stats()["abl"]`, the vLLM backend announces it as `ABL_ATTN_ACTIVE` next to the
  route sentinel, `figure/bench.py` records `meta.ommx_env` / `meta.ommx_abl_active` /
  `meta.ommx_launch_stats`, `bench_e2e_a100` records `abl_env_active` per arm and fails a
  non-ablation arm that carries any, `figure/collect.py` refuses an ablated `ommx_hf.json`,
  and `hf_abl_to_breakdown.py` checks that each ablation JSON carries exactly its own flags.
- The demo's HF-eager arms computed logits for every prompt position on prefill (a 25 GB
  tensor at 98K tokens), which inflated their peak memory and TTFT; the prefill now keeps
  the last position only. Its OMMX-vLLM arm reads the route file through the bench's own
  evidence reader (FIRED whitelist, `*_DEAD` / `*_NOFIRE` refusal, in-process degrade check)
  and runs the same CUDA-graph route the published bars ran; a dead arm now leaves a red
  line instead of a missing row. The HF prefill runs the decoder trunk and applies
  `lm_head` to the last hidden state (the official KIVI model ignores `logits_to_keep`),
  the engine's own log is saved next to the results as provenance for the backend
  evidence, and the graph-route sentinel carries `repr=` like the eager one.
- `OMMX_W_SPLIT=0` reached the kernel as an empty grid and returned zeros; values below 1
  are refused. `encode_fp4_e2m1f` refuses non-finite input instead of encoding it as zero.
- The KV packer's FP4 E2M1 encode (`attention/codec.py::encode_fp4_e2m1f`) rounded to the
  nearest level in LINEAR magnitude while the vendored fakequant (`fp16_to_fp4_e2m1`), which
  produced every accuracy number, rounds in LOG magnitude (geometric-mean boundaries, ties
  down). About 3% of the served K outliers sat one FP4 level away from the evaluated ones.
  The codec now uses fakequant's rule; `tests/test_fakequant_algorithm_parity.py` pins the
  FP4 codec, the pow2 scale and its E8M0 byte (including exact-tie vectors), the INT2 affine
  K and V bases and the dedicated-map outlier values bit-exact against fakequant at fixed
  positions, and the weight path against `weight_quantizer` -- the ratio (which positions
  are outliers) is the one thing left to each side.
- `csrc/linear/build_ommx_linear.py` resolves the CUDA env from the running interpreter
  (`sys.prefix`) instead of a login shell's `CONDA_PREFIX`, and adds the cu13 pip-wheel
  header dirs itself when `CUDA_HOME` has no `cusparse.h`. On a node whose base conda
  exports `CONDA_PREFIX=/opt/anaconda3` the sm_80 build died with `cusparse.h: No such
  file` and then `ld: cannot find -lcudart`; it built on the H200 node only because that
  host's system CUDA headers and shell `LIBRARY_PATH` covered for it.
- The preflight page-grid report now goes to the cross-process sentinel file. It is
  computed in the EngineCore worker, so a driver reading `_PREFLIGHT["report"]` got
  `None` for every field and could not tell that from a real null — measured on H200,
  where `--block-size` 16 and 32 both reported null while the route demonstrably fired.
  Now measured: block_size 32 gives `pages_per_group=1`, block_size 16 gives 2 and
  raises the existing misalignment warning.
- Preflight refuses an unknown outlier repr instead of pricing it as a combinadic rank.
- An empty environment string (`export OMMX_ATTN_BITMAP_READ=`) is treated as unset for
  the read tri-states, so the read follows the repr instead of resolving to False.
- A bitmap recipe with more than 8 outliers per group is refused when the engine is
  built. The kernel stages a group's outlier values in one int32 and refused it on the
  first decode step instead, after the engine had reported itself up.
- Partial bf16 coverage (`SW_BYPASS_BF16*`, sliding-window layers) reached only the
  orchestrator's log, so a Mistral/Gemma-2 bar looked identical to a fully-OMMX one.
  `e2e_to_figure` now carries `partial_bf16_coverage` into the arm JSON.
- bench_e2e_a100 downgraded EVERY plugin registration failure to a warning, including
  the foreign name collision plugin.py re-raises precisely so a run cannot measure a
  method nobody chose. It now consults `is_benign_registration_failure` and exits.
- `ommx_route_health()` had no caller: the arm worker now asks it in-process and the
  orchestrator folds the verdict into the arm's ok, catching a process that degraded
  AFTER firing -- the case a clean sentinel file cannot show.
- `figure/data/a100.json` (fp16 runs under bf16 labels, kept as evidence) now carries a
  SUPERSEDED block and per-method markers so it cannot be cited by accident.
- Stage segments were normalized onto the ablation run's total while the bar is drawn at
  the panel run's tpot -- separate measurements 1-4% apart, so `base` printed as 102%.
  Now normalized onto the target file's own tpot; the ablation total is kept in
  breakdown_meta so the discrepancy stays inspectable.
- The A100 HF-eager stage split reported 0% for outlier and scale below 32K because the
  raw differentials came out negative against a ~40 ms base. Re-measured at warmup 64 /
  measure 400: outlier and unpack resolve (0.14-0.65 ms); scale stays unresolved, and
  the notes now say a 0% segment there means unresolved, not zero.
- The README advertised 4.375 average KV bits beside a recipe using flat-bitmask
  positions; the bitmask prices at 4.125 (1.000 index bits/elem against relidx7's
  1.500). test_bit_accounting.py pins relidx7 only, so the default change left the
  figure stale with no failing test. New gate derives it from the resolved default.
- The A100 panel predated the bitmap default and recorded no recipe, so pairing it with
  today's H200 panel repeated across GPUs the mismatch that was just fixed within one.
  Re-measured on the same basis: shipped-kv, bitmap, all seven arms, five ablation arms.
- The vLLM bar's breakdown shipped EMPTY after the shipped-kv re-measurement: only
  abl_attn was run, so the bar drew hollow while the figure looked finished. All five
  ablation arms re-measured at shipped-kv.
- Serving defaults did not match the accuracy recipe: the harness pins 6 outliers with
  pow2 scales, the defaults said 3 with pow2 off, so an engine started with no OMMX
  environment served a different format from the one every accuracy number describes.
- The two OMMX bars in the TPOT figure ran DIFFERENT recipes: ommx_hf at shipped-kv
  (gt=gc=32, recent 32, signed) against ommx_vllm at the bench's fakequant geometry
  (gt=gc=64, recent 8, abs), so their KIVI ratios could not be read against each other.
  The vLLM arm was re-measured at shipped-kv; figures and notes rebuilt on it.
- Figure labels no longer print "?" for a series whose JSON records no dtype; the
  suffix is simply omitted. collect.py still reports the gap on its COLLECT_DONE line.
- `OMMX_ATTN_OUTLIER_REPR=combinadic` could not decode on any path: the store
  allocates `k_crank` and no `k_oidx`, but `combinadic_read` defaulted to a plain
  `False` while `bitmap_read` was tri-state, so it needed an undocumented second env
  var. Adds `resolved_combinadic_read()` and the symmetric mismatched-repr refusal.
- Four claims corrected after an adversarial audit: the two Fig.7 variants were the same
  file, section 8's +inf mechanism is refuted by the pool being `torch.zeros`, section 9's
  "all ten cells" peak claim contradicted its own table, and section 10 named
  `group_channels` where `_pack_bitmap_frames` uses `group_tokens`.
- The HF-eager decode forwarded `combinadic_read` but not `bitmap_read`/`k_obmp`, so a
  bitmap-packed store raised "requires the k_oidx relidx7 sidecar plane" on its first
  quantized group. Both call sites now source it from the resolved config.
- Recorded that `--block-size 32` works on the OMMX backend and that the page-grid
  preflight fires with the right numbers, measured on H200. The report must be read
  from the engine log: it lives in the `EngineCore` process, so a driver-side probe
  reports every field null whether the check ran or not.
- The batched-graph path aborted the CUDA context on any request long enough to
  complete a group. Prefills arrive at B==1, before the batched session latches, so
  the prompt was written elsewhere and the build seam regrouped uninitialised rows
  to vLLM's sequence length. Graph mode now pre-latches, and the seam refuses to
  claim more than one token beyond the pool's own write high-water mark. 6 of 6
  trials pass, previously 2 of 6.
- The KV outlier-gather guard tested only the upper index bound, while ATen asserts
  `idx_dim >= 0` as well; a negative index passed the check and still aborted the
  device with an unattributable message. Both bounds are now checked and reported.
- The solver chose INT2 codes against an f32 zero-point while storing bf16, so every code
  was slightly off. Rounding to the stored dtype before code selection takes the export
  round-trip from 2.4e-4 to 1.5e-8.
- The vLLM plugin reported "registered" from its own bookkeeping rather than from vLLM's
  registry, and one path set the flag without registering anything.

### Changed
- The fp16-era inputs in `figure/data_h200/` and `figure/data_a100/` and the
  `figure/data/h200.json` collected from them carry a `SUPERSEDED` block naming the current
  files; `collect.py` refuses any input that carries one, so a stale directory can no
  longer regenerate the README figure. The HF-eager stage plot's base segment is labelled
  as including framework overhead, which the ablation cannot separate from the kernel's
  own work. The TurboQuant figure JSON records `weights: bf16` like the other vLLM arms.
- `ommx_linear.cu` drops the three single-launch fused decode kernels
  (`decode_gemv_fused_correct_kernel`, its batched twin and the odelta-LUT variant) with
  their host wrappers and bindings: nothing routed to them after the lane was removed. The
  odelta/kpos LUTs stay -- the CuTe packed correction reads them.
- `bench_linear_memory_bound.py` times LLM.int8 in a child process on the same GPU with the
  same CUDA-event timer (`OMMX_INT8_PYTHON` picks its interpreter, recorded in the result):
  bitsandbytes 0.50 died with SIGBUS in-process on the H200 host, and a signal cannot be
  caught, so the arm used to take the whole run with it.
- The H200 HF-eager bars (OMMX, bf16, KIVI, Kitty) were re-measured on the final tree in
  one session so the panel stays one day / one machine after the ladder change; the stage
  ablation arms behind the OMMX bar's split were re-run with it.
- README cut to the results: one licensing line, the TPOT preamble and the KIVI comparison
  paragraph shortened, and the decode-linear section now describes the CuTe (Hopper) and
  two-launch (sm_80) OMMX arms with LLM.int8 back in the peer set. The single-launch fused
  decode lane (`OMMX_W_DECODE_FUSED*`) is gone from `linear_method.py`, the microbench and
  the figures: it won only at batch 1 and lost 2-8x from batch 2 on both GPUs.
- Decode kernel register diet (bit-exact, parity 17/17): the K code extraction no longer
  materialises a replicated int32 tile, and the outlier-map splice finishes the even
  half before starting the odd one. H200, 2 warps: 32K/64K batch-1 -6.7%, batch-8
  neutral (`figure/data_h200_kernel/candidates/`).
- Weight kernel: group index by shift when the group size is a power of two, and the
  outlier position walk emits positions sequentially (relidx7 shift register, bitmap
  ffs) instead of re-scanning per slot; bit-identical FMA order.
- The bit-width claim moved from README prose onto the figure canvases and is now priced
  on the NPU basis (combinadic positions): 4.00 average KV bits, and 3.94/4.39/5.14
  bits per weight for the linear sweep. The three encodings are membership-equivalent,
  so an unbased number is ambiguous; the NPU basis is the substrate-invariant one.
- README keeps four figures: the two paper figures and the H200 / A100 decode-TPOT
  panels. The KIVI-comparison, peak-memory and encoding-A/B figures are still built
  and committed, just not embedded.
- Default KV outlier encoding is now the flat bitmask: bit-identical output to relidx7
  and 0.58-0.67x the TPOT at >=16K on H200 and A100.
- Both OMMX bars carry a per-stage split; the HF-eager one comes from the same four
  kernel ablation flags, via figure/hf_abl_to_breakdown.py.
- README reduced to results: accuracy figures, the TPOT table with both OMMX paths and
  their KIVI ratios, the encoding A/B, and the three caveats that stop a row being
  over-read. Setup, reproduction, gates and recipe definitions live in the code's docstrings.
- `optimized-weight` now packs flat-bitmask positions: 5.3750 bits/weight instead of
  6.1250, at the same measured perplexity (9.0098, per window and in aggregate). The
  recipe's coverage is 25%, which is above the crossover where the mask becomes the
  cheaper plane; the preset does not generalise that to lower coverage.
- `w_packer budget` prints group size 16 and npv 1/32, the corners where the measured
  recipes sit.
- Figure 7 is now drawn from measured data only. Three of its four series previously had
  no reproducer in this repo and were digitized from the printed figure; all four are now
  produced by `repro/fig7_sweep.sh` and rendered by `figure/plot_fig7.py`, which names its
  ratio denominator on the axes and omits (rather than invents) an arm it has no data for.
- The A100/H200 end-to-end bench records the resolved recipe and vLLM version with each
  cell, and demotes a batch size whose sentinel says the request was served by bf16
  FlashAttention instead of the OMMX route.
- KV serving route fails loudly. A route that cannot run raises with the reason named
  instead of falling through to bf16, `ommx_route_health()` reports what actually fired,
  and slot allocation raises `OMMXSlotAllocationError` rather than reusing a slot.
- PACKED-ONLY KV mode: the bf16 page write is skipped so the OMMX sidecar is the only
  copy, which is what turns the format's byte saving into cache capacity.
- Flat-bitmap outlier positions on the KV axis, alongside relidx7 and combinadic.
- Named recipes are selectable with one flag (`OMMX_RECIPE` / `--recipe` / `--preset`) on
  the serving path, the bench and the eval harness.
- HF-eager arms measure batched decode and report a synchronized median step with
  percentiles, with an optional static KV cache so the arm is not paying allocation.
- KIVI baseline arms cover Llama and Mistral end to end, and the KVQuant adapter binds
  through the same registry as the other comparison arms.
- Task-harness comments are English throughout (CoQA and the five RULER utilities), which
  is the last Korean text on the evaluation path.
- LiveCodeBench-v6 KV-quantization comparison (`eval/lcb/`): four KV-quant methods and a
  bf16 control on one generative path, quantized at the `transformers` `Cache.update()`
  boundary so every arm shares a code path and MoE models are supported unchanged.
- `run.sh` validates that every flag was given a value instead of silently consuming the
  next flag, resolves the GPU it was asked for, and moves a stale output aside rather than
  collecting it as this run's result.
- README documents named recipes, the `ommx_w` offline-pack and serving flow, and what a
  quoted number does and does not cover.
- README is an artifact README: what the artifact contains, setup, reproduce, recipes,
  limitations, citation. Provenance and forensic detail live with the code that produced it
  rather than dropped, so no caveat a reader needs is lost.

- The flat-bitmask KV outlier encoding is served, not merely stored. All four decode call
  sites forward `k_obmp` and a resolved `bitmap_read`; a bitmap-packed engine used to pack
  cleanly and then raise at the first decode, asking for the relidx7 plane the pool had
  deliberately not allocated. At the recipe every published KV accuracy number was
  produced under (group 32, 6 outliers) the flat mask is the *smaller* plane -- 4 B against
  relidx7's 6 B, i.e. 4.375 -> 4.125 bit/element -- so it is cheaper exactly in the
  high-coverage regime the format targets. Still refused above 8 outliers per group, which
  is a kernel bound (the value stream is one int32), not a wiring one. VERIFIED ON AN
  H200: both encodings fire `DECODE_ROUTE_FIRED fmt=i2f4` at the shipped recipe over a
  1386-token context and return character-identical text, including the retrieval answer
  buried at the start of it. On the weight axis the same comparison was run end to end: a
  bitmap bundle packed from the same calibration serves at wikitext PPL 9.0098, identical
  to the relidx7 bundle per window and in aggregate, for 5.3750 bits/weight against
  6.1250 (7.0 GB -> 6.4 GB on disk).

- The OMMX page grid follows the engine's `cache_config.block_size` instead of a constant
  16, which happened to equal vLLM's own default and so agreed by coincidence. Preflight
  now reports `pages_per_group` and warns when a quantization group straddles blocks --
  the precondition for any block-granular design, since a group's scale, zero-point and
  outliers are a pure function of its own tokens but vLLM co-locates nothing.

- The weight kernel reads flat-bitmask outlier positions natively. `nc::idx_slot_pos`
  dispatches on an `idx_fmt` argument threaded from the recipe, so a bitmap bundle serves
  with no load-time re-encoding -- the re-encoding was lossless but spent the saving it
  existed to deliver (3.6250 on disk streamed at 4.1250). At the shipped
  `optimized-weight` recipe the mask is the cheaper plane: 8 B/group against relidx7's 14,
  i.e. 6.1250 -> 5.3750 bits/weight. Below about 12.5% coverage it is dearer, so the
  resident encoding is chosen rather than fixed, and only when a bundle is being rewritten
  for another reason. VERIFIED ON AN H200: the shipped parity gate grew a bitmap arm over
  the (group_size, npv) grid it never covered, and the two encodings agree in all 14 cells.

### Known issues
- The A100 linear microbench runs on a shared host and its absolute latencies move
  between runs: the same bf16 4096x4096 matmul measured 0.104, 0.079 and 0.041-0.107 ms
  in three runs (within a run the median-to-p99 spread is under 3%). Arms inside one
  batch point are timed back to back and remain comparable; levels across runs or across
  the two GPU panels are not. The H200 panel's bf16 baseline is flat (0.0345-0.0358 ms).
- `OMMX_ATTN_BATCHED_GRAPH=1` (opt-in, default off) aborts with a CUDA
  `scatter gather kernel index out of bounds` as soon as any request has a completed
  quantization group -- i.e. at any generation longer than `sink + recent + group_tokens`
  (72 tokens at the shipped recipe). Shorter runs pack nothing and pass, which is why no
  previous test of this path reached it. The default write path is unaffected. See
  the reproducer's own docstring for what is and is not known.
  Bisected to the device pack (`OMMX_KV_GPU_PACK=0` removes the assert) and then to a
  SYNCHRONISATION RACE: two host-side bounds checks added while diagnosing it take the run
  from 287 asserts and 0 completions to 0 asserts and 5 completions, while never firing
  themselves. Nothing they inspect is out of range; what changes the outcome is the
  synchronisation they impose, so the defect is an ordering one and the checks are not a
  fix for it.

### Fixed
- A recycled pool slot could serve the previous occupant's packed KV. Releasing a slot is
  bookkeeping only -- `assign_slots` is deliberately pool-free -- and the single reset of
  `seq_len`/`packed_groups` lived in the eager write path, gated on a restart. The
  capture-safe path documents that it touches neither, and `regroup` is monotone, so a
  stale-high count from the previous request was permanent: the new occupant's groups were
  never packed while `b_seq_len` (derived from the window boundary, not from
  `packed_groups`) still said there were packed tokens to read. `assign_slots` now reports
  which assignments are fresh and the caller clears the pool behind each.
- Preflight reports which copy of the package is running. An editable install and a
  `PYTHONPATH` can resolve to different checkouts and the loser leaves no trace, so every
  later result describes code nobody is reading. This project's own runs were served by a
  stale tree until the resolved path was checked by hand.
- Preflight's page-grid check read `_attr`'s `(value, found)` pair as a value, so its
  guard was permanently false and every field it reports stayed "unknown". It was silent
  in exactly the case it exists to flag. Verified on an H200 that `--block-size 32` starts
  the OMMX backend and gives `pages_per_group == 1`.
- The `M > 1` outlier correction is not run-to-run deterministic and never was:
  `sparse_correct_batched_relidx7_kernel` compacts its slots through
  `atomicAdd(&s_count, cnt)`, so the order corrections are summed in varies. Measured:
  one encoding differs from ITSELF by up to 9.8e-04 across two launches. A bit-exact
  assertion on that path would fail on the shipped code, so the parity gate compares the
  cross-encoding gap against the gap an encoding already has with itself.
- The pool-footprint preflight charged a combinadic rank for every non-relidx7 encoding,
  under-pricing a bitmap pool by 0.25 bit/element at group 32 and 0.625 at group 128. It
  is an OOM guard, so an estimate under the real allocation was the harmful direction.

### Removed
- `figure/out_h200canon/`, `out_h200fig7/`, `out_h200fig7_vsvllm/` (stale renders of a
  figure rebuilt from `data/`, each differing from the committed one and from each
  other) and `figure/data_demo/`, orphaned when the demo scripts were dropped.
- README's "Three things that stop a row being over-read" section; the bit-budget and
  dtype facts it carried are folded into the TPOT section rather than dropped.
- `fig7_tpot_vs_ctx_bitmap.png` (bitmap is the default, so it and `fig7_tpot_vs_ctx.png`
  now draw the same thing), `figure/data_repro/` (five of its seven files were
  byte-identical to `figure/data_bm/` and the other two are the pre-bitmap arms), the
  `assets/` demo casts and gifs, and the three demo-recording scripts. Nothing
  referenced any of them.
- README no longer says the `ommx_w` serving path is GPU-unverified; it has been served end to
  end and its accuracy measured through the CUDA kernel.
