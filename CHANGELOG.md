# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- Named-recipe registry entries now record their NPU-basis budget alongside the stored one.

### Fixed
- The solver chose INT2 codes against an f32 zero-point while storing bf16, so every code
  was slightly off. Rounding to the stored dtype before code selection takes the export
  round-trip from 2.4e-4 to 1.5e-8.
- The vLLM plugin reported "registered" from its own bookkeeping rather than from vLLM's
  registry, and one path set the flag without registering anything.

### Changed
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
