# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Named, measured, tested serving recipes — reproducing a published number is a FLAG.

WHY THIS MODULE EXISTS. Two of the release's headline configurations were reachable
only by hand-assembling environment variables:

  * the recipe every SHIPPED accuracy/latency number was produced under is a dict
    duplicated in ``figure/bench.py`` and ``eval/lm_eval/models/ommx_hf_model.py`` —
    the bare env default carries its numeric knobs (k=6, pow2 on, gt=gc=32) but encodes
    outlier positions as bitmap rather than its relidx7 (K+V 8.250 vs 8.750 bit/pair),
    so an operator who set nothing still measured a different footprint;
  * the recipes that hit the bit budgets the ICCAD paper's Table 1 reports (claims E1
    KV AvgBits 2.75 and E2 weight AvgBits 3.63) are neither the defaults nor documented
    anywhere as a unit.

Every entry below therefore carries (a) the exact env dict / packer argv that realises
it, (b) MEASURED bits recomputed from the code that allocates the planes — never a
figure copied out of a paper — and (c) an ``accuracy_status`` saying whether ANY
accuracy number in this repo belongs to it. That last field is the point: the repo's
accuracy statements were produced with the SHIPPED recipe, so Table 1 does NOT transfer
to a paper-budget recipe and the registry must say so rather than let a reader assume.

DEPENDENCIES: this MODULE's own imports are stdlib only — no torch, no triton, no
vLLM, no GPU — and the verification helpers (:func:`measure_kv`, :func:`measure_weight`)
import the measuring code LAZILY, inside the function, so that stays true.
CAVEAT, measured rather than assumed: that does NOT make ``python3 -m
ommx_gpu_serve.recipes`` runnable without torch, because importing it runs the PACKAGE
``__init__``, and ``ommx_gpu_serve/__init__`` -> ``attention.pack`` imports torch. With
torch blocked, ``import ommx_gpu_serve.recipes`` raises ``ImportError: torch``. So a
torch-free CI lane must load this file by path, not by module name; ``run.sh`` is fine
because it resolves the recipe with the ``ommx`` venv's python, which has torch.

ADDITIVITY. Nothing here runs unless a preset is NAMED (``OMMX_RECIPE`` /
``--preset`` / ``run.sh --recipe``). :func:`resolve_env` is a no-op returning
``None`` when ``OMMX_RECIPE`` is unset, and it never overwrites an env var the caller
already set, so shipped defaults and an explicit env are both untouched.

RESOLUTION MODEL (one thing, stated once — full rationale on :func:`resolve_env`):
EVERY reader of a recipe-controlled env var calls the idempotent
:func:`resolve_env` before it reads ``os.environ``; nothing else expands
``OMMX_RECIPE``. Readers, all of them wired: ``integration/vllm/config.py``,
``integration/vllm/packed_only.py``, ``attention/kv_pool.py``,
``attention/kv_store.py``, ``attention/pack.py``, ``linear/w_packer.py`` (weight
axis, via ``--preset``) and ``figure/bench.py``. ``integration/vllm/preflight.py``
takes an already-resolved ``OMMXServingConfig`` as an argument, so it cannot run
before the serving path has resolved.

  python3 -m ommx_gpu_serve.recipes list          # the table
  python3 -m ommx_gpu_serve.recipes show paper-kv # one entry in full
  python3 -m ommx_gpu_serve.recipes env  paper-kv --export   # eval-able export lines
  python3 -m ommx_gpu_serve.recipes verify        # recompute every measured number
"""
from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

#: Env var naming a preset. Read by ``integration/vllm/config.py`` (serving path) and
#: emitted by ``run.sh --recipe``.
RECIPE_ENV_VAR = "OMMX_RECIPE"

# ── env-name aliasing ────────────────────────────────────────────────────────────
#
# ``config.py::_env_int_alias`` and ``packed_only.py::_env_int_alias`` both read the
# CANONICAL (OMMX_KV_*) spelling FIRST and only then the back-compat OMMX_ATTN_* alias.
# So if a preset materialises the canonical name while the operator has exported the
# alias, the operator's explicit value is SILENTLY IGNORED — the exact failure mode this
# whole module exists to remove. Suppress the canonical key whenever any of its aliases
# is explicitly present, which restores "explicit env wins" for both spellings.
#
# NOT listed here on purpose: OMMX_OUTLIER_PERCENT vs OMMX_ATTN_OUTLIERS. That pair is
# already ordered the other way round in BOTH readers (percent is checked first and
# wins), so an explicitly-set percent overrides a preset's absolute npv with no help
# from us — and leaving OMMX_ATTN_OUTLIERS materialised keeps the recipe readable in
# `env` output instead of hiding a knob.
_ENV_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "OMMX_KV_GROUP_TOKENS": ("OMMX_ATTN_GROUP_TOKENS",),
    "OMMX_KV_GROUP_CHANNELS": ("OMMX_ATTN_GROUP_CHANNELS",),
    "OMMX_KV_SINK": ("OMMX_ATTN_SINK",),
    "OMMX_KV_RECENT": ("OMMX_ATTN_RECENT",),
}


class UnknownRecipeError(KeyError):
    """Raised for an unregistered preset name. Always carries the known names.

    A KeyError subclass so ``except KeyError`` still catches it, but ``str(exc)`` is a
    full sentence with the alternatives in it — a preset system whose failure mode is
    ``KeyError: 'paper_kv'`` teaches the operator nothing.
    """

    def __str__(self) -> str:  # KeyError.__str__ repr()s its single arg; undo that.
        return self.args[0] if self.args and isinstance(self.args[0], str) \
            else super().__str__()


@dataclass(frozen=True)
class NamedRecipe:
    """One named recipe: how to select it, what it MEASURES, and what it does not have.

    ``measured`` is the contract with the code: ``tests/test_recipes.py`` recomputes
    every value from the live implementation and fails if the registry has drifted.
    Values are exact dyadic rationals (halves/quarters/sixteenths of a bit), so they
    compare with ``==`` in float; only ``compression_ratio`` needs a tolerance.
    """

    name: str
    axis: str                                   # "kv" | "weight"
    summary: str
    #: ICCAD'26 claim ids this recipe answers (see PAPER_GPU_CLAIMS.md). Empty tuple
    #: for a recipe that exists to pin CURRENT behaviour rather than a paper number.
    paper_claims: Tuple[str, ...]
    #: KV: the environment that realises the recipe, materialised by
    #: :func:`resolve_env`. WEIGHT: empty — ``w_packer`` reads no environment at
    #: all (verified: no ``os.environ`` / ``getenv`` in linear/), it is argv-driven.
    env: Mapping[str, str]
    #: WEIGHT: kwargs for ``linear.w_packer.build_recipe``. KV: empty.
    packer_args: Mapping[str, Any]
    #: MEASURED bits. KV: k/v/avg/total bits-per-element + compression_ratio.
    #: WEIGHT: bits_per_weight (+ the unpadded information-theoretic twin).
    measured: Mapping[str, float]
    #: The exact command that produced ``measured`` — so the number is checkable by a
    #: reader, not just by the test.
    measured_by: str
    #: Does ANY accuracy number in this repo belong to this recipe? Read it before
    #: putting an accuracy figure next to a bit budget.
    accuracy_status: str
    #: Can the SHIPPED code actually serve it, or does it only pack/price?
    serving_status: str
    notes: Tuple[str, ...] = field(default_factory=tuple)

    # ── convenience ──────────────────────────────────────────────────────────────
    @property
    def is_kv(self) -> bool:
        return self.axis == "kv"

    def packer_argv(self) -> Tuple[str, ...]:
        """``w_packer pack`` flags equivalent to :attr:`packer_args` (for docs/logs)."""
        argv: list[str] = []
        for k, v in sorted(self.packer_args.items()):
            if v is None:
                continue
            argv += [f"--{k.replace('_', '-')}", str(v)]
        return tuple(argv)


# ════════════════════════════════════════════════════════════════════════════════
# THE REGISTRY
# ════════════════════════════════════════════════════════════════════════════════
#
# Provenance of every ``measured`` value below: recomputed on CPU, this engagement,
# by `python3 -m ommx_gpu_serve.recipes verify` (which calls the same two functions
# tests/test_recipes.py calls). No value here was transcribed from the paper, from
# README.md, or from a previous session's notes.

_SHIPPED_KV_ENV: Dict[str, str] = {
    # VERBATIM the dict that figure/bench.py (OMMX_RECIPE_ENV) and
    # eval/lm_eval/models/ommx_hf_model.py both set — i.e. the recipe behind every
    # published OMMX TPOT, peak-mem and CoQA/PPL number in this release …
    "OMMX_ATTN_K_FORMAT": "i2f4",
    "OMMX_ATTN_OUTLIERS": "6",
    "OMMX_ATTN_POW2": "1",
    "OMMX_ATTN_OUTLIER_SELECT": "signed",
    "OMMX_ATTN_OUTLIER_REPR": "relidx7",
    "OMMX_KV_GROUP_TOKENS": "32",
    "OMMX_KV_GROUP_CHANNELS": "32",
    "OMMX_KV_SINK": "8",
    "OMMX_KV_RECENT": "32",
    "OMMX_KV_RING": "1",
    "OMMX_KV_GPU_PACK": "1",
    # … PLUS one knob those two dicts leave implicit. OMMX_KV_OUTLIER_MAP defaults to
    # True, so setting "1" is behaviour-identical TODAY (test_recipes pins that), but a
    # published recipe must not depend on a default holding still: the two
    # k_fp4_map{scale,center} planes are +1.000 bit on K at gt=32, i.e. the difference
    # between 4.375 and 3.875 average bits. Name it.
    "OMMX_KV_OUTLIER_MAP": "1",
}

_PAPER_KV_ENV: Dict[str, str] = {
    # See "HOW paper-kv WAS CHOSEN" below. Everything not listed matches shipped-kv;
    # the recipe moves exactly three knobs: group_tokens 32->128, group_channels
    # 32->128, outliers 6->10.
    "OMMX_ATTN_K_FORMAT": "i2f4",
    "OMMX_ATTN_OUTLIERS": "10",
    "OMMX_ATTN_POW2": "1",
    "OMMX_ATTN_OUTLIER_SELECT": "signed",
    "OMMX_ATTN_OUTLIER_REPR": "relidx7",
    "OMMX_KV_GROUP_TOKENS": "128",
    "OMMX_KV_GROUP_CHANNELS": "128",
    "OMMX_KV_SINK": "8",
    "OMMX_KV_RECENT": "32",
    "OMMX_KV_RING": "1",
    "OMMX_KV_GPU_PACK": "1",
    "OMMX_KV_OUTLIER_MAP": "1",
}

# HOW paper-kv WAS CHOSEN (measurement, not assumption).
#
# The legal grid — group_tokens x group_channels in {16,32,64,128}^2 x outlier count
# 0..gt x kv_outlier_map on/off x pow2 on/off x outlier_repr in
# {relidx7, bitmap, combinadic} — was enumerated and priced by
# ``packed_only.kv_bits_breakdown``. That is 4*4*2*2 * sum(gt+1) = 3904 recipes per
# encoding and 11712 across the three; the relidx7 slice alone (filter 4 below) is the
# 3904. (Both counts are stated because the earlier draft of this comment listed the
# axes WITHOUT outlier_repr next to the 11712 total, which does not add up.)
# Filters, each with a reason:
#
#   1. |avg_bits - 2.75| == 0 exactly. Claim E1 is a hard number; 2.781 is not 2.75 and
#      must not be labelled as such. EIGHT recipes hit it exactly (below).
#   2. outliers_per_vector >= 1. k=0 is plain INT2 with no FP4 lane — not the
#      dual-resolution format Table 1 evaluates (claims B1/B2/B7).
#   3. pow2 ON. Claim B9 is explicit: "e is the shared power-of-two exponent stored as
#      INT8". A bf16 group scale is a different format.
#   4. outlier_repr = relidx7. Held FIXED, not enumerated, because it is the ONLY
#      position encoding the vLLM serving path decodes end to end; the paper's own
#      choice (B4, flat bitmask) also reaches 2.7500 exactly at gt=gc=128, k=3 or 4
#      — but ONLY with kv_outlier_map OFF (a 16-byte bitmask costs a flat 1.000 bit of
#      K at gt=128, so the two k_fp4_map* planes no longer fit inside 2.7500); stating
#      the k without the map condition would send a reader to a recipe that prices
#      2.8750, not 2.7500 (measured) —
#      but ``config.py``'s module docstring records that a bitmap-packed engine packs
#      fine and then RAISES at the first decode (backend.py never forwards
#      ``bitmap_read=``/``k_obmp=``). A preset that cannot serve is not a preset.
#
# The eight survivors, MEASURED (K / V / avg bit per element; every one totals 5.5000
# bit per (K,V) pair = avg 2.7500 = 5.818x vs bf16), ordered by outlier coverage:
#
#     cov%   gt   gc    k   map        K        V      avg   resid@4k
#     7.81  128  128   10   ON    3.3125   2.1875   2.7500     2.3125   <- CHOSEN
#     6.25   64   64    4   off   3.1250   2.3750   2.7500     1.3125
#     6.25  128   64    8   ON    3.1250   2.3750   2.7500     2.3125
#     5.47  128   64    7   ON    3.1250   2.3750   2.7500     2.3125
#     4.69  128   32    6   off   2.7500   2.7500   2.7500     2.3125
#     3.12   64   32    2   off   2.7500   2.7500   2.7500     1.3125
#     2.34  128   32    3   ON    2.7500   2.7500   2.7500     2.3125
#     1.56   64   64    1   ON    3.1250   2.3750   2.7500     1.3125
#
# CHOSEN gt=128 / gc=128 / k=10 / map ON, on three tie-breaks:
#
#   * HIGHEST OUTLIER COVERAGE (10/128 = 7.81%) of any exact hit. Among recipes that
#     cost the SAME number of bits, the one carrying the most outliers is strictly the
#     one to publish — outlier coverage is the paper's entire thesis ("High-Coverage
#     Outlier Representation"), and it is what an accuracy result would depend on.
#     k=10 is also the last free outlier at this budget: relidx7 pads to whole bytes,
#     so k=9 and k=10 both cost 9 index bytes while k=11 costs 10 and overshoots.
#   * gt=128 IS the paper's own worked-example group size (claim B10, "a group of
#     N = 128 elements").
#   * kv_outlier_map STAYS ON, so the DEQUANTIZATION SEMANTICS are identical to every
#     shipped accuracy number (dedicated FP4 range map, not base-shared outliers).
#     Only the group geometry and the outlier budget move. The map-off alternatives
#     above would change the arithmetic as well as the budget.
#
# HONEST COST, disclosed rather than buried: gt=128 quadruples the bf16 residual ring
# (kv_pool sizes it sink + recent + 2*gt = 296 rows/request vs 104 at gt=32). At
# seq_len=4096 with OMMX_KV_RING=1 that is +2.3125 bit/pair, so the EFFECTIVE ratio is
# 4.096x, against shipped-kv's 3.346x. paper-kv still wins on both figures; it just
# wins by less than 5.818x/3.657x suggests. ``kv_bits_breakdown(seq_len=4096)``
# reports it and :func:`measure_kv` returns it.

_SHIPPED_WEIGHT_ARGS: Dict[str, Any] = {
    # ``w_packer pack``'s own argparse defaults, spelled out. outlier_pct=0.0625 is
    # what `--outlier-pct` defaults to inside build_recipe (npv=None -> pct 0.0625 ->
    # derive_npv(64, 0.0625) = 4), written explicitly so the preset pins a VALUE rather
    # than pointing at a default that could move underneath it. test_recipes asserts
    # this resolves identically to the untouched parser defaults.
    "group_size": 64,
    "outlier_pct": 0.0625,
    "npv": None,
    "outlier_repr": "relidx7",
    "outlier_map": "idx_range",
    "zp_dtype": "bf16",
}

_PAPER_WEIGHT_ARGS: Dict[str, Any] = {
    # Claim E2 = 3.63 bits/weight. The packer's own budget path measures this exact
    # decomposition at gs=64, npv=4, flat bitmask positions, no range map:
    #   base_int2 2.0000 + scale_e8m0 8/64 + zero_point 16/64 (BF16 ZP, claim B9)
    #   + outlier_index 64/64 (the FLAT BITMASK of claim B4) + outlier_fp4 4*4/64
    #   + fp4_range_map 0  =  3.6250
    # Self-consistent with the paper's OWN two claims: B4 says the GPU stores positions
    # as a flat bitmask, and B6's bundle definition lists scale + zero-point + positions
    # + extension codes and NO range-map parameters. The shipped relidx7 + idx_range
    # encoding costs 0.5000 + 1.0000 instead of 1.0000 + 0.0000 and lands on 4.1250,
    # stepping straight past 3.63.
    "group_size": 64,
    "outlier_pct": None,
    "npv": 4,
    "outlier_repr": "bitmap",
    "outlier_map": "none",
    "zp_dtype": "bf16",
}

_OPTIMIZED_WEIGHT_ARGS: Dict[str, Any] = {
    # The knee of a MEASURED bits-vs-perplexity front, not a guess. See
    # "HOW optimized-weight WAS CHOSEN" below for the nine cells and their numbers.
    # relidx7 + idx_range because those are the two encodings the CUDA kernel reads
    # natively AND the only ones a calibrated bundle can use: `--calibrated` gates on
    # `dequant(planes) == the calibrated weight`, and outlier_map="none" re-encodes the
    # nibbles against a degenerate ms=1/mc=0 map, so the gate fails by construction.
    "group_size": 64,
    "outlier_pct": 0.25,
    "npv": None,
    # The flat bitmask, because at this recipe's 25% coverage it is the CHEAPER position
    # plane -- 8 bytes per group against relidx7's 14 -- and the kernel reads both
    # natively since the idx_fmt dispatch landed. Measured, not assumed: a bundle packed
    # from the same calibration serves at wikitext PPL 9.0098, identical to the relidx7
    # bundle per window and in aggregate, for 5.3750 bits/weight against 6.1250.
    # Below about 12.5% coverage relidx7 is the cheaper one; this is a property of high
    # coverage, not of the encoding.
    "outlier_repr": "bitmap",
    "outlier_map": "idx_range",
    "zp_dtype": "bf16",
}

# ── HOW optimized-weight WAS CHOSEN ──────────────────────────────────────────────
# Llama-3.1-8B-Instruct, wikitext-2-raw-v1 test, 4 x 4096-token windows, every quantized
# layer INT2 (--special-first-k 0 --special-last-k 0 --special-projs ""), error-feedback
# calibration exported per module and packed through `--calibrated`. BF16 control on the
# same windows: 6.2711 (simulator) / 6.2433 (served). The PPL column below is the
# SIMULATOR's -- it is what a nine-cell sweep can afford, one GPU-slot cell being ~9
# minutes against ~40 for a pack-and-serve. The two candidate cells were then packed and
# served, and that is where gs=128/npv=32 was rejected; see the paragraph after the table.
# Cells, cheapest NPU basis first:
#
#     gs   npv   %out   stored      NPU      PPL   vs bf16   Pareto
#    128     8   6.2%   3.3750   3.2578  18.8577    3.007x   front
#    128    16  12.5%   4.0625   3.7109  12.2442    1.952x   front
#     64     4   6.2%   4.1250   3.9375  5072.97  808.944x   DESTROYED
#     64     8  12.5%   4.7500   4.3906  11.4614    1.828x   front
#    128    32  25.0%   5.4375   4.4766   9.2593    1.477x   front  REJECTED (serves broken)
#     64    16  25.0%   6.1250   5.1406   8.7400    1.394x   front  <== chosen
#     32     8  25.0%   7.5000   6.5000   8.2845    1.321x   front
#     16     1   6.2%   8.5000   8.2500  10.8541    1.731x   dominated
#     16     4  25.0%  10.5000   9.1875   7.7082    1.229x   front
#
# Two things the grid settles that were previously confounded:
#   * The OUTLIER FRACTION is the variable that keeps the model alive, not the group
#     size: gs=64 goes from PPL 5072.97 at 6.25% to 8.7400 at 25%. Group size is a
#     second-order refinement on top (8.7400 -> 7.7082 going 64 -> 16 at fixed 25%).
#   * Calibration is a refinement, not the rescue. At gs=16/25% the uncalibrated RTN
#     bundle serves at PPL 9.7265 against the calibrated 7.8950 (both through the CUDA
#     kernel) -- degraded, not destroyed. The earlier pairing of "calibrated gs16/25%
#     works" with "RTN gs64/6.25% is destroyed" moved both knobs at once and could not
#     have shown this.
#
# WHY gs=64 AND NOT THE CHEAPER gs=128/npv=32. On the simulator gs=128/npv=32 looks like
# the knee (4.4766 NPU bits, PPL 9.2593 against gs=64/npv=16's 5.1406 and 8.7400). SERVED,
# it is not usable. Per-window perplexity through the CUDA kernel, same four windows,
# reproduced exactly across two runs:
#
#     gs=128/npv=32   [ 9.84, 14.68,  8.61, 248947.69 ]   aggregate 132.65
#     gs=64 /npv=16   [ 9.65, 13.11,  7.32,      7.12 ]   aggregate   9.01
#
# The kernel is NOT at fault, at four levels of check: the pack gate (CPU dequant == the
# calibrated weight, all 224 modules); a synthetic parity sweep over the whole gs x npv
# grid (gs=128/npv=32 cos 0.999997); and the REAL bundle's planes at both ABIs, where the
# two bundles are indistinguishable (M=32 median cos 0.99999774 for gs=128 against
# 0.99999768 for gs=16). Note the shipped gate in csrc/linear/test_ommx_linear_parity.py
# is hardcoded at gs=64/npv=16, so it had never covered gs=128/npv=32 either way.
#
# What the 512-token block profile of the failing window shows is DIVERGENCE, not a spike:
#     gs=128 window 4:  [5.83, 12.29, 13.30, 13.35, 13.14, 13.83, 13.84, 13.82]
#     gs=64  window 4:  [2.02,  1.68,  1.86,  2.06,  2.14,  1.97,  2.02,  1.95]
# The model leaves the rails inside the first block and never returns; only 0.049% of
# positions exceed NLL 20, so it is confidently wrong everywhere rather than broken at one
# token (mean NLL 12.43 is worse than uniform over the 128k vocabulary, ln 128256 = 11.76).
# The SIMULATOR runs the same weights over the same window and stays at ~5.9, so gs=128
# sits on the edge of stability: it survives an eager transformers forward and diverges
# under vLLM's prefill. That is also why simulator perplexity alone cannot select a recipe.
#
# So the knee is taken at gs=64/npv=16: 0.6640 NPU bits more than gs=128/npv=32, and the
# only one of the two that serves. One step further down the front (gs=64/npv=8) saves
# 0.7500 NPU bits and costs 31% perplexity. The front is flat above 25% coverage and falls
# off a cliff below it.


RECIPES: Dict[str, NamedRecipe] = {
    "shipped-kv": NamedRecipe(
        name="shipped-kv",
        axis="kv",
        summary="The canonical published KV recipe: INT2 base + 6 FP4 K-outliers per "
                "32-token group, dedicated FP4 range map, INT8 pow2 scales, relidx7 "
                "positions, sink 8 + recent 32.",
        paper_claims=(),           # pins CURRENT behaviour; it does NOT reach E1's 2.75
        env=dict(_SHIPPED_KV_ENV),
        packer_args={},
        measured={
            "k_bits_per_elem": 6.0,
            "v_bits_per_elem": 2.75,
            "avg_bits_per_elem": 4.375,
            "total_bits_per_elem": 8.75,
            "compression_ratio": 32.0 / 8.75,          # 3.6571...
            "residual_bits_per_elem_at_4k": 0.8125,
            "effective_compression_ratio_at_4k": 32.0 / (8.75 + 0.8125),   # 3.3464...
        },
        measured_by="ommx_gpu_serve.integration.vllm.packed_only.kv_bits_breakdown("
                    "128, **shipped-kv recipe, seq_len=4096)",
        accuracy_status=(
            "MEASURED — this is the ONLY KV recipe with accuracy behind it in this "
            "repo. Every published OMMX accuracy number (CoQA/PPL through "
            "eval/lm_eval/models/ommx_hf_model.py, the 4K/16K needle regression gate) "
            "and every published TPOT/peak-mem number was produced under exactly this "
            "env. NOTE it measures 4.375 avg bits, NOT the paper's Table 1 AvgBits of "
            "2.75 (claim E1) — so these accuracy numbers do not belong to the paper's "
            "bit budget either."),
        serving_status=(
            "SERVABLE — but no longer the bare default: naming it reverts the engine's "
            "outlier_repr from bitmap (the serving default) to relidx7. Both encodings "
            "and the FP4 range map are decoded by backend.py + the Triton decode kernel."),
        notes=(
            "A bare run (no env at all) matches every numeric knob here (k=6, pow2, "
            "gt=gc=32, map on, sink 8, recent 32) and differs in ONE: the position "
            "encoding, bitmap (the serving default, K+V 8.250 / avg 4.125) instead of "
            "this preset's relidx7 (8.750 / 4.375). Same positions, same values, "
            "bit-identical decode; only the footprint moves. Naming this preset is how "
            "you get the published 4.375 rather than the bare 4.125.",
        ),
    ),
    "paper-kv": NamedRecipe(
        name="paper-kv",
        axis="kv",
        summary="Reaches the paper's Table 1 KV budget EXACTLY: 2.7500 average bits. "
                "INT2 base + 10 FP4 K-outliers per 128-token group, gc=128, dedicated "
                "FP4 range map, INT8 pow2 scales, relidx7 positions.",
        paper_claims=("E1", "B9", "B10"),
        env=dict(_PAPER_KV_ENV),
        packer_args={},
        measured={
            "k_bits_per_elem": 3.3125,
            "v_bits_per_elem": 2.1875,
            "avg_bits_per_elem": 2.75,                 # EXACT, not "approximately 2.75"
            "total_bits_per_elem": 5.5,
            "compression_ratio": 32.0 / 5.5,           # 5.8181...
            "residual_bits_per_elem_at_4k": 2.3125,
            "effective_compression_ratio_at_4k": 32.0 / (5.5 + 2.3125),    # 4.0960...
        },
        measured_by="ommx_gpu_serve.integration.vllm.packed_only.kv_bits_breakdown("
                    "128, **paper-kv recipe, seq_len=4096)",
        accuracy_status=(
            "NONE — THIS REPO HAS NO ACCURACY MEASUREMENT AT THIS RECIPE. Every OMMX "
            "accuracy number here was produced with shipped-kv (4.375 avg bits, k=6 in "
            "32-token groups). paper-kv changes the group geometry (32->128 on both "
            "axes) and the outlier budget (6->10), which changes what is quantized "
            "together and how many outliers survive; Table 1's accuracy figures "
            "therefore DO NOT TRANSFER to it. Reproducing 2.75 avg bits is a byte "
            "claim only. Measure accuracy under this preset before pairing the two."),
        serving_status=(
            "SERVABLE — relidx7 positions, dedicated FP4 map and INT8 pow2 scales are "
            "all on the decoded path; MultiSeqKVPool builds at gt=gc=128, k=10 (plane "
            "shapes verified on CPU). UNMEASURED: latency. Larger groups move the "
            "regroup cadence from every 32 decode steps to every 128, which changes "
            "both the spike period and its size, and no GPU was available to time it."),
        notes=(
            "Chosen by enumerating 11712 legal recipes and pricing each one; 8 hit "
            "2.7500 exactly and this is the one with the highest outlier coverage "
            "(7.81%). See 'HOW paper-kv WAS CHOSEN' in this module for the shortlist.",
            "Costs a 4x bigger bf16 residual ring than shipped-kv (296 vs 104 rows per "
            "request): the effective ratio at 4K context is 4.096x, not 5.818x.",
        ),
    ),
    "shipped-weight": NamedRecipe(
        name="shipped-weight",
        axis="weight",
        summary="Today's w_packer defaults: group 64, 4 FP4 outliers per group "
                "(6.25%), relidx7 positions, idx_range FP4 map, BF16 zero-point.",
        paper_claims=(),           # pins CURRENT behaviour; it does NOT reach E2's 3.63
        env={},
        packer_args=dict(_SHIPPED_WEIGHT_ARGS),
        measured={
            "bits_per_weight": 4.125,
            "bits_per_weight_unpadded": 4.0625,
            "bits_per_weight_npu": 3.9375,
        },
        measured_by="python3 -m ommx_gpu_serve.linear.w_packer budget "
                    "--group-sizes 64 --npvs 4   (row: relidx7 / idx_range)",
        accuracy_status=(
            "NO TASK ACCURACY at any weight recipe in this repo. What exists for the "
            "weight axis is a KERNEL PARITY gate (csrc/linear/test_ommx_linear_parity"
            ".py, 5/5 PASS on sm_90a), which shows the kernel dequantizes what the "
            "quantizer packed — not what the quantization costs on a benchmark."),
        serving_status=(
            "SERVABLE — relidx7 + idx_range is exactly what linear_method.py's "
            "KERNEL_OUTLIER_REPRS / KERNEL_OUTLIER_MAPS admit."),
        notes=(
            "4.1250 = 2 (INT2 base) + 0.1250 (E8M0 scale) + 0.2500 (BF16 ZP) + 0.5000 "
            "(relidx7 positions, 4 bytes/group) + 0.2500 (FP4 codes) + 1.0000 (the "
            "fp32 idx_range map pair). The map alone is a quarter of the budget.",
        ),
    ),
    "paper-weight": NamedRecipe(
        name="paper-weight",
        axis="weight",
        summary="Reaches the paper's Table 1 weight budget EXACTLY: 3.6250 "
                "bits/weight. Group 64, 4 FP4 outliers, FLAT BITMASK positions, no "
                "range map, BF16 zero-point.",
        paper_claims=("E2", "B4", "B6", "B9"),
        env={},
        packer_args=dict(_PAPER_WEIGHT_ARGS),
        measured={
            "bits_per_weight": 3.625,                  # EXACT
            "bits_per_weight_unpadded": 3.625,         # bitmap is byte-aligned by design
            # NOT 3.625. See the last note on this entry: on the NPU's own position
            # encoding this recipe costs 2.9375, so "3.63 is the NPU number" and
            # "paper-weight reaches 3.63" cannot both be true of the same recipe.
            "bits_per_weight_npu": 2.9375,
        },
        measured_by="python3 -m ommx_gpu_serve.linear.w_packer budget "
                    "--group-sizes 64 --npvs 4   (row: bitmap / none)",
        accuracy_status=(
            "NONE — no accuracy measurement in this repo at this or any other weight "
            "recipe (see shipped-weight). 3.6250 is a BYTE claim, verified from the "
            "layout; it says nothing about what the model scores."),
        serving_status=(
            "PACKS; SERVES ONLY VIA AN OPT-IN TRANSCODE, AND NOT AT 3.6250 RESIDENT. "
            "`w_packer pack --preset paper-weight` writes a valid bundle and `verify` "
            "accepts it, but the shipped CUDA weight kernel cannot read it directly: "
            "csrc/linear/ommx_linear.cu's sparse_correct decodes a relidx7 slot stream "
            "(no bitmap reader) and takes map_scale / map_center as required arguments. "
            "By default linear_method.py::require_kernel_readable RAISES with the fix "
            "named rather than mis-decoding — loud, never silent. With OMMX_W_TRANSCODE "
            "the bundle is re-encoded losslessly at load (bitmap -> relidx7 positions; "
            "the degenerate ms=1, mc=0 map planes synthesised), and THAT IS WHERE THE "
            "PAPER'S BUDGET STOPS BEING THE SERVED BUDGET: measured on-disk 3.6250 "
            "bit/weight, measured RESIDENT 4.1250 — i.e. exactly shipped-weight's "
            "budget, +0.5000 bit/weight over the disk figure. Quote 3.6250 as a STORAGE "
            "claim; the HBM traffic of a served paper-weight bundle is 4.1250."),
        notes=(
            "3.6250 = 2 (INT2 base) + 0.1250 (E8M0 scale) + 0.2500 (BF16 ZP) + 1.0000 "
            "(flat 64-bit bitmask, claim B4) + 0.2500 (4 FP4 codes) + 0 (no range map, "
            "claim B6). Verified against the packer's own --dry-run plane budget.",
            "ON THE NPU BASIS THIS RECIPE IS 2.9375, NOT 3.6250. The paper specifies "
            "the NPU as storing positions in ceil(log2 C(N,K)) bits per group, which at "
            "gs=64/npv=4 is 20 bits against the flat bitmask's 64 — so the same bundle "
            "prices at 2.9375 there. That matters because 3.63 has been described as an "
            "NPU-encoding figure: it cannot be, for THIS recipe. Scanning the whole "
            "legal grid, an NPU basis of exactly 3.6250 occurs at gs=128 / npv=21 "
            "(16.4% coverage, no range map, f32 ZP), with the near neighbours all "
            "between 9% and 19% coverage — an order of magnitude more outliers than the "
            "6.25% here. Reproduce with `w_packer budget` and read the npu column.",
            "The bitmask is FLAT in k: it costs 1.000 bit/weight at any outlier count, "
            "so raising npv above 4 buys coverage at only 0.0625 bit/weight each.",
            "On-disk 3.6250 != resident 4.1250. The load-time transcode that makes this "
            "bundle servable (OMMX_W_TRANSCODE) re-encodes the flat bitmask back to "
            "relidx7 and materialises the degenerate range-map planes, so the paper's "
            "budget survives in the FILE and not in HBM. Measured with "
            "w_format.plan_transcode; gated by tests/test_recipes.py.",
        ),
    ),
    "optimized-weight": NamedRecipe(
        name="optimized-weight",
        axis="weight",
        summary="The measured bits/perplexity knee that actually SERVES: group 64, 16 "
                "FP4 outliers per group (25%), FLAT BITMASK positions, idx_range FP4 "
                "map, BF16 zero-point. 5.3750 bits/weight stored, 5.1406 on the NPU "
                "basis.",
        paper_claims=(),          # measured HERE; it does not reproduce any published row
        env={},
        packer_args=dict(_OPTIMIZED_WEIGHT_ARGS),
        measured={
            "bits_per_weight": 5.375,
            "bits_per_weight_unpadded": 5.375,
            "bits_per_weight_npu": 5.1406,
        },
        measured_by="python3 -m ommx_gpu_serve.linear.w_packer budget "
                    "--group-sizes 64 --npvs 16   (row: bitmap / idx_range); "
                    "PPL from ommx_fakequant/wq_eval.py --group-size 64 "
                    "--outlier-percent 0.25 --pow2 --special-first-k 0 "
                    "--special-last-k 0 --special-projs '' --ppl-seqlen 4096 --limit 4, "
                    "then packed with --calibrated and re-measured THROUGH vLLM",
        accuracy_status=(
            "MEASURED THROUGH THE SERVING KERNEL, and the only weight recipe here that "
            "is. Llama-3.1-8B-Instruct, wikitext-2-raw-v1 test, 4 x 4096-token windows, "
            "every quantized layer INT2, error-feedback calibration carried into the "
            "bundle by --calibrated: served PPL 9.0098 against a 6.2433 BF16 control "
            "measured on the same windows through the same harness (x1.443). The "
            "simulator scores the same recipe at 8.7400, i.e. the kernel reproduces the "
            "quantizer to within 3.1%. That is ONE model, ONE dataset, and a perplexity "
            "— it is not MMLU/ARC-C/HellaSwag/RULER, so it does not substitute for a "
            "Table 1 row. See 'HOW optimized-weight WAS CHOSEN' for the nine-cell grid "
            "this was selected from and for why the cheaper gs=128/npv=32 point was "
            "rejected despite scoring better on the simulator."),
        serving_status=(
            "SERVED, END TO END, and the only weight recipe in this registry that has "
            "been. relidx7 + idx_range is what linear_method.py's KERNEL_OUTLIER_REPRS / "
            "KERNEL_OUTLIER_MAPS admit, so it loads with no transcode and no environment "
            "variable: `vllm serve <bundle> --quantization ommx_w` resolves it from the "
            "stamped quantization_config. Kernel firing confirmed by sentinel "
            "(OMMX_W_KERNEL_BUILT / WEIGHTS_READY gs=64 npv=16 fmt=i2f4 / PREFILL_FIRED "
            "/ OUTLIER_FIRED / DECODE_FIRED). Both kernel bounds are still comfortably "
            "clear: relidx7 tops out at group_size 128 (7-bit positions, "
            "w_format.RELIDX7_MAX_GROUP_SIZE) and sparse_correct asserts npv <= 32."),
        notes=(
            "5.3750 = 2 (INT2 base) + 0.1250 (E8M0 scale) + 0.2500 (BF16 ZP) + 1.0000 "
            "(flat 64-bit mask) + 1.0000 (16 FP4 codes) + 1.0000 (fp32 idx_range map "
            "pair). The same recipe with relidx7 positions is 6.1250: 14 bytes per group "
            "instead of 8, because 16 outliers at 7 bits each is more than 64 bits.",
            "5.1406 on the NPU basis: the same bundle with positions re-encoded as the "
            "paper's combinatorial rank, ceil(log2 C(64,16)) = 49 bits per group instead "
            "of the mask's 64. Every other plane is unchanged -- that is the paper's own "
            "'position-metadata regeneration per target'. The GPU form is now within "
            "0.23 bit of it.",
            "THE POSITION ENCODING IS NOW A FREE 0.75 BITS. The kernel reads the flat "
            "bitmask natively, and at this recipe's 25% coverage the mask is the cheaper "
            "plane: 8 B/group against relidx7's 14, i.e. 6.1250 -> 5.3750 stored (7.0 GB "
            "-> 6.4 GB on disk for Llama-3.1-8B). Packed from the SAME calibration and "
            "served through the same harness it scores 9.0098 -- identical to four "
            "decimal places, per window as well as in aggregate. `--outlier-repr bitmap` "
            "is therefore strictly better here; it is kept off the preset only because "
            "the shipped bundles are relidx7 and a recipe should describe what was "
            "measured under it, which is both.",
            "It is NOT sub-4-bit. Dropping the range map would take the NPU basis to "
            "4.1406, still above 4, and outlier_map='none' cannot be produced by the "
            "calibration path anyway (the pack gate compares the packed planes against "
            "the calibrated weight, and 'none' re-encodes the nibbles). Sub-4 on the NPU "
            "basis WAS measured, at gs=128/npv=16: 3.7109 bits, simulator PPL 12.2442 "
            "(x1.952 over BF16) -- degraded but not destroyed, and not served.",
            "Against the gs=16/npv=4 bundle served earlier: 42% fewer stored bits and "
            "44% fewer NPU bits (11 GB -> 7.0 GB on disk) for 14% higher served "
            "perplexity (7.8950 -> 9.0098).",
        ),
    ),
}

#: Registration order is the display order; sorted() would interleave the two axes.
RECIPE_NAMES: Tuple[str, ...] = tuple(RECIPES)


# ════════════════════════════════════════════════════════════════════════════════
# lookup + application
# ════════════════════════════════════════════════════════════════════════════════

def names(axis: Optional[str] = None) -> Tuple[str, ...]:
    """Registered preset names, optionally restricted to ``"kv"`` / ``"weight"``."""
    if axis is None:
        return RECIPE_NAMES
    return tuple(n for n, r in RECIPES.items() if r.axis == axis)


def get(name: str, *, axis: Optional[str] = None) -> NamedRecipe:
    """Look up a preset. Unknown names RAISE and LIST the alternatives — never a
    silent fallback to the default recipe, which would produce a plausible number
    under a recipe nobody asked for."""
    key = str(name).strip()
    rec = RECIPES.get(key)
    if rec is None:
        raise UnknownRecipeError(
            f"unknown OMMX recipe {name!r}. Known recipes: "
            f"{', '.join(RECIPE_NAMES)}. "
            f"Run `python3 -m ommx_gpu_serve.recipes list` for the measured bits of "
            f"each, or unset the preset to keep the shipped defaults.")
    if axis is not None and rec.axis != axis:
        # A KV preset passed to the weight packer would set nothing at all and the
        # operator would get the defaults while believing they had selected a recipe.
        raise UnknownRecipeError(
            f"recipe {key!r} is a {rec.axis.upper()} recipe and cannot be applied on "
            f"the {axis.upper()} axis. {axis.upper()} recipes: "
            f"{', '.join(names(axis))}.")
    return rec


def preset_env(name: Optional[str]) -> Dict[str, str]:
    """The env dict a named KV preset materialises (``{}`` for ``None``/empty)."""
    if name is None or str(name).strip() == "":
        return {}
    return dict(get(name, axis="kv").env)


def _present(environ: Mapping[str, str], key: str) -> bool:
    """Mirror of ``config.py::_env_present``: an EMPTY value counts as absent, because
    that is how every reader in this repo treats it."""
    v = environ.get(key)
    return v is not None and str(v).strip() != ""


def pending_env(name: Optional[str], environ: Optional[Mapping[str, str]] = None
                ) -> Dict[str, str]:
    """Keys the preset WOULD set against ``environ`` — i.e. after removing every knob
    the caller has already set explicitly (and every knob whose back-compat alias the
    caller has set; see ``_ENV_ALIASES``). Pure: it mutates nothing.
    """
    env = os.environ if environ is None else environ
    out: Dict[str, str] = {}
    for key, val in preset_env(name).items():
        if _present(env, key):
            continue                                    # explicit env wins
        if any(_present(env, a) for a in _ENV_ALIASES.get(key, ())):
            continue                                    # explicit ALIAS wins
        out[key] = val
    return out


def env_preset_name(axis: Optional[str] = None,
                    environ: Optional[Mapping[str, str]] = None,
                    note: Optional[Any] = None) -> Optional[str]:
    """The preset ``OMMX_RECIPE`` names, restricted to ``axis``. Pure: mutates nothing.

    The WEIGHT axis has no environment of its own (``w_packer`` is argv-driven —
    verified: no ``os.environ`` read anywhere under ``linear/``), so it cannot use
    :func:`resolve_env`. It still needs the SAME name resolution, or ``OMMX_RECIPE``
    means one thing on one axis and nothing on the other — which is the defect this
    work exists to remove, just spelled differently.

    Three outcomes, none of them silent:

      * unset / empty            -> ``None``; nothing is touched.
      * a name on a DIFFERENT axis -> ``None``, after calling ``note(message)`` if the
        caller supplied one. Not a raise: ``OMMX_RECIPE=paper-kv`` is a perfectly
        sensible thing to have exported for a serving run, and it genuinely carries no
        weight knobs, so a weight tool must keep its own defaults — but it says so out
        loud rather than leaving the operator to assume the preset applied.
      * an UNKNOWN name          -> :class:`UnknownRecipeError`, listing the known
        names, exactly as every other entry point does.
    """
    env = os.environ if environ is None else environ
    raw = env.get(RECIPE_ENV_VAR)
    if raw is None or str(raw).strip() == "":
        return None
    want = str(raw).strip()
    rec = get(want)                                     # raises + lists on a typo
    if axis is not None and rec.axis != axis:
        if note is not None:
            note(f"{RECIPE_ENV_VAR}={rec.name} is a {rec.axis.upper()} recipe and "
                 f"carries no {axis.upper()} knobs; keeping the {axis} defaults. "
                 f"{axis.upper()} recipes: {', '.join(names(axis))}.")
        return None
    return rec.name


def resolve_env(environ: Optional[Any] = None, name: Optional[str] = None
                ) -> Optional[str]:
    """THE resolution point for ``OMMX_RECIPE``. Every reader calls this FIRST.

    Materialises a named KV preset into ``environ`` (default ``os.environ``) and
    returns the applied preset name, or ``None`` when no preset was requested — in
    which case NOTHING is touched, which is what makes the feature additive.

    ── THE RESOLUTION MODEL, stated once ────────────────────────────────────────
    Model (a): EVERY reader of a recipe-controlled environment variable calls this
    idempotent function before it reads, and no reader anywhere expands
    ``OMMX_RECIPE`` itself. The alternative — expand exactly once, as early as
    possible, by whoever owns process start — was rejected because THIS PACKAGE HAS
    NO SINGLE PROCESS OWNER. It is entered from at least five disjoint places that
    never see each other: the vLLM backend, the HF-eager modeling files,
    ``figure/bench.py``, the ``w_packer`` CLI, and a bare ``python3 -c`` that imports
    one accounting helper. "Whoever ran first" IS the defect being fixed here:
    ``OMMX_RECIPE=paper-kv`` used to be expanded only inside
    ``config.resolve_serving_config``, so in any process where that had not already
    run the preset silently did nothing and the operator got the SHIPPED numbers
    (measured: ``kv_bits_breakdown(128)['avg_bits_per_elem']`` 4.125 before, 2.75
    after, in one process, same call). Model (a) is immune to import order and to a
    reader added later, because the obligation travels with the READ, not with a
    start-up sequence somebody must remember to run.

    Model (a) COMPOSES with (b) rather than competing with it: ``run.sh --recipe``
    exports the assignments into the shell, so a child process starts with the knobs
    already in its environment; this function then finds every key present and writes
    nothing. Same resolved environment either way.

    ── WHY MATERIALISE INTO THE ENVIRONMENT rather than thread a dict around ──
    The KV recipe has many independent readers that each go to ``os.environ`` on
    their own — ``config.py``, ``packed_only``'s accounting, ``kv_pool``,
    ``kv_store``, ``pack``, ``preflight`` and the HF-eager modeling files. A preset
    that reached only one of them would make the ENGINE run recipe A while the
    ACCOUNTING reported recipe B — a silent, plausible-looking disagreement, i.e.
    exactly the bug class this module exists to remove. Writing the knobs into the
    process environment is what makes all of them agree.

    ── PRECEDENCE (unchanged, and deliberately so) ──
    explicit env  >  preset  >  built-in defaults. The write uses ``pending_env``,
    which drops every key the caller has already set AND every key whose back-compat
    ``OMMX_ATTN_*`` alias is set, so an operator can take a preset and move one knob.

    ── IDEMPOTENCE is structural, not remembered ──
    There is deliberately NO CACHE and NO MODULE-LEVEL LATCH. The function re-reads
    ``environ`` every time and only ever fills keys that are ABSENT, so the second
    call finds ``pending_env`` empty and writes nothing. A latch would be faster by
    one dict lookup and would buy a way for one caller (or one test) to freeze the
    resolution for every later one — which is how a gate ends up passing for the
    wrong reason. The common path when ``OMMX_RECIPE`` is unset is a single
    ``environ.get`` returning ``None``, which is cheap enough for the pack loop.

    An unknown name RAISES :class:`UnknownRecipeError` listing the known names —
    from every reader, not just from the serving path. Falling back to the default
    recipe after the operator asked for a specific one is how you publish a number
    under a recipe nobody selected.
    """
    env = os.environ if environ is None else environ
    want = name if name is not None else env.get(RECIPE_ENV_VAR)
    if want is None or str(want).strip() == "":
        return None                                     # additive: touch nothing
    rec = get(want, axis="kv")                          # raises + lists on a typo
    for key, val in pending_env(rec.name, env).items():
        env[key] = val
    return rec.name


def apply_recipe_env(environ: Optional[Any] = None, name: Optional[str] = None
                     ) -> Optional[str]:
    """Back-compat alias for :func:`resolve_env` — same function, older name.

    Kept because it is the name the first version of this feature shipped under and
    it is referenced from prose in ``run.sh`` and ``config.py``. New code calls
    ``resolve_env``: there must be exactly ONE resolution point with ONE name, or the
    "did anybody resolve yet?" question comes back in a different spelling.
    """
    return resolve_env(environ, name)


# ════════════════════════════════════════════════════════════════════════════════
# verification — recompute every ``measured`` value from the live code
# ════════════════════════════════════════════════════════════════════════════════
#
# Imports are LAZY (inside the functions) so this module keeps its stdlib-only import
# contract: `python3 -m ommx_gpu_serve.recipes env paper-kv` must work in a shell with
# no torch, which is how run.sh calls it.

def measure_kv(rec: NamedRecipe, head_dim: int = 128, seq_len: int = 4096
               ) -> Dict[str, float]:
    """Price a KV preset with the SHIPPED accounting, from its env dict alone.

    Deliberately passes explicit kwargs instead of mutating ``os.environ``: the
    measurement must not depend on ambient state, and a pytest that measured through
    the environment would be measuring its own fixture.
    """
    from ommx_gpu_serve.integration.vllm.packed_only import kv_bits_breakdown

    env = rec.env
    b = kv_bits_breakdown(
        head_dim,
        k_format=env["OMMX_ATTN_K_FORMAT"],
        group_tokens=int(env["OMMX_KV_GROUP_TOKENS"]),
        group_channels=int(env["OMMX_KV_GROUP_CHANNELS"]),
        outliers_per_vector=int(env["OMMX_ATTN_OUTLIERS"]),
        outlier_repr=env["OMMX_ATTN_OUTLIER_REPR"],
        kv_outlier_map=env["OMMX_KV_OUTLIER_MAP"] not in ("0", "false", "off", "no"),
        use_pow2=env["OMMX_ATTN_POW2"] not in ("0", "false", "off", "no"),
    )
    # The residual ring is env-driven inside kv_bits_breakdown (OMMX_KV_RING / SINK /
    # RECENT have no kwargs), so it is derived here from the SAME formula the pool uses
    # — sizing comment in kv_pool.MultiSeqKVPool.__init__, mirrored in
    # kv_bits_breakdown: rows = sink + recent + 2*group_tokens, priced as two bf16
    # tensors over seq_len tokens.
    ring_on = env.get("OMMX_KV_RING", "0") not in ("0", "false", "off", "no")
    rows = (int(env["OMMX_KV_SINK"]) + int(env["OMMX_KV_RECENT"])
            + 2 * int(env["OMMX_KV_GROUP_TOKENS"])) if ring_on else None
    resid = (2 * 2 * 8.0) * (rows / float(seq_len)) if rows is not None else (2 * 2 * 8.0)
    total = float(b["total_bits_per_elem"])
    return {
        "k_bits_per_elem": float(b["k_bits_per_elem"]),
        "v_bits_per_elem": float(b["v_bits_per_elem"]),
        "avg_bits_per_elem": float(b["avg_bits_per_elem"]),
        "total_bits_per_elem": total,
        "compression_ratio": float(b["compression_ratio"]),
        "residual_bits_per_elem_at_4k": resid,
        "effective_compression_ratio_at_4k": 32.0 / (total + resid),
    }


def measure_weight(rec: NamedRecipe) -> Dict[str, float]:
    """Price a weight preset with the packer's OWN budget path (``Recipe`` built by
    ``w_packer.build_recipe``, priced by ``w_format.Recipe.bits_breakdown``) — the same
    two calls `w_packer budget` and `pack --dry-run` make."""
    from ommx_gpu_serve.linear.w_packer import build_recipe

    bb = build_recipe(**dict(rec.packer_args)).bits_breakdown()
    # Rounded to the packer's own printed precision: the NPU basis carries a
    # ceil(log2 C(gs,npv)) term that is not a dyadic rational, so an exact float would
    # make every registry entry a 17-digit literal for no gain.
    return {"bits_per_weight": float(bb["bits_per_weight"]),
            "bits_per_weight_unpadded": float(bb["bits_per_weight_unpadded"]),
            "bits_per_weight_npu": round(float(bb["bits_per_weight_npu"]), 4)}


def measure(rec: NamedRecipe) -> Dict[str, float]:
    """Dispatch on the axis. Returns the same key set as ``rec.measured``."""
    return measure_kv(rec) if rec.is_kv else measure_weight(rec)


# ════════════════════════════════════════════════════════════════════════════════
# rendering
# ════════════════════════════════════════════════════════════════════════════════

def format_table(recipes: Optional[Iterable[NamedRecipe]] = None) -> str:
    """One line per preset: the measured bits and whether accuracy exists for it."""
    rows = list(recipes) if recipes is not None else list(RECIPES.values())
    out = [f"{'recipe':<16}{'axis':<8}{'bits':>9}  {'claim':<12}{'accuracy in repo':<18}"
           "serving",
           "-" * 90]
    for r in rows:
        if r.is_kv:
            bits = f"{r.measured['avg_bits_per_elem']:.4f}"
        else:
            bits = f"{r.measured['bits_per_weight']:.4f}"
        claim = ",".join(r.paper_claims) or "-"
        acc = "yes (shipped)" if r.accuracy_status.startswith("MEASURED") else "NONE"
        # First CLAUSE, not first word: the statuses are sentences and "PACKS;" alone
        # ("PACKS; SERVES ONLY VIA AN OPT-IN TRANSCODE…") is not a status a reader can act
        # on. Truncated to the column width with an ellipsis so the table stays a table.
        serve = r.serving_status.split(".", 1)[0].strip()
        if len(serve) > 46:
            serve = serve[:45] + "…"
        out.append(f"{r.name:<16}{r.axis:<8}{bits:>9}  {claim:<12}{acc:<18}{serve}")
    out.append("")
    out.append("KV bits = AVERAGE bit per element per tensor (compare against bf16's 16); "
               "weight bits = bit per weight.")
    out.append("Full detail (env/argv, accuracy status, provenance): "
               "`python3 -m ommx_gpu_serve.recipes show <name>`, or "
               "`w_packer recipes --name <name>`.")
    return "\n".join(out)


def format_recipe(rec: NamedRecipe) -> str:
    """Full detail for one preset — everything a reader needs to quote it safely."""
    lines = [f"recipe        {rec.name}   ({rec.axis} axis)",
             f"summary       {rec.summary}",
             f"paper claims  {', '.join(rec.paper_claims) or '(none — pins current behaviour)'}",
             "",
             "measured"]
    for k, v in rec.measured.items():
        lines.append(f"    {k:<36} {v:.6g}")
    lines += [f"    measured by                          {rec.measured_by}", ""]
    if rec.env:
        lines.append("env")
        for k, v in rec.env.items():
            lines.append(f"    {k}={v}")
        lines.append("")
        lines.append(f"select with   {RECIPE_ENV_VAR}={rec.name}    (explicit env vars "
                     "still win)")
    if rec.packer_args:
        lines.append("packer")
        lines.append("    python3 -m ommx_gpu_serve.linear.w_packer pack --preset "
                     f"{rec.name} --input <ckpt> --output <bundle>")
        lines.append(f"    equivalent flags: {' '.join(rec.packer_argv())}")
    lines += ["", f"accuracy      {rec.accuracy_status}",
              "", f"serving       {rec.serving_status}"]
    if rec.notes:
        lines.append("")
        lines.append("notes")
        for n in rec.notes:
            lines.append(f"    - {n}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

def _cmd_env(name: str, export: bool) -> int:
    """Print the env a preset applies. With ``--export``, print only the assignments
    that would ACTUALLY be made against the current environment (explicit env and
    explicit aliases already removed), shell-quoted, for ``eval`` in run.sh."""
    rec = get(name, axis="kv")
    if not export:
        for k, v in rec.env.items():
            print(f"{k}={v}")
        return 0
    pend = pending_env(rec.name)
    print(f"export {RECIPE_ENV_VAR}={shlex.quote(rec.name)}")
    for k, v in pend.items():
        print(f"export {k}={shlex.quote(v)}")
    # Report what the caller's own environment overrode, on stderr so `eval` is clean.
    held = [k for k in rec.env if k not in pend]
    if held:
        print(f"# recipe {rec.name}: kept your explicit {', '.join(held)}",
              file=sys.stderr)
    return 0


def _cmd_verify() -> int:
    """Recompute every ``measured`` value from the live code and diff. Exit 1 on drift."""
    bad = 0
    for rec in RECIPES.values():
        got = measure(rec)
        for key, want in rec.measured.items():
            have = got[key]
            ok = abs(have - want) <= 1e-9 * max(1.0, abs(want))
            if not ok:
                bad += 1
            print(f"{'ok  ' if ok else 'DRIFT'} {rec.name:<16}{key:<38}"
                  f"registry={want:.6g} code={have:.6g}")
    print("VERIFY: " + ("PASS" if not bad else f"FAIL ({bad} drifted)"))
    return 0 if not bad else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python3 -m ommx_gpu_serve.recipes",
        description="Named, measured OMMX serving recipes (KV + weight).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="table of every recipe with its measured bits")
    sh = sub.add_parser("show", help="full detail for one recipe")
    sh.add_argument("name")
    ev = sub.add_parser("env", help="the env vars a KV recipe applies")
    ev.add_argument("name")
    ev.add_argument("--export", action="store_true",
                    help="emit `export K=V` lines for eval, omitting anything already "
                         "set in this environment")
    sub.add_parser("verify", help="recompute every measured number from the code")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "list":
            print(format_table())
            return 0
        if args.cmd == "show":
            print(format_recipe(get(args.name)))
            return 0
        if args.cmd == "env":
            return _cmd_env(args.name, args.export)
        if args.cmd == "verify":
            return _cmd_verify()
    except UnknownRecipeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 1


__all__ = [
    "RECIPE_ENV_VAR", "RECIPES", "RECIPE_NAMES", "NamedRecipe", "UnknownRecipeError",
    "names", "get", "preset_env", "pending_env", "resolve_env", "apply_recipe_env",
    "env_preset_name",
    "measure", "measure_kv", "measure_weight", "format_table", "format_recipe", "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
