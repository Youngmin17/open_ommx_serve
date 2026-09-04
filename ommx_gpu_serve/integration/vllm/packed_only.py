# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""PACKED-ONLY capacity mode — the OMMX KV page-budget knob, and what it does NOT prove.

THE PROBLEM (SHADOW mode, see the ``backend.py`` module docstring): the OMMX backend
INHERITS vLLM's bf16 ``get_kv_cache_shape`` (a full bf16 paged cache sized for
``max_model_len`` tokens) AND builds the compressed ``CanonicalKVStore`` sidecar planes
as EXTRA memory on top. So there is NO capacity win as wired — the sidecar is strictly
additive. The honest OMMX superiority over FA3 is CAPACITY: for the CANONICAL PUBLISHED
RECIPE (``OMMX_ATTN_K_FORMAT=i2f4 OMMX_ATTN_OUTLIERS=6 OMMX_ATTN_POW2=1
OMMX_KV_GROUP_TOKENS=32 OMMX_KV_GROUP_CHANNELS=32``; dedicated FP4 outlier map ON =
the default) the ACTUAL allocated plane footprint is

    K = 6.000 bit/elem, V = 2.750 bit/elem  ->  8.750 bit per (K,V) element pair
    vs bf16 32.000 bit/pair                 ->  3.66x KV compression

-> 3.66x more concurrent sequences / longer context in the same HBM. Every term of that
number is enumerated, per plane, by :func:`kv_bits_breakdown` — it is not a magic float.

CORRECTED 2026-08 (audit finding H5/H6). 3.66x REPLACES the "~4.6x" / "<=3-bit" claims
that used to stand in this docstring. The old accounting had three independent errors:

  * the two DEDICATED FP4 outlier-map planes (``k_fp4_mapscale`` / ``k_fp4_mapcenter``,
    bf16, one value per group x head x channel) were OMITTED. They ARE allocated by
    default: ``kv_pool.MultiSeqKVPool.__init__`` gates them on ``kv_outlier_map`` whose
    default is True, ``config.py`` resolves the same default onto ``OMMXServingConfig``
    (which ``metadata.py`` passes down as ``kv_outlier_map=c.kv_outlier_map``), and
    ``pack.ommx_pack_kv_canonical_block`` resolves it again for direct callers.
    +1.00 bit on K at gt=32.
  * the scale width was hardcoded to 2 bytes (bf16). With ``OMMX_ATTN_POW2=1`` — which
    the canonical recipe sets — ``kv_int8_scale`` defaults to ``bool(use_pow2)`` and the
    scale is stored as an int8 pow2 EXPONENT (the ``OMMX_KV_INT8_SCALE`` read in
    ``kv_pool.MultiSeqKVPool.__init__`` / ``pack.ommx_pack_kv_canonical_block``).
    -0.25 bit on K and -0.25 bit on V at gt=gc=32. The ZERO-POINT stays bf16
    (an arbitrary zp is not 2^e, so int8-exp would be lossy) and so do the two map
    planes — all three are ``scale_dtype`` (bf16), never the int8 scale dtype.
  * (unrelated to the three errors below, but load-bearing for any bit figure quoted
    from this module) the OUTLIER-POSITION plane is a CHOICE of three
    membership-equivalent encodings, and the number moves with it. All three decode to
    bit-identical values; only the storage differs. At the canonical recipe (gt=32, k=6,
    map ON, pow2):

        relidx7            k_oidx  1.500 b/elem -> K 6.000 / V 2.750 / avg 4.375 / 3.657x
        bitmap (DEFAULT)   k_obmp  1.000 b/elem -> K 5.500 / V 2.750 / avg 4.125 / 3.879x
        combinadic         k_crank 0.750 b/elem -> K 5.250 / V 2.750 / avg 4.000 / 4.000x

    The ``bitmap`` row is the format the ICCAD paper attributes to the GPU
    implementation ("positions are stored as a flat bitmask (N bits per group), enabling
    simple decoding at the cost of higher metadata overhead"). NOTE THE DIRECTION: at
    this recipe the flat bitmask is CHEAPER than the relidx7 the release shipped by
    default before it — 32 bit/group vs 48 — because k/gt = 6/32 = 18.8% is above the
    1/7 = 14.3% crossover where 1 bit/position beats 7 bits/outlier. The paper's "higher
    metadata overhead" holds against a SPARSER budget, not against this one.

  * "~4.6x" was not reproduced by the repo's own formula at ANY outlier count: that
    formula yielded 4.41x at k=3 and 3.88x at the published k=6. The 4.6x figure had
    mixed a pow2-int8 V (2.75 bit) with a bf16-scale K (4.25 bit at k=3) — two
    different recipes in one ratio.

  net: repo formula K 5.25 / V 3.00 (3.88x)  ->  real planes K 6.00 / V 2.75 (3.66x).

THE "<=3-bit" CLAIM IS A DIFFERENT NUMBER SYSTEM — NOT THIS RECIPE. K <= 3 bit/elem is
reachable, but ONLY with ``OMMX_KV_GROUP_TOKENS=64`` AND ``OMMX_KV_GROUP_CHANNELS=64``
AND ``OMMX_KV_OUTLIER_MAP=0`` (gt=64 with gc=32 gives 6.250 bit/pair = 3.125 avg, NOT <=3)
(base-shared outliers: the FP4 code rides the base scale/zp, no dedicated map). See
``pack.ommx_pack_kv_canonical_block`` ("THE LOW-BIT KV LEVER" in its docstring), which states
that combination explicitly, tabulates the same per-plane terms, and calls the result "its OWN
fakequant oracle ... a different number system" from the dedicated-map recipe.
Measured by :func:`kv_bits_breakdown` at gt=gc=64, map OFF, pow2 ON:

    k=6 -> K 3.500 / V 2.375  (5.875 bit/pair, 5.45x)   <- still ABOVE 3 bit on K
    k=3 -> K 3.000 / V 2.375  (5.375 bit/pair, 5.95x)   <- the "<=3-bit" point

NONE of the published OMMX accuracy results were produced under that recipe (they used
the dedicated-map k=6 gt=32 recipe above). Quoting the <=3-bit ratio next to those
accuracy numbers mixes two number systems; do not do it.

THE LEVER (vLLM 0.21, verified in v1/core/kv_cache_utils.py:945 + v1/worker/gpu/
attn_utils.py:155-160):

    num_gpu_blocks = available_kv_memory // page_size_bytes // num_layers
    page_size_bytes (FullAttentionSpec) = 2 * block_size * num_kv_heads
                                            * head_size * dtype_size

``num_blocks`` (== the KV token capacity) is INVERSELY proportional to
``page_size_bytes``, and ``page_size_bytes`` is linear in ``head_size``. The
allocator builds the paged tensor from ``get_kv_cache_shape(num_blocks, block_size,
H, head_size)`` — i.e. it reads ``head_size`` STRAIGHT FROM THE SPEC, so shrinking
the spec's ``head_size`` shrinks ``page_size_bytes`` AND the allocated bf16 tensor
*consistently* (the contiguous-view path stays valid; ``page_size_padded`` is left
None — that path uses ``torch.as_strided`` and is documented-broken for the standard
``(2, num_blocks, ...)`` attention shape, attn_utils.py:181-184).

So PACKED-ONLY = monkeypatch ``Attention.get_kv_cache_spec`` so the returned
``FullAttentionSpec`` carries a ``head_size`` shrunk by the OMMX compression ratio
``r = ommx_bits_per_elem / 32``. vLLM then admits ``1/r`` more KV blocks -> the budget
shows up directly in vLLM's own log lines:

    "GPU KV cache size: <N> tokens"
    "Maximum concurrency for <ctx> tokens per request: <Y>x"

The OMMX compressed sidecar (``CanonicalKVStore`` / ``MultiSeqKVPool``) is the REAL
backing store for the compressed prefix and serves uniform-decode through the canonical
op; the shrunk bf16 paged cache holds the KIVI sink+recent residual + the prefill pass.

BYTE BUDGET ONLY — WHAT THAT LOG LINE DOES AND DOES NOT PROVE. The shrunk ``head_size``
is a RESERVATION AND NOTHING ELSE. Four separate reasons it is not a measured capacity:

  1. the shrunk pages are never written and never read — ``backend.py`` skips the paged
     write in ``do_kv_cache_update`` (a headdim-D scatter into shrunk pages would
     corrupt) and full-prefill/uniform-decode are served off the in-batch q/k/v and the
     sidecar respectively. So nothing validates that the reserved bytes suffice.
  2. the REAL backing store is ``MultiSeqKVPool``, allocated SEPARATELY per layer from
     ``num_seqs x max_context``. It DOES NOT OBEY THIS BUDGET and is invisible to vLLM's
     allocator, so admitting more blocks does not mean the pool fits.
  3. ``head_size`` is quantized to a multiple of 8, so the budget is not even equal to
     the ratio. At D=128 the exact byte-equivalent is 128 * 8.750/32 = 35.0, rounded to
     32 -> vLLM reports 128/32 = 4.00x while the planes only compress 3.66x. The budget
     is 8.6% OPTIMISTIC (see :func:`shrunk_head_size`).
  4. the pool ALSO allocates the bf16 residual history ``k_hist``/``v_hist``. With
     ``OMMX_KV_RING=1`` (the canonical recipe) that is bounded at
     ``sink + recent + 2*gt`` = 104 rows/request, i.e. +32*104/S bit per (K,V) pair:
     +0.81 bit at S=4096 -> 3.35x effective, not 3.66x. With ``OMMX_KV_RING`` UNSET the
     pool keeps the FULL ``[num_seqs, max_seq_len, H, D]`` bf16 shadow = +32.0 bit/pair
     -> the compression claim is VOID (strictly worse than plain bf16). Pass
     ``seq_len=`` to :func:`kv_bits_breakdown` to get that term as a number.

Therefore vLLM's "Maximum concurrency ... <Y>x" line is NOT a validated OMMX capacity
result — it reports what the allocator was TOLD to budget. A capacity claim needs a
max-batch admission probe that runs to completion with the pool resident.

HONEST SCOPE: this realizes the capacity-BUDGET win only, as above. ``backend.py``
never writes and never reads the shrunk pages — a FULL-PROMPT prefill (q_len ==
seq_len per request, ``detect_full_prefill``) runs varlen FlashAttention directly on
the in-batch q/k/v, and uniform decode is sidecar-served; any other step raises loudly
(no bf16 fallback exists over the shrunk cache). Handled scope = the eager bench
(``bench_capacity.py --mode packed``: enforce_eager, enable_chunked_prefill=False,
enable_prefix_caching=False); chunked/mixed prefill under PACKED-ONLY is the remaining
follow-up. Default OFF (``OMMX_KV_PACKED_ONLY=1`` to enable) so SHADOW stays the safe
baseline.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

from ...recipes import resolve_env as _resolve_recipe_env
from .config import DEFAULT_OUTLIER_REPR, DEFAULT_OUTLIERS, DEFAULT_POW2, OUTLIER_REPRS

# ── OMMX_RECIPE resolution ───────────────────────────────────────────────────────
#
# THE DEFECT THIS EXISTS TO KILL. ``OMMX_RECIPE`` used to be expanded in exactly one
# place, ``config.resolve_serving_config``. This module goes to ``os.environ`` on its
# own, so in any process where that call had not already happened the preset did
# NOTHING and the operator silently got the shipped recipe's numbers. Measured, one
# fresh process, ``OMMX_RECIPE=paper-kv``:
#
#     kv_bits_breakdown(128)['avg_bits_per_elem']  ->  4.125   (shipped, preset ignored)
#     resolve_serving_config()                     ->  (side effect)
#     kv_bits_breakdown(128)['avg_bits_per_elem']  ->  2.75    (preset honoured)
#
# An operator reading a bit budget got a plausible number belonging to a recipe they
# did not select. Fix (resolution model (a); see recipes.resolve_env for why (a) and
# not "expand once at process start"): resolve inside the ENV ACCESSORS below, not at
# the top of a few public functions. Putting it here means a future helper cannot
# reintroduce the bug by reading ``os.environ`` one line before someone remembered to
# resolve — every recipe knob in this module travels through these six functions.
# ``resolve_env`` is idempotent, un-cached and a single dict lookup when OMMX_RECIPE
# is unset, so the no-preset path is untouched.

# ── recipe-derived per-element KV bit accounting ─────────────────────────────────
#
# SINGLE SOURCE OF TRUTH for the plane list: ``MultiSeqKVPool.__init__``
# (ommx_gpu_serve/attention/kv_pool.py). EVERY tensor that constructor allocates must
# appear in :func:`kv_bits_breakdown`, or this accounting understates the real
# footprint (that is exactly how the old "~4.6x" claim happened — see the module
# docstring). When kv_pool.py grows a plane, add it here in the same commit.

# ``MultiSeqKVPool`` takes ``scale_dtype: torch.dtype = torch.bfloat16`` and
# metadata.py never overrides it, so the bf16-width planes are 2 bytes. The int8 pow2
# scale (``scl_dt``) is the ONLY plane that can be 1 byte.
_SDT_BYTES = 2          # scale_dtype = bf16: k_zp, v_zp, k_fp4_mapscale, k_fp4_mapcenter
_BF16_KV_BITS = 32.0    # the reference: bf16 K (16) + bf16 V (16) per element pair
_VALID_GROUPS = (16, 32, 64, 128)


def _env(name: str, default: str = "") -> str:
    _resolve_recipe_env()          # OMMX_RECIPE -> os.environ, before ANY read
    v = os.environ.get(name)
    return default if v is None or str(v).strip() == "" else str(v).strip().lower()


def _env_int(name: str, default: int) -> int:
    """Strict int env read. A malformed value RAISES (law: no silent fallback) — a
    typo'd ``OMMX_KV_GROUP_TOKENS=3s`` silently becoming 32 is precisely the class of
    bug this module was audited for. Mirrors ``config.py::_env_int``, which also raises.
    """
    _resolve_recipe_env()          # OMMX_RECIPE -> os.environ, before ANY read
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(str(v).strip())
    except ValueError as exc:
        raise ValueError(
            f"{name}={v!r} is not an integer (KV bit accounting cannot be resolved). "
            f"Fix: unset {name} or set an integer, e.g. {name}={default}.") from exc


def _env_float(name: str, default: float) -> float:
    """Strict float env read; malformed values RAISE (see :func:`_env_int`)."""
    _resolve_recipe_env()          # OMMX_RECIPE -> os.environ, before ANY read
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(str(v).strip())
    except ValueError as exc:
        raise ValueError(
            f"{name}={v!r} is not a float (KV bit accounting cannot be resolved). "
            f"Fix: unset {name} or set a float, e.g. {name}={default}.") from exc


def _os_present(name: str) -> bool:
    _resolve_recipe_env()          # OMMX_RECIPE -> os.environ, before ANY read
    v = os.environ.get(name)
    return v is not None and str(v).strip() != ""


def _env_int_alias(canonical: str, alias: str, default: int) -> int:
    """``canonical`` (mandate spelling) wins, then ``alias`` (back-compat), then the
    default. Byte-for-byte the same precedence as ``config.py::_env_int_alias`` so the
    accounting cannot drift from the recipe the pool is actually built with."""
    if _os_present(canonical):
        return _env_int(canonical, default)
    return _env_int(alias, default)


def _env_flag_pool(name: str, default: bool) -> bool:
    """Mirror of the RAW-string boolean the KV pool itself uses:

        ``(_raw not in {"0","false","off","no"}) if _raw else <default>``

    (``kv_pool.MultiSeqKVPool.__init__``, ``pack.ommx_pack_kv_canonical_block``).
    NOTE it is deliberately NOT
    ``config.py::_env_bool``: the pool does not strip or lowercase, so ``"FALSE"`` /
    ``" 0 "`` read as TRUE here while config.py reads them as False. This is the
    operative read for a knob the caller does NOT pass down — today only
    ``OMMX_KV_INT8_SCALE`` (``metadata.py`` never passes ``kv_int8_scale=``, so the
    pool re-reads the env itself with these semantics).
    """
    _resolve_recipe_env()          # OMMX_RECIPE -> os.environ, before ANY read
    raw = os.environ.get(name)
    return (raw not in {"0", "false", "off", "no"}) if raw else bool(default)


def _env_flag_config(name: str, default: bool) -> bool:
    """Mirror of ``config.py::_env_bool`` (strip + lowercase, then the falsy set). Used
    for the knobs that reach ``MultiSeqKVPool`` THROUGH ``resolve_serving_config``:
    ``OMMX_ATTN_POW2`` (``metadata.py`` passes ``use_pow2=c.use_pow2``) and
    ``OMMX_KV_OUTLIER_MAP`` (``metadata.py`` passes ``kv_outlier_map=c.kv_outlier_map``).
    Matching each knob to the path it actually travels is what keeps this accounting
    from drifting off the allocation.
    """
    _resolve_recipe_env()          # OMMX_RECIPE -> os.environ, before ANY read
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "off", "no"}


def _env_flag_outlier_map(default: bool = True) -> bool:
    """``OMMX_KV_OUTLIER_MAP`` has TWO readers with DIFFERENT string semantics, and
    which one allocates depends on the call path:

      * ``config.py::_env_bool`` (strip + lowercase) -> ``OMMXServingConfig`` ->
        ``metadata.py`` passes ``kv_outlier_map=c.kv_outlier_map`` to the pool: the
        vLLM serving path.
      * the raw read in ``kv_pool.MultiSeqKVPool.__init__`` /
        ``pack.ommx_pack_kv_canonical_block``: any caller that leaves
        ``kv_outlier_map=None`` (CPU tests, hf_eager, direct ``pack_kv_*`` users).

    They agree for every normalized spelling ("0"/"1"/"false"/"off"/"no"/unset) and
    DISAGREE for un-normalized falsy ones ("FALSE", " 0 ", "No"): config reads False
    (no FP4 map -> K 5.00 bit at gt=32) while the pool reads True (map -> K 6.00).
    A 20% footprint swing decided by letter case is not something this accounting may
    guess at, so an ambiguous spelling RAISES with the fix named (law: no silent
    fallback). Pass ``kv_outlier_map=`` to price either recipe explicitly.
    """
    _resolve_recipe_env()          # OMMX_RECIPE -> os.environ, before ANY read
    raw = os.environ.get("OMMX_KV_OUTLIER_MAP")
    if raw is None or str(raw).strip() == "":
        return bool(default)
    pool_view = raw not in {"0", "false", "off", "no"}
    cfg_view = str(raw).strip().lower() not in {"0", "false", "off", "no"}
    if pool_view != cfg_view:
        raise ValueError(
            f"OMMX_KV_OUTLIER_MAP={raw!r} is read as {cfg_view} by config.py::_env_bool "
            f"(-> metadata.py -> MultiSeqKVPool) but as {pool_view} by kv_pool.py/pack.py's "
            "own raw env read, so the dedicated FP4 map planes may or may not be "
            "allocated (K 6.00 vs 5.00 bit/elem at group_tokens=32). Fix: spell it "
            "exactly OMMX_KV_OUTLIER_MAP=0 or =1 (lowercase, no surrounding spaces).")
    return cfg_view


def packed_only_enabled() -> bool:
    return _env("OMMX_KV_PACKED_ONLY", "0") not in {"0", "false", "off", "no"}


def _combinadic_index_bytes(k: int, gt: int) -> int:
    """Stdlib-only mirror of ``attention/codec.py::combinadic_index_bytes`` (that module
    imports torch; this one must stay importable with no torch / no vLLM / no GPU).
    ``ceil(log2 C(gt, k))`` bits rounded up to whole bytes; 0 when k<=0 or k>=gt —
    identical branch structure to ``combinadic_index_bits``."""
    k, gt = int(k), int(gt)
    if k <= 0 or k >= gt:
        return 0
    return (max(1, (math.comb(gt, k) - 1).bit_length()) + 7) // 8


def _bitmap_index_bytes(gt: int) -> int:
    """Stdlib-only mirror of ``attention/codec.py::bitmap_index_bytes`` (that module
    imports torch; this one must stay importable with no torch / no vLLM / no GPU).

    ``ceil(gt/8)`` bytes — the FLAT bitmask the ICCAD paper attributes to the GPU
    implementation: one bit per token position of the group. NOTE THE ARGUMENT: it is
    the GROUP SIZE, not ``k``. That is the defining property of this encoding — its cost
    does not move when the outlier budget moves, so it prices at ``ceil(gt/8)*8/gt`` =
    exactly 1.0 bit/element for every group size in {16,32,64,128}, against relidx7's
    ``7k/gt``. The two cross at k/gt = 1/7 = 14.3% density."""
    return (int(gt) + 7) // 8


def kv_bits_breakdown(
    head_dim: int = 128,
    *,
    k_format: Optional[str] = None,
    group_tokens: Optional[int] = None,
    group_channels: Optional[int] = None,
    outliers_per_vector: Optional[int] = None,
    outlier_repr: Optional[str] = None,
    kv_outlier_map: Optional[bool] = None,
    kv_int8_scale: Optional[bool] = None,
    use_pow2: Optional[bool] = None,
    scale_bytes: Optional[int] = None,
    seq_len: Optional[int] = None,
) -> Dict[str, Any]:
    """Per-PLANE bits/element for the OMMX packed KV footprint — the auditable form.

    Every keyword defaults to ``None`` = "resolve from the environment exactly the way
    ``MultiSeqKVPool`` / ``pack.py`` / ``config.py`` resolve it", so the default call
    describes the recipe that is actually running; pass any keyword to price a
    hypothetical recipe without touching ``os.environ``.

    Units: bits per (head, token, head-dim channel) — i.e. per ELEMENT of one head's
    head_dim vector. K and V are reported separately; ``total_bits_per_elem`` is their
    SUM (the per (K,V) PAIR figure, comparable against bf16's 32) and
    ``avg_bits_per_elem`` is that halved (the per-tensor figure, comparable against 16).
    Both are reported because conflating them is what produced the retracted "<=3-bit"
    claim.

    ``head_dim`` does NOT change any per-element number: every group plane is sized
    ``[..., D]`` or ``[..., D, fb]`` and every V plane is ``[..., NGV]`` with
    ``NGV = D // group_channels``, so D cancels (``NGV * bytes / D == bytes / gc``). It
    is still validated (``D % 32 == 0`` and ``D % gc == 0``, the two constraints
    ``MultiSeqKVPool.__init__`` enforces) so an unbuildable geometry cannot silently
    produce a number.

    Plane list (the allocation block in ``kv_pool.MultiSeqKVPool.__init__``, right after
    the ``G_cap`` / ``P_cap`` sizing), each with its amortization denominator:

      K  k_base           [P,ps,H,D//4] uint8   -> 2 bit/elem (INT2 base, i2f4/itf4)
         k_scale          [G,H,D]  int8|bf16    -> scale_bytes*8 / gt
         k_zp             [G,H,D]  bf16         -> 2*8 / gt
         k_fp4_mapscale   [G,H,D]  bf16         -> 2*8 / gt   (iff k>0 and outlier map)
         k_fp4_mapcenter  [G,H,D]  bf16         -> 2*8 / gt   (iff k>0 and outlier map)
         k_oidx | k_crank | k_obmp
                          [G,H,D,fb] uint8      -> fb*8 / gt  (relidx7 | combinadic |
                                                   bitmap; EXACTLY ONE is allocated)
         k_oval           [G,H,D,fb] uint8      -> fb*8 / gt  (FP4 nibbles, (k+1)//2 B)
      V  v_main           [P,ps,H,D//4] uint8   -> 2 bit/elem (INT2 base, v_format=i2)
         v_scale          [P,ps,H,NGV] int8|bf16-> scale_bytes*8 / gc
         v_zp             [P,ps,H,NGV] bf16     -> 2*8 / gc

    NOT included in the bits totals (reported separately under ``residual``): the bf16
    ``k_hist``/``v_hist`` history the pool also allocates. It is O(1) per REQUEST under
    ``OMMX_KV_RING=1`` and O(max_seq_len) without it, so it is not a per-token rate;
    pass ``seq_len=`` to have it priced at that context length.

    Raises ValueError (never a silent default) for any recipe ``MultiSeqKVPool`` would
    itself reject: bad k_format, group sizes outside {16,32,64,128}, outlier count
    outside [0, gt], unknown outlier_repr, scale_bytes not in {1,2}, or a head_dim the
    pool's geometry checks forbid.
    """
    # OMMX_RECIPE first, explicitly, not merely as a side effect of the ``_env*``
    # helpers below: a fully-kwarg'd call (``measure_kv`` makes one) reaches the raw
    # ``os.environ.get("OMMX_KV_RING")`` in the residual block before any accessor has
    # fired, and that knob IS recipe-controlled (both presets set OMMX_KV_RING=1).
    _resolve_recipe_env()
    # ── resolve inputs (explicit kwarg wins; else the pool/config env resolution) ──
    kfmt = str(k_format).strip().lower() if k_format is not None \
        else _env("OMMX_ATTN_K_FORMAT", "i2f4")
    gt = int(group_tokens) if group_tokens is not None else _env_int_alias(
        "OMMX_KV_GROUP_TOKENS", "OMMX_ATTN_GROUP_TOKENS", 32)
    gc = int(group_channels) if group_channels is not None else _env_int_alias(
        "OMMX_KV_GROUP_CHANNELS", "OMMX_ATTN_GROUP_CHANNELS", 32)
    if outliers_per_vector is not None:
        k = int(outliers_per_vector)
    elif _os_present("OMMX_OUTLIER_PERCENT"):
        # OMMX_OUTLIER_PERCENT (fraction of the K token-group) wins over the absolute
        # npv when set; same rounding as config.py::resolve_serving_config.
        pct = _env_float("OMMX_OUTLIER_PERCENT", 0.0)
        k = max(1, int(round(gt * pct))) if pct > 0 else 0
    else:
        k = _env_int("OMMX_ATTN_OUTLIERS", DEFAULT_OUTLIERS)
    repr_ = str(outlier_repr).strip().lower() if outlier_repr is not None \
        else _env("OMMX_ATTN_OUTLIER_REPR", DEFAULT_OUTLIER_REPR)   # config.py SSOT
    pow2 = bool(use_pow2) if use_pow2 is not None \
        else _env_flag_config("OMMX_ATTN_POW2", DEFAULT_POW2)   # travels via config.py
    omap = bool(kv_outlier_map) if kv_outlier_map is not None \
        else _env_flag_outlier_map(True)   # travels via config.py -> metadata.py
    # kv_int8_scale: kv_pool.py defaults the env read to ``bool(use_pow2)`` and pack.py
    # defaults it to True, but BOTH then clamp with ``and bool(use_pow2)`` — so the
    # EFFECTIVE value is identical and the pow2 clamp below is what actually decides.
    # Without pow2 the scale is an arbitrary bf16 (not 2^e) and int8-exp storage would
    # be lossy, so no knob can turn it on: that clamp is a correctness gate, not a
    # preference, and it is applied to an explicit kwarg too.
    i8raw = bool(kv_int8_scale) if kv_int8_scale is not None \
        else _env_flag_pool("OMMX_KV_INT8_SCALE", pow2)
    i8 = bool(i8raw) and bool(pow2)
    sb = int(scale_bytes) if scale_bytes is not None else (1 if i8 else _SDT_BYTES)

    # ── validate (mirror of the pool's own checks; loud, with the fix named) ──
    if kfmt not in ("i2f4", "itf4"):
        raise ValueError(
            f"k_format must be i2f4|itf4 (INT2 base + FP4 outlier); got {kfmt!r}. "
            "Fix: set OMMX_ATTN_K_FORMAT=i2f4 or pass k_format=.")
    if gt not in _VALID_GROUPS:
        raise ValueError(
            f"group_tokens (K vector_length) must be in {set(_VALID_GROUPS)}; got {gt}. "
            "Fix: set OMMX_KV_GROUP_TOKENS to one of them (canonical recipe: 32).")
    if gc not in _VALID_GROUPS:
        raise ValueError(
            f"group_channels (V vector_length) must be in {set(_VALID_GROUPS)}; got {gc}. "
            "Fix: set OMMX_KV_GROUP_CHANNELS to one of them (canonical recipe: 32).")
    if k < 0 or k > gt:
        raise ValueError(
            f"outliers_per_vector={k} out of range [0, group_tokens={gt}]. "
            "Fix: lower OMMX_ATTN_OUTLIERS (or OMMX_OUTLIER_PERCENT).")
    if repr_ not in OUTLIER_REPRS:
        raise ValueError(
            f"outlier_repr must be one of {OUTLIER_REPRS}; got {repr_!r}. "
            f"Fix: set OMMX_ATTN_OUTLIER_REPR={DEFAULT_OUTLIER_REPR}.")
    if sb not in (1, 2):
        raise ValueError(
            f"scale_bytes must be 1 (int8 pow2 exponent) or 2 (bf16); got {sb}. "
            "Fix: pass scale_bytes=1|2, or leave it to the use_pow2 resolution.")
    D = int(head_dim)
    if D <= 0 or D % 32 != 0:
        raise ValueError(
            f"head_dim must be a positive multiple of 32 (MultiSeqKVPool.__init__); "
            f"got {D}. Fix: this model geometry cannot use the OMMX KV path.")
    if D % gc != 0:
        raise ValueError(
            f"head_dim ({D}) must be a multiple of group_channels ({gc}) "
            "(MultiSeqKVPool.__init__). Fix: set OMMX_KV_GROUP_CHANNELS to a divisor "
            f"of {D}.")

    # ── K planes ──────────────────────────────────────────────────────────────
    # base: k_base_w = D//4 bytes per (page-slot, token, head) covers D channels
    # -> (D//4)*8/D = 2 bit/elem exactly, for both i2f4 and itf4 (k_base_bits = 2).
    kp: Dict[str, float] = {"k_base": 2.0}
    kp["k_scale"] = (sb * 8.0) / gt                     # int8 pow2 exp, or bf16
    kp["k_zp"] = (_SDT_BYTES * 8.0) / gt                # ALWAYS bf16 (zp is not 2^e)
    if k > 0 and omap:
        # the dedicated FP4 range map — the pair the old formula omitted. Allocated
        # whenever there ARE outliers and kv_outlier_map is on (the default).
        kp["k_fp4_mapscale"] = (_SDT_BYTES * 8.0) / gt
        kp["k_fp4_mapcenter"] = (_SDT_BYTES * 8.0) / gt
    if k > 0:
        # EXACTLY ONE outlier-position plane is allocated (kv_pool.MultiSeqKVPool
        # __init__ / CanonicalKVStore.__init__ leave the other two None), so exactly one
        # term enters the sum. The three are membership-equivalent — same outlier set,
        # same dequantized values — so this is a pure STORAGE choice and the K bit rate
        # is the only thing that moves.
        if repr_ == "relidx7":
            idx_fb = (7 * k + 7) // 8                   # 7-bit LSB-first indices
            kp["k_oidx"] = (idx_fb * 8.0) / gt
        elif repr_ == "bitmap":
            # FLAT in k: ceil(gt/8) bytes/frame -> 1.0 bit/elem for every gt in
            # {16,32,64,128}, vs relidx7's 7k/gt. Cheaper than relidx7 whenever
            # k/gt > 1/7; at the canonical gt=32,k=6 it is 1.000 vs 1.500.
            bmp_fb = _bitmap_index_bytes(gt)
            kp["k_obmp"] = (bmp_fb * 8.0) / gt
        else:
            crank_fb = _combinadic_index_bytes(k, gt)   # ceil(log2 C(gt,k)) bytes
            kp["k_crank"] = (crank_fb * 8.0) / gt
        oval_fb = (k + 1) // 2                          # FP4 nibble per outlier
        kp["k_oval"] = (oval_fb * 8.0) / gt

    # ── V planes (i2 base, per-TOKEN affine over gc channels, no outliers) ────
    vp: Dict[str, float] = {"v_main": 2.0}
    vp["v_scale"] = (sb * 8.0) / gc                     # NGV*sb*8 / D == sb*8 / gc
    vp["v_zp"] = (_SDT_BYTES * 8.0) / gc

    k_bits = float(sum(kp.values()))
    v_bits = float(sum(vp.values()))
    total = k_bits + v_bits

    # ── the bf16 residual history (allocated by the SAME pool; NOT a per-token rate) ──
    # kv_pool.MultiSeqKVPool.__init__ (ring sizing): OMMX_KV_RING=1 -> ring_cap = sink +
    # (recent + 2*gt) rows per request; unset -> the full max_seq_len bf16 shadow.
    # Reported, never folded into
    # the rate above (it is O(1)/request with the ring, O(S) without it).
    ring_raw = os.environ.get("OMMX_KV_RING")
    kv_ring = bool(ring_raw) and ring_raw not in {"0", "false", "off", "no"}
    sink = _env_int_alias("OMMX_KV_SINK", "OMMX_ATTN_SINK", 8)
    recent = _env_int_alias("OMMX_KV_RECENT", "OMMX_ATTN_RECENT", 32)
    rows = (sink + recent + 2 * gt) if kv_ring else None
    resid_bits: Optional[float] = None
    if seq_len is not None and int(seq_len) > 0:
        S = int(seq_len)
        # 2 tensors (K and V) x 2 bytes x rows, spread over S tokens, per element.
        resid_bits = (2 * _SDT_BYTES * 8.0) * (rows / S) if rows is not None \
            else (2 * _SDT_BYTES * 8.0)

    out: Dict[str, Any] = {
        "recipe": {
            "head_dim": D, "k_format": kfmt, "v_format": "i2",
            "group_tokens": gt, "group_channels": gc,
            "outliers_per_vector": k, "outlier_repr": repr_,
            # which plane the outlier-index bits above are charged to, so a caller
            # reading only ``recipe`` can tell WHICH encoding a K figure belongs to
            # (the three reprs give three different K rates for identical numerics).
            "outlier_index_plane": (
                None if k <= 0 else
                {"relidx7": "k_oidx", "combinadic": "k_crank",
                 "bitmap": "k_obmp"}[repr_]),
            "kv_outlier_map": omap, "use_pow2": pow2,
            "kv_int8_scale": i8, "scale_bytes": sb, "zp_bytes": _SDT_BYTES,
        },
        "k_planes": kp,
        "v_planes": vp,
        "k_bits_per_elem": k_bits,
        "v_bits_per_elem": v_bits,
        "total_bits_per_elem": total,          # K + V, compare against bf16's 32
        "avg_bits_per_elem": total / 2.0,      # per tensor, compare against bf16's 16
        "bf16_bits_per_elem": _BF16_KV_BITS,
        "compression_ratio": _BF16_KV_BITS / max(1e-6, total),
        "residual": {
            "kv_ring": kv_ring,
            "sink_tokens": sink,
            "recent_window": recent,
            "bf16_rows_per_seq": rows,         # None => the full max_seq_len shadow
            "seq_len": int(seq_len) if seq_len is not None else None,
            "bits_per_elem": resid_bits,       # None unless seq_len= was given
        },
    }
    if resid_bits is not None:
        out["effective_total_bits_per_elem"] = total + resid_bits
        out["effective_compression_ratio"] = _BF16_KV_BITS / max(
            1e-6, total + resid_bits)
    return out


def ommx_bits_per_elem(head_dim: int = 128, **overrides: Any) -> float:
    """OMMX packed K+V bits per (per-head) element — the SUM over the K and V planes.

    Thin wrapper over :func:`kv_bits_breakdown` (which owns the per-plane derivation);
    ``**overrides`` are forwarded verbatim, so every input can be made explicit instead
    of coming from the environment. bf16 reference = 32 bit/elem (16 K + 16 V), so the
    compression ratio is ``32 / ommx_bits_per_elem()``.

    Canonical published recipe (i2f4, k=6, pow2, gt=gc=32, dedicated map ON): 8.750.
    The bf16 residual history is NOT included — see ``residual`` in the breakdown and
    reason 4 of the module docstring.
    """
    return float(kv_bits_breakdown(head_dim, **overrides)["total_bits_per_elem"])


def packed_compression_ratio(head_dim: int = 128, **overrides: Any) -> float:
    """bf16/OMMX KV-byte ratio over the QUANTIZED PLANES ONLY.

    3.66x for the canonical published recipe (i2f4, OMMX_ATTN_OUTLIERS=6,
    OMMX_ATTN_POW2=1, group_tokens=group_channels=32, dedicated FP4 outlier map ON,
    relidx7 positions); the bare default -- the same recipe with bitmap positions,
    ``DEFAULT_OUTLIER_REPR`` -- is 3.88x. This REPLACES the retracted "~4.6x": that
    figure omitted the two
    ``k_fp4_map*`` planes and was never reproduced by the repo's own formula (module
    docstring, finding H5/H6). The higher 5.45x/5.95x belong to the gt=64 +
    ``OMMX_KV_OUTLIER_MAP=0`` recipe, which is a DIFFERENT number system and is NOT the
    one the published accuracy results used.

    Excludes the bf16 residual history: at seq_len=4096 with OMMX_KV_RING=1 the
    effective ratio is 3.35x (``kv_bits_breakdown(seq_len=4096)``), and with the ring
    OFF there is no compression at all.
    """
    return _BF16_KV_BITS / max(1e-6, ommx_bits_per_elem(head_dim, **overrides))


def shrunk_head_size(head_size: int) -> int:
    """The reduced spec ``head_size`` that makes vLLM budget ``page_size_bytes`` for the
    OMMX packed footprint instead of bf16.

    ``page_size_bytes`` is linear in ``head_size``; we want it scaled by
    ``ommx_bits_per_elem / 32`` (both K and V together). Rounded to the NEAREST multiple
    of 8 (FA shape sanity), floor 8, never larger than the real head_size. Overridable
    by ``OMMX_KV_PACKED_HEADSIZE``.

    THIS IS A BYTE BUDGET ONLY — IT BACKS NOTHING. The returned head_size shrinks what
    vLLM's allocator reserves and therefore how many blocks it admits, but those pages
    are never written and never read, and the REAL store for the compressed prefix is
    the separately allocated ``MultiSeqKVPool``, which does NOT obey this budget and is
    invisible to vLLM's accounting. Consequently vLLM's "Maximum concurrency ... Nx"
    log line is NOT a validated OMMX capacity result (module docstring, reasons 1-4).

    The nearest-multiple-of-8 rounding can land BELOW the exact byte-equivalent, i.e.
    the budget can be OPTIMISTIC relative to the plane footprint. For the canonical
    recipe at head_size=128: exact = 128 * 8.750/32 = 35.0 -> returned 32, so vLLM
    budgets 128/32 = 4.00x while the planes only compress 3.66x (8.6% optimistic).
    That is unchanged from the pre-H5/H6 formula (which gave 33.0 -> 32 as well): the
    corrected, LARGER bit count raises the reservation only where the rounding does not
    absorb it (e.g. head_size=256: 64 -> 72). Rounding is deliberately left as-is so the
    H100-measured 4.00x budget stays reproducible.
    """
    forced = _env_int("OMMX_KV_PACKED_HEADSIZE", 0)
    if forced > 0:
        return max(8, min(int(head_size), forced))
    r = ommx_bits_per_elem(head_size) / _BF16_KV_BITS  # OMMX fraction of bf16 bytes
    hs = head_size * r
    hs = max(8, int(round(hs / 8.0)) * 8)              # nearest multiple of 8, floor 8
    hs = min(int(head_size), hs)
    # Loud invariant (NOT `assert` — python -O strips those): FA shape sanity is what
    # keeps the contiguous-view allocation path valid, so a violation must not slip by.
    if hs <= 0 or hs % 8 != 0 or hs > int(head_size):
        raise ValueError(
            f"shrunk_head_size({head_size}) produced {hs}, which is not a positive "
            "multiple of 8 <= head_size. Fix: check OMMX_KV_PACKED_HEADSIZE and the "
            "recipe env (kv_bits_breakdown() prints every term).")
    return hs


# --- the monkeypatch: shrink FullAttentionSpec.head_size in PACKED-ONLY mode -------

_PATCHED = False


def install_packed_only_spec() -> Optional[float]:
    """Patch ``Attention.get_kv_cache_spec`` so PACKED-ONLY shrinks the bf16 cache
    page budget by the OMMX compression ratio. Idempotent. No-op (returns None) when
    PACKED-ONLY is disabled or the patch is already installed.

    LAW #5 (no silent fallback): the ONLY no-op conditions are the two above, both of
    which are checked BEFORE anything can fail. Past that point the operator has
    explicitly set ``OMMX_KV_PACKED_ONLY``, which makes the sidecar the only backing
    store, so a missing vLLM symbol RAISES here. It used to ``return None``, which left
    ``backend._PACKED_ONLY`` True (it is resolved from the env, independently of this
    patch) against an UNSHRUNK bf16 paged cache that ``do_kv_cache_update`` never
    writes and ``forward()`` refuses to read — the contradictory engine
    ``plugin.py``'s import guard exists to prevent. That guard only covered failure to
    import THIS MODULE; a symbol failure inside it slipped through.

    Always returns None: the realized multiplier is not known until the first layer's
    spec is rewritten, so it is emitted as one-time evidence by
    :func:`_packed_only_evidence` (log line + ``OMMX_FIRE_FILE``) instead of returned.

    THE RAISE IS NOT UNCONDITIONAL — re-verified on a host with no vLLM installed. The
    ``not packed_only_enabled()`` guard runs BEFORE the ``from vllm...`` import, and
    ``packed_only_enabled()`` reads ``OMMX_KV_PACKED_ONLY`` with default ``"0"``, so
    SHADOW mode (the default, which keeps vLLM's full bf16 paged cache and a valid bf16
    fallback for every step) returns None without ever touching a vLLM symbol. Measured
    on this machine, ``vllm`` not importable: env unset -> ``install_packed_only_spec()``
    is ``None``; env ``"0"``/``"off"``/``""`` -> ``None``; env ``"1"`` -> ``RuntimeError
    (ModuleNotFoundError: No module named 'vllm')``. A vLLM tree missing the v1 symbols
    therefore still runs SHADOW; only an operator who asked for PACKED-ONLY is stopped.

    ``plugin.py`` does not double-raise or mask this. Its ``try`` wraps ONLY
    ``from .packed_only import install_packed_only_spec`` (a MODULE import failure, which
    it re-raises just for ``_packed_only_requested()`` and otherwise reports as a SHADOW
    note); the call to this function sits in that ``try``'s ``else:`` branch, so the
    RuntimeError above propagates out of ``register()`` unmodified. The two env
    predicates agree exactly: ``plugin._packed_only_requested`` and
    ``packed_only_enabled`` both strip+lowercase and both treat
    ``{"", "0", "false", "off", "no"}`` (and unset) as OFF.

    UNVERIFIED (no GPU this session): the SUCCESS path — a real vLLM import, the patched
    ``get_kv_cache_spec`` running during KV-cache sizing, and the resulting page-budget
    shrink — was not exercised here. Only the two no-op conditions and the PACKED-ONLY
    raise were executed, on CPU, with no vLLM present.
    """
    global _PATCHED
    if _PATCHED or not packed_only_enabled():
        return None
    try:
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.kv_cache_interface import FullAttentionSpec
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "OMMX_KV_PACKED_ONLY is set but the vLLM v1 attention symbols the "
            "page-budget patch needs could not be imported "
            f"({type(exc).__name__}: {exc}). Needed: "
            "vllm.model_executor.layers.attention.attention.Attention and "
            "vllm.v1.kv_cache_interface.FullAttentionSpec. Without the patch vLLM "
            "reserves FULL bf16 pages that the OMMX sidecar never writes and the "
            "backend refuses to read, so the engine would be self-contradictory. "
            "FIX: install a vLLM >= 0.21 that exposes the v1 attention layer, or "
            "unset OMMX_KV_PACKED_ONLY to use SHADOW mode."
        ) from exc

    _orig = Attention.get_kv_cache_spec

    def _packed_only_spec(self, vllm_config):  # noqa: ANN001
        spec = _orig(self, vllm_config)
        # Only shrink the standard full-attention bf16 KV cache (the OMMX target).
        # Sliding-window / TQ / MLA / fp8 specs are left untouched (not the OMMX path).
        if not isinstance(spec, FullAttentionSpec):
            return spec
        try:
            import torch
            if spec.dtype not in (torch.bfloat16, torch.float16):
                return spec  # quantized-cache layers: leave alone
            # NOTE: an unbuildable recipe (bad env, head_size not a multiple of the V
            # group) now raises out of shrunk_head_size and lands in the except below
            # -> the spec is left at full bf16, i.e. OVER-reserved, never under. The
            # loud failure for that same geometry still happens at pool construction
            # (MultiSeqKVPool.__init__ raises), so no silently-wrong path exists.
            hs = shrunk_head_size(int(spec.head_size))
            if hs >= int(spec.head_size):
                return spec
            from dataclasses import replace
            new = replace(spec, head_size=hs, head_size_v=hs)
            # one-time evidence (law #5): the page budget really shrank.
            _packed_only_evidence(int(spec.head_size), hs)
            return new
        except Exception as exc:  # noqa: BLE001
            # LAW #5: this is the ONE fail-SAFE direction (the spec stays FULL bf16, i.e.
            # OVER-reserved, never under) so it does not raise here — but it must not be
            # silent either, or the operator sees PACKED-ONLY with no PACKED_ONLY_SPEC
            # evidence and no stated reason. Record it once, then leave the spec alone.
            _packed_only_spec_skipped_evidence(exc)
            return spec

    Attention.get_kv_cache_spec = _packed_only_spec  # type: ignore[assignment]
    _PATCHED = True
    return None


_EVIDENCE_DONE = [False]
_SKIP_EVIDENCE_DONE = [False]


def _packed_only_evidence(orig_hs: int, new_hs: int) -> None:
    if _EVIDENCE_DONE[0]:
        return
    _EVIDENCE_DONE[0] = True
    ratio = orig_hs / max(1, new_hs)
    # "page_bytes /Nx" is the BUDGET ratio (quantized to a multiple of 8); "ommx_bits"
    # is the true plane footprint. They differ by the rounding — both are logged so the
    # log line cannot be mistaken for a measured compression result.
    bits = ommx_bits_per_elem(orig_hs)
    line = (f"PACKED_ONLY_SPEC head_size {orig_hs} -> {new_hs} "
            f"(page_bytes /{ratio:.2f}x budget) "
            f"ommx_bits/elem={bits:.3f} planes={_BF16_KV_BITS / bits:.2f}x "
            f"(budget only; MultiSeqKVPool is allocated outside it)")
    try:
        from vllm.logger import init_logger
        init_logger("ommx_gpu_serve").info("[ommx] %s", line)
    except Exception:
        pass
    fire = os.environ.get("OMMX_FIRE_FILE", "/tmp/ommx_route_fired.log")
    try:
        with open(fire, "a") as fh:
            fh.write(line + f" pid={os.getpid()}\n")
    except Exception:
        pass



def _packed_only_spec_skipped_evidence(exc: BaseException) -> None:
    """One-time record that a FullAttentionSpec was left UNSHRUNK and why.

    The page budget then stays at full bf16 (over-reserved, never under), and the
    geometry that broke ``shrunk_head_size`` still raises loudly at pool construction
    (``MultiSeqKVPool.__init__``). This line is what makes the gap between "PACKED-ONLY
    is on" and "no PACKED_ONLY_SPEC fired" attributable instead of silent.
    """
    if _SKIP_EVIDENCE_DONE[0]:
        return
    _SKIP_EVIDENCE_DONE[0] = True
    line = ("PACKED_ONLY_SPEC_SKIPPED head_size left UNSHRUNK (full bf16 pages, "
            f"over-reserved) because {type(exc).__name__}: {exc}")
    try:
        from vllm.logger import init_logger
        init_logger("ommx_gpu_serve").warning("[ommx] %s", line)
    except Exception:
        pass
    fire = os.environ.get("OMMX_FIRE_FILE", "/tmp/ommx_route_fired.log")
    try:
        with open(fire, "a") as fh:
            fh.write(line + f" pid={os.getpid()}\n")
    except Exception:
        pass


__all__ = [
    "packed_only_enabled",
    "kv_bits_breakdown",
    "ommx_bits_per_elem",
    "packed_compression_ratio",
    "shrunk_head_size",
    "install_packed_only_spec",
]
