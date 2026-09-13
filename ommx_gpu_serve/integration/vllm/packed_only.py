# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in B1 KV arena integration and independent packed-plane byte accounting.

For OMMX engines, ``OMMX_KV_PACKED_ONLY=1`` replaces the full BF16 KV allocation with a
positive-byte arena backing the actual packed planes, residual ring and tables.
Only the guarded single-request GRAPH path is supported; unsupported schedulers,
cache formats and mixed backends fail before allocation. Default OFF preserves the
SHADOW control path. Allocation/alias evidence is not a measured capacity claim.

``kv_bits_breakdown`` describes the selected codec, not scheduler admission. At
k=6, gt=gc=32, pow2 and the dedicated FP4 map ON, K/V bit costs are:
relidx7 6.000/2.750, bitmap (default) 5.500/2.750, combinadic 5.250/2.750.
Residuals, finite-context slack, tables and alignment add bytes. The arena planner
inventories the real store instead of deriving its allocation from these ratios.
``shrunk_head_size`` remains a legacy accounting helper; it is not used to allocate
or describe a serving cache. GPU bitmap and NPU combinatorial budgets are distinct.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, Optional

from ...recipes import resolve_env as _resolve_recipe_env
from .config import DEFAULT_OUTLIER_REPR, DEFAULT_OUTLIERS, DEFAULT_POW2, OUTLIER_REPRS

# ── OMMX_RECIPE resolution ───────────────────────────────────────────────────────
#
# WHY RESOLUTION LIVES IN THE ENV ACCESSORS. This module goes to ``os.environ`` on
# its own, so if ``OMMX_RECIPE`` were expanded in exactly one place (say
# ``config.resolve_serving_config``), then in any process where that call has not
# already happened the preset does NOTHING and the operator silently gets the shipped
# recipe's numbers. Measured, one fresh process, ``OMMX_RECIPE=paper-kv``, with
# expansion confined to that one call:
#
#     kv_bits_breakdown(128)['avg_bits_per_elem']  ->  4.125   (shipped, preset ignored)
#     resolve_serving_config()                     ->  (side effect)
#     kv_bits_breakdown(128)['avg_bits_per_elem']  ->  2.75    (preset honoured)
#
# An operator reading a bit budget gets a plausible number belonging to a recipe they
# did not select. Hence resolution model (a) (see recipes.resolve_env for why (a) and
# not "expand once at process start"): resolve inside the ENV ACCESSORS below, not at
# the top of a few public functions. Putting it here means a helper cannot
# reintroduce the bug by reading ``os.environ`` one line before someone remembered to
# resolve — every recipe knob in this module travels through these six functions.
# ``resolve_env`` is idempotent, un-cached and a single dict lookup when OMMX_RECIPE
# is unset, so the no-preset path is untouched.

# ── recipe-derived per-element KV bit accounting ─────────────────────────────────
#
# SINGLE SOURCE OF TRUTH for the plane list: ``MultiSeqKVPool.__init__``
# (ommx_gpu_serve/attention/kv_pool.py). EVERY tensor that constructor allocates must
# appear in :func:`kv_bits_breakdown`, or this accounting understates the real
# footprint (an omitted plane is exactly how a "~4.6x" figure arises — see the module
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
    typo'd ``OMMX_KV_GROUP_TOKENS=3s`` silently becoming 32 would price a pool the
    operator did not configure. Mirrors ``config.py::_env_int``, which also raises.
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
    Both are reported because conflating them is what makes a spurious "<=3-bit" claim
    possible.

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
    ``DEFAULT_OUTLIER_REPR`` -- is 3.88x. Do not quote "~4.6x": that figure omits the
    two ``k_fp4_map*`` planes and is not reproduced by the repo's own formula (module
    docstring). The higher 5.45x/5.95x belong to the gt=64 +
    ``OMMX_KV_OUTLIER_MAP=0`` recipe, which is a DIFFERENT number system and is NOT the
    one the published accuracy results used.

    Excludes the bf16 residual history: at seq_len=4096 with OMMX_KV_RING=1 the
    effective ratio is 3.35x (``kv_bits_breakdown(seq_len=4096)``), and with the ring
    OFF there is no compression at all.
    """
    return _BF16_KV_BITS / max(1e-6, ommx_bits_per_elem(head_dim, **overrides))


def shrunk_head_size(head_size: int) -> int:
    """Legacy reservation-only estimate, retained for reproducible accounting.

    This rounds the packed-plane ratio to a multiple of eight and can under-budget
    the real allocation. No serving path uses it; use ``plan_kv_arena`` for exact
    bytes including residuals, tables, slack and alignment.
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


# --- engine-owned byte arena spec (explicit opt-in) -----------------------------

_PATCHED = False
_EVIDENCE_DONE = [False]


def validate_packed_only_config(vllm_config, layer=None) -> None:
    """Fail closed before allocation: the arena is a single-request SoA store.

    Engine block copying/zeroing is not meaningful for this layout. Hybrid/Mamba,
    transfer, sharing and prefix-cache paths must not reach allocation.
    """
    def require(ok, reason):
        if not ok:
            raise RuntimeError(f"OMMX_KV_PACKED_ONLY requires {reason}")

    require(_env_flag_config("OMMX_ATTN_GRAPH", False), "OMMX_ATTN_GRAPH=1")
    require(_env_flag_config("OMMX_KV_RING", False), "OMMX_KV_RING=1")
    for flag in ("OMMX_ATTN_BATCHED", "OMMX_ATTN_BATCHED_GRAPH", "OMMX_ATTN_V_BF16"):
        require(not _env_flag_config(flag, False), f"{flag}=0")
    require(not _os_present("OMMX_KV_PACKED_HEADSIZE"),
            "no legacy OMMX_KV_PACKED_HEADSIZE override (arena uses exact bytes)")
    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    require(model.max_model_len >= 32, "max_model_len>=32 (profiling cache must stay undersized)")
    require(vllm_config.compilation_config.cudagraph_capture_sizes == [1],
            "cudagraph_capture_sizes=[1] (single-request decode capture only)")
    require(scheduler.max_num_seqs == 1, "max_num_seqs=1")
    require(not scheduler.enable_chunked_prefill, "enable_chunked_prefill=False")
    require(not cache.enable_prefix_caching, "enable_prefix_caching=False")
    require(cache.block_size == 16, "block_size=16")
    require(cache.cache_dtype in {"auto", "float16", "bfloat16"}, "unquantized KV cache")
    require(vllm_config.speculative_config is None, "no speculative decoding")
    require(vllm_config.kv_transfer_config is None, "no KV transfer/connector")
    require(getattr(vllm_config, "ec_transfer_config", None) is None, "no encoder transfer")
    for name in ("tensor_parallel_size", "pipeline_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size"):
        require(getattr(parallel, name) == 1, f"{name}=1")
    require(not getattr(model, "is_hybrid", False), "no hybrid/Mamba layers or block zeroing")
    require(not getattr(model, "is_encoder_decoder", False), "decoder-only full attention")
    require(not getattr(model, "use_mla", False), "no MLA attention")
    if layer is not None:
        require(getattr(layer, "kv_sharing_target_layer_name", None) is None,
                "no cross-layer KV sharing")


def install_packed_only_spec() -> None:
    """Install the exact arena spec once; default SHADOW never imports vLLM here."""
    global _PATCHED
    if _PATCHED or not packed_only_enabled():
        return None
    try:
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.kv_cache_interface import FullAttentionSpec
    except Exception as exc:
        raise RuntimeError(
            "OMMX_KV_PACKED_ONLY is set but the vLLM v1 attention symbols could not "
            f"be imported ({type(exc).__name__}: {exc}). Needed: "
            "vllm.model_executor.layers.attention.attention.Attention and "
            "vllm.v1.kv_cache_interface.FullAttentionSpec. FIX: install a compatible "
            "vLLM 0.21 or unset OMMX_KV_PACKED_ONLY to use SHADOW mode."
        ) from exc
    original = Attention.get_kv_cache_spec

    def packed_spec(self, vllm_config):
        spec = original(self, vllm_config)
        peers = (self, *vllm_config.compilation_config.static_forward_context.values())
        # The patch is process-wide, but only engines containing OMMX opt in.
        # Its backend class is already loaded if any layer uses it; compare by
        # identity without importing the backend for unrelated engines.
        backend = sys.modules.get(f"{__package__}.backend")
        ommx_backend = getattr(backend, "OMMXCanonicalBackend", None)
        if ommx_backend is None or not any(
                getattr(layer, "attn_backend", None) is ommx_backend for layer in peers):
            return spec
        from .arena import make_kv_arena_spec
        from .backend import register_kv_arena_layout

        validate_packed_only_config(vllm_config, self)
        # Check peers too: the runner skips shared layers before invoking this
        # getter, and Mamba/non-Attention layers have their own unpatched getter.
        for layer in peers:
            if layer is not self and not callable(getattr(layer, "get_kv_cache_spec", None)):
                continue
            if (not isinstance(layer, Attention)
                    or getattr(layer, "attn_backend", None) is not ommx_backend
                    or getattr(layer, "kv_sharing_target_layer_name", None) is not None):
                raise RuntimeError("OMMX_KV_PACKED_ONLY requires only unshared OMMX full-attention layers")
        if not isinstance(spec, FullAttentionSpec):
            raise RuntimeError("OMMX_KV_PACKED_ONLY requires FullAttentionSpec for every layer")
        result = make_kv_arena_spec(spec, max_context=vllm_config.model_config.max_model_len,
                                   n_q_heads=self.num_heads)
        register_kv_arena_layout(result.arena_layout)
        self._ommx_arena_layout = result.arena_layout
        _packed_only_evidence(result.arena_layout)
        return result

    Attention.get_kv_cache_spec = packed_spec
    _PATCHED = True
    return None


def _packed_only_evidence(layout) -> None:
    if _EVIDENCE_DONE[0]:
        return
    _EVIDENCE_DONE[0] = True
    line = (f"PACKED_ONLY_SPEC arena head_size={layout.head_dim} "
            f"kv_heads={layout.n_kv_heads} max_context={layout.max_context} "
            f"page_bytes={layout.page_size_bytes} store_bytes={layout.total_store_bytes} "
            f"required_blocks={layout.required_blocks} "
            f"allocation_bytes={layout.allocation_bytes} (engine-owned; binding checked separately)")
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


__all__ = [
    "packed_only_enabled", "kv_bits_breakdown", "ommx_bits_per_elem",
    "packed_compression_ratio", "shrunk_head_size", "validate_packed_only_config",
    "install_packed_only_spec",
]
