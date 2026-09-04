# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Serving-recipe resolution (env -> canonical KV recipe). Pure-python, no vLLM.

Resolves the OMMX KV recipe knobs from environment variables into a
``CanonicalAttentionConfig`` + ``WindowSpec`` for the vLLM backend. Kept separate
from the backend so it imports + tests on CPU without vLLM / triton.

The defaults encode the published accuracy recipe's numeric knobs (K=i2f4 signed
6-of-32 outliers, pow2 int8 group scales, V=i2, sink=8, recent=32, group=32) with
bitmap outlier positions, the serving default — see KV FOOTPRINT below for the one
knob on which that differs from ``shipped-kv``. Knobs exist to drive the M2 fakequant
gap-table reconciliation (sink/recent alignment; pow2 lives in the packer).

Env vars (all optional). The MANDATE-named knobs (OMMX_KV_*) are the canonical
spellings; the OMMX_ATTN_* names are kept as back-compat aliases (the OMMX_KV_*
form wins when both are set):

  OMMX_RECIPE               NAMED PRESET applied before every knob below — the one flag
                            that reproduces a published bit budget instead of a guess.
                            `shipped-kv` is the canonical published recipe (measured
                            K 6.000 / V 2.750 / avg 4.375); `paper-kv` is the recipe
                            that MEASURES the paper's Table 1 AvgBits of 2.75 exactly
                            (claim E1; K 3.3125 / V 2.1875 / avg 2.7500) and has NO
                            accuracy number in this repo. Registry + provenance:
                            ommx_gpu_serve/recipes.py; `python3 -m
                            ommx_gpu_serve.recipes list`. Any env var set explicitly
                            (including an OMMX_ATTN_* alias) WINS over the preset, so a
                            preset can be overridden one knob at a time. An unknown
                            name raises and lists the known ones. Unset = nothing
                            applied, defaults exactly as before.
  OMMX_KV_GROUP_TOKENS      K token-group vector_length {16,32,64,128}  (default 32)
                            (alias: OMMX_ATTN_GROUP_TOKENS). 64 amortizes every per-group
                            K plane (scale, bf16 zp, FP4 map, outlier idx/val) over twice
                            the tokens; it is the FIRST half of the "<=3-bit KV lever".
                            It is NOT what the published recipe runs — see KV FOOTPRINT
                            below before quoting a bit/element or compression number.
  OMMX_KV_GROUP_CHANNELS    V channel-group vector_length {16,32,64,128} (default 32)
                            (alias: OMMX_ATTN_GROUP_CHANNELS; must divide head_dim). 64 ->
                            V = 2.375 bit/elem (vs 2.750 at 32, pow2 scale).
  OMMX_KV_OUTLIER_MAP       1 (dedicated FP4 range-map, default) | 0 (BASE-SHARED outlier: the
                            outlier rides the base scale/zp, dropping the per-group FP4 map's
                            16b mapscale+16b mapcenter -> K outlier 0.375b not the mapped cost,
                            the SECOND half of the "<=3-bit KV lever"). pack.py reads this too;
                            the config carries it so the recipe is explicit + gate-able.
  OMMX_KV_SINK              sink tokens kept bf16          (default 8)
                            (alias: OMMX_ATTN_SINK)
  OMMX_KV_RECENT            recent window kept bf16        (default 32)
                            (alias: OMMX_ATTN_RECENT)
  OMMX_OUTLIER_PERCENT      K outlier fraction per group -> npv = round(gt*pct)
                            (overrides OMMX_ATTN_OUTLIERS when set; default unset)
  OMMX_ATTN_OUTLIERS        K outliers per group (absolute npv)  (default 6)
  OMMX_ATTN_POW2            0 | 1 (pow2 -> int8 group scale)   (default 1)
  OMMX_ATTN_K_FORMAT        i2f4 | itf4                    (default i2f4)
  OMMX_ATTN_OUTLIER_SELECT  signed | abs                   (default signed)
  OMMX_ATTN_OUTLIER_REPR    relidx7 | combinadic | bitmap  (default bitmap)
  OMMX_ATTN_COMBINADIC_READ 0 | 1 (in-kernel rank unrank)  (default: follows
                            OMMX_ATTN_OUTLIER_REPR == combinadic)
  OMMX_ATTN_BITMAP_READ     0 | 1 (kernel reads k_obmp)    (default: follows
                            OMMX_ATTN_OUTLIER_REPR == bitmap)
  OMMX_STRICT               0 | 1 (re-raise instead of bf16 fallback) (default 0)

``outlier_repr="bitmap"`` ON THE vLLM SERVING PATH — what is wired, and the one limit
that is not the wiring's fault:

  * the STORE side works: ``metadata.py`` passes ``outlier_repr`` to ``MultiSeqKVPool``,
    which allocates ``k_obmp`` and no ``k_oidx``;
  * the KERNEL side now does too: all four
    ``ommx_paged_decode_attention_canonical(...)`` call sites in ``backend.py`` forward
    ``bitmap_read=cfg.resolved_bitmap_read()`` and ``k_obmp=pl.get("k_obmp")``. The
    HF-eager paths reach the same kernel through ``attention/reference_op.py``, which
    infers the index source from the pack and needed no change.
  * ``cfg.bitmap_read`` is TRI-STATE and None is the answer, not a default:
    :meth:`OMMXServingConfig.resolved_bitmap_read` follows ``outlier_repr``, because a
    bitmap-packed store carries ``k_obmp`` and NO ``k_oidx`` -- reading the mask is the
    only decode that addresses a plane the pool allocated.
  * REMAINING LIMIT, and it is the kernel's not the wiring's: ``bitmap_read`` is refused
    for ``k > 8`` (the outlier VALUE stream is staged as a single int32, so
    ``ceil(k/2) <= 4`` bytes). ``shipped-kv`` (k=6) fits; ``paper-kv`` (k=10) does not,
    and names relidx7. :func:`resolve_serving_config` refuses a bitmap recipe with
    ``k > 8`` at engine build rather than letting the first decode step find out.

The repr is also usable through ``attention/reference_op.py`` (which infers the index
source from the pack) — that is the path ``tests/test_bitmap_outlier.py`` gates on CPU.
``preflight.py``'s pool-footprint estimator now prices all three encodings separately; it
used to charge a combinadic rank for any non-relidx7 repr, under-pricing a bitmap pool by
0.25 bit/element at gt=32 and 0.625 at gt=128.

VERIFIED ON HARDWARE (H200 NVL, 2026-09-04): the kernel's ``bitmap_read`` branch runs and
matches an independent attention oracle at cos>=0.999 (tests/test_decode_kernel_parity.py,
eight ``bitmap`` parametrisations at group_tokens 32 and 64). ``combinadic_read`` remains unverified on a GPU.

KV FOOTPRINT — the published recipe and the "<=3-bit lever" are DIFFERENT number
systems, and only the first one has accuracy results behind it. Every figure below is a
per-plane sum over the tensors ``MultiSeqKVPool`` actually allocates (not a formula
fitted to them), recomputable on CPU with
``ommx_gpu_serve.integration.vllm.packed_only.kv_bits_breakdown``. PROVENANCE, because
"the gate pins it" is exactly the kind of claim this block exists to stop being loose:
only the CANONICAL row is pinned as a constant by the shipped gate
``ommx_gpu_serve/tests/test_bit_accounting.py`` (``CANON_K_BITS`` 6.0 / ``CANON_V_BITS``
2.75 / avg 4.375 / total 8.75). That gate's second recipe, ``LEVER``, holds only 3
outliers per group — K+V = 5.375, avg 2.688, 5.95x — and exists there to prove the
arithmetic is not a hardcode of the canonical answer, NOT to pin the 6-outlier lever row
below. The lever / map-off / bare-default rows here were each recomputed from a real
allocated pool, but no shipped test would catch them drifting. Units: bit per element
of one head's head_dim vector; "avg" is the per-tensor figure (compare against bf16's
16), "K+V" the per-(K,V)-pair figure (compare against 32).

  * PUBLISHED CANONICAL RECIPE — ``OMMX_ATTN_K_FORMAT=i2f4 OMMX_ATTN_OUTLIERS=6
    OMMX_ATTN_POW2=1`` (pow2 -> int8 group scale) ``OMMX_KV_GROUP_TOKENS=32
    OMMX_KV_GROUP_CHANNELS=32 OMMX_KV_OUTLIER_MAP=1`` (the default map), relidx7:

        K = 6.000   V = 2.750   K+V = 8.750   avg = 4.375   ->  32/8.750 = 3.66x vs bf16

    This is the number system every accuracy result was produced under. 3.66x is the
    compression to quote for it.

  * "<=3-BIT KV LEVER" — requires gt=64 AND gc=64 AND OMMX_KV_OUTLIER_MAP=0 (holding the
    same 6 outliers per K group):

        K = 3.500   V = 2.375   K+V = 5.875   avg = 2.938   ->  32/5.875 = 5.45x vs bf16

    "<=3 bit" refers to that avg 2.938, reached only with all three knobs moved. No
    accuracy number in this repo was measured with them, so pairing 5.45x (or the avg
    "<=3 bit") with the published accuracy is a category error. Flipping only the map
    off (gt=gc=32, OMMX_KV_OUTLIER_MAP=0) lands in between: K+V = 7.750, avg 3.875,
    4.13x.

Nothing below defaults into the lever (gt/gc default to 32 and the map to ON). The bare
defaults carry the published recipe's numeric knobs — ``OMMX_ATTN_OUTLIERS`` 6,
``OMMX_ATTN_POW2`` 1 (``DEFAULT_OUTLIERS`` / ``DEFAULT_POW2`` below) — and differ from
``shipped-kv`` in ONE knob, the position encoding: bitmap (``DEFAULT_OUTLIER_REPR``)
instead of relidx7. So a bare run prices

        K = 5.500   V = 2.750   K+V = 8.250   avg = 4.125   ->  32/8.250 = 3.88x vs bf16

0.25 bit/element under the published 4.375, with bit-identical decoded values (the two
encodings carry the same positions and values). To quote 3.66x, name the recipe —
``OMMX_RECIPE=shipped-kv`` — or set ``OMMX_ATTN_OUTLIER_REPR=relidx7``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from ommx_gpu_serve.attention.config import (
    OUTLIER_REPRS,
    CanonicalAttentionConfig,
    TensorQuantSpec,
)
from ommx_gpu_serve.attention.kv_window import WindowSpec
from ommx_gpu_serve.recipes import (
    RECIPE_ENV_VAR,
    preset_env,
    resolve_env,
)

# ``OUTLIER_REPRS`` -- the outlier-position encodings this serving config accepts -- is
# ``attention/config.py``'s tuple, re-exported here. One definition, shared with the
# packer (``attention.pack``) and the preflight pool projection, so a fourth encoding
# cannot be accepted by one reader and refused by another.
# The serving default for OMMX_ATTN_OUTLIER_REPR (why bitmap: see the comment at the
# read in resolve_serving_config). Every harness and the pool accounting
# (packed_only.py) import THIS rather than restating it, so the priced pool and the
# allocated pool cannot disagree by default.
DEFAULT_OUTLIER_REPR = "bitmap"
#: The other two recipe knobs packed_only used to restate as its own literals (3 / off),
#: which priced a default pool as k=3 pow2-off while the engine allocated k=6 pow2-on.
DEFAULT_OUTLIERS = 6
DEFAULT_POW2 = True

GROUP_TOKENS = 32


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return str(default)
    return str(raw).strip().lower()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "off", "no"}


def _env_present(name: str) -> bool:
    raw = os.environ.get(name)
    return raw is not None and str(raw).strip() != ""


def _env_int_alias(canonical: str, alias: str, default: int) -> int:
    """Read ``canonical`` (the mandate spelling) first, then ``alias`` (back-compat),
    then ``default``. The canonical name wins when both are set."""
    if _env_present(canonical):
        return _env_int(canonical, default)
    return _env_int(alias, default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(raw)


@dataclass
class OMMXServingConfig:
    """Resolved serving recipe: format/window knobs + per-model geometry."""

    k_format: str = "i2f4"
    outliers_per_vector: int = DEFAULT_OUTLIERS
    outlier_select: str = "signed"
    # OUTLIER-POSITION STORAGE. Three membership-equivalent encodings of ONE logical
    # format (they dequantize bit-identically; only the index plane and its size differ):
    #   relidx7            k_oidx  ceil(7k/8) B/frame — 7 bit per outlier slot
    #   combinadic         k_crank the ceil(log2 C(gt,k))-bit storage floor
    #   bitmap (DEFAULT)             k_obmp  ceil(gt/8) B/frame — the FLAT N-bit-per-group mask
    #                              the ICCAD paper attributes to the GPU implementation.
    #                              FLAT in k (1.0 bit/element), so it is CHEAPER than
    #                              relidx7 above the 1/7 = 14.3% density crossover: at
    #                              the canonical gt=32, k=6 recipe it is 32 bit/frame vs
    #                              relidx7's 48, i.e. K 6.000 -> 5.500 bit/element and
    #                              K+V 8.750 -> 8.250 (3.657x -> 3.879x).
    outlier_repr: str = DEFAULT_OUTLIER_REPR
    use_pow2: bool = DEFAULT_POW2
    #: Tri-state exactly like ``bitmap_read``: ``None`` means "follow ``outlier_repr``",
    #: which is the only correct default (see ``resolved_combinadic_read``). An explicit
    #: bool is for A/B only. Kept defaulting to False rather than None so that an
    #: OMMX_ATTN_COMBINADIC_READ that is unset still reads as False for any caller that
    #: touches the field directly; ``resolved_combinadic_read()`` is the accessor to use.
    combinadic_read: Any = None
    # Kernel-side index SOURCE for the bitmap repr. None = follow ``outlier_repr``
    # (the only correct decode for a bitmap-packed store, since it carries no k_oidx);
    # set explicitly only to force an A/B. WIRED AND HARDWARE-VERIFIED: backend.py
    # forwards ``resolved_bitmap_read()`` + ``k_obmp`` to the kernel at four call sites
    # (eager and CUDA-graph, single and batched), and the branch is gated on a GPU by
    # tests/test_decode_kernel_parity.py. Was documented here as "NO CONSUMER YET / NOT
    # YET WIRED" long after both stopped being true, on what is now the DEFAULT repr --
    # which is why the claim is now stated with the evidence that backs it.
    bitmap_read: Optional[bool] = None
    sink_tokens: int = 8
    recent_window: int = 32
    group_tokens: int = GROUP_TOKENS         # K token-group vector_length {16,32,64,128}
    group_channels: int = GROUP_TOKENS       # V channel-group vector_length {16,32,64,128}
    # K outlier value codec: True = dedicated FP4 range-map (per-group mapscale/mapcenter, current
    # default); False = BASE-SHARED outlier (outlier rides the base scale/zp -> drops the 2x bf16
    # map params). pack.py honours kv_outlier_map=; we mirror the env here. NOTE: map=False
    # is only ONE of the three knobs the "<=3-bit lever" needs (gt=64 + gc=64 too) — alone it
    # buys K+V 8.750 -> 7.750 (3.66x -> 4.13x), not <=3 bit. See KV FOOTPRINT in the module
    # docstring; the published accuracy recipe keeps this True.
    kv_outlier_map: bool = True
    strict: bool = False
    # per-model geometry (filled from the vLLM model config at backend init).
    head_dim: int = 128
    n_q_heads: int = 32
    n_kv_heads: int = 8
    page_size: int = 16
    max_context: int = 4096
    sm_arch: int = 90
    extra: dict = field(default_factory=dict)

    def window(self) -> WindowSpec:
        return WindowSpec(sink_tokens=self.sink_tokens,
                          recent_window=self.recent_window,
                          group_tokens=self.group_tokens,
                          page_size=self.page_size)

    def resolved_bitmap_read(self) -> bool:
        """Should the kernel take its outlier INDEX from ``k_obmp``?

        ``bitmap_read`` is tri-state and the None case is not a default, it is the only
        correct answer: a bitmap-packed store allocates ``k_obmp`` and NO ``k_oidx``
        (``kv_pool.py``'s ``elif ... == "bitmap"`` branch), so a kernel that did not read
        the bitmap would be reading a plane that does not exist. Following
        ``outlier_repr`` is therefore the decode, not a preference.

        An explicit True/False is kept for A/B only, and ``resolve_serving_config``
        already refuses ``bitmap_read=True`` against a non-bitmap repr, so the explicit
        branch can never select a plane the store did not allocate.
        """
        if self.bitmap_read is None:
            return self.outlier_repr == "bitmap"
        return bool(self.bitmap_read)

    def resolved_combinadic_read(self) -> bool:
        """Should the kernel take its outlier INDEX from ``k_crank``?

        The same argument as ``resolved_bitmap_read``, for the same reason: a
        combinadic-packed store allocates ``k_crank`` and NO ``k_oidx`` (``kv_pool.py``
        allocates ``k_oidx`` only for ``relidx7`` and ``k_obmp`` only for ``bitmap``), so a
        kernel that did not read the rank would be reading a plane that does not exist and
        would raise "requires the k_oidx relidx7 sidecar plane" on the first quantized
        group. Following ``outlier_repr`` is the decode, not a preference.

        This accessor did NOT exist while ``resolved_bitmap_read`` did, so
        ``OMMX_ATTN_OUTLIER_REPR=combinadic`` needed a SECOND env var
        (``OMMX_ATTN_COMBINADIC_READ=1``) to be usable at all, on every path. That
        asymmetry is the defect it removes.
        """
        if self.combinadic_read is None:
            return self.outlier_repr == "combinadic"
        return bool(self.combinadic_read)

    def attention_config(self) -> CanonicalAttentionConfig:
        """Build the validated ``CanonicalAttentionConfig`` for this recipe."""
        if self.outlier_repr not in OUTLIER_REPRS:
            # No silent downgrade to relidx7: that would build a spec describing a
            # DIFFERENT storage layout than the pool allocates.
            raise ValueError(
                f"outlier_repr must be one of {OUTLIER_REPRS}; got "
                f"{self.outlier_repr!r}")
        has_outlier = self.k_format != "i2"
        k = TensorQuantSpec(
            format=self.k_format, vector_length=self.group_tokens,
            vector_axis="channel",
            outliers_per_vector=self.outliers_per_vector if has_outlier else 0,
            outlier_select=self.outlier_select, outlier_repr=self.outlier_repr)
        v = TensorQuantSpec(
            format="i2", vector_length=self.group_channels, vector_axis="token",
            outliers_per_vector=0, outlier_select="abs",
            outlier_repr=self.outlier_repr)
        return CanonicalAttentionConfig(
            k=k, v=v, head_dim=self.head_dim, n_q_heads=self.n_q_heads,
            n_kv_heads=self.n_kv_heads, sm_arch=self.sm_arch,
            max_context=self.max_context, batch_size=1,
            page_size=self.page_size, sink_window=self.sink_tokens,
            recent_window=self.recent_window)


def resolve_serving_config(
    *,
    head_dim: Optional[int] = None,
    n_q_heads: Optional[int] = None,
    n_kv_heads: Optional[int] = None,
    page_size: Optional[int] = None,
    max_context: Optional[int] = None,
    sm_arch: Optional[int] = None,
) -> OMMXServingConfig:
    """Read the env knobs (+ model geometry overrides) into an OMMXServingConfig."""
    # ── named recipe (OMMX_RECIPE), applied BEFORE any knob is read ──────────────
    # A no-op returning None unless OMMX_RECIPE names a preset, so the shipped default
    # resolution below is byte-for-byte what it was. When a preset IS named,
    # resolve_env materialises it into os.environ with setdefault semantics — an env
    # var (or its OMMX_ATTN_* alias) the operator set explicitly is left alone, so a
    # preset can be overridden one knob at a time. An unknown name raises
    # UnknownRecipeError listing the known names; it is deliberately NOT caught here,
    # because falling back to the default recipe after the operator asked for a
    # specific one is how you publish a number under a recipe nobody selected.
    #
    # It writes to os.environ rather than being threaded through this function because
    # the recipe has many independent readers that each go to the environment on their
    # own (this module, packed_only.kv_bits_breakdown, kv_pool.MultiSeqKVPool,
    # kv_store.CanonicalKVStore, pack.ommx_pack_kv_canonical_block, preflight.py, the
    # hf_eager modeling files). Reaching only this one would let the engine run recipe
    # A while the bit accounting reported recipe B.
    #
    # THIS CALL IS NO LONGER LOAD-BEARING FOR THE OTHER READERS, and that is the fix.
    # It used to be the ONLY apply_recipe_env call in the repo, so every other reader
    # honoured the preset only if this function happened to have run first in the same
    # process — measured: kv_bits_breakdown(128) returned the shipped 4.125 under
    # OMMX_RECIPE=paper-kv in a fresh process, and 2.75 in the same process after this
    # call. Each reader now calls recipes.resolve_env itself (resolution model (a));
    # this one stays because it must resolve before the knob reads below, and because
    # `applied_recipe` is what fills cfg.extra's provenance markers.
    applied_recipe = resolve_env()
    group_tokens = _env_int_alias("OMMX_KV_GROUP_TOKENS", "OMMX_ATTN_GROUP_TOKENS",
                                  GROUP_TOKENS)
    group_channels = _env_int_alias("OMMX_KV_GROUP_CHANNELS", "OMMX_ATTN_GROUP_CHANNELS",
                                    GROUP_TOKENS)
    # outlier count: OMMX_OUTLIER_PERCENT (fraction of the K token-group) wins when set,
    # else the absolute OMMX_ATTN_OUTLIERS npv. round to nearest, floor 1 for pct>0.
    if _env_present("OMMX_OUTLIER_PERCENT"):
        pct = _env_float("OMMX_OUTLIER_PERCENT", 0.0)
        npv = max(1, int(round(group_tokens * pct))) if pct > 0 else 0
    else:
        # DEFAULTS ARE THE ACCURACY RECIPE. Every published KV accuracy number is measured
        # under eval/lm_eval/models/ommx_hf_model.py's env block -- i2f4, 6 outliers, pow2
        # scales, signed select, gt=gc=32, sink 8, recent 32 -- and these defaults used to be
        # k=3 with pow2 OFF. So an engine started with no environment served a DIFFERENT
        # format from the one the accuracy table describes, and nothing said so. The knobs
        # below now match that recipe exactly; the harness keeps its setdefault block, which
        # is now a no-op restatement rather than a silent correction.
        npv = _env_int("OMMX_ATTN_OUTLIERS", DEFAULT_OUTLIERS)
    cfg = OMMXServingConfig(
        k_format=_env_str("OMMX_ATTN_K_FORMAT", "i2f4"),
        outliers_per_vector=npv,
        outlier_select=_env_str("OMMX_ATTN_OUTLIER_SELECT", "signed"),
        # bitmap, not relidx7: the two encodings carry the SAME positions and values and
        # produce bit-identical token ids, but the flat bitmask decodes with one wide
        # load plus a masked popcount instead of k dependent 7-bit extractions, and is
        # 0.58-0.71x the TPOT at >=16K on H200 and A100 (figure/data/repr_ab_*_20260903.json;
        # at 1K it is 1.24x SLOWER on H200 -- the win is a long-context one).
        # It is also what the paper describes as the GPU encoding. Bounded by the
        # kernel's k<=8 gate, which the k=6 default satisfies; a recipe with k>8 must
        # set OMMX_ATTN_OUTLIER_REPR=relidx7 explicitly.
        outlier_repr=_env_str("OMMX_ATTN_OUTLIER_REPR", DEFAULT_OUTLIER_REPR),
        use_pow2=_env_bool("OMMX_ATTN_POW2", DEFAULT_POW2),
        combinadic_read=(_env_bool("OMMX_ATTN_COMBINADIC_READ", False)
                         if _env_present("OMMX_ATTN_COMBINADIC_READ") else None),
        # tri-state: unset -> None -> "follow outlier_repr" (see the field comment).
        bitmap_read=(_env_bool("OMMX_ATTN_BITMAP_READ", False)
                     if _env_present("OMMX_ATTN_BITMAP_READ") else None),
        sink_tokens=_env_int_alias("OMMX_KV_SINK", "OMMX_ATTN_SINK", 8),
        recent_window=_env_int_alias("OMMX_KV_RECENT", "OMMX_ATTN_RECENT", 32),
        group_tokens=group_tokens,
        group_channels=group_channels,
        # default ON (dedicated FP4 map = current recipe); OMMX_KV_OUTLIER_MAP=0 -> base-shared
        # outlier; one of the three "<=3-bit lever" knobs, see the module docstring's KV
        # FOOTPRINT). pack.py reads the same env; carrying it on the config
        # makes the recipe explicit so the backend can pass kv_outlier_map=cfg.kv_outlier_map.
        kv_outlier_map=_env_bool("OMMX_KV_OUTLIER_MAP", True),
        strict=_env_bool("OMMX_STRICT", False),
        max_context=_env_int("OMMX_ATTN_MAXCTX", 4096),
    )
    if head_dim is not None:
        cfg.head_dim = int(head_dim)
    if n_q_heads is not None:
        cfg.n_q_heads = int(n_q_heads)
    if n_kv_heads is not None:
        cfg.n_kv_heads = int(n_kv_heads)
    if page_size is not None:
        cfg.page_size = int(page_size)
    if max_context is not None:
        cfg.max_context = int(max_context)
    if sm_arch is not None:
        cfg.sm_arch = int(sm_arch)
    if cfg.k_format not in ("i2f4", "itf4"):
        raise ValueError(f"OMMX_ATTN_K_FORMAT must be i2f4|itf4; got {cfg.k_format!r}")
    if cfg.outlier_repr not in OUTLIER_REPRS:
        raise ValueError(
            f"OMMX_ATTN_OUTLIER_REPR must be one of {OUTLIER_REPRS}; got "
            f"{cfg.outlier_repr!r}")
    # Checked on the RESOLVED values: both are tri-state, so testing the raw fields would
    # let repr=bitmap and repr=combinadic through while each silently resolves to True.
    if cfg.resolved_bitmap_read() and cfg.resolved_combinadic_read():
        raise ValueError(
            "OMMX_ATTN_BITMAP_READ and OMMX_ATTN_COMBINADIC_READ are mutually exclusive "
            "outlier-index sources (a pack carries exactly one of k_obmp / k_crank).")
    if cfg.bitmap_read and cfg.outlier_repr != "bitmap":
        raise ValueError(
            "OMMX_ATTN_BITMAP_READ=1 needs OMMX_ATTN_OUTLIER_REPR=bitmap: the kernel "
            "would read a k_obmp plane the store does not allocate under "
            f"outlier_repr={cfg.outlier_repr!r}.")
    if cfg.combinadic_read and cfg.outlier_repr != "combinadic":
        raise ValueError(
            "OMMX_ATTN_COMBINADIC_READ=1 needs OMMX_ATTN_OUTLIER_REPR=combinadic: the "
            "kernel would read a k_crank plane the store does not allocate under "
            f"outlier_repr={cfg.outlier_repr!r}.")
    # The bitmap decode stages a group's outlier VALUE stream in one int32
    # (ceil(k/2) <= 4 bytes), so the kernel refuses bitmap_read for k > 8 -- on the first
    # quantized group of the first decode step, long after the engine reported itself
    # up. Refuse it here, at engine build. Checked on the RESOLVED read: with the read
    # forced off (the A/B escape hatch) no bitmap value stream is ever staged.
    if cfg.resolved_bitmap_read() and cfg.outliers_per_vector > 8:
        raise ValueError(
            f"outlier_repr=bitmap cannot serve outliers_per_vector="
            f"{cfg.outliers_per_vector}: the decode kernel stages a group's outlier "
            "values in one int32, so bitmap_read is limited to k <= 8. Set "
            "OMMX_ATTN_OUTLIER_REPR=relidx7, or lower OMMX_ATTN_OUTLIERS / "
            "OMMX_OUTLIER_PERCENT.")
    # EVIDENCE, not decoration: which preset (if any) shaped this config, so a bench
    # JSON / log line can record the recipe by NAME instead of by 12 loose env vars.
    # Only set when a preset was actually applied, so `extra` stays empty by default
    # and an un-presetted config compares equal to one from before this feature.
    if applied_recipe is not None:
        cfg.extra[RECIPE_ENV_VAR.lower()] = applied_recipe
        # …and HOW MUCH of it actually applied. A preset only fills knobs the environment
        # does not already carry, which is correct (an operator's explicit export must
        # win) but means the NAME alone is not evidence of what ran: anything that got to
        # os.environ first takes the knob, including another module's defaults.
        #
        # figure/bench.py USED to be that other module — it applied the shipped canonical
        # dict with os.environ.setdefault() at model-build time, before anything expanded
        # OMMX_RECIPE, so `OMMX_RECIPE=paper-kv python3 figure/bench.py --method ommx` ran
        # gt=32/k=6 (shipped-kv, 4.375 avg bit) while this marker said "paper-kv" (2.75).
        # That file now resolves the preset FIRST (figure/bench.py::
        # _install_ommx_recipe_env), so it no longer inverts: re-measured, that same
        # command resolves gt=128/gc=128/k=10, avg 2.75.
        #
        # The ACCURACY arm still does. eval/lm_eval/models/ommx_hf_model.py runs the same
        # eleven-key setdefault loop at MODULE IMPORT with no preset resolution anywhere,
        # so it is unreachable by OMMX_RECIPE and wins over it. Measured by replaying that
        # loop verbatim and asking the accounting: under OMMX_RECIPE=paper-kv it resolves
        # gt=32/gc=32/k=6, avg 4.375 — shipped-kv under the paper-kv name. Until that file
        # calls recipes.resolve_env() before its loop, this marker is the evidence that a
        # run so labelled was not so measured.
        #
        # So name the knobs that did NOT come from the preset, with the value that won.
        # Absent when the preset applied cleanly, so a faithful run's `extra` is
        # unchanged.
        conflicting = {key: os.environ.get(key)
                       for key, val in preset_env(applied_recipe).items()
                       if os.environ.get(key) != val}
        if conflicting:
            cfg.extra[RECIPE_ENV_VAR.lower() + "_overridden"] = conflicting
    return cfg


__all__ = ["OMMXServingConfig", "resolve_serving_config", "GROUP_TOKENS",
           "OUTLIER_REPRS", "DEFAULT_OUTLIER_REPR", "DEFAULT_OUTLIERS", "DEFAULT_POW2"]
