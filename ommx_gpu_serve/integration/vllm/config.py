# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Serving-recipe resolution (env -> canonical KV recipe). Pure-python, no vLLM.

Resolves the OMMX KV recipe knobs from environment variables into a
``CanonicalAttentionConfig`` + ``WindowSpec`` for the vLLM backend. Kept separate
from the backend so it imports + tests on CPU without vLLM / triton.

The defaults encode the validated step1 recipe (K=i2f4 signed 3-of-32 outliers
relidx7, V=i2, sink=8, recent=32, group=32) — the closest canonical match to the
fakequant ``run.sh --phase kv`` reference. Knobs exist to drive the M2 fakequant
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
  OMMX_ATTN_OUTLIERS        K outliers per group (absolute npv)  (default 3)
  OMMX_ATTN_K_FORMAT        i2f4 | itf4                    (default i2f4)
  OMMX_ATTN_OUTLIER_SELECT  signed | abs                   (default signed)
  OMMX_ATTN_OUTLIER_REPR    relidx7 | combinadic | bitmap  (default relidx7)
  OMMX_ATTN_COMBINADIC_READ 0 | 1 (in-kernel rank unrank)  (default 0)
  OMMX_ATTN_BITMAP_READ     0 | 1 (kernel reads k_obmp)    (default: follows
                            OMMX_ATTN_OUTLIER_REPR == bitmap)
  OMMX_STRICT               0 | 1 (re-raise instead of bf16 fallback) (default 0)

NOT YET WIRED ON THE vLLM SERVING PATH — ``outlier_repr="bitmap"``. Read this before
setting either of the two knobs above to "bitmap"/1, because the config accepting a
value is NOT the same as the engine serving it:

  * the STORE side works: ``metadata.py`` passes ``outlier_repr`` to ``MultiSeqKVPool``,
    which allocates ``k_obmp`` and no ``k_oidx``;
  * the KERNEL side does NOT: the four
    ``ommx_paged_decode_attention_canonical(...)`` call sites in ``backend.py`` (and the
    two in ``hf_eager/_ommx_llama_modeling{,_batch}.py``) still pass only
    ``combinadic_read=`` / ``k_crank=``. They need ``bitmap_read=`` + ``k_obmp=`` added.
  * consequence: a bitmap-packed engine PACKS fine and then RAISES at the first decode
    ("requires the k_oidx relidx7 sidecar plane ..."). That is loud, never silent — but
    it is a runtime failure, not a startup refusal.
  * consequence: ``cfg.bitmap_read`` (and therefore ``OMMX_ATTN_BITMAP_READ``) has NO
    consumer yet. Setting it changes nothing on the serving path. The kernel's own
    ``OMMX_VLLM_KV_BITMAP_READ`` is the lever that does reach the launcher.

The repr IS fully usable today through ``attention/reference_op.py`` (which infers the
index source from the pack) — that is the path ``tests/test_bitmap_outlier.py`` gates.
Also note ``preflight.py``'s pool-footprint estimator still charges a ``k_crank`` field
for a bitmap recipe, so it under-prices K by 0.25 bit/element until it is split
three ways.

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

Nothing below defaults into the lever (gt/gc default to 32 and the map to ON), but the
defaults are not the published recipe either: ``OMMX_ATTN_OUTLIERS`` defaults to 3, not
6, and ``OMMX_ATTN_POW2`` defaults to 0, so a bare run is a THIRD number system
(K = 5.250, V = 3.000, K+V = 8.250, avg 4.125, 3.88x). Set the published recipe's env
explicitly — ``figure/bench.py`` and ``eval/lm_eval/models/ommx_hf_model.py`` do — if you
intend to quote 3.66x.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from ommx_gpu_serve.attention.config import (
    OUTLIER_REPRS as ATTENTION_OUTLIER_REPRS,
)
from ommx_gpu_serve.attention.config import (
    CanonicalAttentionConfig,
    TensorQuantSpec,
)
from ommx_gpu_serve.attention.kv_window import WindowSpec
from ommx_gpu_serve.recipes import (
    RECIPE_ENV_VAR,
    preset_env,
    resolve_env,
)

# Outlier-position storage encodings this serving config accepts. Deliberately a LOCAL
# tuple rather than ``attention.config.OUTLIER_REPRS``: this module must stay torch-free
# (``attention.pack``, which owns the packer-side SSOT ``OUTLIER_REPRS``, imports torch)
# and ``attention.config.OUTLIER_REPRS`` still lists only the first two. Widening that
# tuple to include "bitmap" is the ONE change this feature needs in a file outside this
# work's scope — until it lands, :meth:`OMMXServingConfig.attention_config` refuses the
# bitmap recipe loudly (it is a convenience constructor with no caller in the serving
# path, so nothing else is blocked).
OUTLIER_REPRS = ("relidx7", "combinadic", "bitmap")

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
    outliers_per_vector: int = 3
    outlier_select: str = "signed"
    # OUTLIER-POSITION STORAGE. Three membership-equivalent encodings of ONE logical
    # format (they dequantize bit-identically; only the index plane and its size differ):
    #   relidx7 (DEFAULT)  k_oidx  ceil(7k/8) B/frame — 7 bit per outlier slot
    #   combinadic         k_crank the ceil(log2 C(gt,k))-bit storage floor
    #   bitmap             k_obmp  ceil(gt/8) B/frame — the FLAT N-bit-per-group mask
    #                              the ICCAD paper attributes to the GPU implementation.
    #                              FLAT in k (1.0 bit/element), so it is CHEAPER than
    #                              relidx7 above the 1/7 = 14.3% density crossover: at
    #                              the canonical gt=32, k=6 recipe it is 32 bit/frame vs
    #                              relidx7's 48, i.e. K 6.000 -> 5.500 bit/element and
    #                              K+V 8.750 -> 8.250 (3.657x -> 3.879x).
    outlier_repr: str = "relidx7"
    use_pow2: bool = False
    combinadic_read: bool = False
    # Kernel-side index SOURCE for the bitmap repr. None = follow ``outlier_repr``
    # (the only correct decode for a bitmap-packed store, since it carries no k_oidx);
    # set explicitly only to force an A/B. UNVERIFIED ON HARDWARE — the kernel branch
    # has never run on a GPU; see paged_decode.ommx_paged_decode_attention_canonical.
    # NO CONSUMER YET: backend.py does not forward this to the kernel (see "NOT YET
    # WIRED" in the module docstring), so setting it is currently inert on the serving
    # path. It is carried here so the wiring is a one-line change, not a re-design.
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

    def attention_config(self) -> CanonicalAttentionConfig:
        """Build the validated ``CanonicalAttentionConfig`` for this recipe."""
        if self.outlier_repr not in ATTENTION_OUTLIER_REPRS:
            # No silent downgrade to relidx7: that would build a spec describing a
            # DIFFERENT storage layout than the pool allocates. See OUTLIER_REPRS above.
            raise NotImplementedError(
                f"CanonicalAttentionConfig cannot describe outlier_repr="
                f"{self.outlier_repr!r} yet: attention/config.py::OUTLIER_REPRS is "
                f"{ATTENTION_OUTLIER_REPRS} and TensorQuantSpec.validate() rejects "
                "anything else. Fix: add it to that tuple. The serving path itself does "
                "NOT use this method (metadata.py passes outlier_repr straight to the "
                "KV pool), so the recipe is fully usable without it.")
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
        npv = _env_int("OMMX_ATTN_OUTLIERS", 3)
    cfg = OMMXServingConfig(
        k_format=_env_str("OMMX_ATTN_K_FORMAT", "i2f4"),
        outliers_per_vector=npv,
        outlier_select=_env_str("OMMX_ATTN_OUTLIER_SELECT", "signed"),
        outlier_repr=_env_str("OMMX_ATTN_OUTLIER_REPR", "relidx7"),
        use_pow2=_env_bool("OMMX_ATTN_POW2", False),
        combinadic_read=_env_bool("OMMX_ATTN_COMBINADIC_READ", False),
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
    if cfg.bitmap_read and cfg.combinadic_read:
        raise ValueError(
            "OMMX_ATTN_BITMAP_READ and OMMX_ATTN_COMBINADIC_READ are mutually exclusive "
            "outlier-index sources (a pack carries exactly one of k_obmp / k_crank).")
    if cfg.bitmap_read and cfg.outlier_repr != "bitmap":
        raise ValueError(
            "OMMX_ATTN_BITMAP_READ=1 needs OMMX_ATTN_OUTLIER_REPR=bitmap: the kernel "
            "would read a k_obmp plane the store does not allocate under "
            f"outlier_repr={cfg.outlier_repr!r}.")
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
           "OUTLIER_REPRS"]
