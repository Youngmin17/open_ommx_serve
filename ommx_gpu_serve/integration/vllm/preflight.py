# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Startup guards for the OMMX vLLM backend — refuse configs it cannot serve.

Pure-python (no vLLM, no triton, no torch at import time) so it imports and
unit-tests on CPU. ``torch`` is imported LAZILY, only to read free device memory.

WHY THIS FILE EXISTS.  The OMMX KV sidecar (``attention/kv_pool.MultiSeqKVPool``)
is NOT paged: request slot ``b`` owns the STATIC CONTIGUOUS page block
``[b*P_per_seq, (b+1)*P_per_seq)`` and ``req_to_token`` is an identity ``arange``
(the ``rg``/``rt`` ``arange`` tables built in ``kv_pool.MultiSeqKVPool.__init__``).
vLLM's ``slot_mapping`` is never consulted by the sidecar write
path (``backend.do_kv_cache_update``). Three vLLM configurations therefore break
the sidecar in ways that are INVISIBLE at runtime — wrong tokens, not a crash:

  1. ``enable_prefix_caching`` (vLLM v1 default ON).  The pool slot is keyed by the
     request's FIRST physical block id (``metadata._req_keys`` -> ``block_table_tensor
     [:, 0]``).  Prefix caching makes two requests that share a prompt prefix SHARE
     that first physical block, so they hash to the SAME pool slot and overwrite each
     other's K/V.  Cross-request corruption, silent.
  2. ``enable_chunked_prefill`` (vLLM v1 default ON).  The ``n > 1`` write branch
     unconditionally does reset + ``append_block`` (see the "CHUNKED-PREFILL SEAM"
     comment in backend.py), so a CONTINUATION chunk wipes the sequence packed by the
     previous chunk.  Under ``OMMX_KV_RING=1`` the pool raises instead (``ring
     append_block expects a fresh request slot``) — still unsupported, just louder.
  3. Pool oversubscription.  Pool bytes are O(num_seqs * max_model_len) and are
     allocated EAGERLY per layer at the first B>1 step.  At vLLM defaults
     (``OMMX_ATTN_MAX_NUM_SEQS=256``, ``max_model_len=131072``, Llama-3.1-8B geometry,
     canonical recipe) ``ommx_pool_bytes_per_layer`` prices that at 35.122 GiB PER
     LAYER = 1123.9 GiB over 32 layers WITH ``OMMX_KV_RING=1``, and — because the ring
     is OFF by default, so ``k_hist``/``v_hist`` keep the whole bf16 sequence —
     163.020 GiB per layer = 5216.6 GiB with the pool's own defaults.  Either way a
     guaranteed OOM on the first multi-request decode step, thousands of tokens in.
     REACHABILITY (this is check 3's whole subtlety): ``MultiSeqKVPool`` is built
     LAZILY by ``metadata.OMMXBatchedStepManager.pool()`` on the FIRST B>1 step, so a
     run that can never take one — ``--max-num-seqs 1`` with no ``OMMX_ATTN_BATCHED``
     force-on, i.e. every arm of this repo's own single-request bench — never
     allocates a byte of it.  Refusing such a run over a 1.1 TiB projection is a FALSE
     REFUSAL, and a guard that cries wolf gets switched off, which is the silent
     fallback by another route.  The projection is therefore ALWAYS computed and
     ALWAYS reported (``report["pool_projection_informational"]``), but it is only a
     VIOLATION when a B>1 decode step is provably reachable.
  4. Scheduler concurrency above the pool's slot cap.  The pool has exactly
     ``num_seqs`` static slots and ``metadata.assign_slots`` now RAISES
     ``OMMXSlotAllocationError`` rather than recycling a live slot, so
     ``--max-num-seqs`` > ``OMMX_ATTN_MAX_NUM_SEQS`` is a GUARANTEED mid-run hard
     failure (it used to be silent aliasing).  A warning for a guaranteed failure is
     the wrong side of the no-silent-fallback law, so it is a violation too.

The repo's own bench already passes ``enable_prefix_caching=False,
enable_chunked_prefill=False`` (``bench/bench_e2e_a100.py::_make_engine``) — this module turns that
undocumented dependency into a loud, actionable startup failure (project law: NO
SILENT FALLBACK).

Every check is overridable ONLY by an explicit ``OMMX_UNSAFE_ALLOW_*`` env, and an
applied override emits a one-time warning to BOTH stderr and vLLM's logger.  A vLLM
config attribute that is MISSING is reported as ``"unknown"`` and is treated as a
VIOLATION (not as a pass) — a guard that silently skips itself is worse than no guard.

Envs read here:
  OMMX_POOL_BUDGET_FRAC              fraction of FREE device memory the projected pool
                                     may occupy (default 0.5)
  OMMX_UNSAFE_ALLOW_PREFIX_CACHING   skip check 1 (WILL corrupt KV across requests)
  OMMX_UNSAFE_ALLOW_CHUNKED_PREFILL  skip check 2 (WILL drop prefill chunks)
  OMMX_UNSAFE_ALLOW_POOL_OVERSUBSCRIBE  skip check 3 (WILL OOM)
  OMMX_UNSAFE_ALLOW_SLOT_CAP_OVERSUBSCRIBE  skip check 4 (WILL raise mid-serve)
Read but never written: OMMX_ATTN_BATCHED (the batched-route force-on, so check 3
knows whether a B>1 step is reachable) and OMMX_ATTN_MAX_NUM_SEQS (only to say WHERE
the resolved ``num_seqs`` came from in the error text; the backend resolves it as
``max(1, min(OMMX_ATTN_MAX_NUM_SEQS or 256, scheduler_config.max_num_seqs))`` -- a MIN,
not a precedence chain: the env can only LOWER the cap below the engine's, never raise
it above -- and passes the result in as ``num_seqs=``).
Plus the recipe envs the POOL ITSELF reads, mirrored here so the projection matches
what gets allocated: OMMX_KV_OUTLIER_MAP, OMMX_KV_INT8_SCALE, OMMX_KV_RING.

Every env above is parsed with the SAME truthiness as the module that acts on it
(``_pool_env_flag`` mirrors kv_pool.py, ``_backend_env_on`` mirrors backend.py) —
see those helpers for why a "cleaner" parse here is a silent mis-projection.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Optional

# vLLM release that first provides the module layout this backend subclasses
# (vllm.v1.attention.backend.AttentionCGSupport, v1.attention.backends.flash_attn,
# FlashAttentionImpl.do_kv_cache_update). 0.11-era trees have none of them.
MIN_VLLM_VERSION = (0, 21)

# One-time-warning latch (per worker process), keyed by check name.
_WARNED: set = set()

# appended to the pool-budget FIX line when not even a single slot fits the budget
_TOO_SMALL = " — even 1 slot does not fit, so lower max_model_len too"

__all__ = [
    "MIN_VLLM_VERSION",
    "ommx_pool_bytes_per_layer",
    "ommx_preflight_check",
]


# ══════════════════════════════════════════════════════════════════════════════
# small helpers  (no bare try/except that can swallow a real failure)
# ══════════════════════════════════════════════════════════════════════════════

_MISSING = object()


def _attr(obj: Any, name: str):
    """``(value, found)`` — getattr with a sentinel. Never raises, never guesses."""
    if obj is None:
        return None, False
    val = getattr(obj, name, _MISSING)
    if val is _MISSING:
        return None, False
    return val, True


def _chain(root: Any, *names: str):
    """Walk ``root.a.b.c`` defensively; ``(value, found)``. Missing link -> found=False."""
    cur = root
    for n in names:
        cur, ok = _attr(cur, n)
        if not ok:
            return None, False
    return cur, True


def _env_flag(name: str) -> bool:
    """True when ``name`` is set to anything other than an off-ish spelling."""
    raw = os.environ.get(name)
    if raw is None:
        return False
    return str(raw).strip().lower() not in {"", "0", "false", "off", "no"}


# The RAW off-ish spellings the KV pool compares against. NOT stripped, NOT
# lowercased: the OMMX_KV_OUTLIER_MAP / OMMX_KV_INT8_SCALE / OMMX_KV_RING reads in
# kv_pool.MultiSeqKVPool.__init__, and the first two again in
# pack.ommx_pack_kv_canonical_block, test the raw ``os.environ`` string against exactly
# this set.
_POOL_OFF_SPELLINGS = {"0", "false", "off", "no"}


def _pool_env_flag(name: str, default: bool) -> bool:
    """Resolve a RECIPE env exactly the way ``MultiSeqKVPool`` resolves it.

    AUTHORITY: ``ommx_gpu_serve/attention/kv_pool.py`` — ``MultiSeqKVPool.__init__``,
    which reads OMMX_KV_OUTLIER_MAP / OMMX_KV_INT8_SCALE in its recipe-resolution block
    and OMMX_KV_RING in its history-sizing block;
    ``attention/pack.py::ommx_pack_kv_canonical_block`` parses the first two
    identically.
    The pool writes, verbatim::

        raw = os.environ.get(NAME)
        value = (raw not in {"0", "false", "off", "no"}) if raw else <default>
        # OMMX_KV_RING spells the same thing as  bool(raw) and raw not in {...}

    which is CASE-SENSITIVE and does NOT strip.  To the pool, ``OMMX_KV_RING="False"``
    is ON, ``OMMX_KV_OUTLIER_MAP=" 0"`` is ON, and ``OMMX_KV_INT8_SCALE="OFF"`` is ON.
    This module's general-purpose ``_env_flag`` strips and lowercases, so using it for
    these three would price a plane the pool does not allocate (or miss one it does)
    and the OOM guard would check the wrong number while looking healthy.  MIRROR, do
    not improve: if that parsing is ever tightened, tighten kv_pool.py first and this
    helper second, or the two drift apart again.
    """
    raw = os.environ.get(name)
    if not raw:                # None or "" -> the pool's `if raw` / `bool(raw)` guard
        return bool(default)
    return raw not in _POOL_OFF_SPELLINGS


def _backend_env_on(name: str) -> bool:
    """Resolve a BACKEND ROUTE env exactly the way ``backend._env_on`` resolves it.

    AUTHORITY: ``integration/vllm/backend.py``::``_env_on`` — verbatim
    ``os.environ.get(name, "0").strip().lower() not in {"0", "false", "off", ""}``.
    Note what is NOT in that set: ``"no"``.  ``OMMX_ATTN_BATCHED=no`` therefore
    FORCE-ENABLES the batched route in the backend, while ``_env_flag`` here reads the
    same string as OFF.  That divergence would let this module's reachability proof
    declare "no B>1 step is possible" for a run whose backend routes every single step
    through the batched pool — the exact false-negative a startup guard must not have.
    """
    return os.environ.get(name, "0").strip().lower() not in {"0", "false", "off", ""}


def _int_or_none(raw: Any):
    """``int(raw)`` or ``None`` — never raises, never substitutes a default."""
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _valid_num_seqs(num_seqs: Any) -> bool:
    """True only for a genuine positive int pool-slot cap (bools/None/str/float are not)."""
    return (isinstance(num_seqs, int) and not isinstance(num_seqs, bool)
            and num_seqs > 0)


def _gib(nbytes: float) -> str:
    return f"{float(nbytes) / (1024.0 ** 3):.3f} GiB"


def _vllm_logger():
    """``(logger, unavailable_reason)``. vLLM absent -> ``(None, "<reason>")``.

    The reason is NOT swallowed: it is appended to the stderr warning so a missing
    log sink is visible rather than silently dropping the warning.
    """
    try:
        from vllm.logger import init_logger
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        return None, f"{type(exc).__name__}: {exc}"
    return init_logger("ommx_gpu_serve"), ""


def _warn_once(key: str, message: str) -> None:
    """Loud one-time warning: ALWAYS stderr, plus vLLM's logger when importable."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger, why = _vllm_logger()
    text = f"[ommx-preflight] {message}"
    if why:
        text += f"\n[ommx-preflight] (vllm logger unavailable: {why})"
    print(text, file=sys.stderr, flush=True)
    if logger is not None:
        logger.warning("[ommx-preflight] %s", message)


def _parse_version(raw: Any) -> tuple:
    """``"0.21.0"`` -> ``(0, 21, 0)``; stops at the first non-numeric chunk.

    Tolerates ``0.21.0rc1`` / ``0.22.0.dev1+g<sha>`` (-> ``(0, 21, 0)`` / ``(0, 22, 0)``).
    """
    out: list = []
    for chunk in str(raw).split("+")[0].split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if digits == "":
            break
        out.append(int(digits))
    return tuple(out)


def _combinadic_index_bytes(k: int, n: int) -> int:
    """Byte width of one combinadic rank field — mirror of ``codec.combinadic_index_*``.

    Inlined (rather than imported) so this module stays torch-free: ``attention.codec``
    imports torch.
    """
    k, n = int(k), int(n)
    if k <= 0 or k >= n:
        return 0
    bits = max(1, (math.comb(n, k) - 1).bit_length())
    return (bits + 7) // 8


# ══════════════════════════════════════════════════════════════════════════════
# pool byte projection  —  PLANE-FOR-PLANE mirror of MultiSeqKVPool.__init__
# ══════════════════════════════════════════════════════════════════════════════

def ommx_pool_bytes_per_layer(
    head_dim: int,
    n_kv_heads: int,
    num_seqs: int,
    max_seq_len: int,
    *,
    group_tokens: int = 32,
    page_size: int = 16,
    group_channels: int = 32,
    outliers_per_vector: int = 6,
    outlier_repr: str = "relidx7",
    kv_outlier_map: bool = True,
    kv_int8_scale: bool = True,
    kv_ring: bool = True,
    sink_tokens: int = 8,
    recent_window: int = 32,
    scale_bytes: int = 2,
    include_index_tables: bool = True,
) -> dict:
    """Bytes ONE layer's ``MultiSeqKVPool`` allocates, broken down per plane.

    EXACT mirror of ``ommx_gpu_serve/attention/kv_pool.py`` ``MultiSeqKVPool.__init__``
    (the allocation block right after the ``G_per_seq`` / ``P_per_seq`` sizing). Every
    plane below names the attribute it mirrors; a divergence here makes the OOM guard
    useless, so keep the two in lockstep:

      G_per_seq = max(1, ceil(max_seq_len / gt) + 1)   # +1 slack group (== G_cap sizing)
      P_per_seq = G_per_seq * (gt // page_size)
      G_cap     = num_seqs * G_per_seq ; P_cap = num_seqs * P_per_seq

    ``kv_int8_scale`` is the RESOLVED value: the pool computes it as
    ``env(OMMX_KV_INT8_SCALE, default=use_pow2) and use_pow2`` — int8 scale planes exist
    only under pow2 scales. Callers must fold ``use_pow2`` in before calling.
    ``scale_bytes`` is the bf16 scale/zp element width (2); it is NOT the int8 width.

    ``include_index_tables`` (DEFAULT True) counts the pool's two int32 index tables,
    ``rg = arange(G_cap)`` and ``rt = arange(P_cap)`` (``MultiSeqKVPool.__init__``), i.e.
    ``G_cap*4 + P_cap*4`` bytes; ``_req_to_group`` / ``_req_to_token`` are reshape
    VIEWS of those two and cost nothing on top. They are small next to the data planes
    but they are not zero, and ``P_cap`` grows with ``num_seqs * max_seq_len`` like
    everything else here, so omitting them made this projection an UNDER-estimate —
    the one direction an OOM guard must never err in. Pass False ONLY to reproduce the
    historical data-plane-only figure (e.g. against a longhand recomputation written
    before this term existed); the serving path must leave it True.

    Returns the per-plane byte counts (keys named after the pool attributes; a plane the
    recipe does not allocate is 0), ``"total"`` (their sum, the ONLY aggregate), and
    ``"geometry"`` (a nested dict of the derived sizing — not a byte count).
    """
    D = int(head_dim)
    H = int(n_kv_heads)
    B = int(num_seqs)
    S = int(max_seq_len)
    gt = int(group_tokens)
    ps = int(page_size)
    gc = int(group_channels)
    k = int(outliers_per_vector)
    repr_ = str(outlier_repr).lower()
    sbytes = int(scale_bytes)

    # ── mirror the pool's own constructor validation (same errors, same order) ──
    if D % 32 != 0:
        raise ValueError(f"head_dim must be a multiple of 32; got {D}")
    if gt not in (16, 32, 64, 128):
        raise ValueError(f"group_tokens must be in {{16,32,64,128}}; got {gt}")
    if gc not in (16, 32, 64, 128):
        raise ValueError(f"group_channels must be in {{16,32,64,128}}; got {gc}")
    if D % gc != 0:
        raise ValueError(f"head_dim ({D}) must be a multiple of group_channels ({gc})")
    if ps <= 0 or gt % ps != 0:
        raise ValueError(f"group_tokens ({gt}) must be a multiple of page_size ({ps})")
    if H <= 0:
        raise ValueError(f"n_kv_heads must be > 0; got {H}")
    if B <= 0:
        raise ValueError(f"num_seqs must be > 0; got {B}")
    if S <= 0:
        raise ValueError(f"max_seq_len must be > 0; got {S}")
    if sbytes <= 0:
        raise ValueError(f"scale_bytes must be > 0; got {sbytes}")

    pages_per_group = gt // ps
    G_per_seq = max(1, (S + gt - 1) // gt + 1)   # kv_pool.py: +1 slack group
    P_per_seq = G_per_seq * pages_per_group
    G_cap = B * G_per_seq
    P_cap = B * P_per_seq
    k_base_w = D // 4                            # 2-bit base, 4 codes per byte
    NGV = D // gc                                # V scale groups per head row
    scl = 1 if kv_int8_scale else sbytes         # k_scale / v_scale element width
    sdt = sbytes                                 # zp / map / hist bf16 element width

    # bf16 residual history sizing (OMMX_KV_RING). RING keeps only the live set per
    # request (sink U recent U the group being packed U the accumulating group);
    # OFF keeps the WHOLE bf16 sequence — the [num_seqs, max_seq_len, H, D] shadow
    # that is the real B>1 long-context OOM.
    ring_cap = (int(sink_tokens) + int(recent_window) + 2 * gt) if kv_ring else S

    planes = {
        # [P_cap, ps, H, D//4] uint8
        "k_base": P_cap * ps * H * k_base_w,
        # [P_cap, ps, H, D//4] uint8   (V is always i2 in the pool: no v_format knob)
        "v_main": P_cap * ps * H * k_base_w,
        # [G_cap, H, D] int8-or-bf16 / [G_cap, H, D] bf16
        "k_scale": G_cap * H * D * scl,
        "k_zp": G_cap * H * D * sdt,
        # [P_cap, ps, H, NGV] int8-or-bf16 / bf16
        "v_scale": P_cap * ps * H * NGV * scl,
        "v_zp": P_cap * ps * H * NGV * sdt,
        # dedicated FP4 range-map (OMMX_KV_OUTLIER_MAP=1); dropped for the
        # BASE-SHARED outlier variant (the <=3-bit KV lever).
        "k_fp4_mapscale": (G_cap * H * D * sdt) if (k > 0 and kv_outlier_map) else 0,
        "k_fp4_mapcenter": (G_cap * H * D * sdt) if (k > 0 and kv_outlier_map) else 0,
        # outlier index: relidx7 -> ceil(7k/8) bytes; anything else -> combinadic rank
        # (the pool's `elif self.k > 0:` branch treats every non-relidx7 repr as
        # combinadic, so mirror that exactly rather than second-guessing a typo).
        "k_oidx": (G_cap * H * D * ((7 * k + 7) // 8))
                  if (k > 0 and repr_ == "relidx7") else 0,
        "k_crank": (G_cap * H * D * _combinadic_index_bytes(k, gt))
                   if (k > 0 and repr_ != "relidx7") else 0,
        # outlier values: FP4, 2 per byte
        "k_oval": (G_cap * H * D * ((k + 1) // 2)) if k > 0 else 0,
        # [num_seqs, ring_cap, H, D] bf16 x2
        "k_hist": B * ring_cap * H * D * 2,
        "v_hist": B * ring_cap * H * D * 2,
        # fixed-address index tables (MultiSeqKVPool.__init__): rg=arange(G_cap) and
        # rt=arange(P_cap), both int32 (4 B/elem). See include_index_tables above.
        "req_to_group": (G_cap * 4) if include_index_tables else 0,
        "req_to_token": (P_cap * 4) if include_index_tables else 0,
    }
    out = dict(planes)
    out["total"] = sum(planes.values())
    out["geometry"] = {
        "head_dim": D, "n_kv_heads": H, "num_seqs": B, "max_seq_len": S,
        "group_tokens": gt, "page_size": ps, "group_channels": gc,
        "outliers_per_vector": k, "outlier_repr": repr_,
        "kv_outlier_map": bool(kv_outlier_map), "kv_int8_scale": bool(kv_int8_scale),
        "kv_ring": bool(kv_ring), "ring_cap": ring_cap,
        "G_per_seq": G_per_seq, "P_per_seq": P_per_seq,
        "G_cap": G_cap, "P_cap": P_cap, "NGV": NGV,
        "scale_elem_bytes": scl, "bf16_elem_bytes": sdt,
        "include_index_tables": bool(include_index_tables),
    }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# free-memory probe (lazy torch)
# ══════════════════════════════════════════════════════════════════════════════

# Reason PREFIX that marks "this host has NO CUDA DEVICE AT ALL", as opposed to "CUDA
# is here and the query still failed". The caller keys off this prefix, never off the
# prose, so the two cases cannot be told apart by accident (and a caller-supplied fake
# probe, e.g. in a unit test, is treated as the second — stricter — case by default).
_NO_CUDA_PREFIX = "no-cuda: "


def _free_device_bytes(device: Any):
    """``(free_bytes, reason)``. ``free_bytes is None`` -> ``reason`` says why.

    Reads ``torch.cuda.mem_get_info()[0]``. THREE distinct failure shapes, and the
    caller must treat them differently, so the reason string is machine-classifiable:

      * torch missing / ``torch.cuda.is_available()`` False -> reason starts with
        ``_NO_CUDA_PREFIX``. There is no CUDA device on this host, so the pool cannot
        be allocated here EITHER: nothing to budget-check, and the guard must not
        refuse a CPU CI run, a mocked engine, or a torch built without CUDA.
      * an explicit non-CUDA ``device=`` while CUDA IS available -> NOT prefixed. CUDA
        exists and the caller asked about the wrong device; that is a caller/config
        error and we must not guess a budget for it.
      * ``mem_get_info`` raising while CUDA IS available -> NOT prefixed. Something is
        wrong with the driver/context; unknown never counts as a pass.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        return None, f"{_NO_CUDA_PREFIX}torch not importable ({type(exc).__name__}: {exc})"
    if not torch.cuda.is_available():
        return None, f"{_NO_CUDA_PREFIX}torch.cuda.is_available() is False"
    dev = device
    if dev is not None:
        dev = torch.device(dev)
        if dev.type != "cuda":
            # deliberately NOT prefixed: CUDA is present, the request was not for it.
            return None, f"device {dev} is not a CUDA device"
    try:
        free, _total = torch.cuda.mem_get_info(dev) if dev is not None \
            else torch.cuda.mem_get_info()
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        # deliberately NOT prefixed: CUDA is available and the query still failed.
        return None, f"torch.cuda.mem_get_info failed ({type(exc).__name__}: {exc})"
    return int(free), ""


# ══════════════════════════════════════════════════════════════════════════════
# the preflight itself
# ══════════════════════════════════════════════════════════════════════════════

def ommx_preflight_check(
    vllm_config: Any,
    *,
    num_seqs: int,
    cfg: Any,
    num_layers: Optional[int] = None,
    device: Any = None,
    strict: bool = True,
) -> dict:
    """Refuse vLLM configurations the OMMX sidecar cannot serve. Returns a report.

    ``vllm_config``  vLLM's ``VllmConfig`` (read defensively; a missing attribute is a
                     violation, never a pass).
    ``num_seqs``     the pool's slot cap == ``OMMX_ATTN_MAX_NUM_SEQS`` (backend.py).
    ``cfg``          ``OMMXServingConfig`` (duck-typed) — supplies the recipe knobs the
                     pool is built from (group sizes, outliers, window, max_context).
    ``num_layers``   layers whose pools are allocated in THIS process; derived from
                     ``hf_config`` when None.
    ``device``       CUDA device to read free memory from (None -> current device).
    ``strict``       True (the serving default): raise ``RuntimeError`` listing every
                     violation. False: report + print them, do not raise — for offline
                     tooling/tests ONLY; it is never used by the serving path.

    Report keys worth knowing: ``report["pool"]`` carries the budget verdict
    (``status`` is one of ``ok`` / ``over-budget`` / ``not-reachable (...)`` /
    ``skipped: no CUDA device (...)`` / ``unknown (...)``) together with
    ``b_gt_1_reachable``, ``b_gt_1_reachable_because`` and ``num_seqs_origin``;
    ``report["pool_projection_informational"]`` appears ONLY when the pool bytes were
    projected but a B>1 decode step is not reachable, so the number is a fact about a
    hypothetical relaunch rather than a verdict about this one.

    Raises ``RuntimeError`` (strict) naming, for each violation, the cause and the fix.
    """
    report: dict = {
        "strict": bool(strict),
        "min_vllm_version": ".".join(str(x) for x in MIN_VLLM_VERSION),
        "overrides_applied": [],
        "warnings": [],
        "violations": [],
    }
    violations: list = report["violations"]

    # ── 1. vLLM version ───────────────────────────────────────────────────────
    # NOT overridable: below 0.21 the modules this backend subclasses do not exist,
    # so nothing downstream can work.
    try:
        import vllm as _vllm
        raw_ver, _ = _attr(_vllm, "__version__")
    except Exception as exc:  # noqa: BLE001 - reported as a violation, not swallowed
        raw_ver = None
        report["vllm_import_error"] = f"{type(exc).__name__}: {exc}"
    if raw_ver is None:
        report["vllm_version"] = "unknown"
        violations.append(
            "vLLM VERSION UNKNOWN: could not read vllm.__version__ "
            f"({report.get('vllm_import_error', 'attribute missing')}).\n"
            "  The OMMX backend subclasses vLLM v1 internals "
            f"(vllm.v1.attention.*) that exist only in vLLM >= "
            f"{report['min_vllm_version']}. Install it in the serving env: "
            "pip install 'vllm>=0.21'.")
    else:
        parsed = _parse_version(raw_ver)
        report["vllm_version"] = str(raw_ver)
        if parsed < MIN_VLLM_VERSION:
            violations.append(
                f"vLLM {raw_ver} is TOO OLD (need >= {report['min_vllm_version']}).\n"
                "  This backend subclasses vllm.v1.attention.backend.AttentionCGSupport, "
                "vllm.v1.attention.backends.flash_attn.FlashAttention{Backend,Impl,"
                "Metadata,MetadataBuilder} and overrides "
                "FlashAttentionImpl.do_kv_cache_update — none of which exist before "
                f"{report['min_vllm_version']}.\n"
                "  FIX: pip install 'vllm>=0.21' — the same floor both pyproject.toml "
                "files declare (root [ommx] extra, ommx_gpu_serve [vllm] extra), so an "
                "env below it was resolved before those pins or installed around them.")

    # ── 2. pool slot cap sanity ───────────────────────────────────────────────
    # NOT overridable: num_seqs <= 0 is a programming error, and MultiSeqKVPool would
    # silently clamp it to 1 (metadata.py: max(1, int(num_seqs))) and mis-serve.
    report["num_seqs"] = int(num_seqs) if _valid_num_seqs(num_seqs) else repr(num_seqs)
    if not _valid_num_seqs(num_seqs):
        violations.append(
            f"OMMX pool slot cap num_seqs={num_seqs!r} is not a positive integer.\n"
            "  The batched step manager clamps it to 1 (metadata.py "
            "OMMXBatchedStepManager.__init__), so every request would alias pool "
            "slot 0 -> cross-request KV corruption.\n"
            "  FIX: set OMMX_ATTN_MAX_NUM_SEQS to a positive integer (>= the number "
            "of concurrent requests you will serve).")

    # ── 3. prefix caching ─────────────────────────────────────────────────────
    pc_val, pc_found = _chain(vllm_config, "cache_config", "enable_prefix_caching")
    # vLLM's CacheConfig.enable_prefix_caching is Optional[bool]: None means "not
    # resolved yet" (EngineArgs turns an unset knob into True on v1), NOT "disabled".
    # Folding None into False would let the exact config this check exists to refuse
    # pass silently, so None is demoted to UNKNOWN and handled by the not-found branch.
    pc_found = pc_found and pc_val is not None
    report["enable_prefix_caching"] = bool(pc_val) if pc_found else "unknown"
    pc_override = _env_flag("OMMX_UNSAFE_ALLOW_PREFIX_CACHING")
    pc_bad = (not pc_found) or bool(pc_val)
    if pc_bad and pc_override:
        report["overrides_applied"].append("OMMX_UNSAFE_ALLOW_PREFIX_CACHING")
        _warn_once(
            "prefix_caching",
            "OMMX_UNSAFE_ALLOW_PREFIX_CACHING=1 -> running with "
            f"enable_prefix_caching={report['enable_prefix_caching']!r}. The OMMX KV "
            "sidecar keys its pool slot on block_table_tensor[:,0]; requests sharing a "
            "prompt prefix share that block -> SAME pool slot -> CROSS-REQUEST KV "
            "CORRUPTION. Outputs from this run are NOT trustworthy.")
    elif pc_bad:
        state = ("ENABLED" if pc_found else
                 "UNKNOWN (vllm_config.cache_config.enable_prefix_caching is "
                 "missing or None/unresolved)")
        violations.append(
            f"prefix caching is {state} — unsupported by the OMMX KV sidecar.\n"
            "  WHY: the sidecar is NOT paged. A request's pool slot is keyed by its "
            "FIRST physical block id (metadata.py _req_keys -> block_table_tensor[:,0]) "
            "and slot b owns the static block [b*P_per_seq,(b+1)*P_per_seq) with an "
            "identity req_to_token (kv_pool.py). With prefix caching, two requests "
            "sharing a prompt prefix SHARE that first physical block -> they map to the "
            "SAME pool slot -> they overwrite each other's K/V. The corruption is "
            "silent: wrong tokens, no exception.\n"
            "  FIX: launch with --no-enable-prefix-caching (CLI) or "
            "enable_prefix_caching=False (LLM/EngineArgs) — what "
            "bench/bench_e2e_a100.py already does.\n"
            "  Override (UNSAFE, corrupts KV): OMMX_UNSAFE_ALLOW_PREFIX_CACHING=1")

    # ── 4. chunked prefill ────────────────────────────────────────────────────
    # chunked_prefill_enabled is the RESOLVED SchedulerConfig field in vLLM v1;
    # enable_chunked_prefill is the user-facing (possibly None) EngineArgs knob.
    cp_val, cp_found = _chain(vllm_config, "scheduler_config", "chunked_prefill_enabled")
    if not cp_found or cp_val is None:
        cp_val, cp_found = _chain(vllm_config, "scheduler_config",
                                  "enable_chunked_prefill")
    # enable_chunked_prefill is Optional[bool] on EngineArgs — None is UNRESOLVED, not
    # "off" (same reasoning as enable_prefix_caching above): demote it to UNKNOWN.
    cp_found = cp_found and cp_val is not None
    report["chunked_prefill"] = bool(cp_val) if cp_found else "unknown"
    cp_override = _env_flag("OMMX_UNSAFE_ALLOW_CHUNKED_PREFILL")
    cp_bad = (not cp_found) or bool(cp_val)
    if cp_bad and cp_override:
        report["overrides_applied"].append("OMMX_UNSAFE_ALLOW_CHUNKED_PREFILL")
        _warn_once(
            "chunked_prefill",
            "OMMX_UNSAFE_ALLOW_CHUNKED_PREFILL=1 -> running with "
            f"chunked_prefill={report['chunked_prefill']!r}. Every n>1 write does "
            "reset + append_block, so each continuation chunk WIPES the sequence packed "
            "by the previous chunk. Outputs from this run are NOT trustworthy.")
    elif cp_bad:
        state = ("ENABLED" if cp_found else
                 "UNKNOWN (scheduler_config.chunked_prefill_enabled / "
                 "enable_chunked_prefill is missing or None/unresolved)")
        violations.append(
            f"chunked prefill is {state} — unsupported by the OMMX KV sidecar.\n"
            "  WHY: the write path treats EVERY multi-token step (n>1) as a whole new "
            "sequence: it does reset + append_block (backend.py, the "
            "'CHUNKED-PREFILL SEAM' comment; the batched writer resets the slot's "
            "seq_len/packed_groups the same way). A continuation chunk therefore "
            "DISCARDS the KV packed by the previous chunk instead of appending to it. "
            "Under OMMX_KV_RING=1 the pool raises ('ring append_block expects a fresh "
            "request slot') instead of silently truncating — unsupported either way.\n"
            "  FIX: launch with --no-enable-chunked-prefill (CLI) or "
            "enable_chunked_prefill=False (LLM/EngineArgs) — what "
            "bench/bench_e2e_a100.py already does.\n"
            "  Override (UNSAFE, drops prefill chunks): "
            "OMMX_UNSAFE_ALLOW_CHUNKED_PREFILL=1")

    # ── 5. projected pool footprint vs free memory, and scheduler concurrency vs
    #       the pool slot cap. (These are the module docstring's WHY-items 3 and 4;
    #       the numbering here counts the two pure-sanity checks above as 1 and 2.)
    #       The footprint half is REACHABILITY-AWARE — see _pool_budget_check.
    _pool_budget_check(vllm_config, num_seqs=num_seqs, cfg=cfg,
                       num_layers=num_layers, device=device, report=report,
                       violations=violations)

    report["ok"] = not violations
    if violations:
        header = (f"OMMX vLLM preflight FAILED "
                  f"({len(violations)} unsupported configuration"
                  f"{'s' if len(violations) != 1 else ''}):")
        body = "\n\n".join(f"  [{i + 1}] {v}" for i, v in enumerate(violations))
        text = f"{header}\n\n{body}\n"
        # Always print: a vLLM worker subprocess can lose the raised traceback, and a
        # guard nobody sees is the silent fallback this file exists to forbid.
        print(f"[ommx-preflight] {text}", file=sys.stderr, flush=True)
        if strict:
            raise RuntimeError(text)
    return report


def _pool_budget_check(vllm_config: Any, *, num_seqs: int, cfg: Any,
                       num_layers: Optional[int], device: Any, report: dict,
                       violations: list) -> None:
    """Checks 3 and 4: pool bytes vs free memory, and scheduler concurrency vs slots.

    REACHABILITY-AWARE (check 3).  ``MultiSeqKVPool`` is built LAZILY, on the FIRST
    B>1 decode step (``metadata.OMMXBatchedStepManager.pool()``).  A run that can never
    take such a step never allocates one byte of it, so projecting 1.1 TiB at engine
    start and refusing a single-request bench is a FALSE REFUSAL — and a guard that
    cries wolf gets exported out of the launch script, which is the silent fallback
    this file exists to forbid, arriving by a different door.  So: the projection is
    ALWAYS computed and ALWAYS reported (under ``report["pool_projection_informational"]``
    when it is only informational), and it is a VIOLATION only when a B>1 decode step
    is reachable.  "Reachable" is proven from the ENGINE, not guessed: scheduler
    ``max_num_seqs > 1``, or ``OMMX_ATTN_BATCHED`` force-on (which routes even a B==1
    decode through the batched pool).  An UNREADABLE scheduler cap counts as reachable
    — unknown is never safe here.
    """
    override = _env_flag("OMMX_UNSAFE_ALLOW_POOL_OVERSUBSCRIBE")
    pool: dict = {}
    report["pool"] = pool
    unknowns: list = []

    # budget fraction (parse loudly — a typo must not silently become the default)
    raw_frac = os.environ.get("OMMX_POOL_BUDGET_FRAC")
    if raw_frac is None or str(raw_frac).strip() == "":
        frac = 0.5
    else:
        frac = float(raw_frac)   # ValueError here is intentional and loud
        if not (0.0 < frac <= 1.0):
            raise ValueError(
                f"OMMX_POOL_BUDGET_FRAC must be in (0, 1]; got {raw_frac!r}")
    pool["budget_frac"] = frac

    # An invalid slot cap is ALREADY a violation (check 2) and makes the projection
    # meaningless (ommx_pool_bytes_per_layer would raise ValueError over it, masking
    # the actionable report). Stop here; the violation list still carries the cause.
    if not _valid_num_seqs(num_seqs):
        pool["status"] = "not-projected (invalid num_seqs; see the num_seqs violation)"
        return

    model_config, _ = _attr(vllm_config, "model_config")
    hf, hf_found = _attr(model_config, "hf_config")
    if hf_found:
        # multimodal wrappers nest the LM geometry under text_config
        txt, txt_found = _attr(hf, "text_config")
    else:
        txt, txt_found = None, False

    def _hf_int(*names):
        """First present NUMERIC attribute among ``names`` on hf/text_config."""
        for src in (hf, txt if txt_found else None):
            if src is None:
                continue
            for n in names:
                v, ok = _attr(src, n)
                if ok and isinstance(v, (int, float)) and not isinstance(v, bool):
                    return int(v), True
        return None, False

    # ── layer count (whole model; PP>1 shards it across ranks, so this is a
    # conservative UPPER bound for one worker — the safe direction for an OOM guard).
    if num_layers is None:
        num_layers, nl_found = _hf_int("num_hidden_layers", "n_layer", "num_layers")
    elif isinstance(num_layers, int) and not isinstance(num_layers, bool):
        num_layers, nl_found = int(num_layers), True
    else:
        num_layers, nl_found = None, False   # a non-int caller value is NOT a layer count
    pool["num_layers"] = int(num_layers) if nl_found else "unknown"
    if not nl_found or int(num_layers) <= 0:
        unknowns.append("layer count (hf_config.num_hidden_layers)")

    # ── KV geometry, PER TENSOR-PARALLEL RANK (the pool is built from the sharded
    # Impl head counts). vLLM replicates KV heads when n_kv_heads < tp, hence max(1,·).
    tp, tp_found = _chain(vllm_config, "parallel_config", "tensor_parallel_size")
    tp = int(tp) if (tp_found and tp) else 1
    kvh_total, kvh_found = _hf_int("num_key_value_heads", "num_attention_heads")
    head_dim, hd_found = _hf_int("head_dim")
    if not hd_found:
        hidden, h_ok = _hf_int("hidden_size")
        nheads, n_ok = _hf_int("num_attention_heads")
        if h_ok and n_ok and nheads:
            head_dim, hd_found = int(hidden) // int(nheads), True
    n_kv_heads = max(1, int(kvh_total) // tp) if kvh_found else None
    pool["tensor_parallel_size"] = tp
    pool["n_kv_heads_total"] = int(kvh_total) if kvh_found else "unknown"
    pool["n_kv_heads_per_rank"] = n_kv_heads if kvh_found else "unknown"
    pool["head_dim"] = int(head_dim) if hd_found else "unknown"
    if not kvh_found:
        unknowns.append("KV head count (hf_config.num_key_value_heads)")
    if not hd_found:
        unknowns.append("head_dim (hf_config.head_dim / hidden_size//num_attention_heads)")
    # cross-check against the cfg the pool is actually built from (backend.py seeds it
    # from the same hf_config, so a mismatch means one of the two is wrong).
    cfg_hd, _ = _attr(cfg, "head_dim")
    cfg_kvh, _ = _attr(cfg, "n_kv_heads")
    pool["cfg_head_dim"] = cfg_hd
    pool["cfg_n_kv_heads"] = cfg_kvh
    if hd_found and cfg_hd is not None and int(cfg_hd) != int(head_dim):
        _warn_once("geom_hd", f"OMMXServingConfig.head_dim={cfg_hd} != hf_config-derived "
                              f"{head_dim}; the projection uses the LARGER (conservative).")
        head_dim = max(int(cfg_hd), int(head_dim))
    if kvh_found and cfg_kvh is not None and int(cfg_kvh) != int(n_kv_heads):
        _warn_once("geom_kvh", f"OMMXServingConfig.n_kv_heads={cfg_kvh} != hf_config-derived "
                               f"per-rank {n_kv_heads}; using the LARGER (conservative).")
        n_kv_heads = max(int(cfg_kvh), int(n_kv_heads))

    # ── per-slot length: the pool is sized to cfg.max_context (== the captured
    # max_model_len). Take the larger of the two knowns — if they disagree the pool is
    # already mis-sized (the ctx>=4096 index_copy_ OOB), and the bigger number is the
    # one that OOMs.
    cfg_ctx, cfg_ctx_found = _attr(cfg, "max_context")
    mml, mml_found = _attr(model_config, "max_model_len")
    lens = [int(x) for x, ok in ((cfg_ctx, cfg_ctx_found), (mml, mml_found))
            if ok and isinstance(x, (int, float)) and not isinstance(x, bool)
            and int(x) > 0]
    max_seq_len = max(lens) if lens else None
    pool["cfg_max_context"] = int(cfg_ctx) if cfg_ctx_found else "unknown"
    pool["max_model_len"] = int(mml) if mml_found else "unknown"
    pool["max_seq_len_used"] = max_seq_len if max_seq_len else "unknown"
    if max_seq_len is None:
        unknowns.append("max sequence length (model_config.max_model_len / cfg.max_context)")

    # ── recipe knobs. NOTE (metadata.py plumbing), and it decides WHICH parser is
    # authoritative for each knob:
    #   * kv_int8_scale / kv_ring are NOT forwarded by OMMXBatchedStepManager.pool(),
    #     so MultiSeqKVPool resolves OMMX_KV_INT8_SCALE / OMMX_KV_RING from the env
    #     with its OWN raw, case-sensitive, unstripped comparison -> _pool_env_flag.
    #   * kv_outlier_map IS forwarded (`pool(): kv_outlier_map=c.kv_outlier_map`), and
    #     the pool only reads the env when that kwarg is None. Its resolution is NOT
    #     byte-identical to the pool's own read — config._env_bool strips and
    #     lowercases, kv_pool.py does not — so " 0" gives cfg=False and pool-env=True.
    #     Follow the pool's actual order: forwarded cfg value first, raw env second.
    # The projection must mirror what MultiSeqKVPool ACTUALLY allocates, not what the
    # config merely declares and not what a tidier parser would have wanted.
    cfg_missing: list = []

    def _knob(name, default):
        """cfg field, recording (never silently defaulting) a missing one."""
        v, ok = _attr(cfg, name)
        if not ok or v is None:
            cfg_missing.append(f"cfg.{name}")
            return default
        return v

    use_pow2 = bool(_knob("use_pow2", False))
    # kv_outlier_map: metadata.py DOES forward it (``pool(): kv_outlier_map=c.kv_outlier_map``),
    # and MultiSeqKVPool only falls back to its own env read when the kwarg is None. So
    # mirror the pool's ACTUAL resolution order: the forwarded config value first, its
    # raw env parse second. (These two can disagree — config._env_bool strips and
    # lowercases, kv_pool.py does not — so preferring the forwarded value is not a
    # nicety, it is the only way to price the plane the pool really allocates.)
    cfg_omap, cfg_omap_found = _attr(cfg, "kv_outlier_map")
    if cfg_omap_found and cfg_omap is not None:
        kv_outlier_map = bool(cfg_omap)
        omap_src = "cfg.kv_outlier_map (forwarded to the pool by metadata.py)"
    else:
        kv_outlier_map = _pool_env_flag("OMMX_KV_OUTLIER_MAP", True)
        omap_src = "OMMX_KV_OUTLIER_MAP (pool-side default; cfg carries no value)"
    # kv_int8_scale / kv_ring are NOT forwarded by metadata.py — the pool resolves both
    # from the env itself, so these MUST use the pool's own raw truthiness.
    kv_int8_scale = _pool_env_flag("OMMX_KV_INT8_SCALE", use_pow2) and use_pow2
    kv_ring = _pool_env_flag("OMMX_KV_RING", False)  # pool default OFF (full bf16 shadow)
    knobs = {
        "group_tokens": int(_knob("group_tokens", 32)),
        "group_channels": int(_knob("group_channels", 32)),
        "page_size": int(_knob("page_size", 16)),
        "outliers_per_vector": int(_knob("outliers_per_vector", 0)),
        "outlier_repr": str(_knob("outlier_repr", "relidx7")),
        "sink_tokens": int(_knob("sink_tokens", 0)),
        "recent_window": int(_knob("recent_window", 0)),
        "kv_outlier_map": kv_outlier_map,
        "kv_int8_scale": kv_int8_scale,
        "kv_ring": kv_ring,
    }
    pool["knobs"] = knobs
    pool["use_pow2"] = use_pow2
    pool["kv_outlier_map_source"] = omap_src
    if cfg_missing:
        unknowns.append("recipe knobs (" + ", ".join(cfg_missing) + ")")

    # ── is a B>1 decode step — the ONLY thing that allocates this pool — reachable? ──
    # metadata.OMMXBatchedStepManager.pool() builds MultiSeqKVPool lazily, on the first
    # batched step. Two independent ways such a step can occur:
    #   * the engine's scheduler admits more than one sequence at a time
    #     (scheduler_config.max_num_seqs > 1), or
    #   * OMMX_ATTN_BATCHED force-routes even a B==1 decode through the batched pool
    #     (backend.py _BATCHED) — read with the BACKEND's truthiness (_backend_env_on),
    #     because backend.py treats the spelling "no" as ON and this module used to
    #     treat it as OFF, which would have turned a forced-batched run into a
    #     "provably unreachable" one.
    # UNKNOWN counts as REACHABLE: a scheduler cap we cannot read is not a proof that
    # no batched step happens, and "unknown" is never "safe" in this file.
    sched_ns, sched_found = _chain(vllm_config, "scheduler_config", "max_num_seqs")
    sched_found = (sched_found and isinstance(sched_ns, int)
                   and not isinstance(sched_ns, bool))
    pool["scheduler_max_num_seqs"] = int(sched_ns) if sched_found else "unknown"
    batched_forced = _backend_env_on("OMMX_ATTN_BATCHED")
    pool["ommx_attn_batched_forced"] = batched_forced
    if batched_forced:
        reachable = True
        why_reach = ("OMMX_ATTN_BATCHED is on, which force-routes even a B==1 decode "
                     "through the batched pool")
    elif not sched_found:
        reachable = True
        why_reach = ("scheduler_config.max_num_seqs is missing/unreadable, so a B>1 "
                     "step cannot be ruled out (unknown is not 'safe')")
    elif int(sched_ns) > 1:
        reachable = True
        why_reach = f"scheduler_config.max_num_seqs={int(sched_ns)} > 1"
    else:
        reachable = False
        why_reach = (f"scheduler_config.max_num_seqs={int(sched_ns)} <= 1 and "
                     "OMMX_ATTN_BATCHED is off, so no B>1 decode step can be scheduled")
    pool["b_gt_1_reachable"] = reachable
    pool["b_gt_1_reachable_because"] = why_reach

    # ── where did the slot cap we are pricing come from? ─────────────────────
    # This module is NOT the resolver — backend.py is, and it passes the RESULT in as
    # num_seqs=. We reconstruct the provenance only so the error text can name the knob
    # the operator has to turn instead of quoting a bare 256 that appears in no launch
    # command anywhere. MIRROR of ``backend.py::_resolved_max_num_seqs``, verbatim::
    #
    #     v = 256 if OMMX_ATTN_MAX_NUM_SEQS is unset/blank else int(that env)
    #     if the engine's scheduler_config.max_num_seqs is known: v = min(v, engine)
    #     return max(1, v)
    #
    # Note the ``min``: this is NOT the "env WINS over engine" precedence the prose
    # shorthand suggests. An explicit OMMX_ATTN_MAX_NUM_SEQS=512 under a
    # --max-num-seqs 128 engine resolves to 128, and an unset env under a
    # --max-num-seqs 1024 engine resolves to 256 (the default, capped by nothing).
    # Reconstructing it as a precedence chain instead of a min produced two
    # self-contradicting messages: "OMMX_ATTN_MAX_NUM_SEQS='512' ... does NOT match the
    # num_seqs=128 the backend passed in — one of the two is stale" (neither was: the
    # engine capped it), and "no engine max_num_seqs to inherit" printed in a report
    # whose own pool["scheduler_max_num_seqs"] read 1024. Both told the operator to go
    # look for a bug that does not exist, which is how a guard loses its credibility.
    raw_cap = os.environ.get("OMMX_ATTN_MAX_NUM_SEQS")
    env_set = raw_cap is not None and str(raw_cap).strip() != ""
    # backend._env_int falls back to the default on a malformed value rather than
    # raising, so a garbage env spells 256 there and must spell 256 here too.
    env_val = (_int_or_none(raw_cap) if env_set else None)
    env_desc = (f"the explicit env OMMX_ATTN_MAX_NUM_SEQS={raw_cap!r}" if env_set
                else "the built-in default 256 (OMMX_ATTN_MAX_NUM_SEQS unset)")
    if env_set and env_val is None:
        env_val = 256
        env_desc = (f"the built-in default 256 (OMMX_ATTN_MAX_NUM_SEQS={raw_cap!r} is "
                    "not an integer, so backend._env_int falls back to the default)")
    if env_val is None:
        env_val = 256
    if sched_found and int(sched_ns) < int(env_val):
        # the engine, not the env/default, is the binding constraint
        origin = (f"the engine's scheduler_config.max_num_seqs={int(sched_ns)}, which "
                  f"CAPS {env_desc} ({env_val}) — backend._resolved_max_num_seqs takes "
                  "the MIN of the two, because slots the scheduler can never fill are "
                  "pure wasted HBM")
        reconstructed = int(sched_ns)
    else:
        origin = env_desc
        if sched_found:
            origin += (f" (the engine's scheduler_config.max_num_seqs={int(sched_ns)} "
                       "does not cap it)")
        reconstructed = int(env_val)
    reconstructed = max(1, reconstructed)
    if reconstructed != int(num_seqs):
        # Now this really IS a disagreement: the same inputs, resolved the same way,
        # do not reproduce the cap the backend handed us. Say so instead of guessing.
        origin += (f". WARNING: replaying backend._resolved_max_num_seqs on these "
                   f"inputs yields {reconstructed}, not the num_seqs={int(num_seqs)} "
                   "the caller passed in — the two resolvers have drifted, or the env "
                   "changed after backend.py imported it (backend.py reads "
                   "OMMX_ATTN_MAX_NUM_SEQS ONCE, at module import)")
    pool["num_seqs_origin"] = origin
    pool["num_seqs_reconstructed"] = reconstructed

    # ── check 4: scheduler concurrency vs the pool's slot cap ────────────────
    # WAS a warning here, describing an allocator that no longer exists ("_slots:
    # len(self._key_to_slot) % self.num_seqs" wrapped onto an occupied slot). metadata.py
    # now RAISES instead: assign_slots() refuses to recycle a live slot and throws
    # OMMXSlotAllocationError the first time a step needs slot num_seqs+1, and it
    # rejects duplicate request keys before mutating anything. So this condition is no
    # longer "may silently alias" — it is a GUARANTEED hard failure mid-serve, and
    # warning about a guaranteed failure is the wrong side of the no-silent-fallback
    # law. Promoted to a violation, with its own override like the other three.
    # (Note it cannot collide with the unreachable case above: sched_ns > num_seqs >= 1
    # implies sched_ns >= 2, i.e. reachable is already True here.)
    cap_override = _env_flag("OMMX_UNSAFE_ALLOW_SLOT_CAP_OVERSUBSCRIBE")
    if sched_found and int(sched_ns) > int(num_seqs):
        cap_detail = (
            f"scheduler max_num_seqs={int(sched_ns)} EXCEEDS the OMMX pool slot cap "
            f"num_seqs={int(num_seqs)} (resolved from {origin}).\n"
            "  WHY: the batched pool owns exactly num_seqs STATIC slots. The first step "
            "whose live requests need one more than that makes metadata.py assign_slots() "
            "raise OMMXSlotAllocationError — it refuses to recycle a slot a live request "
            "still holds. This is not a risk that might not materialise: with a scheduler "
            "allowed to admit more concurrent sequences than there are slots, it is a hard "
            "failure the moment the engine actually does so, thousands of tokens into the "
            "run.\n"
            f"  FIX (either knob): raise OMMX_ATTN_MAX_NUM_SEQS to >= {int(sched_ns)}, or "
            f"launch with --max-num-seqs <= {int(num_seqs)}. Raising the cap MULTIPLIES "
            "the pool bytes projected by check 3 (every plane is linear in num_seqs); "
            "lowering --max-num-seqs does not.\n"
            "  Override (UNSAFE, the run WILL raise mid-serve): "
            "OMMX_UNSAFE_ALLOW_SLOT_CAP_OVERSUBSCRIBE=1")
        if cap_override:
            report["overrides_applied"].append(
                "OMMX_UNSAFE_ALLOW_SLOT_CAP_OVERSUBSCRIBE")
            _warn_once("slot_cap",
                       "OMMX_UNSAFE_ALLOW_SLOT_CAP_OVERSUBSCRIBE=1 -> " + cap_detail)
        else:
            violations.append(cap_detail)

    if unknowns:
        if not reachable:
            # Nothing will be allocated, so a geometry we cannot read cannot hurt this
            # run. Say so precisely — "not-reachable" plus what was missing — instead
            # of either refusing or pretending the projection succeeded.
            pool["status"] = ("not-reachable (" + why_reach + "); projection "
                              "unavailable (" + "; ".join(unknowns) + ")")
            return
        pool["status"] = "unknown (" + "; ".join(unknowns) + ")"
        if override:
            report["overrides_applied"].append(
                "OMMX_UNSAFE_ALLOW_POOL_OVERSUBSCRIBE")
            _warn_once("pool_unknown",
                       "OMMX_UNSAFE_ALLOW_POOL_OVERSUBSCRIBE=1 -> the OMMX pool "
                       "footprint could NOT be projected (" + "; ".join(unknowns) +
                       ") and the check is being skipped. A B>1 decode step may OOM.")
            return
        violations.append(
            "OMMX pool footprint CANNOT BE PROJECTED: missing " + "; ".join(unknowns) +
            ".\n  The sidecar allocates O(num_seqs * max_model_len) bytes PER LAYER "
            "eagerly at the first B>1 step; without this geometry the guard cannot "
            "tell whether that fits in device memory, and 'unknown' is not 'safe'.\n"
            "  FIX: launch through a normal vLLM engine (the values come from "
            "vllm_config.model_config.hf_config), or, if you accept the OOM risk, set "
            "OMMX_UNSAFE_ALLOW_POOL_OVERSUBSCRIBE=1.")
        return

    per_layer = ommx_pool_bytes_per_layer(
        int(head_dim), int(n_kv_heads), int(num_seqs), int(max_seq_len),
        group_tokens=knobs["group_tokens"], page_size=knobs["page_size"],
        group_channels=knobs["group_channels"],
        outliers_per_vector=knobs["outliers_per_vector"],
        outlier_repr=knobs["outlier_repr"], kv_outlier_map=kv_outlier_map,
        kv_int8_scale=kv_int8_scale, kv_ring=kv_ring,
        sink_tokens=knobs["sink_tokens"], recent_window=knobs["recent_window"],
        # count rg/rt (MultiSeqKVPool.__init__). Explicit rather than defaulted: the
        # serving guard must never be the caller that under-projects.
        include_index_tables=True)
    per_layer_bytes = int(per_layer["total"])
    total_bytes = per_layer_bytes * int(num_layers)
    pool["per_layer"] = per_layer
    pool["per_layer_bytes"] = per_layer_bytes
    pool["total_bytes"] = total_bytes

    if not reachable:
        # PROVEN unreachable -> the pool is never built, so there is nothing to refuse.
        # The number is still worth having (an operator who later raises --max-num-seqs
        # needs to know what it would cost), so report it under a key that cannot be
        # mistaken for a budget VERDICT. No violation, no override consumed.
        info = {
            "note": ("informational only: no B>1 decode step is reachable in this "
                     "configuration, so MultiSeqKVPool is never allocated and these "
                     "bytes are never spent. " + why_reach),
            "becomes_a_violation_if": ("the engine is relaunched with --max-num-seqs > 1 "
                                       "or OMMX_ATTN_BATCHED is turned on"),
            "num_seqs": int(num_seqs), "num_seqs_origin": origin,
            "num_layers": int(num_layers), "max_seq_len": int(max_seq_len),
            "per_layer_bytes": per_layer_bytes, "per_layer": _gib(per_layer_bytes),
            "total_bytes": total_bytes, "total": _gib(total_bytes),
        }
        free_bytes, why = _free_device_bytes(device)
        if free_bytes is None:
            info["free_bytes"] = "unknown"
            info["free_unavailable_reason"] = why
        else:
            info["free_bytes"] = int(free_bytes)
            info["budget_bytes"] = int(free_bytes * frac)
            info["would_fit_at_budget"] = bool(total_bytes <= int(free_bytes * frac))
        report["pool_projection_informational"] = info
        pool["status"] = ("not-reachable (" + why_reach + "); the projection above is "
                          "INFORMATIONAL, not a budget verdict")
        return

    free_bytes, why = _free_device_bytes(device)
    pool["device"] = str(device) if device is not None else "current"
    pool["free_bytes"] = free_bytes if free_bytes is not None else "unknown"
    if free_bytes is None and why.startswith(_NO_CUDA_PREFIX):
        # NOT a violation. There is no CUDA device on this host at all, so the pool
        # cannot be allocated here either — this process will fail loudly inside torch
        # long before anything OOMs, and a CPU CI run / mocked engine / torch-without-
        # CUDA build must not be refused by a GPU memory guard it can never satisfy.
        # Reported (never guessed at), and the projection above is still in the report.
        pool["free_unavailable_reason"] = why
        pool["status"] = ("skipped: no CUDA device (" + why[len(_NO_CUDA_PREFIX):] +
                          ") — the projected pool was NOT checked against any budget")
        return
    if free_bytes is None:
        # CUDA IS available and the query still failed, or the caller asked about a
        # non-CUDA device while CUDA exists. Both mean something is wrong and we must
        # not guess a budget: unknown never counts as a pass.
        pool["free_unavailable_reason"] = why
        pool["status"] = f"unknown (free memory unreadable: {why})"
        if override:
            report["overrides_applied"].append("OMMX_UNSAFE_ALLOW_POOL_OVERSUBSCRIBE")
            _warn_once("pool_free_unknown",
                       "OMMX_UNSAFE_ALLOW_POOL_OVERSUBSCRIBE=1 -> free device memory is "
                       f"unreadable ({why}); the projected OMMX pool "
                       f"({_gib(total_bytes)} over {num_layers} layers) is NOT being "
                       "checked against it. A B>1 decode step may OOM.")
            return
        violations.append(
            f"OMMX pool budget CANNOT BE CHECKED: free device memory is unreadable "
            f"({why}).\n"
            f"  Projected pool: {_gib(per_layer_bytes)} per layer x {num_layers} layers "
            f"= {_gib(total_bytes)} (num_seqs={num_seqs}, from {origin}; "
            f"max_seq_len={max_seq_len}, Hkv={n_kv_heads}, D={head_dim}), and a B>1 "
            f"decode step IS reachable ({why_reach}).\n"
            "  This is NOT the 'no GPU on this box' case — that one is reported as "
            "'skipped: no CUDA device' and does not fail startup. Here a CUDA device "
            "exists (or was named by the caller) and the query still did not answer, so "
            "the guard has no budget to compare against and refuses to guess one.\n"
            "  FIX: point the backend at a working CUDA device (it is CUDA-only) and "
            "investigate the driver/context error above, or accept the risk with "
            "OMMX_UNSAFE_ALLOW_POOL_OVERSUBSCRIBE=1.")
        return

    budget = int(free_bytes * frac)
    pool["budget_bytes"] = budget
    fits = total_bytes <= budget
    pool["status"] = "ok" if fits else "over-budget"
    if fits:
        return

    # linear in num_seqs (every plane scales with it) -> an exact "what fits" hint.
    max_seqs_fit = int(int(num_seqs) * budget // total_bytes) if total_bytes else 0
    detail = (
        "OMMX KV sidecar pool DOES NOT FIT in device memory.\n"
        f"    num_seqs    = {num_seqs}   (resolved from {origin})\n"
        f"    per-layer   = {_gib(per_layer_bytes)}   "
        f"(num_seqs={num_seqs} x max_seq_len={max_seq_len}, Hkv={n_kv_heads}, "
        f"D={head_dim}, group_tokens={knobs['group_tokens']}, "
        f"page_size={knobs['page_size']}, outliers={knobs['outliers_per_vector']}, "
        f"kv_ring={int(kv_ring)}, kv_outlier_map={int(kv_outlier_map)}, "
        f"kv_int8_scale={int(kv_int8_scale)}; includes the int32 req_to_group/"
        "req_to_token tables)\n"
        f"    layers      = {num_layers}\n"
        f"    total       = {_gib(per_layer_bytes)} x {num_layers} "
        f"= {_gib(total_bytes)}\n"
        f"    free        = {_gib(free_bytes)}   "
        f"budget = {frac:g} x free = {_gib(budget)}   "
        "(OMMX_POOL_BUDGET_FRAC)\n"
        "  WHY: the sidecar is NOT paged — every one of the num_seqs slots is "
        "preallocated at the FULL max_model_len, eagerly, per layer, at the first B>1 "
        "decode step. Nothing shrinks it at runtime.\n"
        f"  REACHABLE: this is a real violation and not a hypothetical one because a "
        f"B>1 decode step CAN occur here ({why_reach}). A configuration that can never "
        "take one never allocates the pool, and this guard reports the same projection "
        "as informational instead of refusing to start.\n"
        "  FIX (three knobs; the bytes are linear in the first two):\n"
        f"    * OMMX_ATTN_MAX_NUM_SEQS={max_seqs_fit if max_seqs_fit > 0 else 1}  "
        f"(currently {num_seqs}; this is the largest value that fits the budget"
        f"{_TOO_SMALL if max_seqs_fit < 1 else ''})\n"
        f"    * --max-model-len <L>  (currently {max_seq_len}; e.g. "
        f"{max(1024, int(max_seq_len) // 4)} cuts the pool ~4x)\n"
        "    * --max-num-seqs 1 (with OMMX_ATTN_BATCHED off) — serving strictly one "
        "request at a time makes a B>1 step unreachable, so the B==1 single-sequence "
        "store is used and this pool is never built at all.\n"
        "  Override (UNSAFE, WILL OOM): OMMX_UNSAFE_ALLOW_POOL_OVERSUBSCRIBE=1")
    if override:
        report["overrides_applied"].append("OMMX_UNSAFE_ALLOW_POOL_OVERSUBSCRIBE")
        _warn_once("pool_oversubscribe",
                   "OMMX_UNSAFE_ALLOW_POOL_OVERSUBSCRIBE=1 -> " + detail)
        return
    violations.append(detail)
