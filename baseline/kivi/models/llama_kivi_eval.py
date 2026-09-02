import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kivi.quant.new_pack import triton_quantize_and_pack_along_last_dim, unpack_and_dequant_kcache, unpack_and_dequant_vcache, unpack_and_dequant_triton_packed
from kivi.quant.matmul import cuda_bmm_fA_qB_outer, triton_bmm_fA_qB_outer

from transformers.models.llama.configuration_llama import *
from transformers.models.llama.modeling_llama import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.cache_utils import Cache

_CONFIG_FOR_DOC = "LlamaConfig"


def _normalize_legacy_cache(past_key_values):
    if isinstance(past_key_values, Cache):
        legacy_cache = past_key_values.to_legacy_cache()
        return legacy_cache if len(legacy_cache) > 0 else None
    return past_key_values


def _has_non_finite(tensor: Optional[torch.Tensor]) -> bool:
    return tensor is not None and not torch.isfinite(tensor).all()


import os as _os
import sys as _sys
import traceback as _traceback

# --- GQA extension for the FUSED KIVI quantized-matmul kernel (KIVI_FUSED_GQA=1) ---------
# The upstream KIVI fused kernel `triton_bmm_fA_qB_outer` (or `cuda_bmm_fA_qB_outer`, which
# on sm>=9 delegates to it) fuses dequant+matmul on the packed int cache — but it requires
# the query and the packed key/value to have the SAME head count. Under GQA the query has
# H heads and the KV cache KH=H/n_rep, so a naive fused call shape-mismatches, and the
# obvious `repeat_kv_quant` fix produces non-contiguous strides that the kernel's internal
# .view()/.reshape() reject. Fix: expand the PACKED head dim KH->H *contiguously*, then the
# fused kernel runs (no fp16 KV materialize -> the real KIVI perf/memory path).
#
# Kernel shape contract, per call site (audited against kivi/quant/matmul.py):
#   triton_bmm_fA_qB_outer(fA=[B,H,M,K], qB=[B,H,K,N//feat_per_int]) -> [B,H,M,N]
#   * KEY call:   fA=query [B,H,q_len,head_dim] -> kernel K=head_dim(128),
#                 N=#quantized key tokens. Keys are only ever quantized in whole
#                 residual_length(=128) blocks, so N is a multiple of 128 and the kernel's
#                 `N % 64 == 0` assert holds.
#   * VALUE call: fA=attn_weights[..., :T_vq] -> kernel K=T_vq (grows by 1 per decode step,
#                 NOT a multiple of 64) and N=head_dim(128), so again N%64 holds. The K axis
#                 has no assert and does not need one: the kernel masks both of its K-axis
#                 loads (`offs_ak < K` / `offs_bk < K`, other=0), so a ragged K is exact.
#   * group_size(=32) satisfies the kernel's `group_size % 32 == 0` (== BLOCK_SIZE_N).
#
# PATH ACCOUNTING (added after the fair-comparison audit). The fused branch below is
# DEFAULT-OFF, so a plain run measures the dequant->fp16 + repeat_kv_quant + dense-matmul
# fallback, which materialises the whole dequantised KV expanded n_rep-fold on EVERY decode
# step of EVERY layer (n_rep=4 for Llama-3.1-8B: 32 q heads / 8 kv heads). Whether that is a
# handicap is a MEASURED question, and the measurement says "only at short context". H100 NVL,
# Llama-3.1-8B, both paths prewarmed and every cell taken in both arm orders (fused_fires
# proves which path ran), decode TPOT in ms:
#     ctx=1024    fallback 67.98 / 70.88  ->  fused 60.24 / 59.27   fused/fallback 0.861
#     ctx=4096    fallback 66.63 / 68.33  ->  fused 106.24 / 107.55 fused/fallback 1.584
#     ctx=16384   fallback 144.10/130.03  ->  fused 360.51 / 338.76 fused/fallback 2.551
# So the fused kernel is 1.16x FASTER at 1K and 1.58x / 2.55x SLOWER at 4K / 16K. The
# default-off fallback is therefore the BETTER KIVI at every context this repo publishes
# (>= 4096) - it is not a handicap there, and no text in this file may imply otherwise.
# (An earlier un-prewarmed sweep reported 1.44x for fused at ctx=1024; that cell was
# contaminated by a 57.7 s first-arm triton JIT and is retracted.) What must never again be
# invisible is WHICH path ran, not which one is assumed to be faster:
#   * the default stays OFF, so previously published KIVI numbers remain reproducible;
#   * first use prints a one-time stderr notice naming the path that is actually running and
#     the env var that switches it (silence with KIVI_QUIET=1);
#   * fused fires and fallback fires are counted separately, and any exception that demotes
#     the fused branch to the slow path is RECORDED (site/type/message/frame) instead of
#     being swallowed - `kivi_fused_stats()` is the machine-readable evidence, and
#     KIVI_FUSED_STRICT=1 still re-raises instead of falling back.
#   * the gate covers ONLY the two decode/short-prefill helpers below. Long prefill
#     (q_len > _PREFILL_CHUNK_THRESHOLD) goes through _chunked_prefill_attention, which
#     dequantizes + repeat_kv_quant + dense-matmuls unconditionally and has no fused variant.
#     Those calls are counted too (_kivi_chunked_prefill_fires, folded into fallback_fires),
#     otherwise a 32K/64K run could report path=="fused" with fallback_fires==0 while its
#     whole prefill ran the handicapped path - exactly the invisible handicap this
#     accounting exists to prevent. kivi_fused_stats() derives its `path` label from that
#     counter EXPLICITLY as well, so the label cannot regress to "fused" if the fold into
#     fallback_fires is ever removed.
#   * WHAT THIS MEANS FOR A PUBLISHED fused-vs-fallback COMPARISON: the PREFILL is the same
#     unfused code in both modes. With use_flash=False and q_len > 4096 it is
#     _chunked_prefill_attention (dequant -> fp16 -> dense); with use_flash=True (what
#     figure/bench.py and eval/lm_eval/models/kivi_model.py both set) it is flash_attn on the
#     un-quantised prefill tensors. Neither reads KIVI_FUSED_GQA. So TTFT is identical BY
#     CONSTRUCTION and the ONLY thing the flag can move is DECODE (TPOT): a "fused KIVI is
#     N x" number at 4K/8K/16K/32K is a decode-only number and must never be quoted as an
#     end-to-end speedup. That is exactly the regime the published contexts live in.
#   * both quantized-KV matmul paths are fp16-ONLY - the dequant fallback's
#     unpack_and_dequant_triton_packed hard-casts its output to torch.float16
#     (kivi/quant/new_pack.py:110) and the fused kernel allocates a torch.float16 output
#     (kivi/quant/matmul.py:254). A bfloat16 checkpoint is rejected up front by
#     _kivi_check_fp16 rather than mixing dtypes or dying inside triton, and nothing is cast
#     on the caller's behalf: a silent cast would change the numerics being measured.
_kivi_fused_fires = [0]         # fused dequant+matmul kernel invocations (list: legacy shape)
_kivi_fallback_fires = [0]      # dequant->fp16 + repeat + dense matmul invocations
_kivi_chunked_prefill_fires = [0]  # _chunked_prefill_attention dequant+dense calls (see below)
_kivi_direct_fused_fires = [0]  # UNGATED upstream fused-kernel calls (cuda_bmm_fA_qB_outer
                                # invoked directly, not through _fused_bmm_padM). None exist in
                                # this file; the Mistral sibling has two, and it shares these
                                # counters, so the `path` label must be able to see them -
                                # otherwise a Mistral use_flash=True run reports "unused" while
                                # a fused kernel is firing on every decode step.
_kivi_fused_errors = []         # deduped [{site,type,msg,where,count}] fused->fallback demotions
_kivi_fused_errors_total = [0]  # total demotions, including any dropped by the dedupe cap
_kivi_fused_errors_capped = [False]
_KIVI_MAX_RECORDED_ERRORS = 16  # 32 layers x N steps must not grow the record without bound
_kivi_notice_done = [False]     # one-time "which path am I on" stderr notice
_kivi_direct_notice_done = [False]  # one-time notice for the UNGATED fused call sites
_KIVI_KERNEL_BLOCK_M = 32       # must match BLOCK_SIZE_M in matmul.py::triton_bmm_fA_qB_outer


def _kivi_stderr(msg):
    """Diagnostics go to stderr only: the benchmark harnesses parse stdout."""
    _sys.stderr.write(msg if msg.endswith("\n") else msg + "\n")
    _sys.stderr.flush()


_KIVI_ENV_FALSEY = ("0", "", "false", "off", "no")


def _kivi_env_flag(name):
    """The single truthiness rule for every KIVI_* gate in this file: "0", "", "false", "off"
    and "no" (any case, surrounding whitespace ignored) mean OFF; anything else means ON.

    BEHAVIOUR CHANGE (wave 2), deliberate: KIVI_FUSED_STRICT used to be read as
    bool(_os.environ.get("KIVI_FUSED_STRICT")), and bool("0") is True - so `KIVI_FUSED_STRICT=0`,
    the obvious way to spell "off", actually armed strict mode and turned any fused-path
    demotion into a hard abort. kivi_fused_stats()["strict"] used the same expression and so
    faithfully reported strict=True for a run the user had switched off. Both now go through
    this helper, so the gate and the reported gate state agree with each other and with
    KIVI_FUSED_GQA / KIVI_QUIET. The change can only turn a spurious raise into normal
    (recorded, warned, demoted) operation - never a raise into a silent pass, because
    "1"/"true"/"on"/anything-else still mean strict."""
    return _os.environ.get(name, "0").strip().lower() not in _KIVI_ENV_FALSEY


def _kivi_quiet():
    """KIVI_QUIET=1 silences the informational path notice. It deliberately does NOT
    silence fused-path error warnings: a quiet run may hide a notice, never a failure."""
    return _kivi_env_flag("KIVI_QUIET")


def _kivi_path_notice_once(n_rep):
    """One-time stderr notice stating which quantized-KV matmul path this process runs.

    Emitted on first *use* (not at import), because the gate is an env var that a caller may
    set after importing. It is informational only - it changes no behaviour."""
    if _kivi_notice_done[0]:
        return
    _kivi_notice_done[0] = True
    if _kivi_quiet():
        return
    if _fused_gqa_enabled():
        _kivi_stderr(
            "[kivi] quantized-KV matmul path: FUSED (KIVI_FUSED_GQA=%r).\n"
            "[kivi]   Packed-int dequant+matmul kernel; no fp16 KV materialize. GQA heads are\n"
            "[kivi]   expanded on the PACKED cache (n_rep=%d).\n"
            "[kivi]   Demotions back to the dequant fallback are counted and recorded:\n"
            "[kivi]   kivi_fused_stats(). KIVI_FUSED_STRICT=1 turns a demotion into a raise.\n"
            "[kivi]   Silence this notice with KIVI_QUIET=1."
            % (_os.environ.get("KIVI_FUSED_GQA"), n_rep)
        )
    else:
        _kivi_stderr(
            "[kivi] quantized-KV matmul path: DEQUANT-FALLBACK (KIVI_FUSED_GQA is off).\n"
            "[kivi]   Every step dequantizes the packed low-bit KV to fp16 and expands it\n"
            "[kivi]   n_rep=%d-fold (GQA) before a dense matmul. This is NOT the fused KIVI\n"
            "[kivi]   kernel: it is more memory-hungry than KIVI can be. It is NOT uniformly\n"
            "[kivi]   slower - measured H100 decode TPOT has the fused kernel 1.16x faster at\n"
            "[kivi]   ctx=1024 but 1.58x / 2.55x SLOWER at ctx=4096 / 16384, so at every\n"
            "[kivi]   context this repo publishes THIS path is the faster one.\n"
            "[kivi]   A fused GQA path lives in this same file (_repeat_head_contig +\n"
            "[kivi]   _fused_bmm_padM); enable it with KIVI_FUSED_GQA=1 (add KIVI_FUSED_STRICT=1\n"
            "[kivi]   to make a fused failure raise instead of demoting to this path).\n"
            "[kivi]   Path evidence at runtime: kivi_fused_stats(). Silence: KIVI_QUIET=1."
            % (n_rep,)
        )


def _kivi_record_fused_error(site, exc):
    """Record a fused->fallback demotion instead of swallowing it (no silent fallback).

    Deduped on (site, type, message) so a 32-layer x N-step run cannot spam either the log
    or the record; one stderr warning per unique entry, counts kept for all of them."""
    _kivi_fused_errors_total[0] += 1
    frames = _traceback.extract_tb(exc.__traceback__)
    where = ("%s:%d in %s" % (frames[-1].filename, frames[-1].lineno, frames[-1].name)
             if frames else "<no traceback>")
    etype, emsg = type(exc).__name__, str(exc)
    for rec in _kivi_fused_errors:
        if rec["site"] == site and rec["type"] == etype and rec["msg"] == emsg:
            rec["count"] += 1
            return
    if len(_kivi_fused_errors) < _KIVI_MAX_RECORDED_ERRORS:
        _kivi_fused_errors.append(
            {"site": site, "type": etype, "msg": emsg, "where": where, "count": 1})
        # Under KIVI_FUSED_STRICT the caller re-raises right after this, so do not tell the
        # user a fallback number is coming - nothing is measured at all in that case.
        outcome = ("and KIVI_FUSED_STRICT is set, so this run is about to abort."
                   if _kivi_fused_strict() else
                   "and was demoted to the dequant+dense fallback - the number you are\n"
                   "[kivi]   about to measure is the UNFUSED path, not the one you asked for.\n"
                   "[kivi]   (Unfused is not automatically slower: measured H100 decode TPOT\n"
                   "[kivi]   makes it 1.16x slower at ctx=1024 but 1.58x/2.55x FASTER at\n"
                   "[kivi]   ctx=4096/16384. Either way the label must match the kernel.)")
        _kivi_stderr(
            "[kivi] WARNING: fused GQA path failed %s\n"
            "[kivi]   site=%s  %s: %s\n"
            "[kivi]   at %s\n"
            "[kivi]   Without KIVI_FUSED_STRICT this demotes silently to the unfused path.\n"
            "[kivi]   Full record: kivi_fused_stats()."
            % (outcome, site, etype, emsg, where))
    elif not _kivi_fused_errors_capped[0]:
        _kivi_fused_errors_capped[0] = True
        _kivi_stderr(
            "[kivi] WARNING: fused-path error record capped at %d unique entries; further\n"
            "[kivi]   distinct failures are counted in errors_total only."
            % (_KIVI_MAX_RECORDED_ERRORS,))


def kivi_fused_stats():
    """Which quantized-KV matmul path actually ran, as machine-readable evidence.

    Returns a dict:
      enabled        bool      - KIVI_FUSED_GQA gate state, re-read at call time
      strict         bool      - KIVI_FUSED_STRICT (a fused failure raises instead of demoting)
      gate_env       str|None  - raw KIVI_FUSED_GQA value, for provenance capture
      fused_fires    int       - fused dequant+matmul kernel calls made through the gate
      direct_fused_fires int   - UNGATED fused-kernel calls (the Mistral sibling's use_flash
                                 attention calls cuda_bmm_fA_qB_outer directly); they count as
                                 fused for `path` but are kept separate so the gate's own
                                 effect stays readable
      fallback_fires int       - dequant->fp16 + repeat + dense matmul calls (all sites)
      chunked_prefill_fires int - the subset of those that came from the long-prefill path,
                                  which has NO fused variant at all (see PATH ACCOUNTING)
      path           str       - "fused" | "dequant-fallback" | "mixed" | "unused"
      errors         list      - [{site,type,msg,where,count}] fused->fallback demotions
      errors_total   int       - total demotions (>= sum of counts if the record was capped)

    `path == "mixed"` means some layers/steps ran fused and some did not: any published
    latency from such a run is a blend of two kernels and must not be quoted as either. A run
    with chunked_prefill_fires > 0 can therefore never read "fused", which is correct: the long
    prefill has no fused variant, so such a run IS a blend (and its TTFT is the unfused one -
    see PATH ACCOUNTING)."""
    fused = _kivi_fused_fires[0]
    direct = _kivi_direct_fused_fires[0]
    fallback = _kivi_fallback_fires[0]
    chunked = _kivi_chunked_prefill_fires[0]
    # `chunked` is already folded into `fallback` at the call site, so `fallback or chunked` is
    # redundant TODAY. It is written out anyway: the whole point of this label is that an
    # unfused long prefill can never be reported as "fused", and that guarantee must not depend
    # on a double-count in a different function that a later edit could quietly drop.
    any_fused = fused + direct
    any_unfused = fallback or chunked
    if any_fused and any_unfused:
        path = "mixed"
    elif any_fused:
        path = "fused"
    elif any_unfused:
        path = "dequant-fallback"
    else:
        path = "unused"
    return {
        "enabled": _fused_gqa_enabled(),
        "strict": _kivi_fused_strict(),
        "gate_env": _os.environ.get("KIVI_FUSED_GQA"),
        "fused_fires": fused,
        "direct_fused_fires": direct,
        "fallback_fires": fallback,
        "chunked_prefill_fires": chunked,
        "path": path,
        "errors": [dict(rec) for rec in _kivi_fused_errors],
        "errors_total": _kivi_fused_errors_total[0],
    }


def kivi_reset_fused_stats():
    """Zero the counters/records (e.g. after warmup, before the measured window). The
    one-time path notice is NOT re-armed: it describes the process, not the window."""
    _kivi_fused_fires[0] = 0
    _kivi_direct_fused_fires[0] = 0
    _kivi_fallback_fires[0] = 0
    _kivi_chunked_prefill_fires[0] = 0
    del _kivi_fused_errors[:]
    _kivi_fused_errors_total[0] = 0
    _kivi_fused_errors_capped[0] = False


def _repeat_head_contig(x, n_rep):
    """[B, KH, *rest] -> [B, KH*n_rep, *rest], contiguous (valid strides for the fused kernel).

    Head ordering is torch.repeat_interleave(dim=1) semantics - each KV head repeated n_rep
    times CONSECUTIVELY (out head h reads in head h // n_rep), identical to repeat_kv /
    repeat_kv_quant below. A tiling order (h % KH) would pair every query head with the wrong
    KV head and no shape assert would catch it, so this is proven by
    _selfcheck_repeat_head_contig()."""
    if n_rep == 1:
        return x.contiguous()
    B, KH = x.shape[0], x.shape[1]
    rest = x.shape[2:]
    return x[:, :, None].expand(B, KH, n_rep, *rest).reshape(B, KH * n_rep, *rest).contiguous()


def _fused_bmm_padM(group_size, fA, qB, sc, zp, bits, BM=_KIVI_KERNEL_BLOCK_M):
    """qbmm_kernel assumes M is a multiple of BLOCK_SIZE_M(=32); decode has M=1, which would
    read rows 1..31 of the A operand out of bounds: matmul.py:66 builds a_ptrs with
    offs_am[:, None] over a full BLOCK_SIZE_M tile and matmul.py:84 loads it with mask
    `offs_ak < K` only - the K axis is masked, the M axis of that load is not. (The *store*
    at matmul.py:109-111 is M-masked, `offs_cm < M`, so the risk is an OOB read, not a
    corrupted output.) Pad the query rows up to a multiple of 32 (zeros), run the fused
    dequant+matmul, slice back. This is the missing piece that lets the fused KIVI kernel run
    on the GQA single-token decode.

    The padding cannot pollute the result: the kernel is row-wise in M (accumulator is
    (BLOCK_SIZE_M, BLOCK_SIZE_N) and tl.dot keeps row m dependent on A row m only), the
    appended rows are exact zeros, and they are sliced off. Proven by
    _selfcheck_fused_bmm_padM(). BM must track
    BLOCK_SIZE_M in kivi/quant/matmul.py::triton_bmm_fA_qB_outer, which hardcodes 32 on both
    of its config branches."""
    B, H, M, K = fA.shape
    r = M % BM
    if r != 0:
        fA = torch.cat([fA, fA.new_zeros(B, H, BM - r, K)], dim=2)
    out = triton_bmm_fA_qB_outer(group_size, fA.contiguous(), qB, sc, zp, bits)
    return out[:, :, :M, :]


def _fused_gqa_enabled():
    return _kivi_env_flag("KIVI_FUSED_GQA")


def _kivi_fused_strict():
    """KIVI_FUSED_STRICT=1: a fused->fallback demotion re-raises instead of demoting, so a
    latency can never be measured on a path the caller did not ask for. See _kivi_env_flag for
    the "0 now means off" fix."""
    return _kivi_env_flag("KIVI_FUSED_STRICT")


def _kivi_check_fp16(site, **tensors):
    """Reject a non-fp16 model AT THE BOUNDARY, loudly, instead of casting it or dying deep.

    Both quantized-KV matmul paths in this file are float16-only, and neither of them says so:
      * the dequant fallback calls kivi/quant/new_pack.py::unpack_and_dequant_triton_packed,
        which hard-casts with `data = data.to(torch.float16)` (new_pack.py:110). A bfloat16
        query therefore meets an fp16 key inside torch.matmul and dies with a dtype error
        raised from a line that mentions neither KIVI nor the constraint;
      * the fused path calls kivi/quant/matmul.py::triton_bmm_fA_qB_outer, which allocates its
        output as `torch.empty(..., dtype=torch.float16)` (matmul.py:254) and whose kernel
        finishes with `accumulator.to(tl.float16)` (matmul.py:105) - a bfloat16 operand either
        faults inside triton or produces an fp16 buffer the caller goes on to read as bf16.
    Every other model path in this repo runs bf16 by default, so this is a foot-gun that a
    benchmark hits the first time KIVI is pointed at a non-fp16 checkpoint.

    Note which tensors are checked: the packed cache itself is int32, so its FLOAT dtype lives
    in the scale/zero-point tensors (triton_quantize_and_pack_along_last_dim allocates them
    with dtype=data.dtype, new_pack.py:257-258). Checking query/attn-weights plus the scales
    is what actually catches a bf16 checkpoint.

    This deliberately does NOT cast. A silent cast would change the numerics under measurement
    and hide the fact that KIVI was never evaluated in the dtype that was asked for; the caller
    must load the model with torch_dtype=torch.float16, which is KIVI's own published recipe."""
    for name, tensor in tensors.items():
        if tensor is not None and tensor.dtype != torch.float16:
            raise TypeError(
                "[kivi] %s: the quantized-KV matmul path is float16-ONLY, but %s is %s. "
                "unpack_and_dequant_triton_packed() hard-casts its output to torch.float16 "
                "(kivi/quant/new_pack.py:110) and triton_bmm_fA_qB_outer() allocates a "
                "torch.float16 output (kivi/quant/matmul.py:254), so neither the fused nor the "
                "dequant path can serve %s. Load the model with torch_dtype=torch.float16 "
                "(KIVI's published recipe). No cast is applied on your behalf - that would "
                "silently change the numerics being measured."
                % (site, name, tensor.dtype, tensor.dtype))


def _kivi_direct_fused_notice_once(where):
    """One-time notice for an UNGATED fused-kernel call site (see _kivi_direct_fused_fires).

    It cannot reuse _kivi_path_notice_once: that notice reports the KIVI_FUSED_GQA gate, and an
    ungated site runs the fused kernel whatever the gate says - printing "DEQUANT-FALLBACK"
    there would be an outright false statement about which kernel is running."""
    if _kivi_direct_notice_done[0]:
        return
    _kivi_direct_notice_done[0] = True
    if _kivi_quiet():
        return
    _kivi_stderr(
        "[kivi] quantized-KV matmul path: FUSED (UNGATED) at %s.\n"
        "[kivi]   This call site invokes the upstream cuda_bmm_fA_qB_outer directly and does\n"
        "[kivi]   NOT read KIVI_FUSED_GQA, so the gate does not describe it. Counted as\n"
        "[kivi]   direct_fused_fires in kivi_fused_stats(). Silence this notice with\n"
        "[kivi]   KIVI_QUIET=1." % (where,))


def _fallback_key_quant_matmul(query_states, key_states_quant_trans, key_scale_trans, key_mn_trans, group_size, bits, n_rep):
    # fp16 gate first, before any work and before the notice: both branches below are fp16-only.
    _kivi_check_fp16("_fallback_key_quant_matmul",
                     query_states=query_states, key_scale_trans=key_scale_trans,
                     key_mn_trans=key_mn_trans)
    _kivi_path_notice_once(n_rep)
    if _fused_gqa_enabled():
        try:
            qk = _repeat_head_contig(key_states_quant_trans, n_rep)   # [B,H,D,T//feat]
            sc = _repeat_head_contig(key_scale_trans, n_rep)          # [B,H,D,G]
            zp = _repeat_head_contig(key_mn_trans, n_rep)             # [B,H,D,G]
            out = _fused_bmm_padM(group_size, query_states, qk, sc, zp, bits)
            _kivi_fused_fires[0] += 1
            return out
        except Exception as exc:
            # No silent fallback: record type/message/frame (and warn once) before demoting.
            _kivi_record_fused_error("_fallback_key_quant_matmul", exc)
            if _kivi_fused_strict():
                raise
    # Fallback: dequantize to fp16, expand, dense matmul (works for any GQA layout).
    _kivi_fallback_fires[0] += 1
    key_states_trans = unpack_and_dequant_triton_packed(key_states_quant_trans, key_scale_trans, key_mn_trans, group_size, bits)
    key_states_trans = repeat_kv_quant(key_states_trans, n_rep)
    return torch.matmul(query_states, key_states_trans)


def _fallback_value_quant_matmul(attn_weights, value_states_quant, value_scale, value_mn, group_size, bits, n_rep):
    # fp16 gate first: attn_weights carries the model dtype (softmax is cast back to
    # query_states.dtype at every call site), value_scale/value_mn carry the cache dtype.
    _kivi_check_fp16("_fallback_value_quant_matmul",
                     attn_weights=attn_weights, value_scale=value_scale, value_mn=value_mn)
    _kivi_path_notice_once(n_rep)
    if _fused_gqa_enabled():
        try:
            vq = _repeat_head_contig(value_states_quant, n_rep)       # [B,H,T_v,D//feat]
            sc = _repeat_head_contig(value_scale, n_rep)             # [B,H,T_v,G]
            zp = _repeat_head_contig(value_mn, n_rep)                # [B,H,T_v,G]
            out = _fused_bmm_padM(group_size, attn_weights, vq, sc, zp, bits)
            _kivi_fused_fires[0] += 1
            return out
        except Exception as exc:
            # No silent fallback: record type/message/frame (and warn once) before demoting.
            _kivi_record_fused_error("_fallback_value_quant_matmul", exc)
            if _kivi_fused_strict():
                raise
    _kivi_fallback_fires[0] += 1
    value_states = unpack_and_dequant_triton_packed(value_states_quant, value_scale, value_mn, group_size, bits)
    value_states = repeat_kv_quant(value_states, n_rep)
    return torch.matmul(attn_weights, value_states)



# Chunked prefill attention: avoids materialising the full [B, H, S, S] attention matrix.
# For a 32K sequence with H=32 heads the full matrix is ~68 GB (FP16); processing in
# query-direction chunks of PREFILL_CHUNK_SIZE keeps the peak per-step allocation to ~1 GB.
_PREFILL_CHUNK_THRESHOLD = 4096   # Use chunked path when prefill seq_len exceeds this
_PREFILL_CHUNK_SIZE       = 512   # Query tokens processed per chunk


def _chunked_prefill_attention(
    query_states,            # [B, H, S, D]
    key_states_quant_trans,  # [B, KH, D, T_k_quant/pack] or None  (K transposed then packed)
    key_scale_trans,         # [B, KH, D, num_groups] or None
    key_mn_trans,            # [B, KH, D, num_groups] or None
    key_states_full,         # [B, KH, T_k_full, D] or None        (residual, not quantised)
    value_states_quant,      # [B, KH, T_v_quant, D/pack] or None  (V packed, NOT transposed)
    value_scale,             # [B, KH, T_v_quant, num_groups_D] or None
    value_mn,                # same shape or None
    value_states_full,       # [B, KH, T_v_full, D]
    group_size, k_bits, v_bits, n_rep,
    softmax_scale,           # math.sqrt(head_dim)
    attention_mask,          # [B, 1, S, kv_len] or None
):
    B, H, S, D = query_states.shape
    dtype = query_states.dtype

    # This path has NO fused variant: it always dequantizes the packed KV to fp16 and expands
    # it n_rep-fold before a dense matmul, whatever KIVI_FUSED_GQA says. Count and announce it
    # so a long-context run cannot report path=="fused" while its prefill ran the unfused path
    # (which at these contexts is also the FASTER path - see the measured table in PATH
    # ACCOUNTING; "unfused" is a provenance statement here, never a performance one).
    #
    # State it plainly, because this is the regime the published contexts live in: THE LONG
    # PREFILL IS UNFUSED IN BOTH MODES. Flipping KIVI_FUSED_GQA does not change a single
    # instruction executed here (with use_flash=True - what figure/bench.py and
    # eval/lm_eval/models/kivi_model.py set - prefill does not even reach this function; it
    # runs flash_attn on the un-quantised tensors, equally gate-independent). Therefore TTFT is
    # identical BY CONSTRUCTION between the two modes, and a fused-vs-fallback comparison at
    # ctx > _PREFILL_CHUNK_THRESHOLD is a DECODE-ONLY comparison. Report such a ratio as TPOT,
    # never as an end-to-end or TTFT speedup.
    if key_states_quant_trans is not None or value_states_quant is not None:
        # fp16 gate: unpack_and_dequant_triton_packed below returns fp16 unconditionally, so a
        # bf16 query would fail inside torch.matmul rather than here (see _kivi_check_fp16).
        _kivi_check_fp16("_chunked_prefill_attention",
                         query_states=query_states, key_scale_trans=key_scale_trans,
                         value_scale=value_scale)
        _kivi_path_notice_once(n_rep)
        _kivi_chunked_prefill_fires[0] += 1
        _kivi_fallback_fires[0] += 1

    # --- unpack quantised key (transposed form: [B, KH, D, T_k_quant]) ---
    if key_states_quant_trans is not None:
        key_q = unpack_and_dequant_triton_packed(
            key_states_quant_trans, key_scale_trans, key_mn_trans, group_size, k_bits
        )                                          # [B, KH, D, T_k_quant]
        key_q = repeat_kv_quant(key_q, n_rep)     # [B, H,  D, T_k_quant]
    else:
        key_q = None

    # --- prepare full key (transpose for matmul) ---
    if key_states_full is not None:
        key_f_t = repeat_kv(key_states_full, n_rep).transpose(-2, -1)  # [B, H, D, T_k_full]
    else:
        key_f_t = None

    # --- unpack quantised value ---
    if value_states_quant is not None:
        val_q = unpack_and_dequant_triton_packed(
            value_states_quant, value_scale, value_mn, group_size, v_bits
        )                                         # [B, KH, T_v_quant, D]
        val_q = repeat_kv_quant(val_q, n_rep)    # [B, H,  T_v_quant, D]
    else:
        val_q = None

    # --- prepare full value ---
    val_f     = repeat_kv(value_states_full, n_rep)  # [B, H, T_v_full, D]
    val_f_len = val_f.shape[-2]

    output = torch.zeros(B, H, S, D, device=query_states.device, dtype=dtype)

    for start in range(0, S, _PREFILL_CHUNK_SIZE):
        end = min(start + _PREFILL_CHUNK_SIZE, S)
        q = query_states[:, :, start:end, :]          # [B, H, C, D]

        # key attention logits
        parts = []
        if key_q is not None:
            parts.append(torch.matmul(q, key_q))       # [B, H, C, T_k_quant]
        if key_f_t is not None:
            parts.append(torch.matmul(q, key_f_t))     # [B, H, C, T_k_full]
        attn_logits = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        attn_logits = attn_logits / softmax_scale      # [B, H, C, kv_len]

        kv_len = attn_logits.shape[-1]

        # causal mask: query at position (start+i) may only attend to key positions <= (start+i)
        pos_q = torch.arange(start, end,   device=q.device).unsqueeze(1)  # [C, 1]
        pos_k = torch.arange(kv_len,       device=q.device).unsqueeze(0)  # [1, kv_len]
        attn_logits.masked_fill_(
            (pos_k > pos_q).unsqueeze(0).unsqueeze(0),
            torch.finfo(attn_logits.dtype).min,
        )

        # optional external mask (e.g. padding)
        if attention_mask is not None:
            attn_logits = attn_logits + attention_mask[:, :, start:end, :]

        attn_w = F.softmax(attn_logits.float(), dim=-1).to(dtype)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)

        # value weighted sum
        if val_q is not None:
            out = torch.matmul(attn_w[:, :, :, :-val_f_len], val_q)
            out = out + torch.matmul(attn_w[:, :, :, -val_f_len:], val_f)
        else:
            out = torch.matmul(attn_w, val_f)

        output[:, :, start:end, :] = out
        del attn_logits, attn_w, q, out, parts

    return output


def repeat_kv_quant(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention_KIVI_eval(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.k_bits = config.k_bits
        self.v_bits = config.v_bits
        self.group_size = config.group_size
        self.residual_length = config.residual_length
        #print(f"k_bits: {self.k_bits}, v_bits: {self.v_bits}, group_size: {self.group_size}, residual_length: {self.residual_length}")

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling.get("type", self.config.rope_scaling.get("rope_type"))
            scaling_factor = self.config.rope_scaling.get("factor", 1.0)
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                # Newer Llama-3.x configs use rope_type=llama3 with extra fields.
                self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # [bsz, nh, t, hd]
        if past_key_value is not None:
            key_states_quant_trans = past_key_value[0]
            key_states_full = past_key_value[1]
            key_scale_trans = past_key_value[2]
            key_mn_trans = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full = past_key_value[5]
            value_scale = past_key_value[6]
            value_mn = past_key_value[7]

            if key_states_quant_trans is not None:
                att_qkquant = _fallback_key_quant_matmul(
                    query_states,
                    key_states_quant_trans,
                    key_scale_trans,
                    key_mn_trans,
                    self.group_size,
                    self.k_bits,
                    self.num_key_value_groups,
                )
            else:
                att_qkquant = None

            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states

            key_states_full_repeat = repeat_kv(key_states_full, self.num_key_value_groups)
            att_qkfull = torch.matmul(query_states, key_states_full_repeat.transpose(2, 3))

            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), 
                                                                                                                            self.group_size, 
                                                                                                                            self.k_bits)
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]
            if value_states_quant is None:
                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output = torch.matmul(attn_weights, value_states_full_repeat) # value_states_full need to be repeated
            else:
                attn_output = _fallback_value_quant_matmul(
                    attn_weights[:, :, :, :-value_full_length],
                    value_states_quant,
                    value_scale,
                    value_mn,
                    self.group_size,
                    self.v_bits,
                    self.num_key_value_groups,
                )

                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_states_full_repeat)

            if value_full_length > self.residual_length:
                assert value_full_length == self.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn

        else:
            # Quantize first
            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None

            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(
                                                                    key_states_quant.transpose(2, 3).contiguous(),
                                                                    self.group_size,
                                                                    self.k_bits)
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
            
            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, 
                                                                                                self.group_size, 
                                                                                                self.v_bits)

            # print("key_scale_trans", key_scale_trans)
            # print("key_mn_trans", key_mn_trans)
            # print("value_scale", value_scale)
            # print("value_mn", value_mn)

            if q_len > _PREFILL_CHUNK_THRESHOLD:
                # Memory-efficient chunked attention: avoids [B,H,S,S] materialisation for
                # long sequences (e.g. 32K tokens → ~68 GB in FP16 with 32 heads).
                attn_output = _chunked_prefill_attention(
                    query_states,
                    key_states_quant_trans, key_scale_trans, key_mn_trans,
                    key_states_full,
                    value_states_quant, value_scale, value_mn,
                    value_states_full,
                    self.group_size, self.k_bits, self.v_bits,
                    self.num_key_value_groups,
                    math.sqrt(self.head_dim),
                    attention_mask,
                )
            else:
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                # Calculate attn_weights
                if key_states_quant_trans is not None:
                    att_qkquant = _fallback_key_quant_matmul(
                        query_states,
                        key_states_quant_trans,
                        key_scale_trans,
                        key_mn_trans,
                        self.group_size,
                        self.k_bits,
                        self.num_key_value_groups,
                    )
                else:
                    att_qkquant = None

                if key_states_full is not None:
                    key_states_full_repeat = repeat_kv(key_states_full, self.num_key_value_groups)
                    att_qkfull = torch.matmul(query_states, key_states_full_repeat.transpose(2, 3))
                    if att_qkquant is not None:
                        attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
                    else:
                        attn_weights = att_qkfull / math.sqrt(self.head_dim)
                else:
                    attn_weights = att_qkquant / math.sqrt(self.head_dim)

                if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                        f" {attn_weights.size()}"
                    )

                if attention_mask is not None:
                    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                        )
                    attn_weights = attn_weights + attention_mask
                    attn_weights = torch.max(
                        attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                    )

                attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype).contiguous()

                value_states_full_length = value_states_full.shape[-2]
                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                if value_states_quant is None:
                    attn_output = torch.matmul(attn_weights, value_states_full_repeat)
                else:
                    attn_output = _fallback_value_quant_matmul(
                        attn_weights[:, :, :, :-value_states_full_length],
                        value_states_quant,
                        value_scale,
                        value_mn,
                        self.group_size,
                        self.v_bits,
                        self.num_key_value_groups,
                    )
                    attn_output += torch.matmul(attn_weights[:, :, :, -value_states_full_length:], value_states_full_repeat)
            
        past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len) if use_cache else None
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        attn_weights = None
        return attn_output, attn_weights, past_key_value
    
class LlamaFlashAttention_KIVI_eval(LlamaAttention_KIVI_eval):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]
        if past_key_value is not None:
            key_states_quant_trans = past_key_value[0]
            key_states_full = past_key_value[1]
            key_scale_trans = past_key_value[2]
            key_mn_trans = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full = past_key_value[5]
            value_scale = past_key_value[6]
            value_mn = past_key_value[7]
 
            if key_states_quant_trans is not None:
                att_qkquant = _fallback_key_quant_matmul(query_states, key_states_quant_trans, 
                                 key_scale_trans, key_mn_trans, self.group_size, self.k_bits, self.num_key_value_groups)
                # att_qkquant_ref = triton_bmm_fA_qB_outer(self.group_size, query_states, key_states_quant_trans, 
                #                 key_scale_trans, key_mn_trans, self.k_bits)
                # error = torch.abs(att_qkquant - att_qkquant_ref).float()
                # rel_error = torch.mean(error / (torch.abs(att_qkquant_ref).float()+1e-5))
                # print(f"rel error: {rel_error}")
            else:
                att_qkquant = None
            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states
            key_states_full_repeat = repeat_kv(key_states_full, self.num_key_value_groups)
            att_qkfull = torch.matmul(query_states, key_states_full_repeat.transpose(2, 3))
            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), 
                                                                                                                            self.group_size, 
                                                                                                                            self.k_bits)
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]
            if value_states_quant is None:
                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output = torch.matmul(attn_weights, value_states_full_repeat)
            else:
                attn_output = _fallback_value_quant_matmul(attn_weights[:, :, :, :-value_full_length], value_states_quant, 
                                                value_scale, value_mn, self.group_size, self.v_bits, self.num_key_value_groups)
                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_states_full_repeat)
            attn_output = attn_output.transpose(1, 2).contiguous()
            if value_full_length > self.residual_length:
                assert value_full_length == self.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn

        else:
            # print(f"kivi with flash! {self.k_bits}")
            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                # Handle the case where the model is quantized
                if hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)
            attn_output = self._flash_attention_forward(
                query_states.transpose(1, 2), key_states.transpose(1, 2), 
                value_states.transpose(1, 2), None, q_len, dropout=0.0
            )
            # quantize
            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None
            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
            
            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, 
                                                                                                self.group_size, 
                                                                                                self.v_bits)

        past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, 
                          value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len) if use_cache else None
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        attn_weights = None
        return attn_output, attn_weights, past_key_value


    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        from flash_attn import flash_attn_func, flash_attn_varlen_func

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
            )

        return attn_output


    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
    

class LlamaDecoderLayer_KIVI_eval(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            LlamaAttention_KIVI_eval(config=config)
            if not getattr(config, "use_flash", False)
            else LlamaFlashAttention_KIVI_eval(config=config)
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class LlamaModel_KIVI_eval(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        print(f"This is LlamaModel_KIVI class modified for evaluation")
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer_KIVI_eval(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values = _normalize_legacy_cache(past_key_values)
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][-1]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            # Ensure input_ids are on the same device as embed_tokens (multi-GPU support)
            if input_ids.device != self.embed_tokens.weight.device:
                input_ids = input_ids.to(self.embed_tokens.weight.device)
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM_KIVI_eval(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel_KIVI_eval(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        past_key_values = _normalize_legacy_cache(past_key_values)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        past_key_values = _normalize_legacy_cache(past_key_values)
        if past_key_values is not None:
            past_length = past_key_values[0][-1]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


def _selfcheck_repeat_head_contig(verbose=True):
    """CPU-only proof that _repeat_head_contig reproduces repeat_kv / repeat_kv_quant head
    ordering: torch.repeat_interleave along dim 1 (each KV head repeated n_rep times
    consecutively), never a tile order. Also checks the two properties the fused kernel
    needs: contiguity (it does .view()/.reshape() internally) and dtype preservation (the
    packed cache is int32, the scales are fp16)."""
    ok = True
    cases = [(1, 8, 4, (128, 8)), (2, 4, 2, (5, 3)), (1, 3, 1, (2, 2)),
             (1, 2, 3, (4,)), (2, 8, 4, (16, 4, 2))]
    for i, (B, KH, n_rep, rest) in enumerate(cases):
        torch.manual_seed(1000 + i)
        for dtype in (torch.float32, torch.int32):
            if dtype.is_floating_point:
                x = torch.randn(B, KH, *rest, dtype=dtype)
            else:
                x = torch.randint(-2 ** 30, 2 ** 30, (B, KH) + tuple(rest), dtype=dtype)
            got = _repeat_head_contig(x, n_rep)
            ref = torch.repeat_interleave(x, n_rep, dim=1)
            assert got.shape == ref.shape, "shape %s != %s" % (got.shape, ref.shape)
            assert got.dtype == x.dtype, "dtype changed: %s -> %s" % (x.dtype, got.dtype)
            assert got.is_contiguous(), "fused kernel requires contiguous strides"
            assert torch.equal(got, ref), "head ordering != torch.repeat_interleave(dim=1)"
            if x.dim() == 4 and "repeat_kv_quant" in globals():
                assert torch.equal(got, repeat_kv_quant(x, n_rep)), "!= repeat_kv_quant"
            if x.dim() == 4 and "repeat_kv" in globals():
                assert torch.equal(got, repeat_kv(x, n_rep)), "!= repeat_kv"
        # explicit ordering witness: head kh carries the constant value kh, so out head h
        # must carry h // n_rep (repeat_interleave) and not h % KH (tiling).
        ident = torch.arange(KH, dtype=torch.float32).view(1, KH, *([1] * len(rest)))
        ident = ident.expand(B, KH, *rest).contiguous()
        out = _repeat_head_contig(ident, n_rep)
        H = KH * n_rep
        want = torch.arange(H, dtype=torch.float32).div(n_rep).floor()
        tiled = torch.arange(H, dtype=torch.float32).remainder(KH)
        seen = out.reshape(B, H, -1)[0, :, 0]
        assert torch.equal(seen, want), "head map %s != interleave %s" % (seen, want)
        if KH > 1 and n_rep > 1:
            assert not torch.equal(seen, tiled), "head map is the tiled order - wrong"
        if verbose:
            print("  [ok] repeat_head_contig B=%d KH=%d n_rep=%d rest=%s head_map=%s"
                  % (B, KH, n_rep, tuple(rest), [int(v) for v in seen.tolist()]))
    if verbose:
        print("_selfcheck_repeat_head_contig: PASS (%d cases)" % len(cases))
    return ok


def _selfcheck_fused_bmm_padM(verbose=True):
    """CPU-only proof that the M-padding in _fused_bmm_padM (a) always hands the kernel an M
    that is a multiple of BLOCK_SIZE_M - the kernel does not mask its M axis - (b) appends
    exact zeros, and (c) returns rows [0, M) bit-identical to the unpadded result.

    The triton kernel cannot run on CPU, so it is replaced for the duration of the check by
    a stand-in with the same contract (row-wise batched GEMM, dequant elided). Operands are
    small integers held in fp32, so every product and sum is exact and `torch.equal` is a
    legitimate bit-exact comparison independent of BLAS blocking."""
    g = globals()
    had = "triton_bmm_fA_qB_outer" in g
    saved = g.get("triton_bmm_fA_qB_outer")
    seen = {}

    def _stand_in(group_size, fA, qB, sc, zp, bits):
        assert fA.shape[2] % _KIVI_KERNEL_BLOCK_M == 0, \
            "kernel got M=%d, not a multiple of BLOCK_SIZE_M=%d (would read OOB)" % (
                fA.shape[2], _KIVI_KERNEL_BLOCK_M)
        assert fA.is_contiguous(), "kernel does fA.view(-1, M, K): needs contiguous input"
        seen["fA"] = fA
        return torch.matmul(fA, qB)

    try:
        g["triton_bmm_fA_qB_outer"] = _stand_in
        cases = [(1, 8, 1, 128, 256),   # decode: M=1 (the case the padding exists for)
                 (1, 2, 33, 64, 128),   # M just over a block boundary
                 (2, 4, 32, 64, 64),    # M already a multiple of 32: no padding at all
                 (1, 1, 7, 16, 32)]
        for i, (B, H, M, K, N) in enumerate(cases):
            torch.manual_seed(2000 + i)
            fA = torch.randint(-4, 5, (B, H, M, K)).float()
            qB = torch.randint(-4, 5, (B, H, K, N)).float()
            out = _fused_bmm_padM(32, fA, qB, None, None, 2)
            ref = torch.matmul(fA, qB)
            padded = seen["fA"]
            assert padded.shape[2] % _KIVI_KERNEL_BLOCK_M == 0
            assert padded.shape[2] - M < _KIVI_KERNEL_BLOCK_M, "over-padded"
            if padded.shape[2] > M:
                assert torch.equal(padded[:, :, M:], torch.zeros_like(padded[:, :, M:])), \
                    "pad rows are not exact zeros"
            assert torch.equal(padded[:, :, :M], fA), "padding perturbed the real rows"
            assert out.shape == ref.shape, "%s != %s" % (out.shape, ref.shape)
            assert torch.equal(out, ref), "padded result differs on rows [0, M)"
            if verbose:
                print("  [ok] fused_bmm_padM B=%d H=%d M=%d->%d K=%d N=%d bit-exact"
                      % (B, H, M, padded.shape[2], K, N))
    finally:
        if had:
            g["triton_bmm_fA_qB_outer"] = saved
        else:
            g.pop("triton_bmm_fA_qB_outer", None)
    if verbose:
        print("_selfcheck_fused_bmm_padM: PASS (%d cases)" % len(cases))
    return True


if __name__ == "__main__":
    # CPU-only self-check of the fused-GQA helpers. Running this file directly needs the
    # kivi package + transformers (imported at module top); on a bare machine run the same
    # checks standalone by exec'ing just these helpers, e.g.
    #   python - <<'EOF'
    #   import ast, torch
    #   src = open("baseline/kivi/models/llama_kivi_eval.py").read()
    #   want = {"_repeat_head_contig", "_fused_bmm_padM", "repeat_kv_quant",
    #           "_selfcheck_repeat_head_contig", "_selfcheck_fused_bmm_padM"}
    #   ns = {"torch": torch, "_KIVI_KERNEL_BLOCK_M": 32}
    #   for node in ast.parse(src).body:
    #       if isinstance(node, ast.FunctionDef) and node.name in want:
    #           exec(compile(ast.Module([node], []), "<x>", "exec"), ns)
    #   ns["_selfcheck_repeat_head_contig"](); ns["_selfcheck_fused_bmm_padM"]()
    #   EOF
    _selfcheck_repeat_head_contig()
    _selfcheck_fused_bmm_padM()
    print("kivi_fused_stats():", kivi_fused_stats())
