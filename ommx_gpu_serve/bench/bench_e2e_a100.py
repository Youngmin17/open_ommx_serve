# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""A100 (sm_80) B=1 long-context vLLM E2E comparison + ablation, ONE timing source.

WHY: the headline COMPARISON (OMMX = ommx_w + CUSTOM-attn  vs  FA3  vs  TRITON) and the
ABLATION (linear / attention / outlier-decode / dequant share of TTFT & TPOT) must use the
SAME robust per-step timing so the deltas are physical. We reuse the LLMEngine.step()
CUDA-event-pair method from bench/ablation_matrix.py (NO 2-length subtraction, NO prefill
contamination) and add the ommx_w QUANTIZATION wiring from serve/bench_w_sweep.py
(register_ommx_w + quantization='ommx_w' + hf_overrides + EQUAL-KV-pool pinning).

ARMS (each = its own subprocess so the W/attn/toggle env is captured at vLLM build time and
the GPU memory is clean):

  COMPARISON  (B=1, ctx {1024,4096,16384,65536,131072}):
    OMMX    = ommx_w (i2f4) + CUSTOM attention  (VLLM_ATTENTION_BACKEND=CUSTOM)
    FA3     = bf16-w + FLASH_ATTN
    TRITON  = bf16-w + TRITON_ATTN

  ABLATION  (B=1, ctx {4096,65536}, all FULL-graph):
    bf16    = bf16-w + FLASH_ATTN                 (the floor)
    linear  = ommx_w + FLASH_ATTN                 (delta vs bf16 = OMMX LINEAR cost)
    attn    = bf16-w + CUSTOM                      (delta vs bf16 = OMMX ATTENTION cost)
    full    = ommx_w + CUSTOM                      (== the OMMX comparison arm)
    full_no_outlier = full + OMMX_ABL_NO_OUTLIER   (outlier-decode share = full - this)
    full_no_dequant = full + OMMX_ABL_V_NODEQUANT  (dequant share = full - this)

  ATTENTION BREAKDOWN  (B=1, abl_ctxs, all FULL-graph, bf16-w + CUSTOM; PARITY-BREAKING,
  coh=False — each SKIPs one decode cost so its TPOT delta isolates that share):
    attn                = full attention (the reference for these deltas)
    attn_no_outlier     = attn + NO_OUTLIER                 (skip relidx7 outlier splice)
    attn_no_unpack      = attn + NO_UNPACK                  (skip K/V bit-extract ALU)
    attn_no_dequant     = attn + V_NODEQUANT                (skip V scale/zp dequant)
    attn_no_dequant_all = attn + K&V NODEQUANT              (skip K+V scale/zp dequant)
    attn_no_all         = attn + NO_OUTLIER+K&V NODEQUANT+NO_UNPACK (base math+mem only)
    attn_skipwrite      = attn + OMMX_ABL_SKIP_WRITE        (skip per-step KV pack/write)
  DIFFERENTIAL DECOMPOSITION (TPOT(arm); larger arm => more work kept):
    pack/write share  = TPOT(attn)             - TPOT(attn_skipwrite)
    outlier share     = TPOT(attn)             - TPOT(attn_no_outlier)
    bit-extract share = TPOT(attn)             - TPOT(attn_no_unpack)
    V scale/zp share  = TPOT(attn)             - TPOT(attn_no_dequant)
    K+V scale/zp share= TPOT(attn)             - TPOT(attn_no_dequant_all)
    scale/zp-only ALU = TPOT(attn_no_unpack)   - TPOT(attn_no_dequant_all)
    base attn (math+mem; QK/softmax/PV + KV load) = TPOT(attn_no_all)

FAIR-COMPARE (locked for every arm): enforce_eager=False (CUDA-graph FULL) + inductor
fusion + dtype=bf16 + seed=42 + prefix_caching=False + chunked_prefill=False + EQUAL
geometry. The bf16 arm at the SMALLEST ctx sets num_gpu_blocks; every other arm is pinned to
it (trap C: a quantized arm frees weight bytes and would otherwise get a bigger KV pool ->
fake TTFT/TPOT win). cudagraph_capture_sizes pinned to {1} (this study is B=1) for fast
capture. TPOT = steady-state per-step CUDA-event time (p50+p99 over >=measure steps pooled
over reps). TTFT = the prefill step CUDA-event time. OOM is recorded (capacity story).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" PYTHONNOUSERSITE=1 \
      TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_80" TORCH_CUDA_ARCH_LIST=8.0 \
      OMMX_W_BUNDLE=<path/to/ommx_weights.safetensors> \
      VLLM_PLUGINS=ommx_gpu_serve VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      HF_HUB_OFFLINE=1 \
      python -m ommx_gpu_serve.bench.bench_e2e_a100 --model <llama-3.1-8b-path> \
        --csv runs/e2e_a100.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys


# ── percentile helpers ───────────────────────────────────────────────────────
def _pctl(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _median(xs):
    return _pctl(xs, 0.5)


def _free_mem_gib():
    import torch
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


def _gpu_util(headroom_gib=8.0, hard_cap=0.92):
    import torch
    free, total = torch.cuda.mem_get_info()
    total_g = total / (1024 ** 3)
    target = max(0.30, (free / (1024 ** 3) - headroom_gib) / total_g)
    return min(hard_cap, target)


# ── engine construction (quantization-aware; CUDA-graph FULL fair-compare) ────
def _make_engine(model, backend, quant, max_len, gpu_mem, seed, enforce_eager,
                 num_gpu_blocks=0, capture_sizes="default", tp=1, kv_dtype=""):
    """LLMEngine via from_engine_args so we can drive .step() directly.

    backend -> VLLM_ATTENTION_BACKEND (env path EngineArgs honors).
    quant   -> 'ommx_w' (+ synthetic hf_override so vLLM v1 reaches the runtime-registered
               OMMXWLinearConfig for a BASE checkpoint) or '' (stock bf16 linear).
    num_gpu_blocks -> EQUAL-KV-pool pin (fairness)."""
    from vllm import LLMEngine, EngineArgs
    kw = dict(model=model, max_model_len=max_len, gpu_memory_utilization=gpu_mem,
              tensor_parallel_size=int(tp), enforce_eager=enforce_eager,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              dtype="bfloat16", trust_remote_code=True, seed=seed)
    if backend:
        # 🚨 vLLM 0.21 REMOVED the VLLM_ATTENTION_BACKEND env (it warns "Unknown vLLM
        # environment variable" and silently defaults to FLASH_ATTN). The backend MUST be
        # selected via the EngineArgs.attention_backend field (CLI --attention-backend), or
        # EVERY arm (CUSTOM, TRITON_ATTN, FLASH_ATTN) silently runs FLASH_ATTN -> the OMMX
        # attention never fires and all arms read identical (the bf16 fallback bug).
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        kw["attention_backend"] = backend
    if quant:
        kw["quantization"] = quant
        kw["hf_overrides"] = {"quantization_config": {"quant_method": quant}}
    if kv_dtype:
        # TurboQuant KV-cache-dtype (turboquant_3bit_nc etc.): vLLM auto-selects the turboquant
        # attention backend + Triton decode kernel; do NOT also pass attention_backend.
        kw["kv_cache_dtype"] = kv_dtype
    if num_gpu_blocks and num_gpu_blocks > 0:
        kw["num_gpu_blocks_override"] = int(num_gpu_blocks)
    if not enforce_eager and capture_sizes.strip().lower() != "default":
        try:
            sizes = sorted({int(x) for x in capture_sizes.split(",") if x.strip()})
            kw["compilation_config"] = {"cudagraph_capture_sizes": sizes}
        except (ValueError, TypeError):
            pass
    ea = EngineArgs(**kw)
    return LLMEngine.from_engine_args(ea)


def _kv_blocks(eng):
    for getter in (
        lambda: eng.cache_config.num_gpu_blocks,
        lambda: eng.vllm_config.cache_config.num_gpu_blocks,
        lambda: eng.engine_core.engine_core.scheduler.cache_config.num_gpu_blocks,
        lambda: eng.model_executor.cache_config.num_gpu_blocks,
    ):
        try:
            v = int(getter())
            if v > 0:
                return v
        except Exception:
            continue
    return 0


# ── robust per-step decode timing (LLMEngine.step() loop, NO subtraction) ─────
def _time_cell(eng, ctx, batch, warmup, measure, reps):
    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    total = warmup + measure + 4
    ttft_samples = []
    tpot_samples = []

    def _drain():
        while eng.has_unfinished_requests():
            eng.step()

    for w in range(2):  # warm compile / cudagraph capture / allocator
        sp = SamplingParams(temperature=0.0, max_tokens=total, ignore_eos=True)
        for i in range(batch):
            eng.add_request(f"w{w}_{i}", TokensPrompt(prompt_token_ids=[100] * ctx), sp)
        steps = 0
        while eng.has_unfinished_requests() and steps < total + 2:
            eng.step(); steps += 1
        _drain()

    for rep in range(reps):
        sp = SamplingParams(temperature=0.0, max_tokens=total, ignore_eos=True)
        for i in range(batch):
            eng.add_request(f"r{rep}_{i}", TokensPrompt(prompt_token_ids=[100] * ctx), sp)
        torch.cuda.synchronize()
        prefill_ms = 0.0
        prefill_max = 0.0
        in_decode = False
        decode_idx = 0
        steps = 0
        per_step = []
        while eng.has_unfinished_requests() and steps < total + batch + 4:
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            eng.step()
            e.record(); torch.cuda.synchronize()
            dt = s.elapsed_time(e)
            per_step.append(dt)
            steps += 1
            if not in_decode:
                prefill_max = max(prefill_max, dt)
                if steps >= 1 and dt <= 0.40 * prefill_max and prefill_max > 0:
                    in_decode = True
                    if decode_idx >= warmup and (decode_idx - warmup) < measure:
                        tpot_samples.append(dt)
                    decode_idx += 1
                else:
                    prefill_ms += dt
            else:
                if decode_idx >= warmup and (decode_idx - warmup) < measure:
                    tpot_samples.append(dt)
                decode_idx += 1
        if prefill_ms == 0.0 and per_step:
            prefill_ms = per_step[0]
            for di, dt in enumerate(per_step[1:]):
                if di >= warmup and (di - warmup) < measure:
                    tpot_samples.append(dt)
        ttft_samples.append(prefill_ms)
        _drain()

    if not tpot_samples:
        raise RuntimeError(f"no decode steps captured (ctx={ctx} batch={batch})")
    return {
        "ttft_p50": _median(ttft_samples), "ttft_p99": _pctl(ttft_samples, 0.99),
        "tpot_p50": _median(tpot_samples), "tpot_p99": _pctl(tpot_samples, 0.99),
        "n_steps": len(tpot_samples),
    }


def _coherence(eng, ctx, maxtok=16):
    """Greedy token-id sequence for one length-ctx prompt (law #12 parity check)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(temperature=0.0, max_tokens=maxtok, ignore_eos=True)
    eng.add_request("coh", TokensPrompt(prompt_token_ids=[100] * ctx), sp)
    out_ids = []
    while eng.has_unfinished_requests():
        for o in eng.step():
            if getattr(o, "finished", False) or getattr(o, "outputs", None):
                try:
                    out_ids = list(o.outputs[0].token_ids)
                except Exception:
                    pass
    return out_ids


# ── single-arm worker ────────────────────────────────────────────────────────
def _single_arm(args):
    import torch
    try:
        from ommx_gpu_serve.integration.vllm.plugin import register
        register()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] attn plugin register failed: {e}", flush=True)
    if args.quant == "ommx_w":
        try:
            # register() above already registers ommx_w via the plugin; this is an
            # explicit idempotent re-register + accurate confirmation (the old import
            # pointed at the archived serve/vllm_ommx_linear module, absent here).
            from ommx_gpu_serve.integration.vllm.linear_method import register_ommx_w
            register_ommx_w()
            print("[arm] register_ommx_w() OK", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[arm] register_ommx_w FAILED: {e}", flush=True)

    env_extra = json.loads(args.env) if args.env else {}
    for k, v in env_extra.items():
        os.environ[k] = v
    cells = [(int(p.split(":")[0]), int(p.split(":")[1]))
             for p in args.cells.split(",") if p]
    gm = args.gpu_mem if args.gpu_mem > 0 else _gpu_util()
    enforce_eager = args.eager
    mode = "eager" if enforce_eager else "graph"
    print(f"\n===== ARM={args.single_arm} backend={args.backend} quant={args.quant!r} "
          f"mode={mode} kvpin={args.num_gpu_blocks} free={_free_mem_gib():.1f}GiB "
          f"env={env_extra} =====", flush=True)
    try:
        eng = _make_engine(args.model, args.backend or None, args.quant or "",
                           args.model_max_len, gm, args.seed, enforce_eager,
                           num_gpu_blocks=args.num_gpu_blocks, tp=args.tp,
                           kv_dtype=args.kv_dtype)
    except Exception as e:  # noqa: BLE001
        print(f"[{args.single_arm}] LOAD FAILED ({type(e).__name__}: {e})", flush=True)
        with open(args.arm_out, "w") as f:
            json.dump({"cells": {}, "kv_blocks": 0,
                       "load_err": f"{type(e).__name__}: {e}"}, f)
        return
    kvb = _kv_blocks(eng)
    print(f"[{args.single_arm}] KV gpu_blocks = {kvb}", flush=True)

    coh = []
    if args.coherence_ctx > 0:
        try:
            coh = _coherence(eng, args.coherence_ctx, args.coherence_tok)
            print(f"[{args.single_arm}] COHERENCE ids[:16]={coh[:16]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{args.single_arm}] coherence skip ({type(e).__name__}: {e})", flush=True)

    out = {}
    for (b, c) in cells:
        key = f"b{b}_ctx{c}"
        try:
            r = _time_cell(eng, c, b, args.warmup, args.measure, args.reps)
            r["ok"] = True; r["mode"] = mode; r["oom"] = False
            out[key] = r
            print(f"[{args.single_arm}] {key:14s} TTFT p50={r['ttft_p50']:9.2f} "
                  f"p99={r['ttft_p99']:9.2f}ms | TPOT p50={r['tpot_p50']:8.4f} "
                  f"p99={r['tpot_p99']:8.4f}ms/step (nstep={r['n_steps']})", flush=True)
        except Exception as e:  # noqa: BLE001
            es = f"{type(e).__name__}: {e}"
            is_oom = ("CUDA out of memory" in es or "OutOfMemory" in es
                      or "no decode steps" in es and False)
            out[key] = dict(ok=False, err=es, mode=mode, oom=bool(is_oom))
            print(f"[{args.single_arm}] {key:14s} SKIP ({es})", flush=True)
            torch.cuda.empty_cache()
    del eng; gc.collect(); torch.cuda.empty_cache()
    with open(args.arm_out, "w") as f:
        json.dump({"cells": out, "kv_blocks": kvb, "coherence": coh}, f)
    print(f"[{args.single_arm}] ARM_DONE kvb={kvb}", flush=True)


# ── arm plan ─────────────────────────────────────────────────────────────────
def _fakequant_kv_recipe_env():
    """KV recipe matching fakequant/run.sh --phase kv, with ratio-selectable K outliers."""
    return {
        # fakequant/run.sh KV phase: --group_size 64 --attention_sink_num 8
        # --use_pow2 --outlier_method fp4 --outlier_precision_key fp4
        # --no_detect_outliers_value. Serving expresses the back sink as recent=8.
        "OMMX_KV_GROUP_TOKENS": "64",
        "OMMX_KV_GROUP_CHANNELS": "64",
        "OMMX_KV_SINK": "8",
        "OMMX_KV_RECENT": "8",
        "OMMX_ATTN_POW2": "1",
        "OMMX_ATTN_K_FORMAT": "i2f4",
        "OMMX_ATTN_OUTLIER_SELECT": "abs",
        "OMMX_ATTN_OUTLIER_REPR": "relidx7",
        "OMMX_ATTN_COMBINADIC_READ": "0",
        "OMMX_KV_OUTLIER_MAP": "1",
        # This benchmark is Triton-only; CUDA attention route is intentionally excluded.
        "OMMX_ATTN_CUDA_DECODE": "0",
    }


def _build_plan(cmp_ctxs, abl_ctxs, outlier_env, batches=(1,), kv_token_budget=0):
    """arm name -> dict(backend, quant, eager, cells[(b,ctx)], env, label, group).

    batches: decode batch sizes to sweep (B>1 = high-batch memory-bound regime; the
    OMMX CUSTOM arms need OMMX_ATTN_BATCHED_GRAPH=1 in the env so B>1 fires the captured
    batched route). kv_token_budget: drop (b,ctx) cells whose b*ctx exceeds the equal-KV
    pool token capacity (avoids preempt/OOM; 0 = keep all)."""
    plan = {}

    def _cells(ctxs):
        return [(b, c) for b in batches for c in ctxs
                if kv_token_budget <= 0 or b * c <= kv_token_budget]
    cmp_cells = _cells(cmp_ctxs)
    abl_cells = _cells(abl_ctxs)

    # ── headline COMPARISON ──
    plan["OMMX"] = dict(backend="CUSTOM", quant="ommx_w", eager=False, cells=cmp_cells,
                        env={"OMMX_ATTN_GRAPH": "1", **outlier_env},
                        label="ommx_w+CUSTOM", group="cmp")
    plan["FA3"] = dict(backend="FLASH_ATTN", quant="", eager=False, cells=cmp_cells,
                       env={}, label="bf16+FA3", group="cmp")
    plan["TRITON"] = dict(backend="TRITON_ATTN", quant="", eager=False, cells=cmp_cells,
                          env={}, label="bf16+TRITON", group="cmp")
    # TurboQuant KV-cache quant (3-bit K+V + norm-correction), a KV-quant peer to OMMX/KIVI/Kitty.
    # backend="" -> vLLM auto-selects the turboquant attention backend + Triton decode kernel.
    plan["TURBOQUANT"] = dict(backend="", quant="", kv_dtype="turboquant_3bit_nc", eager=False,
                              cells=cmp_cells, env={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
                              label="turboquant_3bit_nc", group="cmp")
    # v10a head-to-head: bf16-w (W NOT quantized) + OMMX CUSTOM attention with K=i2f4 and
    # V=bf16 (OMMX_ATTN_V_BF16). Isolates the attention kernel vs FA3/TRITON (identical
    # bf16 linear); V=bf16 drops the V=i2 dequant ALU (the M=1 decode loss cause) keeping
    # the K i2f4 byte saving -> the win-path test. Coherence vs FA3 proves it fires.
    plan["OMMX_V10A"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=cmp_cells,
        env={"OMMX_ATTN_GRAPH": "1", "OMMX_KV_RING": "1", "OMMX_KV_GPU_PACK": "1",
             "OMMX_ATTN_V_BF16": "1", **outlier_env},
        label="bf16-w+OMMX-v10a(K=i2f4,V=bf16)", group="cmp")

    # ── ABLATION stack (B=1, abl_ctxs) ──
    plan["abl_bf16"] = dict(backend="FLASH_ATTN", quant="", eager=False, cells=abl_cells,
                            env={}, label="bf16-w+FA3", group="abl")
    # ── FAIR-COMPARISON CANON (change ONLY the part under test; OMMX defaults baked in) ──
    # LINEAR comparison: plain-vLLM (abl_bf16: bf16-w + FA) vs OMMX-linear (ommx_w + FA),
    #   identical FA attention -> the TPOT delta is purely the OMMX LINEAR. SCAN-FREE relidx7
    #   correction ON by default (the in-kernel combinadic unrank was the catastrophic prefill).
    plan["abl_linear"] = dict(backend="FLASH_ATTN", quant="ommx_w", eager=False,
                              cells=abl_cells, env={"OMMX_W_CORR_SCANFREE": "1"},
                              label="ommx_w(scanfree)+FA", group="abl")
    # ATTENTION comparison: paged-triton-v1 (TRITON arm: bf16 attn + bf16-w) vs OMMX-attn
    #   (CUSTOM OMMX attn + bf16-w), identical bf16 linear -> TPOT delta is purely the OMMX
    #   ATTENTION. NO-SHADOW RING ON by default: OMMX K/V is <=3-bit so it MUST use LESS HBM
    #   than bf16; the bf16 k_hist SHADOW (the OOM cause on A100) is removed (ring keeps only
    #   the sink+recent window). GPU-pack on (boundary-step pack on device, no host p99).
    plan["abl_attn"] = dict(backend="CUSTOM", quant="", eager=False, cells=abl_cells,
                            env={"OMMX_ATTN_GRAPH": "1", "OMMX_KV_RING": "1",
                                 "OMMX_KV_GPU_PACK": "1", **outlier_env},
                            label="bf16-w+OMMXattn(noshadow)", group="abl", coh=True)
    plan["abl_full"] = dict(backend="CUSTOM", quant="ommx_w", eager=False, cells=abl_cells,
                            env={"OMMX_ATTN_GRAPH": "1", **outlier_env},
                            label="ommx_w+CUSTOM", group="abl")
    plan["abl_full_no_outlier"] = dict(
        backend="CUSTOM", quant="ommx_w", eager=False, cells=abl_cells,
        env={"OMMX_ATTN_GRAPH": "1", "OMMX_ABL_NO_OUTLIER": "1", **outlier_env},
        label="ommx_w+CUSTOM-outlier", group="abl")
    plan["abl_full_no_dequant"] = dict(
        backend="CUSTOM", quant="ommx_w", eager=False, cells=abl_cells,
        env={"OMMX_ATTN_GRAPH": "1", "OMMX_ABL_V_NODEQUANT": "1", **outlier_env},
        label="ommx_w+CUSTOM-Vdequant", group="abl")
    # PURE attention isolation (bf16-w + CUSTOM): outlier-decode and V-dequant share of the
    # ATTENTION cost alone (scenario-2), not confounded by the ommx_w linear path.
    plan["abl_attn_no_outlier"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={"OMMX_ATTN_GRAPH": "1", "OMMX_ABL_NO_OUTLIER": "1", **outlier_env},
        label="bf16-w+CUSTOM-outlier", group="abl")
    plan["abl_attn_no_dequant"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={"OMMX_ATTN_GRAPH": "1", "OMMX_ABL_V_NODEQUANT": "1", **outlier_env},
        label="bf16-w+CUSTOM-Vdequant", group="abl")
    # ── V=i2 DEQUANT LEVERS (bf16-w + CUSTOM = abl_attn + ONE lever env) ──
    # Attack the V=i2 per-element dequant ALU (the measured 12-23ms per-tile tax) to
    # push the OMMX-attn TPOT from parity toward a >=1.10x win vs TRITON at long ctx.
    # All parity-by-construction (bit-exact codes 0..3) -> coh=True match_vs_TRITON gate.
    _abl_attn_env = {"OMMX_ATTN_GRAPH": "1", "OMMX_KV_RING": "1",
                     "OMMX_KV_GPU_PACK": "1", **outlier_env}
    # ── BREAKDOWN-SWEEP ABL arms (bf16-w + CUSTOM = abl_attn + ONE/MORE timing flag) ──
    # PARITY-BREAKING by design (coh=False): each forces the kernel/backend to SKIP one
    # decode cost so its TPOT delta isolates that share (vs abl_attn = full).
    #   abl_attn            : full  (already defined above)
    #   abl_attn_no_outlier : full + NO_OUTLIER (outlier-decode share)
    #   abl_attn_no_dequant : full + V_NODEQUANT only (V-dequant share)  [already above]
    #   abl_attn_no_unpack  : full + NO_UNPACK (bit-extract ALU share)
    #   abl_attn_no_dequant_all : full + K&V NODEQUANT (full scale/zp dequant share)
    #   abl_attn_no_all     : full + NO_OUTLIER+K&V NODEQUANT+NO_UNPACK (base attn math+mem)
    #   abl_attn_skipwrite  : full + SKIP_WRITE (per-step KV pack/write share)
    plan["abl_attn_no_unpack"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={**_abl_attn_env, "OMMX_ABL_NO_UNPACK": "1"},
        label="bf16-w+CUSTOM-unpack", group="abl", coh=False)
    plan["abl_attn_no_dequant_all"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={**_abl_attn_env, "OMMX_ABL_V_NODEQUANT": "1",
             "OMMX_ABL_K_NODEQUANT": "1"},
        label="bf16-w+CUSTOM-KVdequant", group="abl", coh=False)
    plan["abl_attn_no_all"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={**_abl_attn_env, "OMMX_ABL_NO_OUTLIER": "1",
             "OMMX_ABL_V_NODEQUANT": "1", "OMMX_ABL_K_NODEQUANT": "1",
             "OMMX_ABL_NO_UNPACK": "1"},
        label="bf16-w+CUSTOM-base(math+mem)", group="abl", coh=False)
    plan["abl_attn_skipwrite"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={**_abl_attn_env, "OMMX_ABL_SKIP_WRITE": "1"},
        label="bf16-w+CUSTOM-skipwrite", group="abl", coh=False)
    plan["OMMX_VINTPV"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=cmp_cells,
        env={**_abl_attn_env, "OMMX_ATTN_V_INT_PV": "1"},
        label="bf16-w+OMMXattn+V_INT_PV", group="lever", coh=True)
    plan["OMMX_VLUT"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=cmp_cells,
        env={**_abl_attn_env, "OMMX_ATTN_VLUT": "1"},
        label="bf16-w+OMMXattn+VLUT", group="lever", coh=True)
    plan["OMMX_VINTPV_PIPE"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=cmp_cells,
        env={**_abl_attn_env, "OMMX_ATTN_V_INT_PV": "1",
             "OMMX_ATTN_KV_PIPE": "1", "OMMX_V4_NUM_STAGES": "3"},
        label="bf16-w+OMMXattn+V_INT_PV+pipe(s3)", group="lever", coh=True)
    # BATCHED OMMX (B>1) decode. CRITICAL: OMMX_ATTN_GRAPH MUST be ABSENT — the backend
    # gate is `in_batched = _BATCHED_SESSION and not _GRAPH`, so OMMX_ATTN_GRAPH=1 disables
    # the batched route entirely and a B>1 decode silently falls to bf16 FlashAttention
    # (super().forward) with NO route marker. OMMX_ATTN_BATCHED=1 force-latches the pool;
    # OMMX_ATTN_BATCHED_GRAPH=1 captures the batched decode graph. Firing proof =
    # BATCHED_GRAPH_ROUTE_FIRED in $OMMX_FIRE_FILE.
    plan["abl_attn_bat"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=cmp_cells,
        env={"OMMX_ATTN_BATCHED": "1", "OMMX_ATTN_BATCHED_GRAPH": "1",
             "OMMX_KV_RING": "1", "OMMX_KV_GPU_PACK": "1", **outlier_env},
        label="bf16-w+OMMXattn-BATCHED(graph)", group="lever", coh=True)
    # EAGER batched variant: sidesteps the vLLM multi-size graph-capture ladder (the
    # B=1/prefill-shaped capture dummy crashes the batched write path). enforce_eager=True
    # => no capture => the eager batched route (_route_decode_batched, BATCHED_ROUTE_FIRED)
    # runs. NO OMMX_ATTN_BATCHED_GRAPH (eager). Compared vs TRITON(graph) is conservative
    # (TRITON keeps the launch-amortization edge), so an OMMX win here is a real floor.
    plan["abl_attn_bat_eager"] = dict(
        backend="CUSTOM", quant="", eager=True, cells=cmp_cells,
        env={"OMMX_ATTN_BATCHED": "1", "OMMX_KV_RING": "1",
             "OMMX_KV_GPU_PACK": "1", **outlier_env},
        label="bf16-w+OMMXattn-BATCHED(eager)", group="lever", coh=False)
    # LSE-merge wide-grid + tail-dedup A/B: abl_attn + OMMX_MERGE_WIDE=1 (default-OFF lever
    # that fills SMs in the 2nd/merge launch + shares the GQA tail). Parity bit-exact by
    # design; compare vs abl_attn (same pool, same process) to isolate the merge-launch cost.
    plan["abl_attn_mw"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=abl_cells,
        env={"OMMX_ATTN_GRAPH": "1", "OMMX_KV_RING": "1", "OMMX_KV_GPU_PACK": "1",
             "OMMX_MERGE_WIDE": "1", **outlier_env},
        label="bf16-w+OMMXattn+merge_wide", group="abl", coh=True)
    plan["OMMX_VINTPV_BAT"] = dict(
        backend="CUSTOM", quant="", eager=False, cells=cmp_cells,
        env={"OMMX_ATTN_BATCHED": "1", "OMMX_ATTN_BATCHED_GRAPH": "1",
             "OMMX_ATTN_V_INT_PV": "1", "OMMX_KV_RING": "1",
             "OMMX_KV_GPU_PACK": "1", **outlier_env},
        label="bf16-w+OMMXattn-BATCHED+V_INT_PV", group="lever", coh=True)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cmp-ctxs", default="1024,4096,16384,65536,131072")
    ap.add_argument("--abl-ctxs", default="4096,65536")
    ap.add_argument("--batches", default="1",
                    help="comma list of decode batch sizes to sweep (B>1 high-batch "
                         "regime; OMMX arms need OMMX_ATTN_BATCHED_GRAPH=1 in env)")
    ap.add_argument("--kv-token-budget", type=int, default=0,
                    help="drop (b,ctx) cells where b*ctx exceeds this (equal-KV pool "
                         "fit; avoids preempt/OOM). 0 = keep all")
    ap.add_argument("--model-max-len", type=int, default=131072)
    ap.add_argument("--gpu-mem", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--measure", type=int, default=160)  # >=128 steady-state steps
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--outliers", type=int, default=6)  # canonical 12/64 (6 per 32-ch group)
    ap.add_argument("--outlier-pct", default="",
                    help="K outlier fraction per 64-token fakequant recipe group; e.g. 0.05 or 0.10")
    ap.add_argument("--recipe", choices=("fakequant", "current"), default="fakequant",
                    help="fakequant = match fakequant/run.sh KV recipe; current = only use explicit env")
    ap.add_argument("--coherence-ctx", type=int, default=128,
                    help="run a short greedy decode per arm for token-id parity (0=off)")
    ap.add_argument("--coherence-tok", type=int, default=16)
    ap.add_argument("--csv", default="runs/e2e_a100.csv")
    ap.add_argument("--json", default="runs/e2e_a100.json")
    ap.add_argument("--workdir", default="runs/e2e_a100_arms")
    ap.add_argument("--arch", default="a100")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor_parallel_size. TP>1 shards KV heads per rank; the OMMX "
                         "sidecar store/pool/kernel are per-rank-sharded already. Verify "
                         "firing PER RANK via the rank-suffixed $OMMX_FIRE_FILE.")
    ap.add_argument("--only-arms", default="",
                    help="comma list of arm names to run (kvprobe always runs); empty=all")
    # single-arm worker mode
    ap.add_argument("--single-arm", default="")
    ap.add_argument("--backend", default="")
    ap.add_argument("--quant", default="")
    ap.add_argument("--kv-dtype", default="", dest="kv_dtype")  # e.g. turboquant_3bit_nc
    ap.add_argument("--cells", default="")
    ap.add_argument("--env", default="")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--num-gpu-blocks", type=int, default=0)
    ap.add_argument("--arm-out", default="")
    args = ap.parse_args()

    if args.single_arm:
        _single_arm(args)
        return

    cmp_ctxs = [int(x) for x in args.cmp_ctxs.split(",")
                if x and int(x) + args.warmup + args.measure + 8 <= args.model_max_len]
    abl_ctxs = [int(x) for x in args.abl_ctxs.split(",")
                if x and int(x) + args.warmup + args.measure + 8 <= args.model_max_len]
    outlier_env = _fakequant_kv_recipe_env() if args.recipe == "fakequant" else {}
    outlier_pct = (args.outlier_pct or os.environ.get("OMMX_OUTLIER_PERCENT", "")).strip()
    if outlier_pct:
        pct = float(outlier_pct)
        outlier_env["OMMX_OUTLIER_PERCENT"] = f"{pct:g}"
        outlier_desc = f"{pct * 100:g}%"
    else:
        outlier_env["OMMX_ATTN_OUTLIERS"] = str(args.outliers)
        outlier_desc = f"npv={args.outliers}"
    os.makedirs(args.workdir, exist_ok=True)
    batches = tuple(int(b) for b in args.batches.split(",") if b.strip()) or (1,)
    plan = _build_plan(cmp_ctxs, abl_ctxs, outlier_env, batches=batches,
                       kv_token_budget=args.kv_token_budget)
    n_cells = sum(len(p["cells"]) for p in plan.values())
    print(f"[e2e] model={args.model}\n[e2e] cmp_ctxs={cmp_ctxs} abl_ctxs={abl_ctxs} "
          f"warmup={args.warmup} measure={args.measure} reps={args.reps} "
          f"recipe={args.recipe} outliers={outlier_desc}\n"
          f"[e2e] {len(plan)} arms, {n_cells} cells", flush=True)

    # EQUAL-KV-POOL: run bf16 arm at SMALLEST ctx first to read num_gpu_blocks, then PIN it
    # for every arm. bf16-w is the largest-weight arm so its KV pool is the binding floor.
    results = {"model": args.model, "arms": {}, "coherence": {},
               "config": dict(cmp_ctxs=cmp_ctxs, abl_ctxs=abl_ctxs, warmup=args.warmup,
                              measure=args.measure, reps=args.reps, arch=args.arch,
                              recipe=args.recipe, outliers=args.outliers,
                              outlier_pct=outlier_pct,
                              fair=dict(enforce_eager=False, cudagraph="FULL",
                                        dtype="bfloat16", seed=args.seed,
                                        prefix_caching=False, chunked_prefill=False,
                                        capture_sizes="default",
                                        equal_kv_pool="pinned to bf16 TRITON smallest-ctx arm"),
                              method="LLMEngine.step() CUDA-event per-step; TPOT p50/p99 "
                                     "over measure steps x reps (NO subtraction); TTFT = "
                                     "prefill-step CUDA-event time")}

    def _spawn(arm_name, p, num_gpu_blocks, coherence_ctx):
        cell_str = ",".join(f"{b}:{c}" for (b, c) in p["cells"])
        arm_out = os.path.join(args.workdir, f"{arm_name}.json")
        cmd = [sys.executable, "-m", "ommx_gpu_serve.bench.bench_e2e_a100",
               "--single-arm", arm_name, "--backend", p["backend"],
               "--quant", p["quant"], "--kv-dtype", p.get("kv_dtype", ""),
               "--model", args.model, "--cells", cell_str,
               "--env", json.dumps(p["env"]),
               "--model-max-len", str(args.model_max_len), "--gpu-mem", str(args.gpu_mem),
               "--seed", str(args.seed), "--warmup", str(args.warmup),
               "--measure", str(args.measure), "--reps", str(args.reps),
               "--num-gpu-blocks", str(num_gpu_blocks),
               "--coherence-ctx", str(coherence_ctx),
               "--coherence-tok", str(args.coherence_tok),
               "--tp", str(args.tp),
               "--arm-out", arm_out]
        if p["eager"]:
            cmd.append("--eager")
        print(f"\n########## SPAWN {arm_name} backend={p['backend']} quant={p['quant']!r} "
              f"kvpin={num_gpu_blocks} env={p['env']} ##########", flush=True)
        rc = subprocess.call(cmd)
        d = {}
        if os.path.exists(arm_out):
            with open(arm_out) as f:
                d = json.load(f)
        else:
            print(f"[orch] arm {arm_name} produced no output (rc={rc})", flush=True)
        return d

    # ── probe: bf16 TRITON at smallest ctx to fix the KV pool ──
    # FA3 is intentionally not used as probe/baseline for the attention-only win path.
    probe_ctx = min(cmp_ctxs) if cmp_ctxs else min(abl_ctxs)
    probe_plan = dict(backend="TRITON_ATTN", quant="", eager=False,
                      cells=[(1, probe_ctx)], env={}, label="kv-probe-triton")
    print(f"\n[e2e] KV-POOL PROBE: bf16 TRITON ctx={probe_ctx}", flush=True)
    pd = _spawn("kvprobe", probe_plan, 0, 0)
    pinned_blocks = int(pd.get("kv_blocks", 0))
    print(f"[e2e] pinned num_gpu_blocks = {pinned_blocks}", flush=True)

    # ── all arms, pinned KV pool ──
    order = ["OMMX", "FA3", "TRITON", "TURBOQUANT", "OMMX_V10A", "abl_bf16", "abl_linear", "abl_attn", "abl_full",
             "abl_full_no_outlier", "abl_full_no_dequant",
             "abl_attn_no_outlier", "abl_attn_no_dequant",
             "abl_attn_no_unpack", "abl_attn_no_dequant_all", "abl_attn_no_all",
             "abl_attn_skipwrite", "abl_attn_mw",
             "OMMX_VINTPV", "OMMX_VLUT", "OMMX_VINTPV_PIPE",
             "abl_attn_bat", "OMMX_VINTPV_BAT", "abl_attn_bat_eager"]
    if args.only_arms:
        keep = {a.strip() for a in args.only_arms.split(",") if a.strip()}
        order = [a for a in order if a in keep]
        print(f"[e2e] only-arms filter -> {order}", flush=True)
    for arm_name in order:
        if arm_name not in plan:
            continue
        p = plan[arm_name]
        coh_ctx = args.coherence_ctx if p.get("coh", p["group"] == "cmp") else 0
        d = _spawn(arm_name, p, pinned_blocks, coh_ctx)
        results["arms"][arm_name] = d.get("cells", {})
        if d.get("coherence"):
            results["coherence"][arm_name] = d["coherence"]

    # ── CSV (comparison + ablation rows, single contract) ──
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    fields = ["arch", "group", "arm", "label", "backend", "quant", "batch", "ctx",
              "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99", "n_steps",
              "mode", "ok", "oom", "err"]
    rows = []
    for arm_name, cellmap in results["arms"].items():
        p = plan[arm_name]
        for key, r in cellmap.items():
            b = int(key.split("_")[0][1:]); c = int(key.split("ctx")[1])
            row = dict(arch=args.arch, group=p["group"], arm=arm_name, label=p["label"],
                       backend=p["backend"], quant=p["quant"], batch=b, ctx=c,
                       mode=r.get("mode", ""), ok=bool(r.get("ok")),
                       oom=bool(r.get("oom")), err=r.get("err", ""),
                       n_steps=r.get("n_steps", ""))
            if r.get("ok"):
                for f in ("ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99"):
                    row[f] = round(r[f], 5)
            rows.append(row)
    rows.sort(key=lambda x: (x["group"], x["arm"], x["ctx"]))
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(1 for r in rows if r.get("ok"))
    print(f"\n[e2e] wrote {args.csv} ({n_ok}/{len(rows)} cells ok)", flush=True)
    print(f"[e2e] wrote {args.json}", flush=True)
    # coherence summary: TRITON is the only accepted baseline for this attention-only path.
    base = results["coherence"].get("TRITON")
    for a, ids in results["coherence"].items():
        match = (ids[:16] == base[:16]) if base else None
        print(f"[e2e] coherence {a}: match_vs_TRITON={match} ids[:8]={ids[:8]}", flush=True)
    print("BENCH_E2E_A100_DONE", flush=True)


if __name__ == "__main__":
    main()
