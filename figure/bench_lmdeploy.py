#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""B=1 decode TPOT/TTFT for the LMDeploy **W4A16KV4** arm (AWQ INT4 weights + 4-bit KV).

WHY THIS IS A SEPARATE FILE FROM `figure/bench.py`. Every arm in `bench.py` is a
`transformers` model driven step-by-step in this process, so it can be timed with CUDA events
around a single `model(...)` call. LMDeploy is not that: TurboMind is a C++ inference engine
behind an async request queue, there is no per-step Python call to wrap, and the only
per-token signal it exposes is the arrival of a streamed `Response`. Timing it therefore
means wall-clock timestamps on `stream_infer`, which is a **different measurement** from the
CUDA-event timings in `bench.py`, not a drop-in one.

WHAT THAT COSTS, STATED UP FRONT — both are recorded in the JSON as `meta.timing`:

  * **Wall-clock, not CUDA-event.** Each per-token time includes the engine's queue hop and
    the Python generator resume. That overhead is *additive and small relative to a decode
    step* but it is NOT zero, so a `lmdeploy` bar is an upper bound on the engine's kernel
    time in a way the CUDA-event arms are not.
  * **Chunks may carry more than one token.** TurboMind is free to coalesce. Each chunk
    carries a *cumulative* `generate_token_len`, so this file divides every inter-arrival gap
    by the number of tokens that gap actually delivered instead of assuming one. A run whose
    chunks were never coalesced records `chunk_tokens_max = 1`; anything higher is disclosed.

WHY THE ARM EXISTS AT ALL. It is the paper's `LMDeploy (W4A16KV4)` bar: the one baseline that
quantizes **weights** as well as KV, and the fastest bar in that figure. It was removed from
this repo once (a fairness audit: it was being drawn next to KV-only arms without saying so)
and is restored here with the asymmetry written into the label rather than into a footnote —
`arm_label` is literally `lmdeploy W4A16KV4 (AWQ-INT4 weights + 4-bit KV)`, and every other
arm in the figure runs **bf16 weights**. It is not an iso-weight comparison and must never be
captioned as one.

NOT COMPARABLE ON MEMORY. `peak_gb` is deliberately `null`: TurboMind allocates its KV pool
from its own C++ allocator sized by `cache_max_entry_count` (a *fraction of the whole GPU*),
so `torch.cuda.max_memory_allocated()` sees almost nothing and the pool size reflects the
flag, not the model. Reporting either number next to the torch arms' `peak_gb` would be a
fabrication. `meta.cache_max_entry_count` records the flag so the omission is auditable.

    python figure/bench_lmdeploy.py --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
        --ctxs 1024,4096,8192,16384,32768,65536 --out figure/data_h200/lmdeploy.json
"""
import argparse
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The prompt is built from ONE repeated token rather than random ids because the requirement is
# an EXACT input length: a random-id prompt has to be decoded to text (the only thing
# `stream_infer` accepts) and re-encoded, and that round trip is not length-preserving --
# byte-level BPE merges adjacent pieces, so 8192 random ids routinely come back as ~7900 or
# ~8400 tokens and the x-axis silently stops meaning what it says. A single token repeated is
# a fixed point of that round trip for every tokenizer this runs on, and it is verified below
# against the engine's OWN reported `input_token_len` rather than trusted.
FILLER = " the"


def _percentile(asc, q):
    """Linear-interpolated percentile of an ASCENDING list (same rule as figure/bench.py)."""
    if not asc:
        return None
    if len(asc) == 1:
        return asc[0]
    pos = q * (len(asc) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(asc) - 1)
    return asc[lo] + (asc[hi] - asc[lo]) * (pos - lo)


def build_prompt(tok, n_tokens):
    """A raw string that tokenizes to EXACTLY n_tokens under `tok`, or raise.

    Returned with the count this file measured; the engine's own count is checked against it
    per ctx and any disagreement is recorded, because `do_preprocess=False` is a promise about
    LMDeploy's behaviour that this file cannot verify from the outside.
    """
    ids = tok(FILLER * n_tokens, add_special_tokens=False).input_ids
    # Trim/extend to land exactly on n_tokens. FILLER is a single token for every BPE
    # tokenizer tried, so this converges in one step; the loop is the guard, not the mechanism.
    for _ in range(8):
        if len(ids) == n_tokens:
            return tok.decode(ids), len(ids)
        ids = (ids[:n_tokens] if len(ids) > n_tokens
               else ids + tok(FILLER * (n_tokens - len(ids)),
                              add_special_tokens=False).input_ids)
    raise RuntimeError(f"could not build an exact {n_tokens}-token prompt (got {len(ids)}); "
                       f"the FILLER token is not a fixed point under this tokenizer")


def run_ctx(pipe, gen_cls, prompt, warmup, measure_n, seed):
    """One (prefill ctx -> warmup+measure_n decode steps) generation, timed by token arrival.

    Returns the same field names figure/bench.py's measure() returns, so collect.py needs no
    special case -- EXCEPT peak_gb, which is None here (see the module docstring).
    """
    gen = gen_cls(max_new_tokens=warmup + measure_n, do_sample=False, temperature=1.0,
                  ignore_eos=True,           # a natural EOS would end the run early and the
                                             # measured window would silently shrink
                  random_seed=seed, skip_special_tokens=True)
    marks = []            # (arrival_time, cumulative_generated_tokens)
    inp_len = None
    finish = None
    t0 = time.perf_counter()
    for resp in pipe.stream_infer(prompt, gen_config=gen, do_preprocess=False,
                                  stream_response=True):
        marks.append((time.perf_counter(), int(getattr(resp, "generate_token_len", 0) or 0)))
        if getattr(resp, "input_token_len", None):
            inp_len = int(resp.input_token_len)
        if getattr(resp, "finish_reason", None):
            finish = resp.finish_reason
    if len(marks) < 2:
        raise RuntimeError(f"stream_infer yielded {len(marks)} chunk(s); nothing to time")

    # TTFT = prefill + first decode step, the same thing bench.py's ttft_ms measures.
    ttft_ms = (marks[0][0] - t0) * 1000.0
    # Per-token cost, chunk-size aware: a gap that delivered k tokens contributes k entries of
    # gap/k rather than one entry of gap. Assuming k==1 would inflate every coalesced step.
    per_tok, chunk_max = [], 1
    for (t_prev, n_prev), (t_now, n_now) in zip(marks, marks[1:]):
        k = n_now - n_prev
        if k <= 0:                       # a keep-alive/duplicate chunk delivered no token
            continue
        chunk_max = max(chunk_max, k)
        per_tok.extend([(t_now - t_prev) * 1000.0 / k] * k)
    if not per_tok:
        raise RuntimeError("no chunk delivered a token; cannot time this ctx")

    times = per_tok[warmup:] or per_tok    # drop the warmup prefix, as bench.py does
    asc = sorted(times)
    return {"ttft_ms": round(ttft_ms, 4),
            "tpot_ms": round(statistics.median(asc), 4),
            "mean_ms": round(sum(times) / len(times), 4),
            "p99_ms": round(_percentile(asc, 0.99), 4),
            "max_ms": round(asc[-1], 4),
            "min_ms": round(asc[0], 4),
            "peak_gb": None,               # not measurable here -- see the module docstring
            "engine_input_token_len": inp_len,
            "finish_reason": finish,
            "chunks": len(marks),
            "chunk_tokens_max": chunk_max,
            "cache": "turbomind-paged",
            "cache_class": "TurboMind",
            "times_ms": [round(t, 4) for t in times]}


def main():
    ap = argparse.ArgumentParser(
        description="B=1 decode TPOT/TTFT for the LMDeploy W4A16KV4 arm. Wall-clock timed "
                    "(TurboMind exposes no per-step call to wrap) -- see the module docstring "
                    "before comparing it with the CUDA-event-timed arms in figure/bench.py.")
    ap.add_argument("--model", default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                    help="AWQ-INT4 checkpoint (W4). The bf16 repo would silently give W16.")
    ap.add_argument("--ctxs", default="1024,4096,8192,16384,32768,65536")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--measure", type=int, default=160)
    ap.add_argument("--quant-policy", type=int, default=4,
                    help="4 = 4-bit KV (the paper's KV4). 0 would make this a W4A16 arm with a "
                         "16-bit KV cache and the label would be wrong.")
    ap.add_argument("--model-format", default="awq")
    ap.add_argument("--cache-max-entry-count", type=float, default=0.3,
                    help="fraction of GPU memory TurboMind reserves for its KV pool")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ctxs = [int(c) for c in args.ctxs.split(",") if c.strip()]
    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
    from lmdeploy.version import __version__ as lmd_version
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    session_len = max(ctxs) + args.warmup + args.measure + 64

    # ONE engine for every ctx: a per-ctx engine would re-pay the AWQ->TurboMind conversion
    # (minutes of CPU, not a hang) and would let engine-to-engine variation land in the
    # x-axis. session_len is sized for the LONGEST ctx so the same engine serves them all.
    backend = TurbomindEngineConfig(model_format=args.model_format,
                                    quant_policy=args.quant_policy,
                                    session_len=session_len,
                                    max_batch_size=1,
                                    cache_max_entry_count=args.cache_max_entry_count,
                                    tp=1)
    print(f"[lmdeploy] {lmd_version} model={args.model} quant_policy={args.quant_policy} "
          f"model_format={args.model_format} session_len={session_len}", flush=True)
    pipe = pipeline(args.model, backend_config=backend, log_level="ERROR")

    per_ctx = {}
    try:
        # Untimed warm run: the first request pays JIT/graph/pool warmup that belongs to no ctx.
        wp, _ = build_prompt(tok, min(ctxs))
        run_ctx(pipe, GenerationConfig, wp, 0, 8, args.seed)
        for ctx in ctxs:
            try:
                prompt, built = build_prompt(tok, ctx)
                cell = run_ctx(pipe, GenerationConfig, prompt, args.warmup, args.measure,
                               args.seed)
                cell["prompt_token_len"] = built
                # The engine's own count is the authority on what was prefilled. A mismatch
                # (chat template applied despite do_preprocess=False, BOS added, ...) means the
                # bar is not at the ctx it is plotted at, so it is recorded, not swallowed.
                eng = cell.get("engine_input_token_len")
                if eng is not None and eng != ctx:
                    cell["ctx_mismatch"] = {"requested": ctx, "engine_reported": eng}
                per_ctx[str(ctx)] = cell
                print(f"[lmdeploy] ctx={ctx:6d}  tpot={cell['tpot_ms']:8.3f}  "
                      f"ttft={cell['ttft_ms']:9.3f}  p99={cell['p99_ms']:8.3f}  "
                      f"chunk_max={cell['chunk_tokens_max']}  inp={eng}", flush=True)
            except Exception as e:  # noqa: BLE001
                per_ctx[str(ctx)] = {"error": repr(e)[:300]}
                print(f"[lmdeploy] ctx={ctx:6d}  ERROR {repr(e)[:300]}", flush=True)
    finally:
        try:
            pipe.close()
        except Exception:  # noqa: BLE001
            pass

    import torch
    doc = {
        "method": "lmdeploy",
        "arm_label": "lmdeploy W4A16KV4 (AWQ-INT4 weights + 4-bit KV)",
        "model": args.model,
        "caveat": ("W4A16KV4: this arm quantizes WEIGHTS (AWQ INT4) as well as the KV cache; "
                   "every other arm in the figure runs bf16 weights. Timed by wall-clock "
                   "token arrival from TurboMind's stream, not by CUDA events. peak_gb is "
                   "null on purpose (TurboMind's KV pool is sized by a GPU-memory fraction "
                   "flag, not by the model)."),
        "ctxs": per_ctx,
        "quality": None,
        "meta": {"warmup": args.warmup, "measure": args.measure,
                 "dtype": "float16",           # AWQ INT4 weights, fp16 activations
                 "seed": args.seed,
                 "timing": "wall-clock stream arrival (not CUDA events)",
                 "engine": "lmdeploy-turbomind",
                 "lmdeploy": lmd_version,
                 "quant_policy": args.quant_policy,
                 "model_format": args.model_format,
                 "cache_max_entry_count": args.cache_max_entry_count,
                 "session_len": session_len,
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "python": sys.version.split()[0],
                 "tag": args.tag},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2, default=str)
    print(f"BENCH_DONE method=lmdeploy out={args.out}", flush=True)


if __name__ == "__main__":
    main()
