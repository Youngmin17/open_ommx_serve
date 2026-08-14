#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real streaming generation for the side-by-side demo: 32k-context prefill + N-token decode,
logging per-token (text, cumulative wall-time) so demo_render.py can replay a timing-accurate
4-method race. Reuses figure/bench.py's per-method loaders (run each method in ITS OWN venv).

Usage (per method, in its venv):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<repo>:<repo>/baseline:<repo>/ommx_gpu_serve/hf_eager \
    python figure/demo_gen.py --method ommx --ctx 32768 --out-tokens 1024 \
      --out figure/data_demo/ommx.json
"""
import argparse
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
import bench as B  # noqa: E402  (figure/bench.py: load_model, measure helpers)

TASK = ("Explain OMMX (Outlier-Managed MX), a low-bit MX serving format: how its INT2-base "
        "dense region plus FP4 high-coverage outlier region (shared power-of-two scale, "
        "bundle-packed layout) and paged-decode kernel improve long-context LLM serving.")


def build_prompt_ids(tok, ctx):
    """A ctx-length prompt: a long technical passage (padding for the 32k context) followed by
    the task instruction, so the model streams an explanation under a real long-context load."""
    filler = ("In low-bit LLM inference, outliers expand the quantization range and lower the "
              "effective resolution of the dense majority of weights and KV-cache values. "
              "OMMX (Outlier-Managed MX) assigns MXINT2 to the dense region and MXFP4 to a "
              "high-coverage outlier region under a shared power-of-two scale, preserving a "
              "bundle-packed layout that a paged-decode kernel executes efficiently. ")
    ids = tok(filler * 4000, return_tensors="pt").input_ids[0]
    need = max(0, ctx - 64)
    ids = ids[:need]
    tail = tok("\n\n" + TASK + "\nAnswer:", return_tensors="pt").input_ids[0]
    import torch
    return torch.cat([ids, tail]).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["bf16", "kivi", "kitty", "ommx"])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--out-tokens", type=int, default=1024)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--live", action="store_true",
                    help="stream tokens to the terminal AS the kernel generates them (for a "
                         "genuine asciinema recording of the live inference), with a running tok/s")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    if args.method == "ommx":
        os.environ["OMMX_ATTN_MAXCTX"] = str(args.ctx + args.out_tokens + 16)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model_obj, kind, extra = B.load_model(args.method, args.model, dtype)

    # JIT warmup: compile the Triton kernels on a throwaway generation so the live stream is
    # not interrupted mid-word by the one-time first-call compile (the "O … <pause> … MMX" spike).
    # Warm at the REAL ctx so the exact decode-kernel specialization is compiled (a tiny-ctx warmup
    # can miss a ctx-dependent specialization). OMMX_DEMO_NOWARMUP=1 disables (for measuring the spike).
    import gc
    if os.environ.get("OMMX_DEMO_NOWARMUP", "0") != "1":
        with torch.no_grad():
            # Prefill at the FULL ctx: the split-KV decode kernel specializes on the KV length,
            # so a shorter warmup leaves a residual first-token JIT on a cold cache (measured
            # ~1.3s at 32k from a 64-token warmup). Full-ctx warmup precompiles the exact kernel.
            wctx = int(args.ctx)
            wids = torch.randint(10, model_obj.config.vocab_size - 10, (1, wctx), device="cuda", dtype=torch.long)
            wkv = extra["get_kv"](extra["cfg"], max_batch_size=1, max_length=wctx + 16) if kind == "kitty" else None
            wo = model_obj(input_ids=wids, past_key_values=wkv, use_cache=True) if wkv is not None \
                else model_obj(input_ids=wids, use_cache=True)
            wpast = wkv if wkv is not None else wo.past_key_values
            wnt = wo.logits[:, -1:, :].argmax(-1)
            for _ in range(6):
                wo2 = model_obj(input_ids=wnt, past_key_values=wpast, use_cache=True)
                if wkv is None:
                    wpast = wo2.past_key_values
                wnt = wo2.logits[:, -1:, :].argmax(-1)
        torch.cuda.synchronize()
        del wids, wo, wpast, wnt, wkv
        gc.collect(); torch.cuda.empty_cache()

    ids = build_prompt_ids(tok, args.ctx).cuda()
    kv = None
    if kind == "kitty":
        kv = extra["get_kv"](extra["cfg"], max_batch_size=1,
                             max_length=ids.shape[1] + args.out_tokens + 8)

    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model_obj(input_ids=ids, past_key_values=kv, use_cache=True) if kv is not None \
            else model_obj(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    past = kv if kv is not None else out.past_key_values
    nt = out.logits[:, -1:, :].argmax(-1)
    del out

    if args.live:
        import sys
        sys.stdout.write(f"\n\033[1;32m● {args.method.upper()}\033[0m  Llama-3.1-8B  "
                         f"ctx={args.ctx}  (live paged-decode kernel, real generation)\n"
                         f"\033[90m{'─'*72}\033[0m\n")
        sys.stdout.flush()

    events = []  # (piece_text, cumulative_seconds_since_first_decode)
    dec0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.out_tokens):
            tid = int(nt.item())
            if tid == tok.eos_token_id:
                break
            piece = tok.decode([tid], skip_special_tokens=True)
            now = time.perf_counter() - dec0
            events.append([piece, round(now, 5)])
            if args.live:
                import sys
                sys.stdout.write(piece); sys.stdout.flush()
                if len(events) % 24 == 0:
                    sys.stdout.write(f"\033[90m «{len(events)/now:.1f} tok/s»\033[0m")
                    sys.stdout.flush()
            o = model_obj(input_ids=nt, past_key_values=past, use_cache=True)
            if kv is None:
                past = o.past_key_values
            nt = o.logits[:, -1:, :].argmax(-1)
            del o
    torch.cuda.synchronize()
    total = time.perf_counter() - dec0
    n = len(events)
    # STEADY-STATE rate: drop the first W tokens (they carry the one-time Triton JIT-compile
    # spike, which otherwise makes the fast paths look slow at short output lengths).
    W = min(8, max(0, n // 4))
    if n - 1 > W and events[-1][1] > events[W][1]:
        tok_s = (n - 1 - W) / (events[-1][1] - events[W][1])
    else:
        tok_s = n / total if total > 0 else 0.0
    if args.live:
        import sys
        sys.stdout.write(f"\n\033[90m{'─'*72}\033[0m\n\033[1;32m✓ {args.method.upper()} done\033[0m — "
                         f"{n} tokens, \033[1m{tok_s:.1f} tok/s\033[0m steady-state "
                         f"(ttft {ttft:.1f}s @ {args.ctx} ctx)\n"); sys.stdout.flush()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"method": args.method, "ctx": args.ctx, "n_out": n,
                   "ttft_s": round(ttft, 4), "decode_s": round(total, 4),
                   "tok_per_s": round(tok_s, 3), "tok_per_s_raw": round(n / total, 3) if total > 0 else 0.0,
                   "jit_skip": W, "events": events}, fh)
    print(f"DEMO_DONE method={args.method} n={n} ttft={ttft:.2f}s "
          f"decode={total:.2f}s tok/s={tok_s:.2f}(steady) out={args.out}", flush=True)


if __name__ == "__main__":
    main()
