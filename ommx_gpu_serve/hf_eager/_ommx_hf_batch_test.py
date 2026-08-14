"""OMMX canonical-KV-quant BATCHED (B>=1) decode TPOT + B=1 parity — HF-eager.

The B>1 sibling of baseline/kv/_ommx_hf_tpot_bench.py (the B=1 bench). It drives the
SAME ommx_paged_decode_attention_canonical kernel from an HF-eager full-model forward,
but through LlamaForCausalLM_OMMX_Batch (the shared MultiSeqKVPool path) so a UNIFORM
B={1,4,8,16} decode batch runs in ONE kernel launch over num_seqs=B — the production
batched serving shape, measured full-model the same way KIVI/Kitty/OMMX-B1 are.

SHADOW-FREE: OMMX_KV_RING=1 makes the pool's per-request bf16 k_hist/v_hist a RING of
only sink ∪ recent ∪ the live group (ring_cap rows/req, INDEPENDENT of ctx) instead of
the [num_seqs, max_seq_len, H, D] bf16 SHADOW that OOMs at B>1 long-ctx (the 64GB OOM).
The peak_gb column proves the collapse: it must stay ~flat as B grows (only the ≤3-bit
planes + the small ring scale with B), NOT blow up.

LAW (silent-fallback, CLAUDE.md §2.3 #5): we assert the batched OMMX kernel ACTUALLY
FIRED over num_seqs=B (NOT a B=1 python loop, NOT a bf16 fallback) by reading the
modeling's OMMX_BATCH_EVIDENCE marker counter back after the decode loop: fires>0 AND
max_B == B. A bf16 fallback / B=1 loop would leave max_B<B.

Recipe (set BEFORE importing the modeling — resolve_serving_config reads at construction):
  K_FORMAT=i2f4, OUTLIERS=3, GROUP_TOKENS=32, GROUP_CHANNELS=32, SINK=8, RECENT=32,
  OMMX_KV_RING=1 (shadow-free), OMMX_KV_GPU_PACK=1 (on-device pack).

Env: ommx_fakequant (torch>=2.4 / transformers>=4.51 / triton). The OMMX Triton decode
kernel JIT-builds on the first real CUDA call (absorbed by WARMUP). PYTHONPATH must
include the ommx-serve repo root.
Run:
  CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=<ommx-serve-root> \
      BENCH_BATCHES=1,4,8,16 BENCH_CTX=4096 python _ommx_hf_batch_test.py
"""
import os, sys, gc, json, statistics

# ── OMMX recipe env (mirror _ommx_hf_tpot_bench.py:30-40) ─────────────────────────────
os.environ.setdefault("OMMX_ATTN_K_FORMAT", "i2f4")    # K = INT2 base + FP4 outlier
os.environ.setdefault("OMMX_ATTN_OUTLIERS", "3")       # K outliers per 32-token group
os.environ.setdefault("OMMX_ATTN_OUTLIER_SELECT", "signed")
os.environ.setdefault("OMMX_ATTN_OUTLIER_REPR", "relidx7")
os.environ.setdefault("OMMX_KV_GROUP_TOKENS", "32")    # K token-group vector_length
os.environ.setdefault("OMMX_KV_GROUP_CHANNELS", "32")  # V channel-group vector_length
os.environ.setdefault("OMMX_KV_SINK", "8")             # bf16 sink rows
os.environ.setdefault("OMMX_KV_RECENT", "32")          # bf16 recent rows
os.environ.setdefault("OMMX_KV_RING", "1")             # SHADOW-FREE ring (no bf16 shadow)
os.environ.setdefault("OMMX_KV_GPU_PACK", "1")         # on-device pack (no D2H round-trip)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import torch
from transformers import LlamaConfig, AutoModelForCausalLM

# import the OMMX HF batched modeling (sits in ommx_hf/ next to this file)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ommx_hf"))
from _ommx_llama_modeling_batch import LlamaForCausalLM_OMMX_Batch, OMMX_BATCH_EVIDENCE

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BATCHES = [int(x) for x in os.environ.get("BENCH_BATCHES", "1,4,8,16").split(",") if x]
CTX = int(os.environ.get("BENCH_CTX", "4096"))
WARMUP, MEASURE = 5, 20
PARITY_STEPS = int(os.environ.get("PARITY_STEPS", "16"))   # B=1 greedy tokens to compare
DTYPE = torch.float16
DEV = "cuda"


def synced_batched_tpot(model, B, ctx):
    """Prefill B uniform-ctx sequences then time 1-tok/step BATCHED decode via CUDA events.

    All B requests step together (one model() call per step, q_len==1, batch=B), so the
    decode is exactly the production uniform-batch shape. Returns (median_ms, p99_ms,
    peak_gb)."""
    # size the per-layer pool planes to THIS ctx + decode steps (not 131072 max_pos_emb).
    os.environ["OMMX_ATTN_MAXCTX"] = str(ctx + WARMUP + MEASURE + 8)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    vocab = model.config.vocab_size
    torch.manual_seed(42)
    input_ids = torch.randint(low=10, high=vocab - 10, size=(B, ctx), device=DEV, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)        # [B, 1]
    del out
    torch.cuda.synchronize()
    times = []
    se = torch.cuda.Event(enable_timing=True); ee = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for step in range(WARMUP + MEASURE):
            torch.cuda.synchronize(); se.record()
            o = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            ee.record(); torch.cuda.synchronize()
            ms = se.elapsed_time(ee)
            past = o.past_key_values
            next_tok = o.logits[:, -1:, :].argmax(dim=-1)
            del o
            if step >= WARMUP:
                times.append(ms)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    times.sort()
    med = statistics.median(times)
    p99 = times[min(len(times) - 1, int(0.99 * len(times)))]
    return med, p99, peak


def b1_parity(ommx_model, bf16_model, ctx):
    """B=1 greedy parity: OMMX (batched-path, B=1) vs bf16 first-token + per-step match.

    Quantized KV is NOT bit-exact to bf16, so we report (a) the FIRST decode token match
    (the strongest single signal the batched B=1 path is correct, not garbage/fallback)
    and (b) the greedy-token match RATE over PARITY_STEPS + the first divergence step.
    A correct OMMX decode tracks bf16 closely; a broken pool/ring or a bf16 fallback would
    diverge at token 0."""
    torch.manual_seed(123)
    vocab = ommx_model.config.vocab_size
    ids = torch.randint(low=10, high=vocab - 10, size=(1, ctx), device=DEV, dtype=torch.long)

    def greedy(model, n):
        os.environ["OMMX_ATTN_MAXCTX"] = str(ctx + n + 8)
        toks = []
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            t = out.logits[:, -1:, :].argmax(dim=-1)
            toks.append(int(t.item())); del out
            for _ in range(n - 1):
                out = model(input_ids=t, past_key_values=past, use_cache=True)
                past = out.past_key_values
                t = out.logits[:, -1:, :].argmax(dim=-1)
                toks.append(int(t.item())); del out
        return toks

    o_toks = greedy(ommx_model, PARITY_STEPS)
    b_toks = greedy(bf16_model, PARITY_STEPS)
    first_match = (o_toks[0] == b_toks[0])
    matches = sum(int(a == b) for a, b in zip(o_toks, b_toks))
    diverge = next((i for i, (a, b) in enumerate(zip(o_toks, b_toks)) if a != b), PARITY_STEPS)
    return {
        "first_token_match": bool(first_match),
        "match_rate": matches / len(o_toks),
        "first_divergence_step": diverge,
        "ommx_tokens": o_toks, "bf16_tokens": b_toks,
    }


def load_ommx():
    # PREFILL attention only (decode uses the OMMX batched kernel). flash if available,
    # else sdpa (mem-efficient long-ctx); NOT eager (O(ctx^2) OOM).
    try:
        import flash_attn  # noqa
        prefill_attn = "flash_attention_2"
    except Exception:
        prefill_attn = "sdpa"
    cfg = LlamaConfig.from_pretrained(MODEL)
    cfg._attn_implementation = prefill_attn
    m = LlamaForCausalLM_OMMX_Batch.from_pretrained(
        MODEL, config=cfg, torch_dtype=DTYPE, low_cpu_mem_usage=True,
        attn_implementation=prefill_attn,
    ).to(DEV).eval()
    return m


def load_bf16():
    attn = "sdpa"
    try:
        import flash_attn  # noqa
        attn = "flash_attention_2"
    except Exception:
        attn = "sdpa"
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DTYPE, low_cpu_mem_usage=True, attn_implementation=attn).to(DEV).eval()
    return m


def free(m):
    del m; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def main():
    print("torch", torch.__version__, "device", torch.cuda.get_device_name(0), flush=True)
    import transformers
    print("transformers", transformers.__version__, "CUDA_VISIBLE_DEVICES",
          os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("OMMX recipe:",
          "k_fmt=" + os.environ["OMMX_ATTN_K_FORMAT"],
          "npv=" + os.environ["OMMX_ATTN_OUTLIERS"],
          "gt=" + os.environ["OMMX_KV_GROUP_TOKENS"],
          "gc=" + os.environ["OMMX_KV_GROUP_CHANNELS"],
          "sink=" + os.environ["OMMX_KV_SINK"],
          "recent=" + os.environ["OMMX_KV_RECENT"],
          "ring=" + os.environ["OMMX_KV_RING"],
          "ctx=" + str(CTX), "batches=" + ",".join(map(str, BATCHES)),
          flush=True)

    ommx = load_ommx()

    # ── B>=1 batched TPOT sweep ───────────────────────────────────────────────────────
    results = {}
    for B in BATCHES:
        # reset the fire evidence so we can assert THIS B's batched decode fired over B rows.
        OMMX_BATCH_EVIDENCE["fires"] = 0
        OMMX_BATCH_EVIDENCE["last_B"] = 0
        OMMX_BATCH_EVIDENCE["max_B"] = 0
        OMMX_BATCH_EVIDENCE["printed"] = False
        try:
            med, p99, peak = synced_batched_tpot(ommx, B, CTX)
            ev = dict(OMMX_BATCH_EVIDENCE)
            # LAW #5: the batched OMMX kernel must have fired over num_seqs == B.
            fired_over_B = (ev["fires"] > 0 and ev["max_B"] == B)
            results[B] = {"tpot_ms": med, "p99_ms": p99, "peak_gb": peak,
                          "fires": ev["fires"], "max_B": ev["max_B"],
                          "batched_fired_over_B": bool(fired_over_B)}
            tag = "OK" if fired_over_B else "!! NO-BATCHED-FIRE"
            print(f"[ommx] B={B:3d} ctx={CTX}  tpot={med:7.3f} ms  p99={p99:7.3f}  "
                  f"peak={peak:6.2f} GB  fires={ev['fires']} max_B={ev['max_B']}  {tag}",
                  flush=True)
            assert fired_over_B, (
                f"LAW #5 VIOLATION: B={B} batched OMMX decode did NOT fire over num_seqs=B "
                f"(fires={ev['fires']} max_B={ev['max_B']}) — silent bf16 fallback or B=1 loop")
        except torch.cuda.OutOfMemoryError:
            results[B] = {"tpot_ms": None, "oom": True}
            print(f"[ommx] B={B:3d} ctx={CTX}  OOM (SHADOW NOT free?)", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            results[B] = {"tpot_ms": None, "error": repr(e)[:300]}
            print(f"[ommx] B={B:3d} ctx={CTX}  ERROR {repr(e)[:300]}", flush=True)
            torch.cuda.empty_cache()

    # ── B=1 parity vs bf16 (greedy first-token + match rate) ──────────────────────────
    print("\n### B=1 parity: loading bf16 reference", flush=True)
    bf16 = load_bf16()
    try:
        parity = b1_parity(ommx, bf16, CTX)
        print(f"[parity B=1] first_token_match={parity['first_token_match']} "
              f"match_rate={parity['match_rate']:.3f} "
              f"first_divergence_step={parity['first_divergence_step']}/{PARITY_STEPS}",
              flush=True)
        print(f"[parity B=1] ommx_tokens={parity['ommx_tokens']}", flush=True)
        print(f"[parity B=1] bf16_tokens={parity['bf16_tokens']}", flush=True)
    except Exception as e:
        parity = {"error": repr(e)[:300]}
        print(f"[parity B=1] ERROR {repr(e)[:300]}", flush=True)
    free(bf16)
    free(ommx)

    # ── summary ───────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print(f"Llama-3.1-8B-Instruct  OMMX BATCHED decode TPOT (i2f4 K + i2 V, ring=SHADOW-FREE, "
          f"ctx={CTX}, warmup={WARMUP} measure={MEASURE}, fp16)")
    print("%5s | %11s | %11s | %9s | %7s | %s" %
          ("B", "tpot_ms", "p99_ms", "peak_gb", "fires", "batched_fired_over_B"))
    print("-" * 84)
    base_peak = None
    for B in BATCHES:
        r = results.get(B, {})
        if r.get("oom"):
            print(f"{B:>5} | {'OOM':>11} | {'--':>11} | {'--':>9} | {'--':>7} | --"); continue
        if r.get("tpot_ms") is None:
            print(f"{B:>5} | {'ERR':>11} | {'--':>11} | {'--':>9} | {'--':>7} | --"); continue
        if base_peak is None:
            base_peak = r["peak_gb"]
        print(f"{B:>5} | {r['tpot_ms']:>11.3f} | {r['p99_ms']:>11.3f} | {r['peak_gb']:>9.3f} "
              f"| {r['fires']:>7} | {r['batched_fired_over_B']}")
    print("=" * 84)
    # shadow-free signal: peak must NOT scale ~linearly with B (the bf16 shadow would).
    peaks = [results[B]["peak_gb"] for B in BATCHES if results.get(B, {}).get("peak_gb")]
    if len(peaks) >= 2:
        print(f"peak_gb growth (max/min over B) = {max(peaks) / max(1e-9, min(peaks)):.2f}x "
              f"(shadow-free => sublinear in B; a [B,max_seq] bf16 shadow would be ~Bx)")
    all_fired = all(results.get(B, {}).get("batched_fired_over_B")
                    for B in BATCHES if results.get(B, {}).get("tpot_ms"))
    print("ALL batched B fired the OMMX kernel over num_seqs=B:", all_fired)

    with open("_ommx_hf_batch_result.json", "w") as fh:
        json.dump({"ommx_batched": results, "b1_parity": parity, "model": MODEL,
                   "ctx": CTX, "batches": BATCHES,
                   "k_format": os.environ["OMMX_ATTN_K_FORMAT"],
                   "outliers": os.environ["OMMX_ATTN_OUTLIERS"],
                   "group_tokens": os.environ["OMMX_KV_GROUP_TOKENS"],
                   "sink": os.environ["OMMX_KV_SINK"], "recent": os.environ["OMMX_KV_RECENT"],
                   "ring": os.environ["OMMX_KV_RING"]}, fh, indent=2)
    print("OMMX_HF_BATCH_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
