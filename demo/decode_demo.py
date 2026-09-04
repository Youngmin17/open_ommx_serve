#!/usr/bin/env python3
# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""One long-context request, streamed to the terminal, for one of seven KV-cache arms.

    python demo/decode_demo.py --arm ommx_vllm --ctx 98304 --new-tokens 96

The default prompt (--prompt ommx) is demo/ommx_brief.md, a ~3K-token brief on OMMX itself,
with a question that asks the model to explain the format and quote three facts from it (the
base format, the outlier format, the 64K H200 vLLM latency); --prompt needles instead builds a
--ctx-token filler document with three buried facts. Every generated token is printed
the moment it lands, with the time to first token, the per-token latency (p50 over the decode
steps), tokens/s, peak GPU memory (HF arms) and the evidence that the arm's own kernel ran.

  HF-eager (the model's own cache, one forward per token, figure/bench.py's loaders):
    kivi_hf    KIVI 2-bit KV, fp16, dequantize-and-dense path (KIVI's faster path here)
    kitty_hf   Kitty 2-bit KV, fp16, its Triton qk/sv kernels
    bf16_hf    uncompressed bf16 KV, DynamicCache + FlashAttention-2
    ommx_hf    OMMX i2f4 KV, the same Triton decode kernel vLLM runs
  vLLM (paged KV, CUDA graph, bench_e2e_a100.py's engine construction):
    vllm_bf16        bf16 KV, the engine's default FlashAttention backend
    turboquant_vllm  TurboQuant 3-bit KV (kv_cache_dtype=turboquant_3bit_nc, block 32)
    ommx_vllm        OMMX i2f4 KV (MXINT2 + MXFP4 outliers), CUSTOM attention backend

Nothing here is a benchmark: warm-up is one prompt, the sample is one request. The README's
numbers come from the benches. Appends one JSON line per run to --results.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import logging
import os
import re
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _k, _v in (("TRANSFORMERS_VERBOSITY", "error"), ("HF_HUB_DISABLE_PROGRESS_BARS", "1"),
               ("TOKENIZERS_PARALLELISM", "false"), ("PYTHONWARNINGS", "ignore")):
    os.environ.setdefault(_k, _v)
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

GROUP = {"kivi_hf": "hf", "kitty_hf": "hf", "bf16_hf": "hf", "ommx_hf": "hf",
         "vllm_bf16": "vllm", "turboquant_vllm": "vllm", "ommx_vllm": "vllm"}
LABEL = {
    "kivi_hf": ("KIVI", "2-bit KV · fp16 · dequantize-and-dense path"),
    "kitty_hf": ("Kitty", "2-bit KV · fp16 · Triton qk/sv kernels"),
    "bf16_hf": ("bf16", "uncompressed KV · DynamicCache + FlashAttention-2"),
    "ommx_hf": ("OMMX", "MXINT2 + MXFP4-outlier KV · bitmap positions · Triton decode kernel"),
    "vllm_bf16": ("vLLM bf16", "uncompressed KV · FlashAttention (engine default)"),
    "turboquant_vllm": ("TurboQuant", "3-bit KV · kv_cache_dtype=turboquant_3bit_nc"),
    "ommx_vllm": ("OMMX", "MXINT2 + MXFP4-outlier KV · CUSTOM backend · paged · CUDA graph"),
}
PROMPT_MODE = ["ommx"]
C = {"cyan": "\033[1;36m", "green": "\033[32m", "yellow": "\033[33m", "dim": "\033[2m",
     "bold": "\033[1m", "red": "\033[1;31m", "off": "\033[0m"}
FACTS_NEEDLES = {"codename": "BLUE HERON", "meeting": "Thursday at 4 pm in room 7B", "password": "74915"}
#: the OMMX brief: three answers the question asks for explicitly, checked as substrings
FACTS_OMMX = {"base format": "MXINT2", "outlier format": "MXFP4", "64K H200 vLLM TPOT": "22.6"}
FACTS = dict(FACTS_OMMX)
QUESTION_OMMX = ("Using only the brief above, explain OMMX to an engineer in about 300 words: what the "
                 "format is (name the base format and the outlier format), how the serving recipe "
                 "packs the KV cache, how the decode kernel recovers values, what OMMX measured on the "
                 "H200 at 64K tokens inside vLLM (give the per-token latency in ms), and the one "
                 "limitation the brief is most explicit about.")


def say(s="", end="\n"):
    sys.stdout.write(s + end)
    sys.stdout.flush()


def _load_by_path(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Captured:
    """stdout/stderr into a real temporary file while something noisy loads (vLLM calls
    ``sys.stdout.fileno()``, so a StringIO would break it); the text is kept so evidence
    lines (the backend the engine selected) can be pulled out afterwards."""

    def __init__(self):
        import tempfile
        self.f = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        self.text = ""

    def __enter__(self):
        sys.stdout.flush(); sys.stderr.flush()
        self._o, self._e = sys.stdout, sys.stderr
        self._fd1, self._fd2 = os.dup(1), os.dup(2)
        os.dup2(self.f.fileno(), 1); os.dup2(self.f.fileno(), 2)
        sys.stdout = sys.stderr = self.f
        return self

    def __exit__(self, *exc):
        self.f.flush()
        os.dup2(self._fd1, 1); os.dup2(self._fd2, 2)
        os.close(self._fd1); os.close(self._fd2)
        sys.stdout, sys.stderr = self._o, self._e
        self.f.seek(0); self.text = self.f.read()
        return False

    def grep(self, pattern):
        return [l for l in self.text.splitlines() if re.search(pattern, l)]


FILLER = (
    "The harbour town wakes early; fishing boats leave before the bakeries open. ",
    "By mid-morning the market square fills with stalls of citrus, olives and bolts of cloth. ",
    "The old tram still runs along the seafront, slower than walking on a windy day. ",
    "Schoolchildren cut through the botanical garden, where the gardeners are pruning the figs. ",
    "In the afternoon the light turns copper and the shutters on the west side come down. ",
    "The library keeps its reading room open late for the students preparing for exams. ",
    "A ferry crosses to the island twice a day, weather permitting, carrying mail and goats. ",
    "The lighthouse keeper writes the tide tables by hand and posts them at the pier. ",
    "On Sundays the brass band plays in the bandstand and the cafes run out of chairs. ",
    "Fog settles over the estuary at night, and the foghorn sounds every ninety seconds. ",
    "The town council meets in the stone hall whose clock has been ten minutes fast for years. ",
    "Rain arrives in October, and the streets that were dust become small rivers for a week. ",
)
NEEDLES = [
    (0.15, f"Note to self: the client's project codename is {FACTS_NEEDLES['codename']}. "),
    (0.50, f"Reminder: the review meeting is on {FACTS_NEEDLES['meeting']}. "),
    (0.85, f"The password for the shared drive is {FACTS_NEEDLES['password']}. "),
]
QUESTION = ("Using only the notes in the document above, answer in two sentences: what is the "
            "project codename, when and where is the review meeting, and what is the shared-drive "
            "password? Then add one sentence on why keeping such details in one long document "
            "is risky.")


def build_prompt_ids_ommx(tok):
    """The OMMX brief (demo/ommx_brief.md) + a question that asks for three checkable facts."""
    doc = open(os.path.join(REPO, "demo", "ommx_brief.md"), encoding="utf-8").read()
    msgs = [{"role": "system", "content": "You are a careful engineer who explains from the document given."},
            {"role": "user", "content": "Brief:\n" + doc + "\n\n" + QUESTION_OMMX}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


def build_prompt_ids(tok, ctx, mode="ommx"):
    if mode == "ommx":
        return build_prompt_ids_ommx(tok)
    return build_prompt_ids_needles(tok, ctx)


def build_prompt_ids_needles(tok, ctx):
    """Chat-templated filler document with three buried facts, tuned to within ~0.5 % of ctx tokens."""
    def render(n_sent):
        sents = [FILLER[i % len(FILLER)] for i in range(n_sent)]
        for depth, needle in NEEDLES:
            sents.insert(int(depth * len(sents)), needle)
        doc = "".join(sents)
        msgs = [{"role": "system", "content": "You are a careful assistant who answers from the document."},
                {"role": "user", "content": "Document:\n" + doc + "\n\n" + QUESTION}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False).input_ids
    per = len(tok("".join(FILLER), add_special_tokens=False).input_ids) / len(FILLER)
    n = max(8, int(ctx / per))
    ids = render(n)
    for _ in range(4):
        if abs(len(ids) - ctx) <= max(64, ctx // 200):
            break
        n = max(8, int(n * ctx / len(ids)))
        ids = render(n)
    return ids


class Streamer:
    """Prints token text as it arrives and keeps the per-step times."""

    def __init__(self, tok):
        self.tok, self.ids, self.shown, self.times = tok, [], "", []
        self.t_first = None; self.t_last = None

    def push(self, token_id, now):
        if self.t_first is None:
            self.t_first = now
        else:
            self.times.append(now - self.t_last)
        self.t_last = now
        self.ids.append(int(token_id))
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if text.startswith(self.shown):
            say(text[len(self.shown):], end="")
            self.shown = text


def facts_found(text):
    got = {k: (v.lower() in text.lower()) for k, v in FACTS.items()}
    return sum(got.values()), got


def set_prompt_mode(mode):
    FACTS.clear()
    FACTS.update(FACTS_OMMX if mode == "ommx" else FACTS_NEEDLES)


def _prefill(model_obj, x, kv):
    """One forward over the prompt keeping only the last position's logits. The full
    [1, ctx, vocab] logits tensor is 25 GB at 98K tokens and would dominate peak memory and
    TTFT for reasons that have nothing to do with the KV cache. Runs the decoder trunk and
    applies lm_head to the last hidden state, which every arm's model class (official KIVI
    and Kitty included) exposes as ``.model`` / ``.lm_head``; falls back to the model's own
    ``logits_to_keep`` when the trunk is not exposed."""
    import torch
    kw = dict(past_key_values=kv) if kv is not None else {}
    trunk, head = getattr(model_obj, "model", None), getattr(model_obj, "lm_head", None)
    if trunk is not None and head is not None:
        out = trunk(input_ids=x, use_cache=True, **kw)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        logits = head(h[:, -1:, :])
        return type("O", (), {"logits": logits, "past_key_values": getattr(out, "past_key_values", kv)})()
    for key in ("logits_to_keep", "num_logits_to_keep"):
        try:
            return model_obj(input_ids=x, use_cache=True, **{key: 1}, **kw)
        except TypeError:
            continue
    return model_obj(input_ids=x, use_cache=True, **kw)


def run_hf(arm, model, ctx, n):
    import torch
    bench = _load_by_path("ommx_hf_bench", "figure/bench.py")
    method = {"kivi_hf": "kivi", "kitty_hf": "kitty", "bf16_hf": "bf16", "ommx_hf": "ommx"}[arm]
    dtype = torch.float16 if method in ("kivi", "kitty") else torch.bfloat16
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    t0 = time.perf_counter()
    say(f"   loading {C['dim']}({'fp16' if dtype == torch.float16 else 'bf16'})…{C['off']} ", end="")
    with Captured():
        model_obj, kind, extra = bench.load_model(method, model, dtype)
        torch.cuda.synchronize()
    say(f"{C['green']}ready{C['off']} {time.perf_counter() - t0:.1f} s")
    ids = build_prompt_ids(tok, ctx, PROMPT_MODE[0])
    x = torch.tensor([ids], device="cuda")
    if method == "ommx":
        os.environ["OMMX_ATTN_MAXCTX"] = str(len(ids) + n + 8)
        from ommx_gpu_serve.attention import paged_decode as pd
        pd.reset_launch_stats()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    kv = None
    if kind == "kitty":
        kv = extra["get_kv"](extra["cfg"], max_batch_size=1, max_length=len(ids) + n + 8)
    say(f"   prefill {len(ids):,} tokens… ", end="")
    st = Streamer(tok)
    with torch.no_grad():
        t0 = time.perf_counter()
        out = _prefill(model_obj, x, kv)
        past = kv if kv is not None else out.past_key_values
        nt = out.logits[:, -1:, :].argmax(-1)
        torch.cuda.synchronize(); ttft = time.perf_counter() - t0
        say(f"TTFT {C['bold']}{ttft:.2f} s{C['off']}")
        say(f" {C['yellow']}▶{C['off']} ", end="")
        st.push(nt.item(), time.perf_counter())
        for _ in range(n - 1):
            if int(nt.item()) in (tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")):
                break
            o = model_obj(input_ids=nt, past_key_values=past, use_cache=True)
            if kv is None:
                past = o.past_key_values
            nt = o.logits[:, -1:, :].argmax(-1)
            torch.cuda.synchronize()
            st.push(nt.item(), time.perf_counter())
    say()
    # evidence that the arm's own kernel served the tokens
    if method == "ommx":
        ls = pd.launch_stats()
        abl = pd.abl_flag_names(ls.get("abl", 0))
        ev = f"paged_decode launches={ls['stage1']} bitmap={ls['bitmap']}" + (f" ABLATION={abl}" if abl else "")
        ok = ls["stage1"] >= 1 and ls["bitmap"] == ls["stage1"] and not abl
    elif method == "kivi":
        stats = getattr(bench, "_kivi_path_stats", lambda: None)()
        path = (stats or {}).get("path") if isinstance(stats, dict) else None
        ev = f"kivi path={path or 'dequant-fallback'}"; ok = True
    elif method == "kitty":
        ev = f"cache={type(kv).__name__}"; ok = "Kitty" in type(kv).__name__
    else:
        ev = f"cache={type(past).__name__}"; ok = True
    return dict(ttft=ttft, times=st.times, peak=torch.cuda.max_memory_allocated() / 2**30,
                evidence=ev, evidence_ok=ok, text=st.shown, n_prompt=len(ids))


def run_vllm(arm, model, ctx, n, max_len):
    os.environ.setdefault("VLLM_PLUGINS", "ommx_gpu_serve")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if arm == "ommx_vllm":
        os.environ["OMMX_ATTN_GRAPH"] = "1"        # the route the published vLLM bars ran
    fire = os.environ.setdefault("OMMX_FIRE_FILE", "/tmp/ommx_demo.fire.log")
    with contextlib.suppress(FileNotFoundError):
        os.remove(fire)
    import torch
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    t0 = time.perf_counter()
    say(f"   building the engine {C['dim']}(model load, CUDA-graph capture)…{C['off']} ", end="")
    cap = Captured()
    with cap:
        e2e = _load_by_path("ommx_e2e_bench", "ommx_gpu_serve/bench/bench_e2e_a100.py")
        from transformers import AutoTokenizer
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        backend, kv_dtype, block = {"vllm_bf16": ("", "", 0), "turboquant_vllm": ("", "turboquant_3bit_nc", 32),
                                    "ommx_vllm": ("CUSTOM", "", 0)}[arm]
        eng = e2e._make_engine(model, backend or None, "", max_len, 0.9, 42, False,
                               kv_dtype=kv_dtype, max_num_seqs=1, block_size=block)
        tok = AutoTokenizer.from_pretrained(model)
    say(f"{C['green']}ready{C['off']} {time.perf_counter() - t0:.1f} s")
    # the engine's own log is provenance for the evidence line below; keep it next to the results
    res = os.environ.get("OMMX_DEMO_RESULTS")
    if res:
        with contextlib.suppress(OSError), open(os.path.join(os.path.dirname(res), f"engine_{arm}.log"), "w") as fh:
            fh.write(cap.text)
    logging.disable(logging.WARNING)            # JIT-compile notices during the first steps
    ids = build_prompt_ids(tok, ctx, PROMPT_MODE[0])
    say(f"   prefill {len(ids):,} tokens… ", end="")
    st = Streamer(tok)
    eng.add_request("demo", TokensPrompt(prompt_token_ids=ids),
                    SamplingParams(temperature=0.0, max_tokens=n))
    t0 = time.perf_counter(); seen = 0; ttft = None
    while eng.has_unfinished_requests():
        outs = eng.step(); now = time.perf_counter()
        for o in outs:
            toks = list(o.outputs[0].token_ids)
            for t in toks[seen:]:
                if ttft is None:
                    ttft = now - t0
                    say(f"TTFT {C['bold']}{ttft:.2f} s{C['off']}")
                    say(f" {C['yellow']}▶{C['off']} ", end="")
                st.push(t, now)
            seen = len(toks)
    say()
    cc = eng.vllm_config.cache_config
    if arm == "ommx_vllm":
        # the bench's own reader: FIRED whitelist, *_DEAD / *_NOFIRE refusal, then the
        # in-process health check for a degrade AFTER the route fired (law #5)
        ev_d = e2e._read_fire_evidence(fire, require_attn=True)
        from ommx_gpu_serve.integration.vllm.backend import ommx_route_health
        hz = ommx_route_health()
        tags = [l.strip() for l in open(fire)] if os.path.exists(fire) else []
        fired = [t for t in tags if "ROUTE_FIRED" in t]
        detail = fired[0].split("pid=")[-1].split(" ", 1)[-1] if fired else ""
        abl = [t for t in tags if "ABL_ATTN_ACTIVE" in t]
        ok = bool(ev_d.get("ok")) and not hz.get("degraded") and not abl
        ev = ("route " + detail + f" blk={cc.block_size}") if ok else (
            f"route NOT PROVEN ({ev_d.get('reason') or hz.get('reason') or (abl and 'ablation active')})")
    else:
        # vllm/platforms/cuda.py logs `Using %s backend.` once the backend is chosen
        # vllm/platforms/cuda.py: "Using <AttentionBackendEnum.X | X> backend." -- take the
        # last selection line; fall back to any backend-name token on a line naming a backend
        picked = []
        for l in cap.text.splitlines():
            m = (re.search(r"[Uu]sing\s+(\S+?)\s+(?:attention\s+)?backend", l)
                 or (re.search(r"(FLASH_ATTN|TRITON_ATTN|TURBOQUANT|FLASHINFER|CUSTOM)", l) if "backend" in l.lower() else None))
            if m:
                picked.append(m.group(1).split(".")[-1])
        name = picked[-1] if picked else "not-logged"
        ev = f"backend={name} kv={cc.cache_dtype} blk={cc.block_size}"
        ok = (arm != "turboquant_vllm") or str(cc.cache_dtype).startswith("turboquant")
    return dict(ttft=ttft, times=st.times, peak=None, evidence=ev, evidence_ok=ok, text=st.shown,
                n_prompt=len(ids))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(LABEL))
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", choices=("ommx", "needles"), default="ommx",
                    help="ommx: the OMMX brief + an explain-it question; needles: filler with three buried facts")
    ap.add_argument("--ctx", type=int, default=98304, help="needles mode only: prompt length to build")
    ap.add_argument("--new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=131072, help="vLLM engine max_model_len")
    ap.add_argument("--results", default=None, help="append one JSON line here")
    a = ap.parse_args()
    PROMPT_MODE[0] = a.prompt
    set_prompt_mode(a.prompt)
    if a.results:
        os.environ["OMMX_DEMO_RESULTS"] = a.results
    import torch
    name, sub = LABEL[a.arm]
    gpu = torch.cuda.get_device_name(0)
    say(f"\n{C['cyan']}━━ {name}{C['off']}  {C['dim']}{sub}{C['off']}")
    say(f"   {a.model.split('/')[-1]}   {gpu}")
    r = run_vllm(a.arm, a.model, a.ctx, a.new_tokens, a.max_model_len) if GROUP[a.arm] == "vllm" \
        else run_hf(a.arm, a.model, a.ctx, a.new_tokens)
    times = r["times"]
    p50 = statistics.median(times) * 1000 if times else float("nan")
    tps = 1000.0 / p50 if p50 == p50 else float("nan")
    nf, got = facts_found(r["text"])
    line = (f"   {len(times) + 1} tokens   TPOT p50 {C['bold']}{p50:.1f} ms{C['off']}   {tps:.1f} tok/s"
            + (f"   peak {r['peak']:.1f} GB" if r["peak"] is not None else "")
            + f"   facts {C['green'] if nf == 3 else C['red']}{nf}/3{C['off']}"
            + f"   {C['green'] if r['evidence_ok'] else C['red']}{r['evidence']}{C['off']}")
    say(line)
    if a.results:
        os.environ["OMMX_DEMO_RESULTS"] = a.results
        with open(a.results, "a") as f:
            f.write(json.dumps({"arm": a.arm, "group": GROUP[a.arm], "name": name, "ctx": r["n_prompt"],
                                "ttft_s": r["ttft"], "tpot_p50_ms": p50, "tok_s": tps, "peak_gb": r["peak"],
                                "facts": nf, "facts_detail": got, "evidence": r["evidence"],
                                "evidence_ok": r["evidence_ok"], "gpu": gpu, "n": len(times) + 1,
                                "text": r["text"]}) + "\n")
    sys.stdout.flush(); sys.stderr.flush()
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)     # teardown chatter is not part of the demo


if __name__ == "__main__":
    main()
