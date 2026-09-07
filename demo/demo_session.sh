#!/usr/bin/env bash
# The recorded session: HF-eager arms, then vLLM arms, one long request each, then a summary.
# Env (interpreters, paths, recipe) comes from demo/run_all.sh.
R=${R:?repo root}; PY=${PY:?ommx python}; KP=${KP:?kivi/kitty python}; CTX=${CTX:-98304}; N=${N:-96}
RES=${RES:?results jsonl}; PROMPT=${PROMPT:-ommx}; : > "$RES"
arm() { "$@" || printf '\033[1;31m   arm exited with rc=%s (see the log)\033[0m\n' "$?"; }
if [ "$PROMPT" = ommx ]; then printf '\033[1mOMMX decode demo\033[0m  Llama-3.1-8B-Instruct · a 3K-token brief on OMMX, "explain it" · up to %s tokens\n' "$N"; else printf '\033[1mOMMX decode demo\033[0m  Llama-3.1-8B-Instruct · %sK-token document, three buried facts · up to %s tokens\n' "$((CTX/1024))" "$N"; fi
printf '\n\033[1m── HF-eager ──\033[0m  the model\047s own cache, one forward per token\n'
arm env PYTHONNOUSERSITE=1 PYTHONPATH="${KIVI_EXT:+$KIVI_EXT:}$R:$R/baseline:$R/ommx_gpu_serve/hf_eager" "$KP" demo/decode_demo.py --arm kivi_hf --prompt $PROMPT --ctx $CTX --new-tokens $N --results "$RES"
arm env PYTHONNOUSERSITE=1 PYTHONPATH="${KIVI_EXT:+$KIVI_EXT:}$R:$R/baseline:$R/ommx_gpu_serve/hf_eager" "$KP" demo/decode_demo.py --arm kivi_triton_hf --prompt $PROMPT --ctx $CTX --new-tokens $N --results "$RES"
arm env PYTHONNOUSERSITE=1 PYTHONPATH="$KITTY_PKG_PATH:$KITTY_TF:$R:$R/baseline/kitty" "$KP" demo/decode_demo.py --arm kitty_hf --prompt $PROMPT --ctx $CTX --new-tokens $N --results "$RES"
arm env PYTHONNOUSERSITE=1 PYTHONPATH="$R" "$PY" demo/decode_demo.py --arm bf16_hf --prompt $PROMPT --ctx $CTX --new-tokens $N --results "$RES"
arm env PYTHONNOUSERSITE=1 PYTHONPATH="$R:$R/ommx_gpu_serve/hf_eager" "$PY" demo/decode_demo.py --arm ommx_hf --prompt $PROMPT --ctx $CTX --new-tokens $N --results "$RES"
printf '\n\033[1m── vLLM ──\033[0m  paged KV, CUDA graph\n'
arm env PYTHONNOUSERSITE=1 PYTHONPATH="$R" "$PY" demo/decode_demo.py --arm vllm_bf16 --prompt $PROMPT --ctx $CTX --new-tokens $N --results "$RES"
arm env PYTHONNOUSERSITE=1 PYTHONPATH="$R" "$PY" demo/decode_demo.py --arm turboquant_vllm --prompt $PROMPT --ctx $CTX --new-tokens $N --results "$RES"
arm env PYTHONNOUSERSITE=1 PYTHONPATH="$R" "$PY" demo/decode_demo.py --arm ommx_vllm --prompt $PROMPT --ctx $CTX --new-tokens $N --results "$RES"
"$PY" - "$RES" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
kivi_t = next((r for r in rows if r["arm"] == "kivi_triton_hf"), None)
kivi_c = next((r for r in rows if r["arm"] == "kivi_hf"), None)
print("\n\033[1m   arm            TPOT p50   tok/s   vs KIVI-Tri  vs KIVI-CUDA  vs bf16*  TTFT    peak   facts\033[0m")
for grp, title in (("hf", "HF-eager"), ("vllm", "vLLM")):
    g = [r for r in rows if r["group"] == grp]
    base = next((r for r in g if r["arm"] in ("bf16_hf", "vllm_bf16")), None)
    print(f"   {title}")
    for r in g:
        tpot = r.get("tpot_p50_ms"); ttft = r.get("ttft_s")
        proven = bool(r.get("evidence_ok")) and isinstance(tpot, (int, float)) and tpot == tpot and tpot > 0
        peak = f"{r['peak_gb']:5.1f} GB" if r.get("peak_gb") is not None else "    -   "
        def ratio(ref):
            rt = ref.get("tpot_p50_ms") if ref else None
            ok = proven and ref and ref.get("evidence_ok") and isinstance(rt, (int, float)) and rt == rt and rt > 0
            return f"{rt/tpot:5.2f}x" if ok else "   -  "
        vt, vc, vb = ratio(kivi_t), ratio(kivi_c), ratio(base)
        ttft_s = f"{ttft:5.2f} s" if isinstance(ttft, (int, float)) else "   -   "
        mark = " " if proven else "!"
        tp = tpot if isinstance(tpot, (int, float)) else float("nan")
        ts = r.get("tok_s") if isinstance(r.get("tok_s"), (int, float)) else float("nan")
        print(f"    {mark}{r['name']:<12} {tp:7.1f} ms {ts:6.1f}   {vt}       {vc}      {vb}   {ttft_s}  {peak}  {r['facts']}/3")
    if any(not r.get("evidence_ok") for r in g):
        print("     ! = the arm could not prove its own kernel served the tokens; no ratio is drawn from it")
print("   vs KIVI-Tri = KIVI's Triton kernel, like-for-like with OMMX's Triton kernel; KIVI-CUDA = its CUDA kernel.")
print("   * vs the bf16 arm of the same framework. One request each; README numbers come from the benches.")
PYEOF
