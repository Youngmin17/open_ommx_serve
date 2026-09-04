#!/usr/bin/env bash
# The recorded session: HF-eager arms, then vLLM arms, one long request each, then a summary.
# Env (interpreters, paths, recipe) comes from demo/run_all.sh.
R=${R:?repo root}; PY=${PY:?ommx python}; KP=${KP:?kivi/kitty python}; CTX=${CTX:-98304}; N=${N:-96}
RES=${RES:?results jsonl}; PROMPT=${PROMPT:-ommx}; : > "$RES"
arm() { "$@" || printf '\033[1;31m   arm exited with rc=%s (see the log)\033[0m\n' "$?"; }
if [ "$PROMPT" = ommx ]; then printf '\033[1mOMMX decode demo\033[0m  Llama-3.1-8B-Instruct · a 3K-token brief on OMMX, "explain it" · up to %s tokens\n' "$N"; else printf '\033[1mOMMX decode demo\033[0m  Llama-3.1-8B-Instruct · %sK-token document, three buried facts · up to %s tokens\n' "$((CTX/1024))" "$N"; fi
printf '\n\033[1m── HF-eager ──\033[0m  the model\047s own cache, one forward per token\n'
arm env PYTHONNOUSERSITE=1 PYTHONPATH="$R:$R/baseline:$R/ommx_gpu_serve/hf_eager" "$KP" demo/decode_demo.py --arm kivi_hf --prompt $PROMPT --ctx $CTX --new-tokens $N --results "$RES"
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
kivi = next((r for r in rows if r["arm"] == "kivi_hf"), None)
print("\n\033[1m   arm            TPOT p50   tok/s   vs KIVI  vs bf16*  TTFT    peak   facts\033[0m")
for grp, title in (("hf", "HF-eager"), ("vllm", "vLLM")):
    g = [r for r in rows if r["group"] == grp]
    base = next((r for r in g if r["arm"] in ("bf16_hf", "vllm_bf16")), None)
    print(f"   {title}")
    for r in g:
        peak = f"{r['peak_gb']:5.1f} GB" if r.get("peak_gb") is not None else "    -   "
        vk = f"{kivi['tpot_p50_ms']/r['tpot_p50_ms']:5.2f}x" if kivi else "   -  "
        vb = f"{base['tpot_p50_ms']/r['tpot_p50_ms']:5.2f}x" if base else "   -  "
        print(f"     {r['name']:<12} {r['tpot_p50_ms']:7.1f} ms {r['tok_s']:6.1f}   {vk}   {vb}   {r['ttft_s']:5.2f} s  {peak}  {r['facts']}/3")
print("   * vs the bf16 arm of the same framework. One request each; README numbers come from the benches.")
PYEOF
