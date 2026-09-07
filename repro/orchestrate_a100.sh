#!/usr/bin/env bash
# A100 reproduction (installable subset: bf16/ommx/kivi) on one GPU, writing figure/data_<tag>/<m>_hf.json then collect+plot. This is the concrete
# per-venv instance behind fig2; ../run.sh is the portable single-venv entry point.
#
# Each leg renames the JSON it owns to <file>.prev BEFORE running, so a leg that fails leaves
# no collectable file (nothing stale gets drawn as this run's bar) and nothing is destroyed
# either. Files this run did not produce are named on a "[collect] NOTE" line before the
# figure step rather than silently folded into the figure.
#
# Override before running (defaults assume per-method venvs under $HOME):
#   REPO     repo root                    (default: parent of this script)
#   OMMXPY   python for the ommx venv     (default: $HOME/.venv-ommx/bin/python)
#   KIVIPY   python for the kivi venv     (default: $HOME/.venv-kivi/bin/python)
#   DTYPE      compute dtype for the bf16/ommx arms   (default: bfloat16)
#   DTYPE_KIVI compute dtype for the KIVI arm         (default: float16)
#
# WHY THE KIVI ARM IS PINNED TO fp16. figure/bench.py's --dtype default is bfloat16 (so the arm
# plotted as "bf16" really is bf16), but KIVI is fp16-ONLY: baseline/kivi/quant/new_pack.py
# casts every dequantized K/V to torch.float16 and quant/matmul.py's triton_bmm_fA_qB_outer
# -- the fused kernel models/llama_kivi_eval.py actually imports -- allocates an fp16 output,
# so a bfloat16 KIVI run raises a dtype mismatch instead of a number. The figure is therefore
# MIXED dtype; that is recorded per arm in the JSON (meta.dtype + arm_label) and
# printed in the plot legend. See repro/README.md "Which arm runs in which dtype".
set -o pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CTXS=${CTXS:-1024,4096,8192,16384,32768,65536}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
OMMXPY="${OMMXPY:-$HOME/.venv-ommx/bin/python}"
KIVIPY="${KIVIPY:-$HOME/.venv-kivi/bin/python}"
DTYPE=${DTYPE:-bfloat16}
DTYPE_KIVI=${DTYPE_KIVI:-float16}     # fp16-only packer; see the header
TAG=${TAG:-a100}
DATA="$REPO/figure/data_$TAG"; cd "$REPO"; mkdir -p "$DATA"
echo "=== REPRO A100  CVD=$CUDA_VISIBLE_DEVICES  ctxs=$CTXS  $(date -u) ==="
echo "=== dtype: bf16/ommx=$DTYPE  kivi=$DTYPE_KIVI (mixed = a fairness caveat; recorded per"
echo "===        arm and printed in the plot legend) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null

step(){ echo; echo "######## $1 @ $(date -u +%H:%M:%S) ########"; }

# STALE OUTPUTS ARE MOVED ASIDE BEFORE THE LEG THAT OWNS THEM, NEVER DELETED. figure/data_<tag>/
# is keyed by FILENAME, so a file left behind by an EARLIER run -- or one of the repo's shipped,
# git-tracked historical JSONs, which live under exactly these names -- is indistinguishable
# from one this run produced: collect.py picks it up and plot.py draws it as this run's bar.
# This script used to leave every one of them in place, so a leg that failed still got a bar.
# Renaming to <file>.prev clears the name without costing anyone the file: collect.py's
# FILE2KEY is an EXACT filename map, so a .prev file is never collected. Only one generation is
# kept; a second run overwrites the .prev. Same rule and same suffix as ../run.sh.
PRODUCED=""        # basenames this run OWNS (see the collect-scope note before the figure step)
stash_stale(){
  local f
  for f in "$@"; do
    PRODUCED="$PRODUCED $(basename "$f")"   # owned even if the leg then fails and writes nothing
    [ -e "$f" ] || continue
    mv -f "$f" "$f.prev" || { echo "!! cannot move $f aside -- NOT running this leg, because a"
                              echo "   stale file left in place is collected as this run's"
                              echo "   result. Move it yourself and re-run."; return 1; }
    echo "   stale $(basename "$f") -> $(basename "$f").prev (kept; .prev is never collected)"
  done
}
summ(){ "$OMMXPY" - "$1" <<'PY' 2>/dev/null || true
import json,sys
d=json.load(open(sys.argv[1]))
# arm_label carries the RESOLVED dtype ("kivi (fp16)"), so a mislabelled arm is visible here.
print("  method:",d.get("method"),"arm_label:",d.get("arm_label"),
      "dtype:",(d.get("meta") or {}).get("dtype"))
for c,v in (d.get("ctxs") or {}).items():
    print(f"    ctx={c}: tpot={v.get('tpot_ms')} ttft={v.get('ttft_ms')} peak={v.get('peak_gb')}")
q=d.get("quality") or {}; print("  quality:",repr((q.get("output") or q.get("error") or "")[:160]))
PY
}

# Every leg's exit status is CHECKED and carried to the script's own exit status. Unchecked,
# this script printed ORCHESTRATE_<tag>_DONE and exited 0 after EVERY leg had failed (a wrong
# $OMMXPY, a missing conda env, a CPU-only host: "python: command not found" x5), which is a
# success marker for a run that measured nothing -- the same defect ../run.sh's figure leg had.
# `set -o pipefail` is already on, so the `| tail -14` pipelines report the bench's rc, not
# tail's. A leg that fails is NOT fatal here on purpose (the other arms are still worth
# having); it only has to be visible and to move the exit status.
RC=0
export PYTHONPATH="$REPO:$REPO/ommx_gpu_serve/hf_eager"
step bf16; stash_stale "$DATA/bf16_hf.json" && "$OMMXPY" figure/bench.py --method bf16 --model "$MODEL" --ctxs "$CTXS" --dtype "$DTYPE" --out "$DATA/bf16_hf.json" 2>&1 | tail -14 || { echo "!! [bf16] FAILED"; RC=1; }; summ "$DATA/bf16_hf.json"
step ommx; stash_stale "$DATA/ommx_hf.json" && "$OMMXPY" figure/bench.py --method ommx --model "$MODEL" --ctxs "$CTXS" --dtype "$DTYPE" --out "$DATA/ommx_hf.json" 2>&1 | tail -14 || { echo "!! [ommx] FAILED"; RC=1; }; summ "$DATA/ommx_hf.json"

export PYTHONPATH="$REPO/baseline:$REPO"
step kivi; stash_stale "$DATA/kivi_hf.json" && "$KIVIPY" figure/bench.py --method kivi --model "$MODEL" --ctxs "$CTXS" --dtype "$DTYPE_KIVI" --out "$DATA/kivi_hf.json" 2>&1 | tail -14 || { echo "!! [kivi] FAILED (dtype=$DTYPE_KIVI)"; RC=1; }; summ "$DATA/kivi_hf.json"

step figure
# COLLECT SCOPE. figure/collect.py reads the WHOLE $DATA directory by FILENAME, while this
# script owns only the three HF-eager legs above. Anything else with a collectable name came
# from a DIFFERENT run -- the vLLM arms (../run.sh bench), the kitty arm this script does not
# run at all, or one of the shipped historical JSONs repro/README.md declares uncitable -- and
# is drawn into this figure with nothing on the plot to say so. Name them here; do not delete
# them (they may be exactly what the operator wants combined), just do not let them pass as
# this run's measurement.
for f in bf16_hf.json ommx_hf.json kivi_hf.json kitty_hf.json \
         ommx_vllm.json vllm_bf16.json vllm_flash_attn.json turboquant_vllm.json; do
  case " $PRODUCED " in *" $f "*) continue;; esac
  [ -e "$DATA/$f" ] || continue
  echo "[collect] NOTE $f will be collected but was NOT produced by this run"
  echo "          ($DATA/$f -- earlier run, other leg, or a shipped historical JSON)"
done
"$OMMXPY" figure/collect.py --src "$DATA" --gpu "$(echo "$TAG" | tr a-z A-Z)" --ctxs "$CTXS" --out "figure/data/${TAG}.json" || { echo "!! [figure] collect.py FAILED -> no figure/data/${TAG}.json"; RC=1; }
"$OMMXPY" figure/plot.py --data "figure/data/${TAG}.json" --outdir figure || { echo "!! [figure] plot.py FAILED -> no PNGs (measured JSONs are safe in $DATA/)"; RC=1; }
echo
if [ "$RC" = 0 ]; then echo "ORCHESTRATE_${TAG}_DONE $(date -u)"
else echo "ORCHESTRATE_${TAG}_INCOMPLETE $(date -u) -- at least one leg above FAILED; the"
     echo "  figure is missing those arms. Do NOT cite it as a full reproduction."; fi
ls -la "$DATA/"
exit "$RC"
