#!/usr/bin/env bash
# H200 reproduction: run every method in its own conda env / venv on the assigned GPU
# (CUDA_VISIBLE_DEVICES inherited), writing figure/data_<tag>/<m>_hf.json — the exact names
# figure/collect.py consumes — then normalizing + plotting. This is the concrete multi-env
# instance behind the figures; ../run.sh is the portable single-venv entry point.
#
# Each leg renames the JSON it owns to <file>.prev BEFORE running, so a leg that fails leaves
# no collectable file (nothing stale gets drawn as this run's bar) and nothing is destroyed
# either. Files this run did not produce are named on a "[collect] NOTE" line before the
# figure step rather than silently folded into the figure.
#
# Override any of these before running (defaults assume a conda layout):
#   REPO       repo root                       (default: parent of this script)
#   CONDASH    conda profile.d/conda.sh        (default: /opt/anaconda3/etc/profile.d/conda.sh)
#   ENV_OMMX / ENV_KIVI  conda env names               (default: ommx / kivi)
#   KITTY_SRC  Kitty package src dir           (external `kitty` package)
#   KVENV      python for the Kitty venv       (default: python)
#   HF_HOME    HF cache dir                    (default: $HOME/.cache/huggingface)
#   DTYPE      compute dtype for bf16/ommx/kitty        (default: bfloat16)
#   DTYPE_KIVI compute dtype for the KIVI arm           (default: float16 -- see below)
#   DTYPE_KITTY                                          (default: $DTYPE)
#
# WHY THE KIVI ARM IS PINNED TO fp16. figure/bench.py's --dtype default is bfloat16 (so the
# arm plotted as "bf16" really is bf16 and matches the bf16 vLLM engine arms), but KIVI is
# fp16-ONLY: baseline/kivi/quant/new_pack.py casts every dequantized K/V to torch.float16
# (unpack_and_dequant_{kcache,vcache,triton_packed}) and quant/matmul.py's
# triton_bmm_fA_qB_outer -- the fused kernel models/llama_kivi_eval.py actually imports --
# allocates an fp16 output, so a bfloat16 KIVI run raises a dtype mismatch instead of a
# number. The resulting figure is therefore MIXED dtype; that is recorded per arm in the JSON
# (meta.dtype + arm_label), carried into figure/data/<tag>.json by collect.py, and printed in
# the plot legend. See repro/README.md "Which arm runs in which dtype".
set -o pipefail   # NOT -u: conda activate.d (cuda-nvcc) references unbound NVCC_PREPEND_FLAGS
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDASH="${CONDASH:-/opt/anaconda3/etc/profile.d/conda.sh}"
ENV_OMMX="${ENV_OMMX:-ommx}"; ENV_KIVI="${ENV_KIVI:-kivi}"
KITTY_SRC="${KITTY_SRC:-}"
KVENV="${KVENV:-python}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
CTXS=${CTXS:-1024,4096,16384,32768,65536}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TAG=${TAG:-h200}
DTYPE=${DTYPE:-bfloat16}
DTYPE_KIVI=${DTYPE_KIVI:-float16}     # fp16-only packer; see the header
DTYPE_KITTY=${DTYPE_KITTY:-$DTYPE}    # vendored modeling has no fp16 hard-cast; the EXTERNAL
                                      # kitty package is not vendored -> not pinned here
DATA="$REPO/figure/data_$TAG"; mkdir -p "$DATA"; cd "$REPO"

echo "=== REPRO $TAG  CVD=${CUDA_VISIBLE_DEVICES:-?}  ctxs=$CTXS  $(date -u) ==="
echo "=== dtype: bf16/ommx=$DTYPE  kivi=$DTYPE_KIVI  kitty=$DTYPE_KITTY (mixed = a fairness"
echo "===        caveat; it is recorded per arm and printed in the plot legend) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed -n '1p'

step(){ echo; echo "######## $1  @ $(date -u +%H:%M:%S) ########"; }

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
summ(){ python - "$1" <<'PY' 2>/dev/null || true
import json,sys
d=json.load(open(sys.argv[1]))
# arm_label carries the RESOLVED dtype ("kivi (fp16)"), so a mislabelled arm is visible here.
print("  method:",d.get("method"),"arm_label:",d.get("arm_label"),
      "dtype:",(d.get("meta") or {}).get("dtype"))
for c,v in (d.get("ctxs") or {}).items():
    print(f"    ctx={c}: tpot={v.get('tpot_ms')} ttft={v.get('ttft_ms')} peak={v.get('peak_gb')}")
q=d.get("quality") or {}
o=(q.get("output") or q.get("error") or "")[:160]
print("  quality:",repr(o))
PY
}

# Every leg's exit status is CHECKED and carried to the script's own exit status. Unchecked,
# this script printed ORCHESTRATE_<tag>_DONE and exited 0 after EVERY leg had failed -- a
# missing $CONDASH gives "conda: command not found" and then "python: command not found" for
# each arm, nothing is measured, no JSON is written, and the run still announces DONE. That is
# a success marker for a run that measured nothing, the same defect ../run.sh's figure leg had.
# `set -o pipefail` is already on, so the `| tail -14` pipelines report the bench's rc, not
# tail's. A failed leg is NOT fatal here on purpose (the other arms are still worth having);
# it only has to be visible and to move the exit status.
RC=0
# ---- bf16 + ommx  (env: $ENV_OMMX) ----
source "$CONDASH" || { echo "!! cannot source CONDASH=$CONDASH (set CONDASH to your conda profile.d/conda.sh)"; RC=1; }
conda activate "$ENV_OMMX" || { echo "!! conda activate $ENV_OMMX FAILED"; RC=1; }
export PYTHONPATH="$REPO:$REPO/ommx_gpu_serve/hf_eager"
step bf16;  stash_stale "$DATA/bf16_hf.json" && python figure/bench.py --method bf16 --model "$MODEL" --ctxs "$CTXS" --dtype "$DTYPE" --out "$DATA/bf16_hf.json" 2>&1 | tail -14 || { echo "!! [bf16] FAILED"; RC=1; }; summ "$DATA/bf16_hf.json"
step ommx;  stash_stale "$DATA/ommx_hf.json" && python figure/bench.py --method ommx --model "$MODEL" --ctxs "$CTXS" --dtype "$DTYPE" --out "$DATA/ommx_hf.json" 2>&1 | tail -14 || { echo "!! [ommx] FAILED"; RC=1; }; summ "$DATA/ommx_hf.json"
conda deactivate

# ---- kivi  (env: $ENV_KIVI) ----
source "$CONDASH" 2>/dev/null; conda activate "$ENV_KIVI" || { echo "!! conda activate $ENV_KIVI FAILED"; RC=1; }
export PYTHONPATH="$REPO/baseline:$REPO"
step kivi;  stash_stale "$DATA/kivi_hf.json" && python figure/bench.py --method kivi --model "$MODEL" --ctxs "$CTXS" --dtype "$DTYPE_KIVI" --out "$DATA/kivi_hf.json" 2>&1 | tail -14 || { echo "!! [kivi] FAILED (dtype=$DTYPE_KIVI)"; RC=1; }; summ "$DATA/kivi_hf.json"
conda deactivate

# ---- kitty  (venv $KVENV + external kitty pkg at $KITTY_SRC) ----
# An UNSET $KITTY_SRC is a deliberate skip, not a failure: the kitty package is not vendored,
# and $RC is untouched. A skipped arm is NOT stashed -- it was never requested, so this run has
# no claim on the name -- but any kitty_hf.json already sitting in $DATA/ is still collected and
# still drawn, so the collect-scope note below has to say where it came from.
if [ -n "$KITTY_SRC" ]; then
  export PYTHONPATH="$KITTY_SRC:$REPO/baseline/kitty:$REPO"
  export KITTY_PKG_PATH="$KITTY_SRC"
  step kitty; stash_stale "$DATA/kitty_hf.json" && "$KVENV" figure/bench.py --method kitty --model "$MODEL" --ctxs "$CTXS" --dtype "$DTYPE_KITTY" --out "$DATA/kitty_hf.json" 2>&1 | tail -14 || { echo "!! [kitty] FAILED"; RC=1; }; summ "$DATA/kitty_hf.json"
  unset KITTY_PKG_PATH
else
  echo "[kitty] skipped (set KITTY_SRC to the external kitty package src)"
fi

# ---- normalize + plot (the HF-eager arms; add vLLM arms via ../run.sh bench vllm_bf16 ommx_vllm) ----
source "$CONDASH" 2>/dev/null; conda activate "$ENV_OMMX" 2>/dev/null
step figure
# COLLECT SCOPE. figure/collect.py reads the WHOLE $DATA directory by FILENAME, while this
# script owns only the legs it actually ran. Anything else with a collectable name came from a
# DIFFERENT run -- the vLLM arms (../run.sh bench), a skipped arm, or one of the shipped
# historical JSONs repro/README.md declares uncitable -- and is drawn into this figure with
# nothing on the plot to say so. Name them here; do not delete them (they may be exactly what
# the operator wants combined), just do not let them pass as this run's measurement.
for f in bf16_hf.json ommx_hf.json kivi_hf.json kitty_hf.json \
         ommx_vllm.json vllm_bf16.json vllm_flash_attn.json turboquant_vllm.json; do
  case " $PRODUCED " in *" $f "*) continue;; esac
  [ -e "$DATA/$f" ] || continue
  echo "[collect] NOTE $f will be collected but was NOT produced by this run"
  echo "          ($DATA/$f -- earlier run, other leg, or a shipped historical JSON)"
done
python figure/collect.py --src "$DATA" --gpu "$(echo "$TAG" | tr a-z A-Z)" --ctxs "$CTXS" --out "figure/data/${TAG}.json" || { echo "!! [figure] collect.py FAILED -> no figure/data/${TAG}.json"; RC=1; }
python figure/plot.py --data "figure/data/${TAG}.json" --outdir figure || { echo "!! [figure] plot.py FAILED -> no PNGs (measured JSONs are safe in $DATA/)"; RC=1; }
conda deactivate

echo
if [ "$RC" = 0 ]; then echo "ORCHESTRATE_${TAG}_DONE $(date -u)"
else echo "ORCHESTRATE_${TAG}_INCOMPLETE $(date -u) -- at least one leg above FAILED; the"
     echo "  figure is missing those arms. Do NOT cite it as a full reproduction."; fi
ls -la "$DATA/"
exit "$RC"
