#!/usr/bin/env bash
# H200 reproduction: run every method in its own conda env / venv on the assigned GPU
# (CUDA_VISIBLE_DEVICES inherited), writing figure/data_<tag>/<m>_hf.json — the exact names
# figure/collect.py consumes — then normalizing + plotting. This is the concrete multi-env
# instance behind the figures; ../run.sh is the portable single-venv entry point.
#
# Override any of these before running (defaults assume a conda layout):
#   REPO       repo root                       (default: parent of this script)
#   CONDASH    conda profile.d/conda.sh        (default: /opt/anaconda3/etc/profile.d/conda.sh)
#   ENV_OMMX / ENV_KIVI  conda env names               (default: ommx / kivi)
#   KITTY_SRC  Kitty package src dir           (external `kitty` package)
#   KVENV      python for the Kitty venv       (default: python)
#   HF_HOME    HF cache dir                    (default: $HOME/.cache/huggingface)
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
DATA="$REPO/figure/data_$TAG"; mkdir -p "$DATA"; cd "$REPO"

echo "=== REPRO $TAG  CVD=${CUDA_VISIBLE_DEVICES:-?}  ctxs=$CTXS  $(date -u) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed -n '1p'

step(){ echo; echo "######## $1  @ $(date -u +%H:%M:%S) ########"; }
summ(){ python - "$1" <<'PY' 2>/dev/null || true
import json,sys
d=json.load(open(sys.argv[1]))
print("  method:",d.get("method"))
for c,v in (d.get("ctxs") or {}).items():
    print(f"    ctx={c}: tpot={v.get('tpot_ms')} ttft={v.get('ttft_ms')} peak={v.get('peak_gb')}")
q=d.get("quality") or {}
o=(q.get("output") or q.get("error") or "")[:160]
print("  quality:",repr(o))
PY
}

# ---- bf16 + ommx  (env: $ENV_OMMX) ----
source "$CONDASH"; conda activate "$ENV_OMMX"
export PYTHONPATH="$REPO:$REPO/ommx_gpu_serve/hf_eager"
step bf16;  python figure/bench.py --method bf16 --model "$MODEL" --ctxs "$CTXS" --out "$DATA/bf16_hf.json" 2>&1 | tail -14; summ "$DATA/bf16_hf.json"
step ommx;  python figure/bench.py --method ommx --model "$MODEL" --ctxs "$CTXS" --out "$DATA/ommx_hf.json" 2>&1 | tail -14; summ "$DATA/ommx_hf.json"
conda deactivate

# ---- kivi  (env: $ENV_KIVI) ----
source "$CONDASH"; conda activate "$ENV_KIVI"
export PYTHONPATH="$REPO/baseline:$REPO"
step kivi;  python figure/bench.py --method kivi --model "$MODEL" --ctxs "$CTXS" --out "$DATA/kivi_hf.json" 2>&1 | tail -14; summ "$DATA/kivi_hf.json"
conda deactivate

# ---- kitty  (venv $KVENV + external kitty pkg at $KITTY_SRC) ----
if [ -n "$KITTY_SRC" ]; then
  export PYTHONPATH="$KITTY_SRC:$REPO/baseline/kitty:$REPO"
  export KITTY_PKG_PATH="$KITTY_SRC"
  step kitty; "$KVENV" figure/bench.py --method kitty --model "$MODEL" --ctxs "$CTXS" --out "$DATA/kitty_hf.json" 2>&1 | tail -14; summ "$DATA/kitty_hf.json"
  unset KITTY_PKG_PATH
else
  echo "[kitty] skipped (set KITTY_SRC to the external kitty package src)"
fi

# ---- normalize + plot (the HF-eager arms; add vLLM arms via ../run.sh bench vllm_bf16 ommx_vllm) ----
source "$CONDASH"; conda activate "$ENV_OMMX"
step figure
python figure/collect.py --src "$DATA" --gpu "$(echo "$TAG" | tr a-z A-Z)" --ctxs "$CTXS" --out "figure/data/${TAG}.json"
python figure/plot.py --data "figure/data/${TAG}.json" --outdir figure
conda deactivate

echo; echo "ORCHESTRATE_${TAG}_DONE $(date -u)"
ls -la "$DATA/"
