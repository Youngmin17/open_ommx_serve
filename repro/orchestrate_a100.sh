#!/usr/bin/env bash
# A100 reproduction (installable subset: bf16/ommx/kivi) on one GPU, writing figure/data_<tag>/<m>_hf.json then collect+plot. This is the concrete
# per-venv instance behind fig2; ../run.sh is the portable single-venv entry point.
#
# Override before running (defaults assume per-method venvs under $HOME):
#   REPO     repo root                    (default: parent of this script)
#   OMMXPY   python for the ommx venv     (default: $HOME/.venv-ommx/bin/python)
#   KIVIPY   python for the kivi venv     (default: $HOME/.venv-kivi/bin/python)
set -o pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CTXS=${CTXS:-1024,4096,8192,16384,32768,65536}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
OMMXPY="${OMMXPY:-$HOME/.venv-ommx/bin/python}"
KIVIPY="${KIVIPY:-$HOME/.venv-kivi/bin/python}"
TAG=${TAG:-a100}
DATA="$REPO/figure/data_$TAG"; cd "$REPO"; mkdir -p "$DATA"
echo "=== REPRO A100  CVD=$CUDA_VISIBLE_DEVICES  ctxs=$CTXS  $(date -u) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null

step(){ echo; echo "######## $1 @ $(date -u +%H:%M:%S) ########"; }
summ(){ "$OMMXPY" - "$1" <<'PY' 2>/dev/null || true
import json,sys
d=json.load(open(sys.argv[1]))
print("  method:",d.get("method"))
for c,v in (d.get("ctxs") or {}).items():
    print(f"    ctx={c}: tpot={v.get('tpot_ms')} ttft={v.get('ttft_ms')} peak={v.get('peak_gb')}")
q=d.get("quality") or {}; print("  quality:",repr((q.get("output") or q.get("error") or "")[:160]))
PY
}

export PYTHONPATH="$REPO:$REPO/ommx_gpu_serve/hf_eager"
step bf16; "$OMMXPY" figure/bench.py --method bf16 --model "$MODEL" --ctxs "$CTXS" --out "$DATA/bf16_hf.json" 2>&1 | tail -14; summ "$DATA/bf16_hf.json"
step ommx; "$OMMXPY" figure/bench.py --method ommx --model "$MODEL" --ctxs "$CTXS" --out "$DATA/ommx_hf.json" 2>&1 | tail -14; summ "$DATA/ommx_hf.json"

export PYTHONPATH="$REPO/baseline:$REPO"
step kivi; "$KIVIPY" figure/bench.py --method kivi --model "$MODEL" --ctxs "$CTXS" --out "$DATA/kivi_hf.json" 2>&1 | tail -14; summ "$DATA/kivi_hf.json"

step figure
"$OMMXPY" figure/collect.py --src "$DATA" --gpu "$(echo "$TAG" | tr a-z A-Z)" --ctxs "$CTXS" --out "figure/data/${TAG}.json"
"$OMMXPY" figure/plot.py --data "figure/data/${TAG}.json" --outdir figure
echo; echo "ORCHESTRATE_${TAG}_DONE $(date -u)"; ls -la "$DATA/"
