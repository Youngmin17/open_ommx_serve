#!/usr/bin/env bash
# Record the demo: seven arms (four HF-eager, three vLLM), one request each (the OMMX brief + question), on the GPU CUDA_VISIBLE_DEVICES
# points at. Site paths come from the environment (see the README's Demo section):
#   PY   python with torch + vLLM + this repo's deps        (default: python3)
#   KP   python for the KIVI / Kitty arms (fp16 kernels)    (default: $PY)
#   R    repo root                                          (default: this file's parent)
#   OUT  where the cast, results and sentinel go            (default: $R/demo/out)
#   KITTY_PKG_PATH, KITTY_TF   the Kitty package and its transformers tree (Kitty arm only)
#   HF_HOME                    Hugging Face cache holding meta-llama/Llama-3.1-8B-Instruct
# Render afterwards on any machine: python demo/render_cast.py $OUT/demo_h200.cast figure/demo_h200.gif
export PY=${PY:-python3}; export KP=${KP:-$PY}
export R=${R:-$(cd "$(dirname "$0")/.." && pwd)}; OUT=${OUT:-$R/demo/out}; mkdir -p "$OUT"
export KITTY_PKG_PATH=${KITTY_PKG_PATH:-} KITTY_TF=${KITTY_TF:-}
export TOKENIZERS_PARALLELISM=false KIVI_QUIET=1
export OMMX_KV_GROUP_TOKENS=32 OMMX_KV_GROUP_CHANNELS=32 OMMX_KV_SINK=8 OMMX_KV_RECENT=32
export OMMX_ATTN_POW2=1 OMMX_ATTN_K_FORMAT=i2f4 OMMX_ATTN_OUTLIER_SELECT=signed OMMX_KV_OUTLIER_MAP=1 OMMX_ATTN_OUTLIERS=6
export OMMX_FIRE_FILE=$OUT/demo.fire.log CTX=${CTX:-4096} N=${N:-512} PROMPT=${PROMPT:-ommx} RES=$OUT/results.jsonl
unset OMMX_ATTN_OUTLIER_REPR OMMX_ATTN_OUTLIER_WARPS OMMX_ABL_NO_OUTLIER OMMX_ABL_V_NODEQUANT OMMX_ABL_K_NODEQUANT OMMX_ABL_NO_UNPACK OMMX_ALLOW_BF16_FALLBACK OMMX_ATTN_GRAPH OMMX_ATTN_V_BF16
cd "$R" && "$PY" demo/record.py --cols 100 --rows 40 -o "$OUT/demo_h200.cast" -- bash demo/demo_session.sh
echo DEMO_RUN_DONE
