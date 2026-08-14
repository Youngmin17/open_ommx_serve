#!/usr/bin/env bash
# open_ommx_serve — single reproduction entry point.
#
#   ./run.sh setup                 # print the per-method venv setup commands
#   ./run.sh bench   [opts]        # B=1 decode TPOT/TTFT + peak-mem + quality, all methods
#   ./run.sh figure  [opts]        # normalize figure/data_<tag>/ -> figure/data/<tag>.json + plots
#   ./run.sh all     [opts]        # bench + figure
#
# Methods (each in its OWN venv — KIVI/Kitty/vLLM pin incompatible deps):
#   bf16 ommx kivi kitty  -> figure/bench.py (HF-eager, each method's real low-bit-KV kernel)
#   vllm_bf16 ommx_vllm    -> ommx_gpu_serve/bench/bench_e2e (vLLM TRITON/CUSTOM attn + breakdown)
# Point run.sh at the venvs with VENV_<METHOD> env vars, else it uses .venv-<m>/ then `python`.
#
# Opts: --gpu N  --tag {a100|h200|...}  --model REPO  --ctxs 1024,4096,16384,32768,65536
#       --methods "bf16 ommx kivi kitty vllm_bf16 ommx_vllm"
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO"

GPU="${GPU:-0}"
TAG="${TAG:-run}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CTXS="${CTXS:-1024,4096,16384,32768,65536}"
METHODS="${METHODS:-bf16 ommx kivi kitty vllm_bf16 ommx_vllm}"
KITTY_PKG_PATH="${KITTY_PKG_PATH:-}"     # dir containing the external `kitty` package

CMD="${1:-all}"; shift || true
while [ $# -gt 0 ]; do case "$1" in
  --gpu) GPU="$2"; shift 2;;
  --tag) TAG="$2"; shift 2;;
  --model) MODEL="$2"; shift 2;;
  --ctxs) CTXS="$2"; shift 2;;
  --methods) METHODS="$2"; shift 2;;
  *) echo "unknown opt: $1"; exit 2;;
esac; done

DATA="$REPO/figure/data_$TAG"; mkdir -p "$DATA"

venv_python() {  # echo the python to use for a method
  local m="$1" v; eval "v=\${VENV_$(echo "$m" | tr a-z A-Z):-}"
  [ -z "$v" ] && v="$REPO/.venv-$m"
  if [ -x "$v/bin/python" ]; then echo "$v/bin/python"; else echo python; fi
}

hf_method() {  # bf16|ommx|kivi|kitty -> figure/bench.py -> data_<tag>/<m>_hf.json
  local m="$1" PY; PY="$(venv_python "$m")"
  local PP="$REPO:$REPO/baseline:$REPO/ommx_gpu_serve/hf_eager"
  [ -n "$KITTY_PKG_PATH" ] && PP="$KITTY_PKG_PATH:$PP"
  echo "== [$m] $($PY --version 2>&1) gpu=$GPU =="
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PP" KITTY_PKG_PATH="$KITTY_PKG_PATH" \
    "$PY" figure/bench.py --method "$m" --model "$MODEL" --ctxs "$CTXS" \
    --out "$DATA/${m}_hf.json" || echo "!! [$m] failed"
}

vllm_method() {  # vllm_bf16 + ommx_vllm breakdown from bench_e2e — bf16 weights + OMMX KV, NO weight bundle
  local PY; PY="$(venv_python ommx)"
  # The published OMMX(vLLM) bar is the KV-quant attention path (abl_attn: bf16 weights + OMMX KV),
  # so it needs no weight bundle. Run only the bundle-free figure arms, at EVERY cmp ctx (so the
  # breakdown never silently falls back to a weight-quant arm), with the canonical 12/64 recipe.
  echo "== [vllm_bf16+ommx_vllm] gpu=$GPU (bf16 weights + OMMX KV; ommx_w weight-quant not shipped) =="
  CUDA_VISIBLE_DEVICES="$GPU" VLLM_PLUGINS=ommx_gpu_serve \
    "$PY" -m ommx_gpu_serve.bench.bench_e2e_a100 --model "$MODEL" --cmp-ctxs "$CTXS" \
    --abl-ctxs "$CTXS" --outliers 6 \
    --only-arms "TRITON,abl_attn,abl_attn_no_all,abl_attn_no_unpack,abl_attn_no_dequant,abl_attn_skipwrite" \
    --json "$DATA/_e2e.json" || { echo "!! [vllm] failed"; return; }
  "$PY" ommx_gpu_serve/bench/e2e_to_figure.py --in "$DATA/_e2e.json" \
    --out-vllm-bf16 "$DATA/vllm_bf16.json" --out-ommx-vllm "$DATA/ommx_vllm.json" 2>/dev/null \
    || echo "   (e2e_to_figure adapter absent — see bench_e2e JSON directly)"
}

do_bench() {
  for m in $METHODS; do case "$m" in
    bf16|ommx|kivi|kitty) hf_method "$m";;
    vllm_bf16|ommx_vllm|vllm) vllm_method;;
    *) echo "unknown method: $m";;
  esac; done
}

do_figure() {
  local PY; PY="$(venv_python ommx)"; [ -x "$PY" ] || PY=python
  "$PY" figure/collect.py --src "$DATA" --gpu "$(echo "$TAG" | tr a-z A-Z)" \
    --ctxs "$CTXS" --out "figure/data/${TAG}.json"
  "$PY" figure/plot.py --data "figure/data/${TAG}.json" --outdir figure
  echo "figure -> figure/fig_tpot_${TAG}.png  figure/fig_peakmem_${TAG}.png"
}

case "$CMD" in
  setup) cat <<EOF
Per-method venv setup (uv recommended):
  uv venv .venv-ommx     --python 3.10 && uv pip install -e '.[ommx]'  && uv pip install -e ommx_gpu_serve && uv pip install -e eval
  uv venv .venv-kivi     --python 3.10 && uv pip install -e '.[kivi]'  && uv pip install -e eval
  uv venv .venv-kitty    --python 3.10 && uv pip install -e '.[kitty]' && uv pip install -e eval   # + external kitty pkg -> KITTY_PKG_PATH
bf16 reuses .venv-ommx. vLLM breakdown (ommx_vllm) reuses .venv-ommx (bf16 weights + OMMX KV; no bundle).
EOF
  ;;
  bench)  do_bench ;;
  figure) do_figure ;;
  all)    do_bench; do_figure ;;
  *) echo "usage: ./run.sh {setup|bench|figure|all} [opts]"; exit 2;;
esac
