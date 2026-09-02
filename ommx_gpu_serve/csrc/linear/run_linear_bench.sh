#!/usr/bin/env bash
# run_linear_bench.sh — reproduce the OMMX i2f4 memory-bound decode-linear latency comparison
# (bf16 / CUTLASS INT4 / LLM.int8 / OMMX i2f4 @ 3 outlier ratios) + regenerate the figure.
#
#   ./run_linear_bench.sh [--gpu N] [--tag h200]
#
# Prereqs (the caller sets these; the kernel + CUTLASS arm need them):
#   OMMX_CUTLASS_DIR        CUTLASS 4.4.2 tree (flashinfer bundle: tools/util/.../mixed_dtype_utils.hpp)
#   OMMX_CUTLASS_EX55_BIN   compiled example-55 mixed-input INT4xbf16 binary (optional; else searched)
#   conda env with torch (cu121+) + bitsandbytes for the LLM.int8 arm
# NO `set -u` (cu13 conda activate.d uses unbound vars); NO background/cron; single GPU.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"          # open_ommx_serve root
GPU="${GPU:-0}"; TAG="${TAG:-h200}"
# A value-taking flag is checked for its value FIRST. `shift 2` with only ONE argument left
# fails and shifts NOTHING, and there is no `set -u` here (see the header) to turn the unbound
# $2 into an exit -- so `run_linear_bench.sh --gpu` used to spin in this loop forever, printing
# nothing. Fail with the usage instead.
while [ $# -gt 0 ]; do case "$1" in
  --gpu|--tag) [ $# -ge 2 ] || { echo "$1 needs a value; usage: $0 [--gpu N] [--tag h200]"; exit 2; }
               case "$1" in --gpu) GPU="$2";; --tag) TAG="$2";; esac; shift 2;;
  *) echo "unknown opt: $1"; exit 2;; esac; done

: "${OMMX_CUTLASS_DIR:?set OMMX_CUTLASS_DIR to a CUTLASS 4.4.2 tree (mixed_dtype_utils.hpp)}"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMMX_ENABLE_SM90_CUTLASS=1 OMMX_CUTE_FUSED=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8" PYTHONNOUSERSITE=1
[ -n "$CONDA_PREFIX" ] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"  # cu13 CXXABI

DATA="$REPO/figure/data_$TAG"; mkdir -p "$DATA"
echo "== OMMX i2f4 linear bench  gpu=$GPU tag=$TAG  cutlass=$OMMX_CUTLASS_DIR =="
cd "$HERE"

# 1) build + format-parity assert + latency sweep (bf16/CUTLASS-live/int8/OMMX npv 4/8/16) -> result.json
OMMX_RUN_TS="linear_$(date -u +%Y%m%dT%H%M%SZ)"
python bench_linear_memory_bound.py || { echo "!! bench failed"; exit 1; }
RJ="$(ls -t "$HERE"/runs/*/mem_bound/result.json 2>/dev/null | head -1)"
[ -z "$RJ" ] && { echo "!! no result.json"; exit 1; }
cp "$RJ" "$DATA/linear_mem_bound.json"

# 2) render the figure -> figure/fig_linear_tpot_<tag>.png
python plot_linear_memory_bound.py "$DATA/linear_mem_bound.json" "$REPO/figure/fig_linear_tpot_${TAG}.png"
echo "== done -> $REPO/figure/fig_linear_tpot_${TAG}.png  +  $DATA/linear_mem_bound.json =="
