#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# LiveCodeBench-v6 generation queue for the KV-quant accuracy comparison.
#
#   GPU_CAP=<n> bash eval/lcb/run_lcb_queue.sh
#
# One GPU slot, arms interleaved shard by shard, so a run that is stopped early still leaves
# every arm with comparable coverage instead of one finished arm and three empty ones.
#
# Protocol constants -- changing any of these breaks comparability with the published numbers:
#   batch_size = 1        bs>1 changes left-padding and the fake-quant group boundaries; a bs=4
#                         OMMX run scored exactly 0.0000 on AIME. The bf16 baseline is bs=1, so
#                         every arm must be bs=1.
#   limit = 120           seeded subset (subset_seed=20250729). The loader shuffles with that
#                         seed and truncates, so subsets are prefix-nested: 120 subset-of 420
#                         subset-of the full 1055. A different limit selects different problems.
#   max_new_tokens=16384  a thinking model truncates at this cap; see eval/lcb/README.md, the cap
#                         is a first-order confound and is reported alongside pass@1.
#   LCB_DEFER=1           generation writes <out>.gens.jsonl and defers scoring. No model-written
#                         code is executed on the GPU node; scores come from score_lcb.sh.
#
# The driver (eval_reasoning.py) must forward --sink to build_quantizer and must NOT pass a
# residual_length it did not get on the command line -- an argparse default silently overrides the
# published recipe. kv_fakequant/test_residual_sink.py fails if the region is not enforced.
#
# Every job is retried: transformers opens its own source file on each generate() call, so an
# NFS-hosted environment can kill a shard with OSError [Errno 5] partway through. --resume makes
# a retry cost only the problems not yet generated.
set -uo pipefail

KV="${KV:?set KV to the baseline/kv checkout on the cluster}"
OUT="${OUT:-$KV/runs_reasoning/30b-a3b-thinking}"
BASE="$OUT/livecodebench_v6"
MODEL="${MODEL:-qwen3}"; SIZE="${SIZE:-30b-a3b-thinking}"
LIMIT="${LIMIT:-120}"; SHARDS="${SHARDS:-4}"; TRIES="${TRIES:-4}"
GPU_CAP_SH="${GPU_CAP_SH:-/scratch/<user>/bin/gpu_cap.sh}"
QUEUE="${QUEUE:-ommx:0 kivi:0 kitty:0 turboquant:0 ommx:1 kivi:1 kitty:1 turboquant:1 \
ommx:2 kivi:2 kitty:2 turboquant:2 ommx:3 kivi:3 kitty:3 turboquant:3}"

# Each arm runs its own best low-bit-KV recipe (NOT an iso-bit comparison -- see README).
#
# --sink / --residual-length are the method's fp16 region and are passed EXPLICITLY here. They
# used to be omitted, which silently stripped KIVI's 128-token residual and Kitty's 32-token sink
# while OMMX kept its sink-8 -- an asymmetry that made the baselines look worse than their papers.
# kv_fakequant/quantizers.py PUBLISHED_RECIPE is the source of truth for these values, and
# analyze_lcb.py re-checks the recorded kv_desc against it, so the two cannot drift apart again.
arm_env () {
  case "$1" in
    ommx) echo "OMMX_KV_PREFILL=0 OMMX_KV_OUTLIERS=12 OMMX_KV_GROUP=64 OMMX_KV_SINK=8" ;;
    *)    echo "" ;;
  esac
}
arm_args () {
  case "$1" in
    kivi)       echo "--k-bits 2 --v-bits 2 --group-size 128 --residual-length 128 --sink 0" ;;
    kitty)      echo "--k-bits 2 --v-bits 2 --group-size 128 --outlier-frac 0.02 --residual-length 0 --sink 32" ;;
    turboquant) echo "--k-bits 3 --v-bits 3 --group-size 128 --residual-length 0 --sink 0" ;;
    *)          echo "" ;;
  esac
}
n_gens () { wc -l < "${1}.gens.jsonl" 2>/dev/null || echo 0; }

export PYTHONPATH="$KV:${PYTHONPATH:-}" PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" LCB_DEFER=1
export LCB_CACHE_DIR="${LCB_CACHE_DIR:?set LCB_CACHE_DIR to the directory holding index.json}"
unset LCB_MAX_TESTS LCB_FLOAT_TOL LCB_DATE_MIN LCB_DATE_MAX

run_one () {                                   # arm shard -- claims a GPU through the cap guard
  local arm="$1" si="$2"
  local o="${BASE}__${arm}.shard${si}.json" lg="${BASE}__${arm}.shard${si}.log"
  # A resume target generated at a different batch size would be silently merged into this arm.
  if [ -f "$o" ] && ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$o')).get('batch_size') in (1,None) else 1)"; then
    echo "LCBQ_ABORT arm=$arm shard=$si reason=resume_target_batch_size_not_1"; return
  fi
  local per_shard=$(( LIMIT / SHARDS )) have got
  for t in $(seq 1 "$TRIES"); do
    have=$(n_gens "$o")
    [ "$have" -ge "$per_shard" ] && { echo "LCBQ_SKIP arm=$arm shard=$si already=$have"; break; }
    echo "LCBQ_JOB arm=$arm shard=$si try=$t have=$have $(date -u +%FT%TZ)"
    bash "$GPU_CAP_SH" run "lcb_${arm}_${si}" -- env $(arm_env "$arm") \
      python "$KV/eval_reasoning.py" --model "$MODEL" --size "$SIZE" \
        --quant-method "$arm" --benchmark livecodebench_v6 \
        --batch-size 1 --limit "$LIMIT" --shard "${si}/${SHARDS}" --resume \
        $(arm_args "$arm") --out "$o" >> "$lg" 2>&1
    local rc=$?; got=$(n_gens "$o")
    echo "LCBQ_TRY_DONE arm=$arm shard=$si try=$t rc=$rc gens=$got $(date -u +%FT%TZ)"
    [ "$rc" -eq 0 ] && break
  done
  echo "LCBQ_JOB_DONE arm=$arm shard=$si gens=$(n_gens "$o") $(date -u +%FT%TZ)"
}

mkdir -p "$OUT"
echo "LCBQ_START $(date -u +%FT%TZ) limit=$LIMIT bs=1 shards=$SHARDS tries=$TRIES"
for job in $QUEUE; do run_one "${job%%:*}" "${job##*:}"; done
echo "LCBQ_ALL_DONE $(date -u +%FT%TZ)"
