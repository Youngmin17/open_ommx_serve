#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Offline scoring for the LiveCodeBench-v6 KV-quant comparison.
#
#   KV=<baseline/kv> LCB_CORPUS=<n120 corpus> bash eval/lcb/score_lcb.sh [arm ...]
#
# Generation defers scoring (LCB_DEFER=1), so pass@1 is produced here, on CPU, from
# <base>__<arm>.shard*.json.gens.jsonl. Every arm -- including the bf16 baseline -- must be scored
# under ONE policy: an arm graded on an idle node and an arm graded under load are not comparable.
#
# The policy below is not cosmetic. Each setting fixes a measured failure:
#
#   LCB_TIMEOUT=60        CPU seconds, enforced by the kernel through RLIMIT_CPU. The upstream
#                         harness bounded WALL clock, so a busy node turned correct solutions into
#                         timeouts: the same 90 generations scored pass=43/timeout=8, then
#                         pass=35/timeout=17 thirteen minutes later. CPU time does not move with
#                         node load; after the switch, three independent scorings agreed on every
#                         single verdict (0 flips).
#   LCB_WALL_FACTOR=30    wall clock survives only as a backstop for a child blocked on I/O. At 8x
#                         it still fired under load and re-introduced flips.
#   LCB_MEM_GB=0          RLIMIT_AS off. Capping address space made execve fail in workers that
#                         hold the corpus, inventing a `spawn_error` status that did not exist
#                         before.
#   LCB_SPAWN_RETRY=6     fork() returning ENOMEM is a statement about the machine, not about the
#                         solution being graded. Retry before recording a failure.
#   LCB_SPAWN_MAX_RATE    refuse to publish a result poisoned by spawn failures; write
#                         <dst>.REJECTED instead. Without this gate an out-of-memory episode
#                         published an arm at exactly pass@1 = 0.0000.
#   --workers 2           each worker is a separate copy of the corpus. Use the 120-problem
#                         corpus, not the full 1055-problem one: carrying 935 unused problems in
#                         every worker is what exhausted memory and broke fork() in the first place.
set -uo pipefail

KV="${KV:?set KV to the baseline/kv checkout on the cluster}"
OUT="${OUT:-$KV/runs_reasoning/30b-a3b-thinking}"
BASE="$OUT/livecodebench_v6"
PY="${PY:-python3}"
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(none kivi turboquant kitty ommx)

export LCB_CACHE="${LCB_CORPUS:?set LCB_CORPUS to the scoring corpus .pkl.gz}"
export PYTHONPATH="$KV:${PYTHONPATH:-}" PYTHONNOUSERSITE=1
export LCB_TIMEOUT="${LCB_TIMEOUT:-60}"
export LCB_WALL_FACTOR="${LCB_WALL_FACTOR:-30}"
export LCB_MEM_GB="${LCB_MEM_GB:-0}"
export LCB_SPAWN_RETRY="${LCB_SPAWN_RETRY:-6}"
export LCB_SPAWN_MAX_RATE="${LCB_SPAWN_MAX_RATE:-0.02}"
WORKERS="${WORKERS:-2}"

echo "SCORE_START $(date -u +%FT%TZ) policy=cpu${LCB_TIMEOUT}s/wall${LCB_WALL_FACTOR}x/workers${WORKERS}"
for arm in "${ARMS[@]}"; do
  n=$(cat "${BASE}__${arm}".shard*.gens.jsonl 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || { echo "SCORE_SKIP $arm (no generations)"; continue; }
  echo "SCORE arm=$arm gens=$n $(date -u +%FT%TZ)"
  "$PY" "$KV/score_lcb_offline.py" --base "$BASE" --arm "$arm" --shards 4 --workers "$WORKERS"
  echo "  rc=$?"
done
echo "SCORE_DONE $(date -u +%FT%TZ)"
