#!/usr/bin/env bash
# Paper Fig. 7 — B=1 decode TPOT vs context — MEASURE EVERY BAR, then draw it.
#
# The published figure was drawn by a script whose header says its three baseline bars were
# "CITED from the original ICCAD figure (digitized), NOT re-measured"; only the GPU-OMMX bar
# was a measurement. This script is the missing reproducer: it measures all four series with
# the kernels in this repo and hands them to figure/plot_fig7.py.
#
#   stage vllm      -> vllm_bf16.json + vllm_flash_attn.json + ommx_vllm.json (with breakdown)
#   stage kivi      -> kivi_hf.json
#   stage lmdeploy  -> lmdeploy.json
#   stage figure    -> figure/data/<tag>.json + figure/fig7_tpot_vs_ctx.png
#   stage all       -> the four above, in order, on ONE GPU
#
# The stages are separate because they need three mutually incompatible Python environments
# (vLLM 0.21 + transformers 4.57 / KIVI's pinned transformers 4.43 / LMDeploy's own torch) and
# because `vllm` and `kivi`+`lmdeploy` can then run CONCURRENTLY on two GPUs. Run `figure`
# only after the measuring stages that own the bars you intend to draw.
#
# ONE JOB PER GPU. Two of these stages sharing a GPU produces contended step times -- and a
# contended decode is not slightly wrong, it is wrong in the direction that flatters whichever
# arm ran alone. Each stage reads CUDA_VISIBLE_DEVICES and never sets it, so the caller (on a
# capped cluster: `gpu_cap.sh run <label> -- ...`) owns the assignment. Check min_ms/p99 vs
# the median in the JSONs afterwards: on a clean run they sit close.
#
# NO `set -u`: the conda `ommx` env's activate.d references an unbound NVCC_PREPEND_FLAGS.
set -o pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STAGE="${1:-all}"

TAG="${TAG:-fig7}"
CTXS="${CTXS:-1024,4096,8192,16384,32768,65536}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
# The LMDeploy arm is W4A16KV4, so it needs an AWQ-INT4 checkpoint. Pointing it at the bf16
# repo would quietly produce a W16 arm still labelled W4.
AWQ_MODEL="${AWQ_MODEL:-hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
OMMXPY="${OMMXPY:-$REPO/.venv-ommx/bin/python}"
KIVIPY="${KIVIPY:-$REPO/.venv-kivi/bin/python}"
LMDPY="${LMDPY:-$REPO/.venv-lmdeploy/bin/python}"
# collect.py is stdlib-only, but plot_fig7.py needs matplotlib + numpy, which a serving env
# has no reason to carry (the vLLM env used for this measurement does not). Keep the plotting
# interpreter separable so a missing plotting dependency costs the FIGURE, never the
# measurement: the collected JSON is written first and can be plotted anywhere afterwards.
PLOTPY="${PLOTPY:-$OMMXPY}"
GPU_MEM="${GPU_MEM:-0.70}"        # vLLM util; the ablation arms OOM at 32K/64K near the default
MEASURE="${MEASURE:-160}"         # >=128 steady-state steps
WARMUP="${WARMUP:-16}"
DTYPE_KIVI="${DTYPE_KIVI:-float16}"   # KIVI's packer hard-casts to fp16; bf16 raises

DATA="$REPO/figure/data_$TAG"
mkdir -p "$DATA" "$REPO/runs"
cd "$REPO" || exit 2
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

say(){ echo; echo "######## $* @ $(date -u +%H:%M:%SZ) ########"; }

# Move a leg's outputs aside BEFORE running it. figure/data_<tag>/ is keyed by FILENAME, so a
# file from an earlier run is indistinguishable from one this run produced: collect.py picks
# it up and plot_fig7.py draws it as a measurement of this sweep. Renaming (never deleting)
# clears the name without costing the operator the file -- .prev is never collected.
stash(){ local f; for f in "$@"; do [ -e "$f" ] || continue
  mv -f "$f" "$f.prev" || { echo "!! cannot move $f aside -- NOT running this leg"; return 1; }
  echo "   stale $(basename "$f") -> $(basename "$f").prev"; done; }

check_py(){ [ -x "$1" ] || { echo "!! $2 python not executable: $1"; echo "   set $3=<path>"; return 1; }; }

RC=0

stage_vllm(){
  check_py "$OMMXPY" ommx OMMXPY || return 1
  say "vllm leg (OMMX CUSTOM + bf16 TRITON/FLASH_ATTN + 5-way breakdown)  GPU=$CUDA_VISIBLE_DEVICES"
  stash "$DATA/vllm_bf16.json" "$DATA/vllm_flash_attn.json" "$DATA/ommx_vllm.json" || return 1
  local E2E="$DATA/_e2e.json" WORK="$REPO/runs/e2e_${TAG}_arms"
  # VLLM_USE_FLASHINFER_SAMPLER=0 -- vLLM's default sampler JIT-builds a FlashInfer top-k/top-p
  # kernel at engine start, and on this sm_90a env that ninja build fails, taking the whole
  # EngineCore down before a single token is measured (it dies inside profile_run's
  # _dummy_sampler_run, so even the KV-pool probe returns 0 blocks and the run's cross-arm
  # deltas become meaningless). The bench decodes GREEDY, so the top-k/top-p kernel is not on
  # the measured path at all; falling back to the torch sampler changes no measured number and
  # is applied to EVERY vLLM arm here, bf16 and OMMX alike, so it cannot tilt the comparison.
  # Remove it once the env's FlashInfer JIT builds again.
  VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
  VLLM_PLUGINS=ommx_gpu_serve PYTHONPATH="$REPO" \
    "$OMMXPY" -m ommx_gpu_serve.bench.bench_e2e_a100 --model "$MODEL" \
      --cmp-ctxs "$CTXS" --abl-ctxs "$CTXS" --outliers 6 --gpu-mem "$GPU_MEM" \
      --warmup "$WARMUP" --measure "$MEASURE" \
      --only-arms "TRITON,FA,abl_attn,abl_attn_no_all,abl_attn_no_unpack,abl_attn_no_dequant,abl_attn_skipwrite" \
      --arch "$TAG" --workdir "$WORK" --csv "$DATA/_e2e.csv" --json "$E2E" \
    || { echo "!! [vllm] bench_e2e failed -> no vLLM/OMMX bars"; return 1; }
  # The OMMX route-fired sentinel is the ONLY thing that distinguishes a real OMMX decode from
  # a silent bf16 fall-through (both produce a plausible TPOT and identical text). bench_e2e
  # forces ok=False on an arm that cannot prove it fired; re-read the verdict into this log.
  "$OMMXPY" - "$E2E" <<'PYEV'
import json, sys
res = json.load(open(sys.argv[1])); ev = res.get("route_evidence") or {}
for a, e in sorted(ev.items()):
    print(f"   route_evidence {a}: ok={e.get('ok')} fired={e.get('fired')} "
          f"nofire={e.get('nofire')} reason={e.get('reason')}")
print("OMMX_PROVEN=%d" % (1 if ev and all(e.get("ok") for e in ev.values()) else 0))
PYEV
  local FLASH; FLASH="$("$OMMXPY" - "$E2E" <<'PYFA'
import json, sys
arms = json.load(open(sys.argv[1])).get("arms") or {}
def ok(a): return sum(1 for c in (arms.get(a) or {}).values() if isinstance(c, dict) and c.get("ok"))
print(next((a for a in ("FA", "FA3") if ok(a)), ""))
PYFA
)"
  local -a AD=("$OMMXPY" ommx_gpu_serve/bench/e2e_to_figure.py --in "$E2E"
               --out-vllm-bf16 "$DATA/vllm_bf16.json" --out-ommx-vllm "$DATA/ommx_vllm.json")
  if [ -n "$FLASH" ]; then AD+=(--flash-arm "$DATA/vllm_flash_attn.json" --flash-arm-name "$FLASH")
  else echo "!! [vllm] no FLASH_ATTN arm produced an ok cell -> the figure's 'vLLM (bf16)' bar"
       echo "   will be the TRITON_ATTN arm, which is NOT stock vLLM. Caption accordingly."; fi
  "${AD[@]}" || { echo "!! [vllm] e2e_to_figure failed"; return 1; }
}

# The SAME vLLM leg under the CANONICAL published serving recipe instead of bench_e2e's
# `--recipe fakequant` default. This exists because the two are different number systems —
# fakequant is group 64/64 + recent 8 + `abs` selection (6 K-outliers per 64-token group),
# the canonical recipe is 32/32 + recent 32 + `signed` (12 per 64) and is the one every
# published ACCURACY number was produced under. Quoting a speed figure from one next to an
# accuracy figure from the other is the thing this stage lets you stop doing.
#
# `--recipe current` makes bench_e2e pass NO fakequant dict, so the exports below survive
# into each arm. They are exported rather than passed as a name because the arm inherits the
# environment; the resolved recipe is now recorded per arm in the JSON
# (provenance.<arm>.recipe_resolved), so a reader does not have to trust this comment.
stage_vllm_canon(){
  check_py "$OMMXPY" ommx OMMXPY || return 1
  say "vllm-canon leg (canonical serving recipe)  GPU=$CUDA_VISIBLE_DEVICES"
  local CDATA="$REPO/figure/data_${TAG}canon"
  mkdir -p "$CDATA"
  stash "$CDATA/vllm_bf16.json" "$CDATA/vllm_flash_attn.json" "$CDATA/ommx_vllm.json" || return 1
  local E2E="$CDATA/_e2e.json" WORK="$REPO/runs/e2e_${TAG}canon_arms"
  # shipped-kv, the recipe every published accuracy/latency number was measured under
  # (relidx7 positions, avg 4.375 bit/elem). OMMX_ATTN_OUTLIER_REPR=relidx7 is pinned on
  # purpose: the serving default is bitmap (avg 4.125 -- same positions and values,
  # bit-identical decode, a cheaper index plane), so this leg reproduces the PUBLISHED
  # footprint, not the bare engine's.
  VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
  VLLM_PLUGINS=ommx_gpu_serve PYTHONPATH="$REPO" \
  OMMX_KV_GROUP_TOKENS=32 OMMX_KV_GROUP_CHANNELS=32 OMMX_KV_SINK=8 OMMX_KV_RECENT=32 \
  OMMX_ATTN_POW2=1 OMMX_ATTN_K_FORMAT=i2f4 OMMX_ATTN_OUTLIER_SELECT=signed \
  OMMX_ATTN_OUTLIER_REPR=relidx7 OMMX_ATTN_COMBINADIC_READ=0 OMMX_KV_OUTLIER_MAP=1 \
  OMMX_ATTN_CUDA_DECODE=0 \
    "$OMMXPY" -m ommx_gpu_serve.bench.bench_e2e_a100 --model "$MODEL" \
      --cmp-ctxs "$CTXS" --abl-ctxs "$CTXS" --recipe current --outliers 6 \
      --gpu-mem "$GPU_MEM" --warmup "$WARMUP" --measure "$MEASURE" \
      --only-arms "TRITON,FA,abl_attn,abl_attn_no_all,abl_attn_no_unpack,abl_attn_no_dequant,abl_attn_skipwrite" \
      --arch "${TAG}canon" --workdir "$WORK" --csv "$CDATA/_e2e.csv" --json "$E2E" \
    || { echo "!! [vllm-canon] bench_e2e failed"; return 1; }
  # Assert the recipe the arms RESOLVED to, not the one this function meant to set.
  "$OMMXPY" - "$E2E" <<'PYRC' || return 1
import json, sys
prov = (json.load(open(sys.argv[1])).get("provenance") or {})
r = ((prov.get("abl_attn") or {}).get("recipe_resolved") or {})
want = {"group_tokens": 32, "group_channels": 32, "recent_window": 32,
        "outliers_per_vector": 6, "use_pow2": True, "outlier_select": "signed"}
bad = {k: (r.get(k), v) for k, v in want.items() if r.get(k) != v}
if bad:
    print("!! [vllm-canon] abl_attn did NOT resolve to the canonical recipe:", bad)
    print("   The JSON is kept, but do NOT label these bars 'canonical recipe'.")
    raise SystemExit(1)
print("   canonical recipe CONFIRMED from the run's own provenance:", want)
PYRC
  local -a AD=("$OMMXPY" ommx_gpu_serve/bench/e2e_to_figure.py --in "$E2E"
               --out-vllm-bf16 "$CDATA/vllm_bf16.json" --out-ommx-vllm "$CDATA/ommx_vllm.json")
  local FLASH; FLASH="$("$OMMXPY" - "$E2E" <<'PYFA'
import json, sys
arms = json.load(open(sys.argv[1])).get("arms") or {}
def ok(a): return sum(1 for c in (arms.get(a) or {}).values() if isinstance(c, dict) and c.get("ok"))
print(next((a for a in ("FA", "FA3") if ok(a)), ""))
PYFA
)"
  [ -n "$FLASH" ] && AD+=(--flash-arm "$CDATA/vllm_flash_attn.json" --flash-arm-name "$FLASH")
  "${AD[@]}" || { echo "!! [vllm-canon] e2e_to_figure failed"; return 1; }
  # The KIVI / LMDeploy bars do not depend on the OMMX recipe, so reuse them rather than
  # spending two more GPU-hours measuring identical numbers. Copied (not symlinked) so the
  # directory stays self-contained, and never overwritten if already present.
  local f
  for f in kivi_hf.json lmdeploy.json; do
    [ -e "$CDATA/$f" ] || { [ -e "$DATA/$f" ] && cp "$DATA/$f" "$CDATA/$f" \
      && echo "   reused $f from $DATA (recipe-independent arm)"; }
  done
}

stage_kivi(){
  check_py "$KIVIPY" kivi KIVIPY || return 1
  say "kivi leg (HF-eager, $DTYPE_KIVI)  GPU=$CUDA_VISIBLE_DEVICES"
  stash "$DATA/kivi_hf.json" || return 1
  PYTHONPATH="$REPO:$REPO/baseline:$REPO/ommx_gpu_serve/hf_eager" \
    "$KIVIPY" figure/bench.py --method kivi --model "$MODEL" --ctxs "$CTXS" \
      --dtype "$DTYPE_KIVI" --warmup "$WARMUP" --measure "$MEASURE" \
      --out "$DATA/kivi_hf.json" \
    || { echo "!! [kivi] failed -> no KIVI bar"; return 1; }
}

stage_lmdeploy(){
  check_py "$LMDPY" lmdeploy LMDPY || return 1
  say "lmdeploy leg (W4A16KV4, TurboMind)  GPU=$CUDA_VISIBLE_DEVICES"
  stash "$DATA/lmdeploy.json" || return 1
  PYTHONPATH="$REPO" "$LMDPY" figure/bench_lmdeploy.py --model "$AWQ_MODEL" \
      --ctxs "$CTXS" --warmup "$WARMUP" --measure "$MEASURE" --tag "$TAG" \
      --out "$DATA/lmdeploy.json" \
    || { echo "!! [lmdeploy] failed -> no LMDeploy bar"; return 1; }
}

stage_figure(){
  check_py "$OMMXPY" ommx OMMXPY || return 1
  say "figure leg (collect + plot_fig7)"
  # Name every collectable file this sweep did NOT write. collect.py reads the whole directory
  # by filename, so a leftover from another run is drawn with nothing on the plot to say so.
  local f
  for f in vllm_bf16.json vllm_flash_attn.json ommx_vllm.json kivi_hf.json lmdeploy.json; do
    [ -e "$DATA/$f" ] || { echo "[collect] ABSENT $f -> that bar will be missing"; continue; }
    [ -n "$(find "$DATA/$f" -newermt '-1 day' 2>/dev/null)" ] \
      || echo "[collect] NOTE $f is older than a day -- not from this sweep?"
  done
  "$OMMXPY" figure/collect.py --src "$DATA" --gpu "$(echo "$TAG" | tr a-z A-Z)" \
      --ctxs "$CTXS" --out "figure/data/${TAG}.json" \
    || { echo "!! [figure] collect failed"; return 1; }
  "$PLOTPY" -c 'import matplotlib, numpy' 2>/dev/null || {
    echo "!! [figure] $PLOTPY has no matplotlib/numpy, so NO PNG was rendered."
    echo "   The measurement is safe: figure/data/${TAG}.json is written. Plot it anywhere:"
    echo "     python figure/plot_fig7.py --data figure/data/${TAG}.json --outdir figure"
    echo "   or re-run this stage with PLOTPY=<a python that has matplotlib>."
    return 1; }
  "$PLOTPY" figure/plot_fig7.py --data "figure/data/${TAG}.json" --outdir figure \
    || { echo "!! [figure] plot_fig7 failed"; return 1; }
}

case "$STAGE" in
  vllm)       stage_vllm       || RC=1 ;;
  vllm-canon) stage_vllm_canon || RC=1 ;;
  kivi)       stage_kivi       || RC=1 ;;
  lmdeploy)   stage_lmdeploy   || RC=1 ;;
  figure)     stage_figure     || RC=1 ;;
  # Explicit reassignment, not a `VAR=x stage_figure` command prefix: in a prefix list all
  # assignments are expanded BEFORE any is applied (so a second one referring to the first
  # would silently see the old value), and whether such an assignment survives a *function*
  # call is a bash-version/POSIX-mode detail. Both are avoided by just setting them.
  figure-canon) TAG="${TAG}canon"; DATA="$REPO/figure/data_${TAG}"
                stage_figure || RC=1 ;;
  all)        stage_vllm || RC=1; stage_kivi || RC=1; stage_lmdeploy || RC=1
              stage_figure || RC=1 ;;
  *) echo "usage: $0 {vllm|vllm-canon|kivi|lmdeploy|figure|figure-canon|all}"; exit 2 ;;
esac

echo
if [ "$RC" = 0 ]; then echo "FIG7_${STAGE}_DONE $(date -u)"
else echo "FIG7_${STAGE}_INCOMPLETE $(date -u) -- a leg above FAILED; the figure is missing"
     echo "  those bars. Do NOT cite it as a full reproduction of Fig. 7."; fi
exit "$RC"
