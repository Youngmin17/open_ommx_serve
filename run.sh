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
#   vllm_bf16 ommx_vllm    -> ommx_gpu_serve/bench/bench_e2e (vLLM TRITON + FLASH_ATTN + OMMX
#                             CUSTOM attn + the OMMX internal-kernel breakdown)
# Point run.sh at the venvs with VENV_<METHOD> env vars, else it uses .venv-<m>/ then `python`.
#
# DTYPE IS PER ARM, ON PURPOSE (see repro/README.md "Which arm runs in which dtype"):
#   bf16 / ommx / kitty -> $DTYPE       (default bfloat16, matching the bf16 vLLM engine arms)
#   kivi                -> $DTYPE_KIVI  (default float16 — KIVI's packer is fp16-ONLY:
#                          baseline/kivi/quant/new_pack.py casts every dequantized K/V to
#                          torch.float16 (unpack_and_dequant_{kcache,vcache,triton_packed})
#                          and quant/matmul.py's triton_bmm_fA_qB_outer -- the fused kernel
#                          llama_kivi_eval.py actually imports -- allocates an fp16 output,
#                          so a bfloat16 KIVI run raises a dtype mismatch instead of
#                          producing a number). The mixed dtype is carried into the legend by
#                          figure/collect.py -> figure/plot.py, never hidden.
#
# Opts: --gpu N  --tag {a100|h200|...}  --model REPO  --ctxs 1024,4096,16384,32768,65536
#       --methods "bf16 ommx kivi kitty vllm_bf16 ommx_vllm"
#       --dtype bfloat16   (the shared dtype; --dtype-kivi overrides the KIVI arm alone)
#       --recipe NAME      (named KV recipe from ommx_gpu_serve/recipes.py, so a sweep can
#                           NAME the recipe it ran instead of exporting 12 env vars by hand:
#                           `shipped-kv` = the canonical published recipe (measured avg 4.375
#                           bit/elem), `paper-kv` = the one that MEASURES the ICCAD Table 1
#                           AvgBits of 2.75 exactly and has NO accuracy number in this repo.
#                           Only the OMMX arms read it; bf16/KIVI/Kitty are unaffected. An
#                           env var you exported yourself WINS over the recipe, and an unknown
#                           name aborts with the known names listed. `python3 -m
#                           ommx_gpu_serve.recipes list` prints the registry.)
# A single inherited CUDA_VISIBLE_DEVICES is honoured; a MULTI-GPU mask is REFUSED (pass
# --gpu N), because silently resolving it lands the run on physical GPU 0.
# Each leg renames its own output JSONs to <file>.prev before running, so a failed leg leaves
# no collectable file AND does not destroy the shipped, git-tracked measurement of that name.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO"

# GPU selection, in TWO halves: the inherited mask is classified here, but the fallback (and
# the multi-GPU refusal) is applied in resolve_gpu() below, because --gpu is parsed after this.
# INHERIT an already-masked CUDA_VISIBLE_DEVICES before falling back to 0: under any wrapper
# that allocates a device for you (a cluster GPU-cap guard, SLURM, a container)
# CUDA_VISIBLE_DEVICES is ALREADY set to the device you were given, and that value is
# ABSOLUTE — a child that re-exports CUDA_VISIBLE_DEVICES=0 does not mean "the first device I
# was allocated", it means physical GPU 0. Hardcoding 0 therefore silently moves the run onto
# a GPU the scheduler did not allocate, which both breaks the cap and contaminates the
# measurement with whatever is already running there. Only a single inherited index is
# honoured; an explicit --gpu / GPU= always wins. repro/orchestrate_a100.sh inherits the same
# way.
MASK_MULTI=""      # non-empty = an inherited MULTI-GPU mask that resolve_gpu() must refuse
if [ -n "${GPU:-}" ]; then
  :                                        # explicit GPU= from the environment wins
elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  :                                        # no mask at all -> resolve_gpu() defaults to 0
elif [ "${CUDA_VISIBLE_DEVICES}" = "${CUDA_VISIBLE_DEVICES%,*}" ]; then
  GPU="$CUDA_VISIBLE_DEVICES"              # inherited single-device mask
else
  MASK_MULTI="$CUDA_VISIBLE_DEVICES"       # inherited multi-device mask
fi
TAG="${TAG:-run}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CTXS="${CTXS:-1024,4096,16384,32768,65536}"
METHODS="${METHODS:-bf16 ommx kivi kitty vllm_bf16 ommx_vllm}"
KITTY_PKG_PATH="${KITTY_PKG_PATH:-}"     # dir containing the external `kitty` package
DTYPE="${DTYPE:-bfloat16}"               # bf16/ommx/kitty: match the bf16 vLLM engine arms
DTYPE_KIVI="${DTYPE_KIVI:-float16}"      # KIVI is fp16-only (hard-cast in its packer)
# Kitty's vendored modeling file (baseline/kitty/_kitty_llama_modeling.py) contains no fp16
# hard-cast — it upcasts to fp32 for norm/softmax and casts back to query.dtype — so it is NOT
# pinned here. Its EXTERNAL kernel package ($KITTY_PKG_PATH) is not vendored and cannot be
# checked from this repo; if it turns out to be fp16-only the arm will raise a loud dtype
# mismatch, and DTYPE_KITTY=float16 is the documented override. Never relabel it bf16.
DTYPE_KITTY="${DTYPE_KITTY:-}"           # empty = follow $DTYPE (resolved at use, so a later
                                         # --dtype still moves the kitty arm with the rest)
RECIPE="${RECIPE:-}"                     # empty = apply NO preset; the shipped env resolution
                                         # is then byte-for-byte what it was before this flag
                                         # existed (nothing below runs).

USAGE='usage: ./run.sh {setup|bench|figure|all} [--gpu N] [--tag TAG] [--model REPO]
       [--ctxs L1,L2,...] [--methods "m1 m2 ..."] [--dtype D] [--dtype-kivi D] [--dtype-kitty D]
       [--recipe NAME]'
# Every option below TAKES a value, and under `set -u` a trailing "--gpu" used to abort with
# `run.sh: line NN: $2: unbound variable` -- a bash internals message for what is an ordinary
# usage error. Validate the count first and print the usage instead, matching
# ommx_gpu_serve/csrc/linear/run_linear_bench.sh. Called as `need_val "$1" $#`.
need_val() { [ "$2" -ge 2 ] || { echo "$1 needs a value"; echo "$USAGE"; exit 2; }; }

CMD="${1:-all}"; shift || true
while [ $# -gt 0 ]; do case "$1" in
  --gpu) need_val "$1" $#; GPU="$2"; shift 2;;
  --tag) need_val "$1" $#; TAG="$2"; shift 2;;
  --model) need_val "$1" $#; MODEL="$2"; shift 2;;
  --ctxs) need_val "$1" $#; CTXS="$2"; shift 2;;
  --methods) need_val "$1" $#; METHODS="$2"; shift 2;;
  --dtype) need_val "$1" $#; DTYPE="$2"; shift 2;;
  --dtype-kivi) need_val "$1" $#; DTYPE_KIVI="$2"; shift 2;;
  --dtype-kitty) need_val "$1" $#; DTYPE_KITTY="$2"; shift 2;;
  --recipe) need_val "$1" $#; RECIPE="$2"; shift 2;;
  *) echo "unknown opt: $1"; echo "$USAGE"; exit 2;;
esac; done

# Second half of the GPU selection (see the MASK_MULTI block above). An inherited MULTI-GPU
# mask is REFUSED rather than resolved: this bench runs one arm on one GPU, every index in the
# mask is equally valid, and picking one for the operator is the same silent relocation in a
# nicer costume -- CUDA_VISIBLE_DEVICES=3,5 used to run on physical GPU 0, which is not even a
# member of the mask, with no message at all. Only `bench`/`all` reach a GPU (`figure` is
# collect+plot and `setup` prints text), so the refusal is scoped to those.
resolve_gpu() {
  [ -n "${GPU:-}" ] && return 0            # explicit --gpu / GPU=, or an inherited single mask
  if [ -n "$MASK_MULTI" ]; then
    echo "CUDA_VISIBLE_DEVICES=$MASK_MULTI is a MULTI-GPU mask and this bench is single-GPU."
    echo "Refusing instead of picking for you: a child's CUDA_VISIBLE_DEVICES is ABSOLUTE, so"
    echo "the old fallback put the run on physical GPU 0 -- not even a member of your mask --"
    echo "without saying so. Name ONE index from the mask, e.g."
    echo "  ./run.sh $CMD --gpu ${MASK_MULTI%%,*}"
    exit 2
  fi
  GPU=0                                    # no mask at all: the documented default
}
case "$CMD" in bench|all) resolve_gpu;; esac

DATA="$REPO/figure/data_$TAG"
# The data dir is created only by the subcommands that USE it. Unconditionally, `./run.sh setup`
# -- which does nothing but print venv commands -- left an empty figure/data_run/ behind on every
# invocation (git does not track empty dirs, so `git status` never showed them). `figure` needs
# the dir to READ from: it is not created here either, so a typo'd --tag is refused by
# collect.py instead of silently collecting an empty directory it just made.
case "$CMD" in bench|all) mkdir -p "$DATA";; esac

# STALE OUTPUTS ARE MOVED ASIDE, NEVER DELETED. figure/data_<tag>/ is keyed by FILENAME, so a
# file left behind by an EARLIER run is indistinguishable from one this run produced:
# collect.py picks it up and plot.py draws it. But those same filenames are the repo's SHIPPED,
# git-tracked published measurements (figure/data_h200/, figure/data_a100/), and the first
# documented Reproduce command writes into figure/data_h200/ -- so `rm -f` destroyed shipped
# artifacts before the leg had done anything, and kept them destroyed when the leg failed. A
# tarball user could not get them back. Renaming to <file>.prev satisfies both: the leg starts
# from a clean slate, and collect.py's FILE2KEY is an EXACT filename map ("bf16_hf.json" ->
# bf16_hf), so a .prev file is never collected and never plotted. Only one generation is kept;
# a second run overwrites the .prev.
PRODUCED=""      # basenames this run OWNS (see the collect-scope note at the end of do_bench)
stash_stale() {  # move aside every output file the caller is about to (re)produce
  local f
  for f in "$@"; do
    PRODUCED="$PRODUCED $(basename "$f")"  # owned even if the leg then fails and writes nothing
    [ -e "$f" ] || continue
    mv -f "$f" "$f.prev" || { echo "!! cannot move $f aside -- refusing to run this leg with a"
                              echo "   stale file in place (it would be collected as this run's"
                              echo "   result). Remove or move $f yourself and re-run."; return 1; }
    echo "   stale $(basename "$f") -> $(basename "$f").prev (kept; .prev is never collected)"
  done
}

venv_python() {  # echo the python to use for a method
  local m="$1" v; eval "v=\${VENV_$(echo "$m" | tr a-z A-Z):-}"
  [ -z "$v" ] && v="$REPO/.venv-$m"
  # `run.sh setup` prints "bf16 reuses .venv-ommx" and never tells anyone to create a
  # .venv-bf16 -- but the lookup above only ever tried $REPO/.venv-bf16 and then fell through
  # to the SYSTEM python, which on a correctly set-up host has no torch. The bf16 baseline --
  # the arm every speedup in the figure is divided by -- therefore failed while ommx/kivi/kitty
  # ran, and the figure quietly lost its reference bar. Honour the documented mapping. An
  # explicit VENV_BF16, or a real .venv-bf16, still wins.
  if [ "$m" = bf16 ] && [ ! -x "$v/bin/python" ]; then v="$REPO/.venv-ommx"; fi
  # Bare `python` is the LAST resort and does not exist on a modern distro/venv-less host
  # (only `python3` does), where every leg then died with "python: command not found".
  if [ -x "$v/bin/python" ]; then echo "$v/bin/python"
  elif command -v python >/dev/null 2>&1; then echo python
  else echo python3; fi
}

# ── named KV recipe (--recipe NAME / RECIPE=NAME) ────────────────────────────────────
# Turns "reproduce the published bit budget" into a FLAG. A sweep that says
# `--recipe paper-kv` records WHICH recipe it ran; a sweep that exports twelve OMMX_*
# vars by hand records only that somebody hoped they were the right twelve.
#
# The preset is materialised as ordinary exported env vars rather than passed down as a
# name, for two reasons:
#   * the recipe has several independent readers (figure/bench.py, the hf_eager modeling
#     files, kv_pool.py, packed_only.py's accounting, preflight.py) that each read
#     os.environ on their own, and every one of them must see the SAME recipe or the
#     engine runs recipe A while the accounting prints recipe B. Exporting here is what
#     makes that true: OMMX_RECIPE alone does NOT, because it is expanded lazily inside
#     config.resolve_serving_config, so a reader that runs before that call in the same
#     process still sees the bare defaults (measured: kv_bits_breakdown reports 4.125
#     under OMMX_RECIPE=paper-kv until resolve_serving_config has run, then 2.75);
#   * figure/bench.py applies its own canonical dict with os.environ.SETDEFAULT, so an
#     exported value from here WINS, and its provenance block records os.environ, so the
#     recipe that actually ran is what lands in meta.ommx_recipe — not the one asked for.
#     CAVEAT, because it is a real gap and not a detail: that block enumerates only the
#     ELEVEN keys of figure/bench.py's own OMMX_RECIPE_ENV, so a paper-kv run records its
#     group sizes and outlier count correctly but records NEITHER OMMX_KV_OUTLIER_MAP nor
#     the preset NAME. Closing that is a two-line change in figure/bench.py, which this
#     change does not own; until then read the "recipe … applied" block printed below.
# `recipes env --export` emits only the assignments that are NOT already set in this
# environment (aliases included), so an operator's own export still wins, exactly as it
# does on the serving path. An unknown name aborts the whole run: a bench under the wrong
# recipe is worse than no bench, and the message lists the known names.
# Only the OMMX arms consume these vars; bf16 / KIVI / Kitty are untouched.
apply_recipe() {
  [ -n "$RECIPE" ] || return 0          # unset -> nothing exported, defaults exactly as before
  local PY LINES
  PY="$(venv_python ommx)"
  LINES="$(PYTHONPATH="$REPO" "$PY" -m ommx_gpu_serve.recipes env "$RECIPE" --export)" || {
    echo "!! --recipe $RECIPE was refused (message above). NOTHING was run."
    echo "   List them with: python3 -m ommx_gpu_serve.recipes list"
    exit 2; }
  eval "$LINES"
  echo "== recipe $RECIPE applied to the OMMX arms =="
  printf '%s\n' "$LINES" | sed 's/^export /   /'
}

hf_method() {  # bf16|ommx|kivi|kitty -> figure/bench.py -> data_<tag>/<m>_hf.json
  local m="$1" PY dt; PY="$(venv_python "$m")"
  local PP="$REPO:$REPO/baseline:$REPO/ommx_gpu_serve/hf_eager"
  [ -n "$KITTY_PKG_PATH" ] && PP="$KITTY_PKG_PATH:$PP"
  # --dtype is passed EXPLICITLY for every arm rather than relying on figure/bench.py's
  # default: that default has already flipped once (float16 -> bfloat16), and the flip broke
  # the KIVI arm, whose packer hard-casts to fp16. Spelling it out here means the dtype a run
  # used is readable from this script, and bench.py echoes the resolved dtype into the JSON
  # (meta.dtype + arm_label) so collect.py/plot.py can label it.
  case "$m" in
    kivi)  dt="$DTYPE_KIVI";;
    kitty) dt="${DTYPE_KITTY:-$DTYPE}";;
    *)     dt="$DTYPE";;
  esac
  echo "== [$m] $($PY --version 2>&1) gpu=$GPU dtype=$dt =="
  # Same stale-output rule as the vLLM leg: figure/data_<tag>/ is keyed by filename, so an arm
  # that fails here must leave NO collectable file rather than leave the previous run's file to
  # be collected and plotted as if this run had produced it — including a file written at a
  # DIFFERENT --dtype. Moved aside, not deleted (see stash_stale): these are git-tracked
  # published measurements in figure/data_h200/ and figure/data_a100/.
  stash_stale "$DATA/${m}_hf.json" || return 1
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PP" KITTY_PKG_PATH="$KITTY_PKG_PATH" \
    "$PY" figure/bench.py --method "$m" --model "$MODEL" --ctxs "$CTXS" --dtype "$dt" \
    --out "$DATA/${m}_hf.json" \
    || { echo "!! [$m] failed (dtype=$dt) -> no $m bar this run. A previous ${m}_hf.json, if"
         echo "   there was one, is now ${m}_hf.json.prev in $DATA/ -- rename it back to"
         echo "   republish it as-is, but read repro/README.md on what those JSONs do NOT"
         echo "   contain before citing one."; return 1; }
}

vllm_method() {  # vLLM-engine arms from bench_e2e — bf16 weights + OMMX KV, NO weight bundle
  local PY; PY="$(venv_python ommx)"
  local E2E="$DATA/_e2e.json" WORK="$REPO/runs/e2e_${TAG}_arms"
  # The published OMMX(vLLM) bar is the KV-quant attention path (abl_attn: bf16 weights + OMMX KV),
  # so it needs no weight bundle. Run only the bundle-free figure arms, at EVERY cmp ctx (so the
  # breakdown never silently falls back to a weight-quant arm), with the canonical 12/64 recipe.
  #
  # TWO bf16 baselines, because they are different attention kernels and only ONE of them is
  # what vLLM runs out of the box:
  #   TRITON = TRITON_ATTN  (also the KV-pool probe + the coherence baseline)
  #   FA     = FLASH_ATTN   = vLLM's DEFAULT backend  (bench_e2e_a100._build_plan: plan["FA"])
  # FA is deliberately absent from bench_e2e_a100's default arm order, so it only runs when
  # named in --only-arms. Without it this pipeline could not produce the stock-vLLM baseline at
  # all, and the TRITON arm was published as "bf16 (vLLM)" — a comparison that was never run.
  echo "== [vllm] gpu=$GPU (bf16 weights + OMMX KV; ommx_w weight-quant not shipped) =="
  # MOVE THIS LEG'S OUTPUTS ASIDE FIRST. figure/data_<tag>/ is keyed by FILENAME, so a file
  # left behind by an EARLIER run is indistinguishable from one this run produced: collect.py
  # picks it up and plot.py draws it. That turned every partial failure below into a stale-data
  # publication -- most visibly, run.sh could print "vLLM's DEFAULT baseline will be MISSING"
  # while a vllm_flash_attn.json from a previous sweep was still sitting there being plotted as
  # "bf16 (vLLM, FlashAttn)". Clearing the names up front means a leg that does not finish
  # leaves NO bar, which is the only honest outcome (law: no silent fallback); renaming rather
  # than deleting means it does not cost the operator the shipped file either (stash_stale).
  stash_stale "$DATA/vllm_bf16.json" "$DATA/vllm_flash_attn.json" "$DATA/ommx_vllm.json" \
    || return 1
  CUDA_VISIBLE_DEVICES="$GPU" VLLM_PLUGINS=ommx_gpu_serve \
    "$PY" -m ommx_gpu_serve.bench.bench_e2e_a100 --model "$MODEL" --cmp-ctxs "$CTXS" \
    --abl-ctxs "$CTXS" --outliers 6 \
    --only-arms "TRITON,FA,abl_attn,abl_attn_no_all,abl_attn_no_unpack,abl_attn_no_dequant,abl_attn_skipwrite" \
    --arch "$TAG" --workdir "$WORK" --csv "$DATA/_e2e.csv" \
    --json "$E2E" || { echo "!! [vllm] failed -> no vLLM bars this run. Any previous"
                       echo "   vllm_bf16.json / vllm_flash_attn.json / ommx_vllm.json is at the"
                       echo "   same path with a .prev suffix."; return 1; }

  # ROUTE-FIRED EVIDENCE (law #5: no silent fallback). Every CUSTOM arm gets its own
  # $OMMX_FIRE_FILE sentinel under --workdir; bench_e2e_a100 reads it back and forces ok=False
  # on EVERY cell of an arm that cannot prove the OMMX decode route fired, because a CUSTOM arm
  # that quietly fell back to bf16 FlashAttention still returns a perfectly plausible TPOT.
  # Re-read it here so the verdict is in run.sh's own log, and so the FA baseline is requested
  # from the adapter only when the FA arm actually produced an ok cell (the adapter hard-errors
  # on a missing FA arm by design; that must not take the other two outputs down with it).
  # Human-readable evidence goes to stderr, machine-readable status to stdout.
  local ST; ST="$("$PY" - "$E2E" <<'PYEV'
import json, sys
res = json.load(open(sys.argv[1]))
arms, ev = res.get("arms") or {}, res.get("route_evidence") or {}
def ok_cells(a):
    return sum(1 for c in (arms.get(a) or {}).values() if isinstance(c, dict) and c.get("ok"))
for a, e in sorted(ev.items()):
    print(f"   route_evidence {a}: ok={e.get('ok')} fired={e.get('fired')} "
          f"nofire={e.get('nofire')} procs_fired={e.get('procs_fired')} "
          f"reason={e.get('reason')}", file=sys.stderr)
# `not ev` counts as UNPROVEN: a bench build that records no evidence cannot prove anything.
print("OMMX_PROVEN=%d" % (1 if ev and all(e.get("ok") for e in ev.values()) else 0))
print("FLASH_ARM=%s" % next((a for a in ("FA", "FA3") if ok_cells(a)), ""))
PYEV
)"
  local PROVEN FLASH
  PROVEN="$(printf '%s\n' "$ST" | sed -n 's/^OMMX_PROVEN=//p')"
  FLASH="$(printf '%s\n' "$ST" | sed -n 's/^FLASH_ARM=//p')"

  # Adapter -> the three figure inputs. Its stderr is NOT redirected: it is the only place the
  # "why" of a failed adaptation is printed, and the old `2>/dev/null || echo "adapter absent"`
  # reported every real failure as a missing file.
  local -a ADAPT=("$PY" ommx_gpu_serve/bench/e2e_to_figure.py --in "$E2E"
                  --out-vllm-bf16 "$DATA/vllm_bf16.json"
                  --out-ommx-vllm "$DATA/ommx_vllm.json")
  if [ -n "$FLASH" ]; then
    ADAPT+=(--flash-arm "$DATA/vllm_flash_attn.json" --flash-arm-name "$FLASH")
  else
    echo "!! [vllm] no FLASH_ATTN arm with an ok cell in $E2E -> vLLM's DEFAULT baseline will be"
    echo "   MISSING from this figure. The remaining 'bf16 (vLLM, Triton attn)' bar is the"
    echo "   TRITON_ATTN arm, NOT stock vLLM. Do not caption it as stock vLLM."
  fi
  "${ADAPT[@]}" || { echo "!! [vllm] e2e_to_figure failed (rc=$?) -> no vLLM bars this run"; return 1; }

  if [ "$PROVEN" != "1" ]; then
    echo "!! [vllm] OMMX route NOT PROVEN for at least one CUSTOM arm (evidence above;"
    echo "   sentinels: $WORK/<arm>.fire.log). Asserting the gate actually dropped the cells:"
    "$PY" - "$DATA/ommx_vllm.json" <<'PYGATE' || return 1
import json, os, sys
p = sys.argv[1]
n = len(json.load(open(p)).get("ctxs") or {})
if n:
    # The gate is upstream (bench_e2e_a100 forces ok=False, the adapter drops not-ok cells).
    # If cells survived anyway the gate did not hold, and the only safe action is to remove the
    # file: an unproven arm must never reach collect.py and become a published OMMX bar.
    os.remove(p)
    # Two ways to get here, and this cannot tell them apart from ommx_vllm.json alone:
    #   (a) the FULL arm (abl_attn) is the unproven one and its cells reached the file anyway
    #       -> the upstream ok=False gate did not hold, and the bar is a FlashAttention bar;
    #   (b) abl_attn is proven but a DIFFERENTIAL arm is not -> the total is real, but that
    #       arm's cells are gone, so the adapter substituted 0.0 for its segment and rescaled
    #       the rest, i.e. the stacked breakdown silently absorbs the missing cost.
    # Both make the file unpublishable, so it is removed either way; read the per-arm
    # route_evidence printed above to see which one happened.
    print(f"!! GATE: {p} still carried {n} ctx cell(s) while at least one CUSTOM arm was "
          f"unproven -- either the upstream ok=False gate did not hold, or only a "
          f"differential arm failed and the stacked breakdown is now distorted. Deleted.")
    sys.exit(1)
print("   ok: ommx_vllm.json has 0 ctx cells, so no OMMX(vLLM) bar is plotted this run "
      "(the bf16 vLLM baselines are unaffected).")
PYGATE
    return 1
  fi
}

BENCH_RC=0   # nonzero as soon as any leg refuses to produce a publishable number

do_bench() {
  apply_recipe          # no-op unless --recipe was named; aborts on an unknown name
  # vllm_bf16 / vllm_flash_attn / ommx_vllm are OUTPUTS of ONE bench_e2e sweep, not separate
  # runs. The default METHODS names two of them, which used to run the entire (hours-long)
  # vLLM sweep twice, the second run overwriting the first — so the sweep is fired once.
  local vllm_done=0
  for m in $METHODS; do case "$m" in
    bf16|ommx|kivi|kitty) hf_method "$m" || BENCH_RC=1;;
    vllm_bf16|ommx_vllm|vllm)
      if [ "$vllm_done" = 0 ]; then
        # vllm_method returns 1 when the sweep failed, the adapter failed, or the OMMX route
        # could not be proven. That verdict has to reach run.sh's OWN exit status: an operator
        # (or a CI step) that only checks `./run.sh bench` used to see rc=0 right after
        # "GATE FAILURE", which is the console-only version of a silent fallback.
        vllm_method || BENCH_RC=1
        vllm_done=1
      else echo "== [$m] already emitted by the vLLM sweep above (one sweep -> all vLLM arms) =="; fi;;
    *) echo "unknown method: $m"; BENCH_RC=1;;
  esac; done
  # COLLECT SCOPE. figure/collect.py reads the WHOLE $DATA directory by FILENAME, while a bench
  # run owns only the legs named in --methods. A collectable file this run did not produce came
  # from an EARLIER run, from a method left out of --methods, or is one of the shipped
  # historical JSONs repro/README.md declares uncitable -- and it is drawn into the figure with
  # nothing on the plot to say so. Name them; do not delete them (a split bench+vLLM run
  # legitimately combines two sweeps), just do not let them pass as this run's measurement.
  local f
  for f in bf16_hf.json ommx_hf.json kivi_hf.json kitty_hf.json \
           ommx_vllm.json vllm_bf16.json vllm_flash_attn.json turboquant_vllm.json; do
    case " $PRODUCED " in *" $f "*) continue;; esac
    [ -e "$DATA/$f" ] || continue
    echo "[collect] NOTE $f will be collected but was NOT produced by this run"
    echo "          ($DATA/$f -- earlier run, a method not in --methods, or a shipped JSON)"
  done
  return "$BENCH_RC"
}

do_figure() {
  # Both legs are CHECKED. Unchecked, this function printed "figure -> <two PNG paths>" and
  # returned 0 after collect.py and plot.py had BOTH failed (e.g. `python: command not
  # found`, or plot.py refusing an empty data set) -- naming output files that do not exist,
  # which is the console-only version of a silent fallback. plot.py failing is also NOT the
  # same as bench failing: the measured JSONs are already on disk, so say that instead of
  # sending the operator back to re-run the sweep.
  local PY; PY="$(venv_python ommx)"
  "$PY" figure/collect.py --src "$DATA" --gpu "$(echo "$TAG" | tr a-z A-Z)" \
    --ctxs "$CTXS" --out "figure/data/${TAG}.json" \
    || { echo "!! [figure] collect.py failed (rc=$?) -> no figure/data/${TAG}.json this run"; return 1; }
  "$PY" figure/plot.py --data "figure/data/${TAG}.json" --outdir figure \
    || { echo "!! [figure] plot.py failed (rc=$?) -> no PNGs. The measured data is SAFE in"
         echo "   $DATA/ and figure/data/${TAG}.json; fix the plot side and re-run"
         echo "   './run.sh figure --tag $TAG' -- do NOT re-run the sweep."; return 1; }
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
  # Not a benchmark: the one check whose failure cannot be undone after a push,
  # because a rewrite does not reach forks, reflogs or anyone who already fetched.
  check)  exec bash scripts/check_release_hygiene.sh "$@" ;;
  bench)  do_bench; exit "$BENCH_RC" ;;
  figure) do_figure ;;
  # `all` still draws the figure after a failed leg — the arms that DID prove themselves are
  # still worth plotting, and the missing ones simply have no bar — but the failure is carried
  # into the exit status rather than being erased by a successful plot.
  all)    do_bench; do_figure || BENCH_RC=1; exit "$BENCH_RC" ;;
  *) echo "usage: ./run.sh {setup|bench|figure|all} [opts]"; exit 2;;
esac
