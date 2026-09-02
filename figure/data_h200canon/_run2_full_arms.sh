#!/usr/bin/env bash
# Quantify the recipe asymmetry inside Fig.7's OMMX bar.
#
# bench_e2e_a100 defaults to `--recipe fakequant` (group_tokens/channels 64/64, sink 8,
# recent 8, select "abs", npv 6 -> 6 outliers per 64-token group). That is NOT the canonical
# published serving recipe (32/32, recent 32, select "signed", npv 6 -> 12 per 64), which is
# the number system every accuracy result in the paper was produced under. The bench file
# says so itself. This run measures the SAME abl_attn arm under the canonical recipe so the
# gap between "the speed figure's recipe" and "the accuracy figures' recipe" is a number.
set -o pipefail
PY=/scratch/uceeeee/conda_envs/ommx/bin/python
REPO=/scratch/uceeeee/fig7/open_ommx_serve
cd "$REPO" || exit 2
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/scratch/uceeeee/.cache/huggingface
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_PLUGINS=ommx_gpu_serve PYTHONPATH="$REPO"
# Canonical published serving recipe (README / figure/bench.py OMMX_RECIPE_ENV).
# --recipe current makes bench_e2e pass NO fakequant dict, so these survive into the arm.
export OMMX_KV_GROUP_TOKENS=32 OMMX_KV_GROUP_CHANNELS=32 OMMX_KV_SINK=8 OMMX_KV_RECENT=32
export OMMX_ATTN_POW2=1 OMMX_ATTN_K_FORMAT=i2f4 OMMX_ATTN_OUTLIER_SELECT=signed
export OMMX_ATTN_OUTLIER_REPR=relidx7 OMMX_ATTN_COMBINADIC_READ=0 OMMX_KV_OUTLIER_MAP=1
export OMMX_ATTN_CUDA_DECODE=0
echo "== recipe-delta (canonical 32/32/recent32/signed) CVD=$CUDA_VISIBLE_DEVICES $(date -u) =="
"$PY" -m ommx_gpu_serve.bench.bench_e2e_a100 --model meta-llama/Llama-3.1-8B-Instruct \
  --cmp-ctxs 1024,4096,8192,16384,32768,65536 --abl-ctxs 1024,4096,8192,16384,32768,65536 \
  --recipe current --outliers 6 --gpu-mem 0.70 --warmup 16 --measure 160 \
  --only-arms "TRITON,FA,abl_attn,abl_attn_no_all,abl_attn_no_unpack,abl_attn_no_dequant,abl_attn_skipwrite" --arch h200canon \
  --workdir "$REPO/runs/e2e_h200canon_arms" \
  --csv "$REPO/figure/data_h200canon/_e2e.csv" --json "$REPO/figure/data_h200canon/_e2e.json"
echo "CANON_FULL_EXIT=$?"
"$PY" - "$REPO/figure/data_h200canon/_e2e.json" <<'PYEV'
import json, sys
r = json.load(open(sys.argv[1]))
a = (r.get("arms") or {}).get("abl_attn") or {}
for k in sorted(a, key=lambda x: int(x.split(":")[-1]) if ":" in x else 0):
    v = a[k]
    if isinstance(v, dict):
        print("CANON", k, "ok=", v.get("ok"), "tpot=", v.get("tpot_p50"))
ev = r.get("route_evidence") or {}
print("OMMX_PROVEN=%d" % (1 if ev and all(e.get("ok") for e in ev.values()) else 0))
PYEV
echo "CANON_FULL_DONE $(date -u)"

PY_OMMX=/scratch/uceeeee/conda_envs/ommx/bin/python
D=/scratch/uceeeee/fig7/open_ommx_serve/figure/data_h200canon
"$PY_OMMX" /scratch/uceeeee/fig7/open_ommx_serve/ommx_gpu_serve/bench/e2e_to_figure.py \
  --in "$D/_e2e.json" --out-vllm-bf16 "$D/vllm_bf16.json" --out-ommx-vllm "$D/ommx_vllm.json" \
  --flash-arm "$D/vllm_flash_attn.json" --flash-arm-name FA
echo "CANON_ADAPT_EXIT=$?"
# reuse the already-measured KIVI + LMDeploy bars: they do not depend on the OMMX recipe
cp -n /scratch/uceeeee/fig7/open_ommx_serve/figure/data_h200fig7/kivi_hf.json "$D/" 2>/dev/null
cp -n /scratch/uceeeee/fig7/open_ommx_serve/figure/data_h200fig7/lmdeploy.json "$D/" 2>/dev/null
echo "CANON_FULL_ADAPTED $(date -u)"
