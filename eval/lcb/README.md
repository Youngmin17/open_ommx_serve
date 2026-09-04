# LiveCodeBench-v6 — KV-quantization accuracy comparison

Does low-bit KV quantization damage a reasoning model's **code generation**, and if so, how?

Four KV-quant methods and a bf16 baseline run through one generative path on
**Qwen3-30B-A3B-Thinking-2507**, so every arm is directly comparable. All quantization is
**cache-level fake-quant** — the KV cache is quantize→dequantized at the `transformers`
`Cache.update()` boundary, at the projected pre-RoPE K/V, identically for every method. That is an
accuracy emulation, not a kernel benchmark: it reproduces the numerical error the real CUDA kernel
produces, which is what determines benchmark accuracy, and it is model-agnostic, so KIVI / Kitty /
TurboQuant run unchanged on this MoE even though their upstream stacks cannot.

```
eval/lcb/
├── kv_fakequant/      the cache-level quantizers, and the fp16 region each method's paper specifies
│   ├── cache.py       enforces the front sink and the trailing residual window
│   ├── quantizers.py  KIVI / Kitty / TurboQuant + PUBLISHED_RECIPE
│   └── test_residual_sink.py   CPU self-test: proves the regions are real
├── run_lcb_queue.sh   generation: one GPU slot, arms interleaved shard by shard, per-job retry
├── score_lcb.sh       offline scoring under ONE policy for every arm (CPU only)
├── analyze_lcb.py     recipe-fidelity gate + paired table + McNemar + the figure's data file
└── bit_budget.py      average KV bits for the exact configs run here
```

Figure: `figure/plot_lcb.py`.

## Reproduce

```bash
# 0. the fp16 region every baseline depends on -- run this first, it needs no GPU and no model
python eval/lcb/kv_fakequant/test_residual_sink.py

# 1. generate (GPU). Resumable; safe to stop and restart.
KV=<baseline/kv> LCB_CACHE_DIR=<dir with index.json> bash eval/lcb/run_lcb_queue.sh

# 2. score (CPU only — no model-written code ever runs on the GPU node)
KV=<baseline/kv> LCB_CORPUS=<n120 corpus .pkl.gz> bash eval/lcb/score_lcb.sh

# 3. analyze + figure
python eval/lcb/analyze_lcb.py --runs <runs dir> --ids <lcb_subset_ids.json> \
       --json figure/data/lcb_qwen3_30b.json
python figure/plot_lcb.py --data figure/data/lcb_qwen3_30b.json --outdir figure
```

## What this comparison controls

Fixed and identical across every arm:

| | |
|---|---|
| model + revision | Qwen3-30B-A3B-Thinking-2507, pinned |
| decode | `T=0.6, top_p=0.95, top_k=20, seed=42`, 1 sample per problem |
| batch size | **1** for every arm, baseline included |
| `max_new_tokens` | 16384 |
| problems | one seeded subset, `subset_seed=20250729`, n=120 |
| pairing | each arm compared to bf16 **restricted to that arm's own ids** |
| quantization point | projected pre-RoPE K/V in `Cache.update()` |
| scoring | one policy for all arms, verified reproducible (see below) |

Three of these are not housekeeping — each one was a live defect that silently changed results:

**Batch size.** `bs>1` changes left-padding and the fake-quant group boundaries. An OMMX arm run
at `bs=4` scored exactly 0.0000 on AIME. Generations produced at `bs=4` were quarantined and
regenerated rather than merged into `bs=1` arms; `run_lcb_queue.sh` refuses to resume into an
output whose recorded `batch_size` is not 1.

**Scoring must not depend on machine load.** The upstream harness bounds each test by *wall*
clock. On a shared node that makes the verdict a function of who else is running: the same 90
generations scored `pass=43 / timeout=8`, then `pass=35 / timeout=17` thirteen minutes later —
passes cannot decrease when samples are added. Worse, it was directional: the bf16 baseline had
been graded on an idle node (1.7% timeouts) and the quantized arms under load (18.5%), biasing
every delta against quantization. Scoring now bounds **CPU** time via `RLIMIT_CPU`, which does not
move with load. Three independent scorings under load then agreed on every verdict (0 flips).

**A resource failure is not a wrong answer.** Under memory pressure `fork()` returned `ENOMEM`;
the harness recorded `spawn_error` and the scorer counted it as 0, publishing one arm at exactly
`pass@1 = 0.0000`. Fixed three ways: the scoring corpus carries only the 120 problems actually
scored instead of all 1055 (10× less memory per worker), transient spawn failures are retried, and
a gate refuses to publish any result whose `spawn_error` rate exceeds 2%.

Every arm was also checked against a silent-fallback failure: **zero** generations are
byte-identical to bf16 in any arm (median absolute length difference 4.4k–11.5k tokens), so no arm
is a disguised bf16 passthrough.

## What this comparison does NOT control

**It is not iso-bit.** Each arm runs its own recipe, and the payload budgets differ by up to 50%:

| arm | recipe | avg KV bits (payload) |
|---|---|--:|
| KIVI | plain INT2, group 128 | 2.000 |
| OMMX | INT2 + 12/64 FP4 K-outliers, group 64, sink 8, pow2 | 2.188 |
| Kitty | INT2 + 3/128 bf16 channels, group 128 | 2.328 |
| TurboQuant | Walsh–Hadamard rotate + INT3, group 128 | 3.000 |

Those are **payload only**. Scale/zero-point (+0.25 bit/elem at group 128, +0.5 at group 64) and
OMMX's outlier-*membership* encoding — which 12 of each 64 elements are FP4, up to +1.0 bit/elem
as a bitmap — are excluded. A metadata-inclusive budget is **not established** for the OMMX cache
path, so this table orders payload, not wire format, and does not support "method A uses fewer
bits than method B". Run `bit_budget.py` for the derivation.

**The token cap is a first-order confound.** At `max_new_tokens=16384` a thinking model runs out
of tokens mid-reasoning and never emits its code fence, which the benchmark scores 0. Measured:
`no_code_block` is 97–100% token-capped and 100% fence-less. Quantization inflates generation
length (KIVI +46%, Kitty +37% vs bf16), so raw pass@1 partly measures verbosity. Reporting is
therefore always three numbers — pass@1, cap-hit rate, and pass@1 over pairs where neither side
hit the cap. The cap-free restriction conditions on a post-treatment variable, so treat it as a
bound, not a causal estimate; McNemar on the discordant pairs is the test the question needs.

Also note: the benchmark's own gate counts `no_code_block` as a *harness* failure with a 5%
threshold. That definition is wrong here — the status is a model/protocol outcome, and by it the
bf16 baseline itself fails at 10.8%.

**Each method's fp16 region is now enforced — earlier runs did not have it.** Every method in
this family keeps some tokens out of the quantizer, and that region is part of the recipe:

| arm | high-precision region its paper specifies | enforced by |
|---|---|---|
| KIVI | fixed 128-token fp16 residual | `cache.py`, `PUBLISHED_RECIPE["kivi"]` |
| Kitty | 32-token attention sink | `cache.py`, `PUBLISHED_RECIPE["kitty"]` |
| OMMX | 8-token attention sink | its own quantizer (`attention_sink_num=8`) |
| TurboQuant | none specified | left at 0, and flagged rather than guessed |

The original cache implemented none of this. It committed `(T // group_size) * group_size` tokens
and left the rest fp16, so the residual was `T mod group_size` — a sawtooth between 0 and 127 at
group 128, mean ≈ 63.5, and exactly **zero** whenever T was a multiple of 128. `residual_length`
was stored on the quantizer and printed as `res=128` in every result file, but nothing read it;
`sink` did not exist. Because OMMX's sink lives inside its own quantizer, it survived — so the
defect was directional, stripping the baselines while leaving OMMX intact.

Three changes stop it recurring:

- `cache.py` owns both regions and `describe()` reports only what it enforced, so a result file
  cannot advertise a residual that was not applied;
- `PUBLISHED_RECIPE` is applied by default, so running a method stripped of its region takes an
  explicit `apply_published_recipe=False` rather than an omission;
- `test_residual_sink.py` asserts the regions positionally with a sentinel quantizer, and
  `analyze_lcb.py` re-checks each run's recorded `kv_desc` against the recipe and says so loudly
  when a run predates the stamping.

Measurements taken before this fix are kept as `figure/data/lcb_qwen3_30b_stripped_recipes.json`
with that provenance recorded in the file. **They are not the repo's result**: in them KIVI ran
without its residual and Kitty without its sink, so those two columns bound their methods from
below. A re-run under the corrected cache supersedes them.

**Other limits.** One sample per problem, so there is no within-problem variance estimate; the
subset gives a minimum detectable difference of roughly 10 points on a mean, which is why
arm-vs-arm claims rest on paired McNemar rather than on the gap between two means. The harness is
unofficial and its numbers are **not** comparable to published LiveCodeBench results.

## Scope of the claims this supports

Arms that stop at different coverage are reconciled by the **common subset** — every arm,
baseline included, restricted to the ids every arm generated. On that set all arms share one
problem list, one decoding configuration and one scoring policy, so arm-vs-arm ranking is
licensed. `analyze_lcb.py` prints it as the primary table and prints the per-arm-coverage view
second, labelled as arm-vs-bf16 only.

Two things the common subset does **not** repair, and which every reading has to carry:

- the comparison is **not iso-bit** (2.00–3.00 payload bits/elem, metadata unaccounted), so a
  ranking is a ranking of these five configurations, not of the methods at equal cost;
- TurboQuant has **no** declared fp16 region here. That is recorded as unverified rather than
  guessed, so its column carries an open question the other arms do not.
