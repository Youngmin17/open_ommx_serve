# `ommx_gpu_serve.linear` — OMMX weight quantizer, `OMMX_W_SafeTensor`, and the offline W-Packer

This package is the **offline half of `OMMX_Linear`** — the part of the ICCAD paper's Fig 6
that this release otherwise shipped no code for:

| paper claim | what it says | what lands here |
|---|---|---|
| **C3** | "`OMMX_Linear` applies this execution model to weight tensors stored in **`OMMX_W_SafeTensor`**" | `w_format.py` — the format did not exist; this file **is** the spec |
| **D3** | Fig 6: **HF Model (Safetensor Format) -> OMMX W-Packer -> OMMX Linear Path** | `w_packer.py` + its CLI |
| **D4** | Fig 6: offline packing steps (1) Vector Grouping (2) Min/Max Scaling (3) Top-K Permutation (4) FP4 Outlier Encoding | `quantize.py`, with each step marked by a `--- Fig 6 step (n) ---` banner |

**Scope, corrected.** This note used to read "It does **not** close C8 (a vLLM
`LinearMethodBase` / plugin backend for weights)". That is no longer true of the tree:
`integration/vllm/linear_method.py` defines an `ommx_w` `LinearMethodBase` and
`integration/vllm/plugin.py::_register_ommx_w_quant` registers it on every plugin load, so the
weight path *is* wired into vLLM. What remains true, and matters more than the wiring:

* **nothing on the `ommx_w` path has ever executed against a device.** Not the kernel reading a
  bundle, not `apply()`, not a vLLM engine loading one. There is **no `ommx_w` latency or accuracy
  number anywhere in this repo**, on any GPU, and none of the paper's kernel claims (C1, C5–C7) is
  closed by this package.
* what this package guarantees is **CPU-verified and artefact-level**: the bytes a bundle contains,
  the names those tensors carry, and that the output directory is one a loader can resolve. §5
  itemises verified vs GPU-unverified line by line.

Producing a loadable bundle is not serving one — but as of format version 2 the bundle *is*
loadable in principle, which it was not before (see §2).

```
quantize.py    the OMMX weight recipe as an importable module (was only ever inside a test file)
w_format.py    OMMX_W_SafeTensor: plane naming, JSON manifest, safetensors codec, validator
w_packer.py    HF safetensors checkpoint -> bundle, streaming, with a CLI and a --dry-run budget
```

Pure torch. Imports with no vLLM, no triton, no GPU — and no `safetensors` package either
(the container is implemented in `w_format.py`; see *Dependencies* below).

---

## 1. Why `quantize.py` exists

Before this package the **only** implementation of the OMMX weight recipe was
`csrc/linear/test_ommx_linear_parity.py::quantize_ommx_weight` — inside a test script, in a
directory with no `__init__.py` that is explicitly excluded from the installed package. Nothing
under `ommx_gpu_serve/` could pack a weight.

`quantize.py` is a **lift, not a rewrite**. That test script is a shipped, passing GPU gate
(`PARITY GATE: PASS` 5/5 on sm_90a — base decode cos 1.00000 / max_diff 5.96e-07, E8M0-vs-fp32
max_diff 0.00e+00, prefill cos 0.99999), so its numerics *are* the contract. The gate that
matters here is `tests/test_w_packer.py::test_bit_exact_vs_parity_packer`: every element of
every returned plane, compared with `torch.equal` (never `allclose`), over
shapes x group_size x outlier_pct x seed — 3 x 3 x 4 x 2, none skipped.
**72 parametrisations, 0 tolerances.**

The four format axes, unchanged:

1. group scale = (range **excluding outliers**) / 3, snapped to E8M0 power-of-two;
2. zero-point = group min (asymmetric affine, *not* symmetric);
3. `npv = max(1, int(group_size * outlier_pct))`, block == group;
4. outliers = FP4 E2M1 in **index space** through a per-group range map, `fp_range = 12`.

Two implementation details are load-bearing and documented in the source rather than tidied
away: the FP4 encoder is a **first-wins** nearest scan seeded at `1e30` (so exact midpoints and
the `-0.0` table entry resolve to the *lower* index), and the outlier stage runs in **float64**
because the reference encodes it in a per-element Python loop. Either one, "cleaned up",
silently changes stored nibbles.

Additive axes (all no-ops at their defaults, so the default path stays bit-exact):
`outlier_repr` (`relidx7` | `bitmap`), `outlier_map` (`idx_range` | `none`), `zp_dtype`
(`float32` | `bfloat16`), `reference=False` (skip the `W_base`/`W_ref` full-precision copies).

---

## 2. The `OMMX_W_SafeTensor` format (version 2)

The normative spec is the module docstring of `w_format.py`. Summary:

A bundle is a **self-contained model directory**: one `ommx_w-000NN-of-000MM.safetensors` shard
per input shard, `ommx_w_index.json` (tensor name -> shard, naming, recipe, aux-file record,
totals), and **every non-weight file of the source checkpoint** — `config.json`,
`generation_config.json`, the tokenizer files, the chat template, and anything else. Each shard is
a plain safetensors file whose file-level `__metadata__` carries `format: "ommx_w"`,
`version: "2"`, and `manifest` (the JSON document for that shard).

### 2.1 Plane names bind to a vLLM parameter — the version 1 -> 2 break

A plane is named after the **module** that owns the weight. The quantized tensor's `.weight`
suffix is **replaced**, not extended:

```
model.layers.0.self_attn.q_proj.weight          source checkpoint tensor
   -> model.layers.0.self_attn.q_proj.ommx_code    bundle plane
```

Version 1 wrote `...q_proj.weight.ommx_code`. That name **cannot be loaded by an engine**: vLLM
and transformers match a checkpoint tensor name against `dict(model.named_parameters())`, and a
parameter registered on a Linear module is reachable as `<module>.<attr>` — which is exactly why
AWQ ships `<module>.qweight` and not `<module>.weight.qweight`. A version-1 bundle fails at load
with a missing-parameter error from deep inside a weight loader.

The fix is in the **format**, not in a loader-side remap. `integration/vllm/linear_method.py`
carries `bundle_to_param_name` / `remap_bundle_weight_names` as a CPU-tested workaround for the
v1 names; against a v2 bundle both are identity and can be retired. The format had never shipped
and nothing external consumes it, so correcting it was free and strictly better than carrying a
translation step forever.

Version 1 is refused **by name**, with the reason and the re-pack command — never best-effort
parsed:

```
ommx_w-00001-of-00002.safetensors: this is an OMMX_W_SafeTensor VERSION 1 bundle, and this
build reads version 2 only. In version 1 planes were named '<module>.weight.ommx_<plane>'
(a '.weight' infix that no vLLM/transformers parameter has), so a version-1 bundle cannot be
bound by a model loader. There is no in-place migration (the tensor names themselves changed)
— RE-PACK the source checkpoint:
    python -m ommx_gpu_serve.linear.w_packer pack --input <source-checkpoint> --output <bundle>
```

The convention is **self-describing**: every shard manifest and the index carry a `naming` block
(`w_format.naming_document()`) giving the template, the dropped suffix, a worked example and the
rationale, and the validator refuses a bundle that declares a convention this build does not
implement. It also **recomputes** each plane's tensor name from `(weight, plane)` rather than
trusting the manifest's `name` field, so a shard cannot carry v1-named planes inside a v2
manifest and still be self-consistent.

### 2.2 Planes

One set per quantized Linear weight `[N, K]`, `G = K/group_size`, `<m>` = the owning module
(the tensor name minus `.weight`):

| plane | tensor name | dtype | shape |
|---|---|---|---|
| dense INT2 payload | `<m>.ommx_code` | `U8` | `[N, K/4]` |
| E8M0 shared exponent | `<m>.ommx_scale_exp` | `I8` | `[N, G]` |
| shared zero-point | `<m>.ommx_zp` | `BF16` | `[N, G]` |
| outlier positions | `<m>.ommx_oindex` | `U8` | `[N, G, ceil(npv*7/8)]` (relidx7) or `[N, G, ceil(gs/8)]` (bitmap) |
| FP4 outlier codes | `<m>.ommx_ocode` | `U8` | `[N, G, ceil(npv/2)]` |
| FP4 range map | `<m>.ommx_map_scale`, `<m>.ommx_map_center` | `F32` | `[N, G]` |

Dequant (mirrored by `csrc/linear/ommx_linear.cu`):
`w = code * 2^e + z`, with outlier lanes **overwritten** by `w = (fp4(nib)/ms + mc) * 2^e + z`.

Everything else — embeddings, `lm_head`, norms, biases, anything unrecognised — is copied
through under its original name and dtype and recorded as `kind: "passthrough"` **with a
reason string**. A silently quantized `lm_head` surfaces only as a slightly worse eval score,
so the bundle writes down the decision for every single tensor.

### 2.3 The model files (`aux_files`)

`pack` used to write only the shards and the index, so the documented recipe — pack, then point
vLLM at the bundle directory — could not resolve the model at all. It now copies the source
checkpoint's non-weight files through, **by exclusion rather than by a whitelist**: a file the
packer has never heard of is *copied*, because the set of files HF needs keeps growing and a
missing one shows up as behaviour that reads like quantization damage.

| source file | what happens | why |
|---|---|---|
| `config.json` | **copied; a source without one is a hard `PackError`** | it is the file a loader opens first to decide which architecture to build. Warning instead would move the diagnosis into a transformers traceback an hour later, on a directory the packer had already reported as written |
| `generation_config.json`, `tokenizer_config.json`, `special_tokens_map.json` | copied; a **loud warning** if absent | legal to lack, but each one silently changes behaviour (eos/pad ids, sampling defaults, chat template) |
| `tokenizer.json` / `tokenizer.model` / `vocab.*` / `merges.txt` / … | copied; warning if **none** of them is present | which one a model ships is architecture-dependent; with none, the bundle needs an explicit `--tokenizer` |
| `chat_template.json`, `chat_template.jinja`, `preprocessor_config.json`, anything else | copied | exclusion policy: unrecognised is not a reason to drop |
| `*.safetensors`, `*.bin`, `*.pt`, `*.gguf`, … **and** `*.safetensors.index.json` / `*.bin.index.json` | **never copied**, recorded in `skipped` with the reason | this is the correctness one. A leftover `pytorch_model.bin` beside the ommx planes is a loader candidate, and a loader that picks it serves the **original unquantized weights** while the arm is labelled `ommx_w` — the same class of defect `MEASURED_FACTS` §2 records on the attention side, where a `CUSTOM` run was byte-identical to bf16 and timed under the OMMX label. A stale weight index additionally names shards the bundle does not contain |
| subdirectories | not recursed; recorded in `skipped` | flattening one would rename its files |

The index records the facts: `policy` (`copy` | `no-copy-aux`), `self_contained`, `copied`
(name + bytes + sha256), `skipped` (name + reason), `missing` (name + reason).

**`--no-copy-aux`** opts out, and is warranted: a source directory may hold files that must not
be redistributed with the bundle, an operator may be packing a shard subset for a kernel test and
will point the engine at the original checkpoint, or the config must be *replaced* rather than
copied. It is safe to offer only because it cannot produce a bundle that *looks* self-contained —
the index records `policy: "no-copy-aux"`, `self_contained: false`, both `pack` and `verify` say
"NOT a loadable model directory", and the validator refuses a bundle that claims otherwise.

**The validator** (`validate_bundle` / `validate_shard` / `validate_aux_files`) refuses a bundle
whose manifest and tensors disagree, naming the mismatch: a missing or extra plane, a plane whose
tensor name is not the one the format derives, a shape or dtype that differs from what the recipe
*derives* from `[N, K]` (so a manifest edited to agree with a tampered tensor is still caught), a
tensor present but undeclared, a `packed_bytes` / `bits_per_weight` that does not match the bytes
on disk, shards whose recipes disagree with each other, a foreign `naming` block, a wrong or
absent `version` — and, for the model files, an `aux_files` record that is missing, names a file
that is not in the directory, leaves a required file empty, escapes the directory with a path
separator, or claims a `self_contained` that the facts do not support. The packer runs it on its
own output before returning — a packer that can emit a bundle its own validator rejects is worse
than one that cannot pack at all.

**Where that exclusion is enforced, stated exactly.** It is a PACK-time rule. `validate_bundle`
checks what the index claims against what is in the directory; it does **not** sweep the
directory for files nobody claimed, so a `pytorch_model.bin` dropped into a finished bundle by
hand is not refused by `verify`. The packer can never produce one — but "this bundle validates"
means "the packer's record of it is true", not "nothing else is in this directory". The one case
the packer COULD produce (stale `ommx_w-*.safetensors` from a re-pack at a different shard count)
is refused at pack time by `_stale_shard_guard`; see §5.

Recorded `bytes`/`sha256` are **provenance, not a gate**: editing `config.json` inside a bundle
(rope scaling, `max_position_embeddings`) is legitimate, so `verify` reports which aux files
changed since packing and the bundle stays valid. A validator that failed on that would only
teach operators to skip validation.

---

## 3. Packing

```bash
# no model on this host and no network? generate a checkpoint to pack:
python -m ommx_gpu_serve.linear.w_packer make-synthetic --output /tmp/ckpt

# see the byte budget BEFORE spending an hour packing (writes nothing):
python -m ommx_gpu_serve.linear.w_packer pack --input /tmp/ckpt --output /tmp/bundle --dry-run

# pack, then validate what was written:
python -m ommx_gpu_serve.linear.w_packer pack   --input /tmp/ckpt --output /tmp/bundle
python -m ommx_gpu_serve.linear.w_packer verify --bundle /tmp/bundle

# bits/weight for every recipe, no checkpoint needed:
python -m ommx_gpu_serve.linear.w_packer budget
```

Recipe knobs: `--group-size` (64), `--outlier-pct` (0.0625) **or** `--npv` (never both),
`--outlier-repr relidx7|bitmap`, `--outlier-map idx_range|none`, `--zp-dtype bf16|f32`.
Bundle knobs: `--overwrite`, `--dry-run`, `--json <path>`, and `--no-copy-aux` (§2.3).
`make-synthetic --no-aux` writes shards with **no** `config.json` — i.e. the checkpoint shape
`pack` is supposed to refuse, so the refusal is exercised by the same fixture as the success.

The tail of a `pack` report says what the bundle actually became:

```
  model files   : policy=copy  self_contained=True
    copied      : chat_template.jinja, config.json, generation_config.json,
                  special_tokens_map.json, tokenizer.json, tokenizer_config.json
    skipped     : 3 file(s) (weights / indexes / dirs; see ommx_w_index.json)
```

and `verify` repeats it, so "can I point an engine at this directory?" is answerable without
opening the index by hand.

It **streams**: one input shard open at a time, one tensor materialised at a time, output
payload spilled to a temp file so the safetensors header can be written last. Peak RAM is one
tensor. It **refuses** silently-wrong input rather than working around it — a `K` the group
size does not divide, a dtype it cannot quantize, an output directory that already holds a
bundle (with a *different* manifest: refused outright; with the *same* manifest: needs
`--overwrite`), an `--overwrite` that would strand OMMX shards from an earlier pack because the
shard COUNT changed, an `--output` equal to `--input`, a source checkpoint with no `config.json`, a
non-finite weight, `relidx7` with a group larger than 128 (7 bits cannot address it — the
message points at `bitmap`).

---

## 4. Bit budget, and where the paper's 3.63 comes from

`--dry-run` reports per-tensor packed bytes and bits/weight; `budget` reports the whole grid.
**Stored** includes the per-group byte padding the layout actually requires; **unpadded** is the
information-theoretic figure.

| gs | npv | %out | repr | map | stored | unpadded |
|---:|---:|---:|---|---|---:|---:|
| 64 | 4 | 6.25% | relidx7 | **idx_range** | **4.1250** | 4.0625 |
| 64 | 8 | 12.5% | relidx7 | **idx_range** | **4.7500** | 4.7500 |
| 64 | 16 | 25.0% | relidx7 | **idx_range** | **6.1250** | 6.1250 |
| 64 | 4 | 6.25% | relidx7 | none | 3.1250 | 3.0625 |
| 64 | 8 | 12.5% | relidx7 | none | 3.7500 | 3.7500 |
| 64 | 16 | 25.0% | relidx7 | none | 5.1250 | 5.1250 |
| 64 | 4 | 6.25% | **bitmap** | **none** | **3.6250** | **3.6250** |
| 64 | 4 | 6.25% | bitmap | idx_range | 4.6250 | 4.6250 |
| 128 | 16 | 12.5% | relidx7 | none | 3.5625 | 3.5625 |

Bolded in column *stored* are the three figures `csrc/linear/README.md` publishes —
**4.125 / 4.750 / 6.125** at npv 4/8/16, gs=64, `relidx7` + `idx_range`, which is what
`w_format.Recipe` defaults to. They are reproduced exactly and pinned by
`test_readme_published_bit_budgets_reproduce`.

**Correction, carried in both directions.** That test used to be called
`test_shipped_readme_bit_budgets_reproduce` and pinned 3.06 / 3.75 / 5.125, because that is what
`csrc/linear/README.md` published at the time. It no longer does, and the gate now pins the
current figures *plus the exact size of each of the two errors*, so the prose and the arithmetic
cannot drift apart again:

* the old figures **omitted the `map_scale` / `map_center` planes**, which the timed kernel takes
  as arguments. Two F32 per group is **+1.0 bit/weight exactly** at gs=64 — the `fp4_range_map`
  term. The `map = none` rows above are the like-for-like comparison; the `idx_range` rows are
  what the shipped encoding costs on disk;
* at npv=4 they quoted the **unpadded** column. Four `relidx7` positions are 28 bits and the
  layout pads the stream to 4 bytes per group, so stored is **3.125**, not 3.0625 (and 4.125, not
  4.0625, with the map). At npv 8/16 the stream is already byte-aligned, so `stored == unpadded`
  there and only the npv=4 figure moves for this reason.
* **The paper's weight AvgBits 3.63 (claim E2) is reachable, and the recipes that reach it are
  overwhelmingly the *bitmap, no-range-map* family.**
  `2 (INT2) + 1 (flat bitmask, 1 bit/element) + 8/64 (INT8 E8M0 scale) + 16/64 (BF16 ZP) +
  4*4/64 (FP4 outliers) = 3.625`, which displays as 3.63 under round-half-up. That is
  self-consistent with the paper's own two statements: claim **B4** ("on our GPU implementation,
  positions are stored as a flat bitmask, N bits per group") and claim **B6**, whose bundle
  definition lists a dense payload, SC, ZP, position metadata and extension codes — and **no**
  range-map parameters. At `gs=64` the shipped `relidx7 + no map` ladder steps
  3.125 -> 3.75 straight past 3.63, so the *published* relidx7 grid (npv 4/8/16) cannot land on
  it — but "relidx7 can never reach 3.625" is **false**, and the exhaustive scan below is what
  proves it.

`test_paper_3_63_avgbits_is_the_bitmap_recipe` pins this, and pins the WHOLE solution set rather
than one member of it. Scanning every `(gs, npv, repr, map)` in `{16,32,64,128} x [0..gs]` with
the paper's BF16 zero-point (claim B9) gives exactly five recipes at 3.6250 stored:

| gs | npv | %out | repr | map |
|---:|---:|---:|---|---|
| 64 | 1 | 1.56% | relidx7 | idx_range |
| 64 | 3 | 4.69% | **bitmap** | **none** |
| 64 | 4 | 6.25% | **bitmap** | **none** |
| 128 | 13 | 10.16% | **bitmap** | **none** |
| 128 | 14 | 10.94% | **bitmap** | **none** |

Four of the five are `bitmap + none`, which is the family B4/B6 describe. The fifth
(`gs=64, npv=1, relidx7, idx_range`) reaches 3.625 by coincidence — one outlier per group of 64
is 1.56%, outside the 6.25 / 12.5 / 25% grid this repo publishes — but it exists, so the claim
here is "3.63 is the bitmap/no-map family", NOT "3.63 identifies a unique recipe".
`gs=64, npv=4` is the member quoted above because it is the repo's default outlier budget;
`gs=128, npv=13` or `14` is equally consistent and matches the paper's own `N = 128` worked
example (claim B10). Which one the paper used is **not determined** by the AvgBits figure alone.

---

## 5. What is verified — and what is not

**Verified on CPU, this session** (`python3 -m pytest ommx_gpu_serve/tests/test_w_packer.py -q`,
**147 passed, ~1.4 s**, no GPU / triton / vLLM / safetensors installed):

* bit-exactness of `quantize.py` against the parity gate's packer, every plane, `torch.equal`;
* the relidx7 and bitmap position streams equal `attention/codec.py`'s `pack_relidx7` /
  `pack_bitmap_row` byte for byte;
* pack -> write -> read -> dequant reproduces `W_ref` **exactly** (no tolerance) for five
  recipes including npv=0 and an odd npv;
* **plane names carry no `.weight` infix** — asserted on the helper, on its inverse
  (`split_plane_name` round trip), and on the tensor names read back out of a packed shard, so a
  packer that bypassed the helper is still caught. `plane_name` refuses a tensor that is not a
  `.weight` rather than inventing a name for it;
* **a packed bundle is a self-contained model directory**: `config.json`, the tokenizer files,
  the chat template and an unrecognised extra file all arrive byte-identical to the source, while
  `pytorch_model.bin`, the source's own `*.safetensors` and `*.index.json` are refused entry and
  recorded in `skipped` with a reason;
* a source checkpoint with no `config.json` is a hard refusal that names `--no-copy-aux`, leaves
  no partial directory behind, and fires under `--dry-run` too;
* `--no-copy-aux` produces a bundle that records the opt-out and says, in the pack report, in
  `verify`, and in the index, that it is not loadable on its own;
* the manifest describes every tensor written and nothing else; the validator rejects a dropped
  plane, a corrupted shape, an unknown version, a **version-1 bundle by name with the re-pack
  command**, a **foreign `naming` block**, a **v1-named plane smuggled into a v2 manifest**, a
  smuggled tensor, a cross-shard recipe disagreement, a lying bit budget, and — for the model
  files — a vanished `config.json`, an empty one, a path-escaping entry, an absent `aux_files`
  record, a false `self_contained` claim, an unknown `policy` value, a foreign or absent `naming`
  block **in the index as well as in a shard**, and a plane stored on disk with a dtype the
  manifest does not declare;
* the packer **re-validates its own output** before returning, so "pack reported success" means
  "this directory validates"; a source subdirectory and the packer's own `ommx_w_index.json` are
  recorded in `skipped` rather than copied;
* the read path (`load_planes` / `load_weight`) derives plane names instead of trusting the
  manifest, so the validator and the loader cannot disagree about what a shard contains;
* `lm_head` / embeddings / norms / biases survive byte-identical and un-quantized;
* reported bits/weight == an independent longhand recomputation == bytes actually on disk;
* the safetensors container round-trips every dtype the packer emits and refuses a truncated file.

Every gate added this session was **mutation-tested**: the code under it was deliberately broken
(plane_name restored to the v1 infix; the packer made to bypass `plane_name`; the version check
made to accept v1; the missing-`config.json` refusal turned into a warning; the weight-file
exclusion disabled; `config.json` recorded as copied but never written) and the gate observed to
fail in each case, then reverted. A gate that survives its own mutation is not a gate.

**An adversarial re-run then mutated all of them, plus every guard they touched — 48 mutations,
not a sample.** Thirty-eight were killed by the intended gate. Two were behaviour-equivalent (the
digest taken from the source rather than from the byte-identical copy; the previous-pack shard
exclusion, which `*.safetensors` already covers) and one expectation was simply mis-stated, so
none of those three is a defect. The remaining **seven walked straight through**, and each
SURVIVOR is now closed by a gate of its own (`test_validator_rejects_a_foreign_naming_convention_in_the_INDEX`,
`..._a_shard_manifest_with_no_naming_block`, `..._an_unknown_aux_policy`,
`..._a_plane_stored_with_the_wrong_dtype`, `test_pack_validates_the_bundle_it_just_wrote`,
`test_a_source_subdirectory_is_recorded_and_never_flattened_in`,
`test_a_stale_ommx_w_index_in_the_source_is_not_copied`), each one shown failing under the exact
mutation that had walked through. The two fixes below were themselves mutated six more ways
(each guard removed, each guard made unconditional, each guard moved after the write); every one
was killed. In every case the CODE was already right and only the coverage
was missing — except for two REAL defects, both now fixed and gated:

* **`load_planes` / `load_weight` trusted the manifest's plane `name` field** while
  `validate_shard` recomputed it. A shard carrying v1-named planes inside a v2 manifest was
  therefore REFUSED by the validator and READ ANYWAY by the module's own CPU oracle — the one
  path that could produce clean-looking numbers for a bundle no engine can bind. The read path
  now derives the name the same way and refuses a manifest that declares another; pinned by
  `test_the_read_path_derives_plane_names_too_not_only_the_validator`, which also asserts the
  untampered bundle still loads (so "raise on everything" does not satisfy it).
* **`--overwrite` left OMMX shards from an earlier pack in the bundle.** The shard file name
  encodes the shard COUNT (`ommx_w-00001-of-00003`), so re-packing the same model after the
  source checkpoint was re-sharded wrote a new set of names and left the old set behind. The
  index referenced only the new ones, so `verify` reported OK on a directory holding **two
  complete, disagreeing copies of every plane** — the leftover-weight-file hazard above,
  arriving through the packer's own output instead of the source's, and reachable in one
  command. `_stale_shard_guard` now refuses BEFORE anything is written (it does not delete
  files it did not create in this run) and names both the stale files and the ones the pack
  would have written; pinned by
  `test_repacking_after_a_reshard_refuses_to_leave_stale_ommx_shards`, which also asserts the
  previous bundle is left byte-identical and that a fresh output directory still packs.

**UNVERIFIED (no GPU this session)** — the cluster GPUs were unavailable for the duration and
this host has no device (no utilisation figure was measured; do not quote one). **Nothing on
the `ommx_w` path has ever executed against a device**, so all of the following are open:

* that `csrc/linear/ommx_linear.cu` decodes these planes to the same numbers. That is what
  `csrc/linear/test_ommx_linear_parity.py` measures, and it asserts `torch.cuda.is_available()`
  on entry. The bridge is the bit-exactness gate above: this packer emits the same bytes as the
  packer that gate already blesses, so a passing parity run transfers — but it has not been
  re-run against a bundle produced by *this* code;
* **that a vLLM engine loads a bundle.** The name mismatch that made it impossible is fixed in
  the format and the fix is pinned on the artefact — but "the plane name equals the parameter
  name" and "the copied files make a resolvable model directory" are CPU assertions about bytes
  and file names. No engine has read one. Whether vLLM's loader, TP sharding, and the fused
  `qkv_proj` / `gate_up_proj` mapping actually bind these planes is untested;
* every latency and accuracy question. There is **no `ommx_w` TPOT or eval number in this repo**;
* the `bitmap` position representation has **no kernel reader in this repo**. `ommx_linear.cu`
  decodes `relidx7`. `bitmap` is a format option and a bit-budget statement, not a wired-up
  execution path;
* end-to-end packing of a real checkpoint. The largest weight packed here is 128 x 64 (8192 weights, the synthetic fixture's
  `gate_proj`/`up_proj`), and the largest single-weight round trip is 8 x 256; the
  streaming design is what makes a real model possible, not a measurement that one was packed;
* interop with the official `safetensors` Rust/Python implementation. It is not installed on
  this host. `w_format.py` follows the published container spec (8-byte LE header length, JSON
  header, 8-byte-aligned payload) and self-round-trips, but cross-implementation interop is
  untested.

No accuracy claim is made anywhere in this package. Synthetic Gaussian weights have no heavy
tails, so outlier handling scores near-perfectly on them regardless of `npv` — that is mechanic
parity, not accuracy.

---

## 6. Dependencies and the two build-environment requirements

This package needs **torch only**. The `safetensors` container is implemented in `w_format.py`
(8 bytes of length + a JSON header + a flat byte buffer), so the packer does not add a
dependency and runs on a host where `safetensors` is not installed — which is where it was
written and tested.

The **kernels** are a separate matter. If you go on to run the GPU parity gate against a bundle,
two environment requirements are undeclared elsewhere and both were measured the hard way:

1. **the `ninja` EXECUTABLE must be on `PATH`.** `torch.utils.cpp_extension.load()` shells out
   to the binary, not the wheel. Installing the wheel drops `ninja` into `<env>/bin/`, which is
   on `PATH` only while the env is *activated*; invoking `/path/to/env/bin/python ...` directly
   is not enough. Without it: `RuntimeError: Ninja is required to load C++ extensions`.
2. **`LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64`** on cu13 conda envs. torch emits
   `-L$CUDA_HOME/lib64` while those envs keep `libcudart.so` in `lib/`. Without it:
   `/usr/bin/ld: cannot find -lcudart`.

**Packaging note.** `ommx_gpu_serve/pyproject.toml` lists its packages explicitly rather than
using `find:`, so a new subpackage is invisible to the build until it is named there.
`ommx_gpu_serve.linear` is now listed (with `README.md` as package-data). This mattered more
than it looks: `ommx_gpu_serve.tests` **is** shipped in the wheel and `tests/test_w_packer.py`
imports `ommx_gpu_serve.linear`, so while the subpackage was missing every one of these 147
gates stayed green in a source checkout and raised `ModuleNotFoundError` on an installed one.
Verified by building the wheel and listing its contents, not by reading the toml.

`csrc/` is still deliberately **not** shipped. `test_w_packer.py` loads the reference packer
`csrc/linear/test_ommx_linear_parity.py` by file path, so on an installed tree that one gate
SKIPS with an explicit reason; if `csrc/linear/` exists but the reference file does not, it
FAILS hard instead — a deleted format contract is a regression, not an environment.
