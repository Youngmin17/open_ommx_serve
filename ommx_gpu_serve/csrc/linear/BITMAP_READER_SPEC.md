# `ommx_linear` — native bitmap outlier-position reader

**Status: IMPLEMENTED (2026-09-02), COMPILED, NOT YET RUN.** The reader exists in
`ommx_linear.cu` and `nvcc -arch=sm_90a` accepts it cleanly; nothing in it has executed on
a device. The shape that landed differs from §5.2 on one point: a RUNTIME `idx_fmt`
argument rather than a compile-time traits tag. The branch is uniform across a launch (it
comes from the recipe, not from data) and a missing argument is an arity error, which is
the failure mode worth having when the change cannot be iterated against a GPU. A
second difference is in §5.1's favour: the resident encoding is now CHOSEN as the cheaper
of the two rather than fixed at relidx7, so a bundle is re-encoded only when it is being
rewritten for another reason. Everything below is the design as specified; read it for the
reasoning, not as a description of the current call signatures.
It was written on a host with no GPU, no CUDA toolkit and no network, from a reading of
`ommx_linear.cu`, `../../linear/quantize.py`, `../../linear/w_format.py` and
`../../attention/paged_decode.py`. Everything below that is a *fact about existing code*
carries a `file:line`. Everything that is a *decision left to you* is collected in §9 and
is marked NOT KNOWN. Nothing here is a measurement.

Its purpose is to make the Python-side load-time transcode
(`integration/vllm/linear_method.py`, `OMMX_W_TRANSCODE=1`) unnecessary. That transcode
lets a bitmap bundle be served today, losslessly, but at the relidx7 *resident* cost:
a `gs=64, npv=4` bundle is **3.6250 bits/weight on disk** and streams **4.1250
bits/weight** after transcode. A native reader collapses those two numbers into one.

---

## 1. Scope: what the reader replaces, and what it does not

The OMMX weight bundle stores outlier POSITIONS in one of two encodings. They select the
same set — `tests/test_bitmap_outlier.py` pins that relidx7 / combinadic / bitmap decode
identically, and `tests/test_linear_method.py::test_bitmap_and_relidx7_differ_in_exactly_one_plane`
pins that on the weight path the two bundles differ in **exactly one plane**, `ommx_oindex`.

Everything else is already shared and needs **no** change:

| plane | shared? | why |
|---|---|---|
| `ommx_code` (INT2 payload) | yes | independent of the position encoding |
| `ommx_scale_exp`, `ommx_zp` | yes | per-group, independent |
| `ommx_ocode` (FP4 nibbles) | **yes — load-bearing** | `quantize.py` builds it from `gather(nib_g, pos_sorted)` *outside* the `outlier_repr` branch, so slot `s` is the s-th outlier **in ascending position order for both encodings**. This is exactly what `relidx7_slot_delta` assumes (`ocode_blk[s >> 1]` paired with `relidx7_slot_pos(oindex_blk, s)`), and exactly what the popcount rank produces. |
| `ommx_map_scale`, `ommx_map_center` | yes | absent under `outlier_map="none"`, which is a **separate** axis — see §8 |

So the reader is a swap of **one function**: "given slot `s` of block `bi` of row `n`,
what is the block-local position?" Nothing about the arithmetic downstream of that answer
changes.

---

## 2. The plane the reader consumes

### 2.1 relidx7 (what exists)

```
tensor  <module>.ommx_oindex : uint8 [N, G, IB7]        IB7 = ceil(npv * 7 / 8)
stride  oi_blk = oindex + (size_t(n) * n_blk + bi) * idx_blk_bytes
        idx_blk_bytes = (npv * RELIDX_BITS + 7) / 8      RELIDX_BITS = 7
layout  slot s occupies bits [7*s, 7*s+7) of the block's bit stream, LSB-first;
        the stream is zero-padded to a whole byte per block
decode  ommx_linear.cu:248  relidx7_slot_pos(blk, s)
          v = OR over b in [0,7) of  ((blk[(7s+b)>>3] >> ((7s+b)&7)) & 1) << b
          return v & 0x7F
```

`idx_blk_bytes` is recomputed **inside every kernel** from `npv` alone —
`ommx_linear.cu:512, 596, 654, 704, 752, 821, 862, 931, 1092`, and again in the
`sparse_correct_*` family around `:1883, :1934, :2010`. There is no host-side stride
argument. That is the first thing your change has to alter.

### 2.2 bitmap (what the reader consumes)

```
tensor  <module>.ommx_oindex : uint8 [N, G, IBB]        IBB = ceil(B / 8)
stride  bm_blk = oindex + (size_t(n) * n_blk + bi) * IBB
layout  block-local position p is set  <=>  (bm_blk[p >> 3] >> (p & 7)) & 1
        i.e. LSB-first, byte p>>3, bit p&7
pad     the top (8*IBB - B) bits of the last byte are ZERO
```

Producer: `linear/quantize.py::_pack_bits_lsb_first` over the boolean `omask`. Byte layout
is identical to `attention/codec.py::pack_bitmap_rows` (`p -> byte p>>3, bit p&7`) — the
KV and weight sides share one convention, and `tests/test_bitmap_outlier.py::
test_pack_bitmap_rows_matches_the_pure_python_row_codec` pins it.

### 2.3 The four differences from the relidx7 stream

1. **Width is driven by `B`, not by `npv`.** `IBB = ceil(B/8)`; `IB7 = ceil(npv*7/8)`.
   For the linear path `B == vector_length == group_size` — `linear_method.apply()` passes
   `gs` for both (`mod.sparse_correct(..., N, M, K, gs, npv, gs, KERNEL_FMT)`).
2. **The width is flat in `npv`.** That is the whole point: at `gs=64` it is 8 bytes per
   group whether `npv` is 1 or 32, where relidx7 grows 7 bits per outlier.
3. **The position ceiling disappears.** `relidx7_slot_pos` returns `v & 0x7F`, so relidx7
   cannot address a group wider than 128 (`w_format.RELIDX7_MAX_GROUP_SIZE`, and
   `Recipe.validate` refuses it). A bitmap has no such limit. A native reader is therefore
   not only cheaper at low `npv` — it **unlocks group sizes the format currently forbids**.
4. **Slot order is implied, not stored.** relidx7 stores the ascending positions
   explicitly; the bitmap stores membership and the ordinal has to be computed (§3).

### 2.4 Invariants the reader may assume, and the one it must not

MAY ASSUME (all guaranteed by the packer):
* exactly `npv` bits are set in every block's mask;
* the set positions are all `< B`; padding bits are zero;
* `ocode` slot `s` is the s-th set bit in ascending order.

MUST NOT ASSUME: that `IBB` is a multiple of 4. `w_format.Recipe.validate` only requires
`group_size % 4 == 0`, so `gs=20` gives `IBB=3` and a `uint32` reinterpret would read into
the **next block's bytes** and inflate the popcount. See §4.3 — this is the single most
likely way to ship a silently-wrong reader.

---

## 3. The decode

Two directions exist. **The linear kernels need the one the KV kernel does not use**, and
this is the most important thing to get right before writing any code.

| direction | question | who needs it |
|---|---|---|
| **select** | "block-local position of slot `s`" | **every linear kernel** — they loop `for s in [0,npv)` |
| **rank** | "is token `t` an outlier, and if so which slot" | the KV Triton decode — it loops over tokens |

`relidx7_slot_pos` is a *select*. So the replacement is a *select*:

```
r-th set bit of the mask, ascending  ==  position of slot r
```

and its cost is a masked prefix-popcount, exactly as law #15 prescribes.

### 3.1 The helper already exists (and is currently dead code)

`ommx_linear.cu:283`:

```cuda
// ── BITMAP membership: r-th set bit of a per-(row,block) bitmap via masked
// prefix-popcount (law #15: ONE wide reduce, no per-position __clzll loop).
__device__ __forceinline__ int bitmap_slot_pos(
    const uint32_t* __restrict__ bm_words, int n_words, int r) {
    int acc = 0, wd = 0; uint32_t word = 0u;
    for (; wd < n_words; ++wd) { word = bm_words[wd];
        const int pc = __popc(word); if (acc + pc > r) break; acc += pc; }
    uint32_t w = word;
    #pragma unroll 1
    for (int s = acc; s < r; ++s) w &= (w - 1u);   // drop lower set bits
    return (wd << 5) + (__ffs(w) - 1);
}
```

**No kernel calls it.** `grep -n 'bitmap_slot_pos' ommx_linear.cu` returns only its own
definition; `__popc` appears exactly once in the whole file, inside it. It is scaffolding
someone left for this work. Review it before trusting it — in particular the `w &= (w-1)`
loop assumes `r >= acc`, which holds only because the outer loop breaks at the first word
whose running popcount exceeds `r`, and it returns garbage rather than a sentinel for
`r >= popcount(whole mask)`. §4.4 says what to do about that.

### 3.2 The worked reference to cite

`attention/paged_decode.py::_bitmap_splice_from_bytes` (around line 813) is the only
implementation in this repo that reads a **stored** bitmap plane. Read it. What carries
over and what does not:

**CARRIES OVER**
* the byte→word assembly: word `_w` is bytes `4*_w .. 4*_w+3`, little-endian, and *"NO
  position arithmetic is needed"* because the bit layout is LSB-first (its comment says
  so, and it is the same `codec.pack_bitmap_rows` layout the weight packer emits);
* the multi-word structure `VL_WORDS = ceil(VL/32)`;
* the identity the FP4 gather rests on: *"the set bit's ascending ordinal IS `rank`, and
  pack.py writes `k_oval` in ascending-position order"*. Identical statement, identical
  guarantee, on the weight side with `ocode`.
* the CPU-side gate that proves the arithmetic: `tests/test_bitmap_outlier.py::
  test_kernel_bitmap_arithmetic_mirror_matches_the_packer` re-implements the kernel's
  byte→occupancy→membership→rank chain in Python over real packed frames. Clone it for
  the weight plane (§6, G1) — it is the only gate that catches a wrong formula without a
  device.

**DOES NOT CARRY OVER**
* **the direction.** `_bitmap_splice_from_bytes` computes membership+rank *per token*
  because the KV kernel iterates tokens. The linear kernels iterate slots and need select.
  Do not port its body; port its byte layout and its ordinal guarantee.
* Triton vs CUDA mechanics: `_popcount32` (a Triton helper) vs `__popc`; the "hold the
  words as a PYTHON LIST because Triton can't dynamically index a 2D register tensor"
  contortion is a Triton limitation and has no CUDA analogue.
* the plane's indexing: KV is `(head, block, token-frame)`, weight is `(row n, block bi)`.
* **its verification status.** Its own docstring says: *"UNVERIFIED ON HARDWARE: this
  branch has hardware-verified on an H200 (tests/test_decode_kernel_parity.py)."* Citing it as a design reference is correct;
  citing it as evidence the arithmetic runs is not.

---

## 4. The multi-word case, alignment, and padding

Let `B` = block width = `group_size` on the weight path, `IBB = ceil(B/8)`,
`n_words = ceil(IBB/4)`.

### 4.1 `B <= 32` — one word
`bitmap_slot_pos` runs its outer loop once; `acc` stays 0; `rank` collapses to a single
masked prefix-popcount. This is the same "fast lane" `_bitmap_splice_from_bytes` special-
cases at `VL_WORDS == 1`.

### 4.2 `B > 32` — several words
Words are scanned in ascending order, accumulating full popcounts, until the running total
exceeds `r`. Positions are `(wd << 5) + bit`. At `B=64` that is 2 words, at `B=128` 4
words. Because `npv <= 32` (§4.4) and the words are scanned in order, the loop is bounded
by `n_words`, not by `npv`.

### 4.3 Alignment — the trap

`bitmap_slot_pos` takes `const uint32_t*`. That is only sound when

* `IBB % 4 == 0`, i.e. **`B % 32 == 0`**, and
* the block base pointer is 4-byte aligned, which follows from the above plus torch's
  512-byte-aligned allocations.

For `B % 32 != 0` (e.g. `B=20 -> IBB=3`, `B=12 -> IBB=2`) a word load either misaligns or
reads the next block's first bytes into the high half of the last word — and those bytes
are that block's *low* positions, so the popcount is inflated and `bitmap_slot_pos`
returns a position from the wrong block. It will not fault. It will return plausible
numbers.

**Specified behaviour — pick one, do not leave it implicit:**

* **(a) Constrain.** Host-side `TORCH_CHECK(B % 32 == 0, ...)` in every entry point that
  takes a bitmap plane, and mirror it in `linear_method.require_kernel_readable` so the
  refusal happens at config time rather than at the first token. This matches the KV
  side's own constraint set (`attention/kv_window.py` and `kv_pool.py` already restrict
  group sizes to `{16, 32, 64, 128}`), and 16 is the only one of those it excludes.
* **(b) Byte-accumulate.** A `slot_pos` variant that walks `IBB` bytes with
  `__popc(byte)`, no word reinterpretation. Correct for every `B`, more instructions.

Recommendation: implement **(b)** as the general path and **(a)** as a fast path selected
by a `constexpr bool WORD_ALIGNED = (B % 32 == 0)`; `B` is not a compile-time constant in
these kernels today, so this is a runtime branch unless you also templatize `B`
(NOT KNOWN whether that is worth it — §9).

### 4.4 Padding and out-of-range slots

The packer zero-pads the top `8*IBB - B` bits, so padding cannot be mistaken for a set
bit. Two residual guards:

* `r >= npv` must never be asked. The kernels loop `s < npv`, so this holds by
  construction; assert it in debug builds rather than defending in the hot loop.
* `popcount(mask) == npv` is a packer guarantee the kernel cannot afford to verify per
  block. `linear/quantize.py::outlier_positions` DOES verify it (it raises when the count
  disagrees with the manifest), and the load-time transcode therefore already fails loudly
  on a corrupt plane. A native reader loses that check. Add it once, at load, host-side:
  a `bitmap_validate` kernel (or a torch `popcount` over the plane) run in
  `process_weights_after_loading` costs one pass over `N*G*IBB` bytes and turns a silent
  mis-decode into a startup failure. **Do not skip this** — it is the only thing standing
  between a truncated shard and a plausible model.
* `npv <= 32` is already a hard host limit: `ommx_linear.cu:2346`
  `TORCH_CHECK(npv > 0 && npv <= 32, "npv must be in (0,32] (MAX_NPV)")`, and
  `:1936` stages `int lk[32]; float ld[32]`. The bitmap does not change it.

---

## 5. Where it plugs in

### 5.1 The single funnel

Almost every consumer goes through **one** device function:

`ommx_linear.cu:261 relidx7_slot_delta(n, bi, B, s, K, VL, oindex_blk, ocode_blk, code_row, scale_row, ms_row, mc_row, &k, &delta)`

which is `relidx7_slot_pos` followed by arithmetic that is entirely position-encoding
agnostic:

```
local = relidx7_slot_pos(oindex_blk, s);   //  <-- the ONLY line that changes
k     = bi * B + local;  if (k >= K) return 0;
nib   = (ocode_blk[s >> 1] >> (4 * (s & 1))) & 0xF;
delta = (fp4_decode(nib) / ms_row[k/VL] + mc_row[k/VL] - int2_code_at(code_row, k)) * scale_row[k/VL];
```

Callers of `relidx7_slot_delta`: `sparse_correct_ws_kernel`, `build_odelta_kernel`,
`build_outlier_delta_dense_kernel`, `sparse_correct_gemv_relidx7_kernel`,
`sparse_correct_batched_relidx7_kernel`, `sparse_correct_batched_relidx7_parallel_kernel`.

Callers of `relidx7_slot_pos` **directly** (these need the same swap, individually):
`sparse_correct_ws_lut_kernel`, `build_kpos_kernel`, and the in-tile scan inside
`decode_B_tile` (the prefill fused overlay).

### 5.2 Recommended shape of the change

Add an index-format tag alongside the existing `StoreFmt` tags (`ommx_linear.cu:83`
`StoreI2F4` / `StoreI2`), because the file's whole idiom is already compile-time tags with
a traits struct:

```cuda
namespace ommx { namespace numeric {
struct IdxRelidx7 {};      // 7 bits/slot, ceil(npv*7/8) B per block  (today's default)
struct IdxBitmap  {};      // 1 bit/position, ceil(B/8) B per block   (paper claim B4)

template<class IdxFmt> struct IdxTraits;
template<> struct IdxTraits<IdxRelidx7> {
    static __device__ __forceinline__ int blk_bytes(int npv, int B) {
        return (npv * RELIDX_BITS + 7) / 8; }          // B unused
    static __device__ __forceinline__ int slot_pos(const uint8_t* blk, int s, int B) {
        return relidx7_slot_pos(blk, s); }             // B unused
};
template<> struct IdxTraits<IdxBitmap> {
    static __device__ __forceinline__ int blk_bytes(int npv, int B) {
        return (B + 7) / 8; }                          // npv unused — FLAT in npv
    static __device__ __forceinline__ int slot_pos(const uint8_t* blk, int s, int B) {
        return bitmap_slot_pos_bytes(blk, (B + 7) / 8, s); }
};
}}
```

Then:

1. `relidx7_slot_delta` becomes `template<class IdxFmt> slot_delta(...)` — rename it, and
   keep a `using` alias at the old name bound to `IdxRelidx7` so no existing call site
   changes shape.
2. Every kernel above gains a second template parameter, defaulted to `IdxRelidx7`.
   They are already `template<class StoreFmt>` (and some `template<class StoreFmt, int MAXM>`),
   so this is mechanical.
3. Every `const int idx_blk_bytes = (npv * nc::RELIDX_BITS + 7) / 8;` becomes
   `IdxTraits<IdxFmt>::blk_bytes(npv, B)`. There are nine such lines listed in §2.1 plus
   the ones inside the `sparse_correct_*` family — `grep -n idx_blk_bytes` finds all of
   them and none is anywhere else.

### 5.3 Host entry points

> **Read `SPARSE_CORRECT_ABI.md` before touching `sparse_correct`.** It is TWO functions
> behind one name — `if (M == 1) ... else ...` at `ommx_linear.cu:2392-2393` selects
> between an `A [1,K]` + fp32 `out` convention and an `At = A^T [K,M]` contiguous + bf16
> `C` convention. Adding `idx_fmt` to it means adding it to both arms, and any new gate
> for the bitmap reader must cover both: a five-case gate that exercised only the `M==1`
> arm is exactly how a caller that never passed `At` survived to production, aborting
> every prefill step of every outlier recipe while reporting PASS 5/5.

**Need a new argument** (add a trailing `const std::string& idx_fmt = "relidx7"`, so every
existing call site — including `linear_method.apply()` and
`test_ommx_linear_parity.py` — keeps compiling and behaving identically; the same
additivity rule that governs the Python side):

| entry point | line | why |
|---|---|---|
| `sparse_correct` | `:2340` | the one `OMMXWLinearMethod.apply()` calls |
| `sparse_correct_ws` | `:2175` | weight-stationary onto a cute base |
| `sparse_correct_ws_lut` | `:2204` | reads `oindex` for the position only |
| `build_kpos` | `:2230` | load-time position materialisation |
| `build_odelta` | `:2272` | load-time delta LUT |
| `build_outlier_delta_dense` | `:2295` | load-time dense `dW` |
| `prefill_wmma` | `:2398` | its optional `oindex_opt` overlay |

**Unchanged**: `decode_base` (`:2075` — no positions), `cute_base` (`:1743`),
`reorder_B` (`:1813`), `cast_f32_to_bf16`, `fire_stats`, and `sparse_correct_packed`
(`:2246` — it consumes `kpos`, already decoded; the encoding question was answered by
`build_kpos`).

**Do not confuse** `cute_base_fused`'s existing `bitmask` argument (`:1770`, checked as
`torch::kInt64 [N, n_blk]`) with this plane. That is a per-BLOCK occupancy word — "does
block `bi` of row `n` contain any outlier at all" — not a per-position mask. Different
shape, different dtype, different meaning.

### 5.4 A cheaper first cut (Option B)

If the full template sweep is too much for a first landing: implement `IdxBitmap` **only**
in `build_kpos` and `build_odelta`, and have `linear_method` route bitmap bundles through
the decode-free packed path (`build_kpos` + `build_odelta` once at load, then
`sparse_correct_packed` per token). Two kernels touched instead of twelve, and the steady-
state hot loop never decodes a position in either encoding anyway.

Cost, stated honestly: `kpos` is `int32 [N, n_blk*npv]` and `odelta` is `bf16
[N, n_blk*npv]`, i.e. **6 bytes per outlier resident** — at `gs=64, npv=4` that is
`6*4*8/64 = 3.0` extra bits/weight, far more than the 0.5 the transcode costs. Option B
makes the bitmap *readable*; only §5.2 makes it *cheap*. Do not publish an Option-B build
as "serving at 3.63 bits".

---

## 6. Acceptance gates

Ordered so that each one can only pass if the previous did.

### G1 — CPU, no device: the arithmetic mirror
**Claim:** the byte→word→popcount-select chain recovers exactly the packer's ascending
positions, for every group size, over REAL packed weight planes.
**Where:** `ommx_gpu_serve/tests/test_bitmap_outlier.py`, next to
`test_kernel_bitmap_arithmetic_mirror_matches_the_packer` (which does this for the KV
plane). Parametrize over `gs in {16, 32, 64, 128}` and `npv in {1, 4, 8, 32}`; source the
planes from `linear.quantize.quantize_ommx_weight(..., outlier_repr="bitmap")` and the
truth from `linear.quantize.outlier_positions(..., "bitmap")`.
**Bar:** exact set equality AND exact slot-order equality — `mirror_slot_pos(mask, s)`
must equal `sorted(positions)[s]` for every `s`, not just be a member of the set.
**Also add** the `B % 32 != 0` case (e.g. `gs=20`, `npv=2`) if you chose §4.3(b); if you
chose (a), add the *refusal* test instead.
```
python3 -m pytest ommx_gpu_serve/tests/test_bitmap_outlier.py -q
```

### G2 — GPU: bitmap read is BIT-EXACT to relidx7 read
**Claim:** for the same weight, `sparse_correct` with the bitmap plane and
`sparse_correct` with the relidx7 plane produce **identical fp32 output**.
Not `cos >= 0.999` — **`max_diff == 0.0` exactly**. The two paths compute the same
`(k, delta)` integers and floats from the same `code`/`scale`/`ocode`/map bytes; only the
route to `local` differs. Any nonzero difference is a bug, not a tolerance question.
**Where:** `ommx_gpu_serve/csrc/linear/test_ommx_linear_parity.py::main`, as a sixth
`_gate(...)` after `"decode M=1 base+outlier"`. Build both planes from one
`quantize_ommx_weight` call pair (same `W`, same seed, `outlier_repr` the only difference)
and assert the two `y` tensors are equal.
Add the M>1 twin as a seventh gate only after §9(1) is resolved.
```
python3 ommx_gpu_serve/csrc/linear/test_ommx_linear_parity.py
```

### G3 — GPU: firing evidence (law #5)
**Claim:** the bitmap reader actually executed; the run was not silently the relidx7 path.
`__popc` appears exactly ONCE in `ommx_linear.cu` today, inside the unused
`bitmap_slot_pos`, so a `POPC` in the SASS is a specific signature of this reader.
```
# BEFORE your change, to establish the baseline (expected: 0, the helper is dead code):
cuobjdump --dump-sass "$(python3 -c 'import torch,os;print(os.path.join(torch.utils.cpp_extension._get_build_directory("ommx_linear",False),"ommx_linear.so"))')" | grep -c POPC
# AFTER: must be > 0, and the in-kernel counter must move
python3 -c "import build_ommx_linear as b; m=b.build(); print(dict(m.fire_stats()))"
```
Add a dedicated `g_fire_bitmap` counter next to `g_fire_cute_bf16` / `g_fire_cute_fp8`
(`ommx_linear.cu:312`) and surface it in `fire_stats()`. A host-side counter freezes
under CUDA-graph replay (the file already says so for the cute counters) — for the
definitive proof use the SASS grep, not the counter.

### G4 — GPU: the paper's bundle serves natively, matching the CPU oracle
**Claim:** end to end, with no transcode.
**Where:** `ommx_gpu_serve/tests/test_linear_method.py`, a `@pytest.mark.gpu` sibling of
`test_gpu_end_to_end_linear_matches_the_cpu_dequant_oracle`, driven off the
`bundle_paper` fixture (bitmap + no map) instead of `bundle`.
**Bar:** `cos >= 0.999` against `w_format.load_weight` (law #4: cos first), plus
`ommx_w_fire_stats()["transcode"]["layers"] == 0` — the point is that the transcode did
**not** run.
```
python3 -m pytest ommx_gpu_serve/tests -q -m gpu
```

### G5 — CPU: the resident footprint actually collapses
**Claim:** the reason for all of this.
**Where:** `ommx_gpu_serve/tests/test_linear_method.py`.
Once `"bitmap"` joins `KERNEL_OUTLIER_REPRS`, `require_kernel_readable(paper_recipe)`
must return `None` (no plan, nothing to transcode) and the paper recipe's
`Recipe.bits_breakdown()["bits_per_weight"]` must be the resident figure — **3.6250**, the
same number as on disk. Assert the equality of the two, not just the value.

### G6 — regression: nothing else moved
```
python3 -m pytest ommx_gpu_serve/tests -q          # 515 passed, 3 skipped as of this spec
python3 ommx_gpu_serve/csrc/linear/test_ommx_linear_parity.py   # PARITY GATE: PASS 5/5
```
Both must be unchanged with `idx_fmt` left at its default. That is the CUDA-side
statement of the same additivity rule the Python side is held to.

---

## 7. Build environment — two requirements, both measured

From `MEASURED_FACTS.md` §10, hit on the cluster before the parity gate could run at all.
Neither is discoverable from the raw torch error.

1. **The `ninja` EXECUTABLE must be on `PATH`.** `torch.utils.cpp_extension.load()` shells
   out to it, so installing the wheel is necessary but not sufficient — `<env>/bin` is on
   `PATH` only while the env is ACTIVATED.
   Symptom: `RuntimeError: Ninja is required to load C++ extensions`.
   ```
   conda activate ommx     # or: export PATH="/path/to/env/bin:$PATH"
   ```
2. **`LIBRARY_PATH` must include the conda lib dir on cu13 envs.** torch emits
   `-L$CUDA_HOME/lib64` while those envs keep `libcudart.so` in `lib/`.
   Symptom: `/usr/bin/ld: cannot find -lcudart`.
   ```
   export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64"
   ```

Also from `build_ommx_linear.py`: it asserts `sm >= 80` and pins **one** gencode
(`sm_90a` on Hopper, else `sm_80`) — its own comment warns that without pinning, torch
appends the card's `sm_90` gencode and the runtime may pick the `sm_90` cubin, giving
`ILLEGAL INSTRUCTION` on wgmma/TMA. Build on the target GPU. A `.cu` change requires
purging the torch extension build directory (`ommx_linear` under
`torch.utils.cpp_extension._get_build_directory`) — a stale cached `.so` is how a "verified"
run measures the old kernel.

---

## 8. The other axis: `outlier_map="none"`

The paper's bundle is `(bitmap, none)` — **two** deviations from what the kernel reads, not
one. This spec covers the position encoding only. The map axis is already solved, and
solved cheaply, and does **not** need CUDA work:

`sparse_correct` takes `map_scale` / `map_center` as required `float32` tensors and
dereferences them unconditionally (`ommx_linear.cu:2360-2362`, and `ms_row[g]` / `mc_row[g]`
inside `relidx7_slot_delta`). `quantize.py` documents `outlier_map="none"` as the
**degenerate instance ms=1, mc=0** — it literally initialises those constants and encodes
against them. So handing the kernel constant ones/zeros planes reproduces `none` exactly,
with no ABI change. That is what `w_format.degenerate_map_planes` does today.

**But it costs 1.0 bit/weight at `gs=64`** (two f32 per group), which is precisely the gap
between the paper's 3.6250 and the 4.1250 the transcode lands on. A native reader that
fixes only §1-5 gets a bitmap bundle to `2 + 0.125 + 0.25 + 1.0 (bitmap) + 0.25 (fp4) +
1.0 (synthesised map) = 4.625` b/wt resident — **worse than today's transcode**.

To reach 3.6250 resident you need BOTH: the bitmap reader AND a `MAP_NONE` compile-time
specialisation of `slot_delta` that folds `ms=1, mc=0` and takes null map pointers:

```cuda
// delta at ms=1, mc=0 reduces to:  (fp4_decode(nib) - code) * scale_row[g]
```

That is a smaller change than the reader (one `if constexpr` in one function, plus making
the two host arguments `c10::optional`), but it is a *separate* one, with its own gate:
`sparse_correct` with null map planes must be bit-exact to `sparse_correct` with explicit
ones/zeros planes. Land it in the same PR or the headline number does not move.

### 8.1 …and even then, 3.6250 is a BUNDLE-PLANE figure, not HBM

Every bit budget in this document — 3.6250, 4.1250, 4.625 — counts bundle planes only,
because that is what `Recipe.bits_breakdown()` counts. It is the right basis for comparing
*encodings*. It is **not** an HBM total, and the gap is not small:
`linear_method.process_weights_after_loading` materialises fp32 `scale`/`zp` twins from
the int8 `scale_exp` and bf16 `zp` planes, and `apply()` hands the KERNEL the twins —
`decode_base(x2d, code, scale_f32, zp_f32, …)`, `sparse_correct(…, code, scale_f32, …)`.
At `gs=64` that is **+1.0 b/wt** resident (two f32 per group) on top of the plane figure,
while the 0.125 + 0.25 b/wt of `scale_exp` + bf16 `zp` counted *inside* it are held and
never read on the default path. Measured off the allocated tensors for the `[128,128]`
fixture: 8448 B of planes + 2048 B of twins = 10496 B = **5.1250 b/wt actually in HBM**,
of which **4.7500 b/wt** is what the kernel streams. Reported per layer as
`ommx_w_fire_stats()["transcode"]["twin_bytes"]` and pinned by
`tests/test_linear_method.py::test_the_resident_figure_is_a_delta_not_an_hbm_total`.

So a bitmap reader **plus** `MAP_NONE` lands at 3.6250 b/wt of *planes* and still ~4.625
b/wt of HBM. Closing that last gap is a third change and is not specified here: route the
kernel at the int8 `scale_exp` (the `OMMX_W_E8M0` argument list already exists and the
parity gate measured it BIT-EXACT to the fp32 path, `MEASURED_FACTS.md` §10) and read the
bf16 `zp` directly, then stop materialising the twins. Do not publish 3.6250 as an HBM
number until that lands.

---

## 9. NOT KNOWN — decisions for whoever has the device

These are genuinely open. Do not guess them from this document.

1. **`sparse_correct` at M>1 is currently unreachable from `linear_method.apply()`.**
   `apply()` calls `mod.sparse_correct(y, x2d, None, ...)` — `A_opt=x2d`, `At_opt=None` —
   but the host code takes `else { TORCH_CHECK(At_opt.has_value(), "M>1 correct needs At"); }`
   (`ommx_linear.cu:2393`). So the prefill outlier correction should abort loudly on the
   first `M >= OMMX_W_PREFILL_MIN_M` call. It is a LOUD failure, not a silent one, and it
   is **pre-existing and unrelated to the bitmap work** — but it means G2's M>1 twin cannot
   be run through `apply()` until someone with a device decides whether `apply()` should
   pass `x2d.t().contiguous()` as `At`, or route prefill through `prefill_wmma`'s fused
   `oindex_opt`/`odelta` overlay instead. Resolve it before claiming prefill parity.
2. **Whether `B` should become a template parameter.** §4.3's fast/slow path is a runtime
   branch unless it is. Templatizing over `B in {16,32,64,128}` removes the branch and lets
   `n_words` be a `constexpr`, at the cost of four instantiations of a dozen kernels. No
   measurement exists either way.
3. **Whether the popcount-select is actually faster than relidx7's bit gather at the
   linear path's access pattern.** `bitmap_slot_pos` is a data-dependent loop over words;
   `relidx7_slot_pos` is 7 unrolled bit extracts from a byte or two. The bitmap's win is in
   BYTES STREAMED (flat in `npv`), not obviously in ALU. At `gs=64` the bitmap is a flat 8 B/block
   while relidx7 is `ceil(7*npv/8)` B — so relidx7 is **smaller** for `npv <= 8`, they tie
   at `npv = 9`, and the bitmap only wins from `npv >= 10`. **At the paper's own operating
   point (`gs=64, npv=4`) the bitmap position plane is twice the size of relidx7's** (8 B
   vs 4 B). The paper's 3.6250 is lower than the shipped 4.1250 *because it also drops the
   range map* (§8), not because the bitmap is a smaller index there. Whoever optimises
   this should know which of the two terms they are actually chasing.
4. **Whether freeing the source plane is safe in a real vLLM engine.** The Python transcode
   replaces the `ommx_oindex` parameter's `.data` with an empty tensor after re-encoding.
   That is standard practice for vLLM quant methods, but it has never run inside a real
   engine here. A native reader makes the question moot, which is one more reason to do it.
5. **TP sharding of the bitmap plane.** `create_weights` marks the position plane
   `input_dim=1, packed_dim=1, packed_factor=group_size`. For relidx7 the per-block byte
   count depends on `npv`; for the bitmap it depends on `B`. Both narrow along dim 1 at
   block granularity, so the attribute is probably right for both — **probably** is the
   operative word; no multi-GPU run has ever loaded either.

---

## 10. Provenance

Facts in this document were read from, at the state of the tree it was written against:

* `ommx_gpu_serve/csrc/linear/ommx_linear.cu` — `relidx7_slot_pos` `:248`,
  `relidx7_slot_delta` `:261`, `bitmap_slot_pos` `:283`, `idx_blk_bytes` `:512` et seq,
  `sparse_correct` `:2340`, MAX_NPV `:2346`, `MAX_STAGE_DENSE` `:2366`, `check_fmt` `:2067`.
* `ommx_gpu_serve/linear/quantize.py` — the `outlier_repr` branch, `_pack_bits_lsb_first`,
  `outlier_positions`, and the `outlier_map="none"` degeneracy note.
* `ommx_gpu_serve/linear/w_format.py` — `Recipe.plane_layout`, `bits_breakdown`,
  `plan_transcode` and the transcode primitives.
* `ommx_gpu_serve/attention/paged_decode.py` — `_bitmap_splice_from_word` `:735`,
  `_bitmap_splice_from_bytes` `:813`.
* `ommx_gpu_serve/tests/test_bitmap_outlier.py` — the CPU bitmap gates and the
  `gpu`-marked KV equivalence gate.
* `MEASURED_FACTS.md` §10 — the two build-environment requirements.

Bit budgets quoted (3.6250 on disk, 4.1250 transcoded resident, 4.625 bitmap+synthesised
map, +3.0 for the Option-B `kpos`/`odelta` pair) are computed by
`Recipe.bits_breakdown()` / `TranscodePlan`, not measured on hardware and not taken from
the paper.
