// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// ommx_linear.cu  —  OMMX consolidated low-bit linear kernel (single file)
// ============================================================================
//
// ONE translation unit, ONE pybind module, ONE dequant numeric core.
// Consolidates the previously-scattered csrc/linear/{decode_gemv,dense_decode,
// prefill_wmma,cute_base_*,sparse_correct*}.cu (+ include/ommx/*.cuh) into a
// single source so the dequant math is defined EXACTLY ONCE and provably
// matches the Python fakequant reference (ommx-serve/fakequant + the on-disk
// codec, which is the declared single-source-of-truth the kernel mirrors).
//
// ----------------------------------------------------------------------------
// OMMX format (per-group decomposition = the LLM.int8() 2-path generalized):
//   BASE   : MXINT2 (2-bit) over ALL elements, AFFINE  w = code*scale + zp
//            code in {0,1,2,3}; scale = 2^e (E8M0 power-of-2 byte); zp = group
//            min (bf16).
//   OUTLIER: top-K (<=25%/group) MXFP4 (E2M1) overlay. The kernel decodes
//            delta = (decode_fp4_e2m1f(nibble)/ms + mc - code)*scale, generic
//            in (ms,mc); the shipped weight bundle bakes them in INDEX space
//            (see test). Decoupled correction; OVERWRITES the base lane.
//
// CORRECTNESS CONTRACT (fakequant consistency — see OMMX_LINEAR_CONSISTENCY.md):
//   * base dequant is ASYMMETRIC AFFINE (zp = group min), NOT symmetric — the
//     weight recipe (fakequant/quant_weight.py::weight_quantizer, mode=decode)
//     and the exporter (attention/pack.py) bake an affine (scale,zp) pair.
//     The symmetric (2c-3)/3 level form is retained as a compile/runtime option
//     for legacy/KV formats but is NOT the weight default.
//   * outlier map is parametrized by (ms,mc): delta = (fp4/ms + mc - code)*scale.
//     The shipped weight bundle (test_ommx_linear_parity.py::quantize_ommx_weight,
//     mirroring codec.py primitives) bakes (ms,mc) in INDEX space. pack.py's
//     weight-space KV map (map_scale,o_center) is a DIFFERENT parametrization —
//     convert via ms=scale*map_scale, mc=(o_center-zp)/scale before use; do NOT
//     feed pack.py's raw weight-space params to this kernel.
//   * decode-only: the kernel DECODES the baked bundle; it NEVER re-derives
//     codes/indices on GPU (torch.topk tie-break is device-dependent → the
//     exported bundle is canonical).
//   * rounding (law #10): round-half-to-even (rintf/__float2int_rn), clamp
//     AFTER round, mul-then-add (NOT fused __fmaf) for code*scale+zp to match
//     torch's non-fused op order, bf16 read via hardware __bfloat162float.
//
// ARCH MAP (law #1 / law #2.1):
//   decode  M<=16  : ALU GEMV — NO tensor core (dequant-bound 3.5-22x slower).
//   prefill M>=16  : tensor core. sm_80 = mma.m16n8k16 (PART C1);
//                    sm_90a = wgmma+TMA CUTLASS mixed-input (PART C2, guarded
//                    by OMMX_ENABLE_SM90_CUTLASS; needs CUTLASS 3.5+).
//   outlier        : decoupled sparse correction (PART D), all arches (ALU).
//
// GPU-VALIDATE markers (////[GPU]////) flag what the local (no-GPU) consolidation
// could NOT verify — these are the cluster wave's parity/perf gates
// (cos>=0.999, max_diff<1e-3, bypass=false, SASS mma/wgmma).
// ============================================================================

#include <torch/extension.h>
#include <pybind11/stl.h>            // std::map/vector -> Python (fire_stats return)
#include <map>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAException.h>   // C10_CUDA_CHECK (kernel-launch error check)
#include <cstdint>
#include <cstdlib>                   // std::getenv (OMMX_W_CORR_PARALLEL A/B gate)
#include <string>
#include <initializer_list>          // explicit dep for the split-K range-for

#ifndef OMMX_HOST_DEVICE
#define OMMX_HOST_DEVICE __host__ __device__
#endif

// ============================================================================
// PART A — EMBEDDED DEQUANT NUMERIC CORE  (single source of truth)
// Ported verbatim from include/ommx/ommx_numeric_core.cuh +
// include/ommx/ommx_outlier_relidx7.cuh. Every kernel below calls THESE — no
// path re-inlines the dequant math (the consolidation's central guarantee).
// ============================================================================
namespace ommx {
namespace numeric {

// ── StoreFmt tags — compile-time selectors for the dequant primitives. ──
// The ≤3-bit spec-compliant weight core: INT2 base + FP4 outliers (i2f4) and its
// dense variant (i2). The 4-bit i4f8 quality mode was dropped (2026-07-02).
struct StoreI2F4 {};  // INT2 base + sparse FP4 outliers (canonical weight i2f4)
struct StoreI2   {};  // INT2 only, no outliers (dense_int2)

template<class StoreFmt> struct StoreTraits;
template<> struct StoreTraits<StoreI2F4> {
    static constexpr int  base_bits = 2; static constexpr int elems_per_byte = 4;
    static constexpr bool has_outlier = true;   // FP4
};
template<> struct StoreTraits<StoreI2> {
    static constexpr int  base_bits = 2; static constexpr int elems_per_byte = 4;
    static constexpr bool has_outlier = false;
};

// ── FP4 E2M1 oracle LUT (host reference; device hot path uses fp4_decode). ──
// Order: index = code & 0xF, code = (sign<<3)|(exp<<1)|mant.
#if defined(__CUDACC__)
__device__ __constant__ float OMMX_FP4_E2M1_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
#endif

// ── E8M0 scale decode: int8 power-of-2 exponent -> fp32 (IEEE-754 bit inject).
// scale = 2^exp, exactly representable. Replaces __powf (1 inst). Mirrors
// codec.pow2_scale_exp inverse (pack.py stores round(log2(scale)).int8).
__device__ __forceinline__ float fast_pow2_i8(int8_t exp) {
    int e = static_cast<int>(exp);
    e = e < -126 ? -126 : (e > 127 ? 127 : e);
    return __uint_as_float(static_cast<uint32_t>(127 + e) << 23);
}

// ── INT unpack (bfe on sm_80+, scalar fallback). 4 INT2 / 2 INT4 per byte LE. ──
__device__ __forceinline__ uint32_t int2_unpack(uint32_t word, int pos) {
    uint32_t out;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("bfe.u32 %0, %1, %2, 2;\n" : "=r"(out) : "r"(word), "r"(pos));
#else
    out = (word >> pos) & 0x3u;
#endif
    return out;
}
__device__ __forceinline__ uint32_t int4_unpack(uint32_t word, int pos) {
    uint32_t out;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("bfe.u32 %0, %1, %2, 4;\n" : "=r"(out) : "r"(word), "r"(pos));
#else
    out = (word >> pos) & 0xFu;
#endif
    return out;
}

// Decode one base codeword from the packed stream at flat element index `elem`.
template<class StoreFmt>
__device__ __forceinline__ uint32_t decode_base_code(
    const uint8_t* __restrict__ code, int elem) {
    constexpr int epb = StoreTraits<StoreFmt>::elems_per_byte;
    const uint8_t byte = code[elem / epb];
    if (StoreTraits<StoreFmt>::base_bits == 2)
        return int2_unpack(static_cast<uint32_t>(byte), (elem & 3) * 2);
    else
        return int4_unpack(static_cast<uint32_t>(byte), (elem & 1) * 4);
}

// INT2 base code read by absolute K (used by the outlier -code residual term).
// Mirrors codec int2_code_at: code[n,k] = (row[k>>2] >> (2*(k&3))) & 0x3.
__device__ __forceinline__ int int2_code_at(const uint8_t* __restrict__ row, int k) {
    return (row[k >> 2] >> (2 * (k & 3))) & 0x3;
}

// ── FP4 E2M1 decode (arithmetic, no cmem — divergent __constant__ index
// serializes). Bit-exact vs OMMX_FP4_E2M1_LUT and codec.decode_fp4_e2m1f. ──
__device__ __forceinline__ float fp4_decode(uint8_t code4) {
    const uint32_t c = static_cast<uint32_t>(code4) & 0xFu;
    const int sign = static_cast<int>((c >> 3) & 1u);
    const int exp  = static_cast<int>((c >> 1) & 3u);
    const int mant = static_cast<int>( c       & 1u);
    float v;
    if (exp == 0) {
        v = static_cast<float>(mant) * 0.5f;                   // subnormal 0 / 0.5
    } else {
        const float pow2 = __uint_as_float(static_cast<uint32_t>(127 + (exp - 1)) << 23);
        v = pow2 * (1.0f + static_cast<float>(mant) * 0.5f);   // 2^(exp-1)*(1+0.5m)
    }
    return sign ? -v : v;
}

// FP outlier decode — FP4 e2m1 for the i2f4 weight core (i4f8's FP8 path dropped).
template<class StoreFmt>
__device__ __forceinline__ float outlier_decode(uint8_t code) {
    return fp4_decode(code);
}

// ── Symmetric INT level (LEGACY/KV option; NOT the weight default). ──
// INT2 {-1,-1/3,1/3,1}; INT4 (idx-7.5)/7.5. Arithmetic, bit-exact vs LUT.
template<class StoreFmt>
__device__ __forceinline__ float symmetric_level(uint32_t code) {
    if (StoreTraits<StoreFmt>::base_bits == 2)
        return (static_cast<float>(code & 0x3u) * 2.0f - 3.0f) * (1.0f / 3.0f);
    else
        return (static_cast<float>(code & 0xFu) - 7.5f) / 7.5f;
}

// ── Register-LUT (LEVER 2, INT2-only): fold the 4 scale-applied levels per
// group ONCE, then branchless pick. Bit-exact: levels computed by the IDENTICAL
// fp32 ops the scalar path uses, only HOISTED (runtime loop, NOT literals — a
// compile-time literal lets the host folder evaluate at a precision differing
// by 1 ulp from the device ALU, breaking LUT-vs-scalar bit-exactness, law #10).
struct Lut4 { float l0, l1, l2, l3; };
template<class StoreFmt>
__device__ __forceinline__ Lut4 build_int2_lut(float s, float z, bool symmetric) {
    float lv[4];
    #pragma unroll
    for (uint32_t c = 0; c < 4u; ++c)
        lv[c] = symmetric ? symmetric_level<StoreFmt>(c) * s
                          : static_cast<float>(c) * s + z;   // AFFINE = weight default
    return Lut4{lv[0], lv[1], lv[2], lv[3]};
}
__device__ __forceinline__ float int2_lut_pick(const Lut4& t, uint32_t c) {
    const float lo = (c & 1u) ? t.l1 : t.l0;
    const float hi = (c & 1u) ? t.l3 : t.l2;
    return (c & 2u) ? hi : lo;
}

// ── INT2 -> bf16x2 lop3 splice (the prefill tensor-core decode-to-bf16
// fast-path). magic {+128,+128}; lop3 0xFA = a|c; subtract 128 bias -> (float)c.
static constexpr uint32_t OMMX_INT2_BF16_MAGIC32 = 0x43004300u;
__device__ __forceinline__ __nv_bfloat162 ommx_int2_splice_bf16x2(uint32_t code_packed) {
    uint32_t spliced;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xFA;\n" : "=r"(spliced)
                 : "r"(OMMX_INT2_BF16_MAGIC32), "r"(0u), "r"(code_packed));
#else
    spliced = OMMX_INT2_BF16_MAGIC32 | code_packed;
#endif
    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&spliced);
    const __nv_bfloat162 bias = __floats2bfloat162_rn(128.0f, 128.0f);
    return __hsub2(v, bias);
}

// ── cp.async 16B (sm_80+) + commit/wait, for the prefill double-buffer. ──
__device__ __forceinline__ void cp_async_cg_16B(void* smem, const void* gmem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
#else
    *reinterpret_cast<int4*>(smem) = *reinterpret_cast<const int4*>(gmem);
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n");
#endif
}
template<int N> __device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// ── relidx7 compact-outlier helpers (on-disk sidecar inversion). The exporter
// (pack.py) packs slot positions ALREADY ASCENDING, so position = ONE direct
// 7-bit slice (no combinadic walk, no sort). delta is BYTE-IDENTICAL to the
// Python loader (vllm_ommx_linear._unrank_sidecar_csr) and to sparse_correct. ──
constexpr int RELIDX_BITS = 7;
__device__ __forceinline__ int relidx7_slot_pos(const uint8_t* __restrict__ blk, int s) {
    const int bit = s * RELIDX_BITS;
    int v = 0;
    #pragma unroll
    for (int b = 0; b < RELIDX_BITS; ++b) {
        const int gbit = bit + b;
        v |= (((blk[gbit >> 3] >> (gbit & 7)) & 1) << b);
    }
    return v & 0x7F;
}
// ── BITMAP membership: r-th set bit of a per-(row,block) bitmap via masked
// prefix-popcount (law #15: ONE wide reduce, no per-position __clzll loop).
// Returns the block-local position of the slot-r outlier (ascending), or -1 when the
// mask holds fewer than r+1 outliers.
//
// BYTES, NOT WORDS, and that is not a style choice. The plane's per-block stride is
// ceil(B/8) bytes -- 2 at B=16, 4 at B=32, 8 at B=64 -- so block bi starts at a 2-byte
// boundary for B=16 and a uint32_t* cast would be misaligned. Assemble little-endian
// from bytes instead, which also matches quantize.pack_bitmap_row's layout exactly
// (position p -> byte p>>3, bit p&7).
//
// The -1 return matters: a padded slot (a block whose real outlier count is below npv,
// which the packer emits at the tail) would otherwise index off the end of the mask and
// return a position that decodes a nibble belonging to nobody.
__device__ __forceinline__ int bitmap_slot_pos_bytes(
    const uint8_t* __restrict__ blk, int n_bytes, int r) {
    int acc = 0;
    for (int base = 0; base < n_bytes; base += 4) {
        const int nb = (n_bytes - base) < 4 ? (n_bytes - base) : 4;
        uint32_t w = 0u;
        for (int t = 0; t < nb; ++t)
            w |= static_cast<uint32_t>(blk[base + t]) << (8 * t);
        const int pc = __popc(w);
        if (acc + pc > r) {
            #pragma unroll 1
            for (int s = acc; s < r; ++s) w &= (w - 1u);   // drop lower set bits
            return (base << 3) + (__ffs(w) - 1);
        }
        acc += pc;
    }
    return -1;
}

// ── Outlier POSITION ENCODING, selected at run time ─────────────────────────────
// Two encodings of the same ascending position set. relidx7 spends 7 bits per outlier;
// the flat bitmask spends one bit per position and is therefore FLAT in npv -- cheaper
// above roughly 12.5-14% coverage, dearer below it. Everything downstream of the
// position (the nibble gather, the k bound, the delta arithmetic) is identical, which is
// why this is a two-function seam and not a second code path.
//
// A RUNTIME tag rather than the file's usual compile-time StoreFmt idiom: the branch is
// uniform across every thread of a launch (it comes from the bundle's recipe, not from
// data), so it costs a predicted branch, while a template would double the instantiation
// count of a dozen kernels. Cost unmeasured -- see the ABI doc.
enum IdxFmt : int { IDX_RELIDX7 = 0, IDX_BITMAP = 1 };

__device__ __host__ __forceinline__ int idx_block_bytes(int idx_fmt, int npv, int B) {
    return (idx_fmt == IDX_BITMAP) ? ((B + 7) / 8)
                                   : ((npv * RELIDX_BITS + 7) / 8);
}

// Block-local position of slot s, or -1 if the slot is padding (bitmap only; relidx7
// stores a fixed npv slots per block and signals padding through the k >= K bound).
__device__ __forceinline__ int idx_slot_pos(
    int idx_fmt, const uint8_t* __restrict__ blk, int s, int B) {
    return (idx_fmt == IDX_BITMAP) ? bitmap_slot_pos_bytes(blk, (B + 7) / 8, s)
                                   : relidx7_slot_pos(blk, s);
}

// ── Group index without a runtime integer division. `k / VL` with a RUNTIME divisor is a
// ~18-instruction signed-division sequence on every GPU arch, and the decode kernels
// evaluate it once per packed BYTE of the base walk (K/4 times per row) and once per
// outlier slot. Every shipped group size (16/32/64/128) is a power of two, so the index is
// one shift; a non-power-of-two VL (the format allows any multiple of 4) keeps the exact
// division through the sh < 0 lane, so the result is IDENTICAL for every VL.
__device__ __forceinline__ int group_shift(int VL) {
    return (VL > 0 && (VL & (VL - 1)) == 0) ? (31 - __clz(VL)) : -1;
}
__device__ __forceinline__ int group_of(int k, int VL, int sh) {
    return (sh >= 0) ? (k >> sh) : (k / VL);
}

// ── Sequential POSITION WALKER over one block's ascending outlier positions. The r-th
// next() returns EXACTLY idx_slot_pos(idx_fmt, blk, r, B) -- or -1 once the block's
// population is exhausted (bitmap padding) -- so a caller looping s = 0..npv-1 sees the
// same (slot, position) pairs in the same order as before: per-slot k, delta and the fp32
// FMA order are unchanged. What changes is the WORK per slot (law #15: no per-slot rescan):
//   relidx7 : a 32-bit shift register refilled one byte at a time -> ceil(7*npv/8) byte
//             loads per block instead of 7 per slot, and ~5 ALU per slot instead of the
//             7-iteration bit-gather loop (~28).
//   bitmap  : successive lowest-set-bit extraction on the current 32-bit word (__ffs + a
//             clear) -> O(1) per slot instead of re-assembling the words, re-popcounting
//             and re-dropping r lower bits (O(r)) on EVERY slot.
// Bytes, not words, for the same alignment reason as bitmap_slot_pos_bytes. The refill
// never reads past the block: relidx7 needs ceil(7*(s+1)/8) <= ceil(7*npv/8) bytes for
// slot s, bitmap stops at n_bytes.
struct SlotWalker {
    const uint8_t* blk; int n_bytes; int fmt;
    uint32_t buf; int have; int pos_byte; int wbase;
    __device__ __forceinline__ SlotWalker(int idx_fmt, const uint8_t* __restrict__ b, int npv, int B)
        : blk(b), n_bytes(idx_block_bytes(idx_fmt, npv, B)), fmt(idx_fmt),
          buf(0u), have(0), pos_byte(0), wbase(0) {
        if (fmt == IDX_BITMAP) load_word();
    }
    __device__ __forceinline__ void load_word() {      // bitmap: next little-endian word
        const int nb = (n_bytes - pos_byte) < 4 ? (n_bytes - pos_byte) : 4;
        uint32_t w = 0u;
        for (int t = 0; t < nb; ++t) w |= static_cast<uint32_t>(blk[pos_byte + t]) << (8 * t);
        buf = w; wbase = pos_byte << 3; pos_byte += nb;
    }
    __device__ __forceinline__ int next() {
        if (fmt == IDX_BITMAP) {
            while (buf == 0u) {
                if (pos_byte >= n_bytes) return -1;    // fewer than r+1 outliers in this block
                load_word();
            }
            const int p = wbase + (__ffs(buf) - 1);
            buf &= (buf - 1u);
            return p;
        }
        // relidx7: 7 bits per slot, LSB-first across bytes (== relidx7_slot_pos bit order).
        while (have < 7) { buf |= static_cast<uint32_t>(blk[pos_byte++]) << have; have += 8; }
        const int v = static_cast<int>(buf & 0x7Fu);
        buf >>= 7; have -= 7;
        return v;
    }
};

// (k, delta) for slot s of block bi given its block-local position `local` (from
// idx_slot_pos or SlotWalker::next; < 0 = padded slot). The single source of the delta
// arithmetic -- slot_delta_fmt below and every walker-based loop call THIS.
//   delta = (fp4_decode(nibble)/map_scale[g] + o_center[g] - code[n,k]) * scale[g]
// returns 1 (applied) iff delta!=0 && k<K (padded-tail / 0-delta = no-op).
__device__ __forceinline__ int slot_delta_at(
    int local, int bi, int B, int s, int K, int VL, int vl_sh,
    const uint8_t* __restrict__ ocode_blk,
    const uint8_t* __restrict__ code_row,   const float*   __restrict__ scale_row,
    const float*   __restrict__ ms_row,     const float*   __restrict__ mc_row,
    int* __restrict__ k_out, float* __restrict__ d_out) {
    if (local < 0) return 0;          // bitmap: slot beyond this block's outlier count
    const int k = bi * B + local;
    if (k >= K) return 0;
    const uint8_t byt = ocode_blk[s >> 1];
    const unsigned nib = (byt >> (4 * (s & 1))) & 0xF;
    const int g = group_of(k, VL, vl_sh);
    const float fp4 = fp4_decode(static_cast<uint8_t>(nib));
    const int code  = int2_code_at(code_row, k);
    const float delta = (fp4 / ms_row[g] + mc_row[g] - static_cast<float>(code)) * scale_row[g];
    if (delta == 0.0f) return 0;
    *k_out = k; *d_out = delta; return 1;
}

// Per-slot (k, delta) for relidx7/bitmap slot s of block bi, output row n (random-access
// form: decodes the position of slot s on its own; the sequential kernels use SlotWalker).
__device__ __forceinline__ int slot_delta_fmt(
    int idx_fmt,
    int n, int bi, int B, int s, int K, int VL,
    const uint8_t* __restrict__ oindex_blk, const uint8_t* __restrict__ ocode_blk,
    const uint8_t* __restrict__ code_row,   const float*   __restrict__ scale_row,
    const float*   __restrict__ ms_row,     const float*   __restrict__ mc_row,
    int* __restrict__ k_out, float* __restrict__ d_out) {
    (void)n;
    const int local = idx_slot_pos(idx_fmt, oindex_blk, s, B);
    return slot_delta_at(local, bi, B, s, K, VL, group_shift(VL), ocode_blk,
                         code_row, scale_row, ms_row, mc_row, k_out, d_out);
}

// Back-compat name, bound to the shipped encoding. Every call site that has not yet been
// given an idx_fmt argument keeps its exact previous behaviour through this, so adding
// the bitmap path could not change a relidx7 result by accident.
__device__ __forceinline__ int relidx7_slot_delta(
    int n, int bi, int B, int s, int K, int VL,
    const uint8_t* __restrict__ oindex_blk, const uint8_t* __restrict__ ocode_blk,
    const uint8_t* __restrict__ code_row,   const float*   __restrict__ scale_row,
    const float*   __restrict__ ms_row,     const float*   __restrict__ mc_row,
    int* __restrict__ k_out, float* __restrict__ d_out) {
    return slot_delta_fmt(IDX_RELIDX7, n, bi, B, s, K, VL, oindex_blk, ocode_blk,
                          code_row, scale_row, ms_row, mc_row, k_out, d_out);
}

}  // namespace numeric
}  // namespace ommx

// ============================================================================
// PART B — DECODE PATH  (M<=16, ALU GEMV, NO tensor core — law #1)
// Ported from the ICCAD decode_gemv.cuh (479-LoC superset): split-K lever,
// register-LUT batched path.
// StoreFmt-templated over the i2f4/i2 INT2 core (EPB=4, BB=2).
// ============================================================================
namespace ommx { namespace linear {
namespace nc = ommx::numeric;
using namespace ommx::numeric;   // StoreTraits/StoreI2F4/... template-ids (no ADL → need this)

// law #5 host firing counters for the sm_90a cute_base prefill GEMM act-lanes
// (bf16-act vs fp8-act e4m3). Declared here (BEFORE the cute90 block) so cute_base
// can ++ them and the reopened ommx::linear fire_stats() below can read them. Host-
// side: reflects eager launches; under CUDA-graph replay the counts freeze at
// capture time — the definitive fp8 firing proof is the SASS QGMMA/E4M3 evidence.
static unsigned long long g_fire_cute_bf16 = 0;
static unsigned long long g_fire_cute_fp8  = 0;

// ── Output store for the FUSED decode kernels (OutT = float | __nv_bfloat16).
//    fp32 = the two-launch contract (out[n] = base then += corr). bf16 = the serving
//    dtype stored DIRECTLY: __float2bfloat16 is round-to-nearest-even, the SAME rounding
//    torch's y.to(bfloat16) applies to the fp32 value, so the bf16 lane is bit-identical
//    to "fp32 out, then the cast kernel it removes". ──
__device__ __forceinline__ void store_out(float* __restrict__ out, size_t i, float v) {
    out[i] = v;
}
__device__ __forceinline__ void store_out(__nv_bfloat16* __restrict__ out, size_t i, float v) {
    out[i] = __float2bfloat16(v);
}

// ── B1. M==1 warp-per-channel base GEMV (UF=8 byte-walk, EPB accumulators,
//        shfl tree-reduce, num_k_splits==1 = graph-safe single store). ──
template<class StoreFmt>
__global__ void decode_gemv_kernel(
    const __nv_bfloat16* __restrict__ A,      // [1, K]
    const uint8_t*       __restrict__ code,   // [N, K/EPB]
    const float*         __restrict__ scale,  // [N, G] fp32 (used iff scale_exp==null)
    const int8_t*        __restrict__ scale_exp, // [N, G] E8M0 byte; if !=null scale=2^exp (HBM lever #4)
    const float*         __restrict__ zp,     // [N, G] or nullptr (affine)
    float*               __restrict__ out,    // [1, N]
    int N, int K, int vector_length, bool symmetric, int num_k_splits) {
    constexpr int EPB  = StoreTraits<StoreFmt>::elems_per_byte;
    constexpr int BB   = StoreTraits<StoreFmt>::base_bits;
    constexpr uint32_t MASK = (1u << BB) - 1u;
    constexpr int WARPS = 8;
    const int warp = (blockIdx.x * WARPS) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (warp >= N) return;                                 // whole-warp exit (law #9 safe)
    const int n = warp, G = K / vector_length;
    const int vl_sh = nc::group_shift(vector_length);      // pow2 VL -> shift, else exact div
    const uint8_t* code_row = code + static_cast<size_t>(n) * (K / EPB);
    const int K_bytes = K / EPB;
    // split-K slice [slice_lo, slice_hi) over code bytes.
    const int ksplit = blockIdx.y;
    const int per = (K_bytes + num_k_splits - 1) / num_k_splits;
    const int slice_lo = ksplit * per;
    const int slice_hi = min(slice_lo + per, K_bytes);

    float acc[EPB];
    #pragma unroll
    for (int j = 0; j < EPB; ++j) acc[j] = 0.f;
    constexpr int UF = 8;
    int byte = slice_lo + lane;
    for (; byte + 32 * (UF - 1) < slice_hi; byte += 32 * UF) {
        uint8_t bb[UF];
        #pragma unroll
        for (int u = 0; u < UF; ++u) bb[u] = code_row[byte + 32 * u];
        #pragma unroll
        for (int u = 0; u < UF; ++u) {
            const int k0 = (byte + 32 * u) * EPB, g = nc::group_of(k0, vector_length, vl_sh);
            const float s = scale_exp ? nc::fast_pow2_i8(scale_exp[n * G + g]) : scale[n * G + g];
            const float z = (zp != nullptr) ? zp[n * G + g] : 0.f;
            #pragma unroll
            for (int j = 0; j < EPB; ++j) {
                const uint32_t c = (static_cast<uint32_t>(bb[u]) >> (BB * j)) & MASK;
                const float w = symmetric ? nc::symmetric_level<StoreFmt>(c) * s
                                          : static_cast<float>(c) * s + z;  // mul-then-add
                acc[j] = fmaf(__bfloat162float(A[k0 + j]), w, acc[j]);
            }
        }
    }
    for (; byte < slice_hi; byte += 32) {                  // tail
        const uint8_t b = code_row[byte];
        const int k0 = byte * EPB, g = nc::group_of(k0, vector_length, vl_sh);
        const float s = scale_exp ? nc::fast_pow2_i8(scale_exp[n * G + g]) : scale[n * G + g];
        const float z = (zp != nullptr) ? zp[n * G + g] : 0.f;
        #pragma unroll
        for (int j = 0; j < EPB; ++j) {
            const uint32_t c = (static_cast<uint32_t>(b) >> (BB * j)) & MASK;
            const float w = symmetric ? nc::symmetric_level<StoreFmt>(c) * s
                                      : static_cast<float>(c) * s + z;
            acc[j] = fmaf(__bfloat162float(A[k0 + j]), w, acc[j]);
        }
    }
    float sum = 0.f;
    #pragma unroll
    for (int j = 0; j < EPB; ++j) sum += acc[j];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) { if (num_k_splits > 1) atomicAdd(&out[n], sum); else out[n] = sum; }
}

// ── B2. M=2..16 barrier-free batched GEMV: stream the weight column ONCE and
//        reuse each dequantized weight across all M rows (arithmetic intensity
//        = M FMAs/byte -> overlaps in-flight weight loads with real compute).
//        register-LUT (USE_LUT, INT2-only) hoists the 4 levels per group. ──
template<class StoreFmt, int M_MAX>
__global__ void decode_gemv_batched_kernel(
    const __nv_bfloat16* __restrict__ A,      // [M, K]
    const uint8_t*       __restrict__ code,   // [N, K/EPB]
    const float*         __restrict__ scale,  // [N, G] fp32 (used iff scale_exp==null)
    const int8_t*        __restrict__ scale_exp, // [N, G] E8M0 byte; if !=null scale=2^exp
    const float*         __restrict__ zp,     // [N, G] or nullptr
    float*               __restrict__ out,    // [M, N]
    int N, int M, int K, int vector_length, bool symmetric) {
    constexpr int EPB = StoreTraits<StoreFmt>::elems_per_byte;
    constexpr int BB  = StoreTraits<StoreFmt>::base_bits;
    constexpr uint32_t MASK = (1u << BB) - 1u;
    constexpr bool USE_LUT = (BB == 2);
    constexpr int WARPS = 8;
    const int warp = (blockIdx.x * WARPS) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (warp >= N) return;
    const int n = warp, G = K / vector_length;
    const int vl_sh = nc::group_shift(vector_length);
    const uint8_t* code_row = code + static_cast<size_t>(n) * (K / EPB);
    const int K_bytes = K / EPB;

    float acc[M_MAX];
    #pragma unroll
    for (int m = 0; m < M_MAX; ++m) acc[m] = 0.f;
    int prev_g = -1; nc::Lut4 lut{0,0,0,0}; float s_g = 0.f, z_g = 0.f;
    constexpr int UF = 4;
    int byte = lane;
    for (; byte < K_bytes; byte += 32 * UF) {
        uint8_t bb[UF];
        #pragma unroll
        for (int u = 0; u < UF; ++u) bb[u] = (byte + 32 * u < K_bytes) ? code_row[byte + 32 * u] : 0;
        #pragma unroll
        for (int u = 0; u < UF; ++u) {
            const int bidx = byte + 32 * u; if (bidx >= K_bytes) break;
            const int k0 = bidx * EPB, g = nc::group_of(k0, vector_length, vl_sh);
            if (g != prev_g) { s_g = scale_exp ? nc::fast_pow2_i8(scale_exp[n * G + g]) : scale[n * G + g]; z_g = (zp != nullptr) ? zp[n * G + g] : 0.f;
                if (USE_LUT) lut = nc::build_int2_lut<StoreFmt>(s_g, z_g, symmetric); prev_g = g; }
            #pragma unroll
            for (int j = 0; j < EPB; ++j) {
                const uint32_t c = (static_cast<uint32_t>(bb[u]) >> (BB * j)) & MASK;
                const float w = USE_LUT ? nc::int2_lut_pick(lut, c)
                              : (symmetric ? nc::symmetric_level<StoreFmt>(c) * s_g
                                           : static_cast<float>(c) * s_g + z_g);
                #pragma unroll
                for (int m = 0; m < M_MAX; ++m)
                    if (m < M) acc[m] = fmaf(__bfloat162float(A[(size_t)m * K + k0 + j]), w, acc[m]);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < M_MAX; ++m) {
        float sum = acc[m];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (lane == 0 && m < M) out[(size_t)m * N + n] = sum;
    }
}

// ── WEIGHT-STATIONARY sparse correction ONLY (adds onto an existing bf16 base C[M,N]).
//    Pairs with a tensor-core cute base for decode: the base runs on wgmma (fast, flat over
//    the decode batch), then this reads each sparse outlier ONCE and FMAs it into all M rows
//    (A[m,k] cache-hot) -> the outlier-decode cost amortizes across the batch instead of
//    sparse_correct's per-column Aᵀ re-read (which collapses at M>1). corr[MAXM] only -> low
//    register pressure, high occupancy. C += corr in bf16 (base then +corr order, law #12). ──
template<class StoreFmt, int MAXM>
__global__ void sparse_correct_ws_kernel(
    const __nv_bfloat16* __restrict__ A,        // [M, K]
    const uint8_t*       __restrict__ code, const float* __restrict__ scale,
    const uint8_t*       __restrict__ oindex, const uint8_t* __restrict__ ocode,
    const float*         __restrict__ map_scale, const float* __restrict__ map_center,
    __nv_bfloat16*       __restrict__ C,        // [M, N] base; += correction in-place
    int N, int K, int M, int vector_length, int npv, int B, int idx_fmt) {
    constexpr int EPB = StoreTraits<StoreFmt>::elems_per_byte;
    constexpr int WARPS = 8;
    const int warp = (blockIdx.x * WARPS) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (warp >= N) return;
    const int n = warp, G = K / vector_length;
    const uint8_t* code_row  = code + static_cast<size_t>(n) * (K / EPB);
    const float*   scale_row = scale + static_cast<size_t>(n) * G;
    const float*   ms_row = map_scale  + static_cast<size_t>(n) * G;
    const float*   mc_row = map_center + static_cast<size_t>(n) * G;
    const int n_blk = (K + B - 1) / B;
    const int idx_blk_bytes = nc::idx_block_bytes(idx_fmt, npv, B);
    const int nib_per_blk   = (npv + 1) / 2;
    float corr[MAXM];
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) corr[m] = 0.f;
    if (oindex != nullptr && npv > 0) {
        for (int bi = lane; bi < n_blk; bi += 32) {
            const uint8_t* oi_blk = oindex + (static_cast<size_t>(n) * n_blk + bi) * idx_blk_bytes;
            const uint8_t* oc_blk = ocode  + (static_cast<size_t>(n) * n_blk + bi) * nib_per_blk;
            #pragma unroll 2
            for (int sl = 0; sl < npv; ++sl) {
                int k; float d;
                if (nc::slot_delta_fmt(idx_fmt, n, bi, B, sl, K, vector_length,
                        oi_blk, oc_blk, code_row, scale_row, ms_row, mc_row, &k, &d))
                    #pragma unroll
                    for (int m = 0; m < MAXM; ++m)
                        if (m < M) corr[m] = fmaf(__bfloat162float(A[static_cast<size_t>(m) * K + k]), d, corr[m]);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) {
        if (m >= M) break;
        float cr = corr[m];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) cr += __shfl_down_sync(0xffffffffu, cr, off);
        if (lane == 0) {
            const size_t idx = static_cast<size_t>(m) * N + n;
            C[idx] = __float2bfloat16(__bfloat162float(C[idx]) + cr);
        }
    }
}

// ── LUT + WEIGHT-STATIONARY correction: like sparse_correct_ws_kernel but the per-outlier
//    delta is PRECOMPUTED (odelta LUT, build_odelta) so the hot loop skips int2_code_at +
//    fp4_decode + the division — only relidx7_slot_pos (bit-unpack) + an odelta read + M FMAs
//    remain. odelta slot-major [n, s*n_blk + bi] (matches build_odelta_kernel). ──
template<class StoreFmt, int MAXM>
__global__ void sparse_correct_ws_lut_kernel(
    const __nv_bfloat16* __restrict__ A,        // [M, K]
    const uint8_t*       __restrict__ oindex,   // relidx7 sidecar (positions)
    const __nv_bfloat16* __restrict__ odelta,   // [N, n_blk*npv] precomputed deltas
    __nv_bfloat16*       __restrict__ C,        // [M, N] base; += correction in-place
    int N, int K, int M, int npv, int B, int idx_fmt) {
    constexpr int WARPS = 8;
    const int warp = (blockIdx.x * WARPS) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (warp >= N) return;
    const int n = warp;
    const int n_blk = (K + B - 1) / B;
    const int idx_blk_bytes = nc::idx_block_bytes(idx_fmt, npv, B);
    const size_t od_row = static_cast<size_t>(n) * n_blk * npv;
    float corr[MAXM];
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) corr[m] = 0.f;
    if (oindex != nullptr && npv > 0) {
        for (int bi = lane; bi < n_blk; bi += 32) {
            const uint8_t* oi_blk = oindex + (static_cast<size_t>(n) * n_blk + bi) * idx_blk_bytes;
            #pragma unroll 2
            for (int sl = 0; sl < npv; ++sl) {
                const int local = nc::idx_slot_pos(idx_fmt, oi_blk, sl, B);
                if (local < 0) continue;          // padded slot (bitmap)
                const int k = bi * B + local;
                if (k >= K) continue;
                const float d = __bfloat162float(odelta[od_row + static_cast<size_t>(sl) * n_blk + bi]);
                if (d == 0.f) continue;
                #pragma unroll
                for (int m = 0; m < MAXM; ++m)
                    if (m < M) corr[m] = fmaf(__bfloat162float(A[static_cast<size_t>(m) * K + k]), d, corr[m]);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) {
        if (m >= M) break;
        float cr = corr[m];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) cr += __shfl_down_sync(0xffffffffu, cr, off);
        if (lane == 0) {
            const size_t idx = static_cast<size_t>(m) * N + n;
            C[idx] = __float2bfloat16(__bfloat162float(C[idx]) + cr);
        }
    }
}

// ── LOAD-TIME: precompute the absolute k POSITION per outlier slot (int32, slot-major
//    [n, s*n_blk+bi], -1 for padded/out-of-range). Pairs with odelta so the correction
//    becomes decode-FREE: read (kpos, odelta) + FMA — no relidx7_slot_pos bit-unpack. ──
template<class StoreFmt>
__global__ void build_kpos_kernel(
    const uint8_t* __restrict__ oindex, int* __restrict__ kpos,
    int N, int K, int npv, int B, int idx_fmt) {
    const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int n_blk = (K + B - 1) / B;
    const long total = static_cast<long>(N) * n_blk * npv;
    if (tid >= total) return;
    const int s  =  tid % npv;
    const int bi = (tid / npv) % n_blk;
    const int n  =  tid / (static_cast<long>(n_blk) * npv);
    const int idx_blk_bytes = nc::idx_block_bytes(idx_fmt, npv, B);
    const uint8_t* oi_blk = oindex + (static_cast<size_t>(n) * n_blk + bi) * idx_blk_bytes;
    const int local = nc::idx_slot_pos(idx_fmt, oi_blk, s, B);
    const int k = bi * B + local;                 // local < 0 -> the (k < K) test below
    kpos[static_cast<size_t>(n) * n_blk * npv + static_cast<size_t>(s) * n_blk + bi] =
        (local >= 0 && k < K) ? k : -1;           // -1 is already this table's "no slot"
}

// ── DECODE-FREE weight-stationary correction: reads the precomputed (kpos, odelta) pair
//    per slot and FMAs across the batch. No relidx7/fp4/int2 decode in the hot loop ->
//    memory-bound (coalesced kpos/odelta reads) instead of ALU-bound. C += corr in bf16. ──
template<class StoreFmt, int MAXM>
__global__ void sparse_correct_packed_kernel(
    const __nv_bfloat16* __restrict__ A,        // [M, K]
    const int*           __restrict__ kpos,     // [N, n_blk*npv]
    const __nv_bfloat16* __restrict__ odelta,   // [N, n_blk*npv]
    __nv_bfloat16*       __restrict__ C,        // [M, N] base; += correction
    int N, int K, int M, int npv, int B) {
    constexpr int WARPS = 8;
    const int warp = (blockIdx.x * WARPS) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (warp >= N) return;
    const int n = warp;
    const int n_blk = (K + B - 1) / B;
    const int cnt = n_blk * npv;
    const size_t row = static_cast<size_t>(n) * cnt;
    float corr[MAXM];
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) corr[m] = 0.f;
    for (int i = lane; i < cnt; i += 32) {
        const int k = kpos[row + i];
        if (k < 0) continue;
        const float d = __bfloat162float(odelta[row + i]);
        #pragma unroll
        for (int m = 0; m < MAXM; ++m)
            if (m < M) corr[m] = fmaf(__bfloat162float(A[static_cast<size_t>(m) * K + k]), d, corr[m]);
    }
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) {
        if (m >= M) break;
        float cr = corr[m];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) cr += __shfl_down_sync(0xffffffffu, cr, off);
        if (lane == 0) {
            const size_t idx = static_cast<size_t>(m) * N + n;
            C[idx] = __float2bfloat16(__bfloat162float(C[idx]) + cr);
        }
    }
}

// ── LOAD-TIME odelta LUT build: precompute the activation-INDEPENDENT per-outlier
//    delta = (fp4/ms + mc - code)*scale ONCE (dWt-style, but sparse: only the npv
//    outliers per block, ~4x smaller than dense dWt). Reuses relidx7_slot_delta so
//    the value is BIT-EXACT vs the fused kernel's in-line compute. odelta stored bf16
//    at [n, bi*npv+s]; 0 for not-applied (delta==0 or padded k>=K). One thread/outlier.
template <typename StoreFmt>
__global__ void build_odelta_kernel(
    const uint8_t* __restrict__ code, const float* __restrict__ scale,
    const uint8_t* __restrict__ oindex, const uint8_t* __restrict__ ocode,
    const float* __restrict__ map_scale, const float* __restrict__ map_center,
    __nv_bfloat16* __restrict__ odelta, int N, int K, int VL, int npv, int B, int idx_fmt) {
    constexpr int EPB = StoreTraits<StoreFmt>::elems_per_byte;
    const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int n_blk = (K + B - 1) / B;
    const long total = static_cast<long>(N) * n_blk * npv;
    if (tid >= total) return;
    const int s  =  tid % npv;
    const int bi = (tid / npv) % n_blk;
    const int n  =  tid / (static_cast<long>(n_blk) * npv);
    const int G = K / VL;
    const int idx_blk_bytes = nc::idx_block_bytes(idx_fmt, npv, B);
    const int nib_per_blk   = (npv + 1) / 2;
    const uint8_t* code_row = code + static_cast<size_t>(n) * (K / EPB);
    const float* scale_row = scale + static_cast<size_t>(n) * G;
    const float* ms_row = map_scale + static_cast<size_t>(n) * G;
    const float* mc_row = map_center + static_cast<size_t>(n) * G;
    const uint8_t* oi_blk = oindex + (static_cast<size_t>(n) * n_blk + bi) * idx_blk_bytes;
    const uint8_t* oc_blk = ocode  + (static_cast<size_t>(n) * n_blk + bi) * nib_per_blk;
    int k; float d; float delta = 0.f;
    if (nc::slot_delta_fmt(idx_fmt, n, bi, B, s, K, VL, oi_blk, oc_blk,
                               code_row, scale_row, ms_row, mc_row, &k, &d))
        delta = d;
    // SLOT-MAJOR layout [n, s*n_blk + bi]: the decode reads it with bi=lane (fixed s),
    // so consecutive lanes hit consecutive addresses -> COALESCED (vs bi-major stride-npv).
    odelta[static_cast<long>(n) * n_blk * npv + static_cast<long>(s) * n_blk + bi] =
        __float2bfloat16(delta);
}

// ── LOAD-TIME: build the DENSE outlier-delta weight dW[N,K] bf16 (mostly zero; nonzero only
//    at the ~25% top-K outlier positions). One-time per weight (M-INDEPENDENT). This turns the
//    PREFILL outlier correction into a tensor-core GEMM  C += x @ dW^T  (A is reused across the
//    N-tile) instead of sparse_correct's one-CTA-per-N-column scalar kernel, which re-reads At
//    per column and is DRAM-bound -> collapses at M>1 (measured 100-435x vs CUTLASS INT4). Uses
//    the SAME relidx7_slot_delta value as sparse_correct/build_odelta -> bit-consistent (cos=1).
//    Within a row n, each (bi,s) slot maps to a DISTINCT k (relidx7 gives distinct in-block
//    positions; blocks own disjoint K ranges) -> plain store, no atomicAdd. dW pre-zeroed. ──
template <typename StoreFmt>
__global__ void build_outlier_delta_dense_kernel(
    const uint8_t* __restrict__ code, const float* __restrict__ scale,
    const uint8_t* __restrict__ oindex, const uint8_t* __restrict__ ocode,
    const float* __restrict__ map_scale, const float* __restrict__ map_center,
    __nv_bfloat16* __restrict__ dW, int N, int K, int VL, int npv, int B, int idx_fmt) {
    constexpr int EPB = StoreTraits<StoreFmt>::elems_per_byte;
    const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int n_blk = (K + B - 1) / B;
    const long total = static_cast<long>(N) * n_blk * npv;
    if (tid >= total) return;
    const int s  =  tid % npv;
    const int bi = (tid / npv) % n_blk;
    const int n  =  tid / (static_cast<long>(n_blk) * npv);
    const int G = K / VL;
    const int idx_blk_bytes = nc::idx_block_bytes(idx_fmt, npv, B);
    const int nib_per_blk   = (npv + 1) / 2;
    const uint8_t* code_row = code + static_cast<size_t>(n) * (K / EPB);
    const float* scale_row = scale + static_cast<size_t>(n) * G;
    const float* ms_row = map_scale + static_cast<size_t>(n) * G;
    const float* mc_row = map_center + static_cast<size_t>(n) * G;
    const uint8_t* oi_blk = oindex + (static_cast<size_t>(n) * n_blk + bi) * idx_blk_bytes;
    const uint8_t* oc_blk = ocode  + (static_cast<size_t>(n) * n_blk + bi) * nib_per_blk;
    int k; float d;
    if (nc::slot_delta_fmt(idx_fmt, n, bi, B, s, K, VL, oi_blk, oc_blk,
                               code_row, scale_row, ms_row, mc_row, &k, &d)
        && k >= 0 && k < K)
        dW[static_cast<long>(n) * K + k] = __float2bfloat16(d);
}

// ── C1. sm_80 tensor-core prefill (mma.m16n8k16, base-only). REAL B-stationary.
// Ported from archive prefill_wmma.cuh (verified): 2-stage cp.async A ring +
// double-buffered B decode (INT2 lop3-splice bf16x2 affine code*scale+zp via
// ommx::numeric) + B-stationary MT_BLOCKS m-reuse + EXACT mma.m16n8k16 PTX, f32
// accumulate. Outliers DECOUPLED to PART D (sparse_correct) -> base-only here.
// ////[GPU]//// parity vs fakequant fp32 + SASS `mma.m16n8k16` MUST be confirmed on A100.
//
// stage_A_tile: cooperatively cp.async one 16x16 A tile (2x 16B per row = 8 bf16)
// into a ring slot; scalar tail when the K-tile or M-row is partial. Whole-CTA
// participates (coop over threadIdx.x); no per-lane early-return (law #9).
template<class StoreFmt>
__device__ __forceinline__ void stage_A_tile(
    const __nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ sAslot,
    int m0, int k0, int M, int K, int coop_id, int coop_n) {
    constexpr int MMA_M = 16, MMA_K = 16;
    const bool ktile_full = (k0 + MMA_K) <= K;
    #pragma unroll 8
    for (int r = coop_id; r < MMA_M; r += coop_n) {
        const int m = m0 + r;
        if (m < M && ktile_full) {
            const __nv_bfloat16* gsrc = A + static_cast<size_t>(m) * K + k0;
            nc::cp_async_cg_16B(&sAslot[r * MMA_K + 0], gsrc + 0);   // 8 bf16 = 16B
            nc::cp_async_cg_16B(&sAslot[r * MMA_K + 8], gsrc + 8);   // 8 bf16 = 16B
        } else {
            #pragma unroll
            for (int c = 0; c < MMA_K; ++c) {
                const int k = k0 + c;
                sAslot[r * MMA_K + c] = (m < M && k < K)
                    ? A[static_cast<size_t>(m) * K + k] : __float2bfloat16(0.0f);
            }
        }
    }
}

// decode_B_tile: dequant THIS warp's 8x16 (N x K) B tile into smem as bf16.
// base_bits==2 -> vectorized lop3 splice (2 codes -> bf16x2). AFFINE (weight
// default, symmetric=false): raw{(float)c0,c1} then ONE __hfma2 = code*scale+zp
// (law #10 bf16 mul-then-add). symmetric/INT4 branches preserved. BASE-ONLY:
// the FP4 outlier overlay (decoupled to PART D sparse_correct) is intentionally absent.
template<class StoreFmt>
__device__ __forceinline__ void decode_B_tile(
    __nv_bfloat16* __restrict__ sBw, int n_tile, int k0, int lane,
    const uint8_t* __restrict__ code, const float* __restrict__ scale,
    const float* __restrict__ zp, int N, int K, int G, int vector_length,
    bool symmetric, __nv_bfloat162 LV_SLOPE, __nv_bfloat162 LV_BIAS,
    // FUSED FP4-outlier overlay operands (nullptr/0 -> base-only, byte-identical). odelta =
    // build_odelta (precomputed delta, slot-major [n, s*n_blk+bi]) so the overlay only does
    // relidx7_slot_pos (bit-unpack) + a read — no fp4/int2/division recompute per tile. ──
    const uint8_t* __restrict__ oindex = nullptr, const __nv_bfloat16* __restrict__ odelta = nullptr,
    int npv = 0, int B_blk = 0, int idx_fmt = nc::IDX_RELIDX7) {
    constexpr int MMA_N = 8, MMA_K = 16;
    constexpr int EPB = StoreTraits<StoreFmt>::elems_per_byte;
    constexpr bool VEC = (StoreTraits<StoreFmt>::base_bits == 2);   // INT2 lop3 splice
    if (VEC && symmetric) {
        // SYMMETRIC INT2 (legacy/KV option): splice two adjacent K codes -> raw
        // bf16x2{(float)c0,c1}, hfma2 to level domain {-1,-1/3,1/3,1} (slope 2/3,
        // bias -1), then *scale.
        for (int u = lane; u < MMA_N * (MMA_K / 2); u += 32) {
            const int nn = u / (MMA_K / 2), cp = (u % (MMA_K / 2)) * 2;
            const int n = n_tile + nn, k = k0 + cp;
            __nv_bfloat162 out2 = __floats2bfloat162_rn(0.f, 0.f);
            if (n < N && k + 1 < K) {
                const uint8_t* crow = code + static_cast<size_t>(n) * (K / EPB);
                const uint32_t c0 = nc::decode_base_code<StoreFmt>(crow, k);
                const uint32_t c1 = nc::decode_base_code<StoreFmt>(crow, k + 1);
                const uint32_t pk = (c0 & 0x3u) | ((c1 & 0x3u) << 16);
                const __nv_bfloat162 raw = nc::ommx_int2_splice_bf16x2(pk);
                const __nv_bfloat162 lvl = __hfma2(raw, LV_SLOPE, LV_BIAS);
                const float s = scale[n * G + k / vector_length];
                out2 = __hmul2(lvl, __floats2bfloat162_rn(s, s));
            } else if (n < N && k < K) {
                const uint8_t* crow = code + static_cast<size_t>(n) * (K / EPB);
                const uint32_t c0 = nc::decode_base_code<StoreFmt>(crow, k);
                const float s = scale[n * G + k / vector_length];
                out2 = __floats2bfloat162_rn(nc::symmetric_level<StoreFmt>(c0) * s, 0.f);
            }
            *reinterpret_cast<__nv_bfloat162*>(&sBw[nn * MMA_K + cp]) = out2;
        }
    } else if (VEC && !symmetric && zp != nullptr) {
        // AFFINE INT2 (WEIGHT DEFAULT, law #13): splice two adjacent K codes ->
        // raw bf16x2{(float)c0,c1}, then ONE __hfma2 = code*scale + zp (law #10:
        // bf16 mul-then-add for the VALUE, NOT fmaf). Group-boundary-safe per-elem
        // scale/zp (g0 != g1 at a group edge).
        for (int u = lane; u < MMA_N * (MMA_K / 2); u += 32) {
            const int nn = u / (MMA_K / 2), cp = (u % (MMA_K / 2)) * 2;
            const int n = n_tile + nn, k = k0 + cp;
            __nv_bfloat162 out2 = __floats2bfloat162_rn(0.f, 0.f);
            if (n < N && k + 1 < K) {
                const uint8_t* crow = code + static_cast<size_t>(n) * (K / EPB);
                const uint32_t c0 = nc::decode_base_code<StoreFmt>(crow, k);
                const uint32_t c1 = nc::decode_base_code<StoreFmt>(crow, k + 1);
                const uint32_t pk = (c0 & 0x3u) | ((c1 & 0x3u) << 16);
                const __nv_bfloat162 raw = nc::ommx_int2_splice_bf16x2(pk);
                const int g0 = k / vector_length, g1 = (k + 1) / vector_length;
                const __nv_bfloat162 scale_x2 =
                    __floats2bfloat162_rn(scale[n * G + g0], scale[n * G + g1]);
                const __nv_bfloat162 zp_x2 =
                    __floats2bfloat162_rn(zp[n * G + g0], zp[n * G + g1]);
                out2 = __hfma2(raw, scale_x2, zp_x2);   // code*scale + zp (1 HFMA2)
            } else if (n < N && k < K) {
                const uint8_t* crow = code + static_cast<size_t>(n) * (K / EPB);
                const uint32_t c0 = nc::decode_base_code<StoreFmt>(crow, k);
                const float s = scale[n * G + k / vector_length];
                const float z = zp[n * G + k / vector_length];
                out2 = __floats2bfloat162_rn(static_cast<float>(c0) * s + z, 0.f);
            }
            *reinterpret_cast<__nv_bfloat162*>(&sBw[nn * MMA_K + cp]) = out2;
        }
    } else {
        // SCALAR fallback (INT4 base / affine with zp==null): one element/lane-iter.
        for (int i = lane; i < MMA_N * MMA_K; i += 32) {
            const int nn = i / MMA_K, c = i % MMA_K, n = n_tile + nn, k = k0 + c;
            float wv = 0.f;
            if (n < N && k < K) {
                const float s = scale[n * G + k / vector_length];
                const uint8_t* crow = code + static_cast<size_t>(n) * (K / EPB);
                const uint32_t bc = nc::decode_base_code<StoreFmt>(crow, k);
                wv = symmetric ? nc::symmetric_level<StoreFmt>(bc) * s
                               : static_cast<float>(bc) * s
                                 + (zp ? zp[n * G + k / vector_length] : 0.f);
            }
            sBw[i] = __float2bfloat16(wv);
        }
    }
    // ── FUSED FP4-outlier overlay (memory-preserving in-mainloop fusion oracle): after the
    //    base tile is in smem, add the sparse FP4 delta at THIS 8x16 tile's outlier positions
    //    (in-tile relidx7 scan; no dense weight). delta == relidx7_slot_delta == build_odelta,
    //    so base+overlay == W_full -> the mma consumes the corrected weight (cos=1). Distinct
    //    (nn, k) per outlier -> distinct sBw slot -> race-free. B_blk>=MMA_K & MMA_K|k0 => the
    //    16-wide K-tile lies in one relidx7 block bi=k0/B_blk. ──
    if (oindex != nullptr && npv > 0) {
        __syncwarp();                                   // base writes visible before RMW
        const int n_blk = (K + B_blk - 1) / B_blk;
        const int idx_blk_bytes = nc::idx_block_bytes(idx_fmt, npv, B_blk);
        const int bi = k0 / B_blk;
        for (int u = lane; u < MMA_N * npv; u += 32) {
            const int nn = u / npv, s = u % npv, n = n_tile + nn;
            if (n >= N) continue;
            const uint8_t* oi_blk = oindex + (static_cast<size_t>(n) * n_blk + bi) * idx_blk_bytes;
            const int loc = nc::idx_slot_pos(idx_fmt, oi_blk, s, B_blk);
            const int k = bi * B_blk + loc;
            if (loc >= 0 && k >= k0 && k < k0 + MMA_K) {
                const float dd = __bfloat162float(
                    odelta[static_cast<size_t>(n) * n_blk * npv + static_cast<size_t>(s) * n_blk + bi]);
                if (dd != 0.f)
                    sBw[nn * MMA_K + (k - k0)] = __hadd(sBw[nn * MMA_K + (k - k0)], __float2bfloat16(dd));
            }
        }
    }
    __syncwarp();
}

// EXACT sm_80 tensor-core op: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32.
__device__ __forceinline__ void mma_m16n8k16_bf16(
    const uint32_t (&a)[4], const uint32_t (&b)[2], float (&d)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void)a; (void)b; (void)d;
#endif
}

template<class StoreFmt, int WARPS_PER_CTA = 4, int MT_BLOCKS = 8, int SPLITK = 1>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32)
prefill_wmma_kernel(
    const __nv_bfloat16* __restrict__ A,      // [M, K] row-major
    const uint8_t*       __restrict__ code,   // [N, K/EPB]
    const float*         __restrict__ scale,  // [N, G]
    const float*         __restrict__ zp,     // [N, G] or nullptr
    float*               __restrict__ Cf,     // [M, N] fp32 atomicAdd scratch (SPLITK>1)
    __nv_bfloat16*       __restrict__ C,       // [M, N] bf16 direct store (SPLITK==1)
    int N, int M, int K, int vector_length, bool symmetric, int /*splitk_rt unused*/,
    // FUSED FP4-outlier overlay operands (nullptr/0 -> base-only prefill, byte-identical). ──
    const uint8_t* __restrict__ oindex = nullptr, const __nv_bfloat16* __restrict__ odelta = nullptr,
    int npv = 0, int B_blk = 0, int idx_fmt = nc::IDX_RELIDX7) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    const int warp_id = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n_base = blockIdx.x * (WARPS_PER_CTA * MMA_N);
    const int n_tile = n_base + warp_id * MMA_N;             // PER-WARP 8-col N tile
    const int G = K / vector_length, coop_n = WARPS_PER_CTA * 32;
    const int m_base = blockIdx.y * (MT_BLOCKS * MMA_M);     // CTA-uniform

    __shared__ __align__(16) __nv_bfloat16 sA[2][MT_BLOCKS][MMA_M * MMA_K];
    __shared__ __nv_bfloat16 sB[2][WARPS_PER_CTA][MMA_N * MMA_K];
    const int n_ktiles = (K + MMA_K - 1) / MMA_K;
    const int kt_per = (SPLITK == 1) ? n_ktiles : (n_ktiles / SPLITK);
    const int kt_lo  = (SPLITK == 1) ? 0 : (blockIdx.z * kt_per);
    const int kt_hi  = (SPLITK == 1) ? n_ktiles : (kt_lo + kt_per);

    // symmetric-only level constants ({-1,-1/3,1/3,1} = raw*2/3 - 1).
    const __nv_bfloat162 LV_SLOPE = __floats2bfloat162_rn(2.0f / 3.0f, 2.0f / 3.0f);
    const __nv_bfloat162 LV_BIAS  = __floats2bfloat162_rn(-1.0f, -1.0f);

    float d[MT_BLOCKS][4];
    #pragma unroll
    for (int mb = 0; mb < MT_BLOCKS; ++mb) { d[mb][0]=d[mb][1]=d[mb][2]=d[mb][3]=0.f; }

    // law #9: m_base is CTA-uniform -> a whole-CTA early return is divergence-free.
    // n_tile (= n_base + warp_id*MMA_N) is PER-WARP; we deliberately do NOT
    // early-return on it (the scaffold's bug). All warps stay alive, reach every
    // __syncthreads, and the (n<N) guards in decode_B_tile / the store drop OOB
    // columns when N % (WARPS_PER_CTA*MMA_N) != 0.
    if (m_base >= M) return;

    // prologue: prime stage-0 A (all m-blocks) + decode stage-0 B.
    if (kt_hi > kt_lo) {
        #pragma unroll
        for (int mb = 0; mb < MT_BLOCKS; ++mb)
            stage_A_tile<StoreFmt>(A, sA[0][mb], m_base + mb * MMA_M, kt_lo * MMA_K,
                                   M, K, threadIdx.x, coop_n);
        nc::cp_async_commit();
        decode_B_tile<StoreFmt>(sB[0][warp_id], n_tile, kt_lo * MMA_K, lane, code,
                                scale, zp, N, K, G, vector_length, symmetric,
                                LV_SLOPE, LV_BIAS, oindex, odelta, npv, B_blk, idx_fmt);
    }

    for (int kt = kt_lo; kt < kt_hi; ++kt) {
        const int k0 = kt * MMA_K, slot = (kt - kt_lo) & 1;
        // prefetch next A tile (cp.async) + wait on the OLDEST in-flight group.
        if (kt + 1 < kt_hi) {
            #pragma unroll
            for (int mb = 0; mb < MT_BLOCKS; ++mb)
                stage_A_tile<StoreFmt>(A, sA[(kt + 1 - kt_lo) & 1][mb],
                                       m_base + mb * MMA_M, k0 + MMA_K, M, K,
                                       threadIdx.x, coop_n);
            nc::cp_async_commit();
            nc::cp_async_wait_group<1>();   // keep 1 group (the next) in flight
        } else {
            nc::cp_async_wait_group<0>();    // drain
        }
        __syncthreads();
        // double-buffer B: decode B[kt+1] while we MMA B[kt].
        if (kt + 1 < kt_hi)
            decode_B_tile<StoreFmt>(sB[(kt + 1 - kt_lo) & 1][warp_id], n_tile,
                                    k0 + MMA_K, lane, code, scale, zp, N, K, G,
                                    vector_length, symmetric, LV_SLOPE, LV_BIAS,
                                    oindex, odelta, npv, B_blk, idx_fmt);

        // ── B fragment (col-major B operand of m16n8k16). bcol=lane/4 (8 N-cols),
        // bk=(lane%4)*2; t=0 -> k in [bk,bk+1], t=1 -> k in [bk+8,bk+9].
        const __nv_bfloat16* sBw = sB[slot][warp_id];
        uint32_t b[2];
        { const int bcol = lane / 4, bk = (lane % 4) * 2;
          #pragma unroll
          for (int t = 0; t < 2; ++t) { const int kk = bk + (t ? 8 : 0);
            const __nv_bfloat16 lo = sBw[bcol * MMA_K + kk + 0];
            const __nv_bfloat16 hi = sBw[bcol * MMA_K + kk + 1];
            b[t] = (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&hi)) << 16)
                 |  static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&lo)); } }

        // ── A fragment (row-major). grow=lane/4, gcol=(lane%4)*2; t bit0 -> row+8,
        // t bit1 -> k+8. (B-stationary: one b[] feeds all MT_BLOCKS a[] mmas.)
        #pragma unroll
        for (int mb = 0; mb < MT_BLOCKS; ++mb) {
            const __nv_bfloat16* sAcur = sA[slot][mb];
            uint32_t a[4]; const int grow = lane / 4, gcol = (lane % 4) * 2;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int rr = grow + (t & 1 ? 8 : 0), kk = gcol + (t >> 1 ? 8 : 0);
                const __nv_bfloat16 lo = sAcur[rr * MMA_K + kk + 0];
                const __nv_bfloat16 hi = sAcur[rr * MMA_K + kk + 1];
                a[t] = (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&hi)) << 16)
                     |  static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&lo));
            }
            mma_m16n8k16_bf16(a, b, d[mb]);
        }
        __syncthreads();
    }

    // ── epilogue: write the f32 C fragment. crow=lane/4, ccol=(lane%4)*2;
    // d[mb][t]: t bit1 -> row+8, t bit0 -> col+1.
    const int crow = lane / 4, ccol = (lane % 4) * 2;
    #pragma unroll
    for (int mb = 0; mb < MT_BLOCKS; ++mb) {
        const int m0 = m_base + mb * MMA_M;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int r = crow + ((t & 2) ? 8 : 0), c = ccol + (t & 1);
            const int m = m0 + r, n = n_tile + c;
            if (m < M && n < N) {                        // (n<N) drops OOB warp cols
                if (SPLITK == 1) C[static_cast<size_t>(m) * N + n] = __float2bfloat16(d[mb][t]);
                else             atomicAdd(&Cf[static_cast<size_t>(m) * N + n], d[mb][t]);
            }
        }
    }
#else
    (void)A;(void)code;(void)scale;(void)zp;(void)Cf;(void)C;
    (void)N;(void)M;(void)K;(void)vector_length;(void)symmetric;
#endif  // __CUDA_ARCH__ >= 800
}

// split-K epilogue: cast fp32 atomicAdd scratch -> bf16 (format-independent, once).
__global__ void cast_f32_to_bf16_kernel(const float* __restrict__ src,
                                        __nv_bfloat16* __restrict__ dst, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dst[i] = __float2bfloat16(src[i]);
}

}}  // namespace ommx::linear  (close PART B/C1 before the guarded CUTLASS block)

// ── C2. sm_90a CUTLASS mixed-input prefill (wgmma + TMA), UNIFIED template.
// Consolidates cute_base_i2f4(_fp8) + cute_base_i4f8 into ONE collective
// parameterized on <TileN, QuantType, MmaType>:
//   QuantType in {uint2b_t (i2f4), uint4b_t (i4f8)}  -> AlignmentB, base bits
//   MmaType   in {bfloat16_t (bf16 act), float_e4m3_t (fp8 act)} -> TileShapeK
// ALWAYS the default Data-Parallel persistent scheduler (NO StreamK — its
// device-side reduction spin-wait hangs under CUDA-graph FULL capture, the #2
// graph-safety law). TileN auto by M-band (16 decode-tile / 64 / 128 prefill).
// Persistent caller workspace pre-allocated OUTSIDE capture; k-major scale
// contract (element[n,g] @ g*N+n) validated, NO in-op transpose. Optional
// shuffled-B (compute_memory_reordering_atom + reorder at warmup) is the
// Marlin/Machete memory-bound lever. The narrow weight is the register A
// operand via the swap+transpose Y^T = W^T X^T.
//
// Guarded: requires CUTLASS 3.5+ (mixed_dtype builder) and sm_90a. The
// collective body below is the consolidation TARGET; its exact instantiation is
// lifted from the verified cute_base_i2f4_fp8.cu and ////[GPU]//// MUST be
// compile + parity validated against the pinned CUTLASS tree (cicc OOM budget:
// only the 6 cooperative kernels TileN{16,64,128}x{plain,shuffled}; DROP the
// pingpong/clustered extra schedules from this TU).
#if defined(OMMX_ENABLE_SM90_CUTLASS)
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/util/mixed_dtype_utils.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cute/tensor.hpp>
#if defined(OMMX_CUTE_FUSED)
// OMMX fork of the mixed-input RS collective: fuses the i2f4 FP4-outlier correction into the
// wgmma mainloop (memory-preserving). Opt-in (cicc budget) — the fused GemmFused variant below is
// only instantiated when this is defined. Base-only path stays byte-identical to the stock base.
#include "ommx_sm90_mixed_input_fused.hpp"
#endif

namespace ommx { namespace linear { namespace cute90 {
using namespace cute;

// ----------------------------------------------------------------------------
// CORRECTED unified DP collective (REPLACE the existing OmmxCuteGemmDp struct).
// QuantType in {uint2b_t (i2f4), uint4b_t (i4f8)}; MmaType in {bfloat16_t (bf16
// act -> TileShapeK 64), float_e4m3_t (fp8 act -> TileShapeK 128)}. ALWAYS the
// default persistent Data-Parallel scheduler (NO StreamKScheduler 4th template
// arg) -> CUDA-graph-capturable (law #2 graph-safety; StreamK's device-side
// reduction spin-wait hangs under FULL capture).
template<int TileN, class QuantType, class MmaType>
struct OmmxCuteGemmDp {
    using ElementScale = cutlass::bfloat16_t;
    using ElementZero  = cutlass::bfloat16_t;
    using ElementC     = cutlass::bfloat16_t;
    using ElementD     = cutlass::bfloat16_t;
    using ElementAcc   = float;
    using ArchTag      = cutlass::arch::Sm90;
    using OpClass      = cutlass::arch::OpClassTensorOp;

    // bf16 act -> TileShapeK 64; fp8 act -> 128. (128 * 8 / bits(MmaType).)
    static constexpr int TileShapeK = 128 * 8 / cutlass::sizeof_bits<MmaType>::value;
    using TileShape    = cute::Shape<cute::_128, cute::Int<TileN>, cute::Int<TileShapeK>>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

    // Narrow weight is K-major (TMA 128-bit: bits(QuantType)*alignment % 128 == 0).
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    // Problem is SWAPPED to {N, M, K}: the narrow weight is the A-operand (row in
    // the swapped frame), the activation is B. CUTLASS reads B (= narrow weight)
    // K-major (ColumnMajor of the logical [N,K]) and A (= activation) row-major.
    using LayoutA_act  = cutlass::layout::RowMajor;     // activation logical [M,K]
    using LayoutB_w    = cutlass::layout::ColumnMajor;  // k-major narrow weight logical [N,K]
    using LayoutC_log  = cutlass::layout::RowMajor;     // output logical [M,N]
    using LayoutA_T    = typename cutlass::layout::LayoutTranspose<LayoutA_act>::type; // ColumnMajor
    using LayoutB_T    = typename cutlass::layout::LayoutTranspose<LayoutB_w>::type;   // RowMajor
    using LayoutC_T    = typename cutlass::layout::LayoutTranspose<LayoutC_log>::type; // ColumnMajor

    // ---- SHUFFLED-B (offline memory-reordering) layout ----
    // 16-row Hopper shuffle atom placing the values read by the SAME wgmma thread
    // contiguously. compute_memory_reordering_atom<MmaType> -> tile_to_shape over
    // the col-major [N,K] logical stride. The host reorder_B writes the permuted
    // codes once at warmup; the shuffled mainloop passes the Layout (not a stride)
    // in the B-operand slot. Dequant math UNCHANGED -> parity holds.
    using StrideB_logical   = cutlass::detail::TagToStrideB_t<LayoutB_w>;
    using LayoutAtomQuant   = decltype(cutlass::compute_memory_reordering_atom<MmaType>());
    using LayoutB_Reordered = decltype(cute::tile_to_shape(
        LayoutAtomQuant{}, cute::Layout<cute::Shape<int,int,int>, StrideB_logical>{}));

    // Epilogue: C/D layouts are LayoutC_T (ColumnMajor) — the swapped [N,M] frame.
    // ////[GPU]//// (the previous consolidation used LayoutA_T here; same type as
    // LayoutC_T only by coincidence (both ColumnMajor) for the C/D operands, but
    // sourced from the wrong logical tensor — corrected to LayoutC_T for clarity).
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OpClass, TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, ElementAcc,
        ElementC, LayoutC_T, AlignmentC,
        ElementD, LayoutC_T, AlignmentD,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

    // Mainloop: PLAIN col-major narrow weight (LayoutB_T after transpose-swap).
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        cute::tuple<QuantType, ElementScale, ElementZero>, LayoutB_T, AlignmentB,
        MmaType, LayoutA_T, AlignmentA,
        ElementAcc, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

    // NO 4th (scheduler) template arg -> default persistent Data-Parallel.
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // SHUFFLED twin mainloop (B = LayoutB_Reordered). Same epilogue/scheduler/
    // StageCount, so it shares the plain path's workspace sizing + graph-safety.
    using CollectiveMainloopShuf = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        cute::tuple<QuantType, ElementScale, ElementZero>, LayoutB_Reordered, AlignmentB,
        MmaType, LayoutA_T, AlignmentA,
        ElementAcc, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using GemmKernelShuf = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>, CollectiveMainloopShuf, CollectiveEpilogue>;
    using GemmShuf = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelShuf>;

#if defined(OMMX_CUTE_FUSED)
    // FUSED shuffled mainloop (RF-direct: no smem_D -> auto stages = base occupancy). SAME derived
    // types as CollectiveMainloopShuf, rebound onto the Ommx fused RS dispatch tag. The stock
    // cooperative kernel selects it via is_base_of<...Cooperative, DispatchPolicy::Schedule>.
    using CollectiveMainloopFused = cutlass::gemm::collective::CollectiveMma<
        cutlass::gemm::OmmxMainloopSm90MixedInputFused<
            CollectiveMainloopShuf::DispatchPolicy::Stages,
            typename CollectiveMainloopShuf::DispatchPolicy::ClusterShape,
            typename CollectiveMainloopShuf::DispatchPolicy::Schedule>,
        typename CollectiveMainloopShuf::TileShape,
        cute::tuple<QuantType, ElementScale, ElementZero>,
        typename CollectiveMainloopShuf::StrideA,
        MmaType,
        typename CollectiveMainloopShuf::StrideB,
        typename CollectiveMainloopShuf::TiledMma,
        typename CollectiveMainloopShuf::GmemTiledCopyA,
        typename CollectiveMainloopShuf::SmemLayoutAtomA,
        typename CollectiveMainloopShuf::SmemCopyAtomA,
        typename CollectiveMainloopShuf::TransformA,
        typename CollectiveMainloopShuf::GmemTiledCopyB,
        typename CollectiveMainloopShuf::SmemLayoutAtomB,
        typename CollectiveMainloopShuf::SmemCopyAtomB,
        typename CollectiveMainloopShuf::TransformB>;
    using GemmKernelFused = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>, CollectiveMainloopFused, CollectiveEpilogue>;
    using GemmFused = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelFused>;
#endif
};

// ----------------------------------------------------------------------------
// Persistent EAGER-fallback workspace (grows lazily per size/device). The GRAPH
// path passes a caller-preallocated workspace; this is only used when ws_ptr==null.
static at::Tensor& ommx_cute_workspace_for(int64_t bytes, at::Device dev) {
    static at::Tensor ws;
    if (!ws.defined() || ws.numel() < bytes || ws.device() != dev) {
        ws = torch::empty({bytes > 0 ? bytes : 1},
                          torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    }
    return ws;
}

// k-major group-scale validator: element [n,g] at memory g*N+n (N-contiguous).
// VALIDATE ONLY — NO in-op .transpose().contiguous() (cudaMalloc, illegal under
// CUDA-graph capture). Caller produces it via base_to_machete(..., kmajor=True).
static void ommx_check_kmajor(const torch::Tensor& s, int Ni, int64_t G, const char* nm) {
    TORCH_CHECK(s.dim() == 2 && s.size(0) == Ni && s.size(1) == G,
                "ommx_cute: ", nm, " must be [N,G]=[", Ni, ",", G, "], got [",
                s.size(0), ",", s.size(1), "]");
    TORCH_CHECK(s.transpose(0, 1).is_contiguous(),
                "ommx_cute: ", nm, " must be group-scale (k-major) layout: element "
                "[n,g] at memory g*N+n. Produce it with base_to_machete(kmajor=True).");
}

// ----------------------------------------------------------------------------
// workspace_size probe for ONE TileN bucket: pure host Gemm::get_workspace_size on
// probe-args with NULL operands (reads only problem shape / scheduler config). The
// swapped problem is {N, M, K, L}; strides: A=[M,K], B=[N,K], C/D=[N,M], S=[N,G].
template<int TileN, class QuantType, class MmaType>
static int64_t ommx_cute_workspace_size_for_tile(int M, int N, int K, int group_size) {
    using G_t      = OmmxCuteGemmDp<TileN, QuantType, MmaType>;
    using Gemm     = typename G_t::Gemm;
    using Mainloop = typename G_t::CollectiveMainloop;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using StrideS = typename Mainloop::StrideScale;
    const int G = K / group_size, L = 1;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, L));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(N, M, L));
    auto stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(N, G, L));
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm, {N, M, K, L},
        {nullptr, stride_B, nullptr, stride_A, nullptr, stride_S, group_size, nullptr},
        {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}};
    return static_cast<int64_t>(Gemm::get_workspace_size(arguments));
}

// ----------------------------------------------------------------------------
// run() — PLAIN col-major B. {N,M,K,L} swap; k-major affine (scale*c+zero); DP
// scheduler. ws_ptr non-null = caller persistent workspace (graph path, NO host
// alloc); null = eager static-grow fallback. base-only (beta=0, C unread); a
// fused-correction/residual variant can pass beta=1 + a C-source ptr (see notes).
template<int TileN, class QuantType, class MmaType>
static void ommx_cute_run_dp(torch::Tensor& out, torch::Tensor& A, torch::Tensor& B_codes,
                             torch::Tensor& scales, torch::Tensor& zeros,
                             int M, int N, int K, int group_size,
                             void* ws_ptr, int64_t ws_bytes) {
    using G_t      = OmmxCuteGemmDp<TileN, QuantType, MmaType>;
    using Gemm     = typename G_t::Gemm;
    using Mainloop = typename G_t::CollectiveMainloop;
    using QT       = QuantType;
    using MT       = MmaType;
    using ElementScale = typename G_t::ElementScale;
    using ElementZero  = typename G_t::ElementZero;
    using ElementC     = typename G_t::ElementC;
    using ElementD     = typename G_t::ElementD;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using StrideS = typename Mainloop::StrideScale;
    const int G = K / group_size, L = 1;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, L));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(N, M, L));
    auto stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(N, G, L));
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm, {N, M, K, L},
        {reinterpret_cast<const QT*>(B_codes.data_ptr<uint8_t>()), stride_B,
         reinterpret_cast<const MT*>(A.data_ptr()), stride_A,
         reinterpret_cast<const ElementScale*>(scales.data_ptr()), stride_S,
         group_size, reinterpret_cast<const ElementZero*>(zeros.data_ptr())},
        {{1.0f, 0.0f},
         reinterpret_cast<const ElementC*>(out.data_ptr()), stride_C,
         reinterpret_cast<ElementD*>(out.data_ptr()), stride_D}};
    Gemm gemm;
    auto stream = c10::cuda::getCurrentCUDAStream();
    cutlass::Status can = gemm.can_implement(arguments);
    TORCH_CHECK(can == cutlass::Status::kSuccess,
                "ommx_cute_run_dp: can_implement failed (M=", M, " N=", N, " K=", K,
                " TileN=", TileN, " group_size=", group_size,
                ") status=", cutlassGetStatusString(can));
    size_t ws = Gemm::get_workspace_size(arguments);
    void* ws_data;
    if (ws_ptr != nullptr) {
        TORCH_CHECK(ws_bytes >= static_cast<int64_t>(ws),
                    "ommx_cute_run_dp: provided workspace too small (", ws_bytes, " < ", ws,
                    " bytes for M=", M, " N=", N, " K=", K, " TileN=", TileN,
                    "); size it with workspace_size() BEFORE capture");
        ws_data = ws_ptr;
    } else {
        ws_data = ommx_cute_workspace_for(static_cast<int64_t>(ws), A.device()).data_ptr();
    }
    TORCH_CHECK(gemm.initialize(arguments, ws_data) == cutlass::Status::kSuccess,
                "ommx_cute_run_dp: initialize failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "ommx_cute_run_dp: run failed");
    C10_CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// run() — SHUFFLED B. The B-operand slot carries the full layout_B_reordered
// Layout (NOT a plain stride) — CUTLASS's mixed-input collective accepts a Layout
// for the swizzled operand. A/C/D/scale strides SHARED with the plain DP kernel
// (only B's layout differs) — sourcing them from the plain Gemm avoids the
// reordered-B kernel deducing a divergent stride that trips make_cute_packed_stride.
template<int TileN, class QuantType, class MmaType>
static void ommx_cute_run_shuffled(torch::Tensor& out, torch::Tensor& A, torch::Tensor& B_reordered,
                                   torch::Tensor& scales, torch::Tensor& zeros,
                                   int M, int N, int K, int group_size,
                                   void* ws_ptr, int64_t ws_bytes) {
    using G_t      = OmmxCuteGemmDp<TileN, QuantType, MmaType>;
    using GemmShuf = typename G_t::GemmShuf;
    using QT       = QuantType;
    using MT       = MmaType;
    using ElementScale = typename G_t::ElementScale;
    using ElementZero  = typename G_t::ElementZero;
    using ElementC     = typename G_t::ElementC;
    using ElementD     = typename G_t::ElementD;
    // Strides taken from the PLAIN kernel (shared A/C/D/scale config).
    using StrideA = typename G_t::Gemm::GemmKernel::StrideA;
    using StrideC = typename G_t::Gemm::GemmKernel::StrideC;
    using StrideD = typename G_t::Gemm::GemmKernel::StrideD;
    using StrideS = typename G_t::CollectiveMainloop::StrideScale;
    const int G = K / group_size, L = 1;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, L));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(N, M, L));
    auto stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(N, G, L));
    // tile the 16-row shuffle atom over the (N,K,L) SHAPE (tile_to_shape takes a
    // Shape, not a strided layout). MUST match the warmup reorder_tensor layout.
    auto layout_B_reordered =
        cute::tile_to_shape(typename G_t::LayoutAtomQuant{}, cute::make_shape(N, K, L));
    typename GemmShuf::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm, {N, M, K, L},
        {reinterpret_cast<const QT*>(B_reordered.data_ptr<uint8_t>()), layout_B_reordered,
         reinterpret_cast<const MT*>(A.data_ptr()), stride_A,
         reinterpret_cast<const ElementScale*>(scales.data_ptr()), stride_S,
         group_size, reinterpret_cast<const ElementZero*>(zeros.data_ptr())},
        {{1.0f, 0.0f},
         reinterpret_cast<const ElementC*>(out.data_ptr()), stride_C,
         reinterpret_cast<ElementD*>(out.data_ptr()), stride_D}};
    GemmShuf gemm;
    auto stream = c10::cuda::getCurrentCUDAStream();
    cutlass::Status can = gemm.can_implement(arguments);
    // group_size MUST be a multiple of TileShapeK (bf16:64 / fp8:128) — the k-major
    // group-scale offset + the reordered-B tile must land on a tile boundary.
    TORCH_CHECK(can == cutlass::Status::kSuccess,
                "ommx_cute_run_shuffled: can_implement failed (M=", M, " N=", N, " K=", K,
                " TileN=", TileN, " group_size=", group_size,
                ") status=", cutlassGetStatusString(can));
    size_t ws = GemmShuf::get_workspace_size(arguments);
    void* ws_data;
    if (ws_ptr != nullptr) {
        TORCH_CHECK(ws_bytes >= static_cast<int64_t>(ws),
                    "ommx_cute_run_shuffled: provided workspace too small (", ws_bytes, " < ", ws, ")");
        ws_data = ws_ptr;
    } else {
        ws_data = ommx_cute_workspace_for(static_cast<int64_t>(ws), A.device()).data_ptr();
    }
    TORCH_CHECK(gemm.initialize(arguments, ws_data) == cutlass::Status::kSuccess,
                "ommx_cute_run_shuffled: initialize failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "ommx_cute_run_shuffled: run failed");
    C10_CUDA_CHECK(cudaGetLastError());
}

#if defined(OMMX_CUTE_FUSED)
// ----------------------------------------------------------------------------
// run() — FUSED shuffled B (GemmFused). STEP A: the Arguments are IDENTICAL to the shuffled base
// (no sidecar operands yet) so this MUST be byte-identical to ommx_cute_run_shuffled -> cos==base.
// Later steps add the sparse FP4-outlier sidecar (oindex/odelta) to the mainloop Arguments.
template<int TileN, class QuantType, class MmaType>
static void ommx_cute_run_fused(torch::Tensor& out, torch::Tensor& A, torch::Tensor& B_reordered,
                                torch::Tensor& scales, torch::Tensor& zeros,
                                int M, int N, int K, int group_size, float fused_eps,
                                const uint64_t* bitmask_ptr, const cutlass::bfloat16_t* odelta_ptr,
                                int npv, int b_blk, int fused_ablate,
                                void* ws_ptr, int64_t ws_bytes) {
    using G_t       = OmmxCuteGemmDp<TileN, QuantType, MmaType>;
    using GemmFused = typename G_t::GemmFused;
    using QT        = QuantType;
    using MT        = MmaType;
    using ElementScale = typename G_t::ElementScale;
    using ElementZero  = typename G_t::ElementZero;
    using ElementC     = typename G_t::ElementC;
    using ElementD     = typename G_t::ElementD;
    using StrideA = typename G_t::Gemm::GemmKernel::StrideA;
    using StrideC = typename G_t::Gemm::GemmKernel::StrideC;
    using StrideD = typename G_t::Gemm::GemmKernel::StrideD;
    using StrideS = typename G_t::CollectiveMainloop::StrideScale;
    const int G = K / group_size, L = 1;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, L));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(N, M, L));
    auto stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(N, G, L));
    auto layout_B_reordered =
        cute::tile_to_shape(typename G_t::LayoutAtomQuant{}, cute::make_shape(N, K, L));
    typename GemmFused::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm, {N, M, K, L},
        {reinterpret_cast<const QT*>(B_reordered.data_ptr<uint8_t>()), layout_B_reordered,
         reinterpret_cast<const MT*>(A.data_ptr()), stride_A,
         reinterpret_cast<const ElementScale*>(scales.data_ptr()), stride_S,
         group_size, reinterpret_cast<const ElementZero*>(zeros.data_ptr()), 4u, fused_eps,
         bitmask_ptr, odelta_ptr, npv, b_blk, (b_blk > 0 ? (K + b_blk - 1) / b_blk : 0), N, fused_ablate},
        {{1.0f, 0.0f},
         reinterpret_cast<const ElementC*>(out.data_ptr()), stride_C,
         reinterpret_cast<ElementD*>(out.data_ptr()), stride_D}};
    GemmFused gemm;
    auto stream = c10::cuda::getCurrentCUDAStream();
    cutlass::Status can = gemm.can_implement(arguments);
    TORCH_CHECK(can == cutlass::Status::kSuccess,
                "ommx_cute_run_fused: can_implement failed (M=", M, " N=", N, " K=", K,
                " TileN=", TileN, " group_size=", group_size, ") status=", cutlassGetStatusString(can));
    size_t ws = GemmFused::get_workspace_size(arguments);
    void* ws_data;
    if (ws_ptr != nullptr) {
        TORCH_CHECK(ws_bytes >= static_cast<int64_t>(ws),
                    "ommx_cute_run_fused: provided workspace too small (", ws_bytes, " < ", ws, ")");
        ws_data = ws_ptr;
    } else {
        ws_data = ommx_cute_workspace_for(static_cast<int64_t>(ws), A.device()).data_ptr();
    }
    TORCH_CHECK(gemm.initialize(arguments, ws_data) == cutlass::Status::kSuccess,
                "ommx_cute_run_fused: initialize failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "ommx_cute_run_fused: run failed");
    C10_CUDA_CHECK(cudaGetLastError());
}
#endif  // OMMX_CUTE_FUSED

// ----------------------------------------------------------------------------
// reorder_B() — WARMUP offline weight permutation (Marlin/Machete lever). Pure
// index permutation (cutlass::reorder_tensor over 16-row tiles) -> dequant math
// byte-identical, parity preserved. Call ONCE outside any CUDA-graph capture.
template<class QuantType, class MmaType>
static void ommx_cute_reorder_B(torch::Tensor& B_reordered, torch::Tensor& B_codes, int N, int K) {
    using G_t = OmmxCuteGemmDp<128, QuantType, MmaType>;  // atom is TileN-independent
    using QT  = QuantType;
    const int L = 1;
    auto shape_B = cute::make_shape(N, K, L);
    auto layout_B_logical = cute::make_layout(
        shape_B, cutlass::make_cute_packed_stride(typename G_t::StrideB_logical{}, shape_B));
    auto layout_B_reordered = cute::tile_to_shape(typename G_t::LayoutAtomQuant{}, shape_B);
    cutlass::reorder_tensor(
        reinterpret_cast<const QT*>(B_codes.data_ptr<uint8_t>()), layout_B_logical,
        reinterpret_cast<QT*>(B_reordered.data_ptr<uint8_t>()), layout_B_reordered);
    C10_CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// HOST ENTRY POINTS (the public ABI the serving fusion layer calls). QuantType /
// MmaType are fixed per-entry by the format: i2f4 bf16-act = <uint2b_t,bfloat16_t>,
// i2f4 fp8-act = <uint2b_t,float_e4m3_t>, i4f8 bf16-act = <uint4b_t,bfloat16_t>.
// Add one thin wrapper per (format, act-dtype) pair the build enables; the core
// templates above are instantiated ONCE per (TileN x QuantType x MmaType).
// FP8-ACT NOTE: A must already be per-token-rowmax-scaled fp8 (e4m3) — the
// rowmax/quantize is the CALLER's responsibility (host-side, before this op);
// this entry consumes the pre-quantized fp8 activation directly.
template<class QuantType, class MmaType>
static void ommx_cute_base_impl(
    torch::Tensor out, torch::Tensor A, torch::Tensor B_codes,
    torch::Tensor scales, torch::Tensor zeros,
    int64_t N, int64_t K, int64_t group_size, bool shuffled,
    torch::optional<torch::Tensor> workspace) {
    TORCH_CHECK(B_codes.is_cuda() && B_codes.dtype() == torch::kUInt8,
                "B_codes must be CUDA uint8 (packed sub-byte codes)");
    TORCH_CHECK(scales.is_cuda() && zeros.is_cuda(), "scales/zeros must be CUDA");
    TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kBFloat16, "out must be CUDA bf16 [M,N]");
    const int M  = static_cast<int>(A.size(0));
    const int Ni = static_cast<int>(N), Ki = static_cast<int>(K), gs = static_cast<int>(group_size);
    const int64_t G = Ki / gs;
    ommx_check_kmajor(scales, Ni, G, "scales");
    ommx_check_kmajor(zeros,  Ni, G, "zeros");
    void* ws_ptr = nullptr; int64_t ws_bytes = 0;
    if (workspace.has_value()) {
        const torch::Tensor& w = workspace.value();
        TORCH_CHECK(w.is_cuda() && w.dtype() == torch::kUInt8, "workspace must be CUDA uint8");
        TORCH_CHECK(w.device() == A.device(), "workspace must be on A's device");
        TORCH_CHECK(w.is_contiguous(), "workspace must be contiguous");
        ws_ptr = w.data_ptr(); ws_bytes = w.numel();
    }
    if (shuffled) {
        if      (M <= 16) ommx_cute_run_shuffled<16,  QuantType, MmaType>(out, A, B_codes, scales, zeros, M, Ni, Ki, gs, ws_ptr, ws_bytes);
        else if (M <= 64) ommx_cute_run_shuffled<64,  QuantType, MmaType>(out, A, B_codes, scales, zeros, M, Ni, Ki, gs, ws_ptr, ws_bytes);
        else              ommx_cute_run_shuffled<128, QuantType, MmaType>(out, A, B_codes, scales, zeros, M, Ni, Ki, gs, ws_ptr, ws_bytes);
    } else {
        if      (M <= 16) ommx_cute_run_dp<16,  QuantType, MmaType>(out, A, B_codes, scales, zeros, M, Ni, Ki, gs, ws_ptr, ws_bytes);
        else if (M <= 64) ommx_cute_run_dp<64,  QuantType, MmaType>(out, A, B_codes, scales, zeros, M, Ni, Ki, gs, ws_ptr, ws_bytes);
        else              ommx_cute_run_dp<128, QuantType, MmaType>(out, A, B_codes, scales, zeros, M, Ni, Ki, gs, ws_ptr, ws_bytes);
    }
}

template<class QuantType, class MmaType>
static int64_t ommx_cute_workspace_size_impl(int64_t M, int64_t N, int64_t K, int64_t group_size) {
    const int Mi = (int)M, Ni = (int)N, Ki = (int)K, gs = (int)group_size;
    int64_t b16  = ommx_cute_workspace_size_for_tile<16,  QuantType, MmaType>(Mi, Ni, Ki, gs);
    int64_t b64  = ommx_cute_workspace_size_for_tile<64,  QuantType, MmaType>(Mi, Ni, Ki, gs);
    int64_t b128 = ommx_cute_workspace_size_for_tile<128, QuantType, MmaType>(Mi, Ni, Ki, gs);
    return std::max(b16, std::max(b64, b128));   // shuffled shares this (identical scratch)
}

// ---- Concrete public wrappers (i2f4 fp8-act = production; add bf16/i4f8 as built).
// A-dtype guard differs per MmaType; only the fp8 entry is shown — clone for bf16
// (A.dtype()==kBFloat16, MmaType=bfloat16_t) and i4f8 (QuantType=uint4b_t).
// DEFAULT = bf16-act i2f4 (MmaType=bfloat16_t, CUTLASS 3.x mixed-input; apples-to-apples
// vs cuBLAS bf16). fp8-act lane (float_e4m3_t, +per-token rowmax) is opt-in via OMMX_CUTE_FP8
// to stay within the cicc collective budget (one MmaType x TileN{16,64,128} x {dp,shuffled}).
void cute_base(torch::Tensor out, torch::Tensor A, torch::Tensor B_re_or_codes,
               torch::Tensor scale, torch::Tensor zp,
               int64_t N, int64_t K, int64_t VL, bool shuffled,
               torch::optional<torch::Tensor> workspace) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
#if defined(OMMX_CUTE_FP8)
    if (A.dtype() == torch::kFloat8_e4m3fn) {
        ++g_fire_cute_fp8;
        ommx_cute_base_impl<cutlass::uint2b_t, cutlass::float_e4m3_t>(
            out, A, B_re_or_codes, scale, zp, N, K, VL, shuffled, workspace);
        return;
    }
#endif
    TORCH_CHECK(A.dtype() == torch::kBFloat16,
                "cute_base A must be bf16 (default bf16-act; build with OMMX_CUTE_FP8 for fp8 e4m3)");
    ++g_fire_cute_bf16;
    ommx_cute_base_impl<cutlass::uint2b_t, cutlass::bfloat16_t>(
        out, A, B_re_or_codes, scale, zp, N, K, VL, shuffled, workspace);
}
#if defined(OMMX_CUTE_FUSED)
// cute_base_fused — the i2f4 FUSED path (SHUFFLED base + in-mainloop FP4-outlier correction).
// Sidecar: oindex (relidx7 packed [N,n_blk,idx_blk_bytes]) + odelta (BLOCK-major bf16 [N,n_blk*npv],
// elem [n,bi*npv+s]). npv==0 / undefined oindex -> base-only (byte-identical to cute_base shuffled).
// fused_eps>0 = injection probe. bf16-act i2f4 only; B_reordered = cute_reorder_B(code); b_blk==VL.
void cute_base_fused(torch::Tensor out, torch::Tensor A, torch::Tensor B_reordered,
                     torch::Tensor scale, torch::Tensor zp,
                     int64_t N, int64_t K, int64_t VL, double fused_eps,
                     torch::optional<torch::Tensor> bitmask, torch::optional<torch::Tensor> odelta,
                     int64_t npv, int64_t b_blk,
                     torch::optional<torch::Tensor> workspace) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kBFloat16, "cute_base_fused A must be CUDA bf16");
    TORCH_CHECK(B_reordered.is_cuda() && B_reordered.dtype() == torch::kUInt8, "B_reordered must be CUDA uint8");
    TORCH_CHECK(scale.is_cuda() && zp.is_cuda(), "scale/zp must be CUDA");
    TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kBFloat16, "out must be CUDA bf16 [M,N]");
    using QT = cutlass::uint2b_t; using MT = cutlass::bfloat16_t;
    const int M = static_cast<int>(A.size(0));
    const int Ni = static_cast<int>(N), Ki = static_cast<int>(K), gs = static_cast<int>(VL);
    const int64_t G = Ki / gs;
    ommx_check_kmajor(scale, Ni, G, "scale");
    ommx_check_kmajor(zp,    Ni, G, "zp");
    void* ws_ptr = nullptr; int64_t ws_bytes = 0;
    if (workspace.has_value()) {
        const torch::Tensor& w = workspace.value();
        TORCH_CHECK(w.is_cuda() && w.dtype() == torch::kUInt8 && w.is_contiguous(), "workspace must be CUDA uint8 contiguous");
        ws_ptr = w.data_ptr(); ws_bytes = w.numel();
    }
    const uint64_t* bm_ptr = nullptr; const cutlass::bfloat16_t* od_ptr = nullptr;
    int npv_i = 0, bblk_i = 0;
    if (bitmask.has_value() && odelta.has_value() && npv > 0) {
        TORCH_CHECK(bitmask->is_cuda() && bitmask->dtype() == torch::kInt64, "bitmask must be CUDA int64 [N,n_blk]");
        TORCH_CHECK(odelta->is_cuda() && odelta->dtype() == torch::kBFloat16, "odelta must be CUDA bf16 (block-major, position-ordered)");
        bm_ptr = reinterpret_cast<const uint64_t*>(bitmask->data_ptr<int64_t>());
        od_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(odelta->data_ptr<at::BFloat16>());
        npv_i = static_cast<int>(npv);
        bblk_i = static_cast<int>(b_blk > 0 ? b_blk : VL);
    }
    ++g_fire_cute_bf16;
    const float eps = static_cast<float>(fused_eps);
    const char* abl = std::getenv("OMMX_FUSED_ABLATE");
    const int ablate = abl ? atoi(abl) : 0;   // DIAG bitmask (1 barrier,2 gread,4 scatter)
    if      (M <= 16) ommx_cute_run_fused<16,  QT, MT>(out, A, B_reordered, scale, zp, M, Ni, Ki, gs, eps, bm_ptr, od_ptr, npv_i, bblk_i, ablate, ws_ptr, ws_bytes);
    else if (M <= 64) ommx_cute_run_fused<64,  QT, MT>(out, A, B_reordered, scale, zp, M, Ni, Ki, gs, eps, bm_ptr, od_ptr, npv_i, bblk_i, ablate, ws_ptr, ws_bytes);
    else              ommx_cute_run_fused<128, QT, MT>(out, A, B_reordered, scale, zp, M, Ni, Ki, gs, eps, bm_ptr, od_ptr, npv_i, bblk_i, ablate, ws_ptr, ws_bytes);
}
#endif  // OMMX_CUTE_FUSED
// act_fp8 selects the shuffle atom / workspace of the fp8-act twin (MmaType=float_e4m3_t):
// compute_memory_reordering_atom<MmaType> (~line 905) and the CUTLASS workspace are BOTH
// MmaType-dependent, so the fp8 lane MUST reorder + size with float_e4m3_t (a bf16-atom
// permutation fed to the fp8 mainloop = silent wrong dequant). Only instantiated when the
// fp8 collective is compiled (OMMX_CUTE_FP8); else act_fp8=true honestly errors.
void reorder_B(torch::Tensor B_reordered, torch::Tensor B_codes, int64_t N, int64_t K,
               bool act_fp8) {
    TORCH_CHECK(B_codes.is_cuda() && B_codes.dtype() == torch::kUInt8, "B_codes must be CUDA uint8");
    TORCH_CHECK(B_reordered.is_cuda() && B_reordered.dtype() == torch::kUInt8, "B_reordered must be CUDA uint8");
    TORCH_CHECK(B_reordered.numel() == B_codes.numel(), "B_reordered must match B_codes byte count");
    if (act_fp8) {
#if defined(OMMX_CUTE_FP8)
        ommx_cute_reorder_B<cutlass::uint2b_t, cutlass::float_e4m3_t>(
            B_reordered, B_codes, (int)N, (int)K);
        return;
#else
        TORCH_CHECK(false, "reorder_B(act_fp8=true) needs the fp8-act build (OMMX_CUTE_FP8)");
#endif
    }
    ommx_cute_reorder_B<cutlass::uint2b_t, cutlass::bfloat16_t>(
        B_reordered, B_codes, (int)N, (int)K);
}
int64_t workspace_size(int64_t M, int64_t N, int64_t K, int64_t VL, bool act_fp8) {
    if (act_fp8) {
#if defined(OMMX_CUTE_FP8)
        return ommx_cute_workspace_size_impl<cutlass::uint2b_t, cutlass::float_e4m3_t>(M, N, K, VL);
#else
        TORCH_CHECK(false, "workspace_size(act_fp8=true) needs the fp8-act build (OMMX_CUTE_FP8)");
#endif
    }
    return ommx_cute_workspace_size_impl<cutlass::uint2b_t, cutlass::bfloat16_t>(M, N, K, VL);
}

}}}  // namespace ommx::linear::cute90
#endif  // OMMX_ENABLE_SM90_CUTLASS

// ============================================================================
// PART D — SPARSE OUTLIER CORRECTION  (decoupled, LLM.int8 2-path "addmm")
// C += A @ dW^T where dW is nonzero only at the top-K outliers. Applied AFTER
// the base GEMM (PART B/C ran base-only) into the SAME output buffer. ALU only.
// Unified over M-branch x index-encoding. Default = on-disk relidx7 compact
// (reads the bundle sidecar directly; no CSR materialized at runtime).
// ============================================================================
namespace ommx { namespace linear {
namespace nc = ommx::numeric;
using namespace ommx::numeric;   // StoreTraits + RELIDX_BITS/relidx7_slot_delta visibility

// ── D1. M==1 warp-per-column relidx7 correction (in-place add onto fp32 out). ──
template<class StoreFmt>
__global__ void sparse_correct_gemv_relidx7_kernel(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const uint8_t*       __restrict__ code,     // [N, K/EPB]  (for -code term)
    const float*         __restrict__ scale,    // [N, G]
    const uint8_t*       __restrict__ oindex,   // relidx7 [N, n_blk*idx_blk_bytes]
    const uint8_t*       __restrict__ ocode,    // FP4 nibble [N, n_blk*nib_per_blk]
    const float*         __restrict__ map_scale,// [N, G]
    const float*         __restrict__ map_center,//[N, G]
    float*               __restrict__ out,      // [1, N]  (holds base; += corr)
    int N, int K, int vector_length, int npv, int B,
    unsigned long long*  __restrict__ fired, int idx_fmt) {  // law #5 firing counter (opt)
    constexpr int EPB = StoreTraits<StoreFmt>::elems_per_byte;
    constexpr int WARPS = 8;
    const int warp = (blockIdx.x * WARPS) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (warp >= N) return;
    const int n = warp, G = K / vector_length;
    const uint8_t* code_row = code + static_cast<size_t>(n) * (K / EPB);
    const float* scale_row  = scale + static_cast<size_t>(n) * G;
    const float* ms_row     = map_scale  + static_cast<size_t>(n) * G;
    const float* mc_row     = map_center + static_cast<size_t>(n) * G;
    const int n_blk = (K + B - 1) / B;
    const int idx_blk_bytes = nc::idx_block_bytes(idx_fmt, npv, B);
    const int nib_per_blk   = (npv + 1) / 2;
    const int vl_sh = nc::group_shift(vector_length);
    float corr = 0.f;
    for (int bi = lane; bi < n_blk; bi += 32) {
        const uint8_t* oi_blk = oindex + (static_cast<size_t>(n) * n_blk + bi) * idx_blk_bytes;
        const uint8_t* oc_blk = ocode  + (static_cast<size_t>(n) * n_blk + bi) * nib_per_blk;
        nc::SlotWalker wk(idx_fmt, oi_blk, npv, B);   // sequential positions, O(1)/slot
        #pragma unroll 4
        for (int s = 0; s < npv; ++s) {
            const int local = wk.next();
            if (local < 0) break;                      // bitmap: block population exhausted
            int k; float d;
            if (nc::slot_delta_at(local, bi, B, s, K, vector_length, vl_sh,
                    oc_blk, code_row, scale_row, ms_row, mc_row, &k, &d))
                corr = fmaf(__bfloat162float(A[k]), d, corr);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) corr += __shfl_down_sync(0xffffffffu, corr, off);
    if (lane == 0) {
        out[n] += corr;
        if (fired != nullptr && corr != 0.f) atomicAdd(fired, 1ull);
    }
}

// ── D2. M>1 CTA-per-row: warp0 decodes the row's outliers ONCE into smem, then
//        ALL threads stream the M rows applying the correction (decode amortized
//        over M, not per-(m,n)). C holds the base-GEMM bf16 output; += corr. ──
template<class StoreFmt, int MAX_STAGE>
__global__ void sparse_correct_batched_relidx7_kernel(
    const __nv_bfloat16* __restrict__ At,       // [K, M]  (A^T, coalesced col reads)
    const uint8_t*       __restrict__ code,     // [N, K/EPB]
    const float*         __restrict__ scale,    // [N, G]
    const uint8_t*       __restrict__ oindex, const uint8_t* __restrict__ ocode,
    const float*         __restrict__ map_scale, const float* __restrict__ map_center,
    __nv_bfloat16*       __restrict__ C,        // [M, N] (base; += corr)
    int N, int M, int K, int vector_length, int npv, int B,
    unsigned long long*  __restrict__ fired, int idx_fmt) {
    constexpr int EPB = StoreTraits<StoreFmt>::elems_per_byte;
    const int n = blockIdx.x;
    if (n >= N) return;
    const int G = K / vector_length;
    const uint8_t* code_row = code + static_cast<size_t>(n) * (K / EPB);
    const float* scale_row  = scale + static_cast<size_t>(n) * G;
    const float* ms_row     = map_scale  + static_cast<size_t>(n) * G;
    const float* mc_row     = map_center + static_cast<size_t>(n) * G;
    const int n_blk = (K + B - 1) / B;
    const int idx_blk_bytes = nc::idx_block_bytes(idx_fmt, npv, B);
    const int nib_per_blk   = (npv + 1) / 2;

    __shared__ int   k_sh[MAX_STAGE];
    __shared__ float d_sh[MAX_STAGE];
    __shared__ int   s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();
    // warp0 decodes all of this row's outliers into smem (block-granular order).
    if (threadIdx.x < 32) {
        for (int bi = threadIdx.x; bi < n_blk; bi += 32) {
            const uint8_t* oi_blk = oindex + (static_cast<size_t>(n) * n_blk + bi) * idx_blk_bytes;
            const uint8_t* oc_blk = ocode  + (static_cast<size_t>(n) * n_blk + bi) * nib_per_blk;
            int lk[32]; float ld[32]; int cnt = 0;   // npv<=32 (MAX_NPV); host guards npv
            #pragma unroll 4
            for (int s = 0; s < npv; ++s) {
                int k; float d;
                if (nc::slot_delta_fmt(idx_fmt, n, bi, B, s, K, vector_length,
                        oi_blk, oc_blk, code_row, scale_row, ms_row, mc_row, &k, &d)
                    && cnt < 32) { lk[cnt] = k; ld[cnt] = d; ++cnt; }
            }
            int base = atomicAdd(&s_count, cnt);
            for (int i = 0; i < cnt && base + i < MAX_STAGE; ++i) {
                k_sh[base + i] = lk[i]; d_sh[base + i] = ld[i];
            }
        }
    }
    __syncthreads();
    const int cnt = s_count < MAX_STAGE ? s_count : MAX_STAGE;
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < cnt; ++s)
            acc = fmaf(__bfloat162float(At[(size_t)k_sh[s] * M + m]), d_sh[s], acc);
        if (acc != 0.f) {
            const size_t idx = (size_t)m * N + n;
            C[idx] = __float2bfloat16(__bfloat162float(C[idx]) + acc);
        }
    }
    if (threadIdx.x == 0 && fired != nullptr && cnt > 0) atomicAdd(fired, 1ull);
}

// ── D3. M>1 PARALLEL-staging + warp-per-row apply (DEFAULT; D2 kept env-gated
//        via OMMX_W_CORR_PARALLEL=0). Port of the ICCAD scanfree_parallel fix
//        (H100 forensic: b8 correction 108->18.7 ms = 5.8x,
//        TPOT 119.6->30.4 ms, cos=1.0 vs fp32 ref). D2 collapses at batched
//        decode because (a) warp0-only staging idles 224/256 threads on the
//        25%-density outlier decode EVERY step and (b) the 1-thread-per-row
//        apply serializes all staged FMAs onto M threads.
//        STAGE : every thread fills the DETERMINISTIC dense slot i = bi*npv+s
//                (i strided by blockDim.x) — no atomicAdd compaction, canonical
//                block-major/slot-minor order => stable fp32 sum order (law #10);
//                padded-tail / delta==0 slots keep the d_sh==0 sentinel and are
//                apply no-ops (keep-mask, not compaction).
//        APPLY : warp w owns rows m = w, w+nwarps, ...; its 32 lanes split the
//                staged slots, shfl tree-reduce, lane 0 alone does the single
//                bf16 C[m*N+n] read-modify-write. At is [K,M] (decode M<=16 =>
//                <=64KB per column set, L1/L2-resident) so the lane-scattered
//                At reads hit cache.
//        Law #9: the staging loop has no early-out — ALL threads reach the
//        barrier (the n>=N return is CTA-uniform and dead under grid=N). ──
template<class StoreFmt, int MAX_STAGE>
__global__ void sparse_correct_batched_relidx7_parallel_kernel(
    const __nv_bfloat16* __restrict__ At,       // [K, M]  (A^T, coalesced col reads)
    const uint8_t*       __restrict__ code,     // [N, K/EPB]
    const float*         __restrict__ scale,    // [N, G]
    const uint8_t*       __restrict__ oindex, const uint8_t* __restrict__ ocode,
    const float*         __restrict__ map_scale, const float* __restrict__ map_center,
    __nv_bfloat16*       __restrict__ C,        // [M, N] (base; += corr)
    int N, int M, int K, int vector_length, int npv, int B,
    unsigned long long*  __restrict__ fired, int idx_fmt) {
    constexpr int EPB = StoreTraits<StoreFmt>::elems_per_byte;
    const int n = blockIdx.x;
    if (n >= N) return;                          // CTA-uniform (law #9 safe)
    const int G = K / vector_length;
    const uint8_t* code_row = code + static_cast<size_t>(n) * (K / EPB);
    const float* scale_row  = scale + static_cast<size_t>(n) * G;
    const float* ms_row     = map_scale  + static_cast<size_t>(n) * G;
    const float* mc_row     = map_center + static_cast<size_t>(n) * G;
    const int n_blk = (K + B - 1) / B;
    const int idx_blk_bytes = nc::idx_block_bytes(idx_fmt, npv, B);
    const int nib_per_blk   = (npv + 1) / 2;
    const int total = n_blk * npv;               // host TORCH_CHECKs total <= MAX_STAGE

    __shared__ int   k_sh[MAX_STAGE];
    __shared__ float d_sh[MAX_STAGE];
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int bi = i / npv, s = i % npv;
        const uint8_t* oi_blk = oindex + (static_cast<size_t>(n) * n_blk + bi) * idx_blk_bytes;
        const uint8_t* oc_blk = ocode  + (static_cast<size_t>(n) * n_blk + bi) * nib_per_blk;
        int k; float d;
        const int ok = nc::slot_delta_fmt(idx_fmt, n, bi, B, s, K, vector_length,
                oi_blk, oc_blk, code_row, scale_row, ms_row, mc_row, &k, &d);
        k_sh[i] = ok ? k : 0;
        d_sh[i] = ok ? d : 0.f;                  // 0-delta sentinel -> apply no-op
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = blockDim.x >> 5;
    for (int m = warp; m < M; m += nwarps) {     // whole warp shares m (shfl-safe)
        float acc = 0.f;
        for (int s = lane; s < total; s += 32)
            if (d_sh[s] != 0.f)
                acc = fmaf(__bfloat162float(At[(size_t)k_sh[s] * M + m]), d_sh[s], acc);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0 && acc != 0.f) {
            const size_t idx = (size_t)m * N + n;
            C[idx] = __float2bfloat16(__bfloat162float(C[idx]) + acc);
        }
    }
    // law #5 firing evidence — d_sh is read-only after the barrier, no extra sync needed.
    if (threadIdx.x == 0 && fired != nullptr) {
        for (int i = 0; i < total; ++i)
            if (d_sh[i] != 0.f) { atomicAdd(fired, 1ull); break; }
    }
}

}}  // namespace ommx::linear

// ============================================================================
// PART E — HOST DISPATCH + FIRING COUNTERS + SINGLE PYBIND MODULE
// One torch extension exposing every consolidated kernel. The vLLM/serving
// orchestration (serve/vllm_ommx_linear.py "fusion layer") points its custom_op
// wrappers at THESE symbols (replacing the previous 8 separate .so modules).
// ============================================================================
namespace ommx { namespace linear {
namespace nc = ommx::numeric;

// Process-wide firing evidence (law #5). These are HOST-side launch counters:
// valid only for EAGER / capture-time forwards — a FULL CUDA-graph REPLAY bypasses
// the host increments entirely, so fire_stats() under graph replay reflects only
// the capture-time launches, not replays. The kernels also take an optional device
// `fired` counter (atomicAdd, graph-replay-visible); the host paths currently pass
// nullptr — wiring a device counter tensor through is the TODO for replay-time
// evidence. ////[GPU]//// Do NOT read fire_stats() as replay-time proof.
static unsigned long long g_fire_decode   = 0;
static unsigned long long g_fire_prefill  = 0;
static unsigned long long g_fire_outlier  = 0;

static inline void check_contig_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// Weight core = i2f4 (INT2 base + FP4 outliers) / i2 (dense). i4f8 dropped 2026-07-02.
static inline void check_fmt(const std::string& s) {
    TORCH_CHECK(s == "i2f4" || s == "i2",
                "ommx_linear: fmt must be i2f4|i2 (i4f8 removed); got ", s);
}

// ── Base decode GEMV (M<=16). Dispatches M==1 -> warp-per-channel, M in[2,16]
//    -> barrier-free batched (register-LUT). Returns fp32 [M,N] (caller casts).
//    ////[GPU]//// parity cos>=0.999 vs fakequant; SASS must show NO mma/wgmma. ──
torch::Tensor decode_base(
    torch::Tensor A, torch::Tensor code, torch::Tensor scale,
    c10::optional<torch::Tensor> zp_opt,
    int64_t N, int64_t K, int64_t vector_length, bool symmetric,
    const std::string& fmt, int64_t num_k_splits,
    c10::optional<torch::Tensor> scale_exp_opt) {   // E8M0 int8 scale (HBM lever #4); null=fp32 path
    check_contig_cuda(A, "A"); check_contig_cuda(code, "code"); check_contig_cuda(scale, "scale");
    const int M = A.size(0);
    // The batched launcher tops out at M_MAX=16 and its epilogue writes only rows m < M_MAX;
    // refuse M > 16 here (loud) instead of serving rows 16.. from an unwritten buffer.
    TORCH_CHECK(M >= 1 && M <= 16, "decode_base supports M in [1,16] (batched kernel M_MAX=16); got ", M,
                " -- route M>=16 to prefill_wmma");
    // num_k_splits == 1: BOTH kernels store every out[m*N+n] (m < M, n < N) with a plain
    // store, so the output needs no zero-fill -> torch::empty saves one memset node per
    // call (decode = one call per linear per token). The split-K atomicAdd path is the only
    // reader of a pre-zeroed buffer and keeps torch::zeros.
    auto out = (num_k_splits > 1)
        ? torch::zeros({M, N}, A.options().dtype(torch::kFloat32))
        : torch::empty({M, N}, A.options().dtype(torch::kFloat32));
    const float* zp = zp_opt.has_value() ? zp_opt->data_ptr<float>() : nullptr;
    const auto* Ap = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>());
    const uint8_t* cp = code.data_ptr<uint8_t>();
    const float* sp = scale.data_ptr<float>();
    const int8_t* se = scale_exp_opt.has_value() ? scale_exp_opt->data_ptr<int8_t>() : nullptr;
    float* op = out.data_ptr<float>();
    check_fmt(fmt);
    constexpr int WARPS = 8;
    auto stream = at::cuda::getCurrentCUDAStream();

    auto launch_m1 = [&](auto tag) {
        using SF = decltype(tag);
        dim3 grid((N + WARPS - 1) / WARPS, num_k_splits), block(WARPS * 32);
        decode_gemv_kernel<SF><<<grid, block, 0, stream>>>(
            Ap, cp, sp, se, zp, op, N, K, vector_length, symmetric, num_k_splits);
    };
    auto launch_batched = [&](auto tag) {
        using SF = decltype(tag);
        dim3 grid((N + WARPS - 1) / WARPS), block(WARPS * 32);
        if (M <= 2)  decode_gemv_batched_kernel<SF, 2 ><<<grid, block, 0, stream>>>(Ap, cp, sp, se, zp, op, N, M, K, vector_length, symmetric);
        else if (M <= 4)  decode_gemv_batched_kernel<SF, 4 ><<<grid, block, 0, stream>>>(Ap, cp, sp, se, zp, op, N, M, K, vector_length, symmetric);
        else if (M <= 8)  decode_gemv_batched_kernel<SF, 8 ><<<grid, block, 0, stream>>>(Ap, cp, sp, se, zp, op, N, M, K, vector_length, symmetric);
        else              decode_gemv_batched_kernel<SF, 16><<<grid, block, 0, stream>>>(Ap, cp, sp, se, zp, op, N, M, K, vector_length, symmetric);
    };
    if (M == 1) launch_m1(nc::StoreI2F4{});
    else        launch_batched(nc::StoreI2F4{});
    g_fire_decode++;
    return out;
}

// ── WEIGHT-STATIONARY sparse correction onto a cute-wgmma base C[M,N] (in-place, M in [1,16]).
//    The accelerated decode = cute_base (tensor-core, flat) + this (amortized outlier apply). ──
void sparse_correct_ws(
    torch::Tensor C, torch::Tensor A, torch::Tensor code, torch::Tensor scale,
    torch::Tensor oindex, torch::Tensor ocode, torch::Tensor map_scale, torch::Tensor map_center,
    int64_t N, int64_t K, int64_t vector_length, int64_t npv, int64_t B, const std::string& fmt, int64_t idx_fmt) {
    check_contig_cuda(A, "A"); check_contig_cuda(C, "C");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bf16 [M,N]");
    const int M = static_cast<int>(A.size(0));
    TORCH_CHECK(M >= 1 && M <= 16, "sparse_correct_ws supports M in [1,16], got ", M);
    check_fmt(fmt);
    const auto* Ap = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>());
    auto* Cp = reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>());
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int WARPS = 8;
    dim3 grid((N + WARPS - 1) / WARPS), block(WARPS * 32);
    auto launch = [&](auto maxm_c) {
        constexpr int MAXM = decltype(maxm_c)::value;
        sparse_correct_ws_kernel<nc::StoreI2F4, MAXM><<<grid, block, 0, stream>>>(
            Ap, code.data_ptr<uint8_t>(), scale.data_ptr<float>(),
            oindex.data_ptr<uint8_t>(), ocode.data_ptr<uint8_t>(),
            map_scale.data_ptr<float>(), map_center.data_ptr<float>(),
            Cp, N, K, M, vector_length, npv, B, idx_fmt);
    };
    if (M <= 8) launch(std::integral_constant<int, 8>{});
    else        launch(std::integral_constant<int, 16>{});
    g_fire_outlier++;
}

// ── LUT + WEIGHT-STATIONARY correction onto a cute base C[M,N] (in-place). odelta =
//    build_odelta(...) precomputed once; the hot loop skips the fp4/int2/division. ──
void sparse_correct_ws_lut(
    torch::Tensor C, torch::Tensor A, torch::Tensor oindex, torch::Tensor odelta,
    int64_t N, int64_t K, int64_t npv, int64_t B, const std::string& fmt, int64_t idx_fmt) {
    check_contig_cuda(A, "A"); check_contig_cuda(C, "C");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bf16 [M,N]");
    const int M = static_cast<int>(A.size(0));
    TORCH_CHECK(M >= 1 && M <= 16, "sparse_correct_ws_lut supports M in [1,16], got ", M);
    check_fmt(fmt);
    const auto* Ap = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>());
    const auto* Op = reinterpret_cast<const __nv_bfloat16*>(odelta.data_ptr<at::BFloat16>());
    auto* Cp = reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>());
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int WARPS = 8;
    dim3 grid((N + WARPS - 1) / WARPS), block(WARPS * 32);
    auto launch = [&](auto maxm_c) {
        constexpr int MAXM = decltype(maxm_c)::value;
        sparse_correct_ws_lut_kernel<nc::StoreI2F4, MAXM><<<grid, block, 0, stream>>>(
            Ap, oindex.data_ptr<uint8_t>(), Op, Cp, N, K, M, npv, B, idx_fmt);
    };
    if (M <= 8) launch(std::integral_constant<int, 8>{});
    else        launch(std::integral_constant<int, 16>{});
    g_fire_outlier++;
}

// ── LOAD-TIME: precomputed absolute k positions (int32 [N, n_blk*npv]) to pair with odelta
//    for the decode-FREE packed correction. ──
torch::Tensor build_kpos(
    torch::Tensor oindex, int64_t N, int64_t K, int64_t npv, int64_t B, int64_t idx_fmt) {
    check_contig_cuda(oindex, "oindex");
    const int n_blk = (K + B - 1) / B;
    auto kpos = torch::empty({N, static_cast<int64_t>(n_blk) * npv},
                             oindex.options().dtype(torch::kInt32));
    const long total = static_cast<long>(N) * n_blk * npv;
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int T = 256;
    build_kpos_kernel<nc::StoreI2F4><<<static_cast<unsigned>((total + T - 1) / T), T, 0, stream>>>(
        oindex.data_ptr<uint8_t>(), kpos.data_ptr<int>(), N, K, npv, B, idx_fmt);
    return kpos;
}

// ── DECODE-FREE packed correction onto a cute base C[M,N] (in-place). kpos = build_kpos,
//    odelta = build_odelta (both once). The accelerated decode: cute_base + this. ──
void sparse_correct_packed(
    torch::Tensor C, torch::Tensor A, torch::Tensor kpos, torch::Tensor odelta,
    int64_t N, int64_t K, int64_t npv, int64_t B, const std::string& fmt) {
    check_contig_cuda(A, "A"); check_contig_cuda(C, "C");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bf16 [M,N]");
    const int M = static_cast<int>(A.size(0));
    TORCH_CHECK(M >= 1 && M <= 16, "sparse_correct_packed supports M in [1,16], got ", M);
    check_fmt(fmt);
    const auto* Ap = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>());
    const auto* Op = reinterpret_cast<const __nv_bfloat16*>(odelta.data_ptr<at::BFloat16>());
    auto* Cp = reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>());
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int WARPS = 8;
    dim3 grid((N + WARPS - 1) / WARPS), block(WARPS * 32);
    auto launch = [&](auto maxm_c) {
        constexpr int MAXM = decltype(maxm_c)::value;
        sparse_correct_packed_kernel<nc::StoreI2F4, MAXM><<<grid, block, 0, stream>>>(
            Ap, kpos.data_ptr<int>(), Op, Cp, N, K, M, npv, B);
    };
    if (M <= 8) launch(std::integral_constant<int, 8>{});
    else        launch(std::integral_constant<int, 16>{});
    g_fire_outlier++;
}

// ── LOAD-TIME: build the sparse odelta LUT (bf16 [N, n_blk*npv]). One-time; the
//    fused-LUT decode below then skips the per-outlier delta compute. ──
torch::Tensor build_odelta(
    torch::Tensor code, torch::Tensor scale, torch::Tensor oindex, torch::Tensor ocode,
    torch::Tensor map_scale, torch::Tensor map_center,
    int64_t N, int64_t K, int64_t vector_length, int64_t npv, int64_t B, int64_t idx_fmt) {
    check_contig_cuda(code, "code");
    const int n_blk = (K + B - 1) / B;
    auto odelta = torch::empty({N, static_cast<int64_t>(n_blk) * npv},
                               code.options().dtype(torch::kBFloat16));
    const long total = static_cast<long>(N) * n_blk * npv;
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int T = 256;
    build_odelta_kernel<nc::StoreI2F4><<<static_cast<unsigned>((total + T - 1) / T), T, 0, stream>>>(
        code.data_ptr<uint8_t>(), scale.data_ptr<float>(),
        oindex.data_ptr<uint8_t>(), ocode.data_ptr<uint8_t>(),
        map_scale.data_ptr<float>(), map_center.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(odelta.data_ptr<at::BFloat16>()),
        N, K, vector_length, npv, B, idx_fmt);
    return odelta;
}

// ── LOAD-TIME: dense outlier-delta weight dW[N,K] (bf16). The PREFILL correction is then
//    C += x @ dW^T (torch/cuBLAS bf16 GEMM, A-reuse) — the tractable "option 2" fix that the
//    consolidated sparse_correct was missing. Build ONCE per weight; reuse every forward. ──
torch::Tensor build_outlier_delta_dense(
    torch::Tensor code, torch::Tensor scale, torch::Tensor oindex, torch::Tensor ocode,
    torch::Tensor map_scale, torch::Tensor map_center,
    int64_t N, int64_t K, int64_t vector_length, int64_t npv, int64_t B, int64_t idx_fmt) {
    check_contig_cuda(code, "code");
    auto dW = torch::zeros({N, K}, code.options().dtype(torch::kBFloat16));
    const int n_blk = (K + B - 1) / B;
    const long total = static_cast<long>(N) * n_blk * npv;
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int T = 256;
    build_outlier_delta_dense_kernel<nc::StoreI2F4><<<static_cast<unsigned>((total + T - 1) / T), T, 0, stream>>>(
        code.data_ptr<uint8_t>(), scale.data_ptr<float>(),
        oindex.data_ptr<uint8_t>(), ocode.data_ptr<uint8_t>(),
        map_scale.data_ptr<float>(), map_center.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dW.data_ptr<at::BFloat16>()),
        N, K, vector_length, npv, B, idx_fmt);
    return dW;
}

// ── Decoupled outlier correction onto an existing base output (LLM.int8 addmm).
//    M==1 -> in-place on fp32 out; M>1 -> bf16 C += corr (At = A^T). ──
void sparse_correct(
    torch::Tensor out_or_C, c10::optional<torch::Tensor> A_opt, c10::optional<torch::Tensor> At_opt,
    torch::Tensor code, torch::Tensor scale, torch::Tensor oindex, torch::Tensor ocode,
    torch::Tensor map_scale, torch::Tensor map_center,
    int64_t N, int64_t M, int64_t K, int64_t vector_length, int64_t npv, int64_t B,
    const std::string& fmt, int64_t idx_fmt) {
    TORCH_CHECK(npv > 0 && npv <= 32, "npv must be in (0,32] (MAX_NPV); got ", npv);
    auto stream = at::cuda::getCurrentCUDAStream();
    // M>1 kernel choice, read ONCE (static): default = D3 parallel-staging warp-per-row
    // (ICCAD-proven 5.8x at b8); OMMX_W_CORR_PARALLEL=0 keeps the refuted D2 warp0-staging
    // kernel reachable for GPU A/B (traps history: old variants stay env-gated).
    static const bool corr_parallel = [] {
        const char* e = std::getenv("OMMX_W_CORR_PARALLEL");
        return !(e != nullptr && std::string(e) == "0");
    }();
    auto go1 = [&](auto tag) { using SF = decltype(tag);
        constexpr int WARPS = 8; dim3 grid((N + WARPS - 1) / WARPS), block(WARPS * 32);
        const auto* Ap = reinterpret_cast<const __nv_bfloat16*>(A_opt->data_ptr<at::BFloat16>());
        sparse_correct_gemv_relidx7_kernel<SF><<<grid, block, 0, stream>>>(
            Ap, code.data_ptr<uint8_t>(), scale.data_ptr<float>(),
            oindex.data_ptr<uint8_t>(), ocode.data_ptr<uint8_t>(),
            map_scale.data_ptr<float>(), map_center.data_ptr<float>(),
            out_or_C.data_ptr<float>(), N, K, vector_length, npv, B, nullptr, idx_fmt); };
    auto goM = [&](auto tag) { using SF = decltype(tag);
        const auto* Atp = reinterpret_cast<const __nv_bfloat16*>(At_opt->data_ptr<at::BFloat16>());
        auto* Cp = reinterpret_cast<__nv_bfloat16*>(out_or_C.data_ptr<at::BFloat16>());
        constexpr int MAX_STAGE_DENSE = 5120;    // static smem stage bound (40 KB < 48 KB default); raised 4096->5120 to hold Phi-4 down_proj n_blk*npv=4480 (K=17920,B=128,npv=32) without silent truncation
        if (corr_parallel) {
            const int64_t n_blk = (K + B - 1) / B;
            TORCH_CHECK(n_blk * npv <= MAX_STAGE_DENSE,
                        "sparse_correct(parallel): n_blk*npv=", n_blk * npv,
                        " exceeds smem stage bound ", MAX_STAGE_DENSE,
                        " (set OMMX_W_CORR_PARALLEL=0 for the legacy kernel)");
            // ~1 warp per row while M<=32 (32*M threads), clamped to [256,1024];
            // beyond 32 rows (e.g. the dWt identity probe M=K) warps loop m += nwarps.
            int block = 32 * static_cast<int>(M);
            block = block < 256 ? 256 : (block > 1024 ? 1024 : block);
            sparse_correct_batched_relidx7_parallel_kernel<SF, MAX_STAGE_DENSE>
                <<<dim3(N), dim3(block), 0, stream>>>(
                Atp, code.data_ptr<uint8_t>(), scale.data_ptr<float>(),
                oindex.data_ptr<uint8_t>(), ocode.data_ptr<uint8_t>(),
                map_scale.data_ptr<float>(), map_center.data_ptr<float>(),
                Cp, N, M, K, vector_length, npv, B, nullptr, idx_fmt);
        } else {
            dim3 grid(N), block(256);
            sparse_correct_batched_relidx7_kernel<SF, MAX_STAGE_DENSE><<<grid, block, 0, stream>>>(
                Atp, code.data_ptr<uint8_t>(), scale.data_ptr<float>(),
                oindex.data_ptr<uint8_t>(), ocode.data_ptr<uint8_t>(),
                map_scale.data_ptr<float>(), map_center.data_ptr<float>(),
                Cp, N, M, K, vector_length, npv, B, nullptr, idx_fmt);
        } };
    check_fmt(fmt);
    if (M == 1) { TORCH_CHECK(A_opt.has_value(), "M==1 correct needs A"); go1(nc::StoreI2F4{}); }
    else        { TORCH_CHECK(At_opt.has_value(), "M>1 correct needs At"); goM(nc::StoreI2F4{}); }
    g_fire_outlier++;
}

// ── sm_80 tensor-core prefill (base-only; outliers via sparse_correct). ──
torch::Tensor prefill_wmma(
    torch::Tensor A, torch::Tensor code, torch::Tensor scale, c10::optional<torch::Tensor> zp_opt,
    int64_t N, int64_t M, int64_t K, int64_t vector_length, bool symmetric,
    const std::string& fmt, int64_t splitk,
    // FUSED FP4-outlier overlay (oindex + odelta present -> single-pass fused i2f4; absent -> base-only).
    c10::optional<torch::Tensor> oindex_opt = {}, c10::optional<torch::Tensor> odelta_opt = {},
    int64_t npv = 0, int64_t B_blk = 0) {
    check_contig_cuda(A, "A");
    check_fmt(fmt);
    TORCH_CHECK(code.dtype() == torch::kUInt8, "code must be uint8");
    TORCH_CHECK(scale.dtype() == torch::kFloat32, "scale must be fp32");
    TORCH_CHECK(M >= 16, "prefill_wmma needs M>=16 (m16n8k16 16-row tile); route M<16 to decode_base (law #1)");
    auto C = torch::empty({M, N}, A.options());
    const float* zp = zp_opt.has_value() ? zp_opt->data_ptr<float>() : nullptr;
    const uint8_t* oip = oindex_opt.has_value() ? oindex_opt->data_ptr<uint8_t>() : nullptr;
    const __nv_bfloat16* odp = odelta_opt.has_value()
        ? reinterpret_cast<const __nv_bfloat16*>(odelta_opt->data_ptr<at::BFloat16>()) : nullptr;
    const int npvi = (int)npv, bblki = (int)B_blk;
    const auto* Ap = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>());
    auto* Cp = reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>());
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int WARPS_PER_CTA = 4, MMA_M = 16, MMA_N = 8, MMA_K = 16, BN = WARPS_PER_CTA * MMA_N;
    const int mblk = (int)((M + MMA_M - 1) / MMA_M);
    const int MT = (mblk <= 2) ? 2 : (mblk <= 4 ? 4 : 8);          // m-block reuse band
    const int RPC = MT * MMA_M;
    const int n_ktiles = (int)((K + MMA_K - 1) / MMA_K);
    // split-K = compile-time SPLITK. honor caller splitk>1 (must divide n_ktiles),
    // else SM-saturation auto-fill when the base grid underfills (Marlin-style).
    const int base_ctas = (int)((N + BN - 1) / BN) * (int)((M + RPC - 1) / RPC);
    int sk = (splitk > 1) ? (int)splitk : 1;
    if (sk == 1 && base_ctas < 108 && n_ktiles >= 4) {
        const int want = (216 + base_ctas - 1) / base_ctas;
        for (int s : {8, 4, 2}) if (s <= want && (n_ktiles % s) == 0) { sk = s; break; }
    }
    if (sk > 1 && (n_ktiles % sk) != 0) sk = 1;                    // SPLITK must divide K-tiles
    torch::Tensor Cf; float* Cfp = nullptr;
    if (sk > 1) { Cf = torch::zeros({M, N}, A.options().dtype(torch::kFloat32)); Cfp = Cf.data_ptr<float>(); }
    #define OMMX_LAUNCH(MTv, SKv) do { constexpr int RPCv = (MTv) * MMA_M; \
        dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((M + RPCv - 1) / RPCv), (unsigned)(SKv)); \
        dim3 block(WARPS_PER_CTA * 32); \
        auto go = [&](auto tag) { using SF = decltype(tag); \
            prefill_wmma_kernel<SF, WARPS_PER_CTA, (MTv), (SKv)><<<grid, block, 0, stream>>>( \
                Ap, code.data_ptr<uint8_t>(), scale.data_ptr<float>(), zp, \
                ((SKv) == 1 ? nullptr : Cfp), ((SKv) == 1 ? Cp : nullptr), \
                (int)N, (int)M, (int)K, (int)vector_length, symmetric, (SKv), \
                oip, odp, npvi, bblki); }; \
        go(nc::StoreI2F4{}); } while (0)
    #define OMMX_DISPATCH_SK(MTv) do { \
        if (sk == 8) OMMX_LAUNCH(MTv, 8); else if (sk == 4) OMMX_LAUNCH(MTv, 4); \
        else if (sk == 2) OMMX_LAUNCH(MTv, 2); else OMMX_LAUNCH(MTv, 1); } while (0)
    if (MT == 2) OMMX_DISPATCH_SK(2); else if (MT == 4) OMMX_DISPATCH_SK(4); else OMMX_DISPATCH_SK(8);
    #undef OMMX_DISPATCH_SK
    #undef OMMX_LAUNCH
    C10_CUDA_CHECK(cudaGetLastError());   // launch error check (graph-safe, not a host sync)
    if (sk > 1) { const int total = (int)(M * N); constexpr int CT = 256;
        cast_f32_to_bf16_kernel<<<(total + CT - 1) / CT, CT, 0, stream>>>(Cfp, Cp, total);
        C10_CUDA_CHECK(cudaGetLastError()); }
    g_fire_prefill++;
    return C;
}

std::map<std::string, int64_t> fire_stats() {   // std::map -> pybind Python dict (needs pybind11/stl.h)
    std::map<std::string, int64_t> d;
    d["decode_calls"]  = (int64_t)g_fire_decode;
    d["prefill_calls"] = (int64_t)g_fire_prefill;
    d["outlier_calls"] = (int64_t)g_fire_outlier;
    d["cute_bf16_calls"] = (int64_t)g_fire_cute_bf16;
    d["cute_fp8_calls"]  = (int64_t)g_fire_cute_fp8;
    return d;
}

}}  // namespace ommx::linear

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace L = ommx::linear;
    m.doc() = "OMMX consolidated linear kernels (single TU, single dequant core)";
    m.def("decode_base", &L::decode_base,
          "INT2/INT4 base decode GEMV (M<=16, ALU)",
          py::arg("A"), py::arg("code"), py::arg("scale"), py::arg("zp"),
          py::arg("N"), py::arg("K"), py::arg("vector_length"),
          py::arg("symmetric") = false,   // weight recipe default = affine (law #13)
          py::arg("fmt") = "i2f4", py::arg("num_k_splits") = 1,
          py::arg("scale_exp") = c10::optional<torch::Tensor>());  // E8M0 int8 scale (HBM lever #4); None=fp32
    m.def("sparse_correct_ws", &L::sparse_correct_ws,
          "weight-stationary sparse FP4-outlier correction onto a bf16 base C[M,N] in-place "
          "(M in [1,16]); pair with cute_base for accelerated decode (tensor-core base + this)");
    m.def("sparse_correct_ws_lut", &L::sparse_correct_ws_lut,
          "LUT + weight-stationary correction onto a bf16 base C[M,N] (precomputed odelta -> "
          "skips fp4/int2/division in the hot loop); pair build_odelta + cute_base for decode");
    m.def("build_kpos", &L::build_kpos,
          "precompute absolute k positions int32 [N,n_blk*npv] for the packed correction. "
          "idx_fmt: 0 = relidx7 (7 bits/outlier), 1 = flat bitmask (1 bit/position)");
    m.def("sparse_correct_packed", &L::sparse_correct_packed,
          "DECODE-FREE weight-stationary correction onto a bf16 base C[M,N] using precomputed "
          "(kpos, odelta) -> memory-bound, no relidx7/fp4/int2 decode; the fastest decode correction");
    m.def("build_outlier_delta_dense", &L::build_outlier_delta_dense,
          "dense outlier-delta weight dW[N,K] bf16 (sidecar->dense scatter); prefill correction "
          "C+=x@dW^T on tensor cores, replacing the DRAM-bound scalar sparse_correct at M>1");
    m.def("build_odelta", &L::build_odelta,
          "LOAD-time: precompute sparse per-outlier delta LUT bf16 [N,n_blk*npv]");
    m.def("sparse_correct", &L::sparse_correct,
          "Decoupled FP4/FP8 outlier correction (LLM.int8 2-path addmm)");
    m.def("prefill_wmma", &L::prefill_wmma,
          "sm_80 tensor-core prefill base GEMM (M>=16); pass oindex + odelta(build_odelta) + "
          "npv/B_blk for the SINGLE-PASS FUSED i2f4 (FP4 outlier overlaid onto the mma B-tile)",
          py::arg("A"), py::arg("code"), py::arg("scale"), py::arg("zp"),
          py::arg("N"), py::arg("M"), py::arg("K"), py::arg("vector_length"),
          py::arg("symmetric") = false, py::arg("fmt") = "i2f4", py::arg("splitk") = 1,
          py::arg("oindex") = c10::optional<torch::Tensor>(),
          py::arg("odelta") = c10::optional<torch::Tensor>(),
          py::arg("npv") = 0, py::arg("B_blk") = 0);
    m.def("fire_stats", &L::fire_stats, "law #5 firing evidence counters");
#if defined(OMMX_ENABLE_SM90_CUTLASS)
    // sm_90a CUTLASS prefill (wgmma+TMA, DP graph-safe). A = pre-quantized fp8 e4m3
    // (per-token rowmax scaling is caller-side); workspace pre-allocated OUTSIDE capture.
    m.def("cute_base", &ommx::linear::cute90::cute_base,
          "sm_90a mixed-input wgmma prefill GEMM (i2f4 weight, fp8 act, shuffled-B opt)");
#if defined(OMMX_CUTE_FUSED)
    m.def("cute_base_fused", &ommx::linear::cute90::cute_base_fused,
          "sm_90a i2f4 FUSED wgmma (shuffled base + in-mainloop sparse FP4-outlier correction). "
          "oindex+odelta(block-major)+npv -> memory-preserving fusion; fused_eps>0 = injection probe",
          py::arg("out"), py::arg("A"), py::arg("B_reordered"), py::arg("scale"), py::arg("zp"),
          py::arg("N"), py::arg("K"), py::arg("VL"), py::arg("fused_eps") = 0.0,
          py::arg("bitmask") = torch::optional<torch::Tensor>(),
          py::arg("odelta") = torch::optional<torch::Tensor>(),
          py::arg("npv") = 0, py::arg("b_blk") = 0,
          py::arg("workspace") = torch::optional<torch::Tensor>());
#endif
    m.def("cute_reorder_B", &ommx::linear::cute90::reorder_B,
          "warmup INT2 weight pre-shuffle (Machete 128-bit-load memory-bound lever)",
          py::arg("B_reordered"), py::arg("B_codes"), py::arg("N"), py::arg("K"),
          py::arg("act_fp8") = false);   // fp8-act twin uses the float_e4m3_t shuffle atom
    m.def("cute_workspace_size", &ommx::linear::cute90::workspace_size,
          "persistent CUTLASS workspace bytes (allocate OUTSIDE cudagraph capture)",
          py::arg("M"), py::arg("N"), py::arg("K"), py::arg("VL"),
          py::arg("act_fp8") = false);   // fp8-act twin sizes the float_e4m3_t collective
#endif
}
