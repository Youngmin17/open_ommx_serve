# OMMX: a brief for the decode demo

This document is the prompt of the demo in this directory. Every number in it is taken from
the README, the CHANGELOG or the figure data of this repository, and the model being demoed is
asked to explain OMMX from this text alone.

## 1. What OMMX is

OMMX (Outlier-Managed MX) is a dual-resolution microscaling (MX) number format for the tensors
that dominate LLM inference memory traffic: the KV cache and the weights. Inside every group of
values, the majority is stored as MXINT2, a two-bit integer under a shared power-of-two scale
(an E8M0 exponent), and a small, high-coverage set of outliers is stored as MXFP4, a four-bit
floating-point value (E2M1) under a dedicated range map for that group. The two resolutions
share the group scale, so the dense part costs two bits per element and the outliers add a
few bits per group on top. The format was published at ICCAD 2026 under the title
"High-Coverage Outlier Representation in Low-Bit MX Formats for Efficient LLM Inference".

The point of the design is that low-bit quantization fails on a few large-magnitude values
rather than on the bulk of a tensor. Giving those few values a wider, floating-point
representation while keeping everything else at two bits recovers most of the accuracy of a
much wider format at close to the cost of the two-bit one. Accuracy in the paper is compared
against KIVI, KVQuant, Oaken, AWQ, GPTQ and MXFP4, and on RULER-32K it recovers as outlier
coverage grows.

## 2. The serving recipe

The recipe that every serving number in this repository uses is the same one the accuracy
numbers were measured under. For the key cache, K is MXINT2 with six MXFP4 outliers per group
of 32 tokens by 32 channels; the group scale is a power of two; outlier positions inside the
group are stored as a flat bitmask, which the packer calls the bitmap representation. The
value cache V is INT2 with a per-group scale and zero point. The first 8 tokens of a sequence
(the attention sink) and the most recent 32 tokens are kept in bf16 and never packed. On the
NPU accounting basis this comes to 4.13 bits per KV element instead of 16 for bf16, roughly a
four-fold reduction in KV-cache bytes.

Weights use the same idea: an INT2 base with a power-of-two scale per group of 64, plus FP4
outliers whose positions are packed as a compact slot stream. The repository ships weight
bundles in a safetensors-based format called OMMX_W_SafeTensor, built offline from a Hugging
Face checkpoint.

## 3. The reference implementation

The accuracy evaluation runs on a fake-quantization reference, vendored in this repository as
ommx_fakequant. It quantizes and dequantizes tensors in floating point without any packed
representation, and it is the oracle everything else is checked against. One consequence of
that discipline is a fix recorded in the CHANGELOG: the serving packer used to round FP4
outliers to the nearest level in linear magnitude, while the reference rounds in log magnitude
(the boundaries sit at geometric means and ties go to the lower level). About three percent of
the served outliers sat one FP4 level away from the evaluated ones. The packer now uses the
reference's rule, and a test file pins the FP4 codec, the power-of-two scale and its E8M0
byte, the INT2 bases for K and V, and the outlier values at fixed positions bit-exact against
the reference. The one thing deliberately left to each side is the ratio, that is which
positions are chosen as outliers.

## 4. The decode attention kernel

Decode attention over a packed KV cache is a Triton kernel with a split-KV structure: a stage
one kernel walks the packed pages, dequantizes K and V on the fly, computes the query-key
scores, runs an online softmax and accumulates the probability-weighted values, and a small
merge kernel combines the splits with their log-sum-exp terms. Outlier values are spliced
into the dequantized K from the bitmap and the FP4 payload; the INT2 values are recovered
through a register-resident lookup table.

Whether the kernel uses tensor cores depends on the shape. The query-key and
probability-value products go through a tensor-core tile when the tile height reaches 16
rows; for Llama-3.1-8B, whose grouped-query attention puts four query heads on each KV head,
that happens from 4K tokens of context on. Below that the kernel runs the same reduction on
CUDA cores. This was verified by dumping the generated PTX: no mma instructions at 1K
tokens, 24 of them at 16K. The kernel computes in bf16 with fp32 accumulation on both the
A100 and the H200; an FP8 query-key path exists but is opt-in, Hopper-only and measured to
be no faster.

The number of warps per block for the outlier stage follows a measured ladder rather than a
guess. On the H200 a single request prefers two warps up to 4K tokens, four warps at 8K to
16K and two warps again from 24K; batches of eight prefer four warps below 24K and two above.
On the A100 four warps win everywhere. Eight warps never win on either GPU. A register diet
that removed a replicated integer tile and serialized part of the outlier splice is bit-exact
and is part of the shipped kernel.

## 5. Serving integration

OMMX serves through two paths. The first is a vLLM plugin: it registers a quantization
method named ommx_w for the weight bundles and a custom attention backend for the packed KV
cache. The KV cache keeps vLLM's page abstraction; with the engine's default block size of 16
tokens, one 32-token quantization group spans two pages, and a 32-token block that aligns one
group to one page is supported and verified but costs about 26 percent at 4K tokens, so the
published bars use the default. Decode runs under CUDA graphs. The second path is an
HF-eager integration: a Llama modeling file for Hugging Face transformers whose attention
uses the same Triton decode kernel over the same packed cache, one forward per token, with
no engine around it. It exists so the KV kernel can be compared with baselines that only run
in that framework.

Both paths leave evidence that they actually ran. The vLLM backend writes sentinel tags to a
file: a route-fired tag that carries the K format and the outlier representation, a page-grid
tag with the block size, and an ablation tag whenever an ablation flag was active in the
environment. The kernel launcher keeps per-process counters of launches and bitmap reads.
The benchmarks refuse to publish an arm whose evidence is missing or whose environment
carried an ablation flag, because the failure mode being guarded against is a run that
silently served bf16 attention under an OMMX label.

## 6. Measured results

The main figure is decode time per output token, TPOT, for Llama-3.1-8B-Instruct at batch
one with greedy decoding, taking the median over decode steps. On an H200, at 64K tokens of
context, KIVI takes 163.92 ms per token in HF-eager, OMMX in HF-eager takes 30.43 ms, and
OMMX inside vLLM takes 22.63 ms. Against KIVI that is 5.39 times faster kernel to kernel and
7.24 times faster with the engine included. The same recipe on an A100 80GB PCIe reaches
38.3 ms at 64K, 7.7 times faster than KIVI there. Kitty, another 2-bit KV baseline, takes
133.61 ms at 64K on the H200, and TurboQuant, a 3-bit KV cache served through vLLM, takes
36.72 ms.

OMMX does not beat uncompressed bf16 in latency. vLLM's bf16 Triton attention decodes at
14.31 ms per token at 64K and its default FlashAttention path at 5.8 ms, both faster than
OMMX. The gain over bf16 is KV capacity, four times more context or batch in the same memory,
not speed. A stage ablation at 64K attributes 16.1 of OMMX's 30.4 ms in HF-eager to operand
reconstruction: 7.7 ms for the outlier splice, 5.0 ms for unpacking INT2, 3.4 ms for the
scale and zero point, on top of a 14.3 ms floor that the framework and the bf16 parts
already cost.

For the weight path, a kernel microbenchmark on the square 4096-by-4096 shape at batch one
and 6.25 percent outliers gives, on the H200, 0.0277 ms for the two-launch pair, 0.0400 ms
for the CuTe wgmma base with packed correction, 0.0358 ms for bf16, 0.0404 ms for CUTLASS
mixed-input INT4 and 0.0974 ms for LLM.int8. The two-launch pair wins at batch one and the
CuTe base from batch two on; the outlier correction grows with batch while bf16, CUTLASS and
int8 stay flat. On the A100 the host is shared and the same bf16 matmul moved between 0.041
and 0.104 ms across runs, so that panel is only comparable within one batch point.

## 7. Baselines and how they were run

KIVI runs its official model code in fp16 on its dequantize-and-dense path, which is the
faster of its two paths at every published context; its fused CUDA kernel is routed away by
KIVI's own dispatch on Hopper. Kitty runs the official package's cache and Triton kernels in
fp16, and those kernels do use tensor cores. TurboQuant is the attention backend that ships
inside the installed vLLM 0.21.0, selected through the KV-cache dtype. The bf16 references
are a Hugging Face model with a static cache and vLLM with its Triton attention, with
FlashAttention reported as well. Only the vLLM arms are paged; the HF-eager arms use each
model's own cache.

## 8. Limitations to keep in mind

The kernel is bound by operand reconstruction on the decode critical path, which is why the
advantage over bf16 is capacity rather than latency. The A100 linear microbenchmark runs on
a shared host with visible run-to-run variance. The Kitty baseline directory carries no
upstream licence text, which the release gate holds open as a decision rather than a task.
And the demo that reads this document is a single request per arm, not a benchmark; the
benchmarks behind the README figures are the source of every number above.
