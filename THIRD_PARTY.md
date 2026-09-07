# Third-party code redistributed in this repository

The root `LICENSE` is Apache-2.0 and covers **our** code: `ommx_gpu_serve/`,
`ommx_fakequant/`, `figure/`, `repro/`, `run.sh` -- except where a file inside those
trees says otherwise in its own header. Two do, and both are listed in the table
below: `ommx_gpu_serve/hf_eager/` carries Apache-2.0 code derived from HuggingFace
transformers, and `ommx_gpu_serve/csrc/linear/ommx_sm90_mixed_input_fused.hpp` is
BSD-3-Clause. "Ours" is a claim about provenance, so it has to name its exceptions.

It does not cover the trees below. They were vendored so a reader can reproduce a
number without chasing five upstreams at their commits, and vendoring is
redistribution: the upstream licence text and copyright notices travel with the
code, and a root licence file does not relicense someone else's work by sitting
above it.

`scripts/check_release_hygiene.sh` reads this file. A vendored tree that is not
listed here, or listed as `BLOCKER`, fails the gate; a row whose licence text is
missing must say so in its `state` column (none does today) — so the
question cannot be answered by forgetting about it.

> 루트 `LICENSE`(Apache-2.0)는 **우리 코드**만 덮는다. 아래 트리는 vendoring =
> 재배포이므로 상류 라이선스 원문과 저작권 고지가 함께 가야 하며, 루트 라이선스
> 파일이 위에 있다고 해서 남의 코드가 재라이선스되지는 않는다.

<!-- Read by scripts/check_release_hygiene.sh. One place to change them, so the
     script cannot quietly disagree with this page. -->

    OURS_SPDX: Apache-2.0
    OURS_COPYRIGHT: OMMX Contributors

| path | files | lines | upstream | licence | state |
|---|---:|---:|---|---|---|
| `eval/lm_eval` | 92 | 22333 | EleutherAI / lm-evaluation-harness (v0.4.8 tree) | MIT (`eval/lm_eval/LICENSE.md`, upstream text; files carry NVIDIA / HuggingFace Apache-2.0 notices of their own) | LICENSE vendored 2026-09-06 |
| `eval/lcb` | 10 | 1110 | LiveCodeBench | MIT upstream (`eval/lcb/LICENSE`); the in-tree files are ours and carry Apache-2.0 SPDX headers | LICENSE vendored 2026-09-06 |
| `baseline/kivi` | 20 | 7792 | KIVI ([jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)) | MIT (`baseline/kivi/LICENSE`); the transformers-derived model files keep their Apache-2.0 headers; `quant/csrc/gemv_cuda.cu` derives from AWQ and llama_cu_awq, both MIT (`quant/csrc/LICENSE.awq`, `LICENSE.llama_cu_awq`) | 876b4d2 |
| `baseline/kitty` | 3 | 710 | Kitty ([Summer-Summer/Kitty](https://github.com/Summer-Summer/Kitty)) | MIT (`baseline/kitty/LICENSE`, vendored from upstream); `_kitty_llama_modeling.py` is additionally derived from HuggingFace transformers and carries its Apache-2.0 copyright notice in the file header | dfd2c07; HF notice added 2026-09-07 |
| `ommx_gpu_serve/csrc/linear/ommx_sm90_mixed_input_fused.hpp` | 1 | - | NVIDIA CUTLASS 4.4.2 | BSD-3-Clause (`ommx_gpu_serve/csrc/linear/LICENSE.cutlass`, upstream text); the NVIDIA copyright notice is also in the file header | LICENSE vendored 2026-09-07 |
| `ommx_gpu_serve/csrc/linear/LICENSE.cutlass` | 1 | 28 | NVIDIA CUTLASS 4.4.2 (`LICENSE.txt`) | BSD-3-Clause, verbatim upstream text -- it is the licence the row above points at, and it carries NVIDIA's copyright, so it is declared rather than exempted | LICENSE vendored 2026-09-07 |
| `ommx_gpu_serve/hf_eager` | 2 | 66 | HuggingFace transformers (`models/llama/modeling_llama.py`) | Apache-2.0, same as the root `LICENSE`; the upstream copyright notice is in each file header. Derived symbols: `LlamaRMSNorm`, `LlamaMLP`, `rotate_half`, `apply_rotary_pos_emb`, `repeat_kv`, `eager_attention_forward` | notice added 2026-09-07 |

## What the files themselves say

One of these is not a directory. `ommx_sm90_mixed_input_fused.hpp` is a single file
inside our own tree, and it is the reason the check works file by file: a
directory-level rule flagged all six of `ommx_gpu_serve/`'s subdirectories -- our
own headers say "Copyright" too -- and would still have walked straight past the
one file whose licence actually differs from the root.

Its own header says it plainly: *"OMMX fork of CUTLASS 4.4.2 sm90 mixed-input RS
collective"*, `SPDX-License-Identifier: BSD-3-Clause`. The SPDX tag and the fork
statement are already there and honest. What is missing is the upstream copyright
notice, which BSD-3 asks a redistributor to reproduce.

Measured, not assumed — `git grep` over each tree:

- **`eval/lm_eval`** — 13 self-references to
  `https://github.com/EleutherAI/lm-evaluation-harness`. Individual files carry
  third-party notices of their own: `Copyright (c) 2024, NVIDIA CORPORATION`,
  `Copyright 2020 The HuggingFace Datasets Authors`, both Apache-2.0.
- **`eval/lcb`** — `SPDX-License-Identifier: Apache-2.0`.
- **`baseline/kivi`** — Apache-2.0 headers, with copyrights held by
  `Meta Platforms, Inc.`, `Huawei Technologies Co., Ltd.` and
  `Mistral AI and the HuggingFace Inc. team` (model code derived from
  `transformers`).
- **`baseline/kitty`** — our Llama port of Kitty's Qwen3 modeling, no header of its own;
  the upstream `LICENSE` (MIT, Copyright (c) 2025 Haojun Xia) is vendored beside it.

## `baseline/kitty` and `baseline/kivi`

Both upstreams publish an MIT `LICENSE`; each is vendored inside its tree at the commit the
code was taken from (Kitty dfd2c07, KIVI 876b4d2), and the `state` column carries that
commit. `baseline/kivi`'s model files are derived from `transformers` and keep their
Apache-2.0 headers; the MIT file covers the KIVI kernels and packers.

## The remaining rows

`eval/lm_eval` and `eval/lcb` carry their upstream MIT texts; the CUTLASS-derived header
carries NVIDIA's copyright notice. The KIVI CUDA kernel's own derivation chain (AWQ,
llama_cu_awq) is recorded beside it. Nothing in this table is open.
