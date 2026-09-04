# Third-party code redistributed in this repository

The root `LICENSE` is Apache-2.0 and covers **our** code: `ommx_gpu_serve/`,
`ommx_fakequant/`, `figure/`, `repro/`, `run.sh`.

It does not cover the trees below. They were vendored so a reader can reproduce a
number without chasing four upstreams at four commits, and vendoring is
redistribution: the upstream licence text and copyright notices travel with the
code, and a root licence file does not relicense someone else's work by sitting
above it.

`scripts/check_release_hygiene.sh` reads this file. A vendored tree that is not
listed here, or listed with an unresolved licence, fails the gate — so the
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
| `eval/lm_eval` | 91 | 22311 | EleutherAI / lm-evaluation-harness | see below | **needs upstream LICENSE** |
| `eval/lcb` | 9 | 1085 | LiveCodeBench | Apache-2.0 (SPDX header in-tree) | **needs upstream LICENSE** |
| `baseline/kivi` | 13 | 7043 | KIVI | Apache-2.0 headers in-tree | **needs upstream LICENSE** |
| `baseline/kitty` | 2 | 690 | Kitty | **undetermined** | **BLOCKER** |
| `ommx_gpu_serve/csrc/linear/ommx_sm90_mixed_input_fused.hpp` | 1 | - | NVIDIA CUTLASS 4.4.2 | BSD-3-Clause (SPDX header in-file) | **needs the CUTLASS copyright notice** |

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
- **`baseline/kitty`** — no copyright line, no SPDX tag, no licence header.

## Why `baseline/kitty` is marked a blocker

Redistributing code whose licence is unknown is the one case here that cannot be
fixed by adding a file: nobody can add a licence on the author's behalf. Publishing
it is a decision, not a task, and it has three honest resolutions:

1. establish the upstream and vendor its `LICENSE` alongside the code;
2. remove the tree and reference the upstream by URL and commit, so the numbers
   stay reproducible without redistribution;
3. keep it and record an explicit decision here, with who made it.

Until one of those, the gate stays red for this path, on purpose.

## Closing the other three

For each: place the upstream `LICENSE` file inside the tree at the commit it was
taken from, and change `state` above to that commit hash. Nothing else in this
repository has to move — the code is already separated by directory, and the root
`LICENSE` already scopes itself in its first paragraph above.

**This page reports what the files state. It is not a legal determination, and
whether the result is publishable is not a question a checker answers.**
