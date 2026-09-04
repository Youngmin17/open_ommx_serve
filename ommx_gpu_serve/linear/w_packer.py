# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""OMMX W-Packer — HF safetensors checkpoint -> ``OMMX_W_SafeTensor`` bundle.

This is the "OMMX W-Packer" / "OMMX Safetensor Weight Format Offline Packing" box of
the paper's Fig 6 (claims D3 and D4), which this release otherwise ships no code for.

Fig 6 names four steps; they live in :mod:`ommx_gpu_serve.linear.quantize` under
``--- Fig 6 step (n) ---`` banners and are listed here as :data:`FIG6_STEPS` so the
figure-to-code mapping is checkable rather than asserted:

    (1) Vector Grouping        [N, K] -> [N, K/gs, gs] view along K
    (2) Min/Max Scaling        per-group range EXCLUDING outliers -> E8M0 scale + ZP
    (3) Top-K Permutation      top-npv |w| per group -> position stream (relidx7/bitmap)
    (4) FP4 Outlier Encoding   index-space E2M1 range map -> nibble stream

WHAT THIS PACKER GUARANTEES
---------------------------
* The output is a SELF-CONTAINED MODEL DIRECTORY. As well as the ommx_w shards and
  ``ommx_w_index.json``, every non-weight file of the source checkpoint is copied
  through — ``config.json``, ``generation_config.json``, the tokenizer files, the
  chat template, and anything else, by EXCLUSION rather than by a whitelist (see the
  aux-file constants below). Without them the documented recipe, "pack, then point
  vLLM at the bundle directory", cannot resolve the model at all. Which files were
  copied, skipped or absent is recorded in the index under ``aux_files``. A source
  checkpoint with no ``config.json`` is a hard PackError, and files that carry
  weights are never copied — a leftover ``pytorch_model.bin`` in the bundle is a
  loader candidate that would serve the ORIGINAL unquantized weights under the
  ``ommx_w`` label.
* Only transformer Linear projections are quantized. Embeddings, ``lm_head``,
  norms and biases are COPIED THROUGH byte-for-byte and recorded as such in the
  manifest. A silently quantized ``lm_head`` is a correctness disaster that surfaces
  only as a slightly worse eval score, so every tensor's decision is written down.
* Plane names REPLACE the quantized tensor's ``.weight`` suffix
  (``...q_proj.weight`` -> ``...q_proj.ommx_code``), which is what lets a plane bind
  to the parameter a vLLM Linear registers. See ``w_format``'s SPEC block; this is
  the change that took the format from version 1 to version 2.
* It streams: one input shard is opened at a time, one tensor is materialised at a
  time, and the output payload is spilled to a temp file so the header can be
  written last (see ``w_format.SafeTensorsWriter``). Peak RAM is one tensor.
* It refuses bad input loudly — a K that the group size does not divide, a dtype it
  cannot quantize, an output directory that already holds a bundle, a source with no
  ``config.json``. There is no "skip this and carry on" path; a partially quantized
  or unloadable bundle is worse than no bundle.

WHAT IT DOES NOT DO (no GPU this session)
-----------------------------------------
Nothing here has been fed to a CUDA kernel. The bundle is verified on CPU by
round-tripping it through ``quantize.dequantize_ommx_weight`` and by the validator;
whether ``csrc/linear/ommx_linear.cu`` consumes these planes correctly is what the
GPU parity gate measures, and that needs a device.

CLI
---
    python -m ommx_gpu_serve.linear.w_packer make-synthetic --output /tmp/ckpt
    python -m ommx_gpu_serve.linear.w_packer pack --input /tmp/ckpt --output /tmp/bundle --dry-run
    python -m ommx_gpu_serve.linear.w_packer pack --input /tmp/ckpt --output /tmp/bundle
    python -m ommx_gpu_serve.linear.w_packer verify --bundle /tmp/bundle
    python -m ommx_gpu_serve.linear.w_packer budget
    python -m ommx_gpu_serve.linear.w_packer recipes                 # named presets
    python -m ommx_gpu_serve.linear.w_packer budget --preset paper-weight
    python -m ommx_gpu_serve.linear.w_packer pack --preset paper-weight --input ... --output ...

NAMED PRESETS. ``--preset`` selects a recipe from :mod:`ommx_gpu_serve.recipes` so that
reproducing a published bit budget is a flag rather than five knobs guessed by hand:
``shipped-weight`` is today's defaults (MEASURED 4.1250 bits/weight) and
``paper-weight`` is the recipe that MEASURES the ICCAD Table 1 weight AvgBits of 3.63
(claim E2) exactly — 3.6250, at gs 64 / npv 4 / flat-bitmask positions / no range map.
``OMMX_RECIPE`` selects the same thing without the flag (``pack`` and ``budget`` both
honour it) so ONE selector works on both axes; ``--preset`` wins over it, a KV recipe
name is reported on stderr and leaves the weight defaults alone, and an unknown name is
refused with the known names listed.
Any flag typed explicitly still wins over the preset. NOTE ``paper-weight`` PACKS but
does not SERVE: the shipped CUDA kernel has no bitmap reader and requires the range-map
planes, so ``linear_method.require_kernel_readable`` refuses such a bundle at load
(loudly, by design). ``w_packer recipes`` prints the registry.

``pack --no-copy-aux`` opts out of the model-file copy and records the opt-out;
``make-synthetic --no-aux`` writes shards with no config.json, which is the input the
packer is supposed to refuse.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .quantize import (
    FIG6_STEPS,
    OMMXQuantizeError,
    OUTLIER_MAPS,
    OUTLIER_REPRS,
    derive_npv,
    dequantize_ommx_weight,
    quantize_ommx_weight,
)
from ..recipes import (
    UnknownRecipeError,
    env_preset_name,
    format_recipe,
    format_table,
)
from ..recipes import get as get_recipe
from ..recipes import names as recipe_names
from .w_format import (
    AUX_POLICIES,
    INDEX_FILENAME,
    OMMX_W_FORMAT,
    OMMX_W_FORMAT_VERSION,
    OMMXWFormatError,
    REQUIRED_AUX_FILES,
    Recipe,
    SafeTensorsFile,
    SafeTensorsWriter,
    bits_per_weight,
    naming_document,
    plane_name,
    sha256_file,
    st_dtype_name,
    validate_bundle,
)

PRODUCER = "ommx_gpu_serve.linear.w_packer"

#: Module basenames whose ``.weight`` is quantized. This is the standard
#: Llama/Mistral/Qwen decoder Linear set; anything else is passed through. It is a
#: WHITELIST on purpose — an unrecognised projection in a new architecture should
#: arrive in the bundle un-quantized and visible in the manifest rather than be
#: quantized on a guess.
DEFAULT_QUANT_MODULES: Tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)

#: Names that must NEVER be quantized even if some future whitelist matched them.
#: Kept as an explicit second gate because the cost of getting this wrong is silent.
NEVER_QUANTIZE: Tuple[str, ...] = ("lm_head", "embed_tokens", "embed_out", "wte", "lm_head.weight")

#: Input dtypes a weight may have. All three convert to float32 exactly (f32 is a
#: no-op, bf16/f16 -> f32 widens), which is what the quantizer requires.
QUANTIZABLE_DTYPES: Tuple[str, ...] = ("F32", "F16", "BF16")

_SHARD_FMT = "ommx_w-{:05d}-of-{:05d}.safetensors"

# ════════════════════════════════════════════════════════════════════════════
# non-weight ("aux") checkpoint files — what makes the output a MODEL DIRECTORY
# ════════════════════════════════════════════════════════════════════════════
# The bundle's documented recipe is "pack, then point vLLM at the bundle directory".
# Before this, ``pack`` wrote ONLY the ommx_w shards and ommx_w_index.json, so that
# recipe could not work: with no config.json neither transformers nor vLLM can decide
# which architecture to build, and the run dies before a single plane is read.
#
# POLICY: copy EVERY non-weight regular file through, by EXCLUSION rather than by a
# whitelist. A whitelist is the wrong shape here — the set of files HF needs grows
# (chat_template.jinja, preprocessor_config.json, tokenizer.model, .safetensors-free
# tokenizer variants, per-model extras), and a file this packer has never heard of
# that silently fails to arrive is precisely the "silently-broken bundle" failure
# mode. Anything unrecognised is therefore COPIED, not dropped.
#
# The exclusion list is short and every entry earns its place:

#: Extensions of files that CARRY WEIGHTS. Never copied. This is the one exclusion
#: that is a correctness gate rather than tidiness: a ``pytorch_model.bin`` or a stray
#: ``model-00001-of-00002.safetensors`` sitting next to the ommx shards is a loader
#: candidate, and a loader that picks it serves the ORIGINAL unquantized weights while
#: the arm is labelled ``ommx_w`` — the exact defect MEASURED_FACTS §2 records on the
#: attention side (a CUSTOM run byte-identical to bf16, timed under the OMMX label).
WEIGHT_FILE_SUFFIXES: Tuple[str, ...] = (
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack", ".gguf",
    ".onnx", ".npz", ".pkl",
)

#: Weight INDEX files. Excluded for the same reason plus one more: they name shards
#: (``model-00001-of-00002.safetensors``) that do not exist in the bundle, so a loader
#: that honoured one would fail with a missing-file error about a file the user never
#: asked for. The bundle's own index is ``ommx_w_index.json``.
WEIGHT_INDEX_SUFFIXES: Tuple[str, ...] = (".safetensors.index.json", ".bin.index.json")

#: Files the packer itself produces. Excluded so that packing a bundle-shaped
#: directory cannot smuggle a stale index in as an "aux file".
GENERATED_FILENAMES: Tuple[str, ...] = (INDEX_FILENAME,)

#: Aux files whose absence is worth a LOUD warning even though it is not fatal. Not
#: required (a base model legitimately has no ``generation_config.json``; a
#: sentencepiece model has ``tokenizer.model`` and no ``tokenizer.json``), but each
#: one changes behaviour if it goes missing, and silently different behaviour after
#: quantization reads as "quantization damaged the model".
NOTABLE_AUX_FILES: Tuple[str, ...] = (
    "generation_config.json",       # eos/pad ids, sampling defaults -> generation stops
    "tokenizer_config.json",        # chat template lives here on most models
    "special_tokens_map.json",
)

#: Any ONE of these makes the bundle tokenizable. Warned about as a group, because
#: which member a model ships is architecture-dependent.
TOKENIZER_FILES: Tuple[str, ...] = (
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json",
    "vocab.txt", "merges.txt", "spiece.model", "sentencepiece.bpe.model",
)


class PackError(RuntimeError):
    """The packer refuses to produce a bundle. Message names the offending tensor."""


# ════════════════════════════════════════════════════════════════════════════
# tensor classification
# ════════════════════════════════════════════════════════════════════════════

def classify_tensor(name: str, shape: Sequence[int], dtype: str,
                    quant_modules: Sequence[str] = DEFAULT_QUANT_MODULES
                    ) -> Tuple[str, str]:
    """Decide ``("quantized"|"passthrough", reason)`` for one checkpoint tensor.

    The reason string is written into the manifest verbatim, so a user can audit
    every decision the packer made without re-running it.
    """
    parts = name.split(".")
    if parts[-1] != "weight":
        return "passthrough", f"not a '.weight' tensor (tail={parts[-1]!r})"
    if len(shape) != 2:
        return "passthrough", f"not a 2-D matrix (ndim={len(shape)})"
    if any(tag in name for tag in NEVER_QUANTIZE):
        return "passthrough", ("output head / embedding table — never quantized "
                               "(accuracy-critical, and the kernel path is Linear-only)")
    module = parts[-2] if len(parts) >= 2 else ""
    if "norm" in module or "norm" in parts[-1]:
        return "passthrough", f"normalisation weight ({module!r})"
    if module not in quant_modules:
        return "passthrough", (f"module {module!r} is not a whitelisted transformer "
                               f"Linear projection {tuple(quant_modules)}")
    return "quantized", f"transformer Linear projection ({module})"


# ════════════════════════════════════════════════════════════════════════════
# checkpoint discovery
# ════════════════════════════════════════════════════════════════════════════

def discover_shards(src_dir: str) -> List[str]:
    """Ordered list of safetensors shard paths in an HF checkpoint directory."""
    if not os.path.isdir(src_dir):
        raise PackError(f"input {src_dir!r} is not a directory")
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".safetensors"))
    if not files:
        raise PackError(
            f"{src_dir!r} contains no *.safetensors shard. This packer reads "
            f"safetensors checkpoints only; convert a .bin checkpoint first.")
    return [os.path.join(src_dir, f) for f in files]


# ════════════════════════════════════════════════════════════════════════════
# aux files
# ════════════════════════════════════════════════════════════════════════════

def classify_aux_file(name: str, src_dir: str) -> Tuple[bool, str]:
    """``(copy?, reason)`` for one entry of the source checkpoint directory.

    Exclusion-based (see the POLICY comment above the constants): an entry is copied
    unless there is a NAMED reason not to, so a file this packer has never heard of
    still reaches the bundle.
    """
    path = os.path.join(src_dir, name)
    if os.path.isdir(path):
        # Not recursed on purpose: HF checkpoints are flat, and silently flattening a
        # subdirectory into the bundle root would rename its files. Recorded, not
        # dropped, so an operator with a nested layout sees it in the report.
        return False, "directory (the packer copies top-level files only)"
    if not os.path.isfile(path):
        return False, "not a regular file (symlink target missing, socket, fifo)"
    low = name.lower()
    if name in GENERATED_FILENAMES:
        return False, f"{name} is produced by the packer itself"
    if low.startswith("ommx_w-") and low.endswith(".safetensors"):
        return False, "an OMMX_W_SafeTensor shard from a previous pack"
    for suf in WEIGHT_INDEX_SUFFIXES:
        if low.endswith(suf):
            return False, (f"weight index ({suf}) — it names source shards that are "
                           f"not in the bundle; ommx_w_index.json replaces it")
    for suf in WEIGHT_FILE_SUFFIXES:
        if low.endswith(suf):
            return False, (f"carries weights ({suf}) — copying it would leave the "
                           f"ORIGINAL unquantized weights in the bundle for a loader "
                           f"to pick up and serve under the ommx_w label")
    return True, "non-weight checkpoint file"


def plan_aux_files(src_dir: str, copy_aux: bool = True) -> dict:
    """Decide which source files become part of the bundle. Writes nothing.

    Returns the ``aux_files`` record that goes into ``ommx_w_index.json`` (minus the
    per-file byte counts and digests, which only exist once a file is copied), plus
    ``warnings``: strings the caller must print.

    REFUSAL RULE, and why it is a refusal and not a warning: a missing
    :data:`~ommx_gpu_serve.linear.w_format.REQUIRED_AUX_FILES` entry (``config.json``)
    is a HARD FAIL. Every other outcome here is recoverable by the operator later — a
    missing tokenizer can be supplied with ``--tokenizer``, a missing
    ``generation_config.json`` can be re-added — but a bundle with no ``config.json``
    is a directory that no loader can identify, and producing one while reporting
    success is exactly the "pack succeeded, nothing can load it" defect this function
    exists to end. Warning instead would put the diagnosis an hour downstream, inside
    a transformers traceback about an absent ``architectures`` key.
    """
    entries = sorted(os.listdir(src_dir))
    plan: List[Tuple[str, bool, str]] = [
        (n,) + classify_aux_file(n, src_dir) for n in entries]
    present = {n for n, ok, _ in plan if ok}
    warnings: List[str] = []

    if not copy_aux:
        # The escape hatch still has to be honest about what it produced.
        return {
            "policy": "no-copy-aux",
            "self_contained": False,
            "copied": [],
            "skipped": [{"name": n, "reason": r} for n, ok, r in plan if not ok],
            "missing": [{"name": n, "reason": "--no-copy-aux was passed"}
                        for n in sorted(present)],
            "warnings": [
                "--no-copy-aux: the bundle will contain ONLY the ommx_w shards and "
                f"{INDEX_FILENAME}. It is NOT a loadable model directory — pointing "
                "vLLM or transformers at it fails before any weight is read. Supply "
                "config/tokenizer another way, or re-pack without the flag."],
        }

    missing_required = [f for f in REQUIRED_AUX_FILES if f not in present]
    if missing_required:
        raise PackError(
            f"{src_dir!r} has no {missing_required[0]} (missing: {missing_required}). "
            f"An OMMX_W_SafeTensor bundle is a MODEL DIRECTORY — the documented recipe "
            f"is to point vLLM straight at it — and without {missing_required[0]} "
            f"neither transformers nor vLLM can decide which architecture to build, so "
            f"the bundle would be unloadable. Point --input at the full checkpoint "
            f"directory (the one holding config.json next to the shards), or pass "
            f"--no-copy-aux to deliberately produce a weights-only bundle and supply "
            f"the model files yourself.")

    for name in NOTABLE_AUX_FILES:
        if name not in present:
            warnings.append(
                f"{name} is not in {src_dir!r}, so the bundle will not have one. That "
                f"is legal, but it CHANGES BEHAVIOUR (eos/pad ids and sampling "
                f"defaults, or the chat template) and the difference will look like "
                f"quantization damage. Confirm the source checkpoint really lacks it.")
    if not any(f in present for f in TOKENIZER_FILES):
        warnings.append(
            f"no tokenizer file ({', '.join(TOKENIZER_FILES)}) in {src_dir!r}. The "
            f"bundle will need an explicit --tokenizer when it is served.")

    return {
        "policy": "copy",
        "self_contained": True,
        "copied": [{"name": n} for n in sorted(present)],
        "skipped": [{"name": n, "reason": r} for n, ok, r in plan if not ok],
        "missing": [{"name": n, "reason": "not present in the source checkpoint"}
                    for n in NOTABLE_AUX_FILES if n not in present],
        "warnings": warnings,
    }


def copy_aux_files(src_dir: str, out_dir: str, plan: dict) -> dict:
    """Copy the planned files and fill in each one's byte count and SHA-256.

    Copies in bounded chunks (a ``tokenizer.json`` is tens of MB) and writes through a
    temp file renamed into place, so an interrupted pack cannot leave a half-written
    ``config.json`` that looks complete. The digest is recorded as PROVENANCE — the
    validator does not enforce it, because editing ``config.json`` inside a bundle is
    a legitimate thing to do (see ``w_format.validate_aux_files``).
    """
    copied: List[dict] = []
    for ent in plan["copied"]:
        name = ent["name"]
        src = os.path.join(src_dir, name)
        dst = os.path.join(out_dir, name)
        tmp = dst + ".ommx_partial"
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            while True:
                block = fin.read(1 << 20)
                if not block:
                    break
                fout.write(block)
        os.replace(tmp, dst)
        copied.append({"name": name, "bytes": os.path.getsize(dst),
                       "sha256": sha256_file(dst)})
    out = dict(plan)
    out["copied"] = copied
    return out


#: The key vLLM (and transformers) read out of ``config.json`` to decide which
#: quantization implementation owns a checkpoint.
HF_QUANT_CONFIG_KEY = "quantization_config"

#: The quantization method name this packer stamps. DUPLICATED from
#: ``integration.vllm.linear_method.OMMX_W_METHOD_NAME`` rather than imported: the packer
#: is an OFFLINE tool that must run in an environment with no serving integration at all,
#: and importing the integration package to read one string would couple them. The two are
#: pinned equal by ``tests/test_quant_config_stamp.py`` — if they ever drift, a bundle
#: would advertise a method name the serving side does not answer to, which is exactly the
#: undiscoverable-bundle failure this stamping exists to fix.
OMMX_W_METHOD_NAME = "ommx_w"


def stamp_quantization_config(out_dir: str, recipe, aux: dict) -> dict:
    """Write ``quantization_config: {"quant_method": "ommx_w", ...}`` into the bundle.

    WHY THIS EXISTS — the bundle used to be undiscoverable. ``OMMXWConfig`` registers
    fine under the name ``ommx_w`` (the plugin verifies it against vLLM's own registry),
    and ``from_config`` / ``resolve_bundle_dir`` have always been ready to receive the
    dict. But vLLM only CALLS ``from_config`` when the model's HF config carries a
    ``quantization_config``, and this packer copied ``config.json`` through byte-for-byte
    and never wrote one. So ``vllm serve <bundle> --quantization ommx_w`` — the invocation
    the README documents — could never construct the config, and the whole weight-quant
    path was unreachable from the documented entry point. One key closes that.

    WHAT IS DELIBERATELY *NOT* WRITTEN: a ``bundle`` path. ``resolve_bundle_dir`` accepts
    one, but an absolute path baked into the checkpoint breaks the moment the directory
    is moved or mounted elsewhere. Discovery instead falls to its last rule — "the model
    path vLLM is currently building for, when it IS a bundle" — which is exactly the
    ``vllm serve <bundle>`` case and is relocation-proof. ``$OMMX_W_BUNDLE`` remains the
    override for serving a bundle that sits beside, not at, ``--model``.

    REFUSES to overwrite a FOREIGN quantization_config: packing an already-quantized
    checkpoint (AWQ, GPTQ) and stamping ``ommx_w`` over its config would produce a
    directory that claims a method whose planes it does not contain.

    Returns the refreshed ``aux`` record (config.json's byte count and digest re-taken
    after the edit) so the index describes the file that is actually on disk.
    """
    path = os.path.join(out_dir, "config.json")
    if not os.path.exists(path):
        # plan_aux_files already hard-fails on a missing config.json under the copy
        # policy; under --no-copy-aux there is nothing to stamp and nothing to claim.
        return aux
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    existing = cfg.get(HF_QUANT_CONFIG_KEY)
    if isinstance(existing, dict):
        method = str(existing.get("quant_method", "")) or "<unnamed>"
        if method != OMMX_W_METHOD_NAME:
            raise PackError(
                f"the source checkpoint's config.json already declares "
                f"{HF_QUANT_CONFIG_KEY}.quant_method={method!r}. This is an "
                f"already-quantized checkpoint; stamping {OMMX_W_METHOD_NAME!r} over it "
                f"would claim a method whose planes the bundle does not contain. Pack "
                f"from the unquantized checkpoint, or pass --no-quant-config to write "
                f"the bundle without the discovery key.")

    cfg[HF_QUANT_CONFIG_KEY] = {
        "quant_method": OMMX_W_METHOD_NAME,
        # Echoed for humans and for tools that inspect a checkpoint without opening the
        # index. NOT read back by from_config (which takes only quant_method and then
        # reads the authoritative recipe out of ommx_w_index.json), so these can never
        # disagree with the planes in a way that changes behaviour.
        "format": OMMX_W_FORMAT,
        "version": OMMX_W_FORMAT_VERSION,
        "recipe": recipe.to_json(),
    }
    tmp = path + ".ommx_partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)

    out = dict(aux)
    out["copied"] = [
        ({**e, "bytes": os.path.getsize(path), "sha256": sha256_file(path)}
         if e.get("name") == "config.json" else e)
        for e in aux.get("copied", [])
    ]
    out["quant_config_stamped"] = True
    return out


# ════════════════════════════════════════════════════════════════════════════
# packing
# ════════════════════════════════════════════════════════════════════════════

#: Planes a calibration must supply per module for the encode-only path. These are the
#: DECISIONS -- which lanes are outliers, the group scale/zero-point, the FP4 range map --
#: not the encoded bytes; the packer still does the encoding, so a calibrated bundle and an
#: RTN bundle are the same format written by the same code.
CALIB_PLANES: Tuple[str, ...] = ("omask", "scale", "zp", "map_scale", "map_center")


class CalibrationSource:
    """A directory of per-module quantization DECISIONS produced by a calibrated solver.

    WHY THE PACKER NEEDS THIS AT ALL. The shipped packer decides its INT2 codes by plain
    min/max round-to-nearest: no calibration, no activation-aware saliency. Measured on
    Llama-3.1-8B at the repo's own ``shipped-weight`` recipe, that destroys the model --
    the served output degenerates to a repeated token fragment, and a pure-numeric
    simulation (no kernel, no vLLM) reproduces it exactly, so it is the quantization and
    not the serving path. ``ommx_fakequant/wq_eval.py``'s error-feedback solver is what
    makes 2-bit weights work ("strongest exactly at 2-bit", its own docstring), and until
    now there was no way to get its result into a bundle.

    WHY DECISIONS AND NOT THE CALIBRATED WEIGHT. The solver returns Q, the DEQUANTIZED
    weight. Re-quantizing Q through the min/max path is NOT idempotent -- measured
    max|W_ref2 - W_ref1| = 5.2e-2 with different codes AND different scales -- so packing
    Q the ordinary way would silently discard the calibration and produce a bundle that
    looks fine. Handing over the solver's own per-group scale / zero-point / outlier mask /
    range map is lossless, and :meth:`verify` proves it per module rather than assuming it.

    FILE LAYOUT: ``<dir>/<module>.safetensors`` holding ``omask`` (bool [N, G, gs]),
    ``scale``, ``zp``, ``map_scale``, ``map_center`` ([N, G]) and ``W`` (the calibrated
    weight the decisions were made against). Module name = the checkpoint tensor name
    without the trailing ``.weight``.
    """

    def __init__(self, root: str) -> None:
        if not os.path.isdir(root):
            raise PackError(f"--calibrated {root!r} is not a directory")
        self.root = root
        self.used: List[str] = []
        self.available = sorted(
            f[: -len(".safetensors")] for f in os.listdir(root)
            if f.endswith(".safetensors"))
        if not self.available:
            raise PackError(
                f"--calibrated {root!r} holds no <module>.safetensors files. Export the "
                f"solver's decisions there first (see CALIB_PLANES).")

    def _path(self, module: str) -> str:
        return os.path.join(self.root, module + ".safetensors")

    def decisions_for(self, tensor_name: str, W32: torch.Tensor):
        """``(weight_to_encode, decided)`` for one checkpoint tensor.

        A quantized tensor with NO calibration file is a hard failure, not a fallback to
        RTN: a bundle that is calibrated for some layers and round-to-nearest for others
        is neither recipe, and the difference is invisible once packed.
        """
        module = tensor_name[: -len(".weight")] if tensor_name.endswith(".weight") \
            else tensor_name
        path = self._path(module)
        if not os.path.exists(path):
            raise PackError(
                f"{tensor_name}: --calibrated was given but {path} does not exist. Every "
                f"quantized tensor needs its own decisions; packing this one with plain "
                f"min/max instead would make the bundle a silent mixture of two recipes. "
                f"({len(self.available)} module file(s) present, e.g. "
                f"{self.available[:3]})")
        f = SafeTensorsFile(path)
        names = set(f.names)
        missing = [p for p in CALIB_PLANES if p not in names]
        if missing:
            raise PackError(f"{path}: missing decision plane(s) {missing}; expected "
                            f"{list(CALIB_PLANES)} (+ optional 'W')")
        dec = {p: f.tensor(p) for p in CALIB_PLANES}
        # The solver's Q, when it exported one: the codes were chosen against THESE
        # values, so they -- not the original checkpoint weight -- are what gets
        # encoded. Falling back to W32 keeps the path usable for a decisions-only export.
        Wc = f.tensor("W").to(torch.float32) if "W" in names else W32
        if tuple(Wc.shape) != tuple(W32.shape):
            raise PackError(
                f"{path}: calibrated W has shape {tuple(Wc.shape)} but the checkpoint "
                f"tensor is {tuple(W32.shape)} -- these are different layers.")
        self.used.append(module)
        return Wc, dec

    def verify(self, tensor_name: str, planes: Dict[str, torch.Tensor],
               Wc: torch.Tensor, recipe: Recipe) -> None:
        """The gate: the encoded planes must reconstruct the calibrated weight EXACTLY.

        This is the whole reason the decisions travel instead of the weight. If the
        reconstruction drifts, the bundle is not what the solver measured its accuracy on,
        and every downstream number would be attributed to a calibration that was not
        actually shipped. Exact, not approximate -- the encode-only path is bit-identical
        to the deciding path by construction, so any difference is a real defect.
        """
        N, K = int(Wc.shape[0]), int(Wc.shape[1])
        G = K // recipe.group_size
        rec = dequantize_ommx_weight(
            code=planes["code"], scale_exp=planes["scale_exp"], zp=planes["zp"],
            N=N, K=K, group_size=recipe.group_size, npv=recipe.npv,
            oindex=planes.get("oindex"), ocode=planes.get("ocode"),
            map_scale=planes.get("map_scale"), map_center=planes.get("map_center"),
            outlier_repr=recipe.outlier_repr)
        diff = float((rec.float() - Wc.float()).abs().max())
        # NOT "exactly 0": the solver evaluates code*scale + zp in its own order and the
        # dequantizer in another, so two float32 paths over identical values differ by an
        # ulp (measured 1.5e-8 on ~0.05-magnitude weights). The gate that matters is
        # SEMANTIC -- no code may have changed -- so the tolerance is scaled to float32
        # round-off on this tensor's magnitude. Anything larger is real drift: a decision
        # the packer did not encode the way the solver made it.
        mag = float(Wc.float().abs().max())
        tol = max(1e-6, 64.0 * float(torch.finfo(torch.float32).eps) * max(1.0, mag))
        if diff > tol:
            raise PackError(
                f"{tensor_name}: the packed planes do NOT reconstruct the calibrated "
                f"weight (max |dequant - W| = {diff:.6e} > {tol:.3e}). That is larger "
                f"than float32 round-off, so at least one INT2 code or outlier nibble "
                f"differs from what the calibration chose. The bundle would not be the "
                f"model the calibration was measured on, so the pack is refused rather "
                f"than shipped.\n"
                f"  A common cause: the solver rounded its zero-point AFTER choosing the "
                f"codes. It must decide against the value the format stores "
                f"(wq_eval._store_zp).")

    def report(self) -> Dict[str, object]:
        unused = sorted(set(self.available) - set(self.used))
        return {"root": os.path.abspath(self.root), "modules_used": len(self.used),
                "modules_available": len(self.available), "unused": unused[:8],
                "unused_count": len(unused)}


def _quantize_one(W32: torch.Tensor, name: str, recipe: Recipe,
                  decided: Optional[Dict[str, torch.Tensor]] = None
                  ) -> Dict[str, torch.Tensor]:
    """Run Fig 6 steps (1)-(4) on one weight and shape the planes for storage.

    ``decided`` switches to the ENCODE-ONLY path: the caller (a calibrated solver) has
    already chosen the outlier lanes, the group scale/zero-point and the range map, and
    this only encodes them. The format, the plane shapes and the encoding code are
    identical either way -- see ``quantize.quantize_ommx_weight``'s ``decided`` argument
    for why re-deciding from a calibrated weight would silently discard the calibration.
    """
    N, K = int(W32.shape[0]), int(W32.shape[1])
    try:
        q = quantize_ommx_weight(
            W32, recipe.group_size, recipe.outlier_pct, npv=recipe.npv,
            outlier_repr=recipe.outlier_repr, outlier_map=recipe.outlier_map,
            zp_dtype=recipe.zp_dtype, reference=False, decided=decided)
    except OMMXQuantizeError as exc:
        raise PackError(f"{name}: {exc}") from None
    G = K // recipe.group_size
    planes: Dict[str, torch.Tensor] = {
        "code": q["code"],
        "scale_exp": q["scale_exp"],
        # Stored in the recipe's dtype. The quantizer already rounded the VALUES to
        # this dtype before choosing the INT2 codes, so this cast is exact and the
        # bundle is self-consistent (see quantize.quantize_ommx_weight ``zp_dtype``).
        "zp": q["zp"].to(recipe.zp_dtype),
    }
    if recipe.has_outliers:
        planes["oindex"] = q["oindex"].reshape(N, G, recipe.index_bytes)
        planes["ocode"] = q["ocode"].reshape(N, G, recipe.code_bytes)
    if recipe.has_map:
        planes["map_scale"] = q["map_scale"].to(recipe.map_dtype)
        planes["map_center"] = q["map_center"].to(recipe.map_dtype)
    return planes


def _check_quantizable(name: str, shape: Sequence[int], dtype: str, recipe: Recipe) -> None:
    """Refuse a tensor this recipe cannot pack, naming the property that is wrong."""
    N, K = int(shape[0]), int(shape[1])
    if dtype not in QUANTIZABLE_DTYPES:
        raise PackError(
            f"{name}: dtype {dtype} cannot be quantized (supported: "
            f"{list(QUANTIZABLE_DTYPES)}). Convert the checkpoint or add {name!r} to "
            f"the passthrough set.")
    if K % recipe.group_size:
        raise PackError(
            f"{name}: K={K} is not divisible by group_size={recipe.group_size} "
            f"(K/gs = {K / recipe.group_size:.4f}). Pick a group size that divides "
            f"every quantized K in this checkpoint.")
    if K % 4:
        raise PackError(f"{name}: K={K} must be a multiple of 4 (4 INT2 codes per byte)")
    if N <= 0:
        raise PackError(f"{name}: degenerate shape {(N, K)}")


def _stale_shard_guard(out_dir: str, n_shards: int) -> None:
    """Refuse to leave OMMX shards from an EARLIER pack sitting in the output directory.

    ``--overwrite`` re-packs into a directory that already holds a bundle, and the shard
    file name encodes the shard COUNT (``ommx_w-00001-of-00003``). Re-packing the same
    model after the source checkpoint was re-sharded therefore writes a NEW set of names
    and leaves the old set behind: the index references only the new ones, so
    ``validate_bundle`` reports OK while the directory contains two complete, disagreeing
    copies of every plane under different file names.

    That is the same hazard as a leftover ``pytorch_model.bin`` (see
    ``WEIGHT_FILE_SUFFIXES``) — a loader that globs ``*.safetensors`` in the bundle
    directory finds weight files nobody meant it to read — except arriving through the
    packer's own output instead of through the source. Checked BEFORE anything is
    written, and it refuses rather than deleting: the packer does not remove files it
    did not create in this run.
    """
    if not os.path.isdir(out_dir):
        return
    expected = {_SHARD_FMT.format(i, n_shards) for i in range(1, n_shards + 1)}
    stale = sorted(f for f in os.listdir(out_dir)
                   if f.startswith("ommx_w-") and f.endswith(".safetensors")
                   and f not in expected)
    if stale:
        raise PackError(
            f"{out_dir!r} holds OMMX shard file(s) {stale} from an earlier pack that "
            f"this one will NOT overwrite (it writes {sorted(expected)}). They would "
            f"stay in the bundle carrying a second, disagreeing copy of every plane, "
            f"unreferenced by {INDEX_FILENAME} and therefore invisible to `verify`, "
            f"while a loader that globs *.safetensors in the directory would still find "
            f"them. Delete them, or pack into a fresh output directory.")


def _existing_bundle_guard(out_dir: str, recipe: Recipe, source_model: str,
                           overwrite: bool) -> None:
    """Refuse to write into a directory that already holds a bundle.

    Two distinct failures, deliberately distinguished: a DIFFERENT recipe would
    leave a directory whose shards disagree (and whose stale shards would still be
    referenced by tensor name), while an IDENTICAL recipe is merely a re-pack the
    user should have to ask for.
    """
    index_path = os.path.join(out_dir, INDEX_FILENAME)
    if not os.path.isfile(index_path):
        return
    with open(index_path, "r", encoding="utf-8") as fh:
        old = json.load(fh)
    try:
        old_recipe = Recipe.from_json(old["recipe"])
    except (KeyError, OMMXWFormatError) as exc:
        raise PackError(
            f"{out_dir!r} already contains a {INDEX_FILENAME} this build cannot "
            f"parse ({exc}). Refusing to mix bundles; pick an empty output dir.") from None
    if old_recipe != recipe or old.get("source_model") != source_model:
        raise PackError(
            f"{out_dir!r} already holds a bundle packed with a DIFFERENT manifest "
            f"(recipe {old_recipe} / model {old.get('source_model')!r} vs "
            f"{recipe} / {source_model!r}). Its stale shards would still be listed by "
            f"tensor name. Use a fresh output directory.")
    if not overwrite:
        raise PackError(
            f"{out_dir!r} already holds a bundle with this exact manifest. Pass "
            f"--overwrite to re-pack it.")


def pack_checkpoint(src_dir: str, out_dir: str, recipe: Recipe, *,
                    source_model: Optional[str] = None,
                    quant_modules: Sequence[str] = DEFAULT_QUANT_MODULES,
                    dry_run: bool = False, overwrite: bool = False,
                    copy_aux: bool = True, stamp_quant_config: bool = True,
                    calibrated: Optional[str] = None,
                    verbose: bool = True) -> dict:
    """Pack an HF safetensors checkpoint directory into an OMMX_W_SafeTensor bundle.

    The output is a SELF-CONTAINED MODEL DIRECTORY: the ommx_w shards, the index, and
    every non-weight file of the source checkpoint (config.json, generation_config,
    tokenizer files, chat template, anything else). ``copy_aux=False`` opts out and
    records the opt-out in the index — see :func:`plan_aux_files`.

    Returns a report dict (also what ``--dry-run`` prints): per-tensor rows with
    packed byte size and bits/weight, the aux-file plan, plus bundle totals. With
    ``dry_run=True`` NO file is written and no weight is quantized — only the layout
    is computed, which is the point: seeing the byte budget must not cost an hour of
    packing. The aux-file plan IS evaluated during a dry run (it reads a directory
    listing, nothing more), so a missing ``config.json`` is reported in the second it
    takes rather than after the packing.
    """
    shards = discover_shards(src_dir)
    source_model = source_model or os.path.abspath(src_dir)
    if os.path.abspath(src_dir) == os.path.abspath(out_dir):
        raise PackError(
            f"--input and --output are the same directory ({os.path.abspath(src_dir)}). "
            f"The packer copies the source's non-weight files INTO the output, so this "
            f"would mean reading a checkpoint while overwriting it. Use a fresh "
            f"output directory.")
    # Built BEFORE anything is written: a missing/!malformed calibration directory must
    # stop the pack at argument-validation time, not halfway through the shards.
    calib = CalibrationSource(calibrated) if calibrated else None
    aux = plan_aux_files(src_dir, copy_aux=copy_aux)
    for warning in aux["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if not dry_run:
        _existing_bundle_guard(out_dir, recipe, source_model, overwrite)
        _stale_shard_guard(out_dir, len(shards))
        os.makedirs(out_dir, exist_ok=True)

    rows: List[dict] = []
    weight_map: Dict[str, str] = {}
    t0 = time.time()

    for si, src in enumerate(shards, start=1):
        st = SafeTensorsFile(src)
        shard_file = _SHARD_FMT.format(si, len(shards))
        entries: Dict[str, dict] = {}
        writer = None if dry_run else SafeTensorsWriter(os.path.join(out_dir, shard_file))
        try:
            for name in st.names:
                shape = st.shape(name)
                dtype = st.dtype_name(name)
                kind, reason = classify_tensor(name, shape, dtype, quant_modules)
                if name in weight_map:
                    raise PackError(
                        f"tensor {name!r} appears in more than one input shard "
                        f"({weight_map[name]} and {os.path.basename(src)})")
                if kind == "quantized":
                    _check_quantizable(name, shape, dtype, recipe)
                    N, K = int(shape[0]), int(shape[1])
                    layout = recipe.plane_layout(N, K)
                    if dry_run:
                        nbytes = recipe.packed_bytes(N, K)
                    else:
                        # bf16/f16 -> f32 is exact; f32 is a no-op. The quantizer
                        # requires f32 rather than silently widening, so the cast is
                        # explicit and happens exactly here.
                        W32 = st.tensor(name).to(torch.float32)
                        dec = None
                        if calib is not None:
                            W32, dec = calib.decisions_for(name, W32)
                        planes = _quantize_one(W32, name, recipe, decided=dec)
                        if dec is not None:
                            calib.verify(name, planes, W32, recipe)
                        nbytes = 0
                        pdesc = {}
                        for plane in recipe.planes():
                            pn = plane_name(name, plane)
                            nbytes += writer.add(pn, planes[plane])
                            pdesc[plane] = {"name": pn,
                                            "shape": list(planes[plane].shape),
                                            "dtype": st_dtype_name(planes[plane].dtype)}
                        entries[name] = {
                            "kind": "quantized", "reason": reason,
                            "dtype": dtype, "shape": [N, K],
                            "n_groups": K // recipe.group_size,
                            "planes": pdesc,
                            "packed_bytes": nbytes,
                            "bits_per_weight": bits_per_weight(recipe, N, K, nbytes),
                        }
                    rows.append({
                        "name": name, "kind": "quantized", "shape": [N, K],
                        "src_dtype": dtype, "src_bytes": st.nbytes(name),
                        "packed_bytes": nbytes,
                        "bits_per_weight": bits_per_weight(recipe, N, K, nbytes),
                        "planes": {p: list(layout[p][0]) for p in layout},
                        "shard": shard_file, "reason": reason,
                    })
                else:
                    nbytes = st.nbytes(name)
                    if not dry_run:
                        nbytes = writer.add(name, st.tensor(name))
                        entries[name] = {"kind": "passthrough", "reason": reason,
                                         "dtype": dtype, "shape": list(shape),
                                         "packed_bytes": nbytes}
                    rows.append({
                        "name": name, "kind": "passthrough", "shape": list(shape),
                        "src_dtype": dtype, "src_bytes": st.nbytes(name),
                        "packed_bytes": nbytes, "bits_per_weight": None,
                        "shard": shard_file, "reason": reason,
                    })
                weight_map[name] = shard_file
            if not dry_run:
                manifest = {
                    "format": OMMX_W_FORMAT,
                    "version": OMMX_W_FORMAT_VERSION,
                    "producer": PRODUCER,
                    "source_model": source_model,
                    "shard": {"index": si, "count": len(shards), "file": shard_file},
                    # Replicated into EVERY shard, not just the index: a shard is a
                    # valid safetensors file on its own and gets read on its own.
                    "naming": naming_document(),
                    "recipe": recipe.to_json(),
                    "tensors": entries,
                }
                writer.close({"format": OMMX_W_FORMAT,
                              "version": str(OMMX_W_FORMAT_VERSION),
                              "manifest": json.dumps(manifest, separators=(",", ":"))})
        except BaseException:
            if writer is not None:
                writer.abort()
            raise

    report = _summarise(rows, recipe, source_model, shards, time.time() - t0, dry_run)
    report["aux_files"] = aux
    if calib is not None:
        report["calibration"] = calib.report()
    if not dry_run:
        # Copy the model files BEFORE writing the index, so the index's aux record can
        # carry the real byte counts and digests of files that are already on disk. If
        # a copy fails, no index is written at all and the directory cannot be mistaken
        # for a finished bundle.
        aux = copy_aux_files(src_dir, out_dir, aux)
        # Make the bundle DISCOVERABLE. Without this key vLLM never calls
        # OMMXWConfig.from_config, so `--quantization ommx_w` is rejected by name and the
        # weight-quant path is unreachable from the documented invocation. Runs after the
        # copy (config.json must exist) and before the index (so the recorded digest
        # describes the stamped file).
        if stamp_quant_config:
            aux = stamp_quantization_config(out_dir, recipe, aux)
        report["aux_files"] = aux
        index = {
            "format": OMMX_W_FORMAT,
            "version": OMMX_W_FORMAT_VERSION,
            "producer": PRODUCER,
            "source_model": source_model,
            "naming": naming_document(),
            "recipe": recipe.to_json(),
            "shards": sorted(set(weight_map.values())),
            "weight_map": weight_map,
            # ``warnings`` is advice to the operator at pack time, not a property of
            # the bundle; the index records the FACTS (policy, what was copied,
            # skipped and absent) so a reader can audit the directory it was handed.
            # A calibrated bundle and an RTN bundle are byte-compatible, so the
            # index has to record WHICH one this is -- otherwise the only
            # difference between a model that works and one that does not is
            # undiscoverable after the fact.
            "calibration": (calib.report() if calib is not None else None),
            "aux_files": {k: v for k, v in aux.items() if k != "warnings"},
            "totals": report["totals"],
        }
        with open(os.path.join(out_dir, INDEX_FILENAME), "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2, sort_keys=True)
        # Validate what we just wrote, always. A packer that can emit a bundle its
        # own validator rejects is worse than one that cannot pack at all.
        validate_bundle(out_dir)
    if verbose:
        print(format_report(report))
    return report


def _summarise(rows: List[dict], recipe: Recipe, source_model: str,
               shards: Sequence[str], elapsed: float, dry_run: bool) -> dict:
    q = [r for r in rows if r["kind"] == "quantized"]
    p = [r for r in rows if r["kind"] == "passthrough"]
    q_elems = sum(r["shape"][0] * r["shape"][1] for r in q)
    q_bytes = sum(r["packed_bytes"] for r in q)
    q_src = sum(r["src_bytes"] for r in q)
    p_bytes = sum(r["packed_bytes"] for r in p)
    totals = {
        "n_quantized": len(q), "n_passthrough": len(p),
        "quantized_elements": q_elems,
        "quantized_packed_bytes": q_bytes,
        "quantized_source_bytes": q_src,
        "passthrough_bytes": p_bytes,
        "bundle_bytes": q_bytes + p_bytes,
        "bits_per_weight": (8.0 * q_bytes / q_elems) if q_elems else 0.0,
        "compression_vs_source": (q_src / q_bytes) if q_bytes else 0.0,
    }
    return {"dry_run": dry_run, "source_model": source_model,
            "input_shards": [os.path.basename(s) for s in shards],
            "recipe": recipe.to_json(), "bits_breakdown": recipe.bits_breakdown(),
            "fig6_steps": list(FIG6_STEPS), "rows": rows, "totals": totals,
            "elapsed_s": elapsed}


def format_report(report: dict) -> str:
    """Human-readable per-tensor byte/bit budget — the ``--dry-run`` output."""
    r = report["recipe"]
    bb = report["bits_breakdown"]
    out: List[str] = []
    out.append("=" * 96)
    out.append(f"OMMX W-Packer{'  [DRY RUN — nothing written]' if report['dry_run'] else ''}")
    out.append(f"  source model : {report['source_model']}")
    out.append(f"  input shards : {len(report['input_shards'])}")
    out.append(f"  recipe       : {r['bundle_format']} gs={r['group_size']} npv={r['npv']} "
               f"({100.0 * r['npv'] / r['group_size']:.2f}% outliers) "
               f"repr={r['outlier_repr']} map={r['outlier_map']} zp={r['zp_dtype']}")
    out.append("  Fig 6 steps  : " + " -> ".join(report["fig6_steps"]))
    out.append("-" * 96)
    out.append(f"{'tensor':<58}{'kind':<13}{'bytes':>12}{'bits/wt':>11}")
    for row in report["rows"]:
        bpw = "" if row["bits_per_weight"] is None else f"{row['bits_per_weight']:.4f}"
        nm = row["name"] if len(row["name"]) <= 57 else "..." + row["name"][-54:]
        out.append(f"{nm:<58}{row['kind']:<13}{row['packed_bytes']:>12,}{bpw:>11}")
    t = report["totals"]
    out.append("-" * 96)
    out.append(f"  quantized     : {t['n_quantized']} tensors, "
               f"{t['quantized_elements']:,} weights, {t['quantized_packed_bytes']:,} B")
    out.append(f"  passthrough   : {t['n_passthrough']} tensors, {t['passthrough_bytes']:,} B")
    out.append(f"  bundle total  : {t['bundle_bytes']:,} B")
    out.append(f"  bits / weight : {t['bits_per_weight']:.4f}  "
               f"(recipe model {bb['bits_per_weight']:.4f} stored / "
               f"{bb['bits_per_weight_unpadded']:.4f} unpadded)")
    out.append(f"  vs source     : {t['compression_vs_source']:.3f}x smaller on the "
               f"quantized tensors alone")
    out.append("  plane budget (bits per weight, stored):")
    for k, v in bb["stored"].items():
        if k != "total":
            out.append(f"      {k:<16} {v:8.4f}")
    aux = report.get("aux_files")
    if aux:
        # Printed every run, not only on trouble: "which model files did this bundle
        # end up with" is the question the packer used to leave unanswerable.
        out.append("-" * 96)
        out.append(f"  model files   : policy={aux['policy']}  self_contained="
                   f"{aux['self_contained']}")
        copied = [e["name"] for e in aux["copied"]]
        out.append("    copied      : " + (", ".join(copied) if copied else "(none)"))
        if aux.get("missing"):
            out.append("    NOT present : "
                       + ", ".join(e["name"] for e in aux["missing"]))
        if aux.get("skipped"):
            out.append(f"    skipped     : {len(aux['skipped'])} file(s) "
                       f"(weights / indexes / dirs; see ommx_w_index.json)")
        if not aux["self_contained"]:
            out.append("    NOTE: this bundle is NOT a loadable model directory.")
    out.append("=" * 96)
    return "\n".join(out)


# ════════════════════════════════════════════════════════════════════════════
# synthetic checkpoint (this host has no model and no network)
# ════════════════════════════════════════════════════════════════════════════

#: The non-weight files :func:`make_synthetic_checkpoint` writes. A real HF checkpoint
#: has these; the synthetic one must too, or the packer's aux-file copy-through would
#: be asserted in prose and exercised by nothing. ``config.json`` is filled in from the
#: caller's shapes so it actually describes the tensors next to it.
SYNTHETIC_AUX_FILES: Tuple[str, ...] = (
    "config.json", "generation_config.json", "tokenizer_config.json",
    "special_tokens_map.json", "tokenizer.json", "chat_template.jinja",
)


def _synthetic_aux_documents(layers: int, hidden: int, inter: int, vocab: int,
                             kv: int, dtype: torch.dtype) -> Dict[str, str]:
    """Text of every file in :data:`SYNTHETIC_AUX_FILES`, consistent with the tensors.

    Deliberately consistent (``hidden_size`` really is the generated hidden size, and
    so on) rather than a placeholder blob: the point of the fixture is that a packed
    bundle is a directory a loader would accept, and a config that contradicts the
    shards would not be one. It is still SYNTHETIC — random weights, a two-token
    tokenizer — and says so in ``_comment``.
    """
    cfg = {
        "_comment": "synthetic fixture generated by ommx_gpu_serve.linear.w_packer; "
                    "random weights, not a trained model",
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_hidden_layers": layers,
        "num_attention_heads": max(1, hidden // 8),
        "num_key_value_heads": max(1, kv // 8),
        "vocab_size": vocab,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "torch_dtype": str(dtype).replace("torch.", ""),
        "bos_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": False,
    }
    gen = {"bos_token_id": 0, "eos_token_id": 1, "do_sample": False,
           "max_new_tokens": 16}
    tok_cfg = {"model_max_length": 128, "tokenizer_class": "PreTrainedTokenizerFast",
               "bos_token": "<s>", "eos_token": "</s>",
               "chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}"}
    special = {"bos_token": "<s>", "eos_token": "</s>"}
    tok = {"version": "1.0", "model": {"type": "WordLevel",
                                       "vocab": {"<s>": 0, "</s>": 1}, "unk_token": None},
           "added_tokens": []}
    return {
        "config.json": json.dumps(cfg, indent=2, sort_keys=True),
        "generation_config.json": json.dumps(gen, indent=2, sort_keys=True),
        "tokenizer_config.json": json.dumps(tok_cfg, indent=2, sort_keys=True),
        "special_tokens_map.json": json.dumps(special, indent=2, sort_keys=True),
        "tokenizer.json": json.dumps(tok, indent=2, sort_keys=True),
        "chat_template.jinja": "{% for m in messages %}{{ m['content'] }}{% endfor %}\n",
    }


def make_synthetic_checkpoint(out_dir: str, *, layers: int = 2, hidden: int = 128,
                              inter: int = 256, vocab: int = 512, kv_heads_ratio: int = 2,
                              shards: int = 2, dtype: torch.dtype = torch.bfloat16,
                              seed: int = 0, aux: bool = True) -> List[str]:
    """Write a tiny Llama-shaped safetensors checkpoint. Used by the tests and the CLI.

    Not a nicety: the machine this packer was written on has no model checkpoint and
    no network, so the ONLY way the CLI could be exercised end to end is against a
    checkpoint it generates itself. It carries every tensor class the classifier has
    to get right — projections, an embedding table, an ``lm_head``, norms and a bias —
    AND, since the packer now copies model files through, the non-weight files a real
    checkpoint has (:data:`SYNTHETIC_AUX_FILES`).

    ``aux=False`` writes the shards alone. That is not a convenience: it is how a test
    reaches the packer's "source has no config.json" refusal without hand-deleting a
    file, so the refusal is exercised by the same fixture that exercises the success.

    Returns the SHARD paths (aux files are not weights and are not in the list).
    """
    torch.manual_seed(seed)
    kv = max(1, hidden // kv_heads_ratio)
    tensors: List[Tuple[str, torch.Tensor]] = [
        ("model.embed_tokens.weight", torch.randn(vocab, hidden) * 0.02),
    ]
    for i in range(layers):
        p = f"model.layers.{i}"
        tensors += [
            (f"{p}.self_attn.q_proj.weight", torch.randn(hidden, hidden) * 0.1),
            (f"{p}.self_attn.q_proj.bias", torch.randn(hidden) * 0.01),
            (f"{p}.self_attn.k_proj.weight", torch.randn(kv, hidden) * 0.1),
            (f"{p}.self_attn.v_proj.weight", torch.randn(kv, hidden) * 0.1),
            (f"{p}.self_attn.o_proj.weight", torch.randn(hidden, hidden) * 0.1),
            (f"{p}.mlp.gate_proj.weight", torch.randn(inter, hidden) * 0.1),
            (f"{p}.mlp.up_proj.weight", torch.randn(inter, hidden) * 0.1),
            (f"{p}.mlp.down_proj.weight", torch.randn(hidden, inter) * 0.1),
            (f"{p}.input_layernorm.weight", torch.ones(hidden)),
            (f"{p}.post_attention_layernorm.weight", torch.ones(hidden)),
        ]
    tensors += [("model.norm.weight", torch.ones(hidden)),
                ("lm_head.weight", torch.randn(vocab, hidden) * 0.02)]

    os.makedirs(out_dir, exist_ok=True)
    shards = max(1, min(int(shards), len(tensors)))
    per = (len(tensors) + shards - 1) // shards
    written: List[str] = []
    weight_map: Dict[str, str] = {}
    for si in range(shards):
        chunk = tensors[si * per:(si + 1) * per]
        if not chunk:
            continue
        fn = f"model-{si + 1:05d}-of-{shards:05d}.safetensors"
        w = SafeTensorsWriter(os.path.join(out_dir, fn))
        for name, t in chunk:
            w.add(name, t.to(dtype))
            weight_map[name] = fn
        w.close({"format": "pt"})
        written.append(os.path.join(out_dir, fn))
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"metadata": {"total_size": sum(os.path.getsize(f) for f in written)},
                   "weight_map": weight_map}, fh, indent=2, sort_keys=True)
    if aux:
        for name, text in _synthetic_aux_documents(
                layers, hidden, inter, vocab, kv, dtype).items():
            with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
                fh.write(text)
    return written


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

_ZP_ALIASES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
               "f32": torch.float32, "fp32": torch.float32, "float32": torch.float32}
_SRC_ALIASES = dict(_ZP_ALIASES, f16=torch.float16, fp16=torch.float16,
                    float16=torch.float16)


def build_recipe(group_size: int, outlier_pct: Optional[float], npv: Optional[int],
                 outlier_repr: str, outlier_map: str, zp_dtype: str) -> Recipe:
    """Turn CLI knobs into a :class:`Recipe`, refusing an ambiguous outlier budget."""
    if npv is not None and outlier_pct is not None:
        raise PackError(
            f"--npv {npv} and --outlier-pct {outlier_pct} are both set and they "
            f"disagree unless npv == max(1, int(gs*pct)); pass exactly one.")
    if npv is None:
        pct = 0.0625 if outlier_pct is None else float(outlier_pct)
        npv = derive_npv(group_size, pct)
    else:
        npv = int(npv)
        pct = npv / float(group_size)
    return Recipe(group_size=group_size, npv=npv, outlier_pct=pct,
                  outlier_repr=outlier_repr, outlier_map=outlier_map,
                  zp_dtype=_ZP_ALIASES[zp_dtype])


# ── named presets (--preset) ─────────────────────────────────────────────────────
#
# The `pack` knobs a weight preset may supply. Every one of them keeps its ordinary
# argparse default; the preset only fills the ones the operator did NOT spell out.
_PRESET_DESTS: Tuple[str, ...] = ("group_size", "outlier_pct", "npv", "outlier_repr",
                                  "outlier_map", "zp_dtype")


class _Unset:
    """Sentinel distinguishing "operator typed this flag" from "argparse defaulted it".

    argparse offers no other way to tell the two apart, and the difference IS the
    precedence rule: an explicit flag must beat the preset, a defaulted one must not.
    (Pre-seeding the namespace does not work here — ``_SubParsersAction.__call__``
    parses the subcommand into a FRESH namespace and copies the result over the one you
    handed in, so every seeded sentinel is overwritten by the subparser's own defaults.)
    """

    __slots__ = ()

    def __repr__(self) -> str:                      # readable in a traceback
        return "<unset>"


_UNSET = _Unset()


class _unset_defaults:
    """Context manager: temporarily make ``parser``'s preset-overridable defaults
    ``_UNSET``, exposing exactly which flags the operator typed.

    The REAL defaults are read back off the parser (``get_default``) rather than
    duplicated in this module, so ``pack``'s ``add_argument(default=...)`` literals stay
    the single source of truth and a future default change is picked up here for free
    (and is caught by tests/test_recipes.py, which asserts shipped-weight still equals
    them). Restored in ``finally`` so a SystemExit from a usage error cannot leave the
    parser poisoned for the next ``main()`` call in the same process.
    """

    def __init__(self, parser: argparse.ArgumentParser, dests: Sequence[str]) -> None:
        self.parser, self.dests = parser, tuple(dests)
        self.saved: Dict[str, object] = {}

    def __enter__(self) -> "Dict[str, object]":
        self.saved = {d: self.parser.get_default(d) for d in self.dests}
        self.parser.set_defaults(**{d: _UNSET for d in self.dests})
        return self.saved

    def __exit__(self, *exc: object) -> None:
        self.parser.set_defaults(**self.saved)


def env_preset(default: Optional[str] = None) -> Optional[str]:
    """``--preset`` if the operator typed one, else the WEIGHT recipe ``OMMX_RECIPE``
    names — the weight axis's half of the single ``OMMX_RECIPE`` resolution model.

    WHY THE ENV AT ALL on an argv-driven tool: ``OMMX_RECIPE`` is the one flag that
    selects a named, measured recipe, and a selector that works on the KV axis and is
    inert on the weight axis is the same "silently ignored preset" defect wearing a
    different hat. Precedence mirrors the KV side exactly — EXPLICIT beats preset — so
    ``--preset`` wins here just as an exported ``OMMX_KV_*`` wins over a KV preset.

    Cross-axis and unknown names are handled once, in
    :func:`recipes.env_preset_name`: a KV name prints a note to STDERR (never stdout —
    ``budget``'s stdout is a table a caller may parse) and leaves the packer defaults
    alone; an unknown name RAISES with the known names listed, which is what makes the
    "unknown preset raises from EVERY entry point" rule true of the packer too.
    """
    if default:
        return default
    return env_preset_name("weight",
                           note=lambda msg: print(f"# {msg}", file=sys.stderr))


def resolve_preset_knobs(args: argparse.Namespace,
                         defaults: Mapping[str, object]) -> Optional[str]:
    """Fill every un-typed ``pack`` knob from ``--preset`` / ``OMMX_RECIPE``, else from
    ``defaults``.

    Returns the applied preset name (or None). Precedence, deliberately identical to
    the KV side's ``OMMX_RECIPE``: an EXPLICIT flag always wins, so an operator can
    take a preset and move one knob without hand-assembling the other five.
    """
    preset = env_preset(getattr(args, "preset", None))
    overlay: Dict[str, object] = {}
    if preset:
        # Raises UnknownRecipeError (listing the weight recipes) on a typo or on a KV
        # recipe name — never a silent fall-through to the defaults, which would pack a
        # bundle at a budget the operator did not ask for and could not tell apart.
        overlay = dict(get_recipe(preset, axis="weight").packer_args)
    explicit = {d for d in _PRESET_DESTS if getattr(args, d, _UNSET) is not _UNSET}
    # ``build_recipe`` REFUSES --npv and --outlier-pct together. A preset carries one of
    # them; if the operator typed the other, the preset's must step aside or the
    # override would come back as a confusing "both are set" PackError.
    if "npv" in explicit:
        overlay.pop("outlier_pct", None)
    if "outlier_pct" in explicit:
        overlay.pop("npv", None)
    for dest in _PRESET_DESTS:
        if dest in explicit:
            continue
        setattr(args, dest, overlay[dest] if dest in overlay else defaults[dest])
    return preset or None


def preset_budget_line(name: str) -> str:
    """``budget``-shaped row for one named preset, recomputed from the live layout.

    Printed next to the registry's recorded number so a reader can see the two agree
    (or, if the layout ever moves, that they do not) without running pytest.
    """
    rec = get_recipe(name, axis="weight")
    r = build_recipe(**dict(rec.packer_args))
    bb = r.bits_breakdown()
    return (f"{r.group_size:>4}{r.npv:>5}{100.0 * r.npv / r.group_size:>6.1f}%  "
            f"{r.outlier_repr:<9}{r.outlier_map:<10}"
            f"{bb['bits_per_weight']:>9.4f}{bb['bits_per_weight_unpadded']:>10.4f}")


def budget_table(group_sizes: Sequence[int] = (16, 32, 64, 128),
                 npvs: Sequence[int] = (1, 2, 4, 8, 16, 32),
                 reprs: Sequence[str] = OUTLIER_REPRS,
                 maps: Sequence[str] = OUTLIER_MAPS) -> str:
    """Bits/weight for every recipe, with no checkpoint — the pre-packing budget.

    Answers "which recipe lands on which bit budget" directly from the layout, which
    is the only defensible way to compare against a published AvgBits figure.

    Three columns because a published figure can be quoting any of three things and the
    difference is larger than the figure: ``stored`` is what this packer writes and the
    only memory-traffic statement of the three; ``unpadded`` removes the per-group byte
    padding; ``npu`` re-encodes the positions the way the paper says the Nucleus NPU
    does (``ceil(log2 C(gs, npv))`` per group) and is therefore the column an NPU AvgBits
    claim should be read against. ``npu`` does not vary with ``repr`` -- relidx7 and
    bitmap are two GPU spellings of one position set.

    The grid includes gs=16 and npv 1/32 because the recipes that matter empirically sit
    at its corners: gs=16/npv=4 is the first weight configuration measured to preserve
    the model, and npv=32 is the kernel's MAX_NPV.
    """
    lines = [f"{'gs':>4}{'npv':>5}{'%out':>7}  {'repr':<9}{'map':<10}"
             f"{'stored':>9}{'unpadded':>10}{'npu':>9}"]
    for gs in group_sizes:
        for npv in npvs:
            if npv > gs:
                continue
            for rp in reprs:
                if rp == "relidx7" and gs > 128:
                    continue
                for mp in maps:
                    r = Recipe(gs, npv, npv / gs, rp, mp)
                    bb = r.bits_breakdown()
                    lines.append(
                        f"{gs:>4}{npv:>5}{100.0 * npv / gs:>6.1f}%  {rp:<9}{mp:<10}"
                        f"{bb['bits_per_weight']:>9.4f}"
                        f"{bb['bits_per_weight_unpadded']:>10.4f}"
                        f"{bb['bits_per_weight_npu']:>9.4f}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ommx_gpu_serve.linear.w_packer",
        description="OMMX offline weight packer (HF safetensors -> OMMX_W_SafeTensor)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pk = sub.add_parser("pack", help="pack a checkpoint directory")
    pk.add_argument("--input", required=True, help="HF safetensors checkpoint dir")
    pk.add_argument("--output", required=True, help="OMMX_W_SafeTensor bundle dir")
    # NAMED PRESET. Reproducing the paper's weight bit budget should be a flag, not a
    # guess at five knobs: `--preset paper-weight` is gs 64 / npv 4 / bitmap positions /
    # no range map = a MEASURED 3.6250 bits/weight (claim E2). Any flag typed
    # explicitly still wins over the preset. See ommx_gpu_serve/recipes.py.
    pk.add_argument("--preset", default=None, metavar="NAME",
                    help="named weight recipe: "
                         + ", ".join(recipe_names("weight"))
                         + ". Fills only the knobs you do not pass explicitly; see "
                           "`w_packer recipes`.")
    pk.add_argument("--group-size", type=int, default=64)
    pk.add_argument("--outlier-pct", type=float, default=None,
                    help="outlier fraction per group (default 0.0625 = npv 4 at gs 64)")
    pk.add_argument("--npv", type=int, default=None,
                    help="outliers per group; mutually exclusive with --outlier-pct")
    pk.add_argument("--outlier-repr", choices=OUTLIER_REPRS, default="relidx7")
    pk.add_argument("--outlier-map", choices=OUTLIER_MAPS, default="idx_range")
    pk.add_argument("--zp-dtype", choices=sorted(_ZP_ALIASES), default="bf16")
    pk.add_argument("--source-model", default=None, help="model id recorded in the manifest")
    pk.add_argument("--dry-run", action="store_true",
                    help="report per-tensor packed bytes and bits/weight; write nothing")
    pk.add_argument("--overwrite", action="store_true")
    # WHY AN ESCAPE HATCH AT ALL: three real cases where copying is wrong, and all
    # three are better served by an explicit flag than by the packer guessing.
    #  (1) the source directory holds files that must not be redistributed with the
    #      bundle (private calibration data, a licence-restricted tokenizer);
    #  (2) the operator is packing a shard subset for a kernel test and will point the
    #      engine at the original checkpoint for config/tokenizer;
    #  (3) the config must be REPLACED, not copied (a re-rope'd context length), and
    #      copying first only to overwrite is a trap when the copy is the newer file.
    # It is safe to offer because it cannot produce a bundle that LOOKS self-contained:
    # the index records policy="no-copy-aux" and self_contained=false, `verify` prints
    # it, and the validator refuses a bundle that claims otherwise.
    pk.add_argument("--no-copy-aux", action="store_true",
                    help="do NOT copy the source's non-weight model files (config, "
                         "tokenizer, ...) into the bundle. The result is a weights-only "
                         "directory that no engine can load on its own; the index "
                         "records the opt-out.")
    # Stamping is ON by default because a bundle without it is not discoverable: vLLM
    # only calls OMMXWConfig.from_config when config.json carries a quantization_config,
    # so an unstamped bundle makes `--quantization ommx_w` fail by name. The opt-out
    # exists because the same key makes `transformers` try to resolve a quantizer it does
    # not have -- so a bundle meant to be opened by transformers wants it off.
    pk.add_argument("--calibrated", default=None, metavar="DIR",
                    help="pack from a calibrated solver's DECISIONS in DIR "
                         "(<module>.safetensors with omask/scale/zp/map_scale/map_center "
                         "+ the calibrated W). Without this the packer chooses its INT2 "
                         "codes by plain min/max round-to-nearest, which is MEASURED to "
                         "destroy Llama-3.1-8B at the shipped recipe. Every quantized "
                         "tensor must have a file; a partial directory is refused.")
    pk.add_argument("--no-quant-config", action="store_true",
                    help="do NOT write quantization_config into the bundle's config.json. "
                         "vLLM will then NOT auto-discover the ommx_w method and "
                         "--quantization ommx_w is rejected as an unknown method; use "
                         "this only for a bundle you intend to open with transformers.")
    pk.add_argument("--json", default=None, help="also write the report as JSON here")

    ms = sub.add_parser("make-synthetic", help="write a tiny Llama-shaped checkpoint")
    ms.add_argument("--output", required=True)
    ms.add_argument("--layers", type=int, default=2)
    ms.add_argument("--hidden", type=int, default=128)
    ms.add_argument("--inter", type=int, default=256)
    ms.add_argument("--vocab", type=int, default=512)
    ms.add_argument("--shards", type=int, default=2)
    ms.add_argument("--dtype", choices=sorted(_SRC_ALIASES), default="bf16")
    ms.add_argument("--no-aux", action="store_true",
                    help="omit config.json/tokenizer/... — produces the checkpoint "
                         "shape that `pack` REFUSES, for exercising that refusal")

    vf = sub.add_parser("verify", help="validate a bundle (manifest vs tensors)")
    vf.add_argument("--bundle", required=True)

    bg = sub.add_parser("budget", help="bits/weight per recipe, no checkpoint needed")
    bg.add_argument("--group-sizes", type=int, nargs="+", default=[32, 64, 128])
    bg.add_argument("--npvs", type=int, nargs="+", default=[2, 4, 8, 16])
    bg.add_argument("--preset", default=None, metavar="NAME",
                    help="instead of the grid, print this NAMED recipe in full plus its "
                         "budget row recomputed from the live layout: "
                         + ", ".join(recipe_names("weight")))

    rc = sub.add_parser("recipes", help="the named-recipe registry (KV + weight)")
    rc.add_argument("--name", default=None,
                    help="print one recipe in full instead of the table")

    # Parse with `pack`'s preset-overridable defaults swapped for _UNSET, so anything
    # still holding _UNSET afterwards was NOT typed by the operator and may be filled by
    # --preset. `pack_defaults` carries the real values back out for the no-preset case,
    # which therefore resolves to byte-identical arguments to before this flag existed.
    with _unset_defaults(pk, _PRESET_DESTS) as pack_defaults:
        args = ap.parse_args(argv)
    try:
        if args.cmd == "recipes":
            print(format_recipe(get_recipe(args.name)) if args.name else format_table())
            return 0
        if args.cmd == "pack":
            applied = resolve_preset_knobs(args, pack_defaults)
            if applied:
                # Evidence: the preset name and the knobs it resolved to are printed
                # BEFORE any work, so the packing log records which recipe produced the
                # bundle. (The bundle manifest itself records only the resolved Recipe;
                # carrying the preset NAME into ommx_w_index.json would need a change in
                # w_format.py, which is out of this change's scope.)
                print(f"preset        {applied}  ->  --group-size {args.group_size} "
                      f"--npv {args.npv} --outlier-pct {args.outlier_pct} "
                      f"--outlier-repr {args.outlier_repr} "
                      f"--outlier-map {args.outlier_map} --zp-dtype {args.zp_dtype}")
            recipe = build_recipe(args.group_size, args.outlier_pct, args.npv,
                                  args.outlier_repr, args.outlier_map, args.zp_dtype)
            report = pack_checkpoint(args.input, args.output, recipe,
                                     source_model=args.source_model,
                                     dry_run=args.dry_run, overwrite=args.overwrite,
                                     copy_aux=not args.no_copy_aux,
                                     stamp_quant_config=not args.no_quant_config,
                                     calibrated=args.calibrated)
            if args.json:
                with open(args.json, "w", encoding="utf-8") as fh:
                    json.dump(report, fh, indent=2)
            return 0
        if args.cmd == "make-synthetic":
            files = make_synthetic_checkpoint(
                args.output, layers=args.layers, hidden=args.hidden, inter=args.inter,
                vocab=args.vocab, shards=args.shards, dtype=_SRC_ALIASES[args.dtype],
                aux=not args.no_aux)
            extra = "" if args.no_aux else (
                " + " + ", ".join(SYNTHETIC_AUX_FILES))
            print(f"wrote {len(files)} shard(s) to {args.output}{extra}")
            return 0
        if args.cmd == "verify":
            index = validate_bundle(args.bundle)
            print(f"OK  {args.bundle}: {len(index['weight_map'])} tensors across "
                  f"{len(index['shards'])} shard(s), recipe "
                  f"{Recipe.from_json(index['recipe'])}")
            aux = index["aux_files"]
            copied = [e["name"] for e in aux["copied"]]
            print(f"    model files: policy={aux['policy']} "
                  f"self_contained={aux['self_contained']} "
                  f"[{', '.join(copied) if copied else 'none'}]")
            if not aux["self_contained"]:
                print("    NOTE: not a loadable model directory on its own — supply "
                      "config/tokenizer explicitly when serving it.")
            # Digest drift is INFORMATION, not a failure: editing config.json inside a
            # bundle is legitimate. Reported so an unexplained edit is still visible.
            drift = [e["name"] for e in aux["copied"]
                     if "sha256" in e
                     and sha256_file(os.path.join(args.bundle, e["name"]))
                     != e["sha256"]]
            if drift:
                print(f"    changed since packing (allowed, reported): "
                      f"{', '.join(drift)}")
            return 0
        if args.cmd == "budget":
            # Same resolution as `pack`: --preset first, then a WEIGHT OMMX_RECIPE.
            # Without this, `OMMX_RECIPE=paper-weight w_packer budget` printed the
            # generic sweep table and the operator had to notice that their preset had
            # quietly done nothing.
            budget_preset = env_preset(args.preset)
            if budget_preset:
                rec = get_recipe(budget_preset, axis="weight")
                print(format_recipe(rec))
                print("")
                print(f"{'gs':>4}{'npv':>5}{'%out':>7}  {'repr':<9}{'map':<10}"
                      f"{'stored':>9}{'unpadded':>10}")
                print(preset_budget_line(rec.name))
                return 0
            print(budget_table(args.group_sizes, args.npvs))
            return 0
    except UnknownRecipeError as exc:
        # Same shape as the format/pack errors below: a named message on stderr and a
        # non-zero exit, never a silent fall-through to the default recipe.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (PackError, OMMXWFormatError, OMMXQuantizeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
