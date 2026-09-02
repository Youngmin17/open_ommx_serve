# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""OMMX weight-quant OFFLINE path — quantizer, on-disk format, and packer.

This package is the offline half of ``OMMX_Linear`` (paper §3.3 / Fig 6): it turns a
Hugging Face safetensors checkpoint into an ``OMMX_W_SafeTensor`` bundle that the
i2f4 linear kernels under ``csrc/linear/`` are designed to consume.

  * :mod:`~ommx_gpu_serve.linear.quantize`  — the OMMX weight recipe as an importable
    module, bit-exact to the shipped parity gate
    ``csrc/linear/test_ommx_linear_parity.py::quantize_ommx_weight``.
  * :mod:`~ommx_gpu_serve.linear.w_format`  — the ``OMMX_W_SafeTensor`` spec: plane
    naming, JSON manifest, streaming safetensors reader/writer, validator.
  * :mod:`~ommx_gpu_serve.linear.w_packer`  — the offline packer + CLI
    (``python -m ommx_gpu_serve.linear.w_packer``).

SCOPE, stated precisely. This package is the OFFLINE half: nothing here loads a
kernel and nothing here talks to vLLM. The ONLINE half now exists in a sibling
package — ``integration/vllm/linear_method.py`` defines an ``ommx_w``
``LinearMethodBase`` and ``integration/vllm/plugin.py`` registers it on plugin load —
so "producing a bundle is all this repo can do with weights" is no longer true of the
tree. What that means for a reader here:

  * CPU-VERIFIED: everything this package produces. The quantizer is bit-exact to the
    packer the GPU parity gate blesses, a packed bundle round-trips to ``W_ref`` with
    no tolerance, the validator rejects tampered bundles, and a packed directory is a
    self-contained model directory (config/tokenizer copied through).
  * GPU-UNVERIFIED: the entire execution side. No ``ommx_w`` bundle has ever been read
    by a CUDA kernel or by a vLLM engine — not once, on any device. There is no
    ``ommx_w`` latency or accuracy number anywhere in this repo. The weight kernel's
    own parity gate (``csrc/linear/test_ommx_linear_parity.py``) passes on sm_90a, but
    against tensors it quantizes in-process, not against a bundle this packer wrote.

See ``README.md`` in this directory for the itemised verified/unverified split.

Pure torch. Importable with no vLLM, no triton and no GPU — like every other module
under ``ommx_gpu_serve`` except ``integration/vllm/backend.py``.
"""
from __future__ import annotations

from .quantize import (
    FIG6_STEPS,
    FP4_E2M1,
    FP4_RANGE,
    OMMXQuantizeError,
    OUTLIER_MAPS,
    OUTLIER_REPRS,
    dequantize_ommx_weight,
    derive_npv,
    quantize_ommx_weight,
)
from .w_format import (
    AUX_POLICIES,
    INDEX_FILENAME,
    OMMX_W_FORMAT,
    OMMX_W_FORMAT_VERSION,
    OMMX_W_LEGACY_VERSIONS,
    OMMXWFormatError,
    PLANE_NAME_TEMPLATE,
    QUANTIZED_WEIGHT_SUFFIX,
    REQUIRED_AUX_FILES,
    Recipe,
    SafeTensorsFile,
    SafeTensorsWriter,
    load_planes,
    load_weight,
    naming_document,
    plane_name,
    read_manifest,
    sha256_file,
    split_plane_name,
    validate_aux_files,
    validate_bundle,
    validate_shard,
)

__all__ = [
    # quantize
    "quantize_ommx_weight", "dequantize_ommx_weight", "derive_npv",
    "OMMXQuantizeError", "FIG6_STEPS", "FP4_E2M1", "FP4_RANGE",
    "OUTLIER_REPRS", "OUTLIER_MAPS",
    # format
    "OMMX_W_FORMAT", "OMMX_W_FORMAT_VERSION", "OMMX_W_LEGACY_VERSIONS",
    "INDEX_FILENAME", "OMMXWFormatError", "PLANE_NAME_TEMPLATE",
    "QUANTIZED_WEIGHT_SUFFIX", "REQUIRED_AUX_FILES", "AUX_POLICIES",
    "Recipe", "SafeTensorsFile", "SafeTensorsWriter", "plane_name", "split_plane_name",
    "naming_document", "sha256_file",
    "read_manifest", "validate_shard", "validate_bundle", "validate_aux_files",
    "load_planes", "load_weight",
]
