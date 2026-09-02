# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""``OMMX_W_SafeTensor`` — the on-disk weight-bundle format (paper claim C3).

The paper (§3.3) says ``OMMX_Linear`` reads "weight tensors stored in
``OMMX_W_SafeTensor``" and Fig 6 draws an "OMMX Safetensor Weight Format Offline
Packing" box, but no such format exists anywhere in this release — there is no
prior art in the repo to mirror. This module therefore WRITES THE SPEC, and is the
single source of truth for it.

═══════════════════════════════════════════════════════════════════════════════
SPEC — ommx_w version 2
═══════════════════════════════════════════════════════════════════════════════

A bundle is a SELF-CONTAINED MODEL DIRECTORY:

    <bundle>/
        ommx_w-00001-of-000NN.safetensors    one shard per input checkpoint shard
        ...
        ommx_w_index.json                    tensor name -> shard file, + recipe/naming/
                                             aux-file record/totals
        config.json                          copied from the source checkpoint (REQUIRED)
        generation_config.json               copied if the source has it
        tokenizer*.json / tokenizer.model / special_tokens_map.json / chat_template.*
        ...                                  every other non-weight file of the source

The non-weight files are what make the documented recipe ("pack, then point vLLM at
the bundle directory") work at all: a directory of safetensors shards with no
``config.json`` cannot be resolved as a model by transformers or vLLM. Which files
were copied, which were deliberately skipped, and why, is written into
``ommx_w_index.json`` under ``aux_files`` — see :func:`validate_bundle`. Files that
CARRY WEIGHTS (``*.bin``, ``*.pt``, a stray ``model.safetensors.index.json``) are
never copied: a loader that found them would silently serve the ORIGINAL unquantized
weights while the run was labelled ``ommx_w``.

Each shard is a plain safetensors file (so any safetensors reader can open it), and
each shard's file-level ``__metadata__`` carries:

    "format"      : "ommx_w"
    "version"     : "2"
    "manifest"    : "<the JSON document described below, for THIS shard>"

Tensor naming — one set of planes per quantized Linear weight. A plane is named after
the MODULE that owns it, with the quantized weight's ``.weight`` suffix REPLACED (not
extended) by the plane suffix::

    model.layers.0.self_attn.q_proj.weight   (source checkpoint tensor)
        -> model.layers.0.self_attn.q_proj.ommx_code   (bundle plane)

That is the convention every other quantization format in the ecosystem uses — AWQ
ships ``<module>.qweight``, not ``<module>.weight.qweight`` — and it is not cosmetic:
vLLM (and transformers) match a checkpoint tensor name against
``dict(model.named_parameters())``, and a parameter registered on a Linear module is
reachable as ``<module>.<attr>``. A ``<module>.weight.ommx_code`` plane has no
parameter to bind to and a real engine fails at load. Version 1 of this format used
the ``.weight`` infix; version 2 dropped it, and the validator refuses version 1 BY
NAME rather than best-effort loading it (see :func:`read_manifest`).

    <module>.ommx_code         uint8   [N, K/4]      dense INT2 payload, 4 codes/byte,
                                                     code j at bit 2*(j%4)
    <module>.ommx_scale_exp    int8    [N, G]        E8M0 shared exponent, scale = 2^e
    <module>.ommx_zp           bf16    [N, G]        shared zero-point (paper Eq (2))
    <module>.ommx_oindex       uint8   [N, G, IB]    outlier positions (repr-dependent)
    <module>.ommx_ocode        uint8   [N, G, NB]    FP4 E2M1 nibbles, 2 per byte,
                                                     slot s at bit 4*(s%2), ascending
                                                     position order
    <module>.ommx_map_scale    f32     [N, G]        FP4 index-space range map, ms
    <module>.ommx_map_center   f32     [N, G]        FP4 index-space range map, mc

The convention is not left to be inferred: every shard manifest and the index carry a
``naming`` block (:func:`naming_document`) that states the template, the suffix that
was dropped, and a worked example, and the validator refuses a bundle whose declared
convention is not the one this build implements.

with ``G = K/group_size``, ``IB = ceil(npv*7/8)`` for ``relidx7`` or
``ceil(group_size/8)`` for ``bitmap``, and ``NB = ceil(npv/2)``. The last four planes
are absent when ``npv == 0`` / ``outlier_map == "none"`` respectively, and the
manifest says so; a plane that is declared but missing is a hard validation error.

Everything NOT quantized (embeddings, ``lm_head``, norms, biases, router weights,
anything the packer did not recognise) is copied through under its ORIGINAL name and
ORIGINAL dtype, and is listed in the manifest as ``kind: "passthrough"`` with the
reason. This is deliberate: a silently-quantized ``lm_head`` is a correctness
disaster that shows up only as a slightly worse eval score, so the bundle records
the decision for every single tensor rather than leaving it implicit.

Manifest document (one per shard)::

    {
      "format": "ommx_w", "version": 2,
      "producer": "...", "source_model": "...",
      "shard": {"index": 1, "count": 3, "file": "..."},
      "naming": {plane_name, weight_suffix_dropped, example, rationale},
      "recipe": {group_size, npv, outlier_pct, base_format, outlier_format,
                 bundle_format, outlier_repr, outlier_map, scale_format,
                 zp_dtype, map_dtype, fp4_range},
      "tensors": {"<orig name>": {kind, dtype, shape, [planes], packed_bytes,
                                  bits_per_weight, [reason]}},
      "totals":  {n_quantized, n_passthrough, quantized_elements,
                  quantized_packed_bytes, passthrough_bytes, bits_per_weight}
    }

Index document (``ommx_w_index.json``, one per bundle) carries the same ``format`` /
``version`` / ``naming`` / ``recipe``, plus ``weight_map`` (tensor -> shard file),
``shards``, ``totals``, and::

      "aux_files": {
        "policy": "copy" | "no-copy-aux",
        "self_contained": true | false,
        "copied":  [{"name", "bytes", "sha256"}],
        "skipped": [{"name", "reason"}],
        "missing": [{"name", "reason"}]
      }

Versioning: ``version`` is an integer and this build reads ONLY version 2. An
unknown version is refused by name rather than best-effort parsed — a future format
that moved a plane would otherwise dequantize to plausible garbage — and version 1
specifically is refused with the reason (``.weight`` infix) and the re-pack command,
because that is the one older bundle that exists in the wild-ish and the one whose
failure mode ("parameter not found" deep inside a vLLM loader) is least legible.

═══════════════════════════════════════════════════════════════════════════════

WHY THE SAFETENSORS CONTAINER IS IMPLEMENTED HERE
-------------------------------------------------
The ``safetensors`` package is not a dependency of ``ommx_gpu_serve`` (see its
pyproject: the only hard dep is torch) and is not installed on the host this module
was written and tested on. Rather than add a dependency for a container format that
is 8 bytes of length + a JSON header + a flat byte buffer, the reader and writer are
implemented here in full. They are byte-compatible with the published spec, and the
writer STREAMS (one tensor in RAM at a time, payload spilled to a temp file, header
written last) which is what lets the packer run on a checkpoint larger than RAM.

VERIFIED on CPU this session: self round-trip of every dtype the packer emits,
header byte layout (8-byte LE length, JSON, 8-byte aligned payload start), and the
validator's rejection paths. NOT VERIFIED this session: interop with the official
``safetensors`` Rust/Python implementation — it is not installed on this host. The
layout follows the published spec; treat cross-implementation interop as untested.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .quantize import (
    FP4_RANGE,
    OUTLIER_MAPS,
    OUTLIER_REPRS,
    ZP_DTYPES,
    combinadic_index_bits,
    dequantize_ommx_weight,
    outlier_code_bytes,
    outlier_index_bytes,
)

OMMX_W_FORMAT = "ommx_w"

#: Bumped 1 -> 2 when the ``.weight`` infix was dropped from plane names (see the SPEC
#: block above). This is a HARD break, not a cosmetic one: a version-1 plane cannot
#: bind to a vLLM Linear parameter, so version 1 is refused by name with a re-pack
#: command rather than accepted alongside version 2. Accepting both would mean the
#: loader has to guess which convention a name follows, and guessing wrong on a
#: fused/renamed module is a silent mis-bind.
OMMX_W_FORMAT_VERSION = 2

#: The one older version this build recognises well enough to refuse SPECIFICALLY.
OMMX_W_LEGACY_VERSIONS: Dict[int, str] = {
    1: ("planes were named '<module>.weight.ommx_<plane>' (a '.weight' infix that no "
        "vLLM/transformers parameter has), so a version-1 bundle cannot be bound by a "
        "model loader"),
}

INDEX_FILENAME = "ommx_w_index.json"
PLANE_PREFIX = "ommx_"

#: The suffix a quantized checkpoint tensor carries and a plane REPLACES. The packer
#: only ever quantizes ``*.weight`` tensors (``w_packer.classify_tensor`` passes
#: everything else through), so this is a total function on the quantizable set —
#: :func:`plane_name` raises rather than inventing a name for anything else.
QUANTIZED_WEIGHT_SUFFIX = ".weight"

#: Human/machine readable statement of the naming convention, replicated into every
#: shard manifest and the index so a reader never has to reverse-engineer it from a
#: tensor list. ``{module}`` is the owning module path, i.e. the checkpoint tensor
#: name with :data:`QUANTIZED_WEIGHT_SUFFIX` removed.
PLANE_NAME_TEMPLATE = "{module}.%s{plane}" % PLANE_PREFIX

# ── non-weight ("aux") checkpoint files ──────────────────────────────────────
# A bundle is a MODEL DIRECTORY, not a pile of shards: the documented recipe is
# "pack, then point vLLM at the bundle". Nothing resolves a model without these.

#: Files a bundle MUST contain to be loadable at all. ``config.json`` is the file
#: transformers/vLLM open FIRST to decide which architecture to build; without it the
#: directory is not a model and the packer refuses to produce one (see
#: ``w_packer.plan_aux_files``). Kept to the genuinely load-bearing minimum — a longer
#: required list would turn legitimate checkpoints (sentencepiece-only tokenizers,
#: base models with no generation_config.json) into pack failures.
REQUIRED_AUX_FILES: Tuple[str, ...] = ("config.json",)

#: Aux-file policies recorded in the index. ``no-copy-aux`` is an EXPLICIT opt-out and
#: is written down precisely so a bundle that is not self-contained says so.
AUX_POLICIES: Tuple[str, ...] = ("copy", "no-copy-aux")

#: Planes every quantized weight has.
PLANES_BASE = ("code", "scale_exp", "zp")
#: Planes present iff ``npv > 0``.
PLANES_OUTLIER = ("oindex", "ocode")
#: Planes present iff ``npv > 0 and outlier_map == "idx_range"``.
PLANES_MAP = ("map_scale", "map_center")
ALL_PLANES = PLANES_BASE + PLANES_OUTLIER + PLANES_MAP


class OMMXWFormatError(ValueError):
    """A bundle (or a file claiming to be one) is malformed.

    Every message names the exact mismatch — file, tensor, plane, expected vs
    found — because the failure mode this class exists to prevent is a bundle that
    loads without complaint and produces weights that are merely wrong.
    """


# ════════════════════════════════════════════════════════════════════════════
# safetensors container  (spec: u64 LE header length | JSON header | byte buffer)
# ════════════════════════════════════════════════════════════════════════════

_ST_TO_TORCH: Dict[str, torch.dtype] = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8,
    "I16": torch.int16, "F16": torch.float16, "BF16": torch.bfloat16,
    "I32": torch.int32, "F32": torch.float32,
    "I64": torch.int64, "F64": torch.float64,
}
_TORCH_TO_ST: Dict[torch.dtype, str] = {v: k for k, v in _ST_TO_TORCH.items()}
# U16/U32/U64 are in the safetensors dtype table but torch has no matching dtype, so
# a checkpoint carrying one is refused by name instead of being reinterpreted.
_ST_UNSUPPORTED = ("U16", "U32", "U64", "F8_E5M2", "F8_E4M3")

_HEADER_ALIGN = 8      # official writers pad the JSON header to an 8-byte boundary
_COPY_CHUNK = 8 << 20  # payload copy granularity; bounds writer RAM at 8 MiB


def st_dtype_name(dtype: torch.dtype) -> str:
    """torch dtype -> safetensors dtype string, refusing anything unrepresentable."""
    try:
        return _TORCH_TO_ST[dtype]
    except KeyError:
        raise OMMXWFormatError(
            f"dtype {dtype} has no safetensors encoding (supported: "
            f"{sorted(_TORCH_TO_ST.values())})") from None


def _require_little_endian() -> None:
    if sys.byteorder != "little":
        raise OMMXWFormatError(
            "safetensors payloads are little-endian and this host is "
            f"{sys.byteorder}-endian; byte-swapping is not implemented.")


def _tensor_bytes(t: torch.Tensor) -> bytes:
    """Raw little-endian payload for one tensor (contiguous, CPU)."""
    _require_little_endian()
    t = t.detach().cpu().contiguous()
    if t.numel() == 0:
        return b""
    return t.reshape(-1).view(torch.uint8).numpy().tobytes()


class SafeTensorsWriter:
    """Streaming safetensors writer: one tensor in RAM at a time.

    The header cannot be written until every tensor's byte length is known, so the
    payload is spilled to a sibling temp file as tensors arrive and copied in after
    the header on ``close()``. Peak RAM is one tensor plus ``_COPY_CHUNK``, which is
    what makes ``pack_checkpoint`` able to process a model larger than memory.
    """

    def __init__(self, path: str):
        self.path = path
        self._entries: List[Tuple[str, str, List[int], int, int]] = []
        self._offset = 0
        d = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(d, exist_ok=True)
        fd, self._spill = tempfile.mkstemp(prefix=".ommx_w_payload.", dir=d)
        self._fh = os.fdopen(fd, "wb")
        self._closed = False

    def add(self, name: str, tensor: torch.Tensor) -> int:
        """Append one tensor; returns its byte length."""
        if self._closed:
            raise OMMXWFormatError("SafeTensorsWriter.add() after close()")
        if any(e[0] == name for e in self._entries):
            raise OMMXWFormatError(
                f"duplicate tensor name {name!r} in {os.path.basename(self.path)}")
        blob = _tensor_bytes(tensor)
        self._fh.write(blob)
        self._entries.append((name, st_dtype_name(tensor.dtype),
                              list(tensor.shape), self._offset, self._offset + len(blob)))
        self._offset += len(blob)
        return len(blob)

    def close(self, metadata: Optional[Dict[str, str]] = None) -> None:
        """Write ``header || payload`` to ``self.path`` and drop the spill file."""
        if self._closed:
            return
        self._closed = True
        self._fh.close()
        header: Dict[str, object] = {}
        if metadata:
            header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
        for name, dt, shape, beg, end in self._entries:
            header[name] = {"dtype": dt, "shape": shape, "data_offsets": [beg, end]}
        raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
        pad = (-len(raw)) % _HEADER_ALIGN
        raw = raw + b" " * pad
        try:
            with open(self.path, "wb") as out:
                out.write(len(raw).to_bytes(8, "little"))
                out.write(raw)
                with open(self._spill, "rb") as src:
                    while True:
                        chunk = src.read(_COPY_CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
        finally:
            os.unlink(self._spill)

    def abort(self) -> None:
        """Discard everything written so far (used when packing raises mid-shard)."""
        if self._closed:
            return
        self._closed = True
        self._fh.close()
        if os.path.exists(self._spill):
            os.unlink(self._spill)

    def __enter__(self) -> "SafeTensorsWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()


class SafeTensorsFile:
    """Random-access safetensors reader — reads only the byte range it is asked for."""

    def __init__(self, path: str):
        _require_little_endian()
        self.path = path
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) != 8:
                raise OMMXWFormatError(f"{path}: too short to be a safetensors file")
            n = int.from_bytes(head, "little")
            size = os.path.getsize(path)
            if n <= 0 or 8 + n > size:
                raise OMMXWFormatError(
                    f"{path}: header length {n} does not fit in a {size}-byte file")
            try:
                header = json.loads(fh.read(n).decode("utf-8"))
            except Exception as exc:
                raise OMMXWFormatError(f"{path}: header is not valid JSON ({exc})") from None
        if not isinstance(header, dict):
            raise OMMXWFormatError(f"{path}: header is not a JSON object")
        self._data_start = 8 + n
        self.metadata: Dict[str, str] = dict(header.pop("__metadata__", {}) or {})
        self._header = header

    @property
    def names(self) -> List[str]:
        return list(self._header.keys())

    def info(self, name: str) -> dict:
        try:
            return self._header[name]
        except KeyError:
            raise OMMXWFormatError(
                f"{os.path.basename(self.path)}: no tensor named {name!r}") from None

    def shape(self, name: str) -> List[int]:
        return [int(v) for v in self.info(name)["shape"]]

    def dtype_name(self, name: str) -> str:
        return str(self.info(name)["dtype"])

    def nbytes(self, name: str) -> int:
        beg, end = self.info(name)["data_offsets"]
        return int(end) - int(beg)

    def tensor(self, name: str) -> torch.Tensor:
        """Read one tensor. Only its byte range is touched."""
        info = self.info(name)
        dt = str(info["dtype"])
        if dt in _ST_UNSUPPORTED or dt not in _ST_TO_TORCH:
            raise OMMXWFormatError(
                f"{os.path.basename(self.path)}: tensor {name!r} has dtype {dt}, which "
                f"this reader cannot represent in torch")
        beg, end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
        shape = [int(v) for v in info["shape"]]
        with open(self.path, "rb") as fh:
            fh.seek(self._data_start + beg)
            raw = fh.read(end - beg)
        if len(raw) != end - beg:
            raise OMMXWFormatError(
                f"{os.path.basename(self.path)}: tensor {name!r} declares "
                f"{end - beg} bytes but only {len(raw)} are present (truncated file)")
        torch_dt = _ST_TO_TORCH[dt]
        n_elem = 1
        for s in shape:
            n_elem *= s
        if n_elem == 0:
            return torch.empty(shape, dtype=torch_dt)
        return torch.frombuffer(bytearray(raw), dtype=torch_dt).reshape(shape)


# ════════════════════════════════════════════════════════════════════════════
# Recipe + derived layout
# ════════════════════════════════════════════════════════════════════════════

class Recipe:
    """The quantization recipe a whole bundle was packed with.

    One recipe per bundle, replicated into every shard's manifest and into the
    index. The validator compares them across shards: two shards packed with
    different group sizes would each load fine on their own and produce a model
    whose layers disagree, which is precisely the class of silent failure this
    format is meant to make impossible.
    """

    __slots__ = ("group_size", "npv", "outlier_pct", "outlier_repr", "outlier_map",
                 "zp_dtype", "map_dtype")

    def __init__(self, group_size: int, npv: int, outlier_pct: float,
                 outlier_repr: str = "relidx7", outlier_map: str = "idx_range",
                 zp_dtype: torch.dtype = torch.bfloat16,
                 map_dtype: torch.dtype = torch.float32):
        self.group_size = int(group_size)
        self.npv = int(npv)
        self.outlier_pct = float(outlier_pct)
        self.outlier_repr = str(outlier_repr)
        self.outlier_map = str(outlier_map)
        self.zp_dtype = zp_dtype
        self.map_dtype = map_dtype
        self.validate()

    def validate(self) -> None:
        if self.group_size <= 0 or self.group_size % 4:
            raise OMMXWFormatError(
                f"group_size={self.group_size} must be positive and a multiple of 4")
        if not 0 <= self.npv <= self.group_size:
            raise OMMXWFormatError(
                f"npv={self.npv} must be in [0, group_size={self.group_size}]")
        if self.outlier_repr not in OUTLIER_REPRS:
            raise OMMXWFormatError(
                f"outlier_repr={self.outlier_repr!r} not in {OUTLIER_REPRS}")
        if self.outlier_map not in OUTLIER_MAPS:
            raise OMMXWFormatError(
                f"outlier_map={self.outlier_map!r} not in {OUTLIER_MAPS}")
        if self.zp_dtype not in ZP_DTYPES:
            raise OMMXWFormatError(
                f"zp_dtype={self.zp_dtype} not in {[str(d) for d in ZP_DTYPES]}")
        if self.npv and self.outlier_repr == "relidx7" and self.group_size > 128:
            raise OMMXWFormatError(
                f"relidx7 cannot address a position in a group of {self.group_size} "
                f"(7 bits -> max 128); use outlier_repr='bitmap'")

    # ── derived properties ────────────────────────────────────────────────────
    @property
    def has_outliers(self) -> bool:
        return self.npv > 0

    @property
    def has_map(self) -> bool:
        return self.npv > 0 and self.outlier_map == "idx_range"

    @property
    def bundle_format(self) -> str:
        """The kernel's format tag: INT2 base with (i2f4) or without (i2) FP4 outliers."""
        return "i2f4" if self.has_outliers else "i2"

    @property
    def index_bytes(self) -> int:
        return outlier_index_bytes(self.npv, self.group_size, self.outlier_repr)

    @property
    def code_bytes(self) -> int:
        return outlier_code_bytes(self.npv)

    def planes(self) -> Tuple[str, ...]:
        p = PLANES_BASE
        if self.has_outliers:
            p = p + PLANES_OUTLIER
        if self.has_map:
            p = p + PLANES_MAP
        return p

    def plane_layout(self, N: int, K: int) -> Dict[str, Tuple[Tuple[int, ...], str]]:
        """Expected ``{plane: (shape, safetensors dtype)}`` for a weight ``[N, K]``.

        This is the function the validator uses to recompute what a manifest CLAIMS
        against what the recipe REQUIRES, so a hand-edited shape is caught even when
        the tensor on disk matches the (wrong) manifest.
        """
        if K % self.group_size:
            raise OMMXWFormatError(
                f"weight K={K} is not divisible by group_size={self.group_size}")
        G = K // self.group_size
        out: Dict[str, Tuple[Tuple[int, ...], str]] = {
            "code": ((N, K // 4), "U8"),
            "scale_exp": ((N, G), "I8"),
            "zp": ((N, G), st_dtype_name(self.zp_dtype)),
        }
        if self.has_outliers:
            out["oindex"] = ((N, G, self.index_bytes), "U8")
            out["ocode"] = ((N, G, self.code_bytes), "U8")
        if self.has_map:
            md = st_dtype_name(self.map_dtype)
            out["map_scale"] = ((N, G), md)
            out["map_center"] = ((N, G), md)
        return out

    # ── bit accounting ────────────────────────────────────────────────────────
    def bits_breakdown(self) -> Dict[str, float]:
        """Bits per WEIGHT ELEMENT, per plane, as actually stored (byte padding included).

        Reported two ways because the two differ and the difference is easy to argue
        about:

        ``stored``   — what the packer writes, and the ONLY memory-traffic statement of
                       the two. Position and nibble streams are byte padded per group,
                       which is what the relidx7 layout requires (a slot stream must
                       start on a byte boundary for the kernel's ``relidx7_slot_pos``
                       to index it). This is the column ``csrc/linear/README.md``
                       publishes: 4.125 / 4.750 / 6.125 b/wt at npv 4/8/16, gs=64,
                       ``relidx7`` + ``idx_range``.
        ``npu``      — the SAME bundle re-encoded for the Nucleus NPU, which the paper
                       specifies as differing from the GPU form in the position
                       metadata and nothing else: §3.1 stores positions "into
                       ceil(log2 C(N,K)) bits per group" via an on-chip combinatorial
                       codec, and says "All other bundle fields are platform-invariant,
                       requiring only position-metadata regeneration per target". So
                       this basis substitutes ``combinadic_index_bits(npv, gs)/gs`` for
                       the ``outlier_index`` term and leaves every other plane alone.
                       It is a REPORTING basis only -- this packer never writes a
                       combinadic plane and no kernel here reads one -- and it exists so
                       that a published AvgBits number can be checked against the recipe
                       it is claimed for. Note it is INDEPENDENT of ``outlier_repr``:
                       relidx7 and bitmap are two GPU encodings of the same position
                       set, and the NPU re-derives that set either way.
        ``unpadded`` — the information-theoretic figure (2 + 8/gs + 16/gs + npv*11/gs
                       at relidx7, no range map). It matches ``stored`` whenever
                       ``npv*7`` is a multiple of 8 and is lower otherwise (npv=4,
                       gs=64: 3.0625 unpadded vs 3.125 stored). ``csrc/linear/README.md``
                       used to publish this column MINUS the range map (3.06 / 3.75 /
                       5.125) and explicitly repudiates those figures now; do not
                       reintroduce them as "the" bit budget. Both corrections are
                       pinned by ``tests/test_w_packer.py::
                       test_readme_published_bit_budgets_reproduce``.
        """
        gs = float(self.group_size)
        zp_bits = 8.0 * _itemsize(st_dtype_name(self.zp_dtype))
        map_bits = (2.0 * 8.0 * _itemsize(st_dtype_name(self.map_dtype))
                    if self.has_map else 0.0)
        idx_unpadded = (7.0 * self.npv if self.outlier_repr == "relidx7"
                        else float(self.group_size)) if self.has_outliers else 0.0
        val_unpadded = 4.0 * self.npv if self.has_outliers else 0.0
        stored = {
            "base_int2": 2.0,
            "scale_e8m0": 8.0 / gs,
            "zero_point": zp_bits / gs,
            "outlier_index": self.index_bytes * 8.0 / gs,
            "outlier_fp4": self.code_bytes * 8.0 / gs,
            "fp4_range_map": map_bits / gs,
        }
        stored["total"] = sum(stored.values())
        unpadded = dict(stored)
        unpadded["outlier_index"] = idx_unpadded / gs
        unpadded["outlier_fp4"] = val_unpadded / gs
        unpadded["total"] = (2.0 + 8.0 / gs + zp_bits / gs
                             + idx_unpadded / gs + val_unpadded / gs + map_bits / gs)
        # ONE substitution, and only one. The nibble stream keeps its byte padding here
        # even though the position field loses its own: the paper's warrant for a second
        # budget is "position-metadata regeneration per target", which says nothing about
        # re-packing values, and helping the number along by also unpadding the nibbles
        # would make `npu` a different recipe rather than a different encoding of this
        # one. It matters only at odd npv (ceil(npv/2) bytes vs 4*npv bits).
        npu = dict(stored)
        npu["outlier_index"] = (combinadic_index_bits(self.npv, self.group_size) / gs
                                if self.has_outliers else 0.0)
        npu["total"] = sum(v for k, v in npu.items() if k != "total")
        return {"stored": stored, "unpadded": unpadded, "npu": npu,
                "bits_per_weight": stored["total"],
                "bits_per_weight_unpadded": unpadded["total"],
                "bits_per_weight_npu": npu["total"]}

    def packed_bytes(self, N: int, K: int) -> int:
        """Exact byte count of every plane for a weight ``[N, K]``.

        Derived from the LAYOUT (shape x itemsize), which is what makes it a genuine
        cross-check of the byte totals the writer reports back from the file.
        """
        total = 0
        for shape, dt in self.plane_layout(N, K).values():
            n = 1
            for s in shape:
                n *= s
            total += n * _itemsize(dt)
        return total

    # ── (de)serialisation ─────────────────────────────────────────────────────
    def to_json(self) -> dict:
        return {
            "group_size": self.group_size,
            "npv": self.npv,
            "outlier_pct": self.outlier_pct,
            "base_format": "int2_affine",
            "outlier_format": "fp4_e2m1",
            "bundle_format": self.bundle_format,
            "outlier_repr": self.outlier_repr,
            "outlier_map": self.outlier_map,
            "scale_format": "e8m0_int8",
            "zp_dtype": st_dtype_name(self.zp_dtype),
            "map_dtype": st_dtype_name(self.map_dtype),
            "fp4_range": FP4_RANGE,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Recipe":
        missing = [k for k in ("group_size", "npv", "outlier_repr", "outlier_map",
                               "zp_dtype") if k not in d]
        if missing:
            raise OMMXWFormatError(f"recipe is missing key(s) {missing}")
        for key, want in (("base_format", "int2_affine"), ("scale_format", "e8m0_int8")):
            if key in d and d[key] != want:
                raise OMMXWFormatError(
                    f"recipe {key}={d[key]!r} is not supported by this build "
                    f"(only {want!r})")
        if d["zp_dtype"] not in _ST_TO_TORCH:
            raise OMMXWFormatError(f"recipe zp_dtype={d['zp_dtype']!r} is not a dtype")
        return cls(group_size=int(d["group_size"]), npv=int(d["npv"]),
                   outlier_pct=float(d.get("outlier_pct", 0.0)),
                   outlier_repr=str(d["outlier_repr"]), outlier_map=str(d["outlier_map"]),
                   zp_dtype=_ST_TO_TORCH[d["zp_dtype"]],
                   map_dtype=_ST_TO_TORCH[d.get("map_dtype", "F32")])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Recipe) and self.to_json() == other.to_json()

    def __repr__(self) -> str:
        return (f"Recipe(gs={self.group_size}, npv={self.npv}, "
                f"repr={self.outlier_repr}, map={self.outlier_map}, "
                f"zp={st_dtype_name(self.zp_dtype)})")


def _itemsize(st_dtype: str) -> int:
    """Bytes per element for a safetensors dtype string."""
    return {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "F16": 2, "BF16": 2,
            "I32": 4, "F32": 4, "I64": 8, "F64": 8}[st_dtype]


def naming_document() -> Dict[str, str]:
    """The plane-naming convention, as written into every manifest and the index.

    Self-describing on purpose: a bundle is read by tools that were not built from
    this file, and "infer the convention from the tensor list" is how the ``.weight``
    infix survived a whole release. :func:`validate_shard` / :func:`validate_bundle`
    compare a bundle's declared block against this one and refuse a mismatch, so a
    hand-edited or foreign-produced bundle cannot claim a convention it does not use.
    """
    return {
        "plane_name": PLANE_NAME_TEMPLATE,
        "weight_suffix_dropped": QUANTIZED_WEIGHT_SUFFIX,
        "example": ("model.layers.0.self_attn.q_proj.weight -> "
                    "model.layers.0.self_attn.q_proj." + PLANE_PREFIX + "code"),
        "rationale": ("a parameter registered on a vLLM/transformers Linear module is "
                      "reachable as <module>.<attr>, so a plane must be named for the "
                      "MODULE, not for the tensor it replaced (AWQ ships "
                      "<module>.qweight, not <module>.weight.qweight)"),
    }


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file — PROVENANCE for a copied aux file, not a gate.

    Streaming because a ``tokenizer.json`` is tens of MB and the packer's whole design
    promise is that peak RAM is one tensor.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def plane_name(weight_name: str, plane: str) -> str:
    """Tensor name of one plane: ``...q_proj.weight`` -> ``...q_proj.ommx_code``.

    The ``.weight`` suffix is REPLACED, not extended. See the SPEC block at the top of
    this module for why (short version: ``<module>.weight.ommx_code`` binds to no
    parameter and a real engine fails at load).

    A ``weight_name`` without the ``.weight`` suffix raises: the packer quantizes only
    ``*.weight`` tensors, so falling back to "append the plane suffix" here could only
    ever produce a name no loader will look for — and it would do it silently, which is
    exactly the failure this version bump exists to remove.
    """
    if plane not in ALL_PLANES:
        raise OMMXWFormatError(f"unknown plane {plane!r}; expected one of {ALL_PLANES}")
    if not weight_name.endswith(QUANTIZED_WEIGHT_SUFFIX):
        raise OMMXWFormatError(
            f"cannot name a plane for {weight_name!r}: an OMMX_W_SafeTensor plane is "
            f"named after the MODULE that owns the weight, so the tensor name must end "
            f"in {QUANTIZED_WEIGHT_SUFFIX!r} (got tail "
            f"{weight_name.rpartition('.')[2]!r}). Only '*.weight' tensors are "
            f"quantizable; everything else is passed through under its original name.")
    module = weight_name[: -len(QUANTIZED_WEIGHT_SUFFIX)]
    if not module:
        raise OMMXWFormatError(
            f"cannot name a plane for {weight_name!r}: it is bare "
            f"{QUANTIZED_WEIGHT_SUFFIX!r} with no owning module path")
    return f"{module}.{PLANE_PREFIX}{plane}"


def split_plane_name(tensor_name: str) -> Optional[Tuple[str, str]]:
    """Inverse of :func:`plane_name`: plane tensor -> ``(weight_name, plane)``.

    Returns ``None`` for anything that is not a plane (a passthrough tensor, say), so
    a caller can partition a shard's tensor list without exception control flow. The
    round-trip ``split_plane_name(plane_name(w, p)) == (w, p)`` is a gate in
    ``tests/test_w_packer.py`` — the two must not be able to drift apart.
    """
    module, dot, tail = tensor_name.rpartition(".")
    if not dot or not tail.startswith(PLANE_PREFIX):
        return None
    plane = tail[len(PLANE_PREFIX):]
    if plane not in ALL_PLANES or not module:
        return None
    return module + QUANTIZED_WEIGHT_SUFFIX, plane


def bits_per_weight(recipe: Recipe, N: int, K: int, packed_bytes: int) -> float:
    """Longhand bits/weight from ACTUAL bytes — the number the packer reports.

    Deliberately computed from the byte total rather than from
    ``recipe.bits_breakdown()`` so the two are independent: the test asserts they
    agree, which they only can if both the layout and the accounting are right.
    """
    return 8.0 * float(packed_bytes) / float(N * K)

# ════════════════════════════════════════════════════════════════════════════
# LOAD-TIME TRANSCODE — an on-disk recipe -> the one the CUDA kernel can read
# ════════════════════════════════════════════════════════════════════════════
#
# WHY THIS EXISTS. The packer can write four (outlier_repr, outlier_map) combinations;
# ``csrc/linear/ommx_linear.cu`` reads exactly ONE of them. In particular the paper's own
# GPU weight format — claim B4, "positions are stored as a flat bitmask (N bits per
# group)", with the bundle definition of claim B6 listing no range-map parameters — is
# ``(bitmap, none)``, which is the (64, 4, bitmap, none) row that lands on the paper's
# 3.6250 bits/weight (claim E2, Table 1) and which the serving path refuses at config
# time. Refusing is correct: ``sparse_correct`` would read the bitmask bytes as a relidx7
# slot stream and quietly reconstruct the wrong weights. But it also means the paper's
# format cannot be served AT ALL, which is the gap this section closes.
#
# WHAT IS AND IS NOT EQUIVALENT HERE — the whole justification, in two facts, both
# checked against the code that produces the planes rather than assumed:
#
#   1. POSITIONS. ``quantize.quantize_ommx_weight`` computes ``pos_sorted`` (ASCENDING)
#      once and then branches on ``outlier_repr`` for the ``oindex`` plane ONLY. A bitmap
#      row and a relidx7 slot stream are therefore two encodings of the SAME ascending
#      position set, and ``quantize.outlier_positions`` decodes both to the same
#      ``[N, G, npv]`` tensor. Re-encoding one as the other is lossless.
#   2. NIBBLES. ``ocode`` is built OUTSIDE that branch, from
#      ``nib_sorted = gather(nib_g, pos_sorted)`` — i.e. slot ``s`` of the FP4 stream is
#      the s-th outlier in ASCENDING POSITION ORDER for BOTH representations. That is
#      exactly what the kernel's ``relidx7_slot_delta`` assumes (``ocode_blk[s >> 1]``
#      paired with ``relidx7_slot_pos(oindex_blk, s)``), and it is what the KV path's
#      ``paged_decode._bitmap_splice_from_bytes`` assumes as well (popcount rank == slot).
#      So the nibble plane is SHARED and needs no transcode. Verified for the WEIGHT path
#      specifically by ``tests/test_linear_method.py::
#      test_bitmap_and_relidx7_differ_in_exactly_one_plane``, which packs the same weight
#      both ways and asserts every other plane is byte-identical.
#
#   3. THE RANGE MAP. ``outlier_map="none"`` is documented in ``quantize.py`` as the
#      DEGENERATE INSTANCE ms=1, mc=0 — the packer literally initialises
#      ``map_scale = ones`` / ``map_center = zeros`` and encodes against them, and
#      ``dequantize_ommx_weight`` substitutes ones/zeros when the planes are absent. The
#      kernel's delta is ``(fp4/ms_row[g] + mc_row[g] - code) * scale_row[g]``, so feeding
#      it constant ones/zeros planes reproduces the ``none`` semantics EXACTLY, with no
#      change to the kernel ABI (``sparse_correct`` takes both as required ``float32``
#      arguments and dereferences them unconditionally — checked in ommx_linear.cu).
#      Synthesising them is therefore a materialisation, not a guess.
#
# WHAT IT COSTS, AND WHY THAT MUST BE SAID OUT LOUD. The transcode changes the RESIDENT
# footprint. A (64, 4, bitmap, none) bundle is 3.6250 bits/weight ON DISK and becomes
# 4.1250 bits/weight IN HBM after transcode: the position stream shrinks (1.0 -> 0.5
# b/wt) but two f32 range-map planes appear (+1.0 b/wt). Nobody may read "we served the
# paper's bundle" as "we served at 3.63 bits in HBM", so :class:`TranscodePlan` computes
# BOTH numbers from the same ``Recipe.bits_breakdown()`` path and every caller is expected
# to print both.

#: The single (repr, map) pair ``csrc/linear/ommx_linear.cu`` decodes. Named here rather
#: than only in the serving module because the transcode TARGET is a format fact.
KERNEL_OUTLIER_REPR = "relidx7"
KERNEL_OUTLIER_MAP = "idx_range"

#: ``relidx7_slot_pos`` returns ``v & 0x7F`` — 7 bits, positions 0..127. A group wider
#: than this cannot be re-encoded as relidx7 at all, so a bitmap bundle at gs>128 is
#: genuinely unreadable and is refused rather than transcoded.
RELIDX7_MAX_GROUP_SIZE = 128

#: ``sparse_correct``'s host guard is ``TORCH_CHECK(npv > 0 && npv <= 32, "npv must be in
#: (0,32] (MAX_NPV)")`` and its M>1 kernel stages ``int lk[32]; float ld[32]`` per slot.
#: A transcode that produced a stream the kernel then refuses would move the failure from
#: load time to the first token, so the bound is enforced up front.
KERNEL_MAX_NPV = 32


class TranscodePlan:
    """What a load-time transcode would do to one bundle, and what it costs.

    Constructed by :func:`plan_transcode`. Holds the recipe as PACKED (``source``) and the
    recipe the kernel will actually read (``resident``), so the two bit budgets come from
    one accounting function and cannot drift into two hand-written numbers.

    The plan is deliberately inert — it computes and describes, it does not touch tensors.
    :func:`transcode_oindex_bitmap_to_relidx7` and :func:`degenerate_map_planes` do that,
    and the serving path calls them once per layer at ``process_weights_after_loading``.

    BASIS OF BOTH FIGURES, so they are comparable and neither over-claims: they count the
    BUNDLE PLANES and nothing else — ``code``, ``scale_exp``, ``zp``, ``oindex``,
    ``ocode`` and (when present) the two map planes. The fp32 ``scale``/``zp`` twins that
    ``process_weights_after_loading`` materialises for the kernel are OUTSIDE both
    numbers: they exist identically with and without a transcode, so including them would
    inflate the delta without changing it. The delta between these two numbers is exactly
    what the transcode costs.
    """

    __slots__ = ("source", "resident", "steps")

    def __init__(self, source: Recipe, resident: Recipe, steps: List[str]) -> None:
        self.source = source
        self.resident = resident
        self.steps = list(steps)

    # ── bit accounting: both numbers, from the same code path ─────────────────
    @property
    def on_disk_bits_per_weight(self) -> float:
        """Bits/weight of the bundle AS PACKED. For the paper's recipe: 3.6250."""
        return float(self.source.bits_breakdown()["bits_per_weight"])

    @property
    def resident_bits_per_weight(self) -> float:
        """Bits/weight the kernel streams AFTER the transcode. Paper's recipe: 4.1250."""
        return float(self.resident.bits_breakdown()["bits_per_weight"])

    @property
    def delta_bits_per_weight(self) -> float:
        """Resident minus on-disk. POSITIVE means the transcode costs HBM."""
        return self.resident_bits_per_weight - self.on_disk_bits_per_weight

    def on_disk_bytes(self, N: int, K: int) -> int:
        """Exact bytes of one ``[N, K]`` weight's planes as stored in the bundle."""
        return self.source.packed_bytes(N, K)

    def resident_bytes(self, N: int, K: int) -> int:
        """Exact bytes the transcoded planes of one ``[N, K]`` weight occupy in HBM.

        Assumes the source position plane is FREED after transcode (which the serving
        path does — it replaces the parameter's storage with an empty tensor). If it were
        kept, resident would be this plus ``source.index_bytes * N * G``, and the honest
        figure would be higher; that is why the free and this accounting live together.
        """
        return self.resident.packed_bytes(N, K)

    def describe(self, N: Optional[int] = None, K: Optional[int] = None) -> str:
        """One line, carrying BOTH bit budgets. This is what gets logged.

        A reader of the log must not be able to conclude that serving a 3.63 b/wt bundle
        serves at 3.63 b/wt in HBM, so the resident figure is on the same line as the
        on-disk one, with the delta spelled out and signed.
        """
        head = (f"on-disk {self.on_disk_bits_per_weight:.4f} b/wt "
                f"({self.source.outlier_repr}+{self.source.outlier_map}) -> RESIDENT "
                f"{self.resident_bits_per_weight:.4f} b/wt "
                f"({self.resident.outlier_repr}+{self.resident.outlier_map}), "
                f"delta {self.delta_bits_per_weight:+.4f} b/wt")
        if N is not None and K is not None:
            head += (f"; [{N},{K}] planes {self.on_disk_bytes(N, K)} B on disk -> "
                     f"{self.resident_bytes(N, K)} B resident")
        return head + f"; steps: {', '.join(self.steps)}"

    def to_json(self) -> dict:
        return {
            "source_recipe": self.source.to_json(),
            "resident_recipe": self.resident.to_json(),
            "steps": list(self.steps),
            "on_disk_bits_per_weight": self.on_disk_bits_per_weight,
            "resident_bits_per_weight": self.resident_bits_per_weight,
            "delta_bits_per_weight": self.delta_bits_per_weight,
        }

    def __repr__(self) -> str:
        return f"TranscodePlan({self.describe()})"


def plan_transcode(recipe: Recipe) -> Optional[TranscodePlan]:
    """What (if anything) must be transcoded to serve ``recipe`` through the CUDA kernel.

    Returns ``None`` when the recipe is ALREADY what the kernel reads (including the
    outlier-free case, where no position stream exists and the question does not arise) —
    callers use that ``None`` to keep the shipped path byte-identical. Returns a
    :class:`TranscodePlan` when the gap is closable losslessly, and RAISES
    :class:`OMMXWFormatError` when the recipe cannot reach the kernel at all.

    Three refusals, all derived from the kernel source rather than from taste. The first
    is a bound on the KERNEL and is therefore answered for EVERY outlier recipe, including
    the one the kernel already reads; the other two are bounds on the RE-ENCODING and only
    arise when a re-encoding is what was asked for:

      * ``npv > 32`` — ``sparse_correct`` refuses it host-side (:data:`KERNEL_MAX_NPV`)
        and ``apply()`` calls ``sparse_correct`` for every recipe with outliers, so such a
        bundle aborts on its first served token no matter how its positions are stored;
      * ``group_size > 128`` — relidx7 has 7 bits of position, so a bitmap group wider
        than that simply cannot be re-encoded (:data:`RELIDX7_MAX_GROUP_SIZE`);
      * an ``outlier_repr`` / ``outlier_map`` that is neither what the kernel reads nor
        the one documented lossless re-encoding of it.
    """
    if not recipe.has_outliers:
        return None
    # NPV FIRST, and BEFORE the "is anything to transcode?" question. This is a bound on
    # the KERNEL, not on the re-encoding: ``linear_method.apply()`` calls sparse_correct
    # for EVERY recipe with outliers, so csrc/linear/ommx_linear.cu:2346's
    # TORCH_CHECK(npv > 0 && npv <= 32) aborts every token of a relidx7+idx_range bundle
    # above the bound just as surely as it would a transcoded one — and Recipe.validate
    # only requires npv <= group_size, so the packer emits such a bundle happily
    # (gs=64, npv=40 is a legal --npv). Answering it only on the transcode path left the
    # natively-readable encoding loading fine and aborting on the first served token,
    # which is precisely the "worse place to find out" this check exists to avoid.
    if recipe.npv > KERNEL_MAX_NPV:
        raise OMMXWFormatError(
            f"npv={recipe.npv} exceeds the kernel's MAX_NPV={KERNEL_MAX_NPV}: "
            f"csrc/linear/ommx_linear.cu's sparse_correct asserts "
            f"TORCH_CHECK(npv > 0 && npv <= 32) and its M>1 kernel stages int lk[32] per "
            f"slot. apply() calls sparse_correct on every recipe that HAS outliers, so "
            f"this bundle would abort on its first served token whatever its position "
            f"encoding — a transcode would not rescue it and no transcode is needed to "
            f"hit it. Re-pack with a smaller --npv / --outlier-pct.")
    steps: List[str] = []
    if recipe.outlier_repr != KERNEL_OUTLIER_REPR:
        if recipe.outlier_repr != "bitmap":
            raise OMMXWFormatError(
                f"outlier_repr={recipe.outlier_repr!r} has no transcode to "
                f"{KERNEL_OUTLIER_REPR!r}: only 'bitmap' is a re-encoding of the same "
                f"ascending position set (quantize.outlier_positions decodes both).")
        if recipe.group_size > RELIDX7_MAX_GROUP_SIZE:
            raise OMMXWFormatError(
                f"a bitmap bundle at group_size={recipe.group_size} cannot be transcoded "
                f"to relidx7: relidx7_slot_pos in csrc/linear/ommx_linear.cu returns "
                f"'v & 0x7F', i.e. 7 bits of position, so a group may hold at most "
                f"{RELIDX7_MAX_GROUP_SIZE} positions. Re-pack with "
                f"--group-size <= {RELIDX7_MAX_GROUP_SIZE}.")
        steps.append("oindex: bitmap -> relidx7 (same ascending positions, re-encoded)")
    if recipe.outlier_map != KERNEL_OUTLIER_MAP:
        if recipe.outlier_map != "none":
            raise OMMXWFormatError(
                f"outlier_map={recipe.outlier_map!r} has no materialisation as an "
                f"idx_range map; only 'none' is the documented degenerate instance "
                f"(ms=1, mc=0).")
        steps.append("map_scale/map_center: synthesised as the degenerate ms=1, mc=0 "
                     "planes the 'none' map already means")
    if not steps:
        return None
    resident = Recipe(
        group_size=recipe.group_size, npv=recipe.npv, outlier_pct=recipe.outlier_pct,
        outlier_repr=KERNEL_OUTLIER_REPR, outlier_map=KERNEL_OUTLIER_MAP,
        zp_dtype=recipe.zp_dtype,
        # The synthesised planes are f32 because that is the dtype the kernel
        # dereferences (`map_scale.data_ptr<float>()`); the source recipe's map_dtype is
        # irrelevant here because a 'none' recipe stores no map planes at all.
        map_dtype=torch.float32)
    return TranscodePlan(recipe, resident, steps)


def transcode_oindex_bitmap_to_relidx7(oindex: torch.Tensor, N: int, G: int, npv: int,
                                       group_size: int) -> torch.Tensor:
    """Re-encode a bitmap position plane ``[N, G, ceil(gs/8)]`` as relidx7 ``[N, G, IB]``.

    LOSSLESS BY CONSTRUCTION, and not by a new decoder: the positions come out of
    ``quantize.outlier_positions`` (the one function both representations already share,
    and the one whose bitmap branch RAISES when a group's popcount disagrees with npv) and
    go back in through ``quantize._pack_bits_lsb_first`` — the exact primitive the packer
    itself uses to write a relidx7 stream. A second bit-packing implementation here is
    precisely how the two would drift, so there isn't one.

    Runs on CPU regardless of where ``oindex`` lives, and returns a tensor on the INPUT's
    device. ``outlier_positions`` builds its shift vector with a device-less
    ``torch.arange``, so it cannot be handed a CUDA tensor; this is a one-off at weight
    load, so paying a device round trip is free and removes a class of failure that would
    otherwise only appear on a machine with a GPU.
    """
    from .quantize import (                       # local import: keeps w_format importable
        _pack_bits_lsb_first,                     # standalone and the primitives single-sourced
        outlier_positions,
        relidx7_index_bytes,
    )
    if npv <= 0:
        raise OMMXWFormatError("transcode_oindex_bitmap_to_relidx7 needs npv > 0")
    if group_size > RELIDX7_MAX_GROUP_SIZE:
        raise OMMXWFormatError(
            f"group_size={group_size} > {RELIDX7_MAX_GROUP_SIZE}: a position does not fit "
            f"in relidx7's 7 bits")
    device = oindex.device
    src = oindex.detach().to("cpu")
    want = (N, G, bitmap_index_bytes_for(group_size))
    if tuple(src.reshape(want).shape) != want:    # reshape itself raises on a size mismatch
        raise OMMXWFormatError(f"bitmap oindex plane is not reshapeable to {want}")
    pos = outlier_positions(src, N, G, npv, group_size, "bitmap")      # [N, G, npv] int64
    bits = (pos.unsqueeze(-1) >> torch.arange(7, dtype=pos.dtype)) & 1  # LSB-first, 7/slot
    out = _pack_bits_lsb_first(bits.reshape(N, G, npv * 7), relidx7_index_bytes(npv))
    return out.reshape(N, G, relidx7_index_bytes(npv)).contiguous().to(device)


def bitmap_index_bytes_for(group_size: int) -> int:
    """Bytes of flat bitmask per group — one bit per element, ``ceil(gs/8)``.

    A thin re-export of ``quantize.bitmap_index_bytes`` so this module's transcode helpers
    read without a second import at every call site; the arithmetic lives there.
    """
    from .quantize import bitmap_index_bytes
    return bitmap_index_bytes(group_size)


def degenerate_map_planes(N: int, G: int, device: Optional[torch.device] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The ``(map_scale, map_center)`` pair that MEANS ``outlier_map="none"``.

    ``quantize.py`` documents ``none`` as "degenerate instance ms=1, mc=0": the packer
    initialises exactly these constants, encodes the nibbles against them, and
    ``dequantize_ommx_weight`` substitutes exactly these when the planes are absent. The
    kernel computes ``(fp4/ms + mc - code) * scale``, which at ms=1, mc=0 is
    ``(fp4 - code) * scale`` — the ``none`` arithmetic, unchanged.

    f32 because ``sparse_correct`` dereferences both with ``data_ptr<float>()``.
    """
    ones = torch.ones((N, G), dtype=torch.float32, device=device)
    zeros = torch.zeros((N, G), dtype=torch.float32, device=device)
    return ones, zeros


# ════════════════════════════════════════════════════════════════════════════
# manifest read / validate / load
# ════════════════════════════════════════════════════════════════════════════

#: Printed by every version refusal. One string so the shard path, the index path and
#: the CLI cannot drift into three differently-worded instructions.
REPACK_HINT = ("python -m ommx_gpu_serve.linear.w_packer pack "
               "--input <source-checkpoint> --output <bundle>")


def _version_refusal(where: str, version: int) -> str:
    """Refusal message for a bundle this build cannot read. Names the version.

    A known-legacy version gets the REASON it is unreadable and the exact re-pack
    command; an unknown one gets the version numbers and the same command. Both refuse.
    There is deliberately no compatibility path: version 1 and version 2 differ in
    TENSOR NAMES, so "read it anyway" means either a load-time
    ``parameter not found`` from deep inside a vLLM loader, or — worse, if some future
    loader is lenient — a Linear left at its initialised weights.
    """
    known = OMMX_W_LEGACY_VERSIONS.get(version)
    if known is not None:
        return (f"{where}: this is an OMMX_W_SafeTensor VERSION {version} bundle, and "
                f"this build reads version {OMMX_W_FORMAT_VERSION} only. In version "
                f"{version} {known}. There is no in-place migration (the tensor names "
                f"themselves changed) — RE-PACK the source checkpoint:\n    "
                f"{REPACK_HINT}")
    return (f"{where}: unknown OMMX_W_SafeTensor version {version} — this build reads "
            f"version {OMMX_W_FORMAT_VERSION} only. Upgrade ommx_gpu_serve, or re-pack "
            f"the source checkpoint with this one:\n    {REPACK_HINT}")


def read_manifest(shard_path: str, st: Optional[SafeTensorsFile] = None) -> dict:
    """Parse and version-check the manifest embedded in one shard."""
    st = st or SafeTensorsFile(shard_path)
    md = st.metadata
    base = os.path.basename(shard_path)
    if md.get("format") != OMMX_W_FORMAT:
        raise OMMXWFormatError(
            f"{base}: file-level metadata 'format' is {md.get('format')!r}, not "
            f"{OMMX_W_FORMAT!r} — this is not an OMMX_W_SafeTensor shard")
    try:
        version = int(md.get("version", -1))
    except (TypeError, ValueError):
        raise OMMXWFormatError(
            f"{base}: metadata 'version' is {md.get('version')!r}, not an integer") from None
    if version != OMMX_W_FORMAT_VERSION:
        raise OMMXWFormatError(_version_refusal(base, version))
    if "manifest" not in md:
        raise OMMXWFormatError(f"{base}: metadata has no 'manifest' entry")
    try:
        man = json.loads(md["manifest"])
    except Exception as exc:
        raise OMMXWFormatError(f"{base}: manifest is not valid JSON ({exc})") from None
    for key in ("format", "version", "naming", "recipe", "tensors"):
        if key not in man:
            raise OMMXWFormatError(f"{base}: manifest is missing key {key!r}")
    if int(man["version"]) != OMMX_W_FORMAT_VERSION:
        raise OMMXWFormatError(
            f"{base}: manifest version {man['version']} disagrees with metadata "
            f"version {version}")
    check_naming(base, man["naming"])
    return man


def check_naming(where: str, declared: object) -> None:
    """Refuse a bundle whose declared plane-naming convention is not this build's.

    The block is what makes the format self-describing, so it has to be checked, not
    just carried: a bundle that DECLARES the version-1 convention while claiming
    version 2 would otherwise load its planes by the manifest's explicit ``name``
    fields and bind none of them.
    """
    want = naming_document()
    if not isinstance(declared, dict):
        raise OMMXWFormatError(
            f"{where}: 'naming' is {type(declared).__name__}, not an object")
    diffs = [f"{k}: {declared.get(k)!r} != {want[k]!r}"
             for k in want if declared.get(k) != want[k]]
    if diffs:
        raise OMMXWFormatError(
            f"{where}: the bundle declares a plane-naming convention this build does "
            f"not implement ({'; '.join(diffs)}). Re-pack the source checkpoint:\n    "
            f"{REPACK_HINT}")


def validate_shard(shard_path: str) -> dict:
    """Full cross-check of one shard: manifest <-> recipe <-> tensors actually present.

    Returns the manifest on success; raises :class:`OMMXWFormatError` naming the
    first disagreement otherwise. Checks, in order:

      1. container parses, metadata says ommx_w / version 2, and the declared plane
         naming convention is the one this build implements;
      2. the recipe itself is self-consistent (group size, npv, repr, dtype);
      3. every declared plane EXISTS with the declared shape and dtype;
      4. every declared shape MATCHES what the recipe derives from ``[N, K]``
         (catches a manifest edited to agree with a tampered tensor);
      5. the plane SET matches the recipe (no missing / no extra plane);
      6. every tensor in the file is claimed by exactly one manifest entry;
      7. declared ``packed_bytes`` / ``bits_per_weight`` match the bytes on disk.
    """
    st = SafeTensorsFile(shard_path)
    base = os.path.basename(shard_path)
    man = read_manifest(shard_path, st)
    recipe = Recipe.from_json(man["recipe"])

    claimed: Dict[str, str] = {}          # tensor in file -> owning manifest entry
    for name, ent in man["tensors"].items():
        kind = ent.get("kind")
        if kind == "passthrough":
            if name not in st.names:
                raise OMMXWFormatError(
                    f"{base}: manifest declares passthrough tensor {name!r} but the "
                    f"file does not contain it")
            _check_tensor(st, base, name, tuple(ent["shape"]), str(ent["dtype"]))
            claimed[name] = name
            continue
        if kind != "quantized":
            raise OMMXWFormatError(
                f"{base}: tensor {name!r} has unknown kind {kind!r} "
                f"(expected 'quantized' or 'passthrough')")
        shape = tuple(int(v) for v in ent["shape"])
        if len(shape) != 2:
            raise OMMXWFormatError(
                f"{base}: quantized tensor {name!r} declares shape {shape}, but only "
                f"2-D Linear weights [N, K] can be quantized")
        N, K = shape
        want = recipe.plane_layout(N, K)
        got = ent.get("planes", {})
        if set(got) != set(want):
            raise OMMXWFormatError(
                f"{base}: tensor {name!r} declares planes {sorted(got)} but recipe "
                f"{recipe} requires {sorted(want)} "
                f"(missing {sorted(set(want) - set(got))}, "
                f"extra {sorted(set(got) - set(want))})")
        total_bytes = 0
        for plane, (wshape, wdtype) in want.items():
            # The manifest may spell the plane's tensor name out, but it does NOT get
            # to choose it: the name is a function of (weight, plane) and a loader
            # derives it that way. Without this check a shard could carry v1-named
            # planes inside a v2 manifest, be perfectly self-consistent, and still
            # bind to nothing in a real engine — which is exactly the bug the version
            # bump exists to remove, so it must not be reachable through the manifest.
            want_pn = plane_name(name, plane)
            pn = str(got[plane].get("name", want_pn))
            if pn != want_pn:
                raise OMMXWFormatError(
                    f"{base}: tensor {name!r} plane {plane!r} is stored as {pn!r}, but "
                    f"format version {OMMX_W_FORMAT_VERSION} names it {want_pn!r} "
                    f"({naming_document()['plane_name']}). Re-pack the source "
                    f"checkpoint:\n    {REPACK_HINT}")
            dshape = tuple(int(v) for v in got[plane]["shape"])
            ddtype = str(got[plane]["dtype"])
            if dshape != tuple(wshape) or ddtype != wdtype:
                raise OMMXWFormatError(
                    f"{base}: tensor {name!r} plane {plane!r} is declared as "
                    f"{ddtype}{list(dshape)} but the recipe requires "
                    f"{wdtype}{list(wshape)} for a [{N}, {K}] weight")
            if pn not in st.names:
                raise OMMXWFormatError(
                    f"{base}: tensor {name!r} plane {plane!r} declares tensor {pn!r}, "
                    f"which is not in the file")
            _check_tensor(st, base, pn, tuple(wshape), wdtype)
            if pn in claimed:
                raise OMMXWFormatError(
                    f"{base}: tensor {pn!r} is claimed by both {claimed[pn]!r} and "
                    f"{name!r}")
            claimed[pn] = name
            total_bytes += st.nbytes(pn)
        if int(ent.get("packed_bytes", -1)) != total_bytes:
            raise OMMXWFormatError(
                f"{base}: tensor {name!r} declares packed_bytes="
                f"{ent.get('packed_bytes')} but its planes occupy {total_bytes} bytes")
        want_bpw = bits_per_weight(recipe, N, K, total_bytes)
        if abs(float(ent.get("bits_per_weight", -1.0)) - want_bpw) > 1e-9:
            raise OMMXWFormatError(
                f"{base}: tensor {name!r} declares bits_per_weight="
                f"{ent.get('bits_per_weight')} but {total_bytes} bytes over "
                f"{N * K} weights is {want_bpw:.6f}")

    undeclared = [n for n in st.names if n not in claimed]
    if undeclared:
        raise OMMXWFormatError(
            f"{base}: {len(undeclared)} tensor(s) present but not described by the "
            f"manifest: {sorted(undeclared)[:8]}")
    return man


def _check_tensor(st: SafeTensorsFile, base: str, name: str,
                  shape: Iterable[int], dtype: str) -> None:
    got_shape = tuple(st.shape(name))
    got_dtype = st.dtype_name(name)
    want = tuple(int(v) for v in shape)
    if got_shape != want or got_dtype != dtype:
        raise OMMXWFormatError(
            f"{base}: tensor {name!r} is {got_dtype}{list(got_shape)} on disk but the "
            f"manifest declares {dtype}{list(want)}")


def validate_aux_files(bundle_dir: str, index: dict) -> dict:
    """Check the ``aux_files`` record of an index against the directory on disk.

    WHAT IS ENFORCED, and why exactly this much:

      * the record EXISTS and names a known policy — a version-2 bundle that does not
        say whether it carries its model files is malformed, not "probably fine";
      * every file the index says it copied is actually THERE, and a REQUIRED one is
        non-empty. A bundle whose ``config.json`` was lost (or truncated to 0 bytes by
        a full disk) is unloadable, and the failure otherwise surfaces as an opaque
        transformers error about a missing architecture;
      * under policy ``copy``, every one of :data:`REQUIRED_AUX_FILES` is in the copied
        list, and ``self_contained`` agrees with that fact rather than being a free
        text claim.

    WHAT IS DELIBERATELY *NOT* ENFORCED: the recorded ``bytes``/``sha256`` are NOT
    compared against the files on disk. Editing ``config.json`` in a bundle (rope
    scaling, ``max_position_embeddings``, ``torch_dtype``) is a normal, supported thing
    to do, and a validator that failed on it would train operators to pass a
    ``--no-validate`` flag — the exact reflex this format is built to avoid. The digests
    are PROVENANCE: ``w_packer verify`` reports which aux files changed since packing,
    as information, and the bundle stays valid.
    """
    aux = index.get("aux_files")
    if not isinstance(aux, dict):
        raise OMMXWFormatError(
            f"{INDEX_FILENAME}: missing or malformed 'aux_files' record. A version "
            f"{OMMX_W_FORMAT_VERSION} bundle must state which non-weight checkpoint "
            f"files it carries (policy {list(AUX_POLICIES)}); re-pack it:\n    "
            f"{REPACK_HINT}")
    policy = aux.get("policy")
    if policy not in AUX_POLICIES:
        raise OMMXWFormatError(
            f"{INDEX_FILENAME}: aux_files.policy is {policy!r}, not one of "
            f"{list(AUX_POLICIES)}")
    copied = aux.get("copied", [])
    if not isinstance(copied, list):
        raise OMMXWFormatError(f"{INDEX_FILENAME}: aux_files.copied is not a list")

    names: List[str] = []
    for ent in copied:
        if not isinstance(ent, dict) or "name" not in ent:
            raise OMMXWFormatError(
                f"{INDEX_FILENAME}: aux_files.copied entry {ent!r} has no 'name'")
        name = str(ent["name"])
        # Flat names only. A path separator here would let an index point outside the
        # bundle, and every HF checkpoint file that matters lives at the top level.
        if os.path.basename(name) != name or name in (".", ".."):
            raise OMMXWFormatError(
                f"{INDEX_FILENAME}: aux_files entry {name!r} is not a plain file name "
                f"in the bundle directory")
        path = os.path.join(bundle_dir, name)
        if not os.path.isfile(path):
            raise OMMXWFormatError(
                f"{INDEX_FILENAME}: aux_files lists {name!r} as copied into the bundle, "
                f"but {path} does not exist. The bundle is not the self-contained model "
                f"directory it claims to be.")
        if name in REQUIRED_AUX_FILES and os.path.getsize(path) == 0:
            raise OMMXWFormatError(
                f"{bundle_dir}: {name!r} is required for this directory to load as a "
                f"model and it is EMPTY (0 bytes).")
        names.append(name)

    missing_required = [f for f in REQUIRED_AUX_FILES if f not in names]
    if policy == "copy" and missing_required:
        raise OMMXWFormatError(
            f"{INDEX_FILENAME}: aux_files.policy='copy' but the required file(s) "
            f"{missing_required} were not copied. A bundle without them cannot be "
            f"resolved as a model by transformers or vLLM.")
    want_self_contained = (policy == "copy" and not missing_required)
    if bool(aux.get("self_contained")) != want_self_contained:
        raise OMMXWFormatError(
            f"{INDEX_FILENAME}: aux_files.self_contained="
            f"{aux.get('self_contained')!r} but policy={policy!r} with required files "
            f"{'present' if not missing_required else 'MISSING ' + str(missing_required)}"
            f" means {want_self_contained}")
    return aux


def validate_bundle(bundle_dir: str) -> dict:
    """Validate every shard of a bundle, the index, and the aux-file record.

    Returns the index document. On top of :func:`validate_shard` per shard this
    checks the index's format/version/naming, that the non-weight model files the
    index claims are really in the directory (:func:`validate_aux_files`), that no
    shard's recipe disagrees with the bundle recipe, and that the ``weight_map`` and
    the shard manifests describe exactly the same tensor set.
    """
    index_path = os.path.join(bundle_dir, INDEX_FILENAME)
    if not os.path.isfile(index_path):
        raise OMMXWFormatError(
            f"{bundle_dir}: no {INDEX_FILENAME} — not an OMMX_W_SafeTensor bundle")
    with open(index_path, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    if index.get("format") != OMMX_W_FORMAT:
        raise OMMXWFormatError(f"{INDEX_FILENAME}: format is {index.get('format')!r}")
    try:
        idx_version = int(index.get("version", -1))
    except (TypeError, ValueError):
        raise OMMXWFormatError(
            f"{INDEX_FILENAME}: 'version' is {index.get('version')!r}, "
            f"not an integer") from None
    if idx_version != OMMX_W_FORMAT_VERSION:
        raise OMMXWFormatError(_version_refusal(INDEX_FILENAME, idx_version))
    if "naming" not in index:
        raise OMMXWFormatError(f"{INDEX_FILENAME}: missing key 'naming'")
    check_naming(INDEX_FILENAME, index["naming"])
    validate_aux_files(bundle_dir, index)
    recipe = Recipe.from_json(index["recipe"])
    shards = sorted(set(index["weight_map"].values()))
    n_described = 0
    for shard in shards:
        path = os.path.join(bundle_dir, shard)
        if not os.path.isfile(path):
            raise OMMXWFormatError(
                f"{INDEX_FILENAME}: references shard {shard!r}, which does not exist")
        man = validate_shard(path)
        shard_recipe = Recipe.from_json(man["recipe"])
        if shard_recipe != recipe:
            raise OMMXWFormatError(
                f"{shard}: recipe {shard_recipe} disagrees with the bundle recipe "
                f"{recipe} — shards packed with different recipes cannot be loaded as "
                f"one model")
        for name in man["tensors"]:
            if index["weight_map"].get(name) != shard:
                raise OMMXWFormatError(
                    f"{INDEX_FILENAME}: tensor {name!r} maps to "
                    f"{index['weight_map'].get(name)!r} but is stored in {shard!r}")
        n_described += len(man["tensors"])
    n_index = len(index["weight_map"])
    if n_index != n_described:
        raise OMMXWFormatError(
            f"{INDEX_FILENAME}: weight_map has {n_index} entries but the shards "
            f"describe {n_described} tensors")
    return index


def load_weight(shard_path: str, weight_name: str,
                manifest: Optional[dict] = None) -> torch.Tensor:
    """Read one quantized weight back as a dequantized float32 ``[N, K]`` tensor.

    This is the CPU oracle, not the serving path: it reconstructs exactly what the
    CUDA kernel is supposed to compute from the same planes. On a GPU host the kernel
    would consume ``load_planes()`` directly (``decode_base`` wants ``scale``/``zp``
    as fp32 ``[N, G]``, so the bf16 zero-point is upcast — bf16 -> f32 is exact).

    UNVERIFIED (no GPU this session): that the kernel and this function agree. That
    is what ``csrc/linear/test_ommx_linear_parity.py`` measures, and it needs a device.
    """
    planes, recipe, N, K = load_planes(shard_path, weight_name, manifest)
    return dequantize_ommx_weight(
        code=planes["code"], scale_exp=planes["scale_exp"],
        zp=planes["zp"].to(torch.float32), N=N, K=K,
        group_size=recipe.group_size, npv=recipe.npv,
        oindex=planes.get("oindex"), ocode=planes.get("ocode"),
        map_scale=planes.get("map_scale"), map_center=planes.get("map_center"),
        outlier_repr=recipe.outlier_repr)


def load_planes(shard_path: str, weight_name: str, manifest: Optional[dict] = None
                ) -> Tuple[Dict[str, torch.Tensor], Recipe, int, int]:
    """Read the raw planes of one quantized weight -> ``(planes, recipe, N, K)``."""
    st = SafeTensorsFile(shard_path)
    man = manifest or read_manifest(shard_path, st)
    ent = man["tensors"].get(weight_name)
    if ent is None:
        raise OMMXWFormatError(
            f"{os.path.basename(shard_path)}: no tensor {weight_name!r} in the manifest")
    if ent.get("kind") != "quantized":
        raise OMMXWFormatError(
            f"{weight_name!r} is {ent.get('kind')!r}, not a quantized weight; read it "
            f"with SafeTensorsFile.tensor()")
    recipe = Recipe.from_json(man["recipe"])
    N, K = (int(v) for v in ent["shape"])
    # The plane's tensor name is a FUNCTION of (weight, plane), never a manifest
    # choice — the same rule ``validate_shard`` enforces. Reading it back through the
    # manifest's ``name`` field would make this loader the one place where a v1-named
    # plane inside a v2 manifest still resolves: the validator would refuse the shard
    # while ``load_weight`` returned plausible numbers off it, and "the CPU oracle read
    # it fine" is exactly the evidence nobody should be able to produce for a bundle no
    # engine can bind. So the name is derived, and a manifest that declares a different
    # one is refused here too rather than quietly honoured.
    planes: Dict[str, torch.Tensor] = {}
    for p in ent["planes"]:
        want_pn = plane_name(weight_name, p)
        declared = str(ent["planes"][p].get("name", want_pn))
        if declared != want_pn:
            raise OMMXWFormatError(
                f"{os.path.basename(shard_path)}: tensor {weight_name!r} plane {p!r} is "
                f"declared as {declared!r}, but format version {OMMX_W_FORMAT_VERSION} "
                f"names it {want_pn!r} ({naming_document()['plane_name']}). Re-pack the "
                f"source checkpoint:\n    {REPACK_HINT}")
        planes[p] = st.tensor(want_pn)
    return planes, recipe, N, K
