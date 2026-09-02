# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""``ommx_w`` — the vLLM weight-quant LinearMethod for ``OMMX_W_SafeTensor`` bundles.

This is the SERVING half of ``OMMX_Linear`` (paper §3.3 claim C3, Fig 6 claim D5). The
OFFLINE half — the quantizer, the on-disk format and the packer CLI — already ships in
:mod:`ommx_gpu_serve.linear`; this module is what makes a packed bundle executable inside
a vLLM engine::

    HF checkpoint --[ommx_gpu_serve.linear.w_packer]--> OMMX_W_SafeTensor bundle
                  --[OMMXWConfig / OMMXWLinearMethod (here)]--> csrc/linear kernels

VERIFICATION STATUS — read this before quoting anything from this file
----------------------------------------------------------------------
Authored on a host with NO GPU, NO vLLM and no network. Precisely:

  * CPU-VERIFIED (``tests/test_linear_method.py``: no GPU, ``vllm`` stubbed into
    ``sys.modules``) — ``register_ommx_w()`` is importable/callable/idempotent with no
    vLLM; bundle discovery and manifest/recipe parsing against a REAL bundle produced by
    the packer CLI; every refusal path (no bundle, foreign checkpoint, a recipe the
    kernel cannot read, a failed JIT build) raises with its actionable message; the
    plane shapes ``create_weights`` allocates equal the shapes the manifest declares;
    importing this module does not import vLLM.
  * **UNVERIFIED (no GPU this session)** — that the kernel handle built here executes at
    all; that ``apply()``'s argument order matches the compiled ``ommx_linear`` ABI
    beyond what ``csrc/linear/test_ommx_linear_parity.py`` demonstrates; that vLLM's
    weight loader populates the parameters ``create_weights`` registers (the bundle plane
    name IS the parameter name — see the naming section below);
    that TP sharding of the packed planes is correct; and every latency question. Not
    one line below has run against a device.

NO SILENT FALLBACK (law #5)
---------------------------
Every failure here RAISES. There is deliberately no "quantization unavailable, use bf16"
path: this engagement exists because the attention path had one, and MEASURED_FACTS §2
records the cost — an ``--attention-backend CUSTOM`` run whose output was BYTE-IDENTICAL
to bf16 FlashAttention (228/228 chars) while the bench wrote the timings down under the
OMMX label. A weight arm that degrades to a bf16 Linear and is still reported as
``ommx_w`` is the same defect with a different plane count.

Plane names ARE parameter names (closed; kept documented because it was not always so)
--------------------------------------------------------------------------------------
``w_format.plane_name`` stores a plane as ``<module>.ommx_code`` — the module path of the
tensor it replaced, then the plane attribute, with no ``.weight`` infix. That is exactly
what ``create_weights`` registers on the Linear, and exactly how vLLM's loaders match a
checkpoint tensor name against ``dict(model.named_parameters())`` (the same reason AWQ
ships ``<module>.qweight`` rather than ``<module>.weight.qweight``). So there is NO
translation step between the bundle and the engine.

``OMMX_W_SafeTensor`` **v1** did keep the infix, which bound to no parameter and failed a
real engine run at load. v1 is therefore refused BY NAME by ``w_format``'s validator with
a re-pack instruction rather than silently accepted, so no v1 name can reach a loader.
:func:`bundle_to_param_name` and :func:`remap_bundle_weight_names` survive as identity
functions over v2 names and are retained deliberately: they are the single place the
convention is asserted, and ``tests/test_linear_method.py`` pins that identity so the
invariant fails loudly if either side ever drifts apart again.

Serving a PAPER-FORMAT (bitmap) bundle — the opt-in load-time transcode
-----------------------------------------------------------------------
The paper's own GPU weight format (claim B4: "positions are stored as a flat bitmask,
N bits per group"; claim B6's bundle listing no range-map parameters) is
``outlier_repr="bitmap", outlier_map="none"``, and at ``gs=64, npv=4`` that is exactly
the 3.6250 bits/weight of Table 1's AvgBits 3.63 (claim E2). The shipped kernel cannot
read it — see :data:`KERNEL_OUTLIER_REPRS` — so it is refused at config time.

``OMMX_W_TRANSCODE=1`` (or ``quantization_config["ommx_w_transcode"]``) makes such a
bundle SERVABLE by re-encoding the position plane ONCE, at
``process_weights_after_loading``, into the relidx7 slot stream the kernel decodes, and
materialising the degenerate ms=1 / mc=0 range map that ``outlier_map="none"`` already
means. Both steps are lossless — see the long WHY block above
``w_format.plan_transcode`` for the two facts they rest on — and both are proven on CPU:
the transcoded plane is BYTE-IDENTICAL to packing the same weight with
``outlier_repr="relidx7"`` directly, and the dequantized weight is unchanged.

WHAT IT DOES NOT BUY, stated here because it is the only thing a reader can get wrong:
the bundle's ON-DISK footprint is the paper's 3.6250 b/wt; the RESIDENT footprint after
transcode is the relidx7 one, 4.1250 b/wt at ``gs=64, npv=4`` (the position stream drops
1.0 -> 0.5 b/wt, and two f32 map planes appear, +1.0). **Serving a 3.63 b/wt bundle is
not serving at 3.63 b/wt in HBM.** Both numbers are computed by
:class:`~ommx_gpu_serve.linear.w_format.TranscodePlan` from one accounting path, printed
on the first transcoded layer, and reported by :func:`ommx_w_fire_stats`.

It is OPT-IN, not a default: a format change under the operator's feet is the failure
mode this whole release exists to remove. With the flag unset the bundle is refused exactly as
before — the code path, not merely the outcome; the refusal TEXT gained lines naming the
flag, both bit budgets and the native fix.

Kernel call convention
----------------------
Taken from ``csrc/linear/test_ommx_linear_parity.py``, the only worked example::

    y = mod.decode_base(x, code, scale, zp, N, K, gs, transB, "i2f4", split[, scale_exp])
    y = mod.prefill_wmma(x, code, scale, zp, N, M, K, gs, transB, "i2f4", split)
    mod.sparse_correct(y, x, None, code, scale, oindex, ocode, map_scale, map_center,
                       N, 1, K, gs, npv, block, "i2f4")          # M == 1 ONLY

``sparse_correct`` has TWO call conventions, not one, and the parity gate only ever
showed the first. Read off ``csrc/linear/ommx_linear.cu`` (see
``csrc/linear/SPARSE_CORRECT_ABI.md`` for the full derivation)::

    void sparse_correct(out_or_C, A_opt, At_opt, code, scale, oindex, ocode,
                        map_scale, map_center, N, M, K, vector_length, npv, B, fmt)

    M == 1 :  A_opt  = bf16 [1, K] activation ; out_or_C = fp32 [1, N] (in-place +=)
    M >  1 :  At_opt = bf16 [K, M] activation TRANSPOSED, contiguous
              out_or_C = bf16 [M, N] (in-place +=)

Passing the M==1 argument list at M>1 aborts on ``"M>1 correct needs At"``; passing a
bf16 ``C`` where fp32 is wanted (or vice versa) aborts inside ``data_ptr<T>()``. Both
aborts are loud, so neither can corrupt a number — but both take out every prefill step
of any outlier-carrying recipe, which is what they did until this release.

``scale``/``zp`` are fp32 ``[N, G]`` there, while the bundle stores an int8 E8M0 exponent
and a bf16 zero point — :meth:`process_weights_after_loading` materialises the fp32 twins
(``2**exp`` is exact, and bf16 -> f32 is exact), which is also what
``w_format.load_weight``'s docstring states the kernel wants.

Import contract: imports with NO vllm, NO triton and NO GPU. Every vLLM symbol is imported
inside a function, so ``backend.py`` stays the single module in this package that needs
vLLM at import time.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from ...linear.w_format import (
    INDEX_FILENAME,
    OMMXWFormatError,
    Recipe,
    TranscodePlan,
    degenerate_map_planes,
    plan_transcode,
    st_dtype_name,
    transcode_oindex_bitmap_to_relidx7,
    validate_bundle,
)

#: The vLLM ``--quantization`` name this module registers. ``bench_e2e_a100.py`` and the
#: arm plan in that file spell it out, so it is a CONTRACT, not a preference.
OMMX_W_METHOD_NAME = "ommx_w"

#: Format tag handed to the kernel. The parity gate calls ``decode_base(..., "i2f4", ...)``
#: even for an outlier-free weight (``opct=0``), so this tag names the BUNDLE CLASS — INT2
#: base with an FP4 extension lane — not the presence of outliers in a given tensor.
KERNEL_FMT = "i2f4"

#: Outlier position encodings the WEIGHT-side CUDA kernel can read. ``quantize.py`` can
#: also emit ``bitmap``; ``csrc/linear/ommx_linear.cu``'s ``sparse_correct`` consumes the
#: relidx7 slot stream (``relidx7_slot_pos`` / ``relidx7_slot_delta``) and there is NO
#: bitmap reader on the weight side. Packing one and serving it would be a silent
#: mis-decode, so it is refused at config time instead.
#:
#: This tuple names what the KERNEL DECODES and nothing else. It is deliberately NOT
#: widened by the opt-in transcode below: after a transcode the kernel is still reading
#: relidx7 — the bitmap plane never reaches it — and widening this would turn a statement
#: about the compiled kernel into a statement about the Python loader. The day
#: ``csrc/linear/BITMAP_READER_SPEC.md`` is implemented, "bitmap" is added HERE and the
#: transcode becomes unnecessary rather than merely cheaper.
KERNEL_OUTLIER_REPRS: Tuple[str, ...] = ("relidx7",)

#: Outlier range-map modes the kernel can read. ``sparse_correct`` takes ``map_scale`` and
#: ``map_center`` as required positional arguments and the idx-space map is the only form
#: the parity gate exercises, so ``outlier_map="none"`` (which omits both planes) is
#: refused rather than passed as a guess.
KERNEL_OUTLIER_MAPS: Tuple[str, ...] = ("idx_range",)

#: Environment opt-in for the load-time transcode. Parsed with :func:`_env_on`, so any of
#: 1/true/on/yes enables it and an unset/0/false/off/no leaves the refusal exactly as it
#: was. One name, so the log line, the refusal message and the docs cannot drift.
TRANSCODE_ENV = "OMMX_W_TRANSCODE"

#: ``quantization_config`` keys that request the same thing, for a bundle that wants to
#: ship the decision in its own ``config.json`` rather than in the operator's shell.
TRANSCODE_CONFIG_KEYS: Tuple[str, ...] = ("ommx_w_transcode", "transcode")

#: vLLM parameter attribute per plane. Deliberately the SAME suffix the bundle uses, so
#: that ``plane_name(w, p) == "<module>." + param_name_for_plane(p)`` holds EXACTLY: a
#: bundle tensor name and the parameter it binds to are one string, not two conventions
#: joined by a remap. See the naming section of the module docstring for why v1 was not.
PLANE_PARAM_PREFIX = "ommx_"

#: vLLM fuses these projections into one Linear. Mapping a fused module prefix back to the
#: checkpoint tensors it was built from is the only way to ask the manifest whether a
#: layer is quantized: the bundle names q/k/v separately because the CHECKPOINT does.
FUSED_MODULES: Dict[str, Tuple[str, ...]] = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}

#: Route-evidence sentinel, same file the attention backend writes (``backend._FIRE_FILE``)
#: so ONE ``$OMMX_FIRE_FILE`` carries the proof for both halves of a combined arm. Read
#: lazily (not at import) so a test can point it somewhere writable.
_DEFAULT_FIRE_FILE = "/tmp/ommx_route_fired.log"

# Tags written here are informational to ``bench_e2e_a100.py``: its ``FIRED_TAGS`` is an
# explicit whitelist of ATTENTION decode routes, so these land in ``other_tags`` and change
# no verdict. That is intentional — the linear path proves itself through
# ``ommx_w_fire_stats()``, and silently widening the attention verdict would be a
# fair-compare change. The one rule that IS load-bearing: never end a tag in ``_DEAD`` or
# ``_NOFIRE`` unless it really is a route failure (the bench matches those by SUFFIX).
_FIRE_SEEN: set = set()

#: Process-wide firing counters — the linear path's answer to the attention route
#: sentinels. A bench MUST assert ``apply_calls > 0`` before quoting an ``ommx_w`` number;
#: otherwise it cannot tell an OMMX linear from any other linear that produced logits.
_FIRE: Dict[str, int] = {
    "layers_created": 0,     # create_weights() calls (one per quantized Linear)
    "layers_ready": 0,       # process_weights_after_loading() calls that completed
    "kernel_builds": 0,      # successful JIT builds (expect exactly 1 per process)
    "apply_calls": 0,        # apply() entries
    "decode_calls": 0,       # decode_base() dispatches
    "prefill_calls": 0,      # prefill_wmma() dispatches
    "outlier_calls": 0,      # sparse_correct() dispatches
    "tokens": 0,             # rows of x seen by apply()
    "transcoded_layers": 0,  # layers whose planes were transcoded at load (opt-in)
}

#: Model-level byte accounting for the opt-in transcode, summed over every layer that
#: went through it. Two totals, never one: the whole point of the feature's honesty
#: caveat is that the on-disk bundle and the resident planes are DIFFERENT sizes, so the
#: process keeps both and :func:`ommx_w_fire_stats` reports both. ``freed`` is the source
#: position plane that was released after re-encoding — without it the resident figure
#: would be an understatement of what is actually in HBM.
_TRANSCODE: Dict[str, Any] = {
    "plan": None,            # TranscodePlan.to_json() of the bundle's recipe, or None
    "layers": 0,
    "on_disk_bytes": 0,      # planes as stored in the bundle, for the layers transcoded
    "resident_bytes": 0,     # BUNDLE PLANES only, for the same layers (see twin_bytes)
    "freed_bytes": 0,        # bitmap position plane storage released after re-encoding
    # The fp32 scale/zp twins process_weights_after_loading materialises. They are
    # DELIBERATELY outside on_disk_bytes/resident_bytes — they exist identically with and
    # without a transcode, so counting them there would inflate the delta without changing
    # it — but they ARE in HBM and the kernel reads them IN PLACE OF the int8 scale_exp /
    # bf16 zp planes it was handed. Excluding them from the delta is honest; leaving them
    # unreported anywhere would make ``resident_bytes`` readable as an HBM total, which it
    # is not. So they are reported, separately and always.
    "twin_bytes": 0,
}

#: Cached JIT-built kernel module + the failure that produced it, if any. Built ONCE per
#: process in ``process_weights_after_loading`` (law #6: seed the kernel at weight-load
#: time — a lazy first-call build is never reached on vLLM's traced/compiled path).
_KERNEL: List[Any] = [None]
_KERNEL_ERROR: List[Optional[str]] = [None]

#: Registration latches. ``_REGISTERED_AS`` is the name actually registered, or None when
#: this process has no vLLM (the no-op path).
_REGISTERED = False
_REGISTERED_AS: Optional[str] = None


class OMMXWError(RuntimeError):
    """An ``ommx_w`` serving-path refusal. The message always names the fix."""


class OMMXWBundleUnconfigured(OMMXWError):
    """NOTHING named a bundle — as opposed to something naming a bad one.

    The distinction is load-bearing and must not be re-derived from message text. Only
    this case is DEFERRABLE: it is what ``from_config`` sees when vLLM asks for the
    quantization config from inside ``VllmConfig``'s own construction, before the model
    path is reachable, and the same lookup succeeds later during model loading. Every
    other refusal — a path that is not a bundle, a recipe the kernel cannot decode, a
    foreign ``quant_method`` — is terminal and must surface at once, because deferring it
    would replace a precise message with a confusing one raised somewhere else (or, if
    no layer is ever dispatched, with none at all).
    """


# ════════════════════════════════════════════════════════════════════════════
# safetensors dtype -> torch dtype
# ════════════════════════════════════════════════════════════════════════════

def _st_to_torch_map() -> Dict[str, torch.dtype]:
    """Invert the PUBLIC ``w_format.st_dtype_name``.

    Built by inversion rather than hand-copied so this map cannot drift away from the
    one the packer wrote the manifest with; a hand-copied second table is exactly how a
    plane ends up allocated as the wrong dtype and silently reinterpreted.
    """
    out: Dict[str, torch.dtype] = {}
    for dt in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
               torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.bool):
        try:
            out[st_dtype_name(dt)] = dt
        except Exception:                      # noqa: BLE001 - dtype not in the format
            continue
    return out


_ST_TO_TORCH = _st_to_torch_map()


# ════════════════════════════════════════════════════════════════════════════
# firing evidence
# ════════════════════════════════════════════════════════════════════════════

def _fire_file() -> str:
    return os.environ.get("OMMX_FIRE_FILE", _DEFAULT_FIRE_FILE)


def _fire(tag: str, detail: str = "") -> None:
    """Record one-time route evidence for the LINEAR path.

    Same file format as ``backend._ommx_route_evidence`` (``TAG rank= pid= detail``) so a
    single sentinel file describes a combined ``ommx_w + CUSTOM`` arm. Best-effort: an
    unwritable sentinel must never take down a serving step, and the in-process counters
    in ``_FIRE`` remain authoritative either way.
    """
    key = f"{tag}.{os.getpid()}"
    if key in _FIRE_SEEN:
        return
    _FIRE_SEEN.add(key)
    rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"
    line = f"{tag} rank={rank} pid={os.getpid()} {detail}".rstrip()
    path = _fire_file()
    if rank not in ("0", ""):
        base, ext = os.path.splitext(path)
        path = f"{base}.r{rank}{ext}"
    try:
        with open(path, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    print(f"[ommx_w] {line}", file=sys.stderr, flush=True)


def ommx_w_fire_stats() -> Dict[str, Any]:
    """Firing counters for the OMMX weight-quant linear path.

    The linear-path counterpart of the attention backend's route sentinels: a bench that
    reports an ``ommx_w`` timing without asserting ``apply_calls > 0`` (and, for a recipe
    with outliers, ``outlier_calls > 0``) has not shown that the OMMX linear kernel ran
    at all. ``kernel`` carries the compiled module's own ``fire_stats()`` when a kernel
    has been built — the in-kernel counters, which cannot be faked from Python.
    """
    out: Dict[str, Any] = dict(_FIRE)
    # BOTH byte totals, always. A caller that wants to quote a bits/weight number for a
    # transcoded bundle must be unable to reach for the on-disk one by accident.
    out["transcode"] = dict(_TRANSCODE)
    out["kernel_built"] = _KERNEL[0] is not None
    out["kernel_error"] = _KERNEL_ERROR[0]
    out["fire_file"] = _fire_file()
    if _KERNEL[0] is not None:
        try:
            out["kernel"] = dict(_KERNEL[0].fire_stats())
        except Exception as exc:               # noqa: BLE001
            out["kernel"] = f"unavailable ({type(exc).__name__}: {exc})"
    return out


def ommx_w_health() -> Dict[str, Any]:
    """One-call verdict: did the OMMX linear path actually serve in this process?

    ``ok`` is True only when a kernel was built, at least one layer finished
    ``process_weights_after_loading`` and at least one ``apply()`` ran. ``reason`` names
    the first thing missing, so a bench can print it verbatim instead of guessing.
    """
    stats = ommx_w_fire_stats()
    reasons: List[str] = []
    if not stats["kernel_built"]:
        reasons.append(f"no ommx_linear kernel was built ({_KERNEL_ERROR[0] or 'never attempted'})")
    if not stats["layers_ready"]:
        reasons.append("no layer completed process_weights_after_loading")
    if not stats["apply_calls"]:
        reasons.append("OMMXWLinearMethod.apply() never ran — these are not OMMX linear timings")
    return dict(ok=not reasons, reason="; ".join(reasons), registered_as=_REGISTERED_AS,
                stats=stats)


def _reset_state_for_tests() -> None:
    """Clear every process latch. Called ONLY by ``tests/test_linear_method.py``.

    Module-level latches (registration, the kernel handle, the fire counters) are what
    make this module idempotent in a worker process; they also make test order matter,
    so the reset is explicit and named rather than done by poking globals from outside.
    """
    global _REGISTERED, _REGISTERED_AS
    _REGISTERED = False
    _REGISTERED_AS = None
    _KERNEL[0] = None
    _KERNEL_ERROR[0] = None
    _FIRE_SEEN.clear()
    for k in _FIRE:
        _FIRE[k] = 0
    _TRANSCODE.update(plan=None, layers=0, on_disk_bytes=0, resident_bytes=0,
                      freed_bytes=0, twin_bytes=0)
    _CLASSES.clear()


# ════════════════════════════════════════════════════════════════════════════
# bundle discovery + manifest parsing  (pure python — no vLLM, no GPU)
# ════════════════════════════════════════════════════════════════════════════

def _pack_hint(where: str) -> str:
    """The actionable half of every "no usable bundle" message."""
    return (
        f"{where}\n"
        f"  An ommx_w checkpoint is an OMMX_W_SafeTensor BUNDLE — a directory of "
        f"ommx_w-000NN-of-000MM.safetensors shards plus {INDEX_FILENAME}. Produce one "
        f"with the offline packer:\n"
        f"      python -m ommx_gpu_serve.linear.w_packer pack \\\n"
        f"          --input  <hf safetensors checkpoint dir> \\\n"
        f"          --output <bundle dir> --group-size 64 --outlier-pct 0.0625\n"
        f"      python -m ommx_gpu_serve.linear.w_packer verify --bundle <bundle dir>\n"
        f"  Then point the engine at it: --model <bundle dir> (or set "
        f"OMMX_W_BUNDLE=<bundle dir>, or pass {{\"bundle\": \"<bundle dir>\"}} in the "
        f"checkpoint's quantization_config).\n"
        f"  With no model to hand, 'python -m ommx_gpu_serve.linear.w_packer "
        f"make-synthetic --output <dir>' writes a tiny Llama-shaped checkpoint to pack.\n"
        f"  NOTE: 'pack' writes ONLY the shards and {INDEX_FILENAME}. To pass the bundle "
        f"as --model you must also copy the source checkpoint's config.json / "
        f"generation_config.json / tokenizer files into it, or vLLM fails in "
        f"AutoConfig.from_pretrained before any OMMX code runs."
    )


def is_ommx_w_bundle(path: Optional[str]) -> bool:
    """True iff ``path`` is a directory holding an OMMX_W index. Cheap, no parsing."""
    return bool(path) and os.path.isfile(os.path.join(str(path), INDEX_FILENAME))


def _current_vllm_model_path() -> Optional[str]:
    """The model path of the vLLM config being built, if one is in scope.

    vLLM's ``QuantizationConfig.from_config`` receives only the ``quantization_config``
    dict, never the model path, so the bundle would otherwise be undiscoverable in the
    ordinary case where the bundle IS the checkpoint the user passed to ``--model``.
    ``get_current_vllm_config()`` is the supported way to reach it during engine
    construction; guarded end to end because it is optional (absent in a stub, and it
    raises when no config context is active).
    """
    try:
        from vllm.config import get_current_vllm_config
    except Exception:                          # noqa: BLE001 - no vLLM / older layout
        return None
    try:
        return str(get_current_vllm_config().model_config.model)
    except Exception:                          # noqa: BLE001 - no active config context
        return None


def resolve_bundle_dir(explicit: Optional[str] = None,
                       quant_config: Optional[Dict[str, Any]] = None) -> str:
    """Locate the OMMX_W bundle, in a fixed and reportable order.

        1. ``explicit`` (a direct caller / a test),
        2. ``quant_config['bundle' | 'bundle_dir' | 'ommx_w_bundle']`` — the key an
           ommx_w bundle's own ``config.json`` can carry,
        3. ``$OMMX_W_BUNDLE`` — the operator override, and the only way to serve a
           bundle that lives beside (not at) the ``--model`` path,
        4. the model path vLLM is currently building for, when it IS a bundle.

    Raises :class:`OMMXWError` naming the packer CLI when none of them yields a bundle.
    A candidate that exists but is NOT a bundle is reported separately from "nothing was
    configured": those are different mistakes with different fixes.
    """
    cand: List[Tuple[str, str]] = []
    if explicit:
        cand.append(("explicit argument", str(explicit)))
    for key in ("bundle", "bundle_dir", "ommx_w_bundle"):
        val = (quant_config or {}).get(key)
        if val:
            cand.append((f"quantization_config[{key!r}]", str(val)))
    env = os.environ.get("OMMX_W_BUNDLE", "").strip()
    if env:
        cand.append(("$OMMX_W_BUNDLE", env))

    for source, path in cand:
        if is_ommx_w_bundle(path):
            return os.path.abspath(path)
    # The vLLM model path is consulted LAST and only if nothing else matched, because
    # reaching it imports ``vllm.config`` — a caller that already knows its bundle (a
    # bench, a test, a packer host) must not pay a vLLM import to be told so.
    model = _current_vllm_model_path()
    if model:
        cand.append(("--model", model))
        if is_ommx_w_bundle(model):
            return os.path.abspath(model)
    if not cand:
        # DEFERRABLE (see OMMXWBundleUnconfigured): nothing named a bundle at all, which
        # is the expected state when vLLM asks during VllmConfig construction.
        raise OMMXWBundleUnconfigured(_pack_hint(
            "quantization='ommx_w' was requested but no OMMX_W_SafeTensor bundle was "
            "configured: no bundle key in quantization_config, no $OMMX_W_BUNDLE, and no "
            "vLLM model path in scope."))
    listed = "; ".join(f"{s} -> {p!r}" for s, p in cand)
    raise OMMXWError(_pack_hint(
        f"quantization='ommx_w' was requested but none of the candidate paths is an "
        f"OMMX_W_SafeTensor bundle (no {INDEX_FILENAME} in any of them): {listed}."))


#: Appended to both refusals below. Kept as ONE string so the bitmap message and the
#: no-map message cannot end up describing the opt-in differently.
_TRANSCODE_OFFER = (
    "  ALTERNATIVE (opt-in, and read the caveat): set {env}=1 to TRANSCODE this bundle "
    "at load. The outlier POSITIONS are the only thing the kernel cannot read, and a "
    "bitmap row and a relidx7 slot stream encode the SAME ascending position set (the "
    "FP4 nibble plane is shared and already ascending for both), so the re-encoding is "
    "lossless and is byte-identical to packing this weight with --outlier-repr relidx7 "
    "directly. A missing range map is materialised as the degenerate ms=1 / mc=0 planes "
    "that outlier_map='none' already means.\n"
    "  CAVEAT: the transcode changes the RESIDENT footprint, not the on-disk one. This "
    "bundle is {on_disk:.4f} bits/weight ON DISK and would stream {resident:.4f} "
    "bits/weight in HBM after transcode ({delta:+.4f}). Serving it is NOT serving at "
    "{on_disk:.4f} bits in HBM.\n"
    "  NATIVE FIX (no transcode, no caveat): implement the bitmap reader specified in "
    "ommx_gpu_serve/csrc/linear/BITMAP_READER_SPEC.md."
)


def transcode_requested(quant_config: Optional[Dict[str, Any]] = None) -> bool:
    """Has the operator opted in to the load-time transcode?

    Two sources, both explicit: :data:`TRANSCODE_ENV` in the environment, or one of
    :data:`TRANSCODE_CONFIG_KEYS` in the checkpoint's ``quantization_config``. There is no
    third, and no heuristic — "the bundle is a bitmap bundle, so they must have meant it"
    is exactly the silent format change this flag exists to prevent.
    """
    for key in TRANSCODE_CONFIG_KEYS:
        if key in (quant_config or {}):
            val = (quant_config or {})[key]
            if isinstance(val, str):
                return val.strip().lower() not in ("", "0", "false", "off", "no")
            return bool(val)
    return _env_on(TRANSCODE_ENV)


def require_kernel_readable(recipe: Recipe,
                            allow_transcode: bool = False) -> Optional[TranscodePlan]:
    """Refuse a recipe the shipped CUDA kernel cannot decode. Raises, never warns.

    The packer is deliberately broader than the kernel: it can emit ``bitmap`` positions
    and ``outlier_map='none'`` because the FORMAT admits them. Serving such a bundle
    through ``sparse_correct`` would not fail — it would read the position bytes as a
    relidx7 slot stream and quietly reconstruct the wrong weights, which is the worst
    class of failure this repo has (a plausible-looking number nobody can tell is wrong).

    Returns ``None`` when the recipe is already what the kernel reads — the shipped path,
    unchanged. With ``allow_transcode=True`` it returns the
    :class:`~ommx_gpu_serve.linear.w_format.TranscodePlan` that would close the gap
    losslessly, and STILL RAISES for a gap that cannot be closed (a bitmap bundle at
    ``group_size > 128``, an ``npv`` above the kernel's ``MAX_NPV``): the opt-in buys a
    re-encoding, not a suspension of the rules.
    """
    plan: Optional[TranscodePlan] = None
    if recipe.has_outliers:
        # plan_transcode is the single place that knows which gaps are closable; asking it
        # first means the refusal messages below can quote the REAL bit budgets rather
        # than describing a hypothetical. It raises for a genuinely unreadable recipe, and
        # that raise is the hard refusal — it fires whether or not the operator opted in.
        try:
            plan = plan_transcode(recipe)
        except OMMXWFormatError as exc:
            raise OMMXWError(
                f"this bundle cannot be served by the shipped OMMX weight kernel and "
                f"cannot be transcoded into something it can read: {exc}") from exc
    if plan is None:
        return None
    if allow_transcode:
        return plan
    offer = _TRANSCODE_OFFER.format(
        env=TRANSCODE_ENV, on_disk=plan.on_disk_bits_per_weight,
        resident=plan.resident_bits_per_weight, delta=plan.delta_bits_per_weight)
    if recipe.outlier_repr not in KERNEL_OUTLIER_REPRS:
        raise OMMXWError(
            f"this bundle is packed with outlier_repr={recipe.outlier_repr!r}, which the "
            f"shipped OMMX weight kernel CANNOT read: "
            f"csrc/linear/ommx_linear.cu's sparse_correct decodes the relidx7 slot stream "
            f"(relidx7_slot_pos / relidx7_slot_delta) and there is NO bitmap reader on the "
            f"weight side — the KV/attention path is the one with a bitmap variant. "
            f"Reading a bitmap plane as relidx7 would silently reconstruct the wrong "
            f"weights, so this is refused rather than attempted.\n"
            f"  FIX: re-pack with --outlier-repr relidx7 "
            f"(supported: {', '.join(KERNEL_OUTLIER_REPRS)}).\n"
            f"{offer}")
    raise OMMXWError(
        f"this bundle is packed with outlier_map={recipe.outlier_map!r}, which omits "
        f"the ommx_map_scale / ommx_map_center planes. sparse_correct takes both as "
        f"required arguments and the idx-space range map is the only form the parity "
        f"gate (csrc/linear/test_ommx_linear_parity.py) exercises, so serving this "
        f"recipe would mean guessing at the kernel's ABI.\n"
        f"  FIX: re-pack with --outlier-map idx_range "
        f"(supported: {', '.join(KERNEL_OUTLIER_MAPS)}).\n"
        f"{offer}")


class OMMXWBundle:
    """A validated OMMX_W_SafeTensor bundle: recipe + per-tensor manifest entries.

    Constructed through :meth:`load`, which runs the FULL bundle validator rather than
    just reading the index. That validator is the thing that catches a shard packed with
    a different group size, a hand-edited shape, or a plane the manifest does not claim —
    all of which load fine one tensor at a time and produce a model whose layers disagree.
    Paying for it once at engine start is the cheapest place to find out.
    """

    __slots__ = ("bundle_dir", "index", "recipe", "weight_map", "tensors", "transcode")

    def __init__(self, bundle_dir: str, index: Dict[str, Any], recipe: Recipe,
                 tensors: Dict[str, Dict[str, Any]],
                 transcode: Optional[TranscodePlan] = None) -> None:
        self.bundle_dir = bundle_dir
        self.index = index
        self.recipe = recipe
        self.weight_map: Dict[str, str] = dict(index.get("weight_map", {}))
        self.tensors = tensors
        #: The load-time transcode this bundle needs, or ``None`` when its recipe is
        #: already what the kernel reads. ``None`` is the shipped path and the ONLY path
        #: that touches no plane, which is what makes "unchanged when not requested"
        #: checkable by identity rather than by comparing bytes.
        self.transcode = transcode

    @classmethod
    def load(cls, bundle_dir: str, allow_transcode: bool = False) -> "OMMXWBundle":
        try:
            index = validate_bundle(bundle_dir)
        except OMMXWFormatError as exc:
            raise OMMXWError(_pack_hint(
                f"{bundle_dir!r} carries an {INDEX_FILENAME} but is not a loadable "
                f"OMMX_W_SafeTensor bundle: {exc}")) from exc
        recipe = Recipe.from_json(index["recipe"])
        transcode = require_kernel_readable(recipe, allow_transcode)
        tensors: Dict[str, Dict[str, Any]] = {}
        # Re-read each shard's manifest for the per-tensor entries (kind / shape /
        # planes). validate_bundle has already proven them consistent with the files, so
        # this second pass cannot disagree with what will be loaded.
        from ...linear.w_format import read_manifest      # local: keeps the import cheap
        for shard in sorted(set(index["weight_map"].values())):
            man = read_manifest(os.path.join(bundle_dir, shard))
            for name, ent in man["tensors"].items():
                ent = dict(ent)
                ent["shard"] = shard
                tensors[name] = ent
        return cls(os.path.abspath(bundle_dir), index, recipe, tensors, transcode)

    # ── manifest queries ──────────────────────────────────────────────────────
    def entry(self, tensor_name: str) -> Optional[Dict[str, Any]]:
        return self.tensors.get(tensor_name)

    def source_model(self) -> str:
        return str(self.index.get("source_model", "<unknown>"))

    def candidate_names(self, prefix: str) -> List[str]:
        """Checkpoint tensor names a vLLM Linear at ``prefix`` was built from.

        ``model.layers.0.self_attn.qkv_proj`` -> the three separate projections the
        bundle actually contains, because vLLM fuses them and the checkpoint does not.
        """
        module = prefix.rsplit(".", 1)[-1]
        parts = FUSED_MODULES.get(module)
        if not parts:
            return [f"{prefix}.weight"]
        head = prefix[: -len(module)] if prefix.endswith(module) else prefix + "."
        return [f"{head}{p}.weight" for p in parts]

    def layer_kind(self, prefix: str) -> Tuple[str, List[str]]:
        """``("quantized" | "passthrough" | "unmapped", candidate names)`` for a layer.

        A MIXED fused layer (say a quantized ``q_proj`` beside a passthrough ``k_proj``)
        raises: it cannot be served as one Linear either way, and picking a majority
        would be exactly the silent-wrong-weights outcome this module refuses.
        """
        names = self.candidate_names(prefix)
        kinds = {n: (self.tensors.get(n) or {}).get("kind") for n in names}
        known = [k for k in kinds.values() if k]
        if not known:
            return "unmapped", names
        if len(set(known)) > 1 or len(known) != len(names):
            raise OMMXWError(
                f"vLLM Linear {prefix!r} fuses {names}, but the bundle describes them "
                f"inconsistently ({kinds}). A fused Linear is one parameter set: it is "
                f"either entirely quantized or entirely passthrough. Re-pack the "
                f"checkpoint with one --quant-modules set covering all of them.")
        return str(known[0]), names

    def plane_layout(self, N: int, K: int) -> Dict[str, Tuple[Tuple[int, ...], str]]:
        return self.recipe.plane_layout(N, K)

    def __repr__(self) -> str:
        return (f"OMMXWBundle({self.bundle_dir!r}, {self.recipe!r}, "
                f"{len(self.tensors)} tensors)")


def load_bundle(bundle_dir: Optional[str] = None,
                quant_config: Optional[Dict[str, Any]] = None,
                allow_transcode: Optional[bool] = None) -> OMMXWBundle:
    """Resolve + validate a bundle in one call. The seam the tests drive directly.

    ``allow_transcode=None`` (the default) consults :func:`transcode_requested`, i.e. the
    environment and the checkpoint's ``quantization_config``; pass ``True``/``False`` to
    decide explicitly. Left as a tri-state rather than a plain bool so a caller can be
    explicit WITHOUT having to reproduce the opt-in parsing.
    """
    if allow_transcode is None:
        allow_transcode = transcode_requested(quant_config)
    return OMMXWBundle.load(resolve_bundle_dir(bundle_dir, quant_config),
                            bool(allow_transcode))


# ════════════════════════════════════════════════════════════════════════════
# bundle tensor name  <->  vLLM parameter name  (identity on v2; the invariant, in code)
# ════════════════════════════════════════════════════════════════════════════

def param_name_for_plane(plane: str) -> str:
    """``"code"`` -> ``"ommx_code"``: the attribute ``create_weights`` registers."""
    return f"{PLANE_PARAM_PREFIX}{plane}"


def bundle_to_param_name(tensor_name: str) -> str:
    """Bundle tensor name -> vLLM parameter name. On ``OMMX_W_SafeTensor`` v2: IDENTITY.

    v2 plane names carry no ``.weight`` infix, so there is nothing to rewrite and this
    returns its argument. It is kept rather than deleted because it is the one place the
    bundle-name/parameter-name convention is stated in executable form: the gate
    ``test_bundle_plane_names_remap_to_parameter_names`` asserts the identity, so if
    either ``w_format.plane_name`` or ``PLANE_PARAM_PREFIX`` drifts, that fails loudly
    instead of surfacing as a missing-parameter error inside an engine.

    The ``.weight``-stripping branch below only ever fires for a v1 name, which the
    validator refuses before a loader can see it; it remains as a precise no-op rather
    than a silent pass-through of a name we know binds to nothing.
    """
    head, dot, tail = tensor_name.rpartition(".")
    if not dot or not tail.startswith(PLANE_PARAM_PREFIX):
        return tensor_name
    if head.endswith(".weight"):
        return f"{head[: -len('.weight')]}.{tail}"
    return tensor_name


def remap_bundle_weight_names(weights: Iterable[Tuple[str, Any]]) -> Iterator[Tuple[str, Any]]:
    """Wrap a ``(name, tensor)`` loader stream with :func:`bundle_to_param_name`.

    An identity map over v2 bundles (see that function). Provided so a loader that wants
    to be explicit about the convention can be, and so the stream form is covered by the
    same gate. UNVERIFIED against a real engine (no GPU/engine this session).
    """
    for name, tensor in weights:
        yield bundle_to_param_name(name), tensor


# ════════════════════════════════════════════════════════════════════════════
# kernel JIT build
# ════════════════════════════════════════════════════════════════════════════

_BUILD_ENV_HELP = (
    "  TWO MEASURED ENVIRONMENT REQUIREMENTS (both hit on the cluster before the parity "
    "gate could run at all — MEASURED_FACTS §10):\n"
    "    1. the `ninja` EXECUTABLE must be on PATH. torch.utils.cpp_extension.load() "
    "SHELLS OUT to it, so installing the wheel is necessary but NOT sufficient: "
    "<env>/bin is on PATH only while the env is ACTIVATED. Either activate it "
    "(conda activate ... / source .venv/bin/activate) or "
    "export PATH=\"/path/to/env/bin:$PATH\". Symptom otherwise: "
    "'RuntimeError: Ninja is required to load C++ extensions'.\n"
    "    2. export LIBRARY_PATH=\"$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64\" on cu13 conda "
    "envs. torch emits -L$CUDA_HOME/lib64 while those envs keep libcudart.so in lib/, so "
    "the link dies as '/usr/bin/ld: cannot find -lcudart'.\n"
    "  Install the build deps with: uv pip install -e '.[linear]' (the repo-root extra "
    "that declares ninja)."
)


def _load_build_module():
    """Import ``csrc/linear/build_ommx_linear.py`` BY PATH.

    ``csrc/linear/`` carries no ``__init__.py`` and is excluded from the wheel's package
    list on purpose (see ommx_gpu_serve/pyproject.toml), so it is not importable as
    ``ommx_gpu_serve.csrc.linear.build_ommx_linear``. Loading by file path is what keeps
    that packaging decision intact instead of quietly reversing it here.
    """
    here = os.path.dirname(os.path.abspath(__file__))            # integration/vllm
    pkg_root = os.path.dirname(os.path.dirname(here))            # ommx_gpu_serve
    path = os.path.join(pkg_root, "csrc", "linear", "build_ommx_linear.py")
    if not os.path.isfile(path):
        raise OMMXWError(
            f"the OMMX linear kernel builder is missing: {path} does not exist. The "
            f"ommx_w serving path needs the csrc/linear/ sources, which ship in the git "
            f"checkout but NOT in the ommx-gpu-serve wheel (csrc/ is deliberately not a "
            f"package). Run from a source checkout of open_ommx_serve.")
    spec = importlib.util.spec_from_file_location("ommx_build_ommx_linear", path)
    if spec is None or spec.loader is None:
        raise OMMXWError(f"could not create an import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_kernel(force: bool = False):
    """JIT-build (once per process) and return the ``ommx_linear`` extension module.

    Called from ``process_weights_after_loading`` — law #6: a new kernel cache is seeded
    at WEIGHT-LOAD time, because a lazy "build on the first eager call" is never reached
    on vLLM's compiled/traced path and the build then happens (or fails) somewhere
    unattributable. Failures raise with both measured environment requirements spelled
    out; the failure is also LATCHED, so a second layer gets the same message instead of
    a confusing second traceback from a half-initialised torch extension directory.

    UNVERIFIED (no GPU this session): the build itself. This host has no CUDA device, so
    ``torch.cuda.get_device_capability()`` inside the builder cannot run.
    """
    if _KERNEL[0] is not None and not force:
        return _KERNEL[0]
    if _KERNEL_ERROR[0] is not None and not force:
        raise OMMXWError(_KERNEL_ERROR[0])
    try:
        builder = _load_build_module()
        mod = builder.build()
    except Exception as exc:                   # noqa: BLE001 - every build failure
        msg = (
            f"the OMMX weight-quant linear kernel FAILED TO BUILD "
            f"({type(exc).__name__}: {exc}).\n"
            f"  quantization='ommx_w' cannot serve without it, and this path will NOT "
            f"fall back to a bf16 Linear — a bf16 arm reported under an ommx_w label is "
            f"the exact defect this release removed from the attention path.\n"
            f"{_BUILD_ENV_HELP}\n"
            f"  Verify the toolchain independently first (it is the turnkey gate):\n"
            f"      python ommx_gpu_serve/csrc/linear/test_ommx_linear_parity.py"
        )
        _KERNEL_ERROR[0] = msg
        raise OMMXWError(msg) from exc
    _KERNEL[0] = mod
    _KERNEL_ERROR[0] = None
    _FIRE["kernel_builds"] += 1
    _fire("OMMX_W_KERNEL_BUILT", f"module={getattr(mod, '__name__', type(mod).__name__)}")
    return mod


# ════════════════════════════════════════════════════════════════════════════
# the vLLM classes — built LAZILY so importing this module never needs vLLM
# ════════════════════════════════════════════════════════════════════════════

#: Cache for :func:`_build_classes`. Cleared by ``_reset_state_for_tests``.
_CLASSES: Dict[str, Any] = {}

_VLLM_IMPORT_HELP = (
    "install the serving extra (uv pip install -e '.[ommx]', which pins vllm>=0.21 — the "
    "floor integration/vllm/preflight.py enforces as MIN_VLLM_VERSION)"
)


def _import_vllm_bases():
    """Import the four vLLM symbols the classes need. Raises a NAMED error if absent."""
    try:
        from vllm.model_executor.layers.linear import (
            LinearBase, LinearMethodBase, UnquantizedLinearMethod)
        from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
        from vllm.model_executor.utils import set_weight_attrs
    except ImportError as exc:
        raise OMMXWError(
            f"quantization='ommx_w' needs vLLM's Linear/quantization base classes and "
            f"they are not importable ({exc}). {_VLLM_IMPORT_HELP}.") from exc
    return LinearBase, LinearMethodBase, UnquantizedLinearMethod, QuantizationConfig, \
        set_weight_attrs


def _env_int(name: str, default: int) -> int:
    """Parse an int env var STRICTLY. A typo must not silently become the default.

    (Law #11 in spirit: the clamping helper elsewhere in this repo turned an intended
    OFF=0 into ON=1. Here a malformed value raises instead of being swallowed.)
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise OMMXWError(f"{name}={raw!r} is not an integer") from None


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in ("", "0", "false", "off", "no")


#: Inclusive bounds on ``OMMX_W_PREFILL_MIN_M``, the decode/prefill routing threshold.
#: These are NOT a preference — they are the two kernel limits that bracket the split,
#: read out of ``csrc/linear/ommx_linear.cu``:
#:
#:   * lower bound 16 — ``prefill_wmma`` opens with
#:     ``TORCH_CHECK(M >= 16, "prefill_wmma needs M>=16 (m16n8k16 16-row tile); route
#:     M<16 to decode_base (law #1)")``. A threshold below 16 routes a short prefill into
#:     a kernel that refuses it. LOUD.
#:   * upper bound 17 — ``decode_base`` has NO upper TORCH_CHECK on M, but its batched
#:     launcher tops out at ``decode_gemv_batched_kernel<SF, 16>`` and that kernel's
#:     epilogue writes ``for (int m = 0; m < M_MAX; ++m) ... if (lane == 0 && m < M)``.
#:     At M > 16 rows 16..M-1 are NEVER WRITTEN, and ``out`` came from ``torch::zeros``,
#:     so those tokens silently receive an all-zero activation. SILENT. A threshold of 17
#:     still routes M==16 to decode_base (all 16 rows written) and M>=17 to prefill, so 17
#:     is the largest value that cannot reach the silent case.
#:
#: The env knob therefore validates against BOTH ends. The default, 16, is unaffected.
PREFILL_MIN_M_BOUNDS: Tuple[int, int] = (16, 17)


def prefill_min_m_env() -> int:
    """The decode/prefill routing threshold, validated against the kernel's own bounds.

    Refusing out of range converts a silent wrong answer (see PREFILL_MIN_M_BOUNDS,
    upper bound) into a named error at the first forward. It is a refusal and not a
    clamp on purpose: an operator who set the knob meant something by it, and quietly
    serving a different threshold is the failure mode law #11 names.
    """
    m = _env_int("OMMX_W_PREFILL_MIN_M", PREFILL_MIN_M_BOUNDS[0])
    lo, hi = PREFILL_MIN_M_BOUNDS
    if not lo <= m <= hi:
        raise OMMXWError(
            f"OMMX_W_PREFILL_MIN_M={m} is outside [{lo}, {hi}], the range bracketed by "
            f"csrc/linear/ommx_linear.cu itself. Below {lo}, prefill_wmma aborts on its "
            f"own TORCH_CHECK(M >= 16). Above {hi}, decode_base is handed M > 16 and its "
            f"batched kernel is instantiated at M_MAX=16, whose epilogue writes only "
            f"rows m < 16 into a zero-filled output — every token past the 16th would be "
            f"served an all-zero activation with NO error. Leave the variable unset for "
            f"the shipped threshold ({lo}).")
    return m


def sparse_correct_m_gt_1_operands(y: torch.Tensor,
                                   x2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(C, At)`` for ``sparse_correct``'s M>1 branch. Pure tensor plumbing, no kernel.

    UNVERIFIED on hardware (this repo has had no GPU since the defect was found); the
    argument SHAPES/DTYPES are gated on CPU by
    ``tests/test_linear_method.py::test_apply_builds_the_m_gt_1_sparse_correct_operands``
    and the NUMBERS are gated by the new M>1 cases in
    ``csrc/linear/test_ommx_linear_parity.py``. See ``csrc/linear/SPARSE_CORRECT_ABI.md``
    for the line-by-line derivation and for what would falsify this.

    Two facts from ``ommx_linear.cu``, and this helper exists to satisfy both:

    1. ``At`` is ``A^T``, not ``A``. The M>1 kernels index it as
       ``At[(size_t)k_sh[s] * M + m]`` — element (k, m) at flat offset ``k*M + m`` — so
       the buffer must be a CONTIGUOUS ``[K, M]``. ``x2d.t()`` alone is a stride-(1, K)
       view whose ``data_ptr()`` is still A's row-major storage; the kernel would read
       ``A[m*K + k]`` positions under a ``k*M + m`` index and produce a plausible,
       wrong number. The ``.contiguous()`` is load-bearing, and the kernel does NOT
       check it (there is no ``check_contig_cuda(At)`` in ``sparse_correct``).

       COST, stated because a hidden per-token copy in a serving path is a performance
       defect even when it is correct: this is a full transpose + copy of the
       activation, M*K*2 bytes read and written, EVERY forward that carries outliers.
       At Llama-3.1-8B prefill (M=4096 tokens, K=4096) that is 32 MiB moved twice per
       linear, seven linears per layer, 32 layers. The transpose is a consequence of the
       decoupled base+correct split; the fused entry points
       (``decode_base_correct_fused_batched`` for M<=16, or a prefill_wmma with its
       ``oindex``/``odelta`` overlay) take ``A`` row-major and would remove it entirely.
       That is a measured-perf decision, so it is named here and NOT taken blind.

    2. ``C`` is bf16, where the M==1 branch's ``out`` is fp32. ``goM`` dereferences
       ``out_or_C.data_ptr<at::BFloat16>()`` while ``go1`` uses ``data_ptr<float>()``.
       ``prefill_wmma`` already returns bf16 (``torch::empty({M, N}, A.options())``) so
       nothing is copied on the prefill path; ``decode_base`` returns fp32
       (``torch::zeros(..., kFloat32)``) so the 1 < M < 16 batched-decode path needs the
       cast. Rounding the base to bf16 BEFORE adding the correction (rather than after)
       is a real numeric difference from the M==1 path, but it is the M>1 kernel's own
       contract — the kernel's update is
       ``C[idx] = __float2bfloat16(__bfloat162float(C[idx]) + acc)`` — and the value is
       cast to bf16 on the way out of ``apply()`` regardless.
    """
    if x2d.dtype != torch.bfloat16:
        raise OMMXWError(
            f"sparse_correct's M>1 branch reads At through data_ptr<at::BFloat16>(); "
            f"got activation dtype {x2d.dtype}")
    if y.shape[0] != x2d.shape[0]:
        raise OMMXWError(
            f"base output rows {y.shape[0]} != activation rows {x2d.shape[0]}; the "
            f"kernel indexes C[m*N+n] and At[k*M+m] with the SAME m")
    # bf16 C. Identity (not a copy) when the base kernel already returned bf16, so the
    # prefill path is unchanged and the correction lands in the tensor apply() returns.
    C = y if y.dtype == torch.bfloat16 else y.to(torch.bfloat16)
    # [K, M] contiguous. Derived from the SAME x2d the M==1 path passes as A.
    At = x2d.t().contiguous()
    return C, At


def _build_classes():
    """Define and return ``(OMMXWConfig, OMMXWLinearMethod)``. Cached per process."""
    if "config" in _CLASSES:
        return _CLASSES["config"], _CLASSES["method"]

    (LinearBase, LinearMethodBase, UnquantizedLinearMethod, QuantizationConfig,
     set_weight_attrs) = _import_vllm_bases()

    class OMMXWConfig(QuantizationConfig):
        """vLLM ``QuantizationConfig`` for an OMMX_W_SafeTensor bundle.

        Holds the validated :class:`OMMXWBundle`; every per-layer decision (quantize /
        pass through / refuse) is answered from its MANIFEST rather than from a name
        heuristic, so the serving path and the packer can never disagree about which
        tensors were quantized.
        """

        #: vLLM fuses these on load; naming them here is what lets its loader map a
        #: checkpoint's separate q/k/v planes onto one fused parameter set.
        packed_modules_mapping = {k: list(v) for k, v in FUSED_MODULES.items()}

        def __init__(self, bundle: Optional[OMMXWBundle] = None, *,
                     deferred_quant_config: Optional[Dict[str, Any]] = None) -> None:
            """Either an already-loaded bundle, or a promise to resolve one later.

            DEFERRED RESOLUTION EXISTS BECAUSE OF *WHEN* vLLM ASKS. vLLM builds the
            quantization config inside ``VllmConfig._get_quantization_config`` — which
            runs while the ``VllmConfig`` is still being constructed, so
            ``get_current_vllm_config()`` has nothing to return and
            ``resolve_bundle_dir``'s last rule ("the model path vLLM is building for")
            can never fire there. That rule is the one that makes the documented
            invocation ``vllm serve <bundle> --quantization ommx_w`` work at all, so
            resolving eagerly meant the bundle was findable only via ``$OMMX_W_BUNDLE``
            or an absolute path baked into the checkpoint.

            Deferring moves the lookup to the first ``get_quant_method`` call, which
            happens during model loading, when the config context IS active. Nothing
            else about the config needs the bundle before then — ``get_min_capability``,
            ``get_name`` and ``get_supported_act_dtypes`` are all class-level.
            """
            super().__init__()
            # RE-ASSERT the mapping AFTER super().__init__(). vLLM's QuantizationConfig
            # base sets ``self.packed_modules_mapping = {}`` in its own __init__ (models
            # update it as they initialise), which shadows the class attribute above with
            # an empty dict. Losing it would leave the loader with no idea that a fused
            # qkv_proj is three checkpoint tensors — a load failure, but a puzzling one.
            self.packed_modules_mapping = {k: list(v) for k, v in FUSED_MODULES.items()}
            if bundle is None and deferred_quant_config is None:
                raise OMMXWError("OMMXWConfig needs either a loaded bundle or a "
                                 "quantization_config to resolve one from")
            self._bundle = bundle
            self._deferred = dict(deferred_quant_config or {})
            # ``recipe`` is read by callers that hold the config; it follows the bundle,
            # so it is a property too rather than a snapshot taken before resolution.

        @property
        def bundle(self) -> OMMXWBundle:
            """The loaded bundle, resolving it on first access if construction deferred.

            A failure here raises the SAME OMMXWError ``from_config`` would have raised,
            with the same packer hint — deferring changes WHEN the operator is told, not
            WHAT they are told.
            """
            if self._bundle is None:
                self._bundle = load_bundle(quant_config=dict(self._deferred))
            return self._bundle

        @property
        def recipe(self) -> Recipe:
            return self.bundle.recipe

        # ── vLLM QuantizationConfig surface ──────────────────────────────────
        @classmethod
        def get_name(cls) -> str:
            return OMMX_W_METHOD_NAME

        @classmethod
        def get_supported_act_dtypes(cls) -> List[torch.dtype]:
            # bfloat16 ONLY, and that is a refusal, not a limitation to work around:
            # the kernel's verified call convention takes a bf16 activation
            # (test_ommx_linear_parity casts x to bfloat16 for every gate) and returns
            # fp32. Declaring fp16 here would let an engine start and then hit a dtype
            # mismatch — or worse, an implicit cast — deep inside apply().
            return [torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            # build_ommx_linear.build() asserts sm >= 80 ("targets A100 (sm_80) or
            # H100/H200 (sm_90a)"). Declaring it here makes vLLM refuse at startup
            # instead of at the first JIT build.
            return 80

        @staticmethod
        def get_config_filenames() -> List[str]:
            # Deliberately empty. vLLM reads these files from the MODEL directory to
            # build the config dict, but an OMMX_W bundle's identity is the DIRECTORY
            # (index + shards); handing from_config the parsed ommx_w_index.json without
            # telling it which directory it came from would lose the one fact that
            # matters. resolve_bundle_dir() does the discovery instead.
            return []

        @classmethod
        def from_config(cls, config: Dict[str, Any]) -> "OMMXWConfig":
            """Build from the checkpoint's ``quantization_config`` dict.

            Refuses a config whose ``quant_method`` names some OTHER method: vLLM will
            hand us whatever the HF config says, and quantizing an AWQ checkpoint with
            the OMMX planes is not a recoverable situation.
            """
            method = (config or {}).get("quant_method")
            if method is not None and str(method) != OMMX_W_METHOD_NAME:
                raise OMMXWError(
                    f"this checkpoint's quantization_config says quant_method="
                    f"{method!r}, not {OMMX_W_METHOD_NAME!r}. The ommx_w linear method "
                    f"reads OMMX_W_SafeTensor planes and cannot serve a {method!r} "
                    f"checkpoint. Select --quantization {method} instead, or pack this "
                    f"model with the OMMX packer first.")
            cfg = dict(config or {})
            try:
                return cls(load_bundle(quant_config=cfg))
            except OMMXWBundleUnconfigured:
                # Nothing configured a bundle AT THIS MOMENT. That is the normal case for
                # `vllm serve <bundle> --quantization ommx_w`: this runs inside
                # VllmConfig's own construction, so the model path is not yet reachable
                # through get_current_vllm_config(). Defer rather than refuse — the same
                # lookup runs again at get_quant_method time, with the context active,
                # and raises there with this identical message if it still finds nothing.
                return cls(deferred_quant_config=cfg)

        def get_quant_method(self, layer: Any, prefix: str = "") -> Optional[Any]:
            """Pick the method for one layer, from the MANIFEST.

            Non-Linear layers (embeddings, the LM head's VocabParallelEmbedding, MoE
            experts) return None -> vLLM keeps their own unquantized path, matching the
            packer's NEVER_QUANTIZE list. A Linear the bundle does not describe at all is
            an ERROR: silently serving it bf16 is how a partially-quantized model gets
            published as a fully-quantized measurement.
            """
            if not isinstance(layer, LinearBase):
                return None
            kind, names = self.bundle.layer_kind(prefix)
            if kind == "quantized":
                return OMMXWLinearMethod(self)
            if kind == "passthrough":
                # Recorded in the manifest WITH a reason (lm_head, embeddings, norms).
                # Deliberate, auditable, and not a fallback.
                return UnquantizedLinearMethod()
            if _env_on("OMMX_W_ALLOW_UNMAPPED"):
                _fire("OMMX_W_UNMAPPED_PASSTHROUGH", f"prefix={prefix}")
                print(f"[ommx_w] WARNING: Linear {prefix!r} is not described by the "
                      f"bundle manifest; serving it UNQUANTIZED because "
                      f"OMMX_W_ALLOW_UNMAPPED is set. This layer's weights are bf16 and "
                      f"any bits/weight figure for this model is wrong by its share.",
                      file=sys.stderr, flush=True)
                return UnquantizedLinearMethod()
            raise OMMXWError(
                f"vLLM instantiated a Linear at {prefix!r} but the OMMX_W bundle "
                f"{self.bundle.bundle_dir!r} describes none of {names} — the bundle was "
                f"packed from a different model (source_model="
                f"{self.bundle.source_model()!r}) or with a narrower --quant-modules "
                f"set.\n"
                f"  Serving this layer bf16 would make the model partially quantized "
                f"while still reporting as ommx_w, so it is refused.\n"
                f"  FIX: re-pack THIS model "
                f"(python -m ommx_gpu_serve.linear.w_packer pack --input <ckpt> "
                f"--output <bundle>), or set OMMX_W_ALLOW_UNMAPPED=1 to accept a "
                f"partially quantized model with a loud per-layer warning.")

        def get_scaled_act_names(self) -> List[str]:
            # Present for older vLLM trees that still call it; OMMX quantizes weights
            # only and rescales nothing in the activation path.
            return []

        def __repr__(self) -> str:
            return f"OMMXWConfig({self.bundle!r})"

    class OMMXWLinearMethod(LinearMethodBase):
        """Executes one Linear from OMMX_W planes through the ``csrc/linear`` kernels.

        Lifecycle:
          ``create_weights``                  allocate the packed planes (shapes come
                                              from the manifest's recipe, not from here)
          ``process_weights_after_loading``   JIT-build the kernel + materialise the
                                              fp32 scale/zp twins the kernel wants
          ``apply``                           decode_base / prefill_wmma (+ sparse_correct)

        UNVERIFIED (no GPU this session): everything from the JIT build onwards.
        """

        def __init__(self, quant_config: "OMMXWConfig") -> None:
            self.quant_config = quant_config
            self.recipe = quant_config.recipe

        # ── weight allocation ────────────────────────────────────────────────
        def create_weights(self, layer: torch.nn.Module,
                           input_size_per_partition: int,
                           output_partition_sizes: Sequence[int],
                           input_size: int, output_size: int,
                           params_dtype: torch.dtype, **extra_weight_attrs) -> None:
            """Allocate one parameter per OMMX plane, shaped by the manifest's recipe.

            ``N`` is the SUM of ``output_partition_sizes`` — for a fused ``qkv_proj``
            that is q+k+v rows, which is also how the bundle's three separate ``[N_i, K]``
            weights concatenate along N. ``K`` is the per-partition input size, so a
            TP-sharded Linear allocates only its own slice.

            UNVERIFIED (no GPU/engine this session): TP sharding of the packed planes.
            The dim attributes below are what vLLM's default loader needs to narrow a
            checkpoint tensor, but only a real multi-GPU run proves the packed ``code``
            plane (K/4 bytes along dim 1) and the group planes (K/gs groups along dim 1)
            narrow consistently.
            """
            N = int(sum(output_partition_sizes))
            K = int(input_size_per_partition)
            gs = self.recipe.group_size
            if K % gs:
                raise OMMXWError(
                    f"Linear with input_size_per_partition={K} cannot be served by a "
                    f"bundle packed at group_size={gs}: {K} % {gs} = {K % gs}. Under "
                    f"tensor parallelism the PER-PARTITION K must stay a multiple of the "
                    f"group size, because a group is the unit that owns one scale/zp "
                    f"pair and cannot be split across ranks. Re-pack with a group size "
                    f"that divides {K}, or lower the TP degree.")

            layout = self.quant_config.bundle.plane_layout(N, K)
            for plane, (shape, st_dt) in layout.items():
                torch_dt = _ST_TO_TORCH[st_dt]
                param = torch.nn.Parameter(
                    torch.empty(tuple(shape), dtype=torch_dt), requires_grad=False)
                attrs: Dict[str, Any] = {"output_dim": 0, "ommx_plane": plane}
                if plane == "code":
                    # [N, K/4] — four INT2 codes per byte along the INPUT dim.
                    attrs.update(input_dim=1, packed_dim=1, packed_factor=4)
                else:
                    # [N, G] (or [N, G, bytes]) — one entry per GROUP along the input dim.
                    attrs.update(input_dim=1, packed_dim=1, packed_factor=gs)
                attrs.update(extra_weight_attrs)
                set_weight_attrs(param, attrs)
                layer.register_parameter(param_name_for_plane(plane), param)

            # Geometry the apply() path needs, recorded on the layer rather than on self:
            # one method instance can be shared, a layer is the thing with a shape.
            layer.ommx_w_N = N
            layer.ommx_w_K = K
            layer.ommx_w_group_size = gs
            layer.ommx_w_npv = self.recipe.npv
            # The plane layout above is the ON-DISK one in every case — the loader fills
            # these parameters straight from the bundle, so they must match the manifest
            # exactly whether or not a transcode follows. The transcode happens strictly
            # AFTER loading, in process_weights_after_loading.
            layer.ommx_w_transcode = self.quant_config.bundle.transcode
            layer.ommx_w_ready = False
            _FIRE["layers_created"] += 1

        # ── kernel + derived planes ──────────────────────────────────────────
        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            """Seed the kernel and the fp32 scale/zp twins at LOAD time (law #6).

            vLLM's compiled path never reaches a "build it on the first eager call"
            branch (the traced graph DCEs it), so a lazily built kernel is a kernel that
            never builds. This is the hook that runs once per layer at
            ``process_weights``; the build itself is latched process-wide.

            The bundle stores an int8 E8M0 exponent and a bf16 zero point; the kernel's
            verified call takes fp32 ``[N, G]`` for both. ``2**exp`` is exact (it IS a
            power of two — the parity gate's "E8M0 vs fp32 must be exact" case measured
            max_diff 0.00e+00) and bf16 -> f32 is exact, so materialising the twins costs
            no accuracy. Both are plain attributes, NOT parameters: re-registering them
            would put tensors the checkpoint does not contain into the loader's namespace.
            """
            build_kernel()
            scale_exp = getattr(layer, param_name_for_plane("scale_exp")).data
            zp = getattr(layer, param_name_for_plane("zp")).data
            layer.ommx_w_scale_f32 = torch.pow(
                torch.tensor(2.0, dtype=torch.float32, device=scale_exp.device),
                scale_exp.to(torch.float32)).contiguous()
            layer.ommx_w_zp_f32 = zp.to(torch.float32).contiguous()
            self._resolve_outlier_planes(layer)
            layer.ommx_w_ready = True
            _FIRE["layers_ready"] += 1
            if _FIRE["layers_ready"] == 1:
                _fire("OMMX_W_WEIGHTS_READY",
                      f"N={layer.ommx_w_N} K={layer.ommx_w_K} "
                      f"gs={layer.ommx_w_group_size} npv={layer.ommx_w_npv} "
                      f"fmt={self.recipe.bundle_format}")

        # ── the outlier planes the KERNEL will read (transcode lives here) ────
        def _resolve_outlier_planes(self, layer: torch.nn.Module) -> None:
            """Pin the exact ``oindex`` / ``map_scale`` / ``map_center`` ``apply()`` uses.

            NO TRANSCODE REQUESTED (the shipped path): the three attributes are bound to
            the parameters' OWN ``.data`` tensors — the same objects ``apply()`` used to
            fetch with ``getattr`` inline. Nothing is copied, cast, reshaped or
            reallocated, which is why "unchanged when the feature is not requested" is
            checkable by tensor IDENTITY (``is``) rather than by comparing bytes, and why
            ``tests/test_linear_method.py::
            test_without_the_opt_in_the_kernel_planes_are_the_parameters_themselves``
            asserts exactly that.

            TRANSCODE REQUESTED: the bitmap position plane is re-encoded ONCE into the
            relidx7 slot stream the kernel decodes, and the degenerate ms=1 / mc=0 range
            map is materialised. Both are lossless (see the module docstring and
            ``w_format.plan_transcode``); the source position plane's storage is then
            RELEASED, because a resident-footprint number that quietly excluded a plane
            still sitting in HBM would be the same class of dishonesty this feature is
            documented against.
            """
            if not self.recipe.has_outliers:
                # npv == 0: no position stream, no nibbles, no map. apply() never reaches
                # the correction call, so binding these would only create attributes that
                # promise something the recipe does not have.
                layer.ommx_w_oindex_k = None
                layer.ommx_w_map_scale_k = None
                layer.ommx_w_map_center_k = None
                return
            oindex_p = getattr(layer, param_name_for_plane("oindex"))
            plan: Optional[TranscodePlan] = getattr(layer, "ommx_w_transcode", None)
            if plan is None:
                layer.ommx_w_oindex_k = oindex_p.data
                layer.ommx_w_map_scale_k = getattr(
                    layer, param_name_for_plane("map_scale")).data
                layer.ommx_w_map_center_k = getattr(
                    layer, param_name_for_plane("map_center")).data
                return

            N, K = int(layer.ommx_w_N), int(layer.ommx_w_K)
            gs = int(layer.ommx_w_group_size)
            G = K // gs
            npv = int(self.recipe.npv)
            on_disk_bytes = plan.on_disk_bytes(N, K)
            resident_bytes = plan.resident_bytes(N, K)
            freed = 0

            if self.recipe.outlier_repr != KERNEL_OUTLIER_REPRS[0]:
                # Re-encode. This RAISES (via quantize.outlier_positions) if any group's
                # popcount disagrees with the manifest's npv — a corrupt or mislabelled
                # bitmap must not become a plausible relidx7 stream.
                layer.ommx_w_oindex_k = transcode_oindex_bitmap_to_relidx7(
                    oindex_p.data, N, G, npv, gs)
                freed = oindex_p.data.numel() * oindex_p.data.element_size()
                # Release the source plane. Replacing ``.data`` (rather than deleting the
                # parameter) keeps the parameter object and its name intact, so anything
                # that enumerates the layer's parameters after load still sees the same
                # set — it is only the storage that goes.
                oindex_p.data = torch.empty(0, dtype=oindex_p.data.dtype,
                                            device=oindex_p.data.device)
            else:
                layer.ommx_w_oindex_k = oindex_p.data

            if self.recipe.has_map:
                layer.ommx_w_map_scale_k = getattr(
                    layer, param_name_for_plane("map_scale")).data
                layer.ommx_w_map_center_k = getattr(
                    layer, param_name_for_plane("map_center")).data
            else:
                ms, mc = degenerate_map_planes(N, G, device=oindex_p.data.device)
                layer.ommx_w_map_scale_k = ms
                layer.ommx_w_map_center_k = mc

            # The fp32 twins were materialised a few lines up, in
            # process_weights_after_loading. Measured off the real tensors rather than
            # derived from the recipe, so the number cannot drift from what was allocated.
            twin_bytes = sum(t.numel() * t.element_size()
                             for t in (layer.ommx_w_scale_f32, layer.ommx_w_zp_f32))
            _FIRE["transcoded_layers"] += 1
            _TRANSCODE["plan"] = plan.to_json()
            _TRANSCODE["layers"] += 1
            _TRANSCODE["on_disk_bytes"] += on_disk_bytes
            _TRANSCODE["resident_bytes"] += resident_bytes
            _TRANSCODE["freed_bytes"] += freed
            _TRANSCODE["twin_bytes"] += twin_bytes
            if _TRANSCODE["layers"] == 1:
                # LOUD, once per process, on the first transcoded layer. Both bit budgets
                # are on the line, computed by TranscodePlan, because the one sentence a
                # reader must not be able to write is "we served the paper's bundle, so we
                # served at its bits/weight".
                print(
                    f"[ommx_w] TRANSCODE ACTIVE ({TRANSCODE_ENV}): this bundle is packed "
                    f"outlier_repr={self.recipe.outlier_repr!r} "
                    f"outlier_map={self.recipe.outlier_map!r}, which the CUDA kernel "
                    f"cannot read. Its planes are being re-encoded at LOAD into "
                    f"{KERNEL_OUTLIER_REPRS[0]}+{KERNEL_OUTLIER_MAPS[0]}.\n"
                    f"[ommx_w]   {plan.describe(N, K)}\n"
                    f"[ommx_w]   The transcode is LOSSLESS (identical dequantized "
                    f"weights) but changes the RESIDENT footprint. Do NOT report this "
                    f"run at the bundle's on-disk bits/weight.\n"
                    f"[ommx_w]   Both figures above count BUNDLE PLANES only, which makes "
                    f"the pair a DELTA statement and NOT an HBM total: this layer also "
                    f"holds fp32 scale/zp twins worth "
                    f"{8.0 * twin_bytes / float(N * K):+.4f} b/wt, which the kernel reads "
                    f"IN PLACE OF the int8 scale_exp / bf16 zp planes counted above, so "
                    f"real HBM for this layer is "
                    f"{plan.resident_bits_per_weight + 8.0 * twin_bytes / float(N * K):.4f}"
                    f" b/wt. The twins exist identically with and without a transcode, "
                    f"which is exactly why they are outside the delta and not inside it.",
                    file=sys.stderr, flush=True)
                # Tag deliberately does NOT end in _FIRED: bench_e2e_a100 treats ANY
                # OMMX_W_*_FIRED tag as proof that apply() ran, and a load-time transcode
                # is not evidence that a single token was served.
                _fire("OMMX_W_TRANSCODE_APPLIED",
                      f"on_disk_bpw={plan.on_disk_bits_per_weight:.4f} "
                      f"resident_bpw={plan.resident_bits_per_weight:.4f} "
                      f"steps={len(plan.steps)}")

        # ── execution ────────────────────────────────────────────────────────
        def apply(self, layer: torch.nn.Module, x: torch.Tensor,
                  bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            """``y = x @ dequant(W)^T (+ bias)`` through the OMMX linear kernels.

            Argument order comes from ``csrc/linear/ommx_linear.cu`` itself, cross-read
            against ``csrc/linear/test_ommx_linear_parity.py``. Taking it from the gate
            ALONE is how this method shipped a fatal defect: the gate's single
            ``sparse_correct`` call site is at M=1, ``sparse_correct`` selects its
            argument list on M, and so every M>1 forward of every outlier-carrying recipe
            aborted on ``"M>1 correct needs At"`` while the gate reported PASS 5/5. The
            derivation of both conventions is ``csrc/linear/SPARSE_CORRECT_ABI.md``.

            UNVERIFIED (no GPU): every line of this method, and specifically the M>1
            ``sparse_correct`` operands. What would falsify them is written down in §6 of
            that note and gated by the ``gpu``-marked
            ``test_gpu_apply_at_m_gt_1_with_outliers_matches_the_cpu_dequant_oracle``
            in ``tests/test_linear_method.py``, plus the two new M>1 parity cases.

            ``sparse_correct`` is applied whenever the recipe HAS outliers, at every M.
            Skipping it at M>1 would be a silent accuracy fallback — the run would produce
            base-only INT2 weights and still be labelled i2f4 — so the unverified call is
            made and flagged, not avoided.
            """
            if not getattr(layer, "ommx_w_ready", False):
                raise OMMXWError(
                    "OMMXWLinearMethod.apply() was reached before "
                    "process_weights_after_loading() — the kernel handle and the fp32 "
                    "scale/zp twins do not exist yet. This means vLLM loaded the model "
                    "through a path that skips the process_weights hook; there is no "
                    "safe way to serve the layer, so it refuses rather than guessing.")
            if x.dtype != torch.bfloat16:
                raise OMMXWError(
                    f"the OMMX linear kernel takes a bfloat16 activation (every gate in "
                    f"test_ommx_linear_parity.py casts x to bfloat16); got {x.dtype}. "
                    f"OMMXWConfig.get_supported_act_dtypes() declares bfloat16 only — "
                    f"start the engine with --dtype bfloat16.")

            N, K = int(layer.ommx_w_N), int(layer.ommx_w_K)
            gs = int(layer.ommx_w_group_size)
            if x.shape[-1] != K:
                raise OMMXWError(
                    f"activation last dim {x.shape[-1]} != this layer's K={K}")
            out_shape = tuple(x.shape[:-1]) + (N,)
            x2d = x.reshape(-1, K).contiguous()
            M = int(x2d.shape[0])

            mod = build_kernel()
            code = getattr(layer, param_name_for_plane("code")).data
            scale = layer.ommx_w_scale_f32
            zp = layer.ommx_w_zp_f32
            split = _env_int("OMMX_W_SPLIT", 1)

            # DECODE vs PREFILL. The gate measured decode_base at M=1/8 and prefill_wmma
            # at M=32; the crossover between them was never measured, so the threshold is
            # an env-tunable GUESS sitting between the two verified points and is
            # labelled as such rather than presented as a tuned value.
            # Validated against BOTH kernel bounds (see prefill_min_m_env /
            # PREFILL_MIN_M_BOUNDS): too low aborts inside prefill_wmma, too high hands
            # decode_base an M its batched kernel silently truncates to 16 rows.
            prefill_min_m = prefill_min_m_env()
            if M >= prefill_min_m:
                y = mod.prefill_wmma(x2d, code, scale, zp, N, M, K, gs, False,
                                     KERNEL_FMT, split)
                _FIRE["prefill_calls"] += 1
                _fire("OMMX_W_PREFILL_FIRED", f"M={M} N={N} K={K} gs={gs}")
            elif _env_on("OMMX_W_E8M0"):
                # Lever #4: hand the kernel the int8 exponent so it never reads the fp32
                # scale plane. The gate proved this is BIT-EXACT to the fp32 path
                # (max_diff 0.00e+00), but it is a DIFFERENT kernel argument list, so it
                # is opt-in rather than the default call.
                scale_exp = getattr(layer, param_name_for_plane("scale_exp")).data
                y = mod.decode_base(x2d, code, scale, zp, N, K, gs, False,
                                    KERNEL_FMT, split, scale_exp)
                _FIRE["decode_calls"] += 1
                _fire("OMMX_W_DECODE_FIRED", f"M={M} N={N} K={K} gs={gs} e8m0=1")
            else:
                y = mod.decode_base(x2d, code, scale, zp, N, K, gs, False,
                                    KERNEL_FMT, split)
                _FIRE["decode_calls"] += 1
                _fire("OMMX_W_DECODE_FIRED", f"M={M} N={N} K={K} gs={gs} e8m0=0")

            if self.recipe.has_outliers:
                # Resolved ONCE at load by _resolve_outlier_planes. With no transcode
                # these three ARE the parameters' own .data tensors (identity, not a
                # copy), so this is the same call the shipped path always made; with a
                # transcode they are the re-encoded relidx7 stream and the materialised
                # range map. ocode is untouched either way — the FP4 nibble plane is
                # shared between the two position encodings by construction.
                oindex = layer.ommx_w_oindex_k
                ocode = getattr(layer, param_name_for_plane("ocode")).data
                map_scale = layer.ommx_w_map_scale_k
                map_center = layer.ommx_w_map_center_k
                # block == group_size: the packer writes one relidx7 slot stream per
                # GROUP (w_packer's "block == group (B=group_size)").
                #
                # sparse_correct is TWO functions behind one name. Its last two lines are
                #
                #   if (M == 1) { TORCH_CHECK(A_opt.has_value(),  "M==1 correct needs A");
                #                 go1(...); }
                #   else        { TORCH_CHECK(At_opt.has_value(), "M>1 correct needs At");
                #                 goM(...); }
                #
                # so the argument list is chosen by M, and until this release apply()
                # always sent the M==1 list. Every prefill step of every outlier-carrying
                # recipe therefore died on "M>1 correct needs At". The shipped parity gate
                # passed 5/5 because its single sparse_correct call site is at M=1.
                if M == 1:
                    # SHIPPED PATH, BYTE-IDENTICAL. go1: A bf16 [1,K] as A_opt, At_opt
                    # NULL, out fp32 [1,N] accumulated in place. This is verbatim the
                    # call the parity gate's "decode M=1 base+outlier" case measures.
                    mod.sparse_correct(y, x2d, None, code, scale, oindex, ocode,
                                       map_scale, map_center, N, M, K, gs,
                                       int(self.recipe.npv), gs, KERNEL_FMT)
                else:
                    # goM: At bf16 [K,M] CONTIGUOUS as At_opt, A_opt NULL, C bf16 [M,N].
                    # UNVERIFIED on hardware — see sparse_correct_m_gt_1_operands for the
                    # derivation, the transpose cost, and the falsification criteria.
                    y, at = sparse_correct_m_gt_1_operands(y, x2d)
                    mod.sparse_correct(y, None, at, code, scale, oindex, ocode,
                                       map_scale, map_center, N, M, K, gs,
                                       int(self.recipe.npv), gs, KERNEL_FMT)
                _FIRE["outlier_calls"] += 1
                _fire("OMMX_W_OUTLIER_FIRED", f"npv={self.recipe.npv} M={M}")

            _FIRE["apply_calls"] += 1
            _FIRE["tokens"] += M
            out = y.to(x.dtype).reshape(out_shape)
            if bias is not None:
                out = out + bias
            return out

    _CLASSES["config"] = OMMXWConfig
    _CLASSES["method"] = OMMXWLinearMethod
    return OMMXWConfig, OMMXWLinearMethod


def ommx_w_config_class():
    """The ``OMMXWConfig`` class. Imports vLLM — call only where vLLM is required."""
    return _build_classes()[0]


def ommx_w_linear_method_class():
    """The ``OMMXWLinearMethod`` class. Imports vLLM."""
    return _build_classes()[1]


def __getattr__(name: str):
    """PEP 562 lazy attribute access for the two vLLM-dependent class names.

    ``from ...linear_method import OMMXWConfig`` therefore works where vLLM is present
    and raises a NAMED error where it is not, while a bare ``import linear_method``
    still costs nothing and needs no vLLM.
    """
    if name == "OMMXWConfig":
        return ommx_w_config_class()
    if name == "OMMXWLinearMethod":
        return ommx_w_linear_method_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ════════════════════════════════════════════════════════════════════════════
# registration — the entry point bench_e2e_a100.py contracts on
# ════════════════════════════════════════════════════════════════════════════

def register_ommx_w(method_name: str = OMMX_W_METHOD_NAME) -> Optional[str]:
    """Register the OMMX weight-quant method under ``"ommx_w"``. Idempotent.

    Returns the registered name, or ``None`` when this process has no vLLM
    quantization registry to register into — the same shape as
    ``plugin.register()``'s ``None`` for a missing v1 attention registry, and for the
    same reason: the entry point must be callable on a CPU box (a packer host, a test
    runner) without pretending a serving stack exists.

    A ``None`` return is NOT a licence to serve: nothing can select ``--quantization
    ommx_w`` in a process with no vLLM. The loud failures live one step later, where a
    layer is actually built.

    Must run BEFORE ``EngineArgs`` is constructed — vLLM validates ``--quantization``
    against its registry at argument-parse time, so a late registration reads as an
    unknown method.
    """
    global _REGISTERED, _REGISTERED_AS
    if _REGISTERED:
        return _REGISTERED_AS
    try:
        from vllm.model_executor.layers.quantization import register_quantization_config
    except ImportError:
        # No vLLM (or a tree with no runtime quantization registry). No-op, exactly like
        # plugin.register() does for the v1 attention registry.
        _REGISTERED = True
        _REGISTERED_AS = None
        return None

    cfg_cls = ommx_w_config_class()
    try:
        register_quantization_config(method_name)(cfg_cls)
    except ValueError as exc:
        # vLLM raises ValueError when the name is already taken. Tolerable ONLY when the
        # incumbent is this exact class (a second worker import in the same process);
        # anything else means two different implementations are fighting over the name,
        # and picking one silently is how a run measures a method nobody selected.
        incumbent = _registered_config(method_name)
        if incumbent is not cfg_cls:
            raise OMMXWError(
                f"cannot register quantization method {method_name!r}: it is already "
                f"registered to {incumbent!r}, which is not this module's OMMXWConfig "
                f"({cfg_cls!r}). Two implementations claiming one --quantization name "
                f"means the engine would run whichever won the import race. Original "
                f"error: {exc}") from exc
    _REGISTERED = True
    _REGISTERED_AS = method_name
    return method_name


def _registered_config(method_name: str):
    """The class currently registered under ``method_name``, or None. Best effort."""
    try:
        from vllm.model_executor.layers.quantization import get_quantization_config
        return get_quantization_config(method_name)
    except Exception:                          # noqa: BLE001 - diagnostics only
        return None


__all__ = [
    "OMMX_W_METHOD_NAME",
    "KERNEL_FMT",
    "KERNEL_OUTLIER_REPRS",
    "KERNEL_OUTLIER_MAPS",
    "OMMXWError",
    "OMMXWBundle",
    "register_ommx_w",
    "resolve_bundle_dir",
    "require_kernel_readable",
    "transcode_requested",
    "TRANSCODE_ENV",
    "TRANSCODE_CONFIG_KEYS",
    "load_bundle",
    "is_ommx_w_bundle",
    "build_kernel",
    "bundle_to_param_name",
    "remap_bundle_weight_names",
    "param_name_for_plane",
    "ommx_w_fire_stats",
    "ommx_w_health",
    "ommx_w_config_class",
    "ommx_w_linear_method_class",
]
