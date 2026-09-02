# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""vLLM ``general_plugins`` entry point — register the OMMX backend + quant method.

Registers the OMMX paged-decode attention backend under ``AttentionBackendEnum.CUSTOM``
(KV-quantized decode; select with ``--attention-backend CUSTOM`` / EngineArg
``attention_backend=CUSTOM``).
  WARNING: vLLM 0.21 does NOT read the ``VLLM_ATTENTION_BACKEND`` env var — setting it is
  silently ignored and the platform picks FLASH_ATTN (the OMMX KV backend never engages).
  You MUST pass ``attention_backend`` via EngineArgs/CLI. Confirm engagement in the log:
  ``Using AttentionBackendEnum.CUSTOM backend.`` + a Triton JIT line for
  ``_canonical_splitkv_stage1`` (the OMMX decode kernel firing).

It ALSO registers the OMMX weight-quant linear method under the quantization name
``ommx_w`` (select with ``--quantization ommx_w`` and a model that is an
``OMMX_W_SafeTensor`` bundle — see ``ommx_gpu_serve/linear/w_packer.py``). The two are
independent axes: ``ommx_w`` quantizes WEIGHTS, ``CUSTOM`` quantizes the KV cache, and
either can be selected without the other.
  UNVERIFIED (no GPU this session): the ``ommx_w`` path has never executed against a
  device. What is CPU-verified is listed in ``linear_method.py``'s module docstring.

Runs once per worker (TP + PP) and is idempotent. Declared in ``pyproject.toml``::

    [project.entry-points."vllm.general_plugins"]
    ommx_gpu_serve = "ommx_gpu_serve.integration.vllm.plugin:register"

Importable with no vLLM installed; each registration no-ops when this vLLM lacks
the corresponding registry.
"""
from __future__ import annotations

import os
import sys as _sys
from typing import Optional

_REGISTERED = False
_NO_REGISTRY_NOTED = False

# The single source of truth for the class path we hand to vLLM. Registration and the
# read-back verification below both use it, so a rename cannot make the verifier compare
# against a stale string and report "registered" for a path vLLM does not hold.
_OMMX_BACKEND_PATH = "ommx_gpu_serve.integration.vllm.backend.OMMXCanonicalBackend"


def _attention_registered() -> Optional[bool]:
    """Did the CUSTOM attention override actually land, per vLLM's OWN registry?

    True / False, or None when this vLLM has no v1 attention registry to ask.

    Read back rather than inferred from "register_backend() did not raise": the point of
    this check is to make the log a statement about the registry's state, not about our
    own control flow. Uses ``get_path()`` (a dict lookup returning the string we stored),
    NOT ``get_class()``, which would import the backend module and defeat the lazy-import
    design that keeps vLLM's FlashAttention base out of a plain ``import ommx_gpu_serve``.
    """
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except Exception:  # noqa: BLE001 - verification only
        return None
    try:
        custom = AttentionBackendEnum.CUSTOM
        if not custom.is_overridden():
            return False
        return custom.get_path(include_classname=True) == _OMMX_BACKEND_PATH
    except Exception:  # noqa: BLE001
        return None


def _ommx_w_registered() -> Optional[bool]:
    """Is ``ommx_w`` currently registered to OUR config class, per vLLM's own registry?

    True / False, or None when the question cannot be answered (no quantization registry,
    or our class cannot be constructed to compare against). Distinct from
    :func:`_ommx_w_name_claimed_by_a_foreign_config`, which asks the opposite question
    (does somebody ELSE own the name) and is used to classify a registration failure.
    """
    try:
        from vllm.model_executor.layers.quantization import get_quantization_config
    except Exception:  # noqa: BLE001
        return None
    try:
        from .linear_method import OMMX_W_METHOD_NAME, ommx_w_config_class
        incumbent = get_quantization_config(OMMX_W_METHOD_NAME)
    except Exception:  # noqa: BLE001
        return False                        # unknown name -> not registered
    try:
        return incumbent is ommx_w_config_class()
    except Exception:  # noqa: BLE001
        return None


def _report_registration() -> None:
    """Say what is registered — and only what the registry confirms.

    WHY THIS EXISTS. The plugin used to print on failure only, so a load that registered
    NOTHING and a load that registered BOTH axes looked identical on stdout: silence. The
    two axes are independent and one of them fails quietly in real trees (``ommx_w`` is
    rejected by name when the quantization registry is absent), so "the plugin loaded" is
    not the same claim as "OMMX is registered". Every line below is a read-back of vLLM's
    own registry, never an echo of our own control flow.
    """
    attn, wq = _attention_registered(), _ommx_w_registered()

    def _state(v, yes, no, unknown):
        return yes if v is True else (no if v is False else unknown)

    print(
        "[ommx] "
        + _state(attn,
                 "attention: REGISTERED (AttentionBackendEnum.CUSTOM -> "
                 "OMMXCanonicalBackend; select with --attention-backend CUSTOM)",
                 "attention: NOT registered (CUSTOM does not resolve to OMMX)",
                 "attention: unknown (this vLLM has no v1 attention backend registry)")
        + " | "
        + _state(wq,
                 "ommx_w: REGISTERED (--quantization ommx_w)",
                 "ommx_w: NOT registered (--quantization ommx_w will be rejected "
                 "as an unknown method)",
                 "ommx_w: unknown (no quantization registry to ask)"),
        file=_sys.stderr, flush=True)


def _packed_only_requested() -> bool:
    """True when the operator asked for PACKED-ONLY via the env.

    Mirrors ``packed_only.packed_only_enabled`` WITHOUT importing that module, so the
    plugin can still tell whether the failure it just caught matters. Kept in sync by
    hand; both read ``OMMX_KV_PACKED_ONLY`` with the repo's usual off-ish spellings.
    """
    return os.environ.get("OMMX_KV_PACKED_ONLY", "0").strip().lower() not in {
        "", "0", "false", "off", "no"}


def _ommx_w_name_claimed_by_a_foreign_config() -> Optional[bool]:
    """Is ``"ommx_w"`` currently registered to something that is NOT our config class?

    Returns True (a foreign implementation owns the name), False (nobody else owns it),
    or None when the question cannot be answered in this process — no vLLM
    quantization registry, or the name is simply unregistered (vLLM's
    ``get_quantization_config`` raises for an unknown method).

    WHY A STATE CHECK RATHER THAN AN EXCEPTION TYPE. ``register_ommx_w`` raises
    ``OMMXWError`` for TWO different situations and the type alone cannot separate
    them (see :func:`_register_ommx_w_quant` for the full argument). What separates
    them is exactly this: after the failure, does the name belong to somebody else?
    """
    try:
        from vllm.model_executor.layers.quantization import get_quantization_config
    except Exception:                       # noqa: BLE001 - classification only
        return None                         # no registry -> nothing can be claimed
    try:
        from .linear_method import OMMX_W_METHOD_NAME, ommx_w_config_class
        incumbent = get_quantization_config(OMMX_W_METHOD_NAME)
    except Exception:                       # noqa: BLE001
        return None                         # unknown name (the usual case) -> no claim
    try:
        ours = ommx_w_config_class()
    except Exception:                       # noqa: BLE001
        # We cannot build our own class, so we cannot say whether the incumbent is it.
        # Undeterminable, NOT "no collision" — the caller has a second discriminator.
        return None
    return incumbent is not ours


def _register_ommx_w_quant() -> None:
    """Register the ``ommx_w`` weight-quant method, or explain why it is unavailable.

    THE DECISION THIS FUNCTION ENCODES (audit finding; it used to be a blanket
    ``except Exception`` + stderr note for every failure):

      * BENIGN -> note, do not raise. "This vLLM has no quantization registry", "this
        vLLM has no Linear/quantization base classes", "``linear_method`` is not
        importable in this process". In all of these the name ``ommx_w`` stays
        UNREGISTERED, so vLLM rejects ``--quantization ommx_w`` BY NAME at
        argument-parse time: the operator still gets a loud, accurate refusal, one
        step later and from the component that actually needs it. Raising here would
        instead take down every KV-only run — this function executes in EVERY worker
        of EVERY run, including runs that never touch a weight bundle.
      * FOREIGN NAME COLLISION -> RE-RAISE. A different quantization config already
        owns ``"ommx_w"``. The "vLLM will reject it by name" argument above DOES NOT
        HOLD here, and that asymmetry is the whole point: the name resolves fine, so
        the engine starts and serves with SOMEBODY ELSE'S implementation under the
        name the operator selected. A run then measures a method nobody chose, decided
        by import order. That must not be swallowed.

    WHICH EXCEPTION SIGNALS WHICH — READ FROM ``linear_method.register_ommx_w``:

      * a missing quantization registry is handled INSIDE ``register_ommx_w`` (it
        catches ``ImportError`` and returns None), so it never reaches this except at
        all;
      * ``_import_vllm_bases`` raises ``OMMXWError`` ``from ImportError`` when the
        Linear/quantization bases are absent — registration did NOT happen;
      * the collision branch raises ``OMMXWError`` ``from ValueError`` — registration
        did not happen either, but the NAME IS TAKEN.

    So the exception TYPE does not distinguish the benign case from the collision:
    both are ``OMMXWError``. That is not a blocking defect, because the two are
    distinguishable by the thing that actually matters — whether the name ended up
    owned by a foreign class — which :func:`_ommx_w_name_claimed_by_a_foreign_config`
    asks the registry directly. The ``__cause__`` type is used only as a fallback for
    the case where that question cannot be answered (it is exact against the source
    above: the collision branch is the ONLY ``raise ... from`` a ``ValueError`` in
    ``register_ommx_w``), so an unanswerable registry lookup cannot silently downgrade
    a collision to a note.

    A caller that INTENDS to serve ommx_w must still call
    ``linear_method.register_ommx_w()`` directly and let EVERY failure raise — that is
    what ``bench/bench_e2e_a100.py`` does before it builds an ommx_w arm. This
    function is the plugin-load path, which cannot assume that intent.
    """
    try:
        from .linear_method import register_ommx_w
    except Exception as exc:  # noqa: BLE001
        # The module itself did not import (no torch, a broken tree, ...). Nothing was
        # registered, so the by-name rejection above applies. Note and carry on.
        _ommx_w_note(exc)
        return
    try:
        register_ommx_w()
    except Exception as exc:  # noqa: BLE001
        claimed = _ommx_w_name_claimed_by_a_foreign_config()
        if claimed is None:
            # Registry could not answer. Fall back to the cause-type discriminator
            # rather than assuming "benign" — assuming benign is what let a real
            # conflict be settled by import order.
            claimed = isinstance(exc.__cause__, ValueError)
        if claimed:
            raise
        _ommx_w_note(exc)


def _ommx_w_note(exc: BaseException) -> None:
    """The stderr note for a BENIGN ommx_w registration failure. Never for a collision.

    Says what is unavailable, what the operator will see instead, and what is NOT
    affected — the KV-quant CUSTOM attention backend is a separate axis and keeps
    working.
    """
    print(f"[ommx] note: the ommx_w weight-quant method could not be registered "
          f"({type(exc).__name__}: {exc}); --quantization ommx_w will be rejected by "
          f"vLLM as an unknown method. The KV-quant CUSTOM attention backend is "
          f"unaffected.", file=_sys.stderr, flush=True)


def register(method_name: str = "ommx") -> Optional[str]:
    """Idempotent: record ``OMMXCanonicalBackend`` under ``AttentionBackendEnum.CUSTOM``
    and the ``ommx_w`` weight-quant config.

    Returns the attention enum member name to select ("CUSTOM"), or None when this
    vLLM has no v1 attention-backend registry (too old / unsupported — there is no
    monkeypatch fallback in the standalone package). The ``ommx_w`` quant config is
    registered independently (its own try/except) so it survives even on that path.
    """
    global _REGISTERED
    # WEIGHT-QUANT FIRST, and OUTSIDE the attention latch below. The two registries are
    # independent (a tree can have the quantization registry and not the v1 attention one,
    # or vice versa), and the attention path returns early when its registry is missing —
    # so registering ``ommx_w`` after that early return would silently skip it on exactly
    # the trees where the caller most needs to know. ``ommx_moe`` is still NOT shipped:
    # there is no ``moe_method`` module in this tree.
    _register_ommx_w_quant()
    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
    except ImportError:
        # NOTHING was registered on the attention axis. This used to set
        # ``_REGISTERED = True``, which made the flag mean "we tried" rather than "the
        # backend is registered" -- so a later call on a tree where the import started
        # working would take the `if not _REGISTERED` early-out and skip registration
        # while returning "CUSTOM", i.e. report registered without registering. Leave the
        # flag alone and say the state out loud (once; repeated plugin loads on a tree
        # that will never have the registry must not spam).
        global _NO_REGISTRY_NOTED
        if not _NO_REGISTRY_NOTED:
            _NO_REGISTRY_NOTED = True
            _report_registration()
        return None
    if not _REGISTERED:
        # String class path: the backend module (and vLLM's FlashAttention base)
        # imports lazily only when CUSTOM is actually selected.
        register_backend(AttentionBackendEnum.CUSTOM, _OMMX_BACKEND_PATH)
        # PACKED-ONLY capacity mode (OMMX_KV_PACKED_ONLY=1, default OFF): patch
        # Attention.get_kv_cache_spec to shrink the bf16 paged-cache page budget by the
        # OMMX plane footprint (measured 3.66x for the canonical i2f4+i2 recipe — see
        # packed_only.kv_bits_breakdown; the older "<=3-bit / ~4.6x" figures belong to
        # the group_tokens=64 + OMMX_KV_OUTLIER_MAP=0 recipe, a different number system
        # from the one the published accuracy results used). The shrunk pages are a
        # BYTE BUDGET ONLY — the real backing store is the separately allocated
        # MultiSeqKVPool. Installed here (plugin load, once per worker) so it is in
        # place before get_kv_cache_spec runs during KV-cache sizing.
        #
        # LAW #5 (no silent fallback): the backend decides PACKED-ONLY from the ENV
        # (backend._PACKED_ONLY = packed_only_enabled()), independently of whether this
        # patch installed. So swallowing an install failure produced a CONTRADICTORY
        # engine: an UNSHRUNK bf16 paged cache that do_kv_cache_update deliberately
        # never writes, and a forward() that refuses to read it. Fail at plugin load,
        # where the message can still name the cause, instead of at the first prefill.
        try:
            from .packed_only import install_packed_only_spec
        except Exception as exc:  # noqa: BLE001
            if _packed_only_requested():
                raise RuntimeError(
                    "OMMX_KV_PACKED_ONLY is set but ommx_gpu_serve.integration.vllm."
                    f"packed_only could not be imported ({type(exc).__name__}: {exc}). "
                    "PACKED-ONLY makes the sidecar the ONLY backing store, so running "
                    "without the page-budget patch would serve from an unwritten bf16 "
                    "cache. Unset OMMX_KV_PACKED_ONLY to use SHADOW mode."
                ) from exc
            # SHADOW mode (the default) does not need the patch: vLLM keeps its full
            # bf16 paged cache and every step has a valid fallback. Still say so once.
            print(f"[ommx] note: packed_only unavailable ({type(exc).__name__}); "
                  "SHADOW mode is unaffected.", file=_sys.stderr, flush=True)
        else:
            # Deliberately NOT wrapped: install_packed_only_spec() no-ops when the env
            # is unset, so any exception it raises is a genuine PACKED-ONLY failure and
            # must reach the operator rather than degrade the engine silently.
            install_packed_only_spec()
        _REGISTERED = True
        # Report AFTER the registrations, and only from the registry's own state. Inside
        # the latch so a re-entrant plugin load does not repeat the line.
        _report_registration()
    return "CUSTOM"


__all__ = ["register"]
