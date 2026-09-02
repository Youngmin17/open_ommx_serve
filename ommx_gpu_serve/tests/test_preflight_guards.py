# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Gate for the vLLM startup guards (``integration/vllm/preflight.py``).

Two halves:

  1. ``ommx_pool_bytes_per_layer`` — the pool byte PROJECTION. It calls itself "an
     EXACT mirror of MultiSeqKVPool.__init__", and a projection that has drifted off
     the allocation makes the OOM guard useless in the direction that matters
     (under-projecting, then OOMing thousands of tokens into a run). Checked twice:
     against arithmetic written out longhand here, and — the check that cannot rot —
     against a REAL ``MultiSeqKVPool`` allocated on CPU, plane by plane.

  2. ``ommx_preflight_check`` — the FOUR configurations the OMMX KV sidecar cannot
     serve (prefix caching / chunked prefill / pool oversubscription / a scheduler
     ``max_num_seqs`` above the pool's static slot cap) must RAISE, a MISSING config
     attribute must count as a violation rather than a pass, and each
     ``OMMX_UNSAFE_ALLOW_*`` env must downgrade EXACTLY ONE of them to a warning.
     The first three co-fire in one stub config, so they are parametrized together;
     the slot-cap check needs ``scheduler_config.max_num_seqs > num_seqs``, which
     that shared config deliberately does not set, so it carries its own test.

No vLLM is imported. The config is a plain stub object with the attribute chain the
guard reads (``cache_config.enable_prefix_caching``,
``scheduler_config.chunked_prefill_enabled`` / ``.max_num_seqs``,
``model_config.hf_config`` / ``.max_model_len``,
``parallel_config.tensor_parallel_size``); ``cfg`` is the real, pure-python
``OMMXServingConfig``.

Two module-level seams are patched, both deliberately and both named here:

  * ``sys.modules["vllm"]`` gets a one-attribute stub so the version check has a
    version to read. Without it EVERY call on a CPU box carries a "vLLM VERSION
    UNKNOWN" violation and the three checks under test cannot be isolated. The
    version check itself is exercised directly by
    ``test_vllm_version_floor_is_enforced``.
  * ``preflight._free_device_bytes`` is replaced by a constant, so the memory-budget
    branch is reachable with no GPU. It is a ``(free_bytes, reason)`` probe by
    design, which is exactly the seam that makes this testable.
"""
from __future__ import annotations

import sys
import types

import pytest

from ommx_gpu_serve.attention.kv_pool import MultiSeqKVPool
from ommx_gpu_serve.attention.kv_window import WindowSpec
from ommx_gpu_serve.integration.vllm.config import OMMXServingConfig

_IMPORT_ERR = None
try:
    from ommx_gpu_serve.integration.vllm import preflight as pf
except ImportError as exc:  # pragma: no cover - exercised only before the fix lands
    pf = None
    _IMPORT_ERR = f"{type(exc).__name__}: {exc}"

_XFAIL_MISSING = pytest.mark.xfail(
    _IMPORT_ERR is not None,
    reason=("ommx_gpu_serve.integration.vllm.preflight is not importable "
            f"({_IMPORT_ERR}) — the startup-guard module has not landed. XFAIL, not "
            "skip: the gate is present and failing, not absent."),
    strict=True,
)

GIB = 1024 ** 3


# ── stubs ───────────────────────────────────────────────────────────────────────

_ABSENT = object()      # "this attribute is not on the stub at all"


class _Ns:
    """Attribute bag standing in for a vLLM config node. No vLLM import."""

    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)


def _stub_vllm_config(*, prefix_caching: bool = False, chunked_prefill: bool = False,
                      sched_max_num_seqs: int = 4, max_model_len: int = 8192,
                      num_layers: int = 32, n_kv_heads: int = 8, head_dim: int = 128,
                      tp: int = 1, omit: str = "",
                      enable_chunked_prefill=_ABSENT) -> _Ns:
    """A ``VllmConfig``-shaped stub. ``omit`` drops one top-level node on purpose.

    Geometry defaults are Llama-3.1-8B (head_dim=128, 32 q heads, 8 kv heads,
    32 layers).

    ``chunked_prefill`` is the RESOLVED ``SchedulerConfig.chunked_prefill_enabled``
    field; ``enable_chunked_prefill`` is the user-facing EngineArgs knob preflight
    falls back to when the resolved one is missing or ``None``. It is ABSENT from the
    stub unless passed, because a real ``SchedulerConfig`` may carry either, and a
    stub that always carried both would hide which one the guard actually reads.
    """
    hf = _Ns(num_hidden_layers=num_layers, num_key_value_heads=n_kv_heads,
             num_attention_heads=32, head_dim=head_dim, hidden_size=head_dim * 32)
    nodes = {
        "cache_config": _Ns(enable_prefix_caching=prefix_caching),
        "scheduler_config": _Ns(chunked_prefill_enabled=chunked_prefill,
                                max_num_seqs=sched_max_num_seqs),
        "model_config": _Ns(hf_config=hf, max_model_len=max_model_len),
        "parallel_config": _Ns(tensor_parallel_size=tp),
    }
    if enable_chunked_prefill is not _ABSENT and "scheduler_config" in nodes:
        nodes["scheduler_config"].enable_chunked_prefill = enable_chunked_prefill
    nodes.pop(omit, None)
    return _Ns(**nodes)


def _serving_cfg(max_context: int = 8192) -> OMMXServingConfig:
    """The canonical published recipe as an ``OMMXServingConfig`` (the real class)."""
    return OMMXServingConfig(
        k_format="i2f4", outliers_per_vector=6, outlier_select="signed",
        outlier_repr="relidx7", use_pow2=True, sink_tokens=8, recent_window=32,
        group_tokens=32, group_channels=32, kv_outlier_map=True,
        head_dim=128, n_q_heads=32, n_kv_heads=8, page_size=16,
        max_context=max_context)


@pytest.fixture(autouse=True)
def _reset_warn_latch():
    """Clear preflight's per-process one-time warning latch around each test.

    ``_warn_once`` deduplicates by key for the lifetime of the worker process, so
    without this the SECOND test that triggers a given warning would observe nothing
    on stderr and the assertion would be testing pytest ordering, not the guard.
    """
    if pf is not None:
        pf._WARNED.clear()
    yield
    if pf is not None:
        pf._WARNED.clear()


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch):
    """Install a stub ``vllm`` module so the version check has something to read."""
    def _install(version: str = "0.21.0"):
        mod = types.ModuleType("vllm")
        mod.__version__ = version
        monkeypatch.setitem(sys.modules, "vllm", mod)
        return mod
    return _install


@pytest.fixture
def free_memory(monkeypatch: pytest.MonkeyPatch):
    """Replace the free-device-memory probe with a constant (no GPU needed)."""
    def _set(nbytes):
        monkeypatch.setattr(pf, "_free_device_bytes",
                            lambda device=None: (int(nbytes), ""))
    return _set


# ── violation classification ────────────────────────────────────────────────────

def _tags(report: dict) -> set:
    """Map each violation string to a stable tag so assertions read by CHECK, not text.

    An unrecognised violation fails loudly rather than being ignored — a new guard
    must be classified here on purpose, not slip past every assertion below.
    """
    out = set()
    for v in report["violations"]:
        head = v.splitlines()[0]
        if head.startswith("prefix caching is"):
            out.add("prefix_caching")
        elif head.startswith("chunked prefill is"):
            out.add("chunked_prefill")
        elif ("pool DOES NOT FIT" in head or "pool footprint CANNOT BE PROJECTED" in head
              or "pool budget CANNOT BE CHECKED" in head):
            out.add("pool")
        elif head.startswith("vLLM"):
            out.add("vllm_version")
        elif head.startswith("scheduler max_num_seqs="):
            # check 4 (the slot cap). MUST be tested BEFORE the generic ``num_seqs``
            # branch below: this head also contains the substring "num_seqs", so
            # without its own branch a slot-cap violation is silently ALIASED onto
            # the num_seqs<=0 tag instead of being classified — which is exactly the
            # kind of quiet mis-attribution the raise at the end of this function
            # exists to prevent.
            out.add("slot_cap")
        elif "num_seqs" in head:
            out.add("num_seqs")
        else:
            raise AssertionError(f"unclassified preflight violation: {head!r}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1. the pool byte projection
# ══════════════════════════════════════════════════════════════════════════════

def _pool_bytes_longhand(*, head_dim: int, n_kv_heads: int, num_seqs: int,
                         max_seq_len: int, group_tokens: int, page_size: int,
                         group_channels: int, outliers_per_vector: int,
                         kv_outlier_map: bool, kv_int8_scale: bool, kv_ring: bool,
                         sink_tokens: int = 8, recent_window: int = 32) -> int:
    """Independent recomputation of one layer's pool bytes (relidx7 only).

    Written out here from ``MultiSeqKVPool.__init__``'s allocation block rather than
    imported, so it cannot agree with the projection by construction.
    """
    D, H, B, S = head_dim, n_kv_heads, num_seqs, max_seq_len
    gt, ps, gc, k = group_tokens, page_size, group_channels, outliers_per_vector
    ppg = gt // ps
    g_per_seq = max(1, (S + gt - 1) // gt + 1)      # +1 slack group
    p_per_seq = g_per_seq * ppg
    g_cap, p_cap = B * g_per_seq, B * p_per_seq
    ngv = D // gc
    scale_b = 1 if kv_int8_scale else 2             # int8 pow2 exponent, else bf16
    bf16_b = 2

    total = 0
    total += p_cap * ps * H * (D // 4)              # k_base   uint8
    total += p_cap * ps * H * (D // 4)              # v_main   uint8 (V is always i2)
    total += g_cap * H * D * scale_b                # k_scale
    total += g_cap * H * D * bf16_b                 # k_zp     (never int8)
    total += p_cap * ps * H * ngv * scale_b         # v_scale
    total += p_cap * ps * H * ngv * bf16_b          # v_zp
    if k > 0 and kv_outlier_map:                    # the dedicated FP4 range map
        total += 2 * g_cap * H * D * bf16_b
    if k > 0:
        total += g_cap * H * D * ((7 * k + 7) // 8)  # k_oidx (relidx7)
        total += g_cap * H * D * ((k + 1) // 2)      # k_oval (FP4 nibbles)
    ring_cap = (sink_tokens + recent_window + 2 * gt) if kv_ring else S
    total += 2 * B * ring_cap * H * D * bf16_b      # k_hist + v_hist
    # The two int32 identity page/group tables. kv_pool.py builds them as
    # torch.arange(G_cap, dtype=int32).reshape(num_seqs, G_per_seq) and the same for
    # P_cap, so they are G_cap*4 + P_cap*4 bytes and they scale with num_seqs exactly
    # like the planes above. Small (0.03% at the 256x131072 projection) but real: the
    # projection is a REFUSAL threshold, so it must not under-count anything the pool
    # actually allocates.
    total += (g_cap + p_cap) * 4                   # req_to_group + req_to_token (int32)
    return total


_GEOMETRIES = [
    # (num_seqs, max_seq_len, head_dim, n_kv_heads, gt, gc, ps, outliers, pow2,
    #  outlier_map, ring)
    (2, 256, 128, 2, 32, 32, 16, 6, True, True, False),
    (3, 512, 128, 4, 32, 32, 16, 6, True, True, True),
    (2, 1024, 64, 2, 64, 64, 16, 3, False, False, False),
    # DISCRIMINATING geometry: the three above all have group_tokens == group_channels
    # and outliers in {3, 6}. That is degenerate twice over -- a gt/gc swap in the
    # V-plane sizing (NGV = D // gc) is invisible when they are equal, and relidx7's
    # ceil(7k/8) index bytes coincide with a plain 8-bit ceil(8k/8) at exactly k in
    # {3, 6} (3 and 6 bytes). Both mistakes were mutation-tested and survived without
    # this row. gt=32 != gc=64 and k=8 (7 index bytes, not 8) separate them.
    (2, 512, 128, 2, 32, 64, 16, 8, True, True, False),
]


@_XFAIL_MISSING
@pytest.mark.parametrize("geom", _GEOMETRIES)
def test_pool_byte_projection_matches_longhand_recomputation(geom) -> None:
    (num_seqs, S, D, H, gt, gc, ps, k, pow2, omap, ring) = geom
    got = pf.ommx_pool_bytes_per_layer(
        D, H, num_seqs, S, group_tokens=gt, page_size=ps, group_channels=gc,
        outliers_per_vector=k, outlier_repr="relidx7", kv_outlier_map=omap,
        kv_int8_scale=pow2, kv_ring=ring, sink_tokens=8, recent_window=32)
    exp = _pool_bytes_longhand(
        head_dim=D, n_kv_heads=H, num_seqs=num_seqs, max_seq_len=S, group_tokens=gt,
        page_size=ps, group_channels=gc, outliers_per_vector=k, kv_outlier_map=omap,
        kv_int8_scale=pow2, kv_ring=ring)
    assert got["total"] == exp, (
        f"{geom}: projection total={got['total']} != longhand {exp} "
        f"(diff {got['total'] - exp} bytes)")
    # "total" must really be the sum of the enumerated planes, not a second formula.
    plane_sum = sum(v for key, v in got.items()
                    if key not in ("total", "geometry"))
    assert plane_sum == got["total"], (
        f"{geom}: per-plane sum {plane_sum} != reported total {got['total']}")


@_XFAIL_MISSING
@pytest.mark.parametrize("geom", _GEOMETRIES)
def test_pool_byte_projection_matches_a_real_allocated_pool(
        geom, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plane-for-plane against a real ``MultiSeqKVPool``. The anti-drift gate.

    ``OMMX_KV_RING`` is set on the environment (not passed as a kwarg) because the
    pool reads it at construction time and has no ring kwarg — the same coupling the
    projection has to mirror.
    """
    (num_seqs, S, D, H, gt, gc, ps, k, pow2, omap, ring) = geom
    if ring:
        monkeypatch.setenv("OMMX_KV_RING", "1")
    else:
        monkeypatch.delenv("OMMX_KV_RING", raising=False)
    win = WindowSpec(sink_tokens=8, recent_window=32, group_tokens=gt, page_size=ps)
    pool = MultiSeqKVPool(
        head_dim=D, n_kv_heads=H, num_seqs=num_seqs, max_seq_len=S, k_format="i2f4",
        outliers_per_vector=k, outlier_select="signed", outlier_repr="relidx7",
        use_pow2=pow2, kv_outlier_map=omap, window=win, group_channels=gc,
        device="cpu")
    assert pool.kv_ring is bool(ring), "the pool did not pick up the ring setting"
    # PIN the pool's own scale-dtype resolution against the geometry's declared recipe
    # BEFORE feeding it to the projection below. Without this the comparison is partly
    # circular: ``kv_int8_scale=pool.kv_int8_scale`` makes the projection follow the
    # pool, so a pool that IGNORED use_pow2 and always allocated the int8 pow2 scale
    # would keep every byte total in agreement while silently serving a different
    # recipe (mutation-tested: ``self.kv_int8_scale = True`` survived the whole suite —
    # geom2 is the only use_pow2=False row and nothing else allocates it).
    assert pool.kv_int8_scale is bool(pow2), (
        f"{geom}: use_pow2={pow2} resolved to kv_int8_scale={pool.kv_int8_scale}; the "
        f"pool is not allocating the scale dtype the recipe asks for")

    got = pf.ommx_pool_bytes_per_layer(
        D, H, num_seqs, S, group_tokens=gt, page_size=ps, group_channels=gc,
        outliers_per_vector=k, outlier_repr="relidx7", kv_outlier_map=omap,
        kv_int8_scale=pool.kv_int8_scale, kv_ring=ring, sink_tokens=8,
        recent_window=32)

    planes = ("k_base", "v_main", "k_scale", "k_zp", "v_scale", "v_zp",
              "k_fp4_mapscale", "k_fp4_mapcenter", "k_oidx", "k_crank", "k_oval",
              "k_hist", "v_hist")
    real_total = 0
    for name in planes:
        t = getattr(pool, name)
        real = 0 if t is None else t.numel() * t.element_size()
        real_total += real
        assert got[name] == real, (
            f"{geom}: projected {name}={got[name]} bytes, pool allocated {real}")
    # The two int32 identity tables are allocated by the pool as well (private attrs
    # _req_to_group / _req_to_token). They are part of the per-layer footprint the
    # refusal threshold has to cover, so the projection counts them and this comparison
    # against the REAL tensors must too -- otherwise the guard could under-count.
    for name, key in (("_req_to_group", "req_to_group"),
                      ("_req_to_token", "req_to_token")):
        t = getattr(pool, name)
        real = 0 if t is None else t.numel() * t.element_size()
        real_total += real
        assert got[key] == real, (
            f"{geom}: projected {key}={got[key]} bytes, pool allocated {real}")
    assert got["total"] == real_total, (
        f"{geom}: projected total {got['total']} != allocated {real_total}")
    # the geometry the projection reports must be the pool's own sizing
    geo = got["geometry"]
    assert geo["G_per_seq"] == pool.G_per_seq
    assert geo["P_per_seq"] == pool.P_per_seq
    assert geo["ring_cap"] == pool.ring_cap
    assert geo["NGV"] == pool.NGV


@_XFAIL_MISSING
def test_pool_byte_projection_rejects_geometries_the_pool_rejects() -> None:
    """No silent fallback: a geometry ``MultiSeqKVPool`` would refuse must not price."""
    base = dict(group_tokens=32, page_size=16, group_channels=32,
                outliers_per_vector=6, outlier_repr="relidx7")
    with pytest.raises(ValueError):
        pf.ommx_pool_bytes_per_layer(100, 8, 4, 8192, **base)        # head_dim % 32
    with pytest.raises(ValueError):
        pf.ommx_pool_bytes_per_layer(128, 8, 4, 8192, **{**base, "group_tokens": 48})
    with pytest.raises(ValueError):
        pf.ommx_pool_bytes_per_layer(128, 8, 4, 8192, **{**base, "group_channels": 48})
    with pytest.raises(ValueError):
        pf.ommx_pool_bytes_per_layer(128, 8, 0, 8192, **base)        # num_seqs <= 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. the startup guards
# ══════════════════════════════════════════════════════════════════════════════

@_XFAIL_MISSING
def test_supported_config_passes(fake_vllm, free_memory) -> None:
    """The baseline the three failure tests are measured against.

    Without this, every "it raises" assertion below could be satisfied by a guard
    that simply always raises.
    """
    fake_vllm("0.21.0")
    free_memory(80 * GIB)
    report = pf.ommx_preflight_check(
        _stub_vllm_config(prefix_caching=False, chunked_prefill=False),
        num_seqs=4, cfg=_serving_cfg(), strict=True)
    assert report["ok"] is True, f"unexpected violations: {report['violations']}"
    assert _tags(report) == set()
    assert report["overrides_applied"] == []
    assert report["pool"]["status"] == "ok"


@_XFAIL_MISSING
def test_prefix_caching_raises(fake_vllm, free_memory) -> None:
    fake_vllm("0.21.0")
    free_memory(80 * GIB)
    with pytest.raises(RuntimeError) as ei:
        pf.ommx_preflight_check(_stub_vllm_config(prefix_caching=True),
                                num_seqs=4, cfg=_serving_cfg(), strict=True)
    msg = str(ei.value)
    assert "prefix caching is ENABLED" in msg
    assert "enable_prefix_caching=False" in msg, (
        f"the error must name the fix, not just the problem: {msg}")
    report = pf.ommx_preflight_check(_stub_vllm_config(prefix_caching=True),
                                     num_seqs=4, cfg=_serving_cfg(), strict=False)
    assert _tags(report) == {"prefix_caching"}


@_XFAIL_MISSING
def test_chunked_prefill_raises(fake_vllm, free_memory) -> None:
    fake_vllm("0.21.0")
    free_memory(80 * GIB)
    with pytest.raises(RuntimeError) as ei:
        pf.ommx_preflight_check(_stub_vllm_config(chunked_prefill=True),
                                num_seqs=4, cfg=_serving_cfg(), strict=True)
    msg = str(ei.value)
    assert "chunked prefill is ENABLED" in msg
    assert "enable_chunked_prefill=False" in msg
    report = pf.ommx_preflight_check(_stub_vllm_config(chunked_prefill=True),
                                     num_seqs=4, cfg=_serving_cfg(), strict=False)
    assert _tags(report) == {"chunked_prefill"}


@_XFAIL_MISSING
def test_pool_oversubscription_raises(fake_vllm, free_memory) -> None:
    """The projected pool must be refused when it exceeds the memory budget.

    Same config that passes at 80 GiB free; only the free-memory probe changes, so
    the failure is attributable to the budget check and nothing else.
    """
    fake_vllm("0.21.0")
    free_memory(1 * GIB)                       # budget = 0.5 * 1 GiB, pool ~5.1 GiB
    with pytest.raises(RuntimeError) as ei:
        pf.ommx_preflight_check(_stub_vllm_config(), num_seqs=4,
                                cfg=_serving_cfg(), strict=True)
    msg = str(ei.value)
    assert "DOES NOT FIT" in msg
    assert "OMMX_ATTN_MAX_NUM_SEQS" in msg and "--max-model-len" in msg, (
        f"the OOM error must name both sizing knobs: {msg}")
    report = pf.ommx_preflight_check(_stub_vllm_config(), num_seqs=4,
                                     cfg=_serving_cfg(), strict=False)
    assert _tags(report) == {"pool"}


@_XFAIL_MISSING
def test_unreadable_free_memory_is_a_violation_not_a_pass(fake_vllm,
                                                          monkeypatch) -> None:
    """"unknown" must never count as "safe" — the guard's own stated rule."""
    fake_vllm("0.21.0")
    monkeypatch.setattr(pf, "_free_device_bytes",
                        lambda device=None: (None, "no CUDA device in this test"))
    report = pf.ommx_preflight_check(_stub_vllm_config(), num_seqs=4,
                                     cfg=_serving_cfg(), strict=False)
    assert _tags(report) == {"pool"}
    assert "CANNOT BE CHECKED" in "\n".join(report["violations"])


@_XFAIL_MISSING
@pytest.mark.parametrize("omit,tag", [("cache_config", "prefix_caching"),
                                      ("scheduler_config", "chunked_prefill")])
def test_missing_config_attribute_is_a_violation_not_a_pass(
        omit, tag, fake_vllm, free_memory) -> None:
    """A config node the guard cannot read must fail, never silently pass."""
    fake_vllm("0.21.0")
    free_memory(80 * GIB)
    report = pf.ommx_preflight_check(_stub_vllm_config(omit=omit), num_seqs=4,
                                     cfg=_serving_cfg(), strict=False)
    assert tag in _tags(report), (
        f"omitting {omit} produced {_tags(report)} — the missing attribute was "
        f"treated as a pass")
    assert "UNKNOWN" in "\n".join(report["violations"])


@_XFAIL_MISSING
@pytest.mark.parametrize("kw,tag", [(dict(prefix_caching=None), "prefix_caching"),
                                    (dict(chunked_prefill=None), "chunked_prefill")])
def test_unresolved_none_knob_is_a_violation_not_a_pass(kw, tag, fake_vllm,
                                                        free_memory) -> None:
    """``None`` means UNRESOLVED, not "off" — and unknown is never "safe".

    Both knobs are ``Optional[bool]`` on EngineArgs and vLLM v1 turns an unset one
    into True, so folding ``None`` into ``False`` would let the EXACT default config
    this guard exists to refuse start silently — the original finding. preflight
    demotes ``None`` to UNKNOWN (``pc_found = pc_found and pc_val is not None``);
    without a test, deleting either demotion line passes the whole suite.

    Distinct from ``test_missing_config_attribute_is_a_violation_not_a_pass``: there
    the whole config NODE is gone, here the node is present and the value is None.
    """
    fake_vllm("0.21.0")
    free_memory(80 * GIB)
    report = pf.ommx_preflight_check(_stub_vllm_config(**kw), num_seqs=4,
                                     cfg=_serving_cfg(), strict=False)
    assert _tags(report) == {tag}, (
        f"{kw} produced {_tags(report)} — None was treated as 'disabled'")
    assert "UNKNOWN" in "\n".join(report["violations"])
    assert report["ok"] is False


@_XFAIL_MISSING
@pytest.mark.parametrize("engineargs,expect", [(True, {"chunked_prefill"}),
                                               (False, set()),
                                               # BOTH spellings unresolved: the fallback
                                               # returns None too. That is the ONLY path
                                               # that reaches ``cp_found = cp_found and
                                               # cp_val is not None``; without this row
                                               # that demotion can be deleted and the
                                               # whole suite stays green (mutation-
                                               # tested).
                                               (None, {"chunked_prefill"})])
def test_chunked_prefill_falls_back_to_the_engineargs_knob(engineargs, expect,
                                                           fake_vllm,
                                                           free_memory) -> None:
    """When ``chunked_prefill_enabled`` is None, ``enable_chunked_prefill`` decides.

    preflight reads the RESOLVED SchedulerConfig field first and falls back to the
    EngineArgs knob. All three directions matter: ``True`` must still refuse, ``False``
    must actually clear the violation rather than leaving a permanent UNKNOWN (which
    would refuse every engine that only carries the EngineArgs spelling), and ``None``
    on BOTH spellings must stay UNKNOWN rather than folding into "off".
    """
    fake_vllm("0.21.0")
    free_memory(80 * GIB)
    report = pf.ommx_preflight_check(
        _stub_vllm_config(chunked_prefill=None, enable_chunked_prefill=engineargs),
        num_seqs=4, cfg=_serving_cfg(), strict=False)
    assert _tags(report) == expect, (
        f"chunked_prefill_enabled=None + enable_chunked_prefill={engineargs} gave "
        f"{_tags(report)}, expected {expect}")


# ── check 4: scheduler concurrency vs the pool slot cap ─────────────────────────

@_XFAIL_MISSING
def test_scheduler_concurrency_above_the_slot_cap_is_a_violation(
        fake_vllm, free_memory) -> None:
    """``--max-num-seqs`` above the pool's static slot count must be refused.

    The pool owns exactly ``num_seqs`` slots; the first step that needs one more makes
    ``assign_slots`` raise ``OMMXSlotAllocationError`` thousands of tokens into a run.
    This is preflight's FOURTH violation and nothing else in this file reaches it —
    every other test happens to run with ``sched_max_num_seqs == num_seqs``.
    """
    fake_vllm("0.21.0")
    free_memory(80 * GIB)
    report = pf.ommx_preflight_check(_stub_vllm_config(sched_max_num_seqs=8),
                                     num_seqs=4, cfg=_serving_cfg(), strict=False)
    assert _tags(report) == {"slot_cap"}, f"got {_tags(report)}"
    joined = "\n".join(report["violations"])
    assert "EXCEEDS the OMMX pool slot cap" in joined
    assert "OMMX_ATTN_MAX_NUM_SEQS" in joined and "--max-num-seqs" in joined, (
        f"the error must name both sizing knobs: {joined}")
    assert report["ok"] is False
    # the equal case is the control: it must NOT fire.
    ok = pf.ommx_preflight_check(_stub_vllm_config(sched_max_num_seqs=4), num_seqs=4,
                                 cfg=_serving_cfg(), strict=False)
    assert _tags(ok) == set(), f"sched==cap must be fine; got {_tags(ok)}"


@_XFAIL_MISSING
def test_slot_cap_override_downgrades_that_check_and_warns(
        fake_vllm, free_memory, monkeypatch, capsys) -> None:
    """The FOURTH ``OMMX_UNSAFE_ALLOW_*`` env, which ``_OVERRIDES`` above omits."""
    fake_vllm("0.21.0")
    free_memory(80 * GIB)
    monkeypatch.setenv("OMMX_UNSAFE_ALLOW_SLOT_CAP_OVERSUBSCRIBE", "1")
    report = pf.ommx_preflight_check(_stub_vllm_config(sched_max_num_seqs=8),
                                     num_seqs=4, cfg=_serving_cfg(), strict=False)
    assert _tags(report) == set(), f"the override did not clear it: {_tags(report)}"
    assert report["overrides_applied"] == ["OMMX_UNSAFE_ALLOW_SLOT_CAP_OVERSUBSCRIBE"]
    assert report["ok"] is True
    err = capsys.readouterr().err
    assert "OMMX_UNSAFE_ALLOW_SLOT_CAP_OVERSUBSCRIBE" in err, (
        f"suppressed WITHOUT warning on stderr; captured:\n{err}")


@_XFAIL_MISSING
def test_every_unsafe_override_env_in_preflight_is_covered_here() -> None:
    """No fifth override may be added without a test — this file is the enumeration.

    ``_OVERRIDES`` covers three; ``test_slot_cap_override_downgrades_that_check_and_warns``
    covers the fourth. Reading the module source keeps the list honest.
    """
    import re
    src = open(pf.__file__).read()
    found = {m for m in re.findall(r"OMMX_UNSAFE_ALLOW_[A-Z_]+", src)
             if not m.endswith("ALLOW_")}
    covered = {env for env, _ in _OVERRIDES} | {
        "OMMX_UNSAFE_ALLOW_SLOT_CAP_OVERSUBSCRIBE"}
    assert found == covered, (
        f"preflight.py defines {sorted(found - covered)} with no test here / this "
        f"file claims {sorted(covered - found)} that preflight.py does not define")


# ── B>1 reachability: the switch that turns the pool budget check on and off ────

@_XFAIL_MISSING
def test_unreadable_scheduler_concurrency_still_projects_the_pool(
        fake_vllm, free_memory) -> None:
    """An unreadable ``max_num_seqs`` must count as REACHABLE, not as "no pool".

    ``reachable`` gates the entire pool-OOM check. Flipping the unknown branch to
    ``False`` silently disables the guard that the 1123 GiB projection finding exists
    for, and — without this test — passes the whole suite.
    """
    fake_vllm("0.21.0")
    free_memory(1 * GIB)
    report = pf.ommx_preflight_check(_stub_vllm_config(sched_max_num_seqs=None),
                                     num_seqs=4, cfg=_serving_cfg(), strict=False)
    assert report["pool"]["scheduler_max_num_seqs"] == "unknown"
    assert report["pool"]["b_gt_1_reachable"] is True, (
        f"unknown was treated as 'no batched step': {report['pool']}")
    assert _tags(report) == {"pool"}, f"got {_tags(report)}"


@_XFAIL_MISSING
def test_provably_unreachable_pool_is_informational_not_a_violation(
        fake_vllm, free_memory) -> None:
    """The complement: ``max_num_seqs=1`` with the batched force off allocates nothing.

    Refusing here would refuse every single-sequence run. The projection is still
    reported, under a key that cannot be read as a budget verdict.
    """
    fake_vllm("0.21.0")
    free_memory(1 * GIB)                        # would NOT fit if it were built
    report = pf.ommx_preflight_check(_stub_vllm_config(sched_max_num_seqs=1),
                                     num_seqs=4, cfg=_serving_cfg(), strict=False)
    assert report["pool"]["b_gt_1_reachable"] is False, f"{report['pool']}"
    assert _tags(report) == set(), f"a run that allocates no pool was refused: "\
                                   f"{report['violations']}"
    assert report["ok"] is True
    info = report["pool_projection_informational"]
    assert info["would_fit_at_budget"] is False, (
        "the informational projection must still report that it would NOT fit")
    assert report["overrides_applied"] == [], "an override was consumed for nothing"


@_XFAIL_MISSING
@pytest.mark.parametrize("value", ["1", "no"])
def test_ommx_attn_batched_forces_the_pool_check_back_on(value, fake_vllm,
                                                         free_memory,
                                                         monkeypatch) -> None:
    """``OMMX_ATTN_BATCHED`` force-routes a B==1 decode through the batched pool.

    Same config as ``test_provably_unreachable_pool_is_informational_not_a_violation``
    (``max_num_seqs=1``), so the ONLY difference is the env — the pool must go from
    informational back to a refusal. Ignoring this env re-opens the silent-OOM hole at
    ``max_num_seqs=1``.

    ``"no"`` is parametrized on purpose: ``backend._env_on`` is
    ``... not in {"0", "false", "off", ""}``, so ``OMMX_ATTN_BATCHED=no`` force-ENABLES
    the route. preflight mirrors that with ``_backend_env_on``; reading it with ordinary
    flag truthiness instead would declare "no B>1 step is possible" for a run whose
    backend batches every step.
    """
    fake_vllm("0.21.0")
    free_memory(1 * GIB)
    monkeypatch.setenv("OMMX_ATTN_BATCHED", value)
    report = pf.ommx_preflight_check(_stub_vllm_config(sched_max_num_seqs=1),
                                     num_seqs=4, cfg=_serving_cfg(), strict=False)
    assert report["pool"]["ommx_attn_batched_forced"] is True, (
        f"OMMX_ATTN_BATCHED={value!r} was read as OFF: {report['pool']}")
    assert report["pool"]["b_gt_1_reachable"] is True
    assert _tags(report) == {"pool"}, f"got {_tags(report)}"


@_XFAIL_MISSING
def test_no_cuda_device_is_skipped_not_refused(fake_vllm, monkeypatch) -> None:
    """A host with no CUDA device at all must not be refused by a GPU memory guard.

    Deliberate carve-out (preflight.py, ``_NO_CUDA_PREFIX``): distinct from
    ``test_unreadable_free_memory_is_a_violation_not_a_pass``, where CUDA exists and
    the query still failed. Uses the module's own prefix constant, because the pair
    that must stay in sync is ``_free_device_bytes``'s reason and this branch's test.
    """
    fake_vllm("0.21.0")
    monkeypatch.setattr(
        pf, "_free_device_bytes",
        lambda device=None: (None, pf._NO_CUDA_PREFIX +
                             "torch.cuda.is_available() is False"))
    report = pf.ommx_preflight_check(_stub_vllm_config(), num_seqs=4,
                                     cfg=_serving_cfg(), strict=False)
    assert _tags(report) == set(), (
        f"a CUDA-less host was refused by the memory guard: {report['violations']}")
    assert report["pool"]["status"].startswith("skipped: no CUDA device")
    assert report["ok"] is True


@_XFAIL_MISSING
def test_vllm_version_floor_is_enforced(fake_vllm, free_memory) -> None:
    """A 0.11-era vLLM tree is refused at startup, naming the real floor.

    The backend subclasses ``vllm.v1.attention.*`` internals that only exist from
    0.21, so a 0.11-era tree cannot work at all. Both pyprojects now declare
    ``vllm>=0.21`` (they used to say 0.11); this guard is what covers a tree that
    was installed without honouring that pin, and it must name ``vllm>=0.21`` in
    the refusal rather than failing later inside vLLM. The second half pins the
    other direction: a PRE-RELEASE of the floor version must still be accepted.
    """
    free_memory(80 * GIB)
    fake_vllm("0.11.0")
    report = pf.ommx_preflight_check(_stub_vllm_config(), num_seqs=4,
                                     cfg=_serving_cfg(), strict=False)
    assert _tags(report) == {"vllm_version"}
    joined = "\n".join(report["violations"])
    assert "TOO OLD" in joined and "vllm>=0.21" in joined

    fake_vllm("0.21.0rc1")     # pre-release of the floor version must be accepted
    ok = pf.ommx_preflight_check(_stub_vllm_config(), num_seqs=4,
                                 cfg=_serving_cfg(), strict=False)
    assert "vllm_version" not in _tags(ok), (
        f"0.21.0rc1 was rejected: {ok['violations']}")


@_XFAIL_MISSING
def test_num_seqs_must_be_positive(fake_vllm, free_memory) -> None:
    """``num_seqs <= 0`` clamps to 1 downstream, aliasing every request onto slot 0."""
    fake_vllm("0.21.0")
    free_memory(80 * GIB)
    report = pf.ommx_preflight_check(_stub_vllm_config(), num_seqs=0,
                                     cfg=_serving_cfg(), strict=False)
    assert "num_seqs" in _tags(report)


# ── the unsafe overrides: exactly one check each ────────────────────────────────

_OVERRIDES = [
    ("OMMX_UNSAFE_ALLOW_PREFIX_CACHING", "prefix_caching"),
    ("OMMX_UNSAFE_ALLOW_CHUNKED_PREFILL", "chunked_prefill"),
    ("OMMX_UNSAFE_ALLOW_POOL_OVERSUBSCRIBE", "pool"),
]
_ALL_THREE = {"prefix_caching", "chunked_prefill", "pool"}
# The FOURTH unsafe override, OMMX_UNSAFE_ALLOW_SLOT_CAP_OVERSUBSCRIBE (preflight
# check 4), is deliberately NOT in the list above: it only fires when
# scheduler_config.max_num_seqs EXCEEDS the pool slot cap, and the shared stub the
# parametrized test below uses passes sched_max_num_seqs == num_seqs == 4. It has its
# own pair — test_scheduler_concurrency_above_the_slot_cap_is_a_violation and
# test_slot_cap_override_downgrades_that_check_and_warns — so all four
# OMMX_UNSAFE_ALLOW_* envs are exercised, not three.


@_XFAIL_MISSING
def test_all_three_checks_fire_together_without_any_override(
        fake_vllm, free_memory) -> None:
    """The reference state for the override tests: all three violated at once."""
    fake_vllm("0.21.0")
    free_memory(1 * GIB)
    report = pf.ommx_preflight_check(
        _stub_vllm_config(prefix_caching=True, chunked_prefill=True),
        num_seqs=4, cfg=_serving_cfg(), strict=False)
    assert _tags(report) == _ALL_THREE, f"got {_tags(report)}"
    assert report["ok"] is False


@_XFAIL_MISSING
@pytest.mark.parametrize("env,tag", _OVERRIDES)
def test_each_unsafe_override_downgrades_exactly_one_check(
        env, tag, fake_vllm, free_memory, monkeypatch, capsys) -> None:
    """One ``OMMX_UNSAFE_ALLOW_*`` env removes ONE violation and warns about it.

    Asserted three ways so the override cannot be a silent skip:
      * the tag leaves ``violations`` and the OTHER TWO stay;
      * the env name is recorded in ``overrides_applied``;
      * a warning naming the env is actually printed to stderr.
    """
    fake_vllm("0.21.0")
    free_memory(1 * GIB)
    monkeypatch.setenv(env, "1")
    report = pf.ommx_preflight_check(
        _stub_vllm_config(prefix_caching=True, chunked_prefill=True),
        num_seqs=4, cfg=_serving_cfg(), strict=False)

    tags = _tags(report)
    assert tag not in tags, f"{env}=1 did not clear the {tag} violation (tags={tags})"
    assert tags == _ALL_THREE - {tag}, (
        f"{env}=1 changed more than its own check: expected "
        f"{_ALL_THREE - {tag}}, got {tags}")
    assert report["overrides_applied"] == [env], (
        f"overrides_applied={report['overrides_applied']}, expected exactly [{env!r}]")
    assert report["ok"] is False, "two violations remain; the report must not be ok"

    err = capsys.readouterr().err
    assert env in err, (
        f"{env}=1 suppressed the check WITHOUT warning on stderr; captured:\n{err}")
