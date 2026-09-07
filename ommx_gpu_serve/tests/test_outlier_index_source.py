# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Naming an outlier encoding must be enough to decode it, on every path.

WHY IT EXISTS. The store allocates exactly ONE index plane per encoding: ``k_oidx`` for
relidx7, ``k_obmp`` for bitmap, ``k_crank`` for combinadic (``kv_pool.py``). The kernel
picks which one to read from two booleans. So if a boolean does not follow ``outlier_repr``,
selecting that encoding produces a store the kernel cannot read, and the first quantized
group raises ``k_outliers_per_vector > 0 requires the k_oidx relidx7 sidecar plane``.

That is not hypothetical. It shipped twice:

  * ``bitmap`` on the HF-eager path -- the call sites forwarded ``combinadic_read`` and not
    ``bitmap_read``, so ``OMMX_ATTN_OUTLIER_REPR=bitmap`` raised there while working under
    vLLM. Caught by a benchmark, not by a test.
  * ``combinadic`` on BOTH paths -- ``bitmap_read`` was tri-state with a
    ``resolved_bitmap_read()`` that follows the repr, while ``combinadic_read`` was a plain
    ``False``. Selecting combinadic therefore required a SECOND env var
    (``OMMX_ATTN_COMBINADIC_READ=1``) that nothing told the operator about.

The property below is the one that was missing both times, stated once for all three
encodings so a fourth cannot be added with the same hole.
"""
import pytest

from ommx_gpu_serve.integration.vllm.config import resolve_serving_config

REPRS = ("relidx7", "bitmap", "combinadic")


def _cfg(monkeypatch, repr_, **env):
    for k in ("OMMX_ATTN_OUTLIER_REPR", "OMMX_ATTN_BITMAP_READ",
              "OMMX_ATTN_COMBINADIC_READ"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OMMX_ATTN_OUTLIER_REPR", repr_)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return resolve_serving_config()


# ── the property ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("repr_", REPRS)
def test_naming_an_encoding_selects_exactly_that_index_source(monkeypatch, repr_):
    """Exactly one of the two read flags is true, and it is the one matching the repr.

    relidx7 is the both-false case: it is the only encoding whose plane the kernel reads
    without being told, because ``k_oidx`` is what it looks for by default.
    """
    cfg = _cfg(monkeypatch, repr_)
    got = (cfg.resolved_bitmap_read(), cfg.resolved_combinadic_read())
    want = {"relidx7": (False, False), "bitmap": (True, False),
            "combinadic": (False, True)}[repr_]
    assert got == want, (
        f"outlier_repr={repr_!r} resolves to bitmap_read={got[0]} combinadic_read={got[1]}; "
        f"the store allocates only that encoding's plane, so the kernel would read one that "
        f"does not exist")


@pytest.mark.parametrize("repr_", REPRS)
def test_no_second_env_var_is_needed(monkeypatch, repr_):
    """The regression, stated directly: naming the encoding is sufficient.

    Both historical bugs presented as "it works if you ALSO set the read flag", which is why
    neither was caught -- every internal caller happened to set it.
    """
    bare = _cfg(monkeypatch, repr_)
    flag = {"bitmap": "OMMX_ATTN_BITMAP_READ",
            "combinadic": "OMMX_ATTN_COMBINADIC_READ"}.get(repr_)
    if flag is None:
        return
    explicit = _cfg(monkeypatch, repr_, **{flag: "1"})
    assert (bare.resolved_bitmap_read(), bare.resolved_combinadic_read()) == \
           (explicit.resolved_bitmap_read(), explicit.resolved_combinadic_read()), (
        f"{repr_} needs {flag}=1 to decode; naming the encoding must be enough")


# ── the refusals that keep the explicit flags honest ────────────────────────────

@pytest.mark.parametrize("flag,repr_", [
    ("OMMX_ATTN_BITMAP_READ", "relidx7"),
    ("OMMX_ATTN_BITMAP_READ", "combinadic"),
    ("OMMX_ATTN_COMBINADIC_READ", "relidx7"),
    ("OMMX_ATTN_COMBINADIC_READ", "bitmap"),
])
def test_an_explicit_flag_against_a_mismatched_repr_is_refused(monkeypatch, flag, repr_):
    """An explicit flag may not select a plane the store did not allocate. The combinadic
    half of this matrix had no check at all before."""
    with pytest.raises(ValueError, match=r"OMMX_ATTN_(BITMAP|COMBINADIC)_READ"):
        _cfg(monkeypatch, repr_, **{flag: "1"})


def test_the_two_sources_are_still_mutually_exclusive(monkeypatch):
    """A pack carries exactly one of k_obmp / k_crank, so both flags on is refused --
    checked on the RESOLVED values, since testing the raw fields would miss the case where
    each is None and resolves to True off a different repr."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _cfg(monkeypatch, "bitmap", OMMX_ATTN_COMBINADIC_READ="1",
             OMMX_ATTN_BITMAP_READ="1")


@pytest.mark.parametrize("repr_", ["bitmap", "combinadic"])
def test_an_explicit_false_still_wins(monkeypatch, repr_):
    """The A/B escape hatch survives: an operator can still force the flag off. It then
    produces an unreadable pack, which is the operator's business and is what makes this an
    A/B knob rather than a second required setting."""
    flag = {"bitmap": "OMMX_ATTN_BITMAP_READ",
            "combinadic": "OMMX_ATTN_COMBINADIC_READ"}[repr_]
    cfg = _cfg(monkeypatch, repr_, **{flag: "0"})
    got = cfg.resolved_bitmap_read() if repr_ == "bitmap" else cfg.resolved_combinadic_read()
    assert got is False


@pytest.mark.parametrize("repr_", ["bitmap", "combinadic"])
def test_an_empty_flag_is_absent_not_false(monkeypatch, repr_):
    """``export OMMX_ATTN_COMBINADIC_READ=`` (empty) must leave the tri-state at None so the
    read follows the repr. The combinadic read used a raw ``is not None`` where the bitmap
    read used ``_env_present``, so an empty export resolved it to False and a combinadic
    store failed at the first decode with nothing reading k_crank."""
    flag, field = {"bitmap": ("OMMX_ATTN_BITMAP_READ", "bitmap_read"),
                   "combinadic": ("OMMX_ATTN_COMBINADIC_READ", "combinadic_read")}[repr_]
    cfg = _cfg(monkeypatch, repr_, **{flag: ""})
    assert getattr(cfg, field) is None, f"{flag}='' resolved to {getattr(cfg, field)!r}"
    got = cfg.resolved_bitmap_read() if repr_ == "bitmap" else cfg.resolved_combinadic_read()
    assert got is True


# ── the kernel's bitmap value-stream gate, at resolve time ──────────────────────

@pytest.mark.parametrize("repr_,k,ok", [
    ("bitmap", 8, True),      # the edge the kernel accepts: ceil(8/2) = 4 bytes = one int32
    ("bitmap", 10, False),    # paper-kv's count under the encoding that cannot hold it
    ("relidx7", 10, True),    # the encoding paper-kv actually names
])
def test_a_bitmap_recipe_above_the_kernels_k_gate_is_refused_at_resolve_time(
        monkeypatch, repr_, k, ok):
    """The bitmap decode stages a group's outlier values in one int32, so the kernel refuses
    ``bitmap_read`` for k > 8 -- on the first quantized group of the first decode step, long
    after ``vllm serve`` reported itself up. The same refusal now happens when the config
    resolves, i.e. at engine build, and the message names the two ways out."""
    monkeypatch.delenv("OMMX_OUTLIER_PERCENT", raising=False)
    if ok:
        cfg = _cfg(monkeypatch, repr_, OMMX_ATTN_OUTLIERS=str(k))
        assert cfg.outliers_per_vector == k
    else:
        with pytest.raises(ValueError, match=r"k <= 8"):
            _cfg(monkeypatch, repr_, OMMX_ATTN_OUTLIERS=str(k))


def test_the_k_gate_follows_the_resolved_read_not_the_repr_alone(monkeypatch):
    """``OMMX_ATTN_BITMAP_READ=0`` against a bitmap store is the A/B escape hatch
    (``test_an_explicit_false_still_wins``). With the read forced off no bitmap value stream
    is ever staged, so k=10 resolves: the gate is on the RESOLVED read, like the exclusivity
    check above it."""
    monkeypatch.delenv("OMMX_OUTLIER_PERCENT", raising=False)
    cfg = _cfg(monkeypatch, "bitmap", OMMX_ATTN_OUTLIERS="10", OMMX_ATTN_BITMAP_READ="0")
    assert cfg.resolved_bitmap_read() is False
    assert cfg.outliers_per_vector == 10


# ── the call sites, pinned at source level ─────────────────────────────────────

def test_every_decode_call_site_uses_the_resolved_accessors():
    """Source-level because these files import vllm/torch at module scope. The bug was
    never in the accessor -- it was a call site reading the RAW field, which silently means
    "False" instead of "follow the repr"."""
    import pathlib

    import ommx_gpu_serve

    root = pathlib.Path(ommx_gpu_serve.__file__).parent
    for rel in ("hf_eager/_ommx_llama_modeling.py",
                "hf_eager/_ommx_llama_modeling_batch.py",
                "integration/vllm/backend.py"):
        src = (root / rel).read_text()
        for raw, fixed in (("combinadic_read=cfg.combinadic_read",
                            "cfg.resolved_combinadic_read()"),
                           ("bitmap_read=cfg.bitmap_read", "cfg.resolved_bitmap_read()")):
            assert raw not in src, (
                f"{rel} passes the raw field ({raw}); use {fixed} or the encoding silently "
                f"resolves to False")
        assert ".combinadic_read," not in src.replace("resolved_combinadic_read(),", ""), (
            f"{rel} still reads combinadic_read directly at a call site")
