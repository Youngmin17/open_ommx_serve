# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The flat-bitmask KV outlier encoding, wired from the serving config to the kernel call.

WHY IT EXISTS. Every stage of this encoding already existed -- the codec encodes it, the
packer emits it, ``MultiSeqKVPool`` allocates ``k_obmp`` and (deliberately) no ``k_oidx``,
and the Triton kernel has both the ``k_obmp`` argument and a popcount-rank decode for it.
The four ``ommx_paged_decode_attention_canonical(...)`` call sites in ``backend.py`` did
not forward the plane, so a bitmap-packed engine packed cleanly and then raised at the
first decode, asking for the relidx7 sidecar the pool had not allocated. That is loud
rather than silent, but it made the paper's "positions are stored as a flat bitmask"
false about the served path.

WHAT THESE GATES CAN AND CANNOT SHOW. They are CPU-only: they pin that the config
resolves to the right index source, that the estimator prices the plane the pool really
allocates, and that the two encodings describe the same positions. They do NOT show the
kernel branch is correct -- it has never run on a GPU. ``test_bitmap_outlier.py`` carries
the GPU-marked parity case that would.
"""
import pytest

torch = pytest.importorskip("torch")

from ommx_gpu_serve.attention.codec import (  # noqa: E402
    bitmap_index_bytes,
    combinadic_index_bytes,
)
from ommx_gpu_serve.attention.config import OUTLIER_REPRS  # noqa: E402
from ommx_gpu_serve.integration.vllm.config import OMMXServingConfig  # noqa: E402


def _relidx7_bytes(k: int) -> int:
    return (7 * int(k) + 7) // 8


# ── the index source the kernel is told to read ────────────────────────────────────

def test_bitmap_repr_selects_the_bitmap_index_source_by_default():
    """``bitmap_read=None`` is the ANSWER, not a default to be filled in later: a
    bitmap-packed pool holds ``k_obmp`` and no ``k_oidx``, so following ``outlier_repr``
    is the only decode that addresses a plane that exists."""
    assert OMMXServingConfig(outlier_repr="bitmap").resolved_bitmap_read() is True
    assert OMMXServingConfig(outlier_repr="relidx7").resolved_bitmap_read() is False
    assert OMMXServingConfig(outlier_repr="combinadic").resolved_bitmap_read() is False


@pytest.mark.parametrize("explicit", [True, False])
def test_an_explicit_bitmap_read_overrides_the_repr(explicit):
    cfg = OMMXServingConfig(outlier_repr="bitmap", bitmap_read=explicit)
    assert cfg.resolved_bitmap_read() is explicit


def test_bitmap_is_a_describable_tensor_spec():
    """``TensorQuantSpec.validate`` gates on this tuple. While it omitted "bitmap" the
    convenience constructor refused a recipe the serving path was otherwise willing to
    run, which is the kind of split that makes one of the two wrong."""
    assert "bitmap" in OUTLIER_REPRS


# ── the wiring itself ──────────────────────────────────────────────────────────────

def test_every_decode_call_site_forwards_the_bitmap_plane():
    """Source-level because the alternative is a GPU. The failure this pins is a call
    site that forwards ``combinadic_read``/``k_crank`` and silently drops the bitmap
    pair -- which is exactly what shipped before, at all four sites."""
    import pathlib

    import ommx_gpu_serve

    # Read the FILE, not the module: backend.py imports vllm at module scope, and this
    # whole suite is meant to run with no vllm, no triton and no GPU.
    src = (pathlib.Path(ommx_gpu_serve.__file__).parent
           / "integration" / "vllm" / "backend.py").read_text()
    n_call = src.count("ommx_paged_decode_attention_canonical(")
    assert n_call >= 4, f"expected >= 4 decode call sites, found {n_call}"
    assert src.count("k_obmp=") == n_call, (
        f"{n_call} decode call sites but {src.count('k_obmp=')} forward k_obmp; a site "
        "that omits it raises at the first decode of a bitmap-packed engine")
    assert src.count("bitmap_read=") == n_call
    # Paired with the plane it selects: a bitmap_read=True with no k_obmp is the
    # k_obmp-is-None raise, and a k_obmp with bitmap_read=False never reads it.
    assert src.count("k_crank=") == src.count("k_obmp="), (
        "the two alternative index planes must be forwarded from the same places")


def test_the_kernel_accepts_what_the_backend_now_sends():
    from ommx_gpu_serve.attention.paged_decode import (
        ommx_paged_decode_attention_canonical as launch,
    )
    import inspect

    params = inspect.signature(launch).parameters
    for name in ("bitmap_read", "k_obmp"):
        assert name in params, f"launcher has no {name} parameter"


# ── the footprint estimator ────────────────────────────────────────────────────────

@pytest.mark.parametrize("gt,k", [(32, 6), (128, 10), (64, 8), (16, 3)])
def test_preflight_prices_the_plane_the_pool_allocates(gt, k):
    """The estimator is an OOM guard, so an estimate BELOW the real allocation is the
    direction that hurts. It used to charge a combinadic rank -- the storage floor -- for
    every non-relidx7 repr, so a bitmap pool was priced under its real size."""
    from ommx_gpu_serve.integration.vllm import preflight

    common = dict(head_dim=64, n_kv_heads=2, num_seqs=2, max_seq_len=256,
                  group_tokens=gt, group_channels=32, outliers_per_vector=k,
                  sink_tokens=8, recent_window=32, page_size=16)
    got = {}
    for repr_ in ("relidx7", "bitmap", "combinadic"):
        b = preflight.ommx_pool_bytes_per_layer(outlier_repr=repr_, **common)
        got[repr_] = (b.get("k_oidx", 0), b.get("k_obmp", 0), b.get("k_crank", 0))
    # exactly one index plane is charged per repr, and it is the pool's own choice
    assert got["relidx7"][0] > 0 and got["relidx7"][1] == 0 and got["relidx7"][2] == 0
    assert got["bitmap"][1] > 0 and got["bitmap"][0] == 0 and got["bitmap"][2] == 0
    assert got["combinadic"][2] > 0 and got["combinadic"][0] == 0 and got["combinadic"][1] == 0
    # and the bitmap is charged the flat mask, not the rank
    ratio = got["bitmap"][1] / got["combinadic"][2]
    assert abs(ratio - bitmap_index_bytes(gt) / combinadic_index_bytes(k, gt)) < 1e-9


# ── the arithmetic that decides whether to use it ──────────────────────────────────

@pytest.mark.parametrize("gt,k,winner", [
    (32,   6, "bitmap"),    # shipped-kv: 4 B vs 6 B -- the flat mask is CHEAPER here
    (128, 10, "relidx7"),   # paper-kv:  16 B vs 9 B -- and dearer here
])
def test_which_encoding_is_smaller_at_the_two_published_recipes(gt, k, winner):
    """The paper says the bitmask is chosen 'at the cost of higher metadata overhead'.
    That is true only below ~12.5% coverage. At the recipe every published KV accuracy
    number was produced under (gt=32, k=6, i.e. 18.8%) the bitmask is the SMALLER plane,
    so the stated rationale is backwards for that budget."""
    r7, bm = _relidx7_bytes(k), bitmap_index_bytes(gt)
    assert (bm < r7) == (winner == "bitmap"), f"relidx7 {r7} B vs bitmap {bm} B"


@pytest.mark.parametrize("gt,crossover", [(16, 2), (32, 4), (64, 9), (128, 18)])
def test_the_coverage_crossover_is_where_it_is_claimed(gt, crossover):
    """A flat mask costs ceil(gt/8) bytes regardless of k; relidx7 costs ceil(7k/8). So
    the bitmap wins above roughly 12.5-14% coverage, which is the regime OMMX's own
    high-coverage thesis targets. Pinned so a recipe change cannot silently move it."""
    assert _relidx7_bytes(crossover) >= bitmap_index_bytes(gt)
    assert _relidx7_bytes(crossover - 1) < bitmap_index_bytes(gt)


def test_bitmap_read_is_still_refused_above_the_kernels_value_word():
    """Not a wiring limit: the outlier VALUE stream is staged as one int32, so
    ceil(k/2) <= 4. shipped-kv (k=6) fits and paper-kv (k=10) does not. If this bound is
    ever lifted, the launcher's guard is where it happens -- and this test should be the
    thing that notices."""
    import inspect

    from ommx_gpu_serve.attention import paged_decode

    src = inspect.getsource(paged_decode.ommx_paged_decode_attention_canonical)
    assert "bitmap_read" in src
    # the guard reads on the outliers-per-vector argument, not on some incidental 8
    guard = [ln for ln in src.splitlines()
             if "bitmap_read" in ln and ("> 8" in ln or "<= 8" in ln)]
    assert guard, ("no k<=8 guard found on the bitmap_read path; if it was lifted, the "
                   "value-stream width changed and paper-kv (k=10) may now be servable "
                   "-- re-check the arithmetic before deleting this test")
