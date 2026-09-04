# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""The decode kernel against an independent attention reference.

WHY IT EXISTS. Until now the KV axis had no numeric gate on the kernel's ARITHMETIC. What it
had was:

  * ``test_kv_pool_parity.py`` -- the packed PLANES are byte-identical to a single-sequence
    store. That gates the packer and the slot arithmetic. A kernel that reads correct planes
    and computes the wrong attention passes it untouched.
  * ``test_bitmap_outlier.py`` -- the bitmap decode path agrees with the relidx7 one. That
    gates the two index encodings against EACH OTHER; if both are wrong in the same way, both
    agree and both pass.
  * ``attention/reference_op.py`` -- named "reference", but it CALLS the Triton kernel
    (``_canonical``). It is a torch.library wrapper for graph capture, not an oracle.

So the one question nobody asked was whether the softmax, the sink/packed/tail window split,
the split-KV merge and the outlier splice compose into the right number. This file asks it,
against attention written independently in plain PyTorch over the DEQUANTIZED planes.

WHAT THE ORACLE IS AND IS NOT. The reference dequantizes with ``pack.dequant_kv_canonical``
and then does textbook attention (``attention/oracle.py``, shared with
``bench/bench_decode_kernel.py`` so the bench gates on exactly this). It therefore shares the
DEQUANT arithmetic with the kernel (that half is gated by the pool-parity and bitmap tests)
and shares nothing else: the attention itself, the windowing, and the numerics of the merge
are independent. A kernel that mis-splits the window, drops the sink, mis-normalizes softmax,
or merges splits wrongly fails here and passes everything else in the suite.

REQUIRES A GPU. The kernel is Triton. Skipped, not xfailed, when CUDA is absent -- an xfail
would read as "known broken" on a CPU box, which is the wrong signal for a gate whose subject
simply is not present.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the decode kernel is Triton; needs a CUDA device")

from ommx_gpu_serve.attention.kv_window import WindowSpec  # noqa: E402
from ommx_gpu_serve.attention.oracle import COS_MIN, REL_L2_MAX, compare, reference_output, synth_kv  # noqa: E402
from ommx_gpu_serve.attention.pack import (  # noqa: E402
    dequant_kv_canonical,
    ommx_pack_kv_canonical_block,
)
from ommx_gpu_serve.attention.paged_decode import (  # noqa: E402
    ommx_paged_decode_attention_canonical,
)

H_KV, H_Q, D = 2, 4, 64
GC, SINK, RECENT = 32, 8, 32
PAGE = 16


def _run_kernel(planes, q, seq_len, window, device):
    """Drive the kernel exactly as the serving path does, on ONE request."""
    G = int(planes["n_groups"])
    boundary = window.boundary(seq_len)
    packed = max(0, boundary - window.sink_tokens)
    tail = window.tail_indices(seq_len)

    def dev(x):
        return x.to(device) if torch.is_tensor(x) else x

    P = (packed + PAGE - 1) // PAGE
    req_to_token = torch.arange(max(1, P * PAGE), device=device, dtype=torch.long).unsqueeze(0)
    req_to_group = torch.arange(max(1, G), device=device, dtype=torch.long).unsqueeze(0)
    k_tail = torch.stack([planes["_K_src"][i] for i in tail]).to(device) if tail else \
        torch.zeros(1, H_KV, D, dtype=torch.bfloat16, device=device)
    v_tail = torch.stack([planes["_V_src"][i] for i in tail]).to(device) if tail else \
        torch.zeros(1, H_KV, D, dtype=torch.bfloat16, device=device)
    o = torch.empty(1, H_Q, D, dtype=torch.float32, device=device)
    lse = torch.empty(1, H_Q, dtype=torch.float32, device=device)
    ommx_paged_decode_attention_canonical(
        q.unsqueeze(0).to(device),
        dev(planes["k_base"]), dev(planes["k_scale"]), dev(planes["k_zp"]),
        dev(planes.get("k_oidx")), dev(planes.get("k_oval")),
        dev(planes["v_main"]), dev(planes["v_scale"]), dev(planes["v_zp"]),
        k_tail.unsqueeze(0), v_tail.unsqueeze(0),
        torch.tensor([len(tail)], dtype=torch.int32, device=device),
        o, lse, req_to_token, req_to_group,
        torch.tensor([packed], dtype=torch.int32, device=device),
        sm_scale=1.0 / math.sqrt(D), page_size=PAGE,
        k_outliers_per_vector=int(planes["outliers_per_vector"]),
        k_format=str(planes["k_format"]),
        bitmap_read=planes.get("k_obmp") is not None,
        k_obmp=dev(planes.get("k_obmp")),
        # The pack emits a dedicated FP4 outlier map and int8 scales BY DEFAULT
        # (`kv_outlier_map=True`, `k_fp4_mapscale`/`k_fp4_mapcenter`, `kv_int8_scale=True`).
        # The kernel decodes outliers with the wrong scale unless told, and the first version
        # of this harness passed none of them -- which read as a kernel defect at >=3 groups
        # while k=0 stayed exact. Mirror what the serving paths pass.
        k_fp4_mapscale=dev(planes.get("k_fp4_mapscale")),
        k_fp4_mapcenter=dev(planes.get("k_fp4_mapcenter")),
        kv_outlier_map=bool(planes.get("kv_outlier_map", False)),
        kv_int8_scale=bool(planes.get("kv_int8_scale", False)),
        max_seq_len=max(1, packed), max_tail_len=max(1, len(tail)))
    return o[0].float().cpu()


@pytest.mark.parametrize("GT", [32, 64])
@pytest.mark.parametrize("repr_", ["relidx7", "bitmap"])
@pytest.mark.parametrize("seq_len", [128, 136, 200, 232])
def test_kernel_matches_an_independent_attention_reference(repr_, seq_len, GT):
    """The gate. Both index encodings, four lengths (2/3/5/6 packed groups), against attention
    the kernel did not write.

    GT is not decoration either. The bitmap lane has TWO code paths: VL<=32 fits ONE
    occupancy word and collapses the rank to a single masked prefix-popcount, while
    VL>32 needs ceil(VL/32) words and runs the multi-word loop (full popcount of every
    lower word + masked prefix in the token's own word). Only GT=32 was ever run on a
    GPU, so the multi-word lane -- reachable from any group_tokens=64/128 recipe -- was
    gated on CPU contract tests alone. GT=64 exercises it.

    The lengths are not decoration. The first version of this harness omitted the FP4 outlier
    map and the int8 scale flags, and the result was exact at 2 groups (cos 0.999987) and
    collapsed from 3 onward (0.32 / 0.65 / 0.60) -- with k=0 staying exact at every length, so
    the window, paging, softmax and split merge were all cleared and the finger pointed
    squarely at the kernel outlier splice. It was the harness. Keeping >=3 groups in the sweep
    keeps that discriminating power.
    """
    device = "cuda"
    window = WindowSpec(sink_tokens=SINK, recent_window=RECENT, group_tokens=GT, page_size=PAGE)
    K, V = synth_kv(seq_len, H_KV, D, seed=11)
    boundary = window.boundary(seq_len)
    packed = max(0, boundary - SINK)
    if packed <= 0:
        pytest.skip(f"seq_len={seq_len} packs nothing under this window")

    planes = ommx_pack_kv_canonical_block(
        K[SINK:boundary], V[SINK:boundary],
        outliers_per_vector=6, group_tokens=GT, group_channels=GC,
        page_size=PAGE, k_format="i2f4", outlier_repr=repr_, use_pow2=True,
        outlier_select="signed")
    planes["_K_src"], planes["_V_src"] = K, V

    # the oracle's KV: sink and tail in bf16 exactly as the kernel reads them, the packed
    # middle dequantized from the REAL planes.
    K_dq, V_dq = dequant_kv_canonical(planes)
    tail = window.tail_indices(seq_len)
    K_full = torch.cat([K[SINK:boundary].float().new_tensor(K_dq),
                        torch.stack([K[i] for i in tail]).float()])
    V_full = torch.cat([V[SINK:boundary].float().new_tensor(V_dq),
                        torch.stack([V[i] for i in tail]).float()])

    g = torch.Generator().manual_seed(7)
    q = torch.randn(H_Q, D, generator=g).to(torch.bfloat16)
    got = _run_kernel(planes, q, seq_len, window, device)
    want = reference_output(q, K_full, V_full, H_Q, H_KV, 1.0 / math.sqrt(D))

    assert torch.isfinite(got).all(), "kernel produced non-finite output"
    cos, max_abs, rel_l2 = compare(got, want)
    # cosine alone would pass a wrongly SCALED output (2x the reference is cos 1.0)
    assert rel_l2 <= REL_L2_MAX, (
        f"repr={repr_} GT={GT} seq_len={seq_len}: rel_l2={rel_l2:.4f} against the reference "
        f"(cos={cos:.6f}, max_abs={max_abs:.3e}) -- a scale or bias error, not a direction one")
    assert cos >= COS_MIN, (
        f"repr={repr_} GT={GT} seq_len={seq_len}: cos={cos:.6f} against an independent reference "
        f"(max_abs={max_abs:.3e}). Plane parity and bitmap/relidx7 agreement both pass "
        f"regardless of this, which is why this gate exists")


def test_the_named_reference_is_not_an_oracle():
    """Recorded so the gap cannot be re-closed by pointing at the wrong file.

    ``attention/reference_op.py`` is a torch.library wrapper: its decode op CALLS the Triton
    kernel. Comparing the kernel to it would compare the kernel to itself.
    """
    import pathlib

    import ommx_gpu_serve

    src = (pathlib.Path(ommx_gpu_serve.__file__).parent
           / "attention" / "reference_op.py").read_text()
    assert "ommx_paged_decode_attention_canonical as _canonical" in src, (
        "reference_op.py no longer wraps the kernel -- if it grew an independent "
        "implementation it may now BE an oracle, and this file should use it")
