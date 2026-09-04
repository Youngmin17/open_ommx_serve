#!/usr/bin/env python3
# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark of the OMMX Triton decode-attention path: parity FIRST, then latency,
then the price of the bytes it read.

WHAT THIS MEASURES. One call of ``ommx_paged_decode_attention_canonical`` per decode
step -- the launcher plus the two or three Triton kernels it issues (split-KV stage 1,
the LSE merge, and the bf16 tail or q.zp preamble when those are not fused) -- on
synthetic KV packed by the real packer (``pack.ommx_pack_kv_canonical_block``) under the
canonical recipe, at each ``--seq`` context length. Per (seq, batch, arm) cell:

  1. PARITY. ``o`` and ``lse`` are filled with NaN, the call runs once, and EVERY
     request's output is compared to attention written independently in PyTorch over the
     DEQUANTIZED planes (``attention/oracle.py``, the oracle
     ``tests/test_decode_kernel_parity.py`` gates with). Gate: cos >= 0.999 and rel-L2 <= 0.02 (scale errors pass cosine; the L2 bound does not) -- 0.999 for the worst
     request and all-finite. An arm that fails is recorded with ``parity_ok=false`` and is
     NOT timed -- a fast wrong kernel is not a result.
  2. LATENCY. The call is captured into a ``torch.cuda.CUDAGraph`` and the replay is timed
     with CUDA-event pairs: 3 warmup replays discarded, 20 measured, median and p99 (which
     at n=20 is the max; the table header says so). Replay excludes the Python launcher --
     its env reads, workspace allocations and launch calls -- which at short contexts would
     otherwise dominate the number. If capture raises, the direct call is timed instead
     and the record says ``"timing": "eager"``; the table prints which. Either way ``o`` is
     NaN-filled before the timed runs and checked against the oracle again afterwards
     (``replay_cos``), so the thing that was timed is the thing that passed. Seed 42.
  3. BYTES. The sum of EVERY plane tensor handed to the kernel (``k_base``, ``k_scale``,
     ``k_zp``, the outlier index plane, ``k_oval``, ``k_fp4_mapscale``,
     ``k_fp4_mapcenter``, ``v_main``, ``v_scale``, ``v_zp``) plus the bf16 sink/recent
     tail: the INPUT bytes of one decode step. The kernel's own intermediates (split-KV
     partials, the q.zp buffer) and any re-reads are not counted, so achieved GB/s
     (bytes / median) and % of DRAM peak (``DRAM_PEAK_GBPS``) are LOWER BOUNDS on the
     traffic it generates. Bits per packed K+V element is comparable to bf16's 16 and to
     ``packed_only.kv_bits_breakdown(...)["avg_bits_per_elem"]``.

WHAT IT DOES NOT MEASURE. This is the attention path ALONE -- not TPOT, not the per-step
KV pack/write, not the linear layers, not the vLLM scheduler. Batch > 1 tiles ONE packed
sequence B times along the page/group axis and offsets the page and group tables, so every
request reads DISTINCT memory but IDENTICAL values (each with its own q); it exercises the
kernel's batch lever, not a mixed-length batch.

ARMS. ``--arm NAME=ENV_JSON`` sets the launcher's env knobs (any prefix in
``ARM_ENV_PREFIXES``) for that arm. Every arm runs with every such knob unset except its
own, and the ambient set is restored afterwards; the ambient ``OMMX_*`` environment is
recorded in the JSON ``env`` block. The launcher reads its knobs at every call and Triton
keys the compiled kernel on the resulting constexprs, so an env change recompiles rather
than reusing a stale binary; the graph is captured under the arm's env, so the replay is
that arm's binary. With no ``--arm`` a single ``base`` arm (no knobs) runs.

Usage:
  python -m ommx_gpu_serve.bench.bench_decode_kernel --seq 4096 --seq 65536 \\
      --arm base='{}' --arm splits8='{"OMMX_ATTN_NUM_KV_SPLITS": 8}' --out results.json
Terminal marker on stdout: ``BENCH_DECODE_DONE ok=true|false``; the exit code is 1 when
any arm errored or failed parity.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple
from ommx_gpu_serve.attention.oracle import COS_MIN, REL_L2_MAX

DEFAULT_SEQS = (4096, 16384, 65536)
WARMUP = 3
MEASURE = 20
SEED = 42
# Every env prefix attention/paged_decode.py reads. An arm may set any of these; every
# arm runs with all of them unset except its own.
#: These select what gets PACKED, and the pack happens once per cell before any arm runs,
#: so an arm that sets them would be measured against data packed under the CLI values.
PACK_TIME_KNOBS = ("OMMX_ATTN_OUTLIER_REPR", "OMMX_ATTN_OUTLIERS", "OMMX_ATTN_OUTLIER_SELECT",
                   "OMMX_ATTN_K_FORMAT", "OMMX_ATTN_POW2", "OMMX_ATTN_OUTLIER_PCT")
ARM_ENV_PREFIXES = ("OMMX_ATTN_", "OMMX_V4_", "OMMX_MERGE_", "OMMX_OVERSPLIT_",
                    "OMMX_ABL_", "OMMX_VLLM_KV_", "OMMX_INT2_")
DONE_MARKER = "BENCH_DECODE_DONE"
P99_LABEL = "p99(ms)" if math.ceil(0.99 * MEASURE) < MEASURE else "max(ms)"

# Every plane the kernel signature accepts. At the canonical recipe (gt=32, k=6, bitmap,
# int8 scale) the two FP4 map planes are 1.0 of the 8.25 bit/elem, so a count that omits
# them is not rounding noise. EXACTLY ONE of k_oidx / k_crank / k_obmp is non-None for a
# given outlier_repr.
KERNEL_PLANES = (
    "k_base", "k_scale", "k_zp",
    "k_oidx", "k_crank", "k_obmp", "k_oval",
    "k_fp4_mapscale", "k_fp4_mapcenter",
    "v_main", "v_scale", "v_zp",
)

# Substring match against torch.cuda.get_device_name(), first hit wins. Anything not
# listed prints "peak unknown" rather than a guessed number.
DRAM_PEAK_GBPS: Tuple[Tuple[Tuple[str, ...], float], ...] = (
    (("H200",), 4800.0),
    (("H100", "HBM3"), 3350.0),
    (("A100", "80GB"), 2039.0),
)


def dram_peak_gbps(device_name: str) -> Optional[float]:
    for needles, peak in DRAM_PEAK_GBPS:
        if all(n in device_name for n in needles):
            return peak
    return None


def parse_arm(spec: str) -> Tuple[str, Dict[str, str]]:
    """``NAME=ENV_JSON`` -> (name, {knob: str}). Keys outside ``ARM_ENV_PREFIXES`` are
    refused: the launcher reads only those, and a typo would silently measure the base
    arm."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--arm expects NAME=ENV_JSON; got {spec!r}")
    name, _, raw = spec.partition("=")
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"--arm needs a non-empty NAME; got {spec!r}")
    try:
        env = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--arm {name}: ENV_JSON is not JSON ({exc})")
    if not isinstance(env, dict):
        raise argparse.ArgumentTypeError(f"--arm {name}: ENV_JSON must be an object")
    out: Dict[str, str] = {}
    for k, v in env.items():
        if str(k) in PACK_TIME_KNOBS:
            raise argparse.ArgumentTypeError(
                f"--arm {name}: {k} selects the PACK, which happens once per cell from the "
                f"CLI flags; pass it as --repr/--outliers/... instead of an arm knob")
        if not str(k).startswith(ARM_ENV_PREFIXES):
            raise argparse.ArgumentTypeError(
                f"--arm {name}: {k!r} is not a launcher knob (prefixes: "
                f"{', '.join(ARM_ENV_PREFIXES)})")
        out[str(k)] = str(v)
    return name, out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seq", type=int, action="append",
                    help=f"context length, repeatable (default {list(DEFAULT_SEQS)})")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--repr", choices=("bitmap", "relidx7", "combinadic"), default="bitmap")
    ap.add_argument("--outliers", type=int, default=6)
    ap.add_argument("--arm", type=parse_arm, action="append",
                    help="NAME=ENV_JSON, repeatable; keys must be launcher env knobs")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--q-heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--sink", type=int, default=8)
    ap.add_argument("--recent", type=int, default=32)
    ap.add_argument("--group-tokens", type=int, default=32)
    ap.add_argument("--group-channels", type=int, default=32)
    ap.add_argument("--page", type=int, default=32)
    ap.add_argument("--k-format", default="i2f4")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.seq:
        args.seq = list(DEFAULT_SEQS)
    if not args.arm:
        args.arm = [("base", {})]
    if args.batch < 1:
        ap.error("--batch must be >= 1")
    if args.q_heads % args.kv_heads:
        ap.error("--q-heads must be a multiple of --kv-heads")
    if args.repr == "combinadic" and args.outliers > 4:
        ap.error("--repr combinadic: the kernel's in-register unrank needs --outliers <= 4")
    return args


def pack_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """The packer keywords the recipe flags select (also recorded in the JSON env)."""
    return dict(outliers_per_vector=args.outliers, group_tokens=args.group_tokens,
                group_channels=args.group_channels, page_size=args.page,
                k_format=args.k_format, outlier_repr=args.repr, use_pow2=True,
                outlier_select="signed")


def plane_bytes(planes: Dict[str, Any]) -> Dict[str, int]:
    """Bytes of every KERNEL_PLANES tensor present (None planes contribute nothing)."""
    import torch
    out: Dict[str, int] = {}
    for name in KERNEL_PLANES:
        t = planes.get(name)
        if torch.is_tensor(t):
            out[name] = int(t.numel() * t.element_size())
    return out


def bits_per_elem(packed_bytes: int, packed_tokens: int, kv_heads: int, head_dim: int) -> float:
    """Bits per packed K+V element (K and V counted separately, i.e. per tensor)."""
    return packed_bytes * 8.0 / (2 * packed_tokens * kv_heads * head_dim)


def is_knob(name: str) -> bool:
    return name.startswith(ARM_ENV_PREFIXES)


@contextlib.contextmanager
def arm_env(env: Dict[str, str]) -> Iterator[None]:
    """Every launcher knob scrubbed, the arm's set, the ambient set restored on exit."""
    saved = {k: v for k, v in os.environ.items() if is_knob(k)}
    for k in saved:
        del os.environ[k]
    os.environ.update(env)
    try:
        yield
    finally:
        for k in [k for k in os.environ if is_knob(k)]:
            del os.environ[k]
        os.environ.update(saved)


def graph_or_eager(launch: Callable[[], None]) -> Tuple[Callable[[], None], str]:
    """Capture ``launch`` into a CUDA graph and return its replay; the direct call if
    capture raises."""
    import torch
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            launch()
        return g.replay, "cuda_graph"
    except Exception as exc:
        print(f"  graph capture failed ({type(exc).__name__}: {str(exc)[:120]}); "
              "timing the eager call", flush=True)
        return launch, "eager"


def time_launches(fn: Callable[[], None], warmup: int = WARMUP, measure: int = MEASURE) -> List[float]:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ms: List[float] = []
    for _ in range(measure):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ms.append(a.elapsed_time(b))
    return ms


def median_mean_p99(ms: Sequence[float]) -> Tuple[float, float, float]:
    s = sorted(ms)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    p99 = s[min(n - 1, max(0, math.ceil(0.99 * n) - 1))]
    mean = float(sum(ms) / max(1, len(ms)))
    return med, mean, p99


def build_cell(args: argparse.Namespace, seq: int) -> Dict[str, Any]:
    """Pack one sequence on CPU and derive everything the kernel and the oracle need."""
    import torch
    from ommx_gpu_serve.attention.kv_window import WindowSpec
    from ommx_gpu_serve.attention.oracle import synth_kv
    from ommx_gpu_serve.attention.pack import (
        dequant_kv_canonical,
        ommx_pack_kv_canonical_block,
    )
    window = WindowSpec(sink_tokens=args.sink, recent_window=args.recent,
                        group_tokens=args.group_tokens, page_size=args.page)
    boundary = window.boundary(seq)
    packed = max(0, boundary - args.sink)
    if packed <= 0:
        raise ValueError(f"seq={seq} packs nothing under sink={args.sink} recent={args.recent}")
    tail = window.tail_indices(seq)
    K, V = synth_kv(seq, args.kv_heads, args.head_dim, SEED)
    t0 = time.time()
    planes = ommx_pack_kv_canonical_block(K[args.sink:boundary], V[args.sink:boundary],
                                          **pack_kwargs(args))
    pack_s = time.time() - t0
    K_dq, V_dq = dequant_kv_canonical(planes)
    k_tail = torch.stack([K[i] for i in tail])
    v_tail = torch.stack([V[i] for i in tail])
    K_full = torch.cat([torch.as_tensor(K_dq).float(), k_tail.float()])
    V_full = torch.cat([torch.as_tensor(V_dq).float(), v_tail.float()])
    return dict(seq=seq, packed=packed, tail=len(tail), n_groups=int(planes["n_groups"]),
                planes=planes, k_tail=k_tail, v_tail=v_tail, K_full=K_full, V_full=V_full,
                pack_s=pack_s)


def run_cell(args: argparse.Namespace, cell: Dict[str, Any], device: str) -> List[Dict[str, Any]]:
    import torch
    from ommx_gpu_serve.attention.oracle import compare, reference_output
    from ommx_gpu_serve.attention.paged_decode import ommx_paged_decode_attention_canonical

    B, H_Q, H_KV, D = args.batch, args.q_heads, args.kv_heads, args.head_dim
    planes, packed, G = cell["planes"], cell["packed"], cell["n_groups"]
    P = int(planes["k_base"].shape[0])

    def tile(t):
        return torch.cat([t] * B, 0).to(device) if torch.is_tensor(t) else None

    dev_planes = {n: tile(planes.get(n)) for n in KERNEL_PLANES}
    per_plane = plane_bytes(planes)
    packed_bytes = sum(per_plane.values())
    tail_bytes = sum(int(t.numel() * t.element_size())
                     for t in (cell["k_tail"], cell["v_tail"]))
    bytes_read = B * (packed_bytes + tail_bytes)
    req_to_token = torch.stack(
        [b * P + torch.arange(P, dtype=torch.int32) for b in range(B)]).to(device)
    req_to_group = torch.stack(
        [b * G + torch.arange(G, dtype=torch.int32) for b in range(B)]).to(device)
    k_tail = cell["k_tail"].to(device).unsqueeze(0).repeat(B, 1, 1, 1)
    v_tail = cell["v_tail"].to(device).unsqueeze(0).repeat(B, 1, 1, 1)
    torch.manual_seed(SEED)
    q = torch.randn(B, H_Q, D, dtype=torch.bfloat16, device=device)
    o = torch.empty(B, H_Q, D, dtype=torch.float32, device=device)
    lse = torch.empty(B, H_Q, dtype=torch.float32, device=device)
    b_tail_len = torch.full((B,), cell["tail"], dtype=torch.int32, device=device)
    b_seq_len = torch.full((B,), packed, dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(D)

    def launch():
        ommx_paged_decode_attention_canonical(
            q, dev_planes["k_base"], dev_planes["k_scale"], dev_planes["k_zp"],
            dev_planes["k_oidx"], dev_planes["k_oval"],
            dev_planes["v_main"], dev_planes["v_scale"], dev_planes["v_zp"],
            k_tail, v_tail, b_tail_len, o, lse, req_to_token, req_to_group, b_seq_len,
            sm_scale=sm_scale, page_size=args.page,
            k_outliers_per_vector=int(planes["outliers_per_vector"]),
            k_format=str(planes["k_format"]),
            combinadic_read=planes.get("k_crank") is not None,
            k_crank=dev_planes["k_crank"],
            bitmap_read=planes.get("k_obmp") is not None,
            k_obmp=dev_planes["k_obmp"],
            k_fp4_mapscale=dev_planes["k_fp4_mapscale"],
            k_fp4_mapcenter=dev_planes["k_fp4_mapcenter"],
            kv_outlier_map=bool(planes.get("kv_outlier_map", False)),
            kv_int8_scale=bool(planes.get("kv_int8_scale", False)),
            max_seq_len=packed, max_tail_len=max(1, cell["tail"]))

    want = torch.stack([reference_output(q[b].cpu(), cell["K_full"], cell["V_full"],
                                         H_Q, H_KV, sm_scale) for b in range(B)])
    bpe = bits_per_elem(packed_bytes, packed, H_KV, D)
    peak = dram_peak_gbps(torch.cuda.get_device_name(device))

    def parity() -> Dict[str, Any]:
        """Every request against its oracle; ``cos`` is the worst one."""
        got = o.float().cpu()
        finite = bool(torch.isfinite(got).all())
        if not finite:
            return dict(cos=None, finite=False, max_abs=None, rel_l2=None, parity_ok=False)
        per = [compare(got[i], want[i]) for i in range(got.shape[0])]   # worst request
        cos = min(c for c, _, _ in per); max_abs = max(m for _, m, _ in per)
        rel_l2 = max(r for _, _, r in per)
        # cosine alone passes a wrongly scaled output; the relative L2 bound closes that.
        return dict(cos=cos, finite=True, max_abs=max_abs, rel_l2=rel_l2,
                    parity_ok=bool(cos >= COS_MIN and rel_l2 <= REL_L2_MAX))

    def measure() -> Dict[str, Any]:
        nan = float("nan")
        o.fill_(nan)
        lse.fill_(nan)
        launch()
        torch.cuda.synchronize()
        out = parity()
        if not out["parity_ok"]:
            return out
        fn, mode = graph_or_eager(launch)
        o.fill_(nan)
        ms = time_launches(fn)
        replay = parity()
        out.update(timing=mode, replay_cos=replay["cos"])
        if not replay["parity_ok"]:
            out.update(parity_ok=False, note=f"{mode} replay failed parity")
            return out
        med, mean, p99 = median_mean_p99(ms)
        gbps = bytes_read / (med * 1e-3) / 1e9
        out.update(med_ms=med, mean_ms=mean, p99_ms=p99, samples_ms=ms, gbps=gbps,
                   pct_peak=(100.0 * gbps / peak) if peak else None)
        return out

    records: List[Dict[str, Any]] = []
    for name, env in args.arm:
        rec: Dict[str, Any] = dict(
            arm=name, env=env, seq=cell["seq"], batch=B, repr=args.repr,
            packed_tokens=packed, tail_tokens=cell["tail"], n_groups=G,
            packed_bytes_per_request=packed_bytes, tail_bytes_per_request=tail_bytes,
            bytes_read=bytes_read, plane_bytes=per_plane, bits_per_elem=bpe,
            parity_ok=False)
        try:
            with arm_env(env):
                rec.update(measure())
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        records.append(rec)
    return records


def all_valid(results: Sequence[Dict[str, Any]]) -> bool:
    """True only when every arm passed parity and none raised."""
    return all(r.get("parity_ok") and "error" not in r for r in results)


def git_sha() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
    except Exception:
        return None


def env_block(args: argparse.Namespace, device: str) -> Dict[str, Any]:
    import torch
    try:
        import triton
        triton_v = triton.__version__
    except Exception:
        triton_v = None
    recipe = dict(pack_kwargs(args), q_heads=args.q_heads, kv_heads=args.kv_heads,
                  head_dim=args.head_dim, sink=args.sink, recent=args.recent)
    return {
        "torch": torch.__version__, "triton": triton_v,
        "device": torch.cuda.get_device_name(device),
        "git_sha": git_sha(), "recipe": recipe,
        "protocol": {"warmup": WARMUP, "measure": MEASURE, "seed": SEED,
                     "parity_cos": COS_MIN, "parity_rel_l2": REL_L2_MAX, "timer": "cuda_event"},
        "ommx_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("OMMX_")},
        "argv": sys.argv[1:],
    }


def fmt_header() -> str:
    return (f"{'arm':<12} {'seq':>7} {'B':>3} {'cos':>9} {'par':>5} {'timing':>10} "
            f"{'med(ms)':>9} {P99_LABEL:>9} {'MB read':>9} {'b/elem':>6} {'GB/s':>8} "
            f"{'%peak':>7}")


def fmt_row(r: Dict[str, Any], peak: Optional[float]) -> str:
    if "finite" not in r:
        cos = "n/a"
    elif r["cos"] is None:
        cos = "nan"
    else:
        cos = f"{r['cos']:.6f}"
    head = f"{r['arm']:<12} {r['seq']:>7} {r['batch']:>3} {cos:>9}"
    if r.get("error"):
        return f"{head}  ERROR {r['error']}"
    ok = r["parity_ok"]
    timing = r.get("timing", "-")
    mb = r["bytes_read"] / 1e6
    if ok:
        lat = f"{r['med_ms']:>9.4f} {r['p99_ms']:>9.4f}"
        gb = f"{r['gbps']:>8.1f}"
        pk = f"{r['pct_peak']:>6.1f}%" if peak else "unknown"
    else:
        lat = f"{'invalid':>9} {'invalid':>9}"
        gb, pk = f"{'invalid':>8}", f"{'invalid':>7}"
    row = (f"{head} {'ok' if ok else 'FAIL':>5} {timing:>10} {lat} {mb:>9.2f} "
           f"{r['bits_per_elem']:>6.3f} {gb} {pk}")
    return f"{row}  {r['note']}" if r.get("note") else row


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        print("no CUDA device; the decode kernel is Triton", file=sys.stderr)
        return 2
    device = args.device
    name = torch.cuda.get_device_name(device)
    peak = dram_peak_gbps(name)
    print(f"device={name} dram_peak={'%.0f GB/s' % peak if peak else 'unknown'} "
          f"repr={args.repr} batch={args.batch} outliers={args.outliers}", flush=True)
    results: List[Dict[str, Any]] = []
    for seq in args.seq:
        cell = build_cell(args, seq)
        print(f"seq={seq}: packed={cell['packed']} tail={cell['tail']} "
              f"groups={cell['n_groups']} pack={cell['pack_s']:.1f}s", flush=True)
        results.extend(run_cell(args, cell, device))
    print()
    print(fmt_header())
    for r in results:
        print(fmt_row(r, peak))
    if peak is None:
        print("peak unknown: no DRAM entry for this device; %peak not reported")
    out = {"env": env_block(args, device), "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")
    ok = all_valid(results)
    print(f"{DONE_MARKER} ok={'true' if ok else 'false'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
