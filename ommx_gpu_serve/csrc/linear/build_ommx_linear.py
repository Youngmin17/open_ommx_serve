# SPDX-License-Identifier: Apache-2.0
"""Build the CONSOLIDATED OMMX linear kernel as ONE torch extension.

Replaces the previous per-kernel build (build_serve.py built 8+ separate .so
modules: ommx_linear_cute_base_i2f4, ..._decode_gemv_i2f4, ..._sparse_correct,
...). This builds the single ommx_linear.cu -> ONE module `ommx_linear`.

Arch (CLAUDE.md / build_serve.py flags, verbatim):
  * A100 sm_80  : -arch=sm_80, NO CUTLASS (decode ALU + prefill mma + sparse).
  * H100/H200 sm_90a : -arch=sm_90a; define OMMX_ENABLE_SM90_CUTLASS to compile
    the wgmma+TMA CUTLASS prefill (needs a CUTLASS 3.5+ tree with
    tools/util/include/cutlass/util/mixed_dtype_utils.hpp; set OMMX_CUTLASS_DIR).
Pin ONE gencode: without pinning torch appends the card's sm_90 gencode alongside
-arch=sm_90a and the runtime may pick the sm_90 cubin -> wgmma/TMA = ILLEGAL
INSTRUCTION. Build on the target Hopper GPU (sm_90a).

Usage:
    OMMX_CUTLASS_DIR=/path/to/cutlass python build_ommx_linear.py   # sm_90 + cutlass
    python build_ommx_linear.py                                     # arch auto, no cutlass
"""
from __future__ import annotations   # defer annotations (str | None on py<3.10)

import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def _prepare_cu13_env() -> None:
    """Self-contain the cu130 'ommx' env build at COMPILE time (traps §A).

    The pip cu13 wheels put CUDA headers under
    ``$CONDA_PREFIX/lib/pythonX.Y/site-packages/nvidia/cu13/include`` (NOT
    ``$CONDA_PREFIX/include``), so torch's ``CUDAContext.h`` -> ``cusparse.h``
    will not compile without that dir on ``CPATH`` — added here and inherited by
    the nvcc subprocess (genuinely self-contained for the compile).

    The produced ``.so`` also links conda's newer libstdc++ (``CXXABI_1.3.15``),
    which the system ``/lib64/libstdc++.so.6`` lacks. That is a RUNTIME concern the
    dynamic loader resolves from ``LD_LIBRARY_PATH`` at process START — it CANNOT be
    fixed in-process (torch has already dlopen'd the system libstdc++ by the time
    this runs; an RTLD_GLOBAL preload is too late). So the caller MUST launch with
    ``LD_LIBRARY_PATH=$CONDA_PREFIX/lib`` (gpu_validate.sh + the run_*.sh wrappers
    do). We set it here for nvcc children and warn if the current process is at risk.
    """
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    inc = Path(prefix, "lib", pyver, "site-packages/nvidia/cu13/include")
    if inc.is_dir():
        os.environ["CPATH"] = f"{inc}:{os.environ.get('CPATH', '')}"
    # LINK time (distinct from LD_LIBRARY_PATH below, which is LOAD time). torch emits
    # ``-L$CUDA_HOME/lib64``; a cu13 conda env keeps ``libcudart.so`` in ``lib/``, so the
    # link dies as ``/usr/bin/ld: cannot find -lcudart`` and the parity gate cannot run at
    # all. gcc searches ``LIBRARY_PATH`` for ``-l``, and the nvcc/g++ children inherit this
    # environ, so prepending the env's own lib dirs here fixes it in-process. Guarded on the
    # library actually being there, and only the dirs of THIS env (the one torch itself was
    # loaded from) are added, so it can never pull in a foreign CUDA runtime.
    for _cand in (Path(prefix, "lib"), Path(prefix, "lib64")):
        if not (_cand / "libcudart.so").exists():
            continue
        _cur = os.environ.get("LIBRARY_PATH", "")
        if str(_cand) not in _cur.split(":"):
            os.environ["LIBRARY_PATH"] = f"{_cand}:{_cur}" if _cur else str(_cand)
    libdir = str(Path(prefix, "lib"))
    if Path(libdir, "libstdc++.so.6").exists() and \
            libdir not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{libdir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        print(f"WARN: {libdir} was not on LD_LIBRARY_PATH at startup. If the import "
              f"fails with CXXABI_1.3.15, relaunch with "
              f"LD_LIBRARY_PATH={libdir}:$LD_LIBRARY_PATH (the loader reads it at "
              f"process start; setting it now only helps nvcc child processes).",
              flush=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # ommx_gpu_serve/
SRC = str(HERE / "ommx_linear.cu")


def _sm() -> int:
    cap = torch.cuda.get_device_capability()
    return cap[0] * 10 + cap[1]


def _cutlass_root() -> str | None:
    env = os.environ.get("OMMX_CUTLASS_DIR", "").strip()
    cands = [env] if env else []
    # flashinfer / tilelang bundled trees as a fallback (build_serve.py pattern).
    import site
    for sp in site.getsitepackages():
        cands += [str(Path(sp) / "flashinfer" / "data" / "cutlass"),
                  str(Path(sp) / "tilelang" / "3rdparty" / "cutlass")]
    for r in cands:
        if r and Path(r, "tools/util/include/cutlass/util/mixed_dtype_utils.hpp").exists():
            return r
    return None


def _cutlass_major(root: str) -> int | None:
    """CUTLASS_MAJOR from ``<root>/include/cutlass/version.h`` (None if unreadable)."""
    try:
        for line in Path(root, "include/cutlass/version.h").read_text().splitlines():
            if line.startswith("#define CUTLASS_MAJOR"):
                return int(line.split()[-1])
    except (OSError, ValueError):
        pass
    return None


def build():
    sm = _sm()
    assert sm >= 80, "ommx_linear targets A100 (sm_80) or H100/H200 (sm_90a)"
    sm_arch = "sm_90a" if sm >= 90 else "sm_80"
    flags = ["-O3", "-std=c++17", f"-arch={sm_arch}",
             "--expt-relaxed-constexpr", "--expt-extended-lambda",
             "-DNDEBUG", "--use_fast_math"]
    inc = [str(HERE), str(ROOT / "include")]

    if sm >= 90 and os.environ.get("OMMX_ENABLE_SM90_CUTLASS", "").strip() not in ("", "0", "off"):
        cut = _cutlass_root()
        if cut is None:
            raise RuntimeError("OMMX_ENABLE_SM90_CUTLASS set but no CUTLASS 3.5+ tree "
                               "(mixed_dtype_utils.hpp). Set OMMX_CUTLASS_DIR.")
        flags += ["-DOMMX_ENABLE_SM90_CUTLASS"]
        inc += [f"{cut}/include", f"{cut}/tools/util/include"]
        # fp8-ACT lane (float_e4m3_t MmaType, TileShapeK 128): opt-in via OMMX_CUTE_FP8
        # so the default build stays within the cicc collective budget (one MmaType x
        # TileN{16,64,128} x {dp,shuffled}). When set, cute_base dispatches fp8 A and
        # reorder_B/workspace_size grow the float_e4m3_t twin. Needs a CUTLASS 4.x tree.
        fp8 = os.environ.get("OMMX_CUTE_FP8", "").strip() not in ("", "0", "off")
        if fp8:
            major = _cutlass_major(cut)
            if major is not None and major < 4:
                raise RuntimeError(
                    f"OMMX_CUTE_FP8 needs a CUTLASS 4.x tree for the float_e4m3_t "
                    f"mixed-input twin; {cut} is CUTLASS {major}.x (the 3.5 "
                    f"mixed_dtype_utils.hpp marker is NOT sufficient). Point "
                    f"OMMX_CUTLASS_DIR at a 4.x tree (e.g. the flashinfer bundle).")
            flags += ["-DOMMX_CUTE_FP8"]
        # FUSED lane (in-mainloop FP4-outlier correction, forked RS collective): opt-in via
        # OMMX_CUTE_FUSED so the default build stays within the cicc collective budget (the fused
        # GemmFused variant is only instantiated when set).
        fused = os.environ.get("OMMX_CUTE_FUSED", "").strip() not in ("", "0", "off")
        if fused:
            flags += ["-DOMMX_CUTE_FUSED"]
        print(f"BUILD ommx_linear <- {SRC} | {sm_arch} | CUTLASS {cut} | "
              f"fp8-act={'on' if fp8 else 'off'} | fused={'on' if fused else 'off'}", flush=True)
    else:
        print(f"BUILD ommx_linear <- {SRC} | {sm_arch} | no CUTLASS "
              f"(decode ALU + sm_80 mma prefill + sparse only)", flush=True)

    _prepare_cu13_env()
    mod = load(name="ommx_linear", sources=[SRC],
               extra_include_paths=inc, extra_cuda_cflags=flags, verbose=True)
    # firing evidence smoke (law #5): the module exposes fire_stats.
    print("OMMX_LINEAR_BUILD_DONE rc=0 fire_stats=", dict(mod.fire_stats()), flush=True)
    return mod


if __name__ == "__main__":
    build()
