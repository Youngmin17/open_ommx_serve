"""Rotation (incoherence) utilities for the fakequant rotate-quant A/B.

QuaRot-style: fold an orthonormal Hadamard H into each quantised Linear's input
dim so that  x @ W.T == (x @ H) @ (W @ H).T  exactly (H H.T = I). The low-bit
weight quant then sees the *rotated* weight, whose per-group outliers are spread
out — the hypothesis under test is that this lowers INT2+FP4 quant error at the
fixed OMMX best-recipe. KV rotation applies a head_dim Hadamard to Q/K (post-RoPE)
and V (un-rotated after SDPA) so scores/output stay invariant while K/V quant
sees the flattened values.

Pure-torch dense matmul (fakequant is accuracy emulation, not perf). We use dense
H — verified H@H.T=I at build — instead of the butterfly transform so output
invariance is trivially exact and directly checkable in the parity self-test.
Non-pow2 dims (Llama intermediate 14336 = 28*512) use kron(H28, Sylvester(512))
with the base-28 Hadamard from the QuaRot table.
"""
from __future__ import annotations
import torch

# base-28 Hadamard (H28 @ H28.T == 28 I), copied verbatim from the QuaRot table
# (gptaq/hadamard_utils get_had28) so this module carries no fast_hadamard dep.
_HAD28_ROWS = [
    "++++++++++++++-+++++++++++++",
    "+++-++----++-++-+-++----++-+",
    "++++-++----++-++-+-++----++-",
    "+-+++-++----+++-+-+-++----++",
    "++-+++-++----+++-+-+-++----+",
    "+++-+++-++----+++-+-+-++----",
    "+-++-+++-++---+-++-+-+-++---",
    "+--++-+++-++--+--++-+-+-++--",
    "+---++-+++-++-+---++-+-+-++-",
    "+----++-+++-+++----++-+-+-++",
    "++----++-+++-+++----++-+-+-+",
    "+++----++-+++-+++----++-+-+-",
    "+-++----++-++++-++----++-+-+",
    "++-++----++-++++-++----++-+-",
    "-+++++++++++++--------------",
    "+-+-++----++-+---+--++++--+-",
    "++-+-++----++-----+--++++--+",
    "+-+-+-++----++-+---+--++++--",
    "++-+-+-++----+--+---+--++++-",
    "+++-+-+-++-------+---+--++++",
    "+-++-+-+-++----+--+---+--+++",
    "+--++-+-+-++---++--+---+--++",
    "+---++-+-+-++--+++--+---+--+",
    "+----++-+-+-++-++++--+---+--",
    "++----++-+-+-+--++++--+---+-",
    "+++----++-+-+----++++--+---+",
    "+-++----++-+-+-+--++++--+---",
    "++-++----++-+---+--++++--+--",
]


def _had28(device, dtype) -> torch.Tensor:
    m = torch.tensor(
        [[1.0 if c == "+" else -1.0 for c in row] for row in _HAD28_ROWS],
        device=device, dtype=dtype,
    )
    return m


def _sylvester(n: int, device, dtype) -> torch.Tensor:
    """Unnormalised 2^k Hadamard (H @ H.T == n I), n must be a power of two."""
    assert n & (n - 1) == 0 and n > 0, f"{n} is not a power of two"
    H = torch.ones((1, 1), device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    return H


# odd Hadamard blocks available for non-pow2 dims (extend as models require).
_HAD_BLOCKS = {28: _had28}

_CACHE: dict = {}


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def hadamard_matrix(dim: int, device, dtype=torch.float32,
                    randomize: bool = True, seed: int = 42) -> torch.Tensor:
    """Return a dense orthonormal rotation H (dim x dim, H @ H.T = I).

    pow2 dims use the Sylvester Hadamard; a non-pow2 dim d = K * 2^k (K an odd
    Hadamard block) uses kron(H_K, Sylvester(2^k)). `randomize` right-multiplies
    a seeded +-1 sign diagonal (QuaRot 'random Hadamard' — still orthonormal),
    making the rotation deterministic given `seed` for a reproducible A/B.
    """
    key = (dim, str(device), str(dtype), randomize, seed)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    build_dtype = torch.float64  # build in fp64 for a clean orthonormality check
    if _is_pow2(dim):
        H = _sylvester(dim, device, build_dtype)
    else:
        block = None
        for K, fn in _HAD_BLOCKS.items():
            if dim % K == 0 and _is_pow2(dim // K):
                block = fn(device, build_dtype)
                H = torch.kron(block, _sylvester(dim // K, device, build_dtype))
                break
        if block is None:
            raise ValueError(
                f"no Hadamard construction for dim={dim} "
                f"(not pow2 and no odd block in {sorted(_HAD_BLOCKS)})"
            )
    H = H / (dim ** 0.5)  # normalise -> orthonormal (H @ H.T = I)

    if randomize:
        g = torch.Generator(device="cpu").manual_seed(seed + dim)
        signs = (torch.randint(0, 2, (dim,), generator=g) * 2 - 1).to(build_dtype)
        H = H * signs.to(device).view(1, -1)  # right-multiply diag(signs)

    err = (H @ H.t() - torch.eye(dim, device=device, dtype=build_dtype)).abs().max().item()
    assert err < 1e-6, f"H@H.T != I for dim={dim} (max err {err:.2e})"

    H = H.to(dtype)
    _CACHE[key] = H
    return H


def _rotate_input_pre_hook(dim: int, randomize: bool, seed: int):
    """forward_pre_hook: rotate a Linear's input by H(dim) on the last (feature)
    axis. H is (re)built on the INPUT's device/dtype each call — the model is on
    CPU when the hook is registered but calib/eval forwards run on CUDA, so H must
    follow the input device. hadamard_matrix is cached and deterministic in
    (dim, seed), so this yields the exact same matrix used to rotate the weight."""
    def hook(module, args):
        x = args[0]
        H = hadamard_matrix(dim, x.device, x.dtype, randomize=randomize, seed=seed)
        return (x @ H,) + args[1:]
    return hook


def apply_weight_rotation(model, quant_linear_cls, randomize: bool = True, seed: int = 42):
    """Fold H into every Quant_Linear: W <- W @ H(in_features) and register a
    pre-hook that rotates the module input by the same H. Output-invariant in fp;
    the subsequent low-bit weight quant sees the rotated (outlier-flattened) W.
    Returns the number of rotated linears. The weight matmul runs on GPU when
    available (the model is still on CPU here; a CPU 14336x14336 matmul is slow)."""
    n = 0
    dims = {}
    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    for name, m in model.named_modules():
        if not isinstance(m, quant_linear_cls):
            continue
        w = m.weight
        rot_dev = gpu if gpu is not None else w.device
        H = hadamard_matrix(m.in_features, rot_dev, torch.float32,
                            randomize=randomize, seed=seed)
        with torch.no_grad():
            wr = (w.data.to(rot_dev, torch.float32) @ H).to(w.dtype)
            m.weight.data = wr.to(w.device)
        # device-adaptive pre-hook (H follows the input device at forward time)
        m.register_forward_pre_hook(_rotate_input_pre_hook(m.in_features, randomize, seed))
        dims[m.in_features] = dims.get(m.in_features, 0) + 1
        n += 1
    # drop the fp32 build matrices (up to 14336^2 = 1.6GB on GPU); the eval pre-hooks
    # rebuild the small fp16 H lazily. Deterministic seed => identical matrix.
    _CACHE.clear()
    if gpu is not None:
        torch.cuda.empty_cache()
    print(f"[ROTATE-W] rotated {n} linears; in_features histogram={dims}")
    return n
