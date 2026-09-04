#!/usr/bin/env python
"""
wq_eval.py — OMMX weight quantization + accuracy measurement, all in ONE process.

Applies the OMMX INT2+FP4 format to every linear weight using an error-feedback calibration
solver (closed-form second-order: after quantizing each input channel, the residual is pushed
into the not-yet-quantized channels weighted by the inverse input Hessian, so the LAYER OUTPUT
— not the raw weight — is preserved; strongest exactly at 2-bit). An optional input-correction
term additionally accounts for the layer input itself being quantized at serving time. It then
measures the true INT2+FP4 (decode-path) WikiText perplexity in the same run.

FORMAT IS UNCHANGED. The inner per-channel quantizer reproduces `quant_weight.weight_quantizer(
mode="decode")` bit-for-bit (per-group INT2 asymmetric + per-row FP4 outliers, |W|*act saliency).
Calibration only changes WHICH dense INT2 values are chosen — no extra bits, contiguous groups
preserved (= the serving layout). Special layers (first/last K + special_projs) stay pure FP4.

Run `--selftest` first: it proves the inner quantizer == weight_quantizer(mode="decode") when the
error feedback is disabled (H=I), for both the fp16 and the power-of-two scale.
"""
import os, sys, gc, time, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F

# FP4 E2M1 grid + range — MUST match apply_fp_range_mapped(fp_bits=2) in quant_weight.py.
FP4_VALUES = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
              0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0]
FP4_RANGE = 12.0

# Sequential quantization groups (in dependency order). Each group's input is determined by the
# EARLIER groups, so we quantize a group, then collect the next group's statistics with it quantized.
SEQ_GROUPS = [
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),  # input = input_layernorm(x)
    ("self_attn.o_proj",),                                         # input = attention(quantized q/k/v)
    # leaf-name match so MoE experts (mlp.experts.{i}.gate_proj) ALSO route to the calibrated solver,
    # not just dense mlp.gate_proj. Dense models: leaf strings identical -> bit-identical. Router
    # (mlp.gate) does NOT end with gate_proj so it stays excluded (+ Fix A removes it entirely).
    ("gate_proj", "up_proj"),                                      # input = post_attn_layernorm(x)
    ("down_proj",),                                                # input = act(gate)*up (quantized)
]


#: Special (pure-FP4) layers seen during an export run. They are NOT exportable as
#: calibration decisions -- see _export_decisions -- and a partial export must be visible.
_EXPORT_SKIPPED = []


def _export_decisions(root, name, dec):
    """Write one module's quantization DECISIONS for ``w_packer --calibrated``.

    The packer re-encodes these itself, so what travels is the DECISIONS (which lanes are
    outliers, the group scale/zero-point, the FP4 range map) plus ``W``, the solver's own
    dequantized result. The packer's gate then asserts dequant(planes) == W exactly, which
    is what makes "the bundle is the model the solver measured" checkable rather than
    assumed.

    REFUSES a non-power-of-two scale here rather than at pack time: without ``--pow2`` the
    solver picks full-precision scales, the E8M0 plane can only store an exponent byte, and
    an export that looks fine now would be rejected (or worse, rounded) hours later.
    """
    import os as _os
    from safetensors.torch import save_file as _save
    if dec is None:
        raise RuntimeError(f"{name}: --export-decisions is set but the solver recorded "
                           f"none (record_decisions was not enabled before fasterquant)")
    sc = dec["scale"]
    exp = torch.log2(sc.clamp_min(1e-38))
    off = float((exp - torch.round(exp)).abs().max())
    if off > 1e-6:
        raise RuntimeError(
            f"{name}: the solver chose scales that are not powers of two (max log2 "
            f"deviation {off:.3e}). The OMMX weight format stores ONE EXPONENT BYTE per "
            f"group, so these cannot be packed without discarding exactly what the "
            f"calibration chose. Re-run with --pow2.")
    _os.makedirs(root, exist_ok=True)
    # Module name == the checkpoint tensor name without ".weight", which is what
    # w_packer.CalibrationSource looks up.
    _save({k: v.contiguous() for k, v in dec.items()},
          _os.path.join(root, name + ".safetensors"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--gpu", default="0")
    p.add_argument("--mode", choices=["base", "full"], default="base",
                   help="base = error-feedback only; full = also add the input-correction term "
                        "(accounts for the layer input being quantized at serving).")
    p.add_argument("--bits", type=int, default=2)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--outlier-percent", type=float, default=0.25)
    p.add_argument("--special-first-k", type=int, default=2)
    p.add_argument("--special-last-k", type=int, default=2)
    p.add_argument("--special-projs", default="down_proj", help="comma list, e.g. down_proj,o_proj")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--calib-data", default="pile-10k", choices=["pile-10k", "c4", "wikitext"],
                   help="calibration corpus (default pile-10k)")
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--blocksize", type=int, default=128, help="lazy-batch block (forced to a multiple of group-size)")
    p.add_argument("--alpha", type=float, default=0.25, help="input-correction strength (only used with --mode full)")
    p.add_argument("--pow2", action="store_true",
                   help="OMMX-format-faithful: constrain the per-group INT2 scale to a power of two "
                        "(2^e, the OMMX shared scale). Off = full-precision fp16 scale (format-relaxed).")
    p.add_argument("--scale-search", type=int, default=3,
                   help="with --pow2: per-group search radius over integer exponents e0-R..e0+R, picking "
                        "the one minimizing dense-INT2 reconstruction MSE. 0 = naive round.")
    p.add_argument("--opt-zp", action="store_true",
                   help="also search the BF16 zero-point z (OMMX-allowed) jointly with the scale, over "
                        "zp_steps offsets in [-0.5,0.5]*scale, minimizing the (weighted) dense MSE.")
    p.add_argument("--zp-steps", type=int, default=5, help="number of zero-point offset candidates")
    p.add_argument("--hess-aware", action="store_true",
                   help="weight the scale/zp search MSE by the Hessian diagonal H[i,i] (input energy) -> "
                        "the output-error proxy, instead of plain per-weight MSE.")
    p.add_argument("--sequential", action="store_true",
                   help="quantize the sequential groups in order (q/k/v -> o_proj -> gate/up -> down_proj), "
                        "collecting each group's statistics with the EARLIER groups already quantized "
                        "(captures the intra-layer input shift, e.g. o_proj seeing quantized q/k/v). "
                        "Default (off) = single statistics pass, all linears quantized at once.")
    p.add_argument("--ppl-seqlen", type=int, default=4096)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--decode-ppl", action="store_true",
                   help="also measure DECODE-path PPL: prefill --decode-pf tokens (prefill fp4 KV + "
                        "prefill sparse), then autoregressively decode --decode-len tokens (OMMX INT2+FP4 "
                        "decode KV, q_len=1, sparse OFF). Exercises the full prefill->decode pipeline.")
    p.add_argument("--decode-pf", type=int, default=1024, help="decode-ppl prefill length")
    p.add_argument("--decode-len", type=int, default=256, help="decode-ppl # autoregressive decode tokens")
    p.add_argument("--eval-tasks", default="",
                   help="comma list of lm-eval tasks (e.g. mmlu,arc_challenge,hellaswag,niah). If set, run "
                        "lm-eval on the baked INT2+FP4 model instead of wikitext PPL.")
    p.add_argument("--export-decisions", default=None, metavar="DIR",
                   help="write each calibrated module's quantization DECISIONS to DIR for "
                        "`w_packer pack --calibrated DIR`. Requires --pow2: the OMMX "
                        "weight format stores a power-of-two group scale, so scales chosen "
                        "without it cannot be packed. Special (pure-FP4) layers are not "
                        "exportable and are reported at the end.")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=0, help="limit examples per task (0=all; for smoke)")
    p.add_argument("--eval-batch-size", default="auto", help="lm-eval batch size (int or 'auto')")
    p.add_argument("--tag", default="")
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false",
                    default=True, help="disable for archs natively supported by transformers "
                    "(e.g. phi3) whose remote-code modeling file is version-skewed")
    p.add_argument("--selftest", action="store_true", help="run format-equivalence unit test and exit")
    p.add_argument("--calib-bf16", action="store_true",
                   help="load the model in bf16 (not the default fp16) for the calibration/bake pass -- "
                        "fp16's +-65504 range can silently overflow to inf on outlier activation channels "
                        "during Hessian statistics collection; bf16 matches the eval dtype's range.")
    p.add_argument("--baseline", action="store_true",
                   help="skip quantization entirely; eval the raw (bf16) model — protocol baseline")
    p.add_argument("--kv", action="store_true",
                   help="W+KV mode: also apply KV-cache quantization (INT2+FP4) on top of the calibrated "
                        "weights, then measure the combined PPL. KV is applied BEFORE the weight bake so the "
                        "weight calibration is KV-aware (same order as serving).")
    p.add_argument("--kv-group-size", type=int, default=64, help="KV-cache quantization group size")
    p.add_argument("--kv-sink", type=int, default=8, help="# attention-sink tokens kept in BF16")
    p.add_argument("--kv-outliers-per-group", type=int, default=1, help="FP4 outliers per KV group")
    p.add_argument("--kv-prefill", choices=["fp4", "bf16"], default="fp4",
                   help="prefill-phase KV recipe (decode always uses OMMX INT2+FP4). "
                        "fp4 = FP4 group-quant on prefill KV (default; the historical setup). "
                        "bf16 = leave prefill KV unquantized (prefill_key/value_bits=16) so a fair "
                        "comparison isolates the decode-time KV recipe from any prefill-KV quant.")
    p.add_argument("--sparse", action="store_true",
                   help="fakesparse: apply structured (block N:M tiered) sparsity on top of fakequant. "
                        "Knobs come from OMMX_SPARSE_* env (see fakesparse.config). Off => byte-identical "
                        "to the fakequant baseline. KV sparsity only affects the --kv prefill path.")
    # rotate (QuaRot-style incoherence): fold an orthonormal Hadamard into each
    # quantised Linear's input dim (--rotate-w) and/or a head_dim Hadamard into
    # Q/K/V (--rotate-kv). Output-invariant in fp (see rotation.py); the low-bit
    # weight/KV quant then sees outlier-flattened tensors. Off => byte-identical
    # to the fakequant baseline. A/B vs the best-recipe isolates rotation's effect.
    p.add_argument("--rotate-w", action="store_true",
                   help="apply weight rotation (per-linear input-dim Hadamard) before the weight bake")
    p.add_argument("--rotate-kv", action="store_true",
                   help="apply KV rotation (head_dim Hadamard on Q/K post-RoPE and V; un-rotated after SDPA)")
    p.add_argument("--rotate-seed", type=int, default=42,
                   help="seed for the randomised-Hadamard sign diagonal (reproducible rotation)")
    p.add_argument("--rotate-selftest", action="store_true",
                   help="run rotation math parity (H@H.T=I + per-linear/KV output invariance) and exit")
    return p.parse_args()


# ----------------------------------------------------------------------------------------------------
# OMMX weight quantizer: an error-feedback column loop with the INT2+FP4 decode dequant as the inner
# quantizer. With --mode full, an extra input-correction term is folded into the feedback update.
# ----------------------------------------------------------------------------------------------------
class OMMXWeightQuantizer:
    def __init__(self, rows, columns, group_size, outlier_percent, bits, dev, input_correct=False, alpha=0.25,
                 use_pow2=False, scale_search=3, opt_zp=False, zp_steps=5, hess_aware=False):
        assert columns % group_size == 0, f"in_features {columns} not divisible by group_size {group_size}"
        self.rows, self.columns = rows, columns
        self.gs = group_size
        self.k = max(1, int(group_size * outlier_percent)) if outlier_percent > 0 else 0   # outliers/group/row
        self.outlier_percent = outlier_percent
        self.bits = bits
        self.qmax = (1 << bits) - 1                           # INT2 -> 3 (levels 0..3, asymmetric)
        self.dev = dev
        self.input_correct = input_correct
        self.alpha = alpha
        self.use_pow2 = use_pow2                              # OMMX shared scale = 2^e
        self.scale_search = scale_search                     # exponent search radius
        self.opt_zp = opt_zp                                 # search the BF16 zero-point z (vs z=min)
        self.zp_steps = zp_steps                             # # of z-offset candidates
        self.hess_aware = hess_aware                         # weight scale/z search MSE by diag(H)
        self.Hdiag = None                                    # set in fasterquant (input Hessian diagonal)
        self.H = torch.zeros((columns, columns), device=dev)
        self.nsamples = 0
        self.act_scales = None                               # [I] running max |x| per input channel
        self.fp_values = torch.tensor(FP4_VALUES, device=dev, dtype=torch.float32)
        self.fp_range = FP4_RANGE
        if input_correct:
            self.cov_corr = torch.zeros((columns, columns), device=dev)
            self.fp_inp = []                                 # clean inputs [tokens, I], popped per add_batch
        # DECISION RECORDING (opt-in, off by default so nothing changes for eval runs).
        # fasterquant returns Q, the DEQUANTIZED weight -- and re-quantizing Q with the
        # packer's min/max path is NOT idempotent (measured max|W_ref2-W_ref1|=5.2e-2,
        # different codes AND scales), so a bundle packed from Q silently throws this
        # calibration away. The packer's --calibrated path instead consumes the per-group
        # DECISIONS, which are exactly what _group_params already froze: zero-point,
        # scale, outlier mask and FP4 range map. Recording them here is the only place
        # they exist.
        self.record_decisions = False
        self.decisions = None
        #: Storage dtype of the zero-point, matched to the bundle format (w_packer's
        #: default recipe is BF16). Applied inside _store_zp before the codes are chosen.
        self.zp_dtype = torch.bfloat16

    @torch.no_grad()
    def add_batch(self, inp):
        # inp: [B,T,I] or [T,I] — the (quantized-propagated) input this linear actually receives.
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.to(self.dev)
        # running max |x| per channel (saliency, max-based to match calibrate.set_act_scales)
        cur = inp.abs().amax(dim=0).float()
        self.act_scales = cur if self.act_scales is None else torch.maximum(self.act_scales, cur)

        tmp = inp.shape[0]
        inp_t = inp.t().float()                              # [I, tokens]
        self.H *= self.nsamples / (self.nsamples + tmp)
        if self.input_correct:
            self.cov_corr *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        scl = math.sqrt(2 / self.nsamples)
        inp_s = scl * inp_t
        self.H += inp_s.matmul(inp_s.t())
        if self.input_correct:
            fp = self.fp_inp.pop(0).to(self.dev).t().float() # [I, tokens]  (clean input, same sample)
            dX = fp * scl - inp_s
            self.cov_corr += dX.matmul(inp_s.t())

    def _fp4_map_params(self, idx, outmask):
        # FP4 outlier range-mapping params per row over the row's outlier index range. idx=(W-z)/scale.
        if self.k > 0:
            big = torch.full_like(idx, 1e9)
            small = torch.full_like(idx, -1e9)
            idx_min = torch.where(outmask, idx, big).min(dim=-1, keepdim=True).values
            idx_max = torch.where(outmask, idx, small).max(dim=-1, keepdim=True).values
            idx_range = (idx_max - idx_min).clamp_min(1e-8)
            return self.fp_range / idx_range, (idx_max + idx_min) / 2.0     # mapscale, mapcenter [O,1]
        return torch.ones_like(idx[:, :1]), torch.zeros_like(idx[:, :1])

    @torch.no_grad()
    def _store_zp(self, z):
        """Round the zero-point to the dtype the BUNDLE stores, BEFORE codes are chosen.

        This function used to say "z stays BF16 (OMMX-allowed)" in a comment while keeping
        z in float32. The weight format stores a BF16 zero-point, and the packer rounds it
        on the way to disk -- so codes chosen against an f32 z are every one of them
        fractionally mismatched to the z the kernel reads back. Measured through the
        export path: max|dequant(planes) - solver Q| = 2.4e-4 instead of 0, which the
        packer's calibration gate (correctly) refuses.

        Deciding against the stored value costs nothing and makes the solver's own output
        exactly reproducible from the bundle -- i.e. the accuracy you measure is the
        accuracy you ship.
        """
        if self.zp_dtype is None or self.zp_dtype == torch.float32:
            return z
        return z.to(self.zp_dtype).to(z.dtype)

    def _group_params(self, Wg, col):
        # Wg: [O, gs] CURRENT (error-corrected) weights of this group. Returns frozen group params,
        # mirroring weight_quantizer(mode="decode") restricted to one group. zero-point z=min_vals (BF16,
        # OMMX-allowed); scale is full-precision (relaxed) or a power-of-two 2^e (OMMX-faithful, --pow2).
        big = torch.full_like(Wg, 1e9)
        small = torch.full_like(Wg, -1e9)
        gs = Wg.shape[1]
        if self.k > 0:                                       # |W|*act top-k -> FP4 outliers
            sal = Wg.abs() * self.act_scales[col:col + gs].view(1, gs) if self.act_scales is not None else Wg.abs()
            topk = sal.topk(self.k, dim=-1).indices          # [O, k] per-row outliers
            outmask = torch.zeros_like(Wg, dtype=torch.bool)
            outmask.scatter_(-1, topk, True)
        else:                                                # outlier_percent==0 -> pure INT2 (no FP4)
            outmask = torch.zeros_like(Wg, dtype=torch.bool)
        qmask = ~outmask
        min_vals = torch.where(qmask, Wg, big).min(dim=-1, keepdim=True).values
        max_vals = torch.where(qmask, Wg, small).max(dim=-1, keepdim=True).values
        min_vals = torch.where(min_vals > 1e8, torch.zeros_like(min_vals), min_vals)
        max_vals = torch.where(max_vals < -1e8, torch.zeros_like(max_vals), max_vals)
        s0 = (max_vals - min_vals).clamp_min(1e-8) / self.qmax            # ideal full-precision scale [O,1]

        # candidate scales: pow2 -> 2^e over [e0-R,e0+R] (OMMX-faithful); else the fp16 ideal (relaxed).
        if self.use_pow2:
            e0 = torch.round(torch.log2(s0.clamp_min(1e-12)))            # [O,1]
            scale_cands = [torch.pow(2.0, e0 + d) for d in range(-self.scale_search, self.scale_search + 1)]
        else:
            scale_cands = [s0]
        # candidate zero-point offsets (fraction of scale). z stays BF16 (OMMX-allowed); default = min_vals.
        deltas = (torch.linspace(-0.5, 0.5, self.zp_steps).tolist() if self.opt_zp else [0.0])

        if len(scale_cands) == 1 and len(deltas) == 1:
            scale, z = scale_cands[0], min_vals                         # no search (backward-compatible)
            z = self._store_zp(z)
        else:
            # Per-row search over (scale, z) minimizing dense-INT2 reconstruction error. hess_aware weights
            # each input channel by its Hessian diagonal H[i,i] (input energy) -> the OUTPUT-error proxy
            # ||(W-Q)X||^2 ~ sum_i H[i,i](W-Q)[:,i]^2, instead of plain per-weight MSE.
            if self.hess_aware and self.Hdiag is not None:
                imp = self.Hdiag[col:col + gs].view(1, gs).clamp_min(0)
            else:
                imp = torch.ones((1, gs), device=Wg.device)
            w_imp = imp * qmask.float()                                  # [O,gs] (restrict to dense)
            denom = w_imp.sum(dim=-1, keepdim=True).clamp_min(1e-8)      # [O,1]
            best_scale = scale_cands[0].expand_as(s0).clone()
            best_z = min_vals.clone()
            best_mse = torch.full_like(s0, float("inf"))
            for sc in scale_cands:
                for dl in deltas:
                    z_c = self._store_zp(min_vals + dl * sc)
                    q = torch.round((Wg - z_c) / sc).clamp(0, self.qmax) * sc + z_c
                    mse = (((q - Wg) ** 2) * w_imp).sum(dim=-1, keepdim=True) / denom    # [O,1]
                    take = mse < best_mse
                    best_scale = torch.where(take, sc.expand_as(s0), best_scale)
                    best_z = torch.where(take, z_c, best_z)
                    best_mse = torch.where(take, mse, best_mse)
            scale, z = best_scale, best_z

        mapscale, mapcenter = self._fp4_map_params((Wg - z) / scale, outmask)
        return (z.squeeze(-1), scale.squeeze(-1), outmask,
                mapscale.squeeze(-1), mapcenter.squeeze(-1))

    @torch.no_grad()
    def _quant_col(self, w, jl, mv, sc, outmask, ms, mc):
        # Dequantize one column (all O rows) with frozen group params. outlier -> FP4-mapped, else INT2.
        ol = outmask[:, jl]
        q_int = torch.round((w - mv) / sc).clamp(0, self.qmax) * sc + mv
        idx = (w - mv) / sc
        shifted = (idx - mc) * ms
        d = (shifted.unsqueeze(-1) - self.fp_values.view(1, -1)).abs()    # [O, 16]
        nearest = self.fp_values[d.argmin(dim=-1)]                        # [O]
        q_fp = (nearest / ms + mc) * sc + mv
        return torch.where(ol, q_fp, q_int)

    @torch.no_grad()
    def fasterquant(self, W, blocksize=128, percdamp=0.01):
        # W: [O, I] original weight. Returns Q (dequantized INT2+FP4) [O, I] in the weight's own dtype
        # (fp16 or bf16). Preserving bf16 lets --calib-bf16 bake bf16 caches -> the eval forward runs in
        # bf16 (range ~3e38) instead of fp16 (max 65504), which avoids the layer-2 SwiGLU overflow that
        # turns short-context (arc) logits into NaN on wide models (Qwen3-32B). Solve math stays fp32.
        out_dtype = W.dtype if W.dtype in (torch.float16, torch.bfloat16) else torch.float16
        W = W.float().clone()
        gs = self.gs
        blocksize = gs * max(1, blocksize // gs)             # force multiple of group size

        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        if self.input_correct:
            self.cov_corr[:, dead] = 0
        self.Hdiag = torch.diag(H).clone()                   # input-channel energy for hess-aware search

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        P = None
        if self.input_correct:
            P = self.alpha * ((self.cov_corr @ Hinv.T).triu_(diagonal=1)) @ Hinv

        Q = torch.zeros_like(W)
        gmv = gsc = gom = gms = gmc = None
        rec = None
        if self.record_decisions:
            G = self.columns // gs
            rec = dict(
                zp=torch.zeros(self.rows, G, dtype=torch.float32),
                scale=torch.zeros(self.rows, G, dtype=torch.float32),
                omask=torch.zeros(self.rows, G, gs, dtype=torch.bool),
                map_scale=torch.ones(self.rows, G, dtype=torch.float32),
                map_center=torch.zeros(self.rows, G, dtype=torch.float32))
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2] if self.input_correct else None

            for i in range(count):
                col = i1 + i
                w = W1[:, i]
                d = Hinv1[i, i]
                if col % gs == 0:                            # group boundary: freeze params on live W1
                    gmv, gsc, gom, gms, gmc = self._group_params(W1[:, i:i + gs], col)
                    if rec is not None:
                        g = col // gs
                        rec["zp"][:, g] = gmv.detach().float().cpu()
                        rec["scale"][:, g] = gsc.detach().float().cpu()
                        rec["omask"][:, g] = gom.detach().cpu()
                        rec["map_scale"][:, g] = gms.detach().float().cpu()
                        rec["map_center"][:, g] = gmc.detach().float().cpu()
                q = self._quant_col(w, col % gs, gmv, gsc, gom, gms, gmc)
                Q1[:, i] = q
                err1 = (w - q) / d
                upd = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                if self.input_correct:
                    upd = upd - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                W1[:, i:] -= upd
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            tail = Err1.matmul(Hinv[i1:i2, i2:])
            if self.input_correct:
                tail = tail - W1.matmul(P[i1:i2, i2:])
            W[:, i2:] -= tail

        if torch.any(torch.isnan(Q)) or torch.any(torch.isinf(Q)):
            raise ValueError("NaN/Inf in quantizer output Q")
        if rec is not None:
            # Q is stored alongside the decisions: the packer re-encodes THESE values, and
            # its gate asserts dequant(planes) == Q exactly, so a drift between what the
            # solver measured and what gets shipped cannot pass silently.
            rec["W"] = Q.detach().float().cpu()
            self.decisions = rec
        return Q.to(out_dtype)

    def free(self):
        self.H = None
        if self.input_correct:
            self.cov_corr = None
            self.fp_inp = []
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------
# Rotation parity: H@H.T=I for the real Llama-3.1-8B dims, per-linear output invariance
# x@W.T == (x@H)@(W@H).T, and KV head_dim invariance (Q/K rotation cancels in scores; V
# rotation cancels via the post-SDPA un-rotate). Cheap (no model) law-#12 gate before A/B.
# ----------------------------------------------------------------------------------------------------
def rotate_selftest():
    from rotation import hadamard_matrix
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    ok = True
    for dim in (128, 4096, 14336):                      # head_dim, hidden, intermediate
        H = hadamard_matrix(dim, dev, torch.float32)
        e = (H @ H.t() - torch.eye(dim, device=dev)).abs().max().item()
        print(f"[ROT-SELFTEST] dim={dim:6d} H@H.T-I max={e:.2e}")
        ok &= e < 1e-4  # fp32 accumulation over up to 14336 terms (~1e-5); fp64 build assert is ~1e-13
    # per-linear invariance (fp16, the eval dtype)
    for (O, I) in ((4096, 4096), (4096, 14336)):
        x = torch.randn(8, I, device=dev, dtype=torch.float16)
        W = torch.randn(O, I, device=dev, dtype=torch.float16)
        H = hadamard_matrix(I, dev, torch.float16)
        y = x @ W.t()
        yr = (x @ H) @ (W @ H).t()
        rel = ((y.float() - yr.float()).abs().max() / y.float().abs().max().clamp_min(1e-6)).item()
        print(f"[ROT-SELFTEST] linear O={O} I={I:6d} rel_diff={rel:.2e}")
        ok &= rel < 5e-2
    # KV head_dim invariance: scores (Q@R)(K@R).T == Q@K.T; output (p@(V@R))@R.T == p@V
    R = hadamard_matrix(128, dev, torch.float32)
    Q = torch.randn(2, 4, 16, 128, device=dev); K = torch.randn(2, 4, 16, 128, device=dev)
    V = torch.randn(2, 4, 16, 128, device=dev)
    s0 = Q @ K.transpose(-1, -2); s1 = (Q @ R) @ (K @ R).transpose(-1, -2)
    p = torch.softmax(s0, dim=-1)
    o0 = p @ V; o1 = (p @ (V @ R)) @ R.t()
    es = (s0 - s1).abs().max().item(); eo = (o0 - o1).abs().max().item()
    print(f"[ROT-SELFTEST] KV scores max={es:.2e} value max={eo:.2e}")
    ok &= es < 1e-3 and eo < 1e-3
    print(f"[ROT-SELFTEST] {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


# ----------------------------------------------------------------------------------------------------
# Self-test: with feedback disabled (H=I), OMMXWeightQuantizer must reproduce weight_quantizer(mode="decode").
# ----------------------------------------------------------------------------------------------------
def selftest():
    from quant_weight import weight_quantizer
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # float32 on BOTH sides so the (identical) |W|*act top-k selection can't diverge on fp16 near-ties;
    # this isolates the dequant LOGIC. Residual diff is only the FP4 nearest-grid rounding (~1e-3).
    for pow2 in (False, True):       # pow2=True (search=0=naive) must match weight_quantizer(use_pow2=True)
        for gs in (16, 32):
            for op in (0.25, 0.125, 0.0):
                O, I = 64, 256
                W = torch.randn(O, I, device=dev, dtype=torch.float32)
                act = torch.rand(I, device=dev, dtype=torch.float32) * 3 + 0.1
                ref = weight_quantizer(W, bits=2, group_size=gs, outlier_percent=op, act_scales=act,
                                       use_pow2=pow2, mode="decode").float()
                g = OMMXWeightQuantizer(O, I, gs, op, 2, dev, use_pow2=pow2, scale_search=0)
                g.H = torch.eye(I, device=dev)               # identity -> no error feedback
                g.act_scales = act
                got = g.fasterquant(W, blocksize=128, percdamp=0.0).float()
                md = (got - ref).abs().max().item()
                ok = md < 5e-3
                print(f"[SELFTEST] pow2={pow2} gs={gs} o%={op}  max|Q_calib - Q_ref| = {md:.3e}  {'OK' if ok else 'FAIL'}")
                if not ok:
                    raise SystemExit(f"SELFTEST FAILED pow2={pow2} gs={gs} o%={op} (max diff {md})")
    print("[SELFTEST] PASS — OMMXWeightQuantizer == weight_quantizer(mode=decode) for both fp16 and pow2 scale under H=I")


@torch.no_grad()
def wikitext_ppl(model, tokenizer, seqlen, limit, device):
    from datasets import load_dataset
    import tqdm
    te = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    te = tokenizer("\n\n".join(te["text"]), return_tensors="pt").input_ids.to(device)
    lf = nn.CrossEntropyLoss()
    nlls = []
    # Tokenizers that do NOT auto-prepend BOS (e.g. EXAONE-4.0, add_bos_token=False) collapse
    # without an attention-sink BOS at each window start: raw wikitext PPL 57 vs 31 with BOS, and
    # incoherent greedy generation ("...a capital of France is a capital of France"). Prepend BOS
    # per window for these. add_bos=True models (Llama/Qwen/Mistral) keep the original path exactly.
    needs_bos = (getattr(tokenizer, "add_bos_token", False) is False
                 and tokenizer.bos_token_id is not None)
    if needs_bos:
        bos = torch.tensor([[tokenizer.bos_token_id]], device=te.device)
        win = seqlen - 1  # real tokens per window; +1 BOS => seqlen model inputs
        nsamples = te.numel() // win
        if limit and limit > 0:
            nsamples = min(nsamples, limit)
        for i in tqdm.tqdm(range(nsamples), desc="PPL", leave=False):
            b = torch.cat([bos, te[:, i * win:(i + 1) * win]], dim=1)
            logits = model(b).logits
            sl = logits[:, :-1, :].contiguous().float()
            nlls.append(lf(sl.view(-1, sl.size(-1)), b[:, 1:].reshape(-1)).float() * win)
            if os.environ.get("OMMX_PPL_POSDUMP") and i < 3:
                # per-position NLL: is the blowup at pos0/sink (window-boundary collapse) or flat/spiky?
                import numpy as _np
                lp = nn.CrossEntropyLoss(reduction="none")
                per = lp(sl.view(-1, sl.size(-1)), b[:, 1:].reshape(-1)).float().cpu().numpy()
                print(f"[DIAG-POS] win={i} pos0={per[0]:.2f} pos1={per[1]:.2f} pos2={per[2]:.2f} "
                      f"mean1+={_np.mean(per[1:]):.3f} max={per.max():.1f}@{int(per.argmax())} "
                      f"p99={_np.percentile(per,99):.2f} frac>10={float((per>10).mean()):.4f}")
        return torch.exp(torch.stack(nlls).sum() / (nsamples * win)).item()
    nsamples = te.numel() // seqlen
    if limit and limit > 0:
        nsamples = min(nsamples, limit)
    for i in tqdm.tqdm(range(nsamples), desc="PPL", leave=False):
        b = te[:, i * seqlen:(i + 1) * seqlen].to(device)
        logits = model(b).logits
        sl = logits[:, :-1, :].contiguous().float()
        lab = te[:, i * seqlen:(i + 1) * seqlen][:, 1:]
        nlls.append(lf(sl.view(-1, sl.size(-1)), lab.view(-1)).float() * seqlen)
    return torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()


@torch.no_grad()
def decode_ppl(model, tokenizer, pf_len, dec_len, limit, device):
    """PPL over autoregressive DECODE steps — exercises the FULL prefill->decode KV pipeline that the
    single-forward wikitext_ppl (all prefill, q_len>1) never touches. Prefill ``pf_len`` tokens (prefill
    path: fp4 KV + any prefill Q/P/KV sparse, is_prefill=True), then decode ``dec_len`` tokens one at a
    time with use_cache (decode path: OMMX INT2+FP4 KV, q_len=1, is_prefill=False -> sparse OFF). NLL is
    accumulated over the decoded positions only, so this is the language-modeling quality of the
    sparse-prefilled + OMMX-decoded pipeline. Teacher-forced (feeds the true token each step)."""
    from datasets import load_dataset
    import tqdm
    te = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    te = tokenizer("\n\n".join(te["text"]), return_tensors="pt").input_ids.to(device)
    win = pf_len + dec_len
    nsamples = te.numel() // win
    if limit and limit > 0:
        nsamples = min(nsamples, limit)
    lf = nn.CrossEntropyLoss(reduction="sum")
    tot_nll, tot_tok = 0.0, 0
    for i in tqdm.tqdm(range(nsamples), desc="decodePPL", leave=False):
        b = te[:, i * win:(i + 1) * win]
        out = model(b[:, :pf_len], use_cache=True)            # prefill (q_len>1)
        past = out.past_key_values
        lg = out.logits[:, -1, :].float()                     # predicts token at pf_len
        tot_nll += lf(lg, b[:, pf_len]).item(); tot_tok += 1
        for t in range(pf_len, pf_len + dec_len - 1):         # decode steps (q_len=1)
            out = model(b[:, t:t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            lg = out.logits[:, -1, :].float()                 # predicts token at t+1
            tot_nll += lf(lg, b[:, t + 1]).item(); tot_tok += 1
    return float(torch.exp(torch.tensor(tot_nll / max(1, tot_tok))))


def main():
    a = parse_args()
    # honor an externally-allocated GPU: gpu_cap.sh run exports CUDA_VISIBLE_DEVICES to the
    # capped-and-claimed physical idx, which we must NOT clobber (else the cap is bypassed and
    # two jobs collide on GPU0). Fall back to --gpu only when nothing was pre-assigned.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("SAFETENSORS_DISABLE_MMAP", "1")

    if a.selftest:
        selftest()
        return

    if a.rotate_selftest:
        rotate_selftest()
        return

    # rotate-kv is consumed inside Quant_Attention.__init__ (built by apply_kv_cache_quant
    # below), so publish it to the env before that runs. rotate-w is applied directly here.
    if a.rotate_kv:
        os.environ["OMMX_ROTATE_KV"] = "1"
        os.environ["OMMX_ROTATE_SEED"] = str(a.rotate_seed)

    # provenance: one grep-able line per run so every arm log carries its own env capture
    import json as _json, socket, subprocess, datetime
    try:
        _sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        _sha = ""
    import transformers as _tf
    print("[ENV_JSON] " + _json.dumps({
        "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname().split(".")[0], "git_sha": _sha, "model": a.model,
        "tag": a.tag, "mode": a.mode, "group_size": a.group_size, "nsamples": a.nsamples,
        "ppl_seqlen": a.ppl_seqlen, "kv": bool(a.kv), "kv_prefill": a.kv_prefill, "torch": torch.__version__,
        "transformers": _tf.__version__, "seed": 0,
        "rotate_w": bool(a.rotate_w), "rotate_kv": bool(a.rotate_kv), "rotate_seed": a.rotate_seed,
        "sparse_env": {k: v for k, v in os.environ.items()
                       if k.startswith(("OMMX_SPARSE_", "OMMX_KV_OUTLIER_"))},
    }, sort_keys=True))

    device = torch.device("cuda")
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from quant_weight import apply_weight_quant, Quant_Linear
    from calibrate import get_calib_dataset

    # AUTO bf16 for wide models: fp16 (max 65504) overflows in the layer-2 SwiGLU intermediate on wide
    # hidden (Qwen3-32B/Phi-4/EXAONE-4.5 = 5120) -> short-context (arc_challenge) logits go NaN -> acc
    # collapses to ~random (argmax(NaN)=idx0). bf16 (range ~3e38) removes the overflow with negligible
    # INT2/FP4 precision loss (confirmed: 32B arc 22->47.5, nan 154->0, PPL 8.60->8.23). Narrow models
    # (<5120) keep fp16 (no overflow; preserves the existing matrix). OMMX_NO_AUTO_BF16=1 disables.
    if not a.baseline and not a.calib_bf16 and not os.environ.get("OMMX_NO_AUTO_BF16"):
        try:
            _cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
            _hid = getattr(_cfg, "hidden_size", 0) or getattr(_cfg, "text_config", None) and getattr(_cfg.text_config, "hidden_size", 0)
            if (_hid or 0) >= 5120:
                a.calib_bf16 = True
                print(f"[INFO] auto bf16 (hidden={_hid}>=5120): fp16-overflow guard for wide models")
        except Exception as e:
            print(f"[WARN] auto-bf16 config probe failed ({e}); staying fp16")

    projs = [s for s in a.special_projs.split(",") if s.strip()]
    print("=" * 90)
    print(f" wq_eval {a.tag} | MODE={a.mode.upper()}{'+SEQ' if a.sequential else '+ONESHOT'} bits={a.bits} "
          f"gs={a.group_size} o%={a.outlier_percent} "
          f"special(first={a.special_first_k},last={a.special_last_k},projs={projs})")
    print(f" calib: data={a.calib_data} nsamples={a.nsamples} seqlen={a.seqlen} percdamp={a.percdamp} "
          f"blocksize={a.blocksize}" + (f" alpha={a.alpha}" if a.mode == "full" else "")
          + (f" POW2-scale(search=±{a.scale_search})" if a.pow2 else " fp16-scale(relaxed)")
          + (f" opt-zp({a.zp_steps})" if a.opt_zp else "") + (" hess-aware" if a.hess_aware else ""))
    print("=" * 90)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=(torch.bfloat16 if (a.baseline or a.calib_bf16) else torch.float16),
        trust_remote_code=a.trust_remote_code, low_cpu_mem_usage=True).to("cpu")
    model.eval()
    print(f"[INFO] model loaded (CPU) {time.time()-t0:.0f}s")

    if a.baseline:
        # unquantized reference: skip apply_weight_quant + calib + bake, eval raw weights with the SAME lm-eval path
        model.to(device)
        cfg = f"BASELINE unquantized bf16  {a.model}"
        print(f"[INFO] BASELINE mode -> skipping quantization; evaluating raw bf16 model with same lm-eval harness")
        if a.eval_tasks:
            run_lm_eval(model, tokenizer, a, cfg)
        else:
            ppl = wikitext_ppl(model, tokenizer, a.ppl_seqlen, a.limit, device)
            print(f"\n[RESULT] {a.tag}  {cfg}  ->  bf16 wikitext PPL = {ppl:.4f}")
        return

    if a.kv:
        # W+KV: apply KV-cache quantization FIRST (attention -> Quant_Attention, projections stay nn.Linear),
        # then weight quant below replaces those projections with Quant_Linear. The weight bake then runs
        # through the KV-quantized attention -> KV-aware weight calibration (mirrors the serving order).
        from quantizer_module import apply_kv_cache_quant
        # prefill KV recipe is selectable for fair comparison; decode stays OMMX INT2+FP4 either way.
        # bf16 => prefill_key/value_bits=16 -> _quantize_single_kv returns prefill KV unquantized.
        _pf_bits = 16 if a.kv_prefill == "bf16" else 2
        print(f"[INFO] W+KV mode -> KV quant before weight bake (decode INT2+FP4; prefill={a.kv_prefill})")
        apply_kv_cache_quant(
            model, bits=2, key_bits=2, value_bits=2,
            decode_key_bits=2, decode_value_bits=2,
            prefill_key_bits=_pf_bits, prefill_value_bits=_pf_bits,
            prefill_use_fp4=True, use_pow2=True,
            use_group_quant=True, group_size=a.kv_group_size,
            attention_sink_num=a.kv_sink,
            sliding_window_use_fp4_key=True, sliding_window_use_fp4_value=True,
            outlier_method="fp4", outlier_precision_key="fp4",
            outliers_per_group=a.kv_outliers_per_group,
            detect_outliers_value=False,
        )

    apply_weight_quant(model, bits=a.bits, group_size=a.group_size,
                       outlier_percent=a.outlier_percent, use_pow2=a.pow2,
                       special_first_k=a.special_first_k, special_last_k=a.special_last_k,
                       special_projs=projs)

    # rotate-w: fold H into every Quant_Linear BEFORE calib/bake so the GPTQ Hessian is
    # built on rotated inputs (pre-hook) and fasterquant quantises the rotated weight.
    # Output-invariant, so any PPL/ARC delta vs no-rotate is the rotation x quant interaction.
    if a.rotate_w:
        from rotation import apply_weight_rotation
        apply_weight_rotation(model, Quant_Linear, seed=a.rotate_seed)

    layers = model.model.layers
    embed_tokens = model.model.embed_tokens
    rotary_emb = getattr(model.model, "rotary_emb", None)
    input_correct = (a.mode == "full")

    calib = get_calib_dataset(tokenizer, nsamples=a.nsamples, seqlen=a.seqlen, dataset_name=a.calib_data)

    # fakesparse (opt-in): install offline W keep-masks + KV sparse config BEFORE the
    # weight bake, so the baked/cached q_weights are sparse and the prefill bake forwards
    # see the prefill KV masks. Off (no --sparse / OMMX_SPARSE_ENABLE unset) => identical
    # to the fakequant baseline. The Wanda/SparseGPT calib uses a bounded full-model forward
    # on a few calib samples (the model is moved to device for this then back to CPU).
    if a.sparse or os.environ.get("OMMX_SPARSE_ENABLE", "0") not in ("0", "", "false", "off", "no"):
        from fakesparse import apply_fakesparse, resolve_sparse_config
        scfg = resolve_sparse_config()
        scfg.enable = True
        scfg.validate()
        n_calib = min(16, calib.shape[0])

        def _run_calib():
            model.to(device)
            with torch.no_grad():
                for i in range(n_calib):
                    model(calib[i:i + 1].to(device))
            model.to("cpu")
            torch.cuda.empty_cache()

        run_calib = None if scfg.w_method == "magnitude" else _run_calib
        summary = apply_fakesparse(model, scfg, run_calib=run_calib)
        print(f"[FAKESPARSE] {summary}")

    # initial layer inputs from the embedding (quantized-propagated stream `cur`)
    embed_tokens.to(device)
    cur = []
    with torch.no_grad():
        for i in range(calib.shape[0]):
            cur.append(embed_tokens(calib[i:i + 1].to(device)).cpu())
    embed_tokens.to("cpu")
    torch.cuda.empty_cache()
    cur_fp = [c.clone() for c in cur] if input_correct else None     # clean stream (input-correction only)

    def run_layer_fp16ref(layer, stream, hooks_factory=None):
        """Run all calib samples through `layer` in fp16_ref (clean weights). Optionally install
        per-Quant_Linear forward hooks. Returns the layer outputs (clean propagation)."""
        for m in layer.modules():
            if isinstance(m, Quant_Linear):
                m.cache["mode"] = "fp16_ref"
        handles = []
        if hooks_factory is not None:
            for m in layer.modules():
                if isinstance(m, Quant_Linear):
                    h = hooks_factory(m)
                    if h is not None:
                        handles.append(m.register_forward_hook(h))
        outs = []
        with torch.no_grad():
            for j in range(len(stream)):
                hs = stream[j].to(device)
                pid = torch.arange(hs.shape[1], device=device).unsqueeze(0)
                kw = {"position_ids": pid}
                if rotary_emb is not None:
                    kw["position_embeddings"] = rotary_emb(hs, pid)
                out = layer(hs, **kw)
                outs.append((out[0] if isinstance(out, tuple) else out).cpu())
                del hs
        for h in handles:
            h.remove()
        for m in layer.modules():
            if isinstance(m, Quant_Linear):
                m.cache["mode"] = None
        return outs

    def forward_stream(layer, stream, hooks_factory=None, write_back=False):
        """Run all samples through `layer` using the CURRENT per-module modes (caller manages them;
        used by the sequential path where some linears are baked=decode and others are fp16_ref). Optional
        hooks; if write_back, overwrite stream[j] with the layer outputs."""
        handles = []
        if hooks_factory is not None:
            for m in layer.modules():
                if isinstance(m, Quant_Linear):
                    h = hooks_factory(m)
                    if h is not None:
                        handles.append(m.register_forward_hook(h))
        with torch.no_grad():
            for j in range(len(stream)):
                hs = stream[j].to(device)
                pid = torch.arange(hs.shape[1], device=device).unsqueeze(0)
                kw = {"position_ids": pid}
                if rotary_emb is not None:
                    kw["position_embeddings"] = rotary_emb(hs, pid)
                out = layer(hs, **kw)
                o = (out[0] if isinstance(out, tuple) else out).cpu()
                if write_back:
                    stream[j] = o
                del hs
        for h in handles:
            h.remove()

    def bake_module(name, m, quantizers):
        """Bake one Quant_Linear into INT2 (or special FP4), discard its fp16 weight."""
        if m.is_special_layer:
            q = m._get_special_layer_decode_q_weight().detach()
            kind = "spec"
            # A special layer is PURE FP4 -- a different number system from the INT2+FP4
            # bundle format, with no per-group INT2 codes to record. Exporting it as a
            # calibration decision would claim the packer can reproduce something it
            # cannot, so it is reported instead (see the export summary).
            if a.export_decisions:
                _EXPORT_SKIPPED.append(_FULL_NAME.get(id(m), name))
        else:
            if a.export_decisions:
                quantizers[name].record_decisions = True
            q = quantizers[name].fasterquant(m.weight.data, blocksize=a.blocksize, percdamp=a.percdamp)
            if a.export_decisions:
                _export_decisions(a.export_decisions, _FULL_NAME.get(id(m), name),
                                  quantizers[name].decisions)
            quantizers[name].free()
            kind = "int2"
        m.cache = {"prefill_q_weight": q, "decode_q_weight": q, "mode": "decode"}
        m.calibrating = False
        m.weight.data = torch.zeros((1, 1), device=device, dtype=torch.float16)
        return kind

    # Module names inside the layer loop are LAYER-RELATIVE ("self_attn.q_proj"), so they
    # repeat across all 32 layers. w_packer.CalibrationSource looks a module up by its
    # CHECKPOINT name, and exporting under the relative name silently overwrote every layer
    # into 7 files (measured: "exported 224 modules" but 7 on disk, each holding the last
    # layer's decisions). Resolve the full name by module identity instead.
    _FULL_NAME = {id(mod): nm for nm, mod in model.named_modules()}

    if a.export_decisions and not a.pow2:
        # Checked BEFORE the solve, not after: a calibration run is expensive and its
        # output would be unpackable. The format stores a power-of-two group scale.
        raise SystemExit(
            "--export-decisions requires --pow2: the OMMX weight format stores ONE "
            "exponent byte per group, so the full-precision scales the solver picks "
            "without --pow2 cannot be packed without discarding the calibration.")
    tcal = time.time()
    n_int2, n_spec = 0, 0
    _LAYER_OFFLOAD = bool(os.environ.get("OMMX_LAYER_OFFLOAD"))  # bound GPU mem for big MoE solve
    for li, layer in enumerate(layers):
        layer.to(device)
        if rotary_emb is not None:
            rotary_emb.to(device)

        # identify the INT2 (calibratable) linears in this layer and make an accumulator each
        quantizers = {}
        for name, m in layer.named_modules():
            if isinstance(m, Quant_Linear) and not m.is_special_layer:
                quantizers[name] = OMMXWeightQuantizer(m.out_features, m.in_features, a.group_size,
                                       a.outlier_percent, a.bits, device, input_correct=input_correct, alpha=a.alpha,
                                       use_pow2=a.pow2, scale_search=a.scale_search,
                                       opt_zp=a.opt_zp, zp_steps=a.zp_steps, hess_aware=a.hess_aware)

        # (input-correction) clean pass: advance cur_fp AND record each INT2 linear's clean input as fp_inp
        if input_correct:
            name_by_mod = {m: n for n, m in layer.named_modules() if isinstance(m, Quant_Linear)}
            def clean_factory(m):
                nm = name_by_mod.get(m)
                if nm not in quantizers:
                    return None
                g = quantizers[nm]
                def hook(mod, inp, out):
                    x = inp[0].detach()
                    x = x.reshape(-1, x.shape[-1]) if x.dim() == 3 else x
                    g.fp_inp.append(x.cpu().half())
                return hook
            cur_fp = run_layer_fp16ref(layer, cur_fp, clean_factory)

        name_by_mod = {m: n for n, m in layer.named_modules() if isinstance(m, Quant_Linear)}
        if not a.sequential:
            # ---- ONE-SHOT: one statistics pass over all INT2 linears (clean intra-layer), then bake all ----
            def h_factory(m):
                nm = name_by_mod.get(m)
                if nm not in quantizers:
                    return None
                g = quantizers[nm]
                def hook(mod, inp, out):
                    g.add_batch(inp[0].detach())
                return hook
            run_layer_fp16ref(layer, cur, h_factory)
            for name, m in layer.named_modules():
                if isinstance(m, Quant_Linear):
                    kind = bake_module(name, m, quantizers)
                    n_int2 += (kind == "int2"); n_spec += (kind == "spec")
        else:
            # ---- SEQUENTIAL: quantize groups in order; each group's statistics are collected with the
            #      EARLIER groups already quantized (real intra-layer input shift). fp_inp for the
            #      correction term was cached in the clean pass above (all-clean weights). ----
            for m in layer.modules():
                if isinstance(m, Quant_Linear):
                    m.cache["mode"] = "fp16_ref"            # all clean until baked
            named = dict(layer.named_modules())
            for group in SEQ_GROUPS:
                gnames = [n for n, m in named.items()
                          if isinstance(m, Quant_Linear) and any(n.endswith(s) for s in group)]
                int2_here = [n for n in gnames if n in quantizers]
                if int2_here:                                # collect this group's stats with earlier groups quantized
                    def h_factory(m, _sel=set(int2_here)):
                        nm = name_by_mod.get(m)
                        if nm not in _sel:
                            return None
                        g = quantizers[nm]
                        def hook(mod, inp, out):
                            g.add_batch(inp[0].detach())
                        return hook
                    forward_stream(layer, cur, h_factory)
                for n in gnames:                             # bake -> these become quantized for later groups
                    kind = bake_module(n, named[n], quantizers)
                    n_int2 += (kind == "int2"); n_spec += (kind == "spec")
        if os.environ.get("OMMX_DIAG_MOE"):
            # MoE root-cause: 128 experts x top-8 over a narrow calib set -> some experts get few/0
            # tokens -> singular Hessian -> fasterquant dead-channel path zeros the whole weight
            # (W[:,dead]=0). Count under-sampled experts per layer to confirm.
            ens = {n: int(quantizers[n].nsamples) for n in quantizers if "expert" in n}
            if ens:
                vs = sorted(ens.values())
                dead = sum(v == 0 for v in vs); rare = sum(0 < v < 512 for v in vs)
                print(f"[DIAG-MOE] L{li} experts={len(ens)} dead0={dead} rare<512={rare} "
                      f"min={vs[0]} med={vs[len(vs)//2]} max={vs[-1]}")
        del quantizers
        gc.collect(); torch.cuda.empty_cache()

        # propagate the working stream through the now-quantized layer -> realistic next inputs
        with torch.no_grad():
            for j in range(len(cur)):
                hs = cur[j].to(device)
                pid = torch.arange(hs.shape[1], device=device).unsqueeze(0)
                kw = {"position_ids": pid}
                if rotary_emb is not None:
                    kw["position_embeddings"] = rotary_emb(hs, pid)
                out = layer(hs, **kw)
                cur[j] = (out[0] if isinstance(out, tuple) else out).cpu()
                del hs
        gc.collect(); torch.cuda.empty_cache()
        # OMMX_LAYER_OFFLOAD: solve loop leaves each baked layer resident on GPU (cumulative) -> big
        # MoE (30B, 48 layers) OOMs mid-solve on a single GPU. Offload the just-baked layer back to CPU
        # to bound peak GPU memory; eval then dispatches across GPUs. unset => bit-identical old path.
        if _LAYER_OFFLOAD:
            layer.to("cpu"); gc.collect(); torch.cuda.empty_cache()
        if (li + 1) % 4 == 0 or li == len(layers) - 1:
            print(f"[INFO] {a.mode} solved+baked layer {li+1}/{len(layers)}  ({time.time()-tcal:.0f}s)")

    if _LAYER_OFFLOAD:
        # baked layers are on CPU; spread the model across visible GPUs (accelerate) instead of a
        # single model.to(device) that would re-OOM a 30B on one card. Hooks route eval activations.
        from accelerate import dispatch_model, infer_auto_device_map
        _nosplit = getattr(model, "_no_split_modules", None)
        dm = infer_auto_device_map(model, no_split_module_classes=list(_nosplit) if _nosplit else None)
        dispatch_model(model, device_map=dm)
    else:
        model.to(device)
    gc.collect(); torch.cuda.empty_cache()
    # fused-MoE experts are 3D Parameters (Qwen3MoeExperts.gate_up_proj/down_proj), NOT nn.Linear,
    # so apply_weight_quant skips them -> a 30B MoE would be ~90% unquantized (only attention INT2).
    # Quantize each expert slice here with the same INT2+FP4 recipe. Auto-on when fused experts are
    # present; OMMX_SKIP_MOE_EXPERTS=1 leaves them BF16 (attention-only baseline for A/B).
    if not os.environ.get("OMMX_SKIP_MOE_EXPERTS"):
        from quant_weight import fakequant_fused_experts
        _msp = os.environ.get("OMMX_MOE_EXPERT_NM", "")
        _snm = tuple(int(t) for t in _msp.split(":")) if (_msp and ":" in _msp) else None
        fakequant_fused_experts(model, bits=a.bits, group_size=a.group_size,
                                outlier_percent=a.outlier_percent, use_pow2=a.pow2, sparse_nm=_snm)
        gc.collect(); torch.cuda.empty_cache()
    print(f"[INFO] baked: {n_int2} INT2 layers, {n_spec} special(FP4) layers")
    if a.export_decisions:
        # A partial export is the dangerous outcome: w_packer refuses a directory that
        # does not cover every quantized tensor, but the operator should learn WHY here,
        # where the reason (special layers are pure FP4, a different number system) is
        # still in view.
        _written = len([f for f in os.listdir(a.export_decisions)
                        if f.endswith(".safetensors")]) if os.path.isdir(a.export_decisions) else 0
        print(f"[INFO] exported decisions: {_written} file(s) on disk for {n_int2} "
              f"calibrated module(s) -> {a.export_decisions}")
        if _written != n_int2:
            # Counting the bake calls instead of the files is how a name collision hid:
            # 224 modules were "exported" into 7 files, each holding the last layer's.
            raise SystemExit(
                f"[FATAL] wrote {_written} decision file(s) for {n_int2} calibrated "
                f"module(s). Names must be unique per module (full checkpoint path); a "
                f"collision means later layers overwrote earlier ones and the export is "
                f"unusable.")
        if _EXPORT_SKIPPED:
            print(f"[WARN] {len(_EXPORT_SKIPPED)} special(FP4) layer(s) were NOT exported "
                  f"(pure FP4 has no per-group INT2 codes to record): "
                  f"{_EXPORT_SKIPPED[:4]}{' ...' if len(_EXPORT_SKIPPED) > 4 else ''}")
            print("[WARN] `w_packer pack --calibrated` will REFUSE this directory until "
                  "those layers are covered. Re-run with --special-first-k 0 "
                  "--special-last-k 0 --special-projs '' to calibrate every layer as "
                  "INT2, or pack that subset separately.")

    # ARC NaN RCA (nan-trace: layers.2.mlp.down_proj in_maxabs=inf): the layer-2 SwiGLU intermediate
    # overflows fp16 (max 65504) on short arc sequences -> inf -> down_proj -> NaN; mmlu/wikitext (long,
    # RMSNorm-averaged, smaller activations) stay finite. Two mechanisms, distinguished here:
    #   (a) baked-cache inf: FP4 range-map scale ~1e9 -> fp16 inf, bake gate checks only isnan -> scan finds it.
    #   (b) genuine fp16 activation overflow: OMMX_EVAL_BF16 casts caches+model to bf16 (range ~3e38) -> fix.
    if os.environ.get("OMMX_DIAG_INFWEIGHT"):
        n_inf_w = 0; worst = []
        for nm, m in model.named_modules():
            if isinstance(m, Quant_Linear):
                for k in ("decode_q_weight", "prefill_q_weight"):
                    t = m.cache.get(k)
                    if isinstance(t, torch.Tensor):
                        ni = int(torch.isinf(t).sum())
                        if ni: n_inf_w += ni; worst.append((nm, ni))
        print(f"[DIAG-INFWEIGHT] baked caches with inf: total={n_inf_w} modules={len(worst)} first={worst[:5]}")
    if os.environ.get("OMMX_EVAL_BF16"):
        # memory-efficient: cast each quant cache in place and free the fp16 immediately (the dict held the
        # only ref) with periodic empty_cache, so peak stays ~1x caches (not 2x) -> no OOM on a 32B.
        nc = 0
        qmods = [m for m in model.modules() if isinstance(m, Quant_Linear)]
        for i, m in enumerate(qmods):
            for k in list(m.cache.keys()):
                t = m.cache.get(k)
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    m.cache[k] = t.to(torch.bfloat16); nc += 1
            if i % 16 == 0:
                torch.cuda.empty_cache()
        model.to(torch.bfloat16)                              # non-quant params (norms/embed/lm_head)
        gc.collect(); torch.cuda.empty_cache()
        print(f"[INFO] EVAL_BF16: cast {nc} quant caches + model -> bf16 (fp16-overflow guard)")

    cfg = (f"mode={a.mode}{'+seq' if a.sequential else '+oneshot'}{'+KV' if a.kv else ''} bits={a.bits} gs={a.group_size} "
           f"o%={a.outlier_percent} pow2={a.pow2}{('(s'+str(a.scale_search)+(',zp'+str(a.zp_steps) if a.opt_zp else '')+')') if a.pow2 else ''} "
           f"alpha={a.alpha} special(f{a.special_first_k}/l{a.special_last_k}/{projs})")

    # always report PPL first (cheap), then run benchmarks if requested
    ppl = wikitext_ppl(model, tokenizer, a.ppl_seqlen, a.limit, device)
    print(f"\n[RESULT] {a.tag}  {cfg}  ->  {'W+KV ' if a.kv else ''}INT2+FP4 wikitext PPL = {ppl:.4f}")
    if a.decode_ppl:
        dppl = decode_ppl(model, tokenizer, a.decode_pf, a.decode_len, a.limit, device)
        print(f"[RESULT-DECODE] {a.tag}  (prefill={a.decode_pf} sparse+fp4KV, decode={a.decode_len} "
              f"OMMX INT2+FP4)  ->  decode-path PPL = {dppl:.4f}")

    # fakesparse E0: dump per-layer score_K / e_o (OMMX_SPARSE_SCORE_OUT). The probe caps at
    # OMMX_SPARSE_SCORE_MAX prefill forwards per layer, so it accumulates on the calib-bake
    # forwards above (which run first), NOT on the wikitext PPL prefills.
    score_out = os.environ.get("OMMX_SPARSE_SCORE_OUT", "")
    if score_out and os.environ.get("OMMX_SPARSE_PROFILE_SCORE", "") not in ("", "0"):
        import json as _json
        prof = {}
        for m in model.modules():
            acc = getattr(m, "_score_k_acc", None)
            li = getattr(m, "layer_idx", None)
            if acc and li is not None:
                prof[int(li)] = {"score_K": sum(acc) / len(acc),
                                 "e_o": sum(m._eo_acc) / len(m._eo_acc), "n": len(acc)}
        with open(score_out, "w") as f:
            _json.dump({"profile": prof}, f, indent=2)  # build_alloc_from_sens reads "profile"
        n0 = prof[min(prof)]["n"] if prof else 0
        print(f"[E0-SCORE] wrote {len(prof)} layers (n={n0}) -> {score_out}")

    # K-outlier budget stats (OMMX_KV_OUTLIER_STATS_OUT): per-layer/per-head top-|K| mass
    # fractions on 128-token windows, accumulated by Quant_Attention._capture_k_outlier_stats
    # on the calib-bake prefill forwards. Consumed by fakesparse/build_outlier_profile.py.
    kout_out = os.environ.get("OMMX_KV_OUTLIER_STATS_OUT", "")
    if kout_out:
        import json as _json
        kstats = {}
        for m in model.modules():
            acc = getattr(m, "_kout_stats", None)
            li = getattr(m, "layer_idx", None)
            if acc and li is not None:
                t = torch.stack(acc).mean(0)                      # [Hkv, 4]
                kstats[int(li)] = {
                    "n": len(acc),
                    "mass_topk_per_head": [[round(float(v), 6) for v in row] for row in t],
                    "mass_topk": [round(float(v), 6) for v in t.mean(0)]}
        with open(kout_out, "w") as f:
            _json.dump({"window": 128, "sink_excluded": True, "model": a.model,
                        "stats": kstats}, f, indent=2)
        n0 = kstats[min(kstats)]["n"] if kstats else 0
        print(f"[KOUT-STATS] wrote {len(kstats)} layers (n={n0}) -> {kout_out}")

    if a.eval_tasks:
        run_lm_eval(model, tokenizer, a, cfg)


def run_lm_eval(model, tokenizer, a, cfg):
    """Run lm-evaluation-harness on the already-baked INT2+FP4 model (in memory). The Quant_Linear caches
    make every forward use the INT2+FP4 weights (prefill cache := decode cache was set during baking), so
    this measures the true decode-path INT2 quantization -- consistent with the PPL numbers and sensitive
    to the outlier ratio."""
    import json
    # prefer the repo's editable lm-eval (matching task configs) if present
    repo_eval = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval")
    if os.path.isdir(repo_eval):
        sys.path.insert(0, repo_eval)
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    tasks = [t for t in a.eval_tasks.split(",") if t.strip()]
    bs = a.eval_batch_size
    try:
        bs = int(bs)
    except (ValueError, TypeError):
        pass
    print(f"[INFO] lm-eval (lm_eval {lm_eval.__version__}) tasks={tasks} fewshot={a.num_fewshot} "
          f"batch_size={bs} limit={a.eval_limit or 'all'}")
    model.eval()
    model.config.use_cache = True
    # max_gen_toks default (32 in this lm-eval build) truncates GSM8K CoT (~150-256 tok) mid-reasoning
    # -> no "#### <ans>" -> strict-match=0 universal (RCA: lm-eval task bug, not 2-bit collapse). 256 = task default.
    # softmax_dtype=None -> log_softmax runs in native fp16; large quant logits (e.g. Qwen3-32B W-quant)
    # OVERFLOW in fp16 on some tasks (ARC-C) -> all option logliks tie -> argmax collapses to index 0
    # (acc==acc_norm==266/1172 = frac of docs w/ gold=idx0, NOT quality). fp32 softmax fixes the tie.
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bs, max_gen_toks=256,
              softmax_dtype="float32")
    if os.environ.get("OMMX_DIAG_NAN"):
        # localize the FIRST module whose output is NaN while its input is finite = the NaN SOURCE
        # (arc multi-token forward on 32B W-quant goes all-NaN; mmlu single-token is finite).
        import torch as _t
        _seen = []
        def _mk(nm):
            def _h(mod, inp, out):
                if len(_seen) >= 12:
                    return
                o = out[0] if isinstance(out, (tuple, list)) else out
                if not (isinstance(o, _t.Tensor) and _t.isnan(o).any()):
                    return
                ii = inp[0] if inp else None
                in_nan = bool(_t.isnan(ii).any()) if isinstance(ii, _t.Tensor) else None
                in_max = float(ii.abs().amax()) if (isinstance(ii, _t.Tensor) and not in_nan) else -1
                o_fin = o[~_t.isnan(o)]
                o_max = float(o_fin.abs().amax()) if o_fin.numel() else -1
                _seen.append((nm, type(mod).__name__, in_nan, round(in_max, 1), round(o_max, 1)))
            return _h
        for nm, mod in model.named_modules():
            mod.register_forward_hook(_mk(nm))
        import atexit as _ax
        def _dump():
            print("[NAN-TRACE] first modules emitting NaN (in_nan=False => SOURCE):")
            for r in _seen[:12]:
                print(f"  {r[0]} ({r[1]}) in_nan={r[2]} in_maxabs={r[3]} out_finite_maxabs={r[4]}")
        _ax.register(_dump)
        print("[DIAG] nan-trace hooks installed")
    if os.environ.get("OMMX_DIAG_LOGLIK"):
        # decisive tie-locator: dump per-request loglik + logit nan/inf/maxabs. Compare a
        # BROKEN task (arc_challenge) vs a WORKING one (mmlu) under the SAME W-quant model.
        # tied logliks across a doc's options => degenerate; nan/inf logits => dequant overflow.
        import torch as _t
        _st = {"n": 0, "nan": 0, "inf": 0, "maxabs": 0.0}
        _oc = lm._model_call
        def _dc(inps, *aa, **kw):
            out = _oc(inps, *aa, **kw)
            lg = out[0] if isinstance(out, (tuple, list)) else out
            if _st["n"] < 400:
                _st["n"] += 1
                if _t.isnan(lg).any(): _st["nan"] += 1
                if _t.isinf(lg).any(): _st["inf"] += 1
                _st["maxabs"] = max(_st["maxabs"], float(lg.abs().max()))
            return out
        lm._model_call = _dc
        _oll = lm._loglikelihood_tokens
        def _dll(requests, *aa, **kw):
            r = _oll(requests, *aa, **kw)
            for i, ((ll, g), rq) in enumerate(zip(r, requests)):
                if i >= 48: break
                clen = len(rq[2]) if len(rq) > 2 else -1
                print(f"[DIAG-LL] req{i:03d} loglik={ll:11.4f} greedy={int(g)} clen={clen}")
            return r
        lm._loglikelihood_tokens = _dll
        import atexit as _ax
        _ax.register(lambda: print(f"[DIAG-LOGIT] calls={_st['n']} nan={_st['nan']} inf={_st['inf']} maxabs={_st['maxabs']:.2f}"))
        print("[DIAG] loglik+logit diagnostic hooks installed")
    # --num-fewshot default=0 overrides each task YAML's own num_fewshot (gsm8k=5) -> model never
    # sees the "#### N" convention -> strict-match=0 universal. Pass None to honor the task's own
    # num_fewshot unless the user EXPLICITLY set --num-fewshot (RCA: fewshot bug, not chat-template).
    _nfs = a.num_fewshot if ("--num-fewshot" in sys.argv or "--num_fewshot" in sys.argv) else None
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=_nfs,
                                  limit=(a.eval_limit or None), bootstrap_iters=0)
    results = res["results"]
    print("\n" + "=" * 90)
    for task in sorted(results.keys()):
        m = results[task]
        acc = m.get("acc,none"); accn = m.get("acc_norm,none")
        em = m.get("exact_match,none")
        parts = []
        if acc is not None:  parts.append(f"acc={acc*100:.2f}")
        if accn is not None: parts.append(f"acc_norm={accn*100:.2f}")
        if em is not None:   parts.append(f"exact_match={em*100:.2f}")
        extra = " ".join(f"{k}={v:.4f}" for k, v in m.items()
                         if k not in ("acc,none", "acc_norm,none", "exact_match,none", "alias")
                         and isinstance(v, float)) if not parts else ""
        print(f"[EVAL-RESULT] {a.tag}  {task}  {' '.join(parts) or extra}  ({cfg})")
    print("=" * 90)
    # persist full results
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", f"eval_{a.tag}.json")
    try:
        with open(out, "w") as f:
            json.dump({"config": cfg, "tag": a.tag, "results": results}, f, indent=2, default=str)
        print(f"[INFO] full results -> {out}")
    except Exception as e:
        print(f"[WARN] could not save results json: {e}")


if __name__ == "__main__":
    main()
