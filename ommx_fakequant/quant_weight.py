import torch
import torch.nn.functional as F
from typing import Optional
import math
import os

def round_ste(x: torch.Tensor) -> torch.Tensor:
    """
    Straight-Through Estimator for rounding.
    Forward: round(x), Backward: identity (gradient = 1)
    """
    return (x.round() - x).detach() + x

@torch.no_grad()
def apply_fp_range_mapped(
    data: torch.Tensor,
    fp_bits: int = 2, # 2 for FP4, 4 for FP8
    padding_mask: Optional[torch.Tensor] = None,
    external_scale: Optional[torch.Tensor] = None,
    external_center: Optional[torch.Tensor] = None,
    debug_mode: bool = False
) -> torch.Tensor:
    """
    Min/Max Based on FP Quantification (Range Mapping Method)
    - Data FP Expression Scope(FP4: [-6, 6], FP8: [-448, 448])Recycling to precision polarization
    - Shared Mode(INT + FP): Map INT integer space([0, qmax]) to FP space to secure precision
    """
    if padding_mask is None:
        padding_mask = torch.ones_like(data, dtype=torch.bool)
        
    device = data.device
    dtype = data.dtype
    
    # Set FP Expressionable Value
    if fp_bits == 2:
        # 16 representations of F4 E2M1 (S1 E2 M1)
        fp_values = torch.tensor([
            -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        ], device=device, dtype=dtype)
        fp_range = 12.0
    elif fp_bits == 4:
        # 256 possible values for F8 E4M3FN (S1 E4 M3)
        # Max value: 448.0, Min non-zero: 2^-9 * (1 + 0/8) = 0.001953125
        # Here, instead of creating LUT directly, we're going to build it based on the math range 448.0.
        # Apply the same mapping method as FP4 to keep your load 100%
        # FP8 E4M3FN values: +/- [0.00195, ..., 448.0]
        # To avoid overhead of 256 comparisons, we can use a more efficient way or just the LUT.
        # User requested logic 100% same as FP4, so we follow the LUT search approach.
        
        # simplified E4M3 logic to generate values
        def get_fp8_e4m3_values():
            vals = []
            for s in [1, -1]:
                for e in range(16): # 4 bits exponent
                    for m in range(8): # 3 bits mantissa
                        if e == 0: # subnormal
                            val = s * (2**-6) * (m/8)
                        elif e == 15 and m == 7: # NaN
                            continue
                        else:
                            val = s * (2**(e-7)) * (1 + m/8)
                        vals.append(val)
            vals.append(0.0)
            return sorted(list(set(vals)))
        
        fp8_list = get_fp8_e4m3_values()
        fp_values = torch.tensor(fp8_list, device=device, dtype=dtype)
        fp_range = 896.0 # -448 to 448
    else:
        raise ValueError(f"Unsupported fp_bits: {fp_bits}")
    
    fp_center = 0.0
    
    # 1. Min/ Max and Mapping Parameters Compute
    if external_scale is None or external_center is None:
        # Independent mode: Map the actual range of data into the FP range
        large_val = torch.tensor(1e9, device=device, dtype=dtype)
        small_val = torch.tensor(-1e9, device=device, dtype=dtype)
        
        temp_min = torch.where(padding_mask, data, large_val)
        min_vals = torch.min(temp_min, dim=-1, keepdim=True)[0]
        
        temp_max = torch.where(padding_mask, data, small_val)
        max_vals = torch.max(temp_max, dim=-1, keepdim=True)[0]

        # OMMX_W_SYMMETRIC=1: drop the weight zero-point (symmetric ±max range).
        # W is ~zero-centered so this is free/better (C10: fp4 asym 7.237 -> sym 7.223,
        # -0.20%) and saves the zp in format/HW. KV stays asymmetric (see quant_function).
        if os.environ.get('OMMX_W_SYMMETRIC', '0') == '1':
            _mabs = torch.maximum(min_vals.abs(), max_vals.abs())
            min_vals = -_mabs
            max_vals = _mabs

        data_range = (max_vals - min_vals).clamp_min(1e-8)
        data_center = (max_vals + min_vals) / 2.0
        
        mapping_scale = fp_range / data_range
        mapping_center = data_center
    else:
        # Shared mode: Matching the 'Indices' range of users to FP range
        # external_scale: 1 / shared_scale
        # external_center: shared_min_vals
        
        # 1) Convert the data to an index space over [0, qmix]
        indices = (data - external_center) * external_scale
        
        # 2) Actual range measurement of indexes (the whole range that includes the Autolier)
        large_val = torch.tensor(1e9, device=device, dtype=dtype)
        small_val = torch.tensor(-1e9, device=device, dtype=dtype)
        temp_min = torch.where(padding_mask, indices, large_val)
        idx_min = temp_min.min(dim=-1, keepdim=True)[0]
        temp_max = torch.where(padding_mask, indices, small_val)
        idx_max = temp_max.max(dim=-1, keepdim=True)[0]
        
        idx_range = (idx_max - idx_min).clamp_min(1e-8)
        idx_center = (idx_max + idx_min) / 2.0
        
        # 3) [idx_min, Idx_max] - > FP Map parameter
        mapping_scale = fp_range / idx_range
        mapping_center = idx_center
        
        data = indices # Quantification in Index Space

    # Convert data to FP range
    shifted_data = (data - mapping_center) * mapping_scale + fp_center
    
    # 3. Find the nearest FP value
    best_dist = torch.full_like(shifted_data, float('inf'))
    fp_quantized = torch.zeros_like(shifted_data)
    
    for val in fp_values:
        dist = torch.abs(shifted_data - val)
        mask = dist < best_dist
        best_dist = torch.where(mask, dist, best_dist)
        fp_quantized = torch.where(mask, val, fp_quantized)
    
    # 4. Convert (Dequantize)
    if external_scale is None or external_center is None:
        # Independent Mode Reversion (Prepil)
        quantized = (fp_quantized - fp_center) / mapping_scale + mapping_center
        orig_data = data
    else:
        # Shared mode transformation: FP range - > [idx_min, ix_max] -> Original Space (Decode)
        restored_indices = (fp_quantized - fp_center) / mapping_scale + mapping_center
        quantized = restored_indices / external_scale + external_center
        orig_data = data / external_scale + external_center
    
    # 5. Restore padding
    result = torch.where(padding_mask, quantized, orig_data)
    
    return result

def weight_quantizer(
    weight: torch.Tensor,
    bits: int = 2,
    group_size: int = 16,
    outlier_group_size: Optional[int] = None,
    outlier_percent: float = 0.01,
    act_scales: Optional[torch.Tensor] = None,
    use_pow2: bool = False,
    mode: str = "decode", 
    v: Optional[torch.Tensor] = None,
    min_scale: Optional[torch.Tensor] = None,
    max_scale: Optional[torch.Tensor] = None,
    training: bool = False,
    debug_mode: bool = False,
    init_scale: Optional[torch.Tensor] = None, # New: provide pre-searched scale
    sparse_keep_mask: Optional[torch.Tensor] = None,  # fakesparse: [O,I] bool, True=keep / False=prune
) -> torch.Tensor:
    """
    Weight Quantizer
    - 'prefill' Mode: Total weights Per-group FP (range-mapped)Rotation
    - 'decode' Mode: Base INT + Outlier FP (range-mapped)
    - bits: 2 (INT2 + FP4) or 4 (INT4 + FP8)
    - outlier_group_size: OutlierUnit for Extracting (NonePages group_sizeSame as)
    """
    fp_bits = bits # FP4 if bits=2, FP8 if bits=4
    orig_device = weight.device
    orig_dtype = weight.dtype
    
    if bits == 16:
        return weight

    fp_bits = bits # FP4 if bits=2, FP8 if bits=4
    O, I = weight.shape
    
    if outlier_group_size is None:
        outlier_group_size = group_size
        
    # 1. Padding (Max common multiple approach for simplicity)
    common_size = math.lcm(group_size, outlier_group_size)
    num_common_groups = (I + common_size - 1) // common_size
    pad_len = num_common_groups * common_size - I
    
    if pad_len > 0:
        padded_weight = F.pad(weight, (0, pad_len))
    else:
        padded_weight = weight

    # fakesparse (prune-before-quant): zero pruned weights up front so they are not
    # selected as outliers and contribute 0 to surviving groups; range-exclusion +
    # final zeroing below keep them out of the quant min/max and force exact 0.0.
    if sparse_keep_mask is not None:
        keep_bool = sparse_keep_mask.to(torch.bool)
        keep_padded = F.pad(keep_bool, (0, pad_len), value=True) if pad_len > 0 else keep_bool
        padded_weight = padded_weight * keep_padded.to(padded_weight.dtype)
    else:
        keep_padded = None

    if mode == "prefill":
        # Preferl maintains existing group_ize
        grouped_weight = padded_weight.view(O, -1, group_size)
        padding_mask = torch.ones_like(grouped_weight, dtype=torch.bool)
        if pad_len > 0:
            padding_mask[:, -1, (group_size - (pad_len % group_size if pad_len % group_size != 0 else group_size)):] = False
        if keep_padded is not None:
            padding_mask = padding_mask & keep_padded.view(O, -1, group_size)
        dq_weight = apply_fp_range_mapped(grouped_weight, fp_bits=fp_bits, padding_mask=padding_mask, debug_mode=debug_mode)
    else:
        # Decode style: INT+Otliers
        # 2. Overviewing (utlier_gruup_size based)
        if outlier_percent > 0:
            k = max(1, int(outlier_group_size * outlier_percent))
            
            # [LOLG] Output number of outline numbers (for experimental confirmation)
            if not hasattr(weight_quantizer, "_logged_k"):
                print(f"[INFO] Outlier extraction: {k} outliers per group of {outlier_group_size} (percent: {outlier_percent*100:.1f}%)")
                weight_quantizer._logged_k = True
            
            # Graping for Outlier
            outlier_grouped_weight = padded_weight.view(O, -1, outlier_group_size)
            
            if act_scales is not None:
                if act_scales.dim() == 1:
                    act_scales = act_scales.view(1, -1)
                
                if pad_len > 0:
                    padded_act = F.pad(act_scales, (0, pad_len), value=1.0)
                else:
                    padded_act = act_scales
                
                grouped_act = padded_act.view(1, -1, outlier_group_size)

                # Magnitude-based saliency: abs(W) * act_scales
                saliency = outlier_grouped_weight.abs() * grouped_act
                topk_indices = torch.topk(saliency, k, dim=-1)[1]
            else:
                abs_weight = outlier_grouped_weight.abs()
                topk_indices = torch.topk(abs_weight, k, dim=-1)[1]
                
            outlier_mask = torch.zeros_like(outlier_grouped_weight, dtype=torch.bool)
            outlier_mask.scatter_(-1, topk_indices, True)
            # Expand Again
            outlier_mask = outlier_mask.view(O, -1)
        else:
            outlier_mask = torch.zeros(padded_weight.shape, dtype=torch.bool, device=orig_device)

        # 3. Quanta (group_size based)
        grouped_weight = padded_weight.view(O, -1, group_size)
        outlier_mask_grouped = outlier_mask.view(O, -1, group_size)
        
        padding_mask = torch.ones_like(grouped_weight, dtype=torch.bool)
        if pad_len > 0:
            padding_mask[:, -1, (group_size - (pad_len % group_size if pad_len % group_size != 0 else group_size)):] = False

        quantization_mask = (~outlier_mask_grouped) & padding_mask
        if keep_padded is not None:
            # exclude pruned lanes from the INT min/max range (they were zeroed above
            # and are not outliers since their saliency is 0).
            quantization_mask = quantization_mask & keep_padded.view(O, -1, group_size)

        large_val = torch.tensor(1e9, device=orig_device, dtype=orig_dtype)
        small_val = torch.tensor(-1e9, device=orig_device, dtype=orig_dtype)
        
        temp_min = torch.where(quantization_mask, grouped_weight, large_val)
        min_vals = temp_min.min(dim=-1, keepdim=True)[0]
        temp_max = torch.where(quantization_mask, grouped_weight, small_val)
        max_vals = temp_max.max(dim=-1, keepdim=True)[0]
        
        min_vals = torch.where(min_vals > 1e8, torch.zeros_like(min_vals), min_vals)
        max_vals = torch.where(max_vals < -1e8, torch.zeros_like(max_vals), max_vals)
        
        qmin, qmax = 0, (1 << bits) - 1
        # itf4 ternary lever (zero-kernel): OMMX_N_LEVELS overrides level count.
        # n_levels=4 -> qmax=3 (i2f4 INT2). n_levels=3 -> qmax=2 (itf4 ternary {0,1,2}).
        import os as _os
        _nl = _os.environ.get("OMMX_N_LEVELS")
        if _nl is not None:
            qmax = max(1, int(_nl) - 1)

        # Learnable weight clipping (range shrink): shrink the per-group range by
        # learned factors in [0,1] (1.0 = no clip = original behavior). Tightens the INT2
        # grid onto the bulk of the distribution; extremes are handled by the FP4 outliers.
        if min_scale is not None:
            min_vals = min_vals * torch.clamp(min_scale, 0.0, 1.0)
        if max_scale is not None:
            max_vals = max_vals * torch.clamp(max_scale, 0.0, 1.0)

        if init_scale is not None:
            scale = init_scale
        else:
            data_range = (max_vals - min_vals).clamp_min(1e-8)
            scale = data_range / (qmax - qmin)
        
        if use_pow2:
            scale = torch.pow(2.0, torch.round(torch.log2(scale.clamp_min(1e-12))))
            
        if v is not None:
            # v should be the same shape as grouped_weight or broadcastable
            # v is added before rounding to adjust the rounding direction
            if training:
                # Calibration: STE allows gradient to flow to v
                int_w = round_ste((grouped_weight - min_vals) / scale + v).clamp(qmin, qmax)
            else:
                # Inference: standard round (no gradient needed)
                int_w = torch.round((grouped_weight - min_vals) / scale + v).clamp(qmin, qmax)
        else:
            int_w = torch.round((grouped_weight - min_vals) / scale).clamp(qmin, qmax)
            
        dq_weight = int_w * scale + min_vals
        
        if outlier_percent > 0:
            outlier_data = torch.where(outlier_mask_grouped, grouped_weight, torch.zeros_like(grouped_weight))
            # FP mapping: center = min_vals, mapping_scale = 1/scale, fp_center = 0.0
            refined_outliers = apply_fp_range_mapped(
                outlier_data, 
                fp_bits=fp_bits,
                padding_mask=outlier_mask_grouped, 
                external_scale=1.0/scale,
                external_center=min_vals,
                debug_mode=float(qmax) # Use to compute map scale with forwarding qmx information
            )
            dq_weight = torch.where(outlier_mask_grouped, refined_outliers, dq_weight)

    # fakesparse: force pruned lanes to exact 0.0 (prefill already restores 0 via the
    # padding_mask path; decode needs the explicit zero since INT dq is computed for all).
    if sparse_keep_mask is not None:
        keep_grp = keep_padded.view(O, -1, group_size)
        dq_weight = torch.where(keep_grp, dq_weight, torch.zeros_like(dq_weight))

    final_weight = dq_weight.view(O, -1)[:, :I]
    return final_weight.to(orig_dtype)

class Quant_Linear(torch.nn.Module):
    """
    Dynamic weight-quantization linear layer.
    - Prefill (L>1): FP path (independent mode)
    - Decode (L=1): INT + FP outlier path, with optional learnable-rounding optimization.
    """
    def __init__(self, original_linear, bits=2, group_size=16, outlier_group_size=None, outlier_percent=0.01, use_pow2=False, is_special_layer=False, debug_mode=False):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.bits = bits
        self.group_size = group_size
        self.outlier_group_size = outlier_group_size
        self.outlier_percent = outlier_percent
        self.use_pow2 = use_pow2
        self.is_special_layer = is_special_layer
        self.debug_mode = debug_mode
        
        # The original reference to weights
        self.weight = original_linear.weight 
        
        if original_linear.bias is not None:
            self.bias = original_linear.bias
        else:
            self.bias = None

        # Activation-saliency buffer (per input channel)
        self.register_buffer("act_scales", torch.zeros(self.in_features, device=self.weight.device, dtype=self.weight.dtype))

        # fakesparse: optional offline weight keep-mask ([O,I] bool, True=keep). None => dense.
        # Non-persistent: recomputed offline each run (see fakesparse.weight.prepare_weight_sparsity).
        self.register_buffer("sparse_keep_mask", None, persistent=False)

        # Optional learnable rounding parameter v
        self.v = None
        
        # Since weight quantumization has a lot of computation, it's done in a mode.
        self.cache = {
            "prefill_q_weight": None,
            "decode_q_weight": None,
            "mode": None  # For subcompatibility and special modes (f16_ref)
        }
        self.update_count = 0
        
        # Calibration Mode: Trie, force decod Int2 path and always apply V
        self.calibrating = False

    def set_sparse_mask(self, mask):
        """Install (or clear with None) the fakesparse offline keep-mask [O,I] bool."""
        if mask is None:
            self.sparse_keep_mask = None
            return
        expected = (self.out_features, self.in_features)
        if tuple(mask.shape) != expected:
            raise ValueError(f"sparse mask shape {tuple(mask.shape)} != weight {expected}")
        self.sparse_keep_mask = mask.to(self.weight.device, dtype=torch.bool)

    def _get_special_layer_decode_q_weight(self):
        """Ground Truth FP Range-Mapped logic for special layer decode mode."""
        with torch.no_grad():
            O, I = self.weight.shape
            group_size = self.group_size if self.group_size > 0 else 32
            fp_bits = self.bits
            
            # 1. Padding
            pad_len = (group_size - (I % group_size)) % group_size
            if pad_len > 0:
                padded_weight = torch.nn.functional.pad(self.weight, (0, pad_len), "constant", 0)
            else:
                padded_weight = self.weight
            
            # 2. Grouping
            num_groups = padded_weight.shape[1] // group_size
            grouped_weight = padded_weight.view(O, num_groups, group_size)
            
            # 3. Mask for padding
            padding_mask = torch.ones_like(grouped_weight, dtype=torch.bool)
            if pad_len > 0:
                padding_mask[:, -1, (group_size - (pad_len % group_size if pad_len % group_size != 0 else group_size)):] = False
            
            # 4. Apply FP
            dq_weight = apply_fp_range_mapped(
                grouped_weight,
                fp_bits=fp_bits,
                padding_mask=padding_mask,
                debug_mode=self.debug_mode
            )
            return dq_weight.view(O, -1)[:, :I].to(self.weight.dtype)

    def set_act_scales(self, x):
        """Record the scale of the input activation value (for calibration)"""
        with torch.no_grad():
            if x.dim() == 3:
                cur_scales = x.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                cur_scales = x.abs().max(dim=0)[0]
                
            if self.act_scales is None:
                self.act_scales = cur_scales.to(self.weight.device).to(self.weight.dtype)
            else:
                # Keep max for better outlier detection stability
                self.act_scales = torch.max(self.act_scales, cur_scales.to(self.weight.device).to(self.weight.dtype))
    
    def init_v(self, learn_scales=False):
        """Create V, min_scale, max_scale parameters for calibration (called only during training)"""
        ogsize = self.outlier_group_size if self.outlier_group_size else self.group_size
        common_size = math.lcm(self.group_size, ogsize)
        padded_in = ((self.in_features + common_size - 1) // common_size) * common_size
        num_groups = padded_in // self.group_size
        
        # V: per-element rounding adjustment
        self.v = torch.nn.Parameter(torch.zeros(
            (self.out_features, num_groups, self.group_size),
            device=self.weight.device, dtype=torch.float32
        ))
        
        # Min_scale, math_cale: per-gruup range ringing (previous value 1.0 = no change in range)
        self.min_scale = torch.nn.Parameter(torch.ones(
            (self.out_features, num_groups, 1),
            device=self.weight.device, dtype=torch.float32
        ), requires_grad=learn_scales)
        self.max_scale = torch.nn.Parameter(torch.ones(
            (self.out_features, num_groups, 1),
            device=self.weight.device, dtype=torch.float32
        ), requires_grad=learn_scales)
        
        return self.v, self.min_scale, self.max_scale
    
    def apply_v_and_discard(self, v_data=None, min_scale_data=None, max_scale_data=None, act_scales_data=None):
        """V, min_scale, max_scale, act_scalesSo let's do that. decode weightSo let's do that in advance., Remove From Memory"""
        v = v_data if v_data is not None else (self.v.data if self.v is not None else None)
        ms = min_scale_data if min_scale_data is not None else (self.min_scale.data if hasattr(self, 'min_scale') and self.min_scale is not None else None)
        xs = max_scale_data if max_scale_data is not None else (self.max_scale.data if hasattr(self, 'max_scale') and self.max_scale is not None else None)
        
        # Update to buffer if an action_scales_data is available
        if act_scales_data is not None:
            self.act_scales.copy_(act_scales_data.to(self.act_scales.device))

        if v is None and ms is None and xs is None:
            return  # With nothing, Leaton.
        
        if self.is_special_layer:
            # Special layers use full FP in decode mode (Ground Truth).
            q_weight = self._get_special_layer_decode_q_weight()
            self.cache["decode_q_weight"] = q_weight.detach()
            self.cache["mode"] = "decode"
            self.v = None
            self.min_scale = None
            self.max_scale = None
            return
        with torch.no_grad():
            q_weight = weight_quantizer(
                self.weight,
                bits=self.bits,
                group_size=self.group_size,
                outlier_group_size=self.outlier_group_size,
                outlier_percent=self.outlier_percent,
                act_scales=self.act_scales,
                use_pow2=self.use_pow2,
                mode="decode",
                v=v,
                min_scale=ms,
                max_scale=xs,
                training=False,
                debug_mode=self.debug_mode,
                init_scale=getattr(self, 'best_init_scale', None),
                sparse_keep_mask=self.sparse_keep_mask,
            )
        self.cache["decode_q_weight"] = q_weight.detach()
        self.cache["mode"] = "decode"
        
        # Descending memory (preservation of VRAM)
        self.v = None
        self.min_scale = None
        self.max_scale = None

    def freeze_and_discard_weight(self, prefill=True, decode=True):
        """Pre-calculate quantized weights and discard original weights to save VRAM."""
        with torch.no_grad():
            if self.is_special_layer:
                # Special modules (Layer 0, 31):
                # Prefill = BF16 (Reference), Decode = FP4 (Full Precision FP4)
                if prefill and self.cache["prefill_q_weight"] is None:
                    self.cache["prefill_q_weight"] = self.weight.clone().detach()
                
                if decode and self.cache["decode_q_weight"] is None:
                    # Ground Truth FP4 logic for special layer decode
                    self.cache["decode_q_weight"] = self._get_special_layer_decode_q_weight().detach()
            else:
                # General modules:
                # Prefill = FP4, Decode = INT2 + FP4 Outliers
                if prefill and self.cache["prefill_q_weight"] is None:
                    self.cache["prefill_q_weight"] = weight_quantizer(
                        self.weight, bits=self.bits, group_size=self.group_size,
                        outlier_group_size=self.outlier_group_size, outlier_percent=self.outlier_percent,
                        act_scales=self.act_scales, use_pow2=self.use_pow2, mode="prefill",
                        init_scale=getattr(self, 'best_init_scale', None),
                        sparse_keep_mask=self.sparse_keep_mask,
                    ).detach()
                
                if decode and self.cache["decode_q_weight"] is None:
                    # Calculate decode cache (INT2 + Outliers) using V if calibrated
                    self.cache["decode_q_weight"] = weight_quantizer(
                        self.weight, bits=self.bits, group_size=self.group_size,
                        outlier_group_size=self.outlier_group_size, outlier_percent=self.outlier_percent,
                        act_scales=self.act_scales, use_pow2=self.use_pow2, mode="decode",
                        init_scale=getattr(self, 'best_init_scale', None),
                        sparse_keep_mask=self.sparse_keep_mask,
                    ).detach()

        # Replace original weight with a dummy to save 16GB
        # We use a 1x1 tensor on the same device to keep references valid
        dev = self.weight.device
        dtype = self.weight.dtype
        # Better: just set it to a tiny tensor but keep it as a Parameter if needed
        # Or just delete the data
        if hasattr(self.weight, 'data'):
            self.weight.data = torch.zeros((1, 1), device=dev, dtype=dtype)
        
        # Disable dynamic updates
        self.calibrating = False
        self.act_scales.requires_grad = False

    def forward(self, input):
        # 1. Accession Calls Update (Moving Access)
        # Update only when weight is alive (pass in focus mode)
        if self.weight.numel() > 1:
            with torch.no_grad():
                # Revert to mean activation across batch and sequence dimensions
                # input: [Batch, Seq, Dim] -> mean(dim=(0, 1)) -> [Dim]
                current_scales = input.detach().abs().mean(dim=(0, 1)).view(-1)
                
                if torch.all(self.act_scales == 0):
                    self.act_scales.copy_(current_scales)
                else:
                    # Smoothing: Moving Average with 0.9 momentum (original logic)
                    self.act_scales.copy_(0.9 * self.act_scales + 0.1 * current_scales)

        # Frell vs Decod Panel
        is_prefill = input.shape[1] > 1

        # C3 quality probe (env-gated, default-OFF): emulate serving OMMX_W_ACT_LANE=fp8
        # — per-token-rowmax e4m3 on the prefill activation INTO the quantized linear
        # (matches integration/vllm/linear_method.py, FP8_E4M3_MAX=448). Measures the
        # fp8-act quality cost on top of the i2f4 weight quant (true served numerics).
        if is_prefill and self.weight.numel() > 1 and \
                os.environ.get("OMMX_FAKE_ACT_FP8", "").strip() not in ("", "0", "off"):
            _s = input.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 448.0
            input = (input / _s).to(torch.float8_e4m3fn).to(input.dtype) * _s

        # [Calibration Patch] FP16 reference mode
        if self.cache["mode"] == "fp16_ref":
            return F.linear(input, self.weight, self.bias)

        # If it's in Calivation mode (f16_ref), return original BF16 (for data extraction)
        if self.cache.get("mode") == "fp16_ref":
            return F.linear(input, self.weight, self.bias)

        if self.calibrating:
            # Phase: Optimization loop in calibrate_block
            # - Special layers are NOT optimized (remain FP4 -> must match runtime distribution)
            # - General layers are quantized (mode="decode") to optimize V parameters
            if self.is_special_layer:
                q_weight = self._get_special_layer_decode_q_weight()
                return F.linear(input, q_weight, self.bias)
                
            q_weight = weight_quantizer(
                self.weight,
                bits=self.bits,
                group_size=self.group_size,
                outlier_group_size=self.outlier_group_size,
                outlier_percent=self.outlier_percent,
                act_scales=self.act_scales,
                use_pow2=self.use_pow2,
                mode="decode", # Force decode mode to optimize INT+Outliers
                v=self.v,
                min_scale=self.min_scale if hasattr(self, 'min_scale') else None,
                max_scale=self.max_scale if hasattr(self, 'max_scale') else None,
                training=True,
                debug_mode=self.debug_mode,
                sparse_keep_mask=self.sparse_keep_mask,
            )
            return F.linear(input, q_weight, self.bias)

        # Check the cache
        if is_prefill:
            if self.cache["prefill_q_weight"] is not None:
                return F.linear(input, self.cache["prefill_q_weight"], self.bias)
        else:
            if self.cache["decode_q_weight"] is not None:
                return F.linear(input, self.cache["decode_q_weight"], self.bias)

        # 2. Dynamicly qualitative quantumization (Casi micro-time)
        if self.bits == 16:
            return F.linear(input, self.weight, self.bias)

        mode = "prefill" if is_prefill else "decode"
        if self.is_special_layer:
            # Special layers use FP4 in decode and FP16/FP4 in prefill (Ground Truth).
            if mode == "prefill":
                q_weight = self.weight
            else:
                # Decode: Manual FP4 logic
                q_weight = self._get_special_layer_decode_q_weight()
        else:
            # Normal layer: INT + FP OTIERS
            if self.weight.numel() <= 1:
                raise RuntimeError(f"Weight is discarded but cache is missing for mode {mode}. Did you freeze correctly?")
                
            q_weight = weight_quantizer(
                self.weight,
                bits=self.bits,
                group_size=self.group_size,
                outlier_group_size=self.outlier_group_size,
                outlier_percent=self.outlier_percent,
                act_scales=self.act_scales,
                use_pow2=self.use_pow2,
                mode=mode,
                v=None,
                training=False,
                debug_mode=self.debug_mode,
                init_scale=getattr(self, 'best_init_scale', None),
                sparse_keep_mask=self.sparse_keep_mask,
            )

        # Results Cating
        q_weight_tensor = q_weight.detach() if isinstance(q_weight, torch.nn.Parameter) else q_weight
        if is_prefill:
            self.cache["prefill_q_weight"] = q_weight_tensor
        else:
            self.cache["decode_q_weight"] = q_weight_tensor
        
        self.cache["mode"] = mode
        self.update_count += 1
        
        return F.linear(input, q_weight, self.bias)


# Fused-MoE experts (Qwen3MoeExperts / newer transformers) store every expert as a slice of a 3D
# Parameter (gate_up_proj [E, 2*inter, hidden], down_proj [E, hidden, inter]), NOT as nn.Linear.
# apply_weight_quant only targets nn.Linear, so it silently skips ALL experts (the bulk of a MoE's
# params) -> only attention gets INT2 (PPL looks fine but the model is barely quantized). This
# quantizes each expert slice in place with the SAME INT2+FP4 recipe (decode mode = the eval path).
# RTN / weight-only (no per-expert calibration) -- first fully-quantized milestone; the router
# (a 2D Parameter named "weight" on Qwen3MoeTopKRouter) is left BF16 (never a *_proj, so untouched).
_FUSED_EXPERT_PROJS = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")

@torch.no_grad()
def fakequant_fused_experts(model, bits=2, group_size=128, outlier_percent=0.01,
                            use_pow2=False, sparse_nm=None, debug_mode=False):
    """In-place INT2+FP4 fakequant of 3D fused-MoE expert weights. Returns #expert-slices quantized.
    sparse_nm=(n,m) additionally applies structured N:M (fakesparse) on the input axis per expert."""
    n_q = 0
    for mod_name, module in model.named_modules():
        for pname in _FUSED_EXPERT_PROJS:
            p = getattr(module, pname, None)
            if not (isinstance(p, torch.nn.Parameter) and p.dim() == 3):
                continue
            E, O, I = p.shape
            w2d = p.data.reshape(E * O, I).float()          # per-row group-quant along I (contraction)
            keep = None
            if sparse_nm is not None:
                from fakesparse import mask as _M
                keep = _M.nm_keep_mask(w2d.abs(), sparse_nm[0], sparse_nm[1], axis=-1)
            dq = weight_quantizer(w2d, bits=bits, group_size=group_size,
                                  outlier_percent=outlier_percent, use_pow2=use_pow2,
                                  mode="decode", sparse_keep_mask=keep)
            p.data.copy_(dq.reshape(E, O, I).to(p.dtype))
            n_q += E
            if debug_mode:
                print(f"[DEBUG] fused-expert quant {mod_name}.{pname} [{E},{O},{I}]")
    if n_q:
        tag = f"INT2+FP4 RTN{'+N:M'+str(sparse_nm) if sparse_nm else ''}"
        print(f"[INFO] fakequant fused-MoE experts: {n_q} expert-slices ({tag}, mode=decode)")
    return n_q


def apply_weight_quant(model, bits=2, group_size=64, outlier_group_size=None, outlier_percent=0.01, use_pow2=False, mode="dynamic", v_params_path=None, debug_mode=False,
                       special_first_k=1, special_last_k=1, special_projs=None):
    """
    The model. Linear I'm gonna need a layer. Quant_LinearReplacing.
    VIf you go,: VApplyed decode weightCount in Advance → V Decod memory (Add VRAM = 0)

    Sensitive-layer policy (additive; defaults reproduce the original layer0+last behavior):
      special_first_k : first N decoder layers kept at higher precision (FP4/FP16 via is_special)
      special_last_k  : last  N decoder layers kept at higher precision
      special_projs   : list of projection-name substrings (e.g. ['down_proj']) kept special everywhere
    """
    if special_projs is None:
        special_projs = []
    v_params = None
    if v_params_path and os.path.exists(v_params_path):
        print(f"[INFO] Loading calibrated parameters from {v_params_path}")
        v_params = torch.load(v_params_path, map_location="cpu")

    count = 0
    # Named_modules() will round the target to be replaced
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            targets.append((name, module))

    # Lama chain layer extract (mel.layers. N....)
    import re
    layer_indices = []
    for name, _ in targets:
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            layer_indices.append(int(match.group(1)))
        else:
            layer_indices.append(-1)
            
    max_layer_idx = max(layer_indices) if layer_indices else -1

    v_applied_count = 0
    for (name, module), l_idx in zip(targets, layer_indices):
        # Find Paracent
        name_parts = name.split('.')
        parent = model
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        
        # The lm_had is not from quantumization.
        is_lm_head = "lm_head" in name
        if is_lm_head:
            if debug_mode:
                print(f"[DEBUG] Skipping {name} (lm_head, kept in FP16)")
            continue

        # MoE router / gate: tiny Linear (hidden -> num_experts) whose argmax selects experts.
        # 2-bit-quantizing it corrupts routing -> wrong experts every token -> garbage (PPL 2.8M on
        # qwen3_moe). Keep BF16 (AWQ/GPTQ/vLLM convention: modules_to_not_convert=["gate"]). Match the
        # router leaf == "gate" (NOT gate_proj) or tiny out_features (<=256 >> proj dims). Dense: no-op.
        leaf = name.split(".")[-1]
        is_router = (leaf in ("gate", "router", "wg")) or (getattr(module, "out_features", 1 << 30) <= 256)
        if is_router:
            if debug_mode:
                print(f"[DEBUG] Skipping {name} (MoE router, kept in BF16)")
            continue


        is_first = (l_idx >= 0 and l_idx < special_first_k)
        is_last = (l_idx >= 0 and l_idx > max_layer_idx - special_last_k)
        is_special_proj = any(p in name for p in special_projs)
        is_special = is_first or is_last or is_special_proj
        
        # Replace with Quanta_Linear
        new_module = Quant_Linear(
            module, 
            bits=bits, 
            group_size=group_size, 
            outlier_group_size=outlier_group_size,
            outlier_percent=outlier_percent, 
            use_pow2=use_pow2,
            is_special_layer=is_special,
        )
        
        # Apply calibration parameter and immediately unmask memory
        if v_params and name in v_params:
            entry = v_params[name]
            dev = new_module.weight.device
            
            # New Format vsS format (tensor compatible)
            if isinstance(entry, dict):
                v_data = entry['v'].to(dev)
                ms_data = entry.get('min_scale')
                xs_data = entry.get('max_scale')
                as_data = entry.get('act_scales')
                ms_data = ms_data.to(dev) if ms_data is not None else None
                xs_data = xs_data.to(dev) if xs_data is not None else None
                as_data = as_data.to(dev) if as_data is not None else None
                
                new_module.apply_v_and_discard(
                    v_data=v_data, 
                    min_scale_data=ms_data, 
                    max_scale_data=xs_data,
                    act_scales_data=as_data
                )
                del v_data, ms_data, xs_data, as_data
                v_applied_count += 1
            
        setattr(parent, name_parts[-1], new_module)
        count += 1
    
    if v_params:
        print(f"[INFO] Calibration params applied. Proactively freezing and discarding weights...")
        for name, m in model.named_modules():
            if isinstance(m, Quant_Linear):
                # Prefill and Decode caches are both computed and original weight is discarded
                m.freeze_and_discard_weight(prefill=True, decode=True)
                
        print(f"[INFO] Calibration params applied and weights discarded (extra VRAM = 0)")
        del v_params
        import gc; gc.collect()
        torch.cuda.empty_cache()
        
    return count
