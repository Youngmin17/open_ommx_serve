# quant_function.py
import torch
import os
from typing import Optional
import time
import torch.nn.functional as F


#----------------------------------------------------------
# FP4 Quantization Functions
#----------------------------------------------------------

# Create cache dixnery at top file or module level
_LOG_FP4_CACHE = {}


def fp16_to_fp4_e2m1(inp: torch.Tensor, debug_mode: bool = False) -> torch.Tensor:
    device = inp.device
    dtype = inp.dtype
    cache_key = (device, dtype)
    
    # Reuse fixed FP4 values and log values one time
    if cache_key not in _LOG_FP4_CACHE:
        fp4_vals = torch.tensor([
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        ], device=device, dtype=dtype)
        # Preset log to 1e-8 clamp
        log_fp4_vals = torch.log(fp4_vals.clamp_min(1e-8))
        _LOG_FP4_CACHE[cache_key] = (fp4_vals, log_fp4_vals)
        
    fp4_values, log_fp4_values = _LOG_FP4_CACHE[cache_key]
    
    # Parsing special value (isinf, isnan with a single bit mask reduced by 1 tenor allocation)
    special_mask = ~inp.isfinite()
    
    sign = torch.sign(inp)
    abs_inp = torch.abs(inp)
    
    # From here on, it's the same as the existing code.
    abs_inp_expanded = abs_inp.unsqueeze(-1)
    # Enable cached log_fp4_values
    log_fp4_values_expanded = log_fp4_values.unsqueeze(0).expand(*abs_inp.shape, -1)
    
    log_abs_inp = torch.log(abs_inp_expanded.clamp_min(1e-8))
    distances = torch.abs(log_abs_inp - log_fp4_values_expanded)
    
    closest_indices = torch.argmin(distances, dim=-1)
    result = sign * fp4_values[closest_indices]
    result.masked_fill_(inp == 0.0, 0.0)
    # Restore special value ( Overwrite index instead of place)
    if special_mask.any():
        # Keep original sign and copy
        result[special_mask] = inp[special_mask]
        
    return result

_FP4_GROUP_VALUES_CACHE = {}

def _fp4_group_quant_core(
    grouped_view: torch.Tensor,
    padding_mask: torch.Tensor,
    fp4_values: torch.Tensor,
) -> torch.Tensor:
    """torch.compileThis applied key operation function.
    dict Access, print, if Particles and so on graph break Remove All Inflammatory Factors.
    torch.compile accelerates unsqueeze(-1) -> subtract -> abs -> argmin
    It's a single kernel. 16x Removed memory extension.
    """
    # 1. Min/ Max (using amin/ amax speed function for C++ backend)
    # dtype-safe sentinels (fp16/bf16): fill the non-selected lanes with the dtype's max/min so they
    # are ignored by amin/amax (a raw 1e9 overflows fp16). Matches the finfo usage elsewhere in this file.
    min_vals = torch.where(padding_mask, grouped_view, torch.finfo(grouped_view.dtype).max).amin(dim=-1, keepdim=True)
    max_vals = torch.where(padding_mask, grouped_view, torch.finfo(grouped_view.dtype).min).amax(dim=-1, keepdim=True)

    # OMMX_KV_SYMMETRIC=1: force symmetric (zp-free) KV range. Diagnostic only — KV is
    # skewed/outlier-heavy so asymmetric is essential (C10: prefill fp4 +8%, decode INT2
    # halves coqa f1); the default keeps the zero-point.
    if os.environ.get('OMMX_KV_SYMMETRIC', '0') == '1':
        _mabs = torch.maximum(min_vals.abs(), max_vals.abs())
        min_vals = -_mabs; max_vals = _mabs
    # 2. Compute data range
    data_range = (max_vals - min_vals).clamp_min_(1e-8)
    data_center = (max_vals + min_vals) * 0.5

    # Compute 3.Scale (FP4 range = 12.0)
    scale = 12.0 / data_range

    # Convert to 4.F4 Range
    shifted_data = (grouped_view - data_center).mul_(scale)

    # 5. Find the nearest FP4.
    distances = torch.abs(shifted_data.unsqueeze(-1) - fp4_values)
    closest_indices = torch.argmin(distances, dim=-1)

    # 6. Convert
    quantized = fp4_values[closest_indices].div_(scale).add_(data_center)

    # Restore 7. padding
    quantized = torch.where(padding_mask, quantized, grouped_view)

    return quantized

# Apply torch.compile: prevent memory allocation problems at the point of port with the lazi initialization
# Try compiled only on first call, and fallback as uncompressed function when failed
_fp4_group_quant_compiled = None

def _get_fp4_group_quant_fn():
    """Lazy initialization of torch.compile to avoid fork/memory issues at import time."""
    global _fp4_group_quant_compiled
    if _fp4_group_quant_compiled is None:
        try:
            _fp4_group_quant_compiled = torch.compile(
                _fp4_group_quant_core,
                mode="default",
                fullgraph=True,
            )
        except Exception as e:
            print(f"[WARN] torch.compile failed ({e}), falling back to uncompiled function")
            _fp4_group_quant_compiled = _fp4_group_quant_core
    return _fp4_group_quant_compiled

def _apply_fp4_group_quant(
    grouped_view: torch.Tensor,
    padding_mask: torch.Tensor,
    debug_mode: bool = False
) -> torch.Tensor:
    device = grouped_view.device
    dtype = grouped_view.dtype
    cache_key = (device, dtype)

    if cache_key not in _FP4_GROUP_VALUES_CACHE:
        _FP4_GROUP_VALUES_CACHE[cache_key] = torch.tensor([
            -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        ], device=device, dtype=dtype)
        
    fp4_values = _FP4_GROUP_VALUES_CACHE[cache_key]

    # Raisy-compiled key function call (JIT Compile Warming -30 seconds on first call)
    compiled_fn = _get_fp4_group_quant_fn()
    quantized = compiled_fn(grouped_view, padding_mask, fp4_values)

    if debug_mode:
        valid_mask = padding_mask
        mse = torch.mean((grouped_view[valid_mask] - quantized[valid_mask]) ** 2).item()
        print(f"[FP4 MinMax] MSE = {mse:.6f}")

    return quantized


@torch.no_grad()
def _apply_group_quantization_vectorized(
    inp: torch.Tensor,
    group_size: int,
    bits: int,
    use_pow2: bool,
    attention_sink_num: int,
    outlier_percent_topk: float,
    outlier_percent_bottomk: float,


    is_key: bool,
    debug_mode: bool,
    outlier_method: str = "original",
    outliers_per_group: int = 0,
    is_prefill: bool = False,
    use_fp4: bool = False,
    attention_sink_num_front: Optional[int] = None,  # None, use access_link_num. 0 without front link.
    sparse_keep_mask: Optional[torch.Tensor] = None,  # fakesparse: True=keep, False=prune; same shape as inp
    **kwargs
) -> torch.Tensor:
    """
    - Padding, min/max, topk, Set the last group processing and all the etching cases to match the existing code.
    - attention_sink_num_front: front sink Specify extra number of tokens (tail Quantification Time 0Cannot initialise Evolution's mail component.)
    - sparse_keep_mask: optional fakesparse keep-mask (prefill only). When given, pruned (False)
      lanes are excluded from the FP4 group min/max range AND restored to 0.0. None => identity.
    """
    # ---1. Early settings and early link handling (front/back individual support) ---
    front_sink = attention_sink_num if attention_sink_num_front is None else attention_sink_num_front
    back_sink = attention_sink_num
    has_front = front_sink > 0 and inp.shape[2] > front_sink
    has_back = back_sink > 0 and inp.shape[2] > back_sink

    if has_front or has_back:
        front_end = front_sink if has_front else 0
        sink_part_front = inp[:, :, :front_end, :] if has_front else None
        sink_part_back = inp[:, :, -back_sink:, :] if has_back else None
        if has_back:
            tensor_to_quant = inp[:, :, front_end:-back_sink, :].contiguous()
            sparse_mid = sparse_keep_mask[:, :, front_end:-back_sink, :] if sparse_keep_mask is not None else None
        else:
            tensor_to_quant = inp[:, :, front_end:, :].contiguous()
            sparse_mid = sparse_keep_mask[:, :, front_end:, :] if sparse_keep_mask is not None else None
    else:
        sink_part_front = None
        sink_part_back = None
        tensor_to_quant = inp
        sparse_mid = sparse_keep_mask

    if tensor_to_quant.numel() == 0:
        return inp

    # --- 2. is the same axis as before ---
    if is_key:
        permuted_tensor = tensor_to_quant.permute(0, 1, 3, 2)
    else:
        permuted_tensor = tensor_to_quant
    
    grouping_dim_len = permuted_tensor.shape[-1]
    
    if is_prefill:
        layer_id = kwargs.get('layer_id', -1)
        pad_len = (group_size - grouping_dim_len % group_size) % group_size
        padded_tensor = F.pad(permuted_tensor, [0, pad_len], "constant", 0)
        
        padding_mask = torch.ones_like(padded_tensor, dtype=torch.bool)
        if pad_len > 0:
            padding_mask.narrow(dim=-1, start=grouping_dim_len, length=pad_len).fill_(False)
            
        shape_for_groups = list(padded_tensor.shape[:-1]) + [-1, group_size]
        grouped_view = padded_tensor.reshape(*shape_for_groups)
        padding_mask_grouped = padding_mask.reshape(*shape_for_groups)

        # fakesparse: fold the keep-mask into the FP4 path. Pruned lanes are removed from
        # the group min/max (padding_mask=False) AND zeroed in grouped_view, so the FP4
        # restore (torch.where(padding_mask, quantized, grouped_view)) returns exactly 0.0.
        # NOTE on keys: we permute the mask with the data (the key recipe groups along the
        # token axis = per-channel). The mask is applied ELEMENT-WISE, so a pruned [seq,hd]
        # position is zeroed at exactly that position regardless of the FP4 group axis — the
        # permute must mirror the data, NOT be skipped (skipping it would regroup keys along
        # head_dim and corrupt the key quant format). The mask's intra-block N:M structure is
        # defined on the [seq,hd] plane (fakesparse.kv) and is intentionally independent of the
        # FP4 grouping axis. Verified by test_key_scattered_prune_maps_correctly.
        if sparse_mid is not None:
            sparse_permuted = sparse_mid.permute(0, 1, 3, 2) if is_key else sparse_mid
            sparse_padded = F.pad(sparse_permuted, [0, pad_len], "constant", True)
            sparse_grouped = sparse_padded.reshape(*shape_for_groups)
            padding_mask_grouped = padding_mask_grouped & sparse_grouped
            grouped_view = grouped_view * sparse_grouped.to(grouped_view.dtype)

        # Apply group FP4 quantumization
        quantized_data_grouped = _apply_fp4_group_quant(grouped_view, padding_mask_grouped)
        
        # Results combination and return
        restored_padded = quantized_data_grouped.reshape(*padded_tensor.shape)
        unpadded = restored_padded.narrow(dim=-1, start=0, length=grouping_dim_len)
        
        if is_key:
            final_quantized_part = unpadded.permute(0, 1, 3, 2)
        else:
            final_quantized_part = unpadded

        parts = []
        if sink_part_front is not None:
            parts.append(sink_part_front)
        parts.append(final_quantized_part)
        if sink_part_back is not None:
            parts.append(sink_part_back)
        result = torch.cat(parts, dim=2) if len(parts) > 1 else final_quantized_part
        
        return result.contiguous()
    
    # --- 3. padding and reshape ready--
    if outliers_per_group > 0 and group_size == 64:
        processing_unit_size = 128
    else:
        processing_unit_size = group_size

    pad_len = (processing_unit_size - grouping_dim_len % processing_unit_size) % processing_unit_size
    padded_tensor = F.pad(permuted_tensor, [0, pad_len], "constant", 0)
    
    # Create padding mask
    padding_mask = torch.ones_like(padded_tensor, dtype=torch.bool)
    if pad_len > 0:
        padding_mask.narrow(dim=-1, start=grouping_dim_len, length=pad_len).fill_(False)
    
    shape_for_logic = list(padded_tensor.shape[:-1]) + [-1, processing_unit_size]
    view_for_logic = padded_tensor.reshape(*shape_for_logic)
    padding_mask_reshaped = padding_mask.reshape(*shape_for_logic)
    
    # ---4, create a tyrannized Otlier mask (logic reinforcement) ---
    masked_view_for_op = torch.where(padding_mask_reshaped, view_for_logic, torch.finfo(inp.dtype).min)

    if outliers_per_group > 0:
        if group_size == 64:
            # pairwise reshape
            pairwise_view = view_for_logic.reshape(*shape_for_logic[:-2], -1, 128)
            pairwise_padding_mask = padding_mask_reshaped.reshape(*shape_for_logic[:-2], -1, 128)
            
            # Treat last group if they are not paired
            num_groups = (grouping_dim_len + group_size - 1) // group_size
            if num_groups % 2 == 1 and pairwise_view.numel() > 0:
                # The last pair except for the output detection.
                pairwise_padding_mask[:, :, :, -1, :].fill_(False)

            masked_pairwise_view = torch.where(pairwise_padding_mask, pairwise_view.abs(), torch.finfo(inp.dtype).min)
            k = min(outliers_per_group, pairwise_view.size(-1))
            if k == 1:
                max_indices = torch.argmax(masked_pairwise_view, dim=-1, keepdim=True)
                outlier_mask_128 = torch.zeros_like(pairwise_view, dtype=torch.bool).scatter_(-1, max_indices, True)
            else:
                topk_values, topk_indices = torch.topk(masked_pairwise_view, k, dim=-1)
                threshold = topk_values[..., -1:]
                outlier_mask_128 = (masked_pairwise_view >= threshold)
            
            outlier_mask = outlier_mask_128.reshape(*shape_for_logic[:-2], -1, 2, 64)
        elif group_size == 128:
            max_indices = torch.argmax(masked_view_for_op, dim=-1, keepdim=True)
            outlier_mask = torch.zeros_like(view_for_logic, dtype=torch.bool).scatter_(-1, max_indices, True)
        else: 
            outlier_mask = torch.zeros_like(view_for_logic, dtype=torch.bool)
    else:
        # Ignore padding when working top/ botttom
        masked_view_for_topk = torch.where(padding_mask_reshaped, view_for_logic, torch.finfo(inp.dtype).min)
        masked_view_for_bottomk = torch.where(padding_mask_reshaped, view_for_logic, torch.finfo(inp.dtype).max)
        outlier_mask = torch.zeros_like(view_for_logic, dtype=torch.bool)
        if outlier_percent_topk > 0:
            k = max(1, int(view_for_logic.size(-1) * outlier_percent_topk))
            if k < view_for_logic.size(-1):
                threshold = torch.topk(masked_view_for_topk, k, dim=-1).values[..., -1].unsqueeze(-1)
                outlier_mask |= (view_for_logic >= threshold)
        if outlier_percent_bottomk > 0:
            k = max(1, int(view_for_logic.size(-1) * outlier_percent_bottomk))
            if k < view_for_logic.size(-1):
                threshold = torch.topk(masked_view_for_bottomk, k, dim=-1, largest=False).values[..., -1].unsqueeze(-1)
                outlier_mask |= (view_for_logic <= threshold)

    # -5 .n. . . . . . . . . . . . .
    if outliers_per_group > 0 and group_size == 64:
        data_for_quant = view_for_logic.reshape(*shape_for_logic[:-2], -1, 2, 64)
        padding_mask_for_quant = padding_mask_reshaped.reshape(*shape_for_logic[:-2], -1, 2, 64)
        outlier_mask = outlier_mask.reshape(*shape_for_logic[:-2], -1, 2, 64)
    else:
        data_for_quant = view_for_logic
        padding_mask_for_quant = padding_mask_reshaped
    
    non_outlier_mask = ~outlier_mask
    quantization_mask = non_outlier_mask & padding_mask_for_quant

    # Calculation of Min/ Max (float16 overflow)
    masked_data = data_for_quant * quantization_mask.to(data_for_quant.dtype)
    inverted_mask_float = (~quantization_mask).float()
    temp_data_for_min = masked_data.float() + inverted_mask_float * 1e8
    temp_data_for_max = masked_data.float() - inverted_mask_float * 1e8
    min_vals = torch.min(temp_data_for_min, dim=-1, keepdim=True)[0]
    max_vals = torch.max(temp_data_for_max, dim=-1, keepdim=True)[0]
    min_vals.nan_to_num_(0.0)
    max_vals.nan_to_num_(0.0)

    # Returns an existing type for memory operation
    min_vals = min_vals.to(data_for_quant.dtype)
    max_vals = max_vals.to(data_for_quant.dtype)

    # OMMX_KV_SYMMETRIC=1: symmetric (zp-free) decode-INT path (diagnostic; see above).
    if os.environ.get('OMMX_KV_SYMMETRIC', '0') == '1':
        _mabs = torch.maximum(min_vals.abs(), max_vals.abs())
        min_vals = -_mabs; max_vals = _mabs
    # Cale calculation and quantumization
    qmin, qmax = 0, (1 << bits) - 1
    group_range = max_vals - min_vals
    scale = group_range / (qmax - qmin)
    scale.masked_fill_(group_range <= 1e-8, 1.0)
    if use_pow2:
        scale = torch.pow(2.0, torch.round(torch.log2(torch.clamp(scale, min=1e-12))))
    q_vals = torch.round((data_for_quant - min_vals) / scale).clamp(qmin, qmax)
    dq_vals = q_vals * scale + min_vals

    # ..."The final result combination: the original tenther took the value and guaranteed it would be logical and consistent.
    quantized_data_grouped = torch.where(quantization_mask, dq_vals, data_for_quant)

    # ================================================================
    # Outliner handling: only fp4 or half/oreginal/one support
    # ================================================================
    if outlier_method in ["half_precision", "half", "original", "none"]:
        # Adding jobs are unnecessary because they already contain the original value in quantized_data_Group
        pass

    elif outlier_method == "fp4":

        group_shape = data_for_quant.shape[:-2] if group_size == 64 and outliers_per_group > 0 else data_for_quant.shape[:-1]
        data_reshaped = data_for_quant.reshape(*group_shape, -1) # [..., num_groups, group_per_quant_size]
        mask_reshaped = outlier_mask.reshape(*group_shape, -1)
        
        data_reshaped_float = data_reshaped.float()
        masked_data_for_min = torch.where(mask_reshaped, data_reshaped_float, torch.tensor(1e8, device=data_reshaped.device, dtype=torch.float32))
        masked_data_for_max = torch.where(mask_reshaped, data_reshaped_float, torch.tensor(-1e8, device=data_reshaped.device, dtype=torch.float32))
        
        group_min = torch.min(masked_data_for_min, dim=-1, keepdim=True)[0]
        group_max = torch.max(masked_data_for_max, dim=-1, keepdim=True)[0]
        
        o_range = (group_max - group_min).clamp_min(1e-8)
        o_center = (group_max + group_min) / 2.0
        mapping_scale = 12.0 / o_range
        
        o_center = o_center.to(data_reshaped.dtype)
        mapping_scale = mapping_scale.to(data_reshaped.dtype)

        outlier_values_1d = data_reshaped[mask_reshaped]
        
        if outlier_values_1d.numel() > 0:
            center_1d = o_center.expand_as(data_reshaped)[mask_reshaped]
            scale_1d = mapping_scale.expand_as(data_reshaped)[mask_reshaped]
            mapped_outliers_1d = (outlier_values_1d - center_1d) * scale_1d
            
            quantized_fp4_1d = fp16_to_fp4_e2m1(mapped_outliers_1d)
            
            dq_outliers_1d = (quantized_fp4_1d / scale_1d) + center_1d
            
            quantized_data_grouped[outlier_mask] = dq_outliers_1d

    # --- 6.Share restores the source and returns the final----
    # Extract the quantum part and overwrite the original tener
    quantized_padded = quantized_data_grouped.reshape(*padded_tensor.shape)
    quantized_unpadded = quantized_padded.narrow(dim=-1, start=0, length=grouping_dim_len)
    
    if is_key:
        final_quantized_part = quantized_unpadded.permute(0, 1, 3, 2)
    else:
        final_quantized_part = quantized_unpadded

    parts = []
    if sink_part_front is not None:
        parts.append(sink_part_front)
    parts.append(final_quantized_part)
    if sink_part_back is not None:
        parts.append(sink_part_back)
    result = torch.cat(parts, dim=2) if len(parts) > 1 else final_quantized_part

    return result.contiguous()



def attention_quantizer(
    inp: torch.Tensor,
    *,
    bits: int = 2,
    use_pow2: bool = False,
    use_sliding_window: bool = False,
    window_size: int = 32,
    is_prefill: bool = True,
    attention_sink_num: int = 5,
    outlier_mode: str = "elementwise",
    outlier_percent_topk: float = 0.01,
    outlier_percent_bottomk: float = 0.0,
    use_a_shape: bool = False,
    decode_only: bool = False,
    is_key: bool = True,
    use_symmetric: bool = False,
    distance_ceiling: int = 32,
    outlier_shift: bool = True,
    debug_mode: bool = False,
    layer_id: int = -1,
    workspace: torch.Tensor = None,
    use_fp4: bool = False,
    # Group Quentization parameters
    use_group_quant: bool = True,
    group_size: int = 1024,
    # Sliding-window Settings for Bit/ Forts
    sliding_window_bits: Optional[int] = None,
    sliding_window_use_fp4: bool = False,
    sliding_window_use_fp4_key: bool = False,
    sliding_window_use_fp4_value: bool = False,
    # Configure Bit/ Forms Only for Link Area
    sink_bits: Optional[int] = None,
    sink_use_fp4: bool = False,
    # Force FP4 only for Prepil
    prefill_use_fp4: bool = True,
    # Decod FP4 Force Options
    decode_use_fp4_key: bool = False,
    decode_use_fp4_value: bool = False,
    # Outlier Detection/ Precision Control
    detect_outliers_key: bool = True,
    detect_outliers_value: bool = True,
    outlier_precision: Optional[str] = None,  # "half" | "fp4" | None (deprecated, use outlier_precision_key/value)
    outlier_precision_key: str = "int",
    outlier_precision_value: str = "int",
    # Group out layer policy (6454: 1 sum of 2 groups, 128:1 per group)
    outliers_per_group: int = 0,
    # Select the ideal processing method
    outlier_method: str = "original",  # "original", "int2_shared", "int2_individual", "fp4_no_extra", "fp4_with_extra", "half_precision", "int2_exp_scale"
    # Group catch optimization: front/ back link individual settings
    attention_sink_num_front: Optional[int] = None,  # None is the same as action_link_num. 0 is 0 and no front link is available (til quantumize)
    sparse_keep_mask: Optional[torch.Tensor] = None,  # fakesparse keep-mask (prefill only); None => identity
) -> torch.Tensor:
    """
    attention Quantification Functions (FP4 Add Support)
    
    Args:
        use_fp4: FP4 E2M1 Whether to use quantumization 
    """
    # Decod FP4 Force Decision
    decode_force_fp4 = False
    if not is_prefill:  # Only in Decode Mode
        if is_key and decode_use_fp4_key:
            decode_force_fp4 = True
        elif not is_key and decode_use_fp4_value:
            decode_force_fp4 = True

    if debug_mode and layer_id <= 1:
        kv_type = "Key" if is_key else "Value"
        print(f"[Layer {layer_id}] attention_quantizer: bits={bits}, use_group_quant={use_group_quant}, group_size={group_size}, is_prefill={is_prefill}")
        if decode_force_fp4:
            print(f"  - Decode {kv_type}: FP4 Force (decode_use_fp4_{kv_type.lower()}=True)")
        if use_sliding_window and not is_prefill:
            print(f"  - Sliding window: bits={sliding_window_bits}, fp4_key={sliding_window_use_fp4_key}, fp4_value={sliding_window_use_fp4_value}")
        print(f"  - Outliers: key_precision={outlier_precision_key}, value_precision={outlier_precision_value}, outliers_per_group={outliers_per_group}")
    elif debug_mode and layer_id % 10 == 0:
        print(f"[Layer {layer_id}] Processing... ({'Prefill' if is_prefill else 'Decode'} mode)")

    if decode_only and is_prefill:
        return inp
    
    # Outlair fine-synthetic inter-tasking (backward compactity)
    if outlier_shift and outlier_precision is not None:
        if outlier_precision == "half":
            outlier_method = "half_precision"
        elif outlier_precision == "fp4":
            outlier_method = "fp4_no_extra"
    
    # Parsing New Key/Value
    # Outlier_precision_key and output_precision_value are later passed to the appropriate function

    # Force groupation path to enter when force decod FP4
    if decode_force_fp4 and not is_prefill:
        use_group_quant = True
        if debug_mode and layer_id <= 1:
            print(f"[Layer {layer_id}]  Decode FP4 Force: Enter the group-majorization path.")

    if use_group_quant or (is_prefill and bits <= 4):
        if debug_mode and layer_id <= 1:
            print(f"[Layer {layer_id}] Using vectorized group quantization (group_size={group_size})")

        # KV FP4 fixation in Freell
        group_use_fp4 = use_fp4
        if is_prefill and prefill_use_fp4:
            group_use_fp4 = True  # Force FP4 in Freell
            if debug_mode and layer_id <= 1:
                print(f"[Layer {layer_id}]  Prefill KV FP4 Fixed: FP4 Force")
        
        # Force KV FP4 in Decode
        if decode_force_fp4:
            group_use_fp4 = True  # Force FP4 on Decod
            if debug_mode and layer_id <= 1:
                print(f"[Layer {layer_id}]  Decode KV FP4 Force: FP4 Force activated")

        # Reflecting K/V Star Outlier Detection
        _topk = outlier_percent_topk
        _bottomk = outlier_percent_bottomk
        _outliers_per_group = outliers_per_group
        if (is_key and not detect_outliers_key) or ((not is_key) and not detect_outliers_value):
            _topk, _bottomk = 0.0, 0.0
            _outliers_per_group = 0

        return _apply_group_quantization_vectorized(
            inp,
            group_size=group_size,
            bits=bits,
            use_pow2=use_pow2,
            attention_sink_num=attention_sink_num,
            outlier_percent_topk=_topk,
            outlier_percent_bottomk=_bottomk,
            outlier_shift=outlier_shift,
            is_key=is_key,
            debug_mode=debug_mode,
            layer_id=layer_id,
            is_prefill=is_prefill,
            outlier_method=outlier_method,
            outliers_per_group=_outliers_per_group,
            outlier_precision_key=outlier_precision_key,
            outlier_precision_value=outlier_precision_value,
            detect_outliers_value=detect_outliers_value,
            use_fp4=group_use_fp4,  # Apply Freel/Decode FP4 Locked
            attention_sink_num_front=attention_sink_num_front,  # Group catching Support
            sparse_keep_mask=sparse_keep_mask,  # fakesparse keep-mask (prefill only)
        )

    # use_group_quant is always True → fallback not reachable
    raise RuntimeError(
        f"attention_quantizer: use_group_quant must be True (got {use_group_quant}). "
        "Legacy non-group quantization path has been removed."
    )
