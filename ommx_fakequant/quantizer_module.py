#quantizer_module.py

import torch.nn as nn
import torch
import sys
import os
_QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
if _QUANT_DIR not in sys.path:
    sys.path.insert(0, _QUANT_DIR)

# Use stable Quant_Attention implementation
from quant_attention import Quant_Attention, ModelArgs


def _split_fused_qkv_proj(model):
    """phi3 (and other fused-qkv archs) expose a single `qkv_proj` Linear instead of separate
    q_proj/k_proj/v_proj. The rest of this module's attention-detection/weight-copy code only
    knows how to read separate q/k/v Linears, so split qkv_proj into three real nn.Linear
    submodules (out_features sliced by num_heads*head_dim / num_kv_heads*head_dim) and attach
    them as q_proj/k_proj/v_proj. qkv_proj itself is left in place (unused after this) so nothing
    else that references it breaks.
    """
    config = model.config
    model_type = getattr(config, 'model_type', '').lower()
    if model_type != 'phi3':
        return
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, 'q_proj') or not hasattr(attn, 'qkv_proj'):
            continue
        fused = attn.qkv_proj
        assert fused.weight.shape[0] == q_size + 2 * kv_size, (
            f"qkv_proj out_features {fused.weight.shape[0]} != q+2*kv "
            f"({q_size}+2*{kv_size}); phi3 head-dim/num-heads config mismatch")
        has_bias = fused.bias is not None
        dtype, device = fused.weight.dtype, fused.weight.device
        q_proj = nn.Linear(hidden_size, q_size, bias=has_bias, dtype=dtype, device=device)
        k_proj = nn.Linear(hidden_size, kv_size, bias=has_bias, dtype=dtype, device=device)
        v_proj = nn.Linear(hidden_size, kv_size, bias=has_bias, dtype=dtype, device=device)
        with torch.no_grad():
            q_proj.weight.copy_(fused.weight[:q_size])
            k_proj.weight.copy_(fused.weight[q_size:q_size + kv_size])
            v_proj.weight.copy_(fused.weight[q_size + kv_size:])
            if has_bias:
                q_proj.bias.copy_(fused.bias[:q_size])
                k_proj.bias.copy_(fused.bias[q_size:q_size + kv_size])
                v_proj.bias.copy_(fused.bias[q_size + kv_size:])
        attn.q_proj, attn.k_proj, attn.v_proj = q_proj, k_proj, v_proj


def apply_kv_cache_quant(
    model,
    bits = 2,
    key_bits = None,
    value_bits = None,
    key_fp4 = False,
    prefill_key_bits = None,
    prefill_value_bits = None,
    decode_key_bits = None,
    decode_value_bits = None,
    prefill_use_fp4 = True,
    use_pow2 = True,
    use_sliding_window = False,  
    window_size = 32,
    outlier_percent_topk = 0.01,
    outlier_percent_bottomk = 0.00,
    outlier_shift = False,
    use_a_shape = False,
    decode_only = False,
    debug_mode = False,
    attention_sink_num = 5,
    distance_ceiling = 32,
    outlier_mode = "elementwise",
    outlier_method = "original",
    use_fp4_key = False,
    use_fp4_value = False,
    sliding_window_use_fp4_key = False,
    sliding_window_use_fp4_value = False,
    outlier_precision_key = "int",
    outlier_precision_value = "int",
    detect_outliers_key = True, 
    detect_outliers_value = True,
    outliers_per_group = 0,
    use_symmetric = False,
    use_group_quant = False,
    group_size = 1024,
    decode_use_fp4_key = False, 
    decode_use_fp4_value = False, 
):

    if key_fp4:
        use_fp4_key = True
    
    print("Applying KV Cache Quantization - Replacing attention layers to Quant_Attention")
    
    print("Quantization Configuration:")
    print(f"  - Bits: {bits}")
    print(f"  - Key bits: {key_bits}")
    print(f"  - Value bits: {value_bits}")
    print(f"  - Key FP4: {key_fp4} (mapped to use_fp4_key: {use_fp4_key})")
    print(f"  - Prefill bits: K={prefill_key_bits}, V={prefill_value_bits}, FP4={prefill_use_fp4}")
    print(f"  - Decode bits:  K={decode_key_bits}, V={decode_value_bits}")
    print(f"  - Use pow2: {use_pow2}")
    print(f"  - Sliding window: {use_sliding_window}")
    print(f"  - Window size: {window_size}")
    print(f"  - Decode only: {decode_only}")
    print(f"  - Outlier shift: {outlier_shift}")
    print(f"  - Outlier percentage: {outlier_percent_topk}")
    print(f"  - Use A-shape: {use_a_shape}")
    print(f"  - Attention sink: {attention_sink_num}")
    print(f"  - Use group quantization: {use_group_quant}")
    print(f"  - Group size: {group_size}")
    print(f"  - Debug mode: {debug_mode}")

    layer_counter = [0]

    model_name = getattr(model.config, "_name_or_path", "")

    # Fused-qkv archs (phi3) don't have separate q_proj/k_proj/v_proj -- materialize them from
    # qkv_proj before the detection/copy logic below (which only knows how to read q/k/v Linears).
    _split_fused_qkv_proj(model)

    # Check for existing Quant_Attention layers
    if layer_counter[0] == 0:
        print("Checking for existing quantized layers...")
        
        # Quick check: The first attack layer is checked only
        first_attn_found = False
        for name, child in model.named_modules():
            if 'self_attn' in name and hasattr(child, 'q_proj'):
                if isinstance(child, Quant_Attention):
                    print(f"Model already has quantized attention layers - skipping replacement")
                    return model, None
                first_attn_found = True
                break
        
        if not first_attn_found:
            print("No attention layers found in model")
            return model, None
        
        print("Model is ready for quantization")
    
    # Efficient Attation Layer Collecting ( Direct access method)
    all_attention_layers = []
    
    # Direct access mode: Moveel.model.layers direct access
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        print(f"Found {len(model.model.layers)} layers in model.model.layers")
        
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'self_attn'):
                name = f"model.layers.{i}.self_attn"
                attn_module = layer.self_attn
                
                if not isinstance(attn_module, Quant_Attention):
                    all_attention_layers.append((name, attn_module))
                    if debug_mode and i < 3:  # The first 3 log output
                        print(f" Found attention layer {i}: {name}")
        
        if len(all_attention_layers) > 0:
            print(f"Found {len(all_attention_layers)} attention layers to replace")
        else:
            print("No attention layers found via direct access")
    else:
        print("Model structure not compatible with direct access, falling back to full search")
        # Paulback: Total Search (but more limited)
        for name, child in model.named_modules():
            if name.endswith('self_attn') and hasattr(child, 'q_proj') and not isinstance(child, Quant_Attention):
                all_attention_layers.append((name, child))
                     
    # Secondary search: more extensive access pattern (if not found in car)
    if len(all_attention_layers) == 0:
        if debug_mode:
            print("Primary search failed, trying broader attention detection...")
        
        # Various sets of pattern attempts
        for name, child in model.named_modules():
            # LaMA / Qwen Style Check
            is_attention = (
                hasattr(child, 'q_proj') and
                hasattr(child, 'k_proj') and
                hasattr(child, 'v_proj') and
                hasattr(child, 'o_proj') and
                isinstance(child.q_proj, nn.Linear) and
                isinstance(child.k_proj, nn.Linear) and
                isinstance(child.v_proj, nn.Linear) and
                isinstance(child.o_proj, nn.Linear) and
                not isinstance(child, Quant_Attention) 
            )

            if (is_attention) and ('attn' in name.lower() or 'attention' in name.lower()):
                all_attention_layers.append((name, child))
                if debug_mode:
                    print(f"  - Found via broad search: {name}")
    
    # Thirdary search: Model structure direct navigation (if still not found)
    if len(all_attention_layers) == 0:
        if debug_mode:
            print(" Broad search failed, trying direct model structure exploration...")
        
        # Search for model structure
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for i, layer in enumerate(model.model.layers):
                if hasattr(layer, 'self_attn'):
                    attn_module = layer.self_attn
                    is_attention = (
                        hasattr(attn_module, 'q_proj') and
                        hasattr(attn_module, 'k_proj') and
                        hasattr(attn_module, 'v_proj') and
                        hasattr(attn_module, 'o_proj') and
                        isinstance(attn_module.q_proj, nn.Linear) and
                        isinstance(attn_module.k_proj, nn.Linear) and
                        isinstance(attn_module.v_proj, nn.Linear) and
                        isinstance(attn_module.o_proj, nn.Linear) and
                        not isinstance(attn_module, Quant_Attention) 
                    )

                    if (is_attention):
                        name = f"model.layers.{i}.self_attn"
                        all_attention_layers.append((name, attn_module))
                        if debug_mode:
                            print(f"  - Found via direct exploration: {name}")
    
    if debug_mode:
        print(f" Found {len(all_attention_layers)} attention layers to quantize:")
        for name, child in all_attention_layers:
            print(f"  - {name}: {type(child).__name__}")
            if hasattr(child, 'q_proj'):
                q_shape = child.q_proj.weight.shape
                k_shape = child.k_proj.weight.shape
                v_shape = child.v_proj.weight.shape
                print(f"    Q: {q_shape}, K: {k_shape}, V: {v_shape}")
        
        # Debug: a structure analysis when search failed
        if len(all_attention_layers) == 0:
            print(" Debug: Model structure analysis:")
            print(f"  - Model type: {type(model)}")
            print(f"  - Has model.model: {hasattr(model, 'model')}")
            if hasattr(model, 'model'):
                print(f"  - Model.model type: {type(model.model)}")
                print(f"  - Has model.model.layers: {hasattr(model.model, 'layers')}")
                if hasattr(model.model, 'layers'):
                    print(f"  - Number of layers: {len(model.model.layers)}")
                    if len(model.model.layers) > 0:
                        first_layer = model.model.layers[0]
                        print(f"  - First layer type: {type(first_layer)}")
                        print(f"  - First layer has self_attn: {hasattr(first_layer, 'self_attn')}")
                        if hasattr(first_layer, 'self_attn'):
                            first_attn = first_layer.self_attn
                            print(f"  - First attn type: {type(first_attn)}")
                            print(f"  - First attn has q_proj: {hasattr(first_attn, 'q_proj')}")
            
            # Output all modules (up to ten)
            print(" Debug: All modules (first 10):")
            for i, (name, module) in enumerate(model.named_modules()):
                if i >= 10:
                    break
                print(f"  - {name}: {type(module).__name__}")
                if 'attn' in name.lower():
                    print(f"    → Attention-related module found!")
                    print(f"    → Has q_proj: {hasattr(module, 'q_proj')}")
                    print(f"    → Has k_proj: {hasattr(module, 'k_proj')}")
                    print(f"    → Has v_proj: {hasattr(module, 'v_proj')}")
                    print(f"    → Has o_proj: {hasattr(module, 'o_proj')}")

    # Replace the found attending layers with quantum versions.
    print(f"Replacing {len(all_attention_layers)} attention layers with Quant_Attention...")
    
    for idx, (name, child) in enumerate(all_attention_layers):
        # Find and replace the paracent of the current module
        parent_module = model
        name_parts = name.split('.')
        
        # Navigate to parent
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)
        
        final_name = name_parts[-1]
        old_attn = getattr(parent_module, final_name)
        is_attention_candidate = (
            hasattr(old_attn, 'q_proj') and hasattr(old_attn, 'k_proj') and
            hasattr(old_attn, 'v_proj') and hasattr(old_attn, 'o_proj') and
            isinstance(old_attn.q_proj, nn.Linear) and
            not isinstance(old_attn, Quant_Attention)
        )

        if not is_attention_candidate:
            if debug_mode:
                print(f"Skipping non-standard module: {name}")
            continue

        # Check the model type directly on the config, and divide it clearly
        model_type = getattr(model.config, 'model_type', 'llama').lower()

        # qwen3 + exaone4 share the QK-norm attention branch (copies q_norm/k_norm weights); the
        # per-model RoPE difference (exaone4 = llama3 scaling) is resolved inside Quant_Attention.
        is_qwen_attention = (model_type in ('qwen3', 'exaone4'))
        is_mllama_attention = ("Vision" in model_name)
        is_llama_attention = not (is_qwen_attention or is_mllama_attention)

        if is_llama_attention or is_mllama_attention:
            if debug_mode:
                print(f" Replacing Llama attention layer: {name}")
                
            if is_mllama_attention:
                # Compute Multimedia Lama
                print("Detected Multimodal Llama architecture. Using 'text_config'.")
                config = model.text_config
                hidden_size = config.hidden_size
                num_heads = config.num_attention_heads
                num_kv_heads = config.num_key_value_heads
                model_type = getattr(config, 'model_type', 'llama').lower()   
                if hasattr(config, 'head_dim') and config.head_dim is not None:
                    head_dim = config.head_dim
                else:
                    head_dim = hidden_size // num_heads
                    if debug_mode:
                        print(f" 'head_dim' not found in text_config, calculating manually: {head_dim}")
            else:
                config = model.config
                hidden_size = config.hidden_size
                num_heads = config.num_attention_heads
                num_kv_heads = config.num_key_value_heads
                head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
                    
            if debug_mode:
                print(f"  - Hidden size: {hidden_size}")
                print(f"  - Head dim: {head_dim}")
                print(f"  - Num heads: {num_heads}")
                print(f"  - Num KV heads: {num_kv_heads}")

            original_dtype = old_attn.q_proj.weight.dtype
            
            args = ModelArgs(
                dim = hidden_size,
                n_heads = num_heads,
                n_kv_heads = num_kv_heads,
                max_batch_size = 32,
                max_seq_len = getattr(config, 'max_position_embeddings', 4096),
                torch_dtype = original_dtype,  
                model_type = model_type,
                head_dim = head_dim,
                rope_scaling=getattr(config, 'rope_scaling', None)
            )

            args.bits = bits
            args.key_bits = key_bits
            args.value_bits = value_bits
            # Prefill/Decode specific bits (with overrides)
            args.prefill_key_bits = prefill_key_bits if prefill_key_bits is not None else key_bits
            args.prefill_value_bits = prefill_value_bits if prefill_value_bits is not None else value_bits
            args.decode_key_bits = decode_key_bits if decode_key_bits is not None else key_bits
            args.decode_value_bits = decode_value_bits if decode_value_bits is not None else value_bits
            args.prefill_use_fp4 = prefill_use_fp4 or use_fp4_key
            args.use_pow2 = use_pow2
            args.use_sliding_window = use_sliding_window
            args.window_size = window_size
            args.attention_sink_num = attention_sink_num
            args.outlier_mode = outlier_mode
            args.outlier_method = outlier_method
            args.outlier_percent_topk = outlier_percent_topk
            args.outlier_percent_bottomk = outlier_percent_bottomk
            args.use_a_shape = use_a_shape
            args.decode_only = decode_only
            args.distance_ceiling = distance_ceiling
            args.outlier_shift = outlier_shift
            # Group Quanta Parameters Forward
            args.use_group_quant = use_group_quant
            args.group_size = group_size
            # Send Sliding Window FP4 Parameters
            args.sliding_window_use_fp4_key = sliding_window_use_fp4_key
            args.sliding_window_use_fp4_value = sliding_window_use_fp4_value
            
            # Outliner Parameters Forward
            args.outlier_precision_key = outlier_precision_key
            args.outlier_precision_value = outlier_precision_value
            args.detect_outliers_value = detect_outliers_value
            args.outliers_per_group = outliers_per_group
            
            args.debug_mode = debug_mode
            args.use_fp4_key = use_fp4_key
            args.use_fp4_value = use_fp4_value
            args.use_symmetric = use_symmetric
            
            try:
                if not is_mllama_attention:
                    new_attn = Quant_Attention(args, hf_config=config)
                    new_attn = new_attn.to(old_attn.q_proj.weight.device)
                    new_attn.layer_idx = layer_counter[0]
                    layer_counter[0] += 1
                else:
                    new_attn = Quant_Attention_Multimodal(args, hf_config=config)
                    new_attn = new_attn.to(old_attn.q_proj.weight.device)
                    new_attn.layer_idx = layer_counter[0]
                    layer_counter[0] += 1
                
            except Exception as e:
                print(f"    Failed to create Quant_Attention for {name}: {e}")
                continue

            # Efficient Weight Copy (d type matching guarantee)
            try:
                # dtype Match Check and Adjust
                old_dtype = old_attn.q_proj.weight.dtype
                new_dtype = new_attn.q_proj.weight.dtype
                
                if old_dtype != new_dtype:
                    if debug_mode and idx < 3:
                        print(f"    🔧 Converting weight dtype: {old_dtype} → {new_dtype}")
                    new_attn.q_proj.weight.data.copy_(old_attn.q_proj.weight.data.to(new_dtype))
                    new_attn.k_proj.weight.data.copy_(old_attn.k_proj.weight.data.to(new_dtype))
                    new_attn.v_proj.weight.data.copy_(old_attn.v_proj.weight.data.to(new_dtype))
                    new_attn.o_proj.weight.data.copy_(old_attn.o_proj.weight.data.to(new_dtype))
                else:
                    new_attn.q_proj.weight.data.copy_(old_attn.q_proj.weight.data)
                    new_attn.k_proj.weight.data.copy_(old_attn.k_proj.weight.data)
                    new_attn.v_proj.weight.data.copy_(old_attn.v_proj.weight.data)
                    new_attn.o_proj.weight.data.copy_(old_attn.o_proj.weight.data)
                
                if debug_mode and idx < 3:
                    print(f"     Weights copied successfully (dtype: {new_attn.q_proj.weight.dtype})")
                    
            except Exception as e:
                print(f"    Failed to copy weights for {name}: {e}")
                continue
            
            # Cache Copy (if needed)
            if hasattr(old_attn, 'cache_k'):
                new_attn.cache_k = old_attn.cache_k
            if hasattr(old_attn, 'cache_v'):
                new_attn.cache_v = old_attn.cache_v

            setattr(parent_module, final_name, new_attn)
            
            if debug_mode and idx < 3:  # The first three are detailed.
                print(f"    Replaced {name} (layer_idx={new_attn.layer_idx})")
                print(f"      - Quantization: Key={new_attn.key_bits}bit, Value={new_attn.value_bits}bit")
            elif idx % 10 == 9:  # Simple progress for every 10 of them.
                print(f"  Progress: {idx+1}/{len(all_attention_layers)} layers replaced")

        elif is_qwen_attention:
            
            config = model.config
            hidden_size = config.hidden_size
            num_heads = config.num_attention_heads
            num_kv_heads = config.num_key_value_heads
            head_dim = getattr(config, 'head_dim', hidden_size // num_heads)

            args = ModelArgs(
                dim=hidden_size, n_heads=num_heads, n_kv_heads=num_kv_heads, head_dim=head_dim,
                max_seq_len=getattr(config, 'max_position_embeddings', 4096),
                torch_dtype=old_attn.q_proj.weight.dtype, model_type=model_type,
                rope_scaling=getattr(config, 'rope_scaling', None),
                rope_theta=getattr(config, 'rope_theta', 1000000.0),
                bits=bits, key_bits=key_bits, value_bits=value_bits,
                prefill_key_bits=prefill_key_bits, prefill_value_bits=prefill_value_bits,
                decode_key_bits=decode_key_bits, decode_value_bits=decode_value_bits,
                use_pow2=use_pow2, use_sliding_window=use_sliding_window, window_size=window_size,
                outlier_percent_topk=outlier_percent_topk,
                outlier_percent_bottomk=outlier_percent_bottomk, outlier_shift=outlier_shift,
                use_a_shape=use_a_shape, decode_only=decode_only, attention_sink_num=attention_sink_num,
                distance_ceiling=distance_ceiling, outlier_mode=outlier_mode, outlier_method=outlier_method,
                sliding_window_use_fp4_key=sliding_window_use_fp4_key,
                sliding_window_use_fp4_value=sliding_window_use_fp4_value,
                outlier_precision_key=outlier_precision_key, outlier_precision_value=outlier_precision_value,
                detect_outliers_value=detect_outliers_value, use_symmetric=use_symmetric,
                use_group_quant=use_group_quant, group_size=group_size,
                outliers_per_group=outliers_per_group, debug_mode=debug_mode
            )

            new_attn = Quant_Attention(args, layer_idx=idx, hf_config=config)
            new_attn = new_attn.to(device=old_attn.q_proj.weight.device)

            new_attn.q_proj.weight.data.copy_(old_attn.q_proj.weight.data)
            new_attn.k_proj.weight.data.copy_(old_attn.k_proj.weight.data)
            new_attn.v_proj.weight.data.copy_(old_attn.v_proj.weight.data)
            new_attn.o_proj.weight.data.copy_(old_attn.o_proj.weight.data)

            if hasattr(old_attn, 'q_norm') and new_attn.q_norm is not None:
                new_attn.q_norm.weight.data.copy_(old_attn.q_norm.weight.data)
            if hasattr(old_attn, 'k_norm') and new_attn.k_norm is not None:
                new_attn.k_norm.weight.data.copy_(old_attn.k_norm.weight.data)

            setattr(parent_module, final_name, new_attn)

    if debug_mode:
        quant_count = 0
        for name, submod in model.named_modules():
            if isinstance(submod, Quant_Attention):
                quant_count += 1
                print(f" Found Quant_Attention at {name} (layer_idx={submod.layer_idx})")
                print(f"  - Key bits: {submod.key_bits}, Value bits: {submod.value_bits}")
                print(f"  - Mode: {submod.key_quant_mode if hasattr(submod, 'key_quant_mode') else 'channelwise'} (Key), {submod.value_quant_mode if hasattr(submod, 'value_quant_mode') else 'tokenwise'} (Value)")
        
        print(f" Quantization Summary:")
        print(f"  - Total quantized layers: {quant_count}")
        print(f"  - Original layers replaced: {len(all_attention_layers)}")
        print(f"  - Success rate: {quant_count}/{len(all_attention_layers)} ({100*quant_count/len(all_attention_layers) if all_attention_layers else 0:.1f}%)")

    return model, None

    return True