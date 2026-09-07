import torch
import os
import sys
import transformers
import transformers.modeling_utils
if not hasattr(transformers.modeling_utils, "shard_checkpoint"):
    transformers.modeling_utils.shard_checkpoint = lambda *args, **kwargs: (None, None)
    print("[PATCH] ✅ transformers.modeling_utils.shard_checkpoint dummy injected")

class DummyAutoModel: 
    @staticmethod
    def from_pretrained(*args, **kwargs): raise NotImplementedError("AutoModelForVision2Seq is not available")
# _LazyModule._gettr__i is bypassed by direct injections into _dict_dict_.
transformers.__dict__["AutoModelForVision2Seq"] = DummyAutoModel
print("[PATCH] ✅ transformers.AutoModelForVision2Seq manually injected into __dict__")

# :PytorchGELUNAH Compatibility patches added
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    from transformers.activations import PytorchGELUTanh
except ImportError:
    class PytorchGELUTanh(torch.nn.Module):
        def __init__(self): super().__init__()
        def forward(self, input): return torch.nn.functional.gelu(input, approximate="tanh")
    import transformers.activations
    transformers.activations.PytorchGELUTanh = PytorchGELUTanh
    print("[PATCH] ✅ PytorchGELUTanh successfully injected into transformers.activations")

# Compatibility shim: optimum's QuantizeConfig may be absent in some library versions; provide a dummy.
try:
    import optimum.gptq.quantizer
    if not hasattr(optimum.gptq.quantizer, "QuantizeConfig"):
        class DummyQuantizeConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items(): setattr(self, k, v)
        optimum.gptq.quantizer.QuantizeConfig = DummyQuantizeConfig
        print("[PATCH] ✅ optimum QuantizeConfig shim installed")
except ImportError:
    pass

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.join(os.environ["HF_HOME"], "metrics"))
os.environ.setdefault("HF_EVALUATE_CACHE", os.path.join(os.environ["HF_HOME"], "evaluate"))
os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
LM_EVAL_PATH = os.path.join(os.path.dirname(__file__), "../eval")
if os.path.exists(LM_EVAL_PATH):
    sys.path.insert(0, LM_EVAL_PATH)
    print(f"[INFO] Added lm-eval-harness path: {LM_EVAL_PATH}")

try:
    # kvquint_adpter port in the lm_eval.models path
    from lm_eval.models import kvquant_adapter  # shouldn't be able to register the 'kvquint_v2' model in the regtry if you just report it.
    print("[INFO] kvquant_adapter imported successfully - 'kvquant_v2' model registered")
except ImportError as e:
    print(f"[ERROR] kvquant_adapter import failed: {e}")
    print("[INFO] Trying alternative import method...")
    try:
        import importlib.util
        kvquant_adapter_path = os.path.join(LM_EVAL_PATH, "lm_eval", "models", "kvquant_adapter.py")
        spec = importlib.util.spec_from_file_location("kvquant_adapter", kvquant_adapter_path)
        kvquant_adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kvquant_adapter)
        print("[INFO] kvquant_adapter imported via alternative method - 'kvquant_v2' model registered")
    except Exception as e2:
        print(f"[ERROR] Alternative import method also failed: {e2}")
        raise e

import time
import json
import importlib.util
from tqdm import tqdm
import argparse
import torch._dynamo
import torch.nn as nn
from datasets import load_dataset
from datetime import datetime
import gc

# Disable all symbolic shell tracking
torch._dynamo.config.dynamic_shapes = False
torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
def cleanup_gpu_memory():
    """GPU Memory Summary Function"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print("[INFO] GPU memory cleaned up")

def get_gpu_memory_info(device_id=None):
    """GPU Returns memory information"""
    if not torch.cuda.is_available():
        return None, None
    
    if device_id is None:
        device_id = torch.cuda.current_device()
    
    mem_info = torch.cuda.mem_get_info(device_id)
    free_memory = mem_info[0] / (1024**3)  # GB
    total_memory = mem_info[1] / (1024**3)  # GB
    used_memory = total_memory - free_memory
    
    return free_memory, used_memory

# lm_eval import with better error handling
try:
    import lm_eval
    from lm_eval import simple_evaluate
    print("[INFO] lm-eval-harness imported successfully")
    pass
except ImportError as e:
    print(f"[WARNING] Failed to import lm_eval: {e}")
    print("[WARNING] Make sure lm-evaluation-harness is installed and accessible")
    simple_evaluate = None

# Huging Face Token Settings
HF_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")

def setup_huggingface_token():
    """
    Hugging Face Token Settings
    """
    global HF_TOKEN
    
    # Confirm tokens in environment variables
    env_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        HF_TOKEN = env_token
        print(f"✅ Hugging Face Token is set in environment variables.")
    else:
        print(f"⚠️  Environment Variables HUGGING_FACE_HUB_TOKENThis option is not set.")
        print(f"   Gated models may fail to load. Set: export HUGGING_FACE_HUB_TOKEN=hf_xxx")
    
    return HF_TOKEN

def parse_args():
    """
    A function for parsing the parameter of the command line
    Returns:
        args: Exploreded.
    """
    parser = argparse.ArgumentParser(
        description="KVQ (Key-Value Quantization) - Model Quantification and Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--bits", type=int, default=2, help="Quantization bits (fallback for key_bits/value_bits)")
    parser.add_argument("--key_bits", type=int, default=None, help="Key quantization bits (overrides bits)")
    parser.add_argument("--value_bits", type=int, default=None, help="Value quantization bits (overrides bits)")
    # Prefill/Decode specific bit options
    parser.add_argument("--prefill_key_bits", type=int, default=None, help="Key bits for prefill (overrides key_bits/bits)")
    parser.add_argument("--prefill_value_bits", type=int, default=None, help="Value bits for prefill (overrides value_bits/bits)")
    parser.add_argument("--decode_key_bits", type=int, default=None, help="Key bits for decode (overrides key_bits/bits)")
    parser.add_argument("--decode_value_bits", type=int, default=None, help="Value bits for decode (overrides value_bits/bits)")
    parser.add_argument("--key_fp4", action="store_true", help="Use FP4 quantization for keys")
    parser.add_argument("--use_symmetric", action="store_true", help="Use symmetric quantization")
    parser.add_argument("--use_pow2", action="store_true", help="Use pow2 approximation")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--use_sliding_window", default = True, action="store_true", help="Use sliding window quantization")
    parser.add_argument("--window_size", type=int, default=32, help="Sliding window size (will be unified with group_size if group quant is enabled)")
    parser.add_argument("--attention_sink_num", type=int, default=5, help="Attention sink size")
    parser.add_argument("--outlier_mode", type=str, default="elementwise", help="Outlier detection mode")
    parser.add_argument("--outlier_method", type=str, default="original", help="Outlier quantization method (original, fp4, fp4_shared_scale, etc.)")
    parser.add_argument("--outlier_percent_topk", type=float, default=0.01, help="Top k percent outlier percentage")
    parser.add_argument("--outlier_percent_bottomk", type=float, default=0.01, help="Bottom k percent outlier percentage")
    parser.add_argument("--decode_only",  action="store_true", help="Decode only")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--outlier_shift", action="store_true", help="Use outlier shift")
    parser.add_argument("--use_a_shape", action="store_true", help="Use A-Shape mask")
    parser.add_argument("--distance_ceiling", type=int, default=32,
                        help="Quantise distances beyond this to a constant value")

    # --------------------------------------------------------------------------
    # Group Quantization Options
    # --------------------------------------------------------------------------
    parser.add_argument("--debug_mode", action="store_true", default=False, help="Debug mode")
    parser.add_argument("--eval_task", type=str, default="ppl", help="Evaluation task")
    parser.add_argument("--stride", type=int, default=512, help="Stride for evaluation")
    parser.add_argument("--eval_only", action="store_true", help="Skip quantization and evaluate with kvquant model directly")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--confirm_run_unsafe_code", action="store_true", help="Confirm running tasks marked as unsafe (e.g., HumanEval)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples per task")
    parser.add_argument("--max_gen_toks", type=int, default=32, help="Maximum generation tokens")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Number of examples in few-shot context")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum sequence length for evaluation (defaults to model's max)")

    parser.add_argument("--use_group_quant", action="store_true", help="Enable group quantization for prefill")
    parser.add_argument("--group_size", type=int, default=1024, help="Group size for group quantization (also used as window_size when group quant is enabled)")
    
    parser.add_argument("--sliding_window_use_fp4_key", action="store_true", help="Use FP4 for sliding window keys")
    parser.add_argument("--sliding_window_use_fp4_value", action="store_true", help="Use FP4 for sliding window values")
    parser.add_argument("--no_detect_outliers_key", action="store_true", default=False, help="Disable outlier detection for keys")
    parser.add_argument("--no_detect_outliers_value", action="store_true", default=False, help="Disable outlier detection for values")
    parser.add_argument("--outlier_precision_key", type=str, default=None, help="Outlier precision for keys (fp4, half, etc.)")
    parser.add_argument("--outlier_precision_value", type=str, default=None, help="Outlier precision for values (fp4, half, etc.)")
    parser.add_argument("--outliers_per_group", type=int, default=0, help="Number of outliers to extract per group")
    
    parser.add_argument("--decode_use_fp4_key", action="store_true", help="Force FP4 for all decode key tokens")
    parser.add_argument("--decode_use_fp4_value", action="store_true", help="Force FP4 for all decode value tokens")
    parser.add_argument("--prefill_use_fp4", action="store_true", help="Force FP4 quantization for KV in prefill mode")
    
    # Weight Quantization arguments
    parser.add_argument("--use_weight_quant", action="store_true", help="Enable weight quantization")
    parser.add_argument("--weight_bits", type=int, default=2, help="Weight quantization bits")
    parser.add_argument("--weight_group_size", type=int, default=64, help="Weight quantization group size")
    parser.add_argument("--weight_outlier_group_size", type=int, default=None, help="Weight quantization outlier group size")
    parser.add_argument("--weight_outlier_percent", type=float, default=0.01, help="Weight quantization outlier percentage")
    parser.add_argument("--weight_lr_path", type=str, default=None, help="Path to calibrated V parameters")
    
    parser.add_argument("--disable_exllama", action="store_true", help="Disable exllama backend")

    args = parser.parse_args()
    
    return args
    
def evaluate_with_lm_eval(args):
    """
    Direct lm-eval evaluation without double model loading
    """
    import traceback
    try:
        print("[INFO] Cleaning up GPU memory before evaluation...")
        cleanup_gpu_memory()
        
        if torch.cuda.is_available():
            current_gpu = torch.cuda.current_device()
            free_mem, used_mem = get_gpu_memory_info(current_gpu)
            print(f"[INFO] Current GPU {current_gpu} - Free: {free_mem:.2f}GB / Used: {used_mem:.2f}GB")
        
        # Prepare model arguments for kvquant
        
        model_args_list = [
            f"pretrained={args.model}",
            f"bits={args.bits}",
            f"key_bits={args.key_bits if args.key_bits is not None else args.bits}",
            f"value_bits={args.value_bits if args.value_bits is not None else args.bits}",
            f"prefill_key_bits={args.prefill_key_bits if args.prefill_key_bits is not None else ''}",
            f"prefill_value_bits={args.prefill_value_bits if args.prefill_value_bits is not None else ''}",
            f"decode_key_bits={args.decode_key_bits if args.decode_key_bits is not None else ''}",
            f"decode_value_bits={args.decode_value_bits if args.decode_value_bits is not None else ''}",
            f"use_pow2={1 if args.use_pow2 else 0}",
            f"use_sliding_window={1 if args.use_sliding_window else 0}",
            f"window_size={args.window_size}",
            f"attention_sink_num={args.attention_sink_num}",
            f"outlier_mode={args.outlier_mode}",
            f"outlier_percent_topk={args.outlier_percent_topk}",
            f"outlier_percent_bottomk={args.outlier_percent_bottomk}",
            f"decode_only={1 if args.decode_only else 0}",
            f"outlier_shift={1 if args.outlier_shift else 0}",  
            f"use_a_shape={1 if args.use_a_shape else 0}",
            f"distance_ceiling={args.distance_ceiling}",
            f"debug_mode={1 if args.debug_mode else 0}",
            f"disable_exllama={args.disable_exllama}",
            f"trust_remote_code=True",
            f"max_gen_toks={args.max_gen_toks}",
            f"use_group_quant={1 if args.use_group_quant else 0}",
            f"group_size={args.group_size}",
            f"sliding_window_use_fp4_key={1 if args.sliding_window_use_fp4_key else 0}",
            f"sliding_window_use_fp4_value={1 if args.sliding_window_use_fp4_value else 0}",
            f"outlier_method={args.outlier_method}",
            f"outlier_precision_key={args.outlier_precision_key}",
            f"outlier_precision_value={args.outlier_precision_value}",
            f"detect_outliers_key={0 if args.no_detect_outliers_key else 1}",
            f"detect_outliers_value={0 if args.no_detect_outliers_value else 1}",
            f"outliers_per_group={args.outliers_per_group}",
            f"decode_use_fp4_key={1 if args.decode_use_fp4_key else 0}",
            f"decode_use_fp4_value={1 if args.decode_use_fp4_value else 0}",
            f"prefill_use_fp4={1 if args.prefill_use_fp4 else 0}",  # FP4 fixed.
            f"use_weight_quant={1 if args.use_weight_quant else 0}",
            f"weight_bits={args.weight_bits}",
            f"weight_group_size={args.weight_group_size}",
            f"weight_outlier_group_size={args.weight_outlier_group_size if args.weight_outlier_group_size is not None else ''}",
            f"weight_outlier_percent={args.weight_outlier_percent}",
            f"weight_lr_path={args.weight_lr_path if args.weight_lr_path else ''}",
            "trust_remote_code=True",
            "low_cpu_mem_usage=True",
            "device_map=auto",
        ]
        
        # max_length: Forward only when explicitly specified (i. e. md. md use mdel.config.max_image_embradings)
        if args.max_length:
            model_args_list.append(f"max_length={args.max_length}")
        
        model_args = ",".join(model_args_list)

        # Evaluation kwargs with memory optimizations
        eval_kwargs = {
            'model': 'kvquant_v2',
            #'model': 'hf',
            'model_args': model_args,
            'tasks': [args.eval_task],
            'num_fewshot': args.num_fewshot if args.num_fewshot is not None else 0, # Fallback to 0 if None to be safe, but can also pass None
            'batch_size': args.batch_size if args.batch_size is not None else 1,
            'confirm_run_unsafe_code': args.confirm_run_unsafe_code
        }
        
        # Add limit if specified
        if args.limit:
            eval_kwargs['limit'] = args.limit
            
        print(f"[INFO] Evaluation kwargs: {eval_kwargs}")
        
        if args.eval_task == "wikitext":
            print("[INFO] Bypassing lm-eval for wikitext to measure 4K Chunked Perplexity...")
            from lm_eval.utils import simple_parse_args_string
            from lm_eval.api.registry import get_model
            parsed_args = simple_parse_args_string(model_args)
            lm_model = get_model("kvquant_v2")(**parsed_args)
            model = lm_model.model
            tokenizer = lm_model.tokenizer
            
            # Load wikitext-2
            from datasets import load_dataset
            import torch.nn as nn
            testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
            
            seqlen = 4096
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // seqlen
            model = model.eval()
            nlls = []
            
            import tqdm
            for i in tqdm.tqdm(range(nsamples), desc="Evaluating chunked wikitext(4k)"):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
            print(f"[INFO] 4K Chunked PPL: {ppl.item()}")
            results = {"results": {"wikitext": {"word_perplexity": ppl.item(), "chunked_ppl": ppl.item()}}}
        else:
            # Direct evaluation with kvquant model (no double loading)
            results = simple_evaluate(**eval_kwargs)
            print("[INFO] Evaluation completed successfully!")
            
        return results
        
    except KeyboardInterrupt:
        print("[ERROR] Evaluation interrupted by user")
        raise
    except SystemExit as e:
        print(f"[ERROR] System exit with code: {e.code}")
        raise
    except BaseException as e:  # All exceptions, catch.
        print(f"[ERROR] Unexpected error type: {type(e).__name__}")
        print(f"[ERROR] Error message: {str(e)}")
        print(f"[ERROR] Full traceback:")
        traceback.print_exc()
        
        # Save debug info
        output_dir = create_output_directory()
        debug_info = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc(),
            'args': vars(args)
        }
        save_debug_info(output_dir, args, error=debug_info)
        raise e

def create_output_directory():
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
    
def save_debug_info(output_dir, args, error=None):
    """Save debug information for troubleshooting"""
    debug_info = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'python_path': sys.path[:5],  # First 5 entries
        'error': str(error) if error else None
    }
    
    debug_file = os.path.join(output_dir, f"debug_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    with open(debug_file, 'w') as f:
        json.dump(debug_info, f, indent=2)
    
    print(f"Debug info saved to: {debug_file}")

def main():
    args = parse_args()
    
    # Huging Face Token Settings
    setup_huggingface_token()

    # CUDA Environment Debugging Information and the best GPU selection
    
    print("=== CUDA Environment Check ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        # Preliminary Memory Cleanup
        cleanup_gpu_memory()
        
        # Auto select GPU with the least memory usage
        best_gpu = 0
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            mem_info = torch.cuda.mem_get_info(i)
            free_memory = mem_info[0] / (1024**3)  # GB
            total_memory = mem_info[1] / (1024**3)  # GB
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} - Free: {free_memory:.2f}GB / Total: {total_memory:.2f}GB")
            
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = i
        
        # Make sure that the optimal GPU settings and extra memory are set
        torch.cuda.set_device(best_gpu)
        cleanup_gpu_memory()  # Add to the selected GPU
       
        # Confirm Last Memory Status
        final_free, final_used = get_gpu_memory_info(best_gpu)
        print(f"Selected GPU {best_gpu} - Free: {final_free:.2f}GB / Used: {final_used:.2f}GB")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")    
    print("[INFO] Running KVQ Quantizer")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Bits: {args.bits} (Key: {args.key_bits}, Value: {args.value_bits})")
    print(f"[INFO] Key FP4: {args.key_fp4}")
    print(f"[INFO] Use Symmetric: {args.use_symmetric}")
    print(f"[INFO] Outlier Shift: {args.outlier_shift}")
    print(f"[INFO] Use Pow2: {args.use_pow2}")
    print(f"[INFO] Sliding Window: {args.use_sliding_window} (Size: {args.window_size})")
    print(f"[INFO] Eval Task: {args.eval_task}")
    print(f"[INFO] Debug Mode: {args.debug_mode}")

    # Skip model loading in quantizer
    # Model will be loaded only once in kvquant.py
    print("(1) Skipping model loading - will be handled by kvquant...")
    print("(2) Skipping tokenizer loading - will be handled by kvquant...")
    print("Model loading will be handled by kvquant model.")
    print("(3) Skipping KV Cache Quantization setup - will be handled by kvquant...")
    print("KV Cache Quantization will be applied by kvquant model.")
    print("Model preparation will be handled by kvquant model.")
    
    # Check if eval_only mode is enabled
    if args.eval_only:
        print("(4) Eval-only mode: Skipping quantization setup, evaluating with kvquant model directly...")
    else:
        print("(4) Normal mode: Applying quantization before evaluation...")
    
    # 4. Evaluate with lm-eval-harness (MEMORY OPTIMIZED)
    print("(5) Evaluating with lm-eval-harness...")
    try:
        results = evaluate_with_lm_eval(args)
        
        # Save results with proper serialization
        output_dir = create_output_directory()
        
        # Update args.output_dir to use the timestamped directory
        args.output_dir = output_dir
        
        # Use save_lm_eval_results function for proper dtype handling
        save_lm_eval_results(args, results)
        
        print(f"[INFO]  Results saved to: {output_dir}")
        
        # Print summary
        if 'results' in results and args.eval_task in results['results']:
            task_results = results['results'][args.eval_task]
            print(f"\n Results Summary:")
            print(f"Task: {args.eval_task}")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        print(f"[ERROR] Full traceback:")
        traceback.print_exc()
        
        # Save Debug Information
        try:
            save_debug_info(args, None, None, error_msg=str(e))
        except:
            pass
        
        return 1
    
    print("[INFO]  KVQ Evaluation completed successfully")
    return 0

def save_lm_eval_results(args, results):
    """
    lm-eval-harness The result. JSON Save to File
    """
    def convert_to_serializable(obj):
        """Convert any object to JSON-serializable format"""
        import numpy as np
        import torch
        
        # Handle None
        if obj is None:
            return None
        
        # Handle basic Python types
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # 🔧 Handle dtype objects FIRST (before other checks)
        if hasattr(obj, 'name') and hasattr(obj, 'kind'):  # numpy/torch dtype
            return str(obj)
        
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.dtype):
            return str(obj)
        
        # Handle torch types
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, torch.dtype):
            return str(obj)
        
        # Handle other dtype-like objects
        elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
            try:
                return obj.item()
            except:
                return str(obj)
        elif hasattr(obj, 'dtype') and not hasattr(obj, '__len__'):
            return str(obj)
        
        # Handle collections
        elif isinstance(obj, dict):
            return {str(key): convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(convert_to_serializable(item) for item in obj)
        
        # Handle other objects by converting to string
        else:
            try:
                # Check if it's a dtype-like object by type name
                type_name = type(obj).__name__
                if 'dtype' in type_name.lower():
                    return str(obj)
                
                # Try to convert to basic type if possible
                if hasattr(obj, '__dict__'):
                    return str(obj)
                else:
                    return str(obj)
            except Exception as e:
                return f"<{type(obj).__name__} object: {str(e)[:50]}>"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a filename that contains the Timestamps
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    task_name = args.eval_task.replace(",", "_").replace("/", "_")
    output_file = os.path.join(args.output_dir, f"kvq_results_{task_name}_{timestamp}.json")
    
    # Save result with evaluation settings
    full_results = {
        "model": args.model,
        "evaluation_config": {
            "bits": args.bits,
            "key_bits": args.key_bits,
            "value_bits": args.value_bits,
            "key_fp4": args.key_fp4,
            "use_symmetric": args.use_symmetric,
            "use_pow2": args.use_pow2,
            "use_sliding_window": args.use_sliding_window,
            "window_size": args.window_size,
            "attention_sink_num": args.attention_sink_num,
            "outlier_mode": args.outlier_mode,
            "outlier_percent_topk": args.outlier_percent_topk,
            "outlier_percent_bottomk": args.outlier_percent_bottomk,
            "outlier_shift": args.outlier_shift,
            "use_a_shape": args.use_a_shape,
            "distance_ceiling": args.distance_ceiling,
            "decode_only": args.decode_only,
            "debug_mode": args.debug_mode,
            "use_weight_quant": args.use_weight_quant,
            "weight_bits": args.weight_bits,
            "weight_group_size": args.weight_group_size,
            "weight_outlier_group_size": args.weight_outlier_group_size,
            "weight_outlier_percent": args.weight_outlier_percent,
            "tasks": args.eval_task,
            "batch_size": getattr(args, 'batch_size', None),
            "limit": getattr(args, 'limit', None)
        },
        "results": convert_to_serializable(results),
        "timestamp": timestamp
    }
    
    # Convert results to JSON-serializable format
    serializable_results = convert_to_serializable(full_results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)
    
    print(f"[INFO] Evaluation results saved to: {output_file}")
    
    # Main result output
    if isinstance(results, dict) and "results" in results:
        print("\n=== Evaluation Results ===")
        for task_name, task_results in results["results"].items():
            if isinstance(task_results, dict):
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        print(f"{task_name} {metric}: {value:.4f}")
        print("=" * 25)

if __name__ == "__main__":
    main()