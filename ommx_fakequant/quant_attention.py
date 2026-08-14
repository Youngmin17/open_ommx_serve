# quant_attention.py

from __future__ import annotations

import os
import sys
if sys.stdin is None or sys.stdin.closed:         
    sys.stdin = open(os.devnull) 
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
torch._dynamo.config.disable = True
# Add the current directory to PYTHONPATH so that local imports resolve when the
# file is executed as a script.
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.append(_current_dir)

# Quantisation helpers (vectorised variant is optional).
from quant_function import attention_quantizer
from transformers.cache_utils import Cache  # Hugging Face >= 4.40
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3RotaryEmbedding,
        apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
    )
except ImportError:
    print("Warning: Qwen3 modules not available.")

try:
    # phi3 uses partial RoPE (config.partial_rotary_factor < 1.0, e.g. 96 of head_dim=128):
    # cos/sin come back sized to the rotary_dim only, and only the first rotary_dim channels of
    # q/k are rotated (the rest pass through unrotated). The generic llama apply_rotary_pos_emb
    # assumes cos/sin cover the full head_dim, so phi3 needs this dedicated split/concat variant.
    from transformers.models.phi3.modeling_phi3 import (
        apply_rotary_pos_emb as phi3_apply_rotary_pos_emb,
    )
except ImportError:
    print("Warning: Phi3 modules not available.")

logger = get_logger("quant.quant_attention")


# -----------------------------------------------------------------------------
# Configuration dataclass ------------------------------------------------------
# -----------------------------------------------------------------------------


@dataclass
class ModelArgs:
    """Hyper‑parameters and runtime flags required by :class:`Quant_Attention`."""

    # Architecture 
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None  
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 4096
    
    # RoPE
    rope_theta: float = 500000.0 
    rope_scaling: Optional[dict] = None  
    rope_local_base_freq: Optional[float] = 10000.0 

    # Quantisation - Base
    bits: int = 2  
    key_bits: Optional[int] = None  
    value_bits: Optional[int] = None

    # Quantization - Prefill/Decode specific
    prefill_key_bits: Optional[int] = None
    prefill_value_bits: Optional[int] = None
    decode_key_bits: Optional[int] = None
    decode_value_bits: Optional[int] = None

    # Quantization - Sliding window
    sliding_window_key_bits: Optional[int] = None
    sliding_window_value_bits: Optional[int] = None
    sliding_window_use_fp4_key: bool = False
    sliding_window_use_fp4_value: bool = False

    # Quantization - Options
    use_pow2: bool = False  

    # Sliding‑window 
    use_sliding_window: bool = False
    window_size: int = 32
    distance_ceiling: int = 32  # Quantise distances beyond this to a constant

    # Outlier handling 
    attention_sink_num: int = 5
    outlier_mode: str = "elementwise"
    outlier_percent_topk: float = 0.01
    outlier_percent_bottomk: float = 0.00
    outlier_shift: bool = False
    detect_outliers_key: bool = True
    detect_outliers_value: bool = True
    outlier_precision_key: Optional[str] = None  # "half" | "fp4" | None
    outlier_precision_value: Optional[str] = None
    outlier_method: str = "original"

    # Misc flags 
    use_a_shape: bool = False
    decode_only: bool = False  # Disable quant. during *prefill*
    use_symmetric: bool = False  # Use symmetric quantization
    debug_mode: bool = False
    model_type: str = "llama"

    # Group Quantization 
    use_group_quant: bool = False  
    group_size: int = 64  
    outliers_per_group: int = 0

    # Model-specific
    head_dim: Optional[int] = None
    torch_dtype: torch.dtype = torch.bfloat16

    # fakesparse: SparseConfig for prefill GQA KV sparsity (None => no sparsity).
    # Decode GQA is never sparsified (kept dense quant K=i2f4 / V=i2).
    sparse_config: object = None

    def get_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        return self.dim // self.n_heads

    # Convenience helpers ------------------------------------------------------------
    def get_key_bits(self) -> int:
        return self.key_bits if self.key_bits is not None else self.bits

    def get_value_bits(self) -> int:
        return self.value_bits if self.value_bits is not None else self.bits


# -----------------------------------------------------------------------------
# Attention Module -------------------------------------------------------------
# -----------------------------------------------------------------------------


class Quant_Attention(nn.Module):
    """Llama‑compatible multi‑head attention with selective KV quantisation."""
    # , layer_idx: Optional[int] = None
    def __init__(self, args: ModelArgs, layer_idx: Optional[int] = None, hf_config=None):
        super().__init__()
        self.config = hf_config
        # Validate GQA before debugging information output
        # Removed [DEBUG] logs
        if args.n_kv_heads is None:
            args.n_kv_heads = args.n_heads
        
        # GQA Validated
        if args.n_heads % args.n_kv_heads != 0:
            raise ValueError(
                f"GQA validation failed at layer {layer_idx}: "
                f"n_heads ({args.n_heads}) % n_kv_heads ({args.n_kv_heads}) = "
                f"{args.n_heads % args.n_kv_heads} (must be 0). "
                f"Model type: {args.model_type}"
            )
        
        pass
        # --- Basic geometry -----------------------------------------------------
        self.model_type = args.model_type
        self.layer_idx = layer_idx
        self.hidden_size = args.dim
        self.num_heads = args.n_heads
        self.head_dim = args.get_head_dim()
        self.num_kv_heads = args.n_kv_heads or args.n_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.scaling = self.head_dim ** -0.5
        self.attention_type = "full_attention"
        self.is_sliding = False

        # rotate-kv (QuaRot-style, output-invariant): a head_dim Hadamard rotates Q/K
        # (post-RoPE) so scores (Q@R)(K@R).T == Q@K.T, and V (un-rotated after SDPA) so
        # (p@(V@R))@R.T == p@V. K/V quant then sees the flattened rotated tensors. R is
        # built lazily in forward once the device is known. See rotation.hadamard_matrix.
        self._rotate_kv = bool(os.environ.get("OMMX_ROTATE_KV"))
        self._rotate_seed = int(os.environ.get("OMMX_ROTATE_SEED", "42"))
        self._kv_R = None

        # --- Quantisation settings ---------------------------------------------
        self.bits = args.bits
        self.key_bits = args.get_key_bits()
        self.value_bits = args.get_value_bits()
        self.prefill_key_bits = args.prefill_key_bits if args.prefill_key_bits is not None else self.key_bits
        self.prefill_value_bits = args.prefill_value_bits if args.prefill_value_bits is not None else self.value_bits
        self.decode_key_bits = args.decode_key_bits if args.decode_key_bits is not None else self.key_bits
        self.decode_value_bits = args.decode_value_bits if args.decode_value_bits is not None else self.value_bits
        self.sliding_window_key_bits = args.sliding_window_key_bits
        self.sliding_window_value_bits = args.sliding_window_value_bits
        self.sliding_window_use_fp4_key = args.sliding_window_use_fp4_key
        self.sliding_window_use_fp4_value = args.sliding_window_use_fp4_value
        self.use_pow2 = args.use_pow2

        # --- Streaming / sliding‑window ----------------------------------------
        self.use_sliding_window = args.use_sliding_window
        self.window_size = args.window_size
        self.distance_ceiling = args.distance_ceiling

        # --- Outlier handling ---------------------------------------------------
        self.attention_sink_num = args.attention_sink_num
        self.outlier_mode = args.outlier_mode
        self.outlier_percent_topk = args.outlier_percent_topk
        self.outlier_percent_bottomk = args.outlier_percent_bottomk
        self.outlier_shift = args.outlier_shift
        self.detect_outliers_key = args.detect_outliers_key
        self.detect_outliers_value = args.detect_outliers_value
        self.outlier_precision_key = args.outlier_precision_key
        self.outlier_precision_value = args.outlier_precision_value
        self.outlier_method = getattr(args, 'outlier_method', 'original')

        # --- fakesparse (prefill GQA KV sparsity; None => off) ------------------
        self.sparse_config = getattr(args, "sparse_config", None)

        # --- Misc flags ---------------------------------------------------------
        self.use_a_shape = args.use_a_shape
        self.decode_only = args.decode_only
        self.debug_mode = args.debug_mode
        self.use_symmetric = args.use_symmetric
        
        # Group Quantization --------------------------------------------------
        self.use_group_quant = args.use_group_quant
        self.group_size = args.group_size
        _opg = getattr(args, 'outliers_per_group', None)
        self.outliers_per_group = (1 if self.group_size == 64 else 0) if _opg is None else _opg
        # K-outlier budget redistribution (env OMMX_KV_OUTLIER_PROFILE; unset => plan None and
        # the OFF quantization path below is byte-identical). Resolution is DEFERRED to first use
        # because the Llama replacement path sets .layer_idx AFTER __init__ (quantizer_module.py:347)
        # — resolving here would see layer_idx=None and collapse every layer to the fallback budget.
        self._k_outlier_profile_on = bool(os.environ.get("OMMX_KV_OUTLIER_PROFILE", ""))
        self._k_outlier_plan = None
        self._k_outlier_max_opg = 0
        self._k_outlier_resolved = not self._k_outlier_profile_on
        # --- Attention dropout / mask ------------------------------------------
        self.is_causal = True
        self.attention_dropout = 0.0  # HF handles training‑time dropout in SDPA

        # --- Rotary embedding ----------------------------------------------------
        self._init_rotary_embedding(args)

        # --- Projections --------------------------------------------------------
        # Moveel star bias image is unified by -bflot16
        target_dtype = getattr(args, 'torch_dtype', torch.bfloat16)
        bias = getattr(args, "attention_bias", False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias, dtype=target_dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias, dtype=target_dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias, dtype=target_dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias, dtype=target_dtype)

        # --- QK Normalization for Qwen3 -----------------------------------
        # QK-norm models: qwen3 and exaone4 both apply a per-head RMSNorm to Q/K on the head_dim
        # axis, pre-RoPE (Exaone4RMSNorm == Qwen3RMSNorm in form). exaone4 differs only in RoPE
        # (llama3 scaling) which is handled by the default llama RoPE path below.
        if self.model_type in ("qwen3", "exaone4"):
            self.q_norm = self._create_norm(self.head_dim, args)
            self.k_norm = self._create_norm(self.head_dim, args)
        else:
            self.q_norm = None
            self.k_norm = None

        # --- Runtime state ------------------------------------------------------
        self.is_prefill: bool = True  # Switch before calling ``forward``
        self.last_key_mse: float = 0.0  # Diagnostics
        self.last_value_mse: float = 0.0

        # Workspaces for in‑place quantisation (reduce reallocs during decoding)
        self._key_workspace: Optional[torch.Tensor] = None
        self._value_workspace: Optional[torch.Tensor] = None

        # ---roup quantification calling--
        # Prevent recalculating by compensating the resulting quantumization of a completedjaming unit
        self._cached_quant_k: Optional[torch.Tensor] = None
        self._cached_quant_v: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0

        if self.debug_mode:
            logger.info("Quant_Attention initialised – heads=%d/%d, bits(Q/K/V)=%d/%d/%d" % (
                self.num_heads,
                self.num_kv_heads,
                self.bits,
                self.key_bits,
                self.value_bits,
            ))
    
    def _init_rotary_embedding(self, args: ModelArgs):
        """By the type of model, RoPE Reset"""
        self._init_llama_rope(args)


    def _init_llama_rope(self, args: ModelArgs):
        """Llama RoPE Reset (hf_config Add Support)"""
        try:
            if self.model_type == "qwen3":
                from types import SimpleNamespace
                rope_theta = getattr(args, 'rope_theta', 1000000.0)
                rope_config = SimpleNamespace()
                rope_config.hidden_size = self.hidden_size
                rope_config.num_attention_heads = self.num_heads
                rope_config.num_key_value_heads = self.num_kv_heads  
                rope_config.head_dim = self.head_dim  
                rope_config.max_position_embeddings = args.max_seq_len
                rope_config.rope_theta = rope_theta
                rope_config.rope_scaling = getattr(args, 'rope_scaling', None)
                rope_config.vocab_size = getattr(args, 'vocab_size', 151936)
                rope_config.intermediate_size = getattr(args, 'intermediate_size', None)
                rope_config.rms_norm_eps = getattr(args, 'norm_eps', 1e-6)
                # New transformers requires rope_parameters dict
                rope_config.rope_parameters = {
                    "rope_type": "default",
                    "rope_theta": rope_theta,
                }
                self.rotary_emb = Qwen3RotaryEmbedding(config=rope_config)
                return

            # Llama 3.1 and so on and so forth, forwarding unfig to support the RoPe Scaling of the new model.
            if self.config is not None:
                # If there is rope_scaling in the config
                rope_scaling = getattr(self.config, "rope_scaling", None)
                rope_theta = getattr(self.config, "rope_theta", args.rope_theta)
                
                if self.debug_mode:
                    print(f"[DEBUG_ROPE] Layer {self.layer_idx}: Config detected!")
                    print(f"  - rope_theta: {rope_theta}")
                    print(f"  - rope_scaling: {rope_scaling}")
                    if rope_scaling is not None:
                         print(f"  - rope_scaling type: {type(rope_scaling)}")
                         print(f"  - rope_scaling dict: {getattr(rope_scaling, '__dict__', 'No __dict__')}")
                    print(f"  - max_position_embeddings: {getattr(self.config, 'max_position_embeddings', 'N/A')}")
                
                self.rotary_emb = LlamaRotaryEmbedding(
                    config=self.config,
                    device=None # HFLM will handle device
                )
                if self.debug_mode:
                    print(f"[ROPE] Initialized LlamaRotaryEmbedding with config.")
            else:
                # Build a proper config-like object for new transformers API
                from types import SimpleNamespace
                rope_theta = getattr(args, 'rope_theta', 500000.0)
                rope_config = SimpleNamespace()
                rope_config.hidden_size = self.hidden_size
                rope_config.num_attention_heads = self.num_heads
                rope_config.num_key_value_heads = self.num_kv_heads
                rope_config.head_dim = self.head_dim
                rope_config.max_position_embeddings = args.max_seq_len
                rope_config.rope_theta = rope_theta
                rope_config.rope_scaling = getattr(args, 'rope_scaling', None)
                rope_config.rope_parameters = {
                    "rope_type": "default",
                    "rope_theta": rope_theta,
                }
                self.rotary_emb = LlamaRotaryEmbedding(config=rope_config)
        except (TypeError, Exception) as e:
            if self.debug_mode:
                print(f"[ROPE] Fallback due to error: {e}")
            from types import SimpleNamespace
            rope_theta = getattr(args, 'rope_theta', 500000.0)
            rope_config = SimpleNamespace()
            rope_config.hidden_size = self.hidden_size
            rope_config.num_attention_heads = self.num_heads
            rope_config.num_key_value_heads = self.num_kv_heads
            rope_config.head_dim = self.head_dim
            rope_config.max_position_embeddings = args.max_seq_len
            rope_config.rope_theta = rope_theta
            rope_config.rope_scaling = getattr(args, 'rope_scaling', None)
            rope_config.rope_parameters = {
                "rope_type": "default",
                "rope_theta": rope_theta,
            }
            
            # Use appropriate RoPE class based on model type
            if self.model_type == "qwen3":
                self.rotary_emb = Qwen3RotaryEmbedding(config=rope_config)
            else:
                self.rotary_emb = LlamaRotaryEmbedding(config=rope_config)


    def _create_norm(self, dim: int, args: ModelArgs):
        """Modeling RMSNorm Create"""
        if self.model_type in ("qwen3", "exaone4"):
            try:
                # Qwen3RMSNorm is a plain RMSNorm; reused for exaone4 (Exaone4RMSNorm is identical).
                # The learned q_norm/k_norm weights are copied from the source model in quantizer_module.
                from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
                eps = getattr(args, 'rms_norm_eps', 1e-6)
                return Qwen3RMSNorm(hidden_size=dim, eps=eps)
            except ImportError:
                # Fallback to standard RMSNorm
                print("RMASNorm Failed!")
                return nn.LayerNorm(dim, eps=getattr(args, 'rms_norm_eps', 1e-6))
           
    # ------------------------------------------------------------------ helpers
    def update_mode(self, is_prefill: bool) -> None:
        """Switch between *prefill* and *decode* modes."""
        self.is_prefill = is_prefill

    def _detect_prefill_mode(self, past_key_value, q_len: int) -> bool:
        """
        Simple prefill/decode Mode Detection
        Args:
            past_key_value: KV cache
            q_len: Current query Length
        Returns:
            bool: True if prefill, False if decode
        """
        # Default judgment: if past_key_value is None
        if past_key_value is None:
            return True
        
        # If it's multiple tokens, most of the time it's a prefil.
        if q_len > 1:
            return True
        
        # Code for single tokens and caches
        return False

    # ------------------------ Quantisation utilities ---------------------------

    def _get_workspace(self, template: torch.Tensor, is_key: bool) -> torch.Tensor:
        """Allocate (or reuse) a tensor matching *template* for in‑place quantisation."""
        attr = "_key_workspace" if is_key else "_value_workspace"
        ws = getattr(self, attr)
        if ws is None or ws.shape != template.shape or ws.dtype != template.dtype:
            ws = torch.empty_like(template)
            setattr(self, attr, ws)
        return ws

    def _quantize_single_kv(
        self, kv_states: torch.Tensor, *, is_key: bool,
        attention_sink_num_front: Optional[int] = None,
        sparse_keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply *per‑tensor* quantisation to a Key **or** Value tensor.
        
        Args:
            attention_sink_num_front: front sink Specify token number(s).
                NonePages self.attention_sink_num Enabled (Default Action).
                0Pages front sink Quantification Without (tail For Quantification).
        """
        # In 16-bit case of quantumization (bypass)
        if hasattr(self, 'is_prefill') and self.is_prefill:
            bits = self.prefill_key_bits if is_key else self.prefill_value_bits
        else:
            bits = self.decode_key_bits if is_key else self.decode_value_bits
            
        if bits == 16:
            return kv_states

        if self.model_type in ["qwen3"]:
            original_dtype = kv_states.dtype
            
            if original_dtype == torch.bfloat16:
                kv_states = kv_states.float()
        # Select a bit according to performance/ Decode mode
        if self.is_prefill:
            if is_key:
                bits = self.prefill_key_bits if hasattr(self, 'prefill_key_bits') and self.prefill_key_bits is not None else self.key_bits
            else:
                bits = self.prefill_value_bits if hasattr(self, 'prefill_value_bits') and self.prefill_value_bits is not None else self.value_bits
        else:
            if is_key:
                bits = self.decode_key_bits if hasattr(self, 'decode_key_bits') and self.decode_key_bits is not None else self.key_bits
            else:
                bits = self.decode_value_bits if hasattr(self, 'decode_value_bits') and self.decode_value_bits is not None else self.value_bits

        # Force sliding‑window when decoding with sub‑8‑bit precision to limit
        # memory footprint of extremely long sequences.
        use_sliding_window = (
            self.use_sliding_window or (not self.is_prefill and bits < 16)
        )

        # attention_quantizer supports both tensor‑wise and channel‑wise quantisation.
        # _quant() with default args reproduces the historical call EXACTLY (OFF path is
        # byte-identical); the K-outlier plan overrides only (opg, is_prefill-routing, mask).
        def _quant(x, _opg=self.outliers_per_group, _prefill=self.is_prefill, _mask=sparse_keep_mask,
                   _opct=self.outlier_percent_topk):
            return attention_quantizer(
                inp=x,
                bits=bits,
                use_pow2=self.use_pow2,
                use_sliding_window=use_sliding_window,
                window_size=self.window_size,
                is_prefill=_prefill,
                attention_sink_num=self.attention_sink_num,
                outlier_mode=self.outlier_mode,
                outlier_percent_topk=_opct,
                outlier_percent_bottomk=self.outlier_percent_bottomk,
                use_a_shape=self.use_a_shape,
                decode_only=self.decode_only,
                is_key=is_key,
                distance_ceiling=self.distance_ceiling,
                outlier_shift=self.outlier_shift,
                # The Sliding Window FP4 connection.
                sliding_window_use_fp4_key=self.sliding_window_use_fp4_key,
                sliding_window_use_fp4_value=self.sliding_window_use_fp4_value,
                # Outlier connection
                outlier_precision_key=self.outlier_precision_key,
                outlier_precision_value=self.outlier_precision_value,
                detect_outliers_value=self.detect_outliers_value,
                outliers_per_group=_opg,
                outlier_method=self.outlier_method,
                debug_mode=self.debug_mode,
                layer_id=self.layer_idx if self.layer_idx is not None else -1,
                use_symmetric=self.use_symmetric,
                # Group Quanta Parameters Forward
                use_group_quant=self.use_group_quant,
                group_size=self.group_size,
                # Group catching support: Personal settings
                attention_sink_num_front=attention_sink_num_front,
                # fakesparse keep-mask (prefill GQA only; None => identity)
                sparse_keep_mask=_mask,
            )

        self._ensure_k_outlier_plan()
        _plan = self._k_outlier_plan if is_key else None
        if _plan is None:
            quantised = _quant(kv_states)
        else:
            # K-outlier profile active: route K through the DECODE recipe (INT2 base + per-128
            # pair FP4 outliers) with the layer/head opg — the stock prefill branch is pure FP4
            # min/max and ignores opg, so prefill-only PPL would otherwise be flat across arms.
            # V (plan None) keeps the stock path untouched (quantization-axis isolation).
            if sparse_keep_mask is not None:
                raise ValueError(
                    "OMMX_KV_OUTLIER_PROFILE cannot be combined with prefill K sparsity: the "
                    "decode-recipe branch ignores sparse_keep_mask (it would silently no-op)")
            from kv_outlier_profile import quantize_k_with_plan
            # opg==0 must be a TRUE no-outlier group: also zero the percent top-k mask, else the
            # default outlier_percent_topk (0.01) still FP4-encodes a slot and ko_uni0 / zero-budget
            # layers are not the pure-INT2 baseline the profile schema claims.
            quantised = quantize_k_with_plan(
                kv_states, _plan,
                lambda x, opg: _quant(x, _opg=opg, _prefill=False, _mask=None,
                                      _opct=(0.0 if opg == 0 else self.outlier_percent_topk)))

        if self.model_type in ["qwen3"] and original_dtype == torch.bfloat16:
            quantised = quantised.to(original_dtype)
        return quantised

    def _ensure_k_outlier_plan(self):
        """Resolve the K-outlier per-layer/per-head plan lazily on first use, when self.layer_idx
        is finally set (the Llama replacement path assigns it AFTER __init__). Idempotent; a no-op
        when OMMX_KV_OUTLIER_PROFILE is unset (OFF path stays byte-identical)."""
        if self._k_outlier_resolved:
            return
        from kv_outlier_profile import load_profile, plan_max_opg, resolve_layer
        self._k_outlier_plan = resolve_layer(load_profile(), self.layer_idx, self.outliers_per_group)
        self._k_outlier_max_opg = plan_max_opg(self._k_outlier_plan)
        self._k_outlier_resolved = True

    def _get_processing_unit_size(self) -> int:
        """From Current Configuration processing_unit_size Return (group caching boundary Compute)

        K-outlier profile: any positive per-layer/per-head K opg also pairs 64-groups into
        128-wide processing units, so the safe-reuse boundary must stay 128-aligned even when
        the uniform self.outliers_per_group is 0 (128 is 64-aligned, so this is safe for V too).
        OFF path (_k_outlier_max_opg == 0) is unchanged."""
        self._ensure_k_outlier_plan()
        if (self.outliers_per_group > 0 or self._k_outlier_max_opg > 0) and self.group_size == 64:
            return 128
        return self.group_size

    def _compute_safe_reuse_boundary(self, old_seq_len: int, new_seq_len: int, is_key: bool) -> int:
        """We can use it safely in the cache. seq Calculates the edge boundary of the position.
        
        Returns:
            safe_reuse_end: [0 : safe_reuse_end] Reuse results from cache to quantumize.
                            After this value, we need to recapitulate..
        """
        sink = self.attention_sink_num
        unit_size = self._get_processing_unit_size()
        
        if is_key:
            # Key: group formed along the threeq axis.
            # Reuse a unit complete [link: old_seq - link].
            old_middle_len = max(0, old_seq_len - 2 * sink)
            complete_units = old_middle_len // unit_size
            safe_reuse_end = sink + complete_units * unit_size
        else:
            # Value: group formed along the head_dim axis.
            # Each token is quantumized independently, so it can be reused before back_link.
            safe_reuse_end = max(0, old_seq_len - sink)
        
        return safe_reuse_end

    def _prefill_kv_sparse_masks(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Online magnitude keep-masks for prefill K/V, or (None, None) when off.

        Lazy-imports the fakesparse leaf package so a missing/disabled config costs
        nothing. K uses the conservative tier budget, V the aggressive one.
        """
        cfg = self.sparse_config
        if cfg is None or not getattr(cfg, "enable", False):
            return None, None
        from fakesparse.kv import kv_keep_mask  # leaf import (torch-only); no cycle
        li = self.layer_idx if self.layer_idx is not None else None
        k_mask = kv_keep_mask(key_states, is_key=True, cfg=cfg, layer_idx=li)
        v_mask = kv_keep_mask(value_states, is_key=False, cfg=cfg, layer_idx=li)
        return k_mask, v_mask

    @torch.no_grad()
    def _profile_kv_score(self, q, k, v, k_hat, v_hat):
        """fakesparse E0 — cheap per-layer KV sensitivity from ONE masked prefill forward.

        Computes the softmax-aware key metric `score_K = mean_i sum_j p_ij*|q_i·dK_j|/sqrt(d)` and the
        value output-error `e_o = mean(|o - o_full|/|o_full|)`, THROUGH the deployed 2-bit path:
        dK = quant(K, no-mask) - quant(K, mask)=k_hat is exactly the head_dim channels the keep-mask
        zeroed on the 2-bit K, and p_ij is the model's own attention over the masked-quant k_hat — so
        softmax exp-amplification (invisible to L1 mass) is captured by construction. Accumulated across
        calib batches into per-layer lists that wq_eval.py dumps as json via OMMX_SPARSE_SCORE_OUT
        (consumed by build_alloc_from_sens.py --score-profile). No gradients.
        q:[B,Hq,T,D]; k,v,k_hat,v_hat:[B,Hkv,T,D] (pre-GQA-repeat). Gated by OMMX_SPARSE_PROFILE_SCORE.
        """
        # cap accumulated forwards per layer (the O(T^2) probe is only needed on a few calib seqs)
        if not hasattr(self, "_score_k_acc"):
            self._score_k_acc, self._eo_acc = [], []
        if len(self._score_k_acc) >= int(os.environ.get("OMMX_SPARSE_SCORE_MAX", "") or "8"):
            return
        B, Hkv, T, D = k.shape
        g = max(1, q.shape[1] // Hkv)
        scale = 1.0 / (D ** 0.5)
        k_full = self._quantize_single_kv(k.clone(), is_key=True, sparse_keep_mask=None)
        v_full = self._quantize_single_kv(v.clone(), is_key=False, sparse_keep_mask=None)
        dK = k_full - k_hat
        causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
        sk_sum = eo_sum = 0.0
        for h in range(Hkv):
            qh = q[:, h * g:(h + 1) * g].float()                       # [B,g,T,D]
            khat = k_hat[:, h:h + 1].float()                           # [B,1,T,D] (matmul broadcasts g)
            dk = dK[:, h:h + 1].float()
            logit = torch.matmul(qh, khat.transpose(-1, -2)) * scale    # [B,g,T,T]
            logit = logit.masked_fill(~causal, float("-inf"))
            p = torch.softmax(logit, dim=-1)
            dlogit = (torch.matmul(qh, dk.transpose(-1, -2)) * scale).abs()
            sk_sum += float((p * dlogit).sum(-1).mean().item())        # sum_j p*|dlogit|, mean over i
            vhat = v_hat[:, h:h + 1].float()
            vful = v_full[:, h:h + 1].float()
            o, ofu = torch.matmul(p, vhat), torch.matmul(p, vful)
            # row-wise relative L2 per query-position output vector (element-wise |o-ofu|/|ofu|
            # blows up wherever |ofu| ~ 0)
            eo_sum += float((torch.linalg.norm(o - ofu, dim=-1)
                             / torch.linalg.norm(ofu, dim=-1).clamp_min(1e-6)).mean().item())
        self._score_k_acc.append(sk_sum / Hkv)
        self._eo_acc.append(eo_sum / Hkv)

    @torch.no_grad()
    def _capture_k_outlier_stats(self, k: torch.Tensor) -> None:
        """K per-window top-|x| mass capture for the outlier-budget profile builder.

        Gated by OMMX_KV_OUTLIER_STATS_OUT (wq_eval dumps the accumulators after PPL, like the
        E0 probe). Accumulates on prefill forwards — the calib-bake forwards run first, so the
        cap (OMMX_KV_OUTLIER_STATS_MAX, default 8, 0 = off) is filled by calib data, not the
        PPL prefills. Window = 128 tokens along the seq axis per (b, head, channel) lane — the
        decode-recipe pairwise outlier-selection domain — with front/back attention-sink tokens
        excluded, matching what the quantizer actually sees. Stored per forward as a [Hkv, 4]
        tensor: mean fraction of window |mass| captured by the top-1..top-4 elements.
        """
        if not hasattr(self, "_kout_stats"):
            self._kout_stats = []
        # direct int parse — 0 is a valid "capture off" value, never clamp (law 11)
        cap = int(os.environ.get("OMMX_KV_OUTLIER_STATS_MAX", "") or "8")
        if len(self._kout_stats) >= cap:
            return
        sink = self.attention_sink_num
        B, H, T, D = k.shape
        win = 128
        if T - 2 * sink < win:
            return
        x = k[:, :, sink:T - sink, :].abs().float()
        nw = x.shape[2] // win
        x = x[:, :, :nw * win, :].permute(0, 1, 3, 2).reshape(B, H, D, nw, win)
        tot = x.sum(-1, keepdim=True).clamp_min(1e-12)
        frac = x.topk(4, dim=-1).values.cumsum(-1) / tot          # [B,H,D,nw,4]
        self._kout_stats.append(frac.mean(dim=(0, 2, 3)).cpu())   # [Hkv,4]

    def _apply_quantisation_for_gqa(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantise Key/Value tensors *only* for attention computation.

        Group caching optimization: the already completed processing unit at Decode time
        It can catalyze the results of quantumization to prevent recalculating..
        - Key: seq axis complete unit Cassing (unit_size=128 Unit)
        - Value: Each token is independent. back_sink All the way back to Kathy's.
        """
        if self.decode_only and self.is_prefill:
            self.last_key_mse = self.last_value_mse = 0.0
            return key_states, value_states

        seq_len = key_states.shape[2]
        sink = self.attention_sink_num

        # = = = = = = = = =fpell: total quantumization (does not save in case) ==================================================+(============(=============(===========(=========(======(=========((((=====================((((=====================(((=========(((((======================================((((((((((((((========================(((((((((((((((===================================(((((((((((((((((((((((((((((((((================================================================================================================================================
        if self.is_prefill:
            # K-outlier stats capture (env-gated; raw pre-quant K, the exact |x| signal the
            # decode-recipe top-k outlier selection ranks on)
            if os.environ.get("OMMX_KV_OUTLIER_STATS_OUT", ""):
                self._capture_k_outlier_stats(key_states)
            key_clone = key_states.clone()
            value_clone = value_states.clone()
            # fakesparse: online magnitude keep-masks for prefill K/V (decode untouched).
            k_mask, v_mask = self._prefill_kv_sparse_masks(key_clone, value_clone)
            quant_k = self._quantize_single_kv(key_clone, is_key=True, sparse_keep_mask=k_mask)
            quant_v = self._quantize_single_kv(value_clone, is_key=False, sparse_keep_mask=v_mask)
            # FP4 result does not save to cache.
            # In the first decod step, inINT2 total quantumization and then cassing from the source.
            self._cached_quant_k = None
            self._cached_quant_v = None
            self._cached_seq_len = 0
            return quant_k, quant_v

        # == sync, corrected by elderman == @elder_man
        # Reuse INDI2 results (save from previous decod) if cache is available.
        # If no cache is specified, the entire INT2 → cache will be saved.
        can_use_cache = (
            self._cached_quant_k is not None
            and self._cached_seq_len > 0
            and seq_len > self._cached_seq_len
            and self.use_group_quant
        )

        if can_use_cache:
            old_len = self._cached_seq_len
            
            # Calculating the safe reuse boundary of each Key/Value
            safe_k_end = self._compute_safe_reuse_boundary(old_len, seq_len, is_key=True)
            safe_v_end = self._compute_safe_reuse_boundary(old_len, seq_len, is_key=False)

            # ---key processing ---
            if safe_k_end > 0:
                # Toil Rip (salve_k_end after)
                tail_k = key_states[:, :, safe_k_end:, :].clone()
                # Tilt quantumization: Apply only back link, without front link
                quant_tail_k = self._quantize_single_kv(
                    tail_k, is_key=True, attention_sink_num_front=0
                )
                # Results combination: cache + newly quantumized til
                quant_k = torch.cat([
                    self._cached_quant_k[:, :, :safe_k_end, :],
                    quant_tail_k
                ], dim=2)
            else:
                # No reusable part → Total quantumization
                quant_k = self._quantize_single_kv(key_states.clone(), is_key=True)

            # ---value processing--
            if safe_v_end > 0:
                tail_v = value_states[:, :, safe_v_end:, :].clone()
                quant_tail_v = self._quantize_single_kv(
                    tail_v, is_key=False, attention_sink_num_front=0
                )
                quant_v = torch.cat([
                    self._cached_quant_v[:, :, :safe_v_end, :],
                    quant_tail_v
                ], dim=2)
            else:
                quant_v = self._quantize_single_kv(value_states.clone(), is_key=False)

            # Update Cache
            self._cached_quant_k = quant_k.clone()
            self._cached_quant_v = quant_v.clone()
            self._cached_seq_len = seq_len

            return quant_k, quant_v

        # = ====Fallback: no cache → total Int2 quantumization after cassing==============
        key_clone = key_states.clone()
        value_clone = value_states.clone()
        quant_k = self._quantize_single_kv(key_clone, is_key=True)
        quant_v = self._quantize_single_kv(value_clone, is_key=False)
        # Save Int2 results to cache
        self._cached_quant_k = quant_k.clone()
        self._cached_quant_v = quant_v.clone()
        self._cached_seq_len = seq_len
        return quant_k, quant_v

    # ------------------------------- Public API --------------------------------

    def get_kv_cache_mse(self) -> Tuple[float, float]:
        """Return the *last* key/value MSE for monitoring/logging."""
        return self.last_key_mse, self.last_value_mse

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        # ── New transformers API compatibility ──
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Llama‑compatible SDPA with optional KV quantisation."""
        # Handle both old API (past_key_value) and new API (past_key_values)
        if past_key_value is None and past_key_values is not None:
            past_key_value = past_key_values

        # Check for input tenor safety
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        
        # Track Check
        if any(s < 0 for s in hidden_states.stride()):
            hidden_states = hidden_states.clone()

        batch_size, q_len, _ = hidden_states.shape
        
        # devices consistent validation and GPU priority handling
        model_device = next(self.parameters()).device
        input_device = hidden_states.device
        
        # If the model is in the CPU and the input is in the GPU, move the model to the GPU
        if model_device.type == 'cpu' and input_device.type == 'cuda':
            if self.debug_mode:
                print(f"[DEVICE] Layer {self.layer_idx}: Model on CPU but input on {input_device}, moving model to {input_device}")
            self.to(input_device)
            model_device = input_device
        elif input_device != model_device:
            if self.debug_mode:
                print(f"[DEVICE] Layer {self.layer_idx}: Input device mismatch - input: {input_device}, model: {model_device}")
                print(f"[DEVICE] Moving input to model device: {model_device}")
            hidden_states = hidden_states.to(model_device)
        
        # Adding teners confirm and move the device
        if attention_mask is not None and attention_mask.device != model_device:
            attention_mask = attention_mask.to(model_device)
        if position_ids is not None and position_ids.device != model_device:
            position_ids = position_ids.to(model_device)
        if cache_position is not None and cache_position.device != model_device:
            cache_position = cache_position.to(model_device)
        
        if self.decode_only:
            self.is_prefill = False

        if self.debug_mode:
            logger.info(
                "Layer %s – forward (prefill=%s, seq_len=%d)",
                self.layer_idx,
                self.is_prefill,
                q_len,
            )

        self.is_prefill = self._detect_prefill_mode(past_key_value, q_len)
        # ------------------------------------------------------------------ (1) Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # ------------------------------------------------------------------ (2) Reshape to [B, H, T, D]
        # qwen3 + exaone4: apply per-head QK-norm (pre-RoPE) during the reshape. RoPE below then
        # branches on model_type — qwen3 uses the qwen3 rotary, exaone4 falls to the llama path
        # (its llama3 rope_scaling is honoured by the config-built LlamaRotaryEmbedding).
        if self.model_type in ("qwen3", "exaone4"):
            # Like formula:
            q_shape = (batch_size, q_len, self.num_heads, self.head_dim)
            kv_shape = (batch_size, q_len, self.num_kv_heads, self.head_dim)

            query_states = self.q_norm(query_states.view(q_shape)).transpose(1, 2)
            key_states = self.k_norm(key_states.view(kv_shape)).transpose(1, 2)
            value_states = value_states.view(kv_shape).transpose(1, 2)
        else:
        
            # Traditional mode: rechars →nom
            query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(batch_size, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # NOTE: Qwen3 q_norm/k_norm is applied once during the reshape above (per-head,
        # pre-RoPE, matching HF Qwen3Attention). A second application here would double the
        # RMSNorm and corrupt q/k directions (wikitext PPL 9.03 -> 420). Do not re-add.

        if self.model_type == "qwen3":
            # Qween3 is used directly because the presumption_embedings are already passed
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                # Fallback: Compute directly
                if position_ids is None:
                    past = past_key_value.get_seq_length() if past_key_value else 0
                    position_ids = torch.arange(past, past+q_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0)
                cos, sin = self.rotary_emb(value_states, position_ids)  # Use value_status instead of key_sets
            
            query_states, key_states = qwen3_apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # Apply LLAMA RoPE
        else:
            # ..even if the message_embedings are delivered from the outside,
            # The layer itself is forced to use the lotary_emb because the model's runry_emb may not have been defeated.
            # (Lamamamole.Forwards often give in advance.)
            if True: # Always use internal rotary_emb to ensure scaling is applied
                cos, sin = self.rotary_emb(key_states, position_ids)
            else:
                cos, sin = position_embeddings
            if self.model_type == "phi3":
                query_states, key_states = phi3_apply_rotary_pos_emb(query_states, key_states, cos, sin)
            else:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # rotate-kv: apply the head_dim Hadamard to Q/K (post-RoPE) and V, before the KV
        # cache update + quant. R is orthonormal so scores (Q@R)(K@R).T == Q@K.T are exactly
        # preserved; V is un-rotated after SDPA (step 8). Built lazily now that device/dtype
        # are known, then cached. Applies on the last (head_dim) axis, identically per head.
        if self._rotate_kv:
            from rotation import hadamard_matrix
            if self._kv_R is None or self._kv_R.device != query_states.device:
                self._kv_R = hadamard_matrix(self.head_dim, query_states.device,
                                             query_states.dtype, seed=self._rotate_seed)
            R = self._kv_R.to(query_states.dtype)
            query_states = query_states @ R
            key_states = key_states @ R
            value_states = value_states @ R

        # ------------------------------------------------------------------ (4) Update / append to KV cache
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)  # type: ignore[arg-type]

        # ------------------------------------------------------------------ (5) Quantise copies for attention compute
        key_states_q, value_states_q = self._apply_quantisation_for_gqa(key_states, value_states)
        # fakesparse E0 probe on the pre-GQA-repeat masked-quant K/V — must precede repeat_kv below.
        if self.is_prefill and os.environ.get("OMMX_SPARSE_PROFILE_SCORE", "") not in ("", "0"):
            self._profile_kv_score(query_states, key_states, value_states, key_states_q, value_states_q)
        # ------------------------------------------------------------------ (6) GQA head replication
        if self.model_type == "qwen3":
            from transformers.models.qwen3.modeling_qwen3 import repeat_kv as qwen3_repeat_kv
            key_states_q = qwen3_repeat_kv(key_states_q, self.num_kv_groups)  # [B, H, T, D]
            value_states_q = qwen3_repeat_kv(value_states_q, self.num_kv_groups)
        else:    
            key_states_q = repeat_kv(key_states_q, self.num_kv_groups)  # [B, H, T, D]
            value_states_q = repeat_kv(value_states_q, self.num_kv_groups)

        # ------------------------------------------------------------------ (7) Attention via PyTorch SDPA (flash‑aware)
        query_states = query_states.to(value_states_q.dtype)
        key_states_q = key_states_q.to(value_states_q.dtype)
        # fakesparse Q/P sparsity (prefill only; gate + math live in fakesparse.attn_qp).
        _sc = getattr(self, "sparse_config", None)
        _q_on = _p_on = False
        if self.is_prefill and _sc is not None:
            from fakesparse.attn_qp import manual_attention_p_sparse, q_keep_mask, qp_enabled
            _q_on, _p_on = qp_enabled(_sc, self.layer_idx,
                                      getattr(self.config, "num_hidden_layers", None))
        if _q_on:
            # Cached-prefix coordinates: the first query corresponds to key Tk-Tq.
            # Dense Q sink/recent exemptions therefore stay aligned with the global
            # sequence rather than restarting at every call.
            _tk = key_states_q.shape[-2]
            _tq = query_states.shape[-2]
            query_states = query_states * q_keep_mask(
                query_states, _sc, q_offset=_tk - _tq, total_q_len=_tk
            ).to(query_states.dtype)
        if _p_on:
            attn_output = manual_attention_p_sparse(
                query_states, key_states_q, value_states_q, _sc,
                attention_mask=attention_mask,
                is_causal=self.is_causal and q_len > 1)
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states_q,
                value_states_q,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=self.is_causal and attention_mask is None and q_len > 1,
            )

        # rotate-kv: un-rotate the V-rotation out of the attention output (last dim = head_dim)
        # so o_proj sees the original basis. attn_output is [B, H, T, head_dim] here; (p@(V@R))@R.T == p@V.
        if self._rotate_kv and self._kv_R is not None:
            attn_output = attn_output @ self._kv_R.to(attn_output.dtype).t()

        # ------------------------------------------------------------------ (8) Output projection back to [B, T, D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, -1)
        attn_output = attn_output.to(self.o_proj.weight.dtype)
        attn_output = self.o_proj(attn_output)
        attn_output = attn_output.contiguous()

        if not use_cache:
            past_key_value = None
        
        attn_weights = None if not output_attentions else torch.zeros_like(query_states[:, :, 0:1, 0:1])
        
        return attn_output, attn_weights
