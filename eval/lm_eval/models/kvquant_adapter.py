
from __future__ import annotations

import os
import sys
from typing import Optional, Union, Literal, Any
import logging

import torch
import transformers
from lm_eval.api.registry import register_model

from lm_eval.models.huggingface import HFLM  

# Add the quant directory to the Python path to resolve local imports (utils, quantizer_module)
QUANT_DIR = os.environ.get(
    "OMMX_FAKEQUANT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../ommx_fakequant')))
if QUANT_DIR not in sys.path:
    sys.path.insert(0, QUANT_DIR)

from quantizer_module import apply_kv_cache_quant
from quant_weight import apply_weight_quant

logger = logging.getLogger(__name__)


def _to_bool(x: Any) -> bool:
    """lm‑eval passes every model_arg as *str* unless explicitly numeric.
    This helper converts common truthy strings/ints to bool."""
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x).lower() in {"1", "true", "t", "yes", "y"}

def _to_int_or_none(x: Any) -> Optional[int]:
    """Convert string/int to int or None if empty/invalid."""
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        x = x.strip()
        if x == '' or x.lower() == 'none':
            return None
        try:
            return int(x)
        except ValueError:
            return None
    return None


@register_model("kvquant_v2")
class KVQuantLM(HFLM):
    """HuggingFace model with KV‑cache quantisation."""

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM  # always decoder‑only

    # fmt: off
    def __init__(
        self,
        pretrained: str,
        *,
        # Quantisation hyper‑parameters -----------------------------
        bits: int = 2,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        prefill_key_bits: Optional[int] = None,
        prefill_value_bits: Optional[int] = None,
        decode_key_bits: Optional[int] = None,
        decode_value_bits: Optional[int] = None,
        prefill_use_fp4: Union[bool, str, int] = False,
        use_pow2: Union[bool, str, int] = False,
        use_sliding_window: Union[bool, str, int] = False,
        window_size: int = 32,
        #  Group quantisation options
        use_group_quant: Union[bool, str, int] = False,
        group_size: int = 1024,
        attention_sink_num: int = 5,
        outlier_mode: str = "elementwise",
        outlier_percent_topk: float = 0.01,
        outlier_percent_bottomk: float = 0.0,
        outlier_shift: Union[bool, str, int] = False,
        use_a_shape: Union[bool, str, int] = False,
        decode_only: Union[bool, str, int] = False,
        distance_ceiling: int = 32,
        debug_mode: Union[bool, str, int] = False,
        max_gen_toks: int = 32,  # 생성 관련 파라미터 (HFLM에 전달하지 않음)
        #  Sliding Window FP4 관련 파라미터
        sliding_window_use_fp4_key: Union[bool, str, int] = False,
        sliding_window_use_fp4_value: Union[bool, str, int] = False,
        #  Outlier 관련 파라미터
        outlier_precision_key: str = "int",
        outlier_precision_value: str = "int",
        detect_outliers_key: Union[bool, str, int] = True,
        detect_outliers_value: Union[bool, str, int] = True,
        outlier_method: str = "original",
        outliers_per_group: Union[int, str] = 0,
        #  Decode FP4 강제 옵션
        decode_use_fp4_key: Union[bool, str, int] = False,
        decode_use_fp4_value: Union[bool, str, int] = False,
        #  Weight Quantization options
        use_weight_quant: Union[bool, str, int] = False,
        weight_bits: int = 2,
        weight_group_size: int = 64,
        weight_outlier_group_size: Optional[int] = None,
        weight_outlier_percent: float = 0.01,
        weight_lr_path: Optional[str] = None,
        # ---- All *other* kwargs forwarded to HFLM ------------------
        **hf_kwargs,
    ) -> None:
        # fmt: on
        # ------------------------------------------------------------------
        # 1.  Convert stringy booleans coming from harness into actual bools
        # ------------------------------------------------------------------
        use_pow2 = _to_bool(use_pow2)
        use_sliding_window = _to_bool(use_sliding_window)
        outlier_shift = _to_bool(outlier_shift)
        use_a_shape = _to_bool(use_a_shape)
        decode_only = _to_bool(decode_only)
        debug_mode = _to_bool(debug_mode)
        prefill_use_fp4 = _to_bool(prefill_use_fp4)
        use_group_quant = _to_bool(use_group_quant)
        use_weight_quant = _to_bool(use_weight_quant)
        
        #  Sliding Window FP4 파라미터 파싱
        sliding_window_use_fp4_key = _to_bool(sliding_window_use_fp4_key)
        sliding_window_use_fp4_value = _to_bool(sliding_window_use_fp4_value)
        
        #  Outlier 파라미터 파싱
        detect_outliers_key = _to_bool(detect_outliers_key)
        detect_outliers_value = _to_bool(detect_outliers_value)
        outliers_per_group = _to_int_or_none(outliers_per_group) or 0
        
        # 🔧 INT 파라미터 타입 변환 (빈 문자열 처리)
        prefill_key_bits = _to_int_or_none(prefill_key_bits)
        prefill_value_bits = _to_int_or_none(prefill_value_bits)
        decode_key_bits = _to_int_or_none(decode_key_bits)
        decode_value_bits = _to_int_or_none(decode_value_bits)
        key_bits = _to_int_or_none(key_bits)
        value_bits = _to_int_or_none(value_bits)
        weight_outlier_group_size = _to_int_or_none(weight_outlier_group_size)
        # weight_lr_path: 빈 문자열을 None으로 변환
        if isinstance(weight_lr_path, str) and weight_lr_path.strip() == '':
            weight_lr_path = None
        if weight_lr_path:
            logger.info(f"[KVQuantLM] Loading calibrated V params from: {weight_lr_path}")
        
        # ✅ disable_exllama 파라미터 처리 (HFLM에 직접 전달하면 TypeError 발생 가능)
        disable_exllama = hf_kwargs.pop("disable_exllama", False)
        if isinstance(disable_exllama, str):
            disable_exllama = disable_exllama.lower() == "true"

        # 생성 관련 파라미터 저장 (HFLM에 전달하지 않음)
        self._max_gen_toks = max_gen_toks

        # ------------------------------------------------------------------
        # 2.  Let the parent class do the heavy lifting – downloads model,
        #     builds tokenizer, sets up Accelerate, etc.
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # 2. Load and modify config BEFORE calling super().__init__
        # ------------------------------------------------------------------
        from transformers import AutoConfig
        
        logger.info(f"[KVQuantLM] Loading config for {pretrained}")
        config = AutoConfig.from_pretrained(pretrained)
        
        # ================================================================
        # 🔧 FIX: Auto-detect and apply RoPE scaling for 32K models
        # ================================================================
        is_together_32k = "togethercomputer" in pretrained.lower() and "32k" in pretrained.lower()
        
        if is_together_32k:
            logger.info(f"[KVQuantLM] Detected Together 32K model")
            
            # 1. RoPE scaling
            if not hasattr(config, 'rope_scaling') or config.rope_scaling is None:
                config.rope_scaling = {"type": "linear", "factor": 8.0}
                logger.info(f"[KVQuantLM] Applied RoPE scaling: {config.rope_scaling}")
            
            # 2. rope_theta
            if not hasattr(config, 'rope_theta'):
                config.rope_theta = 10000
                logger.info(f"[KVQuantLM] Set rope_theta: 10000")
            
            # 3. max_position_embeddings
            if config.max_position_embeddings < 32768:
                config.max_position_embeddings = 32768
                logger.info(f"[KVQuantLM] Set max_position_embeddings: 32768")
        
        # ✅ GPTQ disable_exllama 적용
        if disable_exllama and hasattr(config, "quantization_config"):
            logger.info("[KVQuantLM] Disabling Exllama backend as requested")
            if isinstance(config.quantization_config, dict):
                config.quantization_config["disable_exllama"] = True
            else:
                setattr(config.quantization_config, "disable_exllama", True)
             
        #  캐시 디렉토리 설정 (환경변수 우선, 미설정시 기본값 사용)
        os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        os.environ.setdefault('TORCH_HOME', os.path.expanduser('~/.cache/torch'))
        os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(os.environ['HF_HOME'], 'datasets'))
        logger.info(f"[KVQuantLM] Cache directories: HF_HOME={os.environ['HF_HOME']}")
        
        #  GPU 메모리 최적화 설정
        if 'device_map' not in hf_kwargs:
            # 병렬 처리 대신 단일 GPU 사용 (사용자가 요청한 대로 단순화)
            hf_kwargs['device_map'] = "auto"
            logger.info(f"[KVQuantLM] Using default device_map='auto'")

        # ✅ Low CPU memory 옵션 (샤드별 순차 로딩으로 Commit 공간 절약)
        if 'low_cpu_mem_usage' not in hf_kwargs:
            hf_kwargs['low_cpu_mem_usage'] = True

        # ✅ Offload folder 설정 (극단적 OOM 대비)
        if 'offload_folder' not in hf_kwargs:
            offload_folder = os.path.join(QUANT_DIR, '.offload')
            os.makedirs(offload_folder, exist_ok=True)
            hf_kwargs['offload_folder'] = offload_folder
        
        #  HFLM의 dtype 매개변수를 사용하여 torch_dtype 설정 (중복 전달 방지)
        if 'dtype' in hf_kwargs:
            del hf_kwargs['dtype']

        super().__init__(
            pretrained=pretrained,
            config=config,
            backend="causal",  # kvquant currently targets decoder‑only LLMs
            dtype=torch.bfloat16,
            **hf_kwargs,
        )
        
        # 모델 로드 후 상태 확인
        if hasattr(self, '_model') and self._model is not None:
            model_dtype = next(self._model.parameters()).dtype
            logger.info(f"[KVQuantLM] Model loaded. Dtype: {model_dtype}, Device: {self._model.device}")
            
            if model_dtype == torch.float32:
                logger.warning("[KVQuantLM] Model is in FP32! Converting to BFloat16 to save memory...")
                self._model = self._model.to(torch.bfloat16)

            # ✅ Gradient checkpointing 활성화 (메모리 절약)
            try:
                self._model.gradient_checkpointing_enable()
                logger.info("[KVQuantLM] Gradient checkpointing enabled")
            except Exception as e:
                logger.warning(f"[KVQuantLM] Could not enable gradient checkpointing: {e}")

        # ------------------------------------------------------------------
        # 3.  Apply our KV‑cache quantisation in‑place.  We operate on the
        #     *unwrapped* model (self.model property does that for us).
        # ------------------------------------------------------------------
        logger.info("[KVQuantLM] Applying KV‑cache quantisation …")
        config = getattr(self.model, 'config', None)
        model_name_or_path = getattr(config, '_name_or_path', pretrained) if config else pretrained
        
        if bits!=16:
            apply_kv_cache_quant(
                self.model,
                bits=bits,
                key_bits=key_bits,
                value_bits=value_bits,
                prefill_key_bits=prefill_key_bits,
                prefill_value_bits=prefill_value_bits,
                decode_key_bits=decode_key_bits,
                decode_value_bits=decode_value_bits,
                prefill_use_fp4=prefill_use_fp4,
                use_pow2=use_pow2,
                use_sliding_window=use_sliding_window,
                window_size=window_size,
                # forward group quant options
                use_group_quant=use_group_quant,
                group_size=group_size,
                #  Sliding Window FP4 매개변수 추가
                sliding_window_use_fp4_key=sliding_window_use_fp4_key,
                sliding_window_use_fp4_value=sliding_window_use_fp4_value,
                #  Outlier 매개변수 추가
                outlier_precision_key=outlier_precision_key,
                outlier_precision_value=outlier_precision_value,
                detect_outliers_key=detect_outliers_key,
                detect_outliers_value=detect_outliers_value,
                outlier_method=outlier_method,
                outliers_per_group=outliers_per_group,
                decode_use_fp4_key=decode_use_fp4_key,
                decode_use_fp4_value=decode_use_fp4_value,
                outlier_percent_topk=outlier_percent_topk,
                outlier_percent_bottomk=outlier_percent_bottomk,
                outlier_shift=outlier_shift,
                use_a_shape=use_a_shape,
                decode_only=decode_only,
                debug_mode=debug_mode,
                attention_sink_num=attention_sink_num,
                distance_ceiling=distance_ceiling,
                outlier_mode=outlier_mode,
            )

        # ------------------------------------------------------------------
        # 4. Apply Weight Quantization if requested
        # ------------------------------------------------------------------
        if use_weight_quant:
            logger.info(f"[KVQuantLM] Applying weight quantisation (bits={weight_bits}, group={weight_group_size}, outlier={weight_outlier_percent}) …")
            apply_weight_quant(
                self.model,
                bits=weight_bits,
                group_size=weight_group_size,
                outlier_group_size=weight_outlier_group_size,
                outlier_percent=weight_outlier_percent,
                use_pow2=use_pow2,
                v_params_path=weight_lr_path,
                debug_mode=debug_mode
            )

        # Tie weights, switch to eval, etc. (HFLM already did this, but we
        # might have swapped modules – be safe.)
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            try:
                self.model.tie_weights()
            except Exception:
                pass

        logger.info("[KVQuantLM] Quantisation complete – ready for evaluation.")

    @property
    def max_gen_toks(self) -> int:
        """Override max_gen_toks property to return our configured value"""
        return getattr(self, '_max_gen_toks', 32)

    @max_gen_toks.setter
    def max_gen_toks(self, value: int):
        """Allow setting max_gen_toks"""
        self._max_gen_toks = value
