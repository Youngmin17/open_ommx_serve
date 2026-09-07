import os
import sys
import copy
import logging
import torch
from typing import List, Optional, Union, Dict, Any, Tuple
from transformers import AutoTokenizer, LlamaConfig

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance
from lm_eval.models.utils import configure_pad_token

eval_logger = logging.getLogger(__name__)


@register_model("kivi")
class KIVIModel(TemplateLM):
    """
    KIVI (A Tuning-Free Asymmetric 2bit Quantization for KV Cache) model wrapper for lm-evaluation-harness.
    """
    
    _DEFAULT_MAX_LENGTH = 2048
    
    def __init__(
        self,
        pretrained: str = "meta-llama/Llama-2-7b-hf",
        k_bits: int = 2,
        v_bits: int = 2,
        group_size: int = 32,
        residual_length: int = 128,
        use_flash: bool = True,
        device: str = "auto",
        dtype: Optional[Union[str, torch.dtype]] = "float16",
        batch_size: int = 1,
        max_length: Optional[int] = None,
        trust_remote_code: bool = True,
        kivi_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        # kivi_path = the vendored KIVI package dir (contains models/ + quant/).
        # Default to <repo>/baseline/kivi; override via KIVI_PATH env or model_args.
        if kivi_path is None:
            _here = os.path.dirname(os.path.abspath(__file__))
            kivi_path = os.environ.get(
                "KIVI_PATH",
                os.path.abspath(os.path.join(_here, "..", "..", "..", "baseline", "kivi")))

        # `kivi.models.*` / `kivi.quant.*` are package-qualified imports, so the PARENT
        # of the kivi package must be on sys.path.
        _kivi_parent = os.path.dirname(kivi_path)
        for p in (_kivi_parent, kivi_path):
            if p and p not in sys.path:
                sys.path.insert(0, p)
        
        # Store parameters
        self.pretrained = pretrained
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.use_flash = use_flash
        self.trust_remote_code = trust_remote_code
        self.kivi_path = kivi_path
        self.hf_token = hf_token or os.environ.get('HUGGINGFACE_HUB_TOKEN')
        self._max_length = max_length
        self.batch_size_per_gpu = batch_size
        
        # Set device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
            
        # Set dtype
        if isinstance(dtype, str):
            if dtype == "float16":
                self.dtype = torch.float16
            elif dtype == "float32":
                self.dtype = torch.float32
            elif dtype == "bfloat16":
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = dtype
        
        eval_logger.info(f"Initializing KIVI model: {pretrained}")
        eval_logger.info(f"KIVI config: k_bits={k_bits}, v_bits={v_bits}, group_size={group_size}")
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
        
        # Configure pad token
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)
        
        eval_logger.info("✓ KIVI model initialized successfully")
    
    def _load_model(self):
        """Load KIVI model"""
        # Import KIVI model class (package-qualified; the vendored kivi/ parent is on
        # sys.path from __init__). The plain llama_kivi.LlamaForCausalLM_KIVI asserts
        # num_key_value_groups==1 (MHA only) -> crashes on GQA models (Llama-3.1-8B =
        # 32 q / 8 kv). The _eval variant supports GQA + decode (generate_until/CoQA).
        from kivi.models.llama_kivi_eval import (
            LlamaForCausalLM_KIVI_eval as LlamaForCausalLM_KIVI)

        # Create KIVI config
        config = LlamaConfig.from_pretrained(
            self.pretrained,
            token=self.hf_token,
            trust_remote_code=self.trust_remote_code
        )
        config.k_bits = self.k_bits
        config.v_bits = self.v_bits
        config.group_size = self.group_size
        config.residual_length = self.residual_length
        config.use_flash = self.use_flash

        self._config = config

        # Load KIVI model
        self._model = LlamaForCausalLM_KIVI.from_pretrained(
            pretrained_model_name_or_path=self.pretrained,
            config=config,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=self.hf_token,
            trust_remote_code=self.trust_remote_code
        )

        self._model.eval()
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained, 
            use_fast=False, 
            token=self.hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @property
    def config(self):
        return self._config
    
    @property
    def model(self):
        return self._model
    
    @property
    def device(self):
        return self._device
    
    @property
    def batch_size(self):
        return self.batch_size_per_gpu
    
    @property
    def max_length(self):
        if self._max_length:
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH
    
    @property
    def max_gen_toks(self) -> int:
        return 256
    
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def prefix_token_id(self):
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
        
    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")
    
    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """Tokenize string"""
        if add_special_tokens is None:
            add_special_tokens = False
        
        encoding = self.tokenizer(
            string,
            add_special_tokens=add_special_tokens,
            truncation=False,
            return_tensors=None,
        )
        tokens = encoding["input_ids"]
        
        if left_truncate_len:
            tokens = tokens[-left_truncate_len:]
        
        return tokens
    
    def tok_decode(self, tokens, skip_special_tokens=True):
        """Decode tokens to string"""
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def _model_call(self, inps, attn_mask=None, labels=None):
        """Forward pass through the model"""
        with torch.no_grad():
            if attn_mask is not None:
                kwargs = {"attention_mask": attn_mask}
            else:
                kwargs = {}
            
            if labels is not None:
                kwargs["labels"] = labels
            
            return self.model(inps, **kwargs)
    
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """Generate text using the model"""
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)
        
        # Handle temperature and sampling
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        
        # Set max_new_tokens instead of max_length
        generation_kwargs["max_new_tokens"] = max_length
        generation_kwargs.pop("max_length", None)
        
        # Set pad_token_id only if not already present
        if "pad_token_id" not in generation_kwargs:
            generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                context,
                **generation_kwargs
            )
        
        return outputs
    
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        """Compute log-likelihood for tokens (required abstract method)"""
        def _collate(req):
            # req is a tuple of (context, continuation) strings and token lists
            toks = req[1] + req[2]  # context_tokens + continuation_tokens  
            return (-len(req[1]), tuple(toks))
        
        results = []
        
        # Process requests in batches
        for request in requests:
            context_tokens, continuation_tokens = request[1], request[2]
            
            # Combine context and continuation
            input_tokens = context_tokens + continuation_tokens
            input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
            
            # Create labels (only compute loss on continuation tokens)
            labels = input_ids.clone()
            labels[0, :len(context_tokens)] = -100  # Ignore context tokens in loss computation
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids, labels=labels)
                logits = outputs.logits
            
            # Compute log probabilities for continuation tokens
            continuation_logits = logits[0, len(context_tokens)-1:-1]  # Shift by 1 for autoregressive
            continuation_labels = torch.tensor(continuation_tokens, dtype=torch.long, device=self.device)
            
            # Calculate log probabilities
            log_probs = torch.nn.functional.log_softmax(continuation_logits, dim=-1)
            token_log_probs = log_probs.gather(1, continuation_labels.unsqueeze(1)).squeeze(1)
            
            # Sum log probabilities for the continuation
            total_log_prob = token_log_probs.sum().item()
            
            # Check if all tokens were predicted correctly (greedy decoding check)
            predicted_tokens = continuation_logits.argmax(dim=-1)
            is_greedy = torch.equal(predicted_tokens, continuation_labels)
            
            results.append((total_log_prob, is_greedy))
        
        return results
    
    def loglikelihood_rolling(self, requests: List[Instance], disable_tqdm: bool = False) -> List[float]:
        """Compute rolling window log-likelihood"""
        loglikelihoods = []
        
        for request in requests:
            string = request.args[0]
            
            # Tokenize
            encoding = self.tokenizer(
                string,
                add_special_tokens=True,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(self.device)
            
            # Compute log-likelihood
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                log_likelihood = -outputs.loss.item() * input_ids.size(1)
            
            loglikelihoods.append(log_likelihood)
        
        return loglikelihoods
    
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        """Generate text until stopping criteria"""
        def _collate(req):
            context, gen_kwargs = req.args[0], req.args[1]
            context_enc = self.tok_encode(context)
            return context, context_enc, gen_kwargs
        
        results = []
        
        for request in requests:
            context, context_enc, gen_kwargs = _collate(request)
            
            # Prepare input
            context_tokens = torch.tensor([context_enc], dtype=torch.long, device=self.device)
            
            # Set generation parameters
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)
            do_sample = gen_kwargs.get("do_sample", False)
            
            generation_kwargs = {
                "max_new_tokens": max_gen_toks,
                "temperature": temperature,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            
            # Generate
            outputs = self._model_generate(
                context_tokens,
                max_length=max_gen_toks,
                stop=gen_kwargs.get("until", []),
                **generation_kwargs
            )
            
            # Decode generated tokens (exclude context)
            generated_tokens = outputs[0][len(context_enc):]
            generated_text = self.tok_decode(generated_tokens.tolist())
            
            # Handle stop sequences
            stop_sequences = gen_kwargs.get("until", [])
            for stop in stop_sequences:
                if stop in generated_text:
                    generated_text = generated_text.split(stop)[0]
                    break
            
            results.append(generated_text)
        
        return results 