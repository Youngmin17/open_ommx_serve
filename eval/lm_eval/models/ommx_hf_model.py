"""lm-eval adapter for the HF-eager SHADOW-FREE OMMX KV-quant Llama (LlamaForCausalLM_OMMX).

Measures the REAL OMMX decode kernel's generation accuracy (CoQA), comparable to the
kivi/kitty adapters (same lm-eval coqa task). OMMX's modeling integrates an HF Cache
(OmmxSeqCounterCache) so HF `.generate()` works (verified: greedy output == bf16). Each
request rebuilds the per-layer CanonicalKVStore at prefill, so requests don't pollute.

SHADOW-FREE: env OMMX_KV_RING=1 (store keeps only sink+recent bf16 + quantized bulk, no
full bf16 shadow). Requires the `ommx` env (ommx_gpu_serve + transformers>=4.52) and the
synced repo on PYTHONPATH. Prefill uses sdpa (flash_attn optional).

model_args: pretrained, ommx_modeling_path (dir with _ommx_llama_modeling.py), dtype,
            max_length, device.
"""
import os
import sys
from typing import List, Optional

import torch

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance
from ommx_gpu_serve.integration.vllm.config import DEFAULT_OUTLIER_REPR

# Canonical best recipe — 12/64 FP4 K-outliers (6 per 32-ch group), pow2 scales — IDENTICAL
# to figure/bench.py:90 and README §"What reproduces", so the shipped CoQA accuracy arm
# matches every published OMMX number. (Set before the modeling import reads them.)
for _k, _v in {
    "OMMX_ATTN_K_FORMAT": "i2f4", "OMMX_ATTN_OUTLIERS": "6", "OMMX_ATTN_POW2": "1",
    "OMMX_ATTN_OUTLIER_SELECT": "signed", "OMMX_ATTN_OUTLIER_REPR": DEFAULT_OUTLIER_REPR,
    "OMMX_KV_GROUP_TOKENS": "32", "OMMX_KV_GROUP_CHANNELS": "32",
    "OMMX_KV_SINK": "8", "OMMX_KV_RECENT": "32",
    "OMMX_KV_RING": "1", "OMMX_KV_GPU_PACK": "1",
}.items():
    os.environ.setdefault(_k, _v)


@register_model("ommx_hf")
class OMMXHFModel(TemplateLM):
    _DEFAULT_MAX_LENGTH = 4096

    def __init__(
        self,
        pretrained: str = "meta-llama/Llama-3.1-8B-Instruct",
        ommx_modeling_path: Optional[str] = None,
        dtype: str = "float16",
        device: str = "cuda",
        batch_size: int = 1,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.pretrained = pretrained
        self._device = device
        self._max_length = int(max_length) if max_length else None
        self._dtype = getattr(torch, dtype)
        if ommx_modeling_path is None:
            _here = os.path.dirname(os.path.abspath(__file__))
            ommx_modeling_path = os.environ.get(
                "OMMX_MODELING_PATH",
                os.path.abspath(os.path.join(
                    _here, "..", "..", "..", "ommx_gpu_serve", "hf_eager")))
        if ommx_modeling_path and ommx_modeling_path not in sys.path:
            sys.path.insert(0, ommx_modeling_path)
        from transformers import LlamaConfig, AutoTokenizer
        from _ommx_llama_modeling import LlamaForCausalLM_OMMX  # noqa: E402

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            import flash_attn  # noqa
            attn = "flash_attention_2"
        except Exception:
            attn = "sdpa"
        cfg = LlamaConfig.from_pretrained(pretrained)
        cfg._attn_implementation = attn
        self._config = cfg
        self._model = LlamaForCausalLM_OMMX.from_pretrained(
            pretrained, config=cfg, torch_dtype=self._dtype, low_cpu_mem_usage=True,
            attn_implementation=attn).to(device).eval()

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def max_length(self):
        if self._max_length:
            return self._max_length
        for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
            if hasattr(self._config, attr):
                return getattr(self._config, attr)
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        return self.tokenizer.bos_token_id or self.tokenizer.eos_token_id

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        ids = self.tokenizer(string, add_special_tokens=bool(add_special_tokens),
                             truncation=False, return_tensors=None)["input_ids"]
        if left_truncate_len:
            ids = ids[-left_truncate_len:]
        return ids

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        results = []
        for request in requests:
            context, gen_kwargs = request.args[0], request.args[1]
            until = list(gen_kwargs.get("until", []) or [])
            max_gen = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            ctx_ids = self.tok_encode(context, add_special_tokens=True)
            ctx_ids = ctx_ids[-(self.max_length - max_gen - 8):]
            ids = torch.tensor([ctx_ids], dtype=torch.long, device=self._device)
            # OMMX modeling rebuilds its per-layer store at prefill -> fresh KV per request.
            out = self._model.generate(
                ids, max_new_tokens=max_gen, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id)
            text = self.tok_decode(out[0][len(ctx_ids):])
            for s in until:
                if s and s in text:
                    text = text.split(s)[0]
                    break
            results.append(text)
        return results

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("OMMXHFModel supports generate_until tasks only (coqa).")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("OMMXHFModel supports generate_until tasks only.")
