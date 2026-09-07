"""lm-eval adapter for Kitty (2-bit paged KV). generate_until only (CoQA / generation).

Kitty's KittyCache is a static-page cache driven by a MANUAL decode loop (see
baseline/kitty/_kitty_tpot_bench.py) — it does NOT integrate with HF `.generate()`
(which would allocate its own cache and bypass the 2-bit quant). So generate_until here
prefills the context into a fresh KittyCache then greedily decodes token-by-token through
the Kitty 2-bit attention kernel, which is exactly the quantized-KV decode path under test.

Requires transformers 4.53.2 (the version the Kitty modeling pins: masking_utils +
LossKwargs + CacheConfig). Pass it on PYTHONPATH ahead of the env transformers.

model_args: pretrained, kitty_modeling_path (dir with _kitty_llama_modeling.py),
            kitty_pkg_path (dir with the `kitty` package), dtype, max_length, device.
"""
import os
import sys
from typing import List, Optional

import torch

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance


@register_model("kitty")
class KittyModel(TemplateLM):
    _DEFAULT_MAX_LENGTH = 4096

    def __init__(
        self,
        pretrained: str = "meta-llama/Llama-3.1-8B-Instruct",
        kitty_modeling_path: Optional[str] = None,
        kitty_pkg_path: Optional[str] = None,
        dtype: str = "float16",
        device: str = "cuda",
        batch_size: int = 1,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.pretrained = pretrained
        self._device = device
        self.batch_size_per_gpu = int(batch_size)
        self._max_length = int(max_length) if max_length else None
        self._dtype = getattr(torch, dtype)

        if kitty_modeling_path is None:
            _here = os.path.dirname(os.path.abspath(__file__))
            kitty_modeling_path = os.path.abspath(os.path.join(
                _here, "..", "..", "..", "baseline", "kitty"))
        if kitty_pkg_path is None:
            # the upstream `kitty` package is NOT vendored; provide it via env or install.
            kitty_pkg_path = os.environ.get("KITTY_PKG_PATH", "")
        for p in (kitty_pkg_path, kitty_modeling_path):
            if p and p not in sys.path:
                sys.path.insert(0, p)
        from transformers import LlamaConfig, AutoTokenizer
        from _kitty_llama_modeling import LlamaForCausalLM_Kitty  # noqa: E402

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cfg = LlamaConfig.from_pretrained(pretrained)
        cfg._attn_implementation = "flash_attention_2"
        self._config = cfg
        self._model = LlamaForCausalLM_Kitty.from_pretrained(
            pretrained, config=cfg, torch_dtype=self._dtype, low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2").to(device).eval()
        from kitty.kvcache import get_kvcache_kitty  # noqa: E402
        self._get_kv = get_kvcache_kitty

    # ── properties ───────────────────────────────────────────────────────────
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

    # ── generation (manual Kitty decode) ─────────────────────────────────────
    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        results = []
        for request in requests:
            context, gen_kwargs = request.args[0], request.args[1]
            until = list(gen_kwargs.get("until", []) or [])
            max_gen = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            ctx_ids = self.tok_encode(context, add_special_tokens=True)
            # leave room: prompt + generated tokens must fit the static Kitty pages
            ctx_ids = ctx_ids[-(self.max_length - max_gen - 8):]
            ids = torch.tensor([ctx_ids], dtype=torch.long, device=self._device)
            kv = self._get_kv(self._config, max_batch_size=1,
                              max_length=len(ctx_ids) + max_gen + 8)
            out = self._model(input_ids=ids, past_key_values=kv, use_cache=True)
            nt = out.logits[:, -1:, :].argmax(dim=-1)
            gen_ids = []
            for _ in range(max_gen):
                tid = int(nt.item())
                if tid == self.eot_token_id:
                    break
                gen_ids.append(tid)
                text_so_far = self.tok_decode(gen_ids)
                if any(s in text_so_far for s in until):
                    break
                o = self._model(input_ids=nt, past_key_values=kv, use_cache=True)
                nt = o.logits[:, -1:, :].argmax(dim=-1)
            text = self.tok_decode(gen_ids)
            for s in until:
                if s and s in text:
                    text = text.split(s)[0]
                    break
            results.append(text)
        return results

    # ── CoQA needs only generate_until; loglikelihood is not implemented ──────
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("KittyModel supports generate_until tasks only (e.g. coqa).")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("KittyModel supports generate_until tasks only.")
