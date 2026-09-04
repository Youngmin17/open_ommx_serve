# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Mistral model."""
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from kivi.quant.new_pack import triton_quantize_and_pack_along_last_dim, unpack_and_dequant_kcache, unpack_and_dequant_vcache, unpack_and_dequant_triton_packed
from kivi.quant.matmul import cuda_bmm_fA_qB_outer, triton_bmm_fA_qB_outer

from transformers.models.mistral.configuration_mistral import *
from transformers.models.mistral.modeling_mistral import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.cache_utils import Cache

_CONFIG_FOR_DOC = "MistralConfig"

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def repeat_kv_quant(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _normalize_legacy_cache(past_key_values):
    if isinstance(past_key_values, Cache):
        legacy_cache = past_key_values.to_legacy_cache()
        return legacy_cache if len(legacy_cache) > 0 else None
    return past_key_values


def _apply_mistral_rotary(rotary_emb, value_states, position_ids, seq_len):
    try:
        if position_ids is not None:
            return rotary_emb(value_states, position_ids)
        return rotary_emb(value_states, seq_len=seq_len)
    except TypeError:
        return rotary_emb(value_states, seq_len=seq_len)


def _has_non_finite(tensor: Optional[torch.Tensor]) -> bool:
    return tensor is not None and not torch.isfinite(tensor).all()


# --- fused-KIVI path accounting, SHARED with the Llama sibling ---------------------------
# llama_kivi_eval.py owns the fused-GQA extension, the fp16 gate and the provenance counters
# (read its PATH ACCOUNTING block for the full contract). This module imports them instead of
# cloning them, for one reason: the counters must be process-wide. A duplicated counter set
# would let a Mistral run report path=="unused" from kivi.models.llama_kivi_eval.kivi_fused_stats
# - the accessor every harness in this repo actually calls (figure/bench.py:260) - while a
# fused kernel was firing on every decode step of every layer. That is exactly the misreport
# this accounting exists to prevent, so the two modules deliberately share one set of globals.
#
# Import direction is clean: llama_kivi_eval imports nothing from this module (nor from any
# other kivi.models module), so there is no cycle. Both files are siblings inside the
# kivi.models package (baseline/kivi/models/__init__.py exists; baseline/kivi itself is a PEP
# 420 namespace package reached with baseline/ on sys.path). The relative import is tried
# first because it is independent of what the top-level package is named; the absolute form
# covers a caller that put baseline/kivi/models on sys.path directly. If BOTH fail the
# ImportError propagates - a Mistral run whose path evidence is missing must not start.
try:
    from .llama_kivi_eval import (
        _fused_bmm_padM,
        _fused_gqa_enabled,
        _kivi_chunked_prefill_fires,
        _kivi_check_fp16,
        _kivi_direct_fused_fires,
        _kivi_direct_fused_notice_once,
        _kivi_fallback_fires,
        _kivi_fused_fires,
        _kivi_fused_strict,
        _kivi_path_notice_once,
        _kivi_record_fused_error,
        _repeat_head_contig,
        kivi_fused_stats,
        kivi_reset_fused_stats,
    )
except ImportError:
    from kivi.models.llama_kivi_eval import (
        _fused_bmm_padM,
        _fused_gqa_enabled,
        _kivi_chunked_prefill_fires,
        _kivi_check_fp16,
        _kivi_direct_fused_fires,
        _kivi_direct_fused_notice_once,
        _kivi_fallback_fires,
        _kivi_fused_fires,
        _kivi_fused_strict,
        _kivi_path_notice_once,
        _kivi_record_fused_error,
        _repeat_head_contig,
        kivi_fused_stats,
        kivi_reset_fused_stats,
    )

# kivi_fused_stats / kivi_reset_fused_stats are re-exported on purpose: a Mistral harness that
# does `from kivi.models.mistral_kivi_eval import kivi_fused_stats` gets the same counters as a
# Llama one, not a second empty set. __all__ is not defined in this module (it uses star
# imports from transformers), so the names being module attributes is what makes that work.


def _fallback_key_quant_matmul(query_states, key_states_quant_trans, key_scale_trans, key_mn_trans, group_size, bits, n_rep):
    # fp16 gate first, before any work and before the notice: both branches below are fp16-only
    # (unpack_and_dequant_triton_packed hard-casts to torch.float16 at new_pack.py:110, and the
    # fused kernel allocates a torch.float16 output at matmul.py:254). Nothing is cast here.
    _kivi_check_fp16("mistral._fallback_key_quant_matmul",
                     query_states=query_states, key_scale_trans=key_scale_trans,
                     key_mn_trans=key_mn_trans)
    _kivi_path_notice_once(n_rep)
    if _fused_gqa_enabled():
        try:
            qk = _repeat_head_contig(key_states_quant_trans, n_rep)   # [B,H,D,T//feat]
            sc = _repeat_head_contig(key_scale_trans, n_rep)          # [B,H,D,G]
            zp = _repeat_head_contig(key_mn_trans, n_rep)             # [B,H,D,G]
            out = _fused_bmm_padM(group_size, query_states, qk, sc, zp, bits)
            _kivi_fused_fires[0] += 1
            return out
        except Exception as exc:
            # No silent fallback: record type/message/frame (and warn once) before demoting.
            # The site string is prefixed "mistral." so a record cannot be confused with the
            # identically-named Llama helper when both models ran in one process.
            _kivi_record_fused_error("mistral._fallback_key_quant_matmul", exc)
            if _kivi_fused_strict():
                raise
    # Fallback: dequantize to fp16, expand, dense matmul — safe for GQA head expansion.
    _kivi_fallback_fires[0] += 1
    key_states_trans = unpack_and_dequant_triton_packed(key_states_quant_trans, key_scale_trans, key_mn_trans, group_size, bits)
    key_states_trans = repeat_kv_quant(key_states_trans, n_rep)
    return torch.matmul(query_states, key_states_trans)


def _fallback_value_quant_matmul(attn_weights, value_states_quant, value_scale, value_mn, group_size, bits, n_rep):
    # fp16 gate first: attn_weights carries the model dtype (softmax is cast back to
    # query_states.dtype at every call site), value_scale/value_mn carry the cache dtype.
    _kivi_check_fp16("mistral._fallback_value_quant_matmul",
                     attn_weights=attn_weights, value_scale=value_scale, value_mn=value_mn)
    _kivi_path_notice_once(n_rep)
    if _fused_gqa_enabled():
        try:
            vq = _repeat_head_contig(value_states_quant, n_rep)       # [B,H,T_v,D//feat]
            sc = _repeat_head_contig(value_scale, n_rep)             # [B,H,T_v,G]
            zp = _repeat_head_contig(value_mn, n_rep)                # [B,H,T_v,G]
            out = _fused_bmm_padM(group_size, attn_weights, vq, sc, zp, bits)
            _kivi_fused_fires[0] += 1
            return out
        except Exception as exc:
            # No silent fallback: record type/message/frame (and warn once) before demoting.
            _kivi_record_fused_error("mistral._fallback_value_quant_matmul", exc)
            if _kivi_fused_strict():
                raise
    # Fallback: dequantize to fp16, expand, dense matmul — safe for GQA head expansion.
    _kivi_fallback_fires[0] += 1
    value_states = unpack_and_dequant_triton_packed(value_states_quant, value_scale, value_mn, group_size, bits)
    value_states = repeat_kv_quant(value_states, n_rep)
    return torch.matmul(attn_weights, value_states)


_PREFILL_CHUNK_THRESHOLD = 4096
_PREFILL_CHUNK_SIZE       = 512


def _chunked_prefill_attention(
    query_states, key_states_quant_trans, key_scale_trans, key_mn_trans,
    key_states_full, value_states_quant, value_scale, value_mn, value_states_full,
    group_size, k_bits, v_bits, n_rep, softmax_scale, attention_mask,
):
    B, H, S, D = query_states.shape
    dtype = query_states.dtype

    # This path has NO fused variant: it always dequantizes the packed KV to fp16 and expands
    # it n_rep-fold before a dense matmul, whatever KIVI_FUSED_GQA says. Count and announce it
    # so a long-context run cannot report path=="fused" while its prefill ran the unfused path
    # (which at these contexts is also the FASTER path - see the measured table in the Llama
    # sibling's PATH ACCOUNTING; "unfused" here is provenance, never a performance claim).
    #
    # State it plainly, because this is the regime the published contexts live in: THE LONG
    # PREFILL IS UNFUSED IN BOTH MODES. Flipping KIVI_FUSED_GQA does not change a single
    # instruction executed here (and with use_flash=True prefill does not even reach this
    # function - MistralFlashAttention_KIVI runs flash_attn on the un-quantised tensors, which
    # is equally gate-independent). Therefore TTFT is identical BY CONSTRUCTION between the two
    # modes, and a fused-vs-fallback comparison at ctx > _PREFILL_CHUNK_THRESHOLD is a
    # DECODE-ONLY comparison. Report such a ratio as TPOT, never as end-to-end or TTFT.
    if key_states_quant_trans is not None or value_states_quant is not None:
        _kivi_check_fp16("mistral._chunked_prefill_attention",
                         query_states=query_states, key_scale_trans=key_scale_trans,
                         value_scale=value_scale)
        _kivi_path_notice_once(n_rep)
        _kivi_chunked_prefill_fires[0] += 1
        _kivi_fallback_fires[0] += 1

    if key_states_quant_trans is not None:
        key_q = unpack_and_dequant_triton_packed(key_states_quant_trans, key_scale_trans, key_mn_trans, group_size, k_bits)
        key_q = repeat_kv_quant(key_q, n_rep)
    else:
        key_q = None
    if key_states_full is not None:
        key_f_t = repeat_kv(key_states_full, n_rep).transpose(-2, -1)
    else:
        key_f_t = None
    if value_states_quant is not None:
        val_q = unpack_and_dequant_triton_packed(value_states_quant, value_scale, value_mn, group_size, v_bits)
        val_q = repeat_kv_quant(val_q, n_rep)
    else:
        val_q = None
    val_f     = repeat_kv(value_states_full, n_rep)
    val_f_len = val_f.shape[-2]
    output = torch.zeros(B, H, S, D, device=query_states.device, dtype=dtype)
    for start in range(0, S, _PREFILL_CHUNK_SIZE):
        end = min(start + _PREFILL_CHUNK_SIZE, S)
        q = query_states[:, :, start:end, :]
        parts = []
        if key_q   is not None: parts.append(torch.matmul(q, key_q))
        if key_f_t is not None: parts.append(torch.matmul(q, key_f_t))
        attn_logits = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        attn_logits = attn_logits / softmax_scale
        kv_len = attn_logits.shape[-1]
        pos_q = torch.arange(start, end, device=q.device).unsqueeze(1)
        pos_k = torch.arange(kv_len,    device=q.device).unsqueeze(0)
        attn_logits.masked_fill_((pos_k > pos_q).unsqueeze(0).unsqueeze(0), torch.finfo(attn_logits.dtype).min)
        if attention_mask is not None:
            attn_logits = attn_logits + attention_mask[:, :, start:end, :]
        attn_w = F.softmax(attn_logits.float(), dim=-1).to(dtype)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)
        if val_q is not None:
            out = torch.matmul(attn_w[:, :, :, :-val_f_len], val_q) + torch.matmul(attn_w[:, :, :, -val_f_len:], val_f)
        else:
            out = torch.matmul(attn_w, val_f)
        output[:, :, start:end, :] = out
        del attn_logits, attn_w, q, out, parts
    return output


class MistralAttention_KIVI_eval(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.k_bits = config.k_bits
        self.v_bits = config.v_bits
        self.group_size = config.group_size
        self.residual_length = config.residual_length

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]
        cos, sin = _apply_mistral_rotary(self.rotary_emb, value_states, position_ids, kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states_quant_trans = past_key_value[0]
            key_states_full = past_key_value[1]
            key_scale_trans = past_key_value[2]
            key_mn_trans = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full = past_key_value[5]
            value_scale = past_key_value[6]
            value_mn = past_key_value[7]

            if key_states_quant_trans is not None:
                att_qkquant = _fallback_key_quant_matmul(
                    query_states,
                    key_states_quant_trans,
                    key_scale_trans,
                    key_mn_trans,
                    self.group_size,
                    self.k_bits,
                    self.num_key_value_groups,
                )
            else:
                att_qkquant = None

            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states
            
            key_states_full_repeat = repeat_kv(key_states_full, self.num_key_value_groups)
            att_qkfull = torch.matmul(query_states, key_states_full_repeat.transpose(2, 3)) # key_states_full need to be repeated
            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), 
                                                                                                                            self.group_size, 
                                                                                                                            self.k_bits)
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]
            if value_states_quant is None:
                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output = torch.matmul(attn_weights, value_states_full_repeat) # value_states_full need to be repeated
            else:
                attn_output = _fallback_value_quant_matmul(
                    attn_weights[:, :, :, :-value_full_length],
                    value_states_quant,
                    value_scale,
                    value_mn,
                    self.group_size,
                    self.v_bits,
                    self.num_key_value_groups,
                )

                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_states_full_repeat)

            if value_states_full.shape[-2] > self.residual_length:
                assert value_states_full.shape[-2] == self.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn
        
        else:
            # quantize
            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None
            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
            
            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
        
            if q_len > _PREFILL_CHUNK_THRESHOLD:
                attn_output = _chunked_prefill_attention(
                    query_states,
                    key_states_quant_trans, key_scale_trans, key_mn_trans,
                    key_states_full,
                    value_states_quant, value_scale, value_mn,
                    value_states_full,
                    self.group_size, self.k_bits, self.v_bits,
                    self.num_key_value_groups,
                    math.sqrt(self.head_dim),
                    attention_mask,
                )
            else:
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                if key_states_quant_trans is not None:
                    att_qkquant = _fallback_key_quant_matmul(
                        query_states,
                        key_states_quant_trans,
                        key_scale_trans,
                        key_mn_trans,
                        self.group_size,
                        self.k_bits,
                        self.num_key_value_groups,
                    )
                else:
                    att_qkquant = None

                if key_states_full is not None:
                    key_states_full_repeat = repeat_kv(key_states_full, self.num_key_value_groups)
                    att_qkfull = torch.matmul(query_states, key_states_full_repeat.transpose(2, 3))
                    if att_qkquant is not None:
                        attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
                    else:
                        attn_weights = att_qkfull / math.sqrt(self.head_dim)
                else:
                    attn_weights = att_qkquant / math.sqrt(self.head_dim)

                if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                        f" {attn_weights.size()}"
                    )

                if attention_mask is not None:
                    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                        )
                    attn_weights = attn_weights + attention_mask
                    attn_weights = torch.max(
                        attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                    )

                attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)

                value_states_full_length = value_states_full.shape[-2]
                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                if value_states_quant is None:
                    attn_output = torch.matmul(attn_weights, value_states_full_repeat)
                else:
                    attn_output = _fallback_value_quant_matmul(
                        attn_weights[:, :, :, :-value_states_full_length],
                        value_states_quant,
                        value_scale,
                        value_mn,
                        self.group_size,
                        self.v_bits,
                        self.num_key_value_groups,
                    )
                    attn_output += torch.matmul(attn_weights[:, :, :, -value_states_full_length:], value_states_full_repeat)
        past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len) if use_cache else None
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MistralFlashAttention_KIVI(MistralAttention_KIVI_eval):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = _apply_mistral_rotary(self.rotary_emb, value_states, position_ids, rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # use_sliding_windows = (
        #     _flash_supports_window_size
        #     and hasattr(self.config, "sliding_window") is not None
        #     and kv_seq_len > self.config.sliding_window
        # )

        # if not _flash_supports_window_size:
        #     logger.warning_once(
        #         "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
        #         " make sure to upgrade flash-attn library."
        #     )

        if past_key_value is not None:
            key_states_quant_trans = past_key_value[0]
            key_states_full = past_key_value[1]
            key_scale_trans = past_key_value[2]
            key_mn_trans = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full = past_key_value[5]
            value_scale = past_key_value[6]
            value_mn = past_key_value[7]

            # import ipdb; ipdb.set_trace()

            if key_states_quant_trans is not None:
                # UNGATED fused call site. Unlike the _fallback_* helpers above, this upstream
                # code invokes cuda_bmm_fA_qB_outer whatever KIVI_FUSED_GQA says, so the gate
                # does not describe it and the gated notice would misname it. It is left
                # exactly as upstream wrote it - rerouting it through the gated helpers would
                # change which kernel a published Mistral number was measured on - but it is no
                # longer invisible: it is counted as direct_fused_fires, and the fp16-only
                # constraint of the kernel (matmul.py:254 allocates a torch.float16 output) is
                # checked here rather than being discovered inside triton.
                _kivi_check_fp16("MistralFlashAttention_KIVI.key(cuda_bmm_fA_qB_outer)",
                                 query_states=query_states, key_scale_trans=key_scale_trans,
                                 key_mn_trans=key_mn_trans)
                _kivi_direct_fused_notice_once("MistralFlashAttention_KIVI.forward (key)")
                _kivi_direct_fused_fires[0] += 1
                key_states_quant_trans_repeat = repeat_kv_quant(key_states_quant_trans, self.num_key_value_groups)
                key_scale_trans_repeat = repeat_kv_quant(key_scale_trans, self.num_key_value_groups)
                key_mn_trans_repeat = repeat_kv_quant(key_mn_trans, self.num_key_value_groups)
                att_qkquant = cuda_bmm_fA_qB_outer(self.group_size, query_states, key_states_quant_trans_repeat, 
                                key_scale_trans_repeat, key_mn_trans_repeat, self.k_bits) # key_states_quant_trans, key_scale_trans, key_mn_trans need to be repeated
            else:
                att_qkquant = None
            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states

            key_states_full_repeat = repeat_kv(key_states_full, self.num_key_value_groups)
            att_qkfull = torch.matmul(query_states, key_states_full_repeat.transpose(2, 3))

            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), 
                                                                                                                            self.group_size, 
                                                                                                                            self.k_bits)
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]
            if value_states_quant is None:
                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output = torch.matmul(attn_weights, value_states_full_repeat)
            else:
                # UNGATED fused call site (see the key-side comment above): counted, dtype
                # checked, kernel choice left untouched.
                _kivi_check_fp16("MistralFlashAttention_KIVI.value(cuda_bmm_fA_qB_outer)",
                                 attn_weights=attn_weights, value_scale=value_scale,
                                 value_mn=value_mn)
                _kivi_direct_fused_notice_once("MistralFlashAttention_KIVI.forward (value)")
                _kivi_direct_fused_fires[0] += 1
                value_states_quant_repeat = repeat_kv_quant(value_states_quant, self.num_key_value_groups)
                value_scale_repeat = repeat_kv_quant(value_scale, self.num_key_value_groups)
                value_mn_repeat = repeat_kv_quant(value_mn, self.num_key_value_groups)
                attn_output = cuda_bmm_fA_qB_outer(self.group_size, attn_weights[:, :, :, :-value_full_length], value_states_quant_repeat, 
                                                value_scale_repeat, value_mn_repeat, self.v_bits) # value_states_quant, value_scale, value_mn need to be repeated
                
                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:].contiguous(), value_states_full_repeat.contiguous()) # value_states_full need to be repeated

            attn_output = attn_output.transpose(1, 2).contiguous()

            if value_full_length > self.residual_length:
                assert value_full_length == self.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn


        else:
            # print(f"kivi with flash! {self.k_bits}")

            key_states_repeat = repeat_kv(key_states, self.num_key_value_groups)
            value_states_repeat = repeat_kv(value_states, self.num_key_value_groups)

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                # Handle the case where the model is quantized
                if hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states_repeat = key_states_repeat.to(target_dtype)
                value_states_repeat = value_states_repeat.to(target_dtype)
            attn_output = self._flash_attention_forward(
                query_states.transpose(1, 2), key_states_repeat.transpose(1, 2), 
                value_states_repeat.transpose(1, 2), None, q_len, dropout=0.0
            )
            # quantize
            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None
            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
            
            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
            
        past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, 
                          value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len) if use_cache else None

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        # if not output_attentions:
        attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class MistralDecoderLayer_KIVI_eval(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            MistralAttention_KIVI_eval(config=config)
            if not getattr(config, "use_flash", False)
            else MistralFlashAttention_KIVI(config)
        )
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralModel_KIVI_eval(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        print("This is modified MistralModel_KIVI")
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MistralDecoderLayer_KIVI_eval(config) for _ in range(config.num_hidden_layers)])
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values = _normalize_legacy_cache(past_key_values)
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            # past_key_values_length = past_key_values[0][0].shape[2]
            past_key_values_length = past_key_values[0][-1]

            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (
            attention_mask is not None
            and hasattr(self.config, "_flash_attn_2_enabled")
            and self.config._flash_attn_2_enabled
            and past_key_values is not None
        ):
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MistralForCausalLM_KIVI_eval(MistralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel_KIVI_eval(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        past_key_values = _normalize_legacy_cache(past_key_values)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        past_key_values = _normalize_legacy_cache(past_key_values)
        # Omit tokens covered by past_key_values
        if past_key_values:
            # past_length = past_key_values[0][0].shape[2]
            past_length = past_key_values[0][-1]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
