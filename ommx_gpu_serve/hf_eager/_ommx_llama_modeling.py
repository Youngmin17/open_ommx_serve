# OMMX canonical-KV-quant modeling for Llama (HF-eager, full-model, B=1).
#
# Mirrors baseline/kv/kitty/_kitty_llama_modeling.py one-for-one, swapping Kitty's
# 2-bit paged decode kernel for the OMMX canonical paged-decode attention. The OMMX
# decode kernel is vLLM-INDEPENDENT (pure tensor ABI) — this file drives it directly
# from an HF eager forward so OMMX decode can be measured full-model, the same way
# KIVI / Kitty are, for a fair B=1 long-context TPOT comparison.
#
# Split (identical to Kitty's prefill(flash) / decode(custom-kernel) contract):
#   * PREFILL (q_len > 1): flash_attention_2 over the bf16 K/V (same as Kitty), then
#     pack the whole prompt into a per-layer CanonicalKVStore via append_block().
#   * DECODE  (q_len == 1): append the new bf16 K/V row, regroup any completed
#     32-token scale-group, and call ommx_paged_decode_attention_canonical over the
#     stored QUANTIZED prefix + the bf16 sink/recent tail.
#
# SHADOW-FREE: with OMMX_KV_RING=1 (set by the bench) the store keeps only the live
# bf16 set (sink ∪ recent ∪ the group being packed); the bulk prefix is quantized
# (i2f4 K + i2 V). Each layer owns its own store.
#
# Sources verified while writing (file:line cited in the bench RISKS):
#   - ommx_gpu_serve/attention/kv_store.py            CanonicalKVStore API
#   - ommx_gpu_serve/attention/paged_decode.py:2374   kernel signature
#   - ommx_gpu_serve/integration/vllm/backend.py:792  _route_decode (eager call mirror)
#   - ommx_gpu_serve/integration/vllm/config.py:153   resolve_serving_config (env recipe)
#   - ommx_gpu_serve/attention/tests/test_reference_op_parity.py:63  append_block+decode_inputs flow

from typing import Callable, Optional, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging
try:  # LossKwargs removed in transformers>=4.55; only a kwargs type-hint base here
    from transformers.utils import LossKwargs
except ImportError:
    from typing import TypedDict
    class LossKwargs(TypedDict, total=False):  # TypedDict shim so multi-inherit w/
        pass                                    # FlashAttentionKwargs (also TypedDict) works

from transformers.models.llama import LlamaConfig

logger = logging.get_logger(__name__)

# OMMX canonical paged-decode attention (Triton; vLLM-independent pure tensor ABI).
# kv_store.py / paged_decode.py / config.py are CPU-importable; the Triton kernel
# JIT-builds lazily on the FIRST real CUDA call (fine on GPU).
from ommx_gpu_serve.attention.kv_store import CanonicalKVStore
from ommx_gpu_serve.attention.paged_decode import (
    _auto_num_kv_splits,
    ommx_paged_decode_attention_canonical,
)
from ommx_gpu_serve.integration.vllm.config import resolve_serving_config


class OmmxSeqCounterCache(DynamicCache):
    """Minimal HF Cache that ONLY tracks a logical sequence length.

    The real K/V live in each layer's CanonicalKVStore (the OMMX single source of
    truth); this object exists only so HF's cache_position / RoPE position bookkeeping
    keeps advancing across decode steps (``model(input_ids=tok, past_key_values=past)``).
    It subclasses DynamicCache so every ``isinstance(x, Cache)`` / DynamicCache check in
    the HF model + masking utils passes, but stores NOTHING (no per-layer tensors), so
    there is no second bf16 KV shadow (SHADOW-FREE end-to-end).

    VERIFY (transformers version): we override get_seq_length() to read our own counter
    and make update() a no-op that only advances it. If the installed transformers calls
    other Cache methods during decode (e.g. get_max_cache_shape / reorder_cache), they
    inherit DynamicCache's behaviour over the (empty) layer list — harmless for B=1 greedy
    decode. Tested contract: get_seq_length() + the model never reading real K/V back.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ommx_seq_len = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self._ommx_seq_len)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # NO-OP pass-through: the OMMX attention path never calls this (prefill uses flash
        # over its own K/V; decode uses the CanonicalKVStore), and LlamaModel.forward bumps
        # the counter directly. We keep update() inert (no tensor retention, no counter
        # change here to avoid double-counting) purely so any incidental HF call is safe.
        return key_states, value_states


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Llama attention with the OMMX canonical-KV-quant paged decode kernel.

    Prefill = flash_attention_2 over bf16 K/V (like Kitty), then pack the prompt into
    this layer's CanonicalKVStore. Decode = append the new bf16 K/V row, regroup, and
    run ommx_paged_decode_attention_canonical over the stored quantized prefix + bf16
    sink/recent tail. Each layer owns one store (created lazily on the first prefill).
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        # NOTE: Llama has NO q_norm / k_norm (unlike Qwen3).
        self.sliding_window = None  # Llama-3.1: no sliding window

        # OMMX per-layer state (lazy: built on the first prefill once we know device).
        self._ommx_store: Optional[CanonicalKVStore] = None
        self._ommx_cfg = None          # resolved OMMXServingConfig (env recipe)
        self._ommx_ws = None           # fixed-address decode workspace (o/lse/attn_logits/qzp_buf)

    # ── OMMX store + workspace lifecycle ────────────────────────────────────────

    def _ommx_resolve_cfg(self, *, force: bool = False):
        # mirror backend.py:335 resolve_serving_config(head_dim, n_q_heads, n_kv_heads,
        # max_context). max_context drives k_hist/plane sizing + the kv-split workspace.
        #
        # IMPORTANT (memory): the planes are sized to max_context, so sizing it from the
        # model's max_position_embeddings (131072 for Llama-3.1) would allocate ~17GB of
        # ≤3-bit planes PER LAYER-sweep. The bench sets OMMX_ATTN_MAXCTX per ctx (= ctx +
        # decode steps); resolve_serving_config(max_context=None) reads that env. We pass
        # max_context=None so the env wins, and re-resolve on ``force`` (a new prefill /
        # new ctx) so the store is rebuilt at the right size for each ctx in the sweep.
        if self._ommx_cfg is None or force:
            self._ommx_cfg = resolve_serving_config(
                head_dim=self.head_dim,
                n_q_heads=self.num_attention_heads,
                n_kv_heads=self.num_key_value_heads,
                page_size=16,
                max_context=None,   # let OMMX_ATTN_MAXCTX env drive it (default 4096)
            )
        return self._ommx_cfg

    def _ommx_make_store(self, device) -> CanonicalKVStore:
        # mirror backend.py:356-363 CanonicalKVStore(...) construction. v_format follows
        # OMMX_ATTN_V_BF16 (default i2 = shadow-free quantized V). The window (sink/recent/
        # group/page) + group_channels come straight from the resolved env recipe.
        import os
        cfg = self._ommx_resolve_cfg(force=True)
        v_bf16 = os.environ.get("OMMX_ATTN_V_BF16", "0").strip().lower() not in {"0", "false", "off", "no", ""}
        st = CanonicalKVStore(
            head_dim=self.head_dim,
            n_kv_heads=self.num_key_value_heads,
            max_seq_len=int(cfg.max_context),
            k_format=cfg.k_format,
            v_format=("bf16" if v_bf16 else "i2"),
            outliers_per_vector=cfg.outliers_per_vector,
            outlier_select=cfg.outlier_select,
            outlier_repr=cfg.outlier_repr,
            use_pow2=cfg.use_pow2,
            window=cfg.window(),
            group_channels=cfg.group_channels,
            device=device,
        )
        return st

    def _ommx_make_ws(self, device):
        # mirror backend.py:763-790 _ommx_ws: ONE fixed-address decode workspace per layer,
        # sized at worst-case geometry (num_kv_splits from max_context). No per-step alloc.
        if self._ommx_ws is not None:
            return self._ommx_ws
        H, D = self.num_attention_heads, self.head_dim
        cfg = self._ommx_resolve_cfg()
        max_seq = max(1, int(cfg.max_context))
        nsplits = max(1, int(_auto_num_kv_splits(max_seq, 1, device=device,
                                                 kv_heads=self.num_key_value_heads)))
        max_groups = max(1, (max_seq + 31) // 32)
        self._ommx_ws = {
            "o": torch.empty(1, H, D, dtype=torch.float32, device=device),
            "lse": torch.empty(1, H, dtype=torch.float32, device=device),
            # last dim MUST be head_dim+1 (per split: head_dim o-lanes + 1 lse lane), matching
            # paged_decode._make_attn_logits. Allocating D (not D+1) makes stride_mid_os=D so
            # each split's lse write clobbers the NEXT split's o_v -> decode garbage for
            # num_kv_splits>1 (cos 0.75 @ns=2), the ROOT CAUSE of the whole decode-garbage saga.
            "attn_logits": torch.empty(1, H, nsplits + 1, D + 1, dtype=torch.float32, device=device),
            "qzp_buf": torch.empty(1, H, max_groups, dtype=torch.float32, device=device),
            "num_kv_splits": nsplits,
        }
        return self._ommx_ws

    # ── decode kernel call (mirror backend.py:792-836 _route_decode EXACTLY) ─────

    def _ommx_decode(self, query_states: torch.Tensor) -> torch.Tensor:
        """One OMMX decode step. ``query_states`` is HF-layout [B=1, Hq, q_len=1, D]
        (post-RoPE). Returns attn output [B=1, q_len=1, Hq, D] (so the caller's
        ``.reshape(*input_shape, -1)`` lands the heads contiguously, matching Kitty)."""
        H, D = self.num_attention_heads, self.head_dim
        st = self._ommx_store
        ws = self._ommx_make_ws(query_states.device)

        # q -> [1, Hq, D] (batch=1, head_num=Hq, D): the kernel's q layout (paged_decode.py
        # :2542 `batch, head_num = q.shape[0], q.shape[1]`). query_states is HF-layout
        # [B=1, Hq, q_len=1, D]; slice the single decode token then make contiguous (post-RoPE
        # query_states need not be contiguous, so use an explicit slice+contiguous, NOT .view).
        q = query_states[:, :, 0, :].contiguous()       # [1, Hq, D]

        di = st.decode_inputs()                        # nested planes + tail + seq bufs
        pl = di["planes"]
        sm_scale = float(self.scaling)
        # FIXED num_kv_splits (max-context-sized, constant across a decode sequence). Now that
        # attn_logits is correctly D+1-strided, this fixed count is EXACT (test_long_nsplits.py:
        # cos 0.99997 for ns in {1,2,4,8,16,254}). Keep it FIXED (not per-seq auto) so the Triton
        # kernel's num_kv_splits constexpr never changes mid-sequence -> no JIT recompile churn
        # during decode (a per-seq value recompiled as seq grew -> p99 spikes, inflated TPOT).
        cur_num_kv_splits = int(ws["num_kv_splits"])

        ommx_paged_decode_attention_canonical(
            q, pl["k_base"], pl["k_scale"], pl["k_zp"],
            pl.get("k_oidx"), pl.get("k_oval"),
            pl["v_main"], pl["v_scale"], pl["v_zp"],
            di["k_tail"], di["v_tail"], di["b_tail_len"],
            ws["o"], ws["lse"],
            pl["req_to_token"], pl["req_to_group"], di["b_seq_len"],
            sm_scale=sm_scale, page_size=int(pl["page_size"]),
            k_outliers_per_vector=int(pl["outliers_per_vector"]),
            k_format=str(pl["k_format"]),
            combinadic_read=bool(self._ommx_resolve_cfg().combinadic_read),
            k_crank=pl.get("k_crank"),
            k_fp4_mapscale=pl.get("k_fp4_mapscale"),
            k_fp4_mapcenter=pl.get("k_fp4_mapcenter"),
            kv_outlier_map=bool(pl.get("kv_outlier_map", False)),
            kv_int8_scale=bool(pl.get("kv_int8_scale", False)),
            num_kv_splits=cur_num_kv_splits,
            max_seq_len=di["max_seq_len"], max_tail_len=di["max_tail_len"],
            attn_logits=ws["attn_logits"], qzp_buf=ws["qzp_buf"],
            packed_start_offset=int(di["packed_start_offset"]),
        )
        # ws["o"] is [1, Hq, D] f32 (backend.py:783). Return as [1(B), 1(q_len), Hq, D]
        # so the LlamaAttention reshape `.reshape(*input_shape, -1)` (input_shape=(B,q_len))
        # concatenates the head dim — same memory layout Kitty/eager produce after their
        # transpose(1,2) (i.e. [B, q_len, Hq, D]).
        return ws["o"].to(query_states.dtype).view(1, 1, H, D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Llama: plain linear projections, NO per-head RMSNorm on Q/K
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # query_states/key_states/value_states: [B, n_heads, q_len, D]

        q_len = query_states.shape[2]
        is_prefill = q_len > 1

        if is_prefill:
            # New sequence (prefill): (RE)BUILD this layer's store + workspace at the right
            # size for THIS ctx. The bench sweeps multiple ctxs against ONE model instance
            # and sets OMMX_ATTN_MAXCTX per ctx; the planes are max_context-sized, so we must
            # rebuild (not reset_inplace) to (a) pick up the new max_context and (b) free the
            # previous ctx's planes. Drop the old store/ws first so its memory is reclaimed.
            self._ommx_store = None
            self._ommx_ws = None
            self._ommx_store = self._ommx_make_store(hidden_states.device)

            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
            # Pack the prompt into the quantized store. CanonicalKVStore expects bf16
            # K/V as [T, Hkv, D] (test_reference_op_parity.py:53-63). key/value_states
            # are [B=1, Hkv, T, D] -> squeeze batch, permute to [T, Hkv, D].
            K_pack = key_states[0].transpose(0, 1).contiguous()    # [T, Hkv, D]
            V_pack = value_states[0].transpose(0, 1).contiguous()  # [T, Hkv, D]
            self._ommx_store.append_block(K_pack.to(torch.bfloat16), V_pack.to(torch.bfloat16))
        else:
            # DECODE (q_len == 1). Append the new bf16 K/V row, regroup any completed
            # 32-token group, then run the OMMX canonical decode kernel.
            assert self._ommx_store is not None, "OMMX decode before prefill"
            k_row = key_states[0, :, 0, :]      # [Hkv, D]
            v_row = value_states[0, :, 0, :]    # [Hkv, D]
            self._ommx_store.append(k_row.to(torch.bfloat16), v_row.to(torch.bfloat16))
            self._ommx_store.maybe_regroup()     # HOST seam: pack newly completed group(s)
            attn_output = self._ommx_decode(query_states)   # [B=1, q_len=1, Hq, D]
            attn_weights = None

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # All Llama-3.1 layers are full attention
        self.attention_type = "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # OMMX owns the KV state per-layer (CanonicalKVStore), not a HF Cache. We still
        # create a lightweight counter cache ONLY so HF's cache_position / mask plumbing
        # has a get_seq_length() to read; the actual K/V never lives in it (the OMMX
        # store is the single source of truth). The model RETURNS this so the decode
        # loop's `past_key_values=` keeps cache_position advancing across steps.
        if use_cache and past_key_values is None:
            past_key_values = OmmxSeqCounterCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Advance the cache_position bookkeeping by the tokens we just consumed, WITHOUT
        # storing real K/V (the OMMX per-layer stores hold them). The next forward then
        # derives cache_position from get_seq_length() exactly like a stock HF run. For
        # OmmxSeqCounterCache this just bumps an int; for any other passed-in cache we fall
        # back to its own update() (a stock DynamicCache, e.g. the bf16 arm, advances itself).
        if use_cache and isinstance(past_key_values, OmmxSeqCounterCache):
            past_key_values._ommx_seq_len += int(inputs_embeds.shape[1])

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM_OMMX(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["LlamaForCausalLM_OMMX"]
