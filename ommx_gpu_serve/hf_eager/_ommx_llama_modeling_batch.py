# OMMX canonical-KV-quant modeling for Llama (HF-eager, full-model, B>=1 BATCHED).
#
# The B>1 sibling of _ommx_llama_modeling.py (the working B=1 file). Where the B=1
# modeling owns a per-layer single-sequence CanonicalKVStore, this owns a per-layer
# SHARED MultiSeqKVPool (kv_pool.py) — ONE quantized-plane pool for all B requests +
# per-request tables — and drives the SAME ommx_paged_decode_attention_canonical op
# over num_seqs=B in ONE kernel launch (mirroring the vLLM backend's batched route:
# OMMXCanonicalImpl._route_decode_batched + OMMXCanonicalImpl._batched_kv_update in
# ommx_gpu_serve/integration/vllm/backend.py).
#
# Split (identical contract to the B=1 file, per layer):
#   * PREFILL (q_len > 1): eager/flash/sdpa attention over the bf16 K/V for each of the
#     B sequences (uniform length here), then pack each request's prompt into the pool
#     via append_block(req, ...). The B sequences share the pool's plane tensors; each
#     request owns a contiguous group/page slot block.
#   * DECODE  (q_len == 1): append all B new bf16 K/V rows in ONE device scatter
#     (append_decode_batched(slots, ...)), GPU-regroup any completed 32-token group, and
#     call ommx_paged_decode_attention_canonical with q [B,H_q,D] + the pool's batched
#     planes/req-tables + b_seq_len [B] -> attn output [B,1,H_q,D].
#
# SHADOW-FREE: with OMMX_KV_RING=1 (set by the bench) the pool's k_hist/v_hist are a
# per-request RING of only sink ∪ recent ∪ the live group (ring_cap ~= sink+recent+2*gt
# rows/req, INDEPENDENT of max_seq_len) instead of the [num_seqs, max_seq_len, H, D] bf16
# SHADOW that OOMs at B>1 long-ctx. (No measurement of that OOM is shipped in this
# release, so it is stated as the shape argument it is — [num_seqs, max_seq_len, H, D]
# bf16 grows with B AND with ctx — not as a GPU-capacity number.) The quantized
# planes hold the rest. ("Quantized", not "≤3-bit": this file packs whatever recipe the
# caller's env resolves to, and under the published canonical one the benches here set —
# npv=6, gt=32, gc=32, outlier map ON, pow2 — the pool's own planes sum to 4.375 bit/elem
# avg. ≤3 bit needs a DIFFERENT recipe, gt=64 + gc=64 + map OFF, that no accuracy result
# here was measured under.) This file works WITH or WITHOUT the ring (the
# pool sizing is gated internally); the bench sets the ring.
#
# LAW #5 (no silent fallback — the convention the serving code uses throughout): the
# batched decode must actually fire the OMMX batched kernel over num_seqs=B (NOT a
# B=1 python loop, NOT a bf16 fallback). We emit a one-time BATCHED_DECODE_FIRED
# marker from the decode path + assert B>1 reached the single-launch op (the bench
# reads the marker counter back).
#
# Sources verified while writing. Cited by SYMBOL, not by line: these files are under
# active edit and a line number rots silently, while a symbol stays greppable
# (`grep -n "def <symbol>" <file>`).
#   - ommx_gpu_serve/attention/kv_pool.py         MultiSeqKVPool API (ring + batched)
#   - ommx_gpu_serve/attention/paged_decode.py    ommx_paged_decode_attention_canonical
#                                                 (kernel signature + plane ABI docstring)
#   - ommx_gpu_serve/integration/vllm/backend.py  OMMXCanonicalImpl._route_decode_batched
#                                                 (call mirror) and ._batched_kv_update
#                                                 (its UNIFORM single-token fast path)
#   - ommx_gpu_serve/hf_eager/_ommx_llama_modeling.py  the working B=1 modeling (mirror)

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
    class LossKwargs(TypedDict, total=False):
        pass

from transformers.models.llama import LlamaConfig

logger = logging.get_logger(__name__)

# OMMX canonical paged-decode attention (Triton; vLLM-independent pure tensor ABI).
from ommx_gpu_serve.attention.kv_pool import MultiSeqKVPool
from ommx_gpu_serve.attention.paged_decode import (
    _auto_num_kv_splits,
    ommx_paged_decode_attention_canonical,
)
from ommx_gpu_serve.integration.vllm.config import resolve_serving_config

# ── silent-fallback evidence (LAW #5): module-global one-time markers the bench reads ──
# BATCHED_DECODE_FIRES counts the number of (layer, step) batched OMMX kernel launches
# that ran over B>1 rows. A B=1 python loop or a bf16 fallback would leave this at 0 (or
# never report B>1) — the bench asserts it is > 0 AND the recorded B matches num_seqs.
OMMX_BATCH_EVIDENCE = {"fires": 0, "last_B": 0, "max_B": 0, "printed": False}


def _record_batched_fire(B: int) -> None:
    OMMX_BATCH_EVIDENCE["fires"] += 1
    OMMX_BATCH_EVIDENCE["last_B"] = int(B)
    OMMX_BATCH_EVIDENCE["max_B"] = max(OMMX_BATCH_EVIDENCE["max_B"], int(B))
    if not OMMX_BATCH_EVIDENCE["printed"]:
        OMMX_BATCH_EVIDENCE["printed"] = True
        import sys
        print(f"BATCHED_DECODE_FIRED B={B} (OMMX MultiSeqKVPool single-launch decode)",
              file=sys.stderr, flush=True)


class OmmxSeqCounterCache(DynamicCache):
    """Minimal HF Cache that ONLY tracks a logical sequence length (no real K/V).

    Identical to the B=1 file's counter cache: the real K/V live in each layer's
    MultiSeqKVPool (the OMMX single source of truth); this exists only so HF's
    cache_position / RoPE bookkeeping advances across decode steps. Stores NO per-layer
    tensors, so there is no second bf16 KV shadow (SHADOW-FREE end-to-end)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ommx_seq_len = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self._ommx_seq_len)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
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
    """Llama attention with the OMMX canonical-KV-quant BATCHED paged decode kernel.

    Prefill = eager/flash/sdpa over bf16 K/V for each of the B sequences (like the B=1
    file), then pack each request's prompt into this layer's shared MultiSeqKVPool via
    append_block(req, ...). Decode = append all B new bf16 K/V rows in ONE device scatter
    (append_decode_batched), GPU-regroup, and run ONE ommx_paged_decode_attention_canonical
    over all B rows. Each layer owns one pool (created lazily on the first prefill).
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
        self.sliding_window = None  # Llama-3.1: no sliding window

        # OMMX per-layer state (lazy: built on the first prefill once we know device + B).
        self._ommx_pool: Optional[MultiSeqKVPool] = None
        self._ommx_cfg = None          # resolved OMMXServingConfig (env recipe)
        self._ommx_ws = None           # fixed-address batched decode workspace
        self._ommx_B = 0               # batch size of the current pool
        # DEVICE decode path gate (OMMX_DECODE_DEVICE, default ON): drives the per-step
        # decode through the pool's device/captured path (closed-form device tail/seq
        # tables, no per-step python loop / alloc / torch.tensor). The old host path
        # (append_decode_batched + decode_inputs_batched) stays available for A/B when
        # OMMX_DECODE_DEVICE=0. `_ommx_seq_seeded` tracks whether the device seq buffer
        # has been seeded for the current prefill (reset on each new prefill).
        import os as _os
        self._ommx_decode_device = (
            _os.environ.get("OMMX_DECODE_DEVICE", "1") not in {"0", "false", "off", "no"})
        self._ommx_seq_seeded = False

    # ── OMMX pool + workspace lifecycle ─────────────────────────────────────────

    def _ommx_resolve_cfg(self, *, force: bool = False):
        # mirror LlamaAttention._ommx_resolve_cfg in _ommx_llama_modeling.py (the B=1 file)
        # — max_context=None lets OMMX_ATTN_MAXCTX drive the per-request plane sizing (the
        # bench sets it per ctx).
        if self._ommx_cfg is None or force:
            self._ommx_cfg = resolve_serving_config(
                head_dim=self.head_dim,
                n_q_heads=self.num_attention_heads,
                n_kv_heads=self.num_key_value_heads,
                page_size=16,
                max_context=None,
            )
        return self._ommx_cfg

    def _ommx_make_pool(self, device, num_seqs: int) -> MultiSeqKVPool:
        # mirror the pool construction the served path uses (OMMXBatchedStepManager.pool in
        # integration/vllm/metadata.py). The pool
        # holds num_seqs plane blocks + per-request bf16 ring (ring_cap rows/req under
        # OMMX_KV_RING) — NOT the [num_seqs, max_seq_len, H, D] bf16 shadow.
        import os
        cfg = self._ommx_resolve_cfg(force=True)
        # NOTE: MultiSeqKVPool always packs V as i2 (MultiSeqKVPool.__init__ allocates
        # ``self.v_main`` as [P_cap, ps, H, D//4] uint8, i.e. 4 codes/byte) — it
        # has NO bf16-V branch (unlike CanonicalKVStore). So OMMX_ATTN_V_BF16 is IGNORED in
        # the batched pool; V is always quantized i2 here. (# VERIFY: if a bf16-V batched
        # variant is needed, MultiSeqKVPool must grow a v_format arg first.)
        pool = MultiSeqKVPool(
            head_dim=self.head_dim,
            n_kv_heads=self.num_key_value_heads,
            num_seqs=int(num_seqs),
            max_seq_len=int(cfg.max_context),
            k_format=cfg.k_format,
            outliers_per_vector=cfg.outliers_per_vector,
            outlier_select=cfg.outlier_select,
            outlier_repr=cfg.outlier_repr,
            use_pow2=cfg.use_pow2,
            window=cfg.window(),
            group_channels=cfg.group_channels,
            device=device,
        )
        return pool

    def _ommx_make_ws(self, device, B: int):
        # mirror backend.py's OMMXCanonicalImpl._ommx_ws_batched: ONE fixed-address batched
        # decode workspace, sized at worst-case geometry (num_kv_splits from max_context, cap
        # rows = B). No per-step alloc; reused every decode step.
        if self._ommx_ws is not None and self._ommx_ws["o"].shape[0] >= B:
            return self._ommx_ws
        H, D = self.num_attention_heads, self.head_dim
        cfg = self._ommx_resolve_cfg()
        max_seq = max(1, int(cfg.max_context))
        nsplits = max(1, int(_auto_num_kv_splits(max_seq, B, device=device,
                                                 kv_heads=self.num_key_value_heads)))
        max_groups = max(1, (max_seq + 31) // 32)
        self._ommx_ws = {
            "o": torch.empty(B, H, D, dtype=torch.float32, device=device),
            "lse": torch.empty(B, H, dtype=torch.float32, device=device),
            # last dim = D+1: D o-lanes + 1 lse lane (paged_decode._make_attn_logits uses
            # head_dim+1). Allocating D makes stride_mid_os=D so each split's lse write
            # clobbers the next split's o_v -> decode garbage for num_kv_splits>1 (B>1 path).
            "attn_logits": torch.empty(B, H, nsplits + 1, D + 1, dtype=torch.float32, device=device),
            "qzp_buf": torch.empty(B, H, max_groups, dtype=torch.float32, device=device),
            "num_kv_splits": nsplits,
        }
        return self._ommx_ws

    # ── batched decode kernel call (mirror OMMXCanonicalImpl._route_decode_batched) ──

    def _ommx_decode_batched(self, query_states: torch.Tensor) -> torch.Tensor:
        """One BATCHED OMMX decode step. ``query_states`` is HF-layout [B, Hq, q_len=1, D]
        (post-RoPE). Returns attn output [B, q_len=1, Hq, D] (so the caller's
        ``.reshape(*input_shape, -1)`` lands the heads contiguously)."""
        H, D = self.num_attention_heads, self.head_dim
        B = int(query_states.shape[0])
        pool = self._ommx_pool
        ws = self._ommx_make_ws(query_states.device, B)
        slots = list(range(B))   # batch row b -> pool slot b (contiguous; uniform batch)

        # q -> [B, Hq, D] (mirror `q = query.view(-1, H, D)[:B]` in
        # OMMXCanonicalImpl._route_decode_batched). HF-layout
        # query_states is [B, Hq, q_len=1, D]; slice the single decode token -> [B, Hq, D].
        q = query_states[:, :, 0, :].contiguous()       # [B, Hq, D]

        if self._ommx_decode_device:
            # DEVICE path: closed-form device tail/seq tables filled into the pool's
            # persistent buffers (NO per-step python loop / torch.tensor / dict rebuild).
            # The per-step KV write + seq advance already ran in forward() via
            # append_decode_device; here we only build the read bundle on device. The
            # grid uses the FIXED worst-case max_context (no .max().item() sync).
            max_ctx = max(1, int(self._ommx_resolve_cfg().max_context))
            di = pool.decode_inputs_device(B, max_ctx)
        else:
            di = pool.decode_inputs_batched(slots)      # shared planes + per-req tables/tail
        pl = di["planes"]
        o = ws["o"][:B]
        lse = ws["lse"][:B]
        attn_logits = ws["attn_logits"][:B]
        qzp_buf = ws["qzp_buf"][:B]
        sm_scale = float(self.scaling)

        ommx_paged_decode_attention_canonical(
            q, pl["k_base"], pl["k_scale"], pl["k_zp"],
            pl.get("k_oidx"), pl.get("k_oval"),
            pl["v_main"], pl["v_scale"], pl["v_zp"],
            di["k_tail"], di["v_tail"], di["b_tail_len"],
            o, lse,
            di["req_to_token"], di["req_to_group"], di["b_seq_len"],
            sm_scale=sm_scale, page_size=int(pl["page_size"]),
            k_outliers_per_vector=int(pl["outliers_per_vector"]),
            k_format=str(pl["k_format"]),
            combinadic_read=bool(self._ommx_resolve_cfg().resolved_combinadic_read()),
            # The bitmap encoding was reachable in the PACKER but not here: the store
            # allocates k_obmp and NO k_oidx under outlier_repr=bitmap, so a decode that
            # did not also take the bitmap read raised "requires the k_oidx relidx7
            # sidecar plane" on the first quantized group. Sourced from the resolved
            # config exactly like combinadic_read, so OMMX_ATTN_OUTLIER_REPR=bitmap means
            # the same thing on this path as it does on the vLLM backend.
            bitmap_read=bool(self._ommx_resolve_cfg().resolved_bitmap_read()),
            k_obmp=pl.get("k_obmp"),
            k_crank=pl.get("k_crank"),
            k_fp4_mapscale=pl.get("k_fp4_mapscale"),
            k_fp4_mapcenter=pl.get("k_fp4_mapcenter"),
            kv_outlier_map=bool(pl.get("kv_outlier_map", False)),
            kv_int8_scale=bool(pl.get("kv_int8_scale", False)),
            num_kv_splits=ws["num_kv_splits"],
            max_seq_len=di["max_seq_len"], max_tail_len=di["max_tail_len"],
            attn_logits=attn_logits, qzp_buf=qzp_buf,
            packed_start_offset=int(di["packed_start_offset"]),
        )
        # LAW #5: prove the SINGLE-LAUNCH batched op fired over B rows (NOT a B=1 loop /
        # bf16 fallback). o is [B, Hq, D]; the op wrote all B rows in one launch.
        _record_batched_fire(B)
        # Return as [B, q_len=1, Hq, D] so LlamaAttention's reshape concatenates the head dim.
        return ws["o"][:B].to(query_states.dtype).view(B, 1, H, D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]        # (B, q_len)
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # query/key/value_states: [B, n_heads, q_len, D]

        B = query_states.shape[0]
        q_len = query_states.shape[2]
        is_prefill = q_len > 1

        if is_prefill:
            # New batch (prefill): (RE)BUILD this layer's pool + workspace at the right size
            # for THIS ctx and THIS B. Drop the old pool/ws first so its memory is reclaimed
            # (the bench sweeps ctx/B against one model instance; OMMX_ATTN_MAXCTX is set per
            # ctx). Mirror the prefill branch of LlamaAttention.forward in
            # _ommx_llama_modeling.py (the B=1 file) but for the shared pool.
            self._ommx_pool = None
            self._ommx_ws = None
            self._ommx_B = int(B)
            self._ommx_seq_seeded = False   # device seq buffer reseeds at first decode
            self._ommx_pool = self._ommx_make_pool(hidden_states.device, num_seqs=int(B))

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
            # Pack each request's prompt into the shared pool. key/value_states are
            # [B, Hkv, T, D]; per request permute to [T, Hkv, D] — the layout
            # MultiSeqKVPool.append_block documents ("Stash request ``req``'s prefill block
            # bf16 K/V ``[T, H, D]`` then regroup."). Each request owns its contiguous
            # pool slot block.
            for r in range(B):
                K_pack = key_states[r].transpose(0, 1).contiguous()    # [T, Hkv, D]
                V_pack = value_states[r].transpose(0, 1).contiguous()  # [T, Hkv, D]
                self._ommx_pool.append_block(r, K_pack.to(torch.bfloat16), V_pack.to(torch.bfloat16))
        else:
            # DECODE (q_len == 1). Append all B new bf16 K/V rows in ONE device scatter,
            # GPU-regroup any completed group, then run the SINGLE-LAUNCH batched OMMX op.
            assert self._ommx_pool is not None, "OMMX decode before prefill"
            assert B == self._ommx_B, (
                f"OMMX batched decode B={B} != pool num_seqs={self._ommx_B} (the pool is "
                f"built per-batch at prefill; B must be stable across the decode loop)")
            # key/value_states: [B, Hkv, q_len=1, D] -> [B, Hkv, D] (one new row per request).
            k_rows = key_states[:, :, 0, :].contiguous()     # [B, Hkv, D]
            v_rows = value_states[:, :, 0, :].contiguous()   # [B, Hkv, D]
            # ONE scatter + GPU regroup over the B requests (mirror the UNIFORM fast path of
            # OMMXCanonicalImpl._batched_kv_update, which calls pool.append_decode_batched
            # once instead of looping per request). This
            # also advances each request's seq_len + packs completed groups into the pool.
            if self._ommx_decode_device:
                # DEVICE path: seed the pool's device seq buffer ONCE (first decode step
                # after prefill), then the per-step write uses the persistent device
                # row/col buffers (no per-step torch.tensor / _ring_slot_host list-comp).
                if not self._ommx_seq_seeded:
                    self._ommx_pool._ensure_decode_buffers(B)
                    self._ommx_pool.decode_seed_seq_device(list(range(B)))
                    self._ommx_seq_seeded = True
                self._ommx_pool.append_decode_device(k_rows, v_rows, B)
            else:
                self._ommx_pool.append_decode_batched(list(range(B)), k_rows, v_rows)
            attn_output = self._ommx_decode_batched(query_states)   # [B, q_len=1, Hq, D]
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

        # OMMX owns the KV state per-layer (MultiSeqKVPool). A lightweight counter cache
        # ONLY so HF's cache_position / mask plumbing has a get_seq_length() to read. For a
        # UNIFORM batch every request has the same length, so the single counter is correct.
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

        if use_cache and isinstance(past_key_values, OmmxSeqCounterCache):
            past_key_values._ommx_seq_len += int(inputs_embeds.shape[1])

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM_OMMX_Batch(LlamaPreTrainedModel, GenerationMixin):
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


__all__ = ["LlamaForCausalLM_OMMX_Batch", "OMMX_BATCH_EVIDENCE"]
