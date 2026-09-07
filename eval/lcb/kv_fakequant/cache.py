# SPDX-License-Identifier: Apache-2.0
"""Cache-level KV fake-quantization that actually honours each method's high-precision region.

A KV-quant method is not just "N bits". Every method in this family keeps some tokens out of the
quantizer, and that region is part of the published recipe:

    KIVI        a fixed ``residual_length`` window of the most recent tokens stays fp16
    Kitty       a front ``sink`` of tokens stays fp16 (plus its per-channel precision boost)
    OMMX        an attention sink of 8

An earlier revision of this cache ignored both. It committed ``(T // group_size) * group_size``
tokens and left whatever happened to be past the last block boundary in fp16, so the fp16 tail was
``T mod group_size`` -- a sawtooth between 0 and group_size-1 (mean ~63.5 at group 128, and exactly
zero whenever T was a multiple of 128). ``residual_length`` was stored on the quantizer and printed
in its description, but never read, so runs recorded ``res=128`` while running an average residual
of about half that. ``sink`` did not exist at all. The effect was directional: it silently weakened
the baselines whose recipes depend on those regions.

This implementation makes the regions real and reports what it actually did.

Layout of the cache at length T, given sink S, residual R, group g:

    [0, S)                      fp16   -- front sink, never quantized
    [S, commit)                 quant  -- committed in whole g-blocks
    [commit, T)                 fp16   -- trailing residual, at least R tokens

    commit = S + floor(max(0, T - R - S) / g) * g

A token is quantized exactly once, when it falls out of the residual window and completes a block,
so the cost stays O(T). Single-beam greedy/sampling only (no beam reorder/crop).
"""

import torch
from transformers import DynamicCache


class FakeQuantKVCache(DynamicCache):
    """DynamicCache that quantize->dequantizes committed KV at the ``Cache.update()`` boundary.

    sink / residual_length default to whatever the quantizer carries, so an arm configured with
    ``--sink 32`` gets a real 32-token fp16 front, and one with ``--residual-length 128`` gets a
    real 128-token fp16 tail.
    """

    def __init__(self, quantizer, config=None, sink=None, residual_length=None):
        super().__init__(config=config)
        self.quantizer = quantizer
        q = quantizer
        self.sink = int(sink if sink is not None else getattr(q, "sink", 0) or 0)
        self.residual_length = int(
            residual_length if residual_length is not None
            else getattr(q, "residual_length", 0) or 0)
        self._qk = {}       # layer_idx -> committed quantized keys  [B,H,commit-sink,D]
        self._qv = {}
        self._clen = {}     # layer_idx -> absolute index one past the committed region

    # -- introspection ------------------------------------------------------ #
    def describe(self):
        """Single source of truth: the cache owns the regions, so the cache names them.

        The previous revision let the quantizer advertise a residual the cache did not implement;
        anything the cache does not enforce must not appear in this string.
        """
        if self.quantizer is None:
            return "bf16"
        return "%s,sink=%d,res=%d" % (self.quantizer.describe(), self.sink, self.residual_length)

    def reset(self):
        super().reset()
        self._qk.clear(); self._qv.clear(); self._clen.clear()

    # -- core --------------------------------------------------------------- #
    def _commit_end(self, T):
        g = self.quantizer.group_size
        avail = T - self.residual_length - self.sink
        if avail < g:
            return self.sink
        return self.sink + (avail // g) * g

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        k_full, v_full = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        q = self.quantizer
        if q is None:
            return k_full, v_full

        T = k_full.shape[2]
        commit = self._commit_end(T)
        clen = self._clen.get(layer_idx, self.sink)

        if commit > clen:
            kb = k_full[:, :, clen:commit, :]
            vb = v_full[:, :, clen:commit, :]
            kqb, vqb = q.quant_k(kb), q.quant_v(vb)
            if layer_idx not in self._qk:
                self._qk[layer_idx], self._qv[layer_idx] = kqb, vqb
            else:
                self._qk[layer_idx] = torch.cat([self._qk[layer_idx], kqb], dim=2)
                self._qv[layer_idx] = torch.cat([self._qv[layer_idx], vqb], dim=2)
            self._clen[layer_idx] = commit

        if commit <= self.sink:                    # nothing committed yet: all fp16
            return k_full, v_full

        read_k = torch.cat([k_full[:, :, :self.sink, :], self._qk[layer_idx],
                            k_full[:, :, commit:, :]], dim=2)
        read_v = torch.cat([v_full[:, :, :self.sink, :], self._qv[layer_idx],
                            v_full[:, :, commit:, :]], dim=2)
        return read_k, read_v
