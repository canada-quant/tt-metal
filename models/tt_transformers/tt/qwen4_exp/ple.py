# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal)
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next n-gram PLE layer (`Qwen4ExpTextPLELayer`, layer 1 only).

Two halves per docs/qwen38-flash-next-ple-host.md (tt-rd main, PR #299) and the
normative contracts docs/qwen38-flash-next-kernel-contracts.md §2.8/§2.9:

1. `PleHostGather` — HOST side. The 102.4 GiB n-gram table NEVER leaves host
   RAM/disk: pure-int64 hash + zero-copy mmap row-gather (safetensors get_slice
   on the 128 raw shard tensors, row->shard arithmetic r // 2_500_012 /
   r % 2_500_012 — PROVEN bitwise-equal to the HF embedding rows on box by
   scripts/flash-next/capture_ple1.py, HARD GATE n_elem_diff=0 over 80 rows x
   160). ZERO re-implementation risk: the hash pipeline runs through the HF
   class itself (`Qwen4ExpTextNGramEmbedding` instantiated meta — its derived
   buffers + `_shift_right_ignore_eos` are real int64 tensors/methods even with
   the 51.2B-row embedding weight on meta, since `init_empty_weights(
   include_buffers=False)` computes buffers for real).

2. `PleLayer` — DEVICE side. Mirrors the torch-composed CPU spec
   scripts/flash-next/pcc_ple1.py (PR #302 — box-verified bit-exact, all 7
   intermediates max_abs=0.0) op-for-op:

     key   = rmsnorm_g(key_proj(emb),    norm_key,   group=2560)   # zero-centered (1+w)
     query = rmsnorm_g(hidden,           norm_query, group=2560)
     gate  = (key.unflatten(4,2560) * query.unflatten(4,2560)).sum(-1,keepdim)/sqrt(2560)
     gate  = |gate|.clamp_min(1e-6).sqrt() * sign(gate)            # signed-sqrt (unusual!)
     gated = sigmoid(gate) * value.unsqueeze(-2)                   # value = value_proj(emb)
     conv  = silu(depthwise_conv1d(rmsnorm_g(gated, norm_conv), k=4, dilation=3, groups=10240))
     out   = gated.flatten(-2) + conv                              # (…,10240)

   The depthwise dilated conv is COMPOSED (no custom kernel): causal left-pad
   (k-1)*d = 9 zeros, then 4 shifted slices x[..., s-3k] ⊙ tap_k summed — the
   exact F.conv1d(dilation=3, groups=10240) decomposition. Decode-time conv
   state (9 slots x 10240, bf16) is a cache slot per contracts §3; the decode
   update path is phase 2 (this module is the prefill/single-layer PCC target).

Injection (model.py, contracts §2.10): layer 1 ONLY (`layer_idx + 1 in
ple_layer_ids`, ple_layer_ids=[2]); `hidden = hidden + PLE(hidden, input_ids)`
BEFORE attn. Phase-1 layout: all weights + activations mesh-REPLICATED
(ReplicateTensorToMesh) — zero collectives, same as hyper_connections.py.

Host-importable by design: no top-level ttnn/torch/transformers imports (the
P4 harness imports this package before devices are up). ttnn resolves lazily in
the device class; transformers/safetensors lazily in PleHostGather.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

HIDDEN = 2560  # config hidden_size
HC_COUNT = 4  # config hc_count
HCW = HIDDEN * HC_COUNT  # 10240 hyper stream width
EPS = 1e-06  # rms_norm_eps (config.json)
CONV_KERNEL = 4  # ple_conv_kernel_size (config.json)
CONV_DILATION = 3  # HF hardcodes dilation=3 (contracts §2.9)
CONV_STATE = (CONV_KERNEL - 1) * CONV_DILATION  # 9 — decode cache slot length
GATE_CLAMP = 1e-6  # signed-sqrt clamp (HF verbatim)

# 102.4 GiB table on-disk layout (P2(d) discovery, capture_ple1.py verified):
# 128 sharded tensors `…ple_embedding.ngram_embedding.shard_{i}.weight`, each
# (2_500_012, 160) bf16; merged dim-0 in shard-index order == nn.Embedding rows.
SHARD_ROWS = 2_500_012
N_SHARDS = 128
TABLE_ROWS = N_SHARDS * SHARD_ROWS  # 320,001,536
EMB_DIM = 160  # head_dim_per_ngram = 2560 / 16
NGRAM_HEADS = 16  # (ngram_size - 1) * heads_per_ngram = 2 * 8

CKPT_PREFIX = "model.language_model."


def _ttnn():
    """Lazy ttnn import — keeps the package importable host-side (no ttnn in the CI container)."""
    import ttnn  # noqa: PLC0415

    return ttnn


# ---------------------------------------------------------------------------
# HOST SIDE — hash + mmap row-gather
# ---------------------------------------------------------------------------
class PleHostGather:
    """n-gram hash + zero-copy mmap row-gather (host only; the table never goes to device).

    The hash math runs through the genuine HF module: we instantiate
    `Qwen4ExpTextNGramEmbedding` under `init_empty_weights(include_buffers=False)`
    so every derived buffer (`layer_multipliers`, `ngram_heads_vocab_sizes`,
    `ngram_heads_offsets`) and the EOS-segment-aware `_shift_right_ignore_eos`
    are the HF-verbatim implementations — only the meta embedding weight itself
    is never touched.

    Row-gather is zero-copy: `safetensors.safe_open(...).get_slice(name)[r:r+1]`
    per looked-up row reads only the touched bytes (OS page cache warms the
    working rows; ~5 KiB per token — ple-host doc §3).
    """

    def __init__(
        self,
        snapshot_dir: str,
        config: Any,
        layer_idx: int = 1,
        ple_layer_index: int = 0,
        ckpt_prefix: str = CKPT_PREFIX,
    ):
        import torch  # noqa: PLC0415 — host-side int64 hash + bf16 rows only
        from accelerate import init_empty_weights  # noqa: PLC0415
        from safetensors import safe_open  # noqa: PLC0415
        from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: PLC0415

        self._torch = torch
        self._safe_open = safe_open
        self.snapshot_dir = snapshot_dir
        self.layer_idx = layer_idx
        self.table_base = f"{ckpt_prefix}layers.{layer_idx}.ple.ple_embedding.ngram_embedding"

        cfg = config.text_config if hasattr(config, "text_config") else config
        with init_empty_weights(include_buffers=False):
            # embedding_dim=2560 -> head_dim_per_ngram 160; the meta weight is never read.
            self.emb_mod = M.Qwen4ExpTextNGramEmbedding(cfg, HIDDEN, layer_idx, ple_layer_index)
        assert self.emb_mod.ngram_embedding.weight.shape == (TABLE_ROWS, EMB_DIM), (
            f"padded table {tuple(self.emb_mod.ngram_embedding.weight.shape)} != ({TABLE_ROWS}, {EMB_DIM})"
        )

        with open(os.path.join(snapshot_dir, "model.safetensors.index.json")) as f:
            self._weight_map = json.load(f)["weight_map"]
        self._handles: Dict[str, Any] = {}

        # sanity: the on-disk shard layout matches the merged-row assumption
        h0 = self._handle_for(self._weight_map[f"{self.table_base}.shard_0.weight"])
        rows, dim = h0.get_slice(f"{self.table_base}.shard_0.weight").get_shape()
        assert (rows, dim) == (SHARD_ROWS, EMB_DIM), f"shard_0 shape {(rows, dim)} != ({SHARD_ROWS}, {EMB_DIM})"

    def _handle_for(self, shard_file: str):
        if shard_file not in self._handles:
            self._handles[shard_file] = self._safe_open(
                os.path.join(self.snapshot_dir, shard_file), framework="pt"
            )
        return self._handles[shard_file]

    def ngram_ids(self, ple_input_ids, previous_context=None):
        """(B,S) int64 token ids -> (B,S,16) int64 embedding rows — verbatim HF algorithm.

        `ple_input_ids` must already be EOS-segment-masked by the caller
        (torch.where(conv_mask, ids, eos) — model.py / harness duty). For chunked
        prefill/decode pass the last `context_len`=2 previous unmasked ids as
        `previous_context` (B,2); defaults to EOS-filled (sequence start).
        """
        torch = self._torch
        m = self.emb_mod
        ids = ple_input_ids.long()
        B = ids.shape[0]
        if previous_context is None:
            prev = ids.new_full((B, m.context_len), m.eos_token_id)
        else:
            prev = previous_context.long()
            assert prev.shape == (B, m.context_len), f"previous_context {tuple(prev.shape)} != ({B}, {m.context_len})"
        token_history = torch.cat([prev, ids], dim=-1)
        shifted = [m._shift_right_ignore_eos(token_history, s) for s in range(m.ngram_size)]
        blocks = []
        for ngram in range(2, m.ngram_size + 1):
            s0 = (ngram - 2) * m.heads_per_ngram
            e0 = s0 + m.heads_per_ngram
            mixed = shifted[0] * m.layer_multipliers[0]
            for p in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[p] * m.layer_multipliers[p])
            hvs = m.ngram_heads_vocab_sizes[s0:e0]
            hoff = m.ngram_heads_offsets[s0:e0]
            nids = torch.remainder(mixed.unsqueeze(-1), hvs.view(1, 1, -1))
            blocks.append(nids + hoff.view(1, 1, -1))
        ngram_ids = torch.cat(blocks, dim=-1)[:, -ids.shape[1]:]
        return ngram_ids  # (B,S,16) int64

    def gather_rows(self, ngram_ids):
        """(B,S,16) int64 rows -> (B,S,2560) bf16 (16 heads x 160 flattened, HF order)."""
        torch = self._torch
        B, S = ngram_ids.shape[0], ngram_ids.shape[1]
        gathered = torch.empty((B, S, NGRAM_HEADS, EMB_DIM), dtype=torch.bfloat16)
        flat = gathered.view(-1, EMB_DIM)
        for flat_i, r in enumerate(ngram_ids.reshape(-1).tolist()):
            shard_idx = r // SHARD_ROWS
            row_in_shard = r % SHARD_ROWS
            name = f"{self.table_base}.shard_{shard_idx}.weight"
            row = self._handle_for(self._weight_map[name]).get_slice(name)[row_in_shard : row_in_shard + 1]
            flat[flat_i] = row[0]
        return gathered.flatten(-2)  # (B,S,2560)

    def ple_embeddings(self, ple_input_ids, previous_context=None):
        """Convenience: ids -> (ngram_ids, gathered rows (B,S,2560) bf16)."""
        ids = self.ngram_ids(ple_input_ids, previous_context=previous_context)
        return ids, self.gather_rows(ids)


# ---------------------------------------------------------------------------
# DEVICE SIDE — composed PLE math (no custom kernel)
# ---------------------------------------------------------------------------
@dataclass
class PleWeights:
    """Device-resident small weights (everything except the host-offloaded table)."""

    key_proj: Any  # (HIDDEN, HCW) bf16 transposed for ttnn.linear
    value_proj: Any  # (HIDDEN, HIDDEN) bf16 transposed
    norm_key_plus1: Any  # (1, 1, HC_COUNT, HIDDEN) bf16 — (1 + norm_key.weight) precomputed
    norm_query_plus1: Any  # (1, 1, HC_COUNT, HIDDEN) bf16
    norm_conv_plus1: Any  # (1, 1, HC_COUNT, HIDDEN) bf16
    conv_taps: List[Any]  # 4 x (1, 1, 1, HCW) bf16 — depthwise taps w[:, 0, k]


class PleLayer:
    """The layer-1 PLE block (device side). Phase-1: mesh-REPLICATED, zero collectives.

    Input tensors are (1, 1, S, HCW/…)-family bf16 TILE_LAYOUT on the mesh. Callers
    pad the token dim to a tile multiple (right-pad zeros; causal conv reads only
    leftward so right-pad columns are garbage-but-harmless — slice back after).
    """

    def __init__(self, mesh_device, state_dict: dict, prefix: str, tensor_cache_path=None):
        ttnn = _ttnn()
        self.device = mesh_device
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,  # fp32 accumulate — matches the fp32-stats norm contract
            packer_l1_acc=False,
        )
        self.weights = self._load_weights(ttnn, state_dict, prefix, tensor_cache_path)

    def _to_device(self, ttnn, tensor_torch, shape=None):
        t = ttnn.from_torch(
            tensor_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),  # phase-1: replicate, zero collectives
        )
        if shape is not None:
            t = ttnn.reshape(t, shape)
        return t

    def _load_weights(self, ttnn, state_dict, prefix, tensor_cache_path):
        import torch  # noqa: PLC0415 — host-side weight transforms only

        def sd(name):
            return state_dict[prefix + name]

        key_w = sd("key_proj.weight").to(torch.bfloat16).t().contiguous()  # (10240,2560)->(2560,10240)
        value_w = sd("value_proj.weight").to(torch.bfloat16).t().contiguous()  # (2560,2560)

        def norm_plus1(name):
            w = sd(name).float()  # (10240,)
            return (1.0 + w).to(torch.bfloat16).reshape(1, 1, HC_COUNT, HIDDEN)

        conv_w = sd("conv1d.weight").to(torch.bfloat16)  # (10240,1,4)
        taps = [conv_w[:, 0, k].contiguous().reshape(1, 1, 1, HCW) for k in range(CONV_KERNEL)]

        return PleWeights(
            key_proj=self._to_device(ttnn, key_w),
            value_proj=self._to_device(ttnn, value_w),
            norm_key_plus1=self._to_device(ttnn, norm_plus1("norm_key.weight")),
            norm_query_plus1=self._to_device(ttnn, norm_plus1("norm_query.weight")),
            norm_conv_plus1=self._to_device(ttnn, norm_plus1("norm_conv.weight")),
            conv_taps=[self._to_device(ttnn, t) for t in taps],
        )

    def _grouped_rms_norm(self, ttnn, x, weight_plus1):
        """Per-2560-group RMSNorm (fp32 stats, eps 1e-6) with zero-centered (1+w) scale.

        x: (…, 10240) -> (…, 10240). Mirrors pcc_ple1.py `rmsnorm_grouped`.
        """
        lead = list(x.shape)[:-1]
        x4 = ttnn.reshape(x, (*lead, HC_COUNT, HIDDEN))
        n4 = ttnn.rms_norm(x4, epsilon=EPS, compute_kernel_config=self.compute_kernel_config)
        n4 = ttnn.multiply(n4, weight_plus1)  # zero-centered scale, broadcast (1,1,4,2560)
        return ttnn.reshape(n4, (*lead, HCW)), n4

    def _depthwise_dilated_conv(self, ttnn, x_normed):
        """silu(causal depthwise conv k=4 d=3) composed from 4 shifted taps.

        x_normed: (1, 1, S, 10240) -> returns (1, 1, S, 10240). Exact
        F.conv1d(dilation=3, groups=10240) decomposition: causal left-pad
        CONV_STATE=9 zeros, out[s] = sum_k tap_k ⊙ x[s - 3k].
        """
        S = x_normed.shape[-2]
        zeros = ttnn.zeros((1, 1, CONV_STATE, HCW), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        xp = ttnn.concat([zeros, x_normed], dim=2)  # (1,1,S+9,10240)
        acc = None
        for k in range(CONV_KERNEL):
            start = CONV_STATE - CONV_DILATION * k
            xs = ttnn.slice(xp, (0, 0, start, 0), (1, 1, start + S, HCW))  # x[s-3k]
            term = ttnn.multiply(xs, self.weights.conv_taps[k])
            acc = term if acc is None else ttnn.add(acc, term)
        return ttnn.silu(acc)

    def forward(self, emb, hidden):
        """emb: (1,1,S,2560) host-gathered n-gram rows; hidden: (1,1,S,10240) hyper stream.

        Op order identical to pcc_ple1.py (bit-exact reference order). Returns
        (1,1,S,10240) — the PLE residual the decoder ADDS to the stream.
        """
        ttnn = _ttnn()
        w = self.weights

        key = ttnn.linear(emb, w.key_proj, compute_kernel_config=self.compute_kernel_config)  # (1,1,S,10240)
        key_n, key_n4 = self._grouped_rms_norm(ttnn, key, w.norm_key_plus1)
        query_n, query_n4 = self._grouped_rms_norm(ttnn, hidden, w.norm_query_plus1)

        prod = ttnn.multiply(key_n4, query_n4)  # (1,1,S,4,2560)
        gate = ttnn.sum(prod, dim=-1, keepdim=True)  # (1,1,S,4,1)
        gate = ttnn.multiply(gate, 1.0 / math.sqrt(HIDDEN))

        # signed-sqrt (contracts §2.9 — unusual, do not "simplify"):
        g_abs = ttnn.abs(gate)
        g_clamped = ttnn.clamp(g_abs, min=GATE_CLAMP)
        g_sqrt = ttnn.sqrt(g_clamped)
        g_sign = ttnn.sign(gate)
        gate_ss = ttnn.multiply(g_sqrt, g_sign)

        value = ttnn.linear(emb, w.value_proj, compute_kernel_config=self.compute_kernel_config)  # (1,1,S,2560)
        value4 = ttnn.unsqueeze(value, -2)  # (1,1,S,1,2560)
        gated = ttnn.multiply(ttnn.sigmoid(gate_ss), value4)  # broadcast (1,1,S,4,2560)

        lead = list(emb.shape)[:-1]
        gated_flat = ttnn.reshape(gated, (*lead, HCW))  # (1,1,S,10240)
        gated_normed, _ = self._grouped_rms_norm(ttnn, gated_flat, w.norm_conv_plus1)

        conv_branch = self._depthwise_dilated_conv(ttnn, gated_normed)  # (1,1,S,10240)
        return ttnn.add(gated_flat, conv_branch)
