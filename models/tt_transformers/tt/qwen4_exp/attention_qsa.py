# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal)
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next QSA phase-1 full-attention stand-in.

Thin ADAPT wrapper over the qwen36 gated full-attention package
``models.demos.blackhole.qwen36.tt.attention.Qwen36GatedAttention``. Per
``docs/qwen38-flash-next-qsa-phase1.md`` (tt-rd): for prompts with <= 2048
visible tokens the QSA token indexer selects EVERY block (indexer_budget 2048 /
compress_ratio 4 -> block_topk 512 blocks x 4 tokens = 2048), so sparse == dense
and QSA == full softmax attention EXACTLY. The indexer kernel (the genuinely
net-new piece) is deferred to phase 2 with zero loss of correctness for bring-up.

The qwen36 fused path ``gated_attention_forward_ttnn`` already implements the
exact flash-next QSA contract (proven bit-exact vs the HF layer-3 reference in
``scripts/flash-next/pcc_qsa3.py``, PASS mrope q/k post-rope PCC 1.0-class):

  q_proj (12288 = q || sigmoid-gate) -> reshape [B,T,H,2*head_dim] -> chunk(2,-1)
    -> query_states [B,T,24,256], gate [B,T,24,256]
  per-head q/k RMSNorm, ZERO-CENTERED (norm_weights_pre_offset=True path)
  partial RoPE on the first rotary_dim = 64 dims (partial_rotary_factor 0.25)
  GQA SDPA (the fused ttnn kernel handles 24 Q / 2 KV internally, groups = 12)
  sigmoid output gate -> o_proj (6144 -> 2560)

The only deltas vs qwen36 (16 Q / 4 KV) are (a) config values — 24 Q / 2 KV,
GQA 12, head_dim 256, eps 1e-6, max_seq 262144 — and (b) the mrope interleave,
which is a PROVEN no-op for text-only (qsa-phase1 §4.2): HF
``apply_interleaved_mrope`` overwrites identical T-row values when all 3 mrope
position rows equal the plain sequence index, so HF cos/sin == standard
duplicate-half RoPE (theta=1e7, 64 rotary dims). ``duplicate_half_rope`` below
reproduces the fork ``compute_rope_freqs`` convention with ZERO permutation.

Spec provenance (all on tt-rd main): docs/qwen38-flash-next-qsa-phase1.md,
docs/qwen38-flash-next-kernel-contracts.md §2.5; executable CPU spec
scripts/flash-next/pcc_qsa3.py. Host-side authoring only; device execution is
owner-gated (tt-commandr preemption decision pending).

Vision/multimodal mrope grids (the only case where interleave != identity) are
SKIP v1 per the port-scoping doc.
"""

from __future__ import annotations

import torch
import ttnn

from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention

# Safetensors weight_map prefix for the flash-next text tower (matches
# scripts/flash-next/pcc_qsa3.py CKPT_PREFIX).
CKPT_PREFIX = "model.language_model."

# Bare keys the qwen36 load_attention_weights expects (q_norm/k_norm get the
# +1.0 zero-centered RMSNorm pre-offset inside the loader).
EXPECTED_BARE_KEYS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "q_norm.weight",
    "k_norm.weight",
)


RMS_NORM_EPS = 1e-6


def build_attention_config(args) -> AttentionConfig:
    """Map Qwen4ExpTextArgs -> qwen36 ``AttentionConfig`` (config-driven delta only).

    ``head_dim`` is the FULL 256 (the 2x-wide q_proj split and the 64-dim partial
    rope live inside the fused op), matching the qwen36 convention.
    """
    return AttentionConfig(
        num_heads=args.num_attention_heads,        # 24
        num_kv_heads=args.num_key_value_heads,     # 2
        head_dim=args.head_dim,                    # 256
        norm_eps=RMS_NORM_EPS,                       # 1e-6 — config.json value (verified in
                                                     # capture_hc.py); the args dataclass omits
                                                     # rms_norm_eps by design (see model.py), so
                                                     # supply the constant like hc/ple modules do.
        max_seq_len=args.max_position_embeddings,  # 262144
    )


def remap_qsa_state_dict(state_dict, layer_idx: int, ckpt_prefix: str = CKPT_PREFIX) -> dict:
    """Strip ``{ckpt_prefix}layers.{i}.self_attn.`` -> bare weight names.

    Accepts the full text-tower state_dict (or any mapping containing the layer's
    self_attn keys) and returns the six bare keys ``load_attention_weights`` reads.
    Raises KeyError listing any missing expected tensor.
    """
    prefix = f"{ckpt_prefix}layers.{layer_idx}.self_attn."
    out = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    missing = [k for k in EXPECTED_BARE_KEYS if k not in out]
    if missing:
        raise KeyError(
            f"remap_qsa_state_dict: layer {layer_idx} missing {missing} "
            f"(looked under prefix {prefix!r}); got {sorted(out)}"
        )
    return out


def duplicate_half_rope(seq_len: int, rotary_dim: int, theta: float, dtype=torch.bfloat16):
    """cos/sin of shape [1, seq_len, rotary_dim], DUPLICATE-HALF convention.

    Mirrors fork ``compute_rope_freqs`` (emb = cat(freqs, freqs), rotate pairs
    (i, i + rotary_dim/2)) and the proven pcc_qsa3.py host convention exactly.
    For flash-next ``rotary_dim = int(partial_rotary_factor * head_dim) = 64``.

    Returns torch tensors; the device caller stages them to ttnn (e.g. via
    ``ttnn.as_tensor``) before ``forward``.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)        # (S, rotary_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)   # duplicate-half (S, rotary_dim)
    cos = emb.cos()[None].to(dtype)           # (1, S, rotary_dim)
    sin = emb.sin()[None].to(dtype)
    return cos, sin


class Qwen4ExpQSAAttention:
    """Phase-1 QSA full-attention layer — ADAPT of qwen36 gated attention.

    Delegates the device op sequence to ``Qwen36GatedAttention`` (concat / paged
    KV-cache branches unchanged). Cos/sin come from ``duplicate_half_rope``
    (text-only exactness per qsa-phase1 §4.2); the device caller stages them to
    ttnn before ``forward``.
    """

    def __init__(self, mesh_device, args, state_dict, layer_idx: int, tensor_cache_path=None):
        self.device = mesh_device
        self.args = args
        self.layer_idx = layer_idx
        self.config = build_attention_config(args)
        remapped = remap_qsa_state_dict(state_dict, layer_idx)
        self.inner = Qwen36GatedAttention(
            mesh_device, self.config, remapped, tensor_cache_path=tensor_cache_path
        )

    @property
    def rotary_dim(self) -> int:
        """Rotary dims per head: int(partial_rotary_factor * head_dim) = 64."""
        return int(self.args.partial_rotary_factor * self.args.head_dim)

    def build_rope(self, seq_len: int, dtype=torch.bfloat16):
        """Duplicate-half cos/sin [1, seq_len, rotary_dim] for this layer's theta."""
        return duplicate_half_rope(seq_len, self.rotary_dim, self.args.rope_theta, dtype=dtype)

    def __call__(self, x, cos, sin, **kwargs):
        """nn.Module-style callable — model.py wires HC block_fns as callables (moe/mixer precedent)."""
        return self.forward(x, cos, sin, **kwargs)

    def forward(self, x, cos, sin, **kwargs):
        return self.inner.forward(x, cos, sin, **kwargs)

    def reset_cache(self):
        self.inner.reset_cache()

    def set_paged_kv_cache(self, k_cache, v_cache):
        self.inner.set_paged_kv_cache(k_cache, v_cache)
