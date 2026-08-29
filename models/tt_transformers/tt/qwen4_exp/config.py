# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal)
# SPDX-License-Identifier: Apache-2.0
"""HF config plumbing for Qwen3.8-Flash-Next (qwen4_exp_text).

Every field below is grounded VERBATIM in the on-box snapshot
`de4b8e4d43b917e7706784d8bb445c9af86a3540` config.json (text_config) and the
P2(c) kernel-contracts extraction (tt-rd `docs/qwen38-flash-next-kernel-contracts.md`
§0 — transformers 5.16.1 `modeling_qwen4_exp.py`). Do not hand-edit values without
re-verifying against those two sources.

Stdlib-only by design: the P4 harness imports this before ttnn/torch are up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Qwen4ExpTextArgs:
    """Text-tower config for Qwen3.8-Flash-Next (vision + MTP skipped in v1)."""

    model_type: str = "qwen4_exp_text"

    # --- core dims (config.json verbatim) ---
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False
    dtype: str = "bfloat16"
    full_attention_interval: int = 4  # layer_types: 3x linear + 1x qwen_sparse_attention

    # --- QSA-layer full attention ---
    num_attention_heads: int = 24
    num_key_value_heads: int = 2  # GQA 24Q/2KV (num_key_value_groups=12)
    head_dim: int = 256
    partial_rotary_factor: float = 0.25  # 64 rotary dims per head
    # rope_parameters verbatim: {'mrope_interleaved': True, 'mrope_section': [11,11,10],
    # 'partial_rotary_factor': 0.25, 'rope_theta': 10000000, 'rope_type': 'default'}
    rope_theta: float = 10_000_000.0
    mrope_interleaved: bool = True
    mrope_section: tuple = (11, 11, 10)  # partitions 32 mrope pairs across T/H/W
    output_gate_type: str = "sigmoid"  # GDN gated norm + attention output gate

    # --- GDN linear attention (36 layers) ---
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128  # key_dim 2048
    linear_num_value_heads: int = 48
    linear_value_head_dim: int = 128  # value_dim 6144
    linear_conv_kernel_dim: int = 4
    mamba_ssm_dtype: str = "float32"  # recurrent-state dtype contract

    # --- MoE (every layer) ---
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True  # fp32 softmax -> top-10 -> L1 renorm (contracts §2.1)
    hidden_act: str = "silu"  # plain SiLU-gate experts, NO SwiGLU clip (contracts §2.2)

    # --- QSA indexer (NET-NEW kernel, contracts §2.4) ---
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4

    # --- hyper-connections (contracts §2.7) ---
    hc_count: int = 4
    hc_lowrank: int = 320

    # --- n-gram PLE (layer 1 only, contracts §2.8/§2.9) ---
    ngram_size: int = 3
    ngram_vocab_size_base: int = 20_000_000
    heads_per_ngram: int = 8
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ple_conv_dilation: int = 3
    split_ngram_parts: int = 128
    make_ngram_vocab_size_divisible_by: int = 128
    ple_layer_ids: tuple = (2,)  # config is 1-based vs (layer_idx + 1) -> inject at idx 1

    # --- cache manager (contracts: 6 state kinds; 3 conv-state slots) ---
    number_of_conv_states: int = 3  # [GDN conv, PLE short-conv, n-gram token history]

    layer_types: tuple = field(default_factory=tuple)  # 48 entries, verbatim from config

    # ---------- derived facts ----------
    @property
    def hyper_width(self) -> int:
        """Residual-stream width end-to-end: hc_count * hidden_size = 10240."""
        return self.hc_count * self.hidden_size

    @property
    def rotary_dim(self) -> int:
        """Rotary dims per attention head: partial_rotary_factor * head_dim = 64."""
        return int(self.partial_rotary_factor * self.head_dim)

    @property
    def gva_factor(self) -> int:
        """GDN group-value-attention expansion: 48 V heads / 16 K heads = 3."""
        return self.linear_num_value_heads // self.linear_num_key_heads

    @property
    def block_topk(self) -> int:
        """QSA indexer block budget: indexer_budget / indexer_compress_ratio = 512."""
        return self.indexer_budget // self.indexer_compress_ratio

    @property
    def qsa_layer_indices(self) -> tuple:
        """0-based indices of qwen_sparse_attention layers: 3, 7, 11, ..., 47."""
        return tuple(range(self.full_attention_interval - 1, self.num_hidden_layers, self.full_attention_interval))

    @property
    def gdn_layer_indices(self) -> tuple:
        """0-based indices of linear_attention (GDN) layers: all others (36)."""
        qsa = set(self.qsa_layer_indices)
        return tuple(i for i in range(self.num_hidden_layers) if i not in qsa)

    @property
    def ple_inject_layer_idx(self) -> int:
        """0-based decoder layer that injects PLE: ple_layer_ids[0] - 1 = 1.

        HF matches `layer_idx + 1 in ple_layer_ids` — the config's [2] means the
        layer whose 0-based index is 1 (off-by-one resolved in contracts §0/§2.9).
        """
        return self.ple_layer_ids[0] - 1

    @property
    def ple_short_conv_state_len(self) -> int:
        """Dilated PLE conv cache slots: (k-1)*dilation = (4-1)*3 = 9."""
        return (self.ple_conv_kernel_size - 1) * self.ple_conv_dilation

    def is_qsa(self, layer_idx: int) -> bool:
        return layer_idx in self.qsa_layer_indices

    def is_gdn(self, layer_idx: int) -> bool:
        return not self.is_qsa(layer_idx)

    @classmethod
    def from_hf_config(cls, cfg: dict[str, Any]) -> "Qwen4ExpTextArgs":
        """Build from the model's config.json dict (accepts root or text_config).

        The snapshot config is composite (root `qwen4_exp` + `text_config`);
        pass either — nested keys win. Unknown/absent keys fall back to the
        snapshot-verified defaults above.
        """
        tc = dict(cfg.get("text_config") or cfg)
        rope = dict(tc.get("rope_parameters") or {})
        mtp = tc.get("mtp")  # recorded for completeness; MTP is SKIP v1
        _ = mtp
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        out = {}
        for k in list(tc):
            if k in known and k not in {"layer_types", "mrope_section", "ple_layer_ids"}:
                out[k] = tc[k]
        if "layer_types" in tc:
            out["layer_types"] = tuple(tc["layer_types"])
        if rope:
            out["rope_theta"] = rope.get("rope_theta", cls.rope_theta)
            out["mrope_interleaved"] = rope.get("mrope_interleaved", cls.mrope_interleaved)
            if "mrope_section" in rope:
                out["mrope_section"] = tuple(rope["mrope_section"])
            out["partial_rotary_factor"] = rope.get("partial_rotary_factor", cls.partial_rotary_factor)
        if "ple_layer_ids" in tc:
            out["ple_layer_ids"] = tuple(tc["ple_layer_ids"])
        args = cls(**out)
        # sanity pins (snapshot de4b8e4d) — fail loud if a future checkpoint drifts
        assert args.hyper_width == 10240, args.hyper_width
        assert args.gva_factor == 3, args.gva_factor
        assert args.block_topk == 512, args.block_topk
        assert len(args.qsa_layer_indices) == 12 and len(args.gdn_layer_indices) == 36
        return args


# Snapshot-pinned singleton for the QB2 checkpoint (tests import this).
QWEN38_FLASH_NEXT_DE4B8E4D = Qwen4ExpTextArgs()
