# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal)
# SPDX-License-Identifier: Apache-2.0
"""Static configuration for the Qwen3.8-Flash-Next MoE-512 block (qwen4_exp_text).

Every value is grounded VERBATIM in the on-box snapshot
`de4b8e4d43b917e7706784d8bb445c9af86a3540` config.json (text_config) and the build
spec `docs/qwen38-flash-next-moe-bfp4.md` (tt-rd main, PR #296):

  num_experts=512, num_experts_per_tok=10, moe_intermediate_size=640,
  shared_expert_intermediate_size=640, hidden_size=2560, norm_topk_prob=True,
  hidden_act="silu"  (plain SiLU-gate experts — NO gpt_oss SwiGLU clip/alpha).

Correctness-critical contracts this config encodes (do NOT "fix" without
re-reading the spec):
  * Router: softmax over ALL 512 experts (fp32) -> top-10 -> L1 renorm
    (norm_topk_prob=True). NOT the gpt_oss topk-then-softmax order.
  * Fused gate_up weight uses CONTIGUOUS halves (gate=w[:640], up=w[640:]),
    NOT the gpt_oss interleave ([..., ::2] / [..., 1::2]).
  * ZERO biases anywhere: no router bias, no gate/up/down expert bias.
  * Expert weights BFP4 (owner-locked); down_proj BFP8 fallback per spec §7
    (deepseek_v3 art: bfloat8_b if hf_name == "down_proj" else bfloat4_b).
"""
from dataclasses import dataclass

import ttnn


@dataclass(frozen=True)
class Qwen4ExpMoEConfig:
    num_experts: int            # 512
    num_experts_per_tok: int    # 10
    intermediate_size: int      # 640 (routed experts)
    shared_intermediate_size: int  # 640 (shared expert)
    hidden_size: int            # 2560
    norm_topk_prob: bool        # True
    weight_dtype: object        # ttnn dtype for gate/up expert weights (BFP4 locked)
    down_dtype: object          # ttnn dtype for down_proj (BFP4; spec §7 fallback BFP8)

    @classmethod
    def from_args(cls, args, weight_dtype=ttnn.bfloat4_b, down_dtype=ttnn.bfloat4_b) -> "Qwen4ExpMoEConfig":
        """Build from Qwen4ExpTextArgs (config.py dataclass) or an HF text config dict/object.

        Accepts either the qwen4_exp skeleton args dataclass (attribute access) or a
        plain dict / HF PretrainedConfig (key or attribute access).
        """
        if isinstance(args, dict):
            g = lambda k, d=None: args.get(k, d)
        else:
            g = lambda k, d=None: getattr(args, k, d)
        cfg = cls(
            num_experts=int(g("num_experts")),
            num_experts_per_tok=int(g("num_experts_per_tok")),
            intermediate_size=int(g("moe_intermediate_size")),
            shared_intermediate_size=int(g("shared_expert_intermediate_size")),
            hidden_size=int(g("hidden_size")),
            norm_topk_prob=bool(g("norm_topk_prob", True)),
            weight_dtype=weight_dtype,
            down_dtype=down_dtype,
        )
        # sanity pins @ snapshot de4b8e4d — fail loud if a future checkpoint drifts
        assert cfg.num_experts == 512, cfg.num_experts
        assert cfg.num_experts_per_tok == 10, cfg.num_experts_per_tok
        assert cfg.intermediate_size == 640, cfg.intermediate_size
        assert cfg.shared_intermediate_size == 640, cfg.shared_intermediate_size
        assert cfg.hidden_size == 2560, cfg.hidden_size
        assert cfg.norm_topk_prob is True
        return cfg
