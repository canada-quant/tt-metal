# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal)
# SPDX-License-Identifier: Apache-2.0
"""MoE-512 block for Qwen3.8-Flash-Next (qwen4_exp) — BFP4 experts, owner-locked.

Build spec: docs/qwen38-flash-next-moe-bfp4.md (tt-rd main, PR #296). The three
correctness-critical contracts (do NOT "fix" without re-reading the spec):
  1. Router: softmax over ALL 512 experts (fp32) -> top-10 -> L1 renorm
     (NOT the gpt_oss topk-then-softmax order; NOT the 128-expert fused kernel).
  2. Fused gate_up weight split as CONTIGUOUS halves (gate=w[:640], up=w[640:]),
     NOT the gpt_oss interleave.
  3. ZERO biases anywhere in the block.

Device budget (spec §6): experts BFP4 59.77 GiB across the mesh; whole-model
device total ≈ 64.7 / 128 GiB. down_proj BFP8 fallback per spec §7.
"""

from models.tt_transformers.tt.qwen4_exp.moe.config import Qwen4ExpMoEConfig
from models.tt_transformers.tt.qwen4_exp.moe.moe import Qwen4ExpMoE
from models.tt_transformers.tt.qwen4_exp.moe.weights import remap_flash_next_moe_state_dict

__all__ = ["Qwen4ExpMoE", "Qwen4ExpMoEConfig", "remap_flash_next_moe_state_dict"]
