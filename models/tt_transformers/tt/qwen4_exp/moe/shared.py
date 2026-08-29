# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal)
# SPDX-License-Identifier: Apache-2.0
"""Shared expert + sigmoid scalar gate for the Qwen3.8-Flash-Next MoE-512 block.

HF semantics (moe-bfp4 spec §5, verbatim from modeling_qwen4_exp.py lines 924–936):

    shared_expert = MLP(intermediate=640)                       # dense SwiGLU
    shared_expert_gate = Linear(hidden, 1, bias=False)          # (1, 2560)
    ...
    shared_out = sigmoid(shared_expert_gate(x)) * shared_expert(x)
    expert_out = expert_out + shared_out

Runs for EVERY token (dense, not sparse). Weights BFP8 per the spec §6 budget;
the (1, 2560) gate scalar is bf16. TP: gate/up col-sharded (dim 3), down row-sharded
(dim 2) — partial sums need a TP all-reduce after down when tp>1 (handled by the
caller together with the routed-expert reduction's all-reduce).
"""

import ttnn

from .weights import SharedExpertWeights


def shared_expert_forward(hidden_states, weights: SharedExpertWeights, config):
    """Dense shared-expert SwiGLU with sigmoid scalar gate.

    Args:
        hidden_states: ttnn [..., hidden_size] (bf16/bfp8, TILE).
        weights: SharedExpertWeights (device tensors).
        config: Qwen4ExpMoEConfig.

    Returns:
        ttnn [..., hidden_size] — sigmoid(gate(x)) * shared(x), pre-all-reduce.
    """
    gate = ttnn.linear(hidden_states, weights.gate_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
    up = ttnn.linear(hidden_states, weights.up_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
    gate_act = ttnn.silu(gate)
    ttnn.deallocate(gate)
    hidden = ttnn.mul(gate_act, up)
    ttnn.deallocate(gate_act)
    ttnn.deallocate(up)
    shared = ttnn.linear(hidden, weights.down_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(hidden)

    gate_scalar = ttnn.linear(
        hidden_states, weights.shared_expert_gate, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    gate_sig = ttnn.sigmoid(gate_scalar)
    ttnn.deallocate(gate_scalar)
    out = ttnn.mul(shared, gate_sig)
    ttnn.deallocate(shared)
    ttnn.deallocate(gate_sig)
    return out
