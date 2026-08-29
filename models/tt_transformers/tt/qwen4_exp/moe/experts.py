# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal)
# SPDX-License-Identifier: Apache-2.0
"""Decode forward for the Qwen3.8-Flash-Next routed experts (seq_len=1).

Adapted from `models/demos/gpt_oss/tt/experts/decode.py::decode_forward`
(verbatim in-tree Blackhole art) with the flash-next deltas from moe-bfp4 spec §4:

  * THREE `ttnn.sparse_matmul` calls (gate, up, down) with **nnz=None** (inferred
    at runtime) — the hard requirement carried verbatim from the gpt_oss comment:
    a static nnz deadlocks the in0-mcast receivers on Blackhole when the actual
    non-zero sparsity count is smaller (tt-metal#45943/#45052, read-only citation).
  * `is_input_a_sparse=True` on the DOWN projection (its input is the sparse
    SwiGLU output).
  * SwiGLU = silu(gate) * up — PLAIN, no clip/alpha (hidden_act="silu",
    contracts §2.2; gpt_oss's swiglu_limit/alpha clamping does NOT apply).
  * ZERO biases — the three gpt_oss `ttnn.add(..., bias)` calls are REMOVED.
  * EP=1 on QB2 (TP=4, experts col/row-sharded, no moe_routing_remap).
  * Activations bfloat8_b; output_tile 32x32; L1 memory config on matmul outputs.

Program configs: reuse the gpt_oss `ProgramConfig` builder
(models/demos/gpt_oss/tt/experts/config.py) — it carries the in0_block_w
snap-to-divisor and out_subblock_w<=8 discipline verbatim; defaults are the
starting point for the device leg (tuning is a P4-device item).
"""

import ttnn
from models.demos.gpt_oss.tt.experts.config import ProgramConfig

from .weights import ExpertWeights


def experts_decode_forward(
    hidden_states,
    routing_weights_512,
    weights: ExpertWeights,
    config,
    mesh_device,
    program_config: ProgramConfig,
    ccl_manager=None,
    mesh_config=None,
    tp: int = 1,
):
    """Decode forward — one token (seq_len=1) through the 512 routed experts.

    Args:
        hidden_states: ttnn [1, batch, 1, hidden_size] (bf16/bfp8, TILE).
        routing_weights_512: ttnn [batch, 512] bf16 — L1-renormed top-10 weights
            scattered to the full 512-wide vector (from Qwen4ExpRouter).
        weights: ExpertWeights (BFP4 device tensors).
        config: Qwen4ExpMoEConfig.
        mesh_device: ttnn mesh device (1,4).
        program_config: gpt_oss ProgramConfig (matmul program-config discipline).
        ccl_manager: optional CCL manager for TP>1 all-reduce.
        tp: tensor-parallel degree (all-reduce skipped when 1).

    Returns:
        ttnn [1, batch, 1, hidden_size] — routed expert reduction (pre-shared-expert).
    """
    activation_dtype = ttnn.bfloat8_b
    batch_dim = 1
    seq_dim = 2
    batch_size = hidden_states.shape[batch_dim]
    seq_len = hidden_states.shape[seq_dim]

    if seq_len != 1:
        raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")
    if batch_size != 1:
        raise NotImplementedError(f"Currently only batch_size=1 supported, got {batch_size}")

    # Sparse routing vector -> ROW_MAJOR 4D sparsity (gpt_oss decode.py verbatim).
    sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights_512), ttnn.ROW_MAJOR_LAYOUT)

    output_tile = ttnn.Tile([32, 32])

    # ---- Gate projection (sparse_matmul #1) ----
    gate = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_proj,
        sparsity=sparsity,
        # nnz intentionally omitted (None -> inferred at runtime). Passing a static
        # nnz makes the sparse_matmul in0-mcast receivers loop a fixed count while the
        # sender only mcasts for the *actual* non-zero `sparsity` entries. The decode
        # routing weights (softmax over top-k, scattered) frequently have <k non-zeros
        # on Blackhole (small weights flush to 0), so a static nnz != actual count and
        # the receivers deadlock in noc_semaphore_wait. Inferring the count is robust.
        # See tenstorrent/tt-metal#45943 (op deadlock) / #45052 (gpt-oss hang).
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(
            hidden_states.shape[2], weights.gate_proj.shape[3], k=hidden_states.shape[-1]
        ),
        dtype=activation_dtype,
    )
    gate = ttnn.reshape(gate, (batch_size, config.num_experts, 1, weights.intermediate_size_per_device))
    gate = ttnn.transpose(gate, 1, 2)
    gate = ttnn.reshape(gate, (batch_size, config.num_experts, weights.intermediate_size_per_device))

    # ---- Up projection (sparse_matmul #2) ----
    up = ttnn.sparse_matmul(
        hidden_states,
        weights.up_proj,
        sparsity=sparsity,
        nnz=None,  # inferred — see gate comment above
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(
            hidden_states.shape[2], weights.up_proj.shape[3], k=hidden_states.shape[-1]
        ),
        dtype=activation_dtype,
    )
    hidden_states.deallocate(True)
    up = ttnn.reshape(up, (batch_size, config.num_experts, 1, weights.intermediate_size_per_device))
    up = ttnn.transpose(up, 1, 2)
    up = ttnn.reshape(up, (batch_size, config.num_experts, weights.intermediate_size_per_device))

    # ---- SwiGLU: silu(gate) * up — PLAIN, no clip/alpha (contracts §2.2) ----
    gate_act = ttnn.silu(gate)
    ttnn.deallocate(gate)
    down_input = ttnn.mul(gate_act, up)
    ttnn.deallocate(gate_act)
    ttnn.deallocate(up)

    down_input = ttnn.transpose(down_input, 1, 0)
    down_input = ttnn.reshape(
        down_input, (1, config.num_experts, seq_len, weights.intermediate_size_per_device)
    )

    # ---- Down projection (sparse_matmul #3; sparse input A) ----
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=None,  # inferred — see gate comment above
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        program_config=program_config.get_decode_down_config(
            down_input.shape[2], weights.down_proj.shape[-1], k=down_input.shape[-1]
        ),
        dtype=activation_dtype,
    )
    down_input.deallocate(True)
    sparsity.deallocate(True)

    # ---- Apply routing weights + reduce over experts ----
    next_states = ttnn.permute(down, (0, 2, 1, 3))
    next_states = ttnn.reshape(next_states, (batch_size, config.num_experts, config.hidden_size))
    rw = ttnn.permute(routing_weights_512, (1, 0))
    rw = ttnn.reshape(rw, (batch_size, config.num_experts, 1))
    next_states = ttnn.mul(next_states, rw)
    ttnn.deallocate(rw)

    next_states = ttnn.sum(next_states, dim=1)
    next_states = ttnn.unsqueeze_to_4D(next_states)

    # ---- TP all-reduce (skipped for tp==1; EP=1 on QB2 → no EP remap) ----
    if tp > 1:
        if ccl_manager is None or mesh_config is None:
            raise ValueError("tp>1 requires ccl_manager AND mesh_config for the TP all-reduce")
        from models.demos.gpt_oss.tt.experts.operations import apply_tensor_parallel_allreduce

        next_states = ttnn.unsqueeze_to_4D(next_states)
        next_states = apply_tensor_parallel_allreduce(
            next_states,
            mesh_config,
            mesh_device,
            seq_len,
            ccl_manager,
        )

    next_states = ttnn.reshape(
        next_states,
        (1, batch_size, seq_len, config.hidden_size),
        (1, batch_size, max(32, seq_len), config.hidden_size),
    )
    return next_states
