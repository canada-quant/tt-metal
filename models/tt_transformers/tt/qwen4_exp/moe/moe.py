# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal)
# SPDX-License-Identifier: Apache-2.0
"""Top-level Qwen3.8-Flash-Next MoE-512 block — router + routed experts + shared expert.

Composes (per moe-bfp4 spec, validated bit-exact torch-side in
scripts/flash-next/pcc_moe0.py, tt-rd PR #297):

    indices, rw512 = router(x)                       # softmax(512,fp32)->top10->L1
    routed       = experts_decode_forward(x, rw512)  # 3x sparse_matmul nnz=None
    shared       = shared_expert_forward(x)          # dense SwiGLU, sigmoid gate
    out          = routed + shared                   # (HF: expert_output + shared)

Every layer (0..47) is an MoE block in this model. Decode path only for the P4
single-layer PCC bring-up; prefill grouped-GEMM is a later leg (spec §8).
"""

import ttnn
from models.demos.gpt_oss.tt.experts.config import ProgramConfig

from .config import Qwen4ExpMoEConfig
from .experts import experts_decode_forward
from .router import Qwen4ExpRouter
from .shared import shared_expert_forward
from .weights import load_expert_weights, load_shared_expert_weights, remap_flash_next_moe_state_dict


class Qwen4ExpMoE:
    """One MoE decoder sub-block (layer N `mlp.`)."""

    def __init__(
        self,
        mesh_device,
        config: Qwen4ExpMoEConfig,
        state_dict=None,
        tensor_cache_path=None,
        program_config: ProgramConfig = None,
        ccl_manager=None,
        mesh_config=None,
        tp: int = 1,
    ):
        """
        Args:
            mesh_device: ttnn mesh device (1,4) or None (host-side stub).
            config: Qwen4ExpMoEConfig.
            state_dict: HF layer-N `mlp.*` tensors (raw HF names) or None.
            tensor_cache_path: optional converted-tensor cache dir.
            program_config: gpt_oss ProgramConfig (defaults: in-tree tuned cores).
            ccl_manager / mesh_config: required only when tp > 1 (all-reduce).
            tp: tensor-parallel degree (QB2 plan: 4 on mesh (1,4)).
        """
        self.config = config
        self.tp = tp
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config
        # Core grids MUST tile num_blocks_total rectangularly (sparse_matmul mcast1d
        # factory: num_cores_with_work == receiver bounding box). gpt_oss defaults
        # (3,4)/(5,6) fail for qwen4_exp dims: gate/up Nt=20 -> pcn=2 -> 10 blocks
        # (10%3!=0 TT_FATAL), down Nt=80 -> pcn=3 -> 27 (27%5!=0). (5,4) gives
        # gate/up pcn=1 -> 20 blocks (20%5==0) and down pcn=4 -> 20 (20%5==0).
        self.program_config = (
            program_config
            if program_config is not None
            else ProgramConfig(decode_gate_up_cores=(5, 4), decode_down_cores=(5, 4))
        )
        sd = remap_flash_next_moe_state_dict(state_dict) if state_dict else None
        self.router = Qwen4ExpRouter(mesh_device, config, sd, tensor_cache_path=tensor_cache_path)
        self.expert_weights = load_expert_weights(
            mesh_device, config, sd, tensor_cache_path=tensor_cache_path
        )
        self.shared_weights = load_shared_expert_weights(
            mesh_device, config, sd, tensor_cache_path=tensor_cache_path
        )

    def __call__(self, hidden_states):
        """x -> routed experts + gated shared expert.

        Args:
            hidden_states: ttnn [1, batch, seq, hidden_size]. seq == 1 is the
                decode path (proven in the moe leg); seq > 1 loops the same
                decode path per token and concats on dim 2 — the exact prefill
                pattern pcc_device_moe.py used (per-token [1,B,1,H] calls) —
                because the router/sparse_matmul/scatter path is seq=1-shaped.
                The model leg feeds the HC-mixed stream as [1,1,Sp,2560].

        Returns:
            ttnn [1, batch, seq, hidden_size] mlp block output.
        """
        seq = hidden_states.shape[2]
        if seq > 1:
            outs = []
            for t in range(seq):
                x_t = hidden_states[:, :, t : t + 1, :]
                outs.append(self(x_t))
                ttnn.deallocate(x_t)
            out = ttnn.concat(outs, dim=2)
            for o in outs:
                ttnn.deallocate(o)
            return out
        indices, rw512 = self.router(hidden_states)
        ttnn.deallocate(indices)  # sparse_matmul consumes only the scattered 512-wide vector
        routed = experts_decode_forward(
            hidden_states,
            rw512,
            self.expert_weights,
            self.config,
            hidden_states.device(),
            self.program_config,
            ccl_manager=self.ccl_manager,
            mesh_config=self.mesh_config,
            tp=self.tp,
        )
        shared = shared_expert_forward(hidden_states, self.shared_weights, self.config)
        # shared down_proj is row-sharded (dim 2: K=640 -> 160/device) so
        # shared_expert_forward returns PARTIAL sums per device. The routed path
        # all-reduces inside experts_decode_forward; the shared partials must be
        # reduced here before the add (separate collective — a fused
        # single-all-reduce for routed+shared is a comms follow-up).
        if self.tp > 1:
            from models.demos.gpt_oss.tt.experts.operations import (
                apply_tensor_parallel_allreduce,
            )

            shared = apply_tensor_parallel_allreduce(
                shared,
                self.mesh_config,
                hidden_states.device(),
                hidden_states.shape[2],
                self.ccl_manager,
            )
        out = ttnn.add(routed, shared)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out
