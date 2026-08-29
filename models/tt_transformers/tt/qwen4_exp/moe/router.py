# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal)
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next MoE router — softmax-all-512(fp32) -> top-10 -> L1 renorm.

CORRECTNESS-CRITICAL ORDER (moe-bfp4 spec §2, verbatim from transformers 5.16.1
modeling_qwen4_exp.py lines 910–912):

    router_probs = softmax(router_logits, dtype=float, dim=-1)   # over ALL 512
    top_value, indices = topk(router_probs, 10, dim=-1)
    if norm_topk_prob:  # True
        top_value = top_value / top_value.sum(dim=-1, keepdim=True)

Do NOT copy the gpt_oss order (models/demos/gpt_oss/tt/topk.py::topk_router does
topk on the LOGITS first, then softmax over the k selected — a different
distribution). The gpt_oss fused matmul+topk+softmax kernel requires exactly 128
experts and is NOT applicable to 512 — we use the plain linear->softmax->topk path.

Router weight: (512, 2560) bf16, NO bias (module_map.txt: `mlp.gate.weight` only).
Transposed to (2560, 512) for ttnn.linear; replicated across the mesh (2.6 MB/layer).
"""

import ttnn


class Qwen4ExpRouter:
    """Top-10-of-softmax router for the 512-expert MoE."""

    def __init__(self, mesh_device, config, state_dict, tensor_cache_path=None):
        """
        Args:
            mesh_device: ttnn mesh device (1,4).
            config: Qwen4ExpMoEConfig.
            state_dict: mapping with key "gate.weight" : torch (512, 2560) bf16
                (the HF `mlp.gate.weight`). No "gate.bias" key may exist.
            tensor_cache_path: optional converted-tensor cache dir.
        """
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        if "gate.bias" in (state_dict or {}):
            raise ValueError("flash-next MoE router has NO bias (spec §1) — refusing to load one")
        torch_weight = None
        if state_dict:
            w = state_dict["gate.weight"]
            assert tuple(w.shape) == (self.num_experts, self.hidden_dim), w.shape
            # ttnn.linear computes x @ W (K,N): transpose HF (out,in) -> (in,out).
            torch_weight = w.transpose(0, 1).contiguous()
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device is not None else None
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,  # routing decisions are gate-critical (spec §7)
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
            cache_file_name=self._cache_name(tensor_cache_path, "router_weight"),
        )
        # fp32 softmax compute config (verbatim from gpt_oss topk.py lines 32–38)
        self.softmax_compute_config = (
            ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi3,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            if mesh_device is not None
            else None
        )

    @staticmethod
    def _cache_name(tensor_cache_path, name):
        if tensor_cache_path is None:
            return None
        import os

        return os.path.join(tensor_cache_path, f"{name}.tensorbin")

    def __call__(self, hidden_states):
        """Route tokens.

        Args:
            hidden_states: ttnn tensor [..., hidden_dim] (bf16, TILE).

        Returns:
            (expert_indices, routing_weights_512):
              expert_indices: ttnn [B, top_k] (uint32-ish indices, sorted desc by weight)
              routing_weights_512: ttnn [B, 512] bf16 — the L1-renormed top-k weights
                  SCATTERED back to the full 512-wide vector (sparse_matmul sparsity).
        """
        actual_tokens = hidden_states.volume() // self.hidden_dim
        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))
        is_decode = actual_tokens <= 128
        mem_config = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG

        router_logits = ttnn.linear(
            hidden_states,
            self.weight,
            memory_config=mem_config,
        )

        # Pre-allocate the 512-wide zeros vector for the scatter BEFORE freeing
        # router_logits (gpt_oss topk.py: ttnn.scatter(ttnn.zeros_like(g), ...)).
        zeros = ttnn.zeros_like(router_logits)

        # 1) softmax over ALL 512 experts, fp32 accumulation — BEFORE topk.
        probs = ttnn.softmax(
            router_logits,
            dim=-1,
            numeric_stable=True,
            compute_kernel_config=self.softmax_compute_config,
        )
        ttnn.deallocate(router_logits)

        # 2) top-10 of the probabilities (sorted so weights/indices align).
        top_w, top_i = ttnn.topk(probs, k=self.top_k, dim=-1, sorted=True)
        ttnn.deallocate(probs)

        # 3) L1 renorm (norm_topk_prob=True): w_i / sum_j w_j over the 10 selected.
        denom = ttnn.sum(top_w, dim=-1, keepdim=True)
        top_w = ttnn.div(top_w, denom)
        ttnn.deallocate(denom)

        # 4) scatter back to the 512-wide sparse routing vector (sparse_matmul input).
        routing_weights_512 = ttnn.scatter(zeros, dim=1, index=top_i, src=top_w)
        ttnn.deallocate(zeros)
        return top_i, routing_weights_512
