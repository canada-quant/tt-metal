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

import os

import ttnn


def _replicated_mesh_to_host(mesh_device, t):
    """Host copy of a mesh-REPLICATED tensor (qwen4_exp model-leg phase-1 layout).

    Plain ttnn.to_torch on a replicate-mapped mesh tensor is TT_FATAL (a mesh
    composer is required): concat along dim 0 stacks the identical replicas —
    take the first. Same capture style as the model.py layer_debug precedent
    (models/tt_transformers/tt/qwen4_exp/model.py::_replicated_mesh_to_host);
    ttnn is module-level here so it is not a parameter.
    """
    if mesh_device is None:
        return ttnn.to_torch(t)
    cat = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return cat[: t.shape[0]]


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
        self.mesh_device = mesh_device  # Window H: needed by the layer_debug host captures
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
        # W2 (tt-rd gdn-fidelity plan §7.2): env-guarded fp32 router path.
        # When set, the router GEMM accumulates in fp32 (fp32_dest_acc_en) and
        # the softmax->topk->renorm->scatter decision path runs in fp32 —
        # matching the HF cast points (modeling_qwen4_exp.py L909–911: bf16
        # logits -> softmax(dtype=float)); the bf16 out tile after the GEMM IS
        # the HF cast point, so we upcast with typecast (proven op on this
        # image: qwen36 gdn fused_chunk.py:174) rather than an fp32 out tile.
        # Default off = zero behavior change (compute_kernel_config=None is
        # the ttnn.linear default).
        self.router_fp32 = os.environ.get("QWEN4EXP_ROUTER_FP32", "0") == "1"
        self.router_linear_compute_config = (
            ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            if (self.router_fp32 and mesh_device is not None)
            else None
        )
        # QWEN4EXP_TOPK_STABLE (pre-staged 2026-08-31; tt-rd gdn-fidelity plan
        # §7.11): env-guarded deterministic near-tie break at the top-10 — see
        # __call__ step 2 and _stable_bias. Default off = zero behavior change.
        self.topk_stable = os.environ.get("QWEN4EXP_TOPK_STABLE", "0") == "1"
        self._stable_bias_t = None

    def _stable_bias(self):
        """Cached fp32 [1, 512] index-priority epsilon ramp on device
        (QWEN4EXP_TOPK_STABLE): bias_i = i * 2e-9. Built lazily, once, via
        torch -> ttnn.from_torch with the same replicate-mesh mapping and
        TILE layout as the router weight, so ttnn.subtract broadcasts it
        across the [tokens, 512] probs tensor."""
        if self._stable_bias_t is None:
            import torch

            ramp = (torch.arange(self.num_experts, dtype=torch.float32) * 2e-9).reshape(1, -1)
            self._stable_bias_t = ttnn.from_torch(
                ramp,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if self.mesh_device is not None else None,
            )
        return self._stable_bias_t

    @staticmethod
    def _cache_name(tensor_cache_path, name):
        if tensor_cache_path is None:
            return None
        import os

        return os.path.join(tensor_cache_path, f"{name}.tensorbin")

    def __call__(self, hidden_states, layer_debug=None):
        """Route tokens.

        Args:
            hidden_states: ttnn tensor [..., hidden_dim] (bf16, TILE).
            layer_debug: optional flat capture dict (Window H §6.14; model-leg
                divergence localization). When set, the bf16 router logits and the
                top-10 expert indices are host-captured (list-appended per call —
                the seq>1 MoE loop fires this router once per token). Default None
                = zero behavior change on every production path.
                W4 (tt-rd gdn-fidelity plan §7.2): when layer_debug carries a
                "force_router_indices" list, each call pops one [1,10] HF ref
                index set and routes to it INSTEAD of the computed top_i (gate
                weights stay device-computed — indices only are forced), the
                teacher-forced routing diagnostic isolating continuous math from
                routing divergence. Absent/empty list = computed indices (zero
                behavior change).
                W2 (tt-rd gdn-fidelity plan §7.2): when the env knob
                QWEN4EXP_ROUTER_FP32=1 is set at load time, the router GEMM
                accumulates in fp32 and the decision path (softmax -> topk ->
                renorm -> scatter) runs in fp32 at the HF cast points; the
                rw512 return stays bf16 (expert-combine input unchanged).
                Unset = zero behavior change.
                QWEN4EXP_TOPK_STABLE=1 (pre-staged 2026-08-31, §7.11): an fp32
                index-priority epsilon ramp (i * 2e-9) is subtracted from the
                softmax probabilities BEFORE the top-10, making near-tie
                selection a deterministic function of expert id (root cause:
                MoE router near-tie chaotic amplification, §7.9). Default off
                = zero behavior change.

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
            compute_kernel_config=self.router_linear_compute_config,
        )
        if layer_debug is not None:  # Window H: host-capture router logits (bf16; fp32 under the W2 knob)
            layer_debug.setdefault("moe_router_logits", []).append(
                _replicated_mesh_to_host(self.mesh_device, router_logits)
            )

        # Pre-allocate the 512-wide zeros vector for the scatter BEFORE freeing
        # router_logits (gpt_oss topk.py: ttnn.scatter(ttnn.zeros_like(g), ...)).
        zeros = ttnn.zeros_like(router_logits)  # bf16 ALWAYS (created pre-typecast):
        # ttnn.scatter TT_FATALs on fp32 TILE input (scatter.cpp:109; hit by the
        # first W2 leg 2026-08-31 02:28:29Z) — the scatter machinery stays bf16.
        if self.router_fp32:  # W2: fp32 decision path from the softmax onward
            # (bf16 out tile == HF cast point; typecast upcasts the tile).
            router_logits = ttnn.typecast(router_logits, ttnn.float32)

        # 1) softmax over ALL 512 experts, fp32 accumulation — BEFORE topk.
        probs = ttnn.softmax(
            router_logits,
            dim=-1,
            numeric_stable=True,
            compute_kernel_config=self.softmax_compute_config,
        )
        ttnn.deallocate(router_logits)

        # 2) top-10 of the probabilities (sorted so weights/indices align).
        if self.topk_stable:
            # QWEN4EXP_TOPK_STABLE=1 (pre-staged 2026-08-31; default off = zero
            # behavior change): deterministic near-tie break — subtract an
            # index-priority epsilon ramp from the probabilities BEFORE topk,
            # in fp32:
            #     prob'_i = prob_i - i * 2e-9     (span over 512 experts ~1e-6)
            # The span sits below any meaningful softmax gap but above fp32
            # exact-tie noise, so selection among experts tied at the numeric
            # floor becomes a deterministic function of expert id (lower id
            # wins) — removing kernel-config/dtype-path dependence of the
            # routing DECISION itself (root cause: MoE router near-tie chaotic
            # amplification, tt-rd gdn-fidelity plan §7.2/§7.9). fp32 only: the
            # ramp is far below bf16 ULP, so a bf16 subtraction would be a
            # silent no-op. Applied post-softmax/pre-topk — the spec §2 order
            # (softmax-all-512 -> top-10) is preserved.
            src_t = probs
            biased_in = src_t if self.router_fp32 else ttnn.typecast(src_t, ttnn.float32)
            biased = ttnn.subtract(biased_in, self._stable_bias())
            ttnn.deallocate(src_t)
            if biased_in is not src_t:
                ttnn.deallocate(biased_in)
            probs = biased
        top_w, top_i = ttnn.topk(probs, k=self.top_k, dim=-1, sorted=True)
        ttnn.deallocate(probs)
        if self.topk_stable and not self.router_fp32:
            # restore the downstream bf16 contract (top_w feeds the renorm and
            # the bf16-only scatter machinery — see the zeros comment above)
            top_w = ttnn.typecast(top_w, ttnn.bfloat16)
        if layer_debug is not None:  # Window H: host-capture top-10 expert indices
            layer_debug.setdefault("moe_router_indices", []).append(
                _replicated_mesh_to_host(self.mesh_device, top_i)
            )
        force_q = layer_debug.get("force_router_indices") if layer_debug is not None else None
        if force_q:  # W4: teacher-forced routing — swap in the HF ref indices
            # for this token (popped in slot order; the computed indices were
            # host-captured above, so the harness still measures computed-vs-ref
            # Jaccard in a force leg). Mirror the live top_i tensor properties
            # (0.19-era getters, defensive fallbacks).
            forced = force_q.pop(0)
            fi_dev = ttnn.from_torch(
                forced,  # torch [1,10] int (harness slices hc{L} moe_router_indices)
                dtype=top_i.get_dtype() if hasattr(top_i, "get_dtype") else ttnn.uint32,
                layout=top_i.get_layout() if hasattr(top_i, "get_layout") else ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=top_i.memory_config() if hasattr(top_i, "memory_config") else ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if self.mesh_device is not None else None,
            )
            ttnn.deallocate(top_i)
            top_i = fi_dev

        # 3) L1 renorm (norm_topk_prob=True): w_i / sum_j w_j over the 10 selected.
        denom = ttnn.sum(top_w, dim=-1, keepdim=True)
        top_w = ttnn.div(top_w, denom)
        ttnn.deallocate(denom)

        # 4) scatter back to the 512-wide sparse routing vector (sparse_matmul input).
        if self.router_fp32:  # W2 expert-combine dtype audit: cast the renormed
            # weights back to bf16 for the scatter (fp32 TILE unsupported there)
            # — rw512 keeps its bf16 contract, the sparse_matmul sparsity input
            # dtype is UNCHANGED (single-variable experiment: router decision
            # precision only; expert-math precision is W1's lever).
            top_w = ttnn.typecast(top_w, ttnn.bfloat16)
        routing_weights_512 = ttnn.scatter(zeros, dim=1, index=top_i, src=top_w)
        ttnn.deallocate(zeros)
        return top_i, routing_weights_512
