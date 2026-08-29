# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal)
# SPDX-License-Identifier: Apache-2.0
"""Expert + shared-expert weight loading for the Qwen3.8-Flash-Next MoE-512 block.

Adapted from `models/demos/gpt_oss/tt/experts/weights.py::load_expert_weights`
(verbatim in-tree art) with the flash-next deltas from moe-bfp4 spec §1/§3:

  * Fused `experts.gate_up_proj` (512, 1280, 2560) split as CONTIGUOUS HALVES —
    gate = w[:, :640, :], up = w[:, 640:, :] — NOT the gpt_oss interleave
    ([..., ::2] / [..., 1::2]). (HF modeling line 889: `.chunk(2, dim=-1)`.)
  * ZERO biases — no gate/up/down bias tensors at all (module_map.txt grep = 0);
    the gpt_oss bias plumbing is dropped, not zero-filled.
  * BFP4 expert weights (owner-locked); down_proj may flip to BFP8 per spec §7
    (deepseek_v3 art: `ttnn.bfloat8_b if hf_name == "down_proj" else ttnn.bfloat4_b`).
  * TP=4 sharding, EP=1: gate/up col-parallel (shard N 640→160/device on dim 3),
    down row-parallel (shard K 640→160/device on dim 2); expert axis replicated.

HF parameter names (per layer N):
  mlp.gate.weight (512, 2560); mlp.experts.gate_up_proj (512, 1280, 2560);
  mlp.experts.down_proj (512, 2560, 640);
  mlp.shared_expert.{gate,up}_proj.weight (640, 2560);
  mlp.shared_expert.down_proj.weight (2560, 640); mlp.shared_expert_gate.weight (1, 2560).
"""
from dataclasses import dataclass

import torch

import ttnn


@dataclass(frozen=True)
class ExpertWeights:
    """Device-resident routed-expert weights (immutable after creation)."""

    gate_proj: ttnn.Tensor  # (1, 512, 2560, 640) col-sharded (dim 3)
    up_proj: ttnn.Tensor    # (1, 512, 2560, 640) col-sharded (dim 3)
    down_proj: ttnn.Tensor  # (1, 512, 640, 2560) row-sharded (dim 2)
    intermediate_size_per_device: int


@dataclass(frozen=True)
class SharedExpertWeights:
    """Device-resident shared-expert weights (dense SwiGLU + sigmoid scalar gate)."""

    gate_proj: ttnn.Tensor        # (1, 1, 2560, 640) col-sharded
    up_proj: ttnn.Tensor          # (1, 1, 2560, 640) col-sharded
    down_proj: ttnn.Tensor        # (1, 1, 640, 2560) row-sharded
    shared_expert_gate: ttnn.Tensor  # (1, 1, 2560, 1) replicated, bf16


def _cache_name(tensor_cache_path, name):
    if tensor_cache_path is None:
        return None
    import os

    return os.path.join(tensor_cache_path, f"{name}.tensorbin")


def load_expert_weights(mesh_device, config, state_dict, weight_dtype=None, tensor_cache_path=None) -> ExpertWeights:
    """Load + shard the 512 routed experts (BFP4; contiguous-half gate_up split).

    Args:
        mesh_device: ttnn mesh device (1,4) or None (host-side construction).
        config: Qwen4ExpMoEConfig.
        state_dict: mapping with keys "experts.gate_up_proj" (512,1280,2560) and
            "experts.down_proj" (512,2560,640), or None for host-side stub.
        weight_dtype: override for gate/up dtype (default config.weight_dtype).
        tensor_cache_path: optional converted-tensor cache dir.
    """
    E = config.num_experts
    I = config.intermediate_size
    H = config.hidden_size
    n_devices = mesh_device.get_num_devices() if mesh_device is not None else 1
    assert I % n_devices == 0, f"intermediate {I} must divide across {n_devices} devices"
    intermediate_per_device = I // n_devices

    gu_dtype = weight_dtype if weight_dtype is not None else config.weight_dtype
    dn_dtype = config.down_dtype

    if state_dict:
        gate_up = state_dict["experts.gate_up_proj"]
        down = state_dict["experts.down_proj"]
        assert tuple(gate_up.shape) == (E, 2 * I, H), gate_up.shape
        assert tuple(down.shape) == (E, H, I), down.shape
        # CONTIGUOUS-HALF split (spec §1 — NOT gpt_oss [::2]/[1::2] interleave).
        gate = gate_up[:, :I, :]   # (E, 640, 2560) = (E, N_out, K_in)
        up = gate_up[:, I:, :]     # (E, 640, 2560)
        # ttnn.sparse_matmul input_b is (1, E, K, N): transpose HF (E,N,K) -> (1,E,K,N).
        gate_t = gate.transpose(-1, -2).contiguous().unsqueeze(0)   # (1, E, 2560, 640)
        up_t = up.transpose(-1, -2).contiguous().unsqueeze(0)       # (1, E, 2560, 640)
        down_t = down.transpose(-1, -2).contiguous().unsqueeze(0)   # (1, E, 640, 2560)
    else:
        gate_t = up_t = down_t = None

    col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3) if mesh_device is not None else None
    row_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2) if mesh_device is not None else None

    gate_tt = ttnn.as_tensor(
        gate_t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=gu_dtype,
        mesh_mapper=col_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=_cache_name(tensor_cache_path, "experts_gate_proj"),
    )
    up_tt = ttnn.as_tensor(
        up_t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=gu_dtype,
        mesh_mapper=col_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=_cache_name(tensor_cache_path, "experts_up_proj"),
    )
    down_tt = ttnn.as_tensor(
        down_t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dn_dtype,
        mesh_mapper=row_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=_cache_name(tensor_cache_path, "experts_down_proj"),
    )

    return ExpertWeights(
        gate_proj=gate_tt,
        up_proj=up_tt,
        down_proj=down_tt,
        intermediate_size_per_device=intermediate_per_device,
    )


def load_shared_expert_weights(mesh_device, config, state_dict, tensor_cache_path=None) -> SharedExpertWeights:
    """Load + shard the dense shared expert (BFP8) + sigmoid scalar gate (bf16).

    Args:
        state_dict: mapping with keys
            "shared_expert.gate_proj.weight" (640, 2560),
            "shared_expert.up_proj.weight"   (640, 2560),
            "shared_expert.down_proj.weight" (2560, 640),
            "shared_expert_gate.weight"      (1, 2560).
    """
    I = config.shared_intermediate_size
    H = config.hidden_size
    if state_dict:
        g = state_dict["shared_expert.gate_proj.weight"]
        u = state_dict["shared_expert.up_proj.weight"]
        d = state_dict["shared_expert.down_proj.weight"]
        sg = state_dict["shared_expert_gate.weight"]
        assert tuple(g.shape) == (I, H), g.shape
        assert tuple(u.shape) == (I, H), u.shape
        assert tuple(d.shape) == (H, I), d.shape
        assert tuple(sg.shape) == (1, H), sg.shape
        g_t = g.transpose(0, 1).contiguous().view(1, 1, H, I)   # (1,1,2560,640)
        u_t = u.transpose(0, 1).contiguous().view(1, 1, H, I)   # (1,1,2560,640)
        d_t = d.transpose(0, 1).contiguous().view(1, 1, I, H)   # (1,1,640,2560)
        sg_t = sg.transpose(0, 1).contiguous().view(1, 1, H, 1)  # (1,1,2560,1)
    else:
        g_t = u_t = d_t = sg_t = None

    col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3) if mesh_device is not None else None
    row_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2) if mesh_device is not None else None
    rep_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device is not None else None

    dense_dtype = ttnn.bfloat8_b  # shared expert is BFP8 per spec §6 budget
    gate_tt = ttnn.as_tensor(
        g_t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dense_dtype,
        mesh_mapper=col_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=_cache_name(tensor_cache_path, "shared_gate_proj"),
    )
    up_tt = ttnn.as_tensor(
        u_t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dense_dtype,
        mesh_mapper=col_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=_cache_name(tensor_cache_path, "shared_up_proj"),
    )
    down_tt = ttnn.as_tensor(
        d_t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dense_dtype,
        mesh_mapper=row_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=_cache_name(tensor_cache_path, "shared_down_proj"),
    )
    sg_tt = ttnn.as_tensor(
        sg_t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        mesh_mapper=rep_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=_cache_name(tensor_cache_path, "shared_expert_gate"),
    )

    return SharedExpertWeights(
        gate_proj=gate_tt, up_proj=up_tt, down_proj=down_tt, shared_expert_gate=sg_tt
    )


def remap_flash_next_moe_state_dict(layer_sd):
    """Map an HF per-layer mlp.* sub-state_dict to this package's loader keys.

    Args:
        layer_sd: dict of the layer-N `mlp.` tensors with HF names, e.g.
            "gate.weight", "experts.gate_up_proj", "experts.down_proj",
            "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight",
            "shared_expert.down_proj.weight", "shared_expert_gate.weight".

    Returns:
        dict with exactly the keys load_expert_weights / load_shared_expert_weights
        / Qwen4ExpRouter expect. Any bias key present raises (spec: zero biases).
    """
    out = {}
    for k, v in layer_sd.items():
        if "bias" in k:
            raise ValueError(f"flash-next MoE has zero biases (spec §1); unexpected key {k!r}")
        if k == "gate.weight":
            out["gate.weight"] = v
        elif k in ("experts.gate_up_proj", "experts.down_proj"):
            out[k] = v
        elif k.startswith("shared_expert.") or k == "shared_expert_gate.weight":
            out[k] = v
        else:
            raise KeyError(f"unmapped MoE weight key {k!r}")
    return out
