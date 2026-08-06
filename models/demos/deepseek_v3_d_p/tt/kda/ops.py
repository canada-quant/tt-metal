# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Python-owned KDA graph orchestration.

The KDA layer keeps collective and state-flow decisions here; device leaves only
perform the bespoke kernels exposed by ``ttnn.transformer.kda_*``.
"""

from __future__ import annotations

import ttnn


def _output_memory_config(memory_config: ttnn.MemoryConfig | None) -> ttnn.MemoryConfig:
    return ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config


def convolution_halo(
    projected_qkv: ttnn.Tensor,
    initial_carry: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    memory_config: ttnn.MemoryConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Exchange causal-convolution carries along the configured SP axis."""
    qkv_shape = tuple(projected_qkv.shape)
    carry_shape = tuple(initial_carry.shape)
    if sequence_parallel_axis not in (0, 1):
        raise ValueError(f"sequence_parallel_axis must be 0 or 1, got {sequence_parallel_axis}")
    if len(qkv_shape) != 3 or len(carry_shape) != 3:
        raise ValueError("KDA convolution halo expects rank-3 tensors")
    if qkv_shape[0] != carry_shape[0] or qkv_shape[2] != carry_shape[2]:
        raise ValueError("KDA convolution halo requires matching batch and channel dimensions")
    history = carry_shape[1]
    if history <= 0 or qkv_shape[1] < history:
        raise ValueError("KDA convolution halo requires 0 < history <= local T")
    if projected_qkv.dtype != initial_carry.dtype or projected_qkv.layout != initial_carry.layout:
        raise ValueError("KDA convolution halo requires matching dtypes and layouts")
    if history > ttnn.TILE_SIZE:
        raise ValueError("KDA convolution history must fit in one tile")

    mesh_device = projected_qkv.device()
    mesh_shape = tuple(mesh_device.shape)
    if len(mesh_shape) != 2 or mesh_shape[sequence_parallel_axis] <= 1:
        raise ValueError("KDA convolution halo requires a 2D mesh with SP > 1")
    sp_size = mesh_shape[sequence_parallel_axis]
    batch, local_sequence, channels = qkv_shape
    out_mem = _output_memory_config(memory_config)

    local_tail = ttnn.slice(
        projected_qkv,
        (0, local_sequence - history, 0),
        (batch, local_sequence, channels),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    padded_tail = ttnn.pad(
        local_tail,
        ((0, 0), (0, ttnn.TILE_SIZE - history), (0, 0)),
        value=0.0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tiled_tail = ttnn.to_layout(padded_tail, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gathered_tails = ttnn.all_gather(
        tiled_tail,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    entry_carries = [initial_carry]
    for rank in range(sp_size - 1):
        tiled_rank_tail = ttnn.slice(
            gathered_tails,
            (0, rank * ttnn.TILE_SIZE, 0),
            (batch, (rank + 1) * ttnn.TILE_SIZE, channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rank_tail = ttnn.to_layout(tiled_rank_tail, ttnn.ROW_MAJOR_LAYOUT)
        entry_carries.append(ttnn.slice(rank_tail, (0, 0, 0), (batch, history, channels), memory_config=out_mem))
    replicated_entries = ttnn.concat(entry_carries, dim=1, memory_config=out_mem)
    partition_carry = ttnn.mesh_partition(
        replicated_entries,
        dim=1,
        cluster_axis=sequence_parallel_axis,
        memory_config=out_mem,
    )

    tiled_final_carry = ttnn.slice(
        gathered_tails,
        (0, (sp_size - 1) * ttnn.TILE_SIZE, 0),
        (batch, sp_size * ttnn.TILE_SIZE, channels),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    final_row_major = ttnn.to_layout(tiled_final_carry, ttnn.ROW_MAJOR_LAYOUT)
    final_carry = ttnn.slice(final_row_major, (0, 0, 0), (batch, history, channels), memory_config=out_mem)
    return partition_carry, final_carry


def chunk_recurrence(*args, **kwargs):
    """Run KDA's chunk recurrence with explicit input and output state tensors."""
    return ttnn.transformer.chunk_kda(*args, **kwargs)


def distributed_affine_prefix(
    transform_a: ttnn.Tensor,
    transform_b: ttnn.Tensor,
    initial_state: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
    affine_summary_dtype: ttnn.DataType = ttnn.float32,
    recurrent_state_dtype: ttnn.DataType = ttnn.float32,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Compose SP partition affine summaries and return entry/final carries."""
    if sequence_parallel_axis not in (0, 1):
        raise ValueError(f"sequence_parallel_axis must be 0 or 1, got {sequence_parallel_axis}")
    shape = tuple(transform_a.shape)
    if tuple(transform_b.shape) != shape or tuple(initial_state.shape) != shape:
        raise ValueError("distributed KDA affine prefix requires equal batched [K,K] tensor shapes")
    if len(shape) not in (3, 4):
        raise ValueError("KDA all-gather affine prefix requires rank-3 production or rank-4 test transforms")
    if shape[-2] != shape[-1]:
        raise ValueError("distributed KDA affine prefix currently requires K == V")
    if affine_summary_dtype not in (ttnn.float32, ttnn.bfloat16):
        raise ValueError("affine_summary_dtype must be FLOAT32 or BFLOAT16")
    if recurrent_state_dtype not in (ttnn.float32, ttnn.bfloat16):
        raise ValueError("recurrent_state_dtype must be FLOAT32 or BFLOAT16")

    mesh_device = transform_a.device()
    mesh_shape = tuple(mesh_device.shape)
    if len(mesh_shape) != 2 or mesh_shape[sequence_parallel_axis] <= 1:
        raise ValueError("distributed KDA affine prefix requires a 2D mesh with SP > 1")
    sp_size = mesh_shape[sequence_parallel_axis]
    has_explicit_sp_dimension = len(shape) == 4
    batch_heads, key_dim = shape[-3], shape[-2]
    value_dim = transform_b.shape[-1]
    out_mem = _output_memory_config(memory_config)

    summary_a = ttnn.typecast(transform_a, affine_summary_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    summary_b = ttnn.typecast(transform_b, affine_summary_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if not has_explicit_sp_dimension:
        summary_a = ttnn.reshape(summary_a, (1, batch_heads, key_dim, key_dim))
        summary_b = ttnn.reshape(summary_b, (1, batch_heads, key_dim, value_dim))
    packed = ttnn.concat([summary_a, summary_b], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gathered = ttnn.all_gather(
        packed,
        dim=0,
        cluster_axis=sequence_parallel_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    carry = ttnn.typecast(initial_state, recurrent_state_dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
    if not has_explicit_sp_dimension:
        carry = ttnn.reshape(carry, (1, batch_heads, key_dim, value_dim))
    entry_states = []
    for rank in range(sp_size):
        entry_states.append(carry)
        rank_a = ttnn.slice(
            gathered,
            (rank, 0, 0, 0),
            (rank + 1, batch_heads, key_dim, key_dim),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        rank_b = ttnn.slice(
            gathered,
            (rank, 0, 0, key_dim),
            (rank + 1, batch_heads, key_dim, key_dim + value_dim),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if rank_a.dtype != recurrent_state_dtype:
            rank_a = ttnn.typecast(rank_a, recurrent_state_dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
        carry = ttnn.matmul(
            rank_a,
            carry,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=recurrent_state_dtype,
            compute_kernel_config=compute_kernel_config,
        )
        if rank_b.dtype != recurrent_state_dtype:
            rank_b = ttnn.typecast(rank_b, recurrent_state_dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
        carry = ttnn.add(carry, rank_b, memory_config=ttnn.L1_MEMORY_CONFIG)

    replicated_entries = ttnn.concat(entry_states, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    entry_state = ttnn.mesh_partition(
        replicated_entries,
        dim=0,
        cluster_axis=sequence_parallel_axis,
        memory_config=out_mem,
    )
    final_state = ttnn.typecast(carry, recurrent_state_dtype, memory_config=out_mem)
    if not has_explicit_sp_dimension:
        entry_state = ttnn.reshape(entry_state, (batch_heads, key_dim, value_dim))
        final_state = ttnn.reshape(final_state, (batch_heads, key_dim, value_dim))
    return entry_state, final_state
