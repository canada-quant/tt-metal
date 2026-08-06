# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Python-owned KDA graph orchestration.

The KDA layer keeps collective and state-flow decisions here; device leaves only
perform the bespoke kernels exposed by ``ttnn.transformer.kda_*``.
"""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.const_tiles import build_kda_const_tiles


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


def _head_split(tensor: ttnn.Tensor, batch: int, sequence: int, heads: int, width: int, dtype) -> ttnn.Tensor:
    if tensor.dtype != dtype:
        tensor = ttnn.typecast(tensor, dtype)
    tensor = ttnn.permute(tensor, (0, 2, 1, 3))
    return ttnn.reshape(tensor, (batch * heads, sequence, width))


def _head_vector_split(tensor: ttnn.Tensor, batch: int, sequence: int, heads: int) -> ttnn.Tensor:
    if tensor.dtype != ttnn.float32:
        tensor = ttnn.typecast(tensor, ttnn.float32)
    tensor = ttnn.permute(tensor, (0, 2, 1))
    return ttnn.reshape(tensor, (batch * heads, sequence))


def _pad_time(tensor: ttnn.Tensor, batch_heads: int, width: int, pad: int) -> ttnn.Tensor:
    if pad == 0:
        return tensor
    zeros = ttnn.zeros(
        (batch_heads, pad, width),
        dtype=tensor.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=tensor.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.concat([tensor, zeros], dim=1)


def _group_summary_memory_config(device, group_heads: int, key_dim: int) -> ttnn.MemoryConfig:
    worker_cores = ttnn.num_cores_to_corerangeset(
        group_heads,
        device.compute_with_storage_grid_size(),
        row_wise=True,
    )
    return ttnn.create_sharded_memory_config(
        (group_heads, key_dim, key_dim),
        core_grid=worker_cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def chunk_recurrence(
    q_input: ttnn.Tensor,
    k_input: ttnn.Tensor,
    v_input: ttnn.Tensor,
    gate_input: ttnn.Tensor,
    beta_input: ttnn.Tensor,
    *,
    scale: float | None = None,
    initial_state: ttnn.Tensor | None = None,
    output_final_state: bool = False,
    output_head_major: bool = False,
    chunk_size: int = 32,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
    eye: ttnn.Tensor | None = None,
    tril: ttnn.Tensor | None = None,
    ones: ttnn.Tensor | None = None,
    masks: ttnn.Tensor | None = None,
    summary_group_chunks: int = 8,
    sequence_parallel_axis: int | None = None,
    affine_summary_dtype: ttnn.DataType = ttnn.float32,
    recurrent_state_dtype: ttnn.DataType = ttnn.float32,
    affine_prefix_compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
    grouped_scan_output_dtype: ttnn.DataType = ttnn.float32,
    grouped_scan_compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
    use_bf16_prep_intermediates: bool = False,
) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
    """Run KDA's Python-owned chunk recurrence with explicit logical state."""
    q_shape = tuple(q_input.shape)
    k_shape = tuple(k_input.shape)
    v_shape = tuple(v_input.shape)
    gate_shape = tuple(gate_input.shape)
    beta_shape = tuple(beta_input.shape)
    if len(beta_shape) != 3:
        raise ValueError("KDA recurrence beta must be [B,T,H]")
    batch, sequence, heads = beta_shape
    flat_qk = len(q_shape) == 3
    flat_v = len(v_shape) == 3
    flat_gate = len(gate_shape) == 3
    if len(q_shape) not in (3, 4) or len(v_shape) not in (3, 4) or len(gate_shape) not in (3, 4):
        raise ValueError("KDA recurrence q/k/v/g must be rank 3 or 4")
    if k_shape != q_shape or q_shape[:2] != (batch, sequence) or v_shape[:2] != (batch, sequence):
        raise ValueError("KDA recurrence q/k/v shapes are inconsistent")
    key_dim = gate_shape[2] // heads if flat_gate else gate_shape[3]
    value_dim = v_shape[2] // heads if flat_v else v_shape[3]
    if flat_qk and q_shape[2] != heads * key_dim:
        raise ValueError("flat q/k width must equal H*K")
    if not flat_qk and q_shape[2:] != (heads, key_dim):
        raise ValueError("q/k must be [B,T,H,K]")
    if flat_v and v_shape[2] != heads * value_dim:
        raise ValueError("flat v width must equal H*V")
    if not flat_v and v_shape[2:] != (heads, value_dim):
        raise ValueError("v must be [B,T,H,V]")
    if flat_gate and gate_shape[2] != heads * key_dim:
        raise ValueError("flat gate width must equal H*K")
    if not flat_gate and gate_shape[2:] != (heads, key_dim):
        raise ValueError("gate must be [B,T,H,K]")
    if chunk_size != ttnn.TILE_SIZE:
        raise ValueError(f"KDA recurrence currently requires chunk_size=32, got {chunk_size}")
    if summary_group_chunks <= 0:
        raise ValueError(f"summary_group_chunks must be positive, got {summary_group_chunks}")
    if sequence_parallel_axis not in (None, 0, 1):
        raise ValueError("sequence_parallel_axis must be 0 or 1")
    if affine_summary_dtype not in (ttnn.float32, ttnn.bfloat16):
        raise ValueError("affine_summary_dtype must be FLOAT32 or BFLOAT16")
    if recurrent_state_dtype not in (ttnn.float32, ttnn.bfloat16):
        raise ValueError("recurrent_state_dtype must be FLOAT32 or BFLOAT16")
    if grouped_scan_output_dtype not in (ttnn.float32, ttnn.bfloat16):
        raise ValueError("grouped_scan_output_dtype must be FLOAT32 or BFLOAT16")

    device = q_input.device()
    batch_heads = batch * heads
    pad = (-sequence) % chunk_size
    padded_sequence = sequence + pad
    num_chunks = padded_sequence // chunk_size
    qk_scale = key_dim**-0.5 if scale is None else scale
    distributed_prefix = sequence_parallel_axis is not None
    if (flat_qk or flat_v or flat_gate) and pad:
        raise ValueError("flat KDA recurrence inputs require T divisible by chunk_size")

    if flat_qk:
        q = q_input if q_input.dtype == ttnn.bfloat16 else ttnn.typecast(q_input, ttnn.bfloat16)
        k = k_input if k_input.dtype == ttnn.bfloat16 else ttnn.typecast(k_input, ttnn.bfloat16)
    else:
        q = _head_split(q_input, batch, sequence, heads, key_dim, ttnn.bfloat16)
        q = ttnn.multiply(q, qk_scale)
        k = _head_split(k_input, batch, sequence, heads, key_dim, ttnn.bfloat16)
    v = (
        v_input
        if flat_v and v_input.dtype == ttnn.bfloat16
        else (
            ttnn.typecast(v_input, ttnn.bfloat16)
            if flat_v
            else _head_split(v_input, batch, sequence, heads, value_dim, ttnn.bfloat16)
        )
    )
    gate = gate_input if flat_gate else _head_split(gate_input, batch, sequence, heads, key_dim, ttnn.float32)
    beta = _head_vector_split(beta_input, batch, sequence, heads)
    if not flat_qk:
        q = _pad_time(q, batch_heads, key_dim, pad)
        k = _pad_time(k, batch_heads, key_dim, pad)
    if not flat_v:
        v = _pad_time(v, batch_heads, value_dim, pad)
    if not flat_gate:
        gate = _pad_time(gate, batch_heads, key_dim, pad)
    if pad:
        beta_zeros = ttnn.zeros(
            (batch_heads, pad),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        beta = ttnn.concat([beta, beta_zeros], dim=1)
    if not flat_qk:
        q = ttnn.reshape(q, (batch_heads, num_chunks, chunk_size, key_dim))
        k = ttnn.reshape(k, (batch_heads, num_chunks, chunk_size, key_dim))
    if not flat_v:
        v = ttnn.reshape(v, (batch_heads, num_chunks, chunk_size, value_dim))
    if not flat_gate:
        gate = ttnn.reshape(gate, (batch_heads, num_chunks, chunk_size, key_dim))
    beta = ttnn.reshape(beta, (batch_heads, num_chunks, chunk_size, 1))

    const_tiles = (eye, tril, ones, masks)
    if any(tile is None for tile in const_tiles):
        if not all(tile is None for tile in const_tiles):
            raise ValueError("eye, tril, ones, and masks must be supplied together")
        const_tiles = build_kda_const_tiles(device)
    eye_tile, tril_tile, ones_tile, mask_tiles = const_tiles

    if initial_state is None:
        state = ttnn.zeros(
            (batch_heads, key_dim, value_dim),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        state = initial_state if initial_state.dtype == ttnn.float32 else ttnn.typecast(initial_state, ttnn.float32)
        state = ttnn.reshape(state, (batch_heads, key_dim, value_dim))

    out_mem = _output_memory_config(memory_config)
    prep_bf16_mask = (1 << 1) | (1 << 2) | (1 << 5) if use_bf16_prep_intermediates else 0
    prep = ttnn.transformer.kda_chunk_preparation(
        q,
        k,
        v,
        gate,
        beta,
        eye_tile,
        tril_tile,
        ones_tile,
        mask_tiles,
        chunk_size=chunk_size,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if distributed_prefix else None,
        compute_kernel_config=compute_kernel_config,
        v_flat=flat_v,
        value_heads=heads,
        normalize_qk=flat_qk,
        scale=qk_scale,
        qk_flat=flat_qk,
        key_heads=heads,
        gate_flat=flat_gate,
        output_bf16_mask=prep_bf16_mask,
    )

    grouped_scan = None
    distributed_final_state = None
    use_group_prefix = num_chunks >= 160 and num_chunks % summary_group_chunks == 0
    if distributed_prefix or use_group_prefix:
        if num_chunks % summary_group_chunks:
            raise ValueError(
                f"local chunk count {num_chunks} must be divisible by summary_group_chunks {summary_group_chunks}"
            )
        if key_dim != value_dim:
            raise ValueError("grouped KDA affine prefix currently requires K == V")
        groups_per_head = num_chunks // summary_group_chunks
        group_heads = batch_heads * groups_per_head
        grid = device.compute_with_storage_grid_size()
        if group_heads > min(grid.x * grid.y, 128):
            raise ValueError(f"grouped KDA needs {group_heads} summary owners, but only 128 are supported")
        grouped = list(prep)
        grouped[0] = ttnn.reshape(grouped[0], (group_heads, summary_group_chunks, chunk_size, value_dim))
        grouped[1] = ttnn.reshape(grouped[1], (group_heads, summary_group_chunks, chunk_size, key_dim))
        grouped[2] = ttnn.reshape(grouped[2], (group_heads, summary_group_chunks, chunk_size, key_dim))
        grouped[3] = ttnn.reshape(grouped[3], (group_heads, summary_group_chunks, chunk_size, chunk_size))
        grouped[4] = ttnn.reshape(grouped[4], (group_heads, summary_group_chunks, key_dim, chunk_size))
        grouped[5] = ttnn.reshape(grouped[5], (group_heads, summary_group_chunks, key_dim, 1))
        grouped[6] = ttnn.reshape(grouped[6], (group_heads, summary_group_chunks, chunk_size, chunk_size))
        summary_mem = _group_summary_memory_config(device, group_heads, key_dim)
        summaries = ttnn.transformer.kda_final_chunk_scan(
            *grouped,
            chunk_size=chunk_size,
            memory_config=summary_mem,
            compute_kernel_config=compute_kernel_config,
            state_only=True,
            identity_tile=eye_tile,
            summary_pair=True,
        )
        summaries[0] = ttnn.subtract(summaries[0], summaries[1], memory_config=summary_mem)
        summary_a = (
            summaries[0]
            if summaries[0].dtype == affine_summary_dtype
            else ttnn.typecast(summaries[0], affine_summary_dtype, memory_config=summary_mem)
        )
        summary_b = (
            summaries[1]
            if summaries[1].dtype == affine_summary_dtype
            else ttnn.typecast(summaries[1], affine_summary_dtype, memory_config=summary_mem)
        )
        prefix_mem = out_mem if distributed_prefix else ttnn.L1_MEMORY_CONFIG
        if distributed_prefix:
            partition_a, partition_b = ttnn.transformer.kda_affine_compose(
                summary_a,
                summary_b,
                groups_per_head,
                memory_config=prefix_mem,
                compute_kernel_config=affine_prefix_compute_kernel_config,
            )
            partition_entry_state, distributed_final_state = distributed_affine_prefix(
                partition_a,
                partition_b,
                state,
                sequence_parallel_axis=sequence_parallel_axis,
                memory_config=out_mem,
                compute_kernel_config=affine_prefix_compute_kernel_config,
                affine_summary_dtype=affine_summary_dtype,
                recurrent_state_dtype=recurrent_state_dtype,
            )
            if partition_entry_state.dtype != ttnn.float32:
                partition_entry_state = ttnn.typecast(partition_entry_state, ttnn.float32, memory_config=prefix_mem)
            group_initial_states = ttnn.transformer.kda_affine_prefix(
                summary_a,
                summary_b,
                partition_entry_state,
                groups_per_head,
                memory_config=prefix_mem,
                compute_kernel_config=affine_prefix_compute_kernel_config,
            )
        else:
            group_initial_states = ttnn.transformer.kda_affine_prefix(
                summary_a,
                summary_b,
                state,
                groups_per_head,
                memory_config=prefix_mem,
                compute_kernel_config=affine_prefix_compute_kernel_config,
            )
        grouped_scan = ttnn.transformer.kda_final_chunk_scan(
            *grouped,
            initial_state=group_initial_states,
            chunk_size=chunk_size,
            memory_config=out_mem,
            compute_kernel_config=grouped_scan_compute_kernel_config,
            output_bf16=grouped_scan_output_dtype == ttnn.bfloat16,
        )
        grouped_scan[0] = ttnn.reshape(grouped_scan[0], (batch_heads, num_chunks, chunk_size, value_dim))
        if distributed_final_state is not None:
            grouped_scan[1] = distributed_final_state
        else:
            all_final_states = ttnn.reshape(grouped_scan[1], (batch_heads, groups_per_head, key_dim, value_dim))
            last_final_state = ttnn.slice(
                all_final_states,
                (0, groups_per_head - 1, 0, 0),
                (batch_heads, groups_per_head, key_dim, value_dim),
                memory_config=out_mem,
            )
            grouped_scan[1] = ttnn.reshape(last_final_state, (batch_heads, key_dim, value_dim))

    scan = (
        grouped_scan
        if grouped_scan is not None
        else ttnn.transformer.kda_final_chunk_scan(
            *prep,
            initial_state=state,
            chunk_size=chunk_size,
            memory_config=out_mem,
            compute_kernel_config=compute_kernel_config,
        )
    )
    final_state = ttnn.reshape(scan[1], (batch, heads, key_dim, value_dim)) if output_final_state else None
    if output_head_major:
        if pad == 0:
            return ttnn.reshape(scan[0], (batch_heads, sequence, value_dim)), final_state
        output = ttnn.to_layout(scan[0], ttnn.ROW_MAJOR_LAYOUT)
        output = ttnn.reshape(output, (batch_heads, padded_sequence, value_dim))
        output = ttnn.slice(output, (0, 0, 0), (batch_heads, sequence, value_dim))
        return ttnn.to_layout(output, ttnn.TILE_LAYOUT), final_state

    output = ttnn.to_layout(scan[0], ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.reshape(output, (batch_heads, padded_sequence, value_dim))
    if pad:
        output = ttnn.slice(output, (0, 0, 0), (batch_heads, sequence, value_dim))
    output = ttnn.reshape(output, (batch, heads, sequence, value_dim))
    output = ttnn.permute(output, (0, 2, 1, 3))
    return output, final_state


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
