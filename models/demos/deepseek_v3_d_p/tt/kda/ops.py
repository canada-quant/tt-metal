# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Python-owned KDA graph orchestration.

The KDA layer keeps collective and state-flow decisions here; device leaves only
perform the bespoke kernels exposed by ``ttnn.transformer.kda_*``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

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


_MIN_CHUNKS_FOR_GROUPED_SCAN = 160


@dataclass(frozen=True)
class _RecurrenceGeometry:
    batch: int
    sequence: int
    heads: int
    key_dim: int
    value_dim: int
    chunk_size: int
    pad: int
    num_chunks: int
    qk_are_flat: bool
    v_is_flat: bool
    gate_is_flat: bool

    @property
    def batch_heads(self) -> int:
        return self.batch * self.heads

    @property
    def padded_sequence(self) -> int:
        return self.sequence + self.pad


@dataclass(frozen=True)
class _ChunkInputs:
    q: ttnn.Tensor
    k: ttnn.Tensor
    v: ttnn.Tensor
    gate: ttnn.Tensor
    beta: ttnn.Tensor


@dataclass(frozen=True)
class _PreparedChunks:
    v_beta: ttnn.Tensor
    kd: ttnn.Tensor
    q_decay: ttnn.Tensor
    intra: ttnn.Tensor
    k_dec_t: ttnn.Tensor
    final_decay: ttnn.Tensor
    t_inv: ttnn.Tensor

    @classmethod
    def from_kernel_outputs(cls, outputs: list[ttnn.Tensor]) -> _PreparedChunks:
        if len(outputs) != 7:
            raise RuntimeError(f"KDA chunk preparation returned {len(outputs)} tensors, expected 7")
        return cls(*outputs)

    def as_kernel_args(self) -> tuple[ttnn.Tensor, ...]:
        return (
            self.v_beta,
            self.kd,
            self.q_decay,
            self.intra,
            self.k_dec_t,
            self.final_decay,
            self.t_inv,
        )


@dataclass(frozen=True)
class _ScanResult:
    output: ttnn.Tensor
    final_state: ttnn.Tensor


@dataclass(frozen=True)
class _ScanConfig:
    summary_group_chunks: int
    output_memory_config: ttnn.MemoryConfig
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None
    affine_summary_dtype: ttnn.DataType
    recurrent_state_dtype: ttnn.DataType
    affine_prefix_compute_kernel_config: ttnn.DeviceComputeKernelConfig | None
    grouped_scan_output_dtype: ttnn.DataType
    grouped_scan_compute_kernel_config: ttnn.DeviceComputeKernelConfig | None


def _validate_recurrence_geometry(
    q_input: ttnn.Tensor,
    k_input: ttnn.Tensor,
    v_input: ttnn.Tensor,
    gate_input: ttnn.Tensor,
    beta_input: ttnn.Tensor,
    *,
    chunk_size: int,
    summary_group_chunks: int,
    sequence_parallel_axis: int | None,
    affine_summary_dtype: ttnn.DataType,
    recurrent_state_dtype: ttnn.DataType,
    grouped_scan_output_dtype: ttnn.DataType,
) -> _RecurrenceGeometry:
    """Validate the existing KDA input contract and derive host-only execution metadata."""
    q_shape = tuple(q_input.shape)
    k_shape = tuple(k_input.shape)
    v_shape = tuple(v_input.shape)
    gate_shape = tuple(gate_input.shape)
    beta_shape = tuple(beta_input.shape)
    if len(beta_shape) != 3:
        raise ValueError("KDA recurrence beta must be [B,T,H]")
    batch, sequence, heads = beta_shape
    qk_are_flat = len(q_shape) == 3
    v_is_flat = len(v_shape) == 3
    gate_is_flat = len(gate_shape) == 3
    if len(q_shape) not in (3, 4) or len(v_shape) not in (3, 4) or len(gate_shape) not in (3, 4):
        raise ValueError("KDA recurrence q/k/v/g must be rank 3 or 4")
    if k_shape != q_shape or q_shape[:2] != (batch, sequence) or v_shape[:2] != (batch, sequence):
        raise ValueError("KDA recurrence q/k/v shapes are inconsistent")
    key_dim = gate_shape[2] // heads if gate_is_flat else gate_shape[3]
    value_dim = v_shape[2] // heads if v_is_flat else v_shape[3]
    if qk_are_flat and q_shape[2] != heads * key_dim:
        raise ValueError("flat q/k width must equal H*K")
    if not qk_are_flat and q_shape[2:] != (heads, key_dim):
        raise ValueError("q/k must be [B,T,H,K]")
    if v_is_flat and v_shape[2] != heads * value_dim:
        raise ValueError("flat v width must equal H*V")
    if not v_is_flat and v_shape[2:] != (heads, value_dim):
        raise ValueError("v must be [B,T,H,V]")
    if gate_is_flat and gate_shape[2] != heads * key_dim:
        raise ValueError("flat gate width must equal H*K")
    if not gate_is_flat and gate_shape[2:] != (heads, key_dim):
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

    pad = (-sequence) % chunk_size
    num_chunks = (sequence + pad) // chunk_size
    if (qk_are_flat or v_is_flat or gate_is_flat) and pad:
        raise ValueError("flat KDA recurrence inputs require T divisible by chunk_size")

    return _RecurrenceGeometry(
        batch=batch,
        sequence=sequence,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=chunk_size,
        pad=pad,
        num_chunks=num_chunks,
        qk_are_flat=qk_are_flat,
        v_is_flat=v_is_flat,
        gate_is_flat=gate_is_flat,
    )


def _prepare_chunk_inputs(
    q_input: ttnn.Tensor,
    k_input: ttnn.Tensor,
    v_input: ttnn.Tensor,
    gate_input: ttnn.Tensor,
    beta_input: ttnn.Tensor,
    geometry: _RecurrenceGeometry,
    *,
    qk_scale: float,
) -> _ChunkInputs:
    if geometry.qk_are_flat:
        q = q_input if q_input.dtype == ttnn.bfloat16 else ttnn.typecast(q_input, ttnn.bfloat16)
        k = k_input if k_input.dtype == ttnn.bfloat16 else ttnn.typecast(k_input, ttnn.bfloat16)
    else:
        q = _head_split(q_input, geometry.batch, geometry.sequence, geometry.heads, geometry.key_dim, ttnn.bfloat16)
        q = ttnn.multiply(q, qk_scale)
        k = _head_split(k_input, geometry.batch, geometry.sequence, geometry.heads, geometry.key_dim, ttnn.bfloat16)
    v = (
        v_input
        if geometry.v_is_flat and v_input.dtype == ttnn.bfloat16
        else (
            ttnn.typecast(v_input, ttnn.bfloat16)
            if geometry.v_is_flat
            else _head_split(
                v_input, geometry.batch, geometry.sequence, geometry.heads, geometry.value_dim, ttnn.bfloat16
            )
        )
    )
    gate = (
        gate_input
        if geometry.gate_is_flat
        else _head_split(gate_input, geometry.batch, geometry.sequence, geometry.heads, geometry.key_dim, ttnn.float32)
    )
    beta = _head_vector_split(beta_input, geometry.batch, geometry.sequence, geometry.heads)
    if not geometry.qk_are_flat:
        q = _pad_time(q, geometry.batch_heads, geometry.key_dim, geometry.pad)
        k = _pad_time(k, geometry.batch_heads, geometry.key_dim, geometry.pad)
    if not geometry.v_is_flat:
        v = _pad_time(v, geometry.batch_heads, geometry.value_dim, geometry.pad)
    if not geometry.gate_is_flat:
        gate = _pad_time(gate, geometry.batch_heads, geometry.key_dim, geometry.pad)
    if geometry.pad:
        beta_zeros = ttnn.zeros(
            (geometry.batch_heads, geometry.pad),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=q_input.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        beta = ttnn.concat([beta, beta_zeros], dim=1)
    if not geometry.qk_are_flat:
        q = ttnn.reshape(q, (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, geometry.key_dim))
        k = ttnn.reshape(k, (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, geometry.key_dim))
    if not geometry.v_is_flat:
        v = ttnn.reshape(v, (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, geometry.value_dim))
    if not geometry.gate_is_flat:
        gate = ttnn.reshape(gate, (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, geometry.key_dim))
    beta = ttnn.reshape(beta, (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, 1))
    return _ChunkInputs(q=q, k=k, v=v, gate=gate, beta=beta)


def _resolve_chunk_constants(
    device,
    *,
    eye: ttnn.Tensor | None,
    tril: ttnn.Tensor | None,
    ones: ttnn.Tensor | None,
    masks: ttnn.Tensor | None,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    const_tiles = (eye, tril, ones, masks)
    if any(tile is None for tile in const_tiles):
        if not all(tile is None for tile in const_tiles):
            raise ValueError("eye, tril, ones, and masks must be supplied together")
        return build_kda_const_tiles(device)
    return const_tiles


def _prepare_initial_state(
    initial_state: ttnn.Tensor | None,
    geometry: _RecurrenceGeometry,
    *,
    device,
) -> ttnn.Tensor:
    if initial_state is None:
        return ttnn.zeros(
            (geometry.batch_heads, geometry.key_dim, geometry.value_dim),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    state = initial_state if initial_state.dtype == ttnn.float32 else ttnn.typecast(initial_state, ttnn.float32)
    return ttnn.reshape(state, (geometry.batch_heads, geometry.key_dim, geometry.value_dim))


def _prepare_chunk_terms(
    inputs: _ChunkInputs,
    constants: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
    geometry: _RecurrenceGeometry,
    *,
    qk_scale: float,
    memory_config: ttnn.MemoryConfig | None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None,
    use_bf16_prep_intermediates: bool,
) -> _PreparedChunks:
    eye_tile, tril_tile, ones_tile, mask_tiles = constants
    prep_bf16_mask = (1 << 1) | (1 << 2) | (1 << 5) if use_bf16_prep_intermediates else 0
    outputs = ttnn.transformer.kda_chunk_preparation(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.gate,
        inputs.beta,
        eye_tile,
        tril_tile,
        ones_tile,
        mask_tiles,
        chunk_size=geometry.chunk_size,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
        v_flat=geometry.v_is_flat,
        value_heads=geometry.heads,
        normalize_qk=geometry.qk_are_flat,
        scale=qk_scale,
        qk_flat=geometry.qk_are_flat,
        key_heads=geometry.heads,
        gate_flat=geometry.gate_is_flat,
        output_bf16_mask=prep_bf16_mask,
    )
    return _PreparedChunks.from_kernel_outputs(outputs)


def _reshape_chunks_for_groups(
    prepared: _PreparedChunks,
    geometry: _RecurrenceGeometry,
    *,
    group_heads: int,
    summary_group_chunks: int,
) -> _PreparedChunks:
    return _PreparedChunks(
        v_beta=ttnn.reshape(
            prepared.v_beta, (group_heads, summary_group_chunks, geometry.chunk_size, geometry.value_dim)
        ),
        kd=ttnn.reshape(prepared.kd, (group_heads, summary_group_chunks, geometry.chunk_size, geometry.key_dim)),
        q_decay=ttnn.reshape(
            prepared.q_decay, (group_heads, summary_group_chunks, geometry.chunk_size, geometry.key_dim)
        ),
        intra=ttnn.reshape(
            prepared.intra, (group_heads, summary_group_chunks, geometry.chunk_size, geometry.chunk_size)
        ),
        k_dec_t=ttnn.reshape(
            prepared.k_dec_t, (group_heads, summary_group_chunks, geometry.key_dim, geometry.chunk_size)
        ),
        final_decay=ttnn.reshape(prepared.final_decay, (group_heads, summary_group_chunks, geometry.key_dim, 1)),
        t_inv=ttnn.reshape(
            prepared.t_inv, (group_heads, summary_group_chunks, geometry.chunk_size, geometry.chunk_size)
        ),
    )


def _summarize_chunk_groups(
    grouped: _PreparedChunks,
    geometry: _RecurrenceGeometry,
    *,
    identity_tile: ttnn.Tensor,
    summary_memory_config: ttnn.MemoryConfig,
    affine_summary_dtype: ttnn.DataType,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    raw_first, raw_second = ttnn.transformer.kda_final_chunk_scan(
        *grouped.as_kernel_args(),
        chunk_size=geometry.chunk_size,
        memory_config=summary_memory_config,
        compute_kernel_config=compute_kernel_config,
        state_only=True,
        identity_tile=identity_tile,
        summary_pair=True,
    )
    affine_a = ttnn.subtract(raw_first, raw_second, memory_config=summary_memory_config)
    summary_a = (
        affine_a
        if affine_a.dtype == affine_summary_dtype
        else ttnn.typecast(affine_a, affine_summary_dtype, memory_config=summary_memory_config)
    )
    summary_b = (
        raw_second
        if raw_second.dtype == affine_summary_dtype
        else ttnn.typecast(raw_second, affine_summary_dtype, memory_config=summary_memory_config)
    )
    return summary_a, summary_b


class _RecurrenceScan(ABC):
    @property
    def preparation_memory_config(self) -> ttnn.MemoryConfig | None:
        return None

    @abstractmethod
    def run(
        self,
        prepared: _PreparedChunks,
        initial_state: ttnn.Tensor,
        identity_tile: ttnn.Tensor,
        geometry: _RecurrenceGeometry,
        config: _ScanConfig,
    ) -> _ScanResult:
        """Run the selected recurrence scan without changing the device operation order."""


class _DirectScan(_RecurrenceScan):
    def run(
        self,
        prepared: _PreparedChunks,
        initial_state: ttnn.Tensor,
        identity_tile: ttnn.Tensor,
        geometry: _RecurrenceGeometry,
        config: _ScanConfig,
    ) -> _ScanResult:
        del identity_tile
        output, final_state = ttnn.transformer.kda_final_chunk_scan(
            *prepared.as_kernel_args(),
            initial_state=initial_state,
            chunk_size=geometry.chunk_size,
            memory_config=config.output_memory_config,
            compute_kernel_config=config.compute_kernel_config,
        )
        return _ScanResult(output=output, final_state=final_state)


class _GroupedScan(_RecurrenceScan):
    def run(
        self,
        prepared: _PreparedChunks,
        initial_state: ttnn.Tensor,
        identity_tile: ttnn.Tensor,
        geometry: _RecurrenceGeometry,
        config: _ScanConfig,
    ) -> _ScanResult:
        if geometry.num_chunks % config.summary_group_chunks:
            raise ValueError(
                f"local chunk count {geometry.num_chunks} must be divisible by "
                f"summary_group_chunks {config.summary_group_chunks}"
            )
        if geometry.key_dim != geometry.value_dim:
            raise ValueError("grouped KDA affine prefix currently requires K == V")
        groups_per_head = geometry.num_chunks // config.summary_group_chunks
        group_heads = geometry.batch_heads * groups_per_head
        grid = prepared.v_beta.device().compute_with_storage_grid_size()
        if group_heads > min(grid.x * grid.y, 128):
            raise ValueError(f"grouped KDA needs {group_heads} summary owners, but only 128 are supported")

        grouped = _reshape_chunks_for_groups(
            prepared,
            geometry,
            group_heads=group_heads,
            summary_group_chunks=config.summary_group_chunks,
        )
        summary_memory_config = _group_summary_memory_config(prepared.v_beta.device(), group_heads, geometry.key_dim)
        summary_a, summary_b = _summarize_chunk_groups(
            grouped,
            geometry,
            identity_tile=identity_tile,
            summary_memory_config=summary_memory_config,
            affine_summary_dtype=config.affine_summary_dtype,
            compute_kernel_config=config.compute_kernel_config,
        )
        group_initial_states, strategy_final_state = self._compute_group_entry_states(
            summary_a,
            summary_b,
            initial_state,
            groups_per_head,
            config,
        )
        grouped_output, grouped_final_states = ttnn.transformer.kda_final_chunk_scan(
            *grouped.as_kernel_args(),
            initial_state=group_initial_states,
            chunk_size=geometry.chunk_size,
            memory_config=config.output_memory_config,
            compute_kernel_config=config.grouped_scan_compute_kernel_config,
            output_bf16=config.grouped_scan_output_dtype == ttnn.bfloat16,
        )
        output = ttnn.reshape(
            grouped_output,
            (geometry.batch_heads, geometry.num_chunks, geometry.chunk_size, geometry.value_dim),
        )
        final_state = self._resolve_final_state(
            grouped_final_states,
            strategy_final_state,
            geometry,
            groups_per_head,
            config.output_memory_config,
        )
        return _ScanResult(output=output, final_state=final_state)

    @abstractmethod
    def _compute_group_entry_states(
        self,
        summary_a: ttnn.Tensor,
        summary_b: ttnn.Tensor,
        initial_state: ttnn.Tensor,
        groups_per_head: int,
        config: _ScanConfig,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        ...

    @abstractmethod
    def _resolve_final_state(
        self,
        grouped_final_states: ttnn.Tensor,
        strategy_final_state: ttnn.Tensor | None,
        geometry: _RecurrenceGeometry,
        groups_per_head: int,
        output_memory_config: ttnn.MemoryConfig,
    ) -> ttnn.Tensor:
        ...


class _LocalGroupedScan(_GroupedScan):
    def _compute_group_entry_states(
        self,
        summary_a: ttnn.Tensor,
        summary_b: ttnn.Tensor,
        initial_state: ttnn.Tensor,
        groups_per_head: int,
        config: _ScanConfig,
    ) -> tuple[ttnn.Tensor, None]:
        group_initial_states = ttnn.transformer.kda_affine_prefix(
            summary_a,
            summary_b,
            initial_state,
            groups_per_head,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=config.affine_prefix_compute_kernel_config,
        )
        return group_initial_states, None

    def _resolve_final_state(
        self,
        grouped_final_states: ttnn.Tensor,
        strategy_final_state: ttnn.Tensor | None,
        geometry: _RecurrenceGeometry,
        groups_per_head: int,
        output_memory_config: ttnn.MemoryConfig,
    ) -> ttnn.Tensor:
        if strategy_final_state is not None:
            raise RuntimeError("local grouped KDA scan unexpectedly produced a distributed final state")
        all_final_states = ttnn.reshape(
            grouped_final_states,
            (geometry.batch_heads, groups_per_head, geometry.key_dim, geometry.value_dim),
        )
        last_final_state = ttnn.slice(
            all_final_states,
            (0, groups_per_head - 1, 0, 0),
            (geometry.batch_heads, groups_per_head, geometry.key_dim, geometry.value_dim),
            memory_config=output_memory_config,
        )
        return ttnn.reshape(last_final_state, (geometry.batch_heads, geometry.key_dim, geometry.value_dim))


class _DistributedGroupedScan(_GroupedScan):
    def __init__(self, sequence_parallel_axis: int) -> None:
        self._sequence_parallel_axis = sequence_parallel_axis

    @property
    def preparation_memory_config(self) -> ttnn.MemoryConfig:
        return ttnn.DRAM_MEMORY_CONFIG

    def _compute_group_entry_states(
        self,
        summary_a: ttnn.Tensor,
        summary_b: ttnn.Tensor,
        initial_state: ttnn.Tensor,
        groups_per_head: int,
        config: _ScanConfig,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        partition_a, partition_b = ttnn.transformer.kda_affine_compose(
            summary_a,
            summary_b,
            groups_per_head,
            memory_config=config.output_memory_config,
            compute_kernel_config=config.affine_prefix_compute_kernel_config,
        )
        partition_entry_state, distributed_final_state = distributed_affine_prefix(
            partition_a,
            partition_b,
            initial_state,
            sequence_parallel_axis=self._sequence_parallel_axis,
            memory_config=config.output_memory_config,
            compute_kernel_config=config.affine_prefix_compute_kernel_config,
            affine_summary_dtype=config.affine_summary_dtype,
            recurrent_state_dtype=config.recurrent_state_dtype,
        )
        if partition_entry_state.dtype != ttnn.float32:
            partition_entry_state = ttnn.typecast(
                partition_entry_state,
                ttnn.float32,
                memory_config=config.output_memory_config,
            )
        group_initial_states = ttnn.transformer.kda_affine_prefix(
            summary_a,
            summary_b,
            partition_entry_state,
            groups_per_head,
            memory_config=config.output_memory_config,
            compute_kernel_config=config.affine_prefix_compute_kernel_config,
        )
        return group_initial_states, distributed_final_state

    def _resolve_final_state(
        self,
        grouped_final_states: ttnn.Tensor,
        strategy_final_state: ttnn.Tensor | None,
        geometry: _RecurrenceGeometry,
        groups_per_head: int,
        output_memory_config: ttnn.MemoryConfig,
    ) -> ttnn.Tensor:
        del grouped_final_states, geometry, groups_per_head, output_memory_config
        if strategy_final_state is None:
            raise RuntimeError("distributed grouped KDA scan did not produce a final state")
        return strategy_final_state


def _select_scan(
    *,
    num_chunks: int,
    summary_group_chunks: int,
    sequence_parallel_axis: int | None,
) -> _RecurrenceScan:
    if sequence_parallel_axis is not None:
        return _DistributedGroupedScan(sequence_parallel_axis)
    if num_chunks >= _MIN_CHUNKS_FOR_GROUPED_SCAN and num_chunks % summary_group_chunks == 0:
        return _LocalGroupedScan()
    return _DirectScan()


def _restore_recurrence_output(
    scan: _ScanResult,
    geometry: _RecurrenceGeometry,
    *,
    output_final_state: bool,
    output_head_major: bool,
) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
    final_state = (
        ttnn.reshape(
            scan.final_state,
            (geometry.batch, geometry.heads, geometry.key_dim, geometry.value_dim),
        )
        if output_final_state
        else None
    )
    if output_head_major:
        if geometry.pad == 0:
            return ttnn.reshape(scan.output, (geometry.batch_heads, geometry.sequence, geometry.value_dim)), final_state
        output = ttnn.to_layout(scan.output, ttnn.ROW_MAJOR_LAYOUT)
        output = ttnn.reshape(output, (geometry.batch_heads, geometry.padded_sequence, geometry.value_dim))
        output = ttnn.slice(output, (0, 0, 0), (geometry.batch_heads, geometry.sequence, geometry.value_dim))
        return ttnn.to_layout(output, ttnn.TILE_LAYOUT), final_state

    output = ttnn.to_layout(scan.output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.reshape(output, (geometry.batch_heads, geometry.padded_sequence, geometry.value_dim))
    if geometry.pad:
        output = ttnn.slice(output, (0, 0, 0), (geometry.batch_heads, geometry.sequence, geometry.value_dim))
    output = ttnn.reshape(output, (geometry.batch, geometry.heads, geometry.sequence, geometry.value_dim))
    output = ttnn.permute(output, (0, 2, 1, 3))
    return output, final_state


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
    """Run KDA chunk recurrence through the selected scan implementation."""
    geometry = _validate_recurrence_geometry(
        q_input,
        k_input,
        v_input,
        gate_input,
        beta_input,
        chunk_size=chunk_size,
        summary_group_chunks=summary_group_chunks,
        sequence_parallel_axis=sequence_parallel_axis,
        affine_summary_dtype=affine_summary_dtype,
        recurrent_state_dtype=recurrent_state_dtype,
        grouped_scan_output_dtype=grouped_scan_output_dtype,
    )
    scan_strategy = _select_scan(
        num_chunks=geometry.num_chunks,
        summary_group_chunks=summary_group_chunks,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    scan_config = _ScanConfig(
        summary_group_chunks=summary_group_chunks,
        output_memory_config=_output_memory_config(memory_config),
        compute_kernel_config=compute_kernel_config,
        affine_summary_dtype=affine_summary_dtype,
        recurrent_state_dtype=recurrent_state_dtype,
        affine_prefix_compute_kernel_config=affine_prefix_compute_kernel_config,
        grouped_scan_output_dtype=grouped_scan_output_dtype,
        grouped_scan_compute_kernel_config=grouped_scan_compute_kernel_config,
    )
    qk_scale = geometry.key_dim**-0.5 if scale is None else scale
    device = q_input.device()
    inputs = _prepare_chunk_inputs(
        q_input,
        k_input,
        v_input,
        gate_input,
        beta_input,
        geometry,
        qk_scale=qk_scale,
    )
    constants = _resolve_chunk_constants(device, eye=eye, tril=tril, ones=ones, masks=masks)
    state = _prepare_initial_state(initial_state, geometry, device=device)
    prepared = _prepare_chunk_terms(
        inputs,
        constants,
        geometry,
        qk_scale=qk_scale,
        memory_config=scan_strategy.preparation_memory_config,
        compute_kernel_config=compute_kernel_config,
        use_bf16_prep_intermediates=use_bf16_prep_intermediates,
    )
    scan = scan_strategy.run(prepared, state, constants[0], geometry, scan_config)
    return _restore_recurrence_output(
        scan,
        geometry,
        output_final_state=output_final_state,
        output_head_major=output_head_major,
    )


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
