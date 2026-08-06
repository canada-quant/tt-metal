// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda.hpp"

#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include "device/kda_phased.hpp"

#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/mesh_partition/mesh_partition.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::transformer {

namespace kda_frontend_detail {

// [B,T,Hh,D] -> [B*Hh, T, D], TILE bf16 (head-major).
ttnn::Tensor kda_head_split_tile(const ttnn::Tensor& x, uint32_t B, uint32_t T, uint32_t Hh, uint32_t D) {
    // TILE-native head-split (permute on TILE via the transpose engine — avoids the
    // untilize-with-unpadding on the small H tile-dim, which hangs in the full op graph).
    // GPU-style mixed precision: q/k/v are bf16 (gate/decay and state stay fp32). Cast here so
    // any caller (fp32 or bf16) feeds the kernel bf16 q/k/v, matching FLA's Triton dtypes.
    ttnn::Tensor t = x;
    if (t.dtype() != DataType::BFLOAT16) {
        t = ttnn::typecast(t, DataType::BFLOAT16);
    }
    t = ttnn::permute(t, ttnn::SmallVector<int64_t>{0, 2, 1, 3});  // [B, Hh, T, D] TILE
    t = ttnn::reshape(t, ttnn::Shape({B * Hh, T, D}));             // [BH, T, D] TILE
    return t;
}

// [B,T,Hn] -> [B*Hn, T], TILE fp32 (permute on TILE, no untilize).

// [B,T,H,D] -> [B*H,T,D], TILE fp32.
ttnn::Tensor kda_head_split_float_tile(const ttnn::Tensor& x, uint32_t B, uint32_t T, uint32_t H, uint32_t D) {
    ttnn::Tensor t = x.dtype() == DataType::FLOAT32 ? x : ttnn::typecast(x, DataType::FLOAT32);
    t = ttnn::permute(t, ttnn::SmallVector<int64_t>{0, 2, 1, 3});
    return ttnn::reshape(t, ttnn::Shape({B * H, T, D}));
}
ttnn::Tensor kda_headvec_split_tile(const ttnn::Tensor& x, uint32_t B, uint32_t T, uint32_t Hn) {
    ttnn::Tensor t = x;
    if (t.dtype() != DataType::FLOAT32) {
        t = ttnn::typecast(t, DataType::FLOAT32);
    }
    t = ttnn::permute(t, ttnn::SmallVector<int64_t>{0, 2, 1});  // [B, Hn, T] TILE
    t = ttnn::reshape(t, ttnn::Shape({B * Hn, T}));             // [BH, T] TILE
    return t;
}

// Pad TILE [BH, T, D] to [BH, L, D] along the time dim with zeros.
ttnn::Tensor kda_pad_time_tile(const ttnn::Tensor& x, uint32_t BH, uint32_t D, uint32_t pad, MeshDevice* dev) {
    if (pad == 0) {
        return x;
    }
    ttnn::Tensor z =
        ttnn::zeros(ttnn::Shape({BH, pad, D}), x.dtype(), Layout::TILE, std::ref(*dev), ttnn::DRAM_MEMORY_CONFIG);
    return ttnn::concat(std::vector<ttnn::Tensor>{x, z}, 1);
}

ttnn::Tensor kda_make_const_cc(const std::vector<float>& data, uint32_t C, MeshDevice* dev) {
    ttnn::Shape shape({1, 1, C, C});
    TensorLayout layout(DataType::FLOAT32, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG);
    tt::tt_metal::TensorSpec spec(shape, layout);
    return ttnn::Tensor::from_vector(data, spec, dev);
}

struct KdaConstTiles {
    ttnn::Tensor eye, tril, ones, masks;
};

size_t kda_chunk_prep_l1_bytes_per_bank(
    uint32_t BH,
    uint32_t NC,
    uint32_t C,
    uint32_t K,
    uint32_t V,
    bool vector_gate,
    uint32_t output_bf16_mask,
    MeshDevice* device) {
    const auto spec = [&](const ttnn::Shape& shape, uint32_t output_index) {
        const auto dtype = (output_bf16_mask & (1u << output_index)) ? DataType::BFLOAT16 : DataType::FLOAT32;
        return tt::tt_metal::TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    };
    const std::vector<tt::tt_metal::TensorSpec> specs = {
        spec(ttnn::Shape({BH, NC, C, V}), 0),
        spec(ttnn::Shape({BH, NC, C, K}), 1),
        spec(ttnn::Shape({BH, NC, C, K}), 2),
        spec(ttnn::Shape({BH, NC, C, C}), 3),
        spec(ttnn::Shape({BH, NC, K, C}), 4),
        spec(ttnn::Shape({BH, NC, vector_gate ? K : 1, 1}), 5),
        spec(ttnn::Shape({BH, NC, C, C}), 6),
    };
    const auto num_banks = device->allocator()->get_num_banks(BufferType::L1);
    const auto alignment = device->allocator()->get_alignment(BufferType::L1);
    size_t bytes_per_bank = 0;
    for (const auto& output_spec : specs) {
        bytes_per_bank += tt::tt_metal::detail::calculate_bank_size_spread(
            output_spec.compute_packed_buffer_size_bytes(),
            output_spec.compute_page_size_bytes(),
            num_banks,
            alignment);
    }
    return bytes_per_bank;
}

ttnn::Tensor kda_slice_group_axis(
    const ttnn::Tensor& tensor, uint32_t start, uint32_t end, const tt::tt_metal::MemoryConfig& memory_config) {
    const auto& shape = tensor.logical_shape();
    TT_FATAL(shape.rank() == 4, "group-axis slice expects a rank-4 tensor");
    return ttnn::slice(
        tensor,
        ttnn::SmallVector<int32_t>{0, static_cast<int32_t>(start), 0, 0},
        ttnn::SmallVector<int32_t>{
            static_cast<int32_t>(shape[0]),
            static_cast<int32_t>(end),
            static_cast<int32_t>(shape[2]),
            static_cast<int32_t>(shape[3])},
        ttnn::SmallVector<int32_t>{1, 1, 1, 1},
        memory_config);
}

// Three 32x32 quadrant masks packed into one [1,1,32,96] tile-row (tile 0 = top-left,
// tile 1 = bottom-right, tile 2 = bottom-left). Used by the prep kernel's 16x16 sub-blocked
// WY inverse to isolate the four 16-quadrants of each 32x32 diagonal block.
ttnn::Tensor kda_make_quadrant_masks(MeshDevice* dev) {
    std::vector<float> m(32 * 96, 0.0f);
    for (uint32_t i = 0; i < 32; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            const bool lo_i = i < 16, lo_j = j < 16;
            m[i * 96 + 0 * 32 + j] = (lo_i && lo_j) ? 1.0f : 0.0f;    // Qtl
            m[i * 96 + 1 * 32 + j] = (!lo_i && !lo_j) ? 1.0f : 0.0f;  // Qbr
            m[i * 96 + 2 * 32 + j] = (!lo_i && lo_j) ? 1.0f : 0.0f;   // Q10
        }
    }
    ttnn::Shape shape({1, 1, 32, 96});
    TensorLayout layout(DataType::FLOAT32, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG);
    return ttnn::Tensor::from_vector(m, tt::tt_metal::TensorSpec(shape, layout), dev);
}

// eye/tril/ones depend only on the chunk size, and the zero initial-state only on shape — none
// depend on runtime data, and all must be device-resident before trace capture (host<->device
// transfers are illegal under trace). The op therefore takes these as optional arguments so the
// CALLER owns them (built once, e.g. on the model/layer object) and their lifetime is tied to the
// device — not to a process-lifetime C++ static, which would deallocate at exit AFTER the device is
// gone and SIGSEGV. These builders are the eager-only fallback for callers that don't supply them
// (a build here does a host upload and so is NOT valid under trace capture — pass the tensors in).
KdaConstTiles kda_build_const_tiles(uint32_t C, MeshDevice* dev) {
    std::vector<float> eye_data(static_cast<size_t>(C) * C, 0.0f);
    std::vector<float> tril_data(static_cast<size_t>(C) * C, 0.0f);
    for (uint32_t i = 0; i < C; i++) {
        eye_data[i * C + i] = 1.0f;
        for (uint32_t j = 0; j <= i; j++) {
            tril_data[i * C + j] = 1.0f;
        }
    }
    std::vector<float> ones_data(static_cast<size_t>(C) * C, 1.0f);
    return KdaConstTiles{
        kda_make_const_cc(eye_data, C, dev),
        kda_make_const_cc(tril_data, C, dev),
        kda_make_const_cc(ones_data, C, dev),
        kda_make_quadrant_masks(dev)};
}

ttnn::Tensor kda_build_zero_state(uint32_t BH, uint32_t K, uint32_t V, MeshDevice* dev) {
    return ttnn::zeros(
        ttnn::Shape({BH, K, V}), DataType::FLOAT32, Layout::TILE, std::ref(*dev), ttnn::DRAM_MEMORY_CONFIG);
}

}  // namespace kda_frontend_detail
using namespace kda_frontend_detail;

std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> chunk_kda(
    const ttnn::Tensor& q_in,
    const ttnn::Tensor& k_in,
    const ttnn::Tensor& v_in,
    const ttnn::Tensor& g_in,
    const ttnn::Tensor& beta_in,
    std::optional<float> scale_opt,
    const std::optional<ttnn::Tensor>& initial_state,
    bool output_final_state,
    bool output_head_major,
    uint32_t chunk_size,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& eye,
    const std::optional<ttnn::Tensor>& tril,
    const std::optional<ttnn::Tensor>& ones,
    const std::optional<ttnn::Tensor>& masks,
    uint32_t summary_group_chunks,
    const std::optional<uint32_t>& sequence_parallel_axis,
    DataType affine_summary_dtype,
    DataType recurrent_state_dtype,
    const std::optional<ttnn::DeviceComputeKernelConfig>& affine_prefix_compute_kernel_config,
    DataType grouped_scan_output_dtype,
    const std::optional<ttnn::DeviceComputeKernelConfig>& grouped_scan_compute_kernel_config,
    bool use_bf16_prep_intermediates) {
    const auto& qs = q_in.logical_shape();
    const auto& vs = v_in.logical_shape();
    const auto& gs = g_in.logical_shape();
    const auto& bs = beta_in.logical_shape();
    const bool flat_v = vs.rank() == 3;
    const bool flat_qk = qs.rank() == 3;
    const bool flat_g = gs.rank() == 3;
    TT_FATAL(flat_g || gs.rank() == 4, "chunk_kda expects rank-3 or rank-4 g");
    TT_FATAL(bs.rank() == 3, "chunk_kda beta must be [B,T,H]");
    const uint32_t B = bs[0], T = bs[1], H = bs[2];
    TT_FATAL(!flat_g || gs[2] % H == 0, "chunk_kda flat g width {} must be divisible by H={}", gs[2], H);
    const uint32_t K = flat_g ? (gs[2] / H) : gs[3];
    TT_FATAL(flat_qk || qs.rank() == 4, "chunk_kda expects rank-3 or rank-4 q/k");
    TT_FATAL(flat_v || vs.rank() == 4, "chunk_kda expects rank-3 or rank-4 v");
    TT_FATAL(!flat_qk || qs[2] == H * K, "chunk_kda flat q/k width {} must equal H*K={}*{}", qs[2], H, K);
    TT_FATAL(!flat_v || vs[2] % H == 0, "chunk_kda flat v width {} must be divisible by H={}", vs[2], H);
    const uint32_t V = flat_v ? (vs[2] / H) : vs[3];
    const bool distributed_prefix = sequence_parallel_axis.has_value();
    TT_FATAL(!distributed_prefix || *sequence_parallel_axis < 2, "sequence_parallel_axis must be 0 or 1");
    TT_FATAL(
        affine_summary_dtype == DataType::FLOAT32 || affine_summary_dtype == DataType::BFLOAT16,
        "affine_summary_dtype must be FLOAT32 or BFLOAT16");
    TT_FATAL(
        recurrent_state_dtype == DataType::FLOAT32 || recurrent_state_dtype == DataType::BFLOAT16,
        "recurrent_state_dtype must be FLOAT32 or BFLOAT16");
    TT_FATAL(
        grouped_scan_output_dtype == DataType::FLOAT32 || grouped_scan_output_dtype == DataType::BFLOAT16,
        "grouped_scan_output_dtype must be FLOAT32 or BFLOAT16");
    TT_FATAL(chunk_size == 32, "chunk_kda currently requires chunk_size=32, got {}", chunk_size);
    TT_FATAL(
        k_in.logical_shape() == qs && qs[0] == B && qs[1] == T &&
            (flat_qk ? qs[2] == H * K : (qs[2] == H && qs[3] == K)) && vs[0] == B && vs[1] == T &&
            (flat_v ? vs[2] == H * V : (vs[2] == H && vs[3] == V)),
        "chunk_kda q/k/v shapes are inconsistent");
    TT_FATAL(
        gs[0] == B && gs[1] == T && (flat_g ? gs[2] == H * K : (gs[2] == H && gs[3] == K)),
        "chunk_kda g must be [B,T,H,K] or flat [B,T,H*K]");
    TT_FATAL(bs[0] == B && bs[1] == T && bs[2] == H, "chunk_kda beta must be [B,T,H]");

    auto* dev = q_in.device();
    const uint32_t BH = B * H;
    const uint32_t C = chunk_size;
    const uint32_t pad = (C - (T % C)) % C;
    const uint32_t L = T + pad;
    const uint32_t NC = L / C;
    const float scale = scale_opt.value_or(1.0f / std::sqrt(static_cast<float>(K)));

    auto kda_as_bf16 = [](const ttnn::Tensor& tensor) {
        return tensor.dtype() == DataType::BFLOAT16 ? tensor : ttnn::typecast(tensor, DataType::BFLOAT16);
    };
    ttnn::Tensor q = flat_qk ? kda_as_bf16(q_in) : ttnn::multiply(kda_head_split_tile(q_in, B, T, H, K), scale);
    ttnn::Tensor k = flat_qk ? kda_as_bf16(k_in) : kda_head_split_tile(k_in, B, T, H, K);
    ttnn::Tensor v = flat_v ? (v_in.dtype() == DataType::BFLOAT16 ? v_in : ttnn::typecast(v_in, DataType::BFLOAT16))
                            : kda_head_split_tile(v_in, B, T, H, V);
    ttnn::Tensor g = flat_g ? g_in : kda_head_split_float_tile(g_in, B, T, H, K);
    ttnn::Tensor beta = kda_headvec_split_tile(beta_in, B, T, H);
    TT_FATAL(!flat_qk || pad == 0, "chunk_kda flat q/k requires T to be divisible by chunk_size");
    TT_FATAL(!flat_v || pad == 0, "chunk_kda flat v requires T to be divisible by chunk_size");
    TT_FATAL(!flat_g || pad == 0, "chunk_kda flat g requires T to be divisible by chunk_size");
    if (!flat_qk) {
        q = kda_pad_time_tile(q, BH, K, pad, dev);
        k = kda_pad_time_tile(k, BH, K, pad, dev);
    }
    if (!flat_v) {
        v = kda_pad_time_tile(v, BH, V, pad, dev);
    }
    if (!flat_g) {
        g = kda_pad_time_tile(g, BH, K, pad, dev);
    }
    if (pad > 0) {
        auto zeros = ttnn::zeros(
            ttnn::Shape({BH, pad}), DataType::FLOAT32, Layout::TILE, std::ref(*dev), ttnn::DRAM_MEMORY_CONFIG);
        beta = ttnn::concat(std::vector<ttnn::Tensor>{beta, zeros}, 1);
    }
    if (!flat_qk) {
        q = ttnn::reshape(q, ttnn::Shape({BH, NC, C, K}));
        k = ttnn::reshape(k, ttnn::Shape({BH, NC, C, K}));
    }
    if (!flat_v) {
        v = ttnn::reshape(v, ttnn::Shape({BH, NC, C, V}));
    }
    if (!flat_g) {
        g = ttnn::reshape(g, ttnn::Shape({BH, NC, C, K}));
    }
    beta = ttnn::reshape(beta, ttnn::Shape({BH, NC, C, 1}));

    const bool has_const_tiles = eye.has_value() && tril.has_value() && ones.has_value() && masks.has_value();
    KdaConstTiles fallback;
    if (!has_const_tiles) {
        fallback = kda_build_const_tiles(C, dev);
    }
    const auto& eye_c = has_const_tiles ? *eye : fallback.eye;
    const auto& tril_c = has_const_tiles ? *tril : fallback.tril;
    const auto& ones_c = has_const_tiles ? *ones : fallback.ones;
    const auto& masks_c = has_const_tiles ? *masks : fallback.masks;

    std::optional<ttnn::Tensor> s0;
    if (initial_state.has_value()) {
        auto state = initial_state->dtype() == DataType::FLOAT32 ? *initial_state
                                                                 : ttnn::typecast(*initial_state, DataType::FLOAT32);
        s0 = ttnn::reshape(state, ttnn::Shape({BH, K, V}));
    } else {
        s0 = kda_build_zero_state(BH, K, V, dev);
    }

    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_cfg = init_device_compute_kernel_config(
        dev->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    const auto affine_prefix_kernel_cfg = init_device_compute_kernel_config(
        dev->arch(),
        affine_prefix_compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    const auto grouped_scan_kernel_cfg = init_device_compute_kernel_config(
        dev->arch(),
        grouped_scan_compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    // Measured Kimi-K3 storage choice: BF16 kd, q_decay, and dl; other prep outputs stay FP32.
    const uint32_t prep_bf16_mask = use_bf16_prep_intermediates ? ((1u << 1) | (1u << 2) | (1u << 5)) : 0u;
    const auto prep_cb_bytes = ttnn::prim::kda_chunk_prep_cb_size_bytes(C, K, V, true, g.dtype(), prep_bf16_mask);
    const auto prep_output_bytes_per_bank =
        kda_chunk_prep_l1_bytes_per_bank(BH, NC, C, K, V, true, prep_bf16_mask, dev);
    const auto l1_budget =
        dev->l1_size_per_core() - dev->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const bool prep_fits_l1 = prep_cb_bytes + prep_output_bytes_per_bank <= l1_budget;
    // Keep memory selection stable across warmup and trace capture. L1 is used only when the
    // geometry fits the device's fixed allocatable budget; distributed summaries stay in DRAM.
    const auto prep_mem = distributed_prefix || !prep_fits_l1 ? ttnn::DRAM_MEMORY_CONFIG : ttnn::L1_MEMORY_CONFIG;
    auto prep = ttnn::prim::kda_chunk_prep(
        q,
        k,
        v,
        g,
        beta,
        eye_c,
        tril_c,
        ones_c,
        masks_c,
        C,
        prep_mem,
        kernel_cfg,
        flat_v,
        H,
        flat_qk,
        scale,
        flat_qk,
        H,
        flat_g,
        true,
        prep_bf16_mask);
    std::optional<std::vector<ttnn::Tensor>> grouped_scan;
    std::optional<ttnn::Tensor> distributed_final_state;
    // Contiguous chunks become one independent pseudo-head. Running the recurrence from zero gives B; running
    // from I gives A+B. State-only mode drains token outputs without materializing them.
    // The retained T=5120 path has 160 local chunks; below that, the extra summary/compose pass is not amortized.
    constexpr uint32_t kPersistentGroupPrefixMinChunks = 160;
    TT_FATAL(summary_group_chunks > 0, "summary_group_chunks must be positive");
    const bool use_persistent_group_prefix = NC >= kPersistentGroupPrefixMinChunks && NC % summary_group_chunks == 0;
    const bool build_group_summaries = distributed_prefix || use_persistent_group_prefix;
    if (build_group_summaries) {
        TT_FATAL(
            NC % summary_group_chunks == 0,
            "local chunk count {} must be divisible by summary_group_chunks {}",
            NC,
            summary_group_chunks);
        TT_FATAL(K == V, "grouped KDA affine prefix currently requires K == V, got K={} and V={}", K, V);
        const uint32_t groups_per_head = NC / summary_group_chunks;
        const uint32_t group_heads = BH * groups_per_head;
        const auto worker_grid = dev->compute_with_storage_grid_size();
        constexpr uint32_t kMaxAffinePrefixWorkers = 128;
        const uint32_t available_group_workers =
            std::min<uint32_t>(worker_grid.x * worker_grid.y, kMaxAffinePrefixWorkers);
        TT_FATAL(
            group_heads <= available_group_workers,
            "grouped KDA needs {} summary owners (B*local_heads*local_groups), but only {} are supported",
            group_heads,
            available_group_workers);
        auto grouped = prep;
        grouped[0] = ttnn::reshape(grouped[0], ttnn::Shape({group_heads, summary_group_chunks, C, V}));
        grouped[1] = ttnn::reshape(grouped[1], ttnn::Shape({group_heads, summary_group_chunks, C, K}));
        grouped[2] = ttnn::reshape(grouped[2], ttnn::Shape({group_heads, summary_group_chunks, C, K}));
        grouped[3] = ttnn::reshape(grouped[3], ttnn::Shape({group_heads, summary_group_chunks, C, C}));
        grouped[4] = ttnn::reshape(grouped[4], ttnn::Shape({group_heads, summary_group_chunks, K, C}));
        grouped[5] = ttnn::reshape(grouped[5], ttnn::Shape({group_heads, summary_group_chunks, K, 1}));
        grouped[6] = ttnn::reshape(grouped[6], ttnn::Shape({group_heads, summary_group_chunks, C, C}));
        const auto summary_cores =
            tt::tt_metal::num_cores_to_corerangeset(group_heads, dev->compute_with_storage_grid_size(), true);
        const auto summary_mem = ttnn::operations::data_movement::create_sharded_memory_config(
            ttnn::Shape({group_heads, K, K}),
            summary_cores,
            ttnn::operations::data_movement::ShardStrategy::HEIGHT,
            ShardOrientation::ROW_MAJOR,
            std::array<uint32_t, 2>{K, K},
            Layout::TILE);
        auto summaries = ttnn::prim::kda_final_scan(
            grouped[0],
            grouped[1],
            grouped[2],
            grouped[3],
            grouped[4],
            grouped[5],
            grouped[6],
            std::nullopt,
            C,
            true,
            summary_mem,
            kernel_cfg,
            true,
            true,
            eye_c,
            true);
        // The summary scan returns A+B (identity seed) and B (zero seed); recover A at the
        // KDA composition layer so the shared scan primitive remains one device operation.
        summaries[0] = ttnn::subtract(summaries[0], summaries[1], std::nullopt, summary_mem);
        auto summary_a = summaries[0].dtype() == affine_summary_dtype
                             ? summaries[0]
                             : ttnn::typecast(summaries[0], affine_summary_dtype, summary_mem);
        auto summary_b = summaries[1].dtype() == affine_summary_dtype
                             ? summaries[1]
                             : ttnn::typecast(summaries[1], affine_summary_dtype, summary_mem);
        TT_FATAL(s0.has_value(), "group-prefix scan requires initial state");
        const auto prefix_mem = distributed_prefix ? out_mem : ttnn::L1_MEMORY_CONFIG;
        const auto run_grouped_scan = [&](const ttnn::Tensor& group_initial_states) {
            return ttnn::prim::kda_final_scan(
                grouped[0],
                grouped[1],
                grouped[2],
                grouped[3],
                grouped[4],
                grouped[5],
                grouped[6],
                group_initial_states,
                C,
                true,
                out_mem,
                grouped_scan_kernel_cfg,
                true,
                false,
                std::nullopt,
                false,
                grouped_scan_output_dtype == DataType::BFLOAT16);
        };
        if (distributed_prefix) {
            auto [partition_a, partition_b] = ttnn::prim::kda_affine_compose(
                summary_a, summary_b, groups_per_head, prefix_mem, affine_prefix_kernel_cfg);
            auto [partition_entry_state, final_state] = kda_distributed_affine_prefix(
                partition_a,
                partition_b,
                *s0,
                *sequence_parallel_axis,
                out_mem,
                affine_prefix_kernel_cfg,
                affine_summary_dtype,
                recurrent_state_dtype);
            distributed_final_state = final_state;
            // The grouped affine-prefix kernel computes and returns FP32 states. The distributed
            // carry remains in recurrent_state_dtype; promote only at this FP32 compute boundary.
            if (partition_entry_state.dtype() != DataType::FLOAT32) {
                partition_entry_state = ttnn::typecast(partition_entry_state, DataType::FLOAT32, prefix_mem);
            }
            auto group_initial_states = ttnn::prim::kda_affine_prefix(
                summary_a, summary_b, partition_entry_state, groups_per_head, prefix_mem, affine_prefix_kernel_cfg);
            grouped_scan = run_grouped_scan(group_initial_states);
        } else {
            auto group_initial_states = ttnn::prim::kda_affine_prefix(
                summary_a, summary_b, *s0, groups_per_head, prefix_mem, affine_prefix_kernel_cfg);
            grouped_scan = run_grouped_scan(group_initial_states);
        }
        (*grouped_scan)[0] = ttnn::reshape((*grouped_scan)[0], ttnn::Shape({BH, NC, C, V}));
        if (distributed_final_state.has_value()) {
            (*grouped_scan)[1] = *distributed_final_state;
        } else {
            auto all_final_states = ttnn::reshape((*grouped_scan)[1], ttnn::Shape({BH, groups_per_head, K, V}));
            (*grouped_scan)[1] = ttnn::reshape(
                kda_slice_group_axis(all_final_states, groups_per_head - 1, groups_per_head, out_mem),
                ttnn::Shape({BH, K, V}));
        }
    }
    std::vector<ttnn::Tensor> scan;
    if (grouped_scan.has_value()) {
        scan = *grouped_scan;
    } else {
        scan = ttnn::prim::kda_final_scan(
            prep[0],
            prep[1],
            prep[2],
            prep[3],
            prep[4],
            prep[5],
            prep[6],
            s0,
            C,
            output_final_state,
            out_mem,
            kernel_cfg,
            true,
            false,
            std::nullopt);
    }

    std::optional<ttnn::Tensor> final_state;
    if (output_final_state) {
        final_state = ttnn::reshape(scan[1], ttnn::Shape({B, H, K, V}));
    }
    if (output_head_major) {
        if (pad == 0) {
            return {ttnn::reshape(scan[0], ttnn::Shape({BH, T, V})), final_state};
        }
        auto output = ttnn::to_layout(scan[0], Layout::ROW_MAJOR);
        output = ttnn::reshape(output, ttnn::Shape({BH, L, V}));
        output = ttnn::slice(
            output,
            ttnn::SmallVector<int32_t>{0, 0, 0},
            ttnn::SmallVector<int32_t>{static_cast<int32_t>(BH), static_cast<int32_t>(T), static_cast<int32_t>(V)},
            ttnn::SmallVector<int32_t>{1, 1, 1});
        return {ttnn::to_layout(output, Layout::TILE), final_state};
    }

    ttnn::Tensor output = ttnn::to_layout(scan[0], Layout::ROW_MAJOR);
    output = ttnn::reshape(output, ttnn::Shape({BH, L, V}));
    if (pad > 0) {
        output = ttnn::slice(
            output,
            ttnn::SmallVector<int32_t>{0, 0, 0},
            ttnn::SmallVector<int32_t>{static_cast<int32_t>(BH), static_cast<int32_t>(T), static_cast<int32_t>(V)},
            ttnn::SmallVector<int32_t>{1, 1, 1});
    }
    output = ttnn::reshape(output, ttnn::Shape({B, H, T, V}));
    output = ttnn::permute(output, ttnn::SmallVector<int64_t>{0, 2, 1, 3});
    return {output, final_state};
}

namespace kda_frontend_detail {

std::tuple<ttnn::Tensor, ttnn::Tensor> kda_detail_all_gather_affine_prefix(
    const ttnn::Tensor& transform_a,
    const ttnn::Tensor& transform_b,
    const ttnn::Tensor& initial_state,
    uint32_t sequence_parallel_axis,
    uint32_t sp_size,
    const ttnn::MemoryConfig& out_mem,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    DataType affine_summary_dtype,
    DataType recurrent_state_dtype) {
    const auto shape = transform_a.logical_shape();
    TT_FATAL(
        shape.rank() == 3 || shape.rank() == 4,
        "KDA all-gather affine prefix requires rank-3 production or rank-4 test transforms");
    const bool has_explicit_sp_dimension = shape.rank() == 4;
    const uint32_t batch_heads = shape[shape.rank() - 3];
    const uint32_t key_dim = shape[shape.rank() - 2];
    const uint32_t value_dim = transform_b.logical_shape()[shape.rank() - 1];

    auto summary_a = ttnn::typecast(transform_a, affine_summary_dtype, ttnn::DRAM_MEMORY_CONFIG);
    auto summary_b = ttnn::typecast(transform_b, affine_summary_dtype, ttnn::DRAM_MEMORY_CONFIG);
    if (!has_explicit_sp_dimension) {
        summary_a = ttnn::reshape(summary_a, ttnn::Shape({1, batch_heads, key_dim, key_dim}));
        summary_b = ttnn::reshape(summary_b, ttnn::Shape({1, batch_heads, key_dim, value_dim}));
    }
    auto packed = ttnn::concat({summary_a, summary_b}, 3, ttnn::DRAM_MEMORY_CONFIG);
    auto gathered = ttnn::all_gather(packed, 0, sequence_parallel_axis, ttnn::DRAM_MEMORY_CONFIG);

    auto carry = ttnn::typecast(initial_state, recurrent_state_dtype, ttnn::L1_MEMORY_CONFIG);
    if (!has_explicit_sp_dimension) {
        carry = ttnn::reshape(carry, ttnn::Shape({1, batch_heads, key_dim, value_dim}));
    }
    std::vector<ttnn::Tensor> entry_states;
    entry_states.reserve(sp_size);
    for (uint32_t rank = 0; rank < sp_size; ++rank) {
        entry_states.push_back(carry);
        const auto begin_a = ttnn::SmallVector<int32_t>{static_cast<int32_t>(rank), 0, 0, 0};
        const auto end_a = ttnn::SmallVector<int32_t>{
            static_cast<int32_t>(rank + 1),
            static_cast<int32_t>(batch_heads),
            static_cast<int32_t>(key_dim),
            static_cast<int32_t>(key_dim)};
        const auto begin_b =
            ttnn::SmallVector<int32_t>{static_cast<int32_t>(rank), 0, 0, static_cast<int32_t>(key_dim)};
        const auto end_b = ttnn::SmallVector<int32_t>{
            static_cast<int32_t>(rank + 1),
            static_cast<int32_t>(batch_heads),
            static_cast<int32_t>(key_dim),
            static_cast<int32_t>(key_dim + value_dim)};
        auto rank_a = ttnn::slice(gathered, begin_a, end_a, {1, 1, 1, 1}, ttnn::L1_MEMORY_CONFIG);
        auto rank_b = ttnn::slice(gathered, begin_b, end_b, {1, 1, 1, 1}, ttnn::L1_MEMORY_CONFIG);
        if (rank_a.dtype() != recurrent_state_dtype) {
            rank_a = ttnn::typecast(rank_a, recurrent_state_dtype, ttnn::L1_MEMORY_CONFIG);
        }
        carry = ttnn::matmul(
            rank_a,
            carry,
            false,
            false,
            ttnn::L1_MEMORY_CONFIG,
            recurrent_state_dtype,
            std::nullopt,
            std::nullopt,
            compute_kernel_config);
        if (rank_b.dtype() != recurrent_state_dtype) {
            rank_b = ttnn::typecast(rank_b, recurrent_state_dtype, ttnn::L1_MEMORY_CONFIG);
        }
        carry = ttnn::add(carry, rank_b, std::nullopt, ttnn::L1_MEMORY_CONFIG);
    }

    auto replicated_entries = ttnn::concat(entry_states, 0, ttnn::DRAM_MEMORY_CONFIG);
    auto entry_state = ttnn::mesh_partition(replicated_entries, 0, sequence_parallel_axis, out_mem);
    auto final_state = ttnn::typecast(carry, recurrent_state_dtype, out_mem);
    if (!has_explicit_sp_dimension) {
        entry_state = ttnn::reshape(entry_state, ttnn::Shape({batch_heads, key_dim, value_dim}));
        final_state = ttnn::reshape(final_state, ttnn::Shape({batch_heads, key_dim, value_dim}));
    }
    return {entry_state, final_state};
}

std::tuple<ttnn::Tensor, ttnn::Tensor> kda_detail_all_gather_convolution_halo(
    const ttnn::Tensor& projected_qkv,
    const ttnn::Tensor& initial_carry,
    uint32_t sequence_parallel_axis,
    uint32_t sp_size,
    uint32_t history,
    const ttnn::MemoryConfig& out_mem) {
    const auto qkv_shape = projected_qkv.logical_shape();
    const uint32_t local_sequence = qkv_shape[1];
    const uint32_t channels = qkv_shape[2];
    auto local_tail = ttnn::slice(
        projected_qkv,
        ttnn::SmallVector<int32_t>{0, static_cast<int32_t>(local_sequence - history), 0},
        ttnn::SmallVector<int32_t>{
            static_cast<int32_t>(qkv_shape[0]), static_cast<int32_t>(local_sequence), static_cast<int32_t>(channels)},
        ttnn::SmallVector<int32_t>{1, 1, 1},
        ttnn::DRAM_MEMORY_CONFIG);
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    TT_FATAL(history <= tile_height, "KDA convolution history must fit in one tile");
    // Gather one padded boundary tile per SP rank. Row-major K3 tails produce 18--36 KiB pages and hang;
    // TILE layout uses 2 KiB pages and follows the stable all-gather path.
    const ttsl::SmallVector<std::array<uint32_t, 2>> padding = {{0, 0}, {0, tile_height - history}, {0, 0}};
    auto tiled_tail =
        ttnn::to_layout(ttnn::pad(local_tail, padding, 0.0F, true, ttnn::DRAM_MEMORY_CONFIG), Layout::TILE);
    auto gathered_tails = ttnn::all_gather(tiled_tail, 1, sequence_parallel_axis, ttnn::DRAM_MEMORY_CONFIG);

    std::vector<ttnn::Tensor> entry_carries{initial_carry};
    entry_carries.reserve(sp_size);
    for (uint32_t rank = 0; rank + 1 < sp_size; ++rank) {
        auto tiled_rank_tail = ttnn::slice(
            gathered_tails,
            ttnn::SmallVector<int32_t>{0, static_cast<int32_t>(rank * tile_height), 0},
            ttnn::SmallVector<int32_t>{
                static_cast<int32_t>(qkv_shape[0]),
                static_cast<int32_t>((rank + 1) * tile_height),
                static_cast<int32_t>(channels)},
            ttnn::SmallVector<int32_t>{1, 1, 1},
            ttnn::DRAM_MEMORY_CONFIG);
        auto rank_tail = ttnn::to_layout(tiled_rank_tail, Layout::ROW_MAJOR);
        entry_carries.push_back(ttnn::slice(
            rank_tail,
            ttnn::SmallVector<int32_t>{0, 0, 0},
            ttnn::SmallVector<int32_t>{
                static_cast<int32_t>(qkv_shape[0]), static_cast<int32_t>(history), static_cast<int32_t>(channels)},
            ttnn::SmallVector<int32_t>{1, 1, 1},
            out_mem));
    }
    auto replicated_entries = ttnn::concat(entry_carries, 1, out_mem);
    auto partition_carry = ttnn::mesh_partition(replicated_entries, 1, sequence_parallel_axis, out_mem);

    auto tiled_final_carry = ttnn::slice(
        gathered_tails,
        ttnn::SmallVector<int32_t>{0, static_cast<int32_t>((sp_size - 1) * tile_height), 0},
        ttnn::SmallVector<int32_t>{
            static_cast<int32_t>(qkv_shape[0]),
            static_cast<int32_t>(sp_size * tile_height),
            static_cast<int32_t>(channels)},
        ttnn::SmallVector<int32_t>{1, 1, 1},
        ttnn::DRAM_MEMORY_CONFIG);
    auto final_carry = ttnn::slice(
        ttnn::to_layout(tiled_final_carry, Layout::ROW_MAJOR),
        ttnn::SmallVector<int32_t>{0, 0, 0},
        ttnn::SmallVector<int32_t>{
            static_cast<int32_t>(qkv_shape[0]), static_cast<int32_t>(history), static_cast<int32_t>(channels)},
        ttnn::SmallVector<int32_t>{1, 1, 1},
        out_mem);
    return {partition_carry, final_carry};
}

}  // namespace kda_frontend_detail
using namespace kda_frontend_detail;

std::tuple<ttnn::Tensor, ttnn::Tensor> kda_distributed_affine_prefix(
    const ttnn::Tensor& transform_a,
    const ttnn::Tensor& transform_b,
    const ttnn::Tensor& initial_state,
    uint32_t sequence_parallel_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    DataType affine_summary_dtype,
    DataType recurrent_state_dtype) {
    TT_FATAL(sequence_parallel_axis < 2, "sequence_parallel_axis must be 0 or 1");
    TT_FATAL(
        transform_a.logical_shape() == transform_b.logical_shape() &&
            transform_a.logical_shape() == initial_state.logical_shape(),
        "distributed KDA affine prefix requires equal batched [K,K] tensor shapes");
    TT_FATAL(transform_a.logical_shape().rank() >= 3, "distributed KDA affine prefix expects batched matrices");
    TT_FATAL(
        transform_a.logical_shape()[-2] == transform_a.logical_shape()[-1],
        "distributed KDA affine prefix currently requires K == V");
    TT_FATAL(
        affine_summary_dtype == DataType::FLOAT32 || affine_summary_dtype == DataType::BFLOAT16,
        "affine_summary_dtype must be FLOAT32 or BFLOAT16");
    TT_FATAL(
        recurrent_state_dtype == DataType::FLOAT32 || recurrent_state_dtype == DataType::BFLOAT16,
        "recurrent_state_dtype must be FLOAT32 or BFLOAT16");
    for (const auto* tensor : {&transform_a, &transform_b, &initial_state}) {
        TT_FATAL(
            tensor->dtype() == DataType::FLOAT32 || tensor->dtype() == DataType::BFLOAT16,
            "distributed KDA affine prefix requires FP32 or BF16 tensors");
        TT_FATAL(tensor->layout() == Layout::TILE, "distributed KDA affine prefix requires TILE tensors");
        TT_FATAL(
            tensor->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "distributed KDA affine prefix requires interleaved tensors");
    }

    auto* mesh_device = transform_a.device();
    TT_FATAL(mesh_device != nullptr, "distributed KDA affine prefix requires a mesh device");
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(mesh_shape.dims() == 2, "distributed KDA affine prefix requires a 2D mesh");
    const uint32_t sp_size = mesh_shape[sequence_parallel_axis];
    TT_FATAL(sp_size > 1, "distributed KDA affine prefix requires SP > 1");
    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    return kda_detail_all_gather_affine_prefix(
        transform_a,
        transform_b,
        initial_state,
        sequence_parallel_axis,
        sp_size,
        out_mem,
        compute_kernel_config,
        affine_summary_dtype,
        recurrent_state_dtype);
}

std::tuple<ttnn::Tensor, ttnn::Tensor> kda_convolution_halo(
    const ttnn::Tensor& projected_qkv,
    const ttnn::Tensor& initial_carry,
    uint32_t sequence_parallel_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    TT_FATAL(sequence_parallel_axis < 2, "sequence_parallel_axis must be 0 or 1");
    const auto qkv_shape = projected_qkv.logical_shape();
    const auto carry_shape = initial_carry.logical_shape();
    TT_FATAL(qkv_shape.rank() == 3 && carry_shape.rank() == 3, "KDA convolution halo expects rank-3 tensors");
    TT_FATAL(
        qkv_shape[0] == carry_shape[0] && qkv_shape[2] == carry_shape[2],
        "KDA convolution halo requires matching batch and channel dimensions");
    const uint32_t history = carry_shape[1];
    TT_FATAL(history > 0 && qkv_shape[1] >= history, "KDA convolution halo requires 0 < history <= local T");
    TT_FATAL(projected_qkv.dtype() == initial_carry.dtype(), "KDA convolution halo requires matching dtypes");
    TT_FATAL(projected_qkv.layout() == initial_carry.layout(), "KDA convolution halo requires matching layouts");
    for (const auto* tensor : {&projected_qkv, &initial_carry}) {
        TT_FATAL(
            tensor->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "KDA convolution halo requires interleaved tensors");
    }

    auto* mesh_device = projected_qkv.device();
    TT_FATAL(mesh_device != nullptr, "KDA convolution halo requires a mesh device");
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(mesh_shape.dims() == 2, "KDA convolution halo requires a 2D mesh");
    const uint32_t sp_size = mesh_shape[sequence_parallel_axis];
    TT_FATAL(sp_size > 1, "KDA convolution halo requires SP > 1");
    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    return kda_detail_all_gather_convolution_halo(
        projected_qkv, initial_carry, sequence_parallel_axis, sp_size, history, out_mem);
}

ttnn::Tensor kda_gated_rms_norm(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& weight,
    uint32_t num_heads,
    float epsilon,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    DataType output_dtype) {
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/true);
    return ttnn::prim::kda_gated_rms_norm(
        input, gate, weight, num_heads, epsilon, output_memory_config, kernel_config, output_dtype);
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> kda_causal_conv1d_split(
    const ttnn::Tensor& input,
    const ttnn::Tensor& state,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    auto outputs = ttnn::prim::kda_causal_conv1d_split(
        input, state, tap0, tap1, tap2, tap3, q_width, k_width, v_width, out_mem, kernel_config);
    return {outputs[0], outputs[1], outputs[2]};
}

}  // namespace ttnn::transformer
