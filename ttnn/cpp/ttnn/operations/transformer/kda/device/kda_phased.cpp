// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_phased.hpp"

#include <cstdlib>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace kda_primitive_detail {
void check(const Tensor& t, const char* name, DataType dt) {
    TT_FATAL(t.layout() == Layout::TILE, "chunk_gdn: {} must be TILE layout", name);
    TT_FATAL(t.dtype() == dt, "chunk_gdn: {} has wrong dtype", name);
    TT_FATAL(t.buffer() != nullptr, "chunk_gdn: {} must be on device", name);
}

void check_intermediate(const Tensor& t, const char* name, bool allow_bf16) {
    TT_FATAL(t.layout() == Layout::TILE, "chunk_gdn: {} must be TILE layout", name);
    TT_FATAL(
        t.dtype() == DataType::FLOAT32 || (allow_bf16 && t.dtype() == DataType::BFLOAT16),
        "chunk_gdn: {} must be FLOAT32{}",
        name,
        allow_bf16 ? " or BFLOAT16" : "");
    TT_FATAL(t.buffer() != nullptr, "chunk_gdn: {} must be on device", name);
}
}  // namespace kda_primitive_detail
using namespace kda_primitive_detail;

// ---------------------------------------------------------------------------
// PREP
// ---------------------------------------------------------------------------
KdaChunkPrepOperation::program_factory_t KdaChunkPrepOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KdaChunkPrepProgramFactory{};
}

void KdaChunkPrepOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    check(in.q, "q", DataType::BFLOAT16);
    check(in.k, "k", DataType::BFLOAT16);
    check(in.v, "v", DataType::BFLOAT16);  // flat [B,T,HV*V] when attrs.v_flat; else [BH,NC,C,V]
    if (attrs.v_flat) {
        TT_FATAL(attrs.HV > 0, "v_flat requires HV > 0");
        const auto& vs = in.v.logical_shape();
        TT_FATAL(vs.rank() == 3, "v_flat expects a flat [B,T,HV*V] v (got rank {})", vs.rank());
        TT_FATAL(vs[2] == attrs.HV * attrs.val_dim, "v_flat width {} != HV*V ({}*{})", vs[2], attrs.HV, attrs.val_dim);
    }
    if (attrs.qk_flat) {
        TT_FATAL(attrs.Hk > 0, "qk_flat requires Hk > 0");
        const auto& qsf = in.q.logical_shape();
        TT_FATAL(qsf.rank() == 3, "qk_flat expects a flat [B,T,Hk*K] q (got rank {})", qsf.rank());
        TT_FATAL(
            qsf[2] == attrs.Hk * attrs.key_dim, "qk_flat width {} != Hk*K ({}*{})", qsf[2], attrs.Hk, attrs.key_dim);
        TT_FATAL(attrs.qk_norm, "qk_flat requires qk_norm (flat q/k are unnormalized; norm is in-kernel)");
    }
    if (attrs.vector_gate) {
        check_intermediate(in.g, "g", true);
    } else {
        check(in.g, "g", DataType::FLOAT32);
    }
    if (attrs.g_flat) {
        TT_FATAL(attrs.vector_gate && attrs.HV > 0, "g_flat requires vector_gate and HV > 0");
        const auto& gs = in.g.logical_shape();
        TT_FATAL(gs.rank() == 3, "g_flat expects a flat [B,T,HV*K] g (got rank {})", gs.rank());
        TT_FATAL(gs[2] == attrs.HV * attrs.key_dim, "g_flat width {} != HV*K ({}*{})", gs[2], attrs.HV, attrs.key_dim);
    }
    check(in.beta, "beta", DataType::FLOAT32);
    check(in.eye_c, "eye_c", DataType::FLOAT32);
    check(in.tril_c, "tril_c", DataType::FLOAT32);
    check(in.ones_c, "ones_c", DataType::FLOAT32);
    check(in.masks_c, "masks_c", DataType::FLOAT32);
    TT_FATAL(attrs.chunk_size % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be a multiple of 32");
    constexpr uint32_t allowed_bf16_mask = 0x37;  // v_beta, kd, q_decay, k_dec_t, dl
    TT_FATAL(
        attrs.vector_gate || attrs.output_bf16_mask == 0,
        "BF16 prep intermediates are supported only for vector-gate KDA");
    TT_FATAL(
        (attrs.output_bf16_mask & ~allowed_bf16_mask) == 0,
        "unsupported KDA prep BF16 mask 0x{:x}",
        attrs.output_bf16_mask);
}

KdaChunkPrepOperation::spec_return_value_t KdaChunkPrepOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto spec = [&](const ttnn::Shape& s, uint32_t output_index) {
        const auto dtype = (attrs.output_bf16_mask & (1u << output_index)) ? DataType::BFLOAT16 : DataType::FLOAT32;
        return tt::tt_metal::TensorSpec(s, TensorLayout(dtype, PageConfig(Layout::TILE), attrs.output_mem_config));
    };
    const uint32_t BH = attrs.BH, NC = attrs.num_chunks, C = attrs.chunk_size, K = attrs.key_dim, V = attrs.val_dim;
    return {
        spec(ttnn::Shape({BH, NC, C, V}), 0),                          // v_beta
        spec(ttnn::Shape({BH, NC, C, K}), 1),                          // kd
        spec(ttnn::Shape({BH, NC, C, K}), 2),                          // q_decay
        spec(ttnn::Shape({BH, NC, C, C}), 3),                          // intra
        spec(ttnn::Shape({BH, NC, K, C}), 4),                          // k_dec_t
        spec(ttnn::Shape({BH, NC, attrs.vector_gate ? K : 1, 1}), 5),  // dl
        spec(ttnn::Shape({BH, NC, C, C}), 6),                          // t_inv
    };
}

KdaChunkPrepOperation::tensor_return_value_t KdaChunkPrepOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.q.device();
    std::vector<Tensor> outs;
    outs.reserve(specs.size());
    for (const auto& spec : specs) {
        outs.push_back(create_device_tensor(spec, device));
    }
    return outs;
}

std::vector<Tensor> kda_chunk_prep(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& eye_c,
    const Tensor& tril_c,
    const Tensor& ones_c,
    const Tensor& masks_c,
    uint32_t chunk_size,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool v_flat,
    uint32_t HV,
    bool qk_norm,
    float scale,
    bool qk_flat,
    uint32_t Hk,
    bool g_flat,
    bool vector_gate,
    uint32_t output_bf16_mask) {
    const auto& q_shape = q.logical_shape();  // [BH,NC,C,K] head-major, or flat [B,T,Hk*K] when qk_flat
    const auto& v_shape = v.logical_shape();  // [BH,NC,C,V] head-major, or flat [B,T,HV*V] when v_flat
    // Derive dims. Head-major q gives BH/NC/K directly; flat q [B,T,Hk*K] gives B/T, so BH=B*HV,
    // NC=T/chunk (pad==0 required), K=flat_width/Hk. val_dim = v_shape[3] or v_flat width / HV.
    const uint32_t BH = qk_flat ? (q_shape[0] * HV) : q_shape[0];
    const uint32_t num_chunks = qk_flat ? (q_shape[1] / chunk_size) : q_shape[1];
    const uint32_t key_dim = qk_flat ? (q_shape[2] / Hk) : q_shape[3];
    const uint32_t val_dim = v_flat ? (v_shape[2] / HV) : v_shape[3];
    auto attrs = KdaChunkPrepOperation::operation_attributes_t{
        .BH = BH,
        .num_chunks = num_chunks,
        .chunk_size = chunk_size,
        .key_dim = key_dim,
        .val_dim = val_dim,
        .v_flat = v_flat,
        .HV = HV,
        .qk_flat = qk_flat,
        .Hk = Hk,
        .g_flat = g_flat,
        .qk_norm = qk_norm,
        .scale = scale,
        .vector_gate = vector_gate,
        .output_bf16_mask = output_bf16_mask,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = KdaChunkPrepOperation::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .g = g,
        .beta = beta,
        .eye_c = eye_c,
        .tril_c = tril_c,
        .ones_c = ones_c,
        .masks_c = masks_c};
    return ttnn::device_operation::launch<KdaChunkPrepOperation>(attrs, tensor_args);
}

// ---------------------------------------------------------------------------
// SCAN
// ---------------------------------------------------------------------------
KdaFinalScanOperation::program_factory_t KdaFinalScanOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KdaFinalScanProgramFactory{};
}

void KdaFinalScanOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    check_intermediate(in.v_beta, "v_beta", attrs.vector_gate);
    check_intermediate(in.kd, "kd", attrs.vector_gate);
    check_intermediate(in.q_decay, "q_decay", attrs.vector_gate);
    check(in.intra, "intra", DataType::FLOAT32);
    check_intermediate(in.k_dec_t, "k_dec_t", attrs.vector_gate);
    check_intermediate(in.dl, "dl", attrs.vector_gate);
    check(in.t_inv, "t_inv", DataType::FLOAT32);
    if (in.initial_state.has_value()) {
        check(*in.initial_state, "initial_state", DataType::FLOAT32);
    }
    if (in.identity_tile.has_value()) {
        check(*in.identity_tile, "identity_tile", DataType::FLOAT32);
        TT_FATAL(attrs.key_dim == attrs.val_dim, "identity initial state requires K == V");
    }
    TT_FATAL(
        !attrs.summary_pair || (attrs.state_only && in.identity_tile.has_value()),
        "summary_pair requires state_only and an identity tile");
    TT_FATAL(!attrs.state_only || attrs.summary_pair, "state_only requires summary_pair");
    TT_FATAL(!(attrs.summary_pair && attrs.output_bf16), "summary_pair does not support a BF16 token output");
    TT_FATAL(
        !(in.initial_state.has_value() && in.identity_tile.has_value()),
        "initial_state and identity_tile are mutually exclusive");
    TT_FATAL(attrs.chunk_size % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be a multiple of 32");
}

KdaFinalScanOperation::spec_return_value_t KdaFinalScanOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    // Token output dtype is configurable; recurrent state and internal state accumulation stay FP32.
    const auto output_dtype = attrs.output_bf16 ? DataType::BFLOAT16 : DataType::FLOAT32;
    const auto o_layout = TensorLayout(output_dtype, PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto s_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    ttnn::Shape o_shape =
        attrs.summary_pair
            ? ttnn::Shape({attrs.BH, attrs.key_dim, attrs.val_dim})
            : (attrs.state_only ? ttnn::Shape({1, 1, 32, 32})
                                : ttnn::Shape({attrs.BH, attrs.num_chunks, attrs.chunk_size, attrs.val_dim}));
    ttnn::Shape s_shape({attrs.BH, attrs.key_dim, attrs.val_dim});
    std::vector<tt::tt_metal::TensorSpec> specs{
        tt::tt_metal::TensorSpec(o_shape, o_layout), tt::tt_metal::TensorSpec(s_shape, s_layout)};
    return specs;
}

KdaFinalScanOperation::tensor_return_value_t KdaFinalScanOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.v_beta.device();
    std::vector<Tensor> outs;
    outs.reserve(specs.size());
    for (const auto& spec : specs) {
        outs.push_back(create_device_tensor(spec, device));
    }
    return outs;
}

std::vector<Tensor> kda_final_scan(
    const Tensor& v_beta,
    const Tensor& kd,
    const Tensor& q_decay,
    const Tensor& intra,
    const Tensor& k_dec_t,
    const Tensor& dl,
    const Tensor& t_inv,
    const std::optional<Tensor>& initial_state,
    uint32_t chunk_size,
    bool output_final_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool vector_gate,
    bool state_only,
    const std::optional<Tensor>& identity_tile,
    bool summary_pair,
    bool output_bf16) {
    const auto& vb_shape = v_beta.logical_shape();  // [BH, NC, C, V]
    const auto& kd_shape = kd.logical_shape();      // [BH, NC, C, K]
    auto attrs = KdaFinalScanOperation::operation_attributes_t{
        .BH = vb_shape[0],
        .num_chunks = vb_shape[1],
        .chunk_size = chunk_size,
        .key_dim = kd_shape[3],
        .val_dim = vb_shape[3],
        .has_initial_state = initial_state.has_value(),
        .identity_initial_state = identity_tile.has_value(),
        .output_final_state = output_final_state,
        .state_only = state_only,
        .output_bf16 = output_bf16,
        .summary_pair = summary_pair,
        .vector_gate = vector_gate,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = KdaFinalScanOperation::tensor_args_t{
        .v_beta = v_beta,
        .kd = kd,
        .q_decay = q_decay,
        .intra = intra,
        .k_dec_t = k_dec_t,
        .dl = dl,
        .t_inv = t_inv,
        .initial_state = initial_state,
        .identity_tile = identity_tile};
    return ttnn::device_operation::launch<KdaFinalScanOperation>(attrs, tensor_args);
}

// ---------------------------------------------------------------------------
// KDA GROUPED AFFINE PREFIX
// ---------------------------------------------------------------------------
#if 0  // Superseded by fixed-purpose affine prefix and composition leaves.
KdaAffinePrefixOperation::program_factory_t KdaAffinePrefixOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KdaAffinePrefixProgramFactory{};
}

void KdaAffinePrefixOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    check_intermediate(in.transform_a, "transform_a", true);
    check_intermediate(in.transform_b, "transform_b", true);
    TT_FATAL(in.transform_a.dtype() == in.transform_b.dtype(), "KDA affine summaries must have the same dtype");
    const auto& as = in.transform_a.logical_shape();
    const auto& bs = in.transform_b.logical_shape();
    TT_FATAL(as.rank() == 3 && bs.rank() == 3, "KDA affine prefix expects rank-3 transforms");
    TT_FATAL(attrs.groups_per_head > 0, "groups_per_head must be positive");
    TT_FATAL(as[0] == attrs.BH * attrs.groups_per_head, "transform_a leading dimension mismatch");
    TT_FATAL(bs[0] == as[0], "transform_a/transform_b leading dimensions must match");
    TT_FATAL(as[1] == attrs.key_dim && as[2] == attrs.key_dim, "transform_a must be [BH*G,K,K]");
    TT_FATAL(bs[1] == attrs.key_dim && bs[2] == attrs.val_dim, "transform_b must be [BH*G,K,V]");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be tile aligned");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be tile aligned");
    if (attrs.compose_only) {
        TT_FATAL(!in.initial_state.has_value(), "compose-only affine prefix does not take an initial state");
    } else {
        TT_FATAL(in.initial_state.has_value(), "affine prefix requires an initial state");
        check(*in.initial_state, "initial_state", DataType::FLOAT32);
        const auto& ss = in.initial_state->logical_shape();
        TT_FATAL(ss.rank() == 3, "KDA affine prefix expects a rank-3 initial state");
        TT_FATAL(ss[0] == attrs.BH && ss[1] == attrs.key_dim && ss[2] == attrs.val_dim, "initial_state shape mismatch");
    }
}

KdaAffinePrefixOperation::spec_return_value_t KdaAffinePrefixOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto layout = [&](const Shape& shape) {
        return TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config));
    };
    if (attrs.compose_only) {
        return {
            layout(Shape({attrs.BH, attrs.key_dim, attrs.key_dim})),
            layout(Shape({attrs.BH, attrs.key_dim, attrs.val_dim}))};
    }
    return {layout(Shape({attrs.BH * attrs.groups_per_head, attrs.key_dim, attrs.val_dim}))};
}

KdaAffinePrefixOperation::tensor_return_value_t KdaAffinePrefixOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.transform_a.device()));
    }
    return outputs;
}

namespace kda_primitive_detail {
KdaAffinePrefixParams affine_prefix_params(
    const Tensor& transform_a,
    const Tensor& transform_b,
    uint32_t groups_per_head,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool compose_only) {
    const auto& as = transform_a.logical_shape();
    const auto& bs = transform_b.logical_shape();
    TT_FATAL(groups_per_head > 0 && as[0] % groups_per_head == 0, "invalid affine group count");
    return {
        .BH = static_cast<uint32_t>(as[0]) / groups_per_head,
        .groups_per_head = groups_per_head,
        .key_dim = static_cast<uint32_t>(as[1]),
        .val_dim = static_cast<uint32_t>(bs[2]),
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
        .compose_only = compose_only};
}
}  // namespace kda_primitive_detail
using namespace kda_primitive_detail;

Tensor kda_affine_prefix(
    const Tensor& transform_a,
    const Tensor& transform_b,
    const Tensor& initial_state,
    uint32_t groups_per_head,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    auto results = ttnn::device_operation::launch<KdaAffinePrefixOperation>(
        affine_prefix_params(
            transform_a,
            transform_b,
            groups_per_head,
            output_mem_config,
            compute_kernel_config,
            /*compose_only=*/false),
        KdaAffinePrefixInputs{.transform_a = transform_a, .transform_b = transform_b, .initial_state = initial_state});
    return results[0];
}

std::pair<Tensor, Tensor> kda_affine_compose(
    const Tensor& transform_a,
    const Tensor& transform_b,
    uint32_t groups_per_head,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    auto results = ttnn::device_operation::launch<KdaAffinePrefixOperation>(
        affine_prefix_params(
            transform_a,
            transform_b,
            groups_per_head,
            output_mem_config,
            compute_kernel_config,
            /*compose_only=*/true),
        KdaAffinePrefixInputs{.transform_a = transform_a, .transform_b = transform_b, .initial_state = std::nullopt});
    return {results[0], results[1]};
}

#endif
}  // namespace ttnn::prim
