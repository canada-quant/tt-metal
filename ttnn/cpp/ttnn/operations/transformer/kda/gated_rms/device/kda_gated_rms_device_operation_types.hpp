// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct KdaGatedRmsParams {
    uint32_t batch;
    uint32_t num_heads;
    uint32_t sequence;
    uint32_t value_dim;
    float epsilon;
    tt::tt_metal::MemoryConfig output_mem_config;
    DataType output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct KdaGatedRmsInputs {
    Tensor input;
    Tensor gate;
    Tensor weight;
};

struct KdaGatedRmsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KdaGatedRmsParams&, const KdaGatedRmsInputs&, std::vector<Tensor>&);
};

struct KdaGatedRmsOperation {
    using operation_attributes_t = KdaGatedRmsParams;
    using tensor_args_t = KdaGatedRmsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<KdaGatedRmsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor kda_gated_rms_norm(
    const Tensor& input,
    const Tensor& gate,
    const Tensor& weight,
    uint32_t num_heads,
    float epsilon,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    DataType output_dtype);

}  // namespace ttnn::prim
