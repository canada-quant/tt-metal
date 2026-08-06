// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_chunk_preparation.hpp"

#include "device/kda_chunk_preparation_device_operation.hpp"

namespace ttnn::transformer {

std::vector<ttnn::Tensor> kda_chunk_preparation(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& eye,
    const ttnn::Tensor& tril,
    const ttnn::Tensor& ones,
    const ttnn::Tensor& masks,
    uint32_t chunk_size,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    bool v_flat,
    uint32_t value_heads,
    bool normalize_qk,
    float scale,
    bool qk_flat,
    uint32_t key_heads,
    bool gate_flat,
    uint32_t output_bf16_mask) {
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        q.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return ttnn::prim::kda_chunk_preparation(
        q,
        k,
        v,
        g,
        beta,
        eye,
        tril,
        ones,
        masks,
        chunk_size,
        output_memory_config,
        kernel_config,
        v_flat,
        value_heads,
        normalize_qk,
        scale,
        qk_flat,
        key_heads,
        gate_flat,
        output_bf16_mask);
}

}  // namespace ttnn::transformer
