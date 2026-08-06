// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_nanobind.hpp"
#include "kda.hpp"

#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

namespace ttnn::operations::transformer {

void bind_kda(nb::module_& mod) {
    ttnn::bind_function<"chunk_kda", "ttnn.transformer.">(
        mod,
        R"doc(
        Chunk-parallel Kimi Delta Attention recurrence with per-key vector decay.

        Rank-4 q/k must be L2-normalized. Rank-3 flat q/k must be raw because the kernel applies
        both L2 normalization and scale. Shapes: q/k/g [B,T,H,K], v [B,T,H,V],
        with rank-3 flat [B,T,H*D] q/k/v/g accepted for tile-aligned sequences;
        beta [B,T,H], initial_state [B,H,K,V]. chunk_size is currently 32.
        summary_group_chunks counts 32-token chunks in each local affine-summary group.
        sequence_parallel_axis enables the all-gather-based cross-rank prefix.
        affine_summary_dtype selects affine transform storage and communication, while recurrent_state_dtype
        selects retained cross-rank and returned recurrent-state storage. grouped_scan_output_dtype selects
        the grouped scan output format;
        the corresponding compute-kernel configs control their prefix and final-scan math.
        use_bf16_prep_intermediates selects the measured BF16 storage for kd, q_decay, and dl.
        At 160 or more local chunks, the grouped affine-prefix path changes reduction order and rounding.
        Returns token-major output [B,T,H,V], or TILE [B*H,T,V] when output_head_major=True,
        and an optional final state.
        )doc",
        &ttnn::transformer::chunk_kda,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("beta").noconvert(),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("output_final_state") = false,
        nb::arg("output_head_major") = false,
        nb::arg("chunk_size") = 32,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("eye") = nb::none(),
        nb::arg("tril") = nb::none(),
        nb::arg("ones") = nb::none(),
        nb::arg("masks") = nb::none(),
        nb::arg("summary_group_chunks") = 8,
        nb::arg("sequence_parallel_axis") = nb::none(),
        nb::arg("affine_summary_dtype") = ttnn::DataType::FLOAT32,
        nb::arg("recurrent_state_dtype") = ttnn::DataType::FLOAT32,
        nb::arg("affine_prefix_compute_kernel_config") = nb::none(),
        nb::arg("grouped_scan_output_dtype") = ttnn::DataType::FLOAT32,
        nb::arg("grouped_scan_compute_kernel_config") = nb::none(),
        nb::arg("use_bf16_prep_intermediates") = false);

    ttnn::bind_function<"_kda_distributed_affine_prefix", "ttnn.transformer.">(
        mod,
        R"doc(
        Compose one affine KDA partition summary per SP rank with a sequential
        rank-by-rank causal prefix. Returns each rank entry state and the global final state
        replicated over the SP mesh axis.
        )doc",
        &ttnn::transformer::kda_distributed_affine_prefix,
        nb::arg("transform_a").noconvert(),
        nb::arg("transform_b").noconvert(),
        nb::arg("initial_state").noconvert(),
        nb::kw_only(),
        nb::arg("sequence_parallel_axis"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("affine_summary_dtype") = ttnn::DataType::FLOAT32,
        nb::arg("recurrent_state_dtype") = ttnn::DataType::FLOAT32);

    ttnn::bind_function<"kda_convolution_halo", "ttnn.transformer.">(
        mod,
        R"doc(
        For projected_qkv [B,T_local,C] and initial_carry [B,history,C], return the
        partition-entry carry and final carry, each [B,history,C], replicated along the SP axis.
        )doc",
        &ttnn::transformer::kda_convolution_halo,
        nb::arg("projected_qkv").noconvert(),
        nb::arg("initial_carry").noconvert(),
        nb::kw_only(),
        nb::arg("sequence_parallel_axis"),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
