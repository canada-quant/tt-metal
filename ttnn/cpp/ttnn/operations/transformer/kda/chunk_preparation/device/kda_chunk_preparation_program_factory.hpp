// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kda_chunk_preparation_device_operation_types.hpp"

namespace ttnn::prim {

struct KdaChunkPreparationProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KdaChunkPreparationParams&, const KdaChunkPreparationInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::prim
