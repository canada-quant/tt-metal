# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Python-owned KDA graph orchestration.

The KDA layer keeps collective and state-flow decisions here; device leaves only
perform the bespoke kernels exposed by ``ttnn.transformer.kda_*``.
"""

from __future__ import annotations

import ttnn


def convolution_halo(*args, **kwargs):
    """Exchange causal-convolution carries along the configured SP axis."""
    return ttnn.transformer.kda_convolution_halo(*args, **kwargs)


def chunk_recurrence(*args, **kwargs):
    """Run KDA's chunk recurrence with explicit input and output state tensors."""
    return ttnn.transformer.chunk_kda(*args, **kwargs)


def distributed_affine_prefix(*args, **kwargs):
    """Compose SP partition affine summaries and return entry/final carries."""
    return ttnn.transformer._kda_distributed_affine_prefix(*args, **kwargs)
