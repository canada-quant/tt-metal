# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet (linear attention) for Qwen3.8-Flash-Next — adapted from the qwen36 demo package.

48 V heads / GVA-3 (vs qwen36 32 / GVA-2): config-computed dims (q_dim=k_dim=2048, v_dim=6144),
HF fused conv1d split into q/k/v views, no conv bias. Everything else (fused ops, precompute,
state lifecycle, forward dispatch) is verbatim from the qwen36 package — see
docs/qwen38-flash-next-gdn-adapt.md (tt-rd) for the constraint audit proving 48V GVA-3 passes.
"""

from models.tt_transformers.tt.qwen4_exp.gdn.config import GDNConfig
from models.tt_transformers.tt.qwen4_exp.gdn.gated_deltanet import Qwen4ExpGatedDeltaNet
from models.tt_transformers.tt.qwen4_exp.gdn.weights import remap_flash_next_state_dict

__all__ = ["Qwen4ExpGatedDeltaNet", "GDNConfig", "remap_flash_next_state_dict"]
