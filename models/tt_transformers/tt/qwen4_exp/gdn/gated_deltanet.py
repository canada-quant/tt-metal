# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8-Flash-Next Gated DeltaNet layer — composes config/weights/state/prefill/decode.

Wraps the experimental ``gated_deltanet_forward_ttnn()`` and the on-device GDN prefill
kernel into a module that manages weight tensors, recurrent state, and conv state.
"""
import ttnn
from models.tt_transformers.tt.qwen4_exp.gdn.config import GDNConfig
from models.tt_transformers.tt.qwen4_exp.gdn.decode import recurrent_forward
from models.tt_transformers.tt.qwen4_exp.gdn.state import init_recurrent_state, restore_split_conv_from_fused
from models.tt_transformers.tt.qwen4_exp.gdn.weights import load_gdn_weights


class Qwen4ExpGatedDeltaNet:
    """Gated DeltaNet (linear attention) layer for Qwen3.8-Flash-Next.

    Maintains fixed-size recurrent state [B, H, K, V] that replaces the KV cache.
    Also maintains conv states [B, kernel_size-1, D] for causal conv1d history.
    Supports two modes:
      - "recurrent": single-token decode (T=1), O(1) memory
      - "chunk": multi-token prefill (T>1), chunked parallel processing
    """

    def __init__(self, mesh_device, config: GDNConfig, state_dict, tensor_cache_path=None):
        self.device = mesh_device
        self.cfg = config

        # Mirror config-derived scalar dims so the forward bodies read them directly.
        self.num_heads = config.num_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.norm_eps = config.norm_eps
        self.long_prefill_chunk_size = config.long_prefill_chunk_size

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.weights = load_gdn_weights(mesh_device, config, state_dict, tensor_cache_path)

        # ---- Runtime state (plain instance attributes, exact same names as before;
        # poked directly by the trace machinery in model.py / qwen36_vllm.py) ----
        self.recurrent_state = None
        # Conv states: ttnn tensors on device [B, kernel_size-1, D]
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None
        # Fused conv state [B, kernel_size-1, D_total] where D_total = q_dim + k_dim + v_dim
        self.fused_conv_state = None
        self.split_conv_state = None
        # Trace capture support
        self.use_inplace_state = False
        # When True (set during chunk-outer traced-prefill capture), the chunk (prefill)
        # path writes recurrent + conv state into the persistent external buffers IN PLACE
        # (ttnn.copy) instead of reassigning a fresh tensor, so the state carries across
        # execute_trace() replays (each replay re-runs the same baked buffer addresses).
        # Eager prefill keeps the reassign path. See Qwen36Model.capture_prefill_trace_chunked.
        self._chunk_inplace_state = False

    def __call__(self, x, mode="recurrent", chunk_size=None, valid_len=None):
        """nn.Module-style callable — model.py wires HC block_fns as callables (moe/mixer precedent)."""
        return self.forward(x, mode=mode, chunk_size=chunk_size, valid_len=valid_len)

    def forward(self, x, mode="recurrent", chunk_size=None, valid_len=None):
        # Shape shim (model-leg crash #2, 19:08Z TT_FATAL slice_end > padded_shape):
        # qwen4_exp model.py feeds the HC-mixed stream in the tt_transformers 4D
        # convention (1,B,T,H); the qwen36-derived kernel is strictly 3D (B,T,H) —
        # T = x.shape[1] and the q/k/v split slices dim 2, so a 4D input makes
        # dim 2 the TOKEN axis. Drop a leading unit dim here and restore it on
        # the way out; 3D inputs (the gdn leg contract) pass through untouched.
        squeeze = len(x.shape) == 4 and x.shape[0] == 1
        if squeeze:
            x = ttnn.reshape(x, (x.shape[1], x.shape[2], x.shape[3]))
        out = recurrent_forward(self, x, mode=mode, chunk_size=chunk_size, valid_len=valid_len)
        if squeeze:
            out = ttnn.reshape(out, (1, out.shape[0], out.shape[1], out.shape[2]))
        return out

    def set_external_state(self, recurrent_state, conv_state):
        """Point layer at externally-allocated state buffers.
        Sets use_inplace_state=True so all forward passes write state inplace (preserving buffer addresses).
        Does NOT create split_conv_state — that happens after prefill when there is real data to split.
        """
        expected_rec = [1, self.num_v_heads, self.head_k_dim, self.head_v_dim]
        assert (
            list(recurrent_state.shape) == expected_rec
        ), f"recurrent_state shape mismatch: {list(recurrent_state.shape)} != {expected_rec}"
        assert (
            conv_state.shape[1] == self.conv_kernel_size - 1
        ), f"conv_state dim 1 mismatch: {conv_state.shape[1]} != {self.conv_kernel_size - 1}"
        self.recurrent_state = recurrent_state
        self.fused_conv_state = conv_state
        self.use_inplace_state = True

    def _restore_split_conv_from_fused(self):
        """Copy fused_conv_state slices into existing split_conv_state buffers.
        Preserves device addresses (critical for trace replay).
        Kept as a method because model.py calls it on the instance.
        """
        restore_split_conv_from_fused(self)

    def reset_state(self, batch_size=None):
        if batch_size is not None:
            init_recurrent_state(self, batch_size)
        else:
            self.recurrent_state = None
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None
        self.fused_conv_state = None
        self.split_conv_state = None
