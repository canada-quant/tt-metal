# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal)
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connections (`Qwen4ExpTextGatedResidual`) for Qwen3.8-Flash-Next.

Mirrors the torch-composed CPU spec `pcc_hc.py` (tt-rd scripts/flash-next/,
PR #303 — box-verified bit-exact vs the hc0.pt capture) op-for-op. Normative
contracts: tt-rd docs/qwen38-flash-next-kernel-contracts.md §2.7 and
docs/qwen38-flash-next-hyper-connections.md §4 (phase-1 replicate-stream plan:
the 10240-wide stream is REPLICATED on all 4 mesh devices — zero collectives).

Composition (verbatim HF order — do not reorder):
  normed = grouped_rms_norm(x)      # (…,10240) per-2560-group fp32 stats, (1+w) scale
  mix_w  = silu(down(normed) / 4)   # bf16 Linear 10240 -> 320
  mix_w  = sigmoid(up(mix_w))       # bf16 Linear 320 -> 10240
  mixed  = (mix_w.unflatten(-1,(4,2560)) * normed.unflatten(-1,(4,2560))).mean(-2)  # (…,2560)
  inj    = 2 * sigmoid(inject(normed) / 4)   # bf16 Linear 10240 -> 4 (blocks only)
Decoder residual (contracts §2.7):
  hidden = x_stream + (block_out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)

The global `hyper_connection_mixer` is the same module WITHOUT block_inject
(use_combine=False): its `mixed` output IS last_hidden / the initial stream.

Host-importable by design: no top-level ttnn/torch imports (the P4 harness
imports this package before devices are up). ttnn resolves lazily in
`load_weights`/`forward`, which REFUSE to run without a live mesh_device.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

HC_COUNT = 4  # config hc_count (contracts §2.7)
HIDDEN = 2560  # config hidden_size
LOWRANK = 320  # config hc_lowrank
EPS = 1e-06  # rms_norm_eps (config.json, verified in capture_hc.py)
MIX_SCALE = 1.0 / HC_COUNT  # HF divides by hc_count BEFORE silu / sigmoid — power of two, bf16-exact

# W1 (tt-rd gdn-fidelity plan §7.2 — fp32 hyper-connection stream carry):
# QWEN4EXP_HC_FP32=1 (default OFF — zero behavior change) carries the 10240-wide
# stream fp32 across the 96 HC boundaries (48 layers × attn/mlp). Branch inputs are
# downcast to bf16 at mix_input entry (HF cast points unchanged: hf normed/mixed/inj
# are bf16 — modeling_qwen4_exp.py L173-177, L952-972) and block outputs are upcast
# to fp32 in apply_residual, so the residual SUM is never re-rounded bf16 per
# boundary. fp32 ops touched: ttnn.add / ttnn.multiply / ttnn.typecast — first-class
# fp32 TILE on this image (binary.cpp:516-530; typecast = the proven qwen36 gdn
# pattern). Root cause: full-chain PCC 0.99333996 deterministic ×4 with the loss
# driven by MoE router near-tie flips on the inherited ~1e-2 stream-noise band
# (Window H); W4/W2 proved the flips are NOT router-local — the noise arrives
# inherited in the stream, and 96 bf16 boundary roundings compound it.
_HC_FP32 = os.environ.get("QWEN4EXP_HC_FP32") == "1"


def _ttnn():
    """Lazy ttnn import — keeps the package importable host-side (no ttnn in the CI container)."""
    import ttnn  # noqa: PLC0415

    return ttnn


@dataclass
class HyperConnectionWeights:
    """Device-resident weights for one HC instance (attn / mlp / global mixer)."""

    norm_plus1: Any  # (1, 1, HC_COUNT, HIDDEN) bf16 — (1 + hc_norm.weight) precomputed host-side
    down: Any  # (HIDDEN*HC_COUNT, LOWRANK) bf16, transposed for ttnn.linear
    up: Any  # (LOWRANK, HIDDEN*HC_COUNT) bf16, transposed for ttnn.linear
    inject: Optional[Any]  # (HIDDEN*HC_COUNT, HC_COUNT) bf16 transposed, or None for the global mixer


class HyperConnection:
    """One `Qwen4ExpTextGatedResidual` (or the global mixer when has_inject=False).

    Phase-1 layout: all weights and the stream are mesh-REPLICATED
    (ReplicateTensorToMesh) — zero collectives, per hyper-connections doc §4.
    """

    def __init__(
        self,
        mesh_device,
        state_dict: dict,
        prefix: str,
        has_inject: bool = True,
        tensor_cache_path=None,
    ):
        ttnn = _ttnn()
        self.device = mesh_device
        self.has_inject = has_inject
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,  # fp32 accumulate — matches the fp32-stats contract
            packer_l1_acc=False,
        )
        self.weights = self._load_weights(ttnn, state_dict, prefix, tensor_cache_path)

    def _to_device(self, ttnn, tensor_torch, shape=None):
        t = ttnn.from_torch(
            tensor_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),  # phase-1: replicate, zero collectives
        )
        if shape is not None:
            t = ttnn.reshape(t, shape)
        return t

    def _load_weights(self, ttnn, state_dict, prefix, tensor_cache_path):
        import torch  # noqa: PLC0415 — host-side weight transforms only

        def sd(name):
            return state_dict[prefix + name]

        norm_w = sd("hc_norm.weight").float()  # (10240,)
        norm_plus1 = (1.0 + norm_w).to(torch.bfloat16).reshape(1, 1, HC_COUNT, HIDDEN)
        down_w = sd("input_mix_weight_down.weight").to(torch.bfloat16).t().contiguous()  # (10240,320)
        up_w = sd("input_mix_weight_up.weight").to(torch.bfloat16).t().contiguous()  # (320,10240)
        inject_w = None
        if self.has_inject:
            key = prefix + "block_inject_weight.weight"
            if key not in state_dict:
                raise KeyError(f"{key} missing but has_inject=True (global mixer must pass has_inject=False)")
            inject_w = sd("block_inject_weight.weight").to(torch.bfloat16).t().contiguous()  # (10240,4)

        return HyperConnectionWeights(
            norm_plus1=self._to_device(ttnn, norm_plus1),
            down=self._to_device(ttnn, down_w),
            up=self._to_device(ttnn, up_w),
            inject=self._to_device(ttnn, inject_w) if inject_w is not None else None,
        )

    def _grouped_rms_norm(self, ttnn, x):
        """Per-2560-group RMSNorm (fp32 stats, eps 1e-6) with zero-centered (1+w) scale.

        x: (…, 10240) -> returns (…, 10240). Mirrors pcc_hc.py `grouped_rms_norm`.
        """
        lead = list(x.shape)[:-1]
        x4 = ttnn.reshape(x, (*lead, HC_COUNT, HIDDEN))
        # ttnn.rms_norm normalizes the last dim per row: groups fall out of the reshape.
        n4 = ttnn.rms_norm(
            x4,
            epsilon=EPS,
            compute_kernel_config=self.compute_kernel_config,
        )
        n4 = ttnn.multiply(n4, self.weights.norm_plus1)  # zero-centered scale, broadcast (1,1,4,2560)
        return ttnn.reshape(n4, (*lead, HC_COUNT * HIDDEN)), n4

    def mix_input(self, x_stream):
        """(…,10240) stream -> (block_input (…,2560), injection (…,4) or None).

        Op order identical to pcc_hc.py `compose_hc` (bit-exact reference order).
        """
        ttnn = _ttnn()
        w = self.weights
        if _HC_FP32 and x_stream.dtype == ttnn.float32:
            # W1: downcast the branch input at the HC boundary — the hf mix path
            # (normed / gates / mixed / inj) is bf16; cast points unchanged.
            x_stream = ttnn.typecast(x_stream, ttnn.bfloat16)
        normed, normed4 = self._grouped_rms_norm(ttnn, x_stream)

        mix_w = ttnn.linear(normed, w.down, compute_kernel_config=self.compute_kernel_config)
        mix_w = ttnn.multiply(mix_w, MIX_SCALE)  # / 4 before silu (HF order; exact power of two)
        mix_w = ttnn.silu(mix_w)
        mix_w = ttnn.linear(mix_w, w.up, compute_kernel_config=self.compute_kernel_config)
        mix_w = ttnn.sigmoid(mix_w)

        lead = list(x_stream.shape)[:-1]
        mix4 = ttnn.reshape(mix_w, (*lead, HC_COUNT, HIDDEN))
        mixed = ttnn.multiply(mix4, normed4)
        mixed = ttnn.mean(mixed, dim=-2)  # (…,2560)

        inj = None
        if self.has_inject:
            inj = ttnn.linear(normed, w.inject, compute_kernel_config=self.compute_kernel_config)
            inj = ttnn.multiply(inj, MIX_SCALE)
            inj = ttnn.sigmoid(inj)
            inj = ttnn.multiply(inj, 2.0)  # 2*sigmoid(·/4) per contracts §2.7
        return mixed, inj

    def apply_residual(self, x_stream, block_out, injection):
        """Decoder residual composition (contracts §2.7):
        hidden = x_stream + (block_out.unsqueeze(-2) * injection.unsqueeze(-1)).flatten(-2)
        x_stream: (…,10240); block_out: (…,2560); injection: (…,4) — all bf16 device tensors.
        """
        ttnn = _ttnn()
        if not self.has_inject or injection is None:
            raise ValueError("apply_residual requires a block HC (has_inject=True) and its injection tensor")
        if _HC_FP32:
            # W1: upcast block output + injection to fp32 and keep the residual SUM
            # fp32 — the stream is never re-rounded bf16 at an HC boundary.
            if x_stream.dtype != ttnn.float32:
                x_stream = ttnn.typecast(x_stream, ttnn.float32)
            block_out = ttnn.typecast(block_out, ttnn.float32)
            injection = ttnn.typecast(injection, ttnn.float32)
        lead = list(x_stream.shape)[:-1]
        b4 = ttnn.unsqueeze(block_out, -2)  # (…,1,2560)
        i4 = ttnn.unsqueeze(injection, -1)  # (…,4,1)
        scaled = ttnn.multiply(b4, i4)  # broadcast (…,4,2560)
        scaled = ttnn.reshape(scaled, (*lead, HC_COUNT * HIDDEN))
        return ttnn.add(x_stream, scaled)

    def forward(self, x_stream, block_fn=None):
        """Full gated-residual: mixed, inj = mix_input(x); out_block = block_fn(mixed);
        return apply_residual(x, out_block, inj). For the global mixer pass block_fn=None
        and use the `mixed` output directly (use_combine=False — it IS the result).
        """
        mixed, inj = self.mix_input(x_stream)
        if block_fn is None:
            return mixed
        return self.apply_residual(x_stream, block_fn(mixed), inj)


class HyperConnectionMixer(HyperConnection):
    """The global `hyper_connection_mixer` — same math, NO block_inject (use_combine=False).

    init stream: repeat(embed, 1, 1, HC_COUNT) — pure broadcast, no weights (host/torch
    side or ttnn.concat of 4 copies; bit-exact either way per capture_hc.py contract c0).
    final: mixed = mix_input(x_stream)[0] -> last_hidden (…,2560).
    """

    def __init__(self, mesh_device, state_dict, prefix, tensor_cache_path=None):
        super().__init__(mesh_device, state_dict, prefix, has_inject=False, tensor_cache_path=tensor_cache_path)

    def forward(self, x_stream):
        mixed, _ = self.mix_input(x_stream)
        return mixed

    def __call__(self, x_stream):
        """nn.Module-style alias: `mixer(x)` == `mixer.forward(x)` (pcc_device_hc.py convention)."""
        return self.forward(x_stream)
