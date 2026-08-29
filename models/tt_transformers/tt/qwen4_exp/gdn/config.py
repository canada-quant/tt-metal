# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Static configuration for the Qwen3.5-9B Gated DeltaNet (linear attention) layer."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GDNConfig:
    num_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    norm_eps: float
    q_dim: int
    k_dim: int
    v_dim: int
    long_prefill_chunk_size: int = 128

    @classmethod
    def from_args(cls, args) -> "GDNConfig":
        return cls(
            num_heads=args.linear_num_key_heads,
            num_v_heads=args.linear_num_value_heads,
            head_k_dim=args.linear_key_head_dim,
            head_v_dim=args.linear_value_head_dim,
            conv_kernel_size=args.linear_conv_kernel_dim,
            norm_eps=args.norm_eps,
            q_dim=args.linear_q_dim,
            k_dim=args.linear_k_dim,
            v_dim=args.linear_v_dim,
        )

    @classmethod
    def from_hf_config(cls, hf_config) -> "GDNConfig":
        """Build from the Qwen3.8-Flash-Next HF text config (qwen4_exp_text).

        Flash-next config.json ships NO linear_q_dim/linear_k_dim/linear_v_dim and no
        norm_eps key (docs/qwen38-flash-next-gdn-adapt.md section 2) — they are COMPUTED:
          q_dim = linear_num_key_heads   * linear_key_head_dim   = 16 * 128 = 2048
          k_dim = linear_num_key_heads   * linear_key_head_dim   = 16 * 128 = 2048
          v_dim = linear_num_value_heads * linear_value_head_dim = 48 * 128 = 6144
          norm_eps = rms_norm_eps (key rename)
        Verbatim flash-next values @ snapshot de4b8e4d: linear_num_key_heads=16,
        linear_num_value_heads=48, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, mamba_ssm_dtype=float32 (beta/g fp32 like qwen36).
        """
        g = hf_config.get if isinstance(hf_config, dict) else lambda k, d=None: getattr(hf_config, k, d)
        nk = int(g("linear_num_key_heads"))
        nv = int(g("linear_num_value_heads"))
        dk = int(g("linear_key_head_dim"))
        dv = int(g("linear_value_head_dim"))
        return cls(
            num_heads=nk,
            num_v_heads=nv,
            head_k_dim=dk,
            head_v_dim=dv,
            conv_kernel_size=int(g("linear_conv_kernel_dim")),
            norm_eps=float(g("rms_norm_eps")),
            q_dim=nk * dk,
            k_dim=nk * dk,
            v_dim=nv * dv,
        )
