# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal)
# SPDX-License-Identifier: Apache-2.0
"""Decoder composition for Qwen3.8-Flash-Next (qwen4_exp) — the full 48-layer text model.

Composes the five P4 device modules per docs/qwen38-flash-next-kernel-contracts.md §2.10
(verbatim order — the normative spec, tt-rd main):

    stream = repeat(embed_tokens(input_ids), 1, 1, hc_count)     # (…,10240) init (capture c0)
    for each of 48 decoder layers:
        if layer_idx == ple_inject_layer_idx:                     # layer 1 only
            stream = stream + PLE(ple_emb, stream)                # n-gram injection (residual)
        stream = attn_hyper_connection(stream, GDN | QSA)         # gated residual §2.7
        stream = mlp_hyper_connection(stream, MoE)                # gated residual §2.7
    last_hidden = global_hyper_connection_mixer(stream)          # (…,2560) use_combine=False
    logits = lm_head(rms_norm(last_hidden))                      # final norm + head

Phase-1 layout (per hyper-connections doc §4): every weight + the stream are
mesh-REPLICATED (ReplicateTensorToMesh) — zero collectives. Embedding + final
norm + lm_head run HOST-side (torch) — the same host-offload philosophy as the
102.4 GiB PLE n-gram table (ple-host doc): only the per-layer device compute is
on the mesh for the bring-up. Tensor-parallel sharding is a later leg.

Import-safety: this module is HOST-importable (stdlib + dataclass only at module
level). Every ttnn-touching sub-module (gdn / moe / attention_qsa import ttnn at
their own top level; hyper_connections / ple are lazy internally) is imported
LAZILY inside the constructors, so `import model` never requires a device.

Device execution is OWNER-GATED: see scripts/flash-next/pcc_device_model.py, which
REFUSEs to open a device without QWEN4EXP_DEVICE_OK=1 (tt-commandr preemption decision).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# --- stdlib-only module constants (host-importable) ---
# On-disk safetensors prefix, verified verbatim against the checkpoint weight_map
# (snapshot de4b8e4d, 1658 keys): the text tower lives under "model.language_model."
# (1293 keys); "model.visual." (333) and "mtp." towers are skipped in v1; "lm_head.weight"
# is TOP-LEVEL (no prefix). NOTE: the transformers meta-device probe (module_map.txt)
# reports the text model's INTERNAL naming "model.*" — that is NOT the on-disk key prefix.
CKPT_PREFIX = "model.language_model."
LM_HEAD_KEY = "lm_head.weight"  # top-level, NOT under CKPT_PREFIX
HIDDEN = 2560  # config hidden_size
HC_COUNT = 4  # config hc_count
HCW = HIDDEN * HC_COUNT  # 10240 hyper-stream width
RMS_NORM_EPS = 1e-6  # config.json rms_norm_eps (verified in capture_hc.py; hc/ple EPS constant)


def _ttnn():
    """Lazy ttnn import — keeps `import model` host-side safe."""
    import ttnn  # noqa: PLC0415

    return ttnn


def _replicated_mesh_to_host(ttnn, mesh_device, t):
    """0.19.0-era fix: plain ttnn.to_torch on a replicate-mapped mesh tensor is TT_FATAL
    (pytensor.cpp buffers.size()==1 — a mesh composer is required). Concat along dim 0
    stacks the 4 identical replicas; take the first."""
    if mesh_device is None:
        return ttnn.to_torch(t)
    cat = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return cat[: t.shape[0]]


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    """Return the sub-dict with ``prefix`` removed from each key."""
    n = len(prefix)
    return {k[n:]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _gdn_config_dict(args) -> dict:
    """Build the plain dict ``GDNConfig.from_hf_config`` consumes (it accepts a dict).

    qwen4_exp config.json ships no linear_{q,k,v}_dim and no norm_eps — from_hf_config
    computes q_dim=k_dim=2048, v_dim=6144 and reads rms_norm_eps (key rename). The args
    dataclass deliberately omits rms_norm_eps, so it is supplied here from the verified
    config.json value (RMS_NORM_EPS).
    """
    return {
        "linear_num_key_heads": args.linear_num_key_heads,
        "linear_num_value_heads": args.linear_num_value_heads,
        "linear_key_head_dim": args.linear_key_head_dim,
        "linear_value_head_dim": args.linear_value_head_dim,
        "linear_conv_kernel_dim": args.linear_conv_kernel_dim,
        "rms_norm_eps": RMS_NORM_EPS,
    }


class Qwen4ExpDecoderLayer:
    """One of 48 decoder layers: attn-HC + (GDN | QSA) + mlp-HC + MoE (+ PLE at idx 1).

    Each gated-residual sub-block follows contracts §2.10 exactly::

        stream = attn_hc.forward(stream, attn_block)   # mix_input -> block -> apply_residual
        stream = mlp_hc.forward(stream, moe_block)

    where ``attn_hc.forward(stream, fn)`` = ``apply_residual(stream, fn(mix_input(stream)[0]), inj)``
    (the hyper-connections.py gated-residual composition, bit-exact vs pcc_hc.py).
    """

    def __init__(self, mesh_device, args, state_dict: dict, layer_idx: int, tensor_cache_path=None):
        from models.tt_transformers.tt.qwen4_exp.hyper_connections import (  # noqa: PLC0415
            HyperConnection,
        )

        self.device = mesh_device
        self.args = args
        self.layer_idx = layer_idx
        self.is_qsa = layer_idx in set(args.qsa_layer_indices)
        self.is_ple = layer_idx == args.ple_inject_layer_idx

        lp = f"{CKPT_PREFIX}layers.{layer_idx}."

        # --- gated residuals (attn + mlp), has_inject=True (block_inject_weight present) ---
        self.attn_hc = HyperConnection(
            mesh_device, state_dict, lp + "attn_hyper_connection.", has_inject=True,
            tensor_cache_path=tensor_cache_path,
        )
        self.mlp_hc = HyperConnection(
            mesh_device, state_dict, lp + "mlp_hyper_connection.", has_inject=True,
            tensor_cache_path=tensor_cache_path,
        )

        # --- attention block: QSA full-attn stand-in (phase-1) or GDN linear attention ---
        if self.is_qsa:
            from models.tt_transformers.tt.qwen4_exp.attention_qsa import (  # noqa: PLC0415
                Qwen4ExpQSAAttention,
            )

            # remap_qsa_state_dict strips {ckpt}layers.{i}.self_attn. internally by layer_idx.
            self.block = Qwen4ExpQSAAttention(
                mesh_device, args, state_dict, layer_idx, tensor_cache_path=tensor_cache_path
            )
        else:
            from models.tt_transformers.tt.qwen4_exp.gdn import (  # noqa: PLC0415
                GDNConfig,
                Qwen4ExpGatedDeltaNet,
            )

            gdn_cfg = GDNConfig.from_hf_config(_gdn_config_dict(args))
            gdn_sd = _strip_prefix(state_dict, lp + "linear_attn.")
            self.block = Qwen4ExpGatedDeltaNet(
                mesh_device, gdn_cfg, gdn_sd, tensor_cache_path=tensor_cache_path
            )

        # --- MoE-512 BFP4 mlp block (every layer) ---
        from models.tt_transformers.tt.qwen4_exp.moe import (  # noqa: PLC0415
            Qwen4ExpMoE,
            Qwen4ExpMoEConfig,
        )

        moe_cfg = Qwen4ExpMoEConfig.from_args(args)
        moe_sd = _strip_prefix(state_dict, lp + "mlp.")
        self.moe = Qwen4ExpMoE(mesh_device, moe_cfg, moe_sd, tensor_cache_path=tensor_cache_path)

        # --- PLE n-gram injection (layer 1 only) ---
        self.ple = None
        if self.is_ple:
            from models.tt_transformers.tt.qwen4_exp.ple import PleLayer  # noqa: PLC0415

            self.ple = PleLayer(mesh_device, state_dict, lp + "ple.", tensor_cache_path=tensor_cache_path)

    def forward(
        self,
        stream,
        ple_emb=None,
        cos=None,
        sin=None,
        gdn_mode: str = "recurrent",
        gdn_chunk_size: Optional[int] = None,
        gdn_valid_len: Optional[int] = None,
    ):
        """Advance the 10240 hyper stream through this decoder layer (contracts §2.10).

        Args:
            stream: (…, 10240) bf16 device tensor (mesh-replicated).
            ple_emb: (…, 2560) host-gathered n-gram rows for THIS layer (layer 1 only);
                ignored on all other layers. Staged to device by the caller.
            cos / sin: duplicate-half partial-rope tables for QSA layers (ignored on GDN).
            gdn_mode / gdn_chunk_size / gdn_valid_len: GDN forward controls (ignored on QSA).

        Returns:
            (…, 10240) device tensor — the stream after PLE + attn-HC + mlp-HC.
        """
        ttnn = _ttnn()

        # 1. PLE residual injection (layer 1 only) — BEFORE the attn gated residual.
        if self.ple is not None and ple_emb is not None:
            stream = ttnn.add(stream, self.ple.forward(ple_emb, stream))

        # 2. attn gated residual: mix_input -> (GDN|QSA)(mixed) -> apply_residual.
        if self.is_qsa:
            stream = self.attn_hc.forward(stream, lambda mixed: self.block(mixed, cos, sin))
        else:
            stream = self.attn_hc.forward(
                stream,
                lambda mixed: self.block(
                    mixed, mode=gdn_mode, chunk_size=gdn_chunk_size, valid_len=gdn_valid_len
                ),
            )

        # 3. mlp gated residual: mix_input -> MoE(mixed) -> apply_residual.
        stream = self.mlp_hc.forward(stream, self.moe)
        return stream


class Qwen4ExpModel:
    """The full 48-layer qwen4_exp text model — composition root (contracts §2.10).

    Owns: host-side embed + final norm + lm_head, the global hyper-connection mixer
    (use_combine=False), the 48 decoder layers, and the layer-1 PLE block. The PLE
    n-gram row-gather itself (102.4 GiB mmap table) is produced by the caller via
    ``PleHostGather`` and passed in per-forward as ``ple_emb`` — model.py stays
    decoupled from the transformers/safetensors host dependency.
    """

    def __init__(self, mesh_device, args, state_dict: dict, tensor_cache_path=None):
        from models.tt_transformers.tt.qwen4_exp.hyper_connections import (  # noqa: PLC0415
            HyperConnectionMixer,
        )

        self.device = mesh_device
        self.args = args

        # Global mixer (use_combine=False — NO block_inject; maps the final stream -> last_hidden).
        self.mixer = HyperConnectionMixer(
            mesh_device, state_dict, CKPT_PREFIX + "hyper_connection_mixer.",
            tensor_cache_path=tensor_cache_path,
        )

        # 48 decoder layers (36 GDN + 12 QSA alternating; PLE folded into layer 1).
        self.layers = [
            Qwen4ExpDecoderLayer(mesh_device, args, state_dict, i, tensor_cache_path=tensor_cache_path)
            for i in range(args.num_hidden_layers)
        ]

        # Host-side weight refs (never staged whole to device in phase-1).
        self._embed_weight = state_dict[CKPT_PREFIX + "embed_tokens.weight"]  # (vocab, 2560)
        # On-disk truth (weight_map, snapshot de4b8e4d): there is NO
        # "model.language_model.norm.weight" — the text model ships no learnable final-norm
        # weight. The global mixer's zero-centered hc_norm already normalizes the stream;
        # last_hidden is therefore the mixer output directly (see final_norm).
        self._final_norm_weight = state_dict.get(CKPT_PREFIX + "norm.weight")  # None on-disk
        self._lm_head_weight = state_dict.get(LM_HEAD_KEY)  # top-level "lm_head.weight"

    # ----- host-side front / back stages (torch, host-offload philosophy) -----

    def embed_to_stream(self, input_ids):
        """(B,S) int64 -> (B,S,10240) host torch stream = repeat(embed,1,1,hc_count).

        Bit-exact vs capture c0 (stream_init_repeat_bitwise=True): the init stream is a
        pure broadcast repeat of the 2560 embedding output across the 4 hyper slots.
        """
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        emb = F.embedding(input_ids, self._embed_weight)  # (B,S,2560)
        return emb.repeat(1, 1, HC_COUNT)  # (B,S,10240)

    def stream_to_host(self, stream_device):
        """Bring the device stream back to host torch (…,10240)."""
        ttnn = _ttnn()
        return _replicated_mesh_to_host(ttnn, self.device, stream_device)

    def final_norm(self, last_hidden):
        """The final pre-logits normalization.

        On-disk truth (weight_map, snapshot de4b8e4d): there is NO
        ``model.language_model.norm.weight`` — the text model ships no learnable
        final-norm weight. The global mixer's zero-centered hc_norm already
        normalizes the stream (contracts §2.7), so the mixer output IS the final
        hidden state and this is the IDENTITY. A zero-centered RMSNorm branch is
        kept only defensively (should a future checkpoint add the weight); it is
        gated in the wiring leg against p2d/mixer.pt + logits_top10.pt.
        """
        if self._final_norm_weight is None:
            return last_hidden  # identity — mixer hc_norm is the final normalization
        import torch  # noqa: PLC0415

        x = last_hidden.float()
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(ms + RMS_NORM_EPS)
        x = x * (1.0 + self._final_norm_weight.float())
        return x.to(last_hidden.dtype)

    def logits(self, last_hidden):
        """Host final norm + lm_head matmul -> (…, vocab) float logits."""
        import torch  # noqa: PLC0415

        w = self._lm_head_weight if self._lm_head_weight is not None else self._embed_weight
        normed = self.final_norm(last_hidden).float()
        return normed @ w.float().t()

    # ----- the composition -----

    def forward(
        self,
        input_ids=None,
        stream_device=None,
        ple_emb=None,
        rope=None,
        gdn_mode: str = "recurrent",
        gdn_chunk_size: Optional[int] = None,
        gdn_valid_len: Optional[int] = None,
        mesh_mapper=None,
        stream_trace=None,
    ):
        """Run the full 48-layer composition.

        Args:
            input_ids: (B,S) int64 host torch — used with embed_to_stream when
                ``stream_device`` is not supplied.
            stream_device: optional pre-staged (B,S,10240) device stream (skips embed).
            ple_emb: optional pre-gathered (B,S,2560) device n-gram rows for layer 1.
            rope: optional (cos, sin) duplicate-half tables shared by the QSA layers.
            gdn_mode / gdn_chunk_size / gdn_valid_len: GDN forward controls.
            mesh_mapper: optional ttnn mapper for staging the stream (default Replicate).

        Returns:
            (B,S,2560) host torch last_hidden (post global mixer, pre final-norm).
        """
        ttnn = _ttnn()

        if stream_device is None:
            if input_ids is None:
                raise ValueError("forward needs input_ids or a pre-staged stream_device")
            stream_host = self.embed_to_stream(input_ids)  # (B,S,10240) host torch
            if mesh_mapper is None:
                mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
            stream = ttnn.from_torch(
                stream_host.to(self._torch_bf16()),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=mesh_mapper,
            )
        else:
            stream = stream_device

        cos = sin = None
        if rope is not None:
            cos, sin = rope

        if stream_trace is not None:
            stream_trace.append(_replicated_mesh_to_host(ttnn, self.device, stream))
        for layer in self.layers:
            stream = layer.forward(
                stream,
                ple_emb=ple_emb if layer.is_ple else None,
                cos=cos,
                sin=sin,
                gdn_mode=gdn_mode,
                gdn_chunk_size=gdn_chunk_size,
                gdn_valid_len=gdn_valid_len,
            )
            if stream_trace is not None:
                stream_trace.append(_replicated_mesh_to_host(ttnn, self.device, stream))

        last_hidden_device = self.mixer.forward(stream)  # (B,S,2560) use_combine=False
        return _replicated_mesh_to_host(ttnn, self.device, last_hidden_device)

    @staticmethod
    def _torch_bf16():
        import torch  # noqa: PLC0415

        return torch.bfloat16
