# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal)
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next (qwen4_exp) TTNN port — org-internal (canada-quant/tt-metal).

Modules are imported lazily by models.tt_transformers.tt.model.Transformer when
args.model_type == "qwen4_exp" (family dispatch in Transformer.__init__), mirroring
the tt/cohere package pattern. Nothing in this package goes upstream without
explicit owner approval.

P4 bring-up order (see README.md):
  1. config.py        — HF config plumbing (this package, DONE)
  2. GDN layer 0      — single-layer PCC vs box CPU reference (layer_00.pt)
  3. MoE-512 BFP4     — ttnn.sparse_matmul grouped GEMM
  4. QSA phase 1      — full-attention stand-in (exact for <=2048 visible tokens)
  5. Hyper-connections / PLE host-offload / n-gram hash / mrope interleaved
"""
