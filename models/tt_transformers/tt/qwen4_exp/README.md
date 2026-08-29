# Qwen3.8-Flash-Next (qwen4_exp) bring-up scaffold — QB2 Blackhole

Org-internal scaffold on `canada-quant/tt-metal` branch `qwen4-exp-bringup`
(off Command-R-era tip `a6a14f7f`). **Nothing here goes upstream without explicit
owner approval.** Duplication audit (2026-08-29): zero upstream `qwen4_exp` support
in tenstorrent/tt-metal — this is a from-scratch port, not a re-implementation of
anyone's work.

Docs of record (tt-rd private repo, `main`): `docs/qwen38-flash-next-bringup.md`
(runbook), `docs/qwen38-flash-next-port-scoping.md` (P3 kernel map + BFP4 lock),
`docs/qwen38-flash-next-kernel-contracts.md` (P2(c) — 9 verbatim kernel contracts
from transformers 5.16.1 `modeling_qwen4_exp.py`; **the normative spec for every
kernel in this package**).

## Grounding (verified 2026-08-29 against the on-box snapshot + HF 5.16.1 source)

| Fact | Value | Source |
|---|---|---|
| Checkpoint | `Qwen/Qwen3.8-Flash-Next`, snapshot `de4b8e4d43b917e7706784d8bb445c9af86a3540` — 144/144 files, 336 GiB bf16, 131 shards + index, on QB2 | box HF cache (`DL_DONE 144`, `du 336G`) |
| Params | 176.94B total (meta probe): 125B-A6B MoE + 51.2B n-gram PLE table + ~4B MTP (skipped v1) | tt-rd `results/flash-next/module_map.txt` |
| Text config | `qwen4_exp_text`: hidden 2560, 48 layers, vocab 248320, ctx 262144, bf16, untied embeddings | on-box `config.json` |
| Layer pattern | 36× `linear_attention` (GDN) + 12× `qwen_sparse_attention` at indices 3,7,…,47 (`full_attention_interval=4`) | config `layer_types` verbatim |
| Residual stream | **hyper-connections: 4×2560 = 10240 wide end-to-end** (`hc_count=4`, `hc_lowrank=320`), grouped RMSNorm (group 2560) | contracts §2.7 |
| MoE | 512 experts, top-10, fp32 softmax + L1 renorm (`norm_topk_prob=True`), intermediate 640; shared expert 640 with sigmoid scalar gate | contracts §2.1–2.3 |
| GDN | 16 K-heads×128, 48 V-heads×128 (GVA 3×), conv-4, fp32 SSM state (`mamba_ssm_dtype=float32`), fused `chunk_gated_delta_rule` exists in-tree (upstream BH PCC 0.999995) | contracts §2.6 |
| QSA | indexer 4Q/1KV×128, budget 2048, compress 4 → block_topk 512; ReLU-sum scoring; **NET-NEW kernel** | contracts §2.4 |
| Attention (QSA layers) | q_proj DOUBLED 12288 = q‖sigmoid-gate, per-head q/k RMSNorm(256), GQA 24Q/2KV, head_dim 256, partial rotary 0.25 → 64 dims, **interleaved mrope sections [11,11,10]**, θ=1e7 | contracts §0/§2.5 |
| PLE n-gram | 51.2B table (320001536×160), 8 heads × prime tables, splitmix64 EOS-segment-aware hash, **injects at layer_idx == 1** (config `ple_layer_ids=[2]` is 1-based), signed-sqrt gate, dilated conv k4 d3 (state 9) | contracts §2.8/§2.9 |

## Quantization (owner-locked 2026-08-29): **BFP4 experts**

bf16 (335 GiB) and BFP8 (~137 GB) do NOT fit 128 GB device DRAM; BFP4 experts
(~60 GB) + bf16 dense + n-gram host-offload ≈ 70–71 GB + KV/state — fits.
BFP4 path = `ttnn.sparse_matmul` + gpt_oss `load_expert_weights` bfloat4_b pattern;
our expert act is plain SiLU-gate (NO SwiGLU clip — contracts §2.2).

## Component → source map

| Component | Plan | Existing art in this tree |
|---|---|---|
| GDN ×36 | ADAPT | `models/demos/blackhole/qwen36/tt/gdn/` (config-driven via `GDNConfig.from_args` on `linear_num_*` keys; 48 V heads GVA-3 vs 32 GVA-2 = config delta) + fused ttnn op `chunk_gated_delta_rule` |
| MoE-512 ×48 | ADAPT | `ttnn.sparse_matmul` (BFP4), `deepseek_v3` moe_gate router pattern |
| QSA ×12 | NET-NEW (biggest kernel) | phase 1: full-attention stand-in — exact for ≤2048 visible tokens (n_complete_blocks ≤ 512 ⇒ top-k selects all ⇒ mask ≡ causal); phase 2: real indexer micro-block kernel |
| Hyper-connections | COMPOSED ttnn | matmul 10240→320→10240, sigmoid/silu, grouped RMSNorm, broadcast mul-add |
| PLE + n-gram hash | HOST-OFFLOAD | int64 hash + gather on host, DMA result; bit-exact integer math only |
| mrope interleaved | ADAPT | `tt/rope.py` — must match interleaved [11,11,10] layout (NOT qwen2-vl non-interleaved) |
| MTP / vision | SKIP v1 | — |

## Cache manager contract (P4 must handle 6 state kinds)

1. GDN conv state (k=4) · 2. GDN fp32 recurrent SSM state · 3. PLE dilated conv
state (len 9) · 4. n-gram token history (context_len 2) · 5. indexer KV cache
(raw token_k, no RoPE at store) · 6. standard QSA KV cache.
`number_of_conv_states=3` covers slots 1/3/4.

## PCC references (box `/root/v4run/results/flash-next/p2d/`, 2026-08-29)

Full 48-layer streamed CPU reference (52 files): `emb.pt`, `layer_00.pt` (GDN),
`layer_03.pt` (QSA), `layer_47.pt`, `mixer.pt`, `logits_top10.pt`, `meta.json`,
`capture_log.txt`; harness `scripts/flash-next/capture_p2d.py` (tt-rd main).
Single-layer bring-up targets **GDN layer 0 first** vs `layer_00.pt` (PCC ≥ 0.99
gate, Command-R pattern).
