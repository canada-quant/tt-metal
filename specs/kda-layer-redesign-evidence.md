# KDA Layer Redesign — Current-State Evidence

This is supporting evidence for the [review brief](kda-layer-redesign.md). It
records current behavior; it is not an implementation plan.

## Current data flow

```text
hidden states
  → fused input projection
  → Q/K/V causal convolution + convolution carry
  → decay/beta/output gates
  → chunked KDA recurrence + recurrent state
  → gated RMS norm
  → output projection + TP reduce-scatter
```

- **Observed:** this sequence is the current Python `forward` flow
  (`models/experimental/kimi_delta_attention/tt/layer.py:539-561`).
- **Observed:** convolution state is updated before recurrence state is committed
  (`models/experimental/kimi_delta_attention/tt/layer.py:283-332,524-537`).
- **Observed:** the layer requires initialized state and tile-aligned local
  sequence lengths (`models/experimental/kimi_delta_attention/tt/layer.py:256-281`).

## Current ownership and drift
Every current-owner entry below is **Observed**. Its evidence column identifies
the repository source; shortened paths are relative to
`models/experimental/kimi_delta_attention` or the transformer operation leaf.


| Concern | Current owner | Evidence | Design implication |
|---|---|---|---|
| Weights/cache | `KDAWeights` plus `KimiDeltaAttention` | `tt/weights.py:54-289`, `tt/layer.py:88-145` | **Approved:** follow ttMLA’s layer-indexed namespace; model integration owns cache-root lifecycle. |
| State lifecycle | Mutable fields on the layer | `tt/layer.py:155-254,524-537` | **Approved:** expose immutable `KdaState` at the model/layer boundary. |
| SP affine prefix | C++ wrapper composed of existing TTNN ops | `chunk_gated_delta_rule.cpp:802-978` | **Approved:** Python orchestration. |
| Convolution halo | C++ wrapper composed of existing TTNN ops | `chunk_gated_delta_rule.cpp:874-1011` | **Approved:** Python orchestration. |
| KDA chunk graph | C++ wrapper dispatching phased primitives | `chunk_gated_delta_rule.cpp:449-797` | **Approved:** Python orchestration. |
| KDA kernels | Dedicated KDA compute/dataflow kernels | `device/kernels/compute/chunk_kda_prep.cpp`, `chunk_kda_scan.cpp`, `kda_affine_prefix.cpp`, `kda_causal_conv1d.cpp`, `kda_gated_rms.cpp` | **Approved:** retain as KDA C++ device operations. |
| Qwen coupling | Shared GDN operation/factory branches by `vector_gate` | `device/chunk_gdn_phased.hpp:28-190`, `device/chunk_gdn_phased_program_factory.cpp:353-570` | **Approved:** revert this shared-leaf path to its `origin/main` baseline. Relocated KDA code is owned by the dedicated KDA leaf; Qwen tests are not a gate. |

## Why ttMLA is the parent analogue

- **Observed:** ttMLA has static cache completeness/build APIs and receives
  layer index, mesh, axes, topology, and cache path in its construction path
  (`models/demos/deepseek_v3_d_p/tt/mla/mla.py:27-255`).
- **Observed:** it owns weight/configuration setup but composes normal TTNN
  operations from a model-layer `forward` (`mla.py:1085`).
- **Inferred:** KDA should mirror this lifecycle/ownership pattern, not MLA’s
  attention or KV-cache algorithm, because KDA carries recurrent and
  convolution state rather than MLA KV state.

## Operation-placement rule

- **Approved C++:** retain only code that defines a custom device operation,
  kernel ABI, program factory, or bespoke inter-core protocol.
- **Approved Python:** place all normal-TTNN operation graph construction,
  collective sequencing, padding/slicing/reshaping, and state selection in the
  KDA layer package.

## Evidence limits

- **Approved:** `models/demos/deepseek_v3_d_p` is the KDA integration
  package; `ttKDA` follows its established layer cache lifecycle.
- **Approved:** KDA is prefill-only; decode is out of scope.
