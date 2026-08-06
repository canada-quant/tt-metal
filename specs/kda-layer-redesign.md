# KDA Layer Redesign — Review Brief

**Status:** approved design; implementation scope is in the linked development specification.

Supporting evidence: [current-state evidence](kda-layer-redesign-evidence.md).

Approved execution scope: [development specification](kda-development.md).

## 1. Problem and desired outcome

- **Observed:** the PR implements KDA as an experimental standalone layer whose
  Python object owns weights, device execution, and mutable recurrent state
  (`models/experimental/kimi_delta_attention/tt/layer.py:85-561`).
- **Approved:** establish KDA as an embeddable K3 prefill layer with clear
  ownership of weights, recurrent state, TP/SP behavior, and custom kernels.
- **Approved:** a DeepSeek V3 D/P transformer block can construct one KDA layer
  per layer index, provide its state explicitly, and use it without depending
  on Qwen GDN implementation details.

## 2. Verified current-state model

- **Observed:** `KimiDeltaAttention` allocates/resets internal recurrent and
  convolution state, or adopts external buffers for trace replay
  (`models/experimental/kimi_delta_attention/tt/layer.py:190-254`).
- **Observed:** its forward is Python composition around projections, gate
  construction, KDA recurrence, RMS/gate, output projection, and state commit
  (`models/experimental/kimi_delta_attention/tt/layer.py:334-561`).
- **Observed:** KDA’s C++ entry points currently live in Qwen GDN’s
  `chunk_gated_delta_rule` operation leaf and its phased factory selects KDA
  kernels with `vector_gate` (`ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule_nanobind.cpp:74-170`, `device/chunk_gdn_phased_program_factory.cpp:353-570`).
- **Observed:** ttMLA provides the closest in-tree layer lifecycle: layer-indexed
  cache construction, mesh/TP/SP ownership, and a TTNN-composed `forward`
  (`models/demos/deepseek_v3_d_p/tt/mla/mla.py:27-255,1085`).

## 3. Proposed design

- **Approved:** introduce a `ttKDA` layer in the eventual K3 model package; it
  owns per-layer weights and immutable execution configuration.
- **Approved:** model/scheduler owns an immutable `KdaState(recurrent,
  convolution)`; `forward(hidden_states, state)` returns the output and next
  immutable logical state. It never mutates the supplied state.
- **Future goal:** [trace-stable state mutation](kda-trace-stable-state-proposal.md)
  is outside this design. The current `KdaState` contract is immutable.
- **Approved:** keep graph composition in Python: KDA recurrence orchestration,
  SP affine-prefix/halo orchestration, layout changes, and collectives.
- **Approved:** retain C++ only for genuine KDA device operations: chunk prep,
  final scan, affine compose/prefix, fused four-tap convolution, and fused
  gated RMS norm.
- **Approved:** move those custom operations to a dedicated `transformer/kda`
  leaf. Qwen GDN remains in `chunk_gated_delta_rule` with no KDA modes.

## 4. Decisions requiring human approval

1. **Approved:** the owning package is `models/demos/deepseek_v3_d_p`, rather than
   `models/experimental/kimi_delta_attention`. `ttKDA` is a plug-and-play
   layer there.
2. **Approved:** state is explicit and immutable at the layer boundary;
   `forward` returns the next immutable logical state without mutating input.
3. **Future goal:** trace-stable mutation is documented separately and is out
   of scope for this immutable-state design.
4. **Approved:** C++ is restricted to device kernels; orchestration moves to
   Python.
5. **Approved:** `ttKDA` integrates as a plug-and-play layer in
   `models/demos/deepseek_v3_d_p`.
6. **Approved:** This milestone remains prefill-only.

## 5. Alternatives and trade-offs

- **Alternative — retain the current shared GDN factory.** Lowest churn, but
  KDA changes keep altering Qwen’s program-factory contract. Rejected.
- **Alternative — fully separate KDA, including all scheduling helpers.** Best
  ownership clarity; duplicates only a small amount of mechanical code. This is
  acceptable and preferred over a premature shared abstraction.
- **Alternative — introduce a generic shared chunk-recurrence package now.**
  **Observed:** the neutral overlap is limited to compute-config creation and
  basic prep work distribution; scan policy and CB layout differ
  (`device/chunk_gdn_phased_program_factory.cpp:72-178,182-385`). Rejected.

## 6. Invariants and acceptance criteria

- **Approved invariant:** final KDA code has no Qwen/GDN implementation
  dependency. All PR-introduced changes under
  `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/` are
  reverted to their `origin/main` baseline; KDA lives in its dedicated leaf.
  Qwen model code and tests have no final diff and are not a regression gate.
- **Approved invariant:** the base `ttKDA.forward` never mutates its supplied
  `KdaState`.
- **Approved invariant:** KDA remains fully device-resident and trace-safe.
- **Approved invariant:** KDA preserves current recurrent/convolution state
  semantics, PCC, finiteness, determinism, and accepted performance envelopes.
- **Approved acceptance criterion:** every KDA custom kernel implementation
  follows the repository-wide custom-operation structure used by ttMLA’s
  dependencies (for example, `topk`, `indexer`, sparse MLA, and ring MLA).
- **Approved acceptance criterion:** every KDA custom kernel has one KDA-owned
  operation leaf; every orchestration-only flow is Python-visible.
- **Approved acceptance criterion:** KDA follows ttMLA’s model-owned cache
  convention: cache names are layer-indexed (for example,
  `layer_{layer_idx}.kda.*`), while checkpoint selection and cache-root
  lifecycle belong to the model integration. KDA does not add a per-weight
  content hash.
- **Approved invariant:** distinct KDA layer indices do not collide within one
  model-managed cache root. Reusing a cache root across checkpoints follows the
  same caller responsibility as ttMLA.

## 7. Blast radius

- **Observed:** the immediate code surface includes the experimental KDA Python
  package and the shared transformer/GDN C++ leaf
  (`models/experimental/kimi_delta_attention/`, `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/`).
- **Approved:** `models/demos/deepseek_v3_d_p` gains the plug-and-play
  `ttKDA` layer, its cache build/load path, trace-state allocator, KDA tests,
  and transformer build registration.
- **Approved:** Qwen/GDN paths are outside the final blast radius: the final
  diff against `origin/main` is empty for them. KDA validation uses KDA-owned
  tests; the known-broken Qwen suite neither gates nor expands this work.

## 8. Unknowns and assumptions

- **Approved:** `ttKDA` is a plug-and-play layer in
  `models/demos/deepseek_v3_d_p`; its integration follows the established
  layer lifecycle used there.
- **Approved:** KDA is prefill-only. Decode is out of scope for this design.
