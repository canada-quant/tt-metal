# KDA Development Specification

**Status:** approved design translated into development scope. This document is
an implementation specification; it does not change the approved immutable
state contract in [KDA Layer Redesign](kda-layer-redesign.md).

## Outcome

Deliver a prefill-only `ttKDA` layer that is plug-and-play in
`models/demos/deepseek_v3_d_p`, follows ttMLA’s layer/cache lifecycle, and has
no final diff in Qwen/GDN paths relative to `origin/main`.

The layer is responsible for KDA weights, immutable configuration, and the
TTNN graph. The caller owns logical state.

```python
@dataclass(frozen=True)
class KdaState:
    recurrent: ttnn.Tensor
    convolution: ttnn.Tensor

class ttKDA(LightweightModule):
    @staticmethod
    def check_cache_complete(...) -> bool: ...

    @staticmethod
    def build_ttnn_cache(...) -> None: ...

    def forward(
        self, hidden_states: ttnn.Tensor, state: KdaState
    ) -> tuple[ttnn.Tensor, KdaState]: ...
```

`forward` must not write to either tensor reachable through the supplied
`KdaState`. It returns the logical next state. Decode and trace-stable mutable
state are out of scope.

## Source and destination map

| Concern | Source in the current PR | Destination | Required end state |
|---|---|---|---|
| TT layer, weights, and constant tiles | `models/experimental/kimi_delta_attention/tt/` and `weight_schema.py` | `models/demos/deepseek_v3_d_p/tt/kda/` | A `ttKDA` package using DeepSeek model-layer conventions. |
| CPU reference | `models/experimental/kimi_delta_attention/reference/` | `models/demos/deepseek_v3_d_p/reference/kda/` | Direct KDA CPU reference move patterned after `reference/cpu_deepseek_v32/`. |
| Model configuration | `models/experimental/kimi_delta_attention/config.py` and `kimi_k3_config.py` | `models/demos/deepseek_v3_d_p/reference/` | A named K3 config module following the existing model-config convention (for example, `deepseek_v3_2_config.py` and `kimi_k2_6_config.py`). |
| Checkpoint helpers and model-level KDA tests | `models/experimental/kimi_delta_attention/{checkpoint.py,tests}` | `models/demos/deepseek_v3_d_p/tests/kda/` | Direct move preserves `checkpoint/`, `model/`, `perf/`, and `reference/`; `operations/` retains Python-orchestration tests only. |
| Retained KDA device-operation tests | `models/experimental/kimi_delta_attention/tests/operations/test_convolution.py`, `test_gated_rms_norm.py`, and focused coverage extracted from `test_chunk.py` / `test_distributed_affine.py` | `tests/ttnn/nightly/unit_tests/operations/transformer/kda/` | TTNN API-level correctness, validation, determinism, trace, and hardware coverage for each retained custom op. |
| Chunk recurrence orchestration | `.../chunk_gated_delta_rule.cpp:449-797` | Python in `tt/kda/` | Python composes prep, affine operations, scans, layout work, and collectives. |
| Distributed affine-prefix orchestration | `.../chunk_gated_delta_rule.cpp:802-1017` | Python in `tt/kda/` | Python owns all-gather, slice, concat, reshape, typecast, and matmul sequencing. |
| Convolution halo orchestration | `.../chunk_gated_delta_rule.cpp:874-1011` | Python in `tt/kda/` | Python owns halo preparation and normal TTNN graph construction. |
| Device kernels | KDA-specific files below the GDN leaf | `ttnn/cpp/ttnn/operations/transformer/kda/` | Dedicated KDA operations with their kernels/program factories. |
| GDN/Qwen paths | `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/` and Qwen paths | No destination | Revert every PR-introduced change to `origin/main`; final diff is empty. |

The source locations above are **Observed** in the current PR:
`models/experimental/kimi_delta_attention/tt/layer.py:324-465` invokes the
KDA convolution, recurrence, and RMS operations; the current C++ wrapper
contains recurrence and prefix orchestration at
`ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp:449-1017`.

## Python package and layer lifecycle

Create `models/demos/deepseek_v3_d_p/tt/kda/` with a small public surface:

- `kda.py`: `ttKDA`, `KdaState`, and state allocation/validation helpers.
- `weights.py`: layer-indexed cache conversion/loading only.
- `ops.py`: Python-only recurrence, distributed prefix, and halo orchestration.
- `const_tiles.py`: KDA constant-tile construction, if it remains necessary.
- `__init__.py`: exports only `ttKDA` and `KdaState`.

Match the lifecycle of `ttMLA`:

1. Accept layer index, mesh, SP/TP axes, topology, cache path, and model
   configuration in construction/build APIs.
2. Provide static `check_cache_complete` and `build_ttnn_cache` methods.
3. Name all cache entries `layer_{layer_idx}.kda.<weight-name>`.
4. Treat cache-root selection and checkpoint invalidation as a parent-model
   responsibility, as ttMLA does. Do not add KDA-specific content hashes.
5. Prepare `ttKDA` for later prefill-block wiring: match the block-facing
   construction, cache, and `forward` boundary used by `ttMLA`, but do not add
   a selectable attention path or modify `TtPrefillBlock` in this scope.

Use `TtPrefillBlock` and `ttMLA` as the interface template
(`models/demos/deepseek_v3_d_p/tt/tt_prefill_block.py:49-130`,
`models/demos/deepseek_v3_d_p/tt/mla/mla.py:27-255`). `ttKDA` must accept the
corresponding layer-level construction and cache inputs, including global
`layer_idx`, but this work does not wire it into `TtPrefillBlock`.

## State and forward semantics

`KdaState` is frozen and contains only the recurrent and convolution carries.
It is a logical-immutability boundary: `ttKDA.forward` may use TTNN operations
to produce replacement tensors but must not call `ttnn.copy` into state tensors
or keep state as mutable fields on `ttKDA`.

Define and validate one canonical state representation at the layer boundary:

| Field | Producer | Consumer | Required property |
|---|---|---|---|
| `recurrent` | KDA final scan / distributed affine prefix | next KDA recurrence | device-resident, configured recurrent dtype, correct TP/SP distribution |
| `convolution` | causal-convolution operation | next causal convolution | device-resident BF16 row-major carry with the established four-tap shape |

The forward order is fixed: fused input projection; Q/K/V causal convolution;
gate construction; KDA chunk recurrence; gated RMS norm; output projection and
TP reduce-scatter; construct and return `KdaState(next_recurrent,
next_convolution)`.

State allocation belongs beside `ttKDA`, not the scheduler. The scheduler/model
selects and retains each returned state. `ttKDA` has no reset, slot, or
`use_inplace_state` flag.

## Custom-operation boundary

Create `ttnn/cpp/ttnn/operations/transformer/kda/`. Each genuine device
operation must follow the repository operation structure demonstrated by
`topk`, `indexer_score`, and ttMLA dependencies:

```text
kda/<operation>/
  CMakeLists.txt or sources.cmake
  <operation>.hpp / <operation>.cpp
  <operation>_nanobind.hpp / <operation>_nanobind.cpp
  device/<operation>_device_operation_types.hpp
  device/<operation>_device_operation.hpp / .cpp
  device/<operation>_program_factory.hpp / .cpp
  device/kernels/{compute,dataflow}/...
```

Use a small, separate operation leaf per kernel family rather than a
`vector_gate` mode on a shared operation. The required KDA operations are:

1. chunk preparation;
2. final chunk scan;
3. affine composition;
4. affine prefix;
5. four-tap causal-convolution split; and
6. gated RMS norm.

Each operation validates device/layout/dtype/shape invariants in its device
operation type, declares output specs, uses a program factory for kernel and
CB setup, exposes an explicit nanobind binding, and owns its CMake/source
registration. Share only a demonstrably neutral utility; do not introduce a
new generic GDN/KDA scheduler abstraction.

The public bindings must be KDA-named and owned by the KDA leaf. `chunk_kda`,
`kda_*` bindings, KDA kernels, `vector_gate`, and KDA-specific CB/work-split
logic must not remain in `chunk_gated_delta_rule`.

## Migration sequence

1. Establish the DeepSeek KDA Python package, immutable state type, cache
   naming, the CPU reference at `reference/kda/`, and the K3 configuration
   module under `reference/`. Directly move the model-level KDA suite to
   `models/demos/deepseek_v3_d_p/tests/kda/`. Preserve its hierarchy and
   fixtures, except for retained public device-operation coverage specified
   below.
2. Move the six device-operation implementations into standalone KDA leaves,
   preserving kernel behavior and public KDA bindings under the new leaf.
3. Reimplement the three orchestration flows in Python against those leaves:
   recurrence graph, SP affine prefix, and convolution halo.
4. Implement `ttKDA.forward` and its block-facing construction/cache API so
   later prefill-block wiring can use the `ttMLA` lifecycle. Do not modify
   `TtPrefillBlock` or add an attention-selection path in this scope.
5. Remove KDA implementation and bindings from the GDN leaf, then restore all
   GDN/Qwen paths to the `origin/main` content.
6. Move retained public device-operation coverage to the TTNN operation-test
   tree; keep Python-orchestration coverage in the DeepSeek KDA suite. Remove
   the superseded experimental test location only after both suites pass.

Do not mix performance changes with the extraction. Preserve the current
algorithm, layouts, precision choices, and collectives first; optimize in a
follow-up only with measured evidence.

## Test placement

Apply the following ownership rule; it mirrors the split between sparse MLA
model tests, ring-MLA composition tests, and globally tested indexer operations.

| Existing test | Final home | Required treatment |
|---|---|---|
| `operations/test_convolution.py` | `tests/ttnn/nightly/unit_tests/operations/transformer/kda/` | Move as direct API coverage for the retained causal-convolution op. |
| `operations/test_gated_rms_norm.py` | `tests/ttnn/nightly/unit_tests/operations/transformer/kda/` | Move as direct API coverage for the retained gated-RMS op. |
| `operations/test_chunk.py` | `models/demos/deepseek_v3_d_p/tests/kda/operations/` | Retain as Python recurrence-orchestration coverage because `chunk_kda` is removed. Add focused TTNN tests for chunk prep and final scan. |
| `operations/test_distributed_affine.py` | Split between both trees | Retain distributed all-gather/prefix orchestration coverage in DeepSeek KDA. Add focused TTNN tests for affine compose and affine prefix. |
| `operations/test_halo.py` | `models/demos/deepseek_v3_d_p/tests/kda/operations/` | Retain as Python halo-orchestration coverage; the halo wrapper is removed from C++. |

The TTNN KDA operation tests must cover operation validation, numerical
correctness, determinism, trace behavior where applicable, and Blackhole
hardware execution. The DeepSeek KDA suite owns layer, state, distributed
orchestration, cache, reference, and KDA-layer prefill behavior.

## Verification gates

The change is complete only when all gates below pass for the supported
prefill configurations.

| Gate | Evidence |
|---|---|
| Boundary | `git diff --exit-code origin/main -- models/demos/blackhole/qwen36 ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule` succeeds; use the actual Qwen paths if the final diff identifies more. |
| Operation structure | Each KDA op has self-owned CMake/source registration, API, device operation types, factory, kernels, and nanobind binding; no shared `vector_gate` branch remains. |
| Cache | Cache build then cache-only construction succeeds using `layer_{layer_idx}.kda.*`; different layer indices produce distinct cache names. |
| State purity | Capture input-state tensor identities/contents before `forward`; verify they are unchanged and the returned `KdaState` supplies the next carries. |
| Numerical | KDA-owned unit, reference, distributed, and real-weight tests meet the established PCC thresholds and report no non-finite values. |
| Determinism | Identical inputs and logical state yield bit-identical output and next logical state under the existing determinism test policy. |
| Device residency | No torch conversion, host transfer, or host fallback occurs along `ttKDA.forward`. |
| Trace | The immutable forward path captures/replays correctly without mutable-state semantics. |
| Performance | Existing accepted prefill performance envelope is met; report command, hardware, configuration, median, and tail timing. |

Run the repository-safe test wrapper for targeted tests, then the relevant
hardware workload. Record exact commands, device topology, PCC, and timings in
the implementation PR. The known-broken Qwen suite is not a gate.

## Explicit non-goals

- Decode support.
- Trace-stable in-place state mutation.
- A generic shared GDN/KDA recurrence framework.
- Changes to Qwen source, behavior, tests, or its GDN operation leaf.
- Wiring `ttKDA` into `TtPrefillBlock` or adding an attention-selection path.
- A KDA-specific checkpoint-content cache hash.

## Review checklist

- Does the final diff leave every Qwen/GDN path at `origin/main`?
- Can `ttKDA` be instantiated, cache-built, and forwarded through the same
  model-layer lifecycle as `ttMLA`?
- Is every input `KdaState` unchanged after a forward call?
- Is all normal TTNN graph composition visible in Python?
- Does each KDA custom kernel have an independently structured repository op?
- Are all claims validated on the actual prefill hardware envelope?
