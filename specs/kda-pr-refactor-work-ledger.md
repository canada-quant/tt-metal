# KDA PR Refactor Work Ledger

Purpose: concise, timestamped execution record for the KDA refactor of PR #51910.

Legend: 🧭 plan · 🛠️ in progress · ✅ verified · ⚠️ issue · ⛔ blocked · 💡 decision

## Current status

- **2026-08-06 13:20 UTC** 🧭 Initialized: refactor work begins on `momcilo/kda-pr-refactor`, tracking `origin/momcilo/feature/kda-pr`. Development specification, redesign specification, and supporting evidence are available in `specs/`.

## Work entries

_Add succinct timestamped entries here for progress, problems, failures, resolutions, and decisions._
- **2026-08-06 13:31 UTC** 🛠️ Migrated tracked KDA Python, CPU reference, K3 configuration, and test hierarchy into `models/demos/deepseek_v3_d_p`; direct TTNN convolution and gated-RMS coverage moved to the KDA operation-test tree.
- **2026-08-06 13:31 UTC** 💡 Replaced layer-owned mutable carries with frozen `KdaState`, explicit allocation/validation, and `ttKDA.forward(hidden_states, state) -> (output, next_state)`. Cache keys now use only `layer_{layer_idx}.kda.<weight-name>`; cache-root/invalidation remains caller-owned.
- **2026-08-06 13:31 UTC** ⚠️ Native KDA kernels and legacy orchestration still reside in the GDN leaf; this is the next extraction concern, so boundary validation is intentionally not yet expected to pass.
- **2026-08-06 13:46 UTC** ✅ Restored `chunk_gated_delta_rule/` byte-for-byte to `origin/main` and registered a KDA-owned native leaf. The prescribed `./build_metal.sh` passes and installs the worktree Python bindings.
- **2026-08-06 13:46 UTC** ⚠️ The KDA leaf is mechanically isolated but still uses the transitional aggregated `chunk_kda` front-end and phased implementation. It must be split into the six specified KDA operation leaves and Python graph orchestration before this concern is complete.
- **2026-08-06 13:54 UTC** ✅ Repaired KDA kernel relocation: runtime paths now target the KDA leaf, the required final-scan writer is present with the seven-argument KDA accessor ABI, and existing vector-gate KDA readers/compute kernels are selected. `./build_metal.sh` passes; `scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/model/test_layer.py -q -s` reports `9 passed` and `SAFE_PYTEST_RESULT: PASS` (first-case PCC: output 0.999961, recurrent 0.999900, convolution 0.999997).
- **2026-08-06 14:00 UTC** ⚠️ The operation-level `T=5120,H=2,K=V=32,summary_group_chunks=8` case reproduces a device-dispatch timeout twice (standard and `--dev` watcher runs), while shorter cases and the composed-layer suite pass. The watcher captured a 44.480s dump. CPU oracle phase logging now flushes cache state and elapsed time before device work; the existing T=5120 real-weight cache path also flushes hit/miss and elapsed time.
- **2026-08-06 14:05 UTC** ✅ Diagnosed and repaired the T=5120 hang: summary scan produces one `[K,V]` pair, while the copied final writer incorrectly awaited `NC` token slabs. The KDA writer now drains that single summary pair. The exact hardware case passes under the unchanged 5s dispatch guard (`CPU=2.237s`; test body `3.25s`; output PCC `0.999962`; state PCC `0.999966`; `SAFE_PYTEST_RESULT: PASS`).
- **2026-08-06 14:08 UTC** ✅ Migrated remaining model, distributed, real-weight, and perf callers to explicit `KdaState` input/output. Updated the moved KDA test instructions. Static compilation and the composed Blackhole suite pass (`9 passed`, `SAFE_PYTEST_RESULT: PASS`); first composed case PCC: output `0.999961`, recurrent `0.999900`, convolution `0.999997`.
- **2026-08-06 14:08 UTC** 💡 Split design confirmed from source: the current KDA leaf remains a transitional monolith (`chunk_kda`, shared `vector_gate` modes, and C++ collective/layout orchestration). The next concern is a six-leaf native extraction—prep, final scan, affine compose, affine prefix, causal convolution, gated RMS—then replace the monolithic calls with Python-owned recurrence, prefix, and halo graphs. No performance tuning will be folded into that extraction.
