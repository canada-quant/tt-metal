# GLM-5.2 Sparse MLA local performance summary

Measured on the local Blackhole LoudBox (SP=2, TP=4), using the explicit `-m perf` GLM-5.2 Sparse MLA matrix on 2026-08-07. Both revisions completed all nine cases.

| Mode / scenario | `origin/main` total | Branch total | Change |
| --- | ---: | ---: | ---: |
| Sparse BF16 / warm | 6.498 ms | 5.963 ms | **8.2% faster** |
| Sparse BF16 / cold | 63.021 ms | 59.587 ms | **5.4% faster** |
| Sparse BF16 / long | 15.716 ms | 10.793 ms | **31.3% faster** |
| Sparse scaled-FP8 / warm | 6.169 ms | 5.705 ms | **7.5% faster** |
| Sparse scaled-FP8 / cold | 60.801 ms | 57.732 ms | **5.0% faster** |
| Sparse scaled-FP8 / long | 15.096 ms | 10.504 ms | **30.4% faster** |
| Dense / warm | 3.674 ms | 3.689 ms | 0.1% slower (noise-scale) |
| Dense / cold | 31.431 ms | 31.045 ms | 1.2% faster |
| Dense / long | 19.718 ms | 19.719 ms | unchanged |

Totals are the sum of per-op real-time-profiler durations in the per-scenario CSVs. Sparse MLA is faster in every measured scenario; the unchanged dense path remains effectively neutral.

Validation completed locally:

- `./build_metal.sh --release`
- `scripts/run_safe_pytest.sh -q -x models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py` — 32 passed in 679.3 s
- `scripts/run_safe_pytest.sh -q -x models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py -k glm_5_2` — 11 passed
- `scripts/run_safe_pytest.sh -q -x models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py -k 'rotated and maxedge and deepseek_v32'` — 1 passed
- `scripts/run_safe_pytest.sh -q -x tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py -k selected_batch_prefix` — 1 passed
- `scripts/run_safe_pytest.sh -q -m perf models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py -k glm_5_2` — 9 passed on both branch and `origin/main`
