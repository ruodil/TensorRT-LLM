# DeepSeek-V4 module-perf microbenchmarks

Per-module perf microbenchmarks for DeepSeek-V4, modeled on `../attention_perf/`.
Each case times a real `torch.ops.trtllm.*` kernel op in isolation and gates it
against a per-`(case_id, GPU)` golden.

| Module | case_id | op timed | kernel family |
|--------|---------|----------|---------------|
| mHC (HyperConnection) | `mhc_ctx_gemm_time` | `mhc_gemm_sqrsum_fma` | `mhcGemmSqrsumFmaKernel` |
| MLA glue (q_norm)     | `mla_ctx_qnorm_time` | `deepseek_v4_q_norm` | `deepseekV4QNormKernel` |
| MoE EP-dispatch       | `moe_ctx_dispatch_time` | `moe_a2a_dispatch` | `moeA2ADispatchKernel` |

The modules/kernels and expected families (`expected_signatures_v4.json`) are
derived from the real DeepSeek-V4 CTX layerwise nsys trace (which kernels each
module runs, and which sit on the wall-clock critical path). These are **continuous
gpu_time** isolated-op gates. Model-level kernel *launch counts* (e.g. 7119 over a
full layer-slice) are not isolated-op signals and are validated against the
layerwise trace separately, not here.

Registered in `../qa/module_test_list.txt` (the QA perf-pipeline catalog, not
pre-merge L0).

## Run

```bash
# no-GPU logic + signature checks (fast, any box):
pytest tests/microbenchmarks/v4_module_perf/test_v4_harness_logic.py \
       tests/microbenchmarks/v4_module_perf/ -k "families or golden or logic or csv or gate" -v

# GPU timing cases (need CUDA + the TRT-LLM v4 build; skip cleanly otherwise):
pytest tests/microbenchmarks/v4_module_perf/ -m continuous -v
```

GPU cases skip only when a genuine prerequisite is missing (no CUDA, no
`tensorrt_llm`, or the op is not registered). The representative op arg shapes in
each test are placeholders to calibrate on the target GPU; a wrong invocation
fails loudly rather than being masked. `golden_v4.json` baselines are placeholders
to bootstrap from a real run.

## Output

Each GPU case appends its row to `v4_module_perf_log.csv` (via
`append_result`; dir overridable with `V4_PERF_LOG_DIR`), columns compatible with
the attention-perf log and keyed by `(case_id, GPU device name)`. The QA perf
pipeline ingests that CSV into `module_perf_result`, which an external correlation
dashboard reads to compare module-perf against layerwise/e2e kernel behaviour.
`expected_signatures_v4.json` records the per-module expected + critical-path kernel
families used by the pure signature checks.
