# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""mHC (HyperConnection) module microbenchmark. Kernel timing calls the real
``torch.ops.trtllm.mhc_gemm_sqrsum_fma`` op on GPU and skips only when a genuine
prerequisite is absent; the signature/naming check is pure (no GPU)."""

import pytest

from v4_module_perf_harness import (CaseResult, append_result, expected_families,
                                    golden_gate, load_golden, measure_gpu_time_ms,
                                    target_families, trtllm_op)

_MODULE = "mhc_glue"
_CASE = "mhc_ctx_gemm_time"
_GOLDEN = load_golden()


# -- no-GPU: signature / naming alignment --------------------------------
def test_mhc_target_families_are_expected():
    targets = target_families(_MODULE)
    assert targets, "mHC must declare target families for the critical-path axis"
    assert targets <= expected_families(_MODULE)


def test_mhc_golden_keyed_by_case_and_gpu():
    assert _CASE in _GOLDEN and "NVIDIA B200" in _GOLDEN[_CASE]


# -- GPU: the actual microbenchmark (skips without CUDA / runtime / op) ---
@pytest.mark.continuous
def test_mhc_ctx_gemm_time():
    op = trtllm_op("mhc_gemm_sqrsum_fma")   # times mhcGemmSqrsumFmaKernel
    import torch
    # Real op signature (tensorrt_llm/_torch/modules/mhc/mhc_cuda.py):
    #   mhc_gemm_sqrsum_fma(x[M,K], w_t[K,N], y_acc[M,N] fp32, r_acc[M] fp32, M, N, K, tile_n, tile_m)
    # FP32 FMA GEMM on CUDA cores. M=tokens (ctx seq), K=hidden, N=mHC width.
    M, K, N = 8192, 7168, 4096
    x = torch.randn(M, K, device="cuda", dtype=torch.float32)
    w_t = torch.randn(K, N, device="cuda", dtype=torch.float32)
    y_acc = torch.empty((M, N), dtype=torch.float32, device="cuda")
    r_acc = torch.empty((M,), dtype=torch.float32, device="cuda")
    median, p99, cv = measure_gpu_time_ms(
        lambda: op(x, w_t, y_acc, r_acc, M, N, K, 0, 0))   # tile_n=tile_m=0 (auto)
    r = CaseResult(_CASE, _MODULE, "context", "continuous", arch="sm100",
                   gpu_name=torch.cuda.get_device_name(0), gpu_time_median_ms=median,
                   gpu_time_p99_ms=p99, observed_cv=cv)
    r.verdict = golden_gate(r, _GOLDEN.get(_CASE, {}).get(r.gpu_name))
    append_result(r)   # -> v4_module_perf_log.csv -> module_perf_result
    assert r.verdict != "REGRESSION", f"mHC gemm regressed: {median:.3f}ms"
