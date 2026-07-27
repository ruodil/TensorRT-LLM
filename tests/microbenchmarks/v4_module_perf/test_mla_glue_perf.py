# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA-glue (RoPE/quant + q_norm) module microbenchmark. Times the real
``torch.ops.trtllm.deepseek_v4_q_norm`` op; the signature check is pure."""

import pytest

from v4_module_perf_harness import (CaseResult, append_result, expected_families,
                                    golden_gate, load_golden, measure_gpu_time_ms,
                                    target_families, trtllm_op)

_MODULE = "MLA"
_CASE = "mla_ctx_qnorm_time"
_GOLDEN = load_golden()


def test_mla_target_families_are_expected():
    assert target_families(_MODULE) <= expected_families(_MODULE)


def test_mla_golden_present():
    assert _CASE in _GOLDEN and "NVIDIA B200" in _GOLDEN[_CASE]


@pytest.mark.continuous
def test_mla_ctx_qnorm_time():
    op = trtllm_op("deepseek_v4_q_norm")    # times deepseekV4QNormKernel
    import torch
    # Real op signature (tensorrt_llm/_torch/modules/mla.py):
    #   deepseek_v4_q_norm(q[tokens, num_heads*head_dim], num_heads, head_dim, eps)
    # kernel requires head_dim == 512; num_heads is per-TP (representative).
    tokens, num_heads, head_dim = 8192, 32, 512
    q = torch.randn(tokens, num_heads * head_dim, device="cuda", dtype=torch.bfloat16)
    median, p99, cv = measure_gpu_time_ms(lambda: op(q, num_heads, head_dim, 1e-6))
    r = CaseResult(_CASE, _MODULE, "context", "continuous", arch="sm100",
                   gpu_name=torch.cuda.get_device_name(0), gpu_time_median_ms=median,
                   gpu_time_p99_ms=p99, observed_cv=cv)
    r.verdict = golden_gate(r, _GOLDEN.get(_CASE, {}).get(r.gpu_name))
    append_result(r)
    assert r.verdict != "REGRESSION"
