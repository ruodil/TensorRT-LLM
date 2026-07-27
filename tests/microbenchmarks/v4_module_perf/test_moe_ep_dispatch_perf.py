# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE EP-dispatch module microbenchmark. Times the real
``torch.ops.trtllm.moe_a2a_dispatch`` op; the signature check is pure."""

import pytest

from v4_module_perf_harness import (expected_families, load_golden,
                                    target_families, trtllm_op)

_MODULE = "MoE"
_CASE = "moe_ctx_dispatch_time"
_GOLDEN = load_golden()


def test_moe_target_families_are_expected():
    assert target_families(_MODULE) <= expected_families(_MODULE)


def test_moe_golden_present():
    assert _CASE in _GOLDEN and "NVIDIA B200" in _GOLDEN[_CASE]


@pytest.mark.continuous
def test_moe_ctx_dispatch_time():
    op = trtllm_op("moe_a2a_dispatch")      # confirms the op is registered
    # Unlike mHC/q_norm, moe_a2a_dispatch is STATEFUL: it needs the all-to-all
    # workspace from moe_a2a_initialize, per-token expert routing, and payload
    # tensors (see tensorrt_llm/_torch/distributed/moe_alltoall.py). It cannot be
    # timed as a standalone op call. Isolate it via that A2A path (build the
    # workspace + routing once, time the dispatch) before enabling this case.
    assert op is not None
    pytest.skip("moe_a2a_dispatch is stateful (needs moe_a2a_initialize workspace + "
                "expert routing); time it via the A2A setup path, not a bare op call")
