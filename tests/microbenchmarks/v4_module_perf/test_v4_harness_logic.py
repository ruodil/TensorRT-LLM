# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No-GPU unit tests for the V4 module-perf harness CSV + golden logic (mirrors
attention_perf's golden-logic test: no CUDA, no runtime)."""

import csv

from v4_module_perf_harness import (CSV_COLUMNS, CaseResult, append_result,
                                    continuous_gate, golden_gate, to_csv_row,
                                    write_csv)


def test_continuous_gate():
    # gate = 1.0 * (1 + max(4*0.03, 0.01)) = 1.12
    assert continuous_gate(1.0, 0.03, 1.10) == "pass"
    assert continuous_gate(1.0, 0.03, 1.20) == "REGRESSION"
    assert continuous_gate(1.0, 0.0, 1.005) == "pass"      # rel_floor dominates
    assert continuous_gate(1.0, 0.0, 1.02) == "REGRESSION"


def test_golden_gate_continuous_discrete_bootstrap():
    cont = CaseResult("mhc_ctx_gemm_time", "mhc_glue", "context", "continuous",
                      gpu_time_median_ms=1.20)
    assert golden_gate(cont, {"baseline_ms": 1.0, "cv": 0.03}) == "REGRESSION"
    assert golden_gate(cont, None) == "bootstrap"

    disc = CaseResult("mhc_ctx_launch_count", "mhc_glue", "context", "discrete",
                      discrete_metric="launch_count", discrete_value=7119)
    assert golden_gate(disc, {"value": 7119}) == "pass"
    assert golden_gate(disc, {"value": 7000}) == "REGRESSION"


def test_golden_gate_discrete_falls_back_to_launch_count():
    r = CaseResult("mhc_ctx_launch_count", "mhc_glue", "context", "discrete",
                   discrete_metric="launch_count", launch_count=7119)  # no discrete_value
    assert golden_gate(r, {"value": 7119}) == "pass"
    assert golden_gate(r, {"value": 7000}) == "REGRESSION"


def test_csv_roundtrip(tmp_path):
    r = CaseResult("mhc_ctx_gemm_time", "mhc_glue", "context", "continuous",
                   arch="sm100", gpu_name="NVIDIA B200", gpu_time_median_ms=1.9,
                   observed_cv=0.03, launch_count=7119, shape="M=8192", verdict="pass")
    path = write_csv([r], tmp_path / "v4.csv")
    row = to_csv_row(r)
    assert row["case_id"] == "mhc_ctx_gemm_time" and row["gpu_time_median_ms"] == 1.9
    back = list(csv.DictReader(path.open()))
    assert back[0]["case_id"] == "mhc_ctx_gemm_time"
    assert set(back[0].keys()) == set(CSV_COLUMNS)


def test_append_result_appends_rows(tmp_path):
    log = tmp_path / "v4_module_perf_log.csv"
    r = CaseResult("mhc_ctx_gemm_time", "mhc_glue", "context", "continuous",
                   gpu_name="NVIDIA B200", gpu_time_median_ms=1.9, verdict="pass")
    append_result(r, log)
    append_result(r, log)   # appends, header written once
    rows = list(csv.DictReader(log.open()))
    assert len(rows) == 2 and rows[0]["case_id"] == "mhc_ctx_gemm_time"
    assert set(rows[0].keys()) == set(CSV_COLUMNS)
