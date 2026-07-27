# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement + CSV + golden logic for DeepSeek-V4 module-perf microbenchmarks.

Modeled on ``attention_perf/attention_perf_harness.py``. Everything except the
actual GPU timing is pure and unit-testable without a GPU or the TRT-LLM runtime:

  - CaseResult / to_csv_row / write_csv    : the module-perf CSV contract
  - continuous_gate / golden_gate          : continuous / discrete verdict logic
  - load_expected_signatures / *_families  : per-module expected kernel families
  - measure_gpu_time_ms / require_gpu       : GPU-only, guarded
  - trtllm_op                               : resolve a real torch.ops.trtllm.* op

The emitted CSV (per (case_id, GPU device name)) is what an external correlation
dashboard ingests to compare module-perf against layerwise/e2e kernel behaviour.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

_HERE = Path(__file__).resolve().parent

# Per-run CSV log the QA perf pipeline ingests into module_perf_result. Path is
# overridable so each nightly pass writes to its own writable dir.
_PERF_LOG = Path(os.environ.get("V4_PERF_LOG_DIR", str(_HERE))) / "v4_module_perf_log.csv"

CSV_COLUMNS = [
    "case_id", "arch", "phase", "discrete_metric", "launch_count",
    "gpu_time_median_ms", "gpu_time_p99_ms", "observed_cv", "gpu_name",
    "node", "shape", "golden", "verdict",
]

DEFAULT_K_SIGMA = 4.0
DEFAULT_REL_FLOOR = 0.01


@dataclass
class CaseResult:
    case_id: str
    module: str
    phase: str                       # context | generation
    metric_type: str                 # continuous | discrete
    shape: str = ""
    arch: str = ""
    gpu_name: str = ""               # torch.cuda.get_device_name -> golden key
    node: str = ""
    gpu_time_median_ms: Optional[float] = None
    gpu_time_p99_ms: Optional[float] = None
    observed_cv: Optional[float] = None
    launch_count: Optional[int] = None
    discrete_metric: str = ""
    discrete_value: Any = None
    verdict: str = "checked"
    golden: Any = None


# -- golden gate ----------------------------------------------------------
def continuous_gate(baseline_ms: float, cv: float, observed_ms: float,
                    k_sigma: float = DEFAULT_K_SIGMA,
                    rel_floor: float = DEFAULT_REL_FLOOR) -> str:
    gate = baseline_ms * (1.0 + max(k_sigma * cv, rel_floor))
    return "REGRESSION" if observed_ms > gate else "pass"


def golden_gate(result: CaseResult, golden: Mapping[str, Any] | None) -> str:
    if not golden:
        return "bootstrap"
    if result.metric_type == "discrete":
        observed = result.discrete_value
        if observed is None and result.discrete_metric == "launch_count":
            observed = result.launch_count
        return "pass" if observed == golden.get("value") else "REGRESSION"
    baseline = golden.get("baseline_ms")
    if baseline is None or result.gpu_time_median_ms is None:
        return "checked"
    return continuous_gate(
        float(baseline), float(golden.get("cv", 0.0)), float(result.gpu_time_median_ms),
        float(golden.get("k_sigma", DEFAULT_K_SIGMA)),
        float(golden.get("rel_floor", DEFAULT_REL_FLOOR)))


# -- CSV -----------------------------------------------------------------
def to_csv_row(result: CaseResult) -> dict[str, Any]:
    d = asdict(result)
    return {c: d.get(c) for c in CSV_COLUMNS}


def write_csv(results: Sequence[CaseResult], path: str | Path) -> Path:
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in results:
            w.writerow(to_csv_row(r))
    return path


def append_result(result: CaseResult, path: str | Path | None = None) -> None:
    """Best-effort append of one run to the module-perf CSV log (the file the QA
    perf pipeline ingests into module_perf_result). A write failure (e.g. read-only
    mount) is warned but never fails the test."""
    p = Path(path) if path is not None else _PERF_LOG
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.exists()
        with p.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(to_csv_row(result))
    except OSError as e:
        print(f"[warn] v4 perf log not written to {p}: {e}")


# -- expected signatures (per module) ------------------------------------
def load_expected_signatures() -> dict[str, Any]:
    return json.loads((_HERE / "expected_signatures_v4.json").read_text())


def expected_families(module: str) -> set[str]:
    return set(load_expected_signatures()[module]["expected_families"])


def target_families(module: str) -> set[str]:
    return set(load_expected_signatures()[module]["target_families"])


def load_golden() -> dict[str, Any]:
    return json.loads((_HERE / "golden_v4.json").read_text())


# -- GPU-only (guarded) --------------------------------------------------
def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    return torch


def trtllm_op(name: str):
    """Resolve a real torch.ops.trtllm.* op for a GPU microbenchmark. Skips only on
    a genuine prerequisite gap (no GPU / no TRT-LLM / op not registered); it does
    not catch-all-skip, so a wrong op invocation on a real runtime fails loudly."""
    import pytest
    try:
        require_gpu()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no GPU: {exc}")
    pytest.importorskip("tensorrt_llm", reason="TRT-LLM runtime not installed")
    import torch
    op = getattr(torch.ops.trtllm, name, None)
    if op is None:
        pytest.skip(f"torch.ops.trtllm.{name} not registered in this build")
    return op


def measure_gpu_time_ms(fn: Callable[[], Any], *, warmup: int = 10, iters: int = 50
                        ) -> tuple[float, float, float]:
    """Median / p99 / cv of a callable's GPU time (ms). GPU-only."""
    torch = require_gpu()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    median = statistics.median(samples)
    p99 = samples[min(len(samples) - 1, int(round(0.99 * (len(samples) - 1))))]
    cv = (statistics.pstdev(samples) / median) if median else 0.0
    return median, p99, cv
