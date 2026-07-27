# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register the perf markers used by the V4 module microbenchmarks."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "continuous: GPU timing case (needs a quiet/locked GPU)")
    config.addinivalue_line(
        "markers", "discrete: exact-match tripwire case (launch_count / path flags)")
