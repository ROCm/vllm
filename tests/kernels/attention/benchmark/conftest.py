# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Pytest configuration for attention benchmark regression tests.

Adds --attn-bench-intermittent flag to enable performance validation
for intermittent (flaky) test cases.
"""

import pytest


def pytest_addoption(parser):
    """Add custom command-line options for attention benchmark tests."""
    parser.addoption(
        "--attn-bench-intermittent",
        action="store_true",
        default=False,
        help=(
            "Enable performance validation for intermittent "
            "attention benchmark test cases"
        ),
    )


@pytest.fixture
def bench_intermittent(request):
    """Fixture to check if intermittent validation is enabled."""
    return request.config.getoption("--attn-bench-intermittent")
