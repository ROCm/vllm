# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared test utilities for ROCm kernel tests.

Provides beta-quality accuracy and determinism helpers modelled on the
skinny GEMM test patterns. Import these into individual test files instead
of duplicating the logic.
"""

from collections.abc import Callable
from typing import Any

import torch


def _assert_accurate(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float = 0.0,
    pass_rate: float = 0.99999,
    max_violation_factor: float = 3.0,
) -> None:
    """3-pronged accuracy check for numeric kernel output.

    1. At least ``pass_rate`` fraction of elements are within ``atol`` + ``rtol *
       |expected|``.
    2. No element violates the bound by more than ``max_violation_factor *
       atol``.
    3. Mean absolute error is less than ``atol * 0.25``.

    Args:
        actual: Kernel output tensor (any dtype; cast to float32 internally).
        expected: Reference tensor of the same shape.
        atol: Absolute tolerance.
        rtol: Relative tolerance (applied to ``|expected|``).
        pass_rate: Minimum fraction of elements that must satisfy the tolerance.
        max_violation_factor: Maximum allowed violation relative to ``atol``.
    """
    a = actual.detach().float().flatten()
    e = expected.detach().float().flatten()

    abs_err = (a - e).abs()
    tol = atol + rtol * e.abs()

    # 1. Pass rate
    rate = (abs_err <= tol).float().mean().item()
    assert rate >= pass_rate, (
        f"Accuracy pass rate {rate:.6f} < {pass_rate} (atol={atol}, rtol={rtol})"
    )

    # 2. Max violation
    max_err = abs_err.max().item()
    assert max_err <= max_violation_factor * atol, (
        f"Max absolute error {max_err:.6f} exceeds {max_violation_factor} * atol={atol}"
    )

    # 3. Mean absolute error
    mean_err = abs_err.mean().item()
    assert mean_err <= atol * 0.25, (
        f"Mean absolute error {mean_err:.6f} >= atol * 0.25 = {atol * 0.25:.6f}"
    )


def _assert_deterministic(
    fn: Callable[..., Any],
    *args: Any,
    n_runs: int = 4,
    **kwargs: Any,
) -> None:
    """Verify that ``fn(*args, **kwargs)`` produces bitwise-identical outputs.

    Runs the function ``n_runs`` times and asserts that every run produces
    the exact same result as the first run.

    Args:
        fn: Callable to test. Must return a :class:`torch.Tensor` or a tuple/
            list of :class:`torch.Tensor`.
        *args: Positional arguments forwarded to ``fn``.
        n_runs: Number of times to call ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.
    """

    def _collect(result: Any):
        if isinstance(result, torch.Tensor):
            return [result.detach().clone()]
        if isinstance(result, (tuple, list)):
            return [t.detach().clone() for t in result if isinstance(t, torch.Tensor)]
        raise TypeError(f"Unexpected return type {type(result)}")

    reference = _collect(fn(*args, **kwargs))

    for run in range(1, n_runs):
        outputs = _collect(fn(*args, **kwargs))
        for idx, (ref, out) in enumerate(zip(reference, outputs)):
            assert torch.equal(ref, out), (
                f"Run {run}: output[{idx}] differs from run 0 "
                f"(max diff = {(out.float() - ref.float()).abs().max().item():.2e})"
            )
