# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The native MX linear kernels must not depend on the number of rows.

Batch-invariant mode leaves the MXFP4 and MXFP8 GEMMs in place rather than
substituting a Triton kernel, because both reduce K sequentially within a single
workgroup: their tile selection is keyed on M, but only cross-workgroup split-K
would reorder an output element's accumulation. That is a property of the
kernels, not a contract they declare, so pin it here -- if AITER or the native
launcher starts splitting K, batch invariance breaks silently otherwise.
"""

import pytest
import torch
from utils import skip_unsupported

from vllm.platforms import current_platform

# Row counts spanning the M buckets the tile selectors switch on.
TOKEN_COUNTS = [1, 32, 64, 65, 128, 256, 257, 512, 1024, 1025, 2048]

requires_mx = pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="requires a ROCm device with native MX support (gfx95x)",
)


def _first_row_classes(rows: dict[int, torch.Tensor]) -> list[list[int]]:
    """Group row counts by bitwise-identical output, most common case: one class."""
    classes: list[tuple[torch.Tensor, list[int]]] = []
    for num_tokens, row in rows.items():
        for representative, members in classes:
            if torch.equal(row, representative):
                members.append(num_tokens)
                break
        else:
            classes.append((row, [num_tokens]))
    return [members for _, members in classes]


@skip_unsupported
@requires_mx
@pytest.mark.parametrize("n,k", [(4096, 2048), (2048, 6144)])
def test_mxfp4_linear_is_batch_invariant(n: int, k: int):
    # Importing the module registers torch.ops.vllm.gemm_with_dynamic_quant.
    import vllm.model_executor.kernels.linear.mxfp4.aiter  # noqa: F401

    pytest.importorskip("aiter")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    weight = torch.randint(
        0, 255, (n, k // 2), dtype=torch.uint8, device=device, generator=generator
    )
    # Stored transposed, matching AiterMxfp4LinearKernel.process_weights_after_loading.
    weight_scale = torch.randint(
        120, 134, (k // 32, n), dtype=torch.uint8, device=device, generator=generator
    )
    x = torch.randn(
        max(TOKEN_COUNTS), k, device=device, dtype=torch.bfloat16, generator=generator
    )

    rows = {
        num_tokens: torch.ops.vllm.gemm_with_dynamic_quant(
            x[:num_tokens].contiguous(), weight, weight_scale, False, torch.bfloat16
        )[0].clone()
        for num_tokens in TOKEN_COUNTS
    }

    classes = _first_row_classes(rows)
    assert len(classes) == 1, (
        f"MXFP4 linear row 0 changed with the row count: {classes} (N={n}, K={k})"
    )


@skip_unsupported
@requires_mx
@pytest.mark.parametrize("n,k", [(4096, 2048), (2048, 6144)])
def test_mxfp8_linear_is_batch_invariant(n: int, k: int):
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        _mxfp8_dot_scaled_linear,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    weight = (torch.randn(n, k, device=device, generator=generator) / 8).to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.randint(
        120, 134, (n, k // 32), dtype=torch.uint8, device=device, generator=generator
    )
    x = torch.randn(
        max(TOKEN_COUNTS), k, device=device, dtype=torch.bfloat16, generator=generator
    )

    rows = {
        num_tokens: _mxfp8_dot_scaled_linear(
            x[:num_tokens].contiguous(), weight, weight_scale
        )[0].clone()
        for num_tokens in TOKEN_COUNTS
    }

    classes = _first_row_classes(rows)
    assert len(classes) == 1, (
        f"MXFP8 linear row 0 changed with the row count: {classes} (N={n}, K={k})"
    )
