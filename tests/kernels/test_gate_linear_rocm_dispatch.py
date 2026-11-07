# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-agnostic eligibility tests for GateLinear's ROCm router GEMMs.

These assert the dispatch flags directly, so they run device-free by mocking
the platform predicates. ``allow_cublas_router_gemm`` selects the
bf16xbf16->fp32 ``torch.mm`` epilogue, while ``allow_fp32_router_gemm`` selects
the gfx950 low-M kernel with fp32 weights and output. On gfx1x the same tier
runs wvSplitK with an fp32 store (``allow_rocm_skinny_router_gemm``).

The ROCm branch is guarded on ``not bias`` because ``torch.mm`` has no bias
term; a biased gate must fall back so the bias is not silently dropped.
"""

import pytest
import torch

import vllm.model_executor.layers.fused_moe.router.gate_linear as gate_linear_mod
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.platforms import current_platform


def _make_gate(
    monkeypatch,
    *,
    is_rocm: bool,
    is_cuda: bool = False,
    bias: bool = False,
    params_dtype: torch.dtype = torch.bfloat16,
    out_dtype: torch.dtype | None = torch.float32,
    input_size: int = 2048,
    output_size: int = 64,
    on_gfx950: bool = False,
    on_gfx1x: bool = False,
    parallel_world_size: int = 1,
) -> GateLinear:
    """Build a GateLinear with platform predicates mocked, no GPU needed."""
    for target in (
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_rank",
            lambda: 0,
        )
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_world_size",
            lambda: parallel_world_size,
        )

    platform = gate_linear_mod.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(platform, "is_rocm", lambda: is_rocm)
    monkeypatch.setattr(platform, "is_device_capability", lambda *a, **k: False)
    monkeypatch.setattr(platform, "is_device_capability_family", lambda *a, **k: False)
    if is_rocm:
        import vllm.platforms.rocm as rocm_platform

        monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: on_gfx950)
        monkeypatch.setattr(rocm_platform, "on_gfx1x", lambda: on_gfx1x)

    return GateLinear(
        input_size=input_size,
        output_size=output_size,
        bias=bias,
        out_dtype=out_dtype,
        params_dtype=params_dtype,
    )


def test_rocm_no_bias_bf16_fp32_enables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=False)
    assert not gate.allow_specialized_router_gemm
    assert gate.allow_cublas_router_gemm


def test_rocm_bias_disables_fused_gemm(monkeypatch):
    # torch.mm cannot add a bias, so a biased gate must not take the fused path.
    gate = _make_gate(monkeypatch, is_rocm=True, bias=True)
    assert not gate.allow_cublas_router_gemm


def test_rocm_fp32_weight_disables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, params_dtype=torch.float32)
    assert not gate.allow_cublas_router_gemm


def test_rocm_non_fp32_out_dtype_disables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, out_dtype=torch.bfloat16)
    assert not gate.allow_cublas_router_gemm


def test_non_rocm_non_cuda_disables_fused_gemm(monkeypatch):
    # Neither the CUDA specialized path nor the ROCm branch applies.
    gate = _make_gate(monkeypatch, is_rocm=False, is_cuda=False)
    assert not gate.allow_cublas_router_gemm


def test_rocm_set_out_dtype_enables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=False, out_dtype=None)
    assert not gate.allow_cublas_router_gemm
    gate.set_out_dtype(torch.float32)
    assert gate.allow_cublas_router_gemm


def test_rocm_set_out_dtype_respects_bias_guard(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=True, out_dtype=None)
    gate.set_out_dtype(torch.float32)
    assert not gate.allow_cublas_router_gemm


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [(3072, 256), (4096, 8), (4096, 192), (6144, 128), (6144, 256)],
)
def test_rocm_gfx950_enables_fp32_router_gemm(
    monkeypatch, input_size: int, output_size: int
) -> None:
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        params_dtype=torch.float32,
        input_size=input_size,
        output_size=output_size,
        on_gfx950=True,
    )
    assert gate.allow_fp32_router_gemm
    assert not gate.allow_cublas_router_gemm


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_rocm_fp32_router_gemm_is_replicated_across_ep_sizes(
    monkeypatch, ep_size: int
) -> None:
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        params_dtype=torch.float32,
        input_size=6144,
        output_size=128,
        on_gfx950=True,
        parallel_world_size=ep_size,
    )
    assert gate.weight.shape == (128, 6144)
    assert gate.allow_fp32_router_gemm


@pytest.mark.parametrize(
    ("input_size", "output_size", "params_dtype", "bias", "on_gfx950"),
    [
        pytest.param(6144, 128, torch.float32, False, False, id="gfx942"),
        pytest.param(2048, 64, torch.float32, False, True, id="shape"),
        pytest.param(6144, 128, torch.bfloat16, False, True, id="bf16-weight"),
        pytest.param(6144, 128, torch.float32, True, True, id="bias"),
    ],
)
def test_rocm_fp32_router_gemm_rejects_unsupported_configs(
    monkeypatch,
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    bias: bool,
    on_gfx950: bool,
) -> None:
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        params_dtype=params_dtype,
        bias=bias,
        input_size=input_size,
        output_size=output_size,
        on_gfx950=on_gfx950,
    )
    assert not gate.allow_fp32_router_gemm


def test_rocm_gfx1x_enables_skinny_router_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, on_gfx1x=True)
    assert gate.allow_cublas_router_gemm
    assert gate.allow_rocm_skinny_router_gemm


def test_rocm_gfx1x_set_out_dtype_enables_skinny_router_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, on_gfx1x=True, out_dtype=None)
    assert not gate.allow_rocm_skinny_router_gemm
    gate.set_out_dtype(torch.float32)
    assert gate.allow_rocm_skinny_router_gemm


@pytest.mark.parametrize(
    ("on_gfx1x", "bias", "input_size"),
    [
        pytest.param(False, False, 2048, id="not-gfx1x"),
        pytest.param(True, True, 2048, id="bias"),
        pytest.param(True, False, 2052, id="k-not-multiple-of-8"),
    ],
)
def test_rocm_skinny_router_gemm_rejects_unsupported_configs(
    monkeypatch, on_gfx1x: bool, bias: bool, input_size: int
) -> None:
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        on_gfx1x=on_gfx1x,
        bias=bias,
        input_size=input_size,
    )
    assert not gate.allow_rocm_skinny_router_gemm


def _on_gfx1x_gpu() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1x

    return on_gfx1x()


@pytest.mark.skipif(not _on_gfx1x_gpu(), reason="needs a gfx1x GPU")
@pytest.mark.parametrize("num_tokens", [1, 5, 6])
def test_rocm_skinny_router_gemm_matches_fp32_reference(num_tokens: int) -> None:
    """Gemma-4-26B-A4B router shape. num_tokens=6 exercises the torch.mm
    fallback; both paths must return fp32 logits from fp32 accumulation."""
    torch.manual_seed(0)
    x = torch.randn(num_tokens, 2816, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(128, 2816, dtype=torch.bfloat16, device="cuda") * 0.02

    out = gate_linear_mod.rocm_skinny_router_gemm_impl(x, weight)

    assert out.dtype == torch.float32
    ref = x.float() @ weight.float().T
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
