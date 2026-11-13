# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-agnostic eligibility tests for GateLinear's ROCm router GEMMs.

These assert the dispatch flags directly, so they run device-free by mocking
the platform predicates. `allow_cublas_router_gemm` selects the
bf16xbf16->fp32 `torch.mm` epilogue, while `allow_fp32_router_gemm` selects
the gfx950 low-M kernel with fp32 weights and output.

The ROCm branch is guarded on `not bias` because `torch.mm` has no bias
term; a biased gate must fall back so the bias is not silently dropped.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.router.gate_linear as gate_linear_mod
from vllm.model_executor.layers.fused_moe.router import (
    bf16x3_router_gemm_rocm as rocm_bf16x3,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.model_loader.reload import (
    finish_reload,
    record_metadata_for_reloading,
    start_reload,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


def _mock_tp(monkeypatch, world_size: int = 1) -> None:
    for target in (
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(f"{target}.get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_world_size", lambda: world_size
        )


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
    parallel_world_size: int = 1,
) -> GateLinear:
    """Build a GateLinear with platform predicates mocked, no GPU needed."""
    _mock_tp(monkeypatch, parallel_world_size)
    platform = gate_linear_mod.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(platform, "is_rocm", lambda: is_rocm)
    monkeypatch.setattr(platform, "is_device_capability", lambda *a, **k: False)
    monkeypatch.setattr(platform, "is_device_capability_family", lambda *a, **k: False)
    if is_rocm:
        import vllm.platforms.rocm as rocm_platform

        monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: on_gfx950)

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
    [
        (3072, 256),
        (4096, 8),
        (4096, 192),
        (6144, 128),
        (6144, 256),
        (8192, 256),
        (2048, 64),
        (2560, 512),
        (7168, 384),
    ],
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


# ---------------------------------------------------------------------------
# ROCm BF16x3 tier (fp32 router weights)
# ---------------------------------------------------------------------------
def _make_bf16x3_gate(monkeypatch, *, supported=True, **kwargs):
    """Build a gate with the ROCm bf16x3 platform check mocked."""
    monkeypatch.setattr(rocm_bf16x3, "platform_supported", lambda: supported)
    kwargs.setdefault("params_dtype", torch.float32)
    return _make_gate(monkeypatch, is_rocm=True, **kwargs)


def test_rocm_bf16x3_enabled_for_fp32_weight(monkeypatch):
    gate = _make_bf16x3_gate(monkeypatch)
    assert gate.allow_rocm_bf16x3_router_gemm
    # Built by process_weights_after_loading, not lazily in forward().
    assert gate._bf16x3_weight is None


def test_rocm_bf16x3_disabled_by_bias(monkeypatch):
    # torch.mm has no bias term.
    gate = _make_bf16x3_gate(monkeypatch, bias=True)
    assert not gate.allow_rocm_bf16x3_router_gemm


def test_rocm_bf16x3_disabled_for_bf16_weight(monkeypatch):
    # A bf16 weight belongs to the cuBLAS tier; there is nothing to split.
    gate = _make_bf16x3_gate(monkeypatch, params_dtype=torch.bfloat16)
    assert not gate.allow_rocm_bf16x3_router_gemm


def test_rocm_bf16x3_disabled_for_non_fp32_out_dtype(monkeypatch):
    gate = _make_bf16x3_gate(monkeypatch, out_dtype=torch.bfloat16)
    assert not gate.allow_rocm_bf16x3_router_gemm


def test_rocm_bf16x3_disabled_on_unsupported_arch(monkeypatch):
    # Only gfx950 is benchmarked; gfx942 has a different fp32:bf16 MFMA ratio.
    gate = _make_bf16x3_gate(monkeypatch, supported=False)
    assert not gate.allow_rocm_bf16x3_router_gemm
    assert not gate._rocm_bf16x3_weight_eligible


def test_rocm_bf16x3_no_split_built_when_tier_disabled(monkeypatch):
    """An ineligible out_dtype must not allocate a split that is never read;
    it is 1.5x the router weight on every layer."""
    gate = _make_bf16x3_gate(monkeypatch, out_dtype=torch.bfloat16)
    gate.quant_method.process_weights_after_loading(gate)
    assert gate._bf16x3_weight is None


def test_rocm_bf16x3_set_out_dtype_enables_tier(monkeypatch):
    gate = _make_bf16x3_gate(monkeypatch, out_dtype=None)
    assert not gate.allow_rocm_bf16x3_router_gemm
    gate.set_out_dtype(torch.float32)
    assert gate.allow_rocm_bf16x3_router_gemm


def test_rocm_bf16x3_dispatch_is_a_custom_op():
    """The num_tokens branch must stay opaque to Dynamo; see the op's impl."""
    assert hasattr(torch.ops.vllm, "rocm_bf16x3_router_gemm_dispatch")
    x = torch.zeros(8192, 6144, dtype=torch.bfloat16, device="meta")
    w = torch.zeros(128, 6144, dtype=torch.float32, device="meta")
    split = torch.zeros(3, 128, 6144, dtype=torch.bfloat16, device="meta")
    out = torch.ops.vllm.rocm_bf16x3_router_gemm_dispatch(x, w, split)
    assert out.shape == (8192, 128)
    assert out.dtype == torch.float32


def test_rocm_bf16x3_is_reachable_for_low_m_shapes(monkeypatch):
    """The gfx950 low-M fp32 tier claims these shapes too and returns
    unconditionally, so bf16x3 is only reachable if checked first."""
    gate = _make_bf16x3_gate(
        monkeypatch, input_size=6144, output_size=128, on_gfx950=True
    )
    assert gate.allow_fp32_router_gemm
    assert gate.allow_rocm_bf16x3_router_gemm

    source = inspect.getsource(type(gate).forward)
    assert source.index("allow_rocm_bf16x3_router_gemm") < source.index(
        "allow_fp32_router_gemm"
    )


# ---------------------------------------------------------------------------
# Weight reload of the ROCm BF16x3 split
# ---------------------------------------------------------------------------
_MODEL_CONFIG = SimpleNamespace(
    word_embeddings_untied_by_checkpoint=False, quantization=None, dtype=torch.float32
)


class _GateModel(torch.nn.Module):
    def __init__(self, gate: GateLinear):
        super().__init__()
        self.gate = gate

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, weight in weights:
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
        return set(params)


def _gate_checkpoint(gate: GateLinear, seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {"gate.weight": torch.randn(gate.weight.shape, generator=gen) * 0.05}


def _reload_gate(model: _GateModel, device: torch.device) -> torch.Tensor:
    """Cold load, then a checkpoint-format reload with different weights, as
    `reload_weights` does. Returns the reloaded weight."""
    gate = model.gate
    model.load_weights(_gate_checkpoint(gate, seed=0).items())
    process_weights_after_loading(model, _MODEL_CONFIG, device)
    split = gate._bf16x3_weight
    assert split is not None
    ptr = split.data_ptr()

    start_reload(model)
    new = _gate_checkpoint(gate, seed=1)
    model.load_weights(new.items())
    finish_reload(model, _MODEL_CONFIG, device)

    w = new["gate.weight"].to(device)
    split = gate._bf16x3_weight
    assert split is not None
    assert torch.equal(gate.weight, w)
    assert torch.equal(split.float().sum(0), w)
    # Refilled in place: captured graphs keep reading the same storage.
    assert split.data_ptr() == ptr
    return w


@pytest.mark.parametrize("mode", ["direct", "layerwise"])
def test_rocm_bf16x3_split_survives_weight_reload(monkeypatch, mode):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    model = _GateModel(_make_bf16x3_gate(monkeypatch))
    record_metadata_for_reloading(model)
    _reload_gate(model, torch.device("cpu"))


@pytest.mark.skipif(
    not (torch.accelerator.is_available() and rocm_bf16x3.platform_supported()),
    reason="ROCm bf16x3 router GEMM requires gfx950",
)
@pytest.mark.parametrize("mode", ["direct", "layerwise"])
def test_rocm_bf16x3_weight_reload_gfx950(monkeypatch, mode):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    _mock_tp(monkeypatch)
    device = torch.device("cuda", 0)
    with device:  # the loader builds and records under the target device
        model = _GateModel(
            GateLinear(
                input_size=6144,
                output_size=128,
                bias=False,
                out_dtype=torch.float32,
                params_dtype=torch.float32,
            )
        )
        record_metadata_for_reloading(model)
    assert model.gate.allow_rocm_bf16x3_router_gemm
    w = _reload_gate(model, device)

    x = torch.randn(rocm_bf16x3.MIN_TOKENS, 6144, device=device).to(torch.bfloat16)
    out = model.gate(x)
    out = out[0] if isinstance(out, tuple) else out
    ref = x.double() @ w.double().T
    assert (out.double() - ref).norm() / ref.norm() < 2e-6
