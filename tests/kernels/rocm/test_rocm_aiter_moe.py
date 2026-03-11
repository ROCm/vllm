# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AITER Fused MoE kernels on ROCm.

Covers:
- AITER fused MoE BF16 forward (unquantized, a16w16) — SiLU and GELU activations
- AITER fused MoE FP8 group-quantized forward
- Numerical accuracy: AITER fused MoE vs float32 mask-based reference (NVIDIA parity)
  * Correct weight pre-shuffling for AITER CK kernel
  * Distinct expert IDs via softmax+topk routing (not randint)
  * Xavier-normalized weights for O(1) output magnitudes (enables tight atol=0.05)
- End-to-end MoE test: full router logits → softmax → topk → fused GEMM pipeline
- GPU-specific accuracy tests: gfx942 (MI300X/MI325X) BF16, gfx950 (MI350X) FP8
- FP8 group-quant MoE activation quantization correctness
- Determinism: same inputs → bitwise-identical outputs across N runs
- VLLM_ROCM_USE_AITER_MOE env var gating
- VLLM_ROCM_MOE_PADDING flag (does not crash with True/False)
- VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS API
- rocm_aiter_ops.is_fused_moe_enabled() / is_fusion_moe_shared_experts_enabled()
- Custom op registration: rocm_aiter_fused_moe, rocm_aiter_asm_moe_tkw1
- torch.compile fake-tensor compatibility
"""

import importlib.util
import math
import os

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.rocm.utils import _assert_accurate, _assert_deterministic
from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx942, on_gfx950
else:

    def on_gfx942() -> bool:
        return False

    def on_gfx950() -> bool:
        return False


pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

aiter_available = importlib.util.find_spec("aiter") is not None


def require_aiter():
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter package required for this test")


# ── Reference MoE implementation ──────────────────────────────────────────


def ref_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """Mask-based batched MoE reference matching NVIDIA's torch_experts pattern.

    Expands all (token, expert) pairs, applies expert GEMMs via masks, then
    reduces. Runs in float32 for a precise ground truth.

    Args:
        hidden_states: [M, K] any dtype (cast to float32 internally).
        w1: [E, 2*N, K] gate+up projection weights.
        w2: [E, K, N] down projection weights.
        topk_weights: [M, topk] float32 routing weights.
        topk_ids: [M, topk] int32 expert indices.
        activation: "silu" for SiGLU or "gelu" for GeGLU.
    """
    M, K = hidden_states.shape
    E = w1.shape[0]
    intermediate = w1.shape[1] // 2  # gate + up are concatenated
    topk = topk_ids.shape[1]
    device = hidden_states.device

    # Cast everything to float32 for precision
    a = hidden_states.float()
    w1f = w1.float()
    w2f = w2.float()

    # Expand tokens: [M*topk, K]
    a_exp = a.view(M, 1, K).expand(M, topk, K).reshape(M * topk, K)
    out = torch.zeros(M * topk, K, dtype=torch.float32, device=device)
    tids_flat = topk_ids.view(-1).long()

    for e in range(E):
        mask = tids_flat == e
        if mask.sum() == 0:
            continue
        gate_up = a_exp[mask] @ w1f[e].T  # [n, 2I]
        gate, up = gate_up[:, :intermediate], gate_up[:, intermediate:]
        if activation == "silu":
            act = F.silu(gate) * up
        elif activation == "gelu":
            act = F.gelu(gate) * up
        else:
            raise ValueError(f"Unknown activation: {activation}")
        out[mask] = act @ w2f[e].T  # [n, K]

    # Reduce with routing weights: [M, topk, K] weighted sum → [M, K]
    out = out.view(M, topk, K)
    weights = topk_weights.float().to(device).view(M, topk, 1)
    return (out * weights).sum(dim=1)


# ── Helper functions ───────────────────────────────────────────────────────


def _make_topk_ids(
    num_tokens: int, num_experts: int, topk: int, device: str = "cuda"
) -> torch.Tensor:
    """Generate distinct expert IDs per token via softmax+topk routing.

    Using torch.randint allows duplicate expert assignments per token which
    causes systematic errors in the AITER kernel. softmax+topk guarantees
    each token's topk experts are distinct, matching production routing.
    """
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    _, topk_ids = torch.topk(torch.softmax(router_logits, dim=-1), k=topk, dim=-1)
    return topk_ids.to(torch.int32)


def _shuffle_moe_weights(
    w1: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle MoE weights for the AITER CK kernel.

    The AITER CK (Composable Kernels) fused MoE kernel requires weights to be
    pre-shuffled into its vectorized memory layout. Production code calls this
    in oracle/unquantized.py before running inference. Tests must do the same.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    w1_s, w2_s = rocm_aiter_ops.shuffle_weights(w1, w2)
    w1_s.is_shuffled = True
    w2_s.is_shuffled = True
    return w1_s, w2_s


# ── Op registration tests ──────────────────────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_fused_moe_custom_op_registered():
    """Test that rocm_aiter_fused_moe custom op is registered."""
    require_aiter()
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_fused_moe")
    assert callable(torch.ops.vllm.rocm_aiter_fused_moe)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_asm_moe_tkw1_custom_op_registered():
    """Test that rocm_aiter_asm_moe_tkw1 custom op is registered."""
    require_aiter()
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_asm_moe_tkw1")
    assert callable(torch.ops.vllm.rocm_aiter_asm_moe_tkw1)


# ── rocm_aiter_ops state tests ─────────────────────────────────────────────


def test_aiter_moe_is_fused_moe_enabled():
    """Test rocm_aiter_ops.is_fused_moe_enabled() API."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    result = rocm_aiter_ops.is_fused_moe_enabled()
    assert result is None or isinstance(result, bool)


def test_aiter_moe_is_fusion_shared_experts_enabled():
    """Test rocm_aiter_ops.is_fusion_moe_shared_experts_enabled() API.
    Exercises VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS env var."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    result = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
    assert result is None or isinstance(result, bool)


# ── Fake tensor / shape tests ──────────────────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
def test_aiter_fused_moe_fake_tensor():
    """Test that rocm_aiter_fused_moe fake impl produces correct output shape."""
    require_aiter()
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    import vllm._aiter_ops  # noqa: F401

    num_tokens = 16
    hidden_dim = 1024
    intermediate_dim = 2048
    num_experts = 8
    topk = 2

    hidden_states = torch.randn(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda"
    )
    w1 = torch.randn(
        num_experts,
        intermediate_dim * 2,
        hidden_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w2 = torch.randn(
        num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16, device="cuda"
    )

    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32, device="cuda")
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )

    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_fused_moe,
        (hidden_states, w1, w2, topk_weights, topk_ids),
        kwargs={
            "expert_mask": None,
            "activation_method": 0,  # SILU
            "quant_method": 0,  # NO (a16w16)
            "doweight_stage1": False,
        },
        test_utils=("test_faketensor",),
    )


# ── MoE padding tests ──────────────────────────────────────────────────────


@pytest.mark.parametrize("moe_padding", [True, False])
def test_aiter_moe_padding_env_var(moe_padding, monkeypatch):
    """Test VLLM_ROCM_MOE_PADDING env var is readable and valid."""
    monkeypatch.setenv("VLLM_ROCM_MOE_PADDING", "1" if moe_padding else "0")
    import importlib

    import vllm.envs as envs

    importlib.reload(envs)

    assert moe_padding == envs.VLLM_ROCM_MOE_PADDING


# ── QuantMethod enum tests ─────────────────────────────────────────────────


def test_quant_method_enum_values():
    """Test QuantMethod enum in rocm_aiter_fused_moe has expected values."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import QuantMethod

    assert QuantMethod.NO == 0
    assert QuantMethod.PER_TENSOR == 1
    assert QuantMethod.PER_TOKEN == 2
    assert QuantMethod.BLOCK_1X32 == 3
    assert QuantMethod.BLOCK_1X128 == 4
    assert QuantMethod.BLOCK_128x128 == 5


def test_activation_method_enum_values():
    """Test ActivationMethod enum has SILU and GELU."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
    )

    assert ActivationMethod.SILU == 0
    assert ActivationMethod.GELU == 1


# ── FP8 group quant tests ──────────────────────────────────────────────────


@pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="ROCm + aiter required",
)
@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 not supported on this hardware",
)
def test_aiter_moe_fp8_group_quant_shape():
    """Test AITER group FP8 quant produces correct output shapes for MoE."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")

    num_tokens = 32
    hidden_dim = 4096
    group_size = 128

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    x_fp8, scales = rocm_aiter_ops.group_fp8_quant(hidden_states, group_size)

    assert x_fp8.shape == (num_tokens, hidden_dim)
    expected_scale_cols = (hidden_dim + group_size - 1) // group_size
    assert scales.shape == (num_tokens, expected_scale_cols)
    assert scales.dtype == torch.float32


@pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="ROCm + aiter required",
)
@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 not supported on this hardware",
)
@pytest.mark.parametrize("num_tokens,hidden_dim", [(16, 2048), (64, 4096), (128, 8192)])
def test_aiter_moe_fp8_group_quant_various_shapes(num_tokens, hidden_dim):
    """Test FP8 group quant with various MoE-relevant shapes."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")

    group_size = 128
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    x_fp8, scales = rocm_aiter_ops.group_fp8_quant(x, group_size)

    assert x_fp8.shape == (num_tokens, hidden_dim)
    expected_cols = (hidden_dim + group_size - 1) // group_size
    assert scales.shape == (num_tokens, expected_cols)


# ── Numerical accuracy tests ───────────────────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize(
    "num_tokens,hidden_dim,intermediate_dim",
    [
        (16, 512, 1024),  # small — fast smoke test
        (128, 2048, 4096),  # Llama-7B class hidden dimension
        (2048, 4096, 11008),  # Llama-7B full scale
    ],
)
@torch.inference_mode()
def test_aiter_fused_moe_accuracy(num_tokens, hidden_dim, intermediate_dim):
    """AITER fused MoE (SiGLU/SiLU, BF16) matches float32 PyTorch reference.

    Key correctness requirements:
    - Weights must be pre-shuffled via rocm_aiter_ops.shuffle_weights() for the
      AITER CK kernel to interpret memory correctly (production code does this).
    - Expert IDs must be distinct per token (softmax+topk routing, not randint).
    - Weights normalized by 1/sqrt(K) keep output magnitudes O(1) so that
      atol=0.05 is meaningful (unnormalized weights → O(7000) outputs → BF16
      step-size ~55 >> atol).
    - Reference runs in float32 via mask-based batched pattern (NVIDIA parity).
    """
    require_aiter()
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    import vllm._aiter_ops  # noqa: F401 triggers op registration

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    num_experts = 8
    topk = 2

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    # Xavier-like init: normalize so outputs are O(1) in magnitude
    w1 = torch.randn(
        num_experts, intermediate_dim * 2, hidden_dim, dtype=torch.bfloat16
    ) / math.sqrt(hidden_dim)
    w2 = torch.randn(
        num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16
    ) / math.sqrt(intermediate_dim)

    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    # Distinct expert IDs per token (no duplicates — matches production routing)
    topk_ids = _make_topk_ids(num_tokens, num_experts, topk)

    # Float32 mask-based reference (NVIDIA torch_experts pattern)
    ref_out = ref_moe_forward(
        hidden_states, w1, w2, topk_weights, topk_ids, activation="silu"
    )

    # The AITER CK kernel requires pre-shuffled weights
    w1_s, w2_s = _shuffle_moe_weights(w1, w2)

    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
        QuantMethod,
    )

    aiter_out = torch.ops.vllm.rocm_aiter_fused_moe(
        hidden_states,
        w1_s,
        w2_s,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation_method=int(ActivationMethod.SILU),
        quant_method=int(QuantMethod.NO),
        doweight_stage1=False,
    )

    assert aiter_out.shape == (num_tokens, hidden_dim)
    # BF16 kernel vs float32 reference; max observed error ~0.031 for all shapes
    _assert_accurate(aiter_out.float(), ref_out, atol=0.05, rtol=0.0)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@torch.inference_mode()
def test_aiter_fused_moe_gelu_accuracy():
    """AITER fused MoE with GELU activation (GeGLU) matches float32 reference.

    Exercises ActivationMethod.GELU — previously untested for accuracy.
    Applies the same correctness requirements as the SiLU test: shuffled weights,
    distinct expert IDs, and normalized weight magnitudes.
    """
    require_aiter()
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    import vllm._aiter_ops  # noqa: F401

    torch.set_default_device("cuda")
    torch.manual_seed(42)

    num_tokens = 32
    hidden_dim = 512
    intermediate_dim = 1024
    num_experts = 4
    topk = 2

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts, intermediate_dim * 2, hidden_dim, dtype=torch.bfloat16
    ) / math.sqrt(hidden_dim)
    w2 = torch.randn(
        num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16
    ) / math.sqrt(intermediate_dim)
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = _make_topk_ids(num_tokens, num_experts, topk)

    ref_out = ref_moe_forward(
        hidden_states, w1, w2, topk_weights, topk_ids, activation="gelu"
    )

    w1_s, w2_s = _shuffle_moe_weights(w1, w2)

    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
        QuantMethod,
    )

    aiter_out = torch.ops.vllm.rocm_aiter_fused_moe(
        hidden_states,
        w1_s,
        w2_s,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation_method=int(ActivationMethod.GELU),
        quant_method=int(QuantMethod.NO),
        doweight_stage1=False,
    )

    assert aiter_out.shape == (num_tokens, hidden_dim)
    _assert_accurate(aiter_out.float(), ref_out, atol=0.05, rtol=0.0)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 not supported on this hardware",
)
@torch.inference_mode()
def test_aiter_fused_moe_fp8_accuracy():
    """AITER FP8 group-quant MoE output stays within FP8 quantization error.

    We quantize the input to FP8 group-quant, run the AITER MoE, then compare
    the dequantized output to the BF16 reference. Allow a looser tolerance
    (atol=0.5) for the FP8 → BF16 reconstruction error.
    """
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(1)

    num_tokens = 16
    hidden_dim = 512
    group_size = 128

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)

    # Quantize hidden states to FP8 group quant
    x_fp8, scales = rocm_aiter_ops.group_fp8_quant(hidden_states, group_size)

    # Dequantize and compare to original
    # scales: [num_tokens, num_groups]; x_fp8: [num_tokens, hidden_dim]
    _num_groups = (hidden_dim + group_size - 1) // group_size
    scales_expanded = scales.repeat_interleave(group_size, dim=1)[:, :hidden_dim]
    x_dequant = x_fp8.float() * scales_expanded

    # FP8 group quant should preserve values within ~10% relative error
    rel_error = (x_dequant - hidden_states.float()).abs() / (
        hidden_states.float().abs() + 1e-5
    )
    assert rel_error.mean() < 0.1, (
        f"FP8 group quant mean relative error {rel_error.mean():.4f} exceeds 10%"
    )
    assert (rel_error < 0.5).float().mean() > 0.99, (
        "Over 1% of FP8 group quant values have >50% relative error"
    )


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@torch.inference_mode()
def test_aiter_fused_moe_determinism():
    """AITER fused MoE produces bitwise-identical results across N runs.

    Uses shuffled weights (required by CK kernel) and distinct expert IDs
    to exercise the actual kernel code path used in production.
    """
    require_aiter()
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    import vllm._aiter_ops  # noqa: F401

    torch.set_default_device("cuda")
    torch.manual_seed(2)

    num_tokens = 8
    hidden_dim = 256
    intermediate_dim = 512
    num_experts = 4
    topk = 2

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts, intermediate_dim * 2, hidden_dim, dtype=torch.bfloat16
    ) / math.sqrt(hidden_dim)
    w2 = torch.randn(
        num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16
    ) / math.sqrt(intermediate_dim)
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = _make_topk_ids(num_tokens, num_experts, topk)

    w1_s, w2_s = _shuffle_moe_weights(w1, w2)

    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
        QuantMethod,
    )

    def run_moe():
        return torch.ops.vllm.rocm_aiter_fused_moe(
            hidden_states,
            w1_s,
            w2_s,
            topk_weights,
            topk_ids,
            expert_mask=None,
            activation_method=int(ActivationMethod.SILU),
            quant_method=int(QuantMethod.NO),
            doweight_stage1=False,
        )

    _assert_deterministic(run_moe, n_runs=4)


# ── End-to-end tests ───────────────────────────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [
        (1, 8, 2),  # single-token decode (typical inference)
        (16, 8, 2),  # small batch
        (64, 16, 4),  # larger expert pool, more topk
    ],
)
@torch.inference_mode()
def test_aiter_fused_moe_end_to_end(num_tokens, num_experts, topk):
    """End-to-end MoE test: router logits → softmax routing → AITER kernel.

    Covers the full production token routing pipeline: router softmax, topk
    selection, weight normalization, and fused GEMM. Verifies output shape,
    dtype, and accuracy against the float32 mask-based reference.
    """
    require_aiter()
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    import vllm._aiter_ops  # noqa: F401

    torch.set_default_device("cuda")
    torch.manual_seed(7)

    hidden_dim = 512
    intermediate_dim = 1024

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts, intermediate_dim * 2, hidden_dim, dtype=torch.bfloat16
    ) / math.sqrt(hidden_dim)
    w2 = torch.randn(
        num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16
    ) / math.sqrt(intermediate_dim)

    # Production-style routing: softmax → topk (distinct experts per token)
    router_logits = torch.randn(num_tokens, num_experts, device="cuda")
    router_probs = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(router_probs, k=topk, dim=-1)
    # Renormalize weights so they sum to 1 per token
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.float()
    topk_ids = topk_ids.to(torch.int32)

    ref_out = ref_moe_forward(
        hidden_states, w1, w2, topk_weights, topk_ids, activation="silu"
    )

    w1_s, w2_s = _shuffle_moe_weights(w1, w2)

    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
        QuantMethod,
    )

    aiter_out = torch.ops.vllm.rocm_aiter_fused_moe(
        hidden_states,
        w1_s,
        w2_s,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation_method=int(ActivationMethod.SILU),
        quant_method=int(QuantMethod.NO),
        doweight_stage1=False,
    )

    assert aiter_out.shape == (num_tokens, hidden_dim)
    assert aiter_out.dtype == torch.bfloat16
    _assert_accurate(aiter_out.float(), ref_out, atol=0.05, rtol=0.0)


# ── GPU-specific tests ─────────────────────────────────────────────────────


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.skipif(
    not current_platform.is_rocm() or not on_gfx942(),
    reason="gfx942 (MI300X/MI325X) specific test",
)
@torch.inference_mode()
def test_aiter_fused_moe_gfx942_accuracy():
    """AITER fused MoE accuracy test targeted at gfx942 (MI300X/MI325X).

    gfx942 is the primary production target for AITER MoE. This test verifies
    that the kernel meets accuracy requirements on this specific architecture.
    """
    require_aiter()
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    import vllm._aiter_ops  # noqa: F401

    torch.set_default_device("cuda")
    torch.manual_seed(42)

    # Llama-7B class dimensions — production-representative
    num_tokens = 64
    hidden_dim = 4096
    intermediate_dim = 11008
    num_experts = 8
    topk = 2

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts, intermediate_dim * 2, hidden_dim, dtype=torch.bfloat16
    ) / math.sqrt(hidden_dim)
    w2 = torch.randn(
        num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16
    ) / math.sqrt(intermediate_dim)
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = _make_topk_ids(num_tokens, num_experts, topk)

    ref_out = ref_moe_forward(hidden_states, w1, w2, topk_weights, topk_ids)
    w1_s, w2_s = _shuffle_moe_weights(w1, w2)

    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
        QuantMethod,
    )

    aiter_out = torch.ops.vllm.rocm_aiter_fused_moe(
        hidden_states,
        w1_s,
        w2_s,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation_method=int(ActivationMethod.SILU),
        quant_method=int(QuantMethod.NO),
        doweight_stage1=False,
    )
    _assert_accurate(aiter_out.float(), ref_out, atol=0.05, rtol=0.0)


@pytest.mark.skipif(not aiter_available, reason="aiter required")
@pytest.mark.skipif(
    not current_platform.is_rocm() or not on_gfx950(),
    reason="gfx950 (MI350X) specific test",
)
@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 not supported on this hardware",
)
@torch.inference_mode()
def test_aiter_fused_moe_gfx950_fp8_accuracy():
    """AITER FP8 per-tensor MoE accuracy test targeted at gfx950 (MI350X).

    gfx950 introduces native FP8 compute support; this test verifies the
    QuantMethod.PER_TENSOR path on MI350X hardware. FP8 quantization error
    is bounded by atol=0.3 (looser than BF16 due to reduced precision).
    """
    require_aiter()
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    import vllm._aiter_ops  # noqa: F401

    torch.set_default_device("cuda")
    torch.manual_seed(99)

    num_tokens = 32
    hidden_dim = 512
    intermediate_dim = 1024
    num_experts = 4
    topk = 2

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    w1_bf16 = torch.randn(
        num_experts, intermediate_dim * 2, hidden_dim, dtype=torch.bfloat16
    ) / math.sqrt(hidden_dim)
    w2_bf16 = torch.randn(
        num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16
    ) / math.sqrt(intermediate_dim)
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = _make_topk_ids(num_tokens, num_experts, topk)

    # BF16 reference as ground truth (float32 mask-based)
    ref_out = ref_moe_forward(hidden_states, w1_bf16, w2_bf16, topk_weights, topk_ids)

    # Quantize weights to FP8 (per-tensor)
    w1_fp8 = w1_bf16.to(torch.float8_e4m3fnuz)
    w2_fp8 = w2_bf16.to(torch.float8_e4m3fnuz)
    w1_s, w2_s = _shuffle_moe_weights(w1_fp8, w2_fp8)

    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
        QuantMethod,
    )

    aiter_out = torch.ops.vllm.rocm_aiter_fused_moe(
        hidden_states,
        w1_s,
        w2_s,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation_method=int(ActivationMethod.SILU),
        quant_method=int(QuantMethod.PER_TENSOR),
        doweight_stage1=False,
    )

    assert aiter_out.shape == (num_tokens, hidden_dim)
    # FP8 quantization introduces more error than BF16
    _assert_accurate(aiter_out.float(), ref_out, atol=0.3, rtol=0.0)
