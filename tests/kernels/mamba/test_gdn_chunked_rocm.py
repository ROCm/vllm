# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the RDNA3.5 gated delta net kernel (csrc/rocm/gdn_chunked.cu)."""

import contextlib
import math

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        reason="gdn_chunked is a ROCm kernel.",
        allow_module_level=True,
    )

from vllm.platforms.rocm import on_gfx1150, on_gfx1151  # noqa: E402

if not (on_gfx1150() or on_gfx1151()):
    pytest.skip(
        reason="gdn_chunked is RDNA3.5 (gfx1150/gfx1151) only.",
        allow_module_level=True,
    )

import vllm.model_executor.layers.fla.ops.chunk_rocm as chunk_rocm  # noqa: E402
from vllm.model_executor.layers.fla.ops import (  # noqa: E402
    chunk_gated_delta_rule,
)

HEAD_DIM = 128


def test_op_is_registered():
    """The op must be present on gfx115x, not merely skipped.

    Without this, a build that omitted the kernel would fall back to Triton and
    every test below would still pass.
    """
    assert hasattr(torch.ops, "_rocm_C")
    assert hasattr(torch.ops._rocm_C, "gdn_chunked")


@contextlib.contextmanager
def _accelerated_paths_disabled():
    """Force ``chunk_gated_delta_rule`` onto the Triton kernels."""
    saved_hip = chunk_rocm._ENABLED
    chunk_rocm._ENABLED = False
    chunk_rocm._available.cache_clear()
    saved_fused = None
    try:
        import vllm.model_executor.layers.fla.ops.chunk_fused as chunk_fused

        saved_fused = chunk_fused._ENABLED
        chunk_fused._ENABLED = False
    except ImportError:
        chunk_fused = None
    try:
        yield
    finally:
        chunk_rocm._ENABLED = saved_hip
        chunk_rocm._available.cache_clear()
        if chunk_fused is not None:
            chunk_fused._ENABLED = saved_fused


def _make_inputs(seq_lens, num_k_heads, gqa_ratio, state_dtype=torch.float32):
    num_v_heads = num_k_heads * gqa_ratio
    num_seqs = len(seq_lens)
    cu_seqlens = torch.zeros(num_seqs + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = torch.tensor(seq_lens, device="cuda", dtype=torch.int32).cumsum(0)
    total = int(cu_seqlens[-1].item())
    dtype = torch.bfloat16

    # q and k reach the kernel l2-normalised, and the conditioning of (I + A)
    # depends on it.
    q = F.normalize(
        torch.randn(1, total, num_k_heads, HEAD_DIM, device="cuda", dtype=dtype),
        p=2,
        dim=-1,
    )
    k = F.normalize(torch.randn_like(q), p=2, dim=-1)
    v = torch.randn(1, total, num_v_heads, HEAD_DIM, device="cuda", dtype=dtype)
    a = torch.randn(1, total, num_v_heads, device="cuda", dtype=dtype)
    b = torch.randn_like(a)

    # Upstream FLA GatedDeltaNet synthetic initialisation.
    A = torch.empty(num_v_heads, device="cuda", dtype=torch.float32).uniform_(0, 16)
    A_log = torch.log(A)
    dt = torch.exp(
        torch.rand(num_v_heads, device="cuda", dtype=torch.float32)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    g = -A_log.exp().view(1, 1, num_v_heads) * F.softplus(
        a.float() + dt_bias.view(1, 1, num_v_heads)
    )
    beta = torch.sigmoid(b.float())
    initial_state = (
        torch.randn(
            num_seqs,
            num_v_heads,
            HEAD_DIM,
            HEAD_DIM,
            device="cuda",
            dtype=state_dtype,
        )
        * 0.05
    )
    return q, k, v, g, beta, initial_state, cu_seqlens


def _rel_rms(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative RMS error, scaled by the tensor as a whole."""
    got32, ref32 = got.float(), ref.float()
    denom = max(ref32.pow(2).mean().sqrt().item(), 1e-9)
    return ((got32 - ref32).pow(2).mean().sqrt() / denom).item()


def _run_both(q, k, v, g, beta, initial_state, cu_seqlens):
    common = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    with _accelerated_paths_disabled():
        ref = chunk_gated_delta_rule(initial_state=initial_state.clone(), **common)
    got = chunk_gated_delta_rule(initial_state=initial_state.clone(), **common)
    return got, ref


@pytest.mark.parametrize("gqa_ratio", [1, 2, 4])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [512],
        [32, 33, 64, 1],  # pins the 32-token chunk boundary
        [1, 300, 64, 65, 200],
        [137, 1, 941, 512],
    ],
)
@torch.inference_mode()
def test_matches_triton_chain(gqa_ratio, seq_lens):
    inputs = _make_inputs(seq_lens, num_k_heads=8, gqa_ratio=gqa_ratio)
    (o, state), (ref_o, ref_state) = _run_both(*inputs)

    # Both paths are bf16 end to end, so they agree only to bf16 precision.
    assert _rel_rms(o, ref_o) < 2e-2
    assert _rel_rms(state, ref_state) < 2e-2


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_initial_state_dtypes(state_dtype):
    """The wrapper casts the incoming state to fp32."""
    inputs = _make_inputs(
        [300, 64], num_k_heads=8, gqa_ratio=2, state_dtype=state_dtype
    )
    (o, state), (ref_o, ref_state) = _run_both(*inputs)
    assert state.dtype == torch.float32
    assert state.shape == ref_state.shape
    assert _rel_rms(o, ref_o) < 2e-2
    assert _rel_rms(state, ref_state) < 2e-2


@torch.inference_mode()
def test_core_attn_out_aliasing():
    """``out`` is a view into a caller-owned buffer; it must not overrun it."""
    from vllm.model_executor.layers.fla.ops.chunk_rocm import chunk_gdn_hip_fwd

    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        [300], num_k_heads=8, gqa_ratio=2
    )
    scale = HEAD_DIM**-0.5
    ref_o, _ = chunk_gdn_hip_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )

    slack = 1024
    buf = torch.full((v.numel() + slack,), float("nan"), device="cuda", dtype=v.dtype)
    o, _ = chunk_gdn_hip_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        core_attn_out=buf,
    )
    assert torch.equal(o.reshape(-1), buf[: v.numel()])
    assert torch.isnan(buf[v.numel() :]).all(), "kernel wrote past the buffer"
    torch.testing.assert_close(o, ref_o)


@torch.inference_mode()
def test_opcheck():
    """Schema, aliasing annotations and fake implementation agree."""
    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        [300], num_k_heads=8, gqa_ratio=2
    )
    out = torch.empty_like(v).squeeze(0)
    final_state = q.new_empty(1, v.shape[-2], HEAD_DIM, HEAD_DIM, dtype=torch.float32)
    torch.library.opcheck(
        torch.ops._rocm_C.gdn_chunked,
        (
            q.squeeze(0),
            k.squeeze(0),
            v.squeeze(0),
            g.squeeze(0).float(),
            beta.squeeze(0).float(),
            initial_state.float().contiguous(),
            cu_seqlens.to(torch.int32),
            out,
            final_state,
            HEAD_DIM**-0.5,
        ),
    )


@torch.inference_mode()
def test_declines_unsupported():
    """The gate must decline every case the kernel asserts on."""
    from vllm.model_executor.layers.fla.ops.chunk_rocm import is_hip_gdn_supported

    q, _, v, _, _, _, cu_seqlens = _make_inputs([64], num_k_heads=8, gqa_ratio=2)

    assert is_hip_gdn_supported(q, v, cu_seqlens)
    assert not is_hip_gdn_supported(q, v, None)
    assert not is_hip_gdn_supported(q.half(), v.half(), cu_seqlens)
    assert not is_hip_gdn_supported(q[..., :64], v[..., :64], cu_seqlens)
    # value heads not a multiple of key heads
    assert not is_hip_gdn_supported(q[:, :, :3], v[:, :, :5], cu_seqlens)
