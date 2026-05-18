# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""

import os
from collections.abc import Callable

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _tiny_dot_kernel(
        x_ptr, w_ptr, out_ptr, M, K, BLOCK: tl.constexpr, APPLY_SIGMOID: tl.constexpr
    ):
        """One program per output scalar (one row of M).  Each program
        loads its K-vector of x, the shared K-vector of w, computes the
        dot, optionally applies sigmoid, stores out[pid].  Supports
        M >= 1 — the original 1-token shared_expert_gate uses M=1, the
        MTP-verify path uses M=2..5 (one row per speculative token)."""
        pid = tl.program_id(0)
        if pid >= M:
            return
        offsets = tl.arange(0, BLOCK)
        mask = offsets < K
        x = tl.load(x_ptr + pid * K + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = tl.sum(x * w, axis=0)
        if APPLY_SIGMOID:
            acc = 1.0 / (1.0 + tl.exp(-acc))
        # tl.store auto-casts acc (fp32) to the dtype of out_ptr;
        # passing a bf16/fp16 ptr eliminates the post-kernel aten::copy_
        # that otherwise fires once per layer per decode token.
        tl.store(out_ptr + pid, acc)

    def _tiny_dot_triton(
        x_flat: torch.Tensor,
        w_flat: torch.Tensor,
        apply_sigmoid: bool = False,
        M: int = 1,
    ) -> torch.Tensor:
        """Compute out[i] = dot(x[i,:], w_flat) for i in [0, M).  Returns
        a 0-D scalar tensor when M==1 (legacy shared_expert_gate path)
        or a 1-D [M] tensor when M>1 (MTP-verify shared_expert_gate path).
        Both call sites in qwen2_moe.py reshape to the expected output
        layout."""
        K = w_flat.numel()
        BLOCK = triton.next_power_of_2(K)
        out_shape = () if M == 1 else (M,)
        out = torch.empty(out_shape, dtype=x_flat.dtype, device=x_flat.device)
        _tiny_dot_kernel[(M,)](
            x_flat, w_flat, out, M=M, K=K, BLOCK=BLOCK, APPLY_SIGMOID=apply_sigmoid
        )
        return out
except ImportError:
    _tiny_dot_triton = None  # type: ignore[assignment]


def tiny_sigmoid_dot(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """sigmoid((x.flatten() * weight.flatten()).sum()) in one Triton kernel.

    Replaces the eager 3-launch chain (aten::mul + aten::sum + aten::sigmoid)
    used for Qwen MoE shared_expert_gate where the gate is
    `Linear(hidden, 1)` and the post-call is sigmoid.  Falls back to the
    eager chain when Triton or the input shape isn't supported.
    Handles M>=1; for M>1 (the MTP-verify path), returns a 1-D [M]
    tensor of sigmoid(dot).
    """
    K = weight.numel()
    if _tiny_dot_triton is None or K > 4096:
        return torch.sigmoid(
            (x.reshape(-1, K) * weight.reshape(-1)).sum(dim=-1, dtype=x.dtype)
        )
    x_2d = x.reshape(-1, K).contiguous()
    M = x_2d.size(0)
    return _tiny_dot_triton(
        x_2d.reshape(-1), weight.reshape(-1).contiguous(), apply_sigmoid=True, M=M
    )


MOE_LAYER_ROUTER_GATE_SUFFIXES = {
    "gate",
    "router",
    "router_gate",
    "shared_expert_gate",
    "expert_gate",
}


def is_layer_moe_router_gate(prefix: str) -> bool:
    if not prefix:
        return False
    return prefix.rsplit(".", 1)[-1] in MOE_LAYER_ROUTER_GATE_SUFFIXES


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts
        are padded to the maximum prompt length within the batch using
        `vocab_size` as the padding value. The value `vocab_size` is used
        for padding because it does not correspond to any valid token ID
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties

    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)


def use_aiter_triton_gemm(n, m, k, dtype):
    if (
        not rocm_aiter_ops.is_triton_gemm_enabled()
        # MI300's - fp8nuz=True
        or current_platform.is_fp8_fnuz()
        or dtype not in [torch.float16, torch.bfloat16]
    ):
        return False

    # use hipblaslt for the larger GEMMs
    if n > 2048 and m > 512:
        return False
    return (
        (m == 5120 and k == 2880)
        or (m == 2880 and k == 4096)
        or (m == 128 and k == 2880)
        or (m == 640 and k == 2880)
        or (m == 2880 and k == 512)
    )


def rocm_unquantized_gemm_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    from vllm.platforms.rocm import on_gfx1x, on_gfx9, on_gfx950

    n = x.numel() // x.size(-1)
    m = weight.shape[0]
    k = weight.shape[1]

    cu_count = num_compute_units()

    # Next ^2 of n
    N_p2 = 1 << (n - 1).bit_length()
    # With 64 Ms per CU (each of 4 SIMDs working on a 16x16 tile),
    # and each working on a 512-shard of K, how many CUs would we need?
    rndup_cus = ((m + 64 - 1) // 64) * ((k + 512 - 1) // 512)
    # How many of 4 waves in a group can work on same 16 Ms at same time?
    # This reduces the Ms each group works on, i.e. increasing the number of CUs needed.
    GrpsShrB = min(N_p2 // 16, 4)
    # Given the above, how many CUs would we need?
    CuNeeded = rndup_cus * GrpsShrB
    # candidate for atomic reduce count splitk?
    fits_wvsplitkrc = (
        N_p2 * m * ((k + 512 - 1) // 512)
    ) <= 128 * 1024 * 12  # deterministic
    fits_wvsplitkrc &= CuNeeded <= cu_count

    use_skinny_reduce_counting = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and on_gfx950()
        and x.dtype in [torch.float16, torch.bfloat16]
        and (
            10 <= n <= 128
            and k % 8 == 0
            and k > 512
            and m % 16 == 0
            and fits_wvsplitkrc
            and weight.is_contiguous()
        )
    )
    if use_skinny_reduce_counting:
        x_view = x.reshape(-1, x.size(-1))
        with record_function_or_nullcontext(f"wvSplitKrc {n}x{m}x{k}"):
            out = ops.wvSplitKrc(weight, x_view, cu_count, bias)
        return out.reshape(*x.shape[:-1], weight.shape[0])

    if use_aiter_triton_gemm(n, m, k, x.dtype):
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

        return gemm_a16w16(x, weight, bias)

    use_skinny = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and (on_gfx9() or on_gfx1x())
        and x.dtype in [torch.float16, torch.bfloat16]
        and k % 8 == 0
    )

    # Tiny scalar projection (e.g. Qwen MoE shared_expert_gate): hipBLASLt
    # runs a 64x96x32 (or with MTP-verify M=3: 128x32x16) macro tile with
    # SplitK + post-pass even though the output per row is a single scalar.
    # Replace with one Triton program per row that does the dot in a single
    # block.  Original trigger was M=1 N=1 (the decode shared_expert_gate);
    # extended to small M when N=1 to also cover the MTP-verify path
    # (M = num_speculative_tokens + 1) where the same shared_expert_gate
    # Linear runs on the verified-token batch.
    # Naming note: in this function `n` is BATCH (input rows) and `m`
    # is the weight's output-feature count, so `m == 1` is the
    # 1-output-scalar-per-row gate and `n <= 8` covers small batches
    # (decode n=1, MTP-verify n=num_speculative_tokens+1).
    if (
        os.environ.get("VLLM_DISABLE_TINY_DOT_GEMM", "0") != "1"
        and envs.VLLM_ROCM_USE_SKINNY_GEMM
        and m == 1
        and n <= 8
        and x.dtype in [torch.float16, torch.bfloat16]
    ):
        # Triton fast path: a single-block fused (x*w).sum kernel collapses
        # the eager 2-launch chain (aten::mul + aten::sum) into one Triton
        # launch + one aten::copy_ cast.  Cudagraph replay sees fewer
        # dispatches per layer per decode token.  Default ON; revert via
        # VLLM_DISABLE_TINY_DOT_TRITON=1.  Restricted to K<=4096 because
        # the kernel is a single-block reduction (BLOCK = next_pow2(K)).
        # Correctness covered by tests/kernels/test_tiny_dot_triton.py.
        if (
            os.environ.get("VLLM_DISABLE_TINY_DOT_TRITON", "0") != "1"
            and bias is None
            and k <= 4096
        ):
            with record_function_or_nullcontext(f"DOT {n}x{m}x{k} [tk]"):
                # `n` here is batch (input rows); `m` is the weight
                # output count and is 1 in this branch.  Pass `n` as
                # M to the Triton kernel which launches n programs
                # (one per input row).
                x_2d = x.reshape(-1, k).contiguous()
                w_flat = weight.reshape(-1).contiguous()
                out = _tiny_dot_triton(
                    x_2d.reshape(-1), w_flat, apply_sigmoid=False, M=n
                )
                return out.reshape(*x.shape[:-1], 1)
        with record_function_or_nullcontext(f"DOT {n}x{m}x{k}"):
            x_2d = x.reshape(-1, k)
            w_flat = weight.reshape(-1)
            out = (x_2d * w_flat).sum(dim=-1, dtype=x.dtype)
            if bias is not None:
                out = out + bias.reshape(-1)[0]
            return out.reshape(*x.shape[:-1], 1)

    if not use_skinny:
        with record_function_or_nullcontext(f"BLAS {n}x{m}x{k}"):
            return torch.nn.functional.linear(x, weight, bias)

    x_view = x.reshape(-1, x.size(-1))
    if m > 8 and 0 < n <= 4:
        cu_count = num_compute_units()
        with record_function_or_nullcontext(f"wvSplitK {n}x{m}x{k}"):
            out = ops.wvSplitK(weight, x_view, cu_count, bias)
        return out.reshape(*x.shape[:-1], weight.shape[0])
    elif m % 4 == 0 and n == 1 and k <= 8192 and bias is None:
        with record_function_or_nullcontext(f"LLMM1 {n}x{m}x{k}"):
            out = ops.LLMM1(weight, x_view, 4)
        return out.reshape(*x.shape[:-1], weight.shape[0])

    with record_function_or_nullcontext(f"BLAS {n}x{m}x{k}"):
        return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def rocm_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm",
    op_func=rocm_unquantized_gemm_impl,
    fake_impl=rocm_unquantized_gemm_fake,
)


def check_cpu_sgl_kernel(n: int, k: int, dtype: torch.dtype) -> bool:
    return (
        torch.cpu._is_amx_tile_supported()
        and (dtype in (torch.bfloat16, torch.int8))
        and k % 32 == 0
        and n % 16 == 0
    )


def dispatch_cpu_unquantized_gemm(
    layer: torch.nn.Module,
    remove_weight: bool,
) -> None:
    # skip for missing layers
    if layer.weight.is_meta:
        layer.cpu_linear = torch.nn.functional.linear
        return

    N, K = layer.weight.size()
    dtype = layer.weight.dtype

    # Zen CPU path: zentorch_linear_unary with optional eager weight prepacking.
    if current_platform.is_zen_cpu() and hasattr(
        torch.ops.zentorch, "zentorch_linear_unary"
    ):
        zen_weight = layer.weight.detach()
        is_prepacked = False

        if envs.VLLM_ZENTORCH_WEIGHT_PREPACK and hasattr(
            torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
        ):
            zen_weight = torch.ops.zentorch.zentorch_weight_prepack_for_linear(
                zen_weight
            )
            is_prepacked = True

        layer.cpu_linear = lambda x, weight, bias, _p=is_prepacked: (
            torch.ops.zentorch.zentorch_linear_unary(
                x, zen_weight, bias, is_weight_prepacked=_p
            )
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        return

    if envs.VLLM_CPU_SGL_KERNEL and check_cpu_sgl_kernel(N, K, dtype):
        packed_weight = torch.ops._C.convert_weight_packed(layer.weight)
        if getattr(layer, "bias", None) is not None:
            bias_f32 = layer.bias.to(torch.float32)
        else:
            bias_f32 = None
        layer.cpu_linear = lambda x, weight, bias: torch.ops._C.weight_packed_linear(
            x, packed_weight, bias_f32 if bias is not None else None, True
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        return
    elif (
        ops._supports_onednn
        and current_platform.get_cpu_architecture() != CpuArchEnum.POWERPC
    ):
        try:
            origin_weight = layer.weight
            handler = ops.create_onednn_mm(origin_weight.t(), 32)
            layer.cpu_linear = lambda x, weight, bias: ops.onednn_mm(handler, x, bias)
            if remove_weight:
                layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            return
        except RuntimeError as e:
            logger.warning_once(
                "Failed to create oneDNN linear, fallback to torch linear."
                f" Exception: {e}"
            )

    # fallback case
    layer.cpu_linear = lambda x, weight, bias: torch.nn.functional.linear(
        x, weight, bias
    )


def cpu_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return layer.cpu_linear(x, weight, bias)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    else:
        return default_unquantized_gemm
