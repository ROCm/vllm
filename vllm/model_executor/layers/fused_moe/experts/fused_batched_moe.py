# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused batched MoE kernel."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe import batched_activation, dynamic_quant
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
    normalize_batched_scales_shape,
    swiglu_limit_func,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    group_broadcast,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton, use_tensor_descriptor
from vllm.triton_utils.allocation import set_triton_allocator


def _is_capturing_or_compiling() -> bool:
    # torch.cuda.is_current_stream_capturing() is unavailable on non-CUDA (XPU) torch.
    return torch.compiler.is_compiling() or (
        current_platform.is_cuda_alike() and torch.cuda.is_current_stream_capturing()
    )


@triton.jit
def moe_mmk(
    a_ptrs,
    b_ptrs,
    K,
    expert_id,
    a_scale_ptr,
    b_scale_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # Offsets and masks
    offs_m,
    offs_n,
    offs_bn,
    mask_m,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
    use_w8a8: tl.constexpr,
    use_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    # TD: a_base_ptr/b_base_ptr are the expert/CTA-offset bases of A[M,K]/B[N,K].
    a_base_ptr=None,
    b_base_ptr=None,
    M=0,
    N=0,
    stride_am: tl.int64 = 0,
    stride_bn: tl.int64 = 0,
    USE_TD: tl.constexpr = False,
):
    offs_k = tl.arange(0, BLOCK_K)

    if USE_TD:
        # make_tensor_descriptor requires the last (K) stride to be a
        # compile-time 1; the launcher only enables USE_TD for K-contiguous A/B.
        a_desc = tl.make_tensor_descriptor(
            a_base_ptr,
            shape=[M, K],
            strides=[stride_am, 1],
            block_shape=[BLOCK_M, BLOCK_K],
        )
        b_desc = tl.make_tensor_descriptor(
            b_base_ptr,
            shape=[N, K],
            strides=[stride_bn, 1],
            block_shape=[BLOCK_N, BLOCK_K],
        )
    if use_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + expert_id * stride_bse + offs_n[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = b_scale_ptr + offs_bsn * stride_bsn

        # per act token
        elif per_act_token_quant:
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=mask_m, other=0.0)[:, None]

            b_scale_ptrs = b_scale_ptr + offs_bn[None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)

        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if USE_TD:
            # B is [N, K]; tile is [BLOCK_N, BLOCK_K], transposed for dot.
            a = a_desc.load([0, k * BLOCK_K])
            b = tl.trans(b_desc.load([0, k * BLOCK_K]))
        else:
            a = tl.load(
                a_ptrs,
                mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # We accumulate along the K dimension.
        if use_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=mask_m, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                # acc used to enable fp8_fast_accum
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if use_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    return accumulator


@triton.jit
def expert_triton_kernel(
    a_ptr,  # [max_tokens, K]
    b_ptr,  # [K, N]
    c_ptr,  # [max_tokens, N]
    expert_id,
    compute_type: tl.constexpr,
    # Dimensions
    M,
    N,
    K,
    # Quantization data
    a_scale_ptr,
    b_scale_ptr,
    b_zp_ptr,
    # strides
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # offsets
    offs_bn,
    # Blockwise quantization data
    group_n,
    group_k,
    # Quantization schemes
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_TD: tl.constexpr = False,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) % N
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    # Make grids of a + b pointers
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = moe_mmk(
        a_ptrs,
        b_ptrs,
        K,
        expert_id,
        a_scale_ptr,
        b_scale_ptr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_ak,
        stride_bk,
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Offsets and masks
        offs_m,
        offs_n,
        offs_bn,
        mask_m,
        # Block size for block-wise quantization
        group_n,
        group_k,
        # Meta-parameters
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
        a_ptr,
        b_ptr,
        M,
        N,
        stride_am,
        stride_bn,
        USE_TD,
    )

    # store in C
    offs_cn = tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = mask_m[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def batched_triton_kernel(
    a_ptr,  # [E, max_num_tokens, K]
    b_ptr,  # [E, K, N]
    c_ptr,  # [E, max_num_tokens, N]
    expert_num_tokens,  # [E]
    compute_type: tl.constexpr,
    # Dimensions
    max_num_tokens,
    K,
    N,
    # Quantization data
    a_scale_ptr,
    b_scale_ptr,
    b_zp_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ae: tl.int64,
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_be: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_ce: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # Blockwise quantization data
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Quantization schemes
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_TD: tl.constexpr = False,
):
    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        # Early exit
        return

    # axis 1 is M_blocks * N_blocks
    pid_mn = tl.program_id(axis=1)
    # num_pid_m = tl.cdiv(max_num_tokens, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    cta_m_start = pid_m * BLOCK_M
    cta_n_start = pid_n * BLOCK_N
    if cta_m_start >= e_num_tokens:
        # Early exit
        return

    cta_m_size = min(BLOCK_M, e_num_tokens - cta_m_start)
    cta_n_size = min(BLOCK_N, N - cta_n_start)

    a_ptr = a_ptr + expert_id * stride_ae + cta_m_start * stride_am
    b_ptr = b_ptr + expert_id * stride_be + cta_n_start * stride_bn
    c_ptr = (
        c_ptr
        + expert_id * stride_ce
        + cta_m_start * stride_cm
        + cta_n_start * stride_cn
    )

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N

    if use_fp8_w8a8:
        a_scale_ptr = a_scale_ptr + expert_id * stride_ase
        b_scale_ptr = b_scale_ptr + expert_id * stride_bse

        # block-wise
        if group_k > 0 and group_n > 0 or per_act_token_quant:
            a_scale_ptr = a_scale_ptr + cta_m_start * stride_asm

    expert_triton_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        expert_id,
        compute_type,
        cta_m_size,  # M
        cta_n_size,  # N
        K,  # K
        a_scale_ptr,
        b_scale_ptr,
        b_zp_ptr,
        # Strides
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # offsets
        offs_bn,
        # Blockwise quantization data
        group_n,
        group_k,
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
        # Kernel config
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        USE_TD,
    )


def invoke_moe_batched_triton_kernel(
    A: torch.Tensor,  # [E, max_tokens, K]
    B: torch.Tensor,  # [E, N, K]
    C: torch.Tensor,  # [E, max_tokens, N]
    expert_num_tokens: torch.Tensor,  # [E]
    compute_type: tl.dtype,
    # Quantization data
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor,
    # Quantization schemes
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    config: dict[str, int],
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
):
    assert not use_int4_w4a16
    max_num_tokens = A.size(1)
    K = A.size(2)
    N = C.size(2)

    BLOCK_M = config["BLOCK_SIZE_M"]
    BLOCK_N = config["BLOCK_SIZE_N"]
    BLOCK_K = config["BLOCK_SIZE_K"]
    if block_shape is not None:
        # `moe_mmk` loads one scale per K tile, at `k_start // group_k`, and
        # applies it to the whole tile -- so a K tile wider than the
        # quantization group silently uses the first group's scale for every
        # group it spans. `invoke_fused_moe_triton_kernel` clamps identically.
        BLOCK_K = min(BLOCK_K, min(block_shape[0], block_shape[1]))

    grid = (
        expert_num_tokens.size(0),
        triton.cdiv(max_num_tokens, BLOCK_M) * triton.cdiv(B.size(1), BLOCK_N),
    )

    A_scale = normalize_batched_scales_shape(A_scale, expert_num_tokens.shape[0])

    if B_scale is not None and B_scale.ndim == 1:
        assert B_scale.numel() == expert_num_tokens.shape[0]
        B_scale = B_scale.view(-1, 1, 1)

    assert A_scale is None or A_scale.ndim == 3, (
        f"{0 if A_scale is None else A_scale.shape}"
    )
    assert B_scale is None or B_scale.ndim == 1 or B_scale.ndim == 3, (
        f"{0 if B_scale is None else B_scale.shape}"
    )
    if use_fp8_w8a8 or use_int8_w8a16:
        assert B_scale is not None
        # A block-quantized weight scale has one entry per (N, K) group. The
        # kernel indexes it without checking, so a mismatch here is silent
        # wrong arithmetic rather than a fault -- the same class of defect as
        # the stride bug below.
        assert block_shape is None or (
            triton.cdiv(B.size(-2), block_shape[0]) == B_scale.size(-2)
            and triton.cdiv(B.size(-1), block_shape[1]) == B_scale.size(-1)
        ), f"{B.shape} and {block_shape} do not agree with {B_scale.shape}"
    else:
        assert A_scale is None, f"unquantized launch with an A scale {A_scale.shape}"
        assert B_scale is None, f"unquantized launch with a B scale {B_scale.shape}"

    if B_scale is not None:
        stride_bse = B_scale.stride(0)
        # `moe_mmk` indexes the weight scale as
        # `b_scale[expert * stride_bse + offs_bn * stride_bsn]`, unmasked, with
        # `offs_bn` spanning N. A per-tensor scale is 1-D `[E]`, viewed as
        # `[E, 1, 1]` above, whose contiguous strides are all 1 -- so that read
        # runs up to N elements past an E-element tensor. A size-1 dimension is
        # a broadcast and its stride must be 0.
        stride_bsn = B_scale.stride(1) if B_scale.size(1) > 1 else 0
        stride_bsk = B_scale.stride(2) if B_scale.size(2) > 1 else 0
    else:
        stride_bse = 0
        stride_bsk = 0
        stride_bsn = 0

    if A_scale is not None:
        stride_ase = A_scale.stride(0)
        stride_asm = A_scale.stride(1)
        stride_ask = A_scale.stride(2)
    else:
        stride_ase = 0
        stride_asm = 0
        stride_ask = 0

    use_td = (
        use_tensor_descriptor()
        and A.stride(2) == 1
        and B.stride(2) == 1
        and (K * A.element_size()) % 16 == 0
        and (BLOCK_M & (BLOCK_M - 1)) == 0
        and (BLOCK_N & (BLOCK_N - 1)) == 0
        and (BLOCK_K & (BLOCK_K - 1)) == 0
    )
    if use_td and K % BLOCK_K != 0:
        # Mirrored from `invoke_fused_moe_triton_kernel`, which documents a TD
        # gather feeding `tl.dot` at a non-block-aligned K miscompiling on real
        # hardware (~74% of output elements wrong). Not reproduced here -- on
        # gfx950, K=1040 with BLOCK_K=32 gives the same answer with TD on and
        # off -- but it is a compiler bug, the fallback costs only the TD path
        # in a case the alignment gate already lets through, and the dense
        # launcher's note was written against hardware this branch has none of.
        use_td = False
    if use_td:
        set_triton_allocator(A.device)

    batched_triton_kernel[grid](
        A,
        B,
        C,
        expert_num_tokens,
        compute_type,
        # Dimensions
        max_num_tokens,
        K,
        N,
        # Quantization data
        A_scale,
        B_scale,
        B_zp,
        # Strides
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Blockwise quantization data
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
        # Kernel config
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        USE_TD=use_td,
    )


class NaiveBatchedExperts(mk.FusedMoEExpertsModular):
    """
    A reference MoE expert class that operates on expert batched format,
    i.e. E x max_num_tokens x K.  This is the format that the batched
    dispatch/combine kernels use.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        assert not self.quant_config.use_int8_w8a8, "NYI"
        assert not self.quant_config.use_int8_w8a16, "NYI"
        assert not self.quant_config.use_int4_w4a16, "NYI"
        assert self.quant_config.ocp_mx_scheme is None, "NYI"

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device() -> bool:
        raise NotImplementedError(
            "NaiveBatchedExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        raise NotImplementedError(
            "NaiveBatchedExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError(
            "NaiveBatchedExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        raise NotImplementedError(
            "NaiveBatchedExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        raise NotImplementedError(
            "NaiveBatchedExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert self.num_dispatchers is not None
        assert self.max_num_tokens is not None
        num_dp = self.num_dispatchers
        num_experts = local_num_experts
        workspace13 = (num_experts, self.max_num_tokens * num_dp, K)
        workspace2 = (self.max_num_tokens * num_dp, N)
        output = workspace13
        return (workspace13, workspace2, output)

    def dequant(self, t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        assert self.quant_config.is_quantized
        f32 = torch.float32
        if self.quant_config.is_per_act_token or self.quant_config.is_per_tensor:
            return t.to(f32) * scale
        else:
            return t.to(f32) * group_broadcast(scale, t.shape)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert hidden_states.dim() == 3
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        num_local_experts = w1.size(0)
        assert num_local_experts == w1.size(0), f"{num_local_experts} == {w1.size(0)}"

        N = w1.size(1) // 2

        for expert in range(num_local_experts):
            # Indexing expert_num_tokens doesn't work w/cudagraphs or inductor
            if _is_capturing_or_compiling():
                num = hidden_states.shape[1]
            else:
                num = int(expert_num_tokens[expert].item())

            if num == 0:
                continue

            tmp = _resize_cache(workspace2, (num, N))

            if self.quant_config.is_quantized:
                assert a1q_scale is not None and self.w1_scale is not None
                input = self.dequant(hidden_states[expert, :, :], a1q_scale[expert])
                w1_dq = self.dequant(w1[expert], self.w1_scale[expert])
                input = input[:num] @ w1_dq.transpose(0, 1)
            else:
                input = hidden_states[expert, :num, :] @ w1[expert].transpose(0, 1)

            self.activation(activation, tmp, input.to(tmp.dtype))

            if self.quant_config.is_quantized:
                assert self.w2_scale is not None
                w2_dq = self.dequant(w2[expert], self.w2_scale[expert])
            else:
                w2_dq = w2[expert]

            output[expert, :num, :] = tmp @ w2_dq.transpose(0, 1).to(tmp.dtype)


def batched_moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    num_tokens: int,
    E: int,
    N: int,
    expert_num_tokens: torch.Tensor,
    qtype: torch.dtype | None,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # One implementation for eager and for capture.  Placed above the capture
    # check on purpose: routing it inside the branch would preserve the
    # divergence it exists to remove.
    #
    # The capture branch below ignores `expert_num_tokens` and takes one amax
    # over the whole [E, max_num_tokens, N] buffer.  Rows past an expert's
    # delivered count were written by no producer this step, and under
    # cudagraphs the buffer is a persistent graph-pool allocation, so they hold
    # the previous replay's contents.  The eager branch bounds the reduction
    # correctly but only by paying an `.item()` host sync per expert, which is
    # why it cannot be captured -- so the two branches computed different
    # scales from the same inputs, and only one of them was right.
    if (
        qtype is not None
        and A_scale is None
        and not per_act_token_quant
        and block_shape is None
        and A.ndim == 3
    ):
        return dynamic_quant.dynamic_quantize(
            A,
            qtype,
            granularity=dynamic_quant.PER_EXPERT,
            mask_mode=dynamic_quant.MASK_DELIVERED,
            expert_num_tokens=expert_num_tokens,
        )

    if _is_capturing_or_compiling():
        # Note: this does a bunch of extra work because expert_num_tokens is
        # ignored but it does support torch.compile + cudagraphs.
        hidden_dim = A.size(-1)
        assert A_scale is None or A_scale.ndim <= 2, (
            f"{A_scale.shape if A_scale is not None else None}"
        )
        A_q, A_q_scale = moe_kernel_quantize_input(
            A.view(-1, hidden_dim), A_scale, qtype, per_act_token_quant, block_shape
        )
        A_q = A_q.view(E, -1, hidden_dim)
        A_q_scale = normalize_batched_scales_shape(A_q_scale, E)

        return A_q, A_q_scale
    elif qtype is None:
        return A, normalize_batched_scales_shape(A_scale, E)
    else:
        A_q = torch.empty_like(A, dtype=qtype)

        if per_act_token_quant:
            assert block_shape is None
            scale_shape = (E, num_tokens, 1)
        elif block_shape is not None:
            _, block_k = block_shape
            k_tiles = (A.shape[-1] + block_k - 1) // block_k
            scale_shape = (E, num_tokens, k_tiles)
        else:
            scale_shape = (E, 1, 1)

        A_q_scale = torch.zeros(scale_shape, dtype=torch.float32, device=A.device)

        num_experts = expert_num_tokens.numel()

        A_scale = normalize_batched_scales_shape(A_scale, num_experts)

        for e in range(E):
            num_tokens = int(expert_num_tokens[e].item())
            if num_tokens > 0:
                if A_scale is not None:
                    scales = A_scale[e, : min(num_tokens, A_scale.shape[1])]
                else:
                    scales = None
                A_q[e, :num_tokens], tmp_scale = moe_kernel_quantize_input(
                    A[e, :num_tokens],
                    scales,
                    qtype,
                    per_act_token_quant,
                    block_shape,
                )
                assert tmp_scale is not None
                A_q_scale[e, : tmp_scale.shape[0]] = tmp_scale

        return A_q, A_q_scale


class BatchedTritonExperts(mk.FusedMoEExpertsModular):
    """
    A Triton based MoE expert class that operates on expert batched format,
    i.e. E x max_num_tokens x K.  This is the format that the batched
    dispatch/combine kernels use.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        assert not self.quant_config.use_int8_w8a8, "NYI"
        assert not self.quant_config.use_int8_w8a16, "NYI"
        assert not self.quant_config.use_int4_w4a16, "NYI"
        assert self.quant_config.ocp_mx_scheme is None, "NYI"

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike() or current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        p = current_platform
        if p.is_rocm():
            from vllm.platforms.rocm import get_cdna_version

            _rocm_support_fp8 = get_cdna_version() > 2
        else:
            _rocm_support_fp8 = False

        device_supports_fp8 = _rocm_support_fp8 or (
            p.is_cuda() and p.has_device_capability((8, 9))
        )

        supported: list[tuple[QuantKey | None, QuantKey | None]] = [(None, None)]
        # No batch-invariance predicate here: all five pairs are admitted under
        # `VLLM_BATCH_INVARIANT`, each measured individually, and
        # `_supports_batch_invariance` below records the numbers. They were
        # withheld until the unmasked out-of-bounds read of the weight scale
        # was found -- and note what the first diagnosis of *that* got wrong,
        # because it is the trap in this file: the kernel replayed
        # deterministically on captured inputs, which read as "the GEMM is
        # fine, the quantization around it is not". The varying input was
        # memory that is not an argument at all.
        #
        # Do not tidy (kFp8StaticTensorSym, kFp8DynamicTokenSym) away as
        # unreachable. It is true that no in-tree method emits it --
        # compressed-tensors rejects the combination outright, and Quark,
        # ModelOpt and both online paths pair per-tensor weights with
        # per-tensor activations -- but it is the form
        # (kFp8StaticTensorSym, kFp8DynamicTensorSym) *executes in* under the
        # mode, because `maybe_promote_act_quant_for_batch_invariance` returns
        # `with_per_token_act_quant()`: per-token activations against a weight
        # scale left per-tensor. So this row is where that pair's evidence
        # lives, and it is also the shape the launcher's weight-scale stride
        # was wrong for. The oracle currently asks with the pre-promotion keys,
        # so deleting this entry would not break selection today; it would
        # break the moment anything queries the promoted config, and it would
        # orphan the measurement either way.
        if device_supports_fp8:
            supported += [
                (kFp8Static128BlockSym, kFp8Dynamic128Sym),
                (kFp8StaticChannelSym, kFp8DynamicTokenSym),
                (kFp8StaticTensorSym, kFp8DynamicTokenSym),
                (kFp8StaticTensorSym, kFp8StaticTensorSym),
                (kFp8StaticTensorSym, kFp8DynamicTensorSym),
            ]
        return (weight_key, activation_key) in supported

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL,
            MoEActivation.GELU_TANH_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    @staticmethod
    def _supports_batch_invariance() -> bool:
        """Unquantized and all five fp8 pairs `_supports_quant_scheme` lists.

        This is the only experts class reachable with
        `FusedMoEActivationFormat.BatchedExperts` on ROCm, so it is what
        decides whether DeepEP low latency can be brought up under the mode.
        It runs a different kernel from `fused_moe_kernel` on an
        `E x max_num_tokens x K` layout, so nothing measured for the plain
        expert GEMM carries over and all of the below was measured directly.

        `batched_triton_kernel` has no split-K, no `tl.atomic_*` and no
        `@triton.autotune` anywhere in this file; each CTA owns a disjoint
        `[BLOCK_M, BLOCK_N]` tile and runs the whole K loop into one fp32
        accumulator. The grid is keyed on `max_num_tokens` -- the dispatch
        buffer's size, a deployment constant -- not on the runtime token count,
        so token r always lands in tile `r // BLOCK_M`. The batch reaches the
        kernel only through `mask_m` and two early exits.

        Measured on gfx950, E=8/K=1024/N=512, over bf16 (exponent spread +-20),
        bf16 flat, fp16 (+-14) and fp32 flat:

          * 41 launch configurations -- each block size varied alone, a
            BLOCK_K x num_warps cross, and extreme tiles down to 16x16 with 8
            warps -- gave exactly 1 bitwise result per shape.
          * 17 token counts from 1 to 256, straddling both BLOCK_M boundaries,
            plus uneven per-expert counts: no row changed.
          * A per-expert row derangement (68% of rows changing tile) moved no
            bits once un-permuted. All2all dispatch assigns slots by atomic
            increment, so this one is load bearing.
          * Whole class through `BatchedPrepareAndFinalize`: 0 of 2426 rows
            moved across 14 batch sizes, both by appending tokens and by
            dropping them from the front (~1750 slot relocations).

        The fp8 pairs were measured the same way, on the same device, per
        scheme -- four distinct scale layouts reach `moe_mmk`, since it
        branches on the scale shapes and not on the pair:

          * 16 identical calls gave 1 result for each. This is the arm that
            used to fail, and it comes first: a path that cannot repeat itself
            cannot be batch invariant, whatever the batch does.
          * 9 to 13 launch configurations per scheme gave 1 bitwise result.
            The blockwise scheme is scoped to the reachable BLOCK_K, which the
            mode pins at 32 for every M: it applies the scales to each K chunk
            *inside* the loop, so the K tile decides how the scaled partials
            group, and BLOCK_K 16/32/64/128 give four correct but different
            answers. The other three are invariant across the whole grid.
          * 17 token counts from 1 to 256, and a row derangement moving 70% of
            rows to another tile with the activation scale travelling with
            them: no row changed.
          * Whole class through `BatchedPrepareAndFinalize`: 0 of 2346 rows
            over 14 batch sizes and 6 front-drops (2179 slot relocations), in
            both branches of `batched_moe_kernel_quantize_input` -- the
            per-expert loop and the whole-buffer one that cudagraphs and
            torch.compile take.
          * Control, since four of the five schemes could not have been batch
            variant here whatever the kernel did: the fifth,
            (kFp8StaticTensorSym, kFp8DynamicTensorSym), has an activation
            amax over the batch and moves 661 of those same 2346 rows with the
            mode off. With it on,
            `maybe_promote_act_quant_for_batch_invariance` makes that scale
            per-token and it moves none.

        Non-vacuity, since an exactly-summable operand set cannot detect a
        reordering: forward, reverse and split-K fp32 reductions over the same
        products disagreed bitwise in every arm, and the same comparisons
        without the un-permute (kernel) or without the slot remap (class)
        reported differences on exactly the rows that moved. fp8 needs that
        guard more than bf16 does, not less -- two e4m3 values multiply to
        seven significand bits, and 1024 of those products sum *exactly* in
        fp32 unless the operands' exponents are deliberately spread.

        Not measured: no fp8 MoE end to end over DeepEP LL (the arm in
        `test_ep_all2all_batch_invariant.py` is bf16, which is what OLMoE is),
        and no int8 or MX scheme -- the constructor above asserts those NYI.
        """
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        **kwargs,
    ) -> None:
        gemm1_clamp_limit = self.quant_config.gemm1_clamp_limit
        if activation == MoEActivation.SILU and gemm1_clamp_limit is not None:
            swiglu_limit_func(output, input, float(gemm1_clamp_limit))
            return

        super().activation(activation, output, input)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert self.num_dispatchers is not None
        assert self.max_num_tokens is not None
        num_dp = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = self.max_num_tokens
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (num_experts, max_num_tokens * num_dp, max(K, N))
        workspace2 = (num_experts, max_num_tokens * num_dp, activation_out_dim)
        output = (num_experts, max_num_tokens * num_dp, K)
        return (workspace13, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]
        assert expert_tokens_meta is not None

        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        E, max_num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        assert w1.size(0) == E
        assert w2.size(0) == E

        config_dtype = self.quant_config.config_name(hidden_states.dtype)

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            config_dtype,
            max_num_tokens,
            block_shape=self.block_shape,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif hidden_states.dtype == current_platform.fp8_dtype():
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # We can reuse the memory between these because by the time we need
        # cache3, we're done with cache1
        intermediate_cache1 = _resize_cache(workspace13, (E, max_num_tokens, N))
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace2, (E, max_num_tokens, activation_out_dim)
        )

        a1q_scale = normalize_batched_scales_shape(a1q_scale, E)

        # MM1
        invoke_moe_batched_triton_kernel(
            A=hidden_states,
            B=w1,
            C=intermediate_cache1,
            expert_num_tokens=expert_num_tokens,
            compute_type=compute_type,
            A_scale=a1q_scale,
            B_scale=self.w1_scale,
            B_zp=self.w1_zp,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            config=config,
            per_act_token_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
        )

        # Neither cache is zero-filled.  `intermediate_cache1.fill_(0)` used to
        # run above under fp8, and it was not defensive: it was what kept the
        # padded rows finite for a a2 amax that spanned the whole buffer.  That
        # reduction is now bounded by `expert_num_tokens`, so the fill has
        # nothing left to protect and the elementwise work below can skip the
        # rows MM1 never wrote -- 80-90% of the buffer at a realistic live
        # fraction.
        #
        # These two changes are not separable.  A pad-aware activation leaves
        # the rows it skips holding whatever the shared workspace held, so
        # making the producer pad-aware while the consumer still reduces over
        # every row recreates the exact defect the bound removes.
        if (
            activation == MoEActivation.SILU
            and batched_activation.silu_mul_batched_is_exact(intermediate_cache2.dtype)
            and intermediate_cache1.dtype == intermediate_cache2.dtype
        ):
            batched_activation.silu_mul_batched(
                intermediate_cache2, intermediate_cache1, expert_num_tokens
            )
        else:
            self.activation(
                activation,
                intermediate_cache2.view(-1, activation_out_dim),
                intermediate_cache1.view(-1, N),
            )

        qintermediate_cache2, a2q_scale = batched_moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            max_num_tokens,
            E,
            N,
            expert_num_tokens,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
        )

        invoke_moe_batched_triton_kernel(
            A=qintermediate_cache2,
            B=w2,
            C=output,
            expert_num_tokens=expert_num_tokens,
            compute_type=compute_type,
            A_scale=a2q_scale,
            B_scale=self.w2_scale,
            B_zp=self.w2_zp,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            config=config,
            per_act_token_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
        )
