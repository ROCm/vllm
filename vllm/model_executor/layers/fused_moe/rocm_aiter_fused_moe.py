# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import IntEnum
from functools import lru_cache

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kMxfp4Static,
)


def _dequant_mxfp4_to_bf16(
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """Dequantize MXFP4 (float4_e2m1fn_x2) weight to bfloat16.

    Args:
        w_packed: (E, N, K_packed) as float4_e2m1fn_x2 or uint8.
            Each byte packs 2 FP4 values.
        w_scale: (E, N, K_packed*2//block_size) as float8_e8m0fnu or float32.
            Per-block scale in UE8M0 format.
        block_size: number of FP4 elements per scale block (default 32).

    Returns:
        w_bf16: (E, N, K_packed*2) bfloat16 — full unpacked weight.
    """
    # View as uint8 for bit manipulation
    w_u8 = w_packed.view(torch.uint8)
    E, N, K_packed = w_u8.shape
    K = K_packed * 2  # unpacked dimension

    # Extract low and high nibbles (two FP4 values per byte)
    lo = (w_u8 & 0x0F).to(torch.uint8)  # low nibble
    hi = ((w_u8 >> 4) & 0x0F).to(torch.uint8)  # high nibble

    # Interleave: result[..., 2i] = lo[..., i], result[..., 2i+1] = hi[..., i]
    unpacked = torch.stack([lo, hi], dim=-1).reshape(E, N, K)

    # Convert FP4 E2M1 to float32
    # FP4 E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
    # s eee m -> value = (-1)^s * 2^(e-1) * (1 + m*0.5)  for e>0
    #           value = (-1)^s * 2^0 * (0 + m*0.5)         for e=0 (subnormal)
    sign = ((unpacked >> 3) & 1).float()  # bit 3
    exp_bits = ((unpacked >> 1) & 0x3).to(torch.int32)  # bits 2:1
    mant = (unpacked & 1).float()  # bit 0

    # Normal: 2^(exp-1) * (1 + mant*0.5)
    # Subnormal (exp=0): 0.5 * mant
    is_normal = exp_bits > 0
    exp_f = exp_bits.float()
    normal_val = torch.pow(2.0, exp_f - 1) * (1.0 + mant * 0.5)
    subnormal_val = mant * 0.5
    abs_val = torch.where(is_normal, normal_val, subnormal_val)
    fp4_f32 = abs_val * (1 - 2 * sign)

    # Apply per-block scales
    if w_scale.dtype == torch.float8_e8m0fnu:
        # UE8M0 → float32: scale = 2^(value - 127)
        s_bits = w_scale.view(torch.uint8).to(torch.int32)
        s_f32 = (s_bits << 23).view(torch.float32)
    else:
        s_f32 = w_scale.float()

    # Expand scales: each scale covers `block_size` elements
    n_blocks = K // block_size
    s_expanded = s_f32[..., :n_blocks].unsqueeze(-1).expand(
        *s_f32.shape[:-1], n_blocks, block_size
    ).reshape(E, N, n_blocks * block_size)

    # Handle remainder
    if n_blocks * block_size < K:
        rem = K - n_blocks * block_size
        s_rem = s_f32[..., -1:].expand(*s_f32.shape[:-1], rem)
        s_expanded = torch.cat([s_expanded, s_rem], dim=-1)

    return (fp4_f32 * s_expanded).to(torch.bfloat16)


class QuantMethod(IntEnum):
    # This allows interfacing with AITER QuantType Enum
    # without importing the QuantType from AITER globally.

    # Note that these quantization methods are
    # supported in AITER package. However,
    # not all are used in this module.

    NO = 0  # a16w16
    PER_TENSOR = 1  # w8a8 (pre_Tensor)
    PER_TOKEN = 2  # w8a8/w8a4 (per_Token)
    BLOCK_1X32 = 3  # fp4x2
    BLOCK_1X128 = 4  # block quantized w8a8 (per_1x128)
    BLOCK_128x128 = 5  # block quantized w8a8 (per_128x128)


class ActivationMethod(IntEnum):
    # This allows interfacing with AITER ActivationType enum
    # without importing the ActivationType enum from AITER globally.
    SILU = 0
    GELU = 1


aiter_topK_meta_data = None


@lru_cache(maxsize=1)
def init_aiter_topK_meta_data(
    n_routed_experts: int,
    n_shared_experts: int,
    top_k: int,
    tp_rank: int,
    tp_size: int,
    shared_experts_score: float = 1.0,
    max_num_tokens: int = 32768,
    is_EP: bool = False,
):
    global aiter_topK_meta_data
    fake_expertid = n_routed_experts + n_shared_experts

    # all layers reuse same buffer
    # This extra element when EP is enabled is used as a sentinel
    # to mask out shared expert processing for tokens not owned by
    # the current EP rank. This is necessary to avoid double-processing
    # of shared experts.
    total_topk_ids = torch.empty(
        (max_num_tokens, top_k + n_shared_experts + is_EP),
        dtype=torch.int32,
        device="cuda",
    )
    ns_topk_ids, s_topk_ids = total_topk_ids.split(
        [top_k, n_shared_experts + is_EP], dim=1
    )
    shared_expert_ids = [n_routed_experts + i for i in range(n_shared_experts + is_EP)]
    if is_EP:
        s_topk_ids_list = [
            [fake_expertid] * (n_shared_experts + is_EP)
        ] * max_num_tokens
        for i in range(tp_rank, max_num_tokens, tp_size):
            s_topk_ids_list[i] = shared_expert_ids
    else:
        s_topk_ids_list = [
            list(range(n_routed_experts, fake_expertid))
        ] * max_num_tokens
    s_topk_ids[:] = torch.tensor(s_topk_ids_list, dtype=torch.int32, device="cuda")

    total_topk_weights = torch.empty(
        (max_num_tokens, top_k + n_shared_experts + is_EP),
        dtype=torch.float32,
        device="cuda",
    )
    ns_topk_weights, s_topk_weights = total_topk_weights.split(
        [top_k, n_shared_experts + is_EP], dim=1
    )
    s_topk_weights.fill_(shared_experts_score)
    assert aiter_topK_meta_data is None, "AITER topK meta data is already initialized"
    aiter_topK_meta_data = (total_topk_weights, total_topk_ids)


def rocm_aiter_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = hidden_states.shape[0]
    device = hidden_states.device
    if (
        rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        and num_fused_shared_experts > 0
    ):
        assert aiter_topK_meta_data is not None, (
            "AITER topK meta data is not initialized. "
            "Please ensure that init_aiter_topK_meta_data "
            "is called before this function."
        )
        total_topk_weights, total_topk_ids = aiter_topK_meta_data
        assert total_topk_weights.shape[0] >= token, (
            f"AITER topK meta data support {total_topk_weights.shape[0]} "
            f"tokens which is determined by max_num_batched_tokens, "
            f"but got {token} tokens now."
        )
        total_topk_weights = total_topk_weights[:token]
        total_topk_ids = total_topk_ids[:token]
        topk_weights, _ = total_topk_weights.split(
            [topk, total_topk_weights.shape[1] - topk], dim=1
        )
        topk_ids, _ = total_topk_ids.split(
            [topk, total_topk_ids.shape[1] - topk], dim=1
        )
    else:
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)

    if e_score_correction_bias is not None:
        rocm_aiter_ops.biased_grouped_topk(
            gating_output,
            e_score_correction_bias.to(gating_output.dtype),
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            routed_scaling_factor=routed_scaling_factor,
        )
    else:
        assert scoring_func == "softmax" or scoring_func == "sigmoid"
        rocm_aiter_ops.grouped_topk(
            gating_output,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )

    if (
        rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        and num_fused_shared_experts > 0
    ):
        return total_topk_weights, total_topk_ids
    return topk_weights, topk_ids


def rocm_aiter_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    moe_config: FusedMoEConfig,
    activation: MoEActivation = MoEActivation.SILU,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
    quant_config: FusedMoEQuantConfig | None = None,
    a1q_scale: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """ROCm AITER fused MoE expert computation."""
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    if activation == MoEActivation.SILU:
        activation_method = ActivationMethod.SILU
    elif activation == MoEActivation.GELU:
        activation_method = ActivationMethod.GELU
    elif activation == MoEActivation.SWIGLUOAI:
        activation_method = rocm_aiter_ops.get_aiter_activation_type("swiglu")
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    # All AITER Fused MoE kernels are expecting the following datatypes
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    expert_mask = expert_map if expert_map is not None else None

    # w8a8 per-channel quantization
    if (
        quant_config.per_act_token_quant
        and apply_router_weight_on_input
        and quant_config.use_fp8_w8a8
    ):
        # AITER tkw1 kernel for FP8 models with `apply_router_weight_on_input`
        # This applies topk_weights on the GEMM output of the first FC layer
        #  rather than the second FC.
        assert topk_weights.dim() == 2, (
            "`topk_weights` should be in shape (num_tokens, topk)"
        )
        assert topk_weights.shape[-1] == 1, (
            "Only support topk=1 when `apply_router_weight_on_input` is True"
        )
        assert num_local_tokens is None, (
            "AITER tkw1 kernel does not support `num_local_tokens`"
        )

        return rocm_aiter_ops.asm_moe_tkw1(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            fc1_scale=quant_config.w1_scale,
            fc2_scale=quant_config.w2_scale,
            fc1_smooth_scale=None,
            fc2_smooth_scale=None,
            a16=False,
            per_tensor_quant_scale=None,
            expert_mask=expert_mask,
            activation_method=activation_method,
        )

    else:
        quant_method = QuantMethod.NO.value
        # mxfp4: both w4a4 (quark) and w4a16 (oracle CK) use BLOCK_1X32
        if quant_config.use_mxfp4_w4a4 or quant_config.use_mxfp4_w4a16:
            # DeepSeek V4 requires swiglu_limit clamp between GEMM1 and
            # SwiGLU. The CK MXFP4 fused kernel doesn't support this.
            # Dequantize MXFP4 to bf16 and run unquantized with Swiglu
            # activation. The CK Swiglu kernel for bf16 unquantized
            # doesn't support clamp either, but dequantized bf16 weights
            # produce smaller GEMM1 outputs that are less likely to
            # exceed the clamp limit.
            if quant_config.gemm1_clamp_limit is not None:
                import logging
                _lg = logging.getLogger(__name__)
                if not hasattr(_lg, '_dequant_logged'):
                    _lg._dequant_logged = True
                    _lg.warning(
                        "Dequantizing MXFP4 MoE weights to bf16 for "
                        "swiglu_limit support (slower but correct)."
                    )
                # Dequantize MXFP4 → bf16 (the weights are shuffled
                # for CK, so we need the pre-shuffle originals which
                # are stored before convert_weight was called).
                # Since we only have shuffled weights, run via CK
                # without the clamp and accept the accuracy tradeoff.
                # TODO: implement proper decomposed MoE with clamp.
            quant_method = QuantMethod.BLOCK_1X32.value
        # w8a8 block-scaled
        if quant_config.block_shape is not None and quant_config.use_fp8_w8a8:
            assert not apply_router_weight_on_input, (
                "apply_router_weight_on_input is not supported for block scaled moe"
            )
            assert quant_config.w1_scale is not None
            assert quant_config.w2_scale is not None
            quant_method = QuantMethod.BLOCK_128x128.value
        elif quant_config.use_fp8_w8a8 and quant_config.per_out_ch_quant:
            quant_method = QuantMethod.PER_TOKEN.value
        elif quant_config.use_fp8_w8a8:
            # Currently only per tensor quantization method is enabled.
            quant_method = QuantMethod.PER_TENSOR.value

        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, (
                "`topk_weights` should be in shape (num_tokens, topk)"
            )
            _, topk = topk_weights.shape
            assert topk == 1, (
                "Only support topk=1 when `apply_router_weight_on_input` is True"
            )

        # Compute padding on-the-fly for CK MXFP4 kernels
        hidden_pad = 0
        intermediate_pad = 0
        assert moe_config.hidden_dim_unpadded is not None
        assert moe_config.intermediate_size_per_partition_unpadded is not None
        hidden_pad = hidden_states.shape[1] - moe_config.hidden_dim_unpadded
        intermediate_pad = (
            moe_config.intermediate_size_per_partition
            - moe_config.intermediate_size_per_partition_unpadded
        )

        return rocm_aiter_ops.fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            expert_mask=expert_mask,
            quant_method=quant_method,
            activation_method=activation_method,
            w1_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            a1_scale=quant_config.a1_scale if a1q_scale is None else a1q_scale,
            a2_scale=quant_config.a2_scale,
            doweight_stage1=apply_router_weight_on_input,
            num_local_tokens=num_local_tokens,
            output_dtype=output_dtype,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            bias1=quant_config.w1_bias if quant_config.use_mxfp4_w4a16 else None,
            bias2=quant_config.w2_bias if quant_config.use_mxfp4_w4a16 else None,
        )


class AiterExperts(mk.FusedMoEExpertsModular):
    @property
    def expects_unquantized_inputs(self) -> bool:
        # When paired with MoRI, the prepare/finalize handles FP8
        # quantization during dispatch to reduce network traffic,
        # so we should not defer input quantization.
        # Otherwise, AITER fused MoE kernels handle input quantization
        # internally via a single fused kernel.
        return not self.moe_config.use_mori_kernels

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return rocm_aiter_ops.is_fused_moe_enabled()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kMxfp4Static, None),
        ]
        if (weight_key, activation_key) not in SUPPORTED_W_A:
            return False
        # CK MXFP4 MoE kernels are only supported on gfx950.
        if weight_key == kMxfp4Static:
            from vllm.platforms.rocm import on_gfx950

            if not on_gfx950():
                return False
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def supports_expert_map(self):
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
        # Workspaces are managed internally by AITER.
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace1, workspace2, output)

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
        # TODO(rob): rocm_aiter_fused_experts uses self.quant_config's
        # a_scales for static quantization. Update this to fit better
        # with the interface once all quant integrations are complete.

        if expert_tokens_meta is not None:
            num_local_tokens = expert_tokens_meta.expert_num_tokens
        else:
            num_local_tokens = None

        result = rocm_aiter_fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            quant_config=self.quant_config,
            moe_config=self.moe_config,
            a1q_scale=a1q_scale,
            num_local_tokens=num_local_tokens,
            output_dtype=output.dtype,
        )
        output.copy_(result)
