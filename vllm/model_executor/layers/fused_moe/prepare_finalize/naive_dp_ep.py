# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.flashinfer import nvfp4_block_scale_interleave


def _quantize_and_setup_dispatch(
    a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor | None]:
    # Defer input quantization to the MoE kernel.
    if defer_input_quant:
        a1q = a1
        a1q_scale = None
    else:
        input_sf = (
            quant_config.a1_gscale
            if quant_config.use_nvfp4_w4a4
            else quant_config.a1_scale
        )

        # NOTE: swizzling pads the scales to multiple of 128
        # which makes the scales tensor different shape than
        # the hidden states, breaking the A2A kernel. So, we
        # delay the swizzling until after the A2A.
        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            input_sf,
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
            is_scale_swizzled=False,
            mx_alignment=quant_config.mx_alignment,
        )

    # Skip gathering scales if we have static quantization
    # (the scale is a scalar, replicated on all ranks) or
    # if quantization is deferred.
    skip_gather_scales = a1q_scale is None or a1q_scale.ndim == 0
    scales = None if skip_gather_scales else [a1q_scale]

    return a1q, scales, a1q_scale


def _check_act_scale_is_dispatchable(fused_experts: mk.FusedMoEExperts) -> None:
    """Refuse a scheme whose activation scale this dispatch cannot carry.

    `prepare` appends the activation scale to the tensors handed to
    `all_gatherv`, which sizes every one of them by the *token* count of the
    rank that contributed it. That holds for a per-token or block scale, which
    has a row per token, and for a calibrated per-tensor scale, which is a 0-dim
    scalar and is skipped. A dynamic per-tensor scale is neither: it is one
    element, computed from this rank's rows alone, so the gather is being asked
    to treat a single amax as a rank's worth of tokens.

    Unevenly loaded ranks then die inside `_all_gather_single` on
    `1 != <this rank's token count>`, after the model has loaded and captured
    graphs. Evenly loaded ranks are worse: `all_gatherv` drops `sizes` when
    every rank contributes the same count, which skips that assertion, and the
    ranks' amaxes are concatenated into a `[world_size]` vector whose first
    element the kernel then applies to everyone's tokens. Refuse here, where
    the scheme is visible, rather than in either of those places.

    Widening the skip in `_quantize_and_setup_dispatch` is not the fix. It would
    leave each rank quantizing the gathered batch with its own local amax, which
    is the silent-wrong-output case above.
    """
    if fused_experts.expects_unquantized_inputs or (
        fused_experts.moe_config is not None
        and fused_experts.moe_config.is_lora_enabled
    ):
        # The experts quantize for themselves, so `prepare` dispatches
        # unquantized activations and produces no scale to gather.
        #
        # `expects_unquantized_inputs` alone is not usable here: it reads
        # `_lora_context`, which `FusedMoEWithLoRA.set_mapping` installs after
        # this hook runs in FusedMoEKernelModular's constructor, so a LoRA
        # layer would be refused on a scheme its experts quantize themselves.
        # `is_lora_enabled` is a config field, settled before any of this.
        return

    quant_config = fused_experts.quant_config
    if quant_config is None or not quant_config.is_dynamic_per_tensor_act:
        return

    raise ValueError(
        f"A dynamic per-tensor {quant_config.quant_dtype} activation scheme "
        f"cannot be served by the AllGather+ReduceScatter MoE dispatch. Its "
        f"scale is one element per rank, computed from that rank's own tokens, "
        f"but the dispatch all-gathers it alongside the activations and sizes "
        f"every tensor by each rank's token count. Use a checkpoint with "
        f"calibrated activation scales, a per-token or block-quantized scheme, "
        f"or an all2all backend that carries the scale itself "
        f"(--all2all-backend=deepep_high_throughput)."
    )


def _unwrap_scale_and_prepare_for_moe(
    scales: list[torch.Tensor] | None,
    quant_config: FusedMoEQuantConfig,
) -> torch.Tensor:
    assert scales is not None and len(scales) == 1
    a1q_scale = scales[0]
    # Apply swizzling after a2a if the MoE kernel needs it.
    if quant_config.quant_dtype == "nvfp4" and quant_config.is_scale_swizzled:
        assert a1q_scale is not None
        if a1q_scale.element_size() == 1:
            a1q_scale = a1q_scale.view(torch.uint8)
        a1q_scale = nvfp4_block_scale_interleave(a1q_scale)

    return a1q_scale


class MoEPrepareAndFinalizeNaiveDPEPModular(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the topk weights and ids.
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers
        # Set by FusedMoEWithLoRA.set_mapping() when LoRA is active. When
        # present, prepare() dispatches the per-token LoRA mapping alongside
        # hidden_states and writes the gathered result back to the context so
        # experts can use the per-rank-local mapping.
        self._lora_context = None

    def set_lora_context(self, ctx) -> None:
        self._lora_context = ctx

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts) -> None:
        _check_act_scale_is_dispatchable(fused_experts)

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        """Quantize and Dispatch Topk Weights and Topk Ids."""

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        # When LoRA is active, dispatch the per-token LoRA id along with
        # hidden_states so every rank receives the correct mapping for the
        # tokens it ends up processing. The punica_wrapper stores indices as
        # int64 but the moe_lora_align_block_size kernel expects int32, so
        # pull the pre-cast view from token_mapping_meta.
        lora_ctx = self._lora_context
        local_token_lora_mapping = None
        if lora_ctx is not None:
            local_token_lora_mapping = (
                lora_ctx.punica_wrapper.token_mapping_meta.token_lora_mapping[
                    : a1.shape[0]
                ]
            )

        extra_tensors: list[torch.Tensor] | None = None
        if scales is not None:
            extra_tensors = list(scales)
        if local_token_lora_mapping is not None:
            if extra_tensors is None:
                extra_tensors = []
            extra_tensors.append(local_token_lora_mapping)

        res = get_ep_group().dispatch(
            a1q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

        if extra_tensors is None:
            assert len(res) == 3
            a1q, topk_weights, topk_ids = res
            a1q_scale = a1q_scale_orig
        else:
            assert len(res) == 4
            a1q, topk_weights, topk_ids, gathered_extras = res
            gathered_extras = list(gathered_extras)
            if local_token_lora_mapping is not None:
                dispatched_lora_mapping = gathered_extras.pop()
                assert lora_ctx is not None
                lora_ctx.local_token_lora_mapping = dispatched_lora_mapping
            if scales is not None:
                a1q_scale = _unwrap_scale_and_prepare_for_moe(
                    gathered_extras, quant_config
                )
            else:
                a1q_scale = a1q_scale_orig

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        out = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        output.copy_(
            get_ep_group().combine(out, is_sequence_parallel=self.is_sequence_parallel)
        )


class MoEPrepareAndFinalizeNaiveDPEPMonolithic(mk.FusedMoEPrepareAndFinalizeMonolithic):
    """
    Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the router logits (the MoE kernel runs the router internally).
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts) -> None:
        _check_act_scale_is_dispatchable(fused_experts)

    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareMonolithicResultType:
        """Quantize and Dispatch Router Logits."""

        a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        res = get_ep_group().dispatch_router_logits(
            a1q,
            router_logits,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )

        if scales is None:
            assert len(res) == 2
            a1q, router_logits = res
            a1q_scale = a1q_scale_orig
        else:
            assert len(res) == 3
            a1q, router_logits, scales = res
            a1q_scale = _unwrap_scale_and_prepare_for_moe(scales, quant_config)

        return a1q, a1q_scale, router_logits

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        out = get_ep_group().combine(
            fused_expert_output, is_sequence_parallel=self.is_sequence_parallel
        )
        return out


def make_moe_prepare_and_finalize_naive_dp_ep(
    use_monolithic: bool,
    is_sequence_parallel: bool = False,
    num_dispatchers: int = 1,
) -> MoEPrepareAndFinalizeNaiveDPEPModular | MoEPrepareAndFinalizeNaiveDPEPMonolithic:
    return (
        MoEPrepareAndFinalizeNaiveDPEPMonolithic(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
        if use_monolithic
        else MoEPrepareAndFinalizeNaiveDPEPModular(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
    )
