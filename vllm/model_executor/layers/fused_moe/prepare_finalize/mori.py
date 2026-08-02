# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mori
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using MoRI kernels.
    """

    def __init__(
        self,
        mori_op: mori.ops.EpDispatchCombineOp,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
    ):
        super().__init__()
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        # MoRI's own view of the dispatched expert ids, handed straight back to
        # combine. See prepare() for why the experts get a different tensor.
        self._combine_topk_ids: torch.Tensor | None = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self):
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return False

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
        """
        Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        - Optional ExpertTokensMetadata containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True now."
        )
        scale = None
        # When defer_input_quant is True, the expert kernel handles
        # quantization internally, so skip FP8 dispatch quantization.
        if self.use_fp8_dispatch and not defer_input_quant:
            from aiter import QuantType, get_hip_quant

            if quant_config.is_block_quantized:
                quant_func = get_hip_quant(QuantType.per_1x128)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())
            elif quant_config.is_per_act_token:
                quant_func = get_hip_quant(QuantType.per_Token)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, expert_num_tokens_cpu=None
        )

        # `dispatch_ids` is a view of MoRI's receive buffer, sized
        # `max_num_tokens_to_recv()` = EP size x max_num_batched_tokens and
        # written only for the rows that actually arrived. The tail is never
        # cleared, so it holds the previous step's expert ids -- which are in
        # range, so `moe_align_block_size` counts them as real tokens,
        # `num_tokens_post_padded` never shrinks below the whole buffer and
        # every experts backend that derives its work from `topk_ids` runs the
        # full padded M on every step.
        #
        # Marking the tail invalid is enough to fix that: the align kernel
        # skips ids outside [0, num_experts) (`get_local_expert_id`), so they
        # drop out of the per-expert counts, and `fused_moe_kernel` already
        # loads `num_tokens_post_padded` from device memory and early-exits
        # the blocks past it. The row count is read on device, so this costs
        # no host sync -- which is the whole reason the buffer is fixed-size.
        #
        # The tail rows themselves stay garbage: nothing writes them, so the
        # experts leave uninitialised workspace in those rows of the output.
        # That is already the contract -- combine only reads
        # `totalRecvTokenNum` rows back.
        rows = torch.arange(
            dispatch_ids.size(0), device=dispatch_ids.device, dtype=torch.int32
        )
        expert_topk_ids = torch.where(
            (rows < dispatch_recv_token_num).unsqueeze(1), dispatch_ids, -1
        )
        # Combine keeps MoRI's untouched ids. The intranode kernel ignores them
        # entirely (it routes off `dispDestTokIdMap`), but the internode ones
        # do read `args.tokenIndices`, and feeding those a -1 is a change this
        # single-node work cannot test.
        self._combine_topk_ids = dispatch_ids

        return (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            expert_topk_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        num_token = output.shape[0]
        combine_topk_ids = self._combine_topk_ids
        assert combine_topk_ids is not None, "finalize called without a prepare"
        self._combine_topk_ids = None
        result = self.mori_op.combine(
            fused_expert_output,
            None,
            combine_topk_ids,
        )[0]
        output.copy_(result[:num_token])
