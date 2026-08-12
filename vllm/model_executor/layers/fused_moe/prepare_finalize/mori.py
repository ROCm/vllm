# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mori
import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _trim_topk_ids_kernel(
    src_ptr,  # [num_rows, topk], MoRI's receive buffer view
    dst_ptr,  # [num_rows, topk]
    recv_token_num_ptr,  # device-side scalar row count
    numel,
    topk,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    recv_token_num = tl.load(recv_token_num_ptr)
    ids = tl.load(src_ptr + offsets, mask=mask, other=0)
    delivered = (offsets // topk) < recv_token_num
    tl.store(dst_ptr + offsets, tl.where(delivered, ids, -1), mask=mask)


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

    def _max_rows_to_recv(self, local_num_tokens: int, buffer_rows: int) -> int:
        """Upper bound on the rows MoRI's dispatch can deliver to this rank.

        MoRI sizes its receive buffer for the worst case, `ep_size *
        max_num_batched_tokens`: every token in the step could route an expert
        onto this rank, and the dispatch sends at most *one* row per (token,
        destination rank) pair -- it deduplicates a token's topk before
        claiming a slot, which is why the worst case is not multiplied by topk.
        So the bound is the total tokens across the EP ranks, and the per-step
        bound is that same formula with the step's actual counts substituted
        for the maxima.

        Those counts are host-known, in `num_tokens_across_dp_cpu`. This uses
        `ep_size * max(...)` rather than the sum: the two agree whenever DP
        padding is on, which is whenever cudagraphs are (see
        `_synchronize_dp_ranks`), so the shape stays static per captured graph;
        the max is still an upper bound when padding is off; and it remains one
        under TP and sequence parallelism, where an EP rank carries at most one
        DP rank's worth of rows. It is the same quantity `DeepEPV2` bounds its
        own worst-case buffer with, minus the power-of-two rounding, which
        exists there because DeepEP compiles a kernel per size. MoRI compiles
        one per kernel type, so an odd bound is free and rounding would only
        cost looseness.
        """
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is None:
            rows_per_sender = local_num_tokens
        else:
            rows_per_sender = int(dp_metadata.num_tokens_across_dp_cpu.max())
        # Never zero. A zero would only arise if every DP rank were empty, but
        # it would hand the modular kernel M == 0, a branch MoRI could not
        # previously reach because its M was the whole buffer. One dead row is
        # cheaper than reasoning about that.
        return max(1, min(self.num_dispatchers_ * rows_per_sender, buffer_rows))

    def _verify_recv_bound(
        self, dispatch_recv_token_num: torch.Tensor, recv_rows: int
    ) -> None:
        """Check the trimmed buffer really covers what arrived, and log slack.

        Gated on `VLLM_MORI_VERIFY_RECV_BOUND`, and a check rather than an
        alternate path: the trimming in `prepare` is identical either way, so
        there are not two behaviours to drift apart. Never runs while a
        cudagraph is capturing -- see the call site.

        Exceeding the bound is the one way this can be wrong, and it is silent:
        the experts would simply never write rows past it, and combine would
        read whatever the workspace happened to hold. Nothing raises. So the
        validation runs check it directly rather than inferring it from output
        quality, at the cost of a device sync per layer.

        The ratio is logged because a formula error shows up as slack trending
        to zero before it shows up as a violation. Well under 1.0 is expected
        even when routing is balanced: the bound assumes every peer sends every
        token, while a rank actually receives only the tokens that picked at
        least one of its experts.
        """
        recv = int(dispatch_recv_token_num.item())
        logger.info(
            "MoRI recv bound: %d / %d rows (%.3f of bound)",
            recv,
            recv_rows,
            recv / recv_rows if recv_rows else 0.0,
        )
        assert recv <= recv_rows, (
            f"MoRI delivered {recv} rows but prepare trimmed its receive "
            f"buffer to {recv_rows}. The rows past the bound were never "
            f"computed by the experts and combine will read uninitialised "
            f"workspace for them."
        )

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

        # Hand the experts a prefix of MoRI's worst-case receive buffer rather
        # than the whole thing. `_max_rows_to_recv` explains why the prefix is
        # still an upper bound on what arrived; the slicing is free because the
        # dispatch outputs are views onto MoRI's symmetric buffers, so a row
        # prefix keeps both the pointer and the row stride.
        #
        # `combine` does not see the narrowing. It takes the input pointer and
        # `size(1)` and reads exactly `totalRecvTokenNum` rows, so as long as
        # the bound holds every row it reads is one the experts computed. The
        # ids it gets are left alone entirely -- see below.
        recv_rows = self._max_rows_to_recv(a1.size(0), dispatch_a1.size(0))
        # Read at call time, not at import: the knob is for validation runs and
        # nothing should have to be re-imported to turn it on.
        #
        # Not during capture. The check reads the count back to the host, and a
        # device sync inside a capturing stream aborts the capture. Skipping
        # capture also loses nothing: the rows are dummy, and at *replay* this
        # Python does not run at all, so
        # under cudagraphs the check only ever covers non-captured steps.
        # Validation runs that need it to cover real serving traffic have to
        # ask for `--enforce-eager`.
        if (
            envs.VLLM_MORI_VERIFY_RECV_BOUND
            and not torch.cuda.is_current_stream_capturing()
        ):
            self._verify_recv_bound(dispatch_recv_token_num, recv_rows)
        dispatch_a1 = dispatch_a1[:recv_rows]
        dispatch_weights = dispatch_weights[:recv_rows]
        if dispatch_scale is not None:
            dispatch_scale = dispatch_scale[:recv_rows]
        recv_ids = dispatch_ids[:recv_rows]

        # The receive buffer is sized to a worst-case bound and only the rows
        # that arrived are written, so the tail still holds the previous step's
        # expert ids -- in range, so `moe_align_block_size` counts them as real
        # tokens and every experts backend runs the full padded M every step.
        # Marking the tail invalid drops it: the align kernel skips ids outside
        # [0, num_experts) and `fused_moe_kernel` early-exits past
        # `num_tokens_post_padded`. The row count is read on device, so this
        # costs no host sync, which is why the buffer is fixed-size at all.
        # The tail rows stay uninitialised, as before -- combine only reads
        # `totalRecvTokenNum` rows back. `DeepEPV2PrepareAndFinalize` marks its
        # own tail for the same reason.
        #
        # One kernel rather than an arange plus a broadcast `torch.where`: this
        # runs on every layer of every step.
        if recv_ids.is_contiguous():
            expert_topk_ids = torch.empty_like(recv_ids)
            BLOCK_SIZE = 1024
            numel = recv_ids.numel()
            _trim_topk_ids_kernel[(triton.cdiv(numel, BLOCK_SIZE),)](
                recv_ids,
                expert_topk_ids,
                dispatch_recv_token_num,
                numel,
                recv_ids.size(1),
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            rows = torch.arange(
                recv_ids.size(0), device=recv_ids.device, dtype=torch.int32
            )
            expert_topk_ids = torch.where(
                (rows < dispatch_recv_token_num).unsqueeze(1), recv_ids, -1
            )

        # Combine keeps MoRI's ids, full length and untouched. The intranode
        # kernel ignores them entirely (it routes off `dispDestTokIdMap`), but
        # the internode ones do read `args.tokenIndices`, and neither the -1
        # marking nor the narrowing above is testable on one node.
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
