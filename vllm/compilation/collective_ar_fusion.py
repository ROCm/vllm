# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .inductor_pass import enable_fake_mode
from .matcher_utils import MatcherFusedAddRMSNorm
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

FP8_DTYPE = current_platform.fp8_dtype()
logger = init_logger(__name__)
if hasattr(torch.ops._C, "scaled_fp4_quant"):
    STATIC_FP4_QUANT_OP = torch.ops._C.scaled_fp4_quant.default


from vllm import _custom_ops as ops

# from aiter.dist.communication_op import (
#     tensor_model_parallel_all_reduce,
#     tensor_model_parallel_fused_allreduce_rmsnorm,
# )


class BasePattern:
    def __init__(self, dtype: torch.dtype, device: str):
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()


class AllReduceFusedAddRMSNormPattern(BasePattern):
    """
    This pattern replaces the allreduce + rms norm (with residual)
    with fused flashinfer implementation.
    Applies to o_proj + rmsnorm after attn and mlp + rmsnorm before attn.
    """

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str):
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(epsilon)

    def get_inputs(self):
        input, residual, weight = self.rmsnorm_matcher.inputs()

        # input goes through allreduce first, always 16-bit
        return [residual, input.to(self.dtype), weight]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor):
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms, residual = self.rmsnorm_matcher(allreduce_output, weight, residual)
            return rms, residual

        def replacement(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ):
            logger.info("AllReduceFusedAddRMSNormPattern called!!")
            group = get_tp_group()
            rank = group.rank_in_group
            world_size = group.world_size
            workspace = group.device_communicator.caf_comm.get_workspace().to(
                input.device
            )
            norm_out = torch.empty_like(input)
            residual_out = torch.empty_like(residual)
            ops.allreduce_rms_fusion(
                rank,
                world_size,
                residual,
                input,
                weight,
                residual_out,
                norm_out,
                self.epsilon,
                workspace,
            )

            # allreduce_in, residual
            return norm_out, residual_out

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllReduceFusionPass(VllmPatternMatcherPass):
    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.disabled = True
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size <= 1:
            return
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="all_reduce_fusion_pass"
        )
        if config.model_config is None:
            return
        self.hidden_dim = config.model_config.get_hidden_size()
        self.group = get_tp_group().device_group
        rank = get_tensor_model_parallel_rank()
        self.register_patterns()
        self.dump_patterns(config, self.patterns)

    @enable_fake_mode
    def register_patterns(self):
        for epsilon in [1e-5, 1e-6]:
            AllReduceFusedAddRMSNormPattern(
                epsilon,
                self.model_dtype,
                self.device,
            ).register(self.patterns)

            # WARNING: This is a hack to clear the pattern matcher cache
            # and allow multiple values of epsilon.
            torch._inductor.pattern_matcher._seen_patterns.clear()

        self.disabled = False

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        if self.disabled:
            logger.info("AllReduceFusionPass disabled")
            return

        self.matched_count = self.patterns.apply(graph)
        logger.info("Replaced %s patterns", self.matched_count)
