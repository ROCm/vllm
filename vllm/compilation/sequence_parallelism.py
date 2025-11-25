# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import is_rocm_aiter_rmsnorm_enabled
from vllm.platforms import current_platform

from .inductor_pass import enable_fake_mode
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


class _RMSNormAndQuantOpHelper:
    """Base helper for RMSNorm and RMSNorm + Quantization functionalization."""

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str,
        quant_op: torch._ops.OpOverload | None = None,
        **kwargs,
    ):
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.quant_op = quant_op

    def _functional_rmsnorm(self, result_buffer, input_tensor, weight_tensor):
        return torch.ops.higher_order.auto_functionalized(
            torch.ops._C.rms_norm.default,
            result=result_buffer,
            input=input_tensor,
            weight=weight_tensor,
            epsilon=self.epsilon,
        )

    def _functional_fused_add_rmsnorm(
        self, input_tensor, residual_tensor, weight_tensor
    ):
        return torch.ops.higher_order.auto_functionalized(
            torch.ops._C.fused_add_rms_norm.default,
            input=input_tensor,
            residual=residual_tensor,
            weight=weight_tensor,
            epsilon=self.epsilon,
        )

    def _functional_rmsnorm_then_quant(
        self,
        rmsnorm_result_buffer,
        quant_result_buffer,
        input_tensor,
        weight_tensor,
        scale_tensor,
    ):
        if self.quant_op is None:
            raise RuntimeError(
                "_RMSNormAndQuantOpHelper was not initialized with a quant_op."
            )
        rmsnorm_out_tuple = self._functional_rmsnorm(
            rmsnorm_result_buffer, input_tensor, weight_tensor
        )
        quant_out_tuple = torch.ops.higher_order.auto_functionalized(
            self.quant_op,
            result=quant_result_buffer,
            input=rmsnorm_out_tuple[1],
            scale=scale_tensor,
        )
        return quant_out_tuple

    def _functional_fused_add_rmsnorm_then_quant(
        self,
        quant_result_buffer,
        input_tensor,
        residual_tensor,
        weight_tensor,
        scale_tensor,
    ):
        if self.quant_op is None:
            raise RuntimeError(
                "_RMSNormAndQuantOpHelper was not initialized with a quant_op."
            )
        fused_add_rmsnorm_out_tuple = self._functional_fused_add_rmsnorm(
            input_tensor, residual_tensor, weight_tensor
        )
        quant_out_tuple = torch.ops.higher_order.auto_functionalized(
            self.quant_op,
            result=quant_result_buffer,
            input=fused_add_rmsnorm_out_tuple[1],
            scale=scale_tensor,
        )
        return quant_out_tuple, fused_add_rmsnorm_out_tuple[2]

    # def _aiter_rmsnorm(self, input_tensor, weight_tensor):
    #     """Aiter RMSNorm (non-functional, returns output directly)."""
    #     return torch.ops.vllm.rocm_aiter_rms_norm.default(
    #         x=input_tensor,
    #         weight=weight_tensor,
    #         variance_epsilon=self.epsilon,
    #     )

    # def _aiter_fused_add_rmsnorm(self, input_tensor, residual_tensor, weight_tensor):
    #     """Aiter Fused Add RMSNorm (returns normalized output and residual)."""
    #     return torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add.default(
    #         x=input_tensor,
    #         residual=residual_tensor,
    #         weight=weight_tensor,
    #         variance_epsilon=self.epsilon,
    #     )

    def _aiter_functional_rmsnorm(self, result_buffer, input_tensor, weight_tensor):
        return torch.ops.vllm.rocm_aiter_rms_norm.default(
            input_tensor,
            weight_tensor,
            self.epsilon,
        )

    def _aiter_functional_fused_add_rmsnorm(
        self, input_tensor, residual_tensor, weight_tensor
    ):
        return torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add.default(
            input_tensor,
            residual_tensor,
            weight_tensor,
            self.epsilon,
        )

    def _aiter_functional_rmsnorm_then_quant(
        self,
        rmsnorm_result_buffer,
        quant_result_buffer,
        input_tensor,
        weight_tensor,
        scale_tensor,
    ):
        out_tuple = torch.ops.higher_order.auto_functionalized(
            torch.ops.vllm.rocm_aiter_rmsnorm_fused_dynamic_quant.default,
            out=quant_result_buffer,
            input=input_tensor,
            weight=weight_tensor,
            y_scale=scale_tensor,
            epsilon=self.epsilon,
        )
        return out_tuple

    def _aiter_functional_fused_add_rmsnorm_then_quant(
        self,
        quant_result_buffer,
        input_tensor,
        residual_tensor,
        weight_tensor,
        scale_tensor,
    ):
        out_tuple = torch.ops.higher_order.auto_functionalized(
            torch.ops.vllm.rocm_aiter_rmsnorm_fused_add_dynamic_quant.default,
            out=quant_result_buffer,
            input=input_tensor,
            residual=residual_tensor,
            weight=weight_tensor,
            y_scale=scale_tensor,
            epsilon=self.epsilon,
        )
        return (out_tuple[0], out_tuple[2]), out_tuple[1]



class _SequenceParallelPatternHelper(_RMSNormAndQuantOpHelper):
    """Helper for sequence parallelism patterns."""

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str,
        quant_op: torch._ops.OpOverload | None = None,
        **kwargs,
    ):
        super().__init__(epsilon, dtype, device, quant_op=quant_op, **kwargs)
        self.tp_group = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_reduce(x)

    def _reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.reduce_scatter.default(
            x, dim=0, world_size=self.tp_size, group_name=self.tp_group.unique_name
        )

    def _all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.all_gather.default(
            x, dim=0, world_size=self.tp_size, group_name=self.tp_group.unique_name
        )


class FirstAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    def get_inputs(self):
        input = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        permute = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        arg3_1 = torch.empty([4], device=self.device, dtype=self.dtype)

        return [input, permute, arg3_1]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
            permute: torch.Tensor,
            arg3_1: torch.Tensor,
        ):
            all_reduce = self._all_reduce(input)
            rmsnorm = self._functional_rmsnorm(permute, all_reduce, arg3_1)

            return rmsnorm[1], all_reduce

        def replacement(
            input: torch.Tensor,
            permute: torch.Tensor,
            arg3_1: torch.Tensor,
        ):
            reduce_scatter = self._reduce_scatter(input)

            rmsnorm_result = torch.empty_like(reduce_scatter)
            rmsnorm = self._functional_rmsnorm(rmsnorm_result, reduce_scatter, arg3_1)

            all_gather = self._all_gather(rmsnorm[1])

            return all_gather, reduce_scatter

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class MiddleAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [
            residual,
            mm_1,
            rms_norm_weights,
        ]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                all_reduce, residual, rms_norm_weights
            )
            return rmsnorm[1], rmsnorm[2]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                reduce_scatter, residual, rms_norm_weights
            )
            all_gather = self._all_gather(rmsnorm[1])
            return all_gather, rmsnorm[2]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class LastAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [
            residual,
            mm_1,
            rms_norm_weights,
        ]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                all_reduce, residual, rms_norm_weights
            )
            return rmsnorm[1]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                reduce_scatter, residual, rms_norm_weights
            )
            normalized = self._all_gather(rmsnorm[1])
            return normalized

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


FP8_DTYPE = current_platform.fp8_dtype()


# # ============================================================================
# # Aiter RMSNorm Patterns (ROCm only)
# # ============================================================================


# class FirstAllReduceAiterRMSNormPattern(_SequenceParallelPatternHelper):
#     """Pattern for: AllReduce → rocm_aiter_rms_norm (first layer, no residual add)."""

#     def get_inputs(self):
#         input = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
#         weight = torch.empty([4], device=self.device, dtype=self.dtype)
#         return [input, weight]

#     def register(self, pm_pass: PatternMatcherPass):
#         def pattern(
#             input: torch.Tensor,
#             weight: torch.Tensor,
#         ):
#             all_reduce = self._all_reduce(input)
#             rmsnorm = self._aiter_rmsnorm(all_reduce, weight)
#             return rmsnorm, all_reduce

#         def replacement(
#             input: torch.Tensor,
#             weight: torch.Tensor,
#         ):
#             reduce_scatter = self._reduce_scatter(input)
#             rmsnorm = self._aiter_rmsnorm(reduce_scatter, weight)
#             all_gather = self._all_gather(rmsnorm)
#             return all_gather, reduce_scatter

#         pm.register_replacement(
#             pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
#         )


# class MiddleAllReduceAiterRMSNormPattern(_SequenceParallelPatternHelper):
#     """Pattern for: AllReduce → rocm_aiter_rmsnorm2d_fwd_with_add (middle layers)."""
    
#     def get_inputs(self):
#         mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)
#         residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
#         rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)
#         return [residual, mm_1, rms_norm_weights]

#     def register(self, pm_pass: PatternMatcherPass):
#         def pattern(
#             residual: torch.Tensor,
#             mm_1: torch.Tensor,
#             rms_norm_weights: torch.Tensor,
#         ) -> tuple[torch.Tensor, torch.Tensor]:
#             all_reduce = self._all_reduce(mm_1)
#             rmsnorm, residual_out = self._aiter_fused_add_rmsnorm(
#                 all_reduce, residual, rms_norm_weights
#             )
#             return rmsnorm, residual_out

#         def replacement(
#             residual: torch.Tensor,
#             mm_1: torch.Tensor,
#             rms_norm_weights: torch.Tensor,
#         ) -> tuple[torch.Tensor, torch.Tensor]:
#             reduce_scatter = self._reduce_scatter(mm_1)
#             rmsnorm, residual_out = self._aiter_fused_add_rmsnorm(
#                 reduce_scatter, residual, rms_norm_weights
#             )
#             all_gather = self._all_gather(rmsnorm)
#             return all_gather, residual_out

#         pm.register_replacement(
#             pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
#         )


# class LastAllReduceAiterRMSNormPattern(_SequenceParallelPatternHelper):
#     """Pattern for: AllReduce → rocm_aiter_rmsnorm2d_fwd_with_add (last layer)."""
    
#     def get_inputs(self):
#         mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)
#         residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
#         rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)
#         return [residual, mm_1, rms_norm_weights]

#     def register(self, pm_pass: PatternMatcherPass):
#         def pattern(
#             residual: torch.Tensor,
#             mm_1: torch.Tensor,
#             rms_norm_weights: torch.Tensor,
#         ) -> torch.Tensor:
#             all_reduce = self._all_reduce(mm_1)
#             rmsnorm, _ = self._aiter_fused_add_rmsnorm(
#                 all_reduce, residual, rms_norm_weights
#             )
#             return rmsnorm

#         def replacement(
#             residual: torch.Tensor,
#             mm_1: torch.Tensor,
#             rms_norm_weights: torch.Tensor,
#         ) -> torch.Tensor:
#             reduce_scatter = self._reduce_scatter(mm_1)
#             rmsnorm, _ = self._aiter_fused_add_rmsnorm(
#                 reduce_scatter, residual, rms_norm_weights
#             )
#             normalized = self._all_gather(rmsnorm)
#             return normalized

#         pm.register_replacement(
#             pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
#         )

class AiterFirstAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    def get_inputs(self):
        input = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4], device=self.device, dtype=self.dtype)

        return [input, weight]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ):
            all_reduce = self._all_reduce(input)
            rmsnorm = self._aiter_functional_rmsnorm(None, all_reduce, weight)
            return rmsnorm, all_reduce

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ):
            logger.info("Aiter FirstAllReduceRMSNormPattern replacement called!")
            reduce_scatter = self._reduce_scatter(input)

            rmsnorm = self._aiter_functional_rmsnorm(None, reduce_scatter, weight)

            all_gather = self._all_gather(rmsnorm)

            return all_gather, reduce_scatter

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AiterMiddleAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [
            residual,
            mm_1,
            rms_norm_weights,
        ]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm = self._aiter_functional_fused_add_rmsnorm(
                all_reduce, residual, rms_norm_weights
            )
            return rmsnorm[0], rmsnorm[1]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            logger.info("Aiter MiddleAllReduceRMSNormPattern replacement called!")
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm = self._aiter_functional_fused_add_rmsnorm(
                reduce_scatter, residual, rms_norm_weights
            )
            all_gather = self._all_gather(rmsnorm[0])
            return all_gather, rmsnorm[1]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AiterLastAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [
            residual,
            mm_1,
            rms_norm_weights,
        ]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> torch.Tensor:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm = self._aiter_functional_fused_add_rmsnorm(
                all_reduce, residual, rms_norm_weights
            )
            return rmsnorm[0]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> torch.Tensor:
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm = self._aiter_functional_fused_add_rmsnorm(
                reduce_scatter, residual, rms_norm_weights
            )
            normalized = self._all_gather(rmsnorm[0])
            return normalized

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )



class AiterFirstAllReduceRMSNormQuantPattern(_SequenceParallelPatternHelper):

    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str
    ):
        super().__init__(epsilon, dtype, device, quant_op=None)

    def get_inputs(self):
        input = torch.empty([1, 8, 7168], device=self.device, dtype=self.dtype)
        rmsnorm_result = torch.empty([1, 8, 7168], device=self.device, dtype=self.dtype)
        weight = torch.empty([7168], device=self.device, dtype=self.dtype)
        # return [input, rmsnorm_result, weight]
        # Preallocate example buffers to let FX bind existing empties
        quant_out_buf = torch.empty([448, 128], device=self.device, dtype=FP8_DTYPE)
        scales_buf = torch.empty([448, 56], device=self.device, dtype=torch.float32)
        return [input, rmsnorm_result, weight, quant_out_buf, scales_buf]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
            rmsnorm_result: torch.Tensor,
            weight: torch.Tensor,
            quant_out_buf: torch.Tensor,
            scales_buf: torch.Tensor,
        ):
            all_reduce = self._all_reduce(input)

            rmsnorm_out = self._aiter_functional_rmsnorm(rmsnorm_result, all_reduce, weight)

            # reshape to [-1, 128] to form per-token groups
            view_grouped = rmsnorm_out.reshape(-1, 128)
            auto_func = torch.ops.higher_order.auto_functionalized(
                # torch.ops.aiter.wrapper_dynamic_per_token_scaled_quant.default,
                # out=quant_out_buf,
                # input=view_grouped,
                # # scales=scales_buf,
                # scale=scales_buf,
                # scale_ub=None,
                # shuffle_scale=False,
                # num_rows=None,
                # num_rows_factor=1,
                torch.ops.vllm.rocm_aiter_per_token_quant.default,
                out=quant_out_buf,
                x=view_grouped,
                scale=scales_buf
            )
            # quant_out_sel = auto_func[1]
            # scales_out = auto_func[3]
            # return quant_out_sel, scales_out
            return auto_func[1]
            # view_2d = rmsnorm_out.view(-1, rmsnorm_out.shape[-1])
            
            # contiguous = view_2d.contiguous()
            # size = contiguous.size()
            # scale = torch.empty(
            #     (size[0], rmsnorm_out.shape[-1] // 128),
            #     device=self.device
            
            # quant_result = torch.empty(
            #     size,
            #     dtype=FP8_DTYPE,
            #     device=self.device
            # )
            
            # view_grouped = contiguous.view(-1, 128)
            
            # torch.ops.aiter.wrapper_dynamic_per_token_scaled_quant(
            #     quant_out_buf, view_grouped, scales_buf,
            #     shuffle_scale=False,
            #     num_rows=None,
            #     num_rows_factor=1
            # )
            # return quant_out_buf, scales_buf

        def replacement(
            input: torch.Tensor,
            rmsnorm_result: torch.Tensor,
            weight: torch.Tensor,
        ):
            logger.info("Aiter FirstAllReduceRMSNormQuantPattern replacement called!")
            reduce_scatter = self._reduce_scatter(input)
            
            batch_size = reduce_scatter.shape[0]
            seq_len = reduce_scatter.shape[1] if reduce_scatter.dim() > 2 else 1
            last_dim = reduce_scatter.shape[-1]
            num_groups = last_dim // 128
            
            rmsnorm_buf = torch.empty_like(reduce_scatter, dtype=self.dtype)
            quant_result = torch.empty_like(reduce_scatter, dtype=FP8_DTYPE)
            scale = torch.empty(
                (batch_size * seq_len, num_groups),
                dtype=torch.float32,
                device=self.device
            )
            
            fused_out = self._aiter_functional_rmsnorm_then_quant(
                rmsnorm_buf, quant_result, reduce_scatter, weight, scale
            )
            all_gather = self._all_gather(fused_out[0])
            return all_gather, fused_out[1]

        # from torch.fx import symbolic_trace
        # gm = symbolic_trace(pattern)
        # print(f"Aiter FirstAllReduceRMSNormQuantPattern: ", pattern)
        # print(gm.graph)
        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AiterMiddleAllReduceRMSNormQuantPattern(_SequenceParallelPatternHelper):

    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str
    ):
        super().__init__(epsilon, dtype, device, quant_op=None)

    def get_inputs(self):
        mm_1 = torch.empty([4, 7168], device=self.device, dtype=self.dtype)
        residual = torch.empty([4, 7168], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([7168], device=self.device, dtype=self.dtype)
        return [residual, mm_1, rms_norm_weights]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            all_reduce_cloned = all_reduce.clone() 

            rmsnorm = self._aiter_functional_fused_add_rmsnorm(
                all_reduce, residual, rms_norm_weights
            )
            
            view_2d = rmsnorm[0].view(-1, rmsnorm[0].shape[-1])
            contiguous = view_2d.contiguous()
            size = contiguous.size()
            
            scale = torch.empty(
                (size[0], rmsnorm[0].shape[-1] // 128),
                dtype=torch.float32,
                device=self.device
            )
            
            quant_result = torch.empty(
                size,
                dtype=FP8_DTYPE,
                device=self.device
            )
            
            view_grouped = contiguous.view(-1, 128)
            torch.ops.aiter.wrapper_dynamic_per_token_scaled_quant(
                quant_result, view_grouped, scale,
                shuffle_scale=False,
                num_rows=None,
                num_rows_factor=1
            )
            return quant_result, scale, rmsnorm[1]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            logger.info("Aiter MiddleAllReduceRMSNormQuantPattern replacement called!")
            reduce_scatter = self._reduce_scatter(mm_1)
            
            batch_size = reduce_scatter.shape[0]
            last_dim = reduce_scatter.shape[-1]
            num_groups = last_dim // 128
            
            quant_result = torch.empty_like(reduce_scatter, dtype=FP8_DTYPE)
            scale = torch.empty(
                (batch_size, num_groups),
                dtype=torch.float32,
                device=self.device
            )
            
            fused_out, residual_out = self._aiter_functional_fused_add_rmsnorm_then_quant(
                quant_result, reduce_scatter, residual, rms_norm_weights, scale
            )
            all_gather = self._all_gather(fused_out[0])
            return all_gather, fused_out[1], residual_out

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AiterLastAllReduceRMSNormQuantPattern(_SequenceParallelPatternHelper):

    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str
    ):
        super().__init__(epsilon, dtype, device, quant_op=None)

    def get_inputs(self):
        mm_1 = torch.empty([4, 7168], device=self.device, dtype=self.dtype)
        residual = torch.empty([4, 7168], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([7168], device=self.device, dtype=self.dtype)
        return [residual, mm_1, rms_norm_weights]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            all_reduce_cloned = all_reduce.clone() 
            rmsnorm = self._aiter_functional_fused_add_rmsnorm(
                all_reduce, residual, rms_norm_weights
            )
            
            view_2d = rmsnorm[0].view(-1, rmsnorm[0].shape[-1])
            contiguous = view_2d.contiguous()
            size = contiguous.size()
            
            scale = torch.empty(
                (size[0], rmsnorm[0].shape[-1] // 128),
                dtype=torch.float32,
                device=self.device
            )
            
            quant_result = torch.empty(
                size,
                dtype=FP8_DTYPE,
                device=self.device
            )
            
            view_grouped = contiguous.view(-1, 128)
            torch.ops.aiter.wrapper_dynamic_per_token_scaled_quant(
                quant_result, view_grouped, scale,
                shuffle_scale=False,
                num_rows=None,
                num_rows_factor=1
            )
            return quant_result, scale

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            logger.info("Aiter LastAllReduceRMSNormQuantPattern replacement called!")
            reduce_scatter = self._reduce_scatter(mm_1)
            
            batch_size = reduce_scatter.shape[0]
            last_dim = reduce_scatter.shape[-1]
            num_groups = last_dim // 128
            
            quant_result = torch.empty_like(reduce_scatter, dtype=FP8_DTYPE)
            scale = torch.empty(
                (batch_size, num_groups),
                dtype=torch.float32,
                device=self.device
            )
            
            fused_out, _ = self._aiter_functional_fused_add_rmsnorm_then_quant(
                quant_result, reduce_scatter, residual, rms_norm_weights, scale
            )
            normalized = self._all_gather(fused_out[0])
            return normalized, fused_out[1]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class FirstAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):
    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str, op: torch._ops.OpOverload
    ):
        super().__init__(epsilon, dtype, device, quant_op=op)

    def get_inputs(self):
        input = torch.zeros([1, 8, 4], device=self.device, dtype=self.dtype)
        rmsnorm_result = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        quant_result = torch.empty([1, 8, 4], device=self.device, dtype=FP8_DTYPE)
        weight = torch.empty([4], device=self.device, dtype=self.dtype)
        scale = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        return [input, rmsnorm_result, quant_result, weight, scale]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
            rmsnorm_result: torch.Tensor,
            quant_result: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            all_reduce = self._all_reduce(input)
            static_fp8 = self._functional_rmsnorm_then_quant(
                rmsnorm_result, quant_result, all_reduce, weight, scale
            )
            return static_fp8[1], all_reduce

        def replacement(
            input: torch.Tensor,
            rmsnorm_result: torch.Tensor,
            quant_result: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            reduce_scatter = self._reduce_scatter(input)

            rmsnorm_result = torch.empty_like(
                reduce_scatter, dtype=rmsnorm_result.dtype
            )
            quant_result = torch.empty_like(
                rmsnorm_result,  # Output of RMSNorm
                dtype=quant_result.dtype,
            )
            static_fp8 = self._functional_rmsnorm_then_quant(
                rmsnorm_result, quant_result, reduce_scatter, weight, scale
            )
            all_gather = self._all_gather(static_fp8[1])

            return all_gather, reduce_scatter

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class MiddleAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):
    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str, op: torch._ops.OpOverload
    ):
        super().__init__(epsilon, dtype, device, quant_op=op)

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        result = torch.empty([4, 4], device=self.device, dtype=FP8_DTYPE)
        scale = torch.empty([1, 1], device=self.device, dtype=torch.float32)

        return [
            result,
            residual,
            mm_1,
            rms_norm_weights,
            scale,
        ]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            static_fp8, rmsnorm_residual_out = (
                self._functional_fused_add_rmsnorm_then_quant(  # noqa: E501
                    result, all_reduce, residual, rms_norm_weights, scale
                )
            )
            return static_fp8[1], rmsnorm_residual_out

        def replacement(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            quant_result_buf = torch.empty_like(reduce_scatter, dtype=result.dtype)
            static_fp8, rmsnorm_residual_out = (
                self._functional_fused_add_rmsnorm_then_quant(  # noqa: E501
                    quant_result_buf, reduce_scatter, residual, rms_norm_weights, scale
                )
            )
            all_gather = self._all_gather(static_fp8[1])
            return all_gather, rmsnorm_residual_out

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class LastAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):
    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str, op: torch._ops.OpOverload
    ):
        super().__init__(epsilon, dtype, device, quant_op=op)

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        result = torch.empty([4, 4], device=self.device, dtype=FP8_DTYPE)
        scale = torch.empty([1, 1], device=self.device, dtype=torch.float32)

        return [
            result,
            residual,
            mm_1,
            rms_norm_weights,
            scale,
        ]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            static_fp8, _ = self._functional_fused_add_rmsnorm_then_quant(
                result, all_reduce, residual, rms_norm_weights, scale
            )
            return static_fp8[1]

        def replacement(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            quant_result_buf = torch.empty_like(reduce_scatter, dtype=result.dtype)
            static_fp8, _ = self._functional_fused_add_rmsnorm_then_quant(
                quant_result_buf, reduce_scatter, residual, rms_norm_weights, scale
            )
            normalized = self._all_gather(static_fp8[1])
            return normalized

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class SequenceParallelismPass(VllmPatternMatcherPass):
    """
    This pass enables sequence parallelism for models.
    It identifies patterns where an AllReduce operation is followed by
    an RMSNorm (or RMSNorm and then Quantization) operation.
    These patterns are replaced with a ReduceScatter operation, followed by
    a local RMSNorm/Quantization, and then an AllGather operation.

    The general transformation is:
    Input -> AllReduce -> RMSNorm -> Output
    becomes
    Input -> ReduceScatter -> RMSNorm -> AllGather -> Output

    While this pass itself does not directly yield performance improvements,
    it lays the groundwork for subsequent fusion passes, such as
    GEMM + ReduceScatter and AllGather + GEMM fusions. These fusions can
    significantly reduce communication overhead and improve overall model
    performance.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="sequence_parallelism_pass"
        )

        use_aiter_rmsnorm = is_rocm_aiter_rmsnorm_enabled()
        logger.info(f"Aiter RMSNorm enabled: {use_aiter_rmsnorm}")

        for epsilon in [1e-5, 1e-6]:
            fp8_quant_op = torch.ops._C.static_scaled_fp8_quant.default


            if use_aiter_rmsnorm:
                logger.info(f"Registering Aiter RMSNorm + Quant fusion patterns for epsilon={epsilon}")
                AiterFirstAllReduceRMSNormQuantPattern(
                    epsilon, self.model_dtype, self.device
                ).register(self.patterns)

                AiterMiddleAllReduceRMSNormQuantPattern(
                    epsilon, self.model_dtype, self.device
                ).register(self.patterns)

                AiterLastAllReduceRMSNormQuantPattern(
                    epsilon, self.model_dtype, self.device
                ).register(self.patterns)

                logger.info(f"Registering Aiter RMSNorm patterns for epsilon={epsilon}")
                AiterFirstAllReduceRMSNormPattern(
                    epsilon, self.model_dtype, self.device
                ).register(self.patterns)

                AiterMiddleAllReduceRMSNormPattern(
                    epsilon, self.model_dtype, self.device
                ).register(self.patterns)

                AiterLastAllReduceRMSNormPattern(
                    epsilon, self.model_dtype, self.device
                ).register(self.patterns)
            else:
                logger.info(f"Registering Standard RMSNorm patterns for epsilon={epsilon}")
                FirstAllReduceRMSNormPattern(
                    epsilon, self.model_dtype, self.device
                ).register(self.patterns)

                MiddleAllReduceRMSNormPattern(
                    epsilon, self.model_dtype, self.device
                ).register(self.patterns)

                LastAllReduceRMSNormPattern(
                    epsilon, self.model_dtype, self.device
                ).register(self.patterns)
                FirstAllReduceRMSNormStaticFP8Pattern(
                    epsilon, self.model_dtype, self.device, fp8_quant_op
                ).register(self.patterns)
                MiddleAllReduceRMSNormStaticFP8Pattern(
                    epsilon, self.model_dtype, self.device, fp8_quant_op
                ).register(self.patterns)
                LastAllReduceRMSNormStaticFP8Pattern(
                    epsilon, self.model_dtype, self.device, fp8_quant_op
                ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    def is_applicable(self, shape: int | None) -> bool:
        # When sequence parallelism is enabled, the residual tensor from RMSNorm
        # needs to be split along the sequence dimension. However, this dimension
        # is symbolic during piecewise compilation, and splitting symbolic shapes
        # is not supported.
        #
        # This pass is therefore only applied when the sequence dimension is
        # concrete:
        # 1. In full-graph compilation mode (no Dynamo splitting ops are used).
        #   For this case we always pad num_tokens to be a multiple of
        #   tensor_parallel_size, so there's no need to check shape % tp_size == 0.
        # 2. For specific shape provided during compilation (e.g., from
        #    `compile_sizes`), which must be divisible by the tensor-parallel
        #    size.
        if (
            not self.compilation_config.splitting_ops
            or self.compilation_config.use_inductor_graph_partition
        ):
            return True
        # if envs.VLLM_ROCM_USE_SEQUENCE_PARALLEL:
        #     print(f"envs.VLLM_ROCM_USE_SEQUENCE_PARALLEL: {envs.VLLM_ROCM_USE_SEQUENCE_PARALLEL}")
        #     return True
        tp_size = get_tensor_model_parallel_world_size()
        print(f"shape: {shape}, tp_size: {tp_size}")
        return shape is not None and shape % tp_size == 0

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        print(f"graph: {graph}")
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
