# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ROCm AITER TRTLLM fused all-reduce + RMSNorm + optional per-token quant pass.

This pass fuses the following patterns using TRTLLM's fused kernel:

Without quantization:
1. all_reduce(x) -> rocm_aiter_rmsnorm2d_fwd_with_add(ar_out, residual)
   -> trtllm_fused_allreduce_rmsnorm
2. all_reduce(x) -> rocm_aiter_rms_norm(ar_out)
   -> trtllm_fused_allreduce_rmsnorm

With FP8 per-token quantization:
3. all_reduce(x) -> rocm_aiter_rmsnorm2d_fwd_with_add(ar_out, residual)
   -> per_token_quant -> trtllm_fused_allreduce_rmsnorm_quant
4. all_reduce(x) -> rocm_aiter_rms_norm(ar_out) -> per_token_quant
   -> trtllm_fused_allreduce_rmsnorm_quant

The fused operation leverages aiter's trtllm_allreduce_rms kernel which
performs all operations in a single kernel launch, reducing memory bandwidth
and kernel launch overhead.
"""

from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.trtllm_allreduce_fusion import AiterCommManager
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .inductor_pass import enable_fake_mode
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


def is_rocm_aiter_trtllm_fusion_enabled() -> bool:
    """Check if ROCm AITER TRTLLM fused all-reduce + RMSNorm is enabled."""
    if not current_platform.is_rocm():
        return False
    if not envs.VLLM_ROCM_USE_AITER:
        return False
    # Check if aiter is available with trtllm fusion support
    try:
        from importlib.util import find_spec

        if find_spec("aiter") is None:
            return False
        # Check if trtllm fusion module is available
        from aiter.ops.trtllm_all_reduce_fusion import (
            AiterDistEnv,  # noqa: F401
        )

        return True
    except ImportError:
        return False


def ensure_aiter_trtllm_comm_initialized(
    trtllm_comm: AiterCommManager,
    group: ProcessGroup,
    device_id: int,
    dtype: torch.dtype
) -> bool:
    if trtllm_comm is None:
        return False

    if (
        not trtllm_comm.initialized
        or trtllm_comm.group != group
        or trtllm_comm.device_id != device_id
        or trtllm_comm.dtype != dtype
    ):
        trtllm_comm.initialize(group=group, device_id=device_id, dtype=dtype)

    return trtllm_comm.initialized


def _rocm_trtllm_fused_allreduce_rmsnorm_impl(
    input_: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.distributed.parallel_state import _groups

    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")

    use_fused_kernel = False
    aiter_trtllm_comm = None

    # Get the pre-initialized AiterCommManager from device_communicator
    device_comm = group.device_communicator
    if device_comm is not None:
        aiter_trtllm_comm = getattr(device_comm, "aiter_trtllm_comm", None)
        device_id = group.device.index
        if aiter_trtllm_comm is not None and device_id is not None:
            # Pass group.device_group (ProcessGroup) not group (ParallelGroup)
            use_fused_kernel = ensure_aiter_trtllm_comm_initialized(
                aiter_trtllm_comm, group.device_group, device_id, input_.dtype
            )

    if not use_fused_kernel:
        # Fallback to separate operations if AiterCommManager is not available
        logger.info("_rocm_trtllm_fused_allreduce_rmsnorm_impl: use_fused_kernel=False, falling back to NCCL")
        from vllm._aiter_ops import rocm_aiter_ops

        ar_out = group._all_reduce_out_place(input_)
        out, residual_out = rocm_aiter_ops.rms_norm2d_with_add(
            ar_out, residual, weight, epsilon
        )
        return out, residual_out

    # Try the fused kernel - it returns None during warmup (within capture context
    # but not actually recording) to signal fallback to NCCL + separate ops.
    # This follows the same pattern as CustomAllreduce.custom_all_reduce().
    logger.info("_rocm_trtllm_fused_allreduce_rmsnorm_impl: calling allreduce_add_rms_fused")
    result = aiter_trtllm_comm.dist_env.allreduce_add_rms_fused(
        input_, residual, weight, epsilon, fp8_out=False
    )

    if result is None:
        # Fallback during warmup within capture context
        logger.info("_rocm_trtllm_fused_allreduce_rmsnorm_impl: result=None (warmup), falling back to NCCL")
        from vllm._aiter_ops import rocm_aiter_ops

        ar_out = group._all_reduce_out_place(input_)
        out, residual_out = rocm_aiter_ops.rms_norm2d_with_add(
            ar_out, residual, weight, epsilon
        )
        return out, residual_out

    logger.info("_rocm_trtllm_fused_allreduce_rmsnorm_impl: fused kernel SUCCESS")
    residual_out, norm_out, _ = result
    return norm_out, residual_out


def _rocm_trtllm_fused_allreduce_rmsnorm_fake(
    input_: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(input_), torch.empty_like(residual)


def _rocm_trtllm_fused_allreduce_rmsnorm_quant_impl(
    input_: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.distributed.parallel_state import _groups

    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")

    use_fused_kernel = False
    aiter_trtllm_comm = None

    # Get the pre-initialized AiterCommManager from device_communicator
    device_comm = group.device_communicator
    if device_comm is not None:
        aiter_trtllm_comm = getattr(device_comm, "aiter_trtllm_comm", None)
        device_id = group.device.index
        if aiter_trtllm_comm is not None and device_id is not None:
            # Pass group.device_group (ProcessGroup) not group (ParallelGroup)
            use_fused_kernel = ensure_aiter_trtllm_comm_initialized(
                aiter_trtllm_comm, group.device_group, device_id, input_.dtype
            )

    if not use_fused_kernel:
        # Fallback to separate operations if AiterCommManager is not available
        logger.info("_rocm_trtllm_fused_allreduce_rmsnorm_quant_impl: use_fused_kernel=False, falling back to NCCL")
        from vllm._aiter_ops import rocm_aiter_ops
        from aiter import dtypes

        ar_out = group._all_reduce_out_place(input_)
        norm_out, residual_out = rocm_aiter_ops.rms_norm2d_with_add(
            ar_out, residual, weight, epsilon
        )
        # Do per-token quantization separately
        quant_out, scale_out = rocm_aiter_ops.rocm_aiter_per_token_quant(
            norm_out, None, dtypes.fp8
        )
        return quant_out, residual_out, scale_out

    # Try the fused kernel - it returns None during warmup (within capture context
    # but not actually recording) to signal fallback to NCCL + separate ops.
    # This follows the same pattern as CustomAllreduce.custom_all_reduce().
    logger.info("_rocm_trtllm_fused_allreduce_rmsnorm_quant_impl: calling allreduce_add_rms_fused fp8_out=True")
    result = aiter_trtllm_comm.dist_env.allreduce_add_rms_fused(
        input_, residual, weight, epsilon, fp8_out=True
    )

    if result is None:
        # Fallback during warmup within capture context
        logger.info("_rocm_trtllm_fused_allreduce_rmsnorm_quant_impl: result=None (warmup), falling back to NCCL")
        from vllm._aiter_ops import rocm_aiter_ops
        from aiter import dtypes

        ar_out = group._all_reduce_out_place(input_)
        norm_out, residual_out = rocm_aiter_ops.rms_norm2d_with_add(
            ar_out, residual, weight, epsilon
        )
        # Do per-token quantization separately
        quant_out, scale_out = rocm_aiter_ops.rocm_aiter_per_token_quant(
            norm_out, None, dtypes.fp8
        )
        return quant_out, residual_out, scale_out

    logger.info("_rocm_trtllm_fused_allreduce_rmsnorm_quant_impl: fused kernel SUCCESS")
    residual_out, norm_out_fp8, scale_out = result
    return norm_out_fp8, residual_out, scale_out


def _rocm_trtllm_fused_allreduce_rmsnorm_quant_fake(
    input_: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter import dtypes

    return (
        torch.empty_like(input_, dtype=dtypes.fp8),
        torch.empty_like(residual),
        torch.empty(input_.shape[0], 1, dtype=torch.float32, device=input_.device),
    )


def _rocm_trtllm_fused_allreduce_rmsnorm_no_residual_impl(
    input_: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    group_name: str,
) -> torch.Tensor:
    # Create a zero residual for the no-residual case
    residual = torch.zeros_like(input_)
    norm_out, _ = _rocm_trtllm_fused_allreduce_rmsnorm_impl(
        input_, residual, weight, epsilon, group_name
    )
    return norm_out


def _rocm_trtllm_fused_allreduce_rmsnorm_no_residual_fake(
    input_: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    group_name: str,
) -> torch.Tensor:
    """Fake implementation for tracing."""
    return torch.empty_like(input_)


def _rocm_trtllm_fused_allreduce_rmsnorm_no_residual_quant_impl(
    input_: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of the fused all-reduce + RMSNorm + FP8 quant without residual.

    Uses TRTLLM kernel with a zero residual tensor internally.

    Args:
        input_: Input tensor to all-reduce
        weight: RMSNorm weight tensor
        epsilon: Epsilon for numerical stability
        group_name: The name of the tensor parallel group

    Returns:
        A tuple of (fp8_norm_output, scale_output)
    """
    # Create a zero residual for the no-residual case
    residual = torch.zeros_like(input_)
    norm_out_fp8, _, scale_out = _rocm_trtllm_fused_allreduce_rmsnorm_quant_impl(
        input_, residual, weight, epsilon, group_name
    )
    return norm_out_fp8, scale_out


def _rocm_trtllm_fused_allreduce_rmsnorm_no_residual_quant_fake(
    input_: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for tracing."""
    from aiter import dtypes

    return (
        torch.empty_like(input_, dtype=dtypes.fp8),
        torch.empty(input_.shape[0], 1, dtype=torch.float32, device=input_.device),
    )


# Register custom ops for ROCm
if current_platform.is_rocm():
    # Fused allreduce + rmsnorm (with residual, no quant)
    direct_register_custom_op(
        op_name="rocm_trtllm_fused_allreduce_rmsnorm",
        op_func=_rocm_trtllm_fused_allreduce_rmsnorm_impl,
        mutates_args=[],
        fake_impl=_rocm_trtllm_fused_allreduce_rmsnorm_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    # Fused allreduce + rmsnorm + quant (with residual)
    direct_register_custom_op(
        op_name="rocm_trtllm_fused_allreduce_rmsnorm_quant",
        op_func=_rocm_trtllm_fused_allreduce_rmsnorm_quant_impl,
        mutates_args=[],
        fake_impl=_rocm_trtllm_fused_allreduce_rmsnorm_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    # Fused allreduce + rmsnorm (no residual, no quant)
    direct_register_custom_op(
        op_name="rocm_trtllm_fused_allreduce_rmsnorm_no_residual",
        op_func=_rocm_trtllm_fused_allreduce_rmsnorm_no_residual_impl,
        mutates_args=[],
        fake_impl=_rocm_trtllm_fused_allreduce_rmsnorm_no_residual_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    # Fused allreduce + rmsnorm + quant (no residual)
    direct_register_custom_op(
        op_name="rocm_trtllm_fused_allreduce_rmsnorm_no_residual_quant",
        op_func=_rocm_trtllm_fused_allreduce_rmsnorm_no_residual_quant_impl,
        mutates_args=[],
        fake_impl=_rocm_trtllm_fused_allreduce_rmsnorm_no_residual_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )


class TRTLLMAllReduceRMSNormWithAddPattern:
    """
    Pattern for fusing all_reduce + rocm_aiter_rmsnorm2d_fwd_with_add using TRTLLM.

    Pattern:
        all_reduce(x) -> rocm_aiter_rmsnorm2d_fwd_with_add(ar_out, residual, weight, eps)
    Replacement:
        trtllm_fused_allreduce_rmsnorm(x, residual, weight, eps) -> (y, residual_out)
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str,
    ):
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def get_inputs(self):
        input_tensor = torch.empty(5, 16, dtype=self.dtype, device=self.device)
        residual = torch.empty(5, 16, dtype=self.dtype, device=self.device)
        weight = torch.empty(16, dtype=self.dtype, device=self.device)
        return [input_tensor, residual, weight]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input_: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
        ):
            allreduce_output = tensor_model_parallel_all_reduce(input_)
            rms_out, residual_out = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
                allreduce_output, residual, weight, self.epsilon
            )
            return rms_out, residual_out

        def replacement(
            input_: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
        ):
            rms_out, residual_out = torch.ops.vllm.rocm_trtllm_fused_allreduce_rmsnorm(
                input_,
                residual,
                weight,
                self.epsilon,
                self.tp.unique_name,
            )
            return rms_out, residual_out

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class TRTLLMAllReduceRMSNormPattern:
    """
    Pattern for fusing all_reduce + rocm_aiter_rms_norm using TRTLLM (no residual).

    Pattern:
        all_reduce(x) -> rocm_aiter_rms_norm(ar_out, weight, eps)
    Replacement:
        trtllm_fused_allreduce_rmsnorm_no_residual(x, weight, eps) -> y
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str,
    ):
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def get_inputs(self):
        input_tensor = torch.empty(5, 16, dtype=self.dtype, device=self.device)
        weight = torch.empty(16, dtype=self.dtype, device=self.device)
        return [input_tensor, weight]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(input_: torch.Tensor, weight: torch.Tensor):
            allreduce_output = tensor_model_parallel_all_reduce(input_)
            rms_out = torch.ops.vllm.rocm_aiter_rms_norm(
                allreduce_output, weight, self.epsilon
            )
            return rms_out

        def replacement(input_: torch.Tensor, weight: torch.Tensor):
            rms_out = (
                torch.ops.vllm.rocm_trtllm_fused_allreduce_rmsnorm_no_residual(
                    input_,
                    weight,
                    self.epsilon,
                    self.tp.unique_name,
                )
            )
            return rms_out

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class TRTLLMAllReduceRMSNormWithAddQuantPattern:
    """
    Pattern for fusing all_reduce + rmsnorm + per-token quant using TRTLLM.

    Pattern:
        all_reduce(x) -> rocm_aiter_rmsnorm2d_fwd_with_add(ar_out, residual, weight, eps)
        -> rocm_aiter_per_token_quant(rms_out)
    Replacement:
        trtllm_fused_allreduce_rmsnorm_quant(x, residual, weight, eps)
        -> (fp8_out, residual_out, scale_out)
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str,
    ):
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def get_inputs(self):
        input_tensor = torch.empty(5, 16, dtype=self.dtype, device=self.device)
        residual = torch.empty(5, 16, dtype=self.dtype, device=self.device)
        weight = torch.empty(16, dtype=self.dtype, device=self.device)
        # Output tensors for per_token_quant (mutating args)
        from aiter import dtypes

        out = torch.empty(5, 16, dtype=dtypes.fp8, device=self.device)
        scale = torch.empty(5, 1, dtype=torch.float32, device=self.device)
        return [input_tensor, residual, weight, out, scale]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input_: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            out: torch.Tensor,
            scale: torch.Tensor,
        ):
            allreduce_output = tensor_model_parallel_all_reduce(input_)
            rms_out, residual_out = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
                allreduce_output, residual, weight, self.epsilon
            )
            # Per-token quantization (mutates out and scale)
            torch.ops.vllm.rocm_aiter_per_token_quant(out, rms_out, scale)
            return out, residual_out, scale

        def replacement(
            input_: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            out: torch.Tensor,
            scale: torch.Tensor,
        ):
            (
                fp8_out,
                residual_out,
                scale_out,
            ) = torch.ops.vllm.rocm_trtllm_fused_allreduce_rmsnorm_quant(
                input_,
                residual,
                weight,
                self.epsilon,
                self.tp.unique_name,
            )
            # Copy to the provided output tensors
            out.copy_(fp8_out)
            scale.copy_(scale_out)
            return out, residual_out, scale

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class TRTLLMAllReduceRMSNormQuantPattern:
    """
    Pattern for fusing all_reduce + rms_norm + per-token quant using TRTLLM.

    Pattern:
        all_reduce(x) -> rocm_aiter_rms_norm(ar_out, weight, eps)
        -> rocm_aiter_per_token_quant(rms_out)
    Replacement:
        trtllm_fused_allreduce_rmsnorm_no_residual_quant(x, weight, eps)
        -> (fp8_out, scale_out)
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str,
    ):
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def get_inputs(self):
        input_tensor = torch.empty(5, 16, dtype=self.dtype, device=self.device)
        weight = torch.empty(16, dtype=self.dtype, device=self.device)
        # Output tensors for per_token_quant (mutating args)
        from aiter import dtypes

        out = torch.empty(5, 16, dtype=dtypes.fp8, device=self.device)
        scale = torch.empty(5, 1, dtype=torch.float32, device=self.device)
        return [input_tensor, weight, out, scale]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input_: torch.Tensor,
            weight: torch.Tensor,
            out: torch.Tensor,
            scale: torch.Tensor,
        ):
            allreduce_output = tensor_model_parallel_all_reduce(input_)
            rms_out = torch.ops.vllm.rocm_aiter_rms_norm(
                allreduce_output, weight, self.epsilon
            )
            # Per-token quantization (mutates out and scale)
            torch.ops.vllm.rocm_aiter_per_token_quant(out, rms_out, scale)
            return out, scale

        def replacement(
            input_: torch.Tensor,
            weight: torch.Tensor,
            out: torch.Tensor,
            scale: torch.Tensor,
        ):
            (
                fp8_out,
                scale_out,
            ) = torch.ops.vllm.rocm_trtllm_fused_allreduce_rmsnorm_no_residual_quant(
                input_,
                weight,
                self.epsilon,
                self.tp.unique_name,
            )
            # Copy to the provided output tensors
            out.copy_(fp8_out)
            scale.copy_(scale_out)
            return out, scale

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ROCmAiterTRTLLMAllReduceFusionPass(VllmPatternMatcherPass):
    """
    Fusion pass for ROCm AITER TRTLLM fused all-reduce + RMSNorm + optional quant.

    This pass fuses:
    - all_reduce + rocm_aiter_rmsnorm2d_fwd_with_add (with residual)
    - all_reduce + rocm_aiter_rms_norm (without residual)
    - all_reduce + rocm_aiter_rmsnorm2d_fwd_with_add + per_token_quant
    - all_reduce + rocm_aiter_rms_norm + per_token_quant

    into fused operations using aiter's trtllm_allreduce_rms kernel.
    """

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.disabled = True
        self.matched_count = 0
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size <= 1:
            logger.info(
                "ROCmAiterTRTLLMAllReduceFusionPass disabled: TP size <= 1"
            )
            return

        if not is_rocm_aiter_trtllm_fusion_enabled():
            logger.info(
                "ROCmAiterTRTLLMAllReduceFusionPass disabled: "
                "ROCm AITER TRTLLM fusion not enabled or not available"
            )
            return
        
        logger.info(
            "ROCmAiterTRTLLMAllReduceFusionPass enabled: TP size = %d",
            self.tp_size,
        )

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_trtllm_allreduce_fusion_pass"
        )

        self.register_patterns()
        self.dump_patterns(config, self.patterns)

    @enable_fake_mode
    def register_patterns(self):
        for epsilon in [1e-5, 1e-6]:
            # Fuse all_reduce + rmsnorm2d_fwd_with_add (with residual, no quant)
            TRTLLMAllReduceRMSNormWithAddPattern(
                epsilon,
                self.model_dtype,
                self.device,
            ).register(self.patterns)

            # Fuse all_reduce + rms_norm (no residual, no quant)
            TRTLLMAllReduceRMSNormPattern(
                epsilon,
                self.model_dtype,
                self.device,
            ).register(self.patterns)

            # TODO: Quant patterns disabled due to mutating ops not working
            # with pattern matcher. The per_token_quant op modifies out/scale
            # in-place, which causes assertion errors in register_replacement.
            # These patterns need manual graph matching instead.
            #
            # # Fuse all_reduce + rmsnorm2d_fwd_with_add + per_token_quant
            # TRTLLMAllReduceRMSNormWithAddQuantPattern(
            #     epsilon,
            #     self.model_dtype,
            #     self.device,
            # ).register(self.patterns)
            #
            # # Fuse all_reduce + rms_norm + per_token_quant
            # TRTLLMAllReduceRMSNormQuantPattern(
            #     epsilon,
            #     self.model_dtype,
            #     self.device,
            # ).register(self.patterns)

            # Clear pattern matcher cache for multiple epsilon values
            torch._inductor.pattern_matcher._seen_patterns.clear()

        self.disabled = False

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        if self.disabled:
            logger.debug("ROCmAiterTRTLLMAllReduceFusionPass disabled")
            logger.info("!!!!!!!!!!!!!!!! ROCmAiterTRTLLMAllReduceFusionPass disabled")
            return

        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        logger.info("!!!!!!!!!!!! Replaced %s patterns", self.matched_count)

    def uuid(self) -> Any:
        return self.hash_source(
            self,
            TRTLLMAllReduceRMSNormWithAddPattern,
            TRTLLMAllReduceRMSNormPattern,
            # Quant patterns disabled - see register_patterns TODO
            # TRTLLMAllReduceRMSNormWithAddQuantPattern,
            # TRTLLMAllReduceRMSNormQuantPattern,
        )
