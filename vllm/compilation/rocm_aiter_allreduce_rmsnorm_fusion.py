# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .inductor_pass import enable_fake_mode
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


def is_rocm_aiter_allreduce_rmsnorm_enabled() -> bool:
    if not current_platform.is_rocm():
        return False
    if not envs.VLLM_ROCM_USE_AITER:
        return False

    try:
        from importlib.util import find_spec

        if find_spec("aiter") is None:
            return False
    except ImportError:
        return False
    return True


def _can_use_fused_ar_rms(input_: torch.Tensor, world_size: int) -> bool:
    """Taken from condition checks in aiter"""
    n = input_.shape[-1]
    return (
        n <= 16384
        and input_.numel() * input_.element_size() < 8 * 1024 * 8192
        and world_size != 6
    )


def _rocm_aiter_fused_allreduce_rmsnorm_impl(
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

    device_comm = group.device_communicator
    if device_comm is not None:
        ca_comm = getattr(device_comm, "ca_comm", None)
        use_aiter_ca = getattr(device_comm, "use_aiter_custom_allreduce", False)

        if (
            use_aiter_ca
            and ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ar(input_)
            and _can_use_fused_ar_rms(input_, device_comm.world_size)
            and hasattr(ca_comm, "custom_fused_ar_rms")
        ):
            out, res_out = ca_comm.custom_fused_ar_rms(
                input_, residual, weight, epsilon
            )
            return out, res_out

    # Fallback: launch all-reduce and rmsnorm separately
    ar_out = group._all_reduce_out_place(input_)

    out, residual_out = rocm_aiter_ops.rms_norm2d_with_add(
        ar_out, residual, weight, epsilon
    )
    return out, residual_out


def _rocm_aiter_fused_allreduce_rmsnorm_fake(
    input_: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(input_), torch.empty_like(residual)


def _rocm_aiter_fused_allreduce_rmsnorm_no_residual_impl(
    input_: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    group_name: str,
) -> torch.Tensor:
    from vllm.distributed.parallel_state import _groups

    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")

    residual = torch.zeros_like(input_)

    device_comm = group.device_communicator
    if device_comm is not None:
        ca_comm = getattr(device_comm, "ca_comm", None)
        use_aiter_ca = getattr(device_comm, "use_aiter_custom_allreduce", False)

        if (
            use_aiter_ca
            and ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ar(input_)
            and _can_use_fused_ar_rms(input_, device_comm.world_size)
            and hasattr(ca_comm, "custom_fused_ar_rms")
        ):
            # Use AITER's fused all-reduce + RMSNorm kernel
            out, _ = ca_comm.custom_fused_ar_rms(input_, residual, weight, epsilon)
            return out

    # Fallback: launch all-reduce and rmsnorm separately
    ar_out = group._all_reduce_out_place(input_)

    out = rocm_aiter_ops.rms_norm(ar_out, weight, epsilon)
    return out


def _rocm_aiter_fused_allreduce_rmsnorm_no_residual_fake(
    input_: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    group_name: str,
) -> torch.Tensor:
    return torch.empty_like(input_)


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_aiter_fused_allreduce_rmsnorm",
        op_func=_rocm_aiter_fused_allreduce_rmsnorm_impl,
        mutates_args=[],
        fake_impl=_rocm_aiter_fused_allreduce_rmsnorm_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_fused_allreduce_rmsnorm_no_residual",
        op_func=_rocm_aiter_fused_allreduce_rmsnorm_no_residual_impl,
        mutates_args=[],
        fake_impl=_rocm_aiter_fused_allreduce_rmsnorm_no_residual_fake,
        dispatch_key=current_platform.dispatch_key,
    )


class AllReduceAiterRMSNormWithAddPattern:
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
        # Create example tensors for pattern matching
        # input tensor (goes through all-reduce)
        input_tensor = torch.empty(5, 16, dtype=self.dtype, device=self.device)
        # residual tensor
        residual = torch.empty(5, 16, dtype=self.dtype, device=self.device)
        # weight tensor for rmsnorm
        weight = torch.empty(16, dtype=self.dtype, device=self.device)
        return [input_tensor, residual, weight]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(input_: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
            # Pattern: all_reduce -> rocm_aiter_rmsnorm2d_fwd_with_add
            allreduce_output = tensor_model_parallel_all_reduce(input_)
            rms_out, residual_out = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
                allreduce_output, residual, weight, self.epsilon
            )
            return rms_out, residual_out

        def replacement(
            input_: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
        ):
            rms_out, residual_out = torch.ops.vllm.rocm_aiter_fused_allreduce_rmsnorm(
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


class AllReduceAiterRMSNormPattern:
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
            rms_out = torch.ops.vllm.rocm_aiter_fused_allreduce_rmsnorm_no_residual(
                input_,
                weight,
                self.epsilon,
                self.tp.unique_name,
            )
            return rms_out

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ROCmAiterAllReduceRMSNormFusionPass(VllmPatternMatcherPass):
    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.disabled = True
        self.matched_count = 0
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size <= 1:
            logger.debug("ROCmAiterAllReduceRMSNormFusionPass disabled: TP size <= 1")
            return

        if not is_rocm_aiter_allreduce_rmsnorm_enabled():
            logger.debug(
                "ROCmAiterAllReduceRMSNormFusionPass disabled: "
                "ROCm AITER not enabled or not available"
            )
            return

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_allreduce_rmsnorm_fusion_pass"
        )

        self.register_patterns()
        self.dump_patterns(config, self.patterns)

    @enable_fake_mode
    def register_patterns(self):
        for epsilon in [1e-5, 1e-6]:
            # with residual
            AllReduceAiterRMSNormWithAddPattern(
                epsilon,
                self.model_dtype,
                self.device,
            ).register(self.patterns)

            # without residual
            AllReduceAiterRMSNormPattern(
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
            logger.debug("ROCmAiterAllReduceRMSNormFusionPass disabled")
            return

        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> Any:
        return self.hash_source(
            self,
            AllReduceAiterRMSNormWithAddPattern,
            AllReduceAiterRMSNormPattern,
        )
