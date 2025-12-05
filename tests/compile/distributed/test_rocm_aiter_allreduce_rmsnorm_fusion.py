# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.util import find_spec

import pytest
import torch

import vllm.envs as envs
from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.compilation.rocm_aiter_allreduce_rmsnorm_fusion import (
    ROCmAiterAllReduceRMSNormFusionPass,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

from ...utils import multi_gpu_test
from ..backend import TestBackend


class AiterRMSNormModel(torch.nn.Module):
    def __init__(self, hidden_size=16, token_num=16, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = [torch.rand(hidden_size) for _ in range(2)]
        self.w = [torch.rand(hidden_size, hidden_size) for _ in range(2)]

    def forward(self, x):
        z = torch.relu(x)
        x = tensor_model_parallel_all_reduce(z)
        y = torch.ops.vllm.rocm_aiter_rms_norm(x, self.weight[0], self.eps)

        z2 = torch.mm(y, self.w[0])
        x2 = tensor_model_parallel_all_reduce(z2)
        y2 = torch.ops.vllm.rocm_aiter_rms_norm(x2, self.weight[1], self.eps)

        return y2

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_reduce.default]

    def ops_in_model_after(self):
        return [torch.ops.vllm.rocm_aiter_fused_allreduce_rmsnorm_no_residual.default]


class AiterRMSNormWithAddModel(torch.nn.Module):
    def __init__(self, hidden_size=16, token_num=16, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = [torch.rand(hidden_size) for _ in range(4)]
        self.w = [torch.rand(hidden_size, hidden_size) for _ in range(3)]

    def forward(self, x):
        z = torch.relu(x)
        x = tensor_model_parallel_all_reduce(z)
        # all_reduce -> rocm_aiter_rms_norm (no residual)
        y = torch.ops.vllm.rocm_aiter_rms_norm(x, self.weight[0], self.eps)
        resid = x.clone()

        z2 = torch.mm(y, self.w[0])
        x2 = tensor_model_parallel_all_reduce(z2)
        # all_reduce -> rocm_aiter_rmsnorm2d_fwd_with_add
        y2, resid = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
            x2, resid, self.weight[1], self.eps
        )

        z3 = torch.mm(y2, self.w[1])
        x3 = tensor_model_parallel_all_reduce(z3)
        # all_reduce -> rocm_aiter_rmsnorm2d_fwd_with_add
        y3, resid = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
            x3, resid, self.weight[2], self.eps
        )

        z4 = torch.mm(y3, self.w[2])
        x4 = tensor_model_parallel_all_reduce(z4)
        # all_reduce -> rocm_aiter_rmsnorm2d_fwd_with_add
        y4, resid = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
            x4, resid, self.weight[3], self.eps
        )

        return y4

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_reduce.default]

    def ops_in_model_after(self):
        # Should have both fused ops after replacement
        return [
            torch.ops.vllm.rocm_aiter_fused_allreduce_rmsnorm_no_residual.default,
            torch.ops.vllm.rocm_aiter_fused_allreduce_rmsnorm.default,
        ]


def is_rocm_aiter_available():
    if not current_platform.is_rocm():
        return False
    return find_spec("aiter") is not None


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "test_model",
    [
        AiterRMSNormModel,
        AiterRMSNormWithAddModel,
    ],
)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["rocm"], reason="Only test on ROCm")
@pytest.mark.skipif(
    not is_rocm_aiter_available(),
    reason="aiter is not found or ROCm AITER is not enabled",
)
def test_rocm_aiter_allreduce_rmsnorm_fusion_pass_replace(
    test_model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                test_model,
                batch_size,
                seq_len,
                hidden_size,
                dtype,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(
        rocm_aiter_allreduce_rmsnorm_fusion_pass_on_test_model, num_processes
    )


def rocm_aiter_allreduce_rmsnorm_fusion_pass_on_test_model(
    local_rank: int,
    world_size: int,
    test_model_cls: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    current_platform.seed_everything(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12346",
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_CUSTOM_ALL_REDUCE": "1",
            "VLLM_ROCM_USE_AITER_RMSNORM": "1",
        }
    )

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    from vllm._aiter_ops import rocm_aiter_ops

    rocm_aiter_ops.register_ops_once()

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    vllm_config.compilation_config.pass_config = PassConfig(
        enable_fusion=True, enable_noop=True
    )
    vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))
    vllm_config.parallel_config.rank = local_rank

    # Use a fake model name to construct the model config
    model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
    vllm_config.model_config = ModelConfig(
        model=model_name, trust_remote_code=True, dtype=dtype, seed=42
    )

    with set_current_vllm_config(vllm_config):
        fusion_pass = ROCmAiterAllReduceRMSNormFusionPass(vllm_config)
        noop_pass = NoOpEliminationPass(vllm_config)
        func_pass = FixFunctionalizationPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(noop_pass, fusion_pass, func_pass, cleanup_pass)

        token_num = batch_size * seq_len
        model = test_model_cls(hidden_size, token_num)

        hidden_states = torch.randn((token_num, hidden_size), requires_grad=False)

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        # Verify that patterns were matched
        # 2 all_reduce + rms_norm patterns (both without residual)
        # 4 all_reduce + rmsnorm patterns:
        # - 1 without residual (rocm_aiter_rms_norm)
        # - 3 with residual (rocm_aiter_rmsnorm2d_fwd_with_add)
        min_expected_matches = 2 if test_model_cls == AiterRMSNormModel else 4

        assert fusion_pass.matched_count >= min_expected_matches, (
            f"Expected at least {min_expected_matches} matches but got "
            f"{fusion_pass.matched_count}"
        )

        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=False)
        backend.check_after_ops(model.ops_in_model_after())

        del fusion_pass


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["rocm"], reason="Only test on ROCm")
@pytest.mark.skipif(
    not is_rocm_aiter_available(),
    reason="aiter is not found or ROCm AITER is not enabled",
)
def test_rocm_aiter_allreduce_rmsnorm_fusion_correctness(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                batch_size,
                seq_len,
                hidden_size,
                dtype,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(rocm_aiter_allreduce_rmsnorm_correctness_test, num_processes)


def rocm_aiter_allreduce_rmsnorm_correctness_test(
    local_rank: int,
    world_size: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    current_platform.seed_everything(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12347",
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_CUSTOM_ALL_REDUCE": "1",
            "VLLM_ROCM_USE_AITER_RMSNORM": "1",
        }
    )

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    from vllm._aiter_ops import rocm_aiter_ops

    rocm_aiter_ops.register_ops_once()

    from vllm.distributed import get_tp_group

    tp = get_tp_group()
    token_num = batch_size * seq_len
    eps = 1e-6

    input_tensor = torch.randn((token_num, hidden_size), dtype=dtype, device=device)
    residual = torch.randn((token_num, hidden_size), dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    input_copy = input_tensor.clone()
    residual_copy = residual.clone()

    ar_out_ref = tensor_model_parallel_all_reduce(input_copy)
    rms_out_ref, residual_out_ref = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
        ar_out_ref, residual_copy, weight, eps
    )

    rms_out_fused, residual_out_fused = (
        torch.ops.vllm.rocm_aiter_fused_allreduce_rmsnorm(
            input_tensor,
            residual,
            weight,
            eps,
            tp.unique_name,
        )
    )

    ATOL, RTOL = (1e-2, 1e-2)
    torch.testing.assert_close(rms_out_fused, rms_out_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        residual_out_fused, residual_out_ref, atol=ATOL, rtol=RTOL
    )
