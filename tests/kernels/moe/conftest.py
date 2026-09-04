# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.distributed import cleanup_dist_env_and_memory
from vllm.v1.worker.workspace import reset_workspace_manager

NEEDS_CLEAN_ENTRY = frozenset(
    {
        "test_moe_layer.py",
        "test_ocp_mx_moe.py",
        "test_zero_expert_moe.py",
        "test_batched_deepgemm.py",
        "test_deepep_deepgemm_moe.py",
        "test_deepep_moe.py",
        "test_deepep_v2_moe.py",
        "test_deepgemm.py",
        "test_grouped_topk.py",
        "test_mxfp4_moe.py",
        "test_shared_fused_moe_routed_transform.py",
        "test_silu_mul_per_token_group_quant_fp8_colmajor.py",
        "test_situ_mul_fp8_quant.py",
        "test_modular_oai_triton_moe.py",
    }
)


def pytest_addoption(parser):
    parser.addoption(
        "--subtests", action="store", type=str, default=None, help="subtest ids"
    )


@pytest.fixture
def subtests(request):
    return request.config.getoption("--subtests")


@pytest.fixture()
def should_do_global_cleanup_after_test() -> bool:
    """Drop the per-test global cleanup for this directory."""
    return False


def pytest_runtest_setup(item):
    if item.path.name in NEEDS_CLEAN_ENTRY:
        cleanup_dist_env_and_memory()
        reset_workspace_manager()
