# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm._custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.hip_w4a16 import (  # noqa: E501
    HipW4A16LinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E501
    MPLinearLayerConfig,
)
from vllm.model_executor.layers.quantization import awq_gemv_config
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.scalar_type import scalar_types


def _ensure_single_process_model_parallel() -> None:
    import torch.distributed as dist

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
        model_parallel_is_initialized,
    )

    if not dist.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="file:///tmp/vllm_test_dist",
            backend="gloo",
        )
    if not model_parallel_is_initialized():
        with set_current_vllm_config(VllmConfig()):
            ensure_model_parallel_initialized(1, 1)


@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
@pytest.mark.parametrize(
    ("weight_type", "zero_points"),
    [
        pytest.param(scalar_types.uint4, True, id="asymmetric_uint4"),
        pytest.param(scalar_types.uint4b8, False, id="symmetric_uint4b8"),
    ],
)
@pytest.mark.parametrize("act_dtype", [torch.float16])  # TODO: +torch.bfloat16
def test_hip_w4a16_can_implement_happy_path(
    act_dtype, weight_type, zero_points, group_size, monkeypatch
):
    monkeypatch.setattr(
        ops, "hip_w4a16_linear_kernel_apply_weights", lambda *a, **kw: None
    )
    config = MPLinearLayerConfig(
        full_weight_shape=(2560, 2560),
        partition_weight_shape=(2560, 2560),
        weight_type=weight_type,
        act_type=act_dtype,
        group_size=group_size,
        zero_points=zero_points,
        has_g_idx=False,
        out_type=None,
    )
    ok, err = HipW4A16LinearKernel.can_implement(config)
    assert ok, err


def test_hip_w4a16_can_implement_fails_without_op(monkeypatch):
    monkeypatch.setattr(ops, "hip_w4a16_linear_kernel_apply_weights", None)
    config = MPLinearLayerConfig(
        full_weight_shape=(4096, 4096),
        partition_weight_shape=(4096, 4096),
        weight_type=scalar_types.uint4,
        act_type=torch.float16,
        group_size=128,
        zero_points=True,
        has_g_idx=False,
        out_type=None,
    )
    ok, err = HipW4A16LinearKernel.can_implement(config)
    assert not ok
    assert err


@pytest.mark.parametrize(
    "overrides",
    [
        {"full_weight_shape": (4097, 4096)},
        {"full_weight_shape": (4096, 4097)},
        {"full_weight_shape": (0, 4096)},
        {"full_weight_shape": (4096, 0)},
        {"partition_weight_shape": (4096, 4097)},
        {"partition_weight_shape": (4097, 4096)},
        {"partition_weight_shape": (4096, 0)},
        {"partition_weight_shape": (0, 4096)},
        {"weight_type": scalar_types.uint4b8, "zero_points": True},
        {"weight_type": scalar_types.uint4, "zero_points": False},
        {"act_type": torch.float32},
        {"group_size": 0},
        {"group_size": 7},
        {"group_size": -1},
        {"group_size": 15},
        {"has_g_idx": True},
        {"out_type": torch.float32},
    ],
)
def test_hip_w4a16_can_implement_rejects_invalid_configs(overrides, monkeypatch):
    monkeypatch.setattr(
        ops, "hip_w4a16_linear_kernel_apply_weights", lambda *a, **kw: None
    )
    config = MPLinearLayerConfig(
        full_weight_shape=(4096, 4096),
        partition_weight_shape=(4096, 4096),
        weight_type=scalar_types.uint4,
        act_type=torch.float16,
        group_size=128,
        zero_points=True,
        has_g_idx=False,
        out_type=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    ok, err = HipW4A16LinearKernel.can_implement(config)
    assert not ok
    assert err


@pytest.mark.parametrize(
    ("group_size", "input_shape", "expected_shape"),
    [
        (32, (2048, 32 * 15), (2048, 32 * 15)),  # 15 groups, split_k=0 (auto), no pad
        (32, (2048, 32 * 16), (2048, 32 * 16)),  # 16 groups, split_k=0 (auto), no pad
        (32, (2048, 32 * 20), (2048, 32 * 20)),  # 20 groups, split_k=0 (auto), no pad
        (32, (8192, 32 * 22), (8192, 32 * 22)),  # 22 groups, split_k=0 (auto), no pad
        (32, (2048, 32 * 80), (2048, 32 * 80)),  # 80 groups, split_k=0 (auto), no pad
        (128, (8, 128 * 1), (8, 128 * 1)),  # minimum shape
        (128, (2048, 128 * 1), (2048, 128 * 1)),  # too much overhead
        (128, (2048, 128 * 15), (2048, 128 * 16)),  # pads to sk=16
        (128, (2048, 128 * 17), (2048, 128 * 17)),  # no viable padding
        (128, (8192, 128 * 22), (8192, 128 * 24)),  # pads to sk=8
        (128, (12288, 128 * 20), (12288, 128 * 20)),  # already divisible
        (128, (16384, 128 * 15), (16384, 128 * 16)),  # pads to sk=8
        (128, (24576, 128 * 15), (24576, 128 * 16)),  # pads to sk=4
        (128, (24576, 128 * 22), (24576, 128 * 24)),  # pads to sk=4
    ],
)
@pytest.mark.parametrize("act_dtype", [torch.float16])  # TODO: +torch.bfloat16
def test_hip_w4a16_process_shapes(
    group_size, input_shape, expected_shape, act_dtype, monkeypatch
):
    monkeypatch.setattr(
        ops, "hip_w4a16_linear_kernel_apply_weights", lambda *a, **kw: None
    )
    # Force heuristic path by simulating "no config" regardless of device.
    monkeypatch.delenv("AWQ_GEMV_SPLIT_K", raising=False)
    monkeypatch.setattr(
        awq_gemv_config,
        "get_awq_gemv_config",
        lambda: None,
    )
    device = "cpu"

    _ensure_single_process_model_parallel()
    pack_factor = 8
    input_n, input_k = input_shape
    expected_n, expected_k = expected_shape

    config = MPLinearLayerConfig(
        full_weight_shape=(input_k, input_n),
        partition_weight_shape=(input_k, input_n),
        weight_type=scalar_types.uint4,
        act_type=act_dtype,
        group_size=group_size,
        zero_points=True,
        has_g_idx=False,
        out_type=None,
    )
    ok, err = HipW4A16LinearKernel.can_implement(config)
    assert ok, err

    w_q_packed = torch.ones(
        (input_n, input_k // pack_factor),
        dtype=torch.int32,
        device=device,
    )
    w_zp_packed = torch.ones(
        (input_n // pack_factor, input_k // group_size),
        dtype=torch.int32,
        device=device,
    )
    w_s_data = torch.ones(
        input_n, input_k // group_size, dtype=act_dtype, device=device
    )

    layer = torch.nn.Module()
    weight_loader = lambda *_args, **_kwargs: None

    w_q = PackedvLLMParameter(
        input_dim=1,
        output_dim=0,
        packed_dim=1,
        packed_factor=pack_factor,
        weight_loader=weight_loader,
        data=w_q_packed,
    )
    w_s = GroupQuantScaleParameter(
        output_dim=0,
        input_dim=1,
        weight_loader=weight_loader,
        data=w_s_data,
    )
    w_zp = PackedvLLMParameter(
        input_dim=1,
        output_dim=0,
        packed_dim=0,
        packed_factor=pack_factor,
        weight_loader=weight_loader,
        data=w_zp_packed,
    )

    layer.register_parameter("weight_packed", w_q)
    layer.register_parameter("weight_scale", w_s)
    layer.register_parameter("weight_zero_point", w_zp)

    kernel = HipW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point",
    )
    kernel.process_weights_after_loading(layer)

    # Just check that the shape was padded to the expected size.  The
    # tests that actually hit the GPU should be sensitive to errors in
    # transforming the values, so we don't bother checking them here.
    assert layer.weight_packed.shape == (expected_k, expected_n // pack_factor)
    assert layer.weight_scale.shape == (expected_k // group_size, expected_n)
    assert layer.weight_zero_point.shape == (
        expected_k // group_size,
        expected_n // pack_factor,
    )


@pytest.mark.parametrize(
    ("group_size", "input_k", "config_dict", "expected_packed_k"),
    [
        (32, 224, {(224, 2048): 4}, 256),
        (128, 896, {(896, 2048): 7}, 1024),
    ],
)
def test_hip_w4a16_env_split_k_override(
    group_size, input_k, config_dict, expected_packed_k, monkeypatch
):
    monkeypatch.setattr(
        ops, "hip_w4a16_linear_kernel_apply_weights", lambda *a, **kw: None
    )
    # Ensure env override wins even if a config is present.
    monkeypatch.setattr(
        awq_gemv_config,
        "get_awq_gemv_config",
        lambda: config_dict,
    )
    monkeypatch.setenv("AWQ_GEMV_SPLIT_K", "8")

    # Env var must override the group_size!=128 bypass too.
    assert awq_gemv_config.get_awq_gemv_split_k(input_k, 2048, group_size) == 8

    device = "cpu"

    _ensure_single_process_model_parallel()
    pack_factor = 8
    input_n = 2048

    config = MPLinearLayerConfig(
        full_weight_shape=(input_k, input_n),
        partition_weight_shape=(input_k, input_n),
        weight_type=scalar_types.uint4,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=True,
        has_g_idx=False,
        out_type=None,
    )
    ok, err = HipW4A16LinearKernel.can_implement(config)
    assert ok, err

    w_q_packed = torch.ones(
        (input_n, input_k // pack_factor),
        dtype=torch.int32,
        device=device,
    )
    w_zp_packed = torch.ones(
        (input_n // pack_factor, input_k // group_size),
        dtype=torch.int32,
        device=device,
    )
    w_s_data = torch.ones(
        input_n, input_k // group_size, dtype=torch.float16, device=device
    )

    layer = torch.nn.Module()
    weight_loader = lambda *_args, **_kwargs: None

    w_q = PackedvLLMParameter(
        input_dim=1,
        output_dim=0,
        packed_dim=1,
        packed_factor=pack_factor,
        weight_loader=weight_loader,
        data=w_q_packed,
    )
    w_s = GroupQuantScaleParameter(
        output_dim=0,
        input_dim=1,
        weight_loader=weight_loader,
        data=w_s_data,
    )
    w_zp = PackedvLLMParameter(
        input_dim=1,
        output_dim=0,
        packed_dim=0,
        packed_factor=pack_factor,
        weight_loader=weight_loader,
        data=w_zp_packed,
    )

    layer.register_parameter("weight_packed", w_q)
    layer.register_parameter("weight_scale", w_s)
    layer.register_parameter("weight_zero_point", w_zp)

    kernel = HipW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point",
    )
    kernel.process_weights_after_loading(layer)

    # split_k=8 forces padding when num_groups isn't divisible by 8.
    assert kernel._split_k == 8
    assert layer.weight_packed.shape == (expected_packed_k, input_n // pack_factor)


@pytest.mark.parametrize(
    ("group_size", "K", "N"),
    [
        (32, 2560, 2560),
        (32, 4096, 4096),
        (32, 4096, 11008),
        (64, 4096, 4096),
    ],
)
def test_compute_awq_gemv_padding_non128_group_size(group_size, K, N, monkeypatch):
    """group_size != 128 should always return split_k=0, no padding."""
    monkeypatch.delenv("AWQ_GEMV_SPLIT_K", raising=False)
    monkeypatch.setattr(
        awq_gemv_config,
        "get_awq_gemv_config",
        lambda: None,
    )
    num_groups = K // group_size
    should_pad, padded_groups, split_k = awq_gemv_config.compute_awq_gemv_padding(
        num_groups, K, N
    )
    assert not should_pad
    assert split_k == 0
    assert padded_groups == num_groups


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_hybrid_w4a16_process_weights_symmetric_with_prefill_dequant():
    """Symmetric weights must survive process_weights_after_loading.

    The dequant-prefill cache is handed the raw-nibble zero points, which only
    exist on the asymmetric path, so the symmetric path has to pass None rather
    than a name that was never bound.  Reachable only with
    ``w4a16_prefill_dequant`` set, which is off by default.
    """
    from types import SimpleNamespace

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (  # noqa: E501
        RDNAHybridW4A16LinearKernel,
    )

    _ensure_single_process_model_parallel()
    N, K, group_size, pack_factor = 256, 512, 128, 8
    weight_loader = lambda *_a, **_kw: None  # noqa: E731

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight_packed",
        PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=pack_factor,
            weight_loader=weight_loader,
            data=torch.ones((N, K // pack_factor), dtype=torch.int32, device="cuda"),
        ),
    )
    layer.register_parameter(
        "weight_scale",
        GroupQuantScaleParameter(
            output_dim=0,
            input_dim=1,
            weight_loader=weight_loader,
            data=torch.ones(N, K // group_size, dtype=torch.float16, device="cuda"),
        ),
    )

    kernel = RDNAHybridW4A16LinearKernel(
        MPLinearLayerConfig(
            full_weight_shape=(K, N),
            partition_weight_shape=(K, N),
            weight_type=scalar_types.uint4b8,
            act_type=torch.float16,
            group_size=group_size,
            zero_points=False,
            has_g_idx=False,
            out_type=None,
        ),
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name=None,
    )

    # A real ModelConfig would need a downloadable model; this path reads one
    # field off it.
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(w4a16_prefill_dequant="soft")
    with set_current_vllm_config(vllm_config):
        kernel.process_weights_after_loading(layer)
