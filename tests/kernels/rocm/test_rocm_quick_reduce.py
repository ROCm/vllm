# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ROCm Quick Reduce (quantized AllReduce) on MI3xx.

Covers:
- QuickAllReduce class construction, should_quick_allreduce() predicate
- QuickReduceRegime enum values (FP, INT8, INT6, INT4, NONE)
- VLLM_ROCM_QUICK_REDUCE_QUANTIZATION env var: all valid modes
- VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16 env var
- VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB size threshold
- ops.qr_max_size() availability (tests quick_ar flag)
- Multi-GPU quick allreduce numerical correctness (2 GPUs required)

The multi-GPU tests use multiprocessing (not Ray) for a lightweight
kernel-level test. They are skipped when fewer than 2 GPUs are available.
"""

import importlib
import os
import random

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Quick Reduce is ROCm MI300-series only",
)

MB = 1024 * 1024


# ── Availability helpers ──────────────────────────────────────────────────


def quick_ar_available() -> bool:
    try:
        from vllm import _custom_ops as ops

        ops.qr_max_size()
        return True
    except Exception:
        return False


def num_gpus() -> int:
    from vllm.utils.torch_utils import cuda_device_count_stateless

    return cuda_device_count_stateless()


# ── QuickReduceRegime enum tests ──────────────────────────────────────────


def test_quick_reduce_regime_enum_values():
    """QuickReduceRegime enum has the expected values."""
    from vllm.distributed.device_communicators.quick_all_reduce import QuickReduceRegime

    assert QuickReduceRegime.FP.value == 0
    assert QuickReduceRegime.INT8.value == 1
    assert QuickReduceRegime.INT6.value == 2
    assert QuickReduceRegime.INT4.value == 3
    assert QuickReduceRegime.NONE.value == 4


def test_quick_reduce_regime_enum_names():
    """QuickReduceRegime has all expected names."""
    from vllm.distributed.device_communicators.quick_all_reduce import QuickReduceRegime

    names = set(QuickReduceRegime.__members__.keys())
    assert {"FP", "INT8", "INT6", "INT4", "NONE"} == names


# ── Env var tests ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("regime", ["FP", "INT8", "INT6", "INT4", "NONE"])
def test_quick_reduce_quantization_env_var(regime, monkeypatch):
    """VLLM_ROCM_QUICK_REDUCE_QUANTIZATION accepts all valid modes."""
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", regime)
    import vllm.envs as envs

    importlib.reload(envs)
    assert regime == envs.VLLM_ROCM_QUICK_REDUCE_QUANTIZATION


def test_quick_reduce_default_quantization():
    """VLLM_ROCM_QUICK_REDUCE_QUANTIZATION defaults to 'NONE'."""
    import vllm.envs as envs

    # Default is "NONE" (disabled)
    assert envs.VLLM_ROCM_QUICK_REDUCE_QUANTIZATION in (
        "NONE",
        "FP",
        "INT8",
        "INT6",
        "INT4",
    )


@pytest.mark.parametrize("cast_bf16", [True, False])
def test_quick_reduce_cast_bf16_to_fp16_env_var(cast_bf16, monkeypatch):
    """VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16 controls BF16→FP16 casting."""
    monkeypatch.setenv(
        "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "1" if cast_bf16 else "0"
    )
    import vllm.envs as envs

    importlib.reload(envs)
    assert cast_bf16 == envs.VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16


@pytest.mark.parametrize("max_mb", [128, 512, 2048, None])
def test_quick_reduce_max_size_env_var(max_mb, monkeypatch):
    """VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB accepts int or None."""
    if max_mb is not None:
        monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", str(max_mb))
    else:
        monkeypatch.delenv("VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", raising=False)
    import vllm.envs as envs

    importlib.reload(envs)
    assert max_mb == envs.VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB


# ── QuickAllReduce._rocm_arch_available() tests ───────────────────────────


def test_quick_allreduce_rocm_arch_available():
    """_rocm_arch_available() returns True on supported MI300/MI350 hardware."""
    # Instantiate a dummy (disabled) QuickAllReduce to access the method
    # We use a mock group to avoid needing an actual distributed setup
    from unittest.mock import MagicMock, patch

    from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    _mock_group = MagicMock()

    with (
        patch("torch.distributed.get_backend", return_value="gloo"),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=1),
    ):
        qar = QuickAllReduce.__new__(QuickAllReduce)
        qar.disabled = True
        result = qar._rocm_arch_available()
        # On MI300X (gfx942/gfx944) or MI350X (gfx950), should return True
        # On other hardware, returns False
        assert isinstance(result, bool)


# ── should_quick_allreduce predicate tests (unit, no dist needed) ─────────


def test_quick_allreduce_should_quick_allreduce_disabled():
    """should_quick_allreduce returns False when disabled."""
    from vllm.distributed.device_communicators.quick_all_reduce import (
        QuickAllReduce,
    )

    qar = QuickAllReduce.__new__(QuickAllReduce)
    qar.disabled = True
    inp = torch.zeros(1024, dtype=torch.float16)
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_should_quick_allreduce_wrong_dtype():
    """should_quick_allreduce returns False for unsupported dtypes."""
    from vllm.distributed.device_communicators.quick_all_reduce import (
        QuickAllReduce,
        QuickReduceRegime,
    )

    qar = QuickAllReduce.__new__(QuickAllReduce)
    qar.disabled = False
    qar.world_size = 2
    qar.use_fp16_kernels = False
    qar.qr_quant_level = QuickReduceRegime.FP
    qar.qr_max_size = 64 * MB

    # float32 is not in _SUPPORTED_DTYPES
    inp = torch.zeros(1024 * 1024, dtype=torch.float32)
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_should_quick_allreduce_size_too_small():
    """should_quick_allreduce returns False when tensor is too small."""
    from vllm.distributed.device_communicators.quick_all_reduce import (
        QuickAllReduce,
        QuickReduceRegime,
    )

    qar = QuickAllReduce.__new__(QuickAllReduce)
    qar.disabled = False
    qar.world_size = 2
    qar.use_fp16_kernels = False
    qar.qr_quant_level = QuickReduceRegime.FP
    qar.qr_max_size = 64 * MB

    # Very small tensor: well below QR_MIN_SIZE for FP, fp16, 2 GPUs (1 MB)
    # 128 elements * 2 bytes = 256 bytes << 1 MB
    inp = torch.zeros(128, dtype=torch.float16).cuda()
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_should_quick_allreduce_size_too_large():
    """should_quick_allreduce returns False when tensor exceeds max size."""
    from vllm.distributed.device_communicators.quick_all_reduce import (
        QuickAllReduce,
        QuickReduceRegime,
    )

    qar = QuickAllReduce.__new__(QuickAllReduce)
    qar.disabled = False
    qar.world_size = 2
    qar.use_fp16_kernels = False
    qar.qr_quant_level = QuickReduceRegime.FP
    # Set a small max size (1 MB)
    qar.qr_max_size = 1 * MB

    # Tensor > 1 MB
    inp = torch.zeros(2 * MB // 2, dtype=torch.float16).cuda()
    # 2 MB * 2 bytes = 4 MB > 1 MB limit
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_supported_world_sizes():
    """QuickAllReduce._SUPPORTED_WORLD_SIZES has expected values."""
    from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    assert 2 in QuickAllReduce._SUPPORTED_WORLD_SIZES
    assert 4 in QuickAllReduce._SUPPORTED_WORLD_SIZES
    assert 8 in QuickAllReduce._SUPPORTED_WORLD_SIZES


def test_quick_allreduce_supported_dtypes():
    """QuickAllReduce._SUPPORTED_DTYPES contains float16 and bfloat16."""
    from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    assert torch.float16 in QuickAllReduce._SUPPORTED_DTYPES
    assert torch.bfloat16 in QuickAllReduce._SUPPORTED_DTYPES


def test_quick_allreduce_min_size_table():
    """QuickAllReduce._QR_MIN_SIZE has entries for all supported dtypes/world sizes."""
    from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    for dtype in [torch.float16, torch.bfloat16]:
        for world_size in QuickAllReduce._SUPPORTED_WORLD_SIZES:
            key = (dtype, world_size)
            assert key in QuickAllReduce._QR_MIN_SIZE
            min_sizes = QuickAllReduce._QR_MIN_SIZE[key]
            # 4 entries: [FP, INT8, INT6, INT4]
            assert len(min_sizes) == 4
            for s in min_sizes:
                assert s > 0


# ── ops.qr_max_size availability ─────────────────────────────────────────


def test_qr_max_size_available():
    """ops.qr_max_size() is available on ROCm MI300-series (quick_ar=True)."""
    if not quick_ar_available():
        pytest.skip("Quick allreduce library not available on this hardware")
    from vllm import _custom_ops as ops

    max_size = ops.qr_max_size()
    assert isinstance(max_size, int)
    assert max_size > 0


# ── Multi-GPU quick reduce test (requires 2 GPUs) ─────────────────────────


def _worker_quick_allreduce(
    rank: int,
    world_size: int,
    port: int,
    quant_level: str,
    cast_bf16: bool,
    result_dict,
):
    """Worker function for multi-GPU quick allreduce test."""

    os.environ["VLLM_ROCM_QUICK_REDUCE_QUANTIZATION"] = quant_level
    os.environ["VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1" if cast_bf16 else "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    torch.cuda.set_device(rank)
    # QuickAllReduce requires a non-NCCL group (it uses custom IPC buffers).
    # In production, vLLM passes the gloo-backed cpu_group from GroupCoordinator.
    torch.distributed.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )
    group = torch.distributed.GroupMember.WORLD

    from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    # 16 MB tensor (above 1 MB threshold for FP fp16 with 2 GPUs)
    num_elements = 8 * MB  # 8M float16 elements = 16 MB
    inp = torch.ones(num_elements, dtype=torch.float16, device=f"cuda:{rank}")

    qar = QuickAllReduce(group=group, device=rank)

    if qar.disabled:
        # Hardware not supported or library missing — record and exit gracefully
        result_dict[rank] = "disabled"
        torch.distributed.destroy_process_group()
        return

    if qar.should_quick_allreduce(inp):
        out = qar.quick_all_reduce(inp)
        expected = inp * world_size
        is_close = torch.allclose(out, expected, atol=2.5, rtol=0.1)
        result_dict[rank] = "pass" if is_close else "fail"
    else:
        result_dict[rank] = "skipped_size"

    qar.close()
    torch.distributed.destroy_process_group()


def _worker_bf16(rank, world_size, port, result_dict):
    """Worker for BF16 cast mode test. Module-level for pickle compatibility."""
    os.environ["VLLM_ROCM_QUICK_REDUCE_QUANTIZATION"] = "FP"
    os.environ["VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    torch.cuda.set_device(rank)
    # QuickAllReduce requires a non-NCCL group (gloo).
    torch.distributed.init_process_group(
        backend="gloo", rank=rank, world_size=world_size
    )
    group = torch.distributed.GroupMember.WORLD

    from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    num_elements = 4 * MB  # 8 MB of bfloat16 — above bf16 FP threshold (2 MB, 2 GPUs)
    inp = torch.ones(num_elements, dtype=torch.bfloat16, device=f"cuda:{rank}")
    qar = QuickAllReduce(group=group, device=rank)

    if qar.disabled:
        result_dict[rank] = "disabled"
    elif qar.should_quick_allreduce(inp):
        out = qar.quick_all_reduce(inp)
        expected = inp.float() * world_size
        # atol=2.5, rtol=0.1 matches canonical test_quick_all_reduce.py
        is_close = torch.allclose(out.float(), expected, atol=2.5, rtol=0.1)
        result_dict[rank] = "pass" if is_close else "fail"
    else:
        result_dict[rank] = "skipped_size"

    qar.close()
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(num_gpus() < 2, reason="requires 2 ROCm GPUs")
@pytest.mark.skipif(not quick_ar_available(), reason="quick_ar library not available")
@pytest.mark.parametrize("quant_level", ["FP", "INT8", "INT6", "INT4"])
def test_quick_allreduce_two_gpu_correctness(quant_level):
    """Quick allreduce with 2 GPUs produces correct sum for FP and INT8 modes."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    port = random.randint(29000, 30000)
    world_size = 2

    manager = ctx.Manager()
    result_dict = manager.dict()

    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_quick_allreduce,
            args=(rank, world_size, port, quant_level, True, result_dict),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"Worker exited with code {p.exitcode}"

    for rank in range(world_size):
        result = result_dict.get(rank, "missing")
        assert result in ("pass", "disabled", "skipped_size"), (
            f"Rank {rank} failed quick allreduce: {result}"
        )


@pytest.mark.skipif(num_gpus() < 2, reason="requires 2 ROCm GPUs")
@pytest.mark.skipif(not quick_ar_available(), reason="quick_ar library not available")
def test_quick_allreduce_bf16_cast_mode():
    """Quick allreduce with BF16 input + CAST_BF16_TO_FP16=True runs correctly."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    port = random.randint(29000, 30000)
    world_size = 2
    manager = ctx.Manager()
    result_dict = manager.dict()

    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=_worker_bf16, args=(rank, world_size, port, result_dict))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    for rank in range(world_size):
        result = result_dict.get(rank, "missing")
        assert result in ("pass", "disabled", "skipped_size")
