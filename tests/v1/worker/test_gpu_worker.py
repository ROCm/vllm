# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import startup_plan
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)

# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


def test_shutdown_does_not_reset_graphs_after_rocm_quiesce_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    worker = object.__new__(Worker)
    worker.profiler = Mock(shutdown=Mock(side_effect=lambda: events.append("profiler")))
    worker.weight_transfer_engine = Mock(
        shutdown=Mock(side_effect=lambda: events.append("weight_transfer"))
    )

    def model_runner_shutdown() -> None:
        events.append("model_runner")
        raise RuntimeError("graph teardown failed")

    worker.model_runner = Mock(shutdown=Mock(side_effect=model_runner_shutdown))
    worker.model_runner._cudagraph_teardown_incomplete = False

    def kv_shutdown() -> None:
        events.append("kv_transfer")
        raise RuntimeError("KV transfer failed")

    monkeypatch.setattr(
        gpu_worker_module.gc,
        "unfreeze",
        lambda: events.append("unfreeze"),
    )
    monkeypatch.setattr(gpu_worker_module, "ensure_kv_transfer_shutdown", kv_shutdown)
    monkeypatch.setattr(
        gpu_worker_module,
        "ensure_ec_transfer_shutdown",
        lambda: events.append("ec_transfer"),
    )
    monkeypatch.setattr(
        gpu_worker_module.current_platform,
        "is_cuda_alike",
        lambda: True,
    )
    monkeypatch.setattr(
        gpu_worker_module.current_platform,
        "is_rocm",
        lambda: True,
    )
    retain_dependency = Mock()
    monkeypatch.setattr(
        gpu_worker_module,
        "retain_cudagraph_dependency_for_terminal_exit",
        retain_dependency,
    )
    allocator = Mock(
        release_pools=Mock(side_effect=lambda: events.append("release_pools"))
    )
    monkeypatch.setattr(CuMemAllocator, "instance", allocator)

    with pytest.raises(RuntimeError, match="KV transfer failed"):
        worker.shutdown()

    assert events == [
        "unfreeze",
        "kv_transfer",
        "ec_transfer",
        "profiler",
        "weight_transfer",
    ]
    # A failed quiescence phase retains the complete dependency set.
    assert worker.profiler is not None
    assert worker.weight_transfer_engine is not None
    assert worker.model_runner is not None
    assert worker._cudagraph_teardown_incomplete
    worker.model_runner.shutdown.assert_not_called()
    allocator.release_pools.assert_not_called()
    retain_dependency.assert_called_once_with(worker)


def test_shutdown_preserves_non_rocm_fail_fast_after_connector_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    worker = object.__new__(Worker)
    worker.profiler = Mock(shutdown=Mock(side_effect=lambda: events.append("profiler")))
    worker.weight_transfer_engine = Mock(
        shutdown=Mock(side_effect=lambda: events.append("weight_transfer"))
    )
    worker.model_runner = Mock(
        shutdown=Mock(side_effect=lambda: events.append("model_runner")),
        _cudagraph_teardown_incomplete=False,
    )
    worker._cudagraph_teardown_incomplete = False

    def kv_shutdown() -> None:
        events.append("kv_transfer")
        raise RuntimeError("KV transfer failed")

    monkeypatch.setattr(
        gpu_worker_module.gc,
        "unfreeze",
        lambda: events.append("unfreeze"),
    )
    monkeypatch.setattr(gpu_worker_module, "ensure_kv_transfer_shutdown", kv_shutdown)
    monkeypatch.setattr(
        gpu_worker_module,
        "ensure_ec_transfer_shutdown",
        lambda: events.append("ec_transfer"),
    )
    monkeypatch.setattr(
        gpu_worker_module.current_platform,
        "is_cuda_alike",
        lambda: True,
    )
    monkeypatch.setattr(
        gpu_worker_module.current_platform,
        "is_rocm",
        lambda: False,
    )
    retain_dependency = Mock()
    monkeypatch.setattr(
        gpu_worker_module,
        "retain_cudagraph_dependency_for_terminal_exit",
        retain_dependency,
    )
    allocator = Mock(
        release_pools=Mock(side_effect=lambda: events.append("release_pools"))
    )
    monkeypatch.setattr(CuMemAllocator, "instance", allocator)

    with pytest.raises(RuntimeError, match="KV transfer failed"):
        worker.shutdown()

    assert events == ["unfreeze", "kv_transfer"]
    assert not worker._cudagraph_teardown_incomplete
    assert worker.model_runner is not None
    assert worker.profiler is not None
    assert worker.weight_transfer_engine is not None
    worker.model_runner.shutdown.assert_not_called()
    worker.profiler.shutdown.assert_not_called()
    worker.weight_transfer_engine.shutdown.assert_not_called()
    allocator.release_pools.assert_not_called()
    retain_dependency.assert_not_called()


@pytest.mark.parametrize("runner_marks_incomplete", [False, True])
def test_shutdown_retains_runner_and_pools_after_runner_failure(
    monkeypatch: pytest.MonkeyPatch,
    runner_marks_incomplete: bool,
) -> None:
    events: list[str] = []
    worker = object.__new__(Worker)
    worker.profiler = Mock(shutdown=Mock(side_effect=lambda: events.append("profiler")))
    worker.weight_transfer_engine = Mock(
        shutdown=Mock(side_effect=lambda: events.append("weight_transfer"))
    )
    model_runner = Mock()
    model_runner._cudagraph_teardown_incomplete = False

    def model_runner_shutdown() -> None:
        events.append("model_runner")
        if runner_marks_incomplete:
            model_runner._cudagraph_teardown_incomplete = True
        raise RuntimeError("graph teardown failed")

    model_runner.shutdown.side_effect = model_runner_shutdown
    worker.model_runner = model_runner

    monkeypatch.setattr(
        gpu_worker_module.gc,
        "unfreeze",
        lambda: events.append("unfreeze"),
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "ensure_kv_transfer_shutdown",
        lambda: events.append("kv_transfer"),
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "ensure_ec_transfer_shutdown",
        lambda: events.append("ec_transfer"),
    )
    monkeypatch.setattr(
        gpu_worker_module.current_platform,
        "is_cuda_alike",
        lambda: True,
    )
    monkeypatch.setattr(
        gpu_worker_module.current_platform,
        "is_rocm",
        lambda: True,
    )
    allocator = Mock(
        release_pools=Mock(side_effect=lambda: events.append("release_pools"))
    )
    monkeypatch.setattr(CuMemAllocator, "instance", allocator)

    with pytest.raises(RuntimeError, match="graph teardown failed"):
        worker.shutdown()

    assert events == [
        "unfreeze",
        "kv_transfer",
        "ec_transfer",
        "profiler",
        "weight_transfer",
        "model_runner",
    ]
    assert worker._cudagraph_teardown_incomplete
    assert worker.model_runner is model_runner
    allocator.release_pools.assert_not_called()
    assert worker.profiler is None
    assert worker.weight_transfer_engine is None
