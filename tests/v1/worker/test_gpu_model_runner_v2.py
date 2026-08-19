# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.compilation.cuda_graph import (
    CUDAGraphTeardownError,
    CUDAGraphTeardownFailure,
    CUDAGraphTeardownStats,
    CUDAGraphWrapper,
)
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_registry_isolated_when_vllm_config_is_reused(
    monkeypatch: pytest.MonkeyPatch, default_vllm_config: VllmConfig
) -> None:
    class StopRunnerInitialization(Exception):
        pass

    def stop_runner_initialization(_vllm_config: VllmConfig) -> None:
        raise StopRunnerInitialization

    monkeypatch.setattr(
        model_runner_module,
        "JitWarmupRegistry",
        stop_runner_initialization,
    )
    monkeypatch.setattr(
        model_runner_module.current_platform,
        "is_rocm",
        lambda: False,
    )
    monkeypatch.setattr(
        model_runner_module.current_platform,
        "get_global_graph_pool",
        object,
    )

    def partially_construct_runner() -> GPUModelRunner:
        runner = object.__new__(GPUModelRunner)
        with pytest.raises(StopRunnerInitialization):
            runner.__init__(default_vllm_config, torch.device("cpu"))
        return runner

    original_compilation_config = default_vllm_config.compilation_config
    vars(original_compilation_config)["_cudagraph_owner_registry_token"] = (
        "caller-owned-token"
    )
    preexisting_layer = object()
    original_compilation_config.static_forward_context["preexisting"] = (
        preexisting_layer
    )
    runner_a = partially_construct_runner()
    runner_b = partially_construct_runner()

    late_owner_a = CUDAGraphWrapper(
        lambda: None,
        runner_a.vllm_config,
        CUDAGraphMode.FULL,
    )
    late_owner_b = CUDAGraphWrapper(
        lambda: None,
        runner_b.vllm_config,
        CUDAGraphMode.FULL,
    )
    later_owner_a = CUDAGraphWrapper(
        lambda: None,
        runner_a.vllm_config,
        CUDAGraphMode.FULL,
    )

    assert default_vllm_config.compilation_config is original_compilation_config
    assert runner_a.vllm_config is not default_vllm_config
    assert runner_b.vllm_config is not default_vllm_config
    assert runner_a.vllm_config is not runner_b.vllm_config
    assert runner_a.compilation_config is not original_compilation_config
    assert runner_b.compilation_config is not original_compilation_config
    assert runner_a.compilation_config is not runner_b.compilation_config
    assert (
        runner_a.compilation_config.static_forward_context
        is not runner_b.compilation_config.static_forward_context
    )
    assert "preexisting" not in runner_a.compilation_config.static_forward_context
    assert "preexisting" not in runner_b.compilation_config.static_forward_context
    assert original_compilation_config.static_forward_context["preexisting"] is (
        preexisting_layer
    )
    assert (
        vars(original_compilation_config)["_cudagraph_owner_registry_token"]
        == "caller-owned-token"
    )
    assert runner_a.cudagraph_owner_registry.owners() == (
        late_owner_a,
        later_owner_a,
    )
    assert runner_b.cudagraph_owner_registry.owners() == (late_owner_b,)


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""

    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size),
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.blocks_per_kv_block = [1]
    block_tables.block_tables = [_BlockTable()]
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3


class _ShutdownEventList(list):
    def __init__(self, events: list[str], name: str):
        super().__init__([object()])
        self.events = events
        self.name = name

    def clear(self) -> None:
        self.events.append(self.name)
        super().clear()


def _make_teardown_stats(
    error: Exception | None = None, *, fallback_required: bool = False
) -> CUDAGraphTeardownStats:
    failures = (
        (CUDAGraphTeardownFailure("reset", "TestOwner", error, "reset failed"),)
        if error is not None
        else ()
    )
    return CUDAGraphTeardownStats(
        graph_count=int(error is not None),
        owner_graph_counts={"TestOwner": int(error is not None)},
        enumeration_duration_s=0,
        reset_duration_s=0,
        clear_duration_s=0,
        total_duration_s=0,
        failures=failures,
        _retained_owner_ids=(frozenset({1}) if fallback_required else frozenset()),
    )


def _make_shutdown_runner(
    events: list[str], stats: CUDAGraphTeardownStats
) -> tuple[GPUModelRunner, Mock, Mock]:
    runner = object.__new__(GPUModelRunner)
    registry = Mock()
    registry.teardown.side_effect = lambda **_: events.append("teardown") or stats
    speculator = Mock()
    speculator.shutdown.side_effect = lambda: events.append("speculator")
    runner.cudagraph_owner_registry = registry
    runner.speculator = speculator
    runner.cudagraph_manager = object()
    runner.kv_caches = _ShutdownEventList(events, "kv")
    runner.attn_groups = _ShutdownEventList(events, "attention")
    runner.vllm_config = SimpleNamespace()
    runner.model = object()
    return runner, registry, speculator


def _patch_shutdown_dependencies(monkeypatch, events: list[str], *, rocm: bool):
    monkeypatch.setattr(model_runner_module.current_platform, "is_rocm", lambda: rocm)
    monkeypatch.setattr(
        model_runner_module.torch.accelerator,
        "synchronize",
        lambda: events.append("synchronize"),
    )
    monkeypatch.setattr(
        model_runner_module.torch.accelerator,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )
    monkeypatch.setattr(
        model_runner_module.gc, "collect", lambda: events.append("gc_collect")
    )
    monkeypatch.setattr(
        model_runner_module,
        "free_before_shutdown",
        lambda _: events.append("workspace"),
    )


def test_rocm_shutdown_tears_down_graphs_before_model_state(monkeypatch):
    events: list[str] = []
    runner, registry, speculator = _make_shutdown_runner(events, _make_teardown_stats())
    _patch_shutdown_dependencies(monkeypatch, events, rocm=True)

    runner.shutdown()

    assert events == [
        "synchronize",
        "teardown",
        "speculator",
        "kv",
        "attention",
        "workspace",
        "gc_collect",
        "empty_cache",
    ]
    registry.teardown.assert_called_once_with(post_reset_sync=True)
    speculator.shutdown.assert_called_once_with()
    assert runner.speculator is None
    assert not hasattr(runner, "model")


def test_rocm_shutdown_finishes_cleanup_before_reporting_graph_failure(monkeypatch):
    events: list[str] = []
    runner, _, _ = _make_shutdown_runner(
        events, _make_teardown_stats(RuntimeError("reset failed"))
    )

    _patch_shutdown_dependencies(monkeypatch, events, rocm=True)

    with pytest.raises(CUDAGraphTeardownError, match="reset failed"):
        runner.shutdown()

    assert events == [
        "synchronize",
        "teardown",
        "speculator",
        "kv",
        "attention",
        "workspace",
        "gc_collect",
        "empty_cache",
    ]
    assert not hasattr(runner, "model")


def test_rocm_shutdown_does_not_mask_model_cleanup_failure(monkeypatch):
    events: list[str] = []
    runner, _, speculator = _make_shutdown_runner(
        events, _make_teardown_stats(RuntimeError("reset failed"))
    )
    log_error = Mock()

    def fail_speculator_shutdown() -> None:
        events.append("speculator")
        raise RuntimeError("speculator cleanup failed")

    speculator.shutdown.side_effect = fail_speculator_shutdown
    _patch_shutdown_dependencies(monkeypatch, events, rocm=True)
    monkeypatch.setattr(model_runner_module.logger, "error", log_error)

    with pytest.raises(RuntimeError, match="speculator cleanup failed"):
        runner.shutdown()

    assert events == ["synchronize", "teardown", "speculator"]
    assert "reset failed" in str(log_error.call_args_list)


def test_rocm_shutdown_preserves_graph_dependencies_for_fallback(monkeypatch):
    events: list[str] = []
    runner, registry, speculator = _make_shutdown_runner(
        events,
        _make_teardown_stats(
            RuntimeError("reset failed"),
            fallback_required=True,
        ),
    )
    model = runner.model
    cudagraph_manager = runner.cudagraph_manager
    kv_caches = runner.kv_caches
    attention_groups = runner.attn_groups
    model_state = SimpleNamespace(
        supports_mm_inputs=True,
        encoder_runner=Mock(),
    )
    runner.model_state = model_state
    runner.kv_cache_config = object()
    kv_cache_config = runner.kv_cache_config
    _patch_shutdown_dependencies(monkeypatch, events, rocm=True)

    with pytest.raises(CUDAGraphTeardownError, match="reset failed"):
        runner.shutdown()

    assert runner._cudagraph_teardown_incomplete
    registry.retain_fallback_dependency.assert_called_once_with(runner)
    speculator.shutdown.assert_not_called()
    model_state.encoder_runner.clear.assert_not_called()
    assert events == ["synchronize", "teardown"]
    assert runner.speculator is speculator
    assert runner.model is model
    assert runner.model_state is model_state
    assert runner.cudagraph_manager is cudagraph_manager
    assert runner.kv_caches is kv_caches
    assert runner.attn_groups is attention_groups
    assert runner.kv_cache_config is kv_cache_config


def test_rocm_shutdown_retains_after_coordinator_exception(monkeypatch):
    events: list[str] = []
    runner, registry, speculator = _make_shutdown_runner(events, _make_teardown_stats())
    model = runner.model
    cudagraph_manager = runner.cudagraph_manager
    kv_caches = runner.kv_caches
    log_exception = Mock()
    registry.teardown.side_effect = RuntimeError("coordinator failed")
    _patch_shutdown_dependencies(monkeypatch, events, rocm=True)
    monkeypatch.setattr(model_runner_module.logger, "exception", log_exception)

    with pytest.raises(RuntimeError, match="coordinator failed"):
        runner.shutdown()

    assert runner._cudagraph_teardown_incomplete
    registry.retain_fallback_dependency.assert_called_once_with(runner)
    speculator.shutdown.assert_not_called()
    assert events == ["synchronize"]
    assert runner.speculator is speculator
    assert runner.model is model
    assert runner.cudagraph_manager is cudagraph_manager
    assert runner.kv_caches is kv_caches
    log_exception.assert_called_once()
    assert "failed unexpectedly" in log_exception.call_args.args[0]


def test_non_rocm_shutdown_preserves_implicit_graph_cleanup(monkeypatch):
    events: list[str] = []
    runner, registry, speculator = _make_shutdown_runner(events, _make_teardown_stats())
    _patch_shutdown_dependencies(monkeypatch, events, rocm=False)

    runner.shutdown()

    registry.teardown.assert_not_called()
    speculator.shutdown.assert_not_called()
    assert events == [
        "synchronize",
        "kv",
        "attention",
        "workspace",
        "gc_collect",
        "empty_cache",
    ]
    assert runner.speculator is None
    assert not hasattr(runner, "model")


def test_non_rocm_shutdown_sync_failure_is_fail_fast(monkeypatch):
    events: list[str] = []
    runner, registry, speculator = _make_shutdown_runner(events, _make_teardown_stats())
    model = runner.model
    kv_caches = runner.kv_caches
    _patch_shutdown_dependencies(monkeypatch, events, rocm=False)

    def fail_sync() -> None:
        events.append("synchronize")
        raise RuntimeError("sync failed")

    monkeypatch.setattr(model_runner_module.torch.accelerator, "synchronize", fail_sync)

    with pytest.raises(RuntimeError, match="sync failed"):
        runner.shutdown()

    assert events == ["synchronize"]
    registry.teardown.assert_not_called()
    speculator.shutdown.assert_not_called()
    assert runner.model is model
    assert runner.kv_caches is kv_caches


def test_rocm_sync_failure_retains_device_state(monkeypatch):
    events: list[str] = []
    runner, registry, speculator = _make_shutdown_runner(events, _make_teardown_stats())
    model = runner.model
    log_exception = Mock()
    _patch_shutdown_dependencies(monkeypatch, events, rocm=True)
    monkeypatch.setattr(model_runner_module.logger, "exception", log_exception)

    monkeypatch.setattr(
        model_runner_module.torch.accelerator,
        "synchronize",
        lambda: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )

    with pytest.raises(RuntimeError, match="sync failed"):
        runner.shutdown()

    registry.teardown.assert_not_called()
    speculator.shutdown.assert_not_called()
    assert runner._cudagraph_teardown_incomplete
    assert runner._terminal_fallback_cycle is runner
    assert runner.model is model
    log_exception.assert_called_once()
    assert "synchronization failed" in log_exception.call_args.args[0]
