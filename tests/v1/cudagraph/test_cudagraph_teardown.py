# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import gc
import threading
import time
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.compilation.cuda_graph as cuda_graph_module
from vllm.compilation.cuda_graph import (
    CUDAGraphOwnerRegistry,
    CUDAGraphTeardownError,
    OwnedCUDAGraph,
    begin_cudagraph_owner_teardown,
    create_cudagraph,
    cudagraph_capture_attempt,
    cudagraph_owner_activity,
    register_cudagraph_owner,
    teardown_cudagraphs,
)


class FakeGraph:
    def __init__(
        self,
        name: str,
        events: list[str],
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.error = error
        self.reset_count = 0

    def reset(self) -> None:
        self.events.append(f"reset:{self.name}")
        self.reset_count += 1
        if self.error is not None:
            raise self.error


class FakeOwner:
    def __init__(
        self,
        name: str,
        events: list[str],
        graphs: list[OwnedCUDAGraph],
        *,
        begin_error: Exception | None = None,
        clear_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.graphs = graphs
        self.begin_error = begin_error
        self.clear_error = clear_error

    def begin_cudagraph_teardown(self) -> None:
        self.events.append(f"begin:{self.name}")
        if self.begin_error is not None:
            raise self.begin_error

    def iter_cudagraphs(self):
        self.events.append(f"enumerate:{self.name}")
        yield from self.graphs

    def clear_cudagraph_state(self) -> None:
        self.events.append(f"clear:{self.name}")
        self.graphs.clear()
        if self.clear_error is not None:
            raise self.clear_error


def fake_config():
    return SimpleNamespace(compilation_config=SimpleNamespace())


def fake_registry(config=None):
    return CUDAGraphOwnerRegistry(
        fake_config() if config is None else config,
        synchronize=lambda _: None,
        device_context=lambda _: nullcontext(),
    )


def test_teardown_orders_phases_groups_devices_and_deduplicates() -> None:
    events: list[str] = []
    graph_0 = FakeGraph("zero", events)
    graph_1 = FakeGraph("one", events)
    owner_a = FakeOwner(
        "a",
        events,
        [
            OwnedCUDAGraph(graph_1, torch.device("cuda:1")),
            OwnedCUDAGraph(graph_0, torch.device("cuda:0")),
        ],
    )
    owner_b = FakeOwner("b", events, [OwnedCUDAGraph(graph_0, torch.device("cuda:0"))])

    stats = teardown_cudagraphs(
        (owner_a, owner_b),
        post_reset_sync=True,
        synchronize=lambda device: events.append(f"sync:{device}"),
        device_context=lambda _: nullcontext(),
    )

    assert events == [
        "begin:a",
        "begin:b",
        "enumerate:a",
        "enumerate:b",
        "sync:cuda:0",
        "sync:cuda:1",
        "reset:zero",
        "reset:one",
        "sync:cuda:0",
        "sync:cuda:1",
        "clear:a",
        "clear:b",
    ]
    assert stats.graph_count == 2
    assert graph_0.reset_count == graph_1.reset_count == 1
    assert not stats.failures


def test_reset_duration_includes_mandatory_pre_and_post_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_ns = [0]

    class TimedGraph(FakeGraph):
        def reset(self) -> None:
            clock_ns[0] += 5
            super().reset()

    graph = TimedGraph("timed", [])
    owner = FakeOwner("owner", [], [OwnedCUDAGraph(graph, torch.device("cuda:0"))])
    monkeypatch.setattr(time, "perf_counter_ns", lambda: clock_ns[0])

    def synchronize(_: torch.device) -> None:
        clock_ns[0] += 10

    stats = teardown_cudagraphs(
        [owner],
        post_reset_sync=True,
        synchronize=synchronize,
        device_context=lambda _: nullcontext(),
        emit_markers=False,
    )

    assert stats.reset_duration_s == pytest.approx(25e-9)


def test_teardown_attempts_every_resource_and_reports_capability_errors() -> None:
    events: list[str] = []
    broken = FakeGraph("broken", events, RuntimeError("reset failed"))
    healthy = FakeGraph("healthy", events)
    unquiesced = FakeGraph("unquiesced", events)
    missing_reset = object()
    owner_a = FakeOwner(
        "a",
        events,
        [
            OwnedCUDAGraph(broken, torch.device("cuda:0")),
            OwnedCUDAGraph(missing_reset, torch.device("cuda:0")),
        ],
    )
    owner_b = FakeOwner(
        "b",
        events,
        [OwnedCUDAGraph(healthy, torch.device("cuda:1"))],
        clear_error=RuntimeError("clear failed"),
    )
    owner_c = FakeOwner(
        "c",
        events,
        [OwnedCUDAGraph(unquiesced, torch.device("cuda:2"))],
        begin_error=RuntimeError("begin failed"),
    )

    stats = teardown_cudagraphs(
        (owner_a, owner_b, owner_c),
        post_reset_sync=False,
        synchronize=lambda device: events.append(f"sync:{device}"),
        device_context=lambda _: nullcontext(),
    )

    assert broken.reset_count == healthy.reset_count == 1
    assert unquiesced.reset_count == 0
    assert "clear:a" not in events
    assert "clear:c" not in events
    assert events[-1] == "clear:b"
    assert {failure.phase for failure in stats.failures} == {
        "begin",
        "reset",
        "clear",
    }
    with pytest.raises(CUDAGraphTeardownError, match=r"4 failure\(s\) total"):
        stats.raise_if_failed()
    assert stats.fallback_required


def test_enumeration_keeps_graphs_yielded_before_an_iterator_failure() -> None:
    events: list[str] = []
    graph = FakeGraph("visible", events)

    class PartiallyEnumerableOwner(FakeOwner):
        def iter_cudagraphs(self):
            yield OwnedCUDAGraph(graph, torch.device("cuda:0"))
            raise RuntimeError("late enumeration failure")

    owner = PartiallyEnumerableOwner("owner", events, [])
    stats = teardown_cudagraphs(
        [owner],
        post_reset_sync=False,
        synchronize=lambda _: None,
        device_context=lambda _: nullcontext(),
    )

    assert graph.reset_count == 1
    assert stats.graph_count == 1
    assert [failure.phase for failure in stats.failures] == ["enumerate"]
    assert "clear:owner" not in events
    assert stats.fallback_required


def test_device_selection_failure_retains_graph_without_resetting() -> None:
    events: list[str] = []
    graph = FakeGraph("graph", events)
    owner = FakeOwner("owner", events, [OwnedCUDAGraph(graph, torch.device("cuda:0"))])

    def fail_device_context(_):
        raise RuntimeError("cannot select device")

    stats = teardown_cudagraphs(
        [owner],
        post_reset_sync=False,
        synchronize=lambda _: None,
        device_context=fail_device_context,
    )

    assert graph.reset_count == 0
    assert [failure.phase for failure in stats.failures] == ["select_device"]
    assert "clear:owner" not in events
    assert stats.fallback_required


def test_sync_failure_retains_only_affected_owner_state() -> None:
    events: list[str] = []
    failed_graph = FakeGraph("failed-device", events)
    healthy_graph = FakeGraph("healthy-device", events)
    failed_owner = FakeOwner(
        "failed",
        events,
        [OwnedCUDAGraph(failed_graph, torch.device("cuda:0"))],
    )
    healthy_owner = FakeOwner(
        "healthy",
        events,
        [OwnedCUDAGraph(healthy_graph, torch.device("cuda:1"))],
    )

    def synchronize(device: torch.device) -> None:
        events.append(f"sync:{device}")
        if device == torch.device("cuda:0"):
            raise RuntimeError("device synchronization failed")

    stats = teardown_cudagraphs(
        [failed_owner, healthy_owner],
        post_reset_sync=False,
        synchronize=synchronize,
        device_context=lambda _: nullcontext(),
    )

    assert failed_graph.reset_count == 0
    assert healthy_graph.reset_count == 1
    assert failed_owner.graphs
    assert not healthy_owner.graphs
    assert "clear:failed" not in events
    assert "clear:healthy" in events
    assert stats.fallback_required
    assert [failure.phase for failure in stats.failures] == ["pre_reset_sync"]


def test_failed_quiescence_retains_every_owner_of_a_shared_graph() -> None:
    events: list[str] = []
    graph = FakeGraph("shared", events)
    failing_owner = FakeOwner(
        "failing",
        events,
        [OwnedCUDAGraph(graph, torch.device("cuda:0"))],
        begin_error=RuntimeError("quiesce failed"),
    )
    healthy_owner = FakeOwner(
        "healthy",
        events,
        [OwnedCUDAGraph(graph, torch.device("cuda:0"))],
    )

    stats = teardown_cudagraphs(
        [failing_owner, healthy_owner],
        post_reset_sync=False,
        synchronize=lambda _: None,
        device_context=lambda _: nullcontext(),
    )

    assert graph.reset_count == 0
    assert failing_owner.graphs and healthy_owner.graphs
    assert "clear:failing" not in events
    assert "clear:healthy" not in events
    assert stats.fallback_required


def test_conflicting_devices_for_shared_graph_are_retained_without_reset() -> None:
    events: list[str] = []
    graph = FakeGraph("shared", events)
    owner_a = FakeOwner("a", events, [OwnedCUDAGraph(graph, torch.device("cuda:0"))])
    owner_b = FakeOwner("b", events, [OwnedCUDAGraph(graph, torch.device("cuda:1"))])

    stats = teardown_cudagraphs(
        [owner_a, owner_b],
        post_reset_sync=False,
        synchronize=lambda _: None,
        device_context=lambda _: nullcontext(),
    )

    assert graph.reset_count == 0
    assert owner_a.graphs and owner_b.graphs
    assert [failure.phase for failure in stats.failures] == ["enumerate"]
    assert "conflicting devices" in str(stats.first_error)
    assert stats.fallback_required


def test_registry_is_engine_scoped_and_idempotent() -> None:
    events: list[str] = []
    config_a = fake_config()
    config_b = fake_config()
    registry_a = fake_registry(config_a)
    registry_b = fake_registry(config_b)
    graph_a = FakeGraph("a", events)
    graph_b = FakeGraph("b", events)
    owner_a = FakeOwner("a", events, [OwnedCUDAGraph(graph_a, torch.device("cuda:0"))])
    owner_b = FakeOwner("b", events, [OwnedCUDAGraph(graph_b, torch.device("cuda:0"))])
    register_cudagraph_owner(owner_a, config_a)
    register_cudagraph_owner(owner_b, config_b)

    first = registry_a.teardown(post_reset_sync=False)
    second = registry_a.teardown(post_reset_sync=False)

    assert first is second
    assert graph_a.reset_count == 1
    assert graph_b.reset_count == 0
    assert registry_b.owners() == (owner_b,)
    with pytest.raises(RuntimeError, match="after teardown"):
        register_cudagraph_owner(
            FakeOwner("late", events, []),
            config_a,
        )


def test_non_owning_registry_does_not_extend_owner_lifetime() -> None:
    config = fake_config()
    registry = CUDAGraphOwnerRegistry(
        config,
        strong_ownership=False,
        synchronize=lambda _: None,
        device_context=lambda _: nullcontext(),
    )
    owner = FakeOwner("owner", [], [])
    owner_ref = weakref.ref(owner)
    register_cudagraph_owner(owner, config)

    del owner
    gc.collect()

    assert owner_ref() is None
    assert not registry.owners()


def test_concurrent_registry_teardown_converges_on_one_reset() -> None:
    events: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    class BlockingGraph(FakeGraph):
        def reset(self) -> None:
            entered.set()
            assert release.wait(timeout=5)
            super().reset()

    registry = fake_registry()
    graph = BlockingGraph("graph", events)
    registry.register(
        FakeOwner("owner", events, [OwnedCUDAGraph(graph, torch.device("cuda:0"))])
    )
    results = []

    def teardown() -> None:
        results.append(registry.teardown(post_reset_sync=False))

    first = threading.Thread(target=teardown)
    second = threading.Thread(target=teardown)
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert graph.reset_count == 1
    assert len(results) == 2
    assert results[0] is results[1]


def test_same_thread_reentrant_teardown_does_not_deadlock_or_reset_twice() -> None:
    events: list[str] = []
    registry = fake_registry()
    graph = FakeGraph("graph", events)

    class ReentrantOwner(FakeOwner):
        def begin_cudagraph_teardown(self) -> None:
            events.append("begin:reentrant")
            registry.teardown(post_reset_sync=False)

    registry.register(
        ReentrantOwner("owner", events, [OwnedCUDAGraph(graph, torch.device("cuda:0"))])
    )

    stats = registry.teardown(post_reset_sync=False)

    assert stats.graph_count == 1
    assert graph.reset_count == 0
    assert [failure.phase for failure in stats.failures] == ["begin"]
    assert "already in progress" in str(stats.first_error)


def test_owner_teardown_waits_for_activity_and_rejects_new_calls() -> None:
    entered = threading.Event()
    release = threading.Event()

    class ActiveOwner:
        def __init__(self) -> None:
            self._cudagraph_teardown_started = False
            self._cudagraph_active_calls = 0
            self._cudagraph_activity_condition = threading.Condition()

        @cudagraph_owner_activity
        def replay(self) -> None:
            entered.set()
            assert release.wait(timeout=5)

        def begin_cudagraph_teardown(self) -> None:
            begin_cudagraph_owner_teardown(self)

    owner = ActiveOwner()
    replay_thread = threading.Thread(target=owner.replay)
    teardown_thread = threading.Thread(target=owner.begin_cudagraph_teardown)
    replay_thread.start()
    assert entered.wait(timeout=5)
    teardown_thread.start()

    assert teardown_thread.is_alive()
    release.set()
    replay_thread.join(timeout=5)
    teardown_thread.join(timeout=5)

    assert not replay_thread.is_alive()
    assert not teardown_thread.is_alive()
    with pytest.raises(RuntimeError, match="owner is closed"):
        owner.replay()


def test_owner_reentrant_teardown_from_active_call_fails_without_deadlock() -> None:
    class ReentrantOwner:
        def __init__(self) -> None:
            self._cudagraph_teardown_started = False

        @cudagraph_owner_activity
        def replay(self) -> None:
            begin_cudagraph_owner_teardown(self)

    owner = ReentrantOwner()
    errors: list[BaseException] = []

    def replay() -> None:
        try:
            owner.replay()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=replay)
    thread.start()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert "active call" in str(errors[0])
    with pytest.raises(RuntimeError, match="owner is closed"):
        owner.replay()


def test_reusable_reset_does_not_close_registry_or_owners() -> None:
    events: list[str] = []
    registry = fake_registry()
    first_graph = FakeGraph("first", events)
    owner = FakeOwner(
        "owner", events, [OwnedCUDAGraph(first_graph, torch.device("cuda:0"))]
    )
    registry.register(owner)

    first_stats = registry.reset_for_reuse(
        [owner],
        post_reset_sync=False,
    )
    second_graph = FakeGraph("second", events)
    owner.graphs.append(OwnedCUDAGraph(second_graph, torch.device("cuda:0")))
    final_stats = registry.teardown(post_reset_sync=False)

    assert first_stats.graph_count == final_stats.graph_count == 1
    assert first_graph.reset_count == second_graph.reset_count == 1
    assert events.index("begin:owner") < events.index("reset:first")
    assert events.count("begin:owner") == 2


def test_reusable_reset_blocks_replay_then_reopens_owner() -> None:
    reset_entered = threading.Event()
    release_reset = threading.Event()
    replay_events: list[str] = []

    class BlockingGraph(FakeGraph):
        def reset(self) -> None:
            reset_entered.set()
            assert release_reset.wait(timeout=5)
            super().reset()

    class ActiveOwner:
        def __init__(self, graph: FakeGraph) -> None:
            self.graphs = [OwnedCUDAGraph(graph, torch.device("cuda:0"))]

        @cudagraph_owner_activity
        def replay(self) -> None:
            replay_events.append("replay")

        def begin_cudagraph_teardown(self) -> None:
            begin_cudagraph_owner_teardown(self)

        def iter_cudagraphs(self):
            yield from self.graphs

        def clear_cudagraph_state(self) -> None:
            self.graphs.clear()

    registry = fake_registry()
    graph = BlockingGraph("blocking", [])
    owner = ActiveOwner(graph)
    registry.register(owner)
    results: list[object] = []
    reset_thread = threading.Thread(
        target=lambda: results.append(
            registry.reset_for_reuse([owner], post_reset_sync=False)
        )
    )
    reset_thread.start()
    assert reset_entered.wait(timeout=5)

    with pytest.raises(RuntimeError, match="owner is closed"):
        owner.replay()
    release_reset.set()
    reset_thread.join(timeout=5)

    assert not reset_thread.is_alive()
    assert results[0].graph_count == 1
    owner.replay()
    assert replay_events == ["replay"]


def test_failed_reusable_reset_closes_registry_and_owner() -> None:
    events: list[str] = []
    registry = fake_registry()
    graph = FakeGraph("failed", events, RuntimeError("reset failed"))
    owner = FakeOwner("owner", events, [OwnedCUDAGraph(graph, torch.device("cuda:0"))])
    registry.register(owner)

    stats = registry.reset_for_reuse([owner], post_reset_sync=False)

    assert stats.fallback_required
    assert graph.reset_count == 1
    assert events.count("begin:owner") == 1
    with pytest.raises(RuntimeError, match="after teardown"):
        registry.register(FakeOwner("late", events, []))
    assert registry.teardown(post_reset_sync=False) is stats
    assert graph.reset_count == 1


def test_failed_reusable_reset_retains_manual_owner() -> None:
    registry = fake_registry()
    graph = FakeGraph("failed", [], RuntimeError("reset failed"))
    owner = FakeOwner("manual", [], [OwnedCUDAGraph(graph, torch.device("cuda:0"))])
    owner._cudagraph_manual_owner = True
    owner_ref = weakref.ref(owner)

    stats = registry.reset_for_reuse([owner], post_reset_sync=False)
    del owner
    gc.collect()

    assert stats.fallback_required
    assert owner_ref() is not None
    assert registry._fallback_dependencies == [owner_ref()]


def test_reusable_reset_rejects_owner_from_another_registry() -> None:
    config_a = fake_config()
    config_b = fake_config()
    registry_a = fake_registry(config_a)
    registry_b = fake_registry(config_b)
    owner_b = FakeOwner("b", [], [])
    register_cudagraph_owner(owner_b, config_b)

    with pytest.raises(RuntimeError, match="another registry"):
        registry_a.reset_for_reuse([owner_b], post_reset_sync=False)

    assert registry_b.owners() == (owner_b,)


def test_terminal_retry_promotes_cached_failure_to_quarantine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quarantine: list[CUDAGraphOwnerRegistry] = []
    monkeypatch.setattr(
        cuda_graph_module, "_INCOMPLETE_TEARDOWN_REGISTRIES", quarantine
    )
    registry = fake_registry()
    graph = FakeGraph("failed", [], RuntimeError("reset failed"))
    registry.register(
        FakeOwner("owner", [], [OwnedCUDAGraph(graph, torch.device("cuda:0"))])
    )

    stats = registry.teardown(
        post_reset_sync=False,
        terminal_fallback=False,
    )
    assert stats.fallback_required
    assert not quarantine

    assert registry.teardown(post_reset_sync=False, terminal_fallback=True) is stats
    registry.teardown(post_reset_sync=False, terminal_fallback=True)
    assert quarantine == [registry]

    registry._terminal_fallback_cycle = None
    quarantine.clear()


def test_terminal_dependency_is_rooted_through_final_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quarantine: list[CUDAGraphOwnerRegistry] = []
    monkeypatch.setattr(
        cuda_graph_module, "_INCOMPLETE_TEARDOWN_REGISTRIES", quarantine
    )
    registry = fake_registry()
    assert not registry.teardown(post_reset_sync=False).fallback_required
    dependency = object()

    registry.retain_terminal_dependency(dependency)
    assert not quarantine
    with cuda_graph_module.terminal_cudagraph_teardown():
        registry.retain_terminal_dependency(dependency)

    assert quarantine == [registry]
    assert registry._fallback_dependencies == [dependency, dependency]
    registry._terminal_fallback_cycle = None
    quarantine.clear()


def test_unquiesced_dependency_is_rooted_only_for_terminal_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies: list[object] = []
    monkeypatch.setattr(
        cuda_graph_module,
        "_INCOMPLETE_TEARDOWN_DEPENDENCIES",
        dependencies,
    )
    dependency = SimpleNamespace()

    cuda_graph_module.retain_cudagraph_dependency_for_terminal_exit(dependency)
    assert dependency._terminal_fallback_cycle is dependency
    assert dependencies == []

    with cuda_graph_module.terminal_cudagraph_teardown():
        cuda_graph_module.retain_cudagraph_dependency_for_terminal_exit(dependency)
        cuda_graph_module.retain_cudagraph_dependency_for_terminal_exit(dependency)

    assert dependencies == [dependency]
    assert dependency._terminal_fallback_quarantined
    dependency._terminal_fallback_cycle = None
    dependencies.clear()


def test_failed_native_teardown_retains_graphs_for_terminal_fallback() -> None:
    events: list[str] = []
    registry = fake_registry()
    graph = FakeGraph("failed", events, RuntimeError("reset failed"))
    graph_ref = weakref.ref(graph)
    owner = FakeOwner("owner", events, [OwnedCUDAGraph(graph, torch.device("cuda:0"))])
    owner_ref = weakref.ref(owner)
    registry.register(owner)

    stats = registry.teardown(post_reset_sync=False)
    del graph
    del owner
    gc.collect()

    assert graph_ref() is not None
    assert registry.owners() == (owner_ref(),)
    assert stats.fallback_required
    assert "reset failed" in stats.failures[0].traceback_text

    del stats
    del registry
    gc.collect()
    assert graph_ref() is None
    assert owner_ref() is None


def test_every_direct_owner_enumerates_and_clears_its_graphs() -> None:
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        _GraphSegment,
    )
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

    events: list[str] = []
    device = torch.device("cuda:0")
    graphs = [FakeGraph(str(index), events) for index in range(6)]

    wrapper = CUDAGraphWrapper.__new__(CUDAGraphWrapper)
    wrapper.concrete_cudagraph_entries = {
        object(): SimpleNamespace(cudagraph=graphs[0], device=device)
    }

    capture = BreakableCUDAGraphCapture()
    capture.segments.append(_GraphSegment(graphs[1], device))

    manager = ModelCudaGraphManager.__new__(ModelCudaGraphManager)
    manager.device = device
    manager.graphs = {object(): graphs[2]}
    manager.breakable_cg_runner = None
    manager.aux_hidden_states = [object()]

    ubatch = UBatchWrapper.__new__(UBatchWrapper)
    ubatch.device = device
    ubatch.cudagraphs = {1: SimpleNamespace(cudagraph=graphs[3])}
    ubatch.cudagraph_wrapper = None

    encoder = EncoderCudaGraphManager.__new__(EncoderCudaGraphManager)
    encoder.device = device
    encoder.budget_graphs = {"default": {1: SimpleNamespace(graph=graphs[4])}}
    encoder.graph_pool = object()

    gemma = Gemma4Proposer.__new__(Gemma4Proposer)
    gemma.device = device
    gemma._centroids_sizes = [1]
    gemma._centroids_graphs = {1: graphs[5]}
    gemma._centroids_inputs = {1: object()}
    gemma._centroids_outputs = {1: object()}

    owners = (wrapper, capture, manager, ubatch, encoder, gemma)
    assert [list(owner.iter_cudagraphs())[0].graph for owner in owners] == graphs

    for owner in owners:
        owner.clear_cudagraph_state()

    assert not wrapper.concrete_cudagraph_entries
    assert not capture.segments
    assert not manager.graphs and not manager.aux_hidden_states
    assert not ubatch.cudagraphs
    assert not encoder.budget_graphs["default"] and encoder.graph_pool is None
    assert not gemma._centroids_graphs
    assert not gemma._centroids_inputs
    assert not gemma._centroids_outputs


def test_factory_rejects_a_protocol_owner_that_never_registered() -> None:
    owner = FakeOwner("owner", [], [])

    with pytest.raises(RuntimeError, match="registry is unavailable"):
        create_cudagraph(owner, torch.device("cuda:0"), lambda _: None)


def test_factory_tracks_graph_when_owner_install_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    config = fake_config()
    registry = fake_registry(config)
    owner = FakeOwner("owner", events, [])
    register_cudagraph_owner(owner, config)
    graph = FakeGraph("pending", events)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)

    def fail_install(_: object) -> None:
        raise RuntimeError("install failed")

    with pytest.raises(RuntimeError, match="install failed"):
        create_cudagraph(owner, torch.device("cuda:0"), fail_install)

    # The ordinary owner metadata was never populated, but the construction
    # helper's pending record still makes the native graph deterministic.
    assert not owner.graphs
    stats = registry.teardown(post_reset_sync=False)
    assert stats.graph_count == 1
    assert stats.owner_graph_counts == {"FakeOwner": 1}
    assert graph.reset_count == 1
    assert not stats.failures


def test_factory_rejects_stale_owner_after_registry_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = fake_config()
    registry = fake_registry(config)
    owner = FakeOwner("owner", [], [])
    register_cudagraph_owner(owner, config)
    registry.teardown(post_reset_sync=False)
    constructor = Mock()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", constructor)

    with pytest.raises(RuntimeError, match="after teardown"):
        create_cudagraph(owner, torch.device("cuda:0"), lambda _: None)

    constructor.assert_not_called()


def test_registry_teardown_waits_for_atomic_graph_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    entered = threading.Event()
    release = threading.Event()
    config = fake_config()
    registry = fake_registry(config)
    owner = FakeOwner("owner", events, [])
    register_cudagraph_owner(owner, config)
    graph = FakeGraph("leased", events)

    def graph_factory() -> FakeGraph:
        entered.set()
        assert release.wait(timeout=5)
        return graph

    monkeypatch.setattr(torch.cuda, "CUDAGraph", graph_factory)
    create_errors: list[BaseException] = []
    teardown_stats: list[object] = []

    def create() -> None:
        try:
            create_cudagraph(
                owner,
                torch.device("cuda:0"),
                lambda created: owner.graphs.append(
                    OwnedCUDAGraph(created, torch.device("cuda:0"))
                ),
            )
        except BaseException as exc:
            create_errors.append(exc)

    creator = threading.Thread(target=create)
    teardown = threading.Thread(
        target=lambda: teardown_stats.append(registry.teardown(post_reset_sync=False))
    )
    creator.start()
    assert entered.wait(timeout=5)
    teardown.start()
    time.sleep(0.01)
    assert teardown.is_alive()
    release.set()
    creator.join(timeout=5)
    teardown.join(timeout=5)

    assert not creator.is_alive() and not teardown.is_alive()
    assert not create_errors
    assert graph.reset_count == 1
    assert teardown_stats[0].graph_count == 1


def test_reusable_reset_waits_for_manual_owner_graph_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    registry = fake_registry()
    owner = FakeOwner("manual", [], [])
    owner._cudagraph_manual_owner = True
    graph = FakeGraph("manual", [])

    def graph_factory() -> FakeGraph:
        entered.set()
        assert release.wait(timeout=5)
        return graph

    monkeypatch.setattr(torch.cuda, "CUDAGraph", graph_factory)
    creator = threading.Thread(
        target=lambda: create_cudagraph(
            owner,
            torch.device("cuda:0"),
            lambda created: owner.graphs.append(
                OwnedCUDAGraph(created, torch.device("cuda:0"))
            ),
        )
    )
    results: list[object] = []
    reset = threading.Thread(
        target=lambda: results.append(
            registry.reset_for_reuse([owner], post_reset_sync=False)
        )
    )
    creator.start()
    assert entered.wait(timeout=5)
    reset.start()
    time.sleep(0.01)
    assert reset.is_alive()
    release.set()
    creator.join(timeout=5)
    reset.join(timeout=5)

    assert not creator.is_alive() and not reset.is_alive()
    assert graph.reset_count == 1
    assert results[0].graph_count == 1


def test_wrapper_release_precedes_owner_clear_end_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    graph_refs: list[weakref.ReferenceType[object]] = []
    config = fake_config()
    registry = fake_registry(config)
    owner = FakeOwner("owner", events, [])
    register_cudagraph_owner(owner, config)

    class FinalizerGraph(FakeGraph):
        def __del__(self) -> None:
            events.append("graph-finalized")

    def graph_factory() -> FinalizerGraph:
        graph = FinalizerGraph("owned", events)
        graph_refs.append(weakref.ref(graph))
        return graph

    monkeypatch.setattr(torch.cuda, "CUDAGraph", graph_factory)
    monkeypatch.setattr(
        cuda_graph_module,
        "emit_shutdown_marker",
        lambda event, **_fields: events.append(event),
    )

    create_cudagraph(
        owner,
        torch.device("cuda:0"),
        lambda graph: owner.graphs.append(
            OwnedCUDAGraph(graph, torch.device("cuda:0"))
        ),
    )
    stats = registry.teardown(post_reset_sync=False)

    assert stats.graph_count == 1
    assert graph_refs[0]() is None
    assert events.index("graph-finalized") < events.index("owner_clear_end")


def test_accelerator_device_helpers_preserve_xpu_graph_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class DeviceContext:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_):
            events.append("exit")

    monkeypatch.setattr(
        torch.accelerator,
        "current_accelerator",
        lambda: torch.device("xpu"),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)
    monkeypatch.setattr(
        torch.accelerator,
        "device_index",
        lambda index: events.append(("select", index)) or DeviceContext(),
    )
    monkeypatch.setattr(
        torch.accelerator,
        "synchronize",
        lambda device: events.append(("sync", device)),
    )

    device = cuda_graph_module.current_cudagraph_device()
    assert device == torch.device("xpu:3")
    cuda_graph_module._default_synchronize(device)

    assert events == [
        ("select", 3),
        "enter",
        ("sync", torch.device("xpu:3")),
        "exit",
    ]


def test_ubatch_pre_capture_failure_wakes_and_joins_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.worker import gpu_ubatch_wrapper as ubatch_module
    from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

    events: list[object] = []

    class Event:
        def set(self) -> None:
            events.append("wake")

    class Thread:
        def __init__(self, **kwargs) -> None:
            events.append("thread-created")

        def start(self) -> None:
            events.append("thread-started")

        def join(self, timeout: float) -> None:
            events.append(("thread-joined", timeout))

        def is_alive(self) -> bool:
            return False

    class Barrier:
        def wait(self, timeout: float) -> None:
            events.append(("barrier", timeout))

    wrapper = UBatchWrapper.__new__(UBatchWrapper)
    wrapper.device = torch.device("cuda:0")
    wrapper.ready_barrier = Barrier()
    wrapper.cudagraphs = {}
    wrapper.cudagraph_wrapper = None
    metadata = SimpleNamespace(
        num_tokens=1,
        context=SimpleNamespace(
            compute_stream=object(),
            cpu_wait_event=Event(),
        ),
    )
    monkeypatch.setattr(ubatch_module.threading, "Thread", Thread)
    monkeypatch.setattr(
        ubatch_module, "override_forward_context", lambda _: nullcontext()
    )
    monkeypatch.setattr(
        ubatch_module,
        "create_cudagraph",
        lambda *_: (_ for _ in ()).throw(RuntimeError("creation failed")),
    )

    with pytest.raises(RuntimeError, match="creation failed"):
        wrapper._capture_ubatches([metadata], object())

    assert events[:3] == [
        "thread-created",
        "thread-started",
        ("barrier", wrapper._THREAD_QUIESCE_TIMEOUT_S),
    ]
    assert "wake" in events
    assert any(
        isinstance(event, tuple) and event[0] == "thread-joined" for event in events
    )


def test_ubatch_later_thread_start_failure_aborts_barrier_and_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.worker import gpu_ubatch_wrapper as ubatch_module
    from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

    events: list[object] = []
    threads: list[object] = []

    class Event:
        def set(self) -> None:
            events.append("wake")

    class Thread:
        def __init__(self, **kwargs) -> None:
            self.index = len(threads)
            self.started = False
            self.alive = False
            threads.append(self)

        def start(self) -> None:
            events.append(("start", self.index))
            if self.index == 1:
                raise RuntimeError("start failed")
            self.started = True
            self.alive = True

        def join(self, timeout: float) -> None:
            assert self.started, "an unstarted thread must never be joined"
            events.append(("join", self.index))
            self.alive = False

        def is_alive(self) -> bool:
            return self.alive

    class Barrier:
        broken = False

        def wait(self, timeout: float) -> None:
            raise AssertionError("main thread must not reach the barrier")

        def abort(self) -> None:
            events.append("abort")
            self.broken = True

        def reset(self) -> None:
            events.append("reset")
            self.broken = False

    wrapper = UBatchWrapper.__new__(UBatchWrapper)
    wrapper.device = torch.device("cuda:0")
    wrapper.ready_barrier = Barrier()
    wrapper.cudagraphs = {}
    wrapper.cudagraph_wrapper = None
    metadata = [
        SimpleNamespace(
            num_tokens=1,
            context=SimpleNamespace(
                compute_stream=object(),
                cpu_wait_event=Event(),
            ),
        )
        for _ in range(2)
    ]
    monkeypatch.setattr(ubatch_module.threading, "Thread", Thread)
    monkeypatch.setattr(
        ubatch_module,
        "override_forward_context",
        lambda _: nullcontext(),
    )

    with pytest.raises(RuntimeError, match="start failed"):
        wrapper._capture_ubatches(metadata, object())

    assert events.index("abort") < events.index(("join", 0))
    assert events[-1] == "reset"
    assert ("join", 1) not in events
    assert wrapper._outstanding_ubatch_threads == []


def test_ubatch_later_thread_start_failure_releases_started_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.worker import gpu_ubatch_wrapper as ubatch_module
    from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

    real_thread = threading.Thread
    sibling_waiting = threading.Event()
    sibling_released = threading.Event()
    unstarted_join_called = threading.Event()
    created_threads: list[threading.Thread] = []

    class Context:
        def __init__(self, barrier: threading.Barrier) -> None:
            self.id = len(created_threads)
            self.ready_barrier = barrier
            self.cpu_wait_event = threading.Event()

        def __enter__(self):
            sibling_waiting.set()
            try:
                self.ready_barrier.wait(timeout=1)
            except threading.BrokenBarrierError:
                sibling_released.set()
            return self

        def __exit__(self, *args) -> None:
            return None

    def make_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        if created_threads:
            real_join = thread.join

            def fail_start() -> None:
                assert sibling_waiting.wait(timeout=1)
                raise RuntimeError("later start failed")

            def tracked_real_join(timeout: float) -> None:
                unstarted_join_called.set()
                real_join(timeout=timeout)

            thread.start = fail_start  # type: ignore[method-assign]
            thread.join = tracked_real_join  # type: ignore[method-assign]
        created_threads.append(thread)
        return thread

    barrier = threading.Barrier(3)
    wrapper = UBatchWrapper.__new__(UBatchWrapper)
    wrapper.ready_barrier = barrier
    metadata = [
        SimpleNamespace(
            context=Context(barrier),
            input_ids=None,
            positions=None,
            intermediate_tensors=None,
            inputs_embeds=None,
        )
        for _ in range(2)
    ]
    monkeypatch.setattr(ubatch_module.threading, "Thread", make_thread)
    monkeypatch.setattr(
        ubatch_module,
        "override_forward_context",
        lambda _: nullcontext(),
    )

    with pytest.raises(RuntimeError, match="later start failed"):
        wrapper._run_ubatches(metadata, lambda **_: object())

    assert sibling_released.is_set()
    assert not created_threads[0].is_alive()
    assert not unstarted_join_called.is_set()
    assert not barrier.broken
    assert wrapper._outstanding_ubatch_threads == []


def test_ubatch_context_enter_failure_clears_thread_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.worker import ubatching
    from vllm.v1.worker.ubatching import UBatchContext

    contexts: list[UBatchContext | None] = [None]
    thread_contexts: dict[int, int] = {}
    monkeypatch.setattr(ubatching, "_CURRENT_CONTEXTS", contexts)
    monkeypatch.setattr(ubatching, "_THREAD_ID_TO_CONTEXT", thread_contexts)

    class BrokenBarrier:
        def wait(self) -> None:
            raise threading.BrokenBarrierError

    context = UBatchContext.__new__(UBatchContext)
    context.id = 0
    context.ready_barrier = BrokenBarrier()  # type: ignore[assignment]

    with pytest.raises(threading.BrokenBarrierError):
        context.__enter__()

    assert contexts == [None]
    assert thread_contexts == {}


def test_ubatch_stuck_thread_blocks_graph_reset_and_is_retained() -> None:
    from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

    class StuckThread:
        def __init__(self) -> None:
            self.join_count = 0

        def join(self, timeout: float) -> None:
            self.join_count += 1

        def is_alive(self) -> bool:
            return True

    events: list[str] = []
    graph = FakeGraph("ubatch", events)
    stuck_thread = StuckThread()
    wrapper = UBatchWrapper.__new__(UBatchWrapper)
    wrapper.device = torch.device("cuda:0")
    wrapper.cudagraphs = {1: SimpleNamespace(cudagraph=graph)}
    wrapper.cudagraph_wrapper = None
    wrapper._activity_condition = threading.Condition()
    wrapper._active_calls = 0
    wrapper._outstanding_ubatch_threads = [stuck_thread]
    wrapper._THREAD_QUIESCE_TIMEOUT_S = 0

    stats = teardown_cudagraphs(
        [wrapper],
        post_reset_sync=False,
        synchronize=lambda _: None,
        device_context=lambda _: nullcontext(),
    )

    assert stuck_thread.join_count == 1
    assert wrapper._outstanding_ubatch_threads == [stuck_thread]
    assert graph.reset_count == 0
    assert wrapper.cudagraphs
    assert stats.fallback_required
    assert stats.failures[0].phase == "begin"


def test_ubatch_registers_parent_before_child_and_quiesces_outer_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker import gpu_ubatch_wrapper as ubatch_module
    from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

    events: list[str] = []
    registrations: list[object] = []

    class NestedWrapper:
        def __init__(self, *args, **kwargs) -> None:
            events.append("child-constructed")
            ubatch_module.register_cudagraph_owner(self, config)

        def begin_cudagraph_teardown(self) -> None:
            events.append("child-begin")

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(num_ubatches=2),
    )
    monkeypatch.setattr(ubatch_module.torch.cuda, "Stream", lambda **kwargs: object())
    monkeypatch.setattr(
        UBatchWrapper,
        "_create_sm_control_context",
        staticmethod(lambda _: nullcontext()),
    )
    monkeypatch.setattr(ubatch_module, "CUDAGraphWrapper", NestedWrapper)

    def record_registration(owner, _) -> bool:
        registrations.append(owner)
        if isinstance(owner, NestedWrapper):
            events.append("child-registered")
        else:
            assert owner.cudagraph_wrapper is None
            events.append("parent-registered")
        return True

    monkeypatch.setattr(
        ubatch_module,
        "register_cudagraph_owner",
        record_registration,
    )

    wrapper = UBatchWrapper(
        lambda: None,
        config,  # type: ignore[arg-type]
        CUDAGraphMode.FULL,
        torch.device("cuda:0"),  # type: ignore[arg-type]
    )

    assert events[:3] == [
        "parent-registered",
        "child-constructed",
        "child-registered",
    ]
    assert registrations == [wrapper, wrapper.cudagraph_wrapper]
    wrapper._cudagraph_activity_condition = threading.Condition()
    wrapper._cudagraph_activity_local = threading.local()
    with wrapper._cudagraph_activity_condition:
        wrapper._cudagraph_active_calls = 1

    def begin_registered_owners() -> None:
        for owner in registrations:
            owner.begin_cudagraph_teardown()  # type: ignore[attr-defined]

    teardown_thread = threading.Thread(target=begin_registered_owners)
    teardown_thread.start()
    deadline = time.monotonic() + 1
    while not wrapper._cudagraph_teardown_started and time.monotonic() < deadline:
        time.sleep(0.001)

    assert wrapper._cudagraph_teardown_started
    assert "child-begin" not in events

    with wrapper._cudagraph_activity_condition:
        events.append("outer-released")
        wrapper._cudagraph_active_calls = 0
        wrapper._cudagraph_activity_condition.notify_all()
    teardown_thread.join(timeout=1)

    assert not teardown_thread.is_alive()
    assert events.index("outer-released") < events.index("child-begin")


def test_breakable_active_capture_is_reported_but_remains_enumerable() -> None:
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    events: list[str] = []
    graph = FakeGraph("partial", events)
    capture = BreakableCUDAGraphCapture()
    capture._current_graph = graph
    capture._current_device = torch.device("cuda:0")
    capture._capturing = True

    with pytest.raises(RuntimeError, match="active CUDA graph capture"):
        capture.begin_cudagraph_teardown()

    assert [owned.graph for owned in capture.iter_cudagraphs()] == [graph]
    capture.clear_cudagraph_state()
    assert not list(capture.iter_cudagraphs())


def test_capture_failure_marks_owner_unsafe_for_teardown() -> None:
    class Owner:
        def begin_cudagraph_teardown(self) -> None:
            begin_cudagraph_owner_teardown(self)

        def iter_cudagraphs(self):
            return ()

        def clear_cudagraph_state(self) -> None:
            pass

    owner = Owner()
    with (
        pytest.raises(RuntimeError, match="capture failed"),
        cudagraph_capture_attempt(owner),
    ):
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="incomplete CUDA graph capture"):
        owner.begin_cudagraph_teardown()


def test_breakable_capture_begin_failure_remains_unclean_and_enumerable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.compilation import breakable_cudagraph as breakable_module

    class FailedGraph(FakeGraph):
        def capture_begin(self, **_kwargs) -> None:
            raise RuntimeError("capture begin failed")

    graph = FailedGraph("partial", [])
    monkeypatch.setattr(breakable_module.torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(
        breakable_module,
        "current_cudagraph_device",
        lambda: torch.device("cuda:0"),
    )
    capture = breakable_module.BreakableCUDAGraphCapture()

    with pytest.raises(RuntimeError, match="capture begin failed"):
        capture._begin_segment()
    with pytest.raises(RuntimeError, match="active CUDA graph capture"):
        capture.begin_cudagraph_teardown()

    assert capture._capturing
    assert [owned.graph for owned in capture.iter_cudagraphs()] == [graph]


def test_production_cudagraph_construction_uses_the_common_factory() -> None:
    repo_root = Path(__file__).parents[3]
    direct_calls: list[tuple[Path, str]] = []

    for path in (repo_root / "vllm").rglob("*.py"):
        tree = ast.parse(path.read_text())
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "CUDAGraph"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "cuda"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "torch"
            ):
                continue
            parent = parents.get(node)
            while parent is not None and not isinstance(
                parent, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                parent = parents.get(parent)
            direct_calls.append(
                (path.relative_to(repo_root), getattr(parent, "name", "<module>"))
            )

    assert direct_calls == [
        (Path("vllm/compilation/cuda_graph.py"), "create_cudagraph")
    ]
