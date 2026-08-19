# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import vllm.v1.executor.multiproc_executor as multiproc_executor


def test_worker_proc_shutdown_orders_clean_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class MQ:
        def __init__(self, name: str) -> None:
            self.name = name

        def shutdown(self) -> None:
            events.append(self.name)

    class OutputQueue:
        def put_nowait(self, value: object) -> None:
            events.append(value)

    class OutputThread:
        def join(self, timeout: float) -> None:
            events.append(("join", timeout))

        def is_alive(self) -> bool:
            return False

    actual_worker = SimpleNamespace(_cudagraph_teardown_incomplete=False)

    class WorkerWrapper:
        worker = actual_worker

        def shutdown(self) -> None:
            events.append("worker")

    @contextmanager
    def terminal_teardown():
        events.append("terminal_enter")
        yield
        events.append("terminal_exit")

    proc = multiproc_executor.WorkerProc.__new__(multiproc_executor.WorkerProc)
    proc.use_async_scheduling = True
    proc.async_output_queue = OutputQueue()
    proc.async_output_copy_thread = OutputThread()
    proc.rpc_broadcast_mq = MQ("broadcast")
    proc.worker_response_mq = MQ("response")
    proc.worker = WorkerWrapper()

    monkeypatch.setattr(
        multiproc_executor, "terminal_cudagraph_teardown", terminal_teardown
    )
    monkeypatch.setattr(
        multiproc_executor, "destroy_model_parallel", lambda: events.append("model")
    )
    monkeypatch.setattr(
        multiproc_executor,
        "destroy_distributed_environment",
        lambda: events.append("distributed"),
    )
    monkeypatch.setattr(
        multiproc_executor,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: True),
    )
    monkeypatch.setattr(multiproc_executor.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(
        multiproc_executor.gc, "freeze", lambda: events.append("freeze")
    )
    proc.shutdown()

    assert proc._terminal_fallback_cycle is proc
    assert proc.rpc_broadcast_mq is None
    assert proc.worker_response_mq is None
    assert events == [
        multiproc_executor.WorkerProc.ASYNC_OUTPUT_SHUTDOWN,
        ("join", 5.0),
        "terminal_enter",
        "worker",
        "terminal_exit",
        "broadcast",
        "response",
        "model",
        "distributed",
        "gc",
        "freeze",
    ]


def test_worker_proc_shutdown_freezes_incomplete_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class MQ:
        def __init__(self, name: str) -> None:
            self.name = name

        def shutdown(self) -> None:
            events.append(self.name)

    actual_worker = SimpleNamespace(_cudagraph_teardown_incomplete=False)

    class WorkerWrapper:
        worker = actual_worker

        def shutdown(self) -> None:
            events.append("worker")
            actual_worker._cudagraph_teardown_incomplete = True
            raise RuntimeError("worker failed")

    proc = multiproc_executor.WorkerProc.__new__(multiproc_executor.WorkerProc)
    proc.use_async_scheduling = False
    proc.rpc_broadcast_mq = MQ("broadcast")
    proc.worker_response_mq = MQ("response")
    proc.worker = WorkerWrapper()

    monkeypatch.setattr(
        multiproc_executor, "destroy_model_parallel", lambda: events.append("model")
    )
    monkeypatch.setattr(
        multiproc_executor,
        "destroy_distributed_environment",
        lambda: events.append("distributed"),
    )
    monkeypatch.setattr(
        multiproc_executor,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: True),
    )
    monkeypatch.setattr(multiproc_executor.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(
        multiproc_executor.gc, "freeze", lambda: events.append("freeze")
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        proc.shutdown()

    assert proc._terminal_fallback_cycle is proc
    assert events == [
        "worker",
        "broadcast",
        "response",
        "gc",
        "freeze",
    ]


def test_worker_proc_queue_failure_does_not_skip_graph_or_distributed_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class MQ:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def shutdown(self) -> None:
            events.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} failed")

    worker = SimpleNamespace(_cudagraph_teardown_incomplete=False)
    proc = multiproc_executor.WorkerProc.__new__(multiproc_executor.WorkerProc)
    proc.use_async_scheduling = False
    proc.rpc_broadcast_mq = MQ("broadcast", fail=True)
    proc.worker_response_mq = MQ("response")
    proc.worker = SimpleNamespace(
        worker=worker,
        shutdown=lambda: events.append("worker"),
    )
    monkeypatch.setattr(
        multiproc_executor, "destroy_model_parallel", lambda: events.append("model")
    )
    monkeypatch.setattr(
        multiproc_executor,
        "destroy_distributed_environment",
        lambda: events.append("distributed"),
    )
    monkeypatch.setattr(
        multiproc_executor,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: False),
    )

    with pytest.raises(RuntimeError, match="broadcast failed"):
        proc.shutdown()

    assert events == ["worker", "broadcast", "response", "model", "distributed"]


def test_worker_proc_does_not_reset_graphs_before_async_output_quiesces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class OutputQueue:
        def put_nowait(self, value: object) -> None:
            events.append(value)

    class StuckThread:
        def join(self, timeout: float) -> None:
            events.append(("join", timeout))

        def is_alive(self) -> bool:
            return True

    class WorkerWrapper:
        worker = SimpleNamespace(_cudagraph_teardown_incomplete=False)

        def shutdown(self) -> None:
            events.append("worker")

    proc = multiproc_executor.WorkerProc.__new__(multiproc_executor.WorkerProc)
    proc.use_async_scheduling = True
    proc.async_output_queue = OutputQueue()
    proc.async_output_copy_thread = StuckThread()
    proc.rpc_broadcast_mq = None
    proc.worker_response_mq = None
    proc.worker = WorkerWrapper()

    monkeypatch.setattr(
        multiproc_executor, "destroy_model_parallel", lambda: events.append("model")
    )
    monkeypatch.setattr(
        multiproc_executor,
        "destroy_distributed_environment",
        lambda: events.append("distributed"),
    )
    monkeypatch.setattr(
        multiproc_executor,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: True),
    )
    monkeypatch.setattr(multiproc_executor.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(
        multiproc_executor.gc, "freeze", lambda: events.append("freeze")
    )

    with pytest.raises(RuntimeError, match="async output thread did not stop"):
        proc.shutdown()

    assert events == [
        multiproc_executor.WorkerProc.ASYNC_OUTPUT_SHUTDOWN,
        ("join", 5.0),
        "freeze",
    ]
    assert proc._terminal_fallback_cycle is proc
    assert proc.worker.worker._cudagraph_teardown_incomplete


def test_executor_reports_nonzero_worker_after_closing_queues() -> None:
    events: list[str] = []

    class Queue:
        def __init__(self, name: str):
            self.name = name

        def shutdown(self) -> None:
            events.append(self.name)

    process = SimpleNamespace(
        name="Worker_0",
        exitcode=1,
        is_alive=lambda: False,
    )
    executor = object.__new__(multiproc_executor.MultiprocExecutor)
    executor.shutting_down = False
    executor.workers = [
        SimpleNamespace(
            death_writer=None,
            proc=process,
            worker_response_mq=Queue("worker_response"),
        )
    ]
    executor.rpc_broadcast_mq = Queue("broadcast")
    executor.response_mqs = [Queue("response")]

    with pytest.raises(RuntimeError, match="Worker_0=1"):
        executor.shutdown()

    assert events == ["worker_response", "broadcast", "response"]
