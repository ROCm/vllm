# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import queue
import signal
import threading
import weakref
from collections.abc import Callable
from contextlib import nullcontext
from multiprocessing import connection
from types import SimpleNamespace

import pytest
import zmq

import vllm.platforms as platforms
import vllm.v1.utils as v1_utils_module
from vllm.v1.engine import core as core_module
from vllm.v1.engine import utils as engine_utils_module
from vllm.v1.engine.core import EngineCore, EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import (
    CoreEngine,
    CoreEngineLaunch,
    CoreEngineProcManager,
    EngineZmqAddresses,
    SignalCallback,
    wait_for_engine_startup,
)

pytestmark = pytest.mark.skip_global_cleanup


def _init_core_shutdown_state(engine_core: EngineCore) -> None:
    engine_core._shutdown_lock = threading.RLock()
    engine_core._shutdown_started = False
    engine_core._cudagraph_teardown_incomplete = False


def _init_proc_shutdown_state(engine_core: EngineCoreProc) -> None:
    _init_core_shutdown_state(engine_core)
    engine_core._io_shutdown_event = threading.Event()
    engine_core._io_shutdown_lock = threading.Lock()
    engine_core._io_shutdown_complete = False
    engine_core._frontend_teardown_incomplete = False
    engine_core.input_queue = queue.Queue()
    engine_core.output_queue = queue.Queue()
    engine_core.input_thread = None
    engine_core.output_thread = None


def test_signal_callback_stop_joins_and_releases_callback():
    callback_finished = threading.Event()

    class Callback:
        def __call__(self):
            callback_finished.set()

    callback = Callback()
    callback_ref = weakref.ref(callback)
    signal_callback = SignalCallback(callback)

    signal_callback.stop()
    signal_callback.stop()

    assert not signal_callback._thread.is_alive()
    assert signal_callback._callback is None
    assert not callback_finished.is_set()
    del callback
    gc.collect()
    assert callback_ref() is None


def test_signal_callback_stop_uses_bounded_join_and_clears_callback():
    join_timeouts: list[float | None] = []

    class StuckThread:
        def join(self, timeout: float | None = None):
            join_timeouts.append(timeout)

        def is_alive(self):
            return True

    signal_callback = object.__new__(SignalCallback)
    signal_callback._callback = lambda: None
    signal_callback._event = SimpleNamespace(set=lambda: None)
    signal_callback._stopped = False
    signal_callback._stop_lock = threading.Lock()
    signal_callback._thread = StuckThread()

    with pytest.raises(RuntimeError, match="failed to stop"):
        signal_callback.stop()

    assert join_timeouts == [SignalCallback._JOIN_TIMEOUT_S]
    assert signal_callback._callback is None


def test_engine_core_shutdown_attempts_all_cleanup_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    class Component:
        def __init__(self, name: str, fail: bool = False):
            self.name = name
            self.fail = fail

        def _cleanup(self):
            calls.append(self.name)
            if self.name == "executor":
                engine_core.shutdown()
            if self.fail:
                raise RuntimeError(f"{self.name} failed")

        clear_backend = _cleanup
        shutdown = _cleanup

    engine_core = object.__new__(EngineCore)
    _init_core_shutdown_state(engine_core)
    engine_core.structured_output_manager = Component("structured", fail=True)
    engine_core.model_executor = Component("executor")  # type: ignore[assignment]
    engine_core.scheduler = Component("scheduler")  # type: ignore[assignment]

    monkeypatch.setattr(core_module.gc, "unfreeze", lambda: calls.append("unfreeze"))
    monkeypatch.setattr(
        core_module,
        "cleanup_dist_env_and_memory",
        lambda: calls.append("distributed"),
    )

    with pytest.raises(RuntimeError, match="structured failed"):
        engine_core.shutdown()

    assert calls == [
        "structured",
        "executor",
        "scheduler",
        "unfreeze",
        "distributed",
    ]
    assert engine_core.structured_output_manager is None
    assert engine_core.model_executor is None
    assert engine_core.scheduler is None
    engine_core.shutdown()
    assert len(calls) == 5


def test_engine_core_withholds_distributed_cleanup_for_incomplete_graphs(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    class Executor:
        _cudagraph_teardown_incomplete = True

        def shutdown(self):
            calls.append("executor")

    engine_core = object.__new__(EngineCore)
    _init_core_shutdown_state(engine_core)
    executor = Executor()
    engine_core.model_executor = executor  # type: ignore[assignment]
    engine_core.structured_output_manager = None  # type: ignore[assignment]
    engine_core.scheduler = None  # type: ignore[assignment]
    monkeypatch.setattr(core_module.gc, "unfreeze", lambda: calls.append("unfreeze"))
    monkeypatch.setattr(
        core_module,
        "cleanup_dist_env_and_memory",
        lambda: calls.append("distributed"),
    )

    with pytest.raises(RuntimeError, match="distributed cleanup was withheld"):
        engine_core.shutdown()

    assert calls == ["executor", "unfreeze"]
    assert engine_core._cudagraph_teardown_incomplete
    assert engine_core.model_executor is executor


def test_engine_core_proc_stops_io_threads_once():
    engine_core = object.__new__(EngineCoreProc)
    engine_core._io_shutdown_event = threading.Event()
    engine_core._io_shutdown_lock = threading.Lock()
    engine_core._io_shutdown_complete = False
    engine_core.output_queue = queue.Queue()
    output_items: list[bytes] = []

    def input_loop():
        engine_core._io_shutdown_event.wait()

    def output_loop():
        while True:
            item = engine_core.output_queue.get()
            output_items.append(item)
            if item == EngineCoreProc._ENGINE_CORE_STOP:
                return

    input_thread = threading.Thread(target=input_loop, name="test-input")
    output_thread = threading.Thread(target=output_loop, name="test-output")
    engine_core.input_thread = input_thread
    engine_core.output_thread = output_thread
    input_thread.start()
    output_thread.start()

    engine_core._shutdown_io_threads()
    engine_core._shutdown_io_threads()

    assert not input_thread.is_alive()
    assert not output_thread.is_alive()
    assert output_items == [EngineCoreProc._ENGINE_CORE_STOP]
    assert EngineCoreProc.ENGINE_CORE_DEAD not in output_items
    assert engine_core.input_thread is None
    assert engine_core.output_thread is None


def test_engine_core_proc_wakes_output_before_join_and_allows_retry():
    events: list[object] = []

    class OutputQueue:
        def put_nowait(self, item):
            events.append(item)

    class Thread:
        def __init__(self, name: str):
            self.name = name
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            events.append((self.name, "join"))

    engine_core = object.__new__(EngineCoreProc)
    _init_proc_shutdown_state(engine_core)
    engine_core.output_queue = OutputQueue()  # type: ignore[assignment]
    input_thread = Thread("input")
    output_thread = Thread("output")
    engine_core.input_thread = input_thread  # type: ignore[assignment]
    engine_core.output_thread = output_thread  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="failed to stop"):
        engine_core._shutdown_io_threads()

    assert events[0] == EngineCoreProc._ENGINE_CORE_STOP
    assert not engine_core._io_shutdown_complete

    input_thread.alive = False
    output_thread.alive = False
    engine_core._shutdown_io_threads()

    assert engine_core._io_shutdown_complete
    assert engine_core.input_thread is None
    assert engine_core.output_thread is None


def test_engine_core_proc_preserves_frontend_state_while_input_thread_is_stuck(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    class StructuredOutputManager:
        def grammar_init(self):
            entered.set()
            release.wait()

        def clear_backend(self):
            calls.append("structured")

    class Scheduler:
        def shutdown(self):
            calls.append("scheduler")

    class Executor:
        _cudagraph_teardown_incomplete = False

        def shutdown(self):
            calls.append("executor")

    engine_core = object.__new__(EngineCoreProc)
    _init_proc_shutdown_state(engine_core)
    engine_core._IO_THREAD_JOIN_TIMEOUT_S = 0.01
    manager = StructuredOutputManager()
    scheduler = Scheduler()
    mm_cache = object()
    engine_core.structured_output_manager = manager  # type: ignore[assignment]
    engine_core.scheduler = scheduler  # type: ignore[assignment]
    engine_core.model_executor = Executor()  # type: ignore[assignment]
    engine_core.mm_receiver_cache = mm_cache  # type: ignore[assignment]
    input_thread = threading.Thread(target=manager.grammar_init)
    engine_core.input_thread = input_thread
    input_thread.start()
    assert entered.wait(timeout=1)
    monkeypatch.setattr(core_module.gc, "unfreeze", lambda: calls.append("unfreeze"))
    monkeypatch.setattr(
        core_module,
        "cleanup_dist_env_and_memory",
        lambda: calls.append("distributed"),
    )

    with pytest.raises(RuntimeError, match="failed to stop"):
        engine_core.shutdown()

    assert calls == ["executor"]
    assert engine_core.structured_output_manager is manager
    assert engine_core.scheduler is scheduler
    assert engine_core.mm_receiver_cache is mm_cache
    assert engine_core._frontend_teardown_incomplete
    assert not engine_core._shutdown_started

    release.set()
    input_thread.join(timeout=1)
    engine_core.shutdown()

    assert calls == [
        "executor",
        "structured",
        "scheduler",
        "unfreeze",
        "distributed",
    ]
    assert not engine_core._frontend_teardown_incomplete


@pytest.mark.parametrize(
    ("sentinel", "expected_messages"),
    [
        (EngineCoreProc._ENGINE_CORE_STOP, []),
        (EngineCoreProc.ENGINE_CORE_DEAD, [EngineCoreProc.ENGINE_CORE_DEAD]),
    ],
)
def test_output_thread_control_sentinel_wire_behavior(
    monkeypatch: pytest.MonkeyPatch,
    sentinel: bytes,
    expected_messages: list[bytes],
):
    sent_messages: list[bytes] = []
    socket = SimpleNamespace(send=sent_messages.append)
    engine_core = object.__new__(EngineCoreProc)
    engine_core.output_queue = queue.Queue()
    engine_core.output_queue.put_nowait(sentinel)

    monkeypatch.setattr(core_module.zmq, "Context", lambda: nullcontext())
    monkeypatch.setattr(
        core_module,
        "make_zmq_socket",
        lambda *args, **kwargs: nullcontext(socket),
    )

    engine_core.process_output_sockets(["inproc://output"], None, 0)

    assert sent_messages == expected_messages


def test_input_thread_uses_bounded_poll(monkeypatch: pytest.MonkeyPatch):
    poll_timeouts: list[int] = []
    socket = SimpleNamespace(send=lambda *_: None)
    engine_core = object.__new__(EngineCoreProc)
    engine_core.tensor_ipc_receiver = None
    engine_core._io_shutdown_event = threading.Event()
    engine_core._make_ready_response = lambda: {}

    class Poller:
        def register(self, *args):
            pass

        def poll(self, timeout: int):
            poll_timeouts.append(timeout)
            engine_core._io_shutdown_event.set()
            return []

    monkeypatch.setattr(core_module, "MsgpackDecoder", lambda *args, **kwargs: None)
    monkeypatch.setattr(core_module.zmq, "Context", lambda: nullcontext())
    monkeypatch.setattr(core_module.zmq, "Poller", Poller)
    monkeypatch.setattr(
        core_module,
        "make_zmq_socket",
        lambda *args, **kwargs: nullcontext(socket),
    )

    ready_event = threading.Event()
    engine_core.process_input_sockets(["inproc://input"], None, b"id", ready_event)

    assert ready_event.is_set()
    assert poll_timeouts == [EngineCoreProc._IO_POLL_TIMEOUT_MS]


def test_input_thread_coordinator_ready_wait_honors_shutdown(
    monkeypatch: pytest.MonkeyPatch,
):
    coord_poll_timeouts: list[int] = []
    input_socket = SimpleNamespace(send=lambda *_: None)
    engine_core = object.__new__(EngineCoreProc)
    _init_proc_shutdown_state(engine_core)
    engine_core.tensor_ipc_receiver = None
    engine_core._make_ready_response = lambda: {}

    class CoordinatorSocket:
        def send(self, *_):
            pass

        def poll(self, timeout: int):
            coord_poll_timeouts.append(timeout)
            engine_core._io_shutdown_event.set()
            return 0

    sockets = iter((input_socket, CoordinatorSocket()))

    class Poller:
        def register(self, *args):
            pass

    monkeypatch.setattr(core_module, "MsgpackDecoder", lambda *args, **kwargs: None)
    monkeypatch.setattr(core_module.zmq, "Context", lambda: nullcontext())
    monkeypatch.setattr(core_module.zmq, "Poller", Poller)
    monkeypatch.setattr(
        core_module,
        "make_zmq_socket",
        lambda *args, **kwargs: nullcontext(next(sockets)),
    )

    ready_event = threading.Event()
    engine_core.process_input_sockets(
        ["inproc://input"],
        "inproc://coordinator",
        b"id",
        ready_event,
    )

    assert not ready_event.is_set()
    assert coord_poll_timeouts == [EngineCoreProc._IO_POLL_TIMEOUT_MS]


def _run_mock_engine_core(
    monkeypatch: pytest.MonkeyPatch,
    *,
    is_rocm: bool = True,
    shutdown_state: EngineShutdownState = EngineShutdownState.SHUTTING_DOWN,
    has_work: bool = False,
    shutdown_timeout: int = 0,
    exit_code: int | None = None,
    fatal: bool = False,
    shutdown_error: bool = False,
    cudagraph_incomplete: bool = False,
):
    calls: list[str | tuple[str, bool]] = []
    engine_ref: list[weakref.ReferenceType | None] = [None]
    installed_handlers: dict[int, Callable] = {}

    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        numa_bind=False,
        reconfigure_for_independent_dp_rank=lambda: None,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        kv_transfer_config=None,
        shutdown_timeout=shutdown_timeout,
    )

    class FakeEngineCore:
        def __init__(self, *args, **kwargs):
            self.shutdown_state = EngineShutdownState.RUNNING
            self.vllm_config = vllm_config
            self.input_queue = queue.Queue()
            self._io_shutdown_event = threading.Event()
            engine_ref[0] = weakref.ref(self)

        def run_busy_loop(self):
            if fatal:
                raise RuntimeError("fatal engine error")
            self.shutdown_state = shutdown_state
            raise SystemExit(exit_code)

        def has_work(self):
            return has_work

        def _send_engine_dead(self):
            calls.append("engine-dead")

        def shutdown(self):
            calls.append("shutdown")
            self._cudagraph_teardown_incomplete = cudagraph_incomplete
            if shutdown_error:
                raise RuntimeError("shutdown failed")

    class FakeSignalCallback:
        def __init__(self, callback):
            self.callback = callback

        def trigger(self):
            self.callback()

        def stop(self):
            calls.append("callback-stop")
            self.callback = None

    def set_signal(signum, handler):
        if handler is signal.SIG_IGN:
            calls.append(f"ignore-{signal.Signals(signum).name}")
        else:
            installed_handlers[signum] = handler

    for name in (
        "maybe_register_config_serialize_by_value",
        "set_process_title",
        "maybe_init_worker_tracer",
        "decorate_logs",
    ):
        monkeypatch.setattr(core_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(core_module, "EngineCoreProc", FakeEngineCore)
    monkeypatch.setattr(core_module, "SignalCallback", FakeSignalCallback)
    monkeypatch.setattr(core_module.signal, "signal", set_signal)
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_rocm=lambda: is_rocm)
    )

    real_gc_collect = gc.collect

    def collect():
        result = real_gc_collect()
        assert engine_ref[0] is not None
        calls.append(("collect-root-cleared", engine_ref[0]() is None))
        return result

    monkeypatch.setattr(core_module.gc, "collect", collect)
    monkeypatch.setattr(core_module.gc, "freeze", lambda: calls.append("freeze"))

    expected_exception = RuntimeError if fatal or shutdown_error else SystemExit
    with pytest.raises(expected_exception) as exc_info:
        EngineCoreProc.run_engine_core(vllm_config=vllm_config)

    if fatal:
        assert str(exc_info.value) == "fatal engine error"
    elif shutdown_error:
        assert str(exc_info.value) == "shutdown failed"

    assert installed_handlers[signal.SIGTERM] is signal.SIG_DFL
    assert installed_handlers[signal.SIGINT] is signal.SIG_DFL

    return calls, engine_ref[0]


def test_run_engine_core_detaches_root_before_final_gc_and_freeze(
    monkeypatch: pytest.MonkeyPatch,
):
    calls, engine_ref = _run_mock_engine_core(monkeypatch)

    assert engine_ref is not None and engine_ref() is None
    assert calls == [
        "ignore-SIGTERM",
        "ignore-SIGINT",
        "callback-stop",
        "shutdown",
        ("collect-root-cleared", True),
        "freeze",
    ]


@pytest.mark.parametrize(
    ("kwargs", "expected_prefix", "expected_freeze"),
    [
        ({"is_rocm": False}, [], False),
        ({"shutdown_state": EngineShutdownState.RUNNING}, [], False),
        ({"has_work": True}, [], False),
        ({"shutdown_timeout": 1}, [], True),
        ({"exit_code": 1}, [], False),
        ({"fatal": True}, ["engine-dead"], False),
        ({"fatal": True, "shutdown_error": True}, ["engine-dead"], False),
        ({"shutdown_error": True}, [], True),
        (
            {"fatal": True, "cudagraph_incomplete": True},
            ["engine-dead"],
            True,
        ),
    ],
)
def test_run_engine_core_freeze_requires_clean_rocm_exit(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict,
    expected_prefix: list[str],
    expected_freeze: bool,
):
    calls, _ = _run_mock_engine_core(monkeypatch, **kwargs)

    assert ("freeze" in calls) is expected_freeze
    for expected in expected_prefix:
        assert expected in calls
    assert ("collect-root-cleared", True) in calls


@pytest.mark.parametrize(
    ("drain_timeout", "expected_wait"),
    [(None, 15.0), (0, 15.0), (10, 25.0)],
)
def test_engine_core_process_shutdown_keeps_resource_cleanup_grace(
    monkeypatch: pytest.MonkeyPatch,
    drain_timeout: float | None,
    expected_wait: float,
) -> None:
    calls: list[tuple[list[object], float | None, bool]] = []
    processes = [object()]

    class Finalizer:
        alive = True

        def detach(self):
            self.alive = False
            return object()

    manager = object.__new__(CoreEngineProcManager)
    manager.processes = processes  # type: ignore[assignment]
    manager.manager_stopped = threading.Event()
    manager._finalizer = Finalizer()  # type: ignore[assignment]
    monkeypatch.setattr(
        engine_utils_module,
        "shutdown",
        lambda procs, timeout=None, raise_on_failure=False: calls.append(
            (procs, timeout, raise_on_failure)
        ),
    )

    manager.shutdown(timeout=drain_timeout, raise_on_failure=True)

    assert manager.manager_stopped.is_set()
    assert calls == [(processes, expected_wait, True)]


def test_forced_engine_process_shutdown_is_reported_nonclean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StuckProcess:
        name = "EngineCore"
        pid = 123
        exitcode: int | None = None
        alive = True

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            pass

        def join(self, _timeout: float) -> None:
            pass

    process = StuckProcess()
    killed: list[int] = []

    def kill(pid: int) -> None:
        killed.append(pid)
        process.alive = False
        process.exitcode = -9

    monkeypatch.setattr(v1_utils_module, "kill_process_tree", kill)

    with pytest.raises(RuntimeError, match="required SIGKILL"):
        v1_utils_module.shutdown(
            [process],  # type: ignore[list-item]
            timeout=0,
            raise_on_failure=True,
        )

    assert killed == [123]


class _FinishedProcess:
    name = "RustFrontend"

    def __init__(self, sentinel):
        self.sentinel = sentinel

    @property
    def exitcode(self):
        return 1


def test_wait_for_engine_startup_reports_watched_process_exit():
    ctx = zmq.Context()
    handshake_socket = ctx.socket(zmq.ROUTER)
    recv, send = connection.Pipe(duplex=False)
    send.close()

    parallel_config = SimpleNamespace(
        data_parallel_size_local=1,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=False,
    )

    try:
        launch = CoreEngineLaunch(
            engine_manager=None,
            coordinator=None,
            addresses=EngineZmqAddresses(inputs=[], outputs=[]),
            tensor_queue=None,
        )
        launch.watched_frontend_processes = [_FinishedProcess(recv)]
        with pytest.raises(RuntimeError) as exc_info:
            wait_for_engine_startup(
                handshake_socket,
                [CoreEngine()],
                parallel_config,  # type: ignore[arg-type]
                coordinated_dp=False,
                cache_config=None,  # type: ignore[arg-type]
                launch=launch,
            )
    finally:
        recv.close()
        handshake_socket.close(linger=0)
        ctx.term()

    assert "Frontend process failed during engine core initialization" in str(
        exc_info.value
    )
    assert "Failed frontend proc(s): {'RustFrontend': 1}" in str(exc_info.value)
