# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import functools
import threading
import time
import traceback
import uuid
import weakref
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from contextlib import (
    AbstractContextManager,
    ExitStack,
    contextmanager,
)
from contextvars import ContextVar
from enum import Enum, auto
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar, cast
from unittest.mock import patch

import torch

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.utils.shutdown_markers import emit_shutdown_marker
from vllm.utils.torch_utils import current_stream, weak_ref_tensors

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclasses.dataclass(frozen=True)
class OwnedCUDAGraph:
    """A native CUDA graph and the device on which it was captured."""

    graph: Any
    device: torch.device


class CUDAGraphOwner(Protocol):
    """Two-phase ownership contract for deterministic CUDA graph teardown."""

    def begin_cudagraph_teardown(self) -> None: ...

    def iter_cudagraphs(self) -> Iterable[OwnedCUDAGraph]: ...

    def clear_cudagraph_state(self) -> None: ...


class _PendingCUDAGraphState:
    """Graphs constructed but not yet durably installed in owner metadata."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.graphs: dict[int, OwnedCUDAGraph] = {}
        self.active_creations = 0
        self.creation_threads: Counter[int] = Counter()
        self.creation_closed = False


_PENDING_STATE_ATTR = "_cudagraph_pending_state"
_CAPTURE_ATTEMPTS_ATTR = "_cudagraph_capture_attempts"
_PENDING_STATE_INIT_LOCK = threading.Lock()
_OWNER_ACTIVITY_INIT_LOCK = threading.Lock()


def _pending_cudagraph_state(owner: CUDAGraphOwner) -> _PendingCUDAGraphState:
    owner_state = cast(Any, owner)
    state = vars(owner_state).get(_PENDING_STATE_ATTR)
    if state is not None:
        return state
    # Owner construction normally initializes this before capture. Keep the
    # lazy path race-free for manual/test owners and partially initialized
    # owners restored from caches.
    with _PENDING_STATE_INIT_LOCK:
        state = vars(owner_state).get(_PENDING_STATE_ATTR)
        if state is None:
            state = _PendingCUDAGraphState()
            setattr(owner_state, _PENDING_STATE_ATTR, state)
    return state


def _iter_pending_cudagraphs(owner: CUDAGraphOwner) -> tuple[OwnedCUDAGraph, ...]:
    state = _pending_cudagraph_state(owner)
    with state.lock:
        return tuple(state.graphs.values())


def _clear_pending_cudagraphs(owner: CUDAGraphOwner) -> None:
    state = _pending_cudagraph_state(owner)
    with state.lock:
        state.graphs.clear()


@contextmanager
def _owner_graph_creation(owner: CUDAGraphOwner) -> Iterator[None]:
    state = _pending_cudagraph_state(owner)
    thread_id = threading.get_ident()
    with state.condition:
        if state.creation_closed:
            raise RuntimeError("cannot create a CUDA graph after teardown")
        state.active_creations += 1
        state.creation_threads[thread_id] += 1
    try:
        yield
    finally:
        with state.condition:
            state.active_creations -= 1
            state.creation_threads[thread_id] -= 1
            if not state.creation_threads[thread_id]:
                del state.creation_threads[thread_id]
            state.condition.notify_all()


def _begin_owner_graph_creation_teardown(owner: CUDAGraphOwner) -> None:
    state = _pending_cudagraph_state(owner)
    thread_id = threading.get_ident()
    with state.condition:
        state.creation_closed = True
        if state.creation_threads.get(thread_id, 0):
            raise RuntimeError(
                "cannot tear down a CUDA graph owner from graph construction"
            )
        while state.active_creations:
            state.condition.wait()


def _resume_owner_graph_creation(owner: CUDAGraphOwner) -> None:
    state = _pending_cudagraph_state(owner)
    with state.condition:
        state.creation_closed = False
        state.condition.notify_all()


@contextmanager
def cudagraph_capture_attempt(owner: CUDAGraphOwner) -> Iterator[None]:
    """Mark a native capture attempt unsafe until its context exits cleanly."""
    state = _pending_cudagraph_state(owner)
    owner_state = cast(Any, owner)
    with state.lock:
        attempts = vars(owner_state).get(_CAPTURE_ATTEMPTS_ATTR, 0)
        setattr(owner_state, _CAPTURE_ATTEMPTS_ATTR, attempts + 1)
    # If the body raises, contextlib throws the exception at ``yield`` and the
    # decrement is intentionally skipped. capture_begin may already have made
    # native capture active, so teardown must retain the graph and its buffers.
    yield
    with state.lock:
        attempts = vars(owner_state).get(_CAPTURE_ATTEMPTS_ATTR, 0)
        setattr(owner_state, _CAPTURE_ATTEMPTS_ATTR, max(0, attempts - 1))


def ensure_cudagraph_capture_complete(owner: CUDAGraphOwner) -> None:
    state = _pending_cudagraph_state(owner)
    with state.lock:
        attempts = vars(owner).get(_CAPTURE_ATTEMPTS_ATTR, 0)
    if attempts:
        raise RuntimeError(
            f"cannot cleanly tear down {attempts} incomplete CUDA graph capture(s)"
        )


def _owner_activity_condition(owner: Any) -> threading.Condition:
    with _OWNER_ACTIVITY_INIT_LOCK:
        owner_attrs = vars(owner)
        condition = owner_attrs.get("_cudagraph_activity_condition")
        if condition is None:
            condition = threading.Condition()
            owner._cudagraph_activity_condition = condition
        if "_cudagraph_active_calls" not in owner_attrs:
            owner._cudagraph_active_calls = 0
        if "_cudagraph_activity_local" not in owner_attrs:
            owner._cudagraph_activity_local = threading.local()
    return condition


def begin_cudagraph_owner_teardown(owner: Any) -> None:
    """Close an owner and wait for its in-flight capture/replay calls."""
    condition = _owner_activity_condition(owner)
    with condition:
        owner._cudagraph_teardown_started = True
        activity_local = owner._cudagraph_activity_local
        if vars(activity_local).get("depth", 0):
            # Waiting for the current thread's own admitted call would
            # deadlock. Treat reentrant teardown as an unclean quiesce: the
            # coordinator records the failure and retains this owner's graphs
            # instead of resetting underneath the still-running call.
            raise RuntimeError(
                "cannot tear down a CUDA graph owner from its active call"
            )
        while owner._cudagraph_active_calls:
            condition.wait()
        ensure_cudagraph_capture_complete(owner)


def _resume_cudagraph_owner_after_reset(owner: CUDAGraphOwner) -> None:
    """Reopen a known owner after a clean, scoped reusable reset."""
    _resume_owner_graph_creation(owner)
    condition = _owner_activity_condition(owner)
    with condition:
        owner._cudagraph_teardown_started = False  # type: ignore[attr-defined]
        condition.notify_all()


def cudagraph_owner_activity(
    method: Callable[Concatenate[Any, _P], _R],
) -> Callable[Concatenate[Any, _P], _R]:
    """Reject new work after teardown and track already-started owner calls."""

    @functools.wraps(method)
    def wrapped(owner: Any, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not getattr(owner, "_cudagraph_activity_enabled", True):
            return method(owner, *args, **kwargs)
        condition = _owner_activity_condition(owner)
        activity_local = owner._cudagraph_activity_local
        depth = getattr(activity_local, "depth", 0)
        if depth:
            # A public owner method may delegate to another decorated method on
            # the same owner. The outer admission remains active, so teardown
            # must not reject that already-started nested work.
            activity_local.depth = depth + 1
            try:
                return method(owner, *args, **kwargs)
            finally:
                activity_local.depth -= 1
        with condition:
            if getattr(owner, "_cudagraph_teardown_started", False):
                raise RuntimeError("CUDA graph owner is closed")
            owner._cudagraph_active_calls += 1
            activity_local.depth = 1
        try:
            return method(owner, *args, **kwargs)
        finally:
            activity_local.depth = 0
            with condition:
                owner._cudagraph_active_calls -= 1
                condition.notify_all()

    return cast(Callable[Concatenate[Any, _P], _R], wrapped)


@dataclasses.dataclass(frozen=True)
class CUDAGraphTeardownFailure:
    phase: str
    owner: str
    error: Exception
    traceback_text: str


@dataclasses.dataclass(frozen=True)
class CUDAGraphTeardownStats:
    graph_count: int
    owner_graph_counts: dict[str, int]
    enumeration_duration_s: float
    reset_duration_s: float
    clear_duration_s: float
    total_duration_s: float
    failures: tuple[CUDAGraphTeardownFailure, ...]
    _retained_owner_ids: frozenset[int] = dataclasses.field(
        default_factory=frozenset, repr=False, compare=False
    )
    _retained_graphs: tuple[OwnedCUDAGraph, ...] = dataclasses.field(
        default=(), repr=False, compare=False
    )

    @property
    def first_error(self) -> Exception | None:
        return self.failures[0].error if self.failures else None

    @property
    def fallback_required(self) -> bool:
        """Whether native resources were retained after incomplete teardown."""
        return bool(self._retained_owner_ids or self._retained_graphs)

    def raise_if_failed(self) -> None:
        if self.failures:
            raise CUDAGraphTeardownError(self.failures)


class CUDAGraphTeardownError(RuntimeError):
    def __init__(self, failures: Iterable[CUDAGraphTeardownFailure]) -> None:
        self.failures = tuple(failures)
        first = self.failures[0]
        super().__init__(
            f"CUDA graph teardown failed during {first.phase} for {first.owner}: "
            f"{first.error} ({len(self.failures)} failure(s) total)"
        )


def _owner_name(owner: CUDAGraphOwner) -> str:
    return type(owner).__qualname__


def _teardown_failure(
    phase: str, owner: str, error: Exception
) -> CUDAGraphTeardownFailure:
    traceback_text = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    if error.__traceback__ is not None:
        traceback.clear_frames(error.__traceback__)
    return CUDAGraphTeardownFailure(
        phase,
        owner,
        error.with_traceback(None),
        traceback_text,
    )


def _unexpected_coordinator_stats(
    owners: Iterable[CUDAGraphOwner],
    *,
    started_ns: int,
    coordinator: str,
    operation: str,
    error: Exception,
) -> CUDAGraphTeardownStats:
    owners = tuple(owners)
    ended_ns = time.perf_counter_ns()
    wrapped_error = RuntimeError(
        f"unexpected {operation} failure: {type(error).__qualname__}: {error}"
    )
    return CUDAGraphTeardownStats(
        graph_count=0,
        owner_graph_counts={},
        enumeration_duration_s=0.0,
        reset_duration_s=0.0,
        clear_duration_s=0.0,
        total_duration_s=(ended_ns - started_ns) / 1e9,
        failures=(_teardown_failure("coordinator", coordinator, wrapped_error),),
        _retained_owner_ids=frozenset(id(owner) for owner in owners),
    )


def _default_synchronize(device: torch.device) -> None:
    with torch.accelerator.device_index(device.index):
        torch.accelerator.synchronize(device)


def _default_device_context(device: torch.device) -> AbstractContextManager[Any]:
    return torch.accelerator.device_index(device.index)


def _device_sort_key(device: torch.device) -> tuple[str, int]:
    return device.type, -1 if device.index is None else device.index


def teardown_cudagraphs(
    owners: Iterable[CUDAGraphOwner],
    *,
    post_reset_sync: bool,
    marker_operation: str | None = None,
    synchronize: Callable[[torch.device], None] = _default_synchronize,
    device_context: Callable[
        [torch.device], AbstractContextManager[Any]
    ] = _default_device_context,
    emit_markers: bool = True,
) -> CUDAGraphTeardownStats:
    """Reset every graph before releasing any owner's graph state.

    Failures are accumulated so one broken graph or owner cannot prevent
    independent resources from being released.
    """

    marker_operation = marker_operation or "shutdown"
    marker_operation_id = uuid.uuid4().hex

    started_ns = time.perf_counter_ns()
    materialized_owners: list[CUDAGraphOwner] = []
    seen_owners: set[int] = set()
    for owner in owners:
        if id(owner) not in seen_owners:
            seen_owners.add(id(owner))
            materialized_owners.append(owner)

    failures: list[CUDAGraphTeardownFailure] = []
    owner_graph_counts: Counter[str] = Counter()

    def emit_marker(event: str, **fields: Any) -> None:
        if emit_markers:
            emit_shutdown_marker(
                event,
                operation=marker_operation,
                operation_id=marker_operation_id,
                **fields,
            )

    retained_owner_ids: set[int] = set()
    begin_failed_owner_ids: set[int] = set()
    owner_graph_ids: dict[int, set[int]] = {
        id(owner): set() for owner in materialized_owners
    }
    graph_owner_ids: dict[int, set[int]] = {}
    unsafe_graph_ids: set[int] = set()

    def retain_graph_owners(graph_ids: Iterable[int]) -> None:
        for graph_id in graph_ids:
            retained_owner_ids.update(graph_owner_ids.get(graph_id, ()))

    quiesce_started_ns = time.perf_counter_ns()
    emit_marker("owner_quiesce_begin", owner_count=len(materialized_owners))
    for owner in materialized_owners:
        try:
            _begin_owner_graph_creation_teardown(owner)
        except Exception as exc:
            failures.append(_teardown_failure("begin", _owner_name(owner), exc))
            retained_owner_ids.add(id(owner))
            begin_failed_owner_ids.add(id(owner))
        try:
            owner.begin_cudagraph_teardown()
        except Exception as exc:
            failures.append(_teardown_failure("begin", _owner_name(owner), exc))
            retained_owner_ids.add(id(owner))
            begin_failed_owner_ids.add(id(owner))
    quiesce_end_ns = time.perf_counter_ns()
    emit_marker(
        "owner_quiesce_end",
        duration_ns=quiesce_end_ns - quiesce_started_ns,
        failure_count=len(failures),
    )

    enumeration_started_ns = time.perf_counter_ns()
    emit_marker("graph_enumeration_begin", owner_count=len(materialized_owners))
    graphs_by_id: dict[int, OwnedCUDAGraph] = {}

    def record_owned_graph(
        owner: CUDAGraphOwner, owner_name: str, owned: OwnedCUDAGraph
    ) -> None:
        try:
            normalized = OwnedCUDAGraph(
                graph=owned.graph, device=torch.device(owned.device)
            )
        except Exception as exc:
            failures.append(_teardown_failure("enumerate", owner_name, exc))
            retained_owner_ids.add(id(owner))
            return
        graph_id = id(normalized.graph)
        if graph_id not in owner_graph_ids[id(owner)]:
            owner_graph_counts[owner_name] += 1
        owner_graph_ids[id(owner)].add(graph_id)
        graph_owner_ids.setdefault(graph_id, set()).add(id(owner))
        existing = graphs_by_id.get(graph_id)
        if existing is not None and existing.device != normalized.device:
            failures.append(
                _teardown_failure(
                    "enumerate",
                    owner_name,
                    RuntimeError(
                        "the same CUDA graph was reported on conflicting "
                        f"devices: {existing.device} and {normalized.device}"
                    ),
                )
            )
            unsafe_graph_ids.add(graph_id)
            retain_graph_owners((graph_id,))
            return
        graphs_by_id.setdefault(graph_id, normalized)

    def enumerate_owner(owner: CUDAGraphOwner) -> None:
        owner_name = _owner_name(owner)
        try:
            for owned in owner.iter_cudagraphs():
                record_owned_graph(owner, owner_name, owned)
        except Exception as exc:
            failures.append(_teardown_failure("enumerate", owner_name, exc))
            retained_owner_ids.add(id(owner))

        # The construction helper tracks a graph before invoking the owner's
        # metadata installer. This preserves graphs from failed installation
        # and from shutdown concurrent with native construction.
        for pending in _iter_pending_cudagraphs(owner):
            record_owned_graph(owner, owner_name, pending)

    for owner in materialized_owners:
        enumerate_owner(owner)

    enumeration_end_ns = time.perf_counter_ns()

    def group_graphs_by_device() -> dict[torch.device, list[OwnedCUDAGraph]]:
        grouped: dict[torch.device, list[OwnedCUDAGraph]] = {}
        for owned in graphs_by_id.values():
            grouped.setdefault(owned.device, []).append(owned)
        return grouped

    graphs_by_device = group_graphs_by_device()
    devices = sorted(graphs_by_device, key=_device_sort_key)
    graph_count = len(graphs_by_id)
    emit_marker(
        "graph_enumeration_end",
        graph_count=graph_count,
        owner_graph_counts=dict(owner_graph_counts),
        duration_ns=enumeration_end_ns - enumeration_started_ns,
    )

    reset_marker_started_ns = time.perf_counter_ns()
    reset_failure_start = len(failures)
    emit_marker("graph_reset_begin", device_count=len(devices))
    unsafe_devices: set[torch.device] = set()
    for device in devices:
        try:
            synchronize(device)
        except Exception as exc:
            failures.append(_teardown_failure("pre_reset_sync", str(device), exc))
            unsafe_devices.add(device)
            retain_graph_owners(id(owned.graph) for owned in graphs_by_device[device])

    def reset_device_graphs(device: torch.device) -> None:
        try:
            with device_context(device):
                for owned in graphs_by_device[device]:
                    graph_id = id(owned.graph)
                    if graph_id in unsafe_graph_ids or (
                        graph_owner_ids.get(graph_id, set()) & begin_failed_owner_ids
                    ):
                        retain_graph_owners((graph_id,))
                        continue
                    reset = getattr(owned.graph, "reset", None)
                    if not callable(reset):
                        failures.append(
                            _teardown_failure(
                                "reset",
                                type(owned.graph).__qualname__,
                                TypeError("CUDAGraph.reset() is unavailable"),
                            )
                        )
                        retain_graph_owners((graph_id,))
                        continue
                    try:
                        reset()
                    except Exception as exc:
                        failures.append(
                            _teardown_failure(
                                "reset", type(owned.graph).__qualname__, exc
                            )
                        )
                        retain_graph_owners((graph_id,))
        except Exception as exc:
            failures.append(_teardown_failure("select_device", str(device), exc))
            unsafe_devices.add(device)
            retain_graph_owners(id(owned.graph) for owned in graphs_by_device[device])

    for device in devices:
        if device not in unsafe_devices:
            reset_device_graphs(device)

    if post_reset_sync:
        for device in devices:
            if device in unsafe_devices:
                continue
            try:
                synchronize(device)
            except Exception as exc:
                failures.append(_teardown_failure("post_reset_sync", str(device), exc))
                retain_graph_owners(
                    id(owned.graph) for owned in graphs_by_device[device]
                )
    reset_end_ns = time.perf_counter_ns()
    emit_marker(
        "graph_reset_end",
        duration_ns=reset_end_ns - reset_marker_started_ns,
        failure_count=len(failures) - reset_failure_start,
    )

    clear_marker_started_ns = time.perf_counter_ns()
    clear_failure_start = len(failures)
    emit_marker("owner_clear_begin")
    for owner in materialized_owners:
        if id(owner) in retained_owner_ids:
            continue
        try:
            owner.clear_cudagraph_state()
        except Exception as exc:
            failures.append(_teardown_failure("clear", _owner_name(owner), exc))
            retained_owner_ids.add(id(owner))
        else:
            _clear_pending_cudagraphs(owner)

    retained_graph_ids: set[int] = set()
    for owner_id in retained_owner_ids:
        retained_graph_ids.update(owner_graph_ids.get(owner_id, ()))
    retained_graphs = tuple(
        graphs_by_id[graph_id]
        for graph_id in retained_graph_ids
        if graph_id in graphs_by_id
    )

    # Include Python wrapper destruction in the measured teardown phase. In
    # current ROCm PyTorch builds, releasing even an already-reset wrapper can
    # issue another device synchronization from CUDAGraph.__del__.
    graphs_by_device.clear()
    graphs_by_id.clear()
    clear_end_ns = time.perf_counter_ns()
    emit_marker(
        "owner_clear_end",
        duration_ns=clear_end_ns - clear_marker_started_ns,
        failure_count=len(failures) - clear_failure_start,
        retained_owner_count=len(retained_owner_ids),
    )

    return CUDAGraphTeardownStats(
        graph_count=graph_count,
        owner_graph_counts=dict(owner_graph_counts),
        enumeration_duration_s=(enumeration_end_ns - enumeration_started_ns) / 1e9,
        reset_duration_s=(reset_end_ns - reset_marker_started_ns) / 1e9,
        clear_duration_s=(clear_end_ns - clear_marker_started_ns) / 1e9,
        total_duration_s=(clear_end_ns - started_ns) / 1e9,
        failures=tuple(failures),
        _retained_owner_ids=frozenset(retained_owner_ids),
        _retained_graphs=retained_graphs,
    )


class _RegistryState(Enum):
    OPEN = auto()
    RESETTING = auto()
    TEARING_DOWN = auto()
    CLOSED = auto()


_REGISTRY_TOKEN_ATTR = "_cudagraph_owner_registry_token"
_REGISTRY_DIRECTORY: dict[str, weakref.ReferenceType["CUDAGraphOwnerRegistry"]] = {}
_REGISTRY_DIRECTORY_LOCK = threading.Lock()
# Incomplete native teardown is not equivalent to releasing Python owner state.
# Keep failed terminal registries rooted until process exit so the exit-only
# gc.freeze() fallback can preserve them through interpreter finalization.
_INCOMPLETE_TEARDOWN_REGISTRIES: list["CUDAGraphOwnerRegistry"] = []
_INCOMPLETE_TEARDOWN_DEPENDENCIES: list[Any] = []
_TERMINAL_CUDAGRAPH_TEARDOWN = ContextVar("terminal_cudagraph_teardown", default=False)


@contextmanager
def terminal_cudagraph_teardown() -> Iterator[None]:
    """Select the exit-only retention policy for nested runner shutdowns."""
    token = _TERMINAL_CUDAGRAPH_TEARDOWN.set(True)
    try:
        yield
    finally:
        _TERMINAL_CUDAGRAPH_TEARDOWN.reset(token)


def retain_cudagraph_dependency_for_terminal_exit(dependency: Any) -> None:
    """Keep unquiesced device state intact through terminal cyclic GC.

    This is the last-resort path used when shutdown cannot safely begin graph
    reset at all (for example, a worker callback thread failed to stop). The
    self-cycle survives module-global clearing after the dedicated process has
    frozen its exit heap. Outside a terminal context, the cycle remains
    collectable so same-process failure tests and callers do not leak forever.
    """
    dependency._terminal_fallback_cycle = dependency
    if not _TERMINAL_CUDAGRAPH_TEARDOWN.get() or getattr(
        dependency, "_terminal_fallback_quarantined", False
    ):
        return
    _INCOMPLETE_TEARDOWN_DEPENDENCIES.append(dependency)
    dependency._terminal_fallback_quarantined = True


class CUDAGraphOwnerRegistry:
    """Engine-scoped registry that keeps graph owners alive until teardown."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        strong_ownership: bool = True,
        synchronize: Callable[[torch.device], None] = _default_synchronize,
        device_context: Callable[
            [torch.device], AbstractContextManager[Any]
        ] = _default_device_context,
    ) -> None:
        self._owners: MutableMapping[int, CUDAGraphOwner] = (
            {} if strong_ownership else weakref.WeakValueDictionary()
        )
        self._strong_ownership = strong_ownership
        self._state = _RegistryState.OPEN
        self._stats: CUDAGraphTeardownStats | None = None
        self._fallback_dependencies: list[Any] = []
        self._terminal_fallback_quarantined = False
        self._operation_thread_id: int | None = None
        self._condition = threading.Condition()
        self._synchronize = synchronize
        self._device_context = device_context
        self._token = uuid.uuid4().hex
        token = self._token

        def remove_dead_registry(
            registry_ref: weakref.ReferenceType[CUDAGraphOwnerRegistry],
            token: str = token,
        ) -> None:
            with _REGISTRY_DIRECTORY_LOCK:
                if _REGISTRY_DIRECTORY.get(token) is registry_ref:
                    _REGISTRY_DIRECTORY.pop(token, None)

        registry_ref = weakref.ref(self, remove_dead_registry)
        with _REGISTRY_DIRECTORY_LOCK:
            setattr(
                vllm_config.compilation_config,
                _REGISTRY_TOKEN_ATTR,
                self._token,
            )
            _REGISTRY_DIRECTORY[self._token] = registry_ref

    def register(self, owner: CUDAGraphOwner) -> None:
        with self._condition:
            if self._state is not _RegistryState.OPEN:
                raise RuntimeError("cannot register a CUDA graph owner after teardown")
            owner_state = cast(Any, owner)
            existing_ref = vars(owner_state).get("_cudagraph_owner_registry_ref")
            existing_registry = existing_ref() if callable(existing_ref) else None
            if existing_registry is not None and existing_registry is not self:
                raise RuntimeError(
                    "CUDA graph owner is already registered to another registry"
                )
            self._owners.setdefault(id(owner), owner)
            owner_state._cudagraph_owner_registered = True
            owner_state._cudagraph_manual_owner = False
            owner_state._cudagraph_owner_registry_ref = weakref.ref(self)

    def owners(self) -> tuple[CUDAGraphOwner, ...]:
        with self._condition:
            return tuple(self._owners.values())

    def owners_of_type(
        self, owner_types: tuple[type, ...]
    ) -> tuple[CUDAGraphOwner, ...]:
        return tuple(owner for owner in self.owners() if isinstance(owner, owner_types))

    def _begin_registry_operation_locked(self, state: _RegistryState) -> None:
        if self._state is not _RegistryState.OPEN:
            raise RuntimeError("CUDA graph owner registry is closing or closed")
        thread_id = threading.get_ident()
        self._state = state
        self._operation_thread_id = thread_id

    def retain_fallback_dependency(self, dependency: Any) -> None:
        """Keep state needed by an incompletely reset graph alive."""
        with self._condition:
            if (
                self._state is not _RegistryState.CLOSED
                or self._stats is None
                or not self._stats.fallback_required
            ):
                raise RuntimeError(
                    "fallback dependencies require an incomplete closed registry"
                )
            self._fallback_dependencies.append(dependency)

    def retain_terminal_dependency(self, dependency: Any) -> None:
        """Retain device state after a terminal safety gate failed.

        This is broader than ``retain_fallback_dependency``: a runner-level
        synchronization can fail even when no graph was enumerated, leaving no
        retained graph in the otherwise clean registry stats. During terminal
        child shutdown, root the dependency through final collection so its
        tensor/native destructors cannot run before the exit-only freeze.
        """
        with self._condition:
            if self._state is not _RegistryState.CLOSED:
                raise RuntimeError(
                    "terminal dependencies require a closed CUDA graph registry"
                )
            self._fallback_dependencies.append(dependency)
            if self._terminal_fallback_requested(None):
                self._quarantine_terminal_fallback_locked()

    @staticmethod
    def _terminal_fallback_requested(explicit: bool | None) -> bool:
        return _TERMINAL_CUDAGRAPH_TEARDOWN.get() if explicit is None else explicit

    def _quarantine_terminal_fallback_locked(self) -> None:
        if self._terminal_fallback_quarantined:
            return
        self._terminal_fallback_cycle = self
        _INCOMPLETE_TEARDOWN_REGISTRIES.append(self)
        self._terminal_fallback_quarantined = True

    def reset_for_reuse(
        self,
        owners: Iterable[CUDAGraphOwner] | None = None,
        *,
        post_reset_sync: bool,
    ) -> CUDAGraphTeardownStats:
        """Reset temporary graphs without closing their reusable owners."""
        explicitly_selected = tuple(owners) if owners is not None else None
        with self._condition:
            while self._state is _RegistryState.RESETTING:
                if self._operation_thread_id == threading.get_ident():
                    raise RuntimeError(
                        "reusable CUDA graph reset is already in progress on "
                        "this thread"
                    )
                self._condition.wait()
            if self._state is not _RegistryState.OPEN:
                raise RuntimeError("CUDA graph owner registry is closing or closed")
            selected = (
                tuple(self._owners.values())
                if explicitly_selected is None
                else explicitly_selected
            )
            for owner in selected:
                owner_attrs = vars(owner)
                registry_ref = owner_attrs.get("_cudagraph_owner_registry_ref")
                registered_registry = registry_ref() if callable(registry_ref) else None
                if owner_attrs.get("_cudagraph_manual_owner", False):
                    continue
                if registered_registry is not self:
                    raise RuntimeError(
                        "cannot reset a CUDA graph owner from another registry"
                    )
            self._begin_registry_operation_locked(_RegistryState.RESETTING)
        started_ns = time.perf_counter_ns()
        try:
            stats = teardown_cudagraphs(
                selected,
                post_reset_sync=post_reset_sync,
                marker_operation="reset_for_reuse",
                synchronize=self._synchronize,
                device_context=self._device_context,
                emit_markers=False,
            )
            if not stats.fallback_required:
                for owner in selected:
                    _resume_cudagraph_owner_after_reset(owner)
        except Exception as exc:
            stats = _unexpected_coordinator_stats(
                selected,
                started_ns=started_ns,
                coordinator=type(self).__qualname__,
                operation="reusable CUDA graph reset",
                error=exc,
            )
            with self._condition:
                self._stats = stats
                self._fallback_dependencies.extend(selected)
                self._state = _RegistryState.CLOSED
                self._operation_thread_id = None
                self._condition.notify_all()
            raise

        if stats.fallback_required:
            stats = dataclasses.replace(
                stats,
                _retained_owner_ids=stats._retained_owner_ids
                | frozenset(id(owner) for owner in selected),
            )

        with self._condition:
            if stats.fallback_required:
                self._stats = stats
                self._fallback_dependencies.extend(selected)
                self._state = _RegistryState.CLOSED
            else:
                self._state = _RegistryState.OPEN
            self._operation_thread_id = None
            self._condition.notify_all()
        return stats

    def teardown(
        self,
        *,
        post_reset_sync: bool,
        terminal_fallback: bool | None = None,
    ) -> CUDAGraphTeardownStats:
        with self._condition:
            while self._state in (
                _RegistryState.RESETTING,
                _RegistryState.TEARING_DOWN,
            ):
                if self._operation_thread_id == threading.get_ident():
                    if self._state is _RegistryState.TEARING_DOWN:
                        raise RuntimeError(
                            "CUDA graph teardown is already in progress on this thread"
                        )
                    raise RuntimeError(
                        "cannot tear down a CUDA graph registry from its "
                        "reusable reset callback"
                    )
                self._condition.wait()
            if self._state is _RegistryState.CLOSED:
                assert self._stats is not None
                if self._stats.fallback_required and self._terminal_fallback_requested(
                    terminal_fallback
                ):
                    self._quarantine_terminal_fallback_locked()
                return self._stats
            self._begin_registry_operation_locked(_RegistryState.TEARING_DOWN)
            owners = tuple(self._owners.values())

        started_ns = time.perf_counter_ns()
        try:
            stats = teardown_cudagraphs(
                owners,
                post_reset_sync=post_reset_sync,
                synchronize=self._synchronize,
                device_context=self._device_context,
            )
        except Exception as exc:
            failed_stats = _unexpected_coordinator_stats(
                owners,
                started_ns=started_ns,
                coordinator=type(self).__qualname__,
                operation="CUDA graph teardown",
                error=exc,
            )
            with self._condition:
                self._stats = failed_stats
                self._fallback_dependencies.extend(owners)
                if self._terminal_fallback_requested(terminal_fallback):
                    self._quarantine_terminal_fallback_locked()
                self._state = _RegistryState.CLOSED
                self._operation_thread_id = None
                self._condition.notify_all()
            raise

        with self._condition:
            self._stats = stats
            if stats.fallback_required:
                retained_owners = {
                    owner_id: owner
                    for owner_id, owner in self._owners.items()
                    if owner_id in stats._retained_owner_ids
                }
                self._owners = retained_owners
                if self._terminal_fallback_requested(terminal_fallback):
                    # Deliberately form a cycle. A clean terminal process
                    # freezes this quarantined state after final collection,
                    # preventing unsafe wrapper destruction during interpreter
                    # shutdown. Same-process callers retain state only through
                    # their registry and receive the teardown error normally.
                    self._quarantine_terminal_fallback_locked()
            else:
                self._owners.clear()
            self._state = _RegistryState.CLOSED
            self._operation_thread_id = None
            self._condition.notify_all()
        return stats


def register_cudagraph_owner(owner: CUDAGraphOwner, vllm_config: VllmConfig) -> bool:
    _pending_cudagraph_state(owner)
    with _REGISTRY_DIRECTORY_LOCK:
        token = getattr(vllm_config.compilation_config, _REGISTRY_TOKEN_ATTR, None)
        registry_ref = _REGISTRY_DIRECTORY.get(token) if token is not None else None
    owner_state = cast(Any, owner)
    owner_state._cudagraph_manual_owner = token is None
    if token is None:
        owner_state._cudagraph_owner_registered = False
        owner_state._cudagraph_owner_registry_ref = None
        return False
    registry = registry_ref() if registry_ref is not None else None
    if registry is not None:
        registry.register(owner)
        owner_state._cudagraph_owner_registered = True
        owner_state._cudagraph_owner_registry_ref = weakref.ref(registry)
        return registry._strong_ownership
    owner_state._cudagraph_owner_registered = False
    owner_state._cudagraph_owner_registry_ref = None
    return False


def current_cudagraph_device() -> torch.device:
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        raise RuntimeError("no active accelerator is available for CUDA graph capture")
    return torch.device(accelerator.type, torch.accelerator.current_device_index())


def create_cudagraph(
    owner: CUDAGraphOwner,
    device: torch.device,
    install: Callable[[torch.cuda.CUDAGraph], None],
) -> torch.cuda.CUDAGraph:
    """Construct and install a graph without an untracked handoff window.

    The graph is added to owner-local pending state before ``install`` runs.
    If installation raises, deterministic teardown can still enumerate the
    graph. A successful installer makes the owner's normal metadata
    authoritative and removes the temporary pending record.
    """
    for method_name in (
        "begin_cudagraph_teardown",
        "iter_cudagraphs",
        "clear_cudagraph_state",
    ):
        if not callable(getattr(owner, method_name, None)):
            raise TypeError(
                "CUDA graphs must be constructed by a CUDAGraphOwner; "
                f"missing {method_name}()"
            )
    owner_attrs = vars(owner)
    if not (
        owner_attrs.get("_cudagraph_owner_registered", False)
        or owner_attrs.get("_cudagraph_manual_owner", False)
    ):
        raise RuntimeError("CUDA graph owner registry is unavailable")
    if owner_attrs.get("_cudagraph_owner_registered", False):
        registry_ref = owner_attrs.get("_cudagraph_owner_registry_ref")
        registry = registry_ref() if callable(registry_ref) else None
        if registry is None:
            raise RuntimeError("CUDA graph owner registry is unavailable")

    with _owner_graph_creation(owner):
        normalized_device = torch.device(device)
        pending = _pending_cudagraph_state(owner)
        graph = torch.cuda.CUDAGraph()
        with pending.lock:
            pending.graphs[id(graph)] = OwnedCUDAGraph(graph, normalized_device)
        # Leave the pending record intact if installation fails. The normal
        # owner metadata becomes authoritative only after a successful handoff.
        install(graph)
        with pending.lock:
            pending.graphs.pop(id(graph), None)
        return graph


@dataclasses.dataclass(frozen=True)
class CUDAGraphStat:
    num_unpadded_tokens: int
    num_padded_tokens: int
    num_paddings: int
    runtime_mode: str


class CUDAGraphLogging:
    """Aggregate and log cudagraph metrics"""

    COLUMN_HEADERS = [
        "Unpadded Tokens",
        "Padded Tokens",
        "Num Paddings",
        "Runtime Mode",
        "Count",
    ]

    def __init__(
        self, cg_mode: CUDAGraphMode, cg_capture_sizes: list[int] | None
    ) -> None:
        self.reset()
        self.cg_mode = str(cg_mode)
        self.cg_capture_sizes = str(cg_capture_sizes or [])

        self.settings_header = (
            "**CUDAGraph Config Settings:**\n\n"
            f"- Mode: {self.cg_mode}\n"
            f"- Capture sizes: {self.cg_capture_sizes}\n\n"
            "**CUDAGraph Stats:**\n\n"
        )

    def reset(self) -> None:
        self.stats: list[CUDAGraphStat] = []

    def observe(self, cudagraph_stat: CUDAGraphStat) -> None:
        self.stats.append(cudagraph_stat)

    def generate_metric_table(self) -> str:
        stats_counts = Counter(self.stats)

        # Convert stats to rows of strings, in descending order of observed frequencies
        rows = []
        for stat, count in sorted(
            stats_counts.items(), key=lambda item: item[1], reverse=True
        ):
            rows.append(
                [
                    str(stat.num_unpadded_tokens),
                    str(stat.num_padded_tokens),
                    str(stat.num_paddings),
                    stat.runtime_mode,
                    str(count),
                ]
            )

        # Calculate column widths (max of header and data)
        col_widths = []
        for i, header_text in enumerate(self.COLUMN_HEADERS):
            max_width = len(header_text)
            for row in rows:
                max_width = max(max_width, len(row[i]))
            col_widths.append(max_width)

        table_header_list = [
            h.ljust(w) for h, w in zip(self.COLUMN_HEADERS, col_widths)
        ]
        table_header = "| " + " | ".join(table_header_list) + " |\n"

        table_separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|\n"

        # Create data rows with proper alignment
        data_rows = []
        for row in rows:
            formatted_row = [
                str(val).ljust(width) for val, width in zip(row, col_widths)
            ]
            data_rows.append("| " + " | ".join(formatted_row) + " |")

        return (
            self.settings_header
            + table_header
            + table_separator
            + "\n".join(data_rows)
            + "\n"
        )

    def log(self, log_fn: Callable[..., Any] = logger.info) -> None:
        if not self.stats:
            return
        log_fn(self.generate_metric_table())
        self.reset()


@dataclasses.dataclass
class CUDAGraphEntry:
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None
    device: torch.device | None = None
    output: Any | None = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: list[int] | None = None


@dataclasses.dataclass
class CUDAGraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True


class CUDAGraphWrapper:
    """Wraps a runnable to add CUDA graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the cudagraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for cudagraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform cudagraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: CUDAGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
    ) -> None:
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

        # assert runtime_mode is not NONE(no cudagraph), otherwise, we don't
        # need to initialize a CUDAGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.cudagraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # cudagraphs for.
        self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry] = {}
        self._cudagraph_teardown_started = False
        self._cudagraph_activity_enabled = register_cudagraph_owner(self, vllm_config)

    def __getattr__(self, key: str) -> Any:
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of "
                f"cudagraph wrapper: {self._runnable_str}"
            )
        raise AttributeError

    def unwrap(self) -> Callable[..., Any]:
        # in case we need to access the original runnable.
        return self.runnable

    @property
    def cudagraph_wrapper(self) -> "CUDAGraphWrapper":
        return self

    def clear_graphs(self) -> None:
        self.clear_cudagraph_state()

    def begin_cudagraph_teardown(self) -> None:
        begin_cudagraph_owner_teardown(self)

    def iter_cudagraphs(self) -> Iterable[OwnedCUDAGraph]:
        for entry in self.concrete_cudagraph_entries.values():
            if entry.cudagraph is not None and entry.device is not None:
                yield OwnedCUDAGraph(entry.cudagraph, entry.device)

    def clear_cudagraph_state(self) -> None:
        self.concrete_cudagraph_entries.clear()

    @cudagraph_owner_activity
    def __call__(self, *args: Any, **kwargs: Any) -> Any | None:
        if not is_forward_context_available():
            # No forward context means we are outside the normal
            # inference path (e.g. a vision encoder forward pass).
            # Just run the underlying function without cudagraphs.
            return self.runnable(*args, **kwargs)

        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        assert batch_descriptor is not None
        if batch_descriptor not in self.concrete_cudagraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_cudagraph_entries[batch_descriptor]

        if entry.cudagraph is None:
            if self.cudagraph_options.debug_log_enable:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug(
                    "Capturing a cudagraph on (%s,%s)",
                    self.runtime_mode.name,
                    entry.batch_descriptor,
                )
            # validate that cudagraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            device = current_cudagraph_device()
            entry.device = device

            def install_graph(graph: torch.cuda.CUDAGraph) -> None:
                entry.cudagraph = graph

            cudagraph = create_cudagraph(
                self,
                device,
                install_graph,
            )

            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    # during every model forward for piecewise cudagraph
                    # mode, we will capture many pieces of cudagraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(
                        patch("gc.collect", lambda *args, **kwargs: None)
                    )
                    stack.enter_context(
                        patch(
                            "torch.accelerator.empty_cache",
                            lambda *args, **kwargs: None,
                        )
                    )

                if self.graph_pool is not None:
                    set_graph_pool_id(self.graph_pool)
                else:
                    set_graph_pool_id(current_platform.graph_pool_handle())

                # Sync offloader's copy stream before capture.
                # Ensure any pre-capture prefetches from offloader are complete.
                get_offloader().sync_prev_onload()

                # mind-exploding: carefully manage the reference and memory.
                with (
                    cudagraph_capture_attempt(self),
                    torch.cuda.graph(
                        cudagraph,
                        pool=self.graph_pool,
                        stream=current_stream(),
                    ),
                ):
                    # `output` is managed by pytorch's cudagraph pool
                    output = self.runnable(*args, **kwargs)
                    # Join offloader's copy stream after forward to avoid
                    # unjoined stream error. The last layer's start_prefetch
                    # forks copy_stream, but wait_prefetch only happens in
                    # the next forward pass.
                    get_offloader().join_after_forward()
                    if self.cudagraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise cuadgraph mode, because
                        # the output of the last graph will not be used by
                        # any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for cudagraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        # Sync offloader before replay - ensures any external dependencies
        # from pre-capture prefetches are satisfied.
        get_offloader().sync_prev_onload()
        entry.cudagraph.replay()
        return entry.output
