#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure PyTorch CUDA graph lifetime and teardown behavior.

This benchmark intentionally imports only PyTorch, not vLLM.  It emits exactly
one JSON result record on stdout.  Phase markers are written with ``os.write``
to stderr by default, which makes it possible to align the result with a
ROCprofiler runtime trace.

Example ROCprofiler invocation::

    rocprofv3 --runtime-trace --stats --output-format json \
      --output-directory rocprof-cudagraph -- \
      .venv/bin/python benchmarks/cudagraph_lifetime.py \
      --num-graphs 64 --mode explicit-reset-one-sync --device cuda:0 \
      2>phase-markers.jsonl >result.jsonl

An already-open append-only file descriptor can be used instead of stderr::

    exec 3>>phase-markers.jsonl
    .venv/bin/python benchmarks/cudagraph_lifetime.py -n 8 \
      --mode in-flight-reset --marker-fd 3
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import json
import os
import platform
import sys
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch

MODES = (
    "del-gc",
    "explicit-reset-one-sync",
    "in-flight-reset",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture, replay, and tear down standalone PyTorch CUDA graphs. "
            "The program writes one JSON result to stdout and phase markers "
            "to stderr (or --marker-fd)."
        ),
        epilog=(
            "ROCprofiler example:\n"
            "  rocprofv3 --runtime-trace --stats --output-format json "
            "--output-directory rocprof-cudagraph -- "
            ".venv/bin/python benchmarks/cudagraph_lifetime.py -n 64 "
            "--mode explicit-reset-one-sync --device cuda:0 "
            "2>phase-markers.jsonl >result.jsonl\n\n"
            "The intended graph-count sweep is N=1,8,64,256."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--num-graphs",
        type=_positive_int,
        required=True,
        help="number of independent graphs to capture",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        required=True,
        help="graph teardown sequence to exercise",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch CUDA device (ROCm uses the CUDA-compatible API)",
    )
    parser.add_argument(
        "--numel",
        type=_positive_int,
        default=4096,
        help="number of float32 elements in each static input/output (default: 4096)",
    )
    parser.add_argument(
        "--marker-fd",
        type=_nonnegative_int,
        default=2,
        help=(
            "inherited descriptor for JSON phase markers (default: 2/stderr); "
            "open it with O_APPEND when several processes share the file"
        ),
    )
    return parser.parse_args(argv)


class PhaseMarkers:
    """Emit small newline-delimited JSON records without buffered I/O."""

    def __init__(
        self,
        fd: int,
        *,
        run_id: str,
        mode: str,
        num_graphs: int,
        device: str,
    ) -> None:
        self.fd = fd
        self.common = {
            "record_type": "cudagraph_phase_marker",
            "run_id": run_id,
            "pid": os.getpid(),
            "mode": mode,
            "num_graphs": num_graphs,
            "device": device,
        }

    def emit(self, event: str, phase: str, **fields: Any) -> None:
        record = {
            **self.common,
            "event": event,
            "phase": phase,
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_ns": time.time_ns(),
            **fields,
        }
        payload = (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        offset = 0
        while offset < len(payload):
            offset += os.write(self.fd, payload[offset:])


@contextmanager
def _timed_phase(
    markers: PhaseMarkers,
    timings_ns: dict[str, int],
    name: str,
    *,
    operation: str,
) -> Iterator[None]:
    markers.emit("begin", name, operation=operation)
    start_ns = time.perf_counter_ns()
    try:
        yield
    except BaseException as exc:
        elapsed_ns = time.perf_counter_ns() - start_ns
        timings_ns[name] = elapsed_ns
        markers.emit(
            "end",
            name,
            operation=operation,
            duration_ns=elapsed_ns,
            status="error",
            error_type=type(exc).__name__,
        )
        raise
    else:
        elapsed_ns = time.perf_counter_ns() - start_ns
        timings_ns[name] = elapsed_ns
        markers.emit(
            "end",
            name,
            operation=operation,
            duration_ns=elapsed_ns,
            status="ok",
        )


@dataclass
class StaticBuffers:
    input: torch.Tensor
    output: torch.Tensor
    expected: float


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _capture_graphs(
    *,
    num_graphs: int,
    numel: int,
    device: torch.device,
    stream: torch.cuda.Stream,
) -> tuple[list[torch.cuda.CUDAGraph], list[StaticBuffers]]:
    graphs: list[torch.cuda.CUDAGraph] = []
    buffers: list[StaticBuffers] = []

    # Prime allocation and kernel-loading paths outside capture.
    with torch.inference_mode(), torch.cuda.stream(stream):
        warmup_input = torch.ones(numel, dtype=torch.float32, device=device)
        warmup_output = torch.empty_like(warmup_input)
        torch.add(warmup_input, 2.0, out=warmup_output)
    stream.synchronize()
    del warmup_input, warmup_output

    with torch.inference_mode():
        for index in range(num_graphs):
            static_input = torch.empty(
                numel,
                dtype=torch.float32,
                device=device,
            )
            static_output = torch.empty_like(static_input)
            expected = float(index) + 2.25

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                torch.add(static_input, 2.0, out=static_output)

            graphs.append(graph)
            buffers.append(
                StaticBuffers(
                    input=static_input,
                    output=static_output,
                    expected=expected,
                )
            )

    return graphs, buffers


def _enqueue_replays(
    graphs: list[torch.cuda.CUDAGraph],
    buffers: list[StaticBuffers],
    stream: torch.cuda.Stream,
) -> None:
    with torch.inference_mode(), torch.cuda.stream(stream):
        for index in range(len(graphs)):
            buffers[index].input.fill_(float(index) + 0.25)
            buffers[index].output.fill_(float("nan"))
            graphs[index].replay()


def _reset_graphs(graphs: list[torch.cuda.CUDAGraph]) -> int:
    for index in range(len(graphs)):
        graphs[index].reset()
    return len(graphs)


def _validate_outputs(
    buffers: list[StaticBuffers],
    stream: torch.cuda.Stream,
) -> dict[str, int | float | bool]:
    # A single stack/copy gives validation one well-marked synchronization
    # rather than N tensor-to-host synchronizations that could obscure the
    # teardown call-count trace.
    with torch.inference_mode(), torch.cuda.stream(stream):
        observed = torch.stack([item.output for item in buffers]).cpu()

    expected = torch.empty_like(observed)
    for index in range(len(buffers)):
        expected[index].fill_(buffers[index].expected)

    is_correct = bool(torch.equal(observed, expected))
    checksum = float(observed.to(dtype=torch.float64).sum().item())
    expected_checksum = float(expected.to(dtype=torch.float64).sum().item())
    return {
        "correct": is_correct,
        "checked_elements": observed.numel(),
        "checksum": checksum,
        "expected_checksum": expected_checksum,
    }


def _safe_attr(obj: object, name: str) -> Any:
    value = getattr(obj, name, None)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode(errors="backslashreplace")
    if isinstance(value, (tuple, list)):
        return list(value)
    return str(value)


def _hip_runtime_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "library": None,
        "runtime_version_raw": None,
        "runtime_version_status": None,
        "driver_version_raw": None,
        "driver_version_status": None,
    }
    candidates = [ctypes.util.find_library("amdhip64"), "libamdhip64.so"]
    errors: list[str] = []
    library = None
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            library = ctypes.CDLL(candidate)
            result["library"] = str(library._name)
            break
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")

    if library is None:
        result["load_errors"] = errors
        return result

    for function_name, value_key, status_key in (
        ("hipRuntimeGetVersion", "runtime_version_raw", "runtime_version_status"),
        ("hipDriverGetVersion", "driver_version_raw", "driver_version_status"),
    ):
        try:
            function = getattr(library, function_name)
            function.argtypes = [ctypes.POINTER(ctypes.c_int)]
            function.restype = ctypes.c_int
            value = ctypes.c_int()
            status = int(function(ctypes.byref(value)))
            result[value_key] = int(value.value)
            result[status_key] = status
        except (AttributeError, OSError) as exc:
            result[f"{function_name}_error"] = str(exc)
    return result


def _torch_metadata() -> dict[str, Any]:
    return {
        "version": torch.__version__,
        "git_version": getattr(torch.version, "git_version", None),
        "hip_build_version": getattr(torch.version, "hip", None),
        "cuda_build_version": getattr(torch.version, "cuda", None),
        "debug_build": bool(getattr(torch.version, "debug", False)),
        "cuda_available": torch.cuda.is_available(),
        "build_config": torch.__config__.show(),
    }


def _process_metadata() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "argv": sys.argv,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "visible_device_environment": {
            name: os.environ.get(name)
            for name in (
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
            )
        },
    }


def _resolve_device(requested: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    device = torch.device(requested)
    if device.type != "cuda":
        raise ValueError(f"--device must select a CUDA/HIP device, got {requested!r}")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"device index {index} is outside the visible range "
            f"[0, {torch.cuda.device_count()})"
        )
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def _device_metadata(
    requested: str,
    device: torch.device,
) -> dict[str, Any]:
    assert device.index is not None
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    try:
        gencode_flags = torch.cuda.get_gencode_flags()
        gencode_flags_error = None
    except Exception as exc:
        # Some ROCm builds return a one-field architecture entry while this
        # CUDA-oriented helper expects ``(kind, arch)`` pairs.  Keep the raw
        # architecture list and report the helper failure instead of making
        # benchmark metadata collection fatal.
        gencode_flags = None
        gencode_flags_error = f"{type(exc).__name__}: {exc}"

    return {
        "requested": requested,
        "resolved": str(device),
        "index": device.index,
        "visible_device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "name": properties.name,
        "capability": list(capability),
        "gcn_arch_name": _safe_attr(properties, "gcnArchName"),
        "uuid": _safe_attr(properties, "uuid"),
        "total_memory_bytes": properties.total_memory,
        "multi_processor_count": properties.multi_processor_count,
        "major": properties.major,
        "minor": properties.minor,
        "pci_domain_id": _safe_attr(properties, "pci_domain_id"),
        "pci_bus_id": _safe_attr(properties, "pci_bus_id"),
        "pci_device_id": _safe_attr(properties, "pci_device_id"),
        "is_integrated": _safe_attr(properties, "is_integrated"),
        "architecture_list": torch.cuda.get_arch_list(),
        "gencode_flags": gencode_flags,
        "gencode_flags_error": gencode_flags_error,
    }


def _emergency_cleanup(
    graphs: list[torch.cuda.CUDAGraph],
    buffers: list[StaticBuffers],
    device: torch.device,
    markers: PhaseMarkers,
) -> None:
    markers.emit("begin", "emergency_cleanup")
    cleanup_errors: list[str] = []
    try:
        _reset_graphs(graphs)
    except BaseException as exc:
        cleanup_errors.append(f"reset: {type(exc).__name__}: {exc}")
    try:
        torch.cuda.synchronize(device)
    except BaseException as exc:
        cleanup_errors.append(f"synchronize: {type(exc).__name__}: {exc}")
    graphs.clear()
    buffers.clear()
    gc.collect()
    markers.emit(
        "end",
        "emergency_cleanup",
        status="ok" if not cleanup_errors else "error",
        cleanup_errors=cleanup_errors,
    )


def run_benchmark(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    device = _resolve_device(args.device)
    markers = PhaseMarkers(
        args.marker_fd,
        run_id=run_id,
        mode=args.mode,
        num_graphs=args.num_graphs,
        device=str(device),
    )
    record: dict[str, Any] = {
        "record_type": "cudagraph_lifetime_result",
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            "num_graphs": args.num_graphs,
            "mode": args.mode,
            "device": args.device,
            "numel": args.numel,
            "marker_fd": args.marker_fd,
        },
        "process": _process_metadata(),
        "torch": _torch_metadata(),
        "hip_runtime": _hip_runtime_metadata(),
        "device": _device_metadata(args.device, device),
        "dtype": str(torch.float32),
        "bytes_per_graph_static_buffers": args.numel * 2 * 4,
        "timings_ns": {
            "capture": 0,
            "replay_enqueue": 0,
            "replay_quiesce": 0,
            "reset_or_release": 0,
            "explicit_sync": 0,
            "wrapper_destruction_gc": 0,
            "validation": 0,
            "buffer_release_gc": 0,
            "total": 0,
        },
        "phase_operations": {},
        "phase_order": [],
        "memory": {},
    }
    timings_ns: dict[str, int] = record["timings_ns"]
    phase_operations: dict[str, str] = record["phase_operations"]
    phase_order: list[str] = record["phase_order"]
    memory: dict[str, dict[str, int]] = record["memory"]

    graphs: list[torch.cuda.CUDAGraph] = []
    buffers: list[StaticBuffers] = []
    replay_stream = torch.cuda.Stream(device=device)
    reset_count = 0
    released_count = 0
    gc_collected = 0

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    memory["before_capture"] = _memory_snapshot(device)
    total_start_ns = time.perf_counter_ns()
    markers.emit("begin", "benchmark")

    def run_phase(name: str, operation: str) -> Any:
        phase_order.append(name)
        phase_operations[name] = operation
        return _timed_phase(
            markers,
            timings_ns,
            name,
            operation=operation,
        )

    try:
        with run_phase("capture", "capture_independent_graphs"):
            graphs, buffers = _capture_graphs(
                num_graphs=args.num_graphs,
                numel=args.numel,
                device=device,
                stream=replay_stream,
            )
        memory["after_capture"] = _memory_snapshot(device)

        with run_phase("replay_enqueue", "enqueue_on_non_default_stream"):
            _enqueue_replays(graphs, buffers, replay_stream)
        memory["after_replay_enqueue"] = _memory_snapshot(device)

        if args.mode == "in-flight-reset":
            phase_order.append("replay_quiesce")
            phase_operations["replay_quiesce"] = "deliberately_omitted"
            markers.emit(
                "skip",
                "replay_quiesce",
                operation="deliberately_omitted",
                duration_ns=0,
            )
        else:
            with run_phase("replay_quiesce", "replay_stream_synchronize_once"):
                replay_stream.synchronize()
        memory["after_replay_quiesce"] = _memory_snapshot(device)

        if args.mode == "del-gc":
            with run_phase(
                "reset_or_release",
                "release_last_graph_wrapper_references",
            ):
                released_count = len(graphs)
                graphs.clear()
            memory["after_reset_or_release"] = _memory_snapshot(device)

            phase_order.append("explicit_sync")
            phase_operations["explicit_sync"] = "not_requested"
            markers.emit(
                "skip",
                "explicit_sync",
                operation="not_requested",
                duration_ns=0,
            )
            memory["after_explicit_sync"] = _memory_snapshot(device)

            with run_phase(
                "wrapper_destruction_gc",
                "collect_after_wrapper_release",
            ):
                gc_collected = gc.collect()
            memory["after_wrapper_destruction_gc"] = _memory_snapshot(device)

        elif args.mode == "explicit-reset-one-sync":
            with run_phase("reset_or_release", "reset_each_graph"):
                reset_count = _reset_graphs(graphs)
            memory["after_reset_or_release"] = _memory_snapshot(device)

            with run_phase("explicit_sync", "device_synchronize_once"):
                torch.cuda.synchronize(device)
            memory["after_explicit_sync"] = _memory_snapshot(device)

            with run_phase(
                "wrapper_destruction_gc",
                "release_reset_wrappers_then_collect",
            ):
                released_count = len(graphs)
                graphs.clear()
                gc_collected = gc.collect()
            memory["after_wrapper_destruction_gc"] = _memory_snapshot(device)

        else:
            with run_phase(
                "reset_or_release",
                "reset_each_graph_while_replay_is_in_flight",
            ):
                reset_count = _reset_graphs(graphs)
            memory["after_reset_or_release"] = _memory_snapshot(device)

            with run_phase(
                "wrapper_destruction_gc",
                "release_reset_wrappers_while_replay_is_in_flight_then_collect",
            ):
                released_count = len(graphs)
                graphs.clear()
                gc_collected = gc.collect()
            memory["after_wrapper_destruction_gc"] = _memory_snapshot(device)

            with run_phase("explicit_sync", "replay_stream_synchronize_once"):
                replay_stream.synchronize()
            memory["after_explicit_sync"] = _memory_snapshot(device)

        with run_phase(
            "validation",
            "stack_outputs_copy_once_to_host_and_compare",
        ):
            validation = _validate_outputs(buffers, replay_stream)
        memory["after_validation"] = _memory_snapshot(device)

        with run_phase(
            "buffer_release_gc",
            "release_retained_static_buffers_then_collect",
        ):
            buffers.clear()
            gc.collect()
        memory["after_buffer_release_gc"] = _memory_snapshot(device)

        timings_ns["total"] = time.perf_counter_ns() - total_start_ns
        record.update(
            {
                "status": "ok" if validation["correct"] else "incorrect",
                "graphs_captured": args.num_graphs,
                "graphs_reset_explicitly": reset_count,
                "graph_wrapper_references_released": released_count,
                "gc_objects_collected_during_wrapper_phase": gc_collected,
                "validation": validation,
                "peak_memory": {
                    "allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "reserved_bytes": torch.cuda.max_memory_reserved(device),
                },
            }
        )
        markers.emit(
            "end",
            "benchmark",
            status=record["status"],
            duration_ns=timings_ns["total"],
        )
        return record
    except BaseException:
        _emergency_cleanup(graphs, buffers, device, markers)
        timings_ns["total"] = time.perf_counter_ns() - total_start_ns
        markers.emit(
            "end",
            "benchmark",
            status="error",
            duration_ns=timings_ns["total"],
        )
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = str(uuid.uuid4())
    try:
        record = run_benchmark(args, run_id)
    except BaseException as exc:
        record = {
            "record_type": "cudagraph_lifetime_result",
            "schema_version": 1,
            "run_id": run_id,
            "status": "error",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": vars(args),
            "process": _process_metadata(),
            "torch": _torch_metadata(),
            "hip_runtime": _hip_runtime_metadata(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        exit_code = 1
    else:
        exit_code = 0 if record["status"] == "ok" else 2

    sys.stdout.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
