#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure vLLM shutdown and post-exit VRAM recovery from a parent process.

The benchmark owns a fresh process group, sends exactly one shutdown signal,
and treats ``waitpid`` in the parent as the authoritative exit observation.
The child receives an inherited marker descriptor in
``VLLM_SHUTDOWN_MARKER_FD`` and an append-only fallback path in
``VLLM_SHUTDOWN_MARKER_PATH``. Instrumented code can write newline-delimited
JSON records with ``os.write`` without depending on logging. Spawned children
that close the descriptor reopen the fallback path.

Example::

    .venv/bin/python benchmarks/shutdown_timing.py \
      --scenario completed --model facebook/opt-125m --gpu 0 \
      --metadata graph_mode=FULL -- \
      .venv/bin/vllm serve facebook/opt-125m --port 8000

The command is never passed through a shell. On timeout, only the verified
process group created for the child can receive the optional forced signal.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MARKER_FD_ENV = "VLLM_SHUTDOWN_MARKER_FD"
MARKER_PATH_ENV = "VLLM_SHUTDOWN_MARKER_PATH"
RUN_ID_ENV = "VLLM_SHUTDOWN_RUN_ID"
SELF_TEST_GRAPH_COUNTS = {
    "legacy_owner": 1,
    "singular_owner": 2,
    "plural_owner": 3,
    "per_owner_record": 4,
}
CONTRACT_MARKERS = (
    "signal_sent",
    "signal_received",
    "work_quiesced",
    "worker_shutdown_begin",
    "graph_enumeration_begin",
    "graph_enumeration_end",
    "graph_reset_begin",
    "graph_reset_end",
    "owner_clear_begin",
    "owner_clear_end",
    "engine_root_detached",
    "gc_collect_begin",
    "gc_collect_end",
    "worker_shutdown_end",
    "child_reaped",
    "vram_below_threshold",
    "vram_stable",
)


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("os.write made no progress")
        offset += written


def _json_line(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parse_assignments(values: Iterable[str], *, option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise ValueError(f"{option} requires KEY=VALUE, got {value!r}")
        if key in result:
            raise ValueError(f"duplicate {option} key: {key}")
        result[key] = item
    return result


def _split_gpu_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        item.strip() for value in values for item in value.split(",") if item.strip()
    )


def _run_capture(command: Sequence[str], *, timeout: float = 10.0) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _run_json(command: Sequence[str], *, timeout: float = 10.0) -> Any | None:
    output = _run_capture(command, timeout=timeout)
    if not output:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_build_metadata() -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "version": _package_version("torch"),
        "git_version": None,
        "hip_version": None,
        "rocm_version": None,
        "cuda_version": None,
    }
    try:
        distribution = importlib.metadata.distribution("torch")
        version_file = Path(distribution.locate_file("torch/version.py"))
        tree = ast.parse(version_file.read_text())
    except (ImportError, OSError, SyntaxError):
        return result

    wanted = {
        "__version__": "version",
        "git_version": "git_version",
        "hip": "hip_version",
        "rocm": "rocm_version",
        "cuda": "cuda_version",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value_node = node.value
        else:
            continue
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError):
            continue
        if value is None or isinstance(value, str):
            result[wanted[target.id]] = value
    return result


def _git_metadata(repo: Path) -> dict[str, Any]:
    sha = _run_capture(("git", "-C", str(repo), "rev-parse", "HEAD"))
    status = _run_capture(("git", "-C", str(repo), "status", "--porcelain"))
    return {
        "sha": sha,
        "dirty": bool(status) if status is not None else None,
        "changed_path_count": len(status.splitlines()) if status else 0,
    }


def _selected_environment() -> dict[str, str]:
    names = (
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_USE_V1",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_USE_ROCM_CUSTOM_PAGED_ATTN",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def _hardware_metadata(gpu_ids: Sequence[str]) -> dict[str, Any]:
    amd_smi = shutil.which("amd-smi")
    if amd_smi:
        selector = ["-g", *gpu_ids] if gpu_ids else []
        return {
            "amd_smi_version": _run_json((amd_smi, "version", "--json")),
            "gpu": _run_json((amd_smi, "static", "-a", *selector, "--json")),
            "firmware": _run_json(
                (amd_smi, "firmware", "-f", *selector, "--json"), timeout=30.0
            ),
        }

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        selector = ["-i", ",".join(gpu_ids)] if gpu_ids else []
        query = "index,name,uuid,pci.bus_id,driver_version,memory.total,compute_cap"
        return {
            "gpu": _run_capture(
                (
                    nvidia_smi,
                    *selector,
                    f"--query-gpu={query}",
                    "--format=csv,noheader",
                )
            )
        }
    return {}


def _rocm_metadata() -> dict[str, Any]:
    version_files: dict[str, str] = {}
    for path in sorted(Path("/opt/rocm/.info").glob("version*")):
        with suppress(OSError):
            version_files[path.name] = path.read_text().strip()
    hipcc = shutil.which("hipcc")
    return {
        "version_files": version_files,
        "hipcc": _run_capture((hipcc, "--version")) if hipcc else None,
    }


class MarkerCollector:
    """Merge child pipe markers and parent markers into one JSONL artifact."""

    def __init__(self, read_fd: int, path: Path, *, run_id: str) -> None:
        self._read_fd = read_fd
        self._output_fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
            0o644,
        )
        self._run_id = run_id
        self._records: list[dict[str, Any]] = []
        self._malformed = 0
        self._closed = False
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stop = threading.Event()
        os.set_blocking(read_fd, False)
        self._thread = threading.Thread(
            target=self._read_child,
            name="shutdown-marker-reader",
            daemon=True,
        )

    @property
    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    @property
    def malformed_count(self) -> int:
        with self._lock:
            return self._malformed

    def start(self) -> None:
        self._thread.start()

    def emit_parent(
        self,
        marker: str,
        *,
        monotonic_ns: int | None = None,
        **fields: Any,
    ) -> int:
        timestamp = monotonic_ns if monotonic_ns is not None else time.monotonic_ns()
        record = {
            "record_type": "shutdown_phase_marker",
            "source": "parent",
            "run_id": self._run_id,
            "pid": os.getpid(),
            "marker": marker,
            "monotonic_ns": timestamp,
            "operation": "shutdown",
            **fields,
        }
        self._store(record)
        return timestamp

    def wait_for(self, marker: str, timeout: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                for record in self._records:
                    if _record_marker(record) == marker:
                        return dict(record)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def merge_records(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Add fallback-file records to the unified marker artifact."""
        for record in records:
            self._store(dict(record))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        with suppress(OSError):
            os.close(self._read_fd)
        with self._condition:
            self._closed = True
            os.close(self._output_fd)

    def _store(self, record: dict[str, Any]) -> None:
        with self._condition:
            if self._closed:
                return
            _write_all(self._output_fd, _json_line(record))
            self._records.append(record)
            self._condition.notify_all()

    def _read_child(self) -> None:
        pending = b""
        while True:
            try:
                chunk = os.read(self._read_fd, 65536)
            except BlockingIOError:
                if self._stop.is_set():
                    break
                self._stop.wait(0.01)
                continue
            except OSError:
                return
            if not chunk:
                break
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                self._decode_child_line(line)
        if pending:
            self._decode_child_line(pending)

    def _decode_child_line(self, line: bytes) -> None:
        received_ns = time.monotonic_ns()
        try:
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise ValueError("marker must be a JSON object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            with self._condition:
                self._malformed += 1
            decoded = {
                "record_type": "malformed_child_marker",
                "text": line.decode(errors="backslashreplace"),
            }
        record = {
            **decoded,
            "source": "child",
            "run_id": decoded.get("run_id", self._run_id),
            "received_monotonic_ns": received_ns,
        }
        self._store(record)


def _record_marker(record: Mapping[str, Any]) -> str | None:
    event = record.get("event")
    if isinstance(event, str) and event in CONTRACT_MARKERS:
        return event
    for key in ("marker", "name"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    phase = record.get("phase")
    if isinstance(phase, str):
        combined = f"{phase}_{event}"
        if isinstance(event, str) and combined in CONTRACT_MARKERS:
            return combined
        return phase
    if isinstance(event, str):
        return event
    return None


class VramReader:
    """Read used VRAM without importing a GPU runtime into the parent."""

    def __init__(self, source: str, gpu_ids: Sequence[str]) -> None:
        self.gpu_ids = tuple(gpu_ids)
        if source == "auto":
            if shutil.which("amd-smi"):
                source = "amd-smi"
            elif shutil.which("rocm-smi"):
                source = "rocm-smi"
            elif shutil.which("nvidia-smi"):
                source = "nvidia-smi"
            else:
                source = "none"
        self.source = source

    def read(self) -> dict[str, float]:
        if self.source == "none":
            return {}
        if self.source == "amd-smi":
            return self._read_amd_smi()
        if self.source == "rocm-smi":
            return self._read_rocm_smi()
        if self.source == "nvidia-smi":
            return self._read_nvidia_smi()
        raise ValueError(f"unsupported VRAM source: {self.source}")

    def _read_amd_smi(self) -> dict[str, float]:
        executable = shutil.which("amd-smi")
        if not executable:
            raise RuntimeError("amd-smi is not installed")
        selector = ["-g", *self.gpu_ids] if self.gpu_ids else []
        payload = _run_json((executable, "metric", "-m", *selector, "--json"))
        if not isinstance(payload, dict):
            raise RuntimeError("amd-smi returned invalid JSON")
        result: dict[str, float] = {}
        for item in payload.get("gpu_data", []):
            gpu = str(item["gpu"])
            used = item["mem_usage"]["used_vram"]
            value = float(used["value"])
            unit = str(used.get("unit", "MB")).upper()
            if unit in {"B", "BYTES"}:
                value /= 1024 * 1024
            elif unit in {"KB", "KIB"}:
                value /= 1024
            elif unit not in {"MB", "MIB"}:
                raise RuntimeError(f"unsupported amd-smi VRAM unit: {unit}")
            result[gpu] = value
        if not result:
            raise RuntimeError("amd-smi returned no GPUs")
        return result

    def _read_rocm_smi(self) -> dict[str, float]:
        executable = shutil.which("rocm-smi")
        if not executable:
            raise RuntimeError("rocm-smi is not installed")
        selector = ["-d", *self.gpu_ids] if self.gpu_ids else []
        payload = _run_json((executable, *selector, "--showmeminfo", "vram", "--json"))
        if not isinstance(payload, dict):
            raise RuntimeError("rocm-smi returned invalid JSON")
        result: dict[str, float] = {}
        for card, item in payload.items():
            used = item.get("VRAM Total Used Memory (B)")
            if used is not None:
                result[str(card).removeprefix("card")] = float(used) / (1024 * 1024)
        if not result:
            raise RuntimeError("rocm-smi returned no GPUs")
        return result

    def _read_nvidia_smi(self) -> dict[str, float]:
        executable = shutil.which("nvidia-smi")
        if not executable:
            raise RuntimeError("nvidia-smi is not installed")
        selector = ["-i", ",".join(self.gpu_ids)] if self.gpu_ids else []
        output = _run_capture(
            (
                executable,
                *selector,
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            )
        )
        if output is None:
            raise RuntimeError("nvidia-smi failed")
        result = {}
        for line in output.splitlines():
            index, used = (part.strip() for part in line.split(",", 1))
            result[index] = float(used)
        if not result:
            raise RuntimeError("nvidia-smi returned no GPUs")
        return result


@dataclass
class VramOutcome:
    source: str
    baseline_mib: dict[str, float]
    threshold_mib: dict[str, float]
    sample_count: int = 0
    error_count: int = 0
    last_error: str | None = None
    below_threshold_ns: int | None = None
    stable_ns: int | None = None
    stable_since_ns: int | None = None


class VramSampler:
    def __init__(
        self,
        reader: VramReader,
        output: Path,
        baseline: Mapping[str, float],
        *,
        allowance_mib: float,
        interval_s: float,
        stable_s: float,
        markers: MarkerCollector,
    ) -> None:
        threshold = {gpu: used + allowance_mib for gpu, used in baseline.items()}
        self.outcome = VramOutcome(reader.source, dict(baseline), threshold)
        self._reader = reader
        self._output = output
        self._interval_s = interval_s
        self._stable_ns = int(stable_s * 1e9)
        self._markers = markers
        self._signal_sent_ns: int | None = None
        self._child_reaped_ns: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="shutdown-vram-sampler",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def note_signal(self, timestamp_ns: int) -> None:
        self._signal_sent_ns = timestamp_ns

    def note_reaped(self, timestamp_ns: int) -> None:
        self._child_reaped_ns = timestamp_ns

    def wait_stable(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while self.outcome.stable_ns is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._stop.wait(min(remaining, 0.05))
        return True

    def stop(self) -> bool:
        self._stop.set()
        self._thread.join(timeout=max(12.0, self._interval_s * 2))
        return not self._thread.is_alive()

    def _run(self) -> None:
        candidate_ns: int | None = None
        output_fd = os.open(
            self._output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
            0o644,
        )
        try:
            while not self._stop.is_set():
                sample_started_ns = time.monotonic_ns()
                try:
                    used = self._reader.read()
                    timestamp_ns = time.monotonic_ns()
                    record: dict[str, Any] = {
                        "monotonic_ns": timestamp_ns,
                        "sample_started_ns": sample_started_ns,
                        "used_mib": used,
                        "source": self._reader.source,
                    }
                    self.outcome.sample_count += 1
                    if self._signal_sent_ns is not None:
                        below = bool(self.outcome.threshold_mib) and all(
                            used.get(gpu, float("inf")) <= threshold
                            for gpu, threshold in self.outcome.threshold_mib.items()
                        )
                        record["below_threshold"] = below
                        if below:
                            if self.outcome.below_threshold_ns is None:
                                self.outcome.below_threshold_ns = timestamp_ns
                                self._markers.emit_parent(
                                    "vram_below_threshold",
                                    monotonic_ns=timestamp_ns,
                                    used_mib=used,
                                )
                            if self._child_reaped_ns is not None:
                                if candidate_ns is None:
                                    candidate_ns = timestamp_ns
                                if (
                                    self.outcome.stable_ns is None
                                    and timestamp_ns - candidate_ns >= self._stable_ns
                                ):
                                    self.outcome.stable_since_ns = candidate_ns
                                    self.outcome.stable_ns = timestamp_ns
                                    self._markers.emit_parent(
                                        "vram_stable",
                                        monotonic_ns=timestamp_ns,
                                        stable_since_ns=candidate_ns,
                                        used_mib=used,
                                    )
                        else:
                            candidate_ns = None
                except Exception as exc:
                    timestamp_ns = time.monotonic_ns()
                    self.outcome.error_count += 1
                    self.outcome.last_error = f"{type(exc).__name__}: {exc}"
                    record = {
                        "monotonic_ns": timestamp_ns,
                        "sample_started_ns": sample_started_ns,
                        "source": self._reader.source,
                        "error": self.outcome.last_error,
                    }
                    candidate_ns = None
                _write_all(output_fd, _json_line(record))
                self._stop.wait(self._interval_s)
        finally:
            os.close(output_fd)


def _calibrate_vram(
    reader: VramReader,
    output: Path,
    *,
    samples: int,
    interval_s: float,
) -> dict[str, float]:
    if reader.source == "none":
        return {}
    baseline: dict[str, float] = {}
    output_fd = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
        0o644,
    )
    try:
        for index in range(samples):
            timestamp_ns = time.monotonic_ns()
            used = reader.read()
            _write_all(
                output_fd,
                _json_line(
                    {
                        "monotonic_ns": timestamp_ns,
                        "source": reader.source,
                        "phase": "baseline",
                        "used_mib": used,
                    }
                ),
            )
            for gpu, value in used.items():
                baseline[gpu] = max(baseline.get(gpu, value), value)
            if index + 1 < samples:
                time.sleep(interval_s)
    finally:
        os.close(output_fd)
    return baseline


@dataclass
class ChildObservation:
    returncode: int | None = None
    reaped_ns: int | None = None


class ChildWaiter:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.observation = ChildObservation()
        self.done = threading.Event()
        self._process = process
        self._thread = threading.Thread(
            target=self._wait,
            name="shutdown-child-waiter",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def wait(self, timeout: float | None) -> bool:
        return self.done.wait(timeout)

    def _wait(self) -> None:
        self.observation.returncode = self._process.wait()
        self.observation.reaped_ns = time.monotonic_ns()
        self.done.set()


def _process_group_snapshot(pgid: int) -> dict[str, Any]:
    members: list[dict[str, int]] = []
    page_size = os.sysconf("SC_PAGE_SIZE")
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        try:
            stat = (proc_path / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            process_group = int(fields[2])
            if process_group != pgid:
                continue
            resident_pages = int((proc_path / "statm").read_text().split()[1])
            members.append(
                {
                    "pid": int(proc_path.name),
                    "ppid": int(fields[1]),
                    "rss_bytes": resident_pages * page_size,
                    "fd_count": len(tuple((proc_path / "fd").iterdir())),
                    "thread_count": len(tuple((proc_path / "task").iterdir())),
                }
            )
        except (IndexError, OSError, ValueError):
            continue
    return {
        "monotonic_ns": time.monotonic_ns(),
        "pgid": pgid,
        "members": sorted(members, key=lambda item: item["pid"]),
    }


class ProcessSampler:
    def __init__(self, pgid: int, output: Path, interval_s: float) -> None:
        self._pgid = pgid
        self._output = output
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="shutdown-process-sampler",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._interval_s * 2))
        return _process_group_snapshot(self._pgid)

    def _run(self) -> None:
        output_fd = os.open(
            self._output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
            0o644,
        )
        try:
            while not self._stop.is_set():
                _write_all(output_fd, _json_line(_process_group_snapshot(self._pgid)))
                self._stop.wait(self._interval_s)
        finally:
            os.close(output_fd)


def _wait_for_health(
    url: str,
    *,
    timeout: float,
    child_done: threading.Event,
    headers: Mapping[str, str],
) -> int:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if child_done.is_set():
            raise RuntimeError("child exited before its health endpoint became ready")
        try:
            request = urllib.request.Request(url, headers=dict(headers))
            with urllib.request.urlopen(request, timeout=min(2.0, timeout)) as response:
                if 200 <= response.status < 300:
                    return time.monotonic_ns()
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise TimeoutError(f"health endpoint did not become ready: {url}: {last_error}")


@dataclass
class RequestOutcome:
    status: int | None = None
    bytes_read: int = 0
    sent_ns: int | None = None
    first_byte_ns: int | None = None
    completed_ns: int | None = None
    error: str | None = None


class RequestWorker:
    def __init__(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        *,
        timeout: float,
        markers: MarkerCollector,
    ) -> None:
        self.outcome = RequestOutcome()
        self.first_byte = threading.Event()
        self.done = threading.Event()
        self._url = url
        self._payload = payload
        self._headers = headers
        self._timeout = timeout
        self._markers = markers
        self._thread = threading.Thread(
            target=self._run,
            name="shutdown-scenario-request",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            data = json.dumps(self._payload).encode()
            headers = {"Content-Type": "application/json", **self._headers}
            request = urllib.request.Request(
                self._url,
                data=data,
                headers=headers,
                method="POST",
            )
            self.outcome.sent_ns = self._markers.emit_parent("request_sent")
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                self.outcome.status = response.status
                first = response.read(1)
                if first:
                    self.outcome.bytes_read = 1
                    self.outcome.first_byte_ns = self._markers.emit_parent(
                        "request_first_byte"
                    )
                    self.first_byte.set()
                while chunk := response.read(65536):
                    self.outcome.bytes_read += len(chunk)
            self.outcome.completed_ns = self._markers.emit_parent("request_completed")
        except Exception as exc:
            self.outcome.error = f"{type(exc).__name__}: {exc}"
            self._markers.emit_parent("request_failed", error=self.outcome.error)
        finally:
            self.done.set()


def _request_payload(args: argparse.Namespace, *, streaming: bool) -> dict[str, Any]:
    if args.request_json:
        raw = args.request_json
        if raw.startswith("@"):
            raw = Path(raw[1:]).read_text()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("--request-json must decode to an object")
    else:
        if not args.model:
            raise ValueError("--model is required for completed/inflight scenarios")
        payload = {
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }
    payload["stream"] = streaming
    return payload


def _auth_headers(args: argparse.Namespace) -> dict[str, str]:
    if not args.api_key_env:
        return {}
    api_key = os.environ.get(args.api_key_env)
    if api_key is None:
        raise ValueError(f"API key environment variable is unset: {args.api_key_env}")
    return {"Authorization": f"Bearer {api_key}"}


def _signal_process_group(
    process: subprocess.Popen[bytes],
    pgid: int,
    signal_number: int,
) -> None:
    if pgid <= 1 or pgid != process.pid:
        raise RuntimeError(f"refusing to signal unsafe process group {pgid}")
    if process.poll() is not None:
        raise ProcessLookupError("child already exited")
    if os.getpgid(process.pid) != pgid:
        raise RuntimeError("child no longer belongs to its dedicated process group")
    os.killpg(pgid, signal_number)


def _signal_child(
    process: subprocess.Popen[bytes],
    signal_number: int,
) -> None:
    """Signal the service process and let it coordinate child shutdown."""
    if process.poll() is not None:
        raise ProcessLookupError("child already exited")
    os.kill(process.pid, signal_number)


def _record_operation(record: Mapping[str, Any]) -> str:
    operation = record.get("operation")
    return operation if isinstance(operation, str) else "shutdown"


def _aggregate_marker_times(
    records: Iterable[Mapping[str, Any]],
    *,
    operation: str = "shutdown",
) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        if _record_operation(record) != operation:
            continue
        marker = _record_marker(record)
        timestamp = record.get("monotonic_ns")
        if marker in CONTRACT_MARKERS and isinstance(timestamp, int):
            use_latest = marker.endswith("_end") or marker in {
                "work_quiesced",
                "engine_root_detached",
                "child_reaped",
                "vram_stable",
            }
            if marker not in result:
                result[marker] = timestamp
            elif use_latest:
                result[marker] = max(result[marker], timestamp)
            else:
                result[marker] = min(result[marker], timestamp)
    return result


def _graph_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        # Accept the names emitted by current and early instrumentation builds.
        # The current plural form wins if a record contains compatibility keys.
        for key in (
            "owner_graph_counts",
            "graph_count_by_owner",
            "graph_counts_by_owner",
        ):
            counts = record.get(key)
            if isinstance(counts, dict):
                for owner, count in counts.items():
                    if isinstance(owner, str) and isinstance(count, int):
                        result[owner] = count
        owner = record.get("owner_type")
        count = record.get("graph_count")
        if isinstance(owner, str) and isinstance(count, int):
            result[owner] = count
    return result


def _operation_summaries(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int | None, str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        pid_value = record.get("pid")
        pid = pid_value if isinstance(pid_value, int) else None
        rank = record.get("rank")
        operation = _record_operation(record)
        key = (pid, json.dumps(rank, sort_keys=True), operation)
        groups.setdefault(key, []).append(record)

    summaries: list[dict[str, Any]] = []
    for (pid, _rank_key, operation), group in groups.items():
        rank = next((record.get("rank") for record in group if "rank" in record), None)
        local_rank = next(
            (record.get("local_rank") for record in group if "local_rank" in record),
            None,
        )
        process_role = next(
            (
                record.get("process_role")
                for record in group
                if "process_role" in record
            ),
            None,
        )
        operation_ids = sorted(
            {
                value
                for record in group
                if isinstance((value := record.get("operation_id")), str)
            }
        )
        owner_counts = _graph_counts(group)
        explicit_graph_counts = [
            count
            for record in group
            if _record_marker(record) == "graph_enumeration_end"
            and isinstance((count := record.get("graph_count")), int)
        ]
        graph_count = (
            max(explicit_graph_counts)
            if explicit_graph_counts
            else sum(owner_counts.values())
        )
        summaries.append(
            {
                "pid": pid,
                "rank": rank,
                "local_rank": local_rank,
                "process_role": process_role,
                "operation": operation,
                "operation_ids": operation_ids,
                "timestamps_ns": _aggregate_marker_times(group, operation=operation),
                "graph_count": graph_count,
                "graph_counts_by_owner": owner_counts,
                "marker_count": len(group),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            item["operation"],
            -1 if item["pid"] is None else item["pid"],
            str(item["rank"]),
        ),
    )


def _shutdown_graph_totals(
    summaries: Iterable[Mapping[str, Any]],
) -> tuple[int, dict[str, int]]:
    graph_count = 0
    owner_counts: Counter[str] = Counter()
    for summary in summaries:
        if summary.get("operation") != "shutdown":
            continue
        count = summary.get("graph_count")
        if isinstance(count, int):
            graph_count += count
        counts = summary.get("graph_counts_by_owner")
        if isinstance(counts, dict):
            owner_counts.update(
                {
                    owner: count
                    for owner, count in counts.items()
                    if isinstance(owner, str) and isinstance(count, int)
                }
            )
    return graph_count, dict(owner_counts)


def _validate_shutdown_contract(
    result: RunResult,
    records: Iterable[Mapping[str, Any]],
    *,
    graph_expected: bool,
) -> list[str]:
    """Validate that a successful parent run observed complete child cleanup."""
    errors: list[str] = []
    shutdown_records = [
        record for record in records if _record_operation(record) == "shutdown"
    ]
    observed = {
        marker
        for record in shutdown_records
        if (marker := _record_marker(record)) is not None
    }
    required = {
        "signal_sent",
        "signal_received",
        "work_quiesced",
        "worker_shutdown_begin",
        "worker_shutdown_end",
        "graph_enumeration_begin",
        "graph_enumeration_end",
        "graph_reset_begin",
        "graph_reset_end",
        "owner_clear_begin",
        "owner_clear_end",
        "engine_root_detached",
        "gc_collect_begin",
        "gc_collect_end",
        "child_reaped",
    }
    missing = sorted(required - observed)
    if missing:
        errors.append("missing shutdown contract markers: " + ", ".join(missing))

    phase_order = (
        "graph_enumeration_begin",
        "graph_enumeration_end",
        "graph_reset_begin",
        "graph_reset_end",
        "owner_clear_begin",
        "owner_clear_end",
    )
    graph_operations: dict[tuple[int | None, str], dict[str, int]] = {}
    for record in shutdown_records:
        marker = _record_marker(record)
        if marker not in phase_order:
            continue
        operation_id = record.get("operation_id")
        operation_key = operation_id if isinstance(operation_id, str) else "<missing>"
        pid = record.get("pid") if isinstance(record.get("pid"), int) else None
        timestamp = record.get("monotonic_ns")
        if not isinstance(timestamp, int):
            errors.append(f"shutdown marker {marker} for pid={pid} has no monotonic_ns")
            continue
        graph_operations.setdefault((pid, operation_key), {})[marker] = timestamp

    for (pid, operation_id), phases in sorted(graph_operations.items()):
        missing_phases = [phase for phase in phase_order if phase not in phases]
        if missing_phases:
            errors.append(
                f"incomplete graph teardown operation pid={pid} "
                f"operation_id={operation_id}: missing {', '.join(missing_phases)}"
            )
            continue
        phase_times = [phases[phase] for phase in phase_order]
        if phase_times != sorted(phase_times):
            errors.append(
                f"out-of-order graph teardown operation pid={pid} "
                f"operation_id={operation_id}"
            )

    for begin, end in (
        ("worker_shutdown_begin", "worker_shutdown_end"),
        ("gc_collect_begin", "gc_collect_end"),
    ):
        begin_pids = {
            record.get("pid")
            for record in shutdown_records
            if _record_marker(record) == begin
        }
        end_pids = {
            record.get("pid")
            for record in shutdown_records
            if _record_marker(record) == end
        }
        if begin_pids != end_pids:
            errors.append(
                f"unpaired {begin}/{end} markers for pids "
                f"{sorted(begin_pids ^ end_pids, key=str)}"
            )

    signal_sent = result.timestamps_ns.get("signal_sent")
    signal_received = result.timestamps_ns.get("signal_received")
    work_quiesced = result.timestamps_ns.get("work_quiesced")
    child_reaped = result.timestamps_ns.get("child_reaped")
    lifecycle = (signal_sent, signal_received, work_quiesced, child_reaped)
    if all(timestamp is not None for timestamp in lifecycle):
        concrete = [int(timestamp) for timestamp in lifecycle]
        if concrete != sorted(concrete):
            errors.append("shutdown lifecycle markers are out of order")

    if graph_expected and result.graph_count_total <= 0:
        errors.append("graph mode requires at least one enumerated shutdown graph")
    if result.malformed_marker_count:
        errors.append(
            f"received {result.malformed_marker_count} malformed shutdown marker(s)"
        )
    if result.process_tree_gone is not True:
        errors.append("child process group still has live members after shutdown")
    return errors


def _read_spawn_markers(path: Path, run_id: str) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    malformed = 0
    for line in path.read_bytes().splitlines():
        try:
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise ValueError("marker must be a JSON object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            malformed += 1
            continue
        decoded.setdefault("source", "spawned_child")
        decoded.setdefault("run_id", run_id)
        records.append(decoded)
    return records, malformed


def _duration_ms(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None or end_ns is None:
        return None
    return (end_ns - start_ns) / 1e6


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@dataclass
class RunResult:
    run_id: str
    artifact_dir: str
    command: list[str]
    scenario: str
    child_pid: int | None = None
    child_pgid: int | None = None
    returncode: int | None = None
    forced_signal: str | None = None
    process_tree_gone: bool | None = None
    timestamps_ns: dict[str, int] = field(default_factory=dict)
    durations_ms: dict[str, float | None] = field(default_factory=dict)
    request: dict[str, Any] | None = None
    vram: dict[str, Any] | None = None
    marker_count: int = 0
    malformed_marker_count: int = 0
    graph_count_total: int = 0
    graph_counts_by_owner: dict[str, int] = field(default_factory=dict)
    operation_summaries: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _create_artifact_dir(root: Path, run_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = root / f"shutdown-{timestamp}-{run_id[:8]}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def run_benchmark(args: argparse.Namespace) -> RunResult:
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("a child command is required after --")

    run_id = str(uuid.uuid4())
    artifact_dir = _create_artifact_dir(Path(args.output_dir), run_id)
    result = RunResult(run_id, str(artifact_dir.resolve()), command, args.scenario)
    gpu_ids = _split_gpu_ids(args.gpu)
    metadata_values = _parse_assignments(args.metadata, option="--metadata")
    child_env_values = _parse_assignments(args.env, option="--env")
    reader = VramReader(args.vram_source, gpu_ids)

    metadata = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "command_cwd": str(Path(args.cwd).resolve()),
        "scenario": args.scenario,
        "model": args.model,
        "request": {
            "url": args.request_url,
            "prompt_length": len(args.prompt),
            "max_tokens": args.max_tokens,
            "custom_json": bool(args.request_json),
        },
        "benchmark_metadata": metadata_values,
        "environment": _selected_environment(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "kernel": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "vllm": {
            "version": _package_version("vllm"),
            **_git_metadata(Path(__file__).resolve().parents[1]),
        },
        "torch": _torch_build_metadata(),
        "rocm": _rocm_metadata(),
        "hardware": {} if args.skip_hardware_metadata else _hardware_metadata(gpu_ids),
        "vram_source": reader.source,
        "gpu_ids": gpu_ids,
    }
    _write_json(artifact_dir / "metadata.json", metadata)

    baseline_path = artifact_dir / "vram_baseline.jsonl"
    try:
        baseline = _calibrate_vram(
            reader,
            baseline_path,
            samples=args.baseline_samples,
            interval_s=args.poll_interval,
        )
    except (Exception, KeyboardInterrupt) as exc:
        result.errors.append(f"VRAM baseline failed: {type(exc).__name__}: {exc}")
        _write_json(artifact_dir / "result.json", asdict(result))
        return result

    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    markers = MarkerCollector(read_fd, artifact_dir / "markers.jsonl", run_id=run_id)
    markers.start()
    spawn_marker_path = artifact_dir / "spawned_child_markers.jsonl"
    spawn_marker_fd = os.open(
        spawn_marker_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
        0o644,
    )
    os.close(spawn_marker_fd)
    child_env = os.environ.copy()
    child_env.update(child_env_values)
    child_env[MARKER_FD_ENV] = str(write_fd)
    child_env[MARKER_PATH_ENV] = str(spawn_marker_path)
    child_env[RUN_ID_ENV] = run_id
    stdout_file = (artifact_dir / "child.stdout.log").open("wb")
    stderr_file = (artifact_dir / "child.stderr.log").open("wb")
    process: subprocess.Popen[bytes] | None = None
    waiter: ChildWaiter | None = None
    process_sampler: ProcessSampler | None = None
    vram_sampler: VramSampler | None = None
    request_worker: RequestWorker | None = None
    lingering: dict[str, Any] | None = None

    try:
        process = subprocess.Popen(
            command,
            cwd=args.cwd,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            pass_fds=(write_fd,),
            start_new_session=True,
        )
        os.close(write_fd)
        write_fd = -1
        pgid = os.getpgid(process.pid)
        if pgid != process.pid:
            raise RuntimeError(
                "child did not enter the requested dedicated process group"
            )
        result.child_pid = process.pid
        result.child_pgid = pgid
        markers.emit_parent("child_started", child_pid=process.pid, child_pgid=pgid)

        waiter = ChildWaiter(process)
        waiter.start()
        process_sampler = ProcessSampler(
            pgid,
            artifact_dir / "process_samples.jsonl",
            args.poll_interval,
        )
        process_sampler.start()
        if reader.source != "none":
            vram_sampler = VramSampler(
                reader,
                artifact_dir / "vram_samples.jsonl",
                baseline,
                allowance_mib=args.vram_allowance_mib,
                interval_s=args.poll_interval,
                stable_s=args.vram_stable_seconds,
                markers=markers,
            )
            vram_sampler.start()

        headers = _auth_headers(args)
        if args.ready_marker:
            ready = markers.wait_for(args.ready_marker, args.startup_timeout)
            if ready is None:
                raise TimeoutError(f"child marker did not arrive: {args.ready_marker}")
        elif args.scenario != "none":
            ready_ns = _wait_for_health(
                args.health_url,
                timeout=args.startup_timeout,
                child_done=waiter.done,
                headers=headers,
            )
            markers.emit_parent("service_ready", monotonic_ns=ready_ns)
        elif args.startup_delay:
            if waiter.wait(args.startup_delay):
                raise RuntimeError("child exited during startup delay")

        if args.scenario in {"completed", "inflight"}:
            streaming = args.scenario == "inflight"
            payload = _request_payload(args, streaming=streaming)
            request_worker = RequestWorker(
                args.request_url,
                payload,
                headers,
                timeout=args.request_timeout,
                markers=markers,
            )
            request_worker.start()
            if streaming:
                deadline = time.monotonic() + args.request_timeout
                while not request_worker.first_byte.is_set():
                    if request_worker.done.is_set():
                        raise RuntimeError(
                            "in-flight request ended before shutdown signal: "
                            f"{request_worker.outcome.error or 'response completed'}"
                        )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "in-flight request produced no response byte"
                        )
                    request_worker.first_byte.wait(min(remaining, 0.05))
            else:
                if not request_worker.done.wait(args.request_timeout + 1):
                    raise TimeoutError("completed-scenario request did not finish")
                if request_worker.outcome.error:
                    raise RuntimeError(request_worker.outcome.error)

        if args.pre_signal_delay:
            time.sleep(args.pre_signal_delay)
        signal_number = getattr(signal, f"SIG{args.signal}")
        signal_sent_ns = time.monotonic_ns()
        # A graceful service signal belongs to the top-level process. It then
        # coordinates EngineCore and worker shutdown through their existing
        # process managers. Keep process-group signaling for bounded error
        # cleanup and SIGKILL escalation only.
        _signal_child(process, signal_number)
        result.timestamps_ns["signal_sent"] = signal_sent_ns
        markers.emit_parent(
            "signal_sent",
            monotonic_ns=signal_sent_ns,
            signal=f"SIG{args.signal}",
            target_pid=process.pid,
        )
        if vram_sampler:
            vram_sampler.note_signal(signal_sent_ns)

        if not waiter.wait(args.shutdown_timeout):
            markers.emit_parent("shutdown_timeout", timeout_s=args.shutdown_timeout)
            if args.force_kill:
                _signal_process_group(process, pgid, signal.SIGKILL)
                result.forced_signal = "SIGKILL"
                markers.emit_parent("forced_signal_sent", signal="SIGKILL")
                if not waiter.wait(args.force_kill_timeout):
                    raise TimeoutError("child was not reaped after scoped SIGKILL")
            else:
                raise TimeoutError(
                    "child exceeded shutdown timeout; --force-kill was not enabled"
                )

        observation = waiter.observation
        result.returncode = observation.returncode
        if observation.reaped_ns is None:
            raise RuntimeError("waiter completed without a reap timestamp")
        result.timestamps_ns["child_reaped"] = observation.reaped_ns
        markers.emit_parent(
            "child_reaped",
            monotonic_ns=observation.reaped_ns,
            returncode=observation.returncode,
        )
        if vram_sampler:
            vram_sampler.note_reaped(observation.reaped_ns)
            if not vram_sampler.wait_stable(args.vram_timeout):
                result.errors.append(
                    "VRAM did not remain below its calibrated threshold within "
                    f"{args.vram_timeout:g}s after child exit"
                )
    except (Exception, KeyboardInterrupt) as exc:
        result.errors.append(f"{type(exc).__name__}: {exc}")
        should_clean = (
            process is not None
            and process.poll() is None
            and result.child_pgid is not None
        )
        if should_clean:
            try:
                _signal_process_group(process, result.child_pgid, signal.SIGTERM)
                markers.emit_parent(
                    "cleanup_signal_sent", signal="SIGTERM", reason="error"
                )
                reaped = waiter is not None and waiter.wait(args.force_kill_timeout)
                if not reaped and args.force_kill:
                    _signal_process_group(process, result.child_pgid, signal.SIGKILL)
                    result.forced_signal = "SIGKILL"
                    markers.emit_parent(
                        "forced_signal_sent", signal="SIGKILL", reason="error"
                    )
                    if waiter is not None:
                        waiter.wait(args.force_kill_timeout)
                elif not reaped:
                    result.errors.append(
                        "scoped child remained alive after cleanup SIGTERM; "
                        "SIGKILL disabled by --no-force-kill"
                    )
            except (OSError, RuntimeError) as cleanup_exc:
                result.errors.append(
                    "scoped child cleanup failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        if request_worker is not None:
            request_worker.done.wait(1.0)
            result.request = asdict(request_worker.outcome)
        if waiter is not None and waiter.done.is_set():
            result.returncode = waiter.observation.returncode
            if (
                waiter.observation.reaped_ns is not None
                and "child_reaped" not in result.timestamps_ns
            ):
                result.timestamps_ns["child_reaped"] = waiter.observation.reaped_ns
                markers.emit_parent(
                    "child_reaped",
                    monotonic_ns=waiter.observation.reaped_ns,
                    returncode=waiter.observation.returncode,
                )
        if vram_sampler is not None:
            if not vram_sampler.stop():
                result.errors.append("VRAM sampler did not stop cleanly")
            result.vram = asdict(vram_sampler.outcome)
            if vram_sampler.outcome.below_threshold_ns is not None:
                result.timestamps_ns["vram_below_threshold"] = (
                    vram_sampler.outcome.below_threshold_ns
                )
            if vram_sampler.outcome.stable_ns is not None:
                result.timestamps_ns["vram_stable"] = vram_sampler.outcome.stable_ns
        else:
            result.vram = asdict(
                VramOutcome(reader.source, dict(baseline), dict(baseline))
            )
        if process_sampler is not None:
            lingering = process_sampler.stop()
            result.process_tree_gone = not lingering["members"]
            _write_json(artifact_dir / "final_process_group.json", lingering)
        stdout_file.close()
        stderr_file.close()
        time.sleep(0.02)
        spawn_records, spawn_malformed = _read_spawn_markers(spawn_marker_path, run_id)
        markers.merge_records(spawn_records)
        markers.close()
        records = markers.records
        result.marker_count = len(records)
        result.malformed_marker_count = markers.malformed_count + spawn_malformed
        shutdown_times = _aggregate_marker_times(records)
        result.operation_summaries = _operation_summaries(records)
        (
            result.graph_count_total,
            result.graph_counts_by_owner,
        ) = _shutdown_graph_totals(result.operation_summaries)
        for name, timestamp in shutdown_times.items():
            result.timestamps_ns.setdefault(name, timestamp)
        graph_mode = metadata_values.get(
            "graph_mode", metadata_values.get("cudagraph_mode", "")
        )
        graph_expected = str(graph_mode).strip().upper() not in {
            "",
            "0",
            "EAGER",
            "FALSE",
            "NONE",
            "OFF",
        }
        result.errors.extend(
            _validate_shutdown_contract(
                result,
                records,
                graph_expected=graph_expected,
            )
        )

    signal_sent = result.timestamps_ns.get("signal_sent")
    child_reaped = result.timestamps_ns.get("child_reaped")
    result.durations_ms = {
        "signal_to_child_reaped": _duration_ms(signal_sent, child_reaped),
        "signal_to_vram_below_threshold": _duration_ms(
            signal_sent, result.timestamps_ns.get("vram_below_threshold")
        ),
        "signal_to_vram_stable": _duration_ms(
            signal_sent, result.timestamps_ns.get("vram_stable")
        ),
        "child_reaped_to_vram_stable": _duration_ms(
            child_reaped, result.timestamps_ns.get("vram_stable")
        ),
        "worker_shutdown_end_to_child_reaped": _duration_ms(
            result.timestamps_ns.get("worker_shutdown_end"), child_reaped
        ),
    }
    _write_json(artifact_dir / "result.json", asdict(result))
    return result


def _emit_self_test_marker(marker: str, **fields: Any) -> None:
    fd = int(os.environ[MARKER_FD_ENV])
    record = {
        "record_type": "shutdown_phase_marker",
        "run_id": os.environ.get(RUN_ID_ENV),
        "pid": os.getpid(),
        "marker": marker,
        "monotonic_ns": time.monotonic_ns(),
        "operation": "shutdown",
        **fields,
    }
    _write_all(fd, _json_line(record))


def _self_test_path_child() -> int:
    marker_fd_valid = False
    try:
        os.fstat(int(os.environ[MARKER_FD_ENV]))
    except (OSError, OverflowError, ValueError):
        pass
    else:
        marker_fd_valid = True

    from vllm.utils.shutdown_markers import emit_shutdown_marker

    emit_shutdown_marker(
        "graph_reset_begin",
        operation="reset_for_reuse",
        operation_id="self-test-startup-reset",
    )
    emit_shutdown_marker(
        "spawn_path_fallback",
        operation_id="self-test-path-fallback",
        marker_fd_valid=marker_fd_valid,
        owner_graph_counts={"legacy_owner": 1},
        graph_count_by_owner={"singular_owner": 2},
        graph_counts_by_owner={"plural_owner": 3},
        owner_type="per_owner_record",
        graph_count=4,
    )
    return 0


def _self_test_child() -> int:
    shutdown = threading.Event()

    def receive_signal(signal_number: int, _frame: Any) -> None:
        _emit_self_test_marker(
            "signal_received", signal=signal.Signals(signal_number).name
        )
        shutdown.set()

    signal.signal(signal.SIGTERM, receive_signal)
    signal.signal(signal.SIGINT, receive_signal)

    spawn_env = os.environ.copy()
    spawn_env[MARKER_FD_ENV] = "-1"
    spawn_env["RANK"] = "7"
    spawn_env["LOCAL_RANK"] = "2"
    spawned = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--_self-test-path-child"],
        check=False,
        close_fds=True,
        env=spawn_env,
        stdin=subprocess.DEVNULL,
    )
    if spawned.returncode != 0:
        raise RuntimeError(
            f"path-fallback self-test child exited with {spawned.returncode}"
        )

    _emit_self_test_marker("child_ready")
    shutdown.wait()
    for marker in (
        "work_quiesced",
        "worker_shutdown_begin",
        "graph_enumeration_begin",
        "graph_enumeration_end",
        "graph_reset_begin",
        "graph_reset_end",
        "owner_clear_begin",
        "owner_clear_end",
        "engine_root_detached",
        "gc_collect_begin",
        "gc_collect_end",
        "worker_shutdown_end",
    ):
        _emit_self_test_marker(marker)
    return 0


def _self_test_second_term_child() -> int:
    shutdown = threading.Event()
    term_count = 0

    def receive_signal(signal_number: int, _frame: Any) -> None:
        nonlocal term_count
        term_count += 1
        _emit_self_test_marker(
            "signal_received",
            signal=signal.Signals(signal_number).name,
            signal_count=term_count,
        )
        if term_count >= 2:
            shutdown.set()

    signal.signal(signal.SIGTERM, receive_signal)
    _emit_self_test_marker("child_ready")
    shutdown.wait()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a child in a dedicated process group and measure shutdown "
            "and VRAM recovery from the parent."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a GPU-free child/marker/process-group smoke test",
    )
    parser.add_argument(
        "--output-dir",
        default="shutdown-artifacts",
        help="directory under which a unique run directory is created",
    )
    parser.add_argument(
        "--scenario",
        choices=("none", "idle", "completed", "inflight"),
        default="idle",
    )
    parser.add_argument("--cwd", default=".", help="working directory for the child")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment entry added to the child; may be repeated",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="topology/executor/graph-mode metadata; may be repeated",
    )
    parser.add_argument("--health-url", default="http://127.0.0.1:8000/health")
    parser.add_argument(
        "--ready-marker",
        help="wait for this dedicated-pipe marker instead of HTTP health",
    )
    parser.add_argument("--startup-timeout", type=_positive_float, default=600.0)
    parser.add_argument(
        "--startup-delay",
        type=_nonnegative_float,
        default=0.0,
        help="delay before signaling when scenario=none and no ready marker is used",
    )
    parser.add_argument("--pre-signal-delay", type=_nonnegative_float, default=0.0)
    parser.add_argument("--request-url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--request-json", help="JSON object or @path to a JSON file")
    parser.add_argument("--model", help="served model name for generated request JSON")
    parser.add_argument(
        "--prompt", default="Write one sentence about deterministic cleanup."
    )
    parser.add_argument("--max-tokens", type=_positive_int, default=4096)
    parser.add_argument("--request-timeout", type=_positive_float, default=300.0)
    parser.add_argument(
        "--api-key-env",
        help="name of an environment variable containing the bearer token",
    )
    parser.add_argument("--signal", choices=("TERM", "INT"), default="TERM")
    parser.add_argument("--shutdown-timeout", type=_positive_float, default=15.0)
    parser.add_argument(
        "--force-kill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "send SIGKILL only to the verified dedicated group after shutdown "
            "timeout (default: enabled)"
        ),
    )
    parser.add_argument("--force-kill-timeout", type=_positive_float, default=5.0)
    parser.add_argument(
        "--vram-source",
        choices=("auto", "amd-smi", "rocm-smi", "nvidia-smi", "none"),
        default="auto",
    )
    parser.add_argument(
        "--gpu",
        action="append",
        default=[],
        help=(
            "physical GPU index/BDF/UUID to poll; comma lists and repeats are accepted"
        ),
    )
    parser.add_argument("--baseline-samples", type=_positive_int, default=5)
    parser.add_argument("--poll-interval", type=_positive_float, default=0.1)
    parser.add_argument("--vram-allowance-mib", type=_nonnegative_float, default=256.0)
    parser.add_argument("--vram-stable-seconds", type=_nonnegative_float, default=2.0)
    parser.add_argument("--vram-timeout", type=_positive_float, default=5.0)
    parser.add_argument(
        "--skip-hardware-metadata",
        action="store_true",
        help="skip amd-smi/nvidia-smi static and firmware queries",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="child argv after --")
    return parser


def _run_self_test(args: argparse.Namespace) -> int:
    synthetic_times = _aggregate_marker_times(
        (
            {
                "marker": "graph_reset_begin",
                "monotonic_ns": 30,
                "operation": "shutdown",
            },
            {
                "marker": "graph_reset_begin",
                "monotonic_ns": 20,
                "operation": "shutdown",
            },
            {
                "marker": "graph_reset_end",
                "monotonic_ns": 40,
                "operation": "shutdown",
            },
            {
                "marker": "graph_reset_end",
                "monotonic_ns": 50,
                "operation": "shutdown",
            },
            {
                "marker": "graph_reset_begin",
                "monotonic_ns": 1,
                "operation": "reset_for_reuse",
            },
            {
                "marker": "graph_reset_end",
                "monotonic_ns": 100,
                "operation": "reset_for_reuse",
            },
        )
    )
    if synthetic_times != {"graph_reset_begin": 20, "graph_reset_end": 50}:
        raise AssertionError(synthetic_times)

    with tempfile.TemporaryDirectory(prefix="vllm-shutdown-self-test-") as directory:
        args.output_dir = directory
        args.scenario = "none"
        args.ready_marker = "child_ready"
        args.startup_timeout = 30.0
        args.shutdown_timeout = 5.0
        args.force_kill = False
        args.vram_source = "none"
        args.skip_hardware_metadata = True
        args.command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_self-test-child",
        ]
        result = run_benchmark(args)
        signal_sent = result.timestamps_ns.get("signal_sent")
        signal_received = result.timestamps_ns.get("signal_received")
        child_reaped = result.timestamps_ns.get("child_reaped")
        if result.errors:
            raise AssertionError(result.errors)
        if result.returncode != 0 or result.forced_signal is not None:
            raise AssertionError(asdict(result))
        if not (
            signal_sent is not None
            and signal_received is not None
            and child_reaped is not None
            and signal_sent <= signal_received <= child_reaped
        ):
            raise AssertionError(result.timestamps_ns)
        if result.process_tree_gone is not True:
            raise AssertionError("self-test process tree remained alive")
        if result.graph_counts_by_owner != SELF_TEST_GRAPH_COUNTS:
            raise AssertionError(result.graph_counts_by_owner)
        if result.graph_count_total != sum(SELF_TEST_GRAPH_COUNTS.values()):
            raise AssertionError(result.graph_count_total)
        graph_reset_begin = result.timestamps_ns.get("graph_reset_begin")
        if graph_reset_begin is None or graph_reset_begin < signal_received:
            raise AssertionError(result.timestamps_ns)

        artifact_dir = Path(result.artifact_dir)
        path_records, malformed = _read_spawn_markers(
            artifact_dir / "spawned_child_markers.jsonl", result.run_id
        )
        fallback_records = [
            record
            for record in path_records
            if _record_marker(record) == "spawn_path_fallback"
        ]
        if malformed or len(fallback_records) != 1:
            raise AssertionError(path_records)
        fallback_record = fallback_records[0]
        if fallback_record.get("marker_fd_valid") is not False:
            raise AssertionError(fallback_record)
        if (
            fallback_record.get("rank") != 7
            or fallback_record.get("local_rank") != 2
            or fallback_record.get("operation") != "shutdown"
            or fallback_record.get("run_id") != result.run_id
        ):
            raise AssertionError(fallback_record)

        reset_summaries = [
            summary
            for summary in result.operation_summaries
            if summary["operation"] == "reset_for_reuse"
        ]
        fallback_summaries = [
            summary
            for summary in result.operation_summaries
            if summary["operation"] == "shutdown" and summary["rank"] == 7
        ]
        if len(reset_summaries) != 1 or len(fallback_summaries) != 1:
            raise AssertionError(result.operation_summaries)
        if fallback_summaries[0]["graph_counts_by_owner"] != SELF_TEST_GRAPH_COUNTS:
            raise AssertionError(fallback_summaries[0])

        merged_records = [
            json.loads(line)
            for line in (artifact_dir / "markers.jsonl").read_text().splitlines()
        ]
        merged_fallback = [
            record
            for record in merged_records
            if _record_marker(record) == "spawn_path_fallback"
        ]
        if len(merged_fallback) != 1:
            raise AssertionError(merged_records)
        if result.marker_count != len(merged_records):
            raise AssertionError((result.marker_count, len(merged_records)))

        cleanup_args = argparse.Namespace(**vars(args))
        cleanup_args.shutdown_timeout = 0.1
        cleanup_args.force_kill_timeout = 2.0
        cleanup_args.command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_self-test-second-term-child",
        ]
        cleanup_result = run_benchmark(cleanup_args)
        if (
            cleanup_result.returncode != 0
            or cleanup_result.forced_signal is not None
            or cleanup_result.process_tree_gone is not True
        ):
            raise AssertionError(asdict(cleanup_result))
        if not any(
            "--force-kill was not enabled" in error for error in cleanup_result.errors
        ):
            raise AssertionError(cleanup_result.errors)
        cleanup_records = [
            json.loads(line)
            for line in (Path(cleanup_result.artifact_dir) / "markers.jsonl")
            .read_text()
            .splitlines()
        ]
        cleanup_markers = {_record_marker(record) for record in cleanup_records}
        if "cleanup_signal_sent" not in cleanup_markers:
            raise AssertionError(cleanup_records)
        if "forced_signal_sent" in cleanup_markers:
            raise AssertionError(cleanup_records)
        print("shutdown_timing self-test passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--_self-test-child"]:
        return _self_test_child()
    if argv == ["--_self-test-path-child"]:
        return _self_test_path_child()
    if argv == ["--_self-test-second-term-child"]:
        return _self_test_second_term_child()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return _run_self_test(args)
    try:
        result = run_benchmark(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(result), sort_keys=True))
    return int(bool(result.errors or result.returncode != 0 or result.forced_signal))


if __name__ == "__main__":
    raise SystemExit(main())
