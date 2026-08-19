# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

import pytest

from vllm.utils import shutdown_markers


@pytest.fixture(autouse=True)
def reset_marker_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(shutdown_markers, "_MARKER_CONTEXT", {})
    monkeypatch.setattr(shutdown_markers, "_MARKER_FD", None)


def test_emit_marker_is_noop_when_disabled(monkeypatch: pytest.MonkeyPatch):
    writes: list[tuple[int, bytes]] = []
    monkeypatch.setattr(os, "write", lambda fd, data: writes.append((fd, data)))

    shutdown_markers.emit_shutdown_marker("graph_reset_begin")

    assert writes == []


def test_emit_marker_writes_context_and_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    read_fd, write_fd = os.pipe()
    try:
        monkeypatch.setattr(shutdown_markers, "_MARKER_FD", write_fd)
        monkeypatch.setenv(shutdown_markers.RUN_ID_ENV, "run-7")
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("LOCAL_RANK", "1")
        shutdown_markers.set_shutdown_marker_context(
            process_role="gpu_worker",
            dp_rank=2,
            omitted=None,
        )

        shutdown_markers.emit_shutdown_marker(
            "graph_reset_end",
            operation="reset_for_reuse",
            graph_count=4,
        )

        record = json.loads(os.read(read_fd, 4096))
        assert record["record_type"] == "vllm_shutdown_marker"
        assert record["event"] == "graph_reset_end"
        assert record["operation"] == "reset_for_reuse"
        assert record["graph_count"] == 4
        assert record["rank"] == 3
        assert record["local_rank"] == 1
        assert record["dp_rank"] == 2
        assert record["process_role"] == "gpu_worker"
        assert record["run_id"] == "run-7"
        assert "omitted" not in record
        assert isinstance(record["monotonic_ns"], int)
    finally:
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.parametrize("error", [OSError("closed"), TypeError("bad value")])
def test_emit_marker_never_breaks_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
):
    monkeypatch.setattr(shutdown_markers, "_MARKER_FD", 42)

    def fail_write(_fd: int, _payload: bytes) -> None:
        raise error

    monkeypatch.setattr(os, "write", fail_write)
    shutdown_markers.emit_shutdown_marker("worker_shutdown_end")


def test_marker_fd_falls_back_to_append_only_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    path = tmp_path / "markers.jsonl"
    monkeypatch.setenv(shutdown_markers.MARKER_FD_ENV, "not-an-fd")
    monkeypatch.setenv(shutdown_markers.MARKER_PATH_ENV, str(path))

    fd = shutdown_markers._marker_fd()
    assert fd is not None
    try:
        os.write(fd, b"one\n")
    finally:
        os.close(fd)

    monkeypatch.delenv(shutdown_markers.MARKER_FD_ENV)
    fd = shutdown_markers._marker_fd()
    assert fd is not None
    try:
        os.write(fd, b"two\n")
    finally:
        os.close(fd)

    assert path.read_text() == "one\ntwo\n"


def test_environment_context_ignores_non_integer_rank(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("RANK", "worker")
    monkeypatch.setenv("VLLM_DP_RANK", "5")

    assert shutdown_markers._environment_context() == {"dp_rank": 5}
