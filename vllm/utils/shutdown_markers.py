# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional low-level shutdown phase markers for lifecycle benchmarks."""

import json
import os
import time
from typing import Any

MARKER_FD_ENV = "VLLM_SHUTDOWN_MARKER_FD"
MARKER_PATH_ENV = "VLLM_SHUTDOWN_MARKER_PATH"
RUN_ID_ENV = "VLLM_SHUTDOWN_RUN_ID"

_MARKER_CONTEXT: dict[str, Any] = {}


def _marker_fd() -> int | None:
    value = os.environ.get(MARKER_FD_ENV)
    if value is not None:
        try:
            fd = int(value)
            if fd < 0:
                raise ValueError("marker descriptor must be nonnegative")
            os.fstat(fd)
        except (OSError, OverflowError, ValueError):
            pass
        else:
            return fd

    path = os.environ.get(MARKER_PATH_ENV)
    if path is None:
        return None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0)
        return os.open(path, flags, 0o644)
    except OSError:
        return None


_MARKER_FD = _marker_fd()


def set_shutdown_marker_context(**fields: Any) -> None:
    """Attach stable process metadata such as worker rank to every marker."""
    global _MARKER_CONTEXT
    _MARKER_CONTEXT = {
        **_MARKER_CONTEXT,
        **{key: value for key, value in fields.items() if value is not None},
    }


def _environment_context() -> dict[str, int]:
    context: dict[str, int] = {}
    for field, names in (
        ("rank", ("RANK",)),
        ("local_rank", ("LOCAL_RANK",)),
        ("dp_rank", ("VLLM_DP_RANK",)),
        ("dp_local_rank", ("VLLM_DP_RANK_LOCAL",)),
    ):
        for name in names:
            value = os.environ.get(name)
            if value is None:
                continue
            try:
                context[field] = int(value)
            except ValueError:
                break
            break
    return context


def emit_shutdown_marker(
    event: str,
    *,
    operation: str = "shutdown",
    **fields: Any,
) -> None:
    """Write one compact JSON record without using logging or buffered I/O."""
    if _MARKER_FD is None:
        return
    record = {
        **_environment_context(),
        **_MARKER_CONTEXT,
        "record_type": "vllm_shutdown_marker",
        "event": event,
        "pid": os.getpid(),
        "monotonic_ns": time.monotonic_ns(),
        "operation": operation,
        **fields,
    }
    run_id = os.environ.get(RUN_ID_ENV)
    if run_id is not None:
        record["run_id"] = run_id
    try:
        payload = (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        os.write(_MARKER_FD, payload)
    except (OSError, TypeError, ValueError):
        # Diagnostics must never make shutdown less reliable.
        return
