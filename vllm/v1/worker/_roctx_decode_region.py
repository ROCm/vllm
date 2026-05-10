# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional roctx profile-region brackets around the decode forward pass.

Used to unblock single-pass PMC under cudagraph + torch.compile (campaign
idea P-5). Behaviour is gated entirely behind environment variables; if the
master env knob is unset, the helpers are no-ops and no library is loaded.

Env knobs:
  VLLM_ROCTX_DECODE_REGION   '1' / 'true' to enable. Default: off.
  VLLM_ROCTX_WARMUP_STEPS    Number of execute_model() calls to skip before
                             beginning to push roctx ranges. Default: 8
                             (covers warmup + first cudagraph replays).
  VLLM_ROCTX_CAPTURE_STEPS   Number of subsequent execute_model() calls to
                             wrap with roctxProfilerResume / Pause. Default: 4.
                             After this count, profile-region stays paused.
  VLLM_ROCTX_LIB             Override the path to librocprofiler-sdk-roctx.so.
                             Default: searched under .venv site-packages.

Use with:
  rocprofv3 --marker-trace --selected-regions \
            --pmc <single-group counters> -- <your vllm-bench cmd>

`--selected-regions` makes rocprofv3 only sample between the
roctxProfilerResume(0) and roctxProfilerPause(0) calls this module emits.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import threading

_lock = threading.Lock()
_initialised = False
_lib = None  # type: ignore[var-annotated]
_step_count = 0
_active = False
_warmup_steps = 8
_capture_steps = 4
_enabled = False


def _truthy(v: str | None) -> bool:
    return v is not None and v.lower() in ("1", "true", "yes", "on")


def _find_libroctx() -> str | None:
    """Locate librocprofiler-sdk-roctx.so.1 in common locations."""
    override = os.environ.get("VLLM_ROCTX_LIB")
    if override:
        return override if os.path.exists(override) else None

    # Search the active venv site-packages first (rocm-sdk wheel).
    candidates = [
        "/scratch/mgehre/vllm/.venv/lib/python3.12/site-packages/"
        "_rocm_sdk_core/lib/librocprofiler-sdk-roctx.so.1",
        "/opt/rocm/lib/librocprofiler-sdk-roctx.so.1",
        "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
    ]
    # Also try the standard SONAME for ld lookup.
    candidates.append("librocprofiler-sdk-roctx.so.1")

    for path in candidates:
        try:
            if "/" in path and not os.path.exists(path):
                continue
            return path
        except OSError:
            continue
    return None


def _init_once() -> bool:
    global _initialised, _lib, _enabled, _warmup_steps, _capture_steps
    if _initialised:
        return _lib is not None
    with _lock:
        if _initialised:
            return _lib is not None
        _initialised = True
        if not _truthy(os.environ.get("VLLM_ROCTX_DECODE_REGION")):
            return False
        try:
            _warmup_steps = int(os.environ.get("VLLM_ROCTX_WARMUP_STEPS", "8"))
        except ValueError:
            _warmup_steps = 8
        try:
            _capture_steps = int(os.environ.get("VLLM_ROCTX_CAPTURE_STEPS", "4"))
        except ValueError:
            _capture_steps = 4
        path = _find_libroctx()
        if path is None:
            print(
                "[roctx_decode_region] librocprofiler-sdk-roctx.so not "
                "found; profile-region disabled.",
                flush=True,
            )
            return False
        try:
            _lib = ctypes.CDLL(path)
            # Argument types (uint32_t correlation id is unused; pass 0).
            _lib.roctxProfilerResume.argtypes = [ctypes.c_uint64]
            _lib.roctxProfilerResume.restype = ctypes.c_int
            _lib.roctxProfilerPause.argtypes = [ctypes.c_uint64]
            _lib.roctxProfilerPause.restype = ctypes.c_int
            _lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
            _lib.roctxRangePushA.restype = ctypes.c_int
            _lib.roctxRangePop.argtypes = []
            _lib.roctxRangePop.restype = ctypes.c_int
        except OSError as exc:
            print(
                f"[roctx_decode_region] failed to load {path}: {exc}; "
                "profile-region disabled.",
                flush=True,
            )
            _lib = None
            return False
        _enabled = True
        # NOTE: do NOT call roctxProfilerPause here. rocprofv3
        # --selected-regions already starts in the paused state; an extra
        # pause unbalances the resume/pause refcount and causes
        # rocprofiler_stop_context(...) to fail at finalize with
        # ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND.
        print(
            f"[roctx_decode_region] enabled "
            f"(warmup={_warmup_steps} capture={_capture_steps} lib={path})",
            flush=True,
        )
        return True


def begin_decode_step() -> None:
    """Call at the very start of the decode forward pass."""
    global _step_count, _active
    if not _initialised and not _init_once():
        return
    if _lib is None or not _enabled:
        return
    _step_count += 1
    if _step_count <= _warmup_steps:
        return
    if _step_count > _warmup_steps + _capture_steps:
        return
    # Resume the profile region for this step.
    _active = False
    with contextlib.suppress(Exception):
        _lib.roctxProfilerResume(0)
        _lib.roctxRangePushA(f"vllm_decode_step_{_step_count}".encode())
        _active = True


def end_decode_step() -> None:
    """Call at the very end of the decode forward pass."""
    global _active
    if _lib is None or not _enabled or not _active:
        return
    with contextlib.suppress(Exception):
        _lib.roctxRangePop()
        _lib.roctxProfilerPause(0)
    _active = False
