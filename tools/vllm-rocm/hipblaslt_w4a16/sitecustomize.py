# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Make the process use a w4a16-capable hipBLASLt instead of the ROCm SDK's.

Put this directory on PYTHONPATH (see hipblaslt_w4a16_env.sh) when running with
VLLM_ROCM_W4A16_HIPBLASLT enabled. Python imports ``sitecustomize`` at
interpreter start-up, which is the only point early enough: ``import torch``
preloads the SDK's ``libhipblaslt.so.1`` with RTLD_GLOBAL, and from then on any
library with that SONAME resolves to it whatever RPATH or LD_LIBRARY_PATH say.

Seeding ``rocm_sdk._ALL_CDLLS`` uses the SDK's own "already preloaded" check, so
torch skips its copy and the process holds exactly one hipBLASLt. That matters
for more than the scale-mode enums: hipBLASLt pulls in liborigami and
libtensilelite-host, and two builds of those cannot coexist in one process.

No-op unless VLLM_HIPBLASLT_W4A16_ROOT is set, so it is safe to leave on
PYTHONPATH.
"""

import os


def _preload_w4a16_hipblaslt() -> None:
    root = os.environ.get("VLLM_HIPBLASLT_W4A16_ROOT")
    if not root:
        return

    lib = os.path.join(
        root, "build", "projects", "hipblaslt", "library", "libhipblaslt.so"
    )
    if not os.path.exists(lib):
        raise RuntimeError(
            f"VLLM_HIPBLASLT_W4A16_ROOT={root} has no built hipBLASLt at {lib}"
        )

    import ctypes

    import rocm_sdk

    rocm_sdk._ALL_CDLLS["hipblaslt"] = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)


_preload_w4a16_hipblaslt()
