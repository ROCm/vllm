#!/usr/bin/env bash
# Run a command with VLLM_ROCM_W4A16_HIPBLASLT's hipBLASLt in place of the one
# PyTorch ships.
#
#   VLLM_HIPBLASLT_W4A16_ROOT=<rocm-libraries checkout> \
#   VLLM_ROCM_W4A16_HIPBLASLT=all \
#     tools/vllm-rocm/hipblaslt_w4a16/hipblaslt_w4a16_env.sh python -m pytest ...
#
# Two load-order problems have to be arranged; neither is about the w4a16 API:
#
#  - sitecustomize.py (via PYTHONPATH) makes torch preload the checkout's
#    libhipblaslt, so the process never holds two of them. Prepended, because
#    .envrc already puts the SDK's amd_smi on PYTHONPATH and vLLM's ROCm
#    platform detection needs it.
#  - LD_LIBRARY_PATH pins the HIP runtime and friends to the SDK that torch was
#    built against. The checkout's RUNPATH points at whichever SDK *it* was
#    built against, and letting that one load a second libamdhip64 would give
#    the process two HIP runtimes with unshared streams. The one library the
#    SDK does not ship, libtensilelite-host, still resolves via that RUNPATH.
#
# This assumes the checkout and this venv are on a compatible ROCm. They share
# liborigami/librocroller SONAMEs, and those are not ABI-stable across ROCm
# major versions -- origami's get_hardware_for_device gained an argument
# between 7.15 and 10.1. If you see an undefined-symbol error naming origami,
# the two are too far apart to share a process; align them rather than
# reordering paths here.
set -euo pipefail

: "${VLLM_HIPBLASLT_W4A16_ROOT:?set it to a built rocm-libraries checkout}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

python=${PYTHON:-python}
sdk=$("$python" - <<'PY'
import pathlib, rocm_sdk
print(pathlib.Path(rocm_sdk.__file__).parent.parent)
PY
)

export PYTHONPATH=$here${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=\
$sdk/_rocm_sdk_devel/lib:\
$sdk/_rocm_sdk_core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

exec "$@"
