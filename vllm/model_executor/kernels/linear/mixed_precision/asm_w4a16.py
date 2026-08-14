# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct-HIP loader/launcher for the hand-editable AMDGCN W4A16 skinny GEMM.

This bypasses the Triton JIT entirely: it loads a prebuilt ``.hsaco`` (assembled
from a hand-editable ``.amdgcn`` under ``asm/``) with ``libamdhip64`` and launches
it via ``hipModuleLaunchKernel`` using the kernarg ABI pinned in
``asm/hybrid_w4a16_skinny_gfx1151.meta.json``.

Workflow: edit ``asm/hybrid_w4a16_skinny_gfx1151.amdgcn`` -> rebuild vLLM (CMake
assembles it) OR just rerun (this module re-assembles on demand when the
``.amdgcn`` is newer than the cached ``.hsaco``). The dispatcher in
``hybrid_w4a16.py`` routes the matching launch here and ``assert``s that every
constexpr/config field in the meta matches the request.

Enabled only when ``VLLM_W4A16_HANDASM=1``.
"""

from __future__ import annotations

import ctypes
import functools
import glob
import json
import math
import os
import struct
import subprocess
from pathlib import Path

import torch

_ASM_DIR = Path(__file__).parent / "asm"
_STEM = "hybrid_w4a16_skinny_gfx1151"
_AMDGCN = _ASM_DIR / f"{_STEM}.amdgcn"
_HSACO = _ASM_DIR / f"{_STEM}.hsaco"
_META = _ASM_DIR / f"{_STEM}.meta.json"

# hipModuleLaunchKernel "extra" sentinels.
_HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
_HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
_HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)


@functools.lru_cache(maxsize=1)
def load_meta() -> dict:
    return json.loads(_META.read_text())


def _find_clang() -> str:
    """Locate an amdgcn-capable clang for on-demand (re)assembly."""
    cand = os.environ.get("VLLM_ROCM_CLANG")
    if cand and Path(cand).exists():
        return cand
    # ROCm SDK wheel shipped in the active venv (next to torch in
    # site-packages), then system ROCm.
    site = os.path.dirname(os.path.dirname(torch.__file__))  # site-packages
    patterns = [
        os.path.join(site, "_rocm_sdk_devel", "lib", "llvm", "bin", "clang"),
        os.path.join(site, "_rocm_sdk_core", "lib", "llvm", "bin", "clang"),
        "/opt/rocm*/llvm/bin/clang",
    ]
    for pat in patterns:
        for hit in glob.glob(pat):
            if Path(hit).exists():
                return str(Path(hit).resolve())
    raise FileNotFoundError(
        "No amdgcn clang found to assemble the W4A16 .amdgcn. Set "
        "VLLM_ROCM_CLANG to a ROCm clang, or build vLLM (CMake assembles it)."
    )


def assemble(force: bool = False) -> Path:
    """Assemble the .amdgcn -> .hsaco if missing/stale. Returns the .hsaco path.

    This is the dev-loop fast path; the CMake build performs the same step at
    build time so installed wheels ship a ready .hsaco.
    """
    if (
        not force
        and _HSACO.exists()
        and _HSACO.stat().st_mtime >= _AMDGCN.stat().st_mtime
    ):
        return _HSACO
    clang = _find_clang()
    arch = load_meta()["arch"]
    cmd = [
        clang,
        "-target",
        "amdgcn-amd-amdhsa",
        f"-mcpu={arch}",
        "-x",
        "assembler",
        str(_AMDGCN),
        "-o",
        str(_HSACO),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return _HSACO


def _open_hip() -> ctypes.CDLL:
    """Open libamdhip64 (already in-process via torch's ROCm SDK) by full path.

    The bare soname is not on the loader path in the SDK-wheel layout, so resolve
    the versioned file directly; dlopen-ing the same path just shares torch's
    handle and HIP context.
    """
    site = os.path.dirname(os.path.dirname(torch.__file__))
    for pat in (
        os.path.join(site, "_rocm_sdk_core", "lib", "libamdhip64.so*"),
        os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so*"),
        "libamdhip64.so",
    ):
        for hit in sorted(glob.glob(pat)) or ([pat] if "*" not in pat else []):
            try:
                return ctypes.CDLL(hit)
            except OSError:
                continue
    raise OSError("could not locate libamdhip64 for the hand-asm W4A16 loader")


class _HandAsmKernel:
    """Loads the hsaco once and launches it via hipModuleLaunchKernel."""

    def __init__(self) -> None:
        self.meta = load_meta()
        self.hip = _open_hip()
        self.hip.hipModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self.hip.hipModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.hip.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,  # grid
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,  # block
            ctypes.c_uint,  # shared
            ctypes.c_void_p,  # stream
            ctypes.POINTER(ctypes.c_void_p),  # kernelParams
            ctypes.POINTER(ctypes.c_void_p),  # extra
        ]
        image = bytearray(assemble().read_bytes())
        buf = (ctypes.c_char * len(image)).from_buffer(image)
        self._image_keepalive = image  # hipModuleLoadData copies, but be safe
        module = ctypes.c_void_p()
        self._check(self.hip.hipModuleLoadData(ctypes.byref(module), buf))
        func = ctypes.c_void_p()
        self._check(
            self.hip.hipModuleGetFunction(
                ctypes.byref(func), module, self.meta["kernel_name"].encode()
            )
        )
        self.module = module
        self.func = func

    def _check(self, err: int) -> None:
        if err != 0:
            raise RuntimeError(f"HIP error {err} in hand-asm W4A16 loader")

    def launch(
        self, a, b_q, scales, c, M, N, K, K8, stride_bn, num_groups, group_size, stream
    ) -> None:
        m = self.meta
        ksize = m["kernarg_size"]
        kbuf = bytearray(ksize)
        # Pack per the pinned kernarg_layout. ptr_null args stay zero.
        vals = {
            "a_ptr": a.data_ptr(),
            "b_ptr": b_q.data_ptr(),
            "scales_ptr": scales.data_ptr(),
            # symmetric: carrier unused, mirror production's dummy (scales).
            "packed_scale_zp_ptr": scales.data_ptr(),
            "c_ptr": c.data_ptr(),
            "M": M,
            "N": N,
            "K": K,
            "K8": K8,
            "stride_bn": stride_bn,
            "num_groups": num_groups,
            "group_size": group_size,
        }
        for arg in m["kernarg_layout"]:
            off, kind = arg["offset"], arg["kind"]
            if kind == "ptr":
                struct.pack_into("<Q", kbuf, off, vals[arg["name"]])
            elif kind == "i32":
                struct.pack_into("<i", kbuf, off, vals[arg["name"]])
            elif kind == "ptr_null":
                struct.pack_into("<Q", kbuf, off, 0)
            else:
                raise ValueError(f"unknown kernarg kind {kind}")

        cbuf = (ctypes.c_char * ksize).from_buffer(kbuf)
        size = ctypes.c_size_t(ksize)
        extra = (ctypes.c_void_p * 5)(
            _HIP_LAUNCH_PARAM_BUFFER_POINTER,
            ctypes.cast(cbuf, ctypes.c_void_p),
            _HIP_LAUNCH_PARAM_BUFFER_SIZE,
            ctypes.cast(ctypes.byref(size), ctypes.c_void_p),
            _HIP_LAUNCH_PARAM_END,
        )
        gx = math.ceil(M / m["block_m"])
        gy = math.ceil(N / m["block_n"])
        block = m["num_warps"] * m["warp_size"]
        self._check(
            self.hip.hipModuleLaunchKernel(
                self.func,
                gx,
                gy,
                1,
                block,
                1,
                1,
                m["shared_bytes"],
                ctypes.c_void_p(stream),
                None,
                extra,
            )
        )


@functools.lru_cache(maxsize=1)
def _kernel() -> _HandAsmKernel:
    return _HandAsmKernel()


def config_matches(
    dtype, has_zp, group_size, block_m, block_n, block_k, num_warps
) -> bool:
    """True iff this launch is exactly what the pinned .amdgcn was built for."""
    m = load_meta()
    return (
        str(dtype) == f"torch.{m['dtype']}"
        and bool(has_zp) == m["has_zp"]
        and group_size == m["group_size"]
        and block_m == m["block_m"]
        and block_n == m["block_n"]
        and block_k == m["block_k"]
        and num_warps == m["num_warps"]
    )


def launch_skinny_w4a16(
    a, b_q, scales, c, M, N, K, K8, stride_bn, num_groups, group_size
) -> None:
    """Assert-checked entry point used by the hybrid_w4a16 dispatcher."""
    m = load_meta()
    assert a.dtype == getattr(torch, m["dtype"]), (
        f"hand-asm W4A16 built for {m['dtype']}, got {a.dtype}"
    )
    assert group_size == m["group_size"], (
        f"hand-asm W4A16 built for group_size={m['group_size']}, got {group_size}"
    )
    assert num_groups == K // group_size
    assert K8 == K // 8
    stream = torch.cuda.current_stream().cuda_stream
    _kernel().launch(
        a, b_q, scales, c, M, N, K, K8, stride_bn, num_groups, group_size, stream
    )
