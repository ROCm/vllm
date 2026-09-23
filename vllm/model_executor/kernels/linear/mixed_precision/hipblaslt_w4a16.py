# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""hipBLASLt W4A16 GEMM, as a drop-in replacement for the RDNAHybrid kernels.

Experimental: gated off by default behind ``VLLM_ROCM_W4A16_HIPBLASLT`` and
built on demand against a hipBLASLt checkout that exposes the w4a16 API
(``VLLM_HIPBLASLT_W4A16_ROOT``), so a stock ROCm SDK build is unaffected.

The weight and activation buffers are handed to hipBLASLt exactly as the
existing kernels see them -- see ``csrc/rocm/hipblaslt_w4a16.cu`` for the index
mapping. The one thing that has to be rebuilt is the zero-point region, because
hipBLASLt wants it appended to the scales in one allocation and ordered
``[rowpair][group]`` rather than vLLM's ``[group][rowpair]``.
"""

import functools
import os
from pathlib import Path

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

MODE_DECODE = 1
MODE_PREFILL = 2

_MODES = {"off": 0, "decode": MODE_DECODE, "prefill": MODE_PREFILL, "all": 3}

# Matches c_blockScaleAZeroPointAlignment in hipBLASLt's tensile_host.cpp.
ZERO_POINT_ALIGNMENT = 256


@functools.cache
def hipblaslt_w4a16_mode() -> int:
    """Bitmask of the paths ``VLLM_ROCM_W4A16_HIPBLASLT`` routes to hipBLASLt."""
    return _MODES[envs.VLLM_ROCM_W4A16_HIPBLASLT]


def _root() -> Path:
    root = envs.VLLM_HIPBLASLT_W4A16_ROOT
    if not root:
        raise RuntimeError(
            "VLLM_ROCM_W4A16_HIPBLASLT is enabled but VLLM_HIPBLASLT_W4A16_ROOT "
            "is unset. Point it at a built rocm-libraries checkout whose "
            "hipBLASLt exposes the w4a16 API."
        )
    return Path(root)


def _paths() -> tuple[list[str], str, str]:
    """(include dirs, hipBLASLt lib dir, Tensile library dir)."""
    root = _root()
    lib_dir = root / "build" / "projects" / "hipblaslt" / "library"
    includes = [
        # The w4a16 enums live here, so it must precede the ROCm SDK's copy.
        str(root / "projects" / "hipblaslt" / "library" / "include"),
        str(lib_dir / "include"),
        str(root / "projects" / "hipblas-common" / "library" / "include"),
    ]
    # A build tree keeps the Tensile artifacts under a per-arch subdirectory,
    # unlike an installed ROCm where they sit flat in hipblaslt/library.
    tensile_dir = root / "build" / "projects" / "hipblaslt" / "Tensile" / "library"
    from vllm.platforms.rocm import _GCN_ARCH

    arch = _GCN_ARCH.split(":")[0]  # gfx1151:xnack- -> gfx1151
    if (tensile_dir / arch).is_dir():
        tensile_dir = tensile_dir / arch

    for p in includes + [str(lib_dir / "libhipblaslt.so"), str(tensile_dir)]:
        if not os.path.exists(p):
            raise RuntimeError(
                f"VLLM_HIPBLASLT_W4A16_ROOT={root} does not look like a built "
                f"rocm-libraries checkout: {p} is missing."
            )
    return includes, str(lib_dir), str(tensile_dir)


def _check_loaded_library(lib_dir: str) -> None:
    """Fail loudly when the process resolved a different libhipblaslt.

    PyTorch links ``libhipblaslt.so.1`` from the ROCm SDK wheel, and ``dlopen``
    of a library with an already-loaded SONAME returns the loaded one whatever
    the RPATH says. Without this check the symptom is an opaque
    HIPBLAS_STATUS_INVALID_VALUE from an unrecognised scale mode.
    """
    loaded = set()
    with open("/proc/self/maps") as f:
        for line in f:
            path = line.rsplit(" ", 1)[-1].strip()
            if "libhipblaslt.so" in path:
                loaded.add(os.path.realpath(path))
    expected = os.path.realpath(os.path.join(lib_dir, "libhipblaslt.so"))
    if expected not in loaded:
        raise RuntimeError(
            f"the process is using {sorted(loaded) or 'no libhipblaslt'} rather "
            f"than {expected}; a same-SONAME library was already loaded. Launch "
            f"with LD_LIBRARY_PATH={lib_dir} (or LD_PRELOAD of that .so) so the "
            "w4a16-capable hipBLASLt wins."
        )


@functools.cache
def _ext():
    from torch.utils.cpp_extension import load

    source = (
        Path(__file__).resolve().parents[5] / "csrc" / "rocm" / "hipblaslt_w4a16.cu"
    )
    if not source.exists():
        raise RuntimeError(
            f"{source} is missing; the hipBLASLt W4A16 path needs an editable "
            "vLLM checkout with csrc/ present."
        )

    includes, lib_dir, tensile_dir = _paths()
    os.environ.setdefault("HIPBLASLT_TENSILE_LIBPATH", tensile_dir)

    module = load(
        name="vllm_hipblaslt_w4a16",
        sources=[str(source)],
        extra_include_paths=includes,
        extra_cflags=["-O3"],
        extra_ldflags=[f"-L{lib_dir}", "-lhipblaslt", f"-Wl,-rpath,{lib_dir}"],
        verbose=False,
    )
    _check_loaded_library(lib_dir)
    logger.info("hipBLASLt W4A16 enabled (mode=%s)", envs.VLLM_ROCM_W4A16_HIPBLASLT)
    return module


def hipblaslt_w4a16_gemm(
    a: torch.Tensor,  # [M, K] fp16/bf16
    w_q: torch.Tensor,  # [N, K//2] int8, ExLlama shuffle
    scale: torch.Tensor,  # [N, K//G] (symmetric) or the combined buffer (asym)
    group_size: int,
    has_zp: bool,
) -> torch.Tensor:
    return _ext().hipblaslt_w4a16_gemm(a, w_q, scale, group_size, has_zp)


def build_scale_buffer(
    w_s: torch.Tensor,  # [N, K//G] fp16/bf16
    w_zp: torch.Tensor | None,  # [N//8, K//G] int32, 8 rows per word
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(buffer_to_pass, scales_view)`` for the hipBLASLt scale pointer.

    Symmetric layers hand ``w_s`` straight through. Asymmetric layers get one
    allocation holding the scales and, at the next 256-byte boundary, the
    zero-points in hipBLASLt's order. ``scales_view`` aliases the head of that
    allocation and is what the layer keeps as ``w_s``, so the HIP skinny and
    Triton kernels read the same bytes and the scales are not duplicated.
    """
    N, num_groups = w_s.shape
    if w_s.stride(0) != num_groups or w_s.stride(1) != 1:
        raise RuntimeError(
            f"hipBLASLt W4A16 needs densely packed scales; got stride(0)="
            f"{w_s.stride(0)} for K/G={num_groups}. The gfx1151 metadata cliff "
            "pad is not expressible in hipBLASLt's scale layout."
        )
    if w_zp is None:
        return w_s, w_s

    # vLLM packs the zero-points [q][g][p] (group outside the 4-byte word);
    # hipBLASLt wants [q][p][g] (each row-pair gets a contiguous group run).
    # Same nibble pairing, so this is a transpose of the last two axes.
    n8 = w_zp.shape[0]
    assert n8 == N // 8, f"zp rows {n8} do not match N={N}"
    zp_hbl = (
        w_zp.contiguous()
        .view(torch.uint8)
        .reshape(n8, num_groups, 4)
        .permute(0, 2, 1)
        .reshape(-1)
    )

    scale_bytes = N * num_groups * w_s.element_size()
    zp_offset = -(-scale_bytes // ZERO_POINT_ALIGNMENT) * ZERO_POINT_ALIGNMENT
    buf = torch.empty(zp_offset + zp_hbl.numel(), dtype=torch.uint8, device=w_s.device)
    buf[:scale_bytes].copy_(w_s.contiguous().view(torch.uint8).reshape(-1))
    buf[zp_offset:].copy_(zp_hbl)
    return buf, buf[:scale_bytes].view(w_s.dtype).reshape(N, num_groups)
