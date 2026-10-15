# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""hipBLASLt W4A16 GEMM, as a drop-in replacement for the RDNAHybrid kernels.

Compiled into ``_rocm_C`` only when CMake's probe finds that the ROCm being
built against has a hipBLASLt exposing the w4a16 API (see the
VLLM_HIPBLASLT_HAS_W4A16 block in CMakeLists.txt); against an older ROCm the
ops simply do not exist. ``VLLM_ROCM_W4A16_HIPBLASLT`` then chooses whether to
use them.

Activations, the output and the zero-points are handed over untouched -- see
``csrc/rocm/hipblaslt_w4a16.cu`` for the index mapping, and note that
hipBLASLt's zero-point region is ``[ceil(N/8)][K/G]`` 32-bit words with row n
in nibble ``n % 8``, which is already how vLLM packs ``qzeros``. Only the
weights are rebuilt at load time: hipBLASLt dropped the ExLlama int4 encoding
and needs plain K order, which is why a layer commits to one kernel family for
both prefill and decode.
"""

import functools

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

MODE_DECODE = 1
MODE_PREFILL = 2
MODE_ALL = MODE_DECODE | MODE_PREFILL

_MODES = {"off": 0, "decode": MODE_DECODE, "prefill": MODE_PREFILL, "all": MODE_ALL}

# Matches c_blockScaleAZeroPointAlignment in hipBLASLt's tensile_host.cpp.
ZERO_POINT_ALIGNMENT = 256

# One block-scale mode per group size, so these are the only ones expressible.
SUPPORTED_GROUP_SIZES = (32, 64, 128)


@functools.cache
def is_built() -> bool:
    """Whether CMake compiled the w4a16 ops into _rocm_C."""
    return hasattr(torch.ops._rocm_C, "hipblaslt_w4a16_gemm")


@functools.cache
def hipblaslt_w4a16_mode() -> int:
    """Bitmask of the paths ``VLLM_ROCM_W4A16_HIPBLASLT`` routes to hipBLASLt.

    "auto" uses hipBLASLt wherever it has a kernel and the in-tree kernels
    elsewhere; "off" and "all" force the choice. The per-path modes are not
    usable: hipBLASLt dropped the ExLlama int4 encoding, so its weights must be
    packed in plain K order, while ``wvSplitK_int4_g`` and the Triton kernel
    read the ExLlama shuffle. One buffer cannot serve both, and duplicating it
    costs a second copy of the whole weight -- so a mode that splits the GEMMs
    of one layer between them would either read garbage or double the weight
    footprint. The choice is therefore per layer, not per call.
    """
    setting = envs.VLLM_ROCM_W4A16_HIPBLASLT
    if setting == "auto":
        return MODE_ALL if is_built() else 0
    mode = _MODES[setting]
    if mode not in (0, MODE_ALL):
        raise ValueError(
            f"VLLM_ROCM_W4A16_HIPBLASLT={setting!r} is no longer supported: "
            "hipBLASLt needs plain-K-order weights and the in-tree kernels "
            "need the ExLlama shuffle, so the two cannot share one weight "
            "buffer. Use 'auto', 'off' or 'all'."
        )
    if mode and not is_built():
        raise RuntimeError(
            "VLLM_ROCM_W4A16_HIPBLASLT=all but vLLM was built against a ROCm "
            "whose hipBLASLt has no w4a16 API, so _rocm_C has no "
            "hipblaslt_w4a16_gemm. Rebuild against a ROCm that exposes it, or "
            "use 'off'."
        )
    return mode


def supports_layer(group_size: int) -> bool:
    """Whether hipBLASLt can own a layer quantised at this group size.

    The scale modes are per group size -- VEC32/VEC64/VEC128, see
    ``scale_mode_of`` in the .cu -- so anything else has no mode to ask for.
    Shapes are unconstrained beyond what the op already checks, and the device
    library covers every (dtype, symmetry) combination, so group size is the
    whole predicate.
    """
    return group_size in SUPPORTED_GROUP_SIZES


def hipblaslt_w4a16_gemm(
    a: torch.Tensor,  # [M, K] fp16/bf16
    w_q: torch.Tensor,  # [N, K//2] int8, nibbles in K order
    scale: torch.Tensor,  # [N, K//G] (symmetric) or the combined buffer (asym)
    group_size: int,
    has_zp: bool,
) -> torch.Tensor:
    return torch.ops._rocm_C.hipblaslt_w4a16_gemm(a, w_q, scale, group_size, has_zp)


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

    n8 = w_zp.shape[0]
    assert n8 == N // 8, f"zp rows {n8} do not match N={N}"
    # vLLM packs [q][g] words with row n in nibble n % 8, which is exactly what
    # hipBLASLt addresses, so the words go over as bytes with no repack.
    zp_hbl = w_zp.contiguous().view(torch.uint8).reshape(-1)

    scale_bytes = N * num_groups * w_s.element_size()
    zp_offset = -(-scale_bytes // ZERO_POINT_ALIGNMENT) * ZERO_POINT_ALIGNMENT
    buf = torch.empty(zp_offset + zp_hbl.numel(), dtype=torch.uint8, device=w_s.device)
    buf[:scale_bytes].copy_(w_s.contiguous().view(torch.uint8).reshape(-1))
    buf[zp_offset:].copy_(zp_hbl)
    return buf, buf[:scale_bytes].view(w_s.dtype).reshape(N, num_groups)
