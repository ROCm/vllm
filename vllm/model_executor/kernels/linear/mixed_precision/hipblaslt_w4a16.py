# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""hipBLASLt W4A16 GEMM, as a drop-in replacement for the RDNAHybrid kernels.

Runs through AITER (``rocm_aiter_ops.hipb_w4a16_*``), which builds it only when
the hipBLASLt it compiles against exposes the w4a16 API.
``VLLM_ROCM_W4A16_HIPBLASLT`` chooses whether to use it.

Activations, the output and the zero-points are handed over untouched:
hipBLASLt's zero-point region is ``[ceil(N/8)][K/G]`` 32-bit words with row n
in nibble ``n % 8``, which is already how vLLM packs ``qzeros``. Only the
weights are rebuilt at load time: hipBLASLt dropped the ExLlama int4 encoding
and needs plain K order, which is why a layer commits to one kernel family for
both prefill and decode.
"""

import functools
from collections.abc import Callable

import torch

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger

logger = init_logger(__name__)

MODE_DECODE = 1
MODE_PREFILL = 2
MODE_ALL = MODE_DECODE | MODE_PREFILL

_MODES = {"off": 0, "decode": MODE_DECODE, "prefill": MODE_PREFILL, "all": MODE_ALL}

# One block-scale mode per group size, so these are the only ones expressible.
SUPPORTED_GROUP_SIZES = (32, 64, 128)


def is_built() -> bool:
    """Whether the installed AITER exposes the hipBLASLt w4a16 GEMM."""
    return rocm_aiter_ops.has_hipb_w4a16()


@functools.cache
def hipblaslt_w4a16_mode() -> int:
    """Bitmask of the paths ``VLLM_ROCM_W4A16_HIPBLASLT`` routes to hipBLASLt.

    "auto" uses hipBLASLt for layers it has a kernel for, except those with a
    dense prefill copy (see ``use_for_layer``); "off" and "all" force the
    choice. The per-path modes are not usable: hipBLASLt dropped the ExLlama
    int4 encoding, so its weights must be packed in plain K order, while
    ``wvSplitK_int4_g`` and the Triton kernel read the ExLlama shuffle. One
    buffer cannot serve both, and duplicating it costs a second copy of the
    whole weight -- so a mode that splits the GEMMs of one layer between them
    would either read garbage or double the weight footprint. The choice is
    therefore per layer, not per call.
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
            "VLLM_ROCM_W4A16_HIPBLASLT=all needs AITER with the hipBLASLt w4a16 "
            "GEMM (aiter.hipb_w4a16_mm). Install such an AITER, or use 'off'."
        )
    return mode


def use_for_layer(
    group_size: int,
    has_dense_prefill: bool,
    has_solution: Callable[[], bool],
    layer_desc: str,
) -> bool:
    """Whether hipBLASLt owns this layer's GEMMs.

    Under "auto", layers that cache a dense prefill copy stay on the in-tree
    kernels: the dense GEMM beats hipBLASLt for prefill and wvSplitK_int4_g
    beats it for decode. "all" takes every supported layer regardless.

    ``has_solution`` asks hipBLASLt whether it has a kernel for the layer on
    this arch. Under "auto" a miss falls back to the in-tree kernels; under
    "all" it is an error, since the GEMM would fail at its first call.
    """
    if not hipblaslt_w4a16_mode() or not supports_layer(group_size):
        return False
    forced = envs.VLLM_ROCM_W4A16_HIPBLASLT == "all"
    if has_dense_prefill and not forced:
        return False
    if has_solution():
        return True
    if forced:
        raise RuntimeError(
            "VLLM_ROCM_W4A16_HIPBLASLT=all but hipBLASLt has no w4a16 kernel "
            f"for {layer_desc} on this GPU. Use 'auto' or 'off'."
        )
    logger.info_once(
        "hipBLASLt has no w4a16 kernel for %s on this GPU; using the in-tree "
        "W4A16 kernels for it.",
        layer_desc,
    )
    return False


# Probe at least this far: encoder towers can exceed max_num_batched_tokens.
MIN_PROBE_M = 1 << 16


def hipblaslt_w4a16_has_solution(
    dtype: torch.dtype,
    N: int,
    K: int,
    lda: int,  # weight row stride in int4 elements
    group_size: int,
    has_zp: bool,
    max_m: int,
) -> bool:
    """Whether hipBLASLt has a kernel for every batch-size bucket up to max_m.

    The GEMM picks its kernel per power-of-two bucket of M, so every bucket a
    call can land in is asked. Only queries the heuristic (host side, no
    launch), so it is cheap enough to run per layer at load time; the plans it
    finds are kept for the GEMM.
    """
    last = 1 << (max(max_m, MIN_PROBE_M) - 1).bit_length()
    buckets = tuple(1 << i for i in range(last.bit_length()))
    return rocm_aiter_ops.hipb_w4a16_is_supported(
        dtype, N, K, group_size, has_zp, lda, buckets
    )


def supports_layer(group_size: int) -> bool:
    """Whether hipBLASLt can own a layer quantised at this group size.

    The scale modes are per group size -- VEC32/VEC64/VEC128 -- so anything
    else has no mode to ask for.
    Shapes are unconstrained beyond what the op already checks, and the device
    library covers every (dtype, symmetry) combination, so group size is the
    whole predicate.
    """
    return group_size in SUPPORTED_GROUP_SIZES


def hipblaslt_w4a16_gemm(
    a: torch.Tensor,  # [M, K] fp16/bf16
    w_q: torch.Tensor,  # [N, K//2] int8, nibbles in K order
    scale: torch.Tensor,  # from build_scale_buffer
    group_size: int,
    has_zp: bool,
    bias: torch.Tensor | None = None,  # [N], in a's dtype
) -> torch.Tensor:
    return rocm_aiter_ops.hipb_w4a16_mm(a, w_q, scale, group_size, has_zp, bias)


def build_scale_buffer(
    w_s: torch.Tensor,  # [N, K//G] fp16/bf16
    w_zp: torch.Tensor | None,  # [N//8, K//G] int32, 8 rows per word
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(buffer_to_pass, scales_view)`` for the hipBLASLt scale pointer.

    Symmetric layers hand ``w_s`` straight through. Asymmetric layers get one
    allocation holding the scales and, at the next 256-byte boundary, the
    zero-points. ``scales_view`` aliases the head of that allocation and is
    what the layer keeps as ``w_s``, so the scales are not duplicated.
    """
    num_groups = w_s.shape[1]
    if w_s.stride(0) != num_groups or w_s.stride(1) != 1:
        raise RuntimeError(
            f"hipBLASLt W4A16 needs densely packed scales; got stride(0)="
            f"{w_s.stride(0)} for K/G={num_groups}. The gfx115x metadata cliff "
            "pad is not expressible in hipBLASLt's scale layout."
        )
    buf = rocm_aiter_ops.hipb_w4a16_scale_buffer(w_s, w_zp)
    if w_zp is None:
        return buf, buf
    scale_bytes = w_s.numel() * w_s.element_size()
    return buf, buf[:scale_bytes].view(w_s.dtype).reshape(w_s.shape)
