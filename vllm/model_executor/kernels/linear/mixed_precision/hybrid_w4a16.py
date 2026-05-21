# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Hybrid W4A16 kernel: Triton for prefill, HIP skinny for decode.

Routes based on batch size M:
  M <= MAX_SKINNY_BATCH_SIZE: HIP skinny GEMM (wvSplitK_int4_g)
  M > MAX_SKINNY_BATCH_SIZE:  Triton W4A16 fused dequant GEMM

Stores weights ONCE in skinny layout [N, K//8] int32 (ExLlama shuffle).
Both the HIP skinny kernel and the triton kernel read from this single
weight copy. The triton kernel transposes tiles in-register.
"""

from contextlib import nullcontext

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (
    permute_param_layout_,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx1x
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

SUPPORTED_GROUP_SIZES = [32, 64, 128]

# Maximum batch size M for the HIP skinny kernel path (C++ supports N_in
# up to 5).  When M exceeds this AND K*M fits in LDS, the skinny kernel is
# used; otherwise the Triton prefill path handles the GEMM.
MAX_SKINNY_BATCH_SIZE = 5
LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2  # 32768 fp16 elements

# AIESW-32176: shapes routed to the CK WMMA b_scale GEMM op (gfx1151 only),
# now dispatched through aiter.ops.gemm_w4a16 instead of an in-tree _rocm_C op.
# Each entry is keyed by (N, K, group_size, dtype) and maps to (min_M, KPerBlock).
# Dispatch fires when M >= min_M for this layer — the kernel handles any M >= 1,
# but min_M sets a lower bound below which fixed launch overhead (~0.4 ms) dominates
# and Triton is comparable. Above the threshold, CK holds 22-31 TFLOPS uniformly
# across the M dimension (measured M=256-16384 — see AIInfo memory
# project_aiesw_32176_phase5c_shapes). This handles arbitrary chunked-prefill
# chunk sizes including the M=1920 second-chunk case for prompt=3968+chunk=2048.
# All four Qwen3-4B prefill linear columns are wired; the same kernel binary
# handles all shapes (M/N/K are runtime args; only KPerBlock is templated).
# Each wired layer costs an extra weight copy (~0.92 GB total for the four
# Qwen3-4B columns on a 36-layer model).
#
# bf16 entries are registered below for A/B measurement only — prior microbench
# (gate_up M=2048 N=19456 K=2560 G=128 on gfx1151) showed CK 19.4 vs Triton
# 24.4 TFLOPS (Triton 1.26x faster on bf16, vs CK 1.36x faster on fp16). Root
# cause is the RDNA3 (gfx11) ISA lacking a packed bf16 multiply instruction
# (no V_PK_FMA_BF16), so CK's bf16 dequant falls back to scalar fp32
# conversion while Triton's compiler schedules fp32 dual-issue more
# aggressively. CK is expected to lose on bf16, but we want the end-to-end
# number on a real bf16 model (RedHatAI/Qwen3-8B-quantized.w4a16) before
# locking the dispatch decision. Remove the bf16 entries (or guard them) once
# the measurement is done.
_CK_W4A16_TARGET_SHAPES: dict[tuple, tuple[int, int]] = {
    # ---- Qwen/Qwen3-4B-AWQ (h=2560 inter=9728 nq=32 nkv=8 hd=128) ----
    (19456, 2560, 128, torch.float16): (256, 32),  # gate_up_proj
    (6144, 2560, 128, torch.float16): (256, 32),  # qkv_proj
    (2560, 4096, 128, torch.float16): (256, 32),  # o_proj
    (2560, 9728, 128, torch.float16): (256, 32),  # down_proj
    # ---- cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit (h=2560 inter=9728 nq=32 nkv=8
    #      hd=128, group_size=32, sym, AIESW-32282). Vision tower is excluded
    #      from quantization (per the model's quantization_config.ignore list),
    #      so only the LLM-side prefill linear columns hit this dispatch. The
    #      activation dtype here is the bench dtype (--dtype float16 forces
    #      fp16); the model ships bf16 weights but they're cast at load time.
    (19456, 2560, 32, torch.float16): (256, 32),  # gate_up_proj
    (6144, 2560, 32, torch.float16): (256, 32),  # qkv_proj
    (2560, 4096, 32, torch.float16): (256, 32),  # o_proj
    (2560, 9728, 32, torch.float16): (256, 32),  # down_proj
    # ---- Qwen/Qwen3-8B-AWQ (h=4096 inter=12288 nq=32 nkv=8 hd=128) ----
    (24576, 4096, 128, torch.float16): (256, 32),  # gate_up_proj
    (6144, 4096, 128, torch.float16): (256, 32),  # qkv_proj
    (4096, 4096, 128, torch.float16): (256, 32),  # o_proj
    (4096, 12288, 128, torch.float16): (256, 32),  # down_proj
    # ---- Qwen/Qwen2.5-3B-Instruct-AWQ (h=2048 inter=11008 nq=16 nkv=2 hd=128) ----
    (22016, 2048, 128, torch.float16): (256, 32),  # gate_up_proj
    (2560, 2048, 128, torch.float16): (256, 32),  # qkv_proj
    (2048, 2048, 128, torch.float16): (256, 32),  # o_proj
    (2048, 11008, 128, torch.float16): (256, 32),  # down_proj
    # ---- Qwen/Qwen2.5-7B-Instruct-AWQ (h=3584 inter=18944 nq=28 nkv=4 hd=128) ----
    (37888, 3584, 128, torch.float16): (256, 32),  # gate_up_proj
    (4608, 3584, 128, torch.float16): (256, 32),  # qkv_proj
    (3584, 3584, 128, torch.float16): (256, 32),  # o_proj
    (3584, 18944, 128, torch.float16): (256, 32),  # down_proj
    # ---- TheBloke/Llama-2-7B-AWQ (h=4096 inter=11008 nq=32 nkv=32 hd=128, MHA) ----
    (22016, 4096, 128, torch.float16): (256, 32),  # gate_up_proj
    (12288, 4096, 128, torch.float16): (256, 32),  # qkv_proj (no GQA, q=k=v=4096)
    # o_proj (4096, 4096) reused from Qwen3-8B
    (4096, 11008, 128, torch.float16): (256, 32),  # down_proj
    # ---- google/gemma-2b-AWQ (h=2048 inter=16384 nq=8 nkv=1 hd=256) ----
    (32768, 2048, 128, torch.float16): (
        256,
        32,
    ),  # gate_up_proj (Gemma uses 2*inter for gate_up)
    # qkv (2560, 2048) reused from Qwen2.5-3B
    # o_proj (2048, 2048) reused from Qwen2.5-3B
    (2048, 16384, 128, torch.float16): (256, 32),  # down_proj
    # ---- RedHatAI/Qwen3-8B-quantized.w4a16 (bf16 g=128 sym, same Qwen3-8B
    #      arch as fp16 row above) ----
    (24576, 4096, 128, torch.bfloat16): (256, 32),  # gate_up_proj
    (6144, 4096, 128, torch.bfloat16): (256, 32),  # qkv_proj
    (4096, 4096, 128, torch.bfloat16): (256, 32),  # o_proj
    (4096, 12288, 128, torch.bfloat16): (256, 32),  # down_proj
    # ---- Orion-zhen/Qwen3-1.7B-AWQ (bf16 g=128 asym; h=2048 inter=6144
    #      nq=16 nkv=8 hd=128) ----
    (12288, 2048, 128, torch.bfloat16): (256, 32),  # gate_up_proj
    (4096, 2048, 128, torch.bfloat16): (256, 32),  # qkv_proj
    (2048, 2048, 128, torch.bfloat16): (256, 32),  # o_proj
    (2048, 6144, 128, torch.bfloat16): (256, 32),  # down_proj
    # ---- RedHatAI/Qwen2.5-VL-7B-Instruct-quantized.w4a16 (bf16 g=128 sym;
    #      shapes match the fp16 Qwen2.5-7B entries above) ----
    (37888, 3584, 128, torch.bfloat16): (256, 32),  # gate_up_proj
    (4608, 3584, 128, torch.bfloat16): (256, 32),  # qkv_proj
    (3584, 3584, 128, torch.bfloat16): (256, 32),  # o_proj
    (3584, 18944, 128, torch.bfloat16): (256, 32),  # down_proj
    # ---- trymirai/SmolLM2-1.7B-Instruct-AWQ (bf16 g=32 asym; h=2048
    #      inter=8192 nq=32 nkv=32 hd=64, MHA) ----
    (16384, 2048, 32, torch.bfloat16): (256, 32),  # gate_up_proj
    (6144, 2048, 32, torch.bfloat16): (256, 32),  # qkv_proj (no GQA)
    # (2048, 2048, 32, torch.bfloat16) reused under SmolLM2 o_proj — fp16
    # entry doesn't apply
    (2048, 2048, 32, torch.bfloat16): (256, 32),  # o_proj
    (2048, 8192, 32, torch.bfloat16): (256, 32),  # down_proj
    # ---- cyankiwi/gemma-4-31B-it-AWQ-4bit (bf16 g=32 asym VLM; h=5376
    #      inter=21504 nq=32 nkv=16 hd=256) ----
    (43008, 5376, 32, torch.bfloat16): (256, 32),  # gate_up_proj
    (16384, 5376, 32, torch.bfloat16): (256, 32),  # qkv_proj
    (5376, 8192, 32, torch.bfloat16): (256, 32),  # o_proj (K = nq*hd = 32*256)
    (5376, 21504, 32, torch.bfloat16): (256, 32),  # down_proj
}


def _is_gfx1151() -> bool:
    """True iff current device is gfx1151 (Strix Halo, compute_cap (11, 5))."""
    if not on_gfx1x():
        return False
    try:
        return torch.cuda.get_device_capability(0) == (11, 5)
    except Exception:
        return False


def _lookup_ck_target(
    N: int, K: int, group_size: int, dtype: torch.dtype
) -> tuple[int, int] | None:
    """Find a registered CK target for this layer's (N, K, group, dtype).
    Returns (min_M, KPerBlock) if any, else None. Called once per layer at
    load time (Python ints, not SymInts) — so dict lookup is safe.

    AIESW-32282: previously the dispatch was strictly opt-in via
    _CK_W4A16_TARGET_SHAPES (each (N, K, group, dtype) had to be listed
    explicitly). After threading the runtime element-op down and verifying
    correctness across all the AIESW-32282 non-MoE shapes, we now also
    accept any shape that meets the CK config's static constraints. The
    explicit table still wins (per-shape (min_M, KPerBlock) overrides);
    the fallback fires only for shapes not listed.

    Constraints for the generic fallback (EXP1_FINAL kernel config):
      - dtype ∈ {fp16, bf16}                 (other dtypes have no CK kernel)
      - group_size ∈ {32, 128}               (wired ScaleBlockK instantiations)
      - K % KPerBlock(=32) == 0              (B tile load divisibility)
      - K % group_size == 0                  (scale layout integrality)
      - N % NPerBlock(=128) == 0             (B tile N divisibility)

    Default returned for the fallback: (min_M=256, KPerBlock=32). Below
    M=256 the kernel's ~0.4 ms fixed launch overhead dominates and Triton
    is comparable on these shapes, so we don't dispatch CK there."""
    if not _is_gfx1151():
        return None
    entry = _CK_W4A16_TARGET_SHAPES.get((N, K, group_size, dtype))
    if entry is not None:
        return entry
    # Generic fallback: any shape meeting the CK config's static constraints.
    if dtype not in (torch.float16, torch.bfloat16):
        return None
    if group_size not in (32, 128):
        return None
    if K % 32 != 0 or K % group_size != 0:
        return None
    if N % 128 != 0:
        return None
    return (256, 32)


def _has_aiter_w4a16_op() -> bool:
    """True iff aiter is importable AND exposes ops.gemm_w4a16. Used to gate
    the CK W4A16 dispatch — we soft-import so non-aiter builds (or builds where
    the op hasn't been added yet) fall back to Triton transparently. Mirrors
    the find_spec("aiter") pattern in vllm/_aiter_ops.py.

    AIESW-32176: this replaces the prior _has_ck_w4a16_op() / _has_ck_w4a16_zp_op()
    pair which probed torch.ops._rocm_C for an in-tree CK kernel. The single
    aiter op handles both symmetric and asymmetric (per-group zero point) routes
    via its optional scaled_zp argument."""
    from importlib.util import find_spec

    if find_spec("aiter") is None:
        return False
    try:
        import aiter

        return hasattr(aiter, "ops") and hasattr(aiter.ops, "gemm_w4a16")
    except Exception:
        return False


# Cached at import time so the find_spec lookup isn't repeated in the hot path
# and so torch.compile can treat the resulting branch as a Python constant.
# vllm/_aiter_ops.py uses the same pattern for IS_AITER_FOUND.
_HAS_AITER_W4A16_OP = _has_aiter_w4a16_op()


def _ck_disabled() -> bool:
    """Set VLLM_DISABLE_CK_W4A16=1 to bypass the CK (aiter) dispatch and stay
    on Triton. Used for A/B benchmarking the CK kernel against the Triton
    baseline without rebuilding. Name kept (not VLLM_DISABLE_AITER_W4A16) to
    match existing benchmark scripts and JIRA notes."""
    import os

    return os.environ.get("VLLM_DISABLE_CK_W4A16", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _ck_pre_dequant_to_lds() -> bool:
    """Set VLLM_CK_W4A16_PRE_DEQUANT=1 to route the CK W4A16 dispatch into
    the PreDequantToLDS=true variant of the aiter.ops.gemm_w4a16 op.

    PreDequantToLDS=true dequants packed int4 weights once per K-block into
    an LDS scratch region of activation-dtype B, so the WMMA inner loop
    reads dequantized bf16/fp16 from LDS instead of dequanting per-tile in
    VGPRs. The goal is to amortize the IEEE-correct round-to-bf16 dequant
    cost — measured on RDNA 3.5 as v_add3_u32 + v_cmp_o_f32 + v_cndmask_b16
    per nibble — which is the structural bottleneck behind the 8.6% bf16 CK
    vs Triton gap on RedHatAI/Qwen3-8B-quantized.w4a16 (3346 vs 3082 ms
    TTFT). See vllm4/notes/ck-w4a16-isa/README.md.

    Status: the template hook is wired end-to-end through aiter, but the
    kernel body is currently a stub that fails a TORCH_CHECK at runtime —
    see TODO(AIESW-32282) in
    aiter/csrc/ck_w4a16/include/gemm_w4a16_common.cuh.
    Until the kernel lands, setting this env var will surface the stub
    error from the CK dispatch path."""
    import os

    return os.environ.get("VLLM_CK_W4A16_PRE_DEQUANT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _triton_use_scaled_zp() -> bool:
    """Set VLLM_TRITON_W4A16_SCALED_ZP=1 to make the Triton W4A16 fallback
    use the CK-style scaled_zp dequant formula -- (nibble - 8) * scale -
    scaled_zp -- instead of the default (nibble - zp_raw) * scale. Only
    fires when the layer's _hybrid_w_scaled_zp_ck precompute exists
    (i.e. when the shape is covered by the CK target table); otherwise
    falls back to the raw-zp path with no behavior change."""
    import os

    return os.environ.get("VLLM_TRITON_W4A16_SCALED_ZP", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _repack_vllm_to_ck_b_scale(
    w_q_skinny_i32: torch.Tensor,  # [N, K//8] int32
    KPerBlock: int,
) -> torch.Tensor:
    """vLLM ExLlama [N, K//8] int32 -> CK pk_i4 [K0, N, K1//2] int8. Pure
    reshape + axis swap (nibble shuffle is byte-identical). Scales pass through
    unchanged — CK's b1_k_n is a stride-quirk view over [N, K/G] row-major bytes."""
    N, K_div_8 = w_q_skinny_i32.shape
    K = K_div_8 * 8
    K0 = K // KPerBlock
    return (
        w_q_skinny_i32.reshape(N, K0, KPerBlock // 8)
        .permute(1, 0, 2)
        .contiguous()
        .view(torch.int8)
    )


# ---------------------------------------------------------------------------
# Triton kernel for the prefill path (reads skinny-format weights [N, K//8])
# ---------------------------------------------------------------------------


@triton.jit
def _triton_w4a16_skinny_fmt_kernel(
    # Pointers
    a_ptr,  # [M, K]  fp16/bf16 activations
    b_ptr,  # [N, K//8]  int32 packed (ExLlama shuffle, K is packed dim)
    scales_ptr,  # [N, K//G]  fp16/bf16 scales (skinny layout)
    zp_ptr,  # [N, K//G]  fp16/bf16 raw zp (HAS_ZP) or scaled_zp (HAS_SCALED_ZP)
    c_ptr,  # [M, N]  fp16/bf16 output
    # Dimensions
    M,
    N,
    K,
    K8,  # K // 8
    num_groups,  # K // group_size
    # Quantization parameters
    group_size,
    ZP_BIAS: tl.constexpr,
    HAS_ZP: tl.constexpr,
    HAS_SCALED_ZP: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused W4A16 GEMM reading weights from skinny format [N, K//8].

    B is stored as [N, K//8] int32 using ExLlama shuffle packing:
      each int32 packs 8 K-values with interleave [0,2,4,6,1,3,5,7]:
        packed = val[0] | (val[2]<<4) | (val[4]<<8) | (val[6]<<12)
               | (val[1]<<16) | (val[3]<<20) | (val[5]<<24) | (val[7]<<28)

    Scales are [N, K//G] (skinny layout, NOT transposed).
    Three mutually-exclusive dequant modes:
      - HAS_ZP=True, HAS_SCALED_ZP=False (default asym):
            (nibble - zp_raw) * scale, with zp_raw loaded from zp_ptr.
      - HAS_SCALED_ZP=True, HAS_ZP=False (CK-style asym):
            (nibble - 8) * scale - scaled_zp, with scaled_zp loaded from
            zp_ptr (precomputed (zp_raw - 8) * scale per group at load time).
            Algebraic identity vs the HAS_ZP path; differs only in fp
            rounding (one extra act-dtype subtract per dequant pack).
      - both False (symmetric): only ZP_BIAS=8 is subtracted, no zp load.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ExLlama unshuffle shifts: shift[j] = (j//2)*4 + (j%2)*16
    # For 8 values: [0, 16, 4, 20, 8, 24, 12, 28]
    exllama_shifts_row = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    # Tile across BLOCK_K: repeat the 8-element pattern BLOCK_K//8 times
    shifts_1d = tl.reshape(
        tl.broadcast_to(exllama_shifts_row[None, :], (BLOCK_K // 8, 8)),
        (BLOCK_K,),
    )
    # Broadcast to [BLOCK_N, BLOCK_K]
    shifts_full = tl.broadcast_to(shifts_1d[None, :], (BLOCK_N, BLOCK_K))

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # ---- Load activations A: [BLOCK_M, BLOCK_K] ----
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # ---- Load packed weights B: [BLOCK_N, BLOCK_K//8] int32 ----
        offs_k8 = k_start * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)
        b_ptrs = b_ptr + offs_n[:, None] * K8 + offs_k8[None, :]
        mask_b = (offs_n[:, None] < N) & (offs_k8[None, :] < K8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        # ---- Unpack int4 weights with ExLlama unshuffle ----
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = (b >> shifts_full) & 0xF  # [BLOCK_N, BLOCK_K]

        # ---- Load scales from [N, K//G] layout ----
        g_idx = (k_start * BLOCK_K) // group_size
        scale_ptrs = scales_ptr + offs_n * num_groups + g_idx
        scale_mask = offs_n < N
        scales = tl.load(scale_ptrs, mask=scale_mask, other=1.0)

        # ---- Dequantize ----
        if HAS_SCALED_ZP:
            # CK-style asymmetric: (nibble - 8) * scale - scaled_zp.
            # scaled_zp = (zp_raw - 8) * scale precomputed at load time.
            szp_ptrs = zp_ptr + offs_n * num_groups + g_idx
            szp = tl.load(szp_ptrs, mask=scale_mask, other=0.0)
            b_fp = (b - ZP_BIAS).to(scales.dtype) * scales[:, None] - szp[:, None]
        elif HAS_ZP:
            # Asymmetric: (nibble - zp_raw) * scale (single subtraction)
            zp_ptrs = zp_ptr + offs_n * num_groups + g_idx
            zp_raw = tl.load(zp_ptrs, mask=scale_mask, other=0.0)
            b_fp = (b.to(scales.dtype) - zp_raw[:, None]) * scales[:, None]
        else:
            # Symmetric: (w - 8) * scale
            b_fp = (b - ZP_BIAS).to(scales.dtype) * scales[:, None]

        # ---- Transpose to [BLOCK_K, BLOCK_N] for matmul ----
        b_fp_t = tl.trans(b_fp)

        # ---- Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] ----
        accumulator += tl.dot(a, b_fp_t, out_dtype=tl.float32)

    # ---- Store output C: [BLOCK_M, BLOCK_N] ----
    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def triton_w4a16_skinny_fmt_gemm(
    a: torch.Tensor,  # [M, K] fp16/bf16
    b_q: torch.Tensor,  # [N, K//8] int32 (ExLlama shuffle packed)
    scales: torch.Tensor,  # [N, K//G] fp16/bf16
    group_size: int,
    zp_bias: int = 8,
    zp: torch.Tensor | None = None,  # [N, K//G] per-group zero-points
    scaled_zp: torch.Tensor | None = None,  # [N, K//G] (zp - 8) * scale
) -> torch.Tensor:
    """
    Fused W4A16 GEMM reading from skinny weight format [N, K//8].

    Args:
        a:          Activation matrix [M, K], float16 or bfloat16.
        b_q:        Packed weight matrix [N, K//8], int32 (ExLlama shuffle).
        scales:     Per-group scales [N, K//G], same dtype as a.
        group_size: Quantization group size (resolved from -1 to K by caller).
        zp_bias:    Constant zero bias (default 8 for unsigned int4).
        zp:         Raw per-group zero-points [N, K//G] (asymmetric),
                    stored as zp_raw in activation dtype. When provided,
                    dequant is (nibble - zp_raw) * scale. Mutually exclusive
                    with `scaled_zp`.
        scaled_zp:  Pre-scaled per-group zero-points [N, K//G], same dtype
                    as a, equal to (zp_raw - 8) * scale. Selects the
                    CK-style dequant path: (nibble - 8) * scale - scaled_zp.
                    Algebraically equivalent to the `zp` path but with
                    different fp rounding.

    Returns:
        Output matrix [M, N], same dtype as a.
    """
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert b_q.is_contiguous(), "Weight matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"
    assert not (zp is not None and scaled_zp is not None), (
        "Pass at most one of zp / scaled_zp"
    )

    M, K = a.shape
    N = b_q.shape[0]
    K8 = K // 8
    num_groups = K // group_size

    assert b_q.shape == (N, K8), f"b_q shape mismatch: {b_q.shape} vs ({N}, {K8})"
    assert scales.shape == (N, num_groups), (
        f"scales shape mismatch: {scales.shape} vs ({N}, {num_groups})"
    )
    if zp is not None:
        assert zp.is_contiguous(), "Zero-points must be contiguous"
        assert zp.shape == (N, num_groups), (
            f"zp shape mismatch: {zp.shape} vs ({N}, {num_groups})"
        )
    if scaled_zp is not None:
        assert scaled_zp.is_contiguous(), "scaled_zp must be contiguous"
        assert scaled_zp.shape == (N, num_groups), (
            f"scaled_zp shape mismatch: {scaled_zp.shape} vs ({N}, {num_groups})"
        )
    has_zp = zp is not None
    has_scaled_zp = scaled_zp is not None
    zp_or_szp = scaled_zp if has_scaled_zp else zp

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    # AMD-specific scheduling hint; only consumed by the HIP backend below
    # (see compiler.py amdgpu-waves-per-eu attribute). Set to 0 by default
    # (no constraint); per-shape branches may override.
    waves_per_eu = 0

    cap = current_platform.get_device_capability()
    if cap is not None and cap.major >= 12:
        # Tuned on gfx1201 (Radeon AI PRO R9700, 32 CUs, 32-wide wavefronts)
        # using Llama-3.1-8B AWQ weight shapes with group_size=128.
        if M <= 32:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 16, 16, 128, 4
        elif M <= 64:
            if K >= 2 * N:  # tall K (e.g. down_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 32, 128, 8
            elif N > K:  # wide N (e.g. qkv_proj, gate_up_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 32, 64, 8
            else:  # N ~= K (e.g. o_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 32, 64, 128, 4
        elif M <= 128:
            if K >= 2 * N:  # tall K (e.g. down_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 16, 64, 1
            elif N >= 2 * K:  # very wide N (e.g. gate_up_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 128, 64, 8
            else:  # N ~= K (e.g. o_proj, qkv_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 64, 64, 8
        elif M <= 512:
            if K >= 2 * N:  # tall K (e.g. down_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 64, 64, 8
            elif N >= 4 * K:  # very wide N (e.g. gate_up_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 128, 64, 8
            else:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 128, 64, 8
        else:
            if K >= 2 * N:  # tall K (e.g. down_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 64, 64, 8
            elif N >= 4 * K:  # very wide N (e.g. gate_up_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 256, 64, 64, 8
            else:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 128, 32, 8
    elif on_gfx1x():
        # Tuned on gfx1151 (Strix Halo, 40 CUs, 32-wide wavefronts)
        # using Qwen3-4B weight shapes with group_size=128.
        # waves_per_eu=0 means no constraint; specific values pin LLVM
        # to a target VGPR budget per occupancy.md (gfx1151 has 1536
        # VGPRs/SIMD; waves_per_eu=N sets max VGPRs to ~1536/N).
        if M <= 32:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 32, 32, 128, 4
        elif M <= 64:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 64, 32, 4
        elif M <= 128:
            # For K >= 4096 AND N >= 4096, a single config (BN=32, BK=128,
            # NW=4) wins on every projection shape across Qwen3-8B and
            # Llama-3.1-8B (down/qkv/gate_up/o_proj all gain +23%..+35% vs
            # prior shape-specific configs). The wider K-tile escapes WMMA
            # latency-bound regime (wmma.md: >= 2 waves/SIMD), and BN=32
            # keeps the workgroup grid large enough to saturate 40 CUs even
            # at N up to ~28k.
            #
            # Small-N or small-K shapes (Qwen3-VL-4B / Qwen3-4B) need the
            # legacy shape-specific configs — at N=2560 the BN=32 grid drops
            # below the saturation point.
            if K >= 4096 and N >= 4096:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 32, 128, 4
                # waves_per_eu=6 matches the natural VGPR-bound occupancy
                # but explicitly pinning the target gives LLVM a single
                # register count to optimize against (compiler.py: "forces
                # LLVM to focus on a single register count, simplifies some
                # heuristics and may improve scheduling"). +5-8% across all
                # 4 K=N=4096 projection shapes.
                waves_per_eu = 6
            elif K >= 2 * N:  # tall K, small-N down (e.g. Qwen3-VL-4B down)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 16, 64, 1
            elif N > K:  # wide N, small K (e.g. Qwen3-VL-4B qkv/gate_up)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 64, 64, 4
            else:  # N ~= K, small K (e.g. Qwen3-VL-4B o_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 32, 64, 4
        elif M <= 1024:
            if K >= 2 * N:  # tall K (e.g. down_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 64, 64, 4
            elif N >= 4 * K:  # very wide N (e.g. gate_up_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 64, 64, 8
            else:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 128, 32, 4
        else:
            if K >= 2 * N:  # tall K (e.g. down_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 512, 32, 16
            else:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 64, 64, 8
    else:
        num_warps = 4
        if M <= 32:
            BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
        elif M <= 64:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    # The kernel loads one scale per BLOCK_K tile, so BLOCK_K must not
    # exceed group_size — otherwise elements in the tile that belong to
    # a different group would get the wrong scale.
    BLOCK_K = min(BLOCK_K, group_size)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _triton_w4a16_skinny_fmt_kernel[grid](
        a,
        b_q,
        scales,
        zp_or_szp if (has_zp or has_scaled_zp) else scales,  # dummy when unused
        c,
        M,
        N,
        K,
        K8,
        num_groups,
        group_size=group_size,
        ZP_BIAS=zp_bias,
        HAS_ZP=has_zp,
        HAS_SCALED_ZP=has_scaled_zp,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        **({"waves_per_eu": waves_per_eu} if waves_per_eu else {}),
    )
    return c


# ---------------------------------------------------------------------------
# Weight packing
# ---------------------------------------------------------------------------


def pack_int4_exllama_shuffle(w_uint4: torch.Tensor) -> torch.Tensor:
    """Pack uint4 values into ExLlama shuffle format: [N, K] -> [N, K//8] int32.

    Each int32 packs 8 K-values with interleave order [0,2,4,6,1,3,5,7].
    """
    N_dim, K_dim = w_uint4.shape
    assert K_dim % 8 == 0
    g = w_uint4.to(torch.uint8).view(N_dim, K_dim // 8, 8).to(torch.int32)
    return (
        g[:, :, 0]
        | (g[:, :, 2] << 4)
        | (g[:, :, 4] << 8)
        | (g[:, :, 6] << 12)
        | (g[:, :, 1] << 16)
        | (g[:, :, 3] << 20)
        | (g[:, :, 5] << 24)
        | (g[:, :, 7] << 28)
    )


# ---------------------------------------------------------------------------
# Hybrid dispatch logic
# ---------------------------------------------------------------------------


def _hybrid_w4a16_apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_q_i32: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
    w_q_ck: torch.Tensor | None = None,
    ck_min_m: int = 0,
    w_scaled_zp_ck: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch between skinny GEMM, CK W4A16 b_scale (sym/asym), and Triton.

    Both skinny and Triton paths read from the same vLLM skinny-format weights:
      w_q:     [N, K//8] int8 (ExLlama shuffle, for skinny kernel)
      w_q_i32: [N, K//8] int32 (same data viewed as int32, for triton)
      w_s:     [N, K//G] fp16/bf16 (skinny-layout scales)
      w_zp:    [N, K//G] raw zero-points (zp_raw) in act dtype,
               or None for symmetric. Both HIP skinny and Triton use this
               single format: dequant = (nibble - zp_raw) * scale.

    AIESW-32176: w_q_ck is the same weights repacked into CK pk_i4 layout
    [K0, N, K1//2] int8. When non-None and M >= ck_min_m the CK
    GEMM kernel is used instead of Triton:
      - symmetric (w_zp is None): ck_w4a16_b_scale_gemm
      - asymmetric (w_zp set, w_scaled_zp_ck = (zp-8)*scale precomputed at
        load time): ck_w4a16_b_scale_zp_gemm

    Registered as a custom op so torch.compile treats it as opaque.
    """
    import vllm._custom_ops as ops

    M = x_2d.shape[0]
    K = x_2d.shape[1]
    N = w_q.shape[0]

    # Use the HIP skinny kernel for small batch sizes (fast decode path),
    # but only when K*M fits in LDS.  Otherwise fall through to Triton.
    if M <= MAX_SKINNY_BATCH_SIZE and K * M <= LDS_CAPACITY_ELEMENTS:
        ctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(f"wvsplitk_int4 {M}x{N}x{K}")
        )
        with ctx:
            return ops.wvSplitK_int4_g(w_q, x_2d, w_s, cu_count, group_size, w_zp, bias)

    # AIESW-32176: CK W4A16 b_scale path (sym or asym), now dispatched through
    # aiter.ops.gemm_w4a16 — a single op covers both routes via its optional
    # scaled_zp argument and the activation dtype of the caller-allocated
    # output. Conditional is inside the custom op so it's opaque to dynamo and
    # the runtime M check is a plain Python int compare against the per-layer
    # min-M threshold. aiter convention is caller-allocates output.
    if w_q_ck is not None and ck_min_m > 0 and ck_min_m <= M and not _ck_disabled():
        ctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(f"ck_w4a16 {M}x{N}x{K}")
        )
        with ctx:
            output: torch.Tensor | None = None
            # VLLM_CK_W4A16_PRE_DEQUANT=1 → route into the (currently
            # STUBBED) PreDequantToLDS=true variant of the aiter op. Read
            # once here so both arms see the same value.
            ck_pdl = _ck_pre_dequant_to_lds()
            # AIESW-32282: bf16 dequant rounding is no longer a runtime
            # axis. CK ships truncate-to-bf16 as the only bf16 behavior
            # (verified statistically indistinguishable from Triton on
            # lm_eval gsm8k); fp16 path is unaffected.
            if w_zp is None:
                # Symmetric (uint4b8 / GPTQ): no zero points.
                from aiter.ops.gemm_w4a16 import gemm_w4a16 as _aiter_gemm_w4a16

                output = torch.empty((M, N), dtype=x_2d.dtype, device=x_2d.device)
                _aiter_gemm_w4a16(
                    x_2d,
                    w_q_ck,
                    w_s,
                    output,
                    group_size,
                    scaled_zp=None,
                    pre_dequant_to_lds=ck_pdl,
                )
            elif w_scaled_zp_ck is not None:
                # Asymmetric (AWQ): scaled_zp = (zp - 8) * scale precomputed at
                # load time and passed through. aiter dispatches the asymmetric
                # CK kernel internally based on scaled_zp being non-None.
                from aiter.ops.gemm_w4a16 import gemm_w4a16 as _aiter_gemm_w4a16

                output = torch.empty((M, N), dtype=x_2d.dtype, device=x_2d.device)
                _aiter_gemm_w4a16(
                    x_2d,
                    w_q_ck,
                    w_s,
                    output,
                    group_size,
                    scaled_zp=w_scaled_zp_ck,
                    pre_dequant_to_lds=ck_pdl,
                )
            # else: asymmetric layer with zp present but scaled_zp not
            # precomputed — fall through to Triton (shouldn't happen if the
            # load-time path is wired, but defensive).
            if output is not None:
                if bias is not None:
                    output.add_(bias)
                return output

    ctx = (
        nullcontext()
        if torch.compiler.is_compiling()
        else torch.profiler.record_function(f"hybrid_triton_w4a16 {M}x{N}x{K}")
    )
    with ctx:
        # A/B knob: when VLLM_TRITON_W4A16_SCALED_ZP=1 and the layer has
        # _hybrid_w_scaled_zp_ck precomputed (i.e. the shape is in the CK
        # target table), feed scaled_zp into the Triton kernel instead of
        # the raw zp. Algebraically equivalent; differs only in fp rounding.
        if w_zp is not None and w_scaled_zp_ck is not None and _triton_use_scaled_zp():
            triton_zp_arg: torch.Tensor | None = None
            triton_scaled_zp_arg: torch.Tensor | None = w_scaled_zp_ck
        else:
            triton_zp_arg = w_zp
            triton_scaled_zp_arg = None
        output = triton_w4a16_skinny_fmt_gemm(
            a=x_2d,
            b_q=w_q_i32,
            scales=w_s,
            group_size=group_size,
            zp=triton_zp_arg,
            scaled_zp=triton_scaled_zp_arg,
        )
        if bias is not None:
            output.add_(bias)
    return output


def _hybrid_w4a16_apply_fake(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_q_i32: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
    w_q_ck: torch.Tensor | None = None,
    ck_min_m: int = 0,
    w_scaled_zp_ck: torch.Tensor | None = None,
) -> torch.Tensor:
    M = x_2d.size(0)
    N = w_q.size(0)
    return torch.empty((M, N), dtype=x_2d.dtype, device=x_2d.device)


direct_register_custom_op(
    op_name="hybrid_w4a16_apply",
    op_func=_hybrid_w4a16_apply_impl,
    mutates_args=[],
    fake_impl=_hybrid_w4a16_apply_fake,
)


class HybridW4A16LinearKernel(MPLinearKernel):
    """Hybrid W4A16 kernel: HIP skinny for decode, Triton for prefill.

    Stores weights once in skinny layout [N, K//8] (ExLlama shuffle packed).
    Both the HIP skinny kernel and the triton kernel read from this single
    weight copy, eliminating the memory overhead of dual weight storage.
    """

    SUPPORTED_QUANT_TYPES = [
        scalar_types.uint4b8,  # symmetric GPTQ (bias=8)
        scalar_types.uint4,  # asymmetric (zero_points)
    ]

    @classmethod
    def get_min_capability(cls) -> int:
        return 110

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "HybridW4A16LinearKernel only targets ROCm"

        # Check HIP skinny op availability
        try:
            if not hasattr(torch.ops, "_rocm_C") or not hasattr(
                torch.ops._rocm_C, "wvSplitK_int4_g"
            ):
                return False, "wvSplitK_int4_g op not available in this build"
        except Exception:
            return False, "ROCm ops not available"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "requires float16 or bfloat16 activations"

        if c.has_g_idx:
            return False, "does not support g_idx reordering"

        gs = c.group_size
        if gs not in SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size {gs} not supported; supported: {SUPPORTED_GROUP_SIZES}",
            )

        K = c.partition_weight_shape[0]
        if K % 16 != 0:
            return False, f"K={K} must be divisible by 16"

        if K % gs != 0:
            return (
                False,
                f"K={K} must be divisible by group_size={gs}",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        w_q_raw = getattr(layer, self.w_q_name)
        w_s_raw = getattr(layer, self.w_s_name)

        # Unpack raw weights and normalize to [N, K] int32
        unpacked = unpack_quantized_values_into_int32(
            w_q_raw.data, c.weight_type, packed_dim=w_q_raw.packed_dim
        )
        # AWQ-converted weights arrive as (K, N) with output_dim=1;
        # compressed-tensors arrive as (N, K) with output_dim=0.
        if getattr(w_q_raw, "output_dim", 0) != 0:
            unpacked = unpacked.t().contiguous()

        # ---- Pack into skinny format: [N, K//8] ExLlama shuffle ----
        shuffled = pack_int4_exllama_shuffle(unpacked)

        # Store as int8 for skinny kernel, keep int32 view for triton kernel
        w_q_skinny_i32 = shuffled.contiguous()
        w_q_skinny = w_q_skinny_i32.view(torch.int8)

        # ---- Prepare skinny scales: normalize to [N, K//G] ----
        permute_param_layout_(w_s_raw, input_dim=1, output_dim=0)
        w_s_skinny = w_s_raw.data.contiguous()

        # ---- Process zero-points for asymmetric quantization ----
        if c.zero_points:
            assert self.w_zp_name is not None
            w_zp_raw = getattr(layer, self.w_zp_name)
            # Normalize zp layout to (N, num_groups)
            permute_param_layout_(w_zp_raw, input_dim=1, output_dim=0, packed_dim=0)
            zp_unpacked = unpack_quantized_values_into_int32(
                w_zp_raw.data, c.weight_type, packed_dim=0
            )
            # zp_unpacked: [N, num_groups] with raw uint4 values [0..15]
            # Store raw zero-points in activation dtype.
            # Both kernels dequant as (nibble - zp_raw) * scale.
            w_zp = zp_unpacked.to(c.act_type).contiguous()
            self._transform_param(layer, self.w_zp_name, lambda x: w_zp)

        # ---- Store on layer ----
        # Replace w_q with skinny int8 (primary weights for skinny kernel)
        self._transform_param(layer, self.w_q_name, lambda x: w_q_skinny)
        # Replace w_s with skinny scales
        self._transform_param(layer, self.w_s_name, lambda x: w_s_skinny)

        # Store int32 view for triton kernel
        layer.register_parameter(
            "_hybrid_w_q_i32",
            torch.nn.Parameter(w_q_skinny_i32, requires_grad=False),
        )

        # AIESW-32176: precompute CK b_scale layout if this layer's (N, K, group,
        # dtype) matches a registered CK target shape on gfx1151. Done once at
        # load time with regular Python ints (not SymInts), so the lookup is safe
        # outside the dynamo trace. aiter.ops.gemm_w4a16 covers both symmetric
        # and asymmetric (per-group zero-point) paths via the same single op,
        # so a single availability check (cached at module import) suffices.
        # Note: precompute runs even when _ck_disabled() is True so that the
        # Triton fallback can opt into the scaled_zp formulation via
        # VLLM_TRITON_W4A16_SCALED_ZP=1 for A/B benchmarking. Dispatch at
        # apply time still gates on _ck_disabled() so the runtime path is
        # unchanged for default users.
        if _HAS_AITER_W4A16_OP:
            N = w_q_skinny_i32.shape[0]
            K = w_q_skinny_i32.shape[1] * 8
            target = _lookup_ck_target(N, K, c.group_size, c.act_type)
            if target is not None:
                min_M, kperblock = target
                w_q_ck = _repack_vllm_to_ck_b_scale(w_q_skinny_i32, kperblock)
                layer.register_parameter(
                    "_hybrid_w_q_ck",
                    torch.nn.Parameter(w_q_ck, requires_grad=False),
                )
                # Plain Python int — safe to compare against SymInt M at apply.
                layer._hybrid_ck_min_M = int(min_M)

                if c.zero_points:
                    # AIESW-32176: precompute scaled_zp = (zp - 8) * scale.
                    # zp is stored on the layer post-process as raw fp16 in
                    # act dtype (see the c.zero_points block above). scale
                    # here is w_s_skinny [N, K/G]. Result shape matches.
                    w_zp_raw = getattr(layer, self.w_zp_name).data
                    scaled_zp = (
                        (
                            (w_zp_raw.to(torch.float32) - 8.0)
                            * w_s_skinny.to(torch.float32)
                        )
                        .to(c.act_type)
                        .contiguous()
                    )
                    layer.register_parameter(
                        "_hybrid_w_scaled_zp_ck",
                        torch.nn.Parameter(scaled_zp, requires_grad=False),
                    )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.platform_utils import num_compute_units

        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)
        w_q_i32 = layer._hybrid_w_q_i32

        x_2d = x.reshape(-1, x.shape[-1])
        N = w_q.shape[0]
        out_shape = x.shape[:-1] + (N,)

        # AIESW-32176: pass CK-format weights + min M (and scaled_zp for
        # asymmetric) to the custom op if registered for this layer. Dispatch
        # decision happens INSIDE the custom op (opaque to dynamo).
        w_q_ck = getattr(layer, "_hybrid_w_q_ck", None)
        ck_min_m = getattr(layer, "_hybrid_ck_min_M", 0)
        w_scaled_zp_ck = getattr(layer, "_hybrid_w_scaled_zp_ck", None)

        cu_count = num_compute_units()
        output = torch.ops.vllm.hybrid_w4a16_apply(
            x_2d,
            w_q,
            w_s,
            w_q_i32,
            w_zp,
            bias,
            cu_count,
            c.group_size,
            w_q_ck,
            ck_min_m,
            w_scaled_zp_ck,
        )
        return output.reshape(out_shape)
