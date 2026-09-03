# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Hybrid W4A16 kernel: Triton for prefill, HIP skinny for decode.

Routes based on batch size M:
  M <= MAX_SKINNY_BATCH_SIZE: HIP skinny GEMM (wvSplitK_int4_g)
  M > MAX_SKINNY_BATCH_SIZE:  Triton W4A16 fused dequant GEMM

Stores the weights ONCE as int8 [N, K//2] (ExLlama shuffle packed). Both
paths read this single buffer: the HIP skinny kernel uses it directly, and
the triton kernel reinterprets it as int32 [N, K//8] via a view (and
transposes tiles in-register). No dual weight storage.
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
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

SUPPORTED_GROUP_SIZES = [32, 64, 128]


def _on_gfx12x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx12x

    return on_gfx12x()


def _on_gfx1x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1x

    return on_gfx1x()


def _on_gfx1151() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1151

    return on_gfx1151()


# Maximum batch size M for the HIP skinny kernel path (C++ supports N_in
# up to 5).  When M is below this AND K*M fits in LDS, the skinny kernel is
# used; otherwise the Triton prefill path handles the GEMM.
MAX_SKINNY_BATCH_SIZE = 5
# 64 KiB per-workgroup LDS limit expressed in fp16 elements.
# (AMD RDNA has 128 KiB total LDS per CU, but 64 KiB per workgroup.)
LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2  # 32768 fp16 elements


# ---------------------------------------------------------------------------
# Triton kernel for the prefill path (reads skinny-format weights [N, K//8])
# ---------------------------------------------------------------------------


@triton.jit
def _int4_pair_to_fp16x2(x):
    """Unpack two packed int4 nibbles into a uint32 holding two fp16 lanes,
    each equal to 1024 + nibble, with one ``v_and_or_b32``
    (``(x & 0x000F000F) | 0x64006400``).

    OR-ing a 4-bit nibble into the low mantissa of fp16 1024.0 (0x6400)
    bitcasts to exactly 1024+n. Doing it on a full 32-bit lane dequants two
    nibbles per instruction, vs the scalar v_and_b16 + v_or_b16 pair Triton
    emits from the elementwise form.
    """
    mask = tl.full(x.shape, 0x000F000F, tl.int32)
    return tl.inline_asm_elementwise(
        asm="v_and_or_b32 $0, $1, $2, 0x64006400",
        constraints="=v,v,v",
        args=[x, mask],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _triton_w4a16_skinny_fmt_kernel(
    # Pointers
    a_ptr,  # [M, K]  fp16/bf16 activations
    b_ptr,  # [N, K//8]  int32 packed (ExLlama shuffle, K is packed dim)
    scales_ptr,  # [N, K//G]  fp16/bf16 scales (sym path, HAS_ZP=False)
    packed_scale_zp_ptr,  # [N, K//G]  int32 scale/zp carrier (asym, HAS_ZP)
    c_ptr,  # [M, N]  fp16/bf16 output
    # Dimensions
    M,
    N,
    K,
    K8,  # K // 8
    num_groups,  # K // group_size
    # Quantization parameters
    group_size,
    HAS_ZP: tl.constexpr,  # asym: read the scale/zp carrier; sym: scales + (-8)
    PACKED_DEQUANT: tl.constexpr,  # one v_and_or_b32 per nibble pair (fp16 only)
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

    Two dequant paths, chosen by the layer's sym/asym nature:
      - HAS_ZP=True (asymmetric): read the carrier ``packed_scale_zp_ptr``
        [N, K//G] (one fp32 per (n, group)) -- it folds the per-group scale AND
        the zero-point offset into a single load, replacing the separate scale +
        zp loads. Layout: fp16 = scale | bias_eff (= -8*scale - scaled_zp),
        dequant (nibble-1024)*scale + bias_eff via the magic-const fp16 unpack;
        bf16 = scale | zp_int, dequant (nibble - zp_int)*scale.
      - HAS_ZP=False (symmetric): the -8 offset is a constant, so there is
        no second load to fold -- read ``scales_ptr`` directly and subtract the
        constant 8. fp16: (nibble - 1032)*scale via the magic unpack; bf16:
        (nibble - 8)*scale. (No carrier overhead for the sym fast path.)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ExLlama unshuffle shifts: shift[j] = (j//2)*4 + (j%2)*16
    # For 8 values: [0, 16, 4, 20, 8, 24, 12, 28]
    exllama_shifts_row = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    shifts_1d = tl.reshape(
        tl.broadcast_to(exllama_shifts_row[None, :], (BLOCK_K // 8, 8)),
        (BLOCK_K,),
    )
    shifts_full = tl.broadcast_to(shifts_1d[None, :], (BLOCK_N, BLOCK_K))

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        offs_k8 = k_start * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)
        b_ptrs = b_ptr + offs_n[:, None] * K8 + offs_k8[None, :]
        mask_b = (offs_n[:, None] < N) & (offs_k8[None, :] < K8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        if PACKED_DEQUANT:
            # The ExLlama int32 holds the paired nibbles val[2p] @ bits[4p:4p+4]
            # and val[2p+1] @ bits[16+4p:20+4p], so for pre-shift 4p (p=0..3),
            #   (x >> 4p) & 0x000F000F | 0x64006400
            # is one v_and_or_b32 producing a half2 = (1024+val[2p],
            # 1024+val[2p+1]) in K order (signed shift is fine: the sign fill
            # lands above bit 20 and is masked out). The interleave(lo, hi) lays
            # b_raw out as half2 so the downstream affine also packs into
            # v_pk_fma_f16. The dequant inner loop is VALU-issue-bound on gfx11,
            # so this ~halves the dequant instruction count per WMMA.
            shifts4 = (tl.arange(0, 4) * 4)[None, None, :]
            bp_shift = tl.reshape(
                b_packed[:, :, None] >> shifts4, (BLOCK_N, BLOCK_K // 2)
            )
            packed_hl = _int4_pair_to_fp16x2(bp_shift)  # u32 half2: 1024+nibble
            lo = (packed_hl & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
            hi = (packed_hl >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
            b_raw = tl.interleave(lo, hi)  # [BLOCK_N, BLOCK_K] fp16 = 1024+nibble
        else:
            # ExLlama unshuffle: replicate each int32 8x then per-lane shift+mask.
            b = tl.interleave(b_packed, b_packed)
            b = tl.interleave(b, b)
            b = tl.interleave(b, b)
            b = (b >> shifts_full) & 0xF  # [BLOCK_N, BLOCK_K]

        g_idx = (k_start * BLOCK_K) // group_size
        scale_mask = offs_n < N

        if HAS_ZP:
            # Asymmetric: one fp32 per group folds the scale and the zero point.
            psz = tl.load(
                packed_scale_zp_ptr + offs_n * num_groups + g_idx,
                mask=scale_mask,
                other=0,
            )
            psz_u = psz.to(tl.uint32, bitcast=True)
            if a.dtype == tl.float16:
                # low16 = scale, high16 = bias_eff (= -8*scale - scaled_zp).
                # ONE fp16 FMA per group via the magic-constant i4->fp16 unpack.
                scale = (psz_u & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
                bias_eff = (psz_u >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
                if not PACKED_DEQUANT:
                    b_raw = (b | 0x6400).to(tl.uint16).to(tl.float16, bitcast=True)
                c1024 = tl.full((), 1024.0, tl.float16)
                b_fp = (b_raw - c1024) * scale[:, None] + bias_eff[:, None]
            else:
                # bf16: low16 = scale, high16 = zp_int. Cheap int-domain subtract
                # before the single bf16 multiply (RDNA3 has no v_pk_fma_bf16).
                scale = (psz_u & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
                zp_int = ((psz_u >> 16) & 0xFFFF).to(b.dtype)
                b_fp = (b - zp_int[:, None]).to(scale.dtype) * scale[:, None]
        else:
            # Symmetric: the -8 offset is constant (no zp to fold), so read the
            # scale directly -- no carrier overhead.
            scales = tl.load(
                scales_ptr + offs_n * num_groups + g_idx, mask=scale_mask, other=1.0
            )
            if a.dtype == tl.float16:
                # (nibble - 8) * scale == (b_raw - (1024+8)) * scale, via magic.
                if not PACKED_DEQUANT:
                    b_raw = (b | 0x6400).to(tl.uint16).to(tl.float16, bitcast=True)
                c1032 = tl.full((), float(1024 + 8), tl.float16)
                b_fp = (b_raw - c1032) * scales[:, None]
            else:
                # bf16: (nibble - 8) * scale, int subtract before the cast.
                b_fp = (b - 8).to(scales.dtype) * scales[:, None]

        b_fp_t = tl.trans(b_fp)
        accumulator += tl.dot(a, b_fp_t, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


# Per-shape (group_size, K, N) -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps,
# num_stages) tile-config overrides for prefill (M <= 128) on gfx1151. Applies
# to the SCALAR (bf16) dequant path only; the packed fp16 path is tuned by the
# ladder in _select_skinny_gfx1151_config and needs no per-shape entries.
# Picked by sweeping benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py + a
# per-config sweep script; only added when better than the generic heuristic
# by > 20% at M=128. Re-run benchmarks after edits.
_GFX1151_BF16_PREFILL_OVERRIDES: dict[
    tuple[int, int, int], tuple[int, int, int, int, int]
] = {
    # SmolLM2-1.7B-Instruct-AWQ (gs=32, K=2048; gs forces BLOCK_K to 32 so
    # widen BLOCK_M and let Triton pipeline 4 stages to amortize the small
    # K-tile).
    (32, 2048, 6144): (128, 32, 32, 4, 4),  # qkv_proj
    (32, 2048, 2048): (128, 32, 32, 4, 4),  # o_proj
    (32, 2048, 16384): (128, 32, 32, 4, 4),  # gate_up_proj
    (32, 8192, 2048): (128, 64, 32, 8, 2),  # down_proj
    # Qwen3-8B-quantized.w4a16 (gs=128, K=4096 / 12288). For these K's a
    # full BLOCK_K=group_size with no software pipelining beats the generic
    # 64x64x64 — single-stage keeps register pressure down.
    (128, 4096, 6144): (128, 64, 128, 8, 1),  # qkv_proj
    (128, 4096, 4096): (128, 32, 128, 4, 1),  # o_proj
    (128, 4096, 24576): (64, 32, 128, 2, 1),  # gate_up_proj
    (128, 12288, 4096): (128, 64, 128, 8, 1),  # down_proj
}


# Explicit gfx1151 prefill tile selection -- DTYPE-AWARE. The kernel takes the
# packed v_and_or/v_pk_fma dequant for fp16 and the scalar dequant for bf16, and
# the two paths want different tiles (most visibly BLOCK_N at deep M: 256 for
# packed fp16 vs 64 for scalar bf16).
#
# fp16 (packed) -- tuned under do_bench_cudagraph with rotating cold weights
# over a broad shape catalog:
#   * M <= 16: BLOCK_M=16 (more M-tiles fill the CUs at tiny M).
#   * 17..64: BLOCK_M=32; small BLOCK_N keeps the grid large (a wide BLOCK_N
#     leaves only ceil(N/BN) workgroups -- an M-blind BLOCK_N=256 was a 1.6-3x
#     regression here). Square mid shapes take BLOCK_N=128/BLOCK_K=64.
#   * 65..256: square -> BLOCK_N=128 (BLOCK_K=32 nw=8 at M>=128); tall -> 128.
#   * 257..2047: the wide distilled BLOCK_N=256/BLOCK_M=128 tile.
#   * M >= 2048: distilled BLOCK_N=256; BLOCK_M=64 for narrow+deep K (N<=2048
#     and K>=4096), else 128.
#
# bf16 (scalar) -- keeps the pre-existing scalar-tuned ladder plus its per-shape
# overrides unchanged, so bf16 holds parity: the packed fp16 table regresses bf16
# by up to ~40% at deep M, where scalar bf16 wants BLOCK_N=64, not 256. The only
# bf16 behaviour change is num_stages=1 on shapes the override table does not
# cover (it already pinned num_stages on the ones it does).
#
# BLOCK_K is capped to group_size so a K-block never straddles a quant group
# (scale aliasing); gs=128 -- the bulk -- passes the table BLOCK_K through.
def _select_skinny_gfx1151_config(
    M: int, N: int, K: int, group_size: int, dtype: torch.dtype
) -> tuple[int, int, int, int, int | None]:
    """Return (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for gfx1151."""
    # >1 stage regresses badly for this kernel (it has no software pipeline);
    # only the bf16 override table opts back out.
    num_stages: int | None = 1
    if dtype == torch.float16:
        tall = K >= 2 * N  # tall-K (down_proj-like)
        # Very wide N with small K (e.g. gemma gate_up 32768x2048): memory-bound,
        # wants the small square tile at tiny M, not BLOCK_M=16.
        vwide_smallk = N >= 8192 and K <= 2048
        if M <= 16:  # BLOCK_M=16: more M-tiles fill the CUs at tiny M
            if N <= 1024 or vwide_smallk:
                block_m, block_n, block_k, num_warps = 32, 32, 128, 4
            else:
                block_m, block_n, block_k, num_warps = 16, 64, 128, 4
        elif M <= 32:
            if vwide_smallk:
                block_m, block_n, block_k, num_warps = 32, 32, 128, 4
            else:
                block_m, block_n, block_k, num_warps = 32, 64, 128, 4
        elif M <= 64:
            if tall or N >= 4 * K:  # tall or very wide
                block_m, block_n, block_k, num_warps = 32, 64, 128, 4
            else:  # square mid
                block_m, block_n, block_k, num_warps = 32, 128, 64, 4
        elif M <= 128:
            if tall:
                block_m, block_n, block_k, num_warps = 32, 128, 64, 4
            elif N >= 32768 and K <= 2048:
                # Extremely wide + tiny K (e.g. gemma gate_up 32768x2048):
                # BLOCK_N=128 collapses to 0.6x, needs 64.
                block_m, block_n, block_k, num_warps = 128, 64, 64, 8
            elif N >= 16384:  # very wide N (K>2048): BLOCK_N=128 wins
                block_m, block_n, block_k, num_warps = 128, 128, 32, 8
            elif K <= 2048:  # small-K square needs BLOCK_K=128
                block_m, block_n, block_k, num_warps = 32, 64, 128, 4
            else:  # larger square
                block_m, block_n, block_k, num_warps = 64, 128, 32, 4
        elif M <= 256:
            block_m, block_n, block_k, num_warps = 128, 128, 32, 8
        elif M < 2048:  # 257..2047 (mostly 512, 1024): wide distilled tile
            block_m, block_n, block_k, num_warps = 128, 256, 32, 8
        else:  # M >= 2048 (deep prefill)
            if N <= 2048 and K >= 4096:  # narrow + deep: halved BM saturates
                block_m, block_n, block_k, num_warps = 64, 256, 32, 8
            else:
                block_m, block_n, block_k, num_warps = 128, 256, 32, 8
        # Very narrow N at small/mid M: a wide BLOCK_N leaves too few N-tiles to
        # fill the CUs, so clamp it. At M>=1024 the M-tiles already saturate.
        if N <= 1024 and M <= 512:
            block_n = min(block_n, 32)
    else:
        # Scalar-dequant path (bf16): the pre-existing scalar-tuned ladder.
        key = (group_size, K, N)
        override = _GFX1151_BF16_PREFILL_OVERRIDES.get(key) if M <= 128 else None
        if override is not None:
            block_m, block_n, block_k, num_warps, num_stages = override
        elif M <= 32:
            block_m, block_n, block_k, num_warps = 32, 32, 128, 4
        elif M <= 64:
            block_m, block_n, block_k, num_warps = 64, 64, 32, 4
        elif M <= 128:
            if K >= 2 * N:  # tall K (down_proj)
                block_m, block_n, block_k, num_warps = 64, 16, 64, 1
            elif N > K:  # wide N (qkv / gate_up)
                block_m, block_n, block_k, num_warps = 64, 64, 64, 4
            else:  # N ~= K (o_proj)
                block_m, block_n, block_k, num_warps = 64, 32, 64, 4
        elif M <= 1024:
            if K >= 2 * N:  # tall K (down_proj)
                block_m, block_n, block_k, num_warps = 64, 64, 64, 4
            elif N >= 4 * K:  # very wide N (gate_up)
                block_m, block_n, block_k, num_warps = 128, 64, 64, 8
            else:
                block_m, block_n, block_k, num_warps = 64, 128, 32, 4
        else:  # M > 1024
            if K >= 2 * N:  # tall K (down_proj)
                block_m, block_n, block_k, num_warps = 128, 512, 32, 16
            else:
                block_m, block_n, block_k, num_warps = 128, 64, 64, 8
    return block_m, block_n, min(block_k, group_size), num_warps, num_stages


def triton_w4a16_skinny_fmt_gemm(
    a: torch.Tensor,  # [M, K] fp16/bf16
    b_q: torch.Tensor,  # [N, K//8] int32 (ExLlama shuffle packed)
    scales: torch.Tensor,  # [N, K//G] fp16/bf16 (used for the symmetric path)
    group_size: int,
    packed_scale_zp: torch.Tensor | None = None,  # [N, K//G] fp32 (asym only)
    packed_dequant: bool | None = None,
) -> torch.Tensor:
    """
    Fused W4A16 GEMM reading from skinny weight format [N, K//8].

    Asymmetric layers pass ``packed_scale_zp`` (the carrier folding scale and
    zero point into one load); symmetric layers leave it None and the kernel
    reads ``scales`` directly with a constant -8 offset (no carrier overhead --
    sym has no second load to fold).

    Args:
        a:          Activation matrix [M, K], float16 or bfloat16.
        b_q:        Packed weight matrix [N, K//8], int32 (ExLlama shuffle).
        scales:     Per-group scales [N, K//G], same dtype as a (symmetric path).
        group_size: Quantization group size (resolved from -1 to K by caller).
        packed_scale_zp: Packed scale/zp carrier [N, K//G] fp32 for asymmetric
                    layers; the layout is dtype-specific (fp16: scale|bias_eff,
                    bf16: scale|zp_int) -- see the kernel docstring. When None,
                    the symmetric path is used.
        packed_dequant: Override for the one-v_and_or_b32-per-nibble-pair
                    dequant. Defaults to fp16 activations on gfx1151, the only
                    combination it is validated and tuned for. Tests pass False
                    to force the scalar unpack; both produce the same values.

    Returns:
        Output matrix [M, N], same dtype as a.
    """
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert b_q.is_contiguous(), "Weight matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    N = b_q.shape[0]
    K8 = K // 8
    num_groups = K // group_size

    assert b_q.shape == (N, K8), f"b_q shape mismatch: {b_q.shape} vs ({N}, {K8})"
    assert scales.shape == (N, num_groups), (
        f"scales shape mismatch: {scales.shape} vs ({N}, {num_groups})"
    )
    has_zp = packed_scale_zp is not None
    if packed_scale_zp is not None:
        assert packed_scale_zp.is_contiguous(), "packed_scale_zp must be contiguous"
        assert packed_scale_zp.shape == (N, num_groups), (
            f"packed_scale_zp shape mismatch: {packed_scale_zp.shape} "
            f"vs ({N}, {num_groups})"
        )
        packed_scale_zp_i32 = packed_scale_zp.view(torch.int32)
    else:
        packed_scale_zp_i32 = scales  # dummy pointer (unused when HAS_ZP=False)

    # The 1024+n magic trick needs fp16's mantissa, and the packed form is only
    # validated and tuned on gfx1151. Everything else takes the scalar unpack.
    # fp16 is a hard requirement, not a tuning choice, so it also bounds the
    # explicit override.
    packed_dequant = a.dtype == torch.float16 and (
        _on_gfx1151() if packed_dequant is None else packed_dequant
    )

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    # num_stages stays None unless the tile table sets it, so the generic
    # heuristics fall back to Triton's default pipeline depth.
    num_stages: int | None = None
    if _on_gfx12x():
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
    elif _on_gfx1151():
        # gfx1151 (Strix Halo, 40 CUs, 32-wide wavefronts): per-(M, N, K) tile
        # config from the dtype-aware table, since the packed fp16 dequant and
        # the scalar bf16 dequant want different tiles. See
        # _select_skinny_gfx1151_config; re-run
        # benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py after edits.
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = (
            _select_skinny_gfx1151_config(M, N, K, group_size, a.dtype)
        )
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

    extra_kwargs = {} if num_stages is None else {"num_stages": num_stages}
    _triton_w4a16_skinny_fmt_kernel[grid](
        a,
        b_q,
        scales,
        packed_scale_zp_i32,
        c,
        M,
        N,
        K,
        K8,
        num_groups,
        group_size=group_size,
        HAS_ZP=has_zp,
        PACKED_DEQUANT=packed_dequant,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        **extra_kwargs,
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


def _build_packed_scale_zp(
    scales: torch.Tensor,  # [N, K//G] act dtype
    zp_raw: torch.Tensor,  # [N, K//G] int32, raw nibbles 0..15
    act_type: torch.dtype,
) -> torch.Tensor:
    """Fold the per-group scale and zero point into one fp32 per (n, group).

    Replaces the Triton prefill path's two per-group loads with one. Built for
    asymmetric layers only -- symmetric ones subtract the constant 8, so there
    is no second load to fold and the carrier would be pure overhead.

    Layout, matching the kernel's HAS_ZP dequant:
      fp16: low16 = scale, high16 = bias_eff = -(8*scale + (zp-8)*scale), i.e.
            -zp*scale. Consumed as one FMA with the magic-constant i4->fp16
            unpack: (b_raw - 1024)*scale + bias_eff == (nibble - zp)*scale.
      bf16: low16 = scale bits, high16 = the raw zp 0..15 as a plain integer,
            consumed by an int-domain subtract (RDNA3 has no v_pk_fma_bf16).
            Bit-identical to separate scale and zp loads.
    """
    scale_u16 = scales.contiguous().view(torch.uint16).to(torch.int32) & 0xFFFF
    if act_type == torch.float16:
        s_f32 = scales.to(torch.float32)
        bias_eff = (-(8.0 * s_f32 + (zp_raw.to(torch.float32) - 8.0) * s_f32)).to(
            act_type
        )
        hi_u16 = bias_eff.contiguous().view(torch.uint16).to(torch.int32) & 0xFFFF
    else:
        hi_u16 = zp_raw.to(torch.int32) & 0xFFFF  # raw zp 0..15
    return ((hi_u16 << 16) | scale_u16).view(torch.float32).contiguous()


# ---------------------------------------------------------------------------
# Hybrid dispatch logic
# ---------------------------------------------------------------------------


def _rdna_hybrid_w4a16_apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
    packed_scale_zp: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch between skinny GEMM and Triton based on batch size M.

    ``packed_scale_zp`` is the [N, K//G] fp32 carrier folding the per-group
    scale and the zero point into a single load. Asymmetric layers build it at
    load time and the Triton prefill path reads it instead of two separate
    loads; symmetric layers pass None (no second load to fold). The HIP skinny
    decode path always reads ``w_zp``.
    """
    import vllm._custom_ops as ops

    M = x_2d.shape[0]
    K = x_2d.shape[1]
    N = w_q.shape[0]

    # Profiler label suffix. The GEMM's memory traffic is not determined by the
    # shape alone: the per-group scale (and, when asymmetric, zero-point)
    # tensors add N * (K/group_size) * 2 bytes each on top of the N*K/2 weight
    # bytes. At g=32 asymmetric that surcharge is ~25% of the weight bytes, so a
    # trace reporting only MxNxK cannot be turned into a bandwidth number. Use
    # the same `key=value` spelling as the other quantized GEMM scopes.
    gz = f"g={group_size} {'asym' if w_zp is not None else 'sym'}"

    if M <= MAX_SKINNY_BATCH_SIZE and K * M <= LDS_CAPACITY_ELEMENTS:
        # record_function is not torch.compile-safe; use nullcontext when
        # compiling to keep the op traceable.
        ctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(f"wvsplitk_int4 {M}x{N}x{K} {gz}")
        )
        with ctx:
            return ops.wvSplitK_int4_g(w_q, x_2d, w_s, cu_count, group_size, w_zp, bias)

    ctx = (
        nullcontext()
        if torch.compiler.is_compiling()
        else torch.profiler.record_function(f"hybrid_triton_w4a16 {M}x{N}x{K} {gz}")
    )
    with ctx:
        output = triton_w4a16_skinny_fmt_gemm(
            a=x_2d,
            b_q=w_q.view(torch.int32),
            scales=w_s,
            group_size=group_size,
            packed_scale_zp=packed_scale_zp,
        )
        if bias is not None:
            output.add_(bias)
    return output


def _rdna_hybrid_w4a16_apply_fake(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
    packed_scale_zp: torch.Tensor | None = None,
) -> torch.Tensor:
    M = x_2d.size(0)
    N = w_q.size(0)
    return torch.empty((M, N), dtype=x_2d.dtype, device=x_2d.device)


direct_register_custom_op(
    op_name="rdna_hybrid_w4a16_apply",
    op_func=_rdna_hybrid_w4a16_apply_impl,
    mutates_args=[],
    fake_impl=_rdna_hybrid_w4a16_apply_fake,
)


class RDNAHybridW4A16LinearKernel(MPLinearKernel):
    """Hybrid W4A16 kernel: HIP skinny for decode, Triton for prefill.

    Stores the weights once as int8 [N, K//2] (ExLlama shuffle packed). The
    HIP skinny kernel reads it directly; the triton kernel reinterprets the
    same buffer as int32 [N, K//8] via a view, so there is no dual weight
    storage.
    """

    SUPPORTED_QUANT_TYPES = [
        scalar_types.uint4b8,  # symmetric GPTQ (bias=8)
        scalar_types.uint4,  # asymmetric (zero_points)
    ]

    @classmethod
    def get_min_capability(cls) -> int:
        # Arch filtering is handled by can_implement (_on_gfx1x check)
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNAHybridW4A16LinearKernel only targets ROCm"

        if not _on_gfx1x():
            return False, "RDNAHybridW4A16LinearKernel only targets gfx11/gfx12"

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

        unpacked = unpack_quantized_values_into_int32(
            w_q_raw.data, c.weight_type, packed_dim=w_q_raw.packed_dim
        )
        # AWQ weights arrive as (K, N) with output_dim=1;
        # compressed-tensors arrive as (N, K) with output_dim=0.
        if getattr(w_q_raw, "output_dim", 0) != 0:
            unpacked = unpacked.t().contiguous()

        shuffled = pack_int4_exllama_shuffle(unpacked)
        # Store as int8; Triton reinterprets via .view(torch.int32) at apply time.
        w_q_skinny = shuffled.contiguous().view(torch.int8)

        permute_param_layout_(w_s_raw, input_dim=1, output_dim=0)
        w_s_skinny = w_s_raw.data.contiguous()

        if c.zero_points:
            assert self.w_zp_name is not None
            w_zp_raw = getattr(layer, self.w_zp_name)
            permute_param_layout_(w_zp_raw, input_dim=1, output_dim=0, packed_dim=0)
            zp_unpacked = unpack_quantized_values_into_int32(
                w_zp_raw.data, c.weight_type, packed_dim=0
            )
            w_zp = zp_unpacked.to(c.act_type).contiguous()
            self._transform_param(layer, self.w_zp_name, lambda x: w_zp)
            layer.register_parameter(
                "_hybrid_w_packed_scale_zp",
                torch.nn.Parameter(
                    _build_packed_scale_zp(w_s_skinny, zp_unpacked, c.act_type),
                    requires_grad=False,
                ),
            )

        self._transform_param(layer, self.w_q_name, lambda x: w_q_skinny)
        self._transform_param(layer, self.w_s_name, lambda x: w_s_skinny)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.platform_utils import num_compute_units

        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)
        # Fused scale/zero-point carrier; asymmetric layers only (None for sym).
        packed_scale_zp = getattr(layer, "_hybrid_w_packed_scale_zp", None)

        x_2d = x.reshape(-1, x.shape[-1])
        N = w_q.shape[0]
        out_shape = x.shape[:-1] + (N,)

        cu_count = num_compute_units()
        output = torch.ops.vllm.rdna_hybrid_w4a16_apply(
            x_2d,
            w_q,
            w_s,
            w_zp,
            bias,
            cu_count,
            c.group_size,
            packed_scale_zp,
        )
        return output.reshape(out_shape)
