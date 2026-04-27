// SPDX-License-Identifier: Apache-2.0
// MoE megakernel (HIP/RDNA3) — fused GEMM2 + topk-weighted reduce for the
// M=1 decode case of AWQ-int4 quantized MoE.
//
// Status: EXPERIMENTAL — gated by VLLM_MOE_HIP_MEGAKERNEL=1.
//
// Scope (this file): one HIP __global__ kernel that does, per layer call:
//
//   for slot in [0, top_k):
//     expert_id = topk_ids[slot]
//     for m in [0, M=K_hidden) striped across workgroups:
//       acc = sum_k act[slot, k] * dequant(W2[expert_id, m, k])
//       out_partial[slot, m] = topk_weights[slot] * acc
//   # then a tiny atomic-reduce of out_partial[:, m] -> out[m]
//
// We reuse the wvSplitK_int4 inner loop (DOT2C macro, marlin-style bf16
// dequant via bf16x2_dequant_sub_finite). The fusion replaces:
//   (1) fused_moe_wvSplitK_int4_gemm  GEMM2 launch
//   (2) the moe_unpermute kernel
// with a single launch, eliminating one kernel boundary and the inv_perm
// scatter.
//
// We keep the simpler scope (post-router fusion) instead of a 7-stage
// persistent kernel because:
//   1. The grid-wide barrier via atomic counter on RDNA3 is risky
//      without a true cooperative-launch API; deadlock if grid size
//      exceeds resident wave count.
//   2. The per-expert GEMM1 path uses a different grid layout
//      (CuCount × num_expert_blocks) than GEMM2 (CuCount × top_k for the
//      scattered case), so a unified persistent kernel would require
//      stage-specific WG-id remapping that is hard to get right
//      first-pass.
//   3. The existing GEMM1 + silu chain is already fairly tight; the
//      biggest residual launch overhead in the chain is GEMM2 + unpermute
//      (two launches that operate on the same data).
//
// If empirics show this regresses, the env-var gate keeps it off-default.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

#include "../cuda_compat.h"
#include "dispatch_utils.h"

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif

#if defined(__GFX11__) || defined(__GFX12__)
  #define __HIP__GFX1X__
#endif

namespace {

// ---------------------------------------------------------------------------
// Re-derived helper bits (kept local to this file to avoid header churn).
// Identical semantics to the versions in skinny_gemms_int4.cu.
// ---------------------------------------------------------------------------

template <typename T>
struct scalar {};

template <>
struct scalar<c10::Half> {
  using type = half;
};

template <>
struct scalar<c10::BFloat16> {
  using type = __hip_bfloat16;
};

template <typename T>
__device__ __forceinline__ float __s2float(T v);

template <>
__device__ __forceinline__ float __s2float(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float __s2float(__hip_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T __float2s(float v);

template <>
__device__ __forceinline__ half __float2s(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ __hip_bfloat16 __float2s(float v) {
  return __float2bfloat16(v);
}

__device__ __forceinline__ uint32_t
bf16x2_dequant_sub_finite(uint32_t a_bits, uint32_t bias_bits) {
  float a_lo = __uint_as_float((a_bits & 0xFFFFu) << 16);
  float a_hi = __uint_as_float(a_bits & 0xFFFF0000u);
  float b_lo = __uint_as_float((bias_bits & 0xFFFFu) << 16);
  float b_hi = __uint_as_float(bias_bits & 0xFFFF0000u);
  float r_lo = a_lo - b_lo;
  float r_hi = a_hi - b_hi;
  uint32_t lo_bits = __float_as_uint(r_lo);
  uint32_t hi_bits = __float_as_uint(r_hi);
  uint32_t lo_round = lo_bits + 0x7FFFu + ((lo_bits >> 16) & 1u);
  uint32_t hi_round = hi_bits + 0x7FFFu + ((hi_bits >> 16) & 1u);
  return (lo_round >> 16) | (hi_round & 0xFFFF0000u);
}

template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
  return __builtin_nontemporal_load(addr);
}

#define DOT2C(V0, V2, V3)                                                   \
  if constexpr (std::is_same_v<scalar_t, half>) {                           \
    V0 = __builtin_amdgcn_fdot2(*((half2*)(&(V2))), *((half2*)(&(V3))), V0, \
                                false);                                     \
  } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {          \
    typedef short __attribute__((ext_vector_type(2))) bf16x2_t;             \
    V0 = __builtin_amdgcn_fdot2_f32_bf16(*((bf16x2_t*)(&(V2))),             \
                                         *((bf16x2_t*)(&(V3))), V0, false); \
  }

}  // namespace

// ---------------------------------------------------------------------------
// Megakernel: fused GEMM2 + topk-weighted reduce.
//
// Grid:   (CuCount,)         — all blocks stay resident, safe barrier.
// Block:  (THRDS, WvPrGrp)
//
// Inputs:
//   act        [top_k, K_in]  bf16/fp16     (silu output, slot-ordered)
//   w          [E, M, K_in/2] uint8         (packed int4)
//   scales     [E, M, K_in/G] bf16/fp16     (per-group scales)
//   topk_ids   [1, top_k]     int32         (this is for M=1 only)
//   topk_w     [1, top_k]     fp32          (router weights)
//   out        [1, M]         bf16/fp16     OUT
//   partial    [top_k, M]     fp32          SCRATCH
//   barrier    [1]            uint32        SCRATCH (zeroed externally)
//
// Each (cu) workgroup loops over slot in [0, top_k) and computes that
// slot's GEMM2 partial result striped across M in YTILE-row chunks.
// After all blocks have written partial[:, :], the last block to arrive
// at the grid-wide barrier performs the per-M reduction across slots
// and writes the bf16 output.
//
// Sizing the grid to exactly CuCount keeps occupancy ≤ 1 WG/CU
// (modulo the device's actual resident limit, which is ≥1), so the
// atomic-counter barrier is guaranteed not to deadlock waiting on
// non-resident blocks.
//
// We accept that this kernel is not bit-exact against the unfused chain
// (the order of the topk-weighted sum differs slightly, and we use
// fp32 accumulation throughout vs bf16 intermediate in the chain).
// ---------------------------------------------------------------------------

#if defined(__HIP__GFX9__) || defined(__HIP__GFX1X__)

constexpr int MEGA_LDS_ELEMS = 8192;

template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int GROUP_SIZE>
__global__ void __launch_bounds__(WvPrGrp* THRDS) moe_megakernel_int4_gemm2_(
    const int K, const int M, const uint8_t* W_base,
    const scalar_t* __restrict__ act_base,
    const scalar_t* __restrict__ scale_base, const int* __restrict__ topk_ids,
    const float* __restrict__ topk_w, scalar_t* __restrict__ out,
    float* __restrict__ partial, unsigned int* __restrict__ barrier,
    const int top_k, const long expert_stride_w, const long expert_stride_s,
    const int _WvPrGrp, const int CuCount) {
  constexpr int N = 1;  // single token decode
  constexpr int max_lds_len = MEGA_LDS_ELEMS;
  const int K_packed = K / 2;

  union bigTypeA {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
  };
  union bigTypeW {
    uint8_t b[A_CHUNK / 2];
    uint32_t u32[A_CHUNK / 8];
    float f[A_CHUNK / 8];
  };

  __shared__ scalar_t s[max_lds_len];
  __shared__ unsigned int sh_done;

  // Outer slot loop: every block iterates all slots.
  for (int slot = 0; slot < top_k; slot++) {
    int expert_id = topk_ids[slot];
    const uint8_t* W = W_base + (long)expert_id * expert_stride_w;
    const scalar_t* S = scale_base + (long)expert_id * expert_stride_s;
    const scalar_t* A = act_base + (long)slot * K;
    float* Cp = partial + (long)slot * M;

    // Stage activations for this slot into LDS.
    __syncthreads();
    for (uint32_t k = 0; k < (uint32_t)min(K * N, max_lds_len);
         k += THRDS * WvPrGrp * A_CHUNK) {
      uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);
      if (k_in >= (uint32_t)min(K * N, max_lds_len)) break;
      *((bigTypeA*)(&s[k_in])) = *((bigTypeA*)(&A[k_in]));
    }
    __syncthreads();

    if (threadIdx.y < _WvPrGrp) {
      uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;
      const int num_groups = K / GROUP_SIZE;

      float sum[N][YTILE];
      bigTypeA bigA[N][UNRL];
      bigTypeW bigB[YTILE][UNRL];

      while (m < (uint32_t)M) {
        for (int i = 0; i < YTILE; i++)
          for (int n = 0; n < N; n++) sum[n][i] = 0;

        for (uint32_t k1 = 0; k1 < (uint32_t)K; k1 += THRDS * A_CHUNK * UNRL) {
  #pragma unroll
          for (uint32_t k2 = 0; k2 < UNRL; k2++) {
            uint32_t k = k1 + k2 * THRDS * A_CHUNK;
            uint32_t k_ = k + threadIdx.x * A_CHUNK;
            if (k_ >= (uint32_t)K) break;
            const uint8_t* B_ = &W[(m + 0) * K_packed + k_ / 2];
            for (int y = 0; y < YTILE; y++) {
              const float* src = (const float*)(&B_[y * K_packed]);
  #pragma unroll
              for (int i = 0; i < A_CHUNK / 8; i++)
                bigB[y][k2].f[i] = loadnt((float*)&src[i]);
            }
          }

  #pragma unroll
          for (uint32_t k2 = 0; k2 < UNRL; k2++) {
            uint32_t k = k1 + k2 * THRDS * A_CHUNK;
            uint32_t k_ = k + threadIdx.x * A_CHUNK;
            if (k_ >= (uint32_t)K) break;
            for (int n = 0; n < N; n++) {
              bigA[n][k2] = *((const bigTypeA*)(&(s[k_ + K * n])));
            }
          }

  #pragma unroll
          for (uint32_t k2 = 0; k2 < UNRL; k2++) {
            uint32_t k = k1 + k2 * THRDS * A_CHUNK;
            uint32_t k_ = k + threadIdx.x * A_CHUNK;
            if (k_ >= (uint32_t)K) break;
  #pragma unroll
            for (uint32_t n = 0; n < N; n++) {
  #pragma unroll
              for (int y = 0; y < YTILE; y++) {
                bigTypeA cvtB;
                if constexpr (std::is_same_v<scalar_t, half>) {
                  constexpr uint32_t FP16_MAGIC = 0x64006400u;
                  constexpr uint32_t BIAS_LO = 0x64086408u;  // sym
                  constexpr uint32_t SCALE16 = 0x2C002C00u;
                  constexpr uint32_t BIAS_HI = 0xD480D480u;  // sym
  #pragma unroll
                  for (uint32_t w = 0; w < A_CHUNK / 8; w++) {
                    uint32_t qa = bigB[y][k2].u32[w];
                    uint32_t lo0 = (qa & 0x000F000Fu) | FP16_MAGIC;
                    uint32_t hi0 = (qa & 0x00F000F0u) | FP16_MAGIC;
                    qa >>= 8;
                    uint32_t lo1 = (qa & 0x000F000Fu) | FP16_MAGIC;
                    uint32_t hi1 = (qa & 0x00F000F0u) | FP16_MAGIC;
                    *(half2*)&cvtB.f[w * 4 + 0] =
                        __hsub2(*(half2*)&lo0, *(const half2*)&BIAS_LO);
                    *(half2*)&cvtB.f[w * 4 + 1] =
                        __hfma2(*(half2*)&hi0, *(const half2*)&SCALE16,
                                *(const half2*)&BIAS_HI);
                    *(half2*)&cvtB.f[w * 4 + 2] =
                        __hsub2(*(half2*)&lo1, *(const half2*)&BIAS_LO);
                    *(half2*)&cvtB.f[w * 4 + 3] =
                        __hfma2(*(half2*)&hi1, *(const half2*)&SCALE16,
                                *(const half2*)&BIAS_HI);
                  }
                } else {
                  constexpr uint32_t BF16_MAGIC = 0x43004300u;
                  constexpr uint32_t BIAS = 0x43084308u;  // sym (subtract 136)
  #pragma unroll
                  for (uint32_t w = 0; w < A_CHUNK / 8; w++) {
                    uint32_t qa = bigB[y][k2].u32[w];
                    uint32_t v0 = (qa & 0x000F000Fu) | BF16_MAGIC;
                    qa >>= 4;
                    uint32_t v1 = (qa & 0x000F000Fu) | BF16_MAGIC;
                    qa >>= 4;
                    uint32_t v2 = (qa & 0x000F000Fu) | BF16_MAGIC;
                    qa >>= 4;
                    uint32_t v3 = (qa & 0x000F000Fu) | BF16_MAGIC;
                    *(uint32_t*)&cvtB.f[w * 4 + 0] =
                        bf16x2_dequant_sub_finite(v0, BIAS);
                    *(uint32_t*)&cvtB.f[w * 4 + 1] =
                        bf16x2_dequant_sub_finite(v1, BIAS);
                    *(uint32_t*)&cvtB.f[w * 4 + 2] =
                        bf16x2_dequant_sub_finite(v2, BIAS);
                    *(uint32_t*)&cvtB.f[w * 4 + 3] =
                        bf16x2_dequant_sub_finite(v3, BIAS);
                  }
                }

                float partial_dot = 0;
  #pragma unroll
                for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                  DOT2C(partial_dot, bigA[n][k2].f[b], cvtB.f[b])
                }
                uint32_t group_idx = k_ / GROUP_SIZE;
                sum[n][y] += partial_dot *
                             __s2float(S[(m + y) * num_groups + group_idx]);
              }
            }
          }
        }

        // Wave-wide reduce of sum across THRDS lanes.
  #if defined(__HIP__GFX1X__)
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            sum[n][y] +=
                __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf, 1);
            sum[n][y] +=
                __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf, 1);
            sum[n][y] +=
                __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf, 1);
            sum[n][y] +=
                __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf, 1);
            sum[n][y] += __shfl_xor(sum[n][y], 16);
          }
        }
        if (threadIdx.x == (THRDS - 1)) {
          float w = topk_w[slot];
          for (int n = 0; n < N; n++) {
            for (int i = 0; i < YTILE; i++) {
              // Write topk-weighted partial directly to scratch.
              Cp[m + i + n * M] = sum[n][i] * w;
            }
          }
        }
  #else
        // gfx9 wave64 path
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
                : "=v"(sum[n][y])
                : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
            asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
                : "=v"(sum[n][y])
                : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
            asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
                : "=v"(sum[n][y])
                : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
            asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
                : "=v"(sum[n][y])
                : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
            asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
                : "=v"(sum[n][y])
                : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
            asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
                : "=v"(sum[n][y])
                : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
          }
        }
        if (threadIdx.x == 63) {
          float w = topk_w[slot];
          for (int n = 0; n < N; n++) {
            for (int i = 0; i < YTILE; i++) {
              Cp[m + i + n * M] = sum[n][i] * w;
            }
          }
        }
  #endif
        m += CuCount * _WvPrGrp * YTILE;
      }  // while (m < M)
    }  // if (threadIdx.y < _WvPrGrp)
  }  // for slot

  // Grid-wide barrier.  We launch CuCount blocks, all guaranteed
  // resident, so this can't deadlock.
  __syncthreads();
  __threadfence();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    unsigned int prev = atomicAdd(barrier, 1u);
    sh_done = prev;
  }
  __syncthreads();

  const unsigned int total_blocks = (unsigned int)CuCount;
  // Last block to arrive performs the reduction.
  if (sh_done != total_blocks - 1u) return;

  // Reset barrier for next call (only one block reaches here).
  *barrier = 0;
  __threadfence();

  // Tiny reducer: (THRDS * WvPrGrp) threads sum partial[:, :] -> out[:].
  const int linear_tid = threadIdx.y * THRDS + threadIdx.x;
  const int total_threads = THRDS * WvPrGrp;
  for (int m = linear_tid; m < M; m += total_threads) {
    float acc = 0.f;
    for (int sl = 0; sl < top_k; sl++) {
      acc += partial[(long)sl * M + m];
    }
    out[m] = __float2s<scalar_t>(acc);
  }
}

#else  // !defined(__HIP__GFX9__) && !defined(__HIP__GFX1X__)

template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int GROUP_SIZE>
__global__ void moe_megakernel_int4_gemm2_(
    const int K, const int M, const uint8_t* W_base,
    const scalar_t* __restrict__ act_base,
    const scalar_t* __restrict__ scale_base, const int* __restrict__ topk_ids,
    const float* __restrict__ topk_w, scalar_t* __restrict__ out,
    float* __restrict__ partial, unsigned int* __restrict__ barrier,
    const int top_k, const long expert_stride_w, const long expert_stride_s,
    const int _WvPrGrp, const int CuCount) {
  // Unsupported arch — kernel is a no-op; the wrapper guards with
  // is_gfx1x_int4().
  assert(false);
}

#endif  // arch guard

// ---------------------------------------------------------------------------
// Host wrapper.
// ---------------------------------------------------------------------------

static bool is_gfx1x_mega() {
  static const bool result = [] {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx11") != std::string::npos ||
           device_arch.find("gfx12") != std::string::npos;
  }();
  return result;
}

void moe_megakernel_int4_persistent(
    torch::Tensor act,       // [top_k, K] (silu output, slot-ordered)
    torch::Tensor w2,        // [E, M, K/8] int32 (skinny-packed)
    torch::Tensor w2_scale,  // [E, M, K/G] bf16/fp16
    torch::Tensor topk_ids,  // [1, top_k] int32
    torch::Tensor topk_w,    // [1, top_k] fp32
    torch::Tensor out,       // [1, M] bf16/fp16  OUT
    torch::Tensor partial,   // [top_k, M] fp32   SCRATCH
    torch::Tensor barrier,   // [1] uint32        SCRATCH (zeroed)
    int64_t CuCount, int64_t group_size) {
  TORCH_CHECK(is_gfx1x_mega(),
              "moe_megakernel_int4_persistent: only gfx11/gfx12 supported");
  TORCH_CHECK(w2.dtype() == torch::kInt32, "w2 must be int32 packed");
  TORCH_CHECK(act.is_contiguous());
  TORCH_CHECK(act.dim() == 2);
  TORCH_CHECK(w2.dim() == 3);
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32);
  TORCH_CHECK(topk_w.dtype() == torch::kFloat32);
  TORCH_CHECK(barrier.dtype() == torch::kInt32 ||
              barrier.dtype() == torch::kUInt32 ||
              barrier.dtype() == torch::kInt32);

  const int top_k = act.size(0);
  const int K_in = act.size(1);
  const int M_in = w2.size(1);
  // K_packed_int32 = K/8; K bytes = K/2.

  TORCH_CHECK(out.size(0) == 1 && out.size(1) == M_in);
  TORCH_CHECK(partial.size(0) == top_k && partial.size(1) == M_in);
  TORCH_CHECK(barrier.numel() >= 1);
  TORCH_CHECK(group_size == 32 || group_size == 128);
  TORCH_CHECK(K_in % group_size == 0);

  const long expert_stride_w = (long)M_in * (K_in / 2);
  const long expert_stride_s = (long)M_in * (K_in / group_size);

  dim3 grid((unsigned)CuCount);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(act));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      act.scalar_type(), "moe_megakernel_int4_persistent", [&] {
        using fptype = typename scalar<scalar_t>::type;
        const uint8_t* wptr = reinterpret_cast<const uint8_t*>(w2.data_ptr());
        const fptype* aptr = reinterpret_cast<const fptype*>(act.data_ptr());
        const fptype* sptr =
            reinterpret_cast<const fptype*>(w2_scale.data_ptr());
        const int* idsptr = reinterpret_cast<const int*>(topk_ids.data_ptr());
        const float* wghptr = reinterpret_cast<const float*>(topk_w.data_ptr());
        fptype* outptr = reinterpret_cast<fptype*>(out.data_ptr());
        float* partptr = reinterpret_cast<float*>(partial.data_ptr());
        unsigned int* barptr =
            reinterpret_cast<unsigned int*>(barrier.data_ptr());

        // Tile config: mirror MoE_WVSPLITK_INT4G_TUNED case 2 (production
        // winner): THRDS=32, YTILE=2, WvPrGrp=4, A_CHUNK=16, UNRL=4.
        constexpr int THRDS = 32;
        constexpr int YTILE = 2;
        constexpr int WvPrGrp = 4;
        constexpr int A_CHUNK = 16;
        constexpr int UNRL = 4;
        dim3 block(THRDS, WvPrGrp);
        const int wvPrGrp = WvPrGrp;
        if (group_size == 32) {
          moe_megakernel_int4_gemm2_<fptype, THRDS, YTILE, WvPrGrp, A_CHUNK,
                                     UNRL, 32><<<grid, block, 0, stream>>>(
              K_in, M_in, wptr, aptr, sptr, idsptr, wghptr, outptr, partptr,
              barptr, top_k, expert_stride_w, expert_stride_s, wvPrGrp,
              (int)CuCount);
        } else {
          moe_megakernel_int4_gemm2_<fptype, THRDS, YTILE, WvPrGrp, A_CHUNK,
                                     UNRL, 128><<<grid, block, 0, stream>>>(
              K_in, M_in, wptr, aptr, sptr, idsptr, wghptr, outptr, partptr,
              barptr, top_k, expert_stride_w, expert_stride_s, wvPrGrp,
              (int)CuCount);
        }
      });
}
