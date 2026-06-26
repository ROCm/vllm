#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <string>
#include <type_traits>

#include "../../cuda_compat.h"

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif

#define LDS_SIZE 64 * 1024

inline int get_lds_size_int8() {
  static bool is_cached = false;
  static int result;
  if (is_cached == false) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    std::string device_arch = dprops->gcnArchName;
    size_t substring = device_arch.find("gfx95");
    result = (substring == std::string::npos ? 64 * 1024 : 160 * 1024);
    is_cached = true;
  }
  return result;
}

inline bool is_gfx11_int8() {
  static const bool result = [] {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx11") != std::string::npos;
  }();
  return result;
}

#if defined(NDEBUG)
  #undef NDEBUG
  #include <assert.h>
  #define UNREACHABLE_CODE assert(false);
  #define NDEBUG
#else
  #define UNREACHABLE_CODE assert(false);
#endif

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

template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
  return __builtin_nontemporal_load(addr);
}

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

#define DOT2C(V0, V2, V3)                                                   \
  if constexpr (std::is_same_v<scalar_t, half>) {                           \
    V0 = __builtin_amdgcn_fdot2(*((half2*)(&(V2))), *((half2*)(&(V3))), V0, \
                                false);                                     \
  } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {          \
    float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *           \
               __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));            \
    V0 += (s.x + s.y);                                                      \
  }

__device__ inline unsigned int min__(uint32_t a, uint32_t b) {
  return min(a, b);
}

inline int mindiv_int8(int N, int div1, int div2) {
  int nPrRnd = div1 * div2;
  int limit = div2 < 13 ? div2 : 13;
  int rnds[16];
  for (int i = 0; i < limit; i++) {
    rnds[i] = (N + nPrRnd - 1) / nPrRnd;
    nPrRnd -= div1;
  }
  for (int i = limit - 1; i >= 0; i--)
    if (rnds[0] == rnds[i]) return (div2 - i);
  return 0;
}

// W8A16 skinny GEMM kernel: int8 weights, fp16/bf16 activations
// Targets the "sml" case where activations fit in LDS.
// A_CHUNK=16: each thread processes 16 int8 weight elements per step.
// GROUP_SIZE: 0 = per-channel scale [M] (one scale per output row, applied
//             once at the end of the K reduction).
//             >0 = per-group scale [M, K/GROUP_SIZE] (one scale per K-group;
//             requires GROUP_SIZE % A_CHUNK == 0 so each thread's A_CHUNK
//             K-elements lie within a single group).
#if defined(__HIP__GFX9__) || defined(__GFX11__)
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, int GROUP_SIZE = 0>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_int8_hf_sml_(const int K, const int M, const int Bx, const int By,
                          const int8_t* B, const scalar_t* __restrict__ A,
                          const scalar_t* scale,
                          const scalar_t* __restrict__ BIAS, scalar_t* C,
                          const int _WvPrGrp, const int CuCount) {
  static_assert(GROUP_SIZE == 0 || GROUP_SIZE >= A_CHUNK,
                "GROUP_SIZE must be >= A_CHUNK so each thread's A_CHUNK "
                "K-elements lie in one group");
  static_assert(GROUP_SIZE == 0 || (GROUP_SIZE % A_CHUNK) == 0,
                "GROUP_SIZE must be a multiple of A_CHUNK");
  constexpr int max_lds_len = LDS_SIZE / 2;

  // Activation union: 16 fp16/bf16 values = 32 bytes
  union bigTypeA {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
  };

  // Weight union: 16 int8 values = 16 bytes
  union bigTypeW {
    int8_t b[A_CHUNK];
    float f[A_CHUNK / 4];
  };

  __shared__ scalar_t s[max_lds_len];

  // Fetch activation matrix to LDS
  // Each thread fetches A_CHUNK fp16 elements = 32 bytes
  for (uint32_t k = 0; k < min__(K * N, max_lds_len);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    if (k_in >= min__(K * N, max_lds_len)) break;

    *((bigTypeA*)(&s[k_in])) = *((bigTypeA*)(&A[k_in]));
  }
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  // For per-group scales, num_groups stride along K.
  [[maybe_unused]] const int num_groups =
      (GROUP_SIZE > 0) ? (K / GROUP_SIZE) : 0;

  float sum[N][YTILE];

  while (m < M) {
    for (int i = 0; i < YTILE; i++)
      for (int n = 0; n < N; n++) sum[n][i] = 0;

    bigTypeA bigA[N][UNRL];
    bigTypeW bigB[YTILE][UNRL];

    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      // Fetch int8 weights from global memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        const int8_t* B_ = &B[(m + 0) * K + k_];
        for (int y = 0; y < YTILE; y++) {
          // 16 bytes = 4 floats worth of int8 data
          const float* src = (const float*)(&B_[y * K]);
  #pragma unroll
          for (int i = 0; i < A_CHUNK / 4; i++)
            bigB[y][k2].f[i] = loadnt((float*)&src[i]);
        }
      }

      // Fetch fp16/bf16 activations from LDS
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        for (int n = 0; n < N; n++) {
          bigA[n][k2] = *((const bigTypeA*)(&(s[k_ + K * n])));
        }
      }

      // Matrix multiply: convert int8 weight pairs to fp16, then DOT2C
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

  #pragma unroll
        for (uint32_t n = 0; n < N; n++) {
  #pragma unroll
          for (int y = 0; y < YTILE; y++) {
            // Convert 16 int8 weights to 8 fp16 pairs stored in a bigTypeA
            // union
            bigTypeA cvtB;
  #pragma unroll
            for (uint32_t b = 0; b < A_CHUNK; b++) {
              cvtB.h[b] = bigB[y][k2].b[b];
            }
            if constexpr (GROUP_SIZE > 0) {
              // Per-group scale: this thread's A_CHUNK K-elements lie in
              // a single group (statically asserted GROUP_SIZE >= A_CHUNK
              // and GROUP_SIZE % A_CHUNK == 0).  Accumulate the partial
              // dot product locally, multiply by the group's scale, then
              // add into sum[n][y].
              float partial = 0.0f;
  #pragma unroll
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(partial, bigA[n][k2].f[b], cvtB.f[b])
              }
              uint32_t group_idx = k_ / GROUP_SIZE;
              sum[n][y] +=
                  partial * __s2float(scale[(m + y) * num_groups + group_idx]);
            } else {
  #pragma unroll
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(sum[n][y], bigA[n][k2].f[b], cvtB.f[b])
              }
            }
          }
        }
      }
    }

    // Reduction
  #if defined(__GFX11__)
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf,
                                              1);  // row_shr8
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf,
                                              1);  // row_shr4
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf,
                                              1);  // row_shr2
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf,
                                              1);  // row_shr1
        sum[n][y] += __shfl_xor(sum[n][y], 16);
      }
    }

    if (threadIdx.x == (THRDS - 1)) {
      for (int n = 0; n < N; n++) {
        for (int i = 0; i < YTILE; i++) {
          if constexpr (GROUP_SIZE == 0) {
            sum[n][i] *= __s2float(scale[m + i]);
          }
          if (BIAS) sum[n][i] += __s2float(BIAS[(m + i) % Bx + (n % By) * M]);
          C[m + i + n * M] = __float2s<scalar_t>(sum[n][i]);
        }
      }
    }
  #else   // GFX9 wave64 path
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
      for (int n = 0; n < N; n++) {
        for (int i = 0; i < YTILE; i++) {
          if constexpr (GROUP_SIZE == 0) {
            sum[n][i] *= __s2float(scale[m + i]);
          }
          if (BIAS) sum[n][i] += __s2float(BIAS[(m + i) % Bx + (n % By) * M]);
          C[m + i + n * M] = __float2s<scalar_t>(sum[n][i]);
        }
      }
    }
  #endif  // defined(__GFX11__)
    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else   // !defined(__HIP__GFX9__) && !defined(__GFX11__)
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, int GROUP_SIZE = 0>
__global__ void wvSplitK_int8_hf_sml_(const int K, const int M, const int Bx,
                                      const int By, const int8_t* B,
                                      const scalar_t* __restrict__ A,
                                      const scalar_t* scale,
                                      const scalar_t* __restrict__ BIAS,
                                      scalar_t* C, const int _WvPrGrp,
                                      const int CuCount) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__GFX9__) || defined(__GFX11__)
