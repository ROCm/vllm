#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <algorithm>

#include "../cuda_compat.h"
#include "dispatch_utils.h"

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif

#define LDS_SIZE 64 * 1024

int get_lds_size_w8a8() {
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

bool is_gfx11_w8a8() {
  static const bool result = [] {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx11") != std::string::npos;
  }();
  return result;
}

// Check if this is a low-bandwidth gfx11 variant (gfx1150, gfx1152, gfx1153)
// or gfx1103, which benefit from optimized heuristics for lower bandwidth.
// Excludes gfx1151 which has higher bandwidth similar to gfx9.
bool is_low_bandwidth_gfx11_w8a8() {
  static const bool result = [] {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    std::string device_arch = dprops->gcnArchName;
    return device_arch.find("gfx1150") != std::string::npos ||
           device_arch.find("gfx1152") != std::string::npos ||
           device_arch.find("gfx1153") != std::string::npos ||
           device_arch.find("gfx1103") != std::string::npos;
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

#if defined(__GFX11__)
  // Int8x4 dot product for RDNA3/4: v_dot4_i32_iu8
  // 4 signed int8 multiplies + int32 accumulate in one instruction.
  #define DOT4_I8(V0, V2, V3)                                  \
    V0 = __builtin_amdgcn_sudot4(true, *((int*)(&(V2))), true, \
                                 *((int*)(&(V3))), V0, false);
#endif

#if defined(__GFX11__)
  #define REDUCE_SUM_WAVE32(val)  \
    do {                          \
      val += __shfl_xor(val, 1);  \
      val += __shfl_xor(val, 2);  \
      val += __shfl_xor(val, 4);  \
      val += __shfl_xor(val, 8);  \
      val += __shfl_xor(val, 16); \
    } while (0)
#endif

__device__ inline unsigned int min_w8a8(uint32_t a, uint32_t b) {
  return min(a, b);
}

// W8A8 skinny GEMM kernel: int8 weights, int8 activations
// Both operands are int8. Activations stored in LDS as int8 (1 byte each),
// giving 2x the LDS capacity compared to the W8A16 variant.
// Epilogue: result = sum * w_scale[m] * a_scale (per-channel weight, per-tensor
// activation)
#if defined(__HIP__GFX9__) || defined(__GFX11__)
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_w8a8_hf_sml_(const int K, const int M, const int Bx, const int By,
                          const int8_t* B, const int8_t* __restrict__ A,
                          const scalar_t* w_scale,
                          const float* __restrict__ a_scale,
                          const scalar_t* __restrict__ BIAS, scalar_t* C,
                          const int _WvPrGrp, const int CuCount) {
  // LDS stores int8 activations: 1 byte each (vs 2 bytes for fp16 in W8A16)
  constexpr int max_lds_ints = LDS_SIZE;

  // Activation load union: A_CHUNK int8 values = A_CHUNK bytes
  union bigTypeA {
    int8_t b[A_CHUNK];
    float f[A_CHUNK / 4];
  };

  // Converted activation values: A_CHUNK fp16/bf16 values = 2*A_CHUNK bytes
  union bigTypeAcvt {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
  };

  // Weight union: A_CHUNK int8 values = A_CHUNK bytes
  union bigTypeW {
    int8_t b[A_CHUNK];
    float f[A_CHUNK / 4];
  };

  __shared__ int8_t s[max_lds_ints];

  // Fetch int8 activation matrix to LDS
  // Each thread fetches A_CHUNK int8 elements = A_CHUNK bytes
  for (uint32_t k = 0; k < min_w8a8(K * N, max_lds_ints);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    if (k_in >= min_w8a8(K * N, max_lds_ints)) break;

    *((bigTypeA*)(&s[k_in])) = *((bigTypeA*)(&A[k_in]));
  }
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  // Load per-tensor activation scale once
  const float a_scale_val = *a_scale;

  #if defined(__GFX11__)
  int32_t sum[N][YTILE];
  #else
  float sum[N][YTILE];
  #endif

  while (m < M) {
    for (int i = 0; i < YTILE; i++)
      for (int n = 0; n < N; n++) sum[n][i] = 0;

  #if defined(__GFX11__)
    bigTypeA bigA[N][UNRL];
  #else
    bigTypeAcvt bigA[N][UNRL];
  #endif
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
          // Cast to float* for 4-byte non-temporal loads (not arithmetic).
          const float* src = (const float*)(&B_[y * K]);
  #pragma unroll
          for (int i = 0; i < A_CHUNK / 4; i++)
            bigB[y][k2].f[i] = loadnt((float*)&src[i]);
        }
      }

      // Fetch int8 activations from LDS
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        for (int n = 0; n < N; n++) {
  #if defined(__GFX11__)
          // Direct int8 load (no conversion needed for int8 dot product)
          bigA[n][k2] = *((const bigTypeA*)(&(s[k_ + K * n])));
  #else
          bigTypeA rawA = *((const bigTypeA*)(&(s[k_ + K * n])));
            // Convert int8 activations to fp16/bf16
    #pragma unroll
          for (uint32_t b = 0; b < A_CHUNK; b++) {
            bigA[n][k2].h[b] = rawA.b[b];
          }
  #endif
        }
      }

      // Matrix multiply
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

  #pragma unroll
        for (uint32_t n = 0; n < N; n++) {
  #pragma unroll
          for (int y = 0; y < YTILE; y++) {
  #if defined(__GFX11__)
              // Direct int8x int8 -> int32 dot product (4 elements per
              // instruction)
    #pragma unroll
            for (uint32_t b = 0; b < A_CHUNK / 4; b++) {
              DOT4_I8(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
            }
  #else
            // Convert int8 weights to fp16/bf16, then DOT2C
            bigTypeAcvt cvtB;
    #pragma unroll
            for (uint32_t b = 0; b < A_CHUNK; b++) {
              cvtB.h[b] = bigB[y][k2].b[b];
            }
    #pragma unroll
            for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
              DOT2C(sum[n][y], bigA[n][k2].f[b], cvtB.f[b])
            }
  #endif
          }
        }
      }
    }

    // Reduction
  #if defined(__GFX11__)
    for (int n = 0; n < N; n++)
      for (int y = 0; y < YTILE; y++) REDUCE_SUM_WAVE32(sum[n][y]);

    if (threadIdx.x == 0) {
      for (int n = 0; n < N; n++) {
        for (int i = 0; i < YTILE; i++) {
          // Convert int32 accumulator to float for scaling
          float sum_f = static_cast<float>(sum[n][i]);
          sum_f *= __s2float(w_scale[m + i]) * a_scale_val;
          if (BIAS) sum_f += __s2float(BIAS[(m + i) % Bx + (n % By) * M]);
          C[m + i + n * M] = __float2s<scalar_t>(sum_f);
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
          sum[n][i] *= __s2float(w_scale[m + i]) * a_scale_val;
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
          int UNRL, int N>
__global__ void wvSplitK_w8a8_hf_sml_(
    const int K, const int M, const int Bx, const int By, const int8_t* B,
    const int8_t* __restrict__ A, const scalar_t* w_scale,
    const float* __restrict__ a_scale, const scalar_t* __restrict__ BIAS,
    scalar_t* C, const int _WvPrGrp, const int CuCount) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__GFX9__) || defined(__GFX11__)

int mindiv_w8a8(int N, int div1, int div2) {
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

torch::Tensor wvSplitK_w8a8(const at::Tensor& in_a, const at::Tensor& in_b,
                            const at::Tensor& in_w_scale,
                            const at::Tensor& in_a_scale,
                            const std::optional<at::Tensor>& in_bias,
                            const int64_t CuCount) {
  // in_a: int8 weights [M, K]
  // in_b: int8 activations [N, K]
  // in_w_scale: per-channel weight scale [M] in fp16/bf16
  // in_a_scale: per-tensor activation scale (scalar) in float32
  // in_bias: optional bias
  auto M_in = in_a.size(0);
  auto K_in = in_a.size(1);
  auto N_in = in_b.size(0);
  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(in_a.dtype() == torch::kInt8, "Weight must be int8");
  TORCH_CHECK(in_b.dtype() == torch::kInt8, "Activation must be int8");
  TORCH_CHECK(in_a_scale.dtype() == torch::kFloat32,
              "Activation scale must be float32");
  TORCH_CHECK(in_w_scale.dtype() == torch::kFloat16 ||
                  in_w_scale.dtype() == torch::kBFloat16,
              "Weight scale must be float16 or bfloat16");
  TORCH_CHECK(in_w_scale.size(0) == M_in, "Weight scale size must match M");
  TORCH_CHECK(K_in % 16 == 0, "K must be divisible by 16 for w8a8 kernel");

  // LDS stores int8 activations: 1 byte each (full LDS capacity)
  const int max_lds_ints = get_lds_size_w8a8();
  TORCH_CHECK(K_in * N_in <= max_lds_ints,
              "K*N exceeds LDS capacity; only sml variant is supported. "
              "K=",
              K_in, " N=", N_in, " K*N=", K_in * N_in, " max=", max_lds_ints);

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_w_scale.dtype()).device(in_b.device()));

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define WVSPLITK_W8A8_LAUNCH(_THRDS, _YTILE, _UNRL, _N)                       \
  {                                                                           \
    dim3 block(_THRDS, 16);                                                   \
    int __wvPrGrp = mindiv_w8a8(M_in, CuCount * _YTILE, 16);                  \
    TORCH_CHECK(M_in % _YTILE == 0, "M must be divisible by YTILE=", _YTILE); \
    wvSplitK_w8a8_hf_sml_<fptype, _THRDS, _YTILE, 16, 16, _UNRL, _N>          \
        <<<grid, block, 0, stream>>>(K_in, M_in, Bx_in, By_in, wptr, aptr,    \
                                     wsptr, asptr, biasptr, cptr, __wvPrGrp,  \
                                     CuCount);                                \
  }

#define WVSPLITK_W8A8(_YTILE, _UNRL, _N)        \
  if (is_gfx11_w8a8())                          \
    WVSPLITK_W8A8_LAUNCH(32, _YTILE, _UNRL, _N) \
  else                                          \
    WVSPLITK_W8A8_LAUNCH(64, _YTILE, _UNRL, _N)

#define WVSPLIT_W8A8_TILE(_sYT, __N)             \
  {                                              \
    if (is_low_bandwidth_gfx11_w8a8()) {         \
      /* Optimized for gfx1150/1152/1153/1103 */ \
      if (K_in > 6000)                           \
        WVSPLITK_W8A8(2, 1, __N)                 \
      else if (M_in >= 19000)                    \
        WVSPLITK_W8A8(4, 4, __N)                 \
      else if (K_in <= 2048 && M_in < 4096)      \
        WVSPLITK_W8A8(4, 1, __N)                 \
      else                                       \
        WVSPLITK_W8A8(4, 1, __N)                 \
    } else {                                     \
      /* Original heuristic for gfx1151, gfx9 */ \
      if (__N >= 4 && _sYT >= 480)               \
        WVSPLITK_W8A8(4, 1, __N)                 \
      else if (K_in <= 1024 && M_in % 2 == 0)    \
        WVSPLITK_W8A8(2, 1, __N)                 \
      else                                       \
        WVSPLITK_W8A8(1, 4, __N)                 \
    }                                            \
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      in_w_scale.scalar_type(), "wvSplitK_w8a8", [&] {
        using fptype = typename scalar<scalar_t>::type;
        const int8_t* wptr = in_a.data_ptr<int8_t>();
        const int8_t* aptr = in_b.data_ptr<int8_t>();
        const fptype* wsptr =
            reinterpret_cast<const fptype*>(in_w_scale.data_ptr());
        const float* asptr = in_a_scale.data_ptr<float>();
        const fptype* biasptr =
            (in_bias.has_value() && in_bias->numel() > 0)
                ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
                : nullptr;
        fptype* cptr = reinterpret_cast<fptype*>(out_c.data_ptr());

        int sYT = (M_in + CuCount * 4 - 1) / (CuCount * 4);

        switch (N_in) {
          case 1:
            WVSPLIT_W8A8_TILE(sYT, 1)
            break;
          case 2:
            WVSPLIT_W8A8_TILE(sYT, 2)
            break;
          case 3:
            WVSPLIT_W8A8_TILE(sYT, 3)
            break;
          case 4:
            WVSPLIT_W8A8_TILE(sYT, 4)
            break;
          case 5:
            WVSPLIT_W8A8_TILE(sYT, 5)
            break;
          default:
            throw std::runtime_error(
                "Unsupported N value: " + std::to_string(M_in) + "," +
                std::to_string(K_in) + "," + std::to_string(N_in));
        }
      });

#undef WVSPLITK_W8A8_LAUNCH
#undef WVSPLITK_W8A8
#undef WVSPLIT_W8A8_TILE

  return out_c;
}

// Sweep function disabled by default to reduce compile time.
// Build with -DVLLM_SKINNY_GEMM_SWEEP to enable.
#ifdef VLLM_SKINNY_GEMM_SWEEP
torch::Tensor wvSplitK_w8a8_sweep(const at::Tensor& in_a,
                                  const at::Tensor& in_b,
                                  const at::Tensor& in_w_scale,
                                  const at::Tensor& in_a_scale,
                                  const std::optional<at::Tensor>& in_bias,
                                  const int64_t CuCount, const int64_t ytile,
                                  const int64_t unrl, const int64_t achunk,
                                  const int64_t wvprgrp) {
  auto M_in = in_a.size(0);
  auto K_in = in_a.size(1);
  auto N_in = in_b.size(0);
  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(in_a.dtype() == torch::kInt8, "Weight must be int8");
  TORCH_CHECK(in_b.dtype() == torch::kInt8, "Activation must be int8");
  TORCH_CHECK(in_a_scale.dtype() == torch::kFloat32,
              "Activation scale must be float32");
  TORCH_CHECK(in_w_scale.dtype() == torch::kFloat16 ||
                  in_w_scale.dtype() == torch::kBFloat16,
              "Weight scale must be float16 or bfloat16");
  TORCH_CHECK(in_w_scale.size(0) == M_in, "Weight scale size must match M");
  TORCH_CHECK(K_in % achunk == 0, "K must be divisible by achunk=", achunk);
  TORCH_CHECK(M_in % ytile == 0, "M must be divisible by ytile=", ytile);

  const int max_lds_ints = get_lds_size_w8a8();
  TORCH_CHECK(K_in * N_in <= max_lds_ints, "K*N exceeds LDS capacity. K=", K_in,
              " N=", N_in);

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_w_scale.dtype()).device(in_b.device()));

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int8_t* wptr = in_a.data_ptr<int8_t>();
  const int8_t* aptr = in_b.data_ptr<int8_t>();
  const float* asptr = in_a_scale.data_ptr<float>();

  const int THRDS = is_gfx11_w8a8() ? 32 : 64;

  // AT_DISPATCH inside the launch macro so that #defines stay outside lambdas
  // (hipify strips #define directives inside lambda bodies).
  #define SWEEP_W8A8_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, _N)  \
    AT_DISPATCH_REDUCED_FLOATING_TYPES(                                    \
        in_w_scale.scalar_type(), "wvSplitK_w8a8_sweep", [&] {             \
          using fptype = typename scalar<scalar_t>::type;                  \
          const fptype* wsptr =                                            \
              reinterpret_cast<const fptype*>(in_w_scale.data_ptr());      \
          const fptype* biasptr =                                          \
              (in_bias.has_value() && in_bias->numel() > 0)                \
                  ? reinterpret_cast<const fptype*>(in_bias->data_ptr())   \
                  : nullptr;                                               \
          fptype* cptr = reinterpret_cast<fptype*>(out_c.data_ptr());      \
          dim3 block(_THRDS, _WVPRGRP);                                    \
          int __wvPrGrp = mindiv_w8a8(M_in, CuCount * _YTILE, _WVPRGRP);   \
          wvSplitK_w8a8_hf_sml_<fptype, _THRDS, _YTILE, _WVPRGRP, _ACHUNK, \
                                _UNRL, _N><<<grid, block, 0, stream>>>(    \
              K_in, M_in, Bx_in, By_in, wptr, aptr, wsptr, asptr, biasptr, \
              cptr, __wvPrGrp, CuCount);                                   \
        });

  #define SWEEP_W8A8_N(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL)              \
    switch (N_in) {                                                           \
      case 1:                                                                 \
        SWEEP_W8A8_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, 1) break; \
      case 2:                                                                 \
        SWEEP_W8A8_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, 2) break; \
      case 3:                                                                 \
        SWEEP_W8A8_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, 3) break; \
      case 4:                                                                 \
        SWEEP_W8A8_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, 4) break; \
      case 5:                                                                 \
        SWEEP_W8A8_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL, 5) break; \
      default:                                                                \
        TORCH_CHECK(false, "Unsupported N=", N_in);                           \
    }

  #define SWEEP_W8A8_UNRL(_THRDS, _YTILE, _WVPRGRP, _ACHUNK) \
    if (unrl == 1) {                                         \
      SWEEP_W8A8_N(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 1)     \
    } else if (unrl == 2) {                                  \
      SWEEP_W8A8_N(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 2)     \
    } else if (unrl == 4) {                                  \
      SWEEP_W8A8_N(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 4)     \
    } else {                                                 \
      TORCH_CHECK(false, "Unsupported unrl=", unrl);         \
    }

  #define SWEEP_W8A8_YTILE(_THRDS, _WVPRGRP, _ACHUNK)  \
    if (ytile == 1) {                                  \
      SWEEP_W8A8_UNRL(_THRDS, 1, _WVPRGRP, _ACHUNK)    \
    } else if (ytile == 2) {                           \
      SWEEP_W8A8_UNRL(_THRDS, 2, _WVPRGRP, _ACHUNK)    \
    } else if (ytile == 4) {                           \
      SWEEP_W8A8_UNRL(_THRDS, 4, _WVPRGRP, _ACHUNK)    \
    } else {                                           \
      TORCH_CHECK(false, "Unsupported ytile=", ytile); \
    }

  #define SWEEP_W8A8_WVPRGRP(_THRDS, _ACHUNK)              \
    if (wvprgrp == 8) {                                    \
      SWEEP_W8A8_YTILE(_THRDS, 8, _ACHUNK)                 \
    } else if (wvprgrp == 12) {                            \
      SWEEP_W8A8_YTILE(_THRDS, 12, _ACHUNK)                \
    } else if (wvprgrp == 16) {                            \
      SWEEP_W8A8_YTILE(_THRDS, 16, _ACHUNK)                \
    } else {                                               \
      TORCH_CHECK(false, "Unsupported wvprgrp=", wvprgrp); \
    }

  if (THRDS == 32) {
    if (achunk == 8) {
      SWEEP_W8A8_WVPRGRP(32, 8)
    } else if (achunk == 16) {
      SWEEP_W8A8_WVPRGRP(32, 16)
    } else if (achunk == 32) {
      SWEEP_W8A8_WVPRGRP(32, 32)
    } else {
      TORCH_CHECK(false, "Unsupported achunk=", achunk);
    }
  } else {
    if (achunk == 8) {
      SWEEP_W8A8_WVPRGRP(64, 8)
    } else if (achunk == 16) {
      SWEEP_W8A8_WVPRGRP(64, 16)
    } else if (achunk == 32) {
      SWEEP_W8A8_WVPRGRP(64, 32)
    } else {
      TORCH_CHECK(false, "Unsupported achunk=", achunk);
    }
  }

  #undef SWEEP_W8A8_LAUNCH
  #undef SWEEP_W8A8_N
  #undef SWEEP_W8A8_UNRL
  #undef SWEEP_W8A8_YTILE
  #undef SWEEP_W8A8_WVPRGRP

  return out_c;
}
#endif  // VLLM_SKINNY_GEMM_SWEEP
