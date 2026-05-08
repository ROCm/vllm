/*
 * Fused Q-RMSNorm + K-RMSNorm + MRoPE (Multimodal RoPE) kernel for Qwen3-VL /
 * Qwen3-Omni decode.
 *
 * Mirrors csrc/fused_qknorm_rope_kernel.cu in structure (one warp per
 * (token, head)).  The only differences are:
 *   - cos/sin per dim is selected from one of 3 axes (T/H/W) of the
 *     cos_sin_cache, based on either an interleaved or chunked
 *     mrope_section split [t, h, w].
 *   - position_ids has shape [3, num_tokens] (T/H/W positions).
 *
 * The rotation itself is the standard Neox-style RoPE (is_neox=true).
 * GPT-J-style (interleave) is not implemented here because Qwen3-VL/Omni use
 * is_neox_style=true.  The mrope_interleaved flag controls only how cos/sin
 * is laid out across dims, not the rotation pairing.
 */

#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

#include <torch/cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "type_convert.cuh"

#define MROPE_CHECK_TYPE(x, st)                                        \
  TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), \
              ", while ", st, " is expected")
#define MROPE_CHECK_TH_CUDA(x) \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define MROPE_CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define MROPE_CHECK_INPUT(x) \
  MROPE_CHECK_TH_CUDA(x);    \
  MROPE_CHECK_CONTIGUOUS(x)

#ifdef USE_ROCM
  #define MROPE_FINAL_MASK 0xffffffffffffffffULL
#else
  #define MROPE_FINAL_MASK 0xffffffff
#endif

namespace vllm_mrope {

template <typename T>
__inline__ __device__ T warpReduceSumMrope(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(MROPE_FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
inline __device__ __host__ T divUpMrope(T m, T n) {
  return (m + n - 1) / n;
}

template <int VEC_BYTES>
struct VecBytes;
template <>
struct VecBytes<4> {
  using type = uint;
};
template <>
struct VecBytes<8> {
  using type = uint2;
};
template <>
struct VecBytes<16> {
  using type = uint4;
};

// Per-token, per-head warp.  Reads Q/K, RMSNorms across head_dim, then applies
// mrope-selected cos/sin to produce the rotated output (in place).
//
// Grid: divUp(num_tokens * (num_q_heads + num_k_heads), warpsPerBlock)
// Block: 256 threads (8 warps).
template <typename scalar_t_in, typename scalar_t_cache, int head_dim,
          bool mrope_interleaved>
__global__ void fusedQKNormMRopeKernel(
    void* __restrict__ q_void,          // [num_tokens, num_q_heads*head_dim]
    void* __restrict__ k_void,          // [num_tokens, num_k_heads*head_dim]
    int const num_q_heads,              //
    int const num_k_heads,              //
    float const eps,                    //
    void const* __restrict__ q_w_void,  // [head_dim]
    void const* __restrict__ k_w_void,  // [head_dim]
    void const* __restrict__ cos_sin_void,  // [max_pos, rotary_dim]
    int64_t const* __restrict__ pos_t,      // [num_tokens]
    int64_t const* __restrict__ pos_h,      // [num_tokens]
    int64_t const* __restrict__ pos_w,      // [num_tokens]
    int const num_tokens,                   //
    int const rotary_dim,                   // == 2 * (t+h+w)
    int const mrope_t,                      //
    int const mrope_h,                      //
    int const mrope_w                       //
) {
  using Converter = vllm::_typeConvert<scalar_t_in>;
  static_assert(Converter::exists,
                "Input dtype is not supported for this CUDA architecture or "
                "toolkit version.");
  using T_in = typename Converter::hip_type;
  using T2_in = typename Converter::packed_hip_type;

  using CacheConverter = vllm::_typeConvert<scalar_t_cache>;
  static_assert(CacheConverter::exists,
                "Cache dtype is not supported for this CUDA architecture or "
                "toolkit version.");
  using T_cache = typename CacheConverter::hip_type;

  T_in* q = reinterpret_cast<T_in*>(q_void);
  T_in* k = reinterpret_cast<T_in*>(k_void);
  T_in const* q_weight = reinterpret_cast<T_in const*>(q_w_void);
  T_in const* k_weight = reinterpret_cast<T_in const*>(k_w_void);
  T_cache const* cos_sin_cache = reinterpret_cast<T_cache const*>(cos_sin_void);

  int const warpsPerBlock = blockDim.x / 32;
  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;

  int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;
  int const total_qk_heads = num_q_heads + num_k_heads;
  int const tokenIdx = globalWarpIdx / total_qk_heads;
  int const localHeadIdx = globalWarpIdx % total_qk_heads;
  if (tokenIdx >= num_tokens) return;

  bool const isQ = localHeadIdx < num_q_heads;
  int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_q_heads;

  static_assert(head_dim % (32 * 2) == 0, "head_dim must be divisible by 64");
  constexpr int numElemsPerThread = head_dim / 32;
  constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
  static_assert(elemSizeBytes % 4 == 0,
                "elemSizeBytes must be a multiple of 4");
  using vec_T = typename VecBytes<elemSizeBytes>::type;

  // Pointer to head's contiguous block.
  T_in* head_ptr;
  if (isQ) {
    head_ptr = q + tokenIdx * num_q_heads * head_dim + headIdx * head_dim;
  } else {
    head_ptr = k + tokenIdx * num_k_heads * head_dim + headIdx * head_dim;
  }
  int const laneOffset = laneId * numElemsPerThread;
  T_in* lane_ptr = head_ptr + laneOffset;

  float elements[numElemsPerThread];
  float sumOfSquares = 0.0f;

  // Load + sum of squares.
  {
    vec_T vec = *reinterpret_cast<vec_T const*>(lane_ptr);
    constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
    for (int i = 0; i < num_packed_elems; i++) {
      T2_in packed_val = *(reinterpret_cast<T2_in*>(&vec) + i);
      float2 vals = Converter::convert(packed_val);
      sumOfSquares += vals.x * vals.x + vals.y * vals.y;
      elements[2 * i] = vals.x;
      elements[2 * i + 1] = vals.y;
    }
  }

  // Reduce across warp.
  sumOfSquares = warpReduceSumMrope(sumOfSquares);
  float const rms_rcp =
      rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

  // Normalize + apply weight.
#pragma unroll
  for (int i = 0; i < numElemsPerThread; i++) {
    int const dim = laneOffset + i;
    float const w = isQ ? Converter::convert(q_weight[dim])
                        : Converter::convert(k_weight[dim]);
    elements[i] *= rms_rcp * w;
  }

  // === MRoPE (Neox style: rotation pair (i, i + rotary_dim/2)). ===
  int const half_rd = rotary_dim / 2;

  // Choose mrope axis (0=T, 1=H, 2=W) for a given half_dim index.
  // Returns 0/1/2.
  auto axis_for_half_dim = [&](int hd) -> int {
    if constexpr (mrope_interleaved) {
      // From apply_interleaved_rope:
      //   h_mask = ((hd % 3) == 1) & (hd <= 3 * mrope_h)  (note: <= matches)
      //   w_mask = ((hd % 3) == 2) & (hd <= 3 * mrope_w)
      //   else T
      int mod3 = hd % 3;
      if (mod3 == 1 && hd <= 3 * mrope_h) return 1;
      if (mod3 == 2 && hd <= 3 * mrope_w) return 2;
      return 0;
    } else {
      if (hd < mrope_t) return 0;
      if (hd < mrope_t + mrope_h) return 1;
      return 2;
    }
  };

  int64_t const positions_per_axis[3] = {pos_t[tokenIdx], pos_h[tokenIdx],
                                         pos_w[tokenIdx]};

  // Rotation: Neox-style.  Each lane handles dims [laneOffset .. laneOffset +
  // numElemsPerThread).  The pair partner of dim d (d < half_rd) is d +
  // half_rd; the partner of d (d >= half_rd) is d - half_rd.  The cos/sin index
  // is min(d, d - half_rd) (i.e. the lower-half index).  Use shfl to swap
  // pair values across the warp.
  //
  // Within numElemsPerThread (e.g. 4), the 4 dims of one lane all live in the
  // same half because head_dim/32 evenly divides half_rd when rotary_dim ==
  // head_dim.  We assume rotary_dim == head_dim (standard for Qwen3-Omni).
  // pairOffset is the lane-offset of the partner.
  int const pairOffset = half_rd / numElemsPerThread;

  float partner[numElemsPerThread];
#pragma unroll
  for (int i = 0; i < numElemsPerThread; i++) {
    partner[i] = __shfl_xor_sync(MROPE_FINAL_MASK, elements[i], pairOffset);
  }

  bool const isLower = laneId < pairOffset;

#pragma unroll
  for (int i = 0; i < numElemsPerThread; i++) {
    int const dim_idx = laneOffset + i;
    int const half_dim = isLower ? dim_idx : (dim_idx - half_rd);
    int const axis = axis_for_half_dim(half_dim);
    int64_t const pos_id = positions_per_axis[axis];
    T_cache const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
    // cos at +half_dim, sin at +half_rd+half_dim.
    float const cos_val =
        CacheConverter::convert(VLLM_LDG(cache_ptr + half_dim));
    float const sin_val =
        CacheConverter::convert(VLLM_LDG(cache_ptr + half_rd + half_dim));

    float const my_val = elements[i];
    float const pa_val = partner[i];
    if (isLower) {
      // x1' = x1*cos - x2*sin
      elements[i] = my_val * cos_val - pa_val * sin_val;
    } else {
      // x2' = x2*cos + x1*sin
      elements[i] = my_val * cos_val + pa_val * sin_val;
    }
  }

  // Store.
  {
    vec_T vec;
    constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
    for (int i = 0; i < num_packed_elems; i++) {
      T2_in packed_val =
          Converter::convert(make_float2(elements[2 * i], elements[2 * i + 1]));
      *(reinterpret_cast<T2_in*>(&vec) + i) = packed_val;
    }
    *reinterpret_cast<vec_T*>(lane_ptr) = vec;
  }
}

#define DISPATCH_MROPE_INTERLEAVE(interleaved, FLAG, ...) \
  if (interleaved) {                                      \
    constexpr bool FLAG = true;                           \
    __VA_ARGS__                                           \
  } else {                                                \
    constexpr bool FLAG = false;                          \
    __VA_ARGS__                                           \
  }

template <typename scalar_t_in, typename scalar_t_cache>
void launchFusedQKNormMRope(void* q, void* k, int const num_tokens,
                            int const num_q_heads, int const num_k_heads,
                            int const head_dim, int const rotary_dim,
                            float const eps, void const* q_weight,
                            void const* k_weight, void const* cos_sin_cache,
                            int64_t const* pos_t, int64_t const* pos_h,
                            int64_t const* pos_w, int const mrope_t,
                            int const mrope_h, int const mrope_w,
                            bool const mrope_interleaved, cudaStream_t stream) {
  constexpr int blockSize = 256;
  int const warpsPerBlock = blockSize / 32;
  int const total_qk_heads = num_q_heads + num_k_heads;
  int const total_warps = num_tokens * total_qk_heads;
  int const gridSize = divUpMrope(total_warps, warpsPerBlock);
  dim3 gridDim(gridSize);
  dim3 blockDim(blockSize);

  switch (head_dim) {
    case 64:
      DISPATCH_MROPE_INTERLEAVE(mrope_interleaved, INTERLEAVED, {
        fusedQKNormMRopeKernel<scalar_t_in, scalar_t_cache, 64, INTERLEAVED>
            <<<gridDim, blockDim, 0, stream>>>(
                q, k, num_q_heads, num_k_heads, eps, q_weight, k_weight,
                cos_sin_cache, pos_t, pos_h, pos_w, num_tokens, rotary_dim,
                mrope_t, mrope_h, mrope_w);
      });
      break;
    case 128:
      DISPATCH_MROPE_INTERLEAVE(mrope_interleaved, INTERLEAVED, {
        fusedQKNormMRopeKernel<scalar_t_in, scalar_t_cache, 128, INTERLEAVED>
            <<<gridDim, blockDim, 0, stream>>>(
                q, k, num_q_heads, num_k_heads, eps, q_weight, k_weight,
                cos_sin_cache, pos_t, pos_h, pos_w, num_tokens, rotary_dim,
                mrope_t, mrope_h, mrope_w);
      });
      break;
    case 256:
      DISPATCH_MROPE_INTERLEAVE(mrope_interleaved, INTERLEAVED, {
        fusedQKNormMRopeKernel<scalar_t_in, scalar_t_cache, 256, INTERLEAVED>
            <<<gridDim, blockDim, 0, stream>>>(
                q, k, num_q_heads, num_k_heads, eps, q_weight, k_weight,
                cos_sin_cache, pos_t, pos_h, pos_w, num_tokens, rotary_dim,
                mrope_t, mrope_h, mrope_w);
      });
      break;
    default:
      TORCH_CHECK(false, "Unsupported head dimension for fused_qk_norm_mrope: ",
                  head_dim);
  }
}

}  // namespace vllm_mrope

// ---- Public entry point. ----
//
// q: [num_tokens, num_q_heads*head_dim]   (in-place)
// k: [num_tokens, num_k_heads*head_dim]   (in-place)
// q_weight, k_weight: [head_dim]
// cos_sin_cache: [max_position, rotary_dim]
//   First half (rotary_dim/2) is cos, second half is sin.  Same layout as the
//   standard cos_sin_cache used by RotaryEmbedding.
// positions: [3, num_tokens]              (T/H/W positions)
// mrope_section: list of 3 ints summing to rotary_dim/2  (e.g. [24, 20, 20])
// eps: RMSNorm epsilon
// mrope_interleaved: true for Qwen3-Omni-style interleaved cos/sin selection.
void fused_qk_norm_mrope(torch::Tensor& q,              //
                         torch::Tensor& k,              //
                         int64_t num_q_heads,           //
                         int64_t num_k_heads,           //
                         int64_t head_dim,              //
                         double eps,                    //
                         torch::Tensor& q_weight,       //
                         torch::Tensor& k_weight,       //
                         torch::Tensor& cos_sin_cache,  //
                         torch::Tensor& positions,      //
                         int64_t mrope_section_t,       //
                         int64_t mrope_section_h,       //
                         int64_t mrope_section_w,       //
                         bool mrope_interleaved         //
) {
  MROPE_CHECK_INPUT(q);
  MROPE_CHECK_INPUT(k);
  MROPE_CHECK_INPUT(positions);
  MROPE_CHECK_INPUT(q_weight);
  MROPE_CHECK_INPUT(k_weight);
  MROPE_CHECK_INPUT(cos_sin_cache);
  MROPE_CHECK_TYPE(positions, torch::kInt64);

  TORCH_CHECK(q.dim() == 2, "q must be 2D");
  TORCH_CHECK(k.dim() == 2, "k must be 2D");
  TORCH_CHECK(positions.dim() == 2 && positions.size(0) == 3,
              "positions must be [3, num_tokens]");
  TORCH_CHECK(q_weight.dim() == 1 && q_weight.size(0) == head_dim,
              "q_weight must be [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1 && k_weight.size(0) == head_dim,
              "k_weight must be [head_dim]");
  TORCH_CHECK(cos_sin_cache.dim() == 2,
              "cos_sin_cache must be [max_position, rotary_dim]");

  int64_t const num_tokens = q.size(0);
  TORCH_CHECK(positions.size(1) == num_tokens,
              "positions.size(1) must equal num_tokens");
  TORCH_CHECK(q.size(1) == num_q_heads * head_dim, "q layout mismatch");
  TORCH_CHECK(k.size(1) == num_k_heads * head_dim, "k layout mismatch");

  int64_t const rotary_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rotary_dim == head_dim,
              "fused_qk_norm_mrope currently requires rotary_dim == head_dim");
  TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even");
  TORCH_CHECK(
      mrope_section_t + mrope_section_h + mrope_section_w == rotary_dim / 2,
      "mrope_section sum must equal rotary_dim / 2");

  TORCH_CHECK(q.scalar_type() == k.scalar_type() &&
                  q.scalar_type() == q_weight.scalar_type() &&
                  q.scalar_type() == k_weight.scalar_type(),
              "q/k/q_weight/k_weight must share dtype");

  auto device_id = q.get_device();
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  int64_t const* pos_ptr =
      reinterpret_cast<int64_t const*>(positions.data_ptr());
  int64_t const* pos_t = pos_ptr + 0 * num_tokens;
  int64_t const* pos_h = pos_ptr + 1 * num_tokens;
  int64_t const* pos_w = pos_ptr + 2 * num_tokens;

  VLLM_DISPATCH_HALF_TYPES(q.scalar_type(), "fused_qk_norm_mrope", [&] {
    using qkv_scalar_t = scalar_t;
    VLLM_DISPATCH_FLOATING_TYPES(
        cos_sin_cache.scalar_type(), "fused_qk_norm_mrope", [&] {
          using cache_scalar_t = scalar_t;
          vllm_mrope::launchFusedQKNormMRope<qkv_scalar_t, cache_scalar_t>(
              q.data_ptr(), k.data_ptr(), static_cast<int>(num_tokens),
              static_cast<int>(num_q_heads), static_cast<int>(num_k_heads),
              static_cast<int>(head_dim), static_cast<int>(rotary_dim),
              static_cast<float>(eps), q_weight.data_ptr(), k_weight.data_ptr(),
              cos_sin_cache.data_ptr(), pos_t, pos_h, pos_w,
              static_cast<int>(mrope_section_t),
              static_cast<int>(mrope_section_h),
              static_cast<int>(mrope_section_w), mrope_interleaved, stream);
        });
  });
}
