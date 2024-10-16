#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#ifdef USE_ROCM
  #include "quantization/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#ifdef USE_ROCM

#include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;

#endif

namespace vllm {

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    const scalar_t* __restrict__ arr, const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr, int rot_offset, int embed_dim,
    int &x_index, scalar_t &x, int &y_index, scalar_t &y) {
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = VLLM_LDG(cos_ptr + x_index);
    sin = VLLM_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = VLLM_LDG(cos_ptr + x_index / 2);
    sin = VLLM_LDG(sin_ptr + x_index / 2);
  }

  x = arr[x_index] * cos - arr[y_index] * sin;
  y = arr[y_index] * cos + arr[x_index] * sin;
}

template <typename scalar_t, typename cache_t, bool isKey, Fp8KVCacheDataType kv_dt>
inline __device__ void store_value_into_cache(
      cache_t* __restrict__ cache, 
      const int head_size, const int num_kv_heads, 
      const int64_t block_idx, const int block_size, const int64_t block_offset,
      const int x, const int64_t idx, scalar_t val, const float kv_scale) {

  const int head_idx = idx / head_size;
  const int head_offset = idx % head_size;

  int64_t tgt_idx;
  if constexpr (isKey) {
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    tgt_idx = 
        block_idx * num_kv_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + 
        x_idx * block_size * x +
        block_offset * x + x_offset;
  } else {
    tgt_idx =
        block_idx * num_kv_heads * head_size * block_size +
        head_idx * head_size * block_size + 
        head_offset * block_size +
        block_offset;
  }

  if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
    cache[tgt_idx] = val;
  } else {
    cache[tgt_idx] =
        fp8::scaled_convert<cache_t, scalar_t, kv_dt>(val, kv_scale);
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt, bool IS_NEOX>
__global__ void fused_rotary_embedding_and_reshape_cache_kernel(
        scalar_t* __restrict__ query,        // [batch_size, seq_len, num_heads, head_size] or 
                                             // [num_tokens, num_heads, head_size]
        scalar_t* __restrict__ key,          // [num_tokens, num_heads, head_size]
        const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
        cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
        cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
        const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim // 2]
        const int64_t* __restrict__ positions,  // [batch_size, seq_len] or [num_tokens]
        const int64_t* __restrict__ slot_mapping,  // [num_tokens]
        const int64_t query_stride, const int key_stride, const int value_stride, 
        const int num_heads, const int num_kv_heads, const int head_size, 
        const int rot_dim, const int block_size, const int x, 
        const float k_scale, const float v_scale) {
    
    // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_size;

    int x_index, y_index;
    scalar_t x_value, y_value; 
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
                                query + token_head, cos_ptr, sin_ptr, 
                                rot_offset, embed_dim,
                                x_index, x_value, 
                                y_index, y_value);
    query[token_head + x_index] = x_value;
    query[token_head + y_index] = y_value;
  }

  const int64_t slot_idx = block_size == 0 ? 0: slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * head_size;

    int x_index, y_index;
    scalar_t x_value, y_value; 
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
                                key + token_head, cos_ptr, sin_ptr, 
                                rot_offset, embed_dim, 
                                x_index, x_value, 
                                y_index, y_value);
    
    // FIXME: probably not needed for decode path???
    key[token_head + x_index] = x_value;
    key[token_head + y_index] = y_value;

    if (block_size != 0) {
      store_value_into_cache<scalar_t, cache_t, true, kv_dt>
                            (key_cache, head_size, num_kv_heads,
                              block_idx, block_size, block_offset, 
                              x, head_idx * head_size + x_index, x_value, k_scale);
      store_value_into_cache<scalar_t, cache_t, true, kv_dt>
                            (key_cache, head_size, num_kv_heads,
                              block_idx, block_size, block_offset,
                              x, head_idx * head_size + y_index, y_value, k_scale);
    }
  }

  const int nv = num_kv_heads * head_size;
  for (int i = threadIdx.x; block_size && i < nv; i += blockDim.x) {
    const int64_t src_value_idx = token_idx * value_stride + i;
    scalar_t tgt_value = value[src_value_idx];

    store_value_into_cache<scalar_t, cache_t, false, kv_dt>
                          (value_cache, head_size, num_kv_heads, 
                            block_idx, block_size, block_offset,
                            0, i, tgt_value, v_scale);
  }
}

} // namespace vllm


 // SRC_DTYPE is the stored data type of qkv.
 // CACHE_T is the data type of in KV cache.
 // KV_DT is actual type of data in KV cache
 // IS_NEOX flag to compute positional encodding.
 #define CALL_ROTARY_EMBEDDING_RESHAPE_AND_CACHE(QKV_T, CACHE_T, KV_DT, IS_NEOX)            \
    vllm::fused_rotary_embedding_and_reshape_cache_kernel<QKV_T, CACHE_T, KV_DT, IS_NEOX>   \
        <<<grid, block, 0, stream>>>(                                                       \
                reinterpret_cast<QKV_T*>(query.data_ptr()),                                 \
                reinterpret_cast<QKV_T*>(key.data_ptr()),                                   \
                reinterpret_cast<QKV_T*>(value.data_ptr()),                                 \
                reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                           \
                reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                         \
                reinterpret_cast<QKV_T*>(cos_sin_cache.data_ptr()),                         \
                positions.data_ptr<int64_t>(),                                              \
                slot_mapping.data_ptr<int64_t>(),                                           \
                query_stride, key_stride, value_stride,                                     \
                num_heads, num_kv_heads, head_size,                                         \
                rot_dim, block_size, x, k_scale, v_scale);


  // The following macro is used to dispatch the conversion function based on
  // the data type of the key and value cache. The FN is a macro that calls a
  // function with template<typename SRC_DTYPE, str CACHE_T, IS_NEOX>.
  #define DISPATCH_ROPE_BY_KV_CACHE_DTYPE(SRC_DTYPE, CACHE_T, IS_NEOX, FN)            \
    if (CACHE_T == "auto") {                                                          \
      if (SRC_DTYPE == at::ScalarType::Float) {                                       \
        FN(float, float, vllm::Fp8KVCacheDataType::kAuto, IS_NEOX);                   \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                                 \
        FN(c10::Half, c10::Half, vllm::Fp8KVCacheDataType::kAuto, IS_NEOX);           \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                             \
        FN(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto, IS_NEOX);   \
      } else {                                                                        \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);        \
      }                                                                               \
    } else {                                                                          \
      if (CACHE_T == "fp8" || CACHE_T == "fp8_e4m3") {                                \
        if (SRC_DTYPE == at::ScalarType::Float) {                                     \
          FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3, IS_NEOX);            \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                               \
          FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3, IS_NEOX);         \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                           \
          FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3, IS_NEOX);    \
        } else {                                                                      \
          TORCH_CHECK(false,                                                          \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);             \
        }                                                                             \
      } else if (CACHE_T == "fp8_e5m2") {                                             \
        if (SRC_DTYPE == at::ScalarType::Float) {                                     \
          FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2, IS_NEOX);            \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                               \
          FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2, IS_NEOX);         \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                           \
          FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2, IS_NEOX);    \
        } else {                                                                      \
          TORCH_CHECK(false,                                                          \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);             \
        }                                                                             \
      } else {                                                                        \
        TORCH_CHECK(false, "Unsupported data type of kv cache: ", CACHE_T);           \
      }                                                                               \
    }


void fused_rotary_embedding_and_reshape_cache(
        torch::Tensor& positions,     // [batch_size, seq_len] or [num_tokens]
        torch::Tensor& query,   // [batch_size, seq_len, num_heads * head_size] or
                                // [num_tokens, num_heads * head_size]
        torch::Tensor& key,     // [batch_size, seq_len, num_kv_heads * head_size] or
                                // [num_tokens, num_kv_heads * head_size]
        torch::Tensor& value,   // [batch_size, seq_len, num_heads * head_size] or
                                // [num_tokens, num_heads * head_size]
        torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
        torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
        const std::string& kv_cache_dtype,
        torch::Tensor& cos_sin_cache, // [max_position, rot_dim]
        torch::Tensor& slot_mapping,  // [num_tokens]
        const int64_t head_size,
        const double k_scale,
        const double v_scale,
        bool is_neox) {

  assert(query.scalar_type() == key.scalar_type());

  int64_t num_tokens = query.numel() / query.size(-1);
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-1) / head_size;
  int num_kv_heads = key.size(-1) / head_size;
  int64_t query_stride = query.stride(-2);
  int64_t key_stride = key.stride(-2);
  int64_t value_stride = value.stride(-2);

  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (is_neox) {
    DISPATCH_ROPE_BY_KV_CACHE_DTYPE(key.scalar_type(),
        kv_cache_dtype, true, CALL_ROTARY_EMBEDDING_RESHAPE_AND_CACHE)
  } else {
    DISPATCH_ROPE_BY_KV_CACHE_DTYPE(key.scalar_type(),
        kv_cache_dtype, false, CALL_ROTARY_EMBEDDING_RESHAPE_AND_CACHE)
  }
}