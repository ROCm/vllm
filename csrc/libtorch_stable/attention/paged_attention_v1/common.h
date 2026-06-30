#pragma once

#include "../../torch_utils.h"
#include "../attention_kernels.cuh"
#include "../../../cuda_compat.h"
#include "../../../quantization/w8a8/fp8/amd/quant_utils.cuh"

#include <algorithm>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                     \
      ((void*)vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE,        \
                                              BLOCK_SIZE, NUM_THREADS,      \
                                              KV_DTYPE, IS_BLOCK_SPARSE>),  \
      shared_mem_size);                                                     \
  vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,        \
                                  NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE>   \
      <<<grid, block, shared_mem_size, stream>>>(                           \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, \
          scale, block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,    \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,      \
          k_scale_ptr, v_scale_ptr, tp_rank, blocksparse_local_blocks,      \
          blocksparse_vert_stride, blocksparse_block_size,                  \
          blocksparse_head_sliding_step);

// Launcher preamble: extract pointers and compute grid/block dims.
// Used by each shard's launcher template before its HEAD_SIZE switch.
#define V1_LAUNCHER_PREAMBLE                                                 \
  int num_seqs = query.size(0);                                              \
  int num_heads = query.size(1);                                             \
  int head_size = query.size(2);                                             \
  int max_num_blocks_per_seq = block_tables.size(1);                         \
  int q_stride = query.stride(0);                                            \
  int kv_block_stride = key_cache.stride(0);                                 \
  int kv_head_stride = key_cache.stride(1);                                  \
                                                                             \
  const float* alibi_slopes_ptr =                                            \
      alibi_slopes                                                           \
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())  \
          : nullptr;                                                         \
                                                                             \
  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());                         \
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());                     \
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr()); \
  CACHE_T* value_cache_ptr =                                                 \
      reinterpret_cast<CACHE_T*>(value_cache.data_ptr());                    \
  int* block_tables_ptr = block_tables.mutable_data_ptr<int>();              \
  int* seq_lens_ptr = seq_lens.mutable_data_ptr<int>();                      \
  const float* k_scale_ptr =                                                 \
      reinterpret_cast<const float*>(k_scale.data_ptr());                    \
  const float* v_scale_ptr =                                                 \
      reinterpret_cast<const float*>(v_scale.data_ptr());                    \
                                                                             \
  const int NUM_WARPS = NUM_THREADS / WARP_SIZE;                             \
  int padded_max_seq_len =                                                   \
      DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE) * BLOCK_SIZE;                 \
  int logits_size = padded_max_seq_len * sizeof(float);                      \
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);            \
  int shared_mem_size = std::max(logits_size, outputs_size);                 \
                                                                             \
  dim3 grid(num_heads, num_seqs, 1);                                         \
  dim3 block(NUM_THREADS);                                                   \
  const torch::stable::accelerator::DeviceGuard device_guard(                \
      query.get_device_index());                                             \
  const cudaStream_t stream = get_current_cuda_stream();

// Launcher function signature (same for all shards).
#define V1_LAUNCHER_PARAMS                                                  \
  torch::stable::Tensor &out, torch::stable::Tensor &query,                 \
      torch::stable::Tensor &key_cache, torch::stable::Tensor &value_cache, \
      int num_kv_heads, float scale, torch::stable::Tensor &block_tables,   \
      torch::stable::Tensor &seq_lens, int max_seq_len,                     \
      const std::optional<torch::stable::Tensor>&alibi_slopes,              \
      torch::stable::Tensor &k_scale, torch::stable::Tensor &v_scale,       \
      const int tp_rank, const int blocksparse_local_blocks,                \
      const int blocksparse_vert_stride, const int blocksparse_block_size,  \
      const int blocksparse_head_sliding_step

#define V1_LAUNCHER_ARGS                                                 \
  out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, \
      seq_lens, max_seq_len, alibi_slopes, k_scale, v_scale, tp_rank,    \
      blocksparse_local_blocks, blocksparse_vert_stride,                 \
      blocksparse_block_size, blocksparse_head_sliding_step

// Dispatch macros used by the entry point (same as original).
#define CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, KV_DTYPE, IS_BLOCK_SPARSE) \
  paged_attention_v1_launcher<T, CACHE_T, BLOCK_SIZE, KV_DTYPE,             \
                              IS_BLOCK_SPARSE>(V1_LAUNCHER_ARGS);

#define CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE) \
  if (is_block_sparse) {                                                   \
    CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, true);       \
  } else {                                                                 \
    CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, false);      \
  }

#define CALL_V1_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)             \
  switch (block_size) {                                               \
    case 8:                                                           \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 8, KV_DTYPE);             \
      break;                                                          \
    case 16:                                                          \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 16, KV_DTYPE);            \
      break;                                                          \
    case 32:                                                          \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 32, KV_DTYPE);            \
      break;                                                          \
    default:                                                          \
      STD_TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                          \
  }

// Per-shard entry point declarations (non-template, called from entry point).
void paged_attention_v1_small(
    torch::stable::Tensor& out, torch::stable::Tensor& query,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    int64_t num_kv_heads, double scale, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::stable::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void paged_attention_v1_medium(
    torch::stable::Tensor& out, torch::stable::Tensor& query,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    int64_t num_kv_heads, double scale, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::stable::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void paged_attention_v1_large(
    torch::stable::Tensor& out, torch::stable::Tensor& query,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    int64_t num_kv_heads, double scale, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::stable::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);
