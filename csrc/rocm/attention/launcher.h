#pragma once

#include "kernels.cuh"
#include "launch_macros.h"

// GFX9 launcher: uses mfma4 for GQA 1-4, mfma16 for 5-16
template <typename T, typename KVT, vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE, int HEAD_SIZE, typename OUTT, int PARTITION_SIZE_OLD,
          bool ALIBI_ENABLED, MFMAType MFMA_TYPE>
void paged_attention_custom_launcher(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, const int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale) {
  int num_seqs = block_tables.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  const int* query_start_loc_ptr =
      query_start_loc
          ? reinterpret_cast<const int*>(query_start_loc.value().data_ptr())
          : nullptr;

  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  KVT* key_cache_ptr = reinterpret_cast<KVT*>(key_cache.data_ptr());
  KVT* value_cache_ptr = reinterpret_cast<KVT*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();
  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale.data_ptr());
  const float* v_scale_ptr = reinterpret_cast<const float*>(v_scale.data_ptr());
  const auto fp8_out_scale_ptr =
      fp8_out_scale
          ? static_cast<const float*>(fp8_out_scale.value().data_ptr())
          : nullptr;
  OUTT* out_ptr = reinterpret_cast<OUTT*>(out.data_ptr());

  const int max_ctx_blocks = DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE);

  constexpr int PARTITION_SIZE = 256;
  const int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
  const int gqa_ratio = num_heads / num_kv_heads;
  assert(num_heads % num_kv_heads == 0);
  assert(head_size == HEAD_SIZE);

  constexpr int NTHR = 256;
  dim3 grid(num_seqs, max_num_partitions, num_kv_heads);
  dim3 block(NTHR);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (gqa_ratio) {
    case 1:
      LAUNCH_CUSTOM_ATTENTION_MFMA4(1);
      break;
    case 2:
      LAUNCH_CUSTOM_ATTENTION_MFMA4(2);
      break;
    case 3:
      LAUNCH_CUSTOM_ATTENTION_MFMA4(3);
      break;
    case 4:
      LAUNCH_CUSTOM_ATTENTION_MFMA4(4);
      break;
    case 5:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(5);
      break;
    case 6:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(6);
      break;
    case 7:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(7);
      break;
    case 8:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(8);
      break;
    case 9:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(9);
      break;
    case 10:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(10);
      break;
    case 11:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(11);
      break;
    case 12:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(12);
      break;
    case 13:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(13);
      break;
    case 14:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(14);
      break;
    case 15:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(15);
      break;
    case 16:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(16);
      break;
    default:
      TORCH_CHECK(false, "Unsupported gqa ratio: ", gqa_ratio);
      break;
  }

  dim3 reduce_grid(num_heads, num_seqs);
  dim3 reduce_block(head_size);
  const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, WARP_SIZE);
  switch (npar_loops) {
    case 1:
      LAUNCH_CUSTOM_REDUCTION(1);
      break;
    case 2:
      LAUNCH_CUSTOM_REDUCTION(2);
      break;
    case 3:
      LAUNCH_CUSTOM_REDUCTION(3);
      break;
    case 4:
      LAUNCH_CUSTOM_REDUCTION(4);
      break;
    case 5:
      LAUNCH_CUSTOM_REDUCTION(5);
      break;
    case 6:
      LAUNCH_CUSTOM_REDUCTION(6);
      break;
    case 7:
      LAUNCH_CUSTOM_REDUCTION(7);
      break;
    case 8:
      LAUNCH_CUSTOM_REDUCTION(8);
      break;
    default:
      TORCH_CHECK(false, "Unsupported npar_loops: ", npar_loops);
      break;
  }
}

// Navi (GFX11/GFX12) launcher: uses mfma16 for all GQA ratios
template <typename T, typename KVT, vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE, int HEAD_SIZE, typename OUTT, int PARTITION_SIZE_OLD,
          bool ALIBI_ENABLED, MFMAType MFMA_TYPE>
void paged_attention_custom_launcher_navi(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, const int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  int num_seqs = block_tables.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  const int* query_start_loc_ptr =
      query_start_loc
          ? reinterpret_cast<const int*>(query_start_loc.value().data_ptr())
          : nullptr;

  const float* alibi_slopes_ptr = nullptr;

  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  KVT* key_cache_ptr = reinterpret_cast<KVT*>(key_cache.data_ptr());
  KVT* value_cache_ptr = reinterpret_cast<KVT*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();

  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale.data_ptr());
  const float* v_scale_ptr = reinterpret_cast<const float*>(v_scale.data_ptr());
  const auto fp8_out_scale_ptr = nullptr;
  OUTT* out_ptr = reinterpret_cast<OUTT*>(out.data_ptr());

  const int max_ctx_blocks = DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE);

  constexpr int PARTITION_SIZE = 256;
  const int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
  const int gqa_ratio = num_heads / num_kv_heads;
  assert(num_heads % num_kv_heads == 0);
  assert(head_size == HEAD_SIZE);

  constexpr int NTHR = 256;
  dim3 grid(num_seqs, max_num_partitions, num_kv_heads);
  dim3 block(NTHR);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (gqa_ratio) {
    case 1:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(1);
      break;
    case 2:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(2);
      break;
    case 3:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(3);
      break;
    case 4:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(4);
      break;
    case 5:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(5);
      break;
    case 6:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(6);
      break;
    case 7:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(7);
      break;
    case 8:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(8);
      break;
    case 9:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(9);
      break;
    case 10:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(10);
      break;
    case 11:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(11);
      break;
    case 12:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(12);
      break;
    case 13:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(13);
      break;
    case 14:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(14);
      break;
    case 15:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(15);
      break;
    case 16:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(16);
      break;
    default:
      TORCH_CHECK(false, "Unsupported gqa ratio: ", gqa_ratio);
      break;
  }

  dim3 reduce_grid(num_heads, num_seqs);
  dim3 reduce_block(head_size);
  const int warp_size = 32;
  const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, warp_size);
  switch (npar_loops) {
    case 1:
      LAUNCH_CUSTOM_REDUCTION(1);
      break;
    case 2:
      LAUNCH_CUSTOM_REDUCTION(2);
      break;
    case 3:
      LAUNCH_CUSTOM_REDUCTION(3);
      break;
    case 4:
      LAUNCH_CUSTOM_REDUCTION(4);
      break;
    case 5:
      LAUNCH_CUSTOM_REDUCTION(5);
      break;
    case 6:
      LAUNCH_CUSTOM_REDUCTION(6);
      break;
    case 7:
      LAUNCH_CUSTOM_REDUCTION(7);
      break;
    case 8:
      LAUNCH_CUSTOM_REDUCTION(8);
      break;
    case 9:
      LAUNCH_CUSTOM_REDUCTION(9);
      break;
    case 10:
      LAUNCH_CUSTOM_REDUCTION(10);
      break;
    case 11:
      LAUNCH_CUSTOM_REDUCTION(11);
      break;
    case 12:
      LAUNCH_CUSTOM_REDUCTION(12);
      break;
    case 13:
      LAUNCH_CUSTOM_REDUCTION(13);
      break;
    case 14:
      LAUNCH_CUSTOM_REDUCTION(14);
      break;
    case 15:
      LAUNCH_CUSTOM_REDUCTION(15);
      break;
    case 16:
      LAUNCH_CUSTOM_REDUCTION(16);
      break;
    default:
      TORCH_CHECK(false, "Unsupported npar_loops: ", npar_loops);
      break;
  }
}
