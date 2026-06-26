// HEAD_SIZE shard: 192, 256

#include "common.h"

template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 128>
void paged_attention_v1_launcher(V1_LAUNCHER_PARAMS) {
  V1_LAUNCHER_PREAMBLE

  switch (head_size) {
    case 192:
      LAUNCH_PAGED_ATTENTION_V1(192);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
      break;
    default:
      STD_TORCH_CHECK(false,
                      "Unsupported head size in large shard: ", head_size);
      break;
  }
}

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
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);
  DISPATCH_BY_KV_CACHE_DTYPE(query.scalar_type(), kv_cache_dtype,
                             CALL_V1_LAUNCHER_BLOCK_SIZE)
}
