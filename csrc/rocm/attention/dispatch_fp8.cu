// Dispatch shard for kv_cache_dtype == "fp8" / "fp8_e4m3" (FP8 KV cache).
// Split from attention.cu to parallelize compilation across dtype combos.

#include "launcher.h"

#define CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT,   \
                             PSIZE, ALIBI_ENABLED, MFMA_TYPE)               \
  if (!is_navi) {                                                           \
    paged_attention_custom_launcher<T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,  \
                                    OUTT, PSIZE, ALIBI_ENABLED, MFMA_TYPE>( \
        out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,  \
        num_kv_heads, scale, block_tables, seq_lens, query_start_loc,       \
        max_seq_len, alibi_slopes, k_scale, v_scale, fp8_out_scale);        \
  } else {                                                                  \
    paged_attention_custom_launcher_navi<T, KVT, KV_DTYPE, BLK_SIZE,        \
                                         HEAD_SIZE, OUTT, PSIZE,            \
                                         ALIBI_ENABLED, MFMA_TYPE>(         \
        out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,  \
        num_kv_heads, scale, block_tables, seq_lens, query_start_loc,       \
        max_seq_len, alibi_slopes, k_scale, v_scale);                       \
  }

#define CALL_CUSTOM_LAUNCHER_ALIBI(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,    \
                                   OUTT, PSIZE, MFMA_TYPE)                   \
  if (alibi_slopes) {                                                        \
    CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, PSIZE, \
                         true, MFMA_TYPE);                                   \
  } else {                                                                   \
    CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, PSIZE, \
                         false, MFMA_TYPE);                                  \
  }

#if defined(__HIPCC__) && defined(__gfx90a__)
  #define CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,  \
                                   MFMA_TYPE)                              \
    if (fp8_out_scale) {                                                   \
      TORCH_CHECK(false, "fp8 out scale unsupported for gfx90a");          \
    } else {                                                               \
      CALL_CUSTOM_LAUNCHER_ALIBI(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T, \
                                 256, MFMA_TYPE);                          \
    }
#else
  #define CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,  \
                                   MFMA_TYPE)                              \
    if (fp8_out_scale) {                                                   \
      CALL_CUSTOM_LAUNCHER_ALIBI(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,    \
                                 uint8_t, 256, MFMA_TYPE);                 \
    } else {                                                               \
      CALL_CUSTOM_LAUNCHER_ALIBI(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T, \
                                 256, MFMA_TYPE);                          \
    }
#endif

#define CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, HEAD_SIZE, MFMA_TYPE)    \
  switch (block_size) {                                                     \
    case 16:                                                                \
      CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, 16, HEAD_SIZE, MFMA_TYPE); \
      break;                                                                \
    case 32:                                                                \
      CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, 32, HEAD_SIZE, MFMA_TYPE); \
      break;                                                                \
    default:                                                                \
      TORCH_CHECK(false, "Unsupported block size: ", block_size);           \
      break;                                                                \
  }

#define CALL_CUSTOM_LAUNCHER_BLK_HEAD(T, KVT, KV_DTYPE, MFMA_TYPE) \
  switch (head_size) {                                             \
    case 64:                                                       \
      CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 64, MFMA_TYPE);   \
      break;                                                       \
    case 128:                                                      \
      CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 128, MFMA_TYPE);  \
      break;                                                       \
    default:                                                       \
      TORCH_CHECK(false, "Unsupported head size: ", head_size);    \
      break;                                                       \
  }

// clang-format off
void paged_attention_dispatch_fp8(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc,
    int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    torch::Tensor& k_scale, torch::Tensor& v_scale,
    const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type,
    bool is_navi) {
  // clang-format on
  const int head_size = query.size(2);
  if (query.dtype() == at::ScalarType::Half) {
    if (mfma_type == "fp8") {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(
          _Float16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3, MFMAType::Fp8);
    } else {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(
          _Float16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3, MFMAType::F16);
    }
  } else if (query.dtype() == at::ScalarType::BFloat16) {
    if (mfma_type == "fp8") {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(__hip_bfloat16, uint8_t,
                                    vllm::Fp8KVCacheDataType::kFp8E4M3,
                                    MFMAType::Fp8);
    } else {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(__hip_bfloat16, uint8_t,
                                    vllm::Fp8KVCacheDataType::kFp8E4M3,
                                    MFMAType::F16);
    }
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

#undef CALL_CUSTOM_LAUNCHER
#undef CALL_CUSTOM_LAUNCHER_ALIBI
#undef CALL_CUSTOM_LAUNCHER_OUT
#undef CALL_CUSTOM_LAUNCHER_BLK
#undef CALL_CUSTOM_LAUNCHER_BLK_HEAD
