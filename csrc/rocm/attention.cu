/*
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#include <string>

// Per-dtype dispatch shards (defined in attention/dispatch_auto.cu and
// attention/dispatch_fp8.cu).
void paged_attention_dispatch_auto(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    torch::Tensor& k_scale, torch::Tensor& v_scale,
    const std::optional<torch::Tensor>& fp8_out_scale, bool is_navi);

void paged_attention_dispatch_fp8(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    torch::Tensor& k_scale, torch::Tensor& v_scale,
    const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type, bool is_navi);

bool is_navi_gpu() {
  static bool is_cached = false;
  static bool result;

  if (!is_cached) {
    int device_id;
    hipDeviceProp_t deviceProp;
    hipGetDevice(&device_id);
    hipGetDeviceProperties(&deviceProp, device_id);

    std::string arch = deviceProp.gcnArchName;
    result = arch.find("gfx11") == 0 || arch.find("gfx12") == 0;
    is_cached = true;
  }

  return result;
}

// clang-format off
void paged_attention(
    torch::Tensor& out,         // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& tmp_out,     // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,       // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor& value_cache, // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables, // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens, // [num_seqs]
    const std::optional<torch::Tensor>& query_start_loc, // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale,
    const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type) {
  // clang-format on
  bool is_navi = is_navi_gpu();
  if (kv_cache_dtype == "auto") {
    paged_attention_dispatch_auto(out, exp_sums, max_logits, tmp_out, query,
                                  key_cache, value_cache, num_kv_heads, scale,
                                  block_tables, seq_lens, query_start_loc,
                                  block_size, max_seq_len, alibi_slopes,
                                  k_scale, v_scale, fp8_out_scale, is_navi);
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    paged_attention_dispatch_fp8(out, exp_sums, max_logits, tmp_out, query,
                                 key_cache, value_cache, num_kv_heads, scale,
                                 block_tables, seq_lens, query_start_loc,
                                 block_size, max_seq_len, alibi_slopes, k_scale,
                                 v_scale, fp8_out_scale, mfma_type, is_navi);
  } else {
    TORCH_CHECK(false, "Unsupported KV cache dtype: ", kv_cache_dtype);
  }
}
