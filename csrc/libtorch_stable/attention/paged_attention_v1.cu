/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "paged_attention_v1/common.h"

void paged_attention_v1(
    torch::stable::Tensor& out,    // [num_seqs, num_heads, head_size]
    torch::stable::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::stable::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::stable::Tensor&
        value_cache,       // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    torch::stable::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::stable::Tensor& seq_lens,      // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::stable::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  int head_size = query.size(2);
  switch (head_size) {
    case 32:
    case 64:
    case 80:
      paged_attention_v1_small(
          out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
          seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype,
          k_scale, v_scale, tp_rank, blocksparse_local_blocks,
          blocksparse_vert_stride, blocksparse_block_size,
          blocksparse_head_sliding_step);
      break;
    case 96:
    case 112:
    case 120:
    case 128:
      paged_attention_v1_medium(
          out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
          seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype,
          k_scale, v_scale, tp_rank, blocksparse_local_blocks,
          blocksparse_vert_stride, blocksparse_block_size,
          blocksparse_head_sliding_step);
      break;
    case 192:
    case 256:
      paged_attention_v1_large(
          out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
          seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype,
          k_scale, v_scale, tp_rank, blocksparse_local_blocks,
          blocksparse_vert_stride, blocksparse_block_size,
          blocksparse_head_sliding_step);
      break;
    default:
      STD_TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}
