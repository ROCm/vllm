#pragma once

// These macros reference kernel names and local variables from the launcher
// scope. They must be #included inside the launcher function body.

#define LAUNCH_CUSTOM_ATTENTION_MFMA16(GQA_RATIO)                              \
  paged_attention_ll4mi_QKV_mfma16_kernel<T, KVT, KV_DTYPE, OUTT, BLOCK_SIZE,  \
                                          HEAD_SIZE, NTHR, ALIBI_ENABLED,      \
                                          GQA_RATIO, MFMA_TYPE>                \
      <<<grid, block, 0, stream>>>(                                            \
          query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale,      \
          block_tables_ptr, seq_lens_ptr, query_start_loc_ptr,                 \
          max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride, \
          kv_head_stride, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, out_ptr,  \
          max_ctx_blocks, k_scale_ptr, v_scale_ptr);

#define LAUNCH_CUSTOM_ATTENTION_MFMA4(GQA_RATIO)                               \
  paged_attention_ll4mi_QKV_mfma4_kernel<T, KVT, KV_DTYPE, OUTT, BLOCK_SIZE,   \
                                         HEAD_SIZE, NTHR, ALIBI_ENABLED,       \
                                         GQA_RATIO>                            \
      <<<grid, block, 0, stream>>>(                                            \
          query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale,      \
          block_tables_ptr, seq_lens_ptr, query_start_loc_ptr,                 \
          max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride, \
          kv_head_stride, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, out_ptr,  \
          max_ctx_blocks, k_scale_ptr, v_scale_ptr);

#define LAUNCH_CUSTOM_REDUCTION(NPAR_LOOPS)                                 \
  paged_attention_ll4mi_reduce_kernel<T, OUTT, HEAD_SIZE, HEAD_SIZE,        \
                                      PARTITION_SIZE, NPAR_LOOPS>           \
      <<<reduce_grid, reduce_block, 0, stream>>>(                           \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, seq_lens_ptr, \
          query_start_loc_ptr, max_num_partitions, fp8_out_scale_ptr);
