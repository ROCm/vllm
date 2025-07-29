#pragma once

#include <torch/all.h>

void paged_attention(torch::Tensor& out, torch::Tensor& exp_sums,
                     torch::Tensor& max_logits, torch::Tensor& tmp_out,
                     torch::Tensor& query, torch::Tensor& key_cache,
                     torch::Tensor& value_cache, int64_t num_kv_heads,
                     double scale, torch::Tensor& block_tables,
                     torch::Tensor& context_lens,
                     const std::optional<torch::Tensor>& query_start_loc,
                     int64_t block_size, int64_t max_context_len,
                     const std::optional<torch::Tensor>& alibi_slopes,
                     const std::string& kv_cache_dtype, torch::Tensor& k_scale,
                     torch::Tensor& v_scale);

void wvSplitKQ(torch::Tensor& in_a, torch::Tensor& in_b, torch::Tensor& out_c,
               torch::Tensor& scale_a, torch::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor wvSplitK(torch::Tensor& in_a, torch::Tensor& in_b,
                       const int64_t CuCount);

// From
// https://github.com/changcui-amd/aiter/commit/2a87205770ab0e3cb5f1e87ab4876ef25835b046#diff-fff478858fb9d2f35dcbb4edcdfa8c6f158f866d3821890594ce57b04b2af59b

void biased_grouped_topk(
    torch::Tensor& gating_output,    // [num_tokens, num_experts]
    torch::Tensor& correction_bias,  // [num_expert]
    torch::Tensor& topk_weights,     // [num_tokens, topk]
    torch::Tensor& topk_ids,         // [num_tokens, topk]
    int64_t num_expert_group, int64_t topk_group, bool renormalize,
    const double routed_scaling_factor = 1.,
    const int64_t num_fused_shared_experts = 0);

void grouped_topk(torch::Tensor& gating_output,  // [num_tokens, num_experts]
                  torch::Tensor& topk_weights,   // [num_tokens, topk]
                  torch::Tensor& topk_ids,       // [num_tokens, topk]
                  int64_t num_expert_group, int64_t topk_grp, bool need_renorm,
                  std::string scoring_func = "softmax",
                  const double routed_scaling_factor = 1.,
                  const int64_t num_fused_shared_experts = 0);