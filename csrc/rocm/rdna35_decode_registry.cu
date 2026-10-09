// Torch ops over the attention variants built into _rocm_C.
//
// rdna35_decode_variant maps a variant's key (its row of
// vllm/v1/attention/ops/rdna35_variants.csv) to an index, once, when the
// backend picks a variant; rdna35_decode_attn launches by that index.
// rdna35_prefill_variant / rdna35_prefill_attn do the same for the rows of
// rdna35_prefill_variants.csv.  The .inc tables are generated from the CSVs
// by csrc/rocm/generate_rdna35_attn.py.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <string>

#include "rdna35_decode_attn.h"
#include "rdna35_prefill_variants.inc"
#include "rdna35_variants.inc"

namespace {

constexpr int64_t kNumVariants =
    sizeof(rdna35::kVariants) / sizeof(rdna35::kVariants[0]);
constexpr int64_t kNumPrefillVariants =
    sizeof(rdna35::kPrefillVariants) / sizeof(rdna35::kPrefillVariants[0]);

// The kernels carry code for gfx1151 only.
bool on_gfx1151() {
  const std::string arch = at::cuda::getCurrentDeviceProperties()->gcnArchName;
  return arch.find("gfx1151") != std::string::npos;
}

}  // namespace

int64_t rdna35_decode_variant(torch::IntArrayRef key) {
  TORCH_CHECK(key.size() == rdna35::kNumFields, "a variant key has ",
              rdna35::kNumFields, " fields, got ", key.size());
  if (!on_gfx1151()) return -1;
  for (int64_t i = 0; i < kNumVariants; ++i) {
    const int* k = rdna35::kVariants[i].key;
    if (std::equal(key.begin(), key.end(), k)) return i;
  }
  return -1;
}

// The scratch buffers (acc/m/l/cnt) are caller-allocated on purpose: this runs
// inside a CUDA-graph capture, and an allocation there would break it.
void rdna35_decode_attn(int64_t variant, torch::Tensor& q,
                        torch::Tensor& kv_cache, torch::Tensor& block_table,
                        torch::Tensor& out, torch::Tensor& acc,
                        torch::Tensor& m, torch::Tensor& l, torch::Tensor& cnt,
                        torch::Tensor& seq_lens, double scale) {
  using namespace rdna35;
  TORCH_CHECK(variant >= 0 && variant < kNumVariants, "no variant ", variant);
  const Variant& v = kVariants[variant];
  const int maxm = v.key[kCol_MAX_QUERY_LEN];
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous(),
              "q and out must be contiguous");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 &&
                  (block_table.dim() == 1 || block_table.dim() == 2) &&
                  block_table.stride(-1) == 1,
              "block table must be int32 rows, contiguous within a row");
  // A tensor, not an int: under CUDA-graph capture a host argument is frozen
  // at its capture-time value, and every replay would attend over that many
  // keys whatever the sequence has grown to.
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 && seq_lens.numel() >= 1,
              "seq_lens must be an int32 tensor holding the sequence length");
  TORCH_CHECK(q.size(0) % maxm == 0 && q.size(0) > 0, "q has ", q.size(0),
              " tokens, not a multiple of MAX_QUERY_LEN=", maxm);
  const int nseq = q.size(0) / maxm;
  TORCH_CHECK(seq_lens.numel() >= nseq, "seq_lens has ", seq_lens.numel(),
              " entries for ", nseq, " sequences");
  TORCH_CHECK(block_table.dim() == 2 ? block_table.size(0) >= nseq : nseq == 1,
              "block table has too few rows for ", nseq, " sequences");
  // Scratch is (sequences, ...) or, for one sequence, without that dim.
  const bool batched = acc.dim() == 3;
  TORCH_CHECK(batched ? acc.size(0) >= nseq && m.size(0) >= nseq &&
                            l.size(0) >= nseq && cnt.size(0) >= nseq
                      : nseq == 1,
              "scratch holds fewer sequences than the ", nseq, " launched");
  TORCH_CHECK(v.key[kCol_BATCHED] || nseq == 1,
              "one-sequence build launched for ", nseq, " sequences");
  TORCH_CHECK(
      q.size(1) == v.key[kCol_NUM_Q_HEADS] && q.size(2) == v.key[kCol_HEAD_DIM],
      "q shape does not match the compiled variant");
  // The loads reinterpret every tensor as the element type the variant was
  // built for.  The other 16-bit type would read the same bits and return
  // finite nonsense, so refuse it here rather than downstream.
  const auto dtype = v.key[kCol_BF16] ? at::kBFloat16 : at::kHalf;
  TORCH_CHECK(q.scalar_type() == dtype && kv_cache.scalar_type() == dtype &&
                  out.scalar_type() == dtype,
              "kernel built for ", dtype, ", got q=", q.scalar_type(),
              " kv_cache=", kv_cache.scalar_type(), " out=", out.scalar_type());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  v.fn({q.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr<int>(),
        acc.data_ptr<float>(), m.data_ptr<float>(), l.data_ptr<float>(),
        cnt.data_ptr<int>(), out.data_ptr(), seq_lens.data_ptr<int>(),
        (int)block_table.size(-1),
        block_table.dim() == 2 ? (int)block_table.stride(0) : 0,
        batched ? (int)acc.stride(0) : 0, batched ? (int)m.stride(0) : 0,
        batched ? (int)cnt.stride(0) : 0, nseq, (float)scale,
        at::cuda::getCurrentCUDAStream()});
}

int64_t rdna35_prefill_variant(torch::IntArrayRef key) {
  TORCH_CHECK(key.size() == rdna35::kPrefillNumFields, "a prefill key has ",
              rdna35::kPrefillNumFields, " fields, got ", key.size());
  if (!on_gfx1151()) return -1;
  for (int64_t i = 0; i < kNumPrefillVariants; ++i) {
    const int* k = rdna35::kPrefillVariants[i].key;
    if (std::equal(key.begin(), key.end(), k)) return i;
  }
  return -1;
}

// One sequence: q holds its num_tokens query tokens, seq_lens[0] its length
// with them; the scratch is make_prefill_scratch's.  Outside CUDA graphs (a
// prefill runs piecewise), so max_seq_len, the host's copy of S, may size the
// split.
void rdna35_prefill_attn(int64_t variant, torch::Tensor& q,
                         torch::Tensor& kv_cache, torch::Tensor& block_table,
                         torch::Tensor& out, torch::Tensor& partial_o,
                         torch::Tensor& partial_ml, torch::Tensor& counters,
                         torch::Tensor& seq_lens, int64_t max_seq_len,
                         double scale) {
  using namespace rdna35;
  TORCH_CHECK(variant >= 0 && variant < kNumPrefillVariants, "no variant ",
              variant);
  const PrefillVariant& v = kPrefillVariants[variant];
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous(),
              "q and out must be contiguous");
  TORCH_CHECK(
      block_table.scalar_type() == torch::kInt32 && block_table.stride(-1) == 1,
      "block table must be an int32 row, contiguous");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 && seq_lens.numel() >= 1,
              "seq_lens must be an int32 tensor holding the sequence length");
  TORCH_CHECK(q.size(0) > 0 && max_seq_len >= q.size(0),
              "max_seq_len must cover the query tokens");
  TORCH_CHECK(q.size(1) == v.key[kPrefillCol_NUM_Q_HEADS] &&
                  q.size(2) == v.key[kPrefillCol_HEAD_DIM],
              "q shape does not match the compiled variant");
  TORCH_CHECK(partial_o.scalar_type() == torch::kFloat32 &&
                  partial_ml.scalar_type() == torch::kFloat32 &&
                  counters.scalar_type() == torch::kInt32,
              "scratch must be float32 partials and int32 counters");
  const auto dtype = v.key[kPrefillCol_BF16] ? at::kBFloat16 : at::kHalf;
  TORCH_CHECK(q.scalar_type() == dtype && kv_cache.scalar_type() == dtype &&
                  out.scalar_type() == dtype,
              "kernel built for ", dtype, ", got q=", q.scalar_type(),
              " kv_cache=", kv_cache.scalar_type(), " out=", out.scalar_type());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  v.fn({q.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr<int>(),
        out.data_ptr(), seq_lens.data_ptr<int>(), partial_o.data_ptr<float>(),
        partial_ml.data_ptr<float>(), counters.data_ptr<int>(), (int)q.size(0),
        (int)max_seq_len, (float)scale, at::cuda::getCurrentCUDAStream()});
}
