// Torch ops over the RDNA3.5 attention variants built into _rocm_C.
// rdna_singleseq_{decode,prefill}_variant maps a CSV row to an index, and
// rdna_singleseq_{decode,prefill}_attn launches by that index.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <initializer_list>
#include <string>

#include "rdna_singleseq_attn_launch.h"
#include "rdna_singleseq_prefill_variants.inc"
#include "rdna_singleseq_decode_variants.inc"

namespace {

constexpr int64_t kNumVariants =
    sizeof(rdna_singleseq::kVariants) / sizeof(rdna_singleseq::kVariants[0]);
constexpr int64_t kNumPrefillVariants =
    sizeof(rdna_singleseq::kPrefillVariants) /
    sizeof(rdna_singleseq::kPrefillVariants[0]);

bool on_gfx1151() {
  const std::string arch = at::cuda::getCurrentDeviceProperties()->gcnArchName;
  return arch.find("gfx1151") != std::string::npos;
}

// The kernels expect (num_pages, NUM_KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM).
void check_kv_cache(const torch::Tensor& kv_cache, int num_kv_heads,
                    int page_size, int head_dim) {
  const int64_t row = 2 * head_dim;
  TORCH_CHECK(kv_cache.dim() == 4 && kv_cache.size(1) == num_kv_heads &&
                  kv_cache.size(2) == page_size && kv_cache.size(3) == row &&
                  kv_cache.stride(3) == 1 && kv_cache.stride(2) == row &&
                  kv_cache.stride(1) == page_size * row &&
                  kv_cache.stride(0) == num_kv_heads * page_size * row,
              "kv_cache must be packed HND (pages, ", num_kv_heads, ", ",
              page_size, ", ", row, "), got sizes ", kv_cache.sizes(),
              " strides ", kv_cache.strides());
}

void check_device(const torch::Tensor& q,
                  std::initializer_list<const torch::Tensor*> others) {
  for (const torch::Tensor* t : others)
    TORCH_CHECK(t->device() == q.device(), "every tensor must be on ",
                q.device(), ", got one on ", t->device());
}

// The built variants as CSV.
template <typename V, int64_t N>
std::string as_csv(const char* const* cols, int fields, const V (&vs)[N]) {
  std::string s;
  for (int c = 0; c < fields; ++c) {
    if (c) s += ',';
    s += cols[c];
  }
  for (const V& v : vs) {
    s += '\n';
    for (int c = 0; c < fields; ++c) {
      if (c) s += ',';
      s += std::to_string(v.key[c]);
    }
  }
  return s;
}

}  // namespace

std::string rdna_singleseq_decode_table() {
  return as_csv(rdna_singleseq::kColumns, rdna_singleseq::kNumFields,
                rdna_singleseq::kVariants);
}

std::string rdna_singleseq_prefill_table() {
  return as_csv(rdna_singleseq::kPrefillColumns,
                rdna_singleseq::kPrefillNumFields,
                rdna_singleseq::kPrefillVariants);
}

int64_t rdna_singleseq_decode_variant(torch::IntArrayRef key) {
  TORCH_CHECK(key.size() == rdna_singleseq::kNumFields, "a variant key has ",
              rdna_singleseq::kNumFields, " fields, got ", key.size());
  if (!on_gfx1151()) return -1;
  for (int64_t i = 0; i < kNumVariants; ++i) {
    const int* k = rdna_singleseq::kVariants[i].key;
    if (std::equal(key.begin(), key.end(), k)) return i;
  }
  return -1;
}

// The caller allocates the scratch: this runs under CUDA-graph capture.
void rdna_singleseq_decode_attn(int64_t variant, torch::Tensor& q,
                                torch::Tensor& kv_cache,
                                torch::Tensor& block_table, torch::Tensor& out,
                                torch::Tensor& acc, torch::Tensor& m,
                                torch::Tensor& l, torch::Tensor& cnt,
                                torch::Tensor& seq_lens, double scale) {
  using namespace rdna_singleseq;
  TORCH_CHECK(variant >= 0 && variant < kNumVariants, "no variant ", variant);
  const Variant& v = kVariants[variant];
  const int maxm = v.key[kCol_MAX_QUERY_LEN];
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous(),
              "q and out must be contiguous");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 &&
                  (block_table.dim() == 1 || block_table.dim() == 2) &&
                  block_table.stride(-1) == 1,
              "block table must be int32 rows, contiguous within a row");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 && seq_lens.numel() >= 1,
              "seq_lens must be an int32 tensor holding the sequence length");
  // One sequence: row 0 of the block table and seq_lens[0].
  TORCH_CHECK(q.size(0) == maxm, "q has ", q.size(0),
              " tokens, not one sequence of MAX_QUERY_LEN=", maxm);
  TORCH_CHECK(block_table.dim() == 1 || block_table.size(0) >= 1,
              "block table has no row");
  TORCH_CHECK(
      q.size(1) == v.key[kCol_NUM_Q_HEADS] && q.size(2) == v.key[kCol_HEAD_DIM],
      "q shape does not match the compiled variant");
  const auto dtype = v.key[kCol_BF16] ? at::kBFloat16 : at::kHalf;
  TORCH_CHECK(q.scalar_type() == dtype && kv_cache.scalar_type() == dtype &&
                  out.scalar_type() == dtype,
              "kernel built for ", dtype, ", got q=", q.scalar_type(),
              " kv_cache=", kv_cache.scalar_type(), " out=", out.scalar_type());
  TORCH_CHECK(out.numel() == q.numel(), "out must hold q's ", q.numel(),
              " elements, got ", out.numel());
  const int hq = v.key[kCol_NUM_Q_HEADS], hkv = v.key[kCol_NUM_KV_HEADS];
  const int d = v.key[kCol_HEAD_DIM];
  check_kv_cache(kv_cache, hkv, v.key[kCol_PAGE_SIZE], d);
  check_device(q,
               {&kv_cache, &block_table, &out, &acc, &m, &l, &cnt, &seq_lens});
  // As KernelVariant.scratch_shapes.
  const int segments = v.key[kCol_MAX_SEGMENTS];
  int64_t rows, counters;
  if (v.key[kCol_DOT_PRODUCT]) {
    rows = (int64_t)hq * segments * maxm;
    counters = hq;
  } else {
    const int groups = v.key[kCol_ROW_GROUPS];
    const int64_t padded = ((hq / hkv) * maxm / groups + 15) / 16 * 16;
    rows = (int64_t)hkv * groups * segments * padded;
    counters = 4 * hkv * groups;
  }
  TORCH_CHECK(acc.scalar_type() == torch::kFloat32 &&
                  m.scalar_type() == torch::kFloat32 &&
                  l.scalar_type() == torch::kFloat32 &&
                  cnt.scalar_type() == torch::kInt32,
              "scratch must be float32 partials and int32 counters");
  TORCH_CHECK(acc.numel() >= rows * d && m.numel() >= rows &&
                  l.numel() >= rows && cnt.numel() >= counters,
              "scratch too small for this variant: needs ", rows * d, ", ",
              rows, ", ", rows, ", ", counters, " elements");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  v.fn({q.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr<int>(),
        acc.data_ptr<float>(), m.data_ptr<float>(), l.data_ptr<float>(),
        cnt.data_ptr<int>(), out.data_ptr(), seq_lens.data_ptr<int>(),
        (int)block_table.size(-1), (float)scale,
        at::cuda::getCurrentCUDAStream()});
}

int64_t rdna_singleseq_prefill_variant(torch::IntArrayRef key) {
  TORCH_CHECK(key.size() == rdna_singleseq::kPrefillNumFields,
              "a prefill key has ", rdna_singleseq::kPrefillNumFields,
              " fields, got ", key.size());
  if (!on_gfx1151()) return -1;
  for (int64_t i = 0; i < kNumPrefillVariants; ++i) {
    const int* k = rdna_singleseq::kPrefillVariants[i].key;
    if (std::equal(key.begin(), key.end(), k)) return i;
  }
  return -1;
}

// One sequence; runs outside CUDA graphs, so max_seq_len may size the split.
void rdna_singleseq_prefill_attn(
    int64_t variant, torch::Tensor& q, torch::Tensor& kv_cache,
    torch::Tensor& block_table, torch::Tensor& out, torch::Tensor& partial_o,
    torch::Tensor& partial_ml, torch::Tensor& counters, torch::Tensor& seq_lens,
    int64_t max_seq_len, double scale) {
  using namespace rdna_singleseq;
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
  TORCH_CHECK(out.numel() == q.numel(), "out must hold q's ", q.numel(),
              " elements, got ", out.numel());
  const int d = v.key[kPrefillCol_HEAD_DIM];
  check_kv_cache(kv_cache, v.key[kPrefillCol_NUM_KV_HEADS],
                 v.key[kPrefillCol_PAGE_SIZE], d);
  check_device(q, {&kv_cache, &block_table, &out, &partial_o, &partial_ml,
                   &counters, &seq_lens});
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  if (v.key[kPrefillCol_MAX_SEGMENTS] > 1) {
    // As PrefillVariant.scratch_rows.
    const int64_t target =
        4 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t block_rows =
        v.key[kPrefillCol_WAVES] /
        (v.key[kPrefillCol_HEAD_DIM_SPLIT] * v.key[kPrefillCol_KEY_WAVES]) * 16;
    const int64_t rows = 2 * target * block_rows;
    TORCH_CHECK(partial_o.numel() >= rows * d &&
                    partial_ml.numel() >= 2 * rows &&
                    counters.numel() >= target,
                "scratch too small for this variant: needs ", rows * d, ", ",
                2 * rows, ", ", target, " elements");
  }
  v.fn({q.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr<int>(),
        out.data_ptr(), seq_lens.data_ptr<int>(), partial_o.data_ptr<float>(),
        partial_ml.data_ptr<float>(), counters.data_ptr<int>(), (int)q.size(0),
        (int)max_seq_len, (float)scale, at::cuda::getCurrentCUDAStream()});
}
