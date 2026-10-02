// Torch ops over the decode-attention variants built into _rocm_C.
//
// rdna35_decode_variant maps a variant's key (the fields of its variants.def
// line) to an index, once, when the backend picks a variant;
// rdna35_decode_attn launches by that index.  The checks are the JIT op's
// (decode_attn_op in rdna35_decode_attn.cu), read from the key instead of
// from the defines.
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>

#include <algorithm>
#include <cstring>

#include "rdna35_attn/variants.h"

namespace {

enum Field {
  kHeadDim,
  kNumQHeads,
  kNumKvHeads,
  kMaxM,
  kBlockSize,
  kLayout,
  kNseg,
  kRg,
  kMinb,
  kNw,
  kDspl,
  kRspl,
  kPf,
  kCpub,
  kVinlds,
  kBatch,
  kDot,
  kBfly,
  kGt,
  kWin,
  kBf16,
};
static_assert(kBf16 + 1 == RDNA35_NFIELDS, "one Field per key field");

constexpr int64_t kNumVariants =
    sizeof(rdna35::kVariants) / sizeof(rdna35::kVariants[0]);

}  // namespace

int64_t rdna35_decode_variant(torch::IntArrayRef key) {
  TORCH_CHECK(key.size() == RDNA35_NFIELDS, "a variant key has ",
              RDNA35_NFIELDS, " fields, got ", key.size());
  // The kernels carry code for gfx1151 only (rdna35_decode_attn.cu).
  int dev;
  hipDeviceProp_t prop;
  if (hipGetDevice(&dev) != hipSuccess ||
      hipGetDeviceProperties(&prop, dev) != hipSuccess ||
      !strstr(prop.gcnArchName, "gfx1151"))
    return -1;
  for (int64_t i = 0; i < kNumVariants; ++i) {
    const int* k = rdna35::kVariants[i].key;
    if (std::equal(key.begin(), key.end(), k)) return i;
  }
  return -1;
}

void rdna35_decode_attn(int64_t variant, torch::Tensor& q,
                        torch::Tensor& kv_cache, torch::Tensor& block_table,
                        torch::Tensor& out, torch::Tensor& acc,
                        torch::Tensor& m, torch::Tensor& l, torch::Tensor& cnt,
                        torch::Tensor& seq_lens, double scale) {
  TORCH_CHECK(variant >= 0 && variant < kNumVariants, "no variant ", variant);
  const rdna35::Variant& v = rdna35::kVariants[variant];
  const int maxm = v.key[kMaxM];
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous(),
              "q and out must be contiguous");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 &&
                  (block_table.dim() == 1 || block_table.dim() == 2) &&
                  block_table.stride(-1) == 1,
              "block table must be int32 rows, contiguous within a row");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 && seq_lens.numel() >= 1,
              "seq_lens must be an int32 tensor holding the sequence length");
  TORCH_CHECK(q.size(0) % maxm == 0 && q.size(0) > 0, "q has ", q.size(0),
              " tokens, not a multiple of MAXM=", maxm);
  const int nseq = q.size(0) / maxm;
  TORCH_CHECK(seq_lens.numel() >= nseq, "seq_lens has ", seq_lens.numel(),
              " entries for ", nseq, " sequences");
  TORCH_CHECK(block_table.dim() == 2 ? block_table.size(0) >= nseq : nseq == 1,
              "block table has too few rows for ", nseq, " sequences");
  const bool batched = acc.dim() == 3;
  TORCH_CHECK(batched ? acc.size(0) >= nseq && m.size(0) >= nseq &&
                            l.size(0) >= nseq && cnt.size(0) >= nseq
                      : nseq == 1,
              "scratch holds fewer sequences than the ", nseq, " launched");
  TORCH_CHECK(v.key[kBatch] || nseq == 1, "one-sequence build launched for ",
              nseq, " sequences");
  TORCH_CHECK(q.size(1) == v.key[kNumQHeads] && q.size(2) == v.key[kHeadDim],
              "q shape does not match the compiled variant");
  const auto dtype = v.key[kBf16] ? at::kBFloat16 : at::kHalf;
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
