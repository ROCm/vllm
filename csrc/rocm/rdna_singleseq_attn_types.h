// The head dims, element types and GPU families the RDNA3.5 attention kernels
// are built for.
#pragma once

namespace rdna_singleseq {

enum class HeadDim : int {
  k64 = 64,
  k96 = 96,
  k128 = 128,
  k256 = 256,
  k512 = 512
};

enum class Dtype : int { kFp16 = 0, kBf16 = 1 };

// The GPU families with their own config tables, each built for its archs
// only (cmake/rocm_rdna_singleseq_attn.cmake): gfx115X is gfx1151, gfx120X
// is RDNA4 (gfx1200, gfx1201).
enum class Arch : int { kGfx115X = 0, kGfx120X = 1 };
inline constexpr int kNumArches = 2;

}  // namespace rdna_singleseq
