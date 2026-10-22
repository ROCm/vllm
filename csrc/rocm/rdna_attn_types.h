// The head dims and element types the RDNA3.5 attention kernels are built for.
#pragma once

namespace rdna {

enum class HeadDim : int {
  k32 = 32,
  k48 = 48,
  k64 = 64,
  k80 = 80,
  k96 = 96,
  k128 = 128,
  k256 = 256,
  k512 = 512
};

enum class Dtype : int { kFp16 = 0, kBf16 = 1 };

}  // namespace rdna
