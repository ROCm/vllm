#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define DECLARE_LAUNCH_BF16(N)                                           \
  template <typename scalar_t>                                           \
  void launch_wvsplitk_n##N(dim3 grid, cudaStream_t stream, int K_in,    \
                            int Kap_in, int Kbp_in, int M_in, int Bx_in, \
                            int By_in, const scalar_t* af4,              \
                            const scalar_t* bf4, const scalar_t* biasf4, \
                            scalar_t* c, int CuCount, int max_lds_len);

DECLARE_LAUNCH_BF16(1)
DECLARE_LAUNCH_BF16(2)
DECLARE_LAUNCH_BF16(3)
DECLARE_LAUNCH_BF16(4)
DECLARE_LAUNCH_BF16(5)

#undef DECLARE_LAUNCH_BF16
