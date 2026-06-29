#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

#define DECLARE_LAUNCH_INT4(N)                                                 \
  template <typename scalar_t>                                                 \
  void launch_int4_n##N(                                                       \
      dim3 grid, cudaStream_t stream, int K_in, int M_in, int Bx_in,           \
      int By_in, const uint8_t* wptr, const scalar_t* aptr,                    \
      const scalar_t* sptr, const scalar_t* zpptr, const scalar_t* biasptr,    \
      scalar_t* cptr, int CuCount, int b_row_stride_bytes_i32,                 \
      int64_t group_size, int max_lds_len, bool has_zp);                       \
                                                                               \
  template <typename scalar_t>                                                 \
  void launch_moe_int4_n##N(                                                   \
      dim3 grid, cudaStream_t stream, int K_in, int M_in, int N_in,            \
      const uint8_t* wptr, const scalar_t* aptr, const scalar_t* sptr,         \
      const scalar_t* zpptr, scalar_t* cptr, const int* eidptr,                \
      const int* stidptr, int top_k_in, long expert_stride_w,                  \
      long expert_stride_s, long expert_stride_zp, int CuCount,                \
      int num_expert_blocks, int64_t group_size, int max_lds_len, bool has_zp, \
      bool fuse_silu_mul);

DECLARE_LAUNCH_INT4(1)
DECLARE_LAUNCH_INT4(2)
DECLARE_LAUNCH_INT4(3)
DECLARE_LAUNCH_INT4(4)
DECLARE_LAUNCH_INT4(5)

#undef DECLARE_LAUNCH_INT4
