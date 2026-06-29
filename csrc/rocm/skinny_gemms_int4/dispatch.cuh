#pragma once

#include "../skinny_gemms_int4_kernels.cuh"

// Per-N dispatch for wvSplitK_int4_g (standard GEMM).
// Wraps WVSPLIT_INT4G_TILE for a single N value.
template <typename scalar_t, int N_VAL>
inline void dispatch_int4_g(dim3 grid, cudaStream_t stream, int K_in, int M_in,
                            int Bx_in, int By_in, const uint8_t* wptr,
                            const scalar_t* aptr, const scalar_t* sptr,
                            const scalar_t* zpptr, const scalar_t* biasptr,
                            scalar_t* cptr, int CuCount,
                            int b_row_stride_bytes_i32, int64_t group_size,
                            int max_lds_len, bool has_zp) {
  using fptype = scalar_t;
  int N_in = N_VAL;
  int sYT = (M_in + CuCount * 4 - 1) / (CuCount * 4);
  if (has_zp) {
    WVSPLIT_INT4G_TILE(sYT, N_VAL, true)
  } else {
    WVSPLIT_INT4G_TILE(sYT, N_VAL, false)
  }
}

// Per-N dispatch for fused_moe_wvSplitK_int4_gemm (MoE GEMM).
// Wraps MOE_WVSPLIT_INT4G_TILE for a single N value.
template <typename scalar_t, int N_VAL>
inline void dispatch_moe_int4_g(dim3 grid, cudaStream_t stream, int K_in,
                                int M_in, int N_in, const uint8_t* wptr,
                                const scalar_t* aptr, const scalar_t* sptr,
                                const scalar_t* zpptr, scalar_t* cptr,
                                const int* eidptr, const int* stidptr,
                                int top_k_in, long expert_stride_w,
                                long expert_stride_s, long expert_stride_zp,
                                int CuCount, int num_expert_blocks,
                                int64_t group_size, int max_lds_len,
                                bool has_zp, bool fuse_silu_mul) {
  using fptype = scalar_t;
  int sYT = (M_in + CuCount * 4 - 1) / (CuCount * 4);
  if (has_zp) {
    MOE_WVSPLIT_INT4G_TILE(sYT, N_VAL, true)
  } else {
    MOE_WVSPLIT_INT4G_TILE(sYT, N_VAL, false)
  }
}
