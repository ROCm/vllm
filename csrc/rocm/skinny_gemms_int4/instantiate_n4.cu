// Per-N kernel instantiation shard for wvSplitK_int4 (N=4).

#include "dispatch.cuh"
#include "launch.h"

template <typename scalar_t>
void launch_int4_n4(dim3 grid, cudaStream_t stream, int K_in, int M_in,
                    int Bx_in, int By_in, const uint8_t* wptr,
                    const scalar_t* aptr, const scalar_t* sptr,
                    const scalar_t* zpptr, const scalar_t* biasptr,
                    scalar_t* cptr, int CuCount, int b_row_stride_bytes_i32,
                    int64_t group_size, int max_lds_len, bool has_zp) {
  dispatch_int4_g<scalar_t, 4>(
      grid, stream, K_in, M_in, Bx_in, By_in, wptr, aptr, sptr, zpptr, biasptr,
      cptr, CuCount, b_row_stride_bytes_i32, group_size, max_lds_len, has_zp);
}

template void launch_int4_n4<half>(dim3, cudaStream_t, int, int, int, int,
                                   const uint8_t*, const half*, const half*,
                                   const half*, const half*, half*, int, int,
                                   int64_t, int, bool);
template void launch_int4_n4<__hip_bfloat16>(
    dim3, cudaStream_t, int, int, int, int, const uint8_t*,
    const __hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*,
    const __hip_bfloat16*, __hip_bfloat16*, int, int, int64_t, int, bool);

template <typename scalar_t>
void launch_moe_int4_n4(dim3 grid, cudaStream_t stream, int K_in, int M_in,
                        int N_in, const uint8_t* wptr, const scalar_t* aptr,
                        const scalar_t* sptr, const scalar_t* zpptr,
                        scalar_t* cptr, const int* eidptr, const int* stidptr,
                        int top_k_in, long expert_stride_w,
                        long expert_stride_s, long expert_stride_zp,
                        int CuCount, int num_expert_blocks, int64_t group_size,
                        int max_lds_len, bool has_zp, bool fuse_silu_mul) {
  dispatch_moe_int4_g<scalar_t, 4>(
      grid, stream, K_in, M_in, N_in, wptr, aptr, sptr, zpptr, cptr, eidptr,
      stidptr, top_k_in, expert_stride_w, expert_stride_s, expert_stride_zp,
      CuCount, num_expert_blocks, group_size, max_lds_len, has_zp,
      fuse_silu_mul);
}

template void launch_moe_int4_n4<half>(dim3, cudaStream_t, int, int, int,
                                       const uint8_t*, const half*, const half*,
                                       const half*, half*, const int*,
                                       const int*, int, long, long, long, int,
                                       int, int64_t, int, bool, bool);
template void launch_moe_int4_n4<__hip_bfloat16>(
    dim3, cudaStream_t, int, int, int, const uint8_t*, const __hip_bfloat16*,
    const __hip_bfloat16*, const __hip_bfloat16*, __hip_bfloat16*, const int*,
    const int*, int, long, long, long, int, int, int64_t, int, bool, bool);
