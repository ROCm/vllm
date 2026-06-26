// Per-N kernel instantiation shard for wvSplitK_int8 (N=1).
// One TU per N value keeps the template instantiation footprint per file
// small and lets make parallelize across the shards. See dispatch.cuh.

#include "dispatch.cuh"
#include "launch.h"

template <typename scalar_t>
void launch_int8_n1(dim3 grid, cudaStream_t stream, int K, int M, int Bx,
                    int By, const int8_t* B, const scalar_t* A,
                    const scalar_t* scale, const scalar_t* BIAS, scalar_t* C,
                    int CuCount, int thrds, int ytile, int unrl,
                    int64_t group_size) {
  dispatch_int8<scalar_t, 1>(grid, stream, K, M, Bx, By, B, A, scale, BIAS, C,
                             CuCount, thrds, ytile, unrl, group_size);
}

template void launch_int8_n1<half>(dim3, cudaStream_t, int, int, int, int,
                                   const int8_t*, const half*, const half*,
                                   const half*, half*, int, int, int, int,
                                   int64_t);
template void launch_int8_n1<__hip_bfloat16>(
    dim3, cudaStream_t, int, int, int, int, const int8_t*,
    const __hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*,
    __hip_bfloat16*, int, int, int, int, int64_t);

#ifdef VLLM_SKINNY_GEMM_SWEEP

template <typename scalar_t>
void launch_int8_n1_sweep(dim3 grid, cudaStream_t stream, int K, int M, int Bx,
                          int By, const int8_t* B, const scalar_t* A,
                          const scalar_t* scale, const scalar_t* BIAS,
                          scalar_t* C, int CuCount, int thrds, int ytile,
                          int wvprgrp, int achunk, int unrl) {
  dispatch_int8_sweep<scalar_t, 1>(grid, stream, K, M, Bx, By, B, A, scale,
                                   BIAS, C, CuCount, thrds, ytile, wvprgrp,
                                   achunk, unrl);
}

template void launch_int8_n1_sweep<half>(dim3, cudaStream_t, int, int, int, int,
                                         const int8_t*, const half*,
                                         const half*, const half*, half*, int,
                                         int, int, int, int, int);
template void launch_int8_n1_sweep<__hip_bfloat16>(
    dim3, cudaStream_t, int, int, int, int, const int8_t*,
    const __hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*,
    __hip_bfloat16*, int, int, int, int, int, int);

#endif  // VLLM_SKINNY_GEMM_SWEEP
