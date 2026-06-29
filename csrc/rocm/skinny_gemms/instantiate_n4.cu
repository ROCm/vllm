// Per-N kernel instantiation shard for wvSplitK bf16/fp16 (N=4).

#include "dispatch.cuh"
#include "launch.h"

template <typename scalar_t>
void launch_wvsplitk_n4(dim3 grid, cudaStream_t stream, int K_in, int Kap_in,
                        int Kbp_in, int M_in, int Bx_in, int By_in,
                        const scalar_t* af4, const scalar_t* bf4,
                        const scalar_t* biasf4, scalar_t* c, int CuCount,
                        int max_lds_len) {
  dispatch_wvsplitk<scalar_t, 4>(grid, stream, K_in, Kap_in, Kbp_in, M_in,
                                 Bx_in, By_in, af4, bf4, biasf4, c, CuCount,
                                 max_lds_len);
}

template void launch_wvsplitk_n4<half>(dim3, cudaStream_t, int, int, int, int,
                                       int, int, const half*, const half*,
                                       const half*, half*, int, int);
template void launch_wvsplitk_n4<__hip_bfloat16>(
    dim3, cudaStream_t, int, int, int, int, int, int, const __hip_bfloat16*,
    const __hip_bfloat16*, const __hip_bfloat16*, __hip_bfloat16*, int, int);
