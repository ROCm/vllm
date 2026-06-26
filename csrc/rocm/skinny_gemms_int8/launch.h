#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

// Per-N launchers for the wvSplitK_int8 kernel. Each instantiate_n{K}.cu TU
// defines launch_int8_n{K}() for a single N value, which keeps the template
// instantiation footprint per TU small and lets make parallelize across the
// shards. The dispatcher in the parent .cu picks (yt, ur) via the heuristic
// and then calls into the appropriate shard.

#define DECLARE_LAUNCH_INT8(N)                                                \
  template <typename scalar_t>                                                \
  void launch_int8_n##N(dim3 grid, cudaStream_t stream, int K, int M, int Bx, \
                        int By, const int8_t* B, const scalar_t* A,           \
                        const scalar_t* scale, const scalar_t* BIAS,          \
                        scalar_t* C, int CuCount, int thrds, int ytile,       \
                        int unrl, int64_t group_size);

DECLARE_LAUNCH_INT8(1)
DECLARE_LAUNCH_INT8(2)
DECLARE_LAUNCH_INT8(3)
DECLARE_LAUNCH_INT8(4)
DECLARE_LAUNCH_INT8(5)

#undef DECLARE_LAUNCH_INT8

#ifdef VLLM_SKINNY_GEMM_SWEEP

  #define DECLARE_LAUNCH_INT8_SWEEP(N)                                        \
    template <typename scalar_t>                                              \
    void launch_int8_n##N##_sweep(                                            \
        dim3 grid, cudaStream_t stream, int K, int M, int Bx, int By,         \
        const int8_t* B, const scalar_t* A, const scalar_t* scale,            \
        const scalar_t* BIAS, scalar_t* C, int CuCount, int thrds, int ytile, \
        int wvprgrp, int achunk, int unrl);

DECLARE_LAUNCH_INT8_SWEEP(1)
DECLARE_LAUNCH_INT8_SWEEP(2)
DECLARE_LAUNCH_INT8_SWEEP(3)
DECLARE_LAUNCH_INT8_SWEEP(4)
DECLARE_LAUNCH_INT8_SWEEP(5)

  #undef DECLARE_LAUNCH_INT8_SWEEP

#endif  // VLLM_SKINNY_GEMM_SWEEP
