#pragma once

#include "kernel.cuh"

// Production launch dispatch for wvSplitK_int8.
// The heuristic in the parent .cu picks (ytile, unrl) and group_size,
// then calls into the per-N shard via launch_int8_nX().
template <typename scalar_t, int N_VAL>
inline void dispatch_int8(dim3 grid, cudaStream_t stream, int K, int M, int Bx,
                          int By, const int8_t* B, const scalar_t* A,
                          const scalar_t* scale, const scalar_t* BIAS,
                          scalar_t* C, int CuCount, int thrds, int ytile,
                          int unrl, int64_t group_size) {
#define LAUNCH_G(_THRDS, _YTILE, _UNRL, _GROUP)                           \
  do {                                                                    \
    dim3 block(_THRDS, 16);                                               \
    int __wvPrGrp = mindiv_int8(M, CuCount * (_YTILE), 16);               \
    wvSplitK_int8_hf_sml_<scalar_t, _THRDS, _YTILE, 16, 16, _UNRL, N_VAL, \
                          _GROUP><<<grid, block, 0, stream>>>(            \
        K, M, Bx, By, B, A, scale, BIAS, C, __wvPrGrp, CuCount);          \
    return;                                                               \
  } while (0)

#define LAUNCH(_THRDS, _YTILE, _UNRL)       \
  do {                                      \
    if (group_size == -1) {                 \
      LAUNCH_G(_THRDS, _YTILE, _UNRL, 0);   \
    } else if (group_size == 32) {          \
      LAUNCH_G(_THRDS, _YTILE, _UNRL, 32);  \
    } else if (group_size == 64) {          \
      LAUNCH_G(_THRDS, _YTILE, _UNRL, 64);  \
    } else if (group_size == 128) {         \
      LAUNCH_G(_THRDS, _YTILE, _UNRL, 128); \
    }                                       \
  } while (0)

  // Enumerate the (thrds, ytile, unrl) tuples WVSPLIT_INT8_TILE picks.
  if (thrds == 32) {
    if (ytile == 4 && unrl == 1) LAUNCH(32, 4, 1);
    if (ytile == 1 && unrl == 4) LAUNCH(32, 1, 4);
  } else {
    if (ytile == 4 && unrl == 1) LAUNCH(64, 4, 1);
    if (ytile == 1 && unrl == 4) LAUNCH(64, 1, 4);
  }
  TORCH_CHECK(false, "wvSplitK_int8: unhandled (thrds=", thrds,
              ", ytile=", ytile, ", unrl=", unrl, ", group_size=", group_size,
              "). Add to dispatch.cuh.");

#undef LAUNCH_G
#undef LAUNCH
}

#ifdef VLLM_SKINNY_GEMM_SWEEP

// Sweep dispatch: full (yt, ur, ac, wv) cross-product.
template <typename scalar_t, int N_VAL>
inline void dispatch_int8_sweep(dim3 grid, cudaStream_t stream, int K, int M,
                                int Bx, int By, const int8_t* B,
                                const scalar_t* A, const scalar_t* scale,
                                const scalar_t* BIAS, scalar_t* C, int CuCount,
                                int thrds, int ytile, int wvprgrp, int achunk,
                                int unrl) {
  #define SWEEP_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, _UNRL)         \
    do {                                                                 \
      dim3 block(_THRDS, _WVPRGRP);                                      \
      int __wvPrGrp = mindiv_int8(M, CuCount * (_YTILE), _WVPRGRP);      \
      wvSplitK_int8_hf_sml_<scalar_t, _THRDS, _YTILE, _WVPRGRP, _ACHUNK, \
                            _UNRL, N_VAL><<<grid, block, 0, stream>>>(   \
          K, M, Bx, By, B, A, scale, BIAS, C, __wvPrGrp, CuCount);       \
      return;                                                            \
    } while (0)

  #define SWEEP_UNRL(_THRDS, _YTILE, _WVPRGRP, _ACHUNK)                  \
    do {                                                                 \
      if (unrl == 1) SWEEP_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 1); \
      if (unrl == 2) SWEEP_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 2); \
      if (unrl == 4) SWEEP_LAUNCH(_THRDS, _YTILE, _WVPRGRP, _ACHUNK, 4); \
    } while (0)

  #define SWEEP_YTILE(_THRDS, _WVPRGRP, _ACHUNK)                \
    do {                                                        \
      if (ytile == 1) SWEEP_UNRL(_THRDS, 1, _WVPRGRP, _ACHUNK); \
      if (ytile == 2) SWEEP_UNRL(_THRDS, 2, _WVPRGRP, _ACHUNK); \
      if (ytile == 4) SWEEP_UNRL(_THRDS, 4, _WVPRGRP, _ACHUNK); \
    } while (0)

  #define SWEEP_WV(_THRDS, _ACHUNK)                        \
    do {                                                   \
      if (wvprgrp == 8) SWEEP_YTILE(_THRDS, 8, _ACHUNK);   \
      if (wvprgrp == 12) SWEEP_YTILE(_THRDS, 12, _ACHUNK); \
      if (wvprgrp == 16) SWEEP_YTILE(_THRDS, 16, _ACHUNK); \
    } while (0)

  #define SWEEP_AC(_THRDS)                    \
    do {                                      \
      if (achunk == 8) SWEEP_WV(_THRDS, 8);   \
      if (achunk == 16) SWEEP_WV(_THRDS, 16); \
      if (achunk == 32) SWEEP_WV(_THRDS, 32); \
    } while (0)

  if (thrds == 32) {
    SWEEP_AC(32);
  } else {
    SWEEP_AC(64);
  }
  TORCH_CHECK(false, "wvSplitK_int8_sweep: unhandled (thrds=", thrds,
              ", ytile=", ytile, ", wvprgrp=", wvprgrp, ", achunk=", achunk,
              ", unrl=", unrl, ")");

  #undef SWEEP_LAUNCH
  #undef SWEEP_UNRL
  #undef SWEEP_YTILE
  #undef SWEEP_WV
  #undef SWEEP_AC
}

#endif  // VLLM_SKINNY_GEMM_SWEEP
