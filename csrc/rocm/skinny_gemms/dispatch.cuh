#pragma once

#include "kernel.cuh"

// Per-N dispatch for wvSplitK (bf16/fp16 GEMM).
// Wraps WVSPLIT_TILE_CFG for a single N value.
template <typename scalar_t, int N_VAL>
inline void dispatch_wvsplitk(dim3 grid, cudaStream_t stream, int K_in,
                              int Kap_in, int Kbp_in, int M_in, int Bx_in,
                              int By_in, const scalar_t* af4,
                              const scalar_t* bf4, const scalar_t* biasf4,
                              scalar_t* c, int CuCount, int max_lds_len) {
  using fptype = scalar_t;
  int N_in = N_VAL;
  int sYT = (M_in + CuCount * 4 - 1) / (CuCount * 4);

  // clang-format off
  // These macros reference local variables above and kernel templates from
  // kernel.cuh.  They are copies of the macros in the original skinny_gemms.cu.

#define WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, _YTILE, _UNRL, _N, _AC)             \
  {                                                                           \
    dim3 block(_THRDS, _WVPRGRP);                                             \
    int __wvPrGrp = mindiv(M_in, CuCount * _YTILE, _WVPRGRP);                 \
    if ((Kbp_in * N_in <= max_lds_len) && (M_in % _YTILE == 0))               \
      wvSplitK_hf_sml_<fptype, _THRDS, _YTILE, _WVPRGRP, _AC, _UNRL, _N>      \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,     \
                                       By_in, af4, bf4, biasf4, c, __wvPrGrp, \
                                       CuCount);                              \
    else if (Kbp_in * N_in <= max_lds_len * 1.2)                              \
      wvSplitK_hf_<fptype, _THRDS, _YTILE, _WVPRGRP, _AC, _UNRL, _N>          \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,     \
                                       By_in, af4, bf4, biasf4, c, __wvPrGrp, \
                                       CuCount);                              \
    else                                                                      \
      wvSplitK_hf_big_<fptype, _THRDS, _YTILE, _WVPRGRP, _AC, _UNRL, _N>      \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,     \
                                       By_in, af4, bf4, biasf4, c, __wvPrGrp, \
                                       CuCount);                              \
  }

#define WVSPLITK_CFG(_THRDS, _WVPRGRP, _YTILE, _UNRL, _N) \
  WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, _YTILE, _UNRL, _N, 8)

#define WVSPLIT_TILE_CFG(_THRDS, _WVPRGRP, _sYT, __N)                        \
  {                                                                          \
    bool fit_lds = (Kbp_in * N_in <= max_lds_len);                           \
    if (is_gfx11()) {                                                        \
      if (_sYT <= 1)                                                         \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 1, 4, __N)                            \
      else if (K_in < 1024)                                                  \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 2, 4, __N)                            \
      else if ((K_in % 1024 == 512) && (_sYT >= 40 || K_in >= 4096))         \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 4, 1, __N)                            \
      else if ((K_in == 2048) && (__N == 1))                                 \
        WVSPLITK_CFG_AC(32, 32, 1, 8, __N, 16)                               \
      else if ((K_in == 2048) && (__N == 2 || __N == 3))                     \
        WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, 1, 2, __N, 32)                     \
      else if ((K_in == 4096) && (__N == 1))                                 \
        WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, 1, 4, __N, 32)                     \
      else if ((K_in == 4096) && (__N == 2) && (M_in < 4096))                \
        WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, 1, 2, __N, 32)                     \
      else if ((K_in == 4096) && (__N == 2) && (M_in >= 4096))               \
        WVSPLITK_CFG_AC(_THRDS, 32, 1, 2, __N, 32)                           \
      else if ((K_in == 4096) && (__N == 3))                                 \
        WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, 1, 2, __N, 32)                     \
      else if ((K_in == 8192) && (__N == 2))                                 \
        WVSPLITK_CFG_AC(_THRDS, 32, 1, 2, __N, 32)                           \
      else if ((K_in % 2048 == 0) && (__N == 2))                             \
        WVSPLITK_CFG_AC(_THRDS, 32, 2, 4, __N, 16)                           \
      else if ((K_in % 2048 == 0) && (__N != 2))                             \
        WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, 1, 4, __N, 16)                     \
      else if (K_in <= 2048)                                                 \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 1, 4, __N)                            \
      else if (__N >= 2 && !fit_lds) {                                       \
        if (K_in % 1024 == 0 && Kbp_in < max_lds_len / 2)                    \
          WVSPLITK_CFG(_THRDS, _WVPRGRP, 2, 4, __N)                          \
        else                                                                 \
          WVSPLITK_CFG(_THRDS, _WVPRGRP, 1, 4, __N)                          \
      } else if (__N == 1)                                                   \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 1, 2, __N)                            \
      else                                                                   \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 1, 1, __N)                            \
    } else {                                                                 \
      if (_sYT <= 1)                                                         \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 1, 4, __N)                            \
      else if ((__N == 1) || (!fit_lds) || (_sYT <= 4 * 2))                  \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 2, 2, __N)                            \
      else if (_sYT <= 4 * 3)                                                \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 3, 2, __N)                            \
      else if (__N == 4)                                                     \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 4, 1, __N)                            \
      else                                                                   \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 4, 2, __N)                            \
    }                                                                        \
  }

  if (on_gfx1x()) {
    WVSPLIT_TILE_CFG(32, 16, sYT, N_VAL)
  } else {
    WVSPLIT_TILE_CFG(64, 16, sYT, N_VAL)
  }

#undef WVSPLIT_TILE_CFG
#undef WVSPLITK_CFG_AC
#undef WVSPLITK_CFG
  // clang-format on
}
