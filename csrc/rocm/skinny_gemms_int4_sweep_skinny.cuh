#pragma once

// Skinny (N=1..5) sweep for wvSplitK_int4_hf_sml_ covering what
// wvSplitK_int4g_sweep cannot: bf16 activations, asymmetric (zero-point)
// checkpoints in both dtypes, N=5, UNRL=8 and WvPrGrp=32.  Wave32 (gfx11/gfx12)
// only. N=1 sweeps UNRL {1,2,4,8} x A_CHUNK {8,16,32}; N>=2, whose activation
// rows multiply the register footprint, sweeps UNRL {1,2,4} x A_CHUNK {16,32}.
// Instantiations are split across TUs so they compile in parallel; see
// skinny_gemms_int4_sweep_skinny_*.cu.
#include "skinny_gemms_int4_kernels.cuh"

template <typename fptype, bool HAS_ZP, int N>
torch::Tensor wvSplitK_int4g_sweep_skinny_impl(
    const at::Tensor& in_w, const at::Tensor& in_x, const at::Tensor& in_scale,
    const std::optional<at::Tensor>& in_zero_points, const int64_t CuCount,
    const int64_t group_size, const int64_t ytile, const int64_t unrl,
    const int64_t achunk, const int64_t wvprgrp) {
  const int M_in = in_w.size(0);
  const int K_in = in_x.size(1);
  TORCH_CHECK(in_x.size(0) == N, "expected N=", N, ", got ", in_x.size(0));
  TORCH_CHECK((on_gfx_int4<11, 12>()), "sweep is wave32 (gfx11/gfx12) only");
  TORCH_CHECK(group_size == 32 || group_size == 128,
              "group_size must be 32 or 128, got ", group_size);
  TORCH_CHECK(K_in % group_size == 0, "K must be divisible by group_size");
  TORCH_CHECK(K_in % achunk == 0, "K must be divisible by achunk=", achunk);
  TORCH_CHECK(M_in % ytile == 0, "M must be divisible by ytile=", ytile);
  TORCH_CHECK(in_x.is_contiguous(), "Activation must be contiguous");
  TORCH_CHECK(K_in * N <= static_cast<int>(LDS_SIZE / in_x.element_size()),
              "K*N exceeds LDS capacity");
  TORCH_CHECK(in_scale.stride(1) == 1, "Scale rows must be contiguous");
  if constexpr (HAS_ZP) {
    TORCH_CHECK(in_zero_points.has_value(), "zero points required");
    TORCH_CHECK(in_zero_points->stride(0) == in_scale.stride(0),
                "Zero points must share the scale row stride");
  } else {
    TORCH_CHECK(!in_zero_points.has_value(), "unexpected zero points");
  }

  const int b_row_stride_bytes = in_w.stride(0) * in_w.element_size();
  const int group_stride = in_scale.stride(0);
  auto out_c = torch::empty(
      {N, M_in},
      torch::TensorOptions().dtype(in_x.dtype()).device(in_x.device()));

  dim3 grid(CuCount);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_w));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const uint8_t* wptr = reinterpret_cast<const uint8_t*>(in_w.data_ptr());
  const fptype* aptr = reinterpret_cast<const fptype*>(in_x.data_ptr());
  const fptype* sptr = reinterpret_cast<const fptype*>(in_scale.data_ptr());
  const uint32_t* zpptr =
      HAS_ZP ? reinterpret_cast<const uint32_t*>(in_zero_points->data_ptr())
             : nullptr;
  fptype* cptr = reinterpret_cast<fptype*>(out_c.data_ptr());

#define SWEEP_SK_LAUNCH(_YT, _W, _AC, _UN, _GS)                          \
  {                                                                      \
    dim3 block(32, _W);                                                  \
    int wv = mindiv_int4(M_in, CuCount * _YT, _W);                       \
    wvSplitK_int4_hf_sml_<fptype, 32, _YT, _W, _AC, _UN, N, _GS, HAS_ZP> \
        <<<grid, block, 0, stream>>>(K_in, M_in, 1, 1, wptr, aptr, sptr, \
                                     zpptr, nullptr, cptr, wv, CuCount,  \
                                     b_row_stride_bytes, group_stride);  \
  }
#define SWEEP_SK_UN(_YT, _W, _AC, _GS)               \
  switch (unrl) {                                    \
    case 1:                                          \
      SWEEP_SK_LAUNCH(_YT, _W, _AC, 1, _GS) break;   \
    case 2:                                          \
      SWEEP_SK_LAUNCH(_YT, _W, _AC, 2, _GS) break;   \
    case 4:                                          \
      SWEEP_SK_LAUNCH(_YT, _W, _AC, 4, _GS) break;   \
    case 8:                                          \
      if constexpr (N == 1) {                        \
        SWEEP_SK_LAUNCH(_YT, _W, _AC, 8, _GS) break; \
      }                                              \
      [[fallthrough]];                               \
    default:                                         \
      TORCH_CHECK(false, "Unsupported unrl=", unrl); \
  }
#define SWEEP_SK_YT(_W, _AC, _GS)                      \
  switch (ytile) {                                     \
    case 1:                                            \
      SWEEP_SK_UN(1, _W, _AC, _GS) break;              \
    case 2:                                            \
      SWEEP_SK_UN(2, _W, _AC, _GS) break;              \
    case 4:                                            \
      SWEEP_SK_UN(4, _W, _AC, _GS) break;              \
    default:                                           \
      TORCH_CHECK(false, "Unsupported ytile=", ytile); \
  }
#define SWEEP_SK_W(_AC, _GS)                               \
  switch (wvprgrp) {                                       \
    case 8:                                                \
      SWEEP_SK_YT(8, _AC, _GS) break;                      \
    case 12:                                               \
      SWEEP_SK_YT(12, _AC, _GS) break;                     \
    case 16:                                               \
      SWEEP_SK_YT(16, _AC, _GS) break;                     \
    case 32:                                               \
      SWEEP_SK_YT(32, _AC, _GS) break;                     \
    default:                                               \
      TORCH_CHECK(false, "Unsupported wvprgrp=", wvprgrp); \
  }
#define SWEEP_SK_AC(_GS)                                 \
  switch (achunk) {                                      \
    case 8:                                              \
      if constexpr (N == 1) {                            \
        SWEEP_SK_W(8, _GS) break;                        \
      }                                                  \
      [[fallthrough]];                                   \
    default:                                             \
      TORCH_CHECK(false, "Unsupported achunk=", achunk); \
      break;                                             \
    case 16:                                             \
      SWEEP_SK_W(16, _GS) break;                         \
    case 32:                                             \
      SWEEP_SK_W(32, _GS) break;                         \
  }

  if (group_size == 32) {
    SWEEP_SK_AC(32)
  } else {
    SWEEP_SK_AC(128)
  }

#undef SWEEP_SK_LAUNCH
#undef SWEEP_SK_UN
#undef SWEEP_SK_YT
#undef SWEEP_SK_W
#undef SWEEP_SK_AC

  return out_c;
}

// Explicitly instantiated by one TU per N, declared everywhere else.
#define SWEEP_SK_DECLARE_ONE(_EXTERN, _T, _ZP, _N)                          \
  _EXTERN template torch::Tensor                                            \
  wvSplitK_int4g_sweep_skinny_impl<_T, _ZP, _N>(                            \
      const at::Tensor&, const at::Tensor&, const at::Tensor&,              \
      const std::optional<at::Tensor>&, int64_t, int64_t, int64_t, int64_t, \
      int64_t, int64_t);
#define SWEEP_SK_DECLARE(_EXTERN, _N)                      \
  SWEEP_SK_DECLARE_ONE(_EXTERN, half, false, _N)           \
  SWEEP_SK_DECLARE_ONE(_EXTERN, half, true, _N)            \
  SWEEP_SK_DECLARE_ONE(_EXTERN, __hip_bfloat16, false, _N) \
  SWEEP_SK_DECLARE_ONE(_EXTERN, __hip_bfloat16, true, _N)
