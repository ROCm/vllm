#ifdef VLLM_SKINNY_GEMM_SWEEP
  #include "skinny_gemms_int4_sweep_skinny.cuh"

template torch::Tensor wvSplitK_int4g_sweep_skinny_impl<half, false, 1>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t);
extern template torch::Tensor wvSplitK_int4g_sweep_skinny_impl<half, true, 1>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t);
SWEEP_SK_DECLARE_ONE(extern, __hip_bfloat16, true, 1)
extern template torch::Tensor
wvSplitK_int4g_sweep_skinny_impl<__hip_bfloat16, false, 1>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t);
SWEEP_SK_DECLARE(extern, 2)
SWEEP_SK_DECLARE(extern, 3)
SWEEP_SK_DECLARE(extern, 4)
SWEEP_SK_DECLARE(extern, 5)

template <int N>
static torch::Tensor sweep_skinny_n(
    const at::Tensor& in_w, const at::Tensor& in_x, const at::Tensor& in_scale,
    const std::optional<at::Tensor>& in_zero_points, const int64_t CuCount,
    const int64_t group_size, const int64_t ytile, const int64_t unrl,
    const int64_t achunk, const int64_t wvprgrp) {
  if (in_x.dtype() == torch::kFloat16) {
    if (in_zero_points.has_value())
      return wvSplitK_int4g_sweep_skinny_impl<half, true, N>(
          in_w, in_x, in_scale, in_zero_points, CuCount, group_size, ytile,
          unrl, achunk, wvprgrp);
    return wvSplitK_int4g_sweep_skinny_impl<half, false, N>(
        in_w, in_x, in_scale, in_zero_points, CuCount, group_size, ytile, unrl,
        achunk, wvprgrp);
  }
  TORCH_CHECK(in_x.dtype() == torch::kBFloat16,
              "sweep supports f16 and bf16 activations");
  if (in_zero_points.has_value())
    return wvSplitK_int4g_sweep_skinny_impl<__hip_bfloat16, true, N>(
        in_w, in_x, in_scale, in_zero_points, CuCount, group_size, ytile, unrl,
        achunk, wvprgrp);
  return wvSplitK_int4g_sweep_skinny_impl<__hip_bfloat16, false, N>(
      in_w, in_x, in_scale, in_zero_points, CuCount, group_size, ytile, unrl,
      achunk, wvprgrp);
}

torch::Tensor wvSplitK_int4g_sweep_skinny(
    const at::Tensor& in_w, const at::Tensor& in_x, const at::Tensor& in_scale,
    const std::optional<at::Tensor>& in_zero_points, const int64_t CuCount,
    const int64_t group_size, const int64_t ytile, const int64_t unrl,
    const int64_t achunk, const int64_t wvprgrp) {
  TORCH_CHECK(in_scale.dtype() == in_x.dtype(),
              "Scale dtype must match activation dtype");
  switch (in_x.size(0)) {
    case 1:
      return sweep_skinny_n<1>(in_w, in_x, in_scale, in_zero_points, CuCount,
                               group_size, ytile, unrl, achunk, wvprgrp);
    case 2:
      return sweep_skinny_n<2>(in_w, in_x, in_scale, in_zero_points, CuCount,
                               group_size, ytile, unrl, achunk, wvprgrp);
    case 3:
      return sweep_skinny_n<3>(in_w, in_x, in_scale, in_zero_points, CuCount,
                               group_size, ytile, unrl, achunk, wvprgrp);
    case 4:
      return sweep_skinny_n<4>(in_w, in_x, in_scale, in_zero_points, CuCount,
                               group_size, ytile, unrl, achunk, wvprgrp);
    case 5:
      return sweep_skinny_n<5>(in_w, in_x, in_scale, in_zero_points, CuCount,
                               group_size, ytile, unrl, achunk, wvprgrp);
    default:
      TORCH_CHECK(false, "sweep supports N=1..5, got ", in_x.size(0));
  }
}

#endif  // VLLM_SKINNY_GEMM_SWEEP
