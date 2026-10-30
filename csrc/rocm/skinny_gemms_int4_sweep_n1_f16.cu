#ifdef VLLM_SKINNY_GEMM_SWEEP
  #include "skinny_gemms_int4_sweep_n1.cuh"

extern template torch::Tensor wvSplitK_int4g_sweep_n1_impl<half, true>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t);
extern template torch::Tensor
wvSplitK_int4g_sweep_n1_impl<__hip_bfloat16, false>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t);

torch::Tensor wvSplitK_int4g_sweep_n1(
    const at::Tensor& in_w, const at::Tensor& in_x, const at::Tensor& in_scale,
    const std::optional<at::Tensor>& in_zero_points, const int64_t CuCount,
    const int64_t group_size, const int64_t ytile, const int64_t unrl,
    const int64_t achunk, const int64_t wvprgrp) {
  TORCH_CHECK(in_scale.dtype() == in_x.dtype(),
              "Scale dtype must match activation dtype");
  if (in_x.dtype() == torch::kFloat16) {
    if (in_zero_points.has_value())
      return wvSplitK_int4g_sweep_n1_impl<half, true>(
          in_w, in_x, in_scale, in_zero_points, CuCount, group_size, ytile,
          unrl, achunk, wvprgrp);
    return wvSplitK_int4g_sweep_n1_impl<half, false>(
        in_w, in_x, in_scale, in_zero_points, CuCount, group_size, ytile, unrl,
        achunk, wvprgrp);
  }
  TORCH_CHECK(in_x.dtype() == torch::kBFloat16 && !in_zero_points.has_value(),
              "sweep_n1 supports f16 (sym/asym) and bf16 (sym)");
  return wvSplitK_int4g_sweep_n1_impl<__hip_bfloat16, false>(
      in_w, in_x, in_scale, in_zero_points, CuCount, group_size, ytile, unrl,
      achunk, wvprgrp);
}

#endif  // VLLM_SKINNY_GEMM_SWEEP
