#ifdef VLLM_SKINNY_GEMM_SWEEP
  #include "skinny_gemms_int4_sweep_n1.cuh"

template torch::Tensor wvSplitK_int4g_sweep_n1_impl<__hip_bfloat16, false>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t);

#endif  // VLLM_SKINNY_GEMM_SWEEP
