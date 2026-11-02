#ifdef VLLM_SKINNY_GEMM_SWEEP
  #include "skinny_gemms_int4_sweep_skinny.cuh"

template torch::Tensor wvSplitK_int4g_sweep_skinny_impl<half, true, 1>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t);

#endif  // VLLM_SKINNY_GEMM_SWEEP
