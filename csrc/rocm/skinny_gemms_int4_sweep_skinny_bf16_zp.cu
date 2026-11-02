#ifdef VLLM_SKINNY_GEMM_SWEEP
  #include "skinny_gemms_int4_sweep_skinny.cuh"

SWEEP_SK_DECLARE_ONE(, __hip_bfloat16, true, 1)

#endif  // VLLM_SKINNY_GEMM_SWEEP
