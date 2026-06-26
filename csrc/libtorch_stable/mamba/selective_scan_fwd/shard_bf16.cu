// Explicit instantiations for bf16 input type.

#include "kernel.cuh"

template void selective_scan_fwd_cuda<torch::headeronly::BFloat16, float,
                                      torch::headeronly::BFloat16>(
    SSMParamsBase& params, cudaStream_t stream);
template void selective_scan_fwd_cuda<torch::headeronly::BFloat16, float,
                                      float>(SSMParamsBase& params,
                                             cudaStream_t stream);
