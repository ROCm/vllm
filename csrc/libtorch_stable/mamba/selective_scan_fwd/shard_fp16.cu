// Explicit instantiations for fp16 and fp32 input types.

#include "kernel.cuh"

template void selective_scan_fwd_cuda<torch::headeronly::Half, float,
                                      torch::headeronly::Half>(
    SSMParamsBase& params, cudaStream_t stream);
template void selective_scan_fwd_cuda<torch::headeronly::Half, float, float>(
    SSMParamsBase& params, cudaStream_t stream);
template void selective_scan_fwd_cuda<float, float, float>(
    SSMParamsBase& params, cudaStream_t stream);
