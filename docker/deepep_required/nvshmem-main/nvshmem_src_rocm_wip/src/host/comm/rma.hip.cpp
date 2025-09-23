/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <hip/hip_runtime.h>
#include "internal/host/nvshmem_internal.h"
#include "internal/non_abi/nvshmemi_h_to_d_rma_defs.hip.h"

int nvshmemi_proxy_rma_launcher(void *args[], hipStream_t cstrm, bool is_nbi, bool is_signal) {
    if (is_signal && is_nbi) {
        return hipLaunchKernel((const void *)nvshmemi_proxy_rma_signal_entrypoint, 1, 1, args, 0,
                                cstrm);
    } else if (is_nbi) {
        return hipLaunchKernel((const void *)nvshmemi_proxy_rma_entrypoint, 1, 1, args, 0, cstrm);
    } else if (is_signal) {
        return hipLaunchKernel((const void *)nvshmemi_proxy_rma_signal_entrypoint_blocking, 1, 1,
                                args, 0, cstrm);
    } else {
        return hipLaunchKernel((const void *)nvshmemi_proxy_rma_entrypoint_blocking, 1, 1, args, 0,
                                cstrm);
    }
}
