/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "internal/host/util.h"
#include "internal/non_abi/nvshmemi_h_to_d_sync_defs.hip.h"

static int nvshmemi_quiet_maxblocksize = -1;

void nvshmemi_call_proxy_quiet_entrypoint(hipStream_t cstrm) {
    if (nvshmemi_quiet_maxblocksize == -1) {
        int tmp;
        CUDA_RUNTIME_CHECK(hipOccupancyMaxPotentialBlockSize(
            &tmp, (int *)&nvshmemi_quiet_maxblocksize, nvshmemi_proxy_quiet_entrypoint));
    }
    int status = hipLaunchKernel((const void *)nvshmemi_proxy_quiet_entrypoint, 1,
                                  nvshmemi_quiet_maxblocksize, NULL, 0, cstrm);
    if (status) {
        NVSHMEMI_ERROR_PRINT("hipLaunchKernel() failed in nvshmem_quiet_on_stream \n");
    }
}
