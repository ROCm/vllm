/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#define NVSHMEMI_HOST_ONLY

#include "barrier.h"
#include <hip/hip_runtime.h>                    // for hipStreamSynchronize
#include "device_host/nvshmem_common.hip.h"    // for NVSHMEM_TEAM_WORLD
#include "device_host/nvshmem_types.h"       // for nvshmem_team_t
#include "host/nvshmem_api.h"                // for nvshmem_quiet
#include "host/nvshmem_coll_api.h"           // for nvshmem_barrier, nvshm...
#include "internal/host/nvshmem_internal.h"  // for nvshmemi_state, nvshme...
#include "internal/host/nvshmemi_types.h"    // for nvshmemi_state
#if !__HIP__
#include "internal/host/nvshmem_nvtx.hpp"    // for nvtx_cond_range, COLL_OPT
#endif
#include "internal/host/util.h"              // for nvshmemi_check_state_a...

void nvshmemi_barrier(nvshmem_team_t team) {
    nvshmem_quiet();
    nvshmemi_call_barrier_on_stream_kernel(team, nvshmemi_state->my_stream);
    CUDA_RUNTIME_CHECK(hipStreamSynchronize(nvshmemi_state->my_stream));
}

void nvshmemi_barrier_all() { nvshmemi_barrier(NVSHMEM_TEAM_WORLD); }

NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_barrier(nvshmem_team_t team) {
#if !__HIP__
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
#endif
    NVSHMEMI_CHECK_INIT_STATUS();
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();

    nvshmemi_barrier(team);

    return 0;
}

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_barrier_all() {
#if !__HIP__
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
#endif
    nvshmemi_check_state_and_init();
    nvshmemi_barrier_all();
    return;
}

void nvshmemi_sync(nvshmem_team_t team) {
    nvshmemi_call_sync_on_stream_kernel(team, nvshmemi_state->my_stream);
    CUDA_RUNTIME_CHECK(hipStreamSynchronize(nvshmemi_state->my_stream));
}

NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_team_sync(nvshmem_team_t team) {
#if !__HIP__
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
#endif
    NVSHMEMI_CHECK_INIT_STATUS();
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();

    nvshmemi_sync(team);

    return 0;
}

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_sync_all() {
#if !__HIP__
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
#endif
    nvshmemi_check_state_and_init();

    nvshmemxi_sync_all_on_stream(nvshmemi_state->my_stream);
    CUDA_RUNTIME_CHECK(hipStreamSynchronize(nvshmemi_state->my_stream));
}
