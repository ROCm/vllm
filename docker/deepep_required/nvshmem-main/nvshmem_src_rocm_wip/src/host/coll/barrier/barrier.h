/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_BARRIER_CPU_H
#define NVSHMEMI_BARRIER_CPU_H
//#include <hip/driver_types.h>
#include <hip/hip_runtime_api.h>
#include "device_host/nvshmem_types.h"

int nvshmemi_call_barrier_on_stream_kernel(nvshmem_team_t team, hipStream_t stream);
int nvshmemi_call_sync_on_stream_kernel(nvshmem_team_t team, hipStream_t stream);
void nvshmemxi_barrier_all_on_stream(hipStream_t);
void nvshmemxi_barrier_on_stream(nvshmem_team_t team, hipStream_t stream);
void nvshmemxi_sync_all_on_stream(hipStream_t);

#endif /* NVSHMEMI_BARRIER_CPU_H */
