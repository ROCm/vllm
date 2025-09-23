/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _P2P_H
#define _P2P_H

#include <stdint.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-uintn.h>
#include <hip/hip_runtime.h>
#include "internal/host_transport/nvshmemi_transport_defines.h"

typedef struct {
    int ndev;
    hipDevice_t *cudev;
    int *devid;
    hipDeviceptr_t *curetval;
    hipDevice_t cudevice;
    int device_id;
    uint64_t hostHash;
    pcie_id_t *pcie_ids;
    char pcie_bdf[NVSHMEM_PCIE_BDF_BUFFER_LEN];
} transport_p2p_state_t;

#endif
