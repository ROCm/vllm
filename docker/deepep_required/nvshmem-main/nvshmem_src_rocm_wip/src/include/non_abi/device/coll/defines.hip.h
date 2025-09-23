/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_DEVICE_COLL_DEFINES_H_
#define _NVSHMEMI_DEVICE_COLL_DEFINES_H_

#include "alltoall.hip.h"
#include "barrier.hip.h"
#include "broadcast.hip.h"
#if !__HIP__
#include "fcollect.hip.h"
#endif
#include "reduce.hip.h"
#if !__HIP__
#include "reducescatter.hip.h"
#endif

#endif /* NVSHMEMI_DEVICE_COLL_DEFINES_H */
