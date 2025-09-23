/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "reducescatter_common.hip.h"
#include "internal/non_abi/nvshmemi_h_to_d_coll_defs.hip.h"

REPT_FOR_BITWISE_TYPES(INSTANTIATE_NVSHMEMI_CALL_REDUCESCATTER_ON_STREAM_KERNEL, OR)
