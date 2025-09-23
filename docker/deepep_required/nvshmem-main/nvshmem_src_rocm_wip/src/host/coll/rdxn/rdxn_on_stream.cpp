/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#define NVSHMEMI_HOST_ONLY

#include <stddef.h>                          // for size_t
#include "device_host/nvshmem_common.hip.h"    // for RDXN_OPS_max, RDXN_OPS...
#include "device_host/nvshmem_types.h"       // for nvshmem_team_t
#include "host/nvshmemx_coll_api.h"          // for nvshmemx_char_max_redu...
#include "internal/host/nvshmem_internal.h"  // for NVSHMEMI_CHECK_INIT_ST...
#if !__HIP__
#include "internal/host/nvshmem_nvtx.hpp"    // for nvtx_cond_range, NVTX_...
#endif
#include "internal/host/util.h"              // for NVSHMEM_API_NOT_SUPPOR...
#include "rdxn.h"                            // for nvshmemi_reduce_on_stream

#define DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM(TYPENAME, TYPE, OP)                     \
    NVSHMEMI_HOSTDEVICE_PREFIX int nvshmemx_##TYPENAME##_##OP##_reduce_on_stream(nvshmem_team_t team, TYPE *dest,     \
                                                      const TYPE *source, size_t nreduce,  \
                                                      hipStream_t stream) {               \
        NVSHMEMI_CHECK_INIT_STATUS();                                                      \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                 \
        return nvshmemi_reduce_on_stream<TYPE, RDXN_OPS_##OP>(team, dest, source, nreduce, \
                                                              stream);                     \
    }

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, prod)
#undef DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM
