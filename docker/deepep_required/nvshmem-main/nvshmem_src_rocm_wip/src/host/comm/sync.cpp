/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#define NVSHMEMI_HOST_ONLY

#include <hip/hip_runtime.h>                  // for hipMemcpy, cud...
#include <stddef.h>                        // for size_t, NULL
#include <stdint.h>                        // for uint64_t
#include "device_host/nvshmem_common.hip.h"  // for NVSHMEMI_REPT_F...
#include "device_host_transport/nvshmem_common_transport.h"
#include "device_host_transport/nvshmem_constants.h"  // for NVSHMEM_SIGNAL_SET
#include "host/nvshmem_api.h"                         // for nvshmem_signal_...
#include "host/nvshmemx_api.h"                        // for nvshmemx_int32_...
#include "non_abi/nvshmemx_error.h"                   // for NVSHMEMI_NZ_EXIT
#include "internal/host/nvshmem_internal.h"           // for nvshmemi_signal...
#include "internal/host/cuda_interface_sync.h"        // for call_nvshmemi_i...
#if !__HIP__
#include "internal/host/nvshmem_nvtx.hpp"             // for nvtx_cond_range
#endif
#include "internal/host/nvshmemi_symmetric_heap.hpp"  // for nvshmemi_symmet...
#include "internal/host/nvshmemi_types.h"             // for nvshmemi_state
#include "internal/host/util.h"                       // for NVSHMEM_API_NOT...

#define NVSHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE)                                     \
    void nvshmemx_##type##_wait_until_on_stream(TYPE *ivar, int cmp, TYPE cmp_value,       \
                                                hipStream_t cstream) {                    \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                 \
        call_nvshmemi_##type##_wait_until_on_stream_kernel(ivar, cmp, cmp_value, cstream); \
    }
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_TYPE_WAIT_UNTIL_ON_STREAM)
#undef NVSHMEMX_TYPE_WAIT_UNTIL_ON_STREAM

#define NVSHMEMX_TYPE_WAIT_UNTIL_ALL_ON_STREAM(type, TYPE)                                         \
    void nvshmemx_##type##_wait_until_all_on_stream(TYPE *ivars, size_t nelems, const int *status, \
                                                    int cmp, TYPE cmp_value,                       \
                                                    hipStream_t cstream) {                        \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                         \
        call_nvshmemi_##type##_wait_until_all_on_stream_kernel(ivars, nelems, status, cmp,         \
                                                               cmp_value, cstream);                \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_TYPE_WAIT_UNTIL_ALL_ON_STREAM)
#undef NVSHMEMX_TYPE_WAIT_UNTIL_ALL_ON_STREAM

#define NVSHMEMX_TYPE_WAIT_UNTIL_ALL_VECTOR_ON_STREAM(type, TYPE)                                 \
    void nvshmemx_##type##_wait_until_all_vector_on_stream(                                       \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_value,                  \
        hipStream_t cstream) {                                                                   \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                        \
        call_nvshmemi_##type##_wait_until_all_vector_on_stream_kernel(ivars, nelems, status, cmp, \
                                                                      cmp_value, cstream);        \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_TYPE_WAIT_UNTIL_ALL_VECTOR_ON_STREAM)
#undef NVSHMEMX_TYPE_WAIT_UNTIL_ALL_VECTOR_ON_STREAM

void nvshmemx_signal_wait_until_on_stream(uint64_t *sig_addr, int cmp, uint64_t cmp_value,
                                          hipStream_t cstream) {
#if !__HIP__
    NVTX_FUNC_RANGE_IN_GROUP(WAIT_ON_STREAM);
#endif
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();
    int status = 0;
    if (((cmp == NVSHMEM_CMP_GE) || (cmp == NVSHMEM_CMP_EQ)) &&
        nvshmemi_can_use_cuda_64_bit_stream_memops &&
        (nvshmemi_can_flush_remote_writes || nvshmemi_options.BYPASS_FLUSH)) {
        if (cmp == NVSHMEM_CMP_GE)
            status = hipStreamWaitValue64(cstream, (hipDeviceptr_t)sig_addr, cmp_value, hipStreamWaitValueGte);
        else {  // cmp == NVSHMEM_CMP_EQ
            status = hipStreamWaitValue64(cstream, (hipDeviceptr_t)sig_addr, cmp_value, hipStreamWaitValueEq);
        }
        NVSHMEMI_NZ_EXIT(status, "hipStreamWaitValue64() failed\n");
    } else {
        call_nvshmemi_signal_wait_until_on_stream_kernel(sig_addr, cmp, cmp_value, cstream);
    }
}

void nvshmemi_signal_op_on_stream(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                  hipStream_t cstrm) {
    int status = 0;
    if (sig_op == NVSHMEMI_AMO_SIGNAL_SET &&
        nvshmemi_state->heap_obj->get_local_pe_base()[pe] != NULL) {
        void *peer_addr;
        NVSHMEMU_MAPPED_PTR_TRANSLATE(peer_addr, sig_addr, pe)
        if (nvshmemi_can_use_cuda_64_bit_stream_memops &&
            nvshmemi_job_connectivity == NVSHMEMI_JOB_GPU_LDST_ATOMICS) {
            status = hipStreamWriteValue64(cstrm, (hipDeviceptr_t)peer_addr, signal, 0);
            NVSHMEMI_NZ_EXIT(status, "hipStreamWriteValue64() failed\n");
        } else {
            status = hipMemcpyAsync(peer_addr, (const void *)&signal, sizeof(uint64_t),
                                     hipMemcpyHostToDevice, cstrm);
            NVSHMEMI_NZ_EXIT(status, "hipMemcpyAsync() failed\n");
        }
    } else {
        call_nvshmemi_signal_op_kernel(sig_addr, signal, sig_op, pe, cstrm);
    }
}

void nvshmemx_signal_op_on_stream(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                  hipStream_t cstrm) {
    nvshmemi_signal_op_on_stream(sig_addr, signal, sig_op, pe, cstrm);
}

NVSHMEMI_HOSTDEVICE_PREFIX uint64_t nvshmem_signal_fetch(uint64_t *sig_addr) {
    uint64_t signal;
    CUDA_RUNTIME_CHECK(hipMemcpy(&signal, sig_addr, sizeof(uint64_t), hipMemcpyDeviceToHost));
    return signal;
}
