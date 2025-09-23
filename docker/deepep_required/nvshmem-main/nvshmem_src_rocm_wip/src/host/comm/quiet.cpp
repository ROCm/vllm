/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#define NVSHMEMI_HOST_ONLY

#include "host/nvshmem_api.h"   // IWYU pragma: keep
#include "host/nvshmemx_api.h"  // IWYU pragma: keep
#include <hip/hip_runtime.h>

#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_types.h"
#if !__HIP__
#include "internal/host/nvshmem_nvtx.hpp"
#endif
#include "non_abi/nvshmemx_error.h"
#include "internal/host_transport/transport.h"
#include "internal/host/util.h"

void nvshmemi_call_proxy_quiet_entrypoint(hipStream_t cstrm);
#ifdef __cplusplus
extern "C" {
#endif
void nvshmemx_quiet_on_stream(hipStream_t cstrm);
#ifdef __cplusplus
}
#endif

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_quiet(void) {
#if !__HIP__
    NVTX_FUNC_RANGE_IN_GROUP(MEMORDER);
#endif
    NVSHMEMI_CHECK_INIT_STATUS();

    int status = 0;

    int tbitmap = nvshmemi_state->transport_bitmap;
    if (nvshmemi_state->used_internal_streams) {
        for (int s = 0; s < MAX_PEER_STREAMS; s++) {
            if (nvshmemi_state->active_internal_streams[s]) {
                hipStream_t custrm = nvshmemi_state->custreams[s];
                CUDA_RUNTIME_CHECK_GOTO(hipStreamSynchronize(custrm), status, out);
                nvshmemi_state->active_internal_streams[s] = 0;
            }
        }
        nvshmemi_state->used_internal_streams = 0;
    }

    for (int j = 0; j < nvshmemi_state->num_initialized_transports; j++) {
        if (tbitmap & 1) {
            struct nvshmem_transport *tcurr =
                ((nvshmem_transport_t *)nvshmemi_state->transports)[j];
            for (int k = 0; k < nvshmemi_state->npes; k++) {
                if (tcurr->host_ops.quiet) {
                    status = tcurr->host_ops.quiet(tcurr, k, 0);
                }
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "nvshmem_quiet() failed \n");
            }
        }
        tbitmap >>= 1;
    }
out:
    return;
}

void nvshmemi_quiesce_internal_streams(hipStream_t cstrm) {
    if (nvshmemi_state->used_internal_streams) {
        for (int s = 0; s < MAX_PEER_STREAMS; s++) {
            hipStream_t custrm = nvshmemi_state->custreams[s];
            hipEvent_t cuev = nvshmemi_state->cuevents[s];

            if (nvshmemi_state->active_internal_streams[s]) {
                CUDA_RUNTIME_CHECK(hipEventRecord(cuev, custrm));
                CUDA_RUNTIME_CHECK(hipStreamWaitEvent(cstrm, cuev, 0));
                nvshmemi_state->active_internal_streams[s] = 0;
            }
        }
        nvshmemi_state->used_internal_streams = 0;
    }
}

void nvshmemx_quiet_on_stream(hipStream_t cstrm) {
#if !__HIP__
    NVTX_FUNC_RANGE_IN_GROUP(QUIET_ON_STREAM);
#endif
    NVSHMEMI_CHECK_INIT_STATUS();

    int in_cuda_graph = 0;
    hipStreamCaptureStatus status;
    CUDA_RUNTIME_CHECK(hipStreamIsCapturing(cstrm, &status));
    if (status == hipStreamCaptureStatusActive) in_cuda_graph = 1;

    nvshmemi_quiesce_internal_streams(cstrm);

    /* If our stream is in a graph, we need to perform the threadfence_system inside quiet */
    if (in_cuda_graph) {
        nvshmemi_call_proxy_quiet_entrypoint(cstrm);
        /* Otherwise, we only need to launch the quiet if we have a remote transport. */
    } else {
        int tbitmap = nvshmemi_state->transport_bitmap;

        for (int j = 0; j < nvshmemi_state->num_initialized_transports; j++) {
            if (tbitmap & 1) {
                struct nvshmem_transport *tcurr =
                    ((nvshmem_transport_t *)nvshmemi_state->transports)[j];
                if (tcurr->attr & NVSHMEM_TRANSPORT_ATTR_CONNECTED) {
                    nvshmemi_call_proxy_quiet_entrypoint(cstrm);
                }
            }
            tbitmap >>= 1;
        }
    }

    return;
}
