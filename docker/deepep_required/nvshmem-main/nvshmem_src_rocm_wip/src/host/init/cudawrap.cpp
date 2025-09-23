/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <hip/hip_runtime.h>  // for hipError_t
#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>
#else
// IWYU pragma: no_include <cudaTypedefs.h>
#endif

#include <hip/hip_runtime.h>                                // for hipDriverGetVersion
#include <dlfcn.h>                                       // for dlsym, dlopen, RTLD...
//#include <hip/driver_types.h>                                // for hipError_t
#include <stdio.h>                                       // for NULL, snprintf
#include "internal/host/debug.h"                         // for WARN, INFO, NVSHMEM...
#include "internal/host/error_codes_internal.h"          // for NVSHMEMI_SYSTEM_ERROR
#include "internal/host/util.h"                          // for nvshmemi_options
#include "internal/host_transport/cudawrap.h"            // for nvshmemi_cuda_fn_table
#include "bootstrap_host_transport/env_defs_internal.h"  // for nvshmemi_options_s

static enum {
    cudaUninitialized,
    cudaInitializing,
    cudaInitialized,
    cudaError
} cudaState = cudaUninitialized;

static void *cudaLib;
static int cudaDriverVersion;

static int cudaPfnFuncLoader(struct nvshmemi_cuda_fn_table *table) {
    hipError_t res;

#define LOAD_SYM(table, symbol, version, sym_suffix, ignore)                                       \
    do {                                                                                           \
        bool not_found = false;                                                                    \
        if (table->pfn_cuGetProcAddress) {                                                         \
            res =                                                                                  \
                table->pfn_cuGetProcAddress(#symbol, (void **)(&table->pfn_##symbol), version, 0); \
            if (res != 0) not_found = true;                                                        \
        } else {                                                                                   \
            table->pfn_##symbol = (PFN_##symbol##_v##version)dlsym(cudaLib, #symbol #sym_suffix);  \
            if (table->pfn_##symbol == NULL) not_found = true;                                     \
        }                                                                                          \
        if (not_found) {                                                                           \
            if (!ignore) {                                                                         \
                WARN("Retrieve %s version %d failed", #symbol #sym_suffix, cudaDriverVersion);     \
                return NVSHMEMI_SYSTEM_ERROR;                                                      \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#if 0
    LOAD_SYM(table, hipCtxGetDevice, 2000, , 0);
    LOAD_SYM(table, hipCtxSynchronize, 2000, , 0);
    LOAD_SYM(table, hipDeviceGet, 2000, , 0);
    LOAD_SYM(table, hipDeviceGetAttribute, 2000, , 0);
    LOAD_SYM(table, hipPointerSetAttribute, 6000, , 0);
    LOAD_SYM(table, hipModuleGetGlobal, 3020, _v2, 0);
    LOAD_SYM(table, hipDrvGetErrorString, 6000, , 0);
    LOAD_SYM(table, hipDrvGetErrorName, 6000, , 0);
    LOAD_SYM(table, hipCtxSetCurrent, 4000, , 0);
    LOAD_SYM(table, hipDevicePrimaryCtxRetain, 7000, , 0);
    LOAD_SYM(table, hipCtxGetCurrent, 4000, , 0);
    LOAD_SYM(table, hipCtxGetFlags, 7000, , 0);
#endif
    //LOAD_SYM(table, cuCtxSetFlags, 12010, , 1);
    //LOAD_SYM(table, cuFlushGPUDirectRDMAWrites, 11030, , 1);
    //LOAD_SYM(table, cuMemGetHandleForAddressRange, 11070, , 1);  // DMA-BUF support
#if 0 
    LOAD_SYM(table, hipMemCreate, 10020, , 1);
    LOAD_SYM(table, hipMemMap, 10020, , 1);
    LOAD_SYM(table, hipMemAddressReserve, 10020, , 1);
    LOAD_SYM(table, hipMemAddressFree, 10020, , 1);
    LOAD_SYM(table, hipMemGetAllocationGranularity, 10020, , 1);
    LOAD_SYM(table, hipMemImportFromShareableHandle, 10020, , 1);
    LOAD_SYM(table, hipMemExportToShareableHandle, 10020, , 1);
    LOAD_SYM(table, hipMemRelease, 10020, , 1);
    LOAD_SYM(table, hipMemSetAccess, 10020, , 1);
    LOAD_SYM(table, hipMemUnmap, 10020, , 1);
#endif
    //LOAD_SYM(table, cuMulticastCreate, 12010, , 1);
    //LOAD_SYM(table, cuMulticastAddDevice, 12010, , 1);
    //LOAD_SYM(table, cuMulticastBindMem, 12010, , 1);
    //LOAD_SYM(table, cuMulticastUnbind, 12010, , 1);
    //LOAD_SYM(table, cuMulticastGetGranularity, 12010, , 1);
#if 0
    LOAD_SYM(table, hipStreamWriteValue64, 11070, _v2, 1);
    LOAD_SYM(table, hipStreamWaitValue64, 11070, _v2, 1);
#endif
    return NVSHMEMI_SUCCESS;
}

int nvshmemi_cuda_library_init(struct nvshmemi_cuda_fn_table *table) {
    hipError_t cuda_err;

    if (cudaState == cudaInitialized) return NVSHMEMI_SUCCESS;
    if (cudaState == cudaError) return NVSHMEMI_SYSTEM_ERROR;

    /*
     * Load CUDA driver library
     */
    char path[1024];
    if (!nvshmemi_options.CUDA_PATH_provided)
        snprintf(path, 1024, "%s", "libcuda.so.1");
    else
        snprintf(path, 1024, "%s/%s", nvshmemi_options.CUDA_PATH, "libcuda.so.1");

    cudaLib = dlopen(path, RTLD_LAZY);
    if (cudaLib == NULL) {
        WARN("Failed to find CUDA library in %s (NVSHMEM_CUDA_PATH=%s)", path,
             nvshmemi_options.CUDA_PATH);
        goto error;
    }

    /*
     * Load initial CUDA functions
     */

#if 0
    table->pfn_hipInit = (PFN_hipInit_v2000)dlsym(cudaLib, "hipInit");
    if (table->pfn_hipInit == NULL) {
        WARN("Failed to load CUDA missing symbol hipInit");
        goto error;
    }

    cuda_err = hipDriverGetVersion(&cudaDriverVersion);
    if (cuda_err != 0) {
        WARN("hipDriverGetVersion failed with %d", cuda_err);
        goto error;
    }
    INFO(NVSHMEM_INIT, "cudaDriverVersion %d", cudaDriverVersion);

    table->pfn_hipGetProcAddress = (PFN_hipGetProcAddress_v11030)dlsym(cudaLib, "hipGetProcAddress");

    /*
     * Required to initialize the CUDA Driver.
     * Multiple calls of hipInit() will return immediately
     * without making any relevant change
     */
    table->pfn_hipInit(0);

    if (cudaPfnFuncLoader(table)) {
        WARN("CUDA some PFN functions not found in the library");
        goto error;
    }
#endif

    cudaState = cudaInitialized;
    return NVSHMEMI_SUCCESS;

error:
    cudaState = cudaError;
    return NVSHMEMI_SYSTEM_ERROR;
}
