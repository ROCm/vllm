/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NVSHMEM_CUDAWRAP_H
#define NVSHMEM_CUDAWRAP_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

#if CUDART_VERSION < 12040
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED 128
#define CU_MEM_HANDLE_TYPE_FABRIC (hipMemAllocationHandleType)0x8
#define CU_CTX_SYNC_MEMOPS 0x80
#endif

#if CUDART_VERSION < 11020
#define hipMemHandleTypeNone (hipMemAllocationHandleType)0x0
#endif

#if CUDART_VERSION < 12010
typedef hipError_t(*PFN_hipCtxSetFlags_v12010)(int flags);
#endif

#if CUDART_VERSION < 11070
#define CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED 124
typedef enum CUmemRangeHandleType_enum {
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 0x1,
    CU_MEM_RANGE_HANDLE_TYPE_MAX = 0x7FFFFFFF
} CUmemRangeHandleType;
typedef hipError_t(*PFN_hipMemGetHandleForAddressRange_v11070)(void *handle, hipDeviceptr_t dptr,
                                                                    size_t size,
                                                                    CUmemRangeHandleType handleType,
                                                                    unsigned long long flags);
#endif

#if CUDART_VERSION < 12010
#define CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED 132
typedef enum CUmulticastGranularity_flags_enum {
    CU_MULTICAST_GRANULARITY_MINIMUM = 0x0,
    CU_MULTICAST_GRANULARITY_RECOMMENDED = 0x1
} CUmulticastGranularity_flags;
typedef struct CUmulticastObjectProp_st {
    unsigned int numDevices;
    size_t size;
    unsigned long long handleTypes;
    unsigned long long flags;
} CUmulticastObjectProp_v1;
typedef CUmulticastObjectProp_v1 CUmulticastObjectProp;
typedef hipError_t(*PFN_hipMulticastCreate_v12010)(hipMemGenericAllocationHandle_t *mcHandle,
                                                        const CUmulticastObjectProp *prop);
typedef hipError_t(*PFN_hipMulticastBindMem_v12010)(hipMemGenericAllocationHandle_t mcHandle,
                                                         size_t mcOffset,
                                                         hipMemGenericAllocationHandle_t memHandle,
                                                         size_t memOffset, size_t size,
                                                         unsigned long long flags);
typedef hipError_t(*PFN_hipMulticastAddDevice_v12010)(hipMemGenericAllocationHandle_t mcHandle,
                                                           hipDevice_t dev);
typedef hipError_t(*PFN_hipMulticastUnbind_v12010)(hipMemGenericAllocationHandle_t mcHandle,
                                                        hipDevice_t dev, size_t mcOffset, size_t size);
typedef hipError_t(*PFN_hipMulticastGetGranularity_v12010)(
    size_t *granularity, const CUmulticastObjectProp *prop, CUmulticastGranularity_flags option);
#endif

#if CUDART_VERSION >= 11030
#include <hipdaTypedefs.h>
#else
typedef enum CUflushGPUDirectRDMAWritesTarget_enum {
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX = 0
} CUflushGPUDirectRDMAWritesTarget;
typedef enum CUflushGPUDirectRDMAWritesScope_enum {
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER = 100,
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES = 200
} CUflushGPUDirectRDMAWritesScope;

#define CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS 117
#define CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING 118
#define hipFlushGPUDirectRDMAWritesOptionHost (1 << 0)
typedef hipError_t(*PFN_hipInit_v2000)(unsigned int Flags);
typedef hipError_t(*PFN_hipGetProcAddress_v11030)(const char *symbol, void **pfn,
                                                       int driverVersion, uint64_t flags);
typedef hipError_t(*PFN_hipDeviceGetAttribute_v2000)(int *pi, hipDeviceAttribute_t attrib,
                                                          hipDevice_t dev);
typedef hipError_t(*PFN_hipPointerSetAttribute_v6000)(const void *value,
                                                           hipPointer_attribute attribute,
                                                           hipDeviceptr_t ptr);
typedef hipError_t(*PFN_hipGetErrorString_v6000)(hipError_t error, const char **pStr);
typedef hipError_t(*PFN_hipGetErrorName_v6000)(hipError_t error, const char **pStr);
typedef hipError_t(*PFN_hipDeviceGet_v2000)(hipDevice_t *device, int ordinal);
typedef hipError_t(*PFN_hipCtxSetCurrent_v4000)(hipCtx_t ctx);
typedef hipError_t(*PFN_hipCtxGetDevice_v2000)(hipDevice_t *device);
typedef hipError_t(*PFN_hipCtxGetCurrent_v4000)(hipCtx_t *pctx);
typedef hipError_t(*PFN_hipCtxGetFlags_v7000)(unsigned int *flags);
typedef hipError_t(*PFN_hipCtxSetFlags_v12010)(int flags);
typedef hipError_t(*PFN_hipDevicePrimaryCtxRetain_v7000)(hipCtx_t *pctx, hipDevice_t dev);
typedef hipError_t(*PFN_hipCtxSynchronize_v2000)();
typedef hipError_t(*PFN_hipFlushGPUDirectRDMAWrites_v11030)(
    CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope);
typedef hipError_t(*PFN_hipModuleGetGlobal_v3020)(hipDeviceptr_t *dptr, size_t *bytes,
                                                       hipModule_t hmod, const char *name);
typedef hipError_t(*PFN_hipMemCreate_v10020)(hipMemGenericAllocationHandle_t *handle, size_t size,
                                                  const hipMemAllocationProp *prop,
                                                  unsigned long long flags);
typedef hipError_t(*PFN_hipMemGetAllocationGranularity_v10020)(
    size_t *granularity, const hipMemAllocationProp *prop, hipMemAllocationGranularity_flags option);
typedef hipError_t(*PFN_hipMemAddressReserve_v10020)(hipDeviceptr_t *ptr, size_t size,
                                                          size_t alignment, hipDeviceptr_t addr,
                                                          unsigned long long flags);
typedef hipError_t(*PFN_hipMemAddressFree_v10020)(hipDeviceptr_t ptr, size_t size);
typedef hipError_t(*PFN_hipMemExportToShareableHandle_v10020)(
    void *shareableHandle, hipMemGenericAllocationHandle_t handle,
    hipMemAllocationHandleType handleType, unsigned long long flags);
typedef hipError_t(*PFN_hipMemImportFromShareableHandle_v10020)(
    hipMemGenericAllocationHandle_t *handle, void *osHandle, hipMemAllocationHandleType shHandleType);
typedef hipError_t(*PFN_hipMemMap_v10020)(hipDeviceptr_t ptr, size_t size, size_t offset,
                                               hipMemGenericAllocationHandle_t handle,
                                               unsigned long long flags);
typedef hipError_t(*PFN_hipMemRelease_v10020)(hipMemGenericAllocationHandle_t handle);
typedef hipError_t(*PFN_hipMemSetAccess_v10020)(hipDeviceptr_t ptr, size_t size,
                                                     const hipMemAccessDesc *desc, size_t count);
typedef hipError_t(*PFN_hipMemUnmap_v10020)(hipDeviceptr_t ptr, size_t size);
typedef hipError_t(*PFN_hipMemGetAccess_v10020)(unsigned long long *flags,
                                                     const hipMemLocation *location,
                                                     hipDeviceptr_t ptr);
typedef hipError_t(*PFN_hipStreamWriteValue64_v11070)(hipStream_t stream, hipDeviceptr_t addr,
                                                           uint64_t value, unsigned int flags);
typedef hipError_t(*PFN_hipStreamWaitValue64_v11070)(hipStream_t stream, hipDeviceptr_t addr,
                                                          uint64_t value, unsigned int flags);
#endif

#define DEFINE_SYM(symbol, version) PFN_##symbol##_v##version pfn_##symbol;
struct nvshmemi_hipda_fn_table {
    DEFINE_SYM(hipCtxGetDevice, 2000)
    DEFINE_SYM(hipCtxSynchronize, 2000)
    DEFINE_SYM(hipDeviceGet, 2000)
    DEFINE_SYM(hipDeviceGetAttribute, 2000)
    DEFINE_SYM(hipPointerSetAttribute, 6000)
    DEFINE_SYM(hipModuleGetGlobal, 3020)
    DEFINE_SYM(hipGetErrorString, 6000)
    DEFINE_SYM(hipGetErrorName, 6000)
    DEFINE_SYM(hipCtxSetCurrent, 4000)
    DEFINE_SYM(hipDevicePrimaryCtxRetain, 7000)
    DEFINE_SYM(hipCtxGetCurrent, 4000)
    DEFINE_SYM(hipCtxGetFlags, 7000)
    DEFINE_SYM(hipCtxSetFlags, 12010)
    DEFINE_SYM(hipFlushGPUDirectRDMAWrites, 11030)     // DMA-BUF support
    DEFINE_SYM(hipMemGetHandleForAddressRange, 11070)  // DMA-BUF support
    DEFINE_SYM(hipMemCreate, 10020)
    DEFINE_SYM(hipMemAddressReserve, 10020)
    DEFINE_SYM(hipMemAddressFree, 10020)
    DEFINE_SYM(hipMemMap, 10020)
    DEFINE_SYM(hipMemGetAllocationGranularity, 10020)
    DEFINE_SYM(hipMemImportFromShareableHandle, 10020)
    DEFINE_SYM(hipMemExportToShareableHandle, 10020)
    DEFINE_SYM(hipMemRelease, 10020)
    DEFINE_SYM(hipMemSetAccess, 10020)
    DEFINE_SYM(hipMemUnmap, 10020)
    DEFINE_SYM(hipMulticastCreate, 12010)
    DEFINE_SYM(hipMulticastAddDevice, 12010)
    DEFINE_SYM(hipMulticastBindMem, 12010)
    DEFINE_SYM(hipMulticastUnbind, 12010)
    DEFINE_SYM(hipMulticastGetGranularity, 12010)
    DEFINE_SYM(hipStreamWriteValue64, 11070)
    DEFINE_SYM(hipStreamWaitValue64, 11070)

    /* CUDA Driver functions loaded with dlsym() */
    DEFINE_SYM(hipInit, 2000)
    DEFINE_SYM(hipGetProcAddress, 11030)
};
#undef DEFINE_SYM

#define CUPFN(table, symbol) table->pfn_##symbol

// Check CUDA PFN driver calls
#define CUCHECKNORETURN(table, cmd)                          \
    do {                                                     \
        hipError_t err = cmd;                     \
        if (err != hipSuccess) {                           \
            const char *errStr = hipGetErrorString(err);                              \
            fprintf(stderr, "Cuda failure '%s'", errStr);    \
        }                                                    \
        assert(err == hipSuccess);                         \
    } while (false)

#define CUASSERTAPIAVAILABLE(table, cmd) \
    do {                                 \
        if (cmd == NULL) {  \
            assert(false);               \
        }                                \
    } while (false)

// Check CUDA PFN driver calls
#define CUCHECK(table, cmd)                                                         \
    do {                                                                            \
        hipError_t err = cmd;                                            \
        if (err != hipSuccess) {                                                  \
            const char *errStr = hipGetErrorString(err);                              \
            fprintf(stderr, "%s:%d Cuda failure '%s'", __FILE__, __LINE__, errStr); \
            return NVSHMEMX_ERROR_INTERNAL;                                         \
        }                                                                           \
    } while (false)

#define CUCHECKGOTO(table, cmd, res, label)                                         \
    do {                                                                            \
        hipError_t err = cmd;                                            \
        if (err != hipSuccess) {                                                  \
            const char *errStr = hipGetErrorString(err);                              \
            fprintf(stderr, "%s:%d Cuda failure '%s'", __FILE__, __LINE__, errStr); \
            res = NVSHMEMX_ERROR_INTERNAL;                                          \
            goto label;                                                             \
        }                                                                           \
    } while (false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(table, cmd)                                                   \
    do {                                                                            \
        hipError_t err = cmd;                                            \
        if (err != hipSuccess) {                                                  \
            const char *errStr = hipGetErrorString(err);                              \
            fprintf(stderr, "%s:%d Cuda failure '%s'", __FILE__, __LINE__, errStr); \
        }                                                                           \
    } while (false)

#define CUCHECKTHREAD(table, cmd, args)                                             \
    do {                                                                            \
        hipError_t err = cmd;                                            \
        if (err != hipSuccess) {                                                  \
            fprintf(stderr, "%s:%d -> %d [Async thread]", __FILE__, __LINE__, err); \
            args->ret = NVSHMEMX_ERROR_INTERNAL;                                    \
            return args;                                                            \
        }                                                                           \
    } while (0)

#endif
