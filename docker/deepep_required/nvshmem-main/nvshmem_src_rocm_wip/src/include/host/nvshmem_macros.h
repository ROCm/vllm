
#ifndef _NVSHMEM_MACROS_H_
#define _NVSHMEM_MACROS_H_

#include <hip/hip_runtime.h>

#if __HIP_DEVICE_COMPILE__
#ifdef NVSHMEMI_HOST_ONLY
#define NVSHMEMI_HOSTDEVICE_PREFIX __host__
#else
#define NVSHMEMI_HOSTDEVICE_PREFIX __host__ __device__
#endif
#else
#ifndef NVSHMEMI_HOSTDEVICE_PREFIX
#define NVSHMEMI_HOSTDEVICE_PREFIX __host__
#endif
#endif

#if defined NVSHMEM_BITCODE_APPLICATION
#undef NVSHMEMI_HOSTDEVICE_PREFIX
#define NVSHMEMI_HOSTDEVICE_PREFIX __host__ __device__ __attribute__((always_inline))
#endif

#endif
