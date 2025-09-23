/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEMX_H_
#define _NVSHMEMX_H_

/* NVRTC only compiles device code. Leave out host headers */
#if !defined __CUDACC_RTC__ && !defined __clang_llvm_bitcode_lib__
#include "host/nvshmemx_api.h"
#include "device/nvshmemx_collective_launch_apis.h"
#endif
#if !defined NVSHMEM_BITCODE_APPLICATION
#include "device/nvshmemx_defines.h"
#include "device/nvshmemx_coll_defines.cuh"
#endif

#endif
