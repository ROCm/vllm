/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef COLL_TEST_H
#define COLL_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <assert.h>
#include <unistd.h>
#include "utils.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <algorithm>

#define MAX_SKIP 16
#define MAX_ITERS 128
#define MAX_NPES 128
#define BARRIER_MAX_ITERS 1000
#define FCOLLECT_MAX_ITERS 1024
#define BARRIER_MAX_SKIP 10

extern int coll_max_iters;

#define alltoall_src_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems * npes)

#define alltoall_dest_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems * npes)

#define fcollect_src_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems)

#define fcollect_dest_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems * npes)

#define broadcast_src_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems)

#define broadcast_dest_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems)

#define call_shmem_broadcast(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root) \
    do {                                                                              \
        nvshmem_##TYPENAME##_broadcast(team, d_dest, d_source, num_elems, root);      \
    } while (0)

#define call_shmem_fcollect(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root) \
    do {                                                                             \
        nvshmem_##TYPENAME##_fcollect(team, d_dest, d_source, num_elems);            \
    } while (0)

#define call_shmem_alltoall(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root) \
    do {                                                                             \
        nvshmem_##TYPENAME##_alltoall(team, d_dest, d_source, num_elems);            \
    } while (0)

#define call_shmem_broadcast_on_stream(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root, \
                                       stream)                                                  \
    do {                                                                                        \
        nvshmemx_##TYPENAME##_broadcast_on_stream(team, d_dest, d_source, num_elems, root,      \
                                                  stream);                                      \
    } while (0)

#define call_shmem_fcollect_on_stream(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root, \
                                      stream)                                                  \
    do {                                                                                       \
        nvshmemx_##TYPENAME##_fcollect_on_stream(team, d_dest, d_source, num_elems, stream);   \
    } while (0)

#define call_shmem_alltoall_on_stream(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root, \
                                      stream)                                                  \
    do {                                                                                       \
        nvshmemx_##TYPENAME##_alltoall_on_stream(team, d_dest, d_source, num_elems, stream);   \
    } while (0)

/* */
#define RUN_COLL_ON_STREAM(coll, COLL, TYPENAME, TYPE, d_source, h_source, d_dest, h_dest, npes, \
                           root, stream, size_array, latency_array)                              \
    do {                                                                                         \
        int array_index = 0;                                                                     \
        size_t min_elems, max_elems;                                                             \
        if (strcmp(#coll, "broadcast") == 0) {                                                   \
            min_elems = max(static_cast<size_t>(1), min_size / sizeof(TYPE));                    \
            max_elems = max(static_cast<size_t>(1), max_size / sizeof(TYPE));                    \
        } else {                                                                                 \
            min_elems = max(static_cast<size_t>(1), min_size / (npes * sizeof(TYPE)));           \
            max_elems = max(static_cast<size_t>(1), max_size / (npes * sizeof(TYPE)));           \
        }                                                                                        \
        for (size_t num_elems = min_elems; num_elems <= max_elems; num_elems *= step_factor) {   \
            float latency = 0;                                                                   \
            cudaEvent_t t_start, t_stop;                                                         \
            CUDA_CHECK(cudaEventCreate(&t_start));                                               \
            CUDA_CHECK(cudaEventCreate(&t_stop));                                                \
            int latency_iters = 0;                                                               \
            auto lat_idx_array = latency_array[array_index];                                     \
            nvshmem_barrier_all();                                                               \
            for (int iter = 0; iter < warmup_iters; iter++) {                                    \
                call_shmem_##coll##_on_stream(TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, d_dest,        \
                                              d_source, num_elems, root, stream);                \
            }                                                                                    \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                           \
            nvshmem_barrier_all();                                                               \
            for (int iter = 0; iter < iters; iter++) {                                           \
                CUDA_CHECK(cudaEventRecord(t_start, stream));                                    \
                call_shmem_##coll##_on_stream(TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, d_dest,        \
                                              d_source, num_elems, root, stream);                \
                CUDA_CHECK(cudaEventRecord(t_stop, stream));                                     \
                CUDA_CHECK(cudaStreamSynchronize(stream));                                       \
                CUDA_CHECK(cudaEventElapsedTime(&latency, t_start, t_stop));                     \
                lat_idx_array[latency_iters] = latency * 1e+3;                                   \
                latency_iters++;                                                                 \
            }                                                                                    \
            CUDA_CHECK(cudaEventDestroy(t_start));                                               \
            CUDA_CHECK(cudaEventDestroy(t_stop));                                                \
            nvshmem_barrier_all();                                                               \
            if (strcmp(#coll, "alltoall") == 0 || strcmp(#coll, "fcollect") == 0)                \
                size_array[array_index] = num_elems * sizeof(TYPE) * npes;                       \
            else                                                                                 \
                size_array[array_index] = num_elems * sizeof(TYPE);                              \
            array_index++;                                                                       \
        }                                                                                        \
    } while (0)

#define RUN_RDXN(coll, TYPENAME, TYPE, OP, team, d_source, d_dest, size_array, latency_array,  \
                 stream)                                                                       \
    do {                                                                                       \
        cudaEvent_t start_event, stop_event;                                                   \
        CUDA_CHECK(cudaEventCreate(&start_event));                                             \
        CUDA_CHECK(cudaEventCreate(&stop_event));                                              \
        float ms = 0.0f;                                                                       \
        if (strcmp(#coll, "reduce") == 0) {                                                    \
            min_elems = max(static_cast<size_t>(1), min_size / sizeof(TYPE));                  \
            max_elems = max(static_cast<size_t>(1), max_size / sizeof(TYPE));                  \
        } else {                                                                               \
            min_elems = max(static_cast<size_t>(1), min_size / (npes * sizeof(TYPE)));         \
            max_elems = max(static_cast<size_t>(1), max_size / (npes * sizeof(TYPE)));         \
        }                                                                                      \
        int idx = 0;                                                                           \
        for (size_t num_elems = min_elems; num_elems <= max_elems; num_elems *= step_factor) { \
            nvshmemx_barrier_all_on_stream(stream);                                            \
            for (int iter = 0; iter < iters + warmup_iters; iter++) {                          \
                if (iter >= warmup_iters) CUDA_CHECK(cudaEventRecord(start_event, stream));    \
                nvshmemx_##TYPENAME##_##OP##_##coll##_on_stream(                               \
                    team, (TYPE *)d_dest, (const TYPE *)d_source, num_elems, stream);          \
                                                                                               \
                if (iter >= warmup_iters) {                                                    \
                    CUDA_CHECK(cudaEventRecord(stop_event, stream));                           \
                    CUDA_CHECK(cudaStreamSynchronize(stream));                                 \
                    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));            \
                    latency_array[idx][iter - warmup_iters] = ms * 1e+3;                       \
                }                                                                              \
            }                                                                                  \
            if (strcmp(#coll, "reduce") == 0)                                                  \
                size_array[idx] = num_elems * sizeof(TYPE);                                    \
            else                                                                               \
                size_array[idx] = num_elems * npes * sizeof(TYPE);                             \
            idx++;                                                                             \
        }                                                                                      \
        CUDA_CHECK(cudaEventDestroy(start_event));                                             \
        CUDA_CHECK(cudaEventDestroy(stop_event));                                              \
    } while (0)

#define RUN_RDXN_BITWISE_DATATYPE(coll, TYPENAME, TYPE, team, d_source, d_dest, num_elems, stream, \
                                  size_array, latency_array)                                       \
    switch (reduce_op.type) {                                                                      \
        case NVSHMEM_SUM:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, sum, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_MIN:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, min, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_MAX:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, max, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_PROD:                                                                         \
            RUN_RDXN(coll, TYPENAME, TYPE, prod, team, d_source, d_dest, size_array,               \
                     latency_array, stream);                                                       \
            break;                                                                                 \
        case NVSHMEM_AND:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, and, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_OR:                                                                           \
            RUN_RDXN(coll, TYPENAME, TYPE, or, team, d_source, d_dest, size_array, latency_array,  \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_XOR:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, xor, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
    }

#define RUN_RDXN_DATATYPE(coll, TYPENAME, TYPE, team, d_source, d_dest, num_elems, stream,         \
                          size_array, latency_array)                                               \
    switch (reduce_op.type) {                                                                      \
        case NVSHMEM_SUM:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, sum, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_MIN:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, min, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_MAX:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, max, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_PROD:                                                                         \
            RUN_RDXN(coll, TYPENAME, TYPE, prod, team, d_source, d_dest, size_array,               \
                     latency_array, stream);                                                       \
            break;                                                                                 \
    }
#endif /*COLL_TEST_H*/
