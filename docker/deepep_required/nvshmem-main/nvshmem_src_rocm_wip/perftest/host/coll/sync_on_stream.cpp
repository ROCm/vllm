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

#include "coll_test.h"

int main(int argc, char *argv[]) {
    int status = 0;
    int mype;
    size_t size = 1;
    struct timeval t_start, t_stop;
    float ms = 0;
    double latency_value;
    hipEvent_t start_event, stop_event;
    hipStream_t stream;

    read_args(argc, argv);

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
#ifdef _NVSHMEM_DEBUG
    int npes = nvshmem_n_pes();
#endif
    CUDA_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    CUDA_CHECK(hipEventCreate(&start_event));
    CUDA_CHECK(hipEventCreate(&stop_event));

    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype, npes);

    for (size_t iter = 0; iter < iters + warmup_iters; iter++) {
        if (iter == warmup_iters) CUDA_CHECK(hipEventRecord(start_event, stream));

        nvshmemx_team_sync_on_stream(NVSHMEM_TEAM_WORLD, stream);
    }
    CUDA_CHECK(hipEventRecord(stop_event, stream));
    CUDA_CHECK(hipStreamSynchronize(stream));
    CUDA_CHECK(hipEventElapsedTime(&ms, start_event, stop_event));

    if (!mype) {
        latency_value = (ms / iters) * 1000;
        print_table_basic("sync_on_stream", "None", "size (Bytes)", "latency", "us", '-', &size,
                          &latency_value, 1);
    }

    nvshmem_barrier_all();

    CUDA_CHECK(hipStreamDestroy(stream));
    CUDA_CHECK(hipEventDestroy(start_event));
    CUDA_CHECK(hipEventDestroy(stop_event));

    finalize_wrapper();

    return status;
}
