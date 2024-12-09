#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE) \
    { \
        /* QWen-57B 
           NK= 4608, 3584 */ \
        {{16, 4608, 3584}, \
         a8w8_rowwise_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE>}, \
        {{32, 4608, 3584}, \
         a8w8_rowwise_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE>}, \
        {{64, 4608, 3584}, \
         a8w8_rowwise_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 4608, 3584}, \
         a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{256, 4608, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{512, 4608, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{1024, 4608, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{2048, 4608, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{4096, 4608, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{8192, 4608, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{16384, 4608, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 4608, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-57B 
           NK= 3584, 3584 */ \
        {{16, 3584, 3584}, \
         a8w8_rowwise_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE>}, \
        {{32, 3584, 3584}, \
         a8w8_rowwise_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE>}, \
        {{64, 3584, 3584}, \
         a8w8_rowwise_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 3584, 3584}, \
         a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{256, 3584, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{512, 3584, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{1024, 3584, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{2048, 3584, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{4096, 3584, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 3584, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{16384, 3584, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{20480, 3584, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-57B 
           NK= 3584, 20480 */ \
          /*Splitk=4*/\
        {{16, 3584, 20480}, \
         a8w8_rowwise_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3_k4<DTYPE>}, \
        {{32, 3584, 20480}, \
         a8w8_rowwise_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_2x1_intrawave_v3_k4<DTYPE>}, \
        {{64, 3584, 20480}, \
         a8w8_rowwise_256x64x192x256_32x32_1x3_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k4<DTYPE>}, \
         /*Splitk=8*/\
        {{128, 3584, 20480}, \
         a8w8_rowwise_256x128x192x128_32x32_2x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k4<DTYPE>}, \
         /*Splitk=4*/\
        {{256, 3584, 20480}, \
         a8w8_rowwise_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k4<DTYPE>}, \
         /*Splitk=2*/\
        {{512, 3584, 20480}, \
         a8w8_rowwise_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<DTYPE>}, \
        {{1024, 3584, 20480}, \
         a8w8_rowwise_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{2048, 3584, 20480}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{4096, 3584, 20480}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{8192, 3584, 20480}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{16384, 3584, 20480}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{20480, 3584, 20480}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        /* QWen-57B 
           NK= 40960, 3584 */ \
        /*TODO: Tune*/\
        {{16, 40960, 3584}, \
         a8w8_rowwise_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE>}, \
        /*TODO: Tune*/\
        {{32, 40960, 3584}, \
         a8w8_rowwise_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{64, 40960, 3584}, \
         a8w8_rowwise_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 40960, 3584}, \
         a8w8_rowwise_256x128x256x128_32x32_4x2_8x32x1_8x32x1_1x8x1x32_8x8x1_1x2_intrawave_v3<DTYPE>}, \
        {{256, 40960, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{512, 40960, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{1024, 40960, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{2048, 40960, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{4096, 40960, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 40960, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 40960, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 40960, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
      /* QWen-57B TP4
           NK= 3584, 4736 */ \
        {{16, 3584, 4736}, \
         a8w8_rowwise_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE>}, \
        {{32, 3584, 4736}, \
         a8w8_rowwise_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE>}, \
        {{64, 3584, 4736}, \
         a8w8_rowwise_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 3584, 4736}, \
         a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{256, 3584, 4736}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{512, 3584, 4736}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{1024, 3584, 4736}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{2048, 3584, 4736}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{4096, 3584, 4736}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 3584, 4736}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{16384, 3584, 4736}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{20480, 3584, 4736}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-72B TP1
           NK= 8192, 29568 */ \
        {{16, 8192, 29568}, \
         a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k16<DTYPE>}, \
        {{32, 8192, 29568}, \
         a8w8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2_k2<DTYPE>}, \
        {{64, 8192, 29568}, \
         a8w8_rowwise_256x64x192x128_32x32_1x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k8<DTYPE>}, \
        {{128, 8192, 29568}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<DTYPE>}, \
        {{256, 8192, 29568}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4_k2<DTYPE>}, \
        {{512, 8192, 29568}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{1024, 8192, 29568}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{2048, 8192, 29568}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4_k2<DTYPE>}, \
        {{4096, 8192, 29568}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 8192, 29568}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 8192, 29568}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 8192, 29568}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-72B TP1
           NK= 59136, 8192 */ \
        {{16, 59136, 8192}, \
         a8w8_rowwise_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE>}, \
        {{32, 59136, 8192}, \
         a8w8_rowwise_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{64, 59136, 8192}, \
         a8w8_rowwise_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 59136, 8192}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{256, 59136, 8192}, \
         a8w8_rowwise_256x256x256x128_32x32_4x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{512, 59136, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{1024, 59136, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{2048, 59136, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{4096, 59136, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 59136, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 59136, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 59136, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-72B TP1/
           NK= 8192, 8192 */ \
        {{16, 8192, 8192}, \
         a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2_k4<DTYPE>}, \
        {{32, 8192, 8192}, \
         a8w8_rowwise_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{64, 8192, 8192}, \
         a8w8_rowwise_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 8192, 8192}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<DTYPE>}, \
        {{256, 8192, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4_k2<DTYPE>}, \
        {{512, 8192, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{1024, 8192, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{2048, 8192, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{4096, 8192, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 8192, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 8192, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 8192, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-72B TP1
           NK= 10240, 8192 */ \
        {{16, 10240, 8192}, \
         a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2<DTYPE>}, \
        {{32, 10240, 8192}, \
         a8w8_rowwise_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{64, 10240, 8192}, \
         a8w8_rowwise_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 10240, 8192}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<DTYPE>}, \
        {{256, 10240, 8192}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{512, 10240, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{1024, 10240, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{2048, 10240, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{4096, 10240, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 10240, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 10240, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 10240, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-72B TP2
           NK= 8192, 14784 */ \
        {{16, 8192, 14784}, \
         a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k8<DTYPE>}, \
        {{32, 8192, 14784}, \
         a8w8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2_k2<DTYPE>}, \
        {{64, 8192, 14784}, \
         a8w8_rowwise_256x64x224x128_16x16_2x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<DTYPE>}, \
        {{128, 8192, 14784}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<DTYPE>}, \
        {{256, 8192, 14784}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4_k2<DTYPE>}, \
        {{512, 8192, 14784}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{1024, 8192, 14784}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{2048, 8192, 14784}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{4096, 8192, 14784}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 8192, 14784}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 8192, 14784}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 8192, 14784}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
         /* QWen-72B TP2
           NK= 29568, 8192 */ \
        {{16, 29568, 8192}, \
         a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k4<DTYPE>}, \
        {{32, 29568, 8192}, \
         a8w8_rowwise_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{64, 29568, 8192}, \
         a8w8_rowwise_256x64x192x128_32x32_1x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 29568, 8192}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{256, 29568, 8192}, \
         a8w8_rowwise_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{512, 29568, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{1024, 29568, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{2048, 29568, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{4096, 29568, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 29568, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 29568, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 29568, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-72B TP2
           NK= 8192, 4096 */ \
        {{16, 8192, 4096}, \
         a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1_k4<DTYPE>}, \
        {{32, 8192, 4096}, \
         a8w8_rowwise_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{64, 8192, 4096}, \
         a8w8_rowwise_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 8192, 4096}, \
         a8w8_rowwise_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{256, 8192, 4096}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{512, 8192, 4096}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{1024, 8192, 4096}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{2048, 8192, 4096}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{4096, 8192, 4096}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 8192, 4096}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 8192, 4096}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 8192, 4096}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
         /* QWen-72B TP2
           NK= 5120, 8192 */ \
        {{16, 5120, 8192}, \
         a8w8_rowwise_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE>}, \
        {{32, 5120, 8192}, \
         a8w8_rowwise_256x32x64x512_16x16_2x1_32x8x1_32x8x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE>}, \
        {{64, 5120, 8192}, \
         a8w8_rowwise_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 5120, 8192}, \
         a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{256, 5120, 8192}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<DTYPE>}, \
        {{512, 5120, 8192}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{1024, 5120, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{2048, 5120, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{4096, 5120, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 5120, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 5120, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 5120, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-72B TP4
           NK= 8192, 7392 */ \
        {{16, 8192, 7392}, \
         a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2_k2<DTYPE>}, \
        {{32, 8192, 7392}, \
         a8w8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2_k2<DTYPE>}, \
        {{64, 8192, 7392}, \
         a8w8_rowwise_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 8192, 7392}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<DTYPE>}, \
        {{256, 8192, 7392}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4_k2<DTYPE>}, \
        {{512, 8192, 7392}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{1024, 8192, 7392}, \
         a8w8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<DTYPE>}, \
        {{2048, 8192, 7392}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{4096, 8192, 7392}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 8192, 7392}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 8192, 7392}, \
         a8w8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE>}, \
        {{20480, 8192, 7392}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
         /* QWen-72B TP4
           NK= 14784, 8192 */ \
        {{16, 14784, 8192}, \
         a8w8_rowwise_256x16x192x256_16x16_1x3_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE>}, \
        {{32, 14784, 8192}, \
         a8w8_rowwise_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE>}, \
        {{64, 14784, 8192}, \
         a8w8_rowwise_256x64x192x128_32x32_1x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<DTYPE>}, \
        {{128, 14784, 8192}, \
         a8w8_rowwise_256x128x192x128_32x32_2x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{256, 14784, 8192}, \
         a8w8_rowwise_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{512, 14784, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4_k2<DTYPE>}, \
        {{1024, 14784, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{2048, 14784, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{4096, 14784, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 14784, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 14784, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 14784, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-72B TP4
           NK= 8192, 2048 */ \
        {{16, 8192, 2048}, \
         a8w8_rowwise_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE>}, \
        {{32, 8192, 2048}, \
         a8w8_rowwise_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{64, 8192, 2048}, \
         a8w8_rowwise_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 8192, 2048}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{256, 8192, 2048}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{512, 8192, 2048}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{1024, 8192, 2048}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{2048, 8192, 2048}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{4096, 8192, 2048}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 8192, 2048}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 8192, 2048}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 8192, 2048}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
         /* QWen-72B TP4
           NK= 2560, 8192 */ \
        {{16, 2560, 8192}, \
         a8w8_rowwise_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3_k4<DTYPE>}, \
        {{32, 2560, 8192}, \
         a8w8_rowwise_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3_k2<DTYPE>}, \
        {{64, 2560, 8192}, \
         a8w8_rowwise_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<DTYPE>}, \
        {{128, 2560, 8192}, \
         a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<DTYPE>}, \
        {{256, 2560, 8192}, \
         a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{512, 2560, 8192}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<DTYPE>}, \
        {{1024, 2560, 8192}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{2048, 2560, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{4096, 2560, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 2560, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 2560, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 2560, 8192}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-7B TP1
           NK= 3584, 18944 */ \
        {{16, 3584, 18944}, \
         a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2_k2<DTYPE>}, \
        {{32, 3584, 18944}, \
         a8w8_rowwise_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3_k2<DTYPE>}, \
        {{64, 3584, 18944}, \
         a8w8_rowwise_256x64x192x128_32x32_1x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k8<DTYPE>}, \
        {{128, 3584, 18944}, \
         a8w8_rowwise_256x128x192x128_32x32_2x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k4<DTYPE>}, \
        {{256, 3584, 18944}, \
         a8w8_rowwise_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k4<DTYPE>}, \
        {{512, 3584, 18944}, \
         a8w8_rowwise_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<DTYPE>}, \
        {{1024, 3584, 18944}, \
         a8w8_rowwise_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{2048, 3584, 18944}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4_k2<DTYPE>}, \
        {{4096, 3584, 18944}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 3584, 18944}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 3584, 18944}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 3584, 18944}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-7B TP1
           NK= 37888, 3584 */ \
        {{16, 37888, 3584}, \
         a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v1<DTYPE>}, \
        {{32, 37888, 3584}, \
         a8w8_rowwise_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{64, 37888, 3584}, \
         a8w8_rowwise_256x64x192x128_32x32_1x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 37888, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{256, 37888, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{512, 37888, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{1024, 37888, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{2048, 37888, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{4096, 37888, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 37888, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 37888, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 37888, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-7B TP1/
           NK= 3584, 3584 */ \
        {{16, 3584, 3584}, \
         a8w8_rowwise_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE>}, \
        {{32, 3584, 3584}, \
         a8w8_rowwise_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE>}, \
        {{64, 3584, 3584}, \
         a8w8_rowwise_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 3584, 3584}, \
         a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{256, 3584, 3584}, \
         a8w8_rowwise_256x128x96x256_32x32_1x3_16x16x1_16x16x1_1x64x1x4_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{512, 3584, 3584}, \
         a8w8_rowwise_256x256x96x128_32x32_2x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<DTYPE>}, \
        {{1024, 3584, 3584}, \
         a8w8_rowwise_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{2048, 3584, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{4096, 3584, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 3584, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 3584, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 3584, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        /* QWen-7B TP1
           NK= 4608, 3584 */ \
        {{16, 4608, 3584}, \
         a8w8_rowwise_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE>}, \
        {{32, 4608, 3584}, \
         a8w8_rowwise_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE>}, \
        {{64, 4608, 3584}, \
         a8w8_rowwise_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{128, 4608, 3584}, \
         a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{256, 4608, 3584}, \
         a8w8_rowwise_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{512, 4608, 3584}, \
         a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DTYPE>}, \
        {{1024, 4608, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{2048, 4608, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{4096, 4608, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{8192, 4608, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{16384, 4608, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
        {{20480, 4608, 3584}, \
         a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DTYPE>}, \
    }

#endif // USE_ROCM
