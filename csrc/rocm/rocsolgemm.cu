// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
// #ifdef __gfx908__
// // Uncomment ifdef and endif only if you need to undef the HIP_HALF ops below just for gfx908 and not for others
// // below lines enable hip float to half conversion which are disabled by default in hip_fp16.h
// #undef __HIP_NO_HALF_OPERATORS__
// #undef __HIP_NO_HALF_CONVERSIONS__
// #endif

#include <hip/hip_runtime.h>

// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <initializer_list>
#include <cstdlib>
#include "core/registration.h"

#include <rocblas/rocblas.h>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                \
  if (error != hipSuccess)                    \
  {                                           \
    fprintf(stderr,                           \
            "Hip error: '%s'(%d) at %s:%d\n", \
            hipGetErrorString(error),         \
            error,                            \
            __FILE__,                         \
            __LINE__);                        \
    exit(EXIT_FAILURE);                       \
  }
#endif

#ifndef CHECK_ROCBLAS_STATUS
#define CHECK_ROCBLAS_STATUS(status)                  \
    if(status != rocblas_status_success)              \
    {                                                 \
        fprintf(stderr,                               \
                "rocBLAS error: '%s'(%d) at %s:%d\n", \
                rocblas_status_to_string(status),     \
                status,                               \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }
#endif


// namespace
// {
//   rocblas_handle r_handle;
//   rocb_create_extension();
//   // /*thread_local*/ cudaStream_t weight_stream;
//   // BUG: DLM has event and stream on different devices error
//   // In multi-GPU scenerio, do names defined in this namespace exist on all devices?
//   // C++ keyword: thread_local <- maybe this can help?
//   // /*thread_local*/ cudaEvent_t event;

//   // // hipBLASLt
//   // hipblasLtHandle_t hipblaslt_handle;
//   // hipblasLtMatmulPreference_t preference;
//   // uint64_t workspace_size = 32 * 1024 * 1024;
//   // // uint64_t workspace_size = 0;
//   // void *d_workspace;
//   // int request_solutions = 1;
//   // int returnedAlgoCount = 0;

//   struct MatMulConfig
//   {
//     hipblasOperation_t op_A;
//     hipblasOperation_t op_B;
//     int M;
//     int N;
//     int K;
//     hipblasDiagType_t dtype;

//     friend auto operator<(const MatMulConfig &left, const MatMulConfig &right) -> bool
//     {
//       return std::tie(left.op_A, left.op_B, left.M, left.N, left.K, left.dtype) < std::tie(right.op_A, right.op_B, right.M, right.N, right.K, right.dtype);
//     }
//   };

//   // std::map<std::tuple<int, int, int, int, int, int>, std::vector<hipblasLtMatmulHeuristicResult_t>> heuristic_map;
//   std::map<MatMulConfig, hipblasLtMatmulHeuristicResult_t> heuristic_map;

//   // hipEvent_t start, stop;
//   // int bench_iters{1};
//   // int warmup_iters{1};

//   // bool cout_print = true;
// }

rocblas_handle r_handle;

torch::Tensor RocSolIdxBlas(
    torch::Tensor &mat1,
    torch::Tensor &mat2,
    int64_t  solution_index)
{
  auto mat1_strides{mat1.strides()};
  auto mat2_strides{mat2.strides()};
  auto mat1_sizes{mat1.sizes()};
  auto mat2_sizes{mat2.sizes()};
  // std::cout << " | mat1 info: size: " << mat1_sizes << " stride: " << mat1_strides << std::endl
  //           << " | mat2 info: size: " << mat2_sizes << " stride: " << mat2_strides << std::endl;

  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(
      mat1.dtype() == mat2.dtype(),
      "expected mat1 and mat2 to have the same dtype, but got: ", mat1.dtype(), " != ", mat2.dtype());
  TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0], "mat1 dim 1 must match mat2 dim 0");

  auto abcType{mat1.options().dtype()};
  auto options{at::TensorOptions().dtype(abcType).device(at::kCUDA)};
  auto result{torch::empty({mat1_sizes[0], mat2_sizes[1]}, options)};
  // std::cout << " | result info: size: " << result.sizes() << " stride: " << result.strides() << std::endl;

  bool transpose_result = true;
  bool transpose_mat1;
  bool transpose_mat2;
  if ((mat2_strides[0] == 1) && (mat2_strides[1] >= std::max<int64_t>(1, mat2_sizes[0])))
  {
    transpose_mat2 = false;
  }
  else if ((mat2_strides[1] == 1) && (mat2_strides[0] >= std::max<int64_t>(1, mat2_sizes[1])))
  {
    transpose_mat2 = true;
  }
  else
  {
    assert(false && "unusual strides detected, may need to clone a contiguous tensor");
  }
  if ((mat1_strides[0] == 1) && (mat1_strides[1] >= std::max<int64_t>(1, mat1_sizes[0])))
  {
    transpose_mat1 = false;
  }
  else if ((mat1_strides[1] == 1) && (mat1_strides[0] >= std::max<int64_t>(1, mat1_sizes[1])))
  {
    transpose_mat1 = true;
  }
  else
  {
    assert(false && "unusual strides detected, may need to clone a contiguous tensor");
  }

  if (transpose_result)
  {
    bool tmp = transpose_mat1;
    transpose_mat1 = !transpose_mat2;
    transpose_mat2 = !tmp;
    mat1_strides = mat2.strides();
    mat2_strides = mat1.strides();
    mat1_sizes = mat2.sizes();
    mat2_sizes = mat1.sizes();
  }
  // std::cout << " | transpose_result: " << (transpose_result ? "true" : "false") << std::endl
  //           << " | transpose_A: " << (transpose_mat1 ? "true" : "false") << std::endl
  //           << " | transpose_B: " << (transpose_mat2 ? "true" : "false") << std::endl;
  // std::cout << " | A matrix: size: " << mat1_sizes << " stride: " << mat1_strides << std::endl
  //           << " | B matrix: size: " << mat2_sizes << " stride: " << mat2_strides << std::endl;

  float one{1.0f};
  float zero{0.0f};
  int64_t m = mat1_sizes[transpose_result ? 1 : 0];
  int64_t k = mat1_sizes[transpose_result ? 0 : 1];
  int64_t n = mat2_sizes[transpose_result ? 0 : 1];
  int64_t mat1_ld = mat1_strides[(transpose_mat1 == transpose_result) ? 1 : 0];
  int64_t mat2_ld = mat2_strides[(transpose_mat2 == transpose_result) ? 1 : 0];
  int64_t result_ld = result.stride(transpose_result ? 0 : 1);
  // std::cout << " | (m, n, k): " << m << ", " << n << ", " << k << std::endl
  //           << " | (lda, ldb, ldc): " << mat1_ld << ", " << mat2_ld << ", " << result_ld << std::endl;

  // std::cout << "Input mat1: size: " << mat1.sizes() << ", stride: " << mat1.strides() << std::endl;
  // std::cout << "Input mat2: size: " << mat2.sizes() << ", stride: " << mat2.strides() << std::endl;
  // std::cout << "mat1 data: ";
  // std::cout << mat1 << std::endl; // 打印 mat1 的数据
  // std::cout << "mat2 data: ";
  // std::cout << mat2 << std::endl; // 打印 mat2 的数据


  void *ptrA{static_cast<void *>((transpose_result ? mat2 : mat1).data_ptr())};
  void *ptrB{static_cast<void *>((transpose_result ? mat1 : mat2).data_ptr())};
  void *ptrC{static_cast<void *>(result.data_ptr())};
  auto current_stream{torch::hip::getCurrentHIPStream().stream()};
  if (r_handle == nullptr)
    rocblas_create_handle(&r_handle);
  rocblas_set_stream(r_handle, current_stream);
  uint32_t flags{0};
  // int32_t solution_index {0};
  rocblas_datatype abcRtype;
  if (abcType == at::kHalf)
  {
    abcRtype = rocblas_datatype_f16_r;
  }
  else if (abcType == at::kBFloat16)
  {
    abcRtype = rocblas_datatype_bf16_r;
  }
  else if (abcType == at::kFloat)
  {
    abcRtype = rocblas_datatype_f32_r;
  }
  else
  {
    assert(false && "Wrong datatype!");
  }

  // CHECK_ROCBLAS_ERROR(
  rocblas_status rstatus = rocblas_gemm_ex(r_handle,
                      transpose_mat1 ? rocblas_operation_transpose : rocblas_operation_none,
                      transpose_mat2 ? rocblas_operation_transpose : rocblas_operation_none,
                      m, n, k,
                      &one, ptrA, abcRtype, mat1_ld, ptrB, abcRtype, mat2_ld,
                      &zero, ptrC, abcRtype, result_ld,
                      ptrC, abcRtype, result_ld,
                      rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, solution_index, flags);
  //);
  CHECK_ROCBLAS_STATUS(rstatus);
  return result;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////

void rocb_create_extension()
{
  rocblas_create_handle(&r_handle);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void rocb_destroy_extension()
{
  rocblas_destroy_handle(r_handle);
}


TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("RocSolIdxBlas", &RocSolIdxBlas);
}