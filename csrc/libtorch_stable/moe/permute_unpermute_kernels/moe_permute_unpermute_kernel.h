#pragma once
// reference from tensorrt_llm moe kernel implementation archive in
// https://github.com/BBuf/tensorrt-llm-moe/tree/master

#include <torch/csrc/stable/tensor.h>
#include "libtorch_stable/moe/permute_unpermute_kernels/dispatch.h"

#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
  #include <hipcub/hipcub.hpp>
#else
  #include <cub/cub.cuh>
  #include <cub/device/device_radix_sort.cuh>
  #include <cub/util_type.cuh>
#endif

// ============================================================================
// cutlass replacements for ROCm (cutlass is not available on HIP)
// ============================================================================
#ifdef USE_ROCM

namespace cutlass {

// Drop in replacement for cutlass::Array with the only strictily required
// methods
template <typename T, int N>
struct Array {
  static constexpr int kElements = N;
  using Element = T;
  T storage[N];

  __host__ __device__ T& operator[](int i) { return storage[i]; }
  __host__ __device__ const T& operator[](int i) const { return storage[i]; }

  __host__ __device__ void fill(T val) {
  #pragma unroll
    for (int i = 0; i < N; ++i) storage[i] = val;
  }

  __host__ __device__ Array operator+(const Array& rhs) const {
    Array result;
  #pragma unroll
    for (int i = 0; i < N; ++i) result[i] = storage[i] + rhs[i];
    return result;
  }
};

template <typename T, int N>
__host__ __device__ Array<T, N> operator*(T scalar, const Array<T, N>& arr) {
  Array<T, N> result;
  #pragma unroll
  for (int i = 0; i < N; ++i) result[i] = scalar * arr[i];
  return result;
}

template <typename T>
struct sizeof_bits {
  static constexpr int value = sizeof(T) * 8;
};

template <>
struct sizeof_bits<hip_bfloat16> {
  static constexpr int value = 16;
};

}  // namespace cutlass

// Type conversion helpers for ROCm (where __HIP_NO_HALF_CONVERSIONS__ is set)
template <typename To, typename From>
__host__ __device__ inline To rocm_moe_convert(From val) {
  return static_cast<To>(val);
}

template <>
__host__ __device__ inline float rocm_moe_convert<float, __half>(__half val) {
  return __half2float(val);
}

template <>
__host__ __device__ inline __half rocm_moe_convert<__half, float>(float val) {
  return __float2half(val);
}

template <>
__host__ __device__ inline float rocm_moe_convert<float, __hip_fp8_e4m3>(
    __hip_fp8_e4m3 val) {
  __half_raw hr = __hip_cvt_fp8_to_halfraw(val.__x, __HIP_E4M3);
  return rocm_moe_convert<float, __half>(__half(hr));
}

template <>
__host__ __device__ inline __hip_fp8_e4m3
rocm_moe_convert<__hip_fp8_e4m3, float>(float val) {
  __hip_fp8_e4m3 result;
  result.__x = __hip_cvt_float_to_fp8(val, __HIP_SATFINITE, __HIP_E4M3);
  return result;
}

template <>
__host__ __device__ inline float rocm_moe_convert<float, __hip_fp8_e5m2>(
    __hip_fp8_e5m2 val) {
  __half_raw hr = __hip_cvt_fp8_to_halfraw(val.__x, __HIP_E5M2);
  return rocm_moe_convert<float, __half>(__half(hr));
}

template <>
__host__ __device__ inline __hip_fp8_e5m2
rocm_moe_convert<__hip_fp8_e5m2, float>(float val) {
  __hip_fp8_e5m2 result;
  result.__x = __hip_cvt_float_to_fp8(val, __HIP_SATFINITE, __HIP_E5M2);
  return result;
}

#else  // !USE_ROCM

  #include "cutlass/numeric_size.h"
  #include "cutlass/array.h"

#endif  // USE_ROCM

template <typename T>
inline T* get_ptr(torch::stable::Tensor& t) {
  return reinterpret_cast<T*>(t.mutable_data_ptr());
}

template <typename T>
inline const T* get_ptr(const torch::stable::Tensor& t) {
  return reinterpret_cast<const T*>(t.const_data_ptr());
}

class CubKeyValueSorter {
 public:
  CubKeyValueSorter();

  CubKeyValueSorter(int const num_experts);

  void updateNumExperts(int const num_experts);

  static size_t getWorkspaceSize(size_t const num_key_value_pairs,
                                 int const num_experts);

  void run(void* workspace, size_t const workspace_size, int const* keys_in,
           int* keys_out, int const* values_in, int* values_out,
           size_t const num_key_value_pairs, cudaStream_t stream);

 private:
  static int expertsToBits(int experts);
  int num_experts_;
  int num_bits_;
};

void computeExpertFirstTokenOffset(int const* sorted_indices,
                                   int const total_indices,
                                   int const num_experts,
                                   int64_t* expert_first_token_offset,
                                   cudaStream_t stream);

void sortAndScanExpert(const int* expert_for_source_row, const int* source_rows,
                       int* permuted_experts, int* permuted_rows,
                       int64_t* expert_first_token_offset, int num_rows,
                       int num_experts, int num_experts_per_node, int k,
                       CubKeyValueSorter& sorter, void* sorter_ws,
                       cudaStream_t stream);

template <typename T>
void expandInputRowsKernelLauncher(
    T const* unpermuted_input, T* permuted_output,
    int const* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row, int* permuted_idx,
    int64_t const* expert_first_token_offset, int64_t const num_rows,
    int64_t const* num_valid_tokens_ptr, int64_t const cols, int const k,
    int num_local_experts, cudaStream_t stream);

template <class T, class OutputType>
void finalizeMoeRoutingKernelLauncher(
    T const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    float const* scales, int const* expanded_source_row_to_expanded_dest_row,
    int64_t const num_rows, int64_t const cols, int64_t const k,
    int64_t const* num_valid_ptr, cudaStream_t stream);

void preprocessTopkIdLauncher(int* topk_id_ptr, int size,
                              const int* expert_map_ptr, int num_experts,
                              cudaStream_t stream);

#include "moe_permute_unpermute_kernel.inl"
