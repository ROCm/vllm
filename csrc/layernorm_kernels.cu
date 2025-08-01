#include "type_convert.cuh"
#include "dispatch_utils.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

namespace vllm {

// This kernel uses the _f16Vec to represent vectorized data.
// A conversion to/from float should exist
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
rms_norm_kernel(scalar_t* __restrict__ out,          // [..., hidden_size]
                const scalar_t* __restrict__ input,  // [..., hidden_size]
                const int64_t input_stride,
                const scalar_t* __restrict__ weight,  // [hidden_size]
                const float epsilon, const int num_tokens,
                const size_t hidden_size, const size_t vec_hidden_size) {
  __shared__ float s_variance;
  float v8_variance_sum = 0.0f;

  const int64_t tx = threadIdx.x;
  const int64_t bx = blockIdx.x;
  const int64_t num_threads = blockDim.x;

  auto* __restrict__ out_v = reinterpret_cast<_f16Vec<scalar_t, width>*>(out);
  auto* __restrict__ input_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(
          input + bx * static_cast<int64_t>(hidden_size));
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  // Compute variance. Be careful, hidden_size should multiple of 4.
  for (size_t idx = tx; idx < vec_hidden_size; idx += num_threads) {
    _f16Vec<scalar_t, width> temp = input_v[idx];
    v8_variance_sum += temp.sum_squares();
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;

  float variance =
      BlockReduce(reduceStore).Reduce(v8_variance_sum, cub::Sum{}, num_threads);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  variance = s_variance;

  for (size_t idx = tx; idx < vec_hidden_size; idx += num_threads) {
    _f16Vec<scalar_t, width> temp = input_v[idx];
    temp *= variance;
    temp *= weight_v[idx];
    out_v[bx * static_cast<int64_t>(vec_hidden_size) + idx] = temp;
  }
}

// Non vectorized kernel for unusual shapes/types without conversion
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
rms_norm_kernel(scalar_t* __restrict__ out,          // [..., hidden_size]
                const scalar_t* __restrict__ input,  // [..., hidden_size]
                const int64_t input_stride,
                const scalar_t* __restrict__ weight,  // [hidden_size]
                const float epsilon, const int num_tokens,
                const size_t hidden_size, const size_t) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (size_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * input_stride + idx];
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (size_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * input_stride + idx];
    out[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  const int64_t vec_input_stride = input_stride / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[strided_id] = temp;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * input_stride + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

}  // namespace vllm

#define LAUNCH_RMS_NORM(width)                                               \
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] { \
    vllm::rms_norm_kernel<scalar_t, width><<<grid, block, 0, stream>>>(      \
        out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), input_stride,  \
        weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size,       \
        vec_hidden_size);                                                    \
  });

void rms_norm(torch::Tensor& out,     // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(input.stride(-1) == 1);
  TORCH_CHECK(weight.is_contiguous());

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int vec_size = 16 / input.element_size();
  int vec_hidden_size = hidden_size / vec_size;
  int64_t input_stride = input.stride(-2);
  bool can_run_vectorize =
      (hidden_size % vec_size) == 0 && input_stride == hidden_size;

  dim3 grid(num_tokens);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (vec_size % 8 == 0 && can_run_vectorize) {
    dim3 block(std::min(vec_hidden_size, 1024));
    LAUNCH_RMS_NORM(8);
  } else {
    dim3 block(std::min(hidden_size, 1024));
    LAUNCH_RMS_NORM(0);
  }
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                    \
  VLLM_DISPATCH_FLOATING_TYPES(                                             \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {               \
        vllm::fused_add_rms_norm_kernel<scalar_t, width>                    \
            <<<grid, block, 0, stream>>>(                                   \
                input.data_ptr<scalar_t>(), input_stride,                   \
                residual.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), \
                epsilon, num_tokens, hidden_size);                          \
      });

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  constexpr int vector_width = 8;
  constexpr int req_alignment_bytes =
      vector_width * 2;  // vector_width * sizeof(bfloat16 or float16) (float32
                         // falls back to non-vectorized version anyway)
  bool ptrs_are_aligned = inp_ptr % req_alignment_bytes == 0 &&
                          res_ptr % req_alignment_bytes == 0 &&
                          wt_ptr % req_alignment_bytes == 0;
  bool offsets_are_multiple_of_vector_width =
      hidden_size % vector_width == 0 && input_stride % vector_width == 0;
  if (ptrs_are_aligned && offsets_are_multiple_of_vector_width) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}
