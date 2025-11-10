#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include "custom_all_reduce_fusion.cuh"

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

fptr_t init_custom_ar_fusion(int64_t rank, int64_t world_size,
                             int64_t max_size_in_bytes) {
  switch (world_size) {
    case 8:
    case 4:
    case 2:
      break;
    default:
      throw std::invalid_argument("world size is not supported");
  }
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");
  return (fptr_t) new vllm::CustomAllreduceFusion(rank, world_size,
                                                  max_size_in_bytes);
}

void destroy_custom_ar_fusion(fptr_t fptr) {
  auto ptr = reinterpret_cast<vllm::CustomAllreduceFusion*>(fptr);
  delete ptr;
}

Tensor get_arfusion_handle(fptr_t fptr) {
  auto ptr = reinterpret_cast<vllm::CustomAllreduceFusion*>(fptr);
  return ptr->get_handle();
}

void open_arfusion_handles(fptr_t fptr, std::vector<Tensor> handles) {
  auto ptr = reinterpret_cast<vllm::CustomAllreduceFusion*>(fptr);
  ptr->open_handles(handles);
}

Tensor get_arfusion_workspace(fptr_t fptr) {
  auto ptr = reinterpret_cast<vllm::CustomAllreduceFusion*>(fptr);
  return ptr->get_workspace();
}

void allreduce_rms_fusion(int64_t rank, int64_t nranks, Tensor& allreduce_in,
                          Tensor& residual_in, Tensor& rms_gamma,
                          Tensor& residual_out, Tensor& norm_out, double eps,
                          Tensor& workspace) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(allreduce_in));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  int size = allreduce_in.numel();
  int hidden_dim = allreduce_in.size(-1);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, allreduce_in.scalar_type(),
      "allreduce_rms_fusion", [&] {
        using k_scalar_t =
            vllm::allreduce_fusion::KernelElementType<scalar_t>::type;
        vllm::allreduce_fusion::allreduce_rms_fusion_impl<k_scalar_t>(
            (void**)workspace.data_ptr(), rank, nranks, size, hidden_dim,
            (void*)allreduce_in.data_ptr<scalar_t>(),
            (void*)residual_in.data_ptr<scalar_t>(),
            (void*)residual_out.data_ptr<scalar_t>(),
            (void*)norm_out.data_ptr<scalar_t>(),
            (void*)rms_gamma.data_ptr<scalar_t>(), eps, stream);
      });
}
