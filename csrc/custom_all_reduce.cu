#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "custom_all_reduce.cuh"

// fake pointer type
using fptr_t = uint64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

fptr_t init_custom_ar(torch::Tensor& meta, torch::Tensor& rank_data,
                      const std::vector<std::string>& handles,
                      const std::vector<int64_t>& offsets, int rank,
                      bool full_nvlink) {
  int world_size = offsets.size();
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (world_size != handles.size())
    throw std::invalid_argument(
        "handles length should equal to offsets length");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");

  cudaIpcMemHandle_t ipc_handles[8];
  for (int i = 0; i < world_size; i++) {
    std::memcpy(&ipc_handles[i], handles[i].data(), sizeof(cudaIpcMemHandle_t));
  }
  return (fptr_t) new vllm::CustomAllreduce(
      reinterpret_cast<vllm::Signal*>(meta.data_ptr()), rank_data.data_ptr(),
      rank_data.numel(), ipc_handles, offsets, rank, full_nvlink);
}

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 *
 * Examples
 * A = torch.zeros(3, 3, 3)
 * 1. A: OK
 * 2. A[1:]: OK
 * 3. A.permute(2, 0, 1): OK
 * 4. A[1:].permute(2, 0, 1): OK
 * 5. A[None].expand(2, -1, -1, -1): Not OK
 * 6. A[:, 1:, 1:]: Not OK
 */
bool _is_weak_contiguous(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() ==
          t.numel() * t.element_size());
}

bool should_custom_ar(torch::Tensor& inp, int max_size, int world_size,
                      bool full_nvlink) {
  auto inp_size = inp.numel() * inp.element_size();
  // custom allreduce requires input byte size to be multiples of 16
  if (inp_size % 16 != 0) return false;
  if (!_is_weak_contiguous(inp)) return false;
  if (world_size == 2 || full_nvlink) return inp_size <= max_size;
  // for 4 or more non NVLink-capable GPUs, custom allreduce provides little
  // performance improvement over NCCL.
  return false;
}

void _all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                 cudaStream_t stream) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  TORCH_CHECK(_is_weak_contiguous(out));
  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->allreduce<float>(stream, reinterpret_cast<float*>(inp.data_ptr()),
                           reinterpret_cast<float*>(out.data_ptr()),
                           out.numel());
      break;
    }
    case at::ScalarType::Half: {
      fa->allreduce<half>(stream, reinterpret_cast<half*>(inp.data_ptr()),
                          reinterpret_cast<half*>(out.data_ptr()), out.numel());
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
      fa->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(inp.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel());
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "custom allreduce only supports float32, float16 and bfloat16");
  }
}

void all_reduce_reg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  _all_reduce(_fa, inp, out, stream);
}

void all_reduce_unreg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& reg_buffer,
                      torch::Tensor& out) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  auto input_size = inp.numel() * inp.element_size();
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(input_size <= reg_buffer.numel() * reg_buffer.element_size(),
              "registered buffer is too small to contain the input");
  AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer.data_ptr(), inp.data_ptr(),
                                input_size, cudaMemcpyDeviceToDevice, stream));
  _all_reduce(_fa, reg_buffer, out, stream);
}

void dispose(fptr_t _fa) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  delete fa;
}

int meta_size() { return sizeof(vllm::Signal); }

void register_buffer(fptr_t _fa, torch::Tensor& t,
                     const std::vector<std::string>& handles,
                     const std::vector<int64_t>& offsets) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  fa->register_buffer(handles, offsets, t.data_ptr());
}

std::pair<std::vector<uint8_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(
    fptr_t _fa) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  return fa->get_graph_buffer_ipc_meta();
}

void register_graph_buffers(fptr_t _fa, const std::vector<std::string>& handles,
                            const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  fa->register_graph_buffers(handles, offsets);
}

#ifdef USE_ROCM

void free_meta_buffer(void* buffer) { hipFree(buffer); }

std::vector<uint8_t> get_meta_buffer_ipc_handle(torch::Tensor inp) {
  std::vector<uint8_t> data_handle(sizeof(cudaIpcMemHandle_t), 0);
  CUDACHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t*)data_handle.data(),
                                inp.data_ptr()));
  return data_handle;
}

torch::Tensor allocate_meta_buffer(int size) {
  auto device_index = c10::cuda::current_device();
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  void* buffer;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  AT_CUDA_CHECK(
      hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached));
  AT_CUDA_CHECK(cudaMemsetAsync(buffer, 0, size, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  auto options = torch::TensorOptions()
                     .dtype(torch::kI8)
                     .device(torch::kCUDA, device_index);
  return torch::from_blob(buffer, {size}, free_meta_buffer, options);
}

std::vector<uint8_t> get_device_bdf(int dev) {
  char busIdStr[] = "0000:00:00.0";
  std::vector<uint8_t> bdf(sizeof(busIdStr), 0);
  CUDACHECK(cudaDeviceGetPCIBusId((char*)bdf.data(), sizeof(busIdStr), dev));
  bdf.resize(bdf.size() - 1);  // remove trailing NULL
  return bdf;
}

#endif
