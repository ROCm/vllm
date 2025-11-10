#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(USE_ROCM)
typedef __hip_bfloat16 __bfloat16;
#else
typedef __nv_bfloat16 __bfloat16;
#endif

#include <iostream>
#include <array>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <cstring>

using Tensor = torch::Tensor;

namespace vllm {

namespace allreduce_fusion {

namespace details {

static constexpr int kBytesPerAccess = 16;
static constexpr int kDefaultNCTA = 256;

}  // namespace details

namespace block_utils {

#if !defined(USE_ROCM)
template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  return val;
}
#else
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
  #pragma unroll
  for (int offset = (32 >> 1); offset > 0; offset >>= 1) {
    val += __shfl_xor(val, offset, 32);
  }
  return val;
}
#endif

template <typename T>
__inline__ __device__ T block_reduce_sum(T val) {
  static __shared__ T shared[32];
  const int tid = threadIdx.x;
  const int w_tid = tid % 32;
  const int wid = tid / 32;
  val = warp_reduce_sum(val);
  if (w_tid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
  val = is_mask ? shared[w_tid] : (T)(0.0f);
  __syncthreads();
  val = warp_reduce_sum(val);
  return val;
}

}  // namespace block_utils

namespace comm {

template <int NRanks>
struct SyncComm {
  __device__ __forceinline__ SyncComm(void** workspace) {
    counter_ptr = (int*)workspace[NRanks * 3 + 0];
    flag_ptr = (int*)workspace[NRanks * 3 + 1];
    flag_value = *flag_ptr;
    for (int r = 0; r < NRanks; ++r) {
      comm_bufs[r] = workspace[r];
      barrier_flags[r] = workspace[NRanks + r];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_flag_value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (atomicAdd(counter_ptr, 0) != gridDim.x) {
      }
      *flag_ptr = new_flag_value;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  void* comm_bufs[NRanks];
  void* barrier_flags[NRanks];
  int flag_value;
};

template <int NRanks>
class Barrier {
 public:
  __device__ __forceinline__ Barrier(int rank, SyncComm<NRanks> const& comm) {
    if (threadIdx.x < NRanks) {
      m_flag_value = comm.flag_value;
      int current_rank = rank;
      int target_rank = threadIdx.x;
      m_target_flag = reinterpret_cast<int*>(comm.barrier_flags[target_rank]) +
                      current_rank;
      m_current_flag =
          reinterpret_cast<int*>(comm.barrier_flags[current_rank]) +
          blockIdx.x * NRanks + target_rank;
    }
  }

  __device__ __forceinline__ void sync() {
    constexpr int kBarrierFlagCount = details::kDefaultNCTA;
    __syncthreads();
    if (threadIdx.x < NRanks) {
      m_flag_value = next_flag(m_flag_value);
      // To avoid the ABA problem, we need to synchronize the correct flag value
      // to all barrier_flags, even if the corresponding CTA has not been
      // launched.
      for (int flag_idx = blockIdx.x; flag_idx < kBarrierFlagCount;
           flag_idx += gridDim.x) {
        st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
      }
      while (ld_flag(m_current_flag) == prev_flag(m_flag_value)) {
      }
    }
    __syncthreads();
  }

 protected:
  __device__ void st_flag(int* addr, int flag) {
#if !defined(USE_ROCM)
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
#else
    __scoped_atomic_store_n(addr, flag, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
#endif
  }

  __device__ int ld_flag(int* addr) {
    int flag;
#if !defined(USE_ROCM)
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];"
                 : "=r"(flag)
                 : "l"(addr));
#else
    flag =
        __scoped_atomic_load_n(addr, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
#endif
    return flag;
  }

  __device__ __forceinline__ int next_flag(int flag) {
    return flag == 2 ? 0 : flag + 1;
  }

  __device__ __forceinline__ int prev_flag(int flag) {
    return flag == 0 ? 2 : flag - 1;
  }

 public:
  volatile int m_flag_value;

 private:
  int* m_target_flag;
  int* m_current_flag;
};

template <int NRanks>
struct LamportComm {
  __device__ __forceinline__ LamportComm(void** workspace, int rank) {
    counter_ptr = (int*)workspace[NRanks * 3 + 0];
    flag_ptr = (int*)workspace[NRanks * 3 + 2];
    int comm_size = *reinterpret_cast<int*>(workspace[NRanks * 3 + 3]);
    clear_ptr = (int*)workspace[NRanks * 3 + 4];
    flag_value = *flag_ptr;
    clear_size = *clear_ptr;
    int data_offset = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) +
                     static_cast<int64_t>(data_offset) * comm_size;
    }
    clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) +
                clear_offset * comm_size;
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_clear_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (atomicAdd(counter_ptr, 0) != gridDim.x) {
      }
      *flag_ptr = (flag_value + 1) % 3;
      *clear_ptr = new_clear_size;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  int* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int clear_size;
  int flag_value;
};

}  // namespace comm

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) vec_t {
  T data[vec_size];
  __device__ __forceinline__ T& operator[](int i) { return data[i]; }
  __device__ __forceinline__ T const& operator[](int i) const {
    return data[i];
  }
  __device__ __forceinline__ void load(T* ptr) {
    *this = *reinterpret_cast<vec_t<T, vec_size>*>(ptr);
  }
  __device__ __forceinline__ void store(T* ptr) {
    *reinterpret_cast<vec_t<T, vec_size>*>(ptr) = *this;
  }
  __device__ __forceinline__ void fill(T val) {
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      data[i] = val;
    }
  }
};

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void vec_add_(vec_t<T, VEC_SIZE>& self,
                                         const vec_t<T, VEC_SIZE>& other) {
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    self[i] = (float)self[i] + (float)other[i];
  }
}

template <typename T>
struct AllReduceFusionParams {
  int nranks;
  int rank;
  int size;
  int hidden_dim;
  void** workspace;
  void* allreduce_in;
  void* residual_in;
  void* residual_out;
  void* norm_out;
  void* rms_gamma;
  float rms_eps;
  float scale_factor;
};

template <typename T, int VEC_SIZE>
__device__ __forceinline__ vec_t<T, VEC_SIZE> rms_norm(
    AllReduceFusionParams<T> const& m_params,
    vec_t<T, VEC_SIZE> const& residual, vec_t<T, VEC_SIZE> const& gamma) {
  __shared__ float s_val;
  vec_t<T, VEC_SIZE> norm_out;
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    float v = static_cast<float>(reinterpret_cast<T const*>(&residual)[i]);
    acc += v * v;
  }
  acc = block_utils::block_reduce_sum<float>(acc);
  if (threadIdx.x == 0) {
    s_val = rsqrtf(acc / m_params.hidden_dim + m_params.rms_eps);
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    reinterpret_cast<T*>(&norm_out)[i] = static_cast<T>(
        static_cast<float>(reinterpret_cast<T const*>(&residual)[i]) * s_val *
        static_cast<float>(reinterpret_cast<T const*>(&gamma)[i]));
  }
  return norm_out;
}

template <typename T, int NRanks>
__global__ void allreduce_fusion_kernel_twoshot_direct(
    AllReduceFusionParams<T> params) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

  int access_id_in_token = threadIdx.x * VEC_SIZE;

  vec_t<T, VEC_SIZE> gamma;
  gamma.load(reinterpret_cast<T*>(params.rms_gamma) + access_id_in_token);

  comm::SyncComm<NRanks> comm(params.workspace);

#pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    for (int idx =
             (blockIdx.x * NRanks + r) * params.hidden_dim + access_id_in_token;
         idx < params.size; idx += gridDim.x * NRanks * params.hidden_dim) {
      reinterpret_cast<float4*>(comm.comm_bufs[params.rank])[idx / VEC_SIZE] =
          reinterpret_cast<float4*>(params.allreduce_in)[idx / VEC_SIZE];
    }
  }

  comm::Barrier<NRanks> barrier(params.rank, comm);
  barrier.sync();

  // allreduce
  for (int idx = (blockIdx.x * NRanks + params.rank) * params.hidden_dim +
                 access_id_in_token;
       idx < params.size; idx += gridDim.x * NRanks * params.hidden_dim) {
    vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      vals[r].load(reinterpret_cast<T*>(comm.comm_bufs[r]) + idx);
    }
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
      vec_add_<T, VEC_SIZE>(vals[0], vals[r]);
    }
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      vals[0].store(reinterpret_cast<T*>(comm.comm_bufs[r]) + params.size +
                    idx);
    }
  }

  barrier.sync();

#pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    int token_id = blockIdx.x * NRanks + r;
    for (int idx = token_id * params.hidden_dim + access_id_in_token;
         idx < params.size; idx += gridDim.x * NRanks * params.hidden_dim) {
      vec_t<T, VEC_SIZE> data[2];
      data[0].load(reinterpret_cast<T*>(params.residual_in) + idx);
      data[1].load(reinterpret_cast<T*>(comm.comm_bufs[params.rank]) +
                   params.size + idx);
      vec_add_<T, VEC_SIZE>(data[0], data[1]);
      data[0].store(reinterpret_cast<T*>(params.residual_out) + idx);
      auto val = rms_norm<T, VEC_SIZE>(params, data[0], gamma);
      val.store(reinterpret_cast<T*>(params.norm_out) + idx);
    }
  }

  comm.update(barrier.m_flag_value);
}

template <typename T, int NRanks>
void allreduce_fusion_kernel_launcher(AllReduceFusionParams<T> const& params,
                                      cudaStream_t stream) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  assert(params.size % params.hidden_dim == 0);
  assert(params.hidden_dim % VEC_SIZE == 0);
  // int token_num = params.size / params.hidden_dim;
  int threads_per_token = params.hidden_dim / VEC_SIZE;
  int threads_per_block = threads_per_token;
  dim3 threadsPerBlock(threads_per_block);
  dim3 numBlocks(details::kDefaultNCTA);
  allreduce_fusion_kernel_twoshot_direct<T, NRanks>
      <<<numBlocks, threadsPerBlock, 0, stream>>>(params);
}

template <typename T>
void allreduce_rms_fusion_impl(void** workspace, int rank, int nranks, int size,
                               int hidden_dim, void* allreduce_in,
                               void* residual_in, void* residual_out,
                               void* norm_out, void* rms_gamma, float eps,
                               cudaStream_t stream = 0) {
  allreduce_fusion::AllReduceFusionParams<T> params;
  params.nranks = nranks;
  params.rank = rank;
  params.size = size;
  params.hidden_dim = hidden_dim;
  params.workspace = workspace;
  params.allreduce_in = allreduce_in;
  params.residual_in = residual_in;
  params.residual_out = residual_out;
  params.norm_out = norm_out;
  params.rms_gamma = rms_gamma;
  params.rms_eps = eps;
  if (nranks == 8) {
    allreduce_fusion_kernel_launcher<T, 8>(params, stream);
  } else if (nranks == 4) {
    allreduce_fusion_kernel_launcher<T, 4>(params, stream);
  } else if (nranks == 2) {
    allreduce_fusion_kernel_launcher<T, 2>(params, stream);
  } else {
    assert(false);
  }
}

template <typename T>
struct KernelElementType {
  using type = T;
};

template <>
struct KernelElementType<c10::Half> {
  using type = __half;
};

template <>
struct KernelElementType<c10::BFloat16> {
  using type = __bfloat16;
};

}  // namespace allreduce_fusion

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

#define MAX_RANKS 32

class CustomAllreduceFusion {
 public:
  CustomAllreduceFusion(int64_t rank, int64_t world_size,
                        int64_t max_size_in_bytes)
      : rank_(rank),
        world_size_(world_size),
        max_size_in_bytes_(max_size_in_bytes) {
    int data_size =
        max_size_in_bytes * 2 +
        allreduce_fusion::details::kDefaultNCTA * world_size * sizeof(int);
    cudaMalloc(&data_, data_size);
    cudaMalloc(&counter_, sizeof(int));
    cudaMalloc(&twoshot_sync_clock_, sizeof(int));
    cudaMemset(counter_, 0, sizeof(int));
    cudaMemset(twoshot_sync_clock_, 0, sizeof(int));
  }

  ~CustomAllreduceFusion() {
    cudaFree(twoshot_sync_clock_);
    cudaFree(counter_);
    cudaFree(data_);
  }

  Tensor get_handle() {
    cudaIpcMemHandle_t handle;
    CUDACHECK(cudaIpcGetMemHandle(&handle, data_));
    auto options =
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto data_handle = torch::empty(
        {static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, options);
    std::memcpy(data_handle.data_ptr(), &handle, sizeof(cudaIpcMemHandle_t));
    return data_handle;
  }

  void open_handles(std::vector<Tensor> handles) {
    std::vector<cudaIpcMemHandle_t> ipc_handles;
    ipc_handles.reserve(world_size_);
    for (auto& handle : handles) {
      // Ensure the tensor is on the same device as the current device.
      cudaIpcMemHandle_t ipc_handle;
      std::memcpy(&ipc_handle, handle.data_ptr(), sizeof(cudaIpcMemHandle_t));
      ipc_handles.push_back(ipc_handle);
    }

    for (int i = 0; i < world_size_; ++i) {
      if (i != rank_) {
        CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_data_[i], ipc_handles[i],
                                       cudaIpcMemLazyEnablePeerAccess));
      } else {
        ipc_data_[i] = data_;
      }
    }

    for (int i = 0; i < world_size_; ++i) {
      twoshot_comm_bufs_[i] = ipc_data_[i];
      twoshot_barrier_flags_[i] =
          (int*)((char*)ipc_data_[i] + 2 * max_size_in_bytes_);
    }
  }

  Tensor get_workspace() {
    std::vector<void*> workspace(world_size_ * 3 + 5);
    for (int r = 0; r < world_size_; ++r) {
      workspace[r] = (void*)twoshot_comm_bufs_[r];
      workspace[world_size_ + r] = (void*)twoshot_barrier_flags_[r];
    }
    workspace[world_size_ * 3 + 0] = (void*)counter_;
    workspace[world_size_ * 3 + 1] = (void*)twoshot_sync_clock_;
    auto options =
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto workspace_tensor = torch::empty(
        {static_cast<int64_t>(workspace.size() * sizeof(void*))}, options);
    std::memcpy(workspace_tensor.data_ptr(), workspace.data(),
                workspace.size() * sizeof(void*));
    return workspace_tensor;
  }

 private:
  // meta
  int rank_;
  int world_size_;
  int max_size_in_bytes_;

  // data
  void* data_;
  void* ipc_data_[MAX_RANKS];

  int* counter_;

  // twoshot
  void* twoshot_comm_bufs_[MAX_RANKS];     // 2 * size * sizeof(T)
  int* twoshot_barrier_flags_[MAX_RANKS];  // nblocks * world_size
  int* twoshot_sync_clock_;
};

}  // namespace vllm
