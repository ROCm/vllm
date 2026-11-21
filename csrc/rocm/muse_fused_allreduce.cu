// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Torch entry points of the Muse Spark fused all-reduces: argument checks and
// the dispatch on the world size (the kernels are in muse_fused_allreduce.cuh).
// The caller owns the inboxes and the per-row state; every rank must issue the
// same sequence of calls. An inbox is device memory of one rank mapped into
// every peer's address space (torch symmetric memory or a CUDA/HIP IPC
// mapping); the kernels reach it with system-scope, cache-bypassing accesses,
// so it may be fine- or coarse-grained. It must be 16-byte aligned and filled
// with the empty patterns before the first call.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <optional>
#include <vector>

#include "muse_fused_allreduce.h"
#include "ops.h"

namespace {

using vllm::muse_far::BoundaryArgs;
using vllm::muse_far::Comm;
using vllm::muse_far::FinalizeArgs;
using vllm::muse_far::MAX_NR;
using vllm::muse_far::Mode;

// `hidden`: the row width of the op; `fp8_hop`: its inbox region also holds
// the FP8 second-hop records.
Comm make_comm(const std::vector<int64_t>& inboxes, torch::Tensor& state,
               const torch::Device& device, int64_t rank, int64_t max_rows,
               int64_t rows, int64_t mode, int64_t hidden, bool fp8_hop) {
  const int64_t world = static_cast<int64_t>(inboxes.size());
  TORCH_CHECK(world == 4 || world == 8, "world size must be 4 or 8, got ",
              world);
  TORCH_CHECK(0 <= rank && rank < world, "rank out of range");
  TORCH_CHECK(Mode::TWO_HOP <= mode && mode <= Mode::MULTI_BLOCK,
              "unknown communication mode ", mode);
  TORCH_CHECK(0 < rows && rows <= max_rows, "rows must be in [1, max_rows]");
  // The kernels address an op's region with 32-bit buffer offsets.
  const int64_t region = 3 * (world + 1) * max_rows * hidden * 2 +
                         (fp8_hop ? 3 * max_rows * (hidden + 16) : 0);
  TORCH_CHECK(region < (int64_t{1} << 31), "max_rows too large: the inbox ",
              "region of ", region, " bytes exceeds 2 GiB");
  for (const int64_t p : inboxes)
    TORCH_CHECK(p != 0 && p % 16 == 0, "inboxes must be 16-byte aligned");
  // One-block-per-row kernels: [max_rows][2]; multi-block: [3][max_rows].
  const int64_t per_row = mode == Mode::MULTI_BLOCK ? 3 : 2;
  TORCH_CHECK(state.scalar_type() == at::kInt && state.is_contiguous() &&
                  state.numel() == per_row * max_rows,
              "state must be ", per_row * max_rows,
              " contiguous int32 for this mode");
  TORCH_CHECK(state.device() == device, "state must be on ", device);
  Comm c{};
  for (int64_t i = 0; i < world; ++i)
    c.inbox[i] = reinterpret_cast<uint8_t*>(inboxes[i]);
  c.state = state.data_ptr<int32_t>();
  c.rank = static_cast<int32_t>(rank);
  c.max_rows = static_cast<uint32_t>(max_rows);
  c.rows = static_cast<int32_t>(rows);
  c.mode = static_cast<int32_t>(mode);
  return c;
}

void check_vector(const torch::Tensor& t, at::ScalarType dtype, int64_t n,
                  const char* name) {
  TORCH_CHECK(t.scalar_type() == dtype && t.is_contiguous() && t.numel() == n,
              name, " must be a contiguous ", dtype, " tensor of ", n,
              " elements");
}

template <class T>
T* data_or_null(const std::optional<torch::Tensor>& t) {
  return t.has_value() ? reinterpret_cast<T*>(t->data_ptr()) : nullptr;
}

__global__ void fill_kernel(uint4* p, uint64_t n16, uint32_t pattern) {
  const uint4 s = make_uint4(pattern, pattern, pattern, pattern);
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n16;
       i += (uint64_t)gridDim.x * blockDim.x)
    p[i] = s;
}

}  // namespace

void muse_fused_allreduce_finalize(
    const torch::Tensor& routes, const torch::Tensor& weight,
    const torch::Tensor& topk_weights, const torch::Tensor& topk_ids,
    const torch::Tensor& expert_map, const std::optional<torch::Tensor>& out,
    const std::optional<torch::Tensor>& q,
    const std::optional<torch::Tensor>& q_scale, double eps, double gain,
    const std::vector<int64_t>& inboxes, torch::Tensor& state, int64_t rank,
    int64_t max_rows, int64_t mode, bool read_all) {
  constexpr int64_t H = 4096, K = 8;
  const int64_t m = topk_ids.size(0);
  TORCH_CHECK(routes.scalar_type() == at::kBFloat16 && routes.is_contiguous() &&
                  routes.dim() == 3 && routes.size(0) == m &&
                  routes.size(1) == K && routes.size(2) == H,
              "routes must be a contiguous bf16 [M, 8, 4096] tensor");
  check_vector(weight, at::kBFloat16, H, "weight");
  check_vector(topk_weights, at::kFloat, m * K, "topk_weights");
  check_vector(topk_ids, at::kInt, m * K, "topk_ids");
  TORCH_CHECK(
      expert_map.scalar_type() == at::kInt && expert_map.is_contiguous(),
      "expert_map must be contiguous int32");
  TORCH_CHECK(out.has_value() != q.has_value(),
              "exactly one of out and q must be given");
  TORCH_CHECK(q.has_value() == q_scale.has_value(), "q needs q_scale");
  if (out.has_value()) check_vector(*out, at::kBFloat16, m * H, "out");
  if (q.has_value()) {
    check_vector(*q, at::kFloat8_e4m3fn, m * H, "q");
    check_vector(*q_scale, at::kFloat, m, "q_scale");
  }
  const at::cuda::OptionalCUDAGuard guard(routes.device());
  FinalizeArgs a{};
  a.y = reinterpret_cast<const uint16_t*>(routes.data_ptr());
  a.gamma = reinterpret_cast<const uint16_t*>(weight.data_ptr());
  a.tw = topk_weights.data_ptr<float>();
  a.ids = topk_ids.data_ptr<int32_t>();
  a.emap = expert_map.data_ptr<int32_t>();
  a.out = data_or_null<uint16_t>(out);
  a.q = data_or_null<uint8_t>(q);
  a.qs = data_or_null<float>(q_scale);
  a.eps = static_cast<float>(eps);
  a.gain = static_cast<float>(gain);
  a.comm = make_comm(inboxes, state, routes.device(), rank, max_rows, m, mode,
                     H, /*fp8_hop=*/mode != Mode::MULTI_BLOCK);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (inboxes.size() == 4)
    vllm::muse_far::launch_finalize<4>(a, read_all, stream);
  else
    vllm::muse_far::launch_finalize<8>(a, read_all, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void muse_fused_allreduce_boundary(
    const torch::Tensor& partial, const torch::Tensor& stream_in,
    const torch::Tensor& alpha, const torch::Tensor& beta,
    const torch::Tensor& post_weight, const torch::Tensor& pre_weight,
    torch::Tensor& new_stream, const std::optional<torch::Tensor>& out,
    torch::Tensor& q, torch::Tensor& q_scale, double post_eps, double pre_eps,
    double post_gain, double pre_gain, bool post_input_scale,
    bool pre_input_scale, const std::vector<int64_t>& inboxes,
    torch::Tensor& state, int64_t rank, int64_t max_rows, int64_t mode) {
  constexpr int64_t H = 8192;
  const int64_t m = partial.size(0);
  TORCH_CHECK(mode != vllm::muse_far::Mode::TWO_HOP_CHUNKED,
              "the boundary has no chunked two-hop mode");
  check_vector(partial, at::kBFloat16, m * H, "partial");
  check_vector(stream_in, at::kFloat, m * H, "stream");
  check_vector(alpha, at::kFloat, H, "alpha");
  check_vector(beta, at::kFloat, H, "beta");
  check_vector(post_weight, at::kBFloat16, H, "post_weight");
  check_vector(pre_weight, at::kBFloat16, H, "pre_weight");
  check_vector(new_stream, at::kFloat, m * H, "new_stream");
  if (out.has_value()) check_vector(*out, at::kBFloat16, m * H, "out");
  check_vector(q, at::kFloat8_e4m3fn, m * H, "q");
  check_vector(q_scale, at::kFloat, m, "q_scale");
  const at::cuda::OptionalCUDAGuard guard(partial.device());
  BoundaryArgs a{};
  a.partial = reinterpret_cast<const uint16_t*>(partial.data_ptr());
  a.stream = stream_in.data_ptr<float>();
  a.alpha = alpha.data_ptr<float>();
  a.beta = beta.data_ptr<float>();
  a.post_w = reinterpret_cast<const uint16_t*>(post_weight.data_ptr());
  a.pre_w = reinterpret_cast<const uint16_t*>(pre_weight.data_ptr());
  a.new_stream = new_stream.data_ptr<float>();
  a.out = data_or_null<uint16_t>(out);
  a.q = reinterpret_cast<uint8_t*>(q.data_ptr());
  a.qs = q_scale.data_ptr<float>();
  a.post_eps = static_cast<float>(post_eps);
  a.pre_eps = static_cast<float>(pre_eps);
  a.post_gain = static_cast<float>(post_gain);
  a.pre_gain = static_cast<float>(pre_gain);
  a.post_input_scale = post_input_scale;
  a.pre_input_scale = pre_input_scale;
  a.comm = make_comm(inboxes, state, partial.device(), rank, max_rows, m, mode,
                     H, /*fp8_hop=*/false);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (inboxes.size() == 4)
    vllm::muse_far::launch_boundary<4>(a, stream);
  else
    vllm::muse_far::launch_boundary<8>(a, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void muse_fused_allreduce_fill(int64_t ptr, int64_t nbytes, int64_t pattern,
                               int64_t device) {
  TORCH_CHECK(ptr != 0 && ptr % 16 == 0 && nbytes % 16 == 0 && nbytes >= 0,
              "fill needs a 16-byte aligned range");
  if (nbytes == 0) return;
  const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fill_kernel<<<1024, 256, 0, stream>>>(reinterpret_cast<uint4*>(ptr),
                                        static_cast<uint64_t>(nbytes / 16),
                                        static_cast<uint32_t>(pattern));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void muse_fused_allreduce_close_ipc_handle(int64_t ptr, int64_t device) {
  const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  C10_CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr)));
}
