// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// hipBLASLt w4a16 GEMM: D = dequant(A) * B, with int4 weights in A, fp16/bf16
// activations in B, and a per-K-group scale (plus optional zero-points).
//
// Built at import time by torch.utils.cpp_extension.load() against a hipBLASLt
// that exposes the w4a16 API -- it is deliberately NOT part of the CMake build,
// so vLLM keeps building against a stock ROCm SDK. See hipblaslt_w4a16.py.
//
// The vLLM W4A16 tensors map onto the API without any repacking. With
// hipBLASLt's m = vLLM's N (output features) and hipBLASLt's n = vLLM's M
// (batch), in TN orientation:
//
//   A  HIP_R_4I  (k x m, lda)  <- w_q [N, K/2] int8, ExLlama shuffle
//   B  fp16/bf16 (k x n, K)    <- x   [M, K]
//   D  fp16/bf16 (m x n, N)    <- out [M, N] row-major
//
// lda comes from the weight's row stride, so the gfx1151 cliff padding applied
// by pack_skinny_int4() is carried through as-is.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hipblaslt/hipblaslt.h>

#include <map>
#include <mutex>
#include <tuple>

namespace {

#define LT_CHECK(expr)                                             \
  do {                                                             \
    hipblasStatus_t _s = (expr);                                   \
    TORCH_CHECK(_s == HIPBLAS_STATUS_SUCCESS, #expr " failed: ", _s); \
  } while (0)

// Matches c_blockScaleAZeroPointAlignment in hipBLASLt's tensile_host.cpp.
constexpr int64_t kZeroPointAlignment = 256;

constexpr size_t kMaxWorkspaceBytes = 32u * 1024u * 1024u;

hipblasLtHandle_t lt_handle() {
  static hipblasLtHandle_t handle = [] {
    hipblasLtHandle_t h = nullptr;
    LT_CHECK(hipblasLtCreate(&h));
    return h;
  }();
  return handle;
}

hipDataType hip_type_of(at::ScalarType t) {
  switch (t) {
    case at::kHalf:
      return HIP_R_16F;
    case at::kBFloat16:
      return HIP_R_16BF;
    default:
      TORCH_CHECK(false, "hipblaslt_w4a16: unsupported activation dtype ", t);
  }
}

hipblasLtMatmulMatrixScale_t scale_mode_of(int64_t group_size, bool has_zp) {
  switch (group_size) {
    case 32:
      return has_zp ? HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_ZP_EXT
                    : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_EXT;
    case 64:
      return has_zp ? HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_ZP_EXT
                    : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_EXT;
    case 128:
      return has_zp ? HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_ZP_EXT
                    : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_EXT;
    default:
      TORCH_CHECK(false, "hipblaslt_w4a16: unsupported group_size ", group_size);
  }
}

// A plan is shared by every call with the same weight shape. The two things
// that do vary -- the batch size (the layouts' column count) and the layer's
// scale pointer -- are set on every call, which keeps the cache bounded by the
// number of distinct weight shapes rather than by the batch sizes seen. The
// w4a16 kernels come from a FreeSize logic file, so the heuristic returns the
// same solution whatever the batch size it was queried with.
struct Plan {
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatrixLayout_t layout_a = nullptr;
  hipblasLtMatrixLayout_t layout_b = nullptr;
  hipblasLtMatrixLayout_t layout_d = nullptr;
  hipblasLtMatmulAlgo_t algo{};
  size_t workspace_bytes = 0;
};

using PlanKey = std::tuple<int, int64_t, int64_t, int64_t, int64_t, bool>;

Plan& get_plan(at::ScalarType dtype, int64_t M, int64_t N, int64_t K, int64_t lda,
               int64_t group_size, bool has_zp, const void* scale_ptr) {
  static std::map<PlanKey, Plan> cache;
  const PlanKey key{static_cast<int>(dtype), N, K, lda, group_size, has_zp};
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  const hipDataType act = hip_type_of(dtype);
  Plan plan;

  LT_CHECK(hipblasLtMatrixLayoutCreate(&plan.layout_a, HIP_R_4I, K, N, lda));
  LT_CHECK(hipblasLtMatrixLayoutCreate(&plan.layout_b, act, K, M, K));
  LT_CHECK(hipblasLtMatrixLayoutCreate(&plan.layout_d, act, N, M, N));

  LT_CHECK(hipblasLtMatmulDescCreate(&plan.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasOperation_t op_a = HIPBLAS_OP_T, op_b = HIPBLAS_OP_N;
  LT_CHECK(hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                           &op_a, sizeof(op_a)));
  LT_CHECK(hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                           &op_b, sizeof(op_b)));

  int32_t encoding = HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXLLAMA_EXT;
  LT_CHECK(hipblasLtMatmulDescSetAttribute(
      plan.desc, HIPBLASLT_MATMUL_DESC_A_INT4_ENCODING_EXT, &encoding,
      sizeof(encoding)));

  // Mode before pointer: a pointer set while the mode is None defaults to Scalar.
  hipblasLtMatmulMatrixScale_t mode = scale_mode_of(group_size, has_zp);
  LT_CHECK(hipblasLtMatmulDescSetAttribute(
      plan.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
  // The heuristic only offers block-scale solutions when the scale pointer is
  // already set, so it goes on before the query and is refreshed per call.
  LT_CHECK(hipblasLtMatmulDescSetAttribute(
      plan.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_ptr,
      sizeof(scale_ptr)));

  hipblasLtMatmulPreference_t pref = nullptr;
  LT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
  size_t max_ws = kMaxWorkspaceBytes;
  LT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws)));

  hipblasLtMatmulHeuristicResult_t heuristic[1];
  int found = 0;
  LT_CHECK(hipblasLtMatmulAlgoGetHeuristic(lt_handle(), plan.desc, plan.layout_a,
                                           plan.layout_b, plan.layout_d,
                                           plan.layout_d, pref, 1, heuristic,
                                           &found));
  LT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
  TORCH_CHECK(found > 0, "hipblaslt_w4a16: no solution for M=", M, " N=", N,
              " K=", K, " lda=", lda, " group_size=", group_size, " ",
              (has_zp ? "asym" : "sym"), " ", dtype,
              " -- this hipBLASLt has no kernel for that combination");

  plan.algo = heuristic[0].algo;
  plan.workspace_bytes = heuristic[0].workspaceSize;
  return cache.emplace(key, plan).first->second;
}

// Allocated once and kept, so that a call made during CUDA-graph capture does
// not take its workspace from the graph's private pool. The w4a16 kernels ask
// for none, so in practice nothing is allocated at all.
void* workspace_for(size_t bytes, const torch::Tensor& like) {
  static torch::Tensor ws;
  if (bytes == 0) return nullptr;
  if (!ws.defined() || static_cast<size_t>(ws.numel()) < bytes) {
    ws = at::empty({static_cast<int64_t>(bytes)}, like.options().dtype(at::kByte));
  }
  return ws.data_ptr();
}

}  // namespace

torch::Tensor hipblaslt_w4a16_gemm(const torch::Tensor& a, const torch::Tensor& b_q,
                                   const torch::Tensor& scale, int64_t group_size,
                                   bool has_zp) {
  TORCH_CHECK(a.dim() == 2 && b_q.dim() == 2, "a and b_q must be 2-D");
  TORCH_CHECK(a.is_contiguous(), "activations must be contiguous");
  TORCH_CHECK(b_q.scalar_type() == at::kChar, "packed weights must be int8");
  TORCH_CHECK(b_q.stride(1) == 1, "packed weight rows must be contiguous");

  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b_q.size(0);
  TORCH_CHECK(b_q.size(1) == K / 2, "b_q must be [N, K/2], got [", N, ", ",
              b_q.size(1), "] for K=", K);
  TORCH_CHECK(group_size > 0 && K % group_size == 0, "K=", K,
              " not divisible by group_size=", group_size);

  const int64_t num_groups = K / group_size;
  if (has_zp) {
    const int64_t zp_offset =
        (num_groups * N * 2 + kZeroPointAlignment - 1) / kZeroPointAlignment *
        kZeroPointAlignment;
    TORCH_CHECK(scale.scalar_type() == at::kByte && scale.dim() == 1,
                "asymmetric scale buffer must be a 1-D uint8 tensor");
    TORCH_CHECK(scale.numel() >= zp_offset + ((N + 1) / 2) * num_groups,
                "asymmetric scale buffer too small");
  } else {
    TORCH_CHECK(scale.dim() == 2 && scale.scalar_type() == a.scalar_type(),
                "symmetric scales must be [N, K/G] in the activation dtype");
    // hipBLASLt derives the scale row stride from K inside the kernel, so a
    // padded scale row is not expressible.
    TORCH_CHECK(scale.stride(0) == num_groups && scale.stride(1) == 1,
                "symmetric scales must be densely packed, got stride(0)=",
                scale.stride(0), " for K/G=", num_groups);
  }
  TORCH_CHECK(scale.is_cuda() && a.is_cuda() && b_q.is_cuda(),
              "all inputs must be on the GPU");

  torch::Tensor out = at::empty({M, N}, a.options());
  if (M == 0) return out;

  const at::cuda::OptionalCUDAGuard guard(device_of(a));
  const int64_t lda = b_q.stride(0) * 2;  // int8 row stride -> int4 elements
  const void* scale_ptr = scale.const_data_ptr();
  Plan& plan =
      get_plan(a.scalar_type(), M, N, K, lda, group_size, has_zp, scale_ptr);

  LT_CHECK(hipblasLtMatmulDescSetAttribute(
      plan.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_ptr,
      sizeof(scale_ptr)));

  // B and D are (K x M) and (N x M): only the column count varies per call.
  const uint64_t cols = static_cast<uint64_t>(M);
  LT_CHECK(hipblasLtMatrixLayoutSetAttribute(
      plan.layout_b, HIPBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols)));
  LT_CHECK(hipblasLtMatrixLayoutSetAttribute(
      plan.layout_d, HIPBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols)));

  const float alpha = 1.0f, beta = 0.0f;
  LT_CHECK(hipblasLtMatmul(lt_handle(), plan.desc, &alpha, b_q.const_data_ptr(),
                           plan.layout_a, a.const_data_ptr(), plan.layout_b, &beta,
                           out.data_ptr(), plan.layout_d, out.data_ptr(),
                           plan.layout_d, &plan.algo,
                           workspace_for(plan.workspace_bytes, a),
                           plan.workspace_bytes,
                           at::cuda::getCurrentCUDAStream()));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hipblaslt_w4a16_gemm", &hipblaslt_w4a16_gemm,
        "W4A16 GEMM through hipBLASLt (ROCm)");
}
