// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// hipBLASLt w4a16 GEMM: D = dequant(A) * B, with int4 weights in A, fp16/bf16
// activations in B, and a per-K-group scale (plus optional zero-points).
//
// Compiled into _rocm_C only when CMake's probe finds a hipBLASLt whose
// headers expose the w4a16 API (VLLM_HIPBLASLT_HAS_W4A16). A stock ROCm SDK
// without it simply does not get these ops. See hipblaslt_w4a16.py.
//
// The vLLM W4A16 tensors map onto the API without any repacking. With
// hipBLASLt's m = vLLM's N (output features) and hipBLASLt's n = vLLM's M
// (batch), in TN orientation:
//
//   A  HIP_R_4I  (k x m, lda)  <- w_q [N, K/2] int8, nibbles in K order
//   B  fp16/bf16 (k x n, K)    <- x   [M, K]
//   D  fp16/bf16 (m x n, N)    <- out [M, N] row-major
//
// lda comes from the weight's row stride, so the gfx1151 cliff padding applied
// by pack_skinny_int4() is carried through as-is.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <map>
#include <mutex>
#include <tuple>
#include <vector>

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

// The scale mode names the scale element type, which hipBLASLt requires to
// equal the type of matrix B -- the activations. The modes used to be
// polymorphic over that type; they are not any more, and the bf16 ones kept
// the old numeric values, so feeding an fp16 layer the bf16 mode is silent
// rather than an error.
hipblasLtMatmulMatrixScale_t scale_mode_of(at::ScalarType dtype,
                                           int64_t group_size, bool has_zp) {
  const bool bf16 = dtype == at::kBFloat16;
  switch (group_size) {
    case 32:
      return bf16 ? (has_zp ? HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_ZP_EXT
                            : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_EXT)
                  : (has_zp ? HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16F_ZP_EXT
                            : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16F_EXT);
    case 64:
      return bf16 ? (has_zp ? HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_16BF_ZP_EXT
                            : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_16BF_EXT)
                  : (has_zp ? HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_16F_ZP_EXT
                            : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_16F_EXT);
    case 128:
      return bf16 ? (has_zp ? HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_16BF_ZP_EXT
                            : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_16BF_EXT)
                  : (has_zp ? HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_16F_ZP_EXT
                            : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_16F_EXT);
    default:
      TORCH_CHECK(false, "hipblaslt_w4a16: unsupported group_size ", group_size);
  }
}

// Once the library carries size-specialised logic -- an Equality table, or a
// decode-tuned solution -- the heuristic's answer depends on the batch size, so
// a plan queried at one batch size must not be reused at another. Reusing one
// silently pins every call to whichever batch size happened to touch the layer
// first, which during start-up is a profile-run prefill: decode then replays a
// prefill-tuned kernel and the library is never asked about n=1 at all.
//
// Keying on the exact batch size would grow the cache with every prefill
// length, so it is bucketed to the next power of two: one entry per shape per
// octave, and decode (n=1) gets its own query, which is the case whose best
// kernel differs most. Within a bucket only the layouts' column count and the
// layer's scale pointer vary, and both are set on every call.
int64_t batch_bucket(int64_t M) {
  int64_t b = 1;
  while (b < M) b <<= 1;
  return b;
}

struct Plan {
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatrixLayout_t layout_a = nullptr;
  hipblasLtMatrixLayout_t layout_b = nullptr;
  hipblasLtMatrixLayout_t layout_d = nullptr;
  hipblasLtMatmulAlgo_t algo{};
  size_t workspace_bytes = 0;
};

using PlanKey =
    std::tuple<int, int64_t, int64_t, int64_t, int64_t, bool, int64_t>;

Plan& get_plan(at::ScalarType dtype, int64_t M, int64_t N, int64_t K, int64_t lda,
               int64_t group_size, bool has_zp, const void* scale_ptr) {
  static std::map<PlanKey, Plan> cache;
  const int64_t bucket = batch_bucket(M);
  const PlanKey key{static_cast<int>(dtype), N,       K,     lda,
                    group_size,              has_zp,  bucket};
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  const hipDataType act = hip_type_of(dtype);
  Plan plan;

  // Queried at the bucket, not at M, so every call in the bucket shares the
  // solution the heuristic picked for a batch size of that magnitude.
  LT_CHECK(hipblasLtMatrixLayoutCreate(&plan.layout_a, HIP_R_4I, K, N, lda));
  LT_CHECK(hipblasLtMatrixLayoutCreate(&plan.layout_b, act, K, bucket, K));
  LT_CHECK(hipblasLtMatrixLayoutCreate(&plan.layout_d, act, N, bucket, N));

  LT_CHECK(hipblasLtMatmulDescCreate(&plan.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasOperation_t op_a = HIPBLAS_OP_T, op_b = HIPBLAS_OP_N;
  LT_CHECK(hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                           &op_a, sizeof(op_a)));
  LT_CHECK(hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                           &op_b, sizeof(op_b)));

  int32_t encoding = HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT;
  LT_CHECK(hipblasLtMatmulDescSetAttribute(
      plan.desc, HIPBLASLT_MATMUL_DESC_A_INT4_ENCODING_EXT, &encoding,
      sizeof(encoding)));

  // Mode before pointer: a pointer set while the mode is None defaults to Scalar.
  hipblasLtMatmulMatrixScale_t mode = scale_mode_of(dtype, group_size, has_zp);
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
  TORCH_CHECK(found > 0, "hipblaslt_w4a16: no solution for M=", M,
              " (queried at ", bucket, ") N=", N,
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

// Runs one planned GEMM: out (N x M) = dequant(w) * x (K x M).
void run_plan(Plan& plan, const void* w, const void* x, void* out, int64_t M,
              const void* scale_ptr, const torch::Tensor& like) {
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
  LT_CHECK(hipblasLtMatmul(
      lt_handle(), plan.desc, &alpha, w, plan.layout_a, x, plan.layout_b, &beta,
      out, plan.layout_d, out, plan.layout_d, &plan.algo,
      workspace_for(plan.workspace_bytes, like), plan.workspace_bytes,
      at::cuda::getCurrentCUDAStream()));
}

// One hipblaslt_ext::GroupedGemm over the experts in `experts`. Returns false
// if hipBLASLt rejects the problem or has no grouped solution. Uses the typed
// API: the descriptor-based GroupedGemm constructor requires typeA == typeB.
bool run_grouped(const torch::Tensor& out, const torch::Tensor& x,
                 const torch::Tensor& w, const torch::Tensor& scale,
                 const std::vector<int64_t>& experts,
                 const std::vector<int64_t>& offsets,
                 const std::vector<int64_t>& counts, int64_t group_size,
                 bool has_zp, int64_t lda) {
  const int64_t N = out.size(1), K = x.size(1);
  const size_t G = experts.size();
  const hipDataType act = hip_type_of(x.scalar_type());
  const float one = 1.0f, zero = 0.0f;
  std::vector<int64_t> m(G, N), n(counts), k(G, K), batch(G, 1);
  std::vector<int64_t> v_lda(G, lda), v_ldb(G, K), v_ldc(G, N);
  std::vector<int64_t> s_a(G, lda * N), s_b(G), s_c(G);
  std::vector<hipblaslt_ext::GemmEpilogue> epilogue(G);
  std::vector<hipblaslt_ext::GemmInputs> inputs(G);
  const auto mode = scale_mode_of(x.scalar_type(), group_size, has_zp);
  for (size_t i = 0; i < G; ++i) {
    s_b[i] = K * n[i];
    s_c[i] = N * n[i];
    epilogue[i].setScalingAType(mode);
    void* o = static_cast<char*>(out.data_ptr()) +
              offsets[i] * N * out.element_size();
    inputs[i].setA(w[experts[i]].const_data_ptr());
    inputs[i].setB(static_cast<const char*>(x.const_data_ptr()) +
                   offsets[i] * K * x.element_size());
    inputs[i].setC(o);
    inputs[i].setD(o);
    inputs[i].setAlpha(&one);
    inputs[i].setBeta(&zero);
    inputs[i].setScaleA(scale[experts[i]].const_data_ptr());
  }
  try {
    hipblaslt_ext::GemmProblemType type(HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_4I,
                                        act, act, act, HIPBLAS_COMPUTE_32F);
    type.setInt4EncodingA(HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT);
    hipblaslt_ext::GroupedGemm gg(lt_handle(), HIPBLAS_OP_T, HIPBLAS_OP_N,
                                  HIP_R_4I, act, act, act, HIPBLAS_COMPUTE_32F);
    if (gg.setProblem(m, n, k, batch, v_lda, v_ldb, v_ldc, v_ldc, s_a, s_b, s_c,
                      s_c, epilogue, inputs, type) != HIPBLAS_STATUS_SUCCESS)
      return false;
    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(kMaxWorkspaceBytes);
    std::vector<hipblasLtMatmulHeuristicResult_t> res;
    if (gg.algoGetHeuristic(1, pref, res) != HIPBLAS_STATUS_SUCCESS ||
        res.empty())
      return false;
    const auto stream = at::cuda::getCurrentCUDAStream();
    LT_CHECK(gg.initialize(res[0].algo, workspace_for(res[0].workspaceSize, x),
                           false, stream));
    LT_CHECK(gg.run(stream));
    return true;
  } catch (const std::exception& e) {
    TORCH_WARN_ONCE("hipBLASLt GroupedGemm failed: ", e.what());
  } catch (...) {
    TORCH_WARN_ONCE("hipBLASLt GroupedGemm threw a non-std exception");
  }
  return false;
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

  run_plan(plan, b_q.const_data_ptr(), a.const_data_ptr(), out.data_ptr(), M,
           scale_ptr, a);
  return out;
}

// MoE expert GEMMs. Rows of x are sorted by expert; counts (CPU int64 [E]) is
// the number of rows per expert. w: [E, N, K/2] int8, nibbles in K order.
// scale: [E, bytes] uint8, one scale (+ zero-point) allocation per expert in
// the layout of hipblaslt_w4a16_gemm. Tries one grouped GEMM over the active
// experts, else one dense GEMM per expert. Returns whether grouped ran.
bool hipblaslt_w4a16_moe_gemm(torch::Tensor& out, const torch::Tensor& x,
                              const torch::Tensor& w,
                              const torch::Tensor& scale,
                              const torch::Tensor& counts, int64_t group_size,
                              bool has_zp) {
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && w.dim() == 3);
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous(),
              "x and out must be contiguous");
  TORCH_CHECK(w.scalar_type() == at::kChar && w.stride(2) == 1,
              "packed weights must be int8 with contiguous rows");
  TORCH_CHECK(scale.dim() == 2 && scale.size(0) == w.size(0));
  TORCH_CHECK(counts.device().is_cpu() && counts.scalar_type() == at::kLong &&
                  counts.numel() == w.size(0),
              "counts must be a CPU int64 tensor with one entry per expert");
  const int64_t N = out.size(1), K = x.size(1);
  TORCH_CHECK(w.size(1) == N && w.size(2) * 2 == K, "w must be [E, N, K/2]");
  TORCH_CHECK(group_size > 0 && K % group_size == 0);
  const int64_t lda = w.stride(1) * 2;

  const int64_t* c = counts.const_data_ptr<int64_t>();
  std::vector<int64_t> experts, offsets, ns;
  int64_t off = 0;
  for (int64_t e = 0; e < counts.numel(); ++e) {
    if (c[e] > 0) {
      experts.push_back(e);
      offsets.push_back(off);
      ns.push_back(c[e]);
    }
    off += c[e];
  }
  TORCH_CHECK(off == x.size(0) && off == out.size(0), "counts sum to ", off,
              ", x has ", x.size(0), " rows");
  if (experts.empty()) return false;

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  if (run_grouped(out, x, w, scale, experts, offsets, ns, group_size, has_zp,
                  lda))
    return true;

  for (size_t i = 0; i < experts.size(); ++i) {
    const void* scale_ptr = scale[experts[i]].const_data_ptr();
    Plan& plan = get_plan(x.scalar_type(), ns[i], N, K, lda, group_size, has_zp,
                          scale_ptr);
    run_plan(plan, w[experts[i]].const_data_ptr(),
             static_cast<const char*>(x.const_data_ptr()) +
                 offsets[i] * K * x.element_size(),
             static_cast<char*>(out.data_ptr()) +
                 offsets[i] * N * out.element_size(),
             ns[i], scale_ptr, x);
  }
  return false;
}
