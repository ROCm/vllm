#define SKINNY_GEMMS_MAIN_TU
#include "skinny_gemms/kernel.cuh"
#include "skinny_gemms/launch.h"
#include "quantization/w8a8/fp8/common.cuh"
#include "core/batch_invariant.hpp"

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount) {
  auto M_in = in_a.size(0);
  auto K_in = in_a.size(1);
  auto N_in = in_b.size(0);
  auto Kap_in = in_a.stride(0);
  auto Kbp_in = in_b.stride(0);
  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(in_a.dtype() == in_b.dtype());
  TORCH_CHECK(K_in % 8 == 0, "k % 8 == 0");
  TORCH_CHECK(in_a.dtype() == torch::kFloat16 ||
              in_a.dtype() == torch::kBFloat16);

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int max_lds_len = get_lds_size() / 2;

  // Dispatch macros removed — dispatch logic moved to per-N shards in
  // skinny_gemms/dispatch.cuh.  The fused variants below still use the
  // kernels directly (N=1 only, not split).

// Kept for fused_silu_mul and fused_silu_gate_mul (N=1 only, not split):
#define WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, _YTILE, _UNRL, _N, _AC)             \
  {                                                                           \
    dim3 block(_THRDS, _WVPRGRP);                                             \
    int __wvPrGrp = mindiv(M_in, CuCount * _YTILE, _WVPRGRP);                 \
    if ((Kbp_in * N_in <= max_lds_len) && (M_in % _YTILE == 0))               \
      wvSplitK_hf_sml_<fptype, _THRDS, _YTILE, _WVPRGRP, _AC, _UNRL, _N>      \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,     \
                                       By_in, af4, bf4, biasf4, c, __wvPrGrp, \
                                       CuCount);                              \
    else if (Kbp_in * N_in <= max_lds_len * 1.2)                              \
      wvSplitK_hf_<fptype, _THRDS, _YTILE, _WVPRGRP, _AC, _UNRL, _N>          \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,     \
                                       By_in, af4, bf4, biasf4, c, __wvPrGrp, \
                                       CuCount);                              \
    else                                                                      \
      wvSplitK_hf_big_<fptype, _THRDS, _YTILE, _WVPRGRP, _AC, _UNRL, _N>      \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,     \
                                       By_in, af4, bf4, biasf4, c, __wvPrGrp, \
                                       CuCount);                              \
  }

#define WVSPLITK_CFG(_THRDS, _WVPRGRP, _YTILE, _UNRL, _N) \
  WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, _YTILE, _UNRL, _N, 8)

#define WVSPLIT_TILE_CFG(_THRDS, _WVPRGRP, _sYT, __N)                        \
  {                                                                          \
    bool fit_lds = (Kbp_in * N_in <= max_lds_len);                           \
    if (is_gfx11()) {                                                        \
      if (_sYT <= 1)                                                         \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 1, 4, __N)                            \
      else if (K_in < 1024)                                                  \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 2, 4, __N)                            \
      else if ((K_in % 1024 == 512) && (_sYT >= 40 || K_in >= 4096))         \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 4, 1, __N)                            \
      else if ((K_in == 2048) && (__N == 1))                                 \
        /* Tuned for gfx1151 (Qwen3.5 decode shapes M ∈ {256, 1024,        \
           248320}, K=2048, N=1): beats the (AC=8, W=16, UN=4) baseline of   \
           the K_in<=2048 branch below by 1.31-1.37x on small/mid M and 3.9% \
           on the lm_head-sized M.  Compiles to 145 VGPRs / occupancy 9 (vs  \
           46 / 8 for the default); zero spills.  VGPR-bound but DRAM-       \
           saturated past 92% of LPDDR5X peak after subtracting the per-     \
           launch dispatch floor.  Verify per shape with                     \
           benchmarks/kernels/sweep_bf16_kernel.py. */                       \
        WVSPLITK_CFG_AC(32, 32, 1, 8, __N, 16)                               \
      /* gfx1151 AC=32 fast paths.  Each cell beats AC=16 by >=2% with       \
         z>1.96 in a 10-rep do_bench A/B (stderr 0.1-1.0us per cell, mean    \
         delta 1.5-3 us per cell).  Other K%2048==0 cells stay on the AC=16  \
         fallbacks below where AC=32 was a tie or lost (notably 4096x4096    \
         N=4 was -2.7%, do not extrapolate to untested cells).  Re-verify    \
         per shape with benchmarks/kernels/sweep_bf16_kernel.py (extend the  \
         ACHUNKS list to include 32 and rebuild with                         \
         VLLM_SKINNY_GEMM_SWEEP_BF16=1). */                                  \
      else if ((K_in == 2048) && (__N == 2 || __N == 3))                     \
        /* M=2560 K=2048 N=2: 1.057x (z=21.5); N=3: 1.049x (z=12.8) */       \
        WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, 1, 2, __N, 32)                     \
      else if ((K_in == 4096) && (__N == 1))                                 \
        /* M=2560 K=4096 N=1: 1.041x (z=3.7); UR=4 not 2 for N=1 */          \
        WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, 1, 4, __N, 32)                     \
      else if ((K_in == 4096) && (__N == 2) && (M_in < 4096))                \
        /* M<4096 K=4096 N=2: 1.057x (z=13.1), W=16 wins at this M */        \
        WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, 1, 2, __N, 32)                     \
      else if ((K_in == 4096) && (__N == 2) && (M_in >= 4096))               \
        /* M>=4096 K=4096 N=2: 1.028x (z=6.2), W=32 wins at larger M */      \
        WVSPLITK_CFG_AC(_THRDS, 32, 1, 2, __N, 32)                           \
      else if ((K_in == 4096) && (__N == 3))                                 \
        /* M=2560 K=4096 N=3: 1.031x (z=4.9) */                              \
        WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, 1, 2, __N, 32)                     \
      else if ((K_in == 8192) && (__N == 2))                                 \
        /* M=2560 K=8192 N=2: 1.040x (z=9.3), W=32 wins at this K */         \
        WVSPLITK_CFG_AC(_THRDS, 32, 1, 2, __N, 32)                           \
      else if ((K_in % 2048 == 0) && (__N == 2))                             \
        /* gfx1151 K%2048==0, N=2 only: YT=2 + W=32 + AC=16 + UR=4.          \
           sweep_bf16_kernel.py 4-axis sweep showed this is the best         \
           N=2 config across K in {2048, 4096, 8192} and 4096x4096,          \
           1.06x (K=8192) to 1.60x (K=2048) over the prior AC=8 default. */  \
        WVSPLITK_CFG_AC(_THRDS, 32, 2, 4, __N, 16)                           \
      else if ((K_in % 2048 == 0) && (__N != 2))                             \
        /* gfx1151 K%2048==0, N in {1, 3, 4} (K=2048 N=1 handled above):     \
           YT=1 + W=16 + AC=16 + UR=4.  N=3/4 want YT=1 not YT=2 (LDS/VGPR   \
           pressure from W=32 hurts them); same config also wins for N=1.    \
           sweep showed 1.11x-1.19x (N=1), 1.23x-1.98x (N=3),                \
           1.59x-2.73x (N=4) over the prior AC=8 defaults. */                \
        WVSPLITK_CFG_AC(_THRDS, _WVPRGRP, 1, 4, __N, 16)                     \
      else if (K_in <= 2048)                                                 \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 1, 4, __N)                            \
      else if (__N >= 2 && !fit_lds) {                                       \
        if (K_in % 1024 == 0 && Kbp_in < max_lds_len / 2)                    \
          WVSPLITK_CFG(_THRDS, _WVPRGRP, 2, 4, __N)                          \
        else                                                                 \
          WVSPLITK_CFG(_THRDS, _WVPRGRP, 1, 4, __N)                          \
      } else if (__N == 1)                                                   \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 1, 2, __N)                            \
      else                                                                   \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 1, 1, __N)                            \
    } else {                                                                 \
      if (_sYT <= 1)                                                         \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 1, 4, __N)                            \
      else if ((__N == 1) || (!fit_lds) || (_sYT <= 4 * 2))                  \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 2, 2, __N)                            \
      else if (_sYT <= 4 * 3)                                                \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 3, 2, __N)                            \
      else if (__N == 4)                                                     \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 4, 1, __N)                            \
      else                                                                   \
        WVSPLITK_CFG(_THRDS, _WVPRGRP, 4, 2, __N)                            \
    }                                                                        \
  }

#define WVSPLIT_TILE(_sYT, __N)                                      \
  {                                                                  \
    bool fit_lds = (Kbp_in * N_in <= max_lds_len);                   \
    if (is_gfx11()) {                                                \
      if (_sYT <= 1)                                                 \
        WVSPLITK_CFG(32, 16, 1, 4, __N)                              \
      else if (K_in < 1024)                                          \
        WVSPLITK_CFG(32, 16, 2, 4, __N)                              \
      else if ((K_in % 1024 == 512) && (_sYT >= 40 || K_in >= 4096)) \
        WVSPLITK_CFG(32, 16, 4, 1, __N)                              \
      else if (K_in <= 2048 && (__N >= 2 || _sYT <= 26))             \
        WVSPLITK_CFG(32, 16, 1, 4, __N)                              \
      else if (__N >= 2 && !fit_lds)                                 \
        WVSPLITK_CFG(32, 16, 1, 4, __N)                              \
      else if (__N == 1)                                             \
        WVSPLITK_CFG(32, 16, 1, 2, __N)                              \
      else                                                           \
        WVSPLITK_CFG(32, 16, 1, 1, __N)                              \
    } else {                                                         \
      if (_sYT <= 1)                                                 \
        WVSPLITK_CFG(64, 16, 1, 4, __N)                              \
      else if ((__N == 1) || (!fit_lds) || (_sYT <= 4 * 2))          \
        WVSPLITK_CFG(64, 16, 2, 2, __N)                              \
      else if (_sYT <= 4 * 3)                                        \
        WVSPLITK_CFG(64, 16, 3, 2, __N)                              \
      else if (__N == 4)                                             \
        WVSPLITK_CFG(64, 16, 4, 1, __N)                              \
      else                                                           \
        WVSPLITK_CFG(64, 16, 4, 2, __N)                              \
    }                                                                \
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(in_b.scalar_type(), "wvSplitK", [&] {
    using fptype = typename scalar<scalar_t>::type;
    fptype* af4 = reinterpret_cast<fptype*>(in_a.data_ptr());
    const fptype* bf4 = reinterpret_cast<const fptype*>(in_b.data_ptr());
    const fptype* biasf4 =
        (in_bias.has_value() && in_bias->numel() > 0)
            ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
            : nullptr;
    fptype* c = reinterpret_cast<fptype*>(out_c.data_ptr());

    switch (N_in) {
      case 1:
        launch_wvsplitk_n1(grid, stream, K_in, Kap_in, Kbp_in, M_in, Bx_in,
                           By_in, af4, bf4, biasf4, c, CuCount, max_lds_len);
        break;
      case 2:
        launch_wvsplitk_n2(grid, stream, K_in, Kap_in, Kbp_in, M_in, Bx_in,
                           By_in, af4, bf4, biasf4, c, CuCount, max_lds_len);
        break;
      case 3:
        launch_wvsplitk_n3(grid, stream, K_in, Kap_in, Kbp_in, M_in, Bx_in,
                           By_in, af4, bf4, biasf4, c, CuCount, max_lds_len);
        break;
      case 4:
        launch_wvsplitk_n4(grid, stream, K_in, Kap_in, Kbp_in, M_in, Bx_in,
                           By_in, af4, bf4, biasf4, c, CuCount, max_lds_len);
        break;
      case 5:
        launch_wvsplitk_n5(grid, stream, K_in, Kap_in, Kbp_in, M_in, Bx_in,
                           By_in, af4, bf4, biasf4, c, CuCount, max_lds_len);
        break;
      default:
        throw std::runtime_error(
            "Unsupported N value: " + std::to_string(M_in) + "," +
            std::to_string(K_in) + "," + std::to_string(N_in));
    }
  });
  return out_c;
}

// META3-2: bf16/fp16 wvSplitK with a FUSED_SILU_MUL preamble.
//
// in_a: weight   [M, K]   (bf16/fp16)         (wvSplitK convention: in_a)
// in_b: activation [N=1, 2*K]  packed [gate(K) | up(K)] per row
// in_bias: optional bias
// out:  [N=1, M]
//
// Mirrors the int4 EXPERIMENT-g `fuse_silu_mul` flag but specialized to a
// single-call entry point so we don't have to retrofit the existing
// wvSplitK dispatch macros (which cover N=1..5 and 3 LDS-residency
// regimes).  Only the N=1 sml path is implemented; the caller (Python)
// is responsible for falling back to the unfused wvSplitK + silu_and_mul
// when preconditions don't hold.
//
// Note on parameter naming: the wvSplitK kernel's C-side argument names
// are confusingly cross-mapped vs. the wrapper's local variable names.
// Inside the kernel, `B` is the weight (= in_a here) and `A` is the
// activation (= in_b).  We mirror the original wvSplitK wrapper's call
// pattern so the kernel sees (B=af4=weight, A=bf4=activation) exactly as
// it does in the unfused path.
torch::Tensor wvSplitK_fused_silu_mul(const at::Tensor& in_a,
                                      const at::Tensor& in_b,
                                      const std::optional<at::Tensor>& in_bias,
                                      const int64_t CuCount) {
  auto M_in = in_a.size(0);      // weight output dim
  auto K_in = in_a.size(1);      // weight K (compute K)
  auto N_in = in_b.size(0);      // batch (must be 1)
  auto two_K = in_b.size(1);     // activation's last dim (must be 2*K)
  auto Kap_in = in_a.stride(0);  // weight row stride (passed as kernel's
                                 // Kbp); = K_in normally
  auto Kbp_in = in_b.stride(0);  // activation row stride (passed as
                                 // kernel's Kap); = 2*K_in normally

  TORCH_CHECK(in_a.dtype() == in_b.dtype(),
              "wvSplitK_fused_silu_mul: dtype mismatch");
  TORCH_CHECK(
      in_a.dtype() == torch::kFloat16 || in_a.dtype() == torch::kBFloat16,
      "wvSplitK_fused_silu_mul: only fp16/bf16 supported");
  TORCH_CHECK(K_in % 8 == 0,
              "wvSplitK_fused_silu_mul: K must be multiple of 8");
  TORCH_CHECK(N_in == 1, "wvSplitK_fused_silu_mul: only N=1 supported, got ",
              N_in);
  TORCH_CHECK(two_K == 2 * K_in,
              "wvSplitK_fused_silu_mul: in_b.size(-1) must equal 2*K=",
              2 * K_in, ", got ", two_K);

  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int max_lds_len = get_lds_size() / 2;

  // sml-path precondition: post-silu_mul K elements per row must fit LDS.
  // (The fused load reads 2*K from global but only K go into LDS.)
  TORCH_CHECK(K_in * N_in <= max_lds_len,
              "wvSplitK_fused_silu_mul: K*N must fit LDS (K=", K_in,
              ", N=", N_in, ", max_lds_len=", max_lds_len, ")");

  dim3 grid(CuCount);

  // Pick a small fixed config that matches what the unfused wvSplitK
  // dispatcher chooses for the canonical Qwen3.5-MoE shared_expert
  // down-projection (N=1, K=512, M=2048 on gfx1151): per the
  // WVSPLIT_TILE_CFG macro for K_in <= 2048 case, this lands at
  // THRDS=32, WvPrGrp=16, YTILE=1, UNRL=4 on wave32 (gfx1x) and
  // THRDS=64, WvPrGrp=16, YTILE=2, UNRL=2 on wave64 (gfx9).
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      in_b.scalar_type(), "wvSplitK_fused_silu_mul", [&] {
        using fptype = typename scalar<scalar_t>::type;
        fptype* af4 = reinterpret_cast<fptype*>(in_a.data_ptr());
        const fptype* bf4 = reinterpret_cast<const fptype*>(in_b.data_ptr());
        const fptype* biasf4 =
            (in_bias.has_value() && in_bias->numel() > 0)
                ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
                : nullptr;
        fptype* c = reinterpret_cast<fptype*>(out_c.data_ptr());

        const bool use_wave32 = on_gfx1x();

        // Mirror the unfused wvSplitK launch site exactly, then flip the
        // FUSED_SILU_MUL template parameter to true.  Argument order is the
        // documented (K, Kap_in, Kbp_in, M, ..., af4=weight, bf4=activation)
        // -- same swap that wvSplitK does under the hood.  Inlined (not
        // macroized) because hipify mishandles the line-continuation '\'
        // in macro bodies that contain literal template args like 'true'.
        if (use_wave32) {
          // gfx11/12 (wave32): YTILE=1, UNRL=4 matches the K_in <= 2048,
          // N=1 branch of WVSPLIT_TILE_CFG -- same per-CU tile shape as
          // the unfused down_proj kernel for this model.
          constexpr int _THRDS = 32, _WVPRGRP = 16, _YTILE = 1, _UNRL = 4;
          dim3 block(_THRDS, _WVPRGRP);
          int __wvPrGrp = mindiv(M_in, CuCount * _YTILE, _WVPRGRP);
          wvSplitK_hf_sml_<fptype, _THRDS, _YTILE, _WVPRGRP, 8, _UNRL, 1, true>
              <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,
                                           By_in, af4, bf4, biasf4, c,
                                           __wvPrGrp, CuCount);
        } else {
          // wave64 (gfx9): YTILE=2, UNRL=2 mirrors the (__N==1) clause of
          // WVSPLIT_TILE_CFG.
          constexpr int _THRDS = 64, _WVPRGRP = 16, _YTILE = 2, _UNRL = 2;
          dim3 block(_THRDS, _WVPRGRP);
          int __wvPrGrp = mindiv(M_in, CuCount * _YTILE, _WVPRGRP);
          wvSplitK_hf_sml_<fptype, _THRDS, _YTILE, _WVPRGRP, 8, _UNRL, 1, true>
              <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,
                                           By_in, af4, bf4, biasf4, c,
                                           __wvPrGrp, CuCount);
        }
      });
  return out_c;
}

// META3-2 Phase 2: bf16/fp16 wvSplitK with both FUSED_SILU_MUL preamble
// AND a per-token FUSED_GATE_MUL epilogue.
//
// Same shape conventions as `wvSplitK_fused_silu_mul`, plus:
//   in_gate: [N, 1] (or [N]) per-token scalar weight that pre-multiplies
//            the GEMM output before write-back to C. Same dtype as in_b.
//
// Used by Qwen2MoeMLP.forward when expert_gate is set: collapses
// silu_and_mul + down_proj + (sigmoid(expert_gate(x)) * out) into one
// kernel.  Output is [N=1, M].
torch::Tensor wvSplitK_fused_silu_gate_mul(
    const at::Tensor& in_a, const at::Tensor& in_b, const at::Tensor& in_gate,
    const std::optional<at::Tensor>& in_bias, const int64_t CuCount) {
  auto M_in = in_a.size(0);      // weight output dim
  auto K_in = in_a.size(1);      // weight K (compute K)
  auto N_in = in_b.size(0);      // batch (must be 1)
  auto two_K = in_b.size(1);     // activation's last dim (must be 2*K)
  auto Kap_in = in_a.stride(0);  // weight row stride (kernel's Kbp)
  auto Kbp_in = in_b.stride(0);  // activation row stride (kernel's Kap)

  TORCH_CHECK(in_a.dtype() == in_b.dtype(),
              "wvSplitK_fused_silu_gate_mul: dtype mismatch a vs b");
  TORCH_CHECK(in_gate.dtype() == in_b.dtype(),
              "wvSplitK_fused_silu_gate_mul: dtype mismatch gate vs b");
  TORCH_CHECK(
      in_a.dtype() == torch::kFloat16 || in_a.dtype() == torch::kBFloat16,
      "wvSplitK_fused_silu_gate_mul: only fp16/bf16 supported");
  TORCH_CHECK(K_in % 8 == 0,
              "wvSplitK_fused_silu_gate_mul: K must be multiple of 8");
  TORCH_CHECK(N_in == 1,
              "wvSplitK_fused_silu_gate_mul: only N=1 supported, got ", N_in);
  TORCH_CHECK(two_K == 2 * K_in,
              "wvSplitK_fused_silu_gate_mul: in_b.size(-1) must equal 2*K=",
              2 * K_in, ", got ", two_K);
  TORCH_CHECK(in_gate.numel() == N_in,
              "wvSplitK_fused_silu_gate_mul: in_gate.numel() must equal N=",
              N_in, ", got ", in_gate.numel());

  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int max_lds_len = get_lds_size() / 2;

  TORCH_CHECK(K_in * N_in <= max_lds_len,
              "wvSplitK_fused_silu_gate_mul: K*N must fit LDS (K=", K_in,
              ", N=", N_in, ", max_lds_len=", max_lds_len, ")");

  dim3 grid(CuCount);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      in_b.scalar_type(), "wvSplitK_fused_silu_gate_mul", [&] {
        using fptype = typename scalar<scalar_t>::type;
        fptype* af4 = reinterpret_cast<fptype*>(in_a.data_ptr());
        const fptype* bf4 = reinterpret_cast<const fptype*>(in_b.data_ptr());
        const fptype* gatef4 =
            reinterpret_cast<const fptype*>(in_gate.data_ptr());
        const fptype* biasf4 =
            (in_bias.has_value() && in_bias->numel() > 0)
                ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
                : nullptr;
        fptype* c = reinterpret_cast<fptype*>(out_c.data_ptr());

        const bool use_wave32 = on_gfx1x();

        // Same tile config as wvSplitK_fused_silu_mul; only the trailing
        // FUSED_GATE_MUL=true template flag and the GATE pointer differ.
        if (use_wave32) {
          constexpr int _THRDS = 32, _WVPRGRP = 16, _YTILE = 1, _UNRL = 4;
          dim3 block(_THRDS, _WVPRGRP);
          int __wvPrGrp = mindiv(M_in, CuCount * _YTILE, _WVPRGRP);
          wvSplitK_hf_sml_<fptype, _THRDS, _YTILE, _WVPRGRP, 8, _UNRL, 1, true,
                           true><<<grid, block, 0, stream>>>(
              K_in, Kap_in, Kbp_in, M_in, Bx_in, By_in, af4, bf4, biasf4, c,
              __wvPrGrp, CuCount, gatef4);
        } else {
          constexpr int _THRDS = 64, _WVPRGRP = 16, _YTILE = 2, _UNRL = 2;
          dim3 block(_THRDS, _WVPRGRP);
          int __wvPrGrp = mindiv(M_in, CuCount * _YTILE, _WVPRGRP);
          wvSplitK_hf_sml_<fptype, _THRDS, _YTILE, _WVPRGRP, 8, _UNRL, 1, true,
                           true><<<grid, block, 0, stream>>>(
              K_in, Kap_in, Kbp_in, M_in, Bx_in, By_in, af4, bf4, biasf4, c,
              __wvPrGrp, CuCount, gatef4);
        }
      });
  return out_c;
}

// Sweep function disabled by default to reduce compile time.
// Build with -DVLLM_SKINNY_GEMM_SWEEP_BF16 (or the umbrella
// -DVLLM_SKINNY_GEMM_SWEEP) to enable.  Compared to the dispatcher's
// (YTILE, UNRL) heuristic this lets the caller also pick A_CHUNK and
// WvPrGrp at runtime, which is necessary to verify whether the (W=32,
// AC=16) trick used by the K=2048 N=1 fast path also recovers other
// K%2048 == 0 shapes that currently route through (W=16, AC=8).
// The YTILE grid is restricted to {1, 2} -- the production dispatcher
// never picks YTILE > 2 for the slow K%2048 shapes (YT=1 for N=1,
// YT=2 for N>=2 + !fit_lds) -- which keeps the template-instantiation
// count to 24 combos x 4 N x 3 kernel variants = 288.
#ifdef VLLM_SKINNY_GEMM_SWEEP_BF16
torch::Tensor wvSplitK_sweep(const at::Tensor& in_a, const at::Tensor& in_b,
                             const std::optional<at::Tensor>& in_bias,
                             const int64_t CuCount, const int64_t ytile,
                             const int64_t unrl, const int64_t achunk,
                             const int64_t wvprgrp) {
  auto M_in = in_a.size(0);
  auto K_in = in_a.size(1);
  auto N_in = in_b.size(0);
  auto Kap_in = in_a.stride(0);
  auto Kbp_in = in_b.stride(0);
  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(in_a.dtype() == in_b.dtype());
  TORCH_CHECK(K_in % 8 == 0, "k % 8 == 0");
  TORCH_CHECK(in_a.dtype() == torch::kFloat16 ||
              in_a.dtype() == torch::kBFloat16);
  TORCH_CHECK(M_in % ytile == 0, "M must be divisible by ytile=", ytile);

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int max_lds_len = get_lds_size() / 2;

  #define WVSPLITK_SWEEP_LAUNCH(_THRDS, _YTILE, _UNRL, _N, _AC, _WVPRGRP)   \
    {                                                                       \
      dim3 block(_THRDS, _WVPRGRP);                                         \
      int __wvPrGrp = mindiv(M_in, CuCount * _YTILE, _WVPRGRP);             \
      if ((Kbp_in * N_in <= max_lds_len) && (M_in % _YTILE == 0))           \
        wvSplitK_hf_sml_<fptype, _THRDS, _YTILE, _WVPRGRP, _AC, _UNRL, _N>  \
            <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in, \
                                         By_in, af4, bf4, biasf4, c,        \
                                         __wvPrGrp, CuCount);               \
      else if (Kbp_in * N_in <= max_lds_len * 1.2)                          \
        wvSplitK_hf_<fptype, _THRDS, _YTILE, _WVPRGRP, _AC, _UNRL, _N>      \
            <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in, \
                                         By_in, af4, bf4, biasf4, c,        \
                                         __wvPrGrp, CuCount);               \
      else                                                                  \
        wvSplitK_hf_big_<fptype, _THRDS, _YTILE, _WVPRGRP, _AC, _UNRL, _N>  \
            <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in, \
                                         By_in, af4, bf4, biasf4, c,        \
                                         __wvPrGrp, CuCount);               \
    }

  #define WVSPLITK_SWEEP_N(_THRDS, _YTILE, _UNRL, _AC, _WVPRGRP)              \
    switch (N_in) {                                                           \
      case 1:                                                                 \
        WVSPLITK_SWEEP_LAUNCH(_THRDS, _YTILE, _UNRL, 1, _AC, _WVPRGRP) break; \
      case 2:                                                                 \
        WVSPLITK_SWEEP_LAUNCH(_THRDS, _YTILE, _UNRL, 2, _AC, _WVPRGRP) break; \
      case 3:                                                                 \
        WVSPLITK_SWEEP_LAUNCH(_THRDS, _YTILE, _UNRL, 3, _AC, _WVPRGRP) break; \
      case 4:                                                                 \
        WVSPLITK_SWEEP_LAUNCH(_THRDS, _YTILE, _UNRL, 4, _AC, _WVPRGRP) break; \
      default:                                                                \
        TORCH_CHECK(false, "Unsupported N=", N_in);                           \
    }

  #define WVSPLITK_SWEEP_WVPRGRP(_THRDS, _YTILE, _UNRL, _AC) \
    if (wvprgrp == 16) {                                     \
      WVSPLITK_SWEEP_N(_THRDS, _YTILE, _UNRL, _AC, 16)       \
    } else if (wvprgrp == 32) {                              \
      WVSPLITK_SWEEP_N(_THRDS, _YTILE, _UNRL, _AC, 32)       \
    } else {                                                 \
      TORCH_CHECK(false, "Unsupported wvprgrp=", wvprgrp,    \
                  "; allowed: 16, 32");                      \
    }

  #define WVSPLITK_SWEEP_AC(_THRDS, _YTILE, _UNRL)                           \
    if (achunk == 8) {                                                       \
      WVSPLITK_SWEEP_WVPRGRP(_THRDS, _YTILE, _UNRL, 8)                       \
    } else if (achunk == 16) {                                               \
      WVSPLITK_SWEEP_WVPRGRP(_THRDS, _YTILE, _UNRL, 16)                      \
    } else {                                                                 \
      TORCH_CHECK(false, "Unsupported achunk=", achunk, "; allowed: 8, 16"); \
    }

  #define WVSPLITK_SWEEP_UNRL(_THRDS, _YTILE)        \
    if (unrl == 1) {                                 \
      WVSPLITK_SWEEP_AC(_THRDS, _YTILE, 1)           \
    } else if (unrl == 2) {                          \
      WVSPLITK_SWEEP_AC(_THRDS, _YTILE, 2)           \
    } else if (unrl == 4) {                          \
      WVSPLITK_SWEEP_AC(_THRDS, _YTILE, 4)           \
    } else {                                         \
      TORCH_CHECK(false, "Unsupported unrl=", unrl); \
    }

  #define WVSPLITK_SWEEP_YTILE(_THRDS)                                    \
    if (ytile == 1) {                                                     \
      WVSPLITK_SWEEP_UNRL(_THRDS, 1)                                      \
    } else if (ytile == 2) {                                              \
      WVSPLITK_SWEEP_UNRL(_THRDS, 2)                                      \
    } else {                                                              \
      TORCH_CHECK(false, "Unsupported ytile=", ytile, "; allowed: 1, 2"); \
    }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(in_b.scalar_type(), "wvSplitK_sweep", [&] {
    using fptype = typename scalar<scalar_t>::type;
    fptype* af4 = reinterpret_cast<fptype*>(in_a.data_ptr());
    const fptype* bf4 = reinterpret_cast<const fptype*>(in_b.data_ptr());
    const fptype* biasf4 =
        (in_bias.has_value() && in_bias->numel() > 0)
            ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
            : nullptr;
    fptype* c = reinterpret_cast<fptype*>(out_c.data_ptr());

    if (is_gfx11()) {
      WVSPLITK_SWEEP_YTILE(32)
    } else {
      WVSPLITK_SWEEP_YTILE(64)
    }
  });

  #undef WVSPLITK_SWEEP_LAUNCH
  #undef WVSPLITK_SWEEP_N
  #undef WVSPLITK_SWEEP_WVPRGRP
  #undef WVSPLITK_SWEEP_AC
  #undef WVSPLITK_SWEEP_UNRL
  #undef WVSPLITK_SWEEP_YTILE

  return out_c;
}
#endif  // VLLM_SKINNY_GEMM_SWEEP_BF16

// This version targets cases skinny where CUs are not filled
// Wave-SplitK is used with reduction done via atomics.
#if defined(__gfx950__)
  #define WVSPLITKRC_1KPASS
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, int GrpsShrB, int CHUNKK, int DTRMNSTC>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    __attribute__((amdgpu_waves_per_eu(1, 1)))
    wvSplitKrc_(const int actlN, const int K, const int Kap, const int M,
                const int Bx, const int By, const scalar_t* __restrict__ A,
                const scalar_t* __restrict__ B,
                const scalar_t* __restrict__ BIAS, float* glbl, int* cntr,
                scalar_t* C, const int CuCount) {
  constexpr int NTILE = 16;
  constexpr int APAD = 1;
  constexpr int ASTRD = 64;
  constexpr int BPAD = 1;
  constexpr int WVLDS_ = THRDS * A_CHUNK / CHUNKK;
  constexpr int WVLDS = ((WVLDS_ + A_CHUNK * BPAD)) * YTILE;

  constexpr int max_lds_len = LDS_SIZE / 2;

  using scalar16 =
      __attribute__((__vector_size__((A_CHUNK * 2) * sizeof(float)))) float;
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  using half4 = __attribute__((__vector_size__(4 * sizeof(__bf16)))) __bf16;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    unsigned int i[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    unsigned long l[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half4 h4[A_CHUNK / 4];
    scalar8 h8;
  };
  using big4 = __attribute__((__vector_size__(4 * sizeof(bigType)))) __bf16;

  __shared__ scalar_t stg[WvPrGrp * WVLDS / GrpsShrB];
  unsigned int* myStg = (unsigned int*)(&stg[WVLDS * (threadIdx.y / GrpsShrB)]);
  __shared__ scalar_t s[max_lds_len - WvPrGrp * WVLDS / GrpsShrB];

  #ifndef WVSPLITKRC_1KPASS
  constexpr int TUC_ = (THRDS * UNRL * A_CHUNK);
  // find biggest k size that fits padded into LDS
  constexpr uint32_t kFit__ = (max_lds_len - WvPrGrp * WVLDS / GrpsShrB) / N;
  constexpr uint32_t kFit_ = (kFit__ * ASTRD) / (APAD + ASTRD);
  uint32_t kFit = kFit_ - (kFit_ % TUC_);
  uint32_t kfitsPerRdc = (K + kFit - 1) / kFit;

  // find best k split to fill the CUs
  if (((K + kfitsPerRdc * kFit - 1) / (kfitsPerRdc * kFit)) * numCuWithFullK <=
      CuCount)
    while (true) {
      while (kFit > TUC_) {
        uint32_t kFit_ = kFit - TUC_;
        if (((K + (kfitsPerRdc * kFit_ - 1)) / (kfitsPerRdc * kFit_)) *
                numCuWithFullK >
            CuCount)
          break;
        kFit = kFit_;
      }
      if (((K + ((kfitsPerRdc - 1) * kFit - 1)) / ((kfitsPerRdc - 1) * kFit)) *
              numCuWithFullK <=
          CuCount)
        kfitsPerRdc--;
      else
        break;
    }
  #else
  int constexpr kFit = 512 / CHUNKK;
  int constexpr kfitsPerRdc = 1;
  #endif

  bool doRdc = true;  // Assuming (kfitsPerRdc * kFit < K) is always true
  uint32_t numCuWithFullK =
      ((M + (WvPrGrp * YTILE / GrpsShrB) - 1) / (WvPrGrp * YTILE / GrpsShrB));
  uint32_t Mmod = numCuWithFullK * (WvPrGrp * YTILE / GrpsShrB);

  // given above k-split, find this wave's position
  uint32_t kFitPdd = kFit * CHUNKK + ((kFit * CHUNKK) / ASTRD) * APAD;
  uint32_t m0 = (blockIdx.x * WvPrGrp / GrpsShrB) * YTILE;
  uint32_t m1 = ((threadIdx.y % WvPrGrp) / GrpsShrB) * YTILE;
  uint32_t m = (m0 + m1) % Mmod;
  const uint32_t k_str = (m0 / Mmod) * kFit * kfitsPerRdc;
  uint32_t k_end = (m0 / Mmod + 1) * kFit * kfitsPerRdc;
  const uint32_t k_rnd = (K + kFit * kfitsPerRdc - 1) / (kFit * kfitsPerRdc);

  scalar8 sum4[N / NTILE / GrpsShrB][1] = {0};
  bigType bigB_[YTILE / GrpsShrB / CHUNKK][UNRL];
  const uint32_t bLoader = (threadIdx.y % GrpsShrB);
  uint32_t kBase = 0;
  if (k_str >= K) return;
  if (m >= Mmod) return;

  bool noreloada = false;
  constexpr bool FAST_UNSAFE_RDC_INIT = false;

  #ifdef WVSPLITKRC_1KPASS
  // Early glbl init, B[] loading, if 1KPASS
  if constexpr (FAST_UNSAFE_RDC_INIT) {
    if (m + (threadIdx.x % 16) < M)
      if (doRdc)
        if (k_str == 0) {
          int mindx = m + (threadIdx.x % 16);
          int nindx_ = (0 + (threadIdx.x / 16) * 4) + 0 * NTILE +
                       (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
          int adr_ = mindx + M * nindx_ / 4;
          __hip_atomic_store(&cntr[adr_], 0, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            for (uint32_t j = 0; j < 4; j++) {
              int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                          (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
              int adr = mindx + M * nindx;
              __hip_atomic_store(&glbl[adr], 0, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
            }
          }
        }
  }

    // Load first B[] chunk
    #pragma unroll
  for (uint32_t k2 = 0; k2 < UNRL; k2++) {
    uint32_t k = k_str + k2 * THRDS * A_CHUNK;
    uint32_t k_ = k + (threadIdx.x % (THRDS / CHUNKK)) * A_CHUNK;
    const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
    #pragma unroll
    for (uint32_t y = 0; y < YTILE / GrpsShrB; y += CHUNKK)
      bigB_[y / CHUNKK][k2].h8 = (loadnt(
          (scalar8*)(&B_[min__((y + threadIdx.x / (THRDS / CHUNKK)) * GrpsShrB +
                                   bLoader + m,
                               M - 1) *
                         K])));
  }
  {
  #else
  while (m < Mmod) {
  #endif

  #ifndef WVSPLITKRC_1KPASS
    if constexpr (FAST_UNSAFE_RDC_INIT) {
      if (m + (threadIdx.x % 16) < M)
        if (doRdc)
          if (k_str == 0) {
            int mindx = m + (threadIdx.x % 16);
            int nindx_ = (0 + (threadIdx.x / 16) * 4) + 0 * NTILE +
                         (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
            int adr_ = mindx + M * nindx_ / 4;
            __hip_atomic_store(&cntr[adr_], 0, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
            for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
              for (uint32_t j = 0; j < 4; j++) {
                int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                            (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
                int adr = mindx + M * nindx;
                __hip_atomic_store(&glbl[adr], 0, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
              }
            }
          }
    }

  #endif

  #ifndef WVSPLITKRC_1KPASS
    for (uint32_t k1 = k_str; k1 < k_end; k1 += THRDS * A_CHUNK * UNRL) {
  #else
    const uint32_t k1 = k_str;
    {
  #endif
  #ifndef WVSPLITKRC_1KPASS
      const bool reloada = (!noreloada) &&
                           ((k1 == k_str) || (k1 == k_str + kBase + kFit)) &&
                           (k1 < k_end);
      // load next chunk of A[] to LDS
      if (reloada) {
        if (k1 != k_str) kBase += kFit;
        __syncthreads();
  #else
      const bool reloada = (!noreloada) &&
                           ((k1 == k_str) || (k1 == k_str + kBase + kFit)) &&
                           (k1 < k_end);
      if (reloada) {
  #endif
        constexpr int sprdN = 4;
        const uint32_t thrd = threadIdx.x % (THRDS / CHUNKK);

  #ifndef WVSPLITKRC_1KPASS
    #pragma unroll
        for (int k = 0; k < kFit;
             k += (THRDS * (WvPrGrp / sprdN) * A_CHUNK) / CHUNKK) {
  #else
        const unsigned int k = 0;
        {
  #endif
          unsigned int kOff = k + (thrd * A_CHUNK);
          unsigned int kOffcp = min__(K - A_CHUNK, k_str + kOff);
          for (unsigned int n = 0; n < N; n += CHUNKK * sprdN) {
            __builtin_amdgcn_global_load_lds(
                (int*)(&A[min__(Kap * actlN - A_CHUNK,
                                kOffcp + Kap * (n / CHUNKK +
                                                (N / CHUNKK) * (threadIdx.x /
                                                                (64 / CHUNKK)) +
                                                (threadIdx.y % sprdN)))]),
                (int*)(&s[(k +
                           kFitPdd * ((n / CHUNKK) + (threadIdx.y % sprdN)))]),
                16, 0, 0);
          }

          // Stage loaded B[] to LDS for MFMA swizzling...
          for (uint32_t k2 = 0; k2 < UNRL; k2++) {
            uint32_t k = k1 + k2 * THRDS * A_CHUNK;
            uint32_t k_ = k + (threadIdx.x % (THRDS / CHUNKK)) * A_CHUNK;
            const bool oob_k = (k_ >= K);
            for (uint32_t y = 0; y < YTILE / GrpsShrB; y += CHUNKK) {
              uint32_t idx =
                  (threadIdx.x % (THRDS / CHUNKK)) * 4 +
                  ((y + threadIdx.x / (THRDS / CHUNKK)) * GrpsShrB + bLoader) *
                      ((THRDS / CHUNKK + BPAD) * 4);
              // zero out if oob
              *((scalar8*)&myStg[idx]) =
                  (oob_k)  // TODO: ever necessary (y*GrpsShrB+bLoader+m>=M) ?
                      ? 0
                      : bigB_[y / CHUNKK][k2].h8;
            }
          }
        }
      }
    }
  #ifndef WVSPLITKRC_1KPASS
    // Fire load of next B[] chunk...
    if ((k1 + THRDS * A_CHUNK * UNRL < k_end) &&
        (k1 + THRDS * A_CHUNK * UNRL < K))
    #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + THRDS * A_CHUNK * UNRL + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
    #pragma unroll
        for (uint32_t y = 0; y < YTILE / GrpsShrB; y += CHUNKK)
          bigB_[y / CHUNKK][k2].h8 = (loadnt(
              (scalar8*)(&B_[min__((y + threadIdx.x / (THRDS / CHUNKK)) *
                                           GrpsShrB +
                                       bLoader + m,
                                   M - 1) *
                             K])));
      }
  #endif

    // B[] staging is cooperative across GrpsShrB, so sync here before reading
    // back. This wait is currently inserted by compiler, but not guaranteed.
    asm volatile("s_waitcnt 0");
    __syncthreads();

    // read back B[] swizzled for MFMA...
    bigType bigB[YTILE / CHUNKK][UNRL];
    for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      for (uint32_t y = 0; y < YTILE / CHUNKK; y++) {
        unsigned int idx =
            (threadIdx.x % YTILE) * ((THRDS / CHUNKK + BPAD) * 4) +
            (threadIdx.x / YTILE) * 4 + y * 16;
        bigB[y][k2].h8 = *((scalar8*)&myStg[idx]);
      }
    }

    // rReadback A[] swizzled for MFMA...
    bigType bigA[N / GrpsShrB / CHUNKK][UNRL];
  #pragma unroll
    for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      uint32_t k = k1 + k2 * THRDS * A_CHUNK - kBase - k_str;
  #pragma unroll
      for (uint32_t nt = 0; nt < N / GrpsShrB; nt += NTILE)
  #pragma unroll
        for (uint32_t n = 0; n < NTILE / CHUNKK; n++) {
          uint32_t idxa =
              ((nt + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) % (N / CHUNKK) +
               (threadIdx.x % NTILE)) *
                  kFitPdd +
              ((nt + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) /
               (N / CHUNKK)) *
                  A_CHUNK * (64 / CHUNKK) +
              A_CHUNK * ((threadIdx.x / NTILE) + n * 4) + k;
          bigA[nt / CHUNKK + n][k2] = *((const bigType*)(&(s[idxa])));
        }
    }

    // Do the MFMAs
  #pragma unroll
    for (uint32_t k2 = 0; k2 < UNRL; k2++) {
  #pragma unroll
      for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
  #pragma unroll
        for (uint32_t j = 0; j < YTILE / CHUNKK; j++) {
          if constexpr (std::is_same_v<scalar_t, half>) {
            sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                bigA[nt * (YTILE / CHUNKK) + j][k2].h8, bigB[j][k2].h8,
                sum4[nt][0], 0, 0, 0);
          } else {  // bf16
            sum4[nt][0] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                bigA[nt * (YTILE / CHUNKK) + j][k2].h8, bigB[j][k2].h8,
                sum4[nt][0], 0, 0, 0);
          }
        }
      }
    }
  }

  union flt4 {
    scalar8 s8;
    float2 f2[2];
    float4 f4;
  };
  if (m + (threadIdx.x % 16) < M) {
    int my_cntr;
    int mindx = m + (threadIdx.x % 16);
    int g_mindx = m * 4 + (threadIdx.x % 64);  // coalesced atomic reduction
    scalar_t biases[N / NTILE / GrpsShrB][4] = {};
    // Atomic add the output, read biases
    for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
      int g_nindx =
          (nt * NTILE + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) / 4;
      int g_adr = g_mindx * 4 + 0 + M * g_nindx * 4;
      if (DTRMNSTC) {
        flt4 flt4_ = {.s8 = sum4[nt][0]};
        __hip_atomic_store((float2*)&glbl[g_adr + M * N * (m0 / Mmod)],
                           flt4_.f2[0], __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store((float2*)&glbl[g_adr + 2 + M * N * (m0 / Mmod)],
                           flt4_.f2[1], __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
      } else {
        for (uint32_t j = 0; j < 4; j++)
          atomicAdd((&glbl[g_adr + j]), sum4[nt][0][j]);
      }
    }

    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    int nindx_ = (0 + (threadIdx.x / 16) * 4) + 0 * NTILE +
                 (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
    int adr_ = mindx + M * nindx_ / 4;
    my_cntr = atomicAdd(&cntr[adr_], 1);

    // make sure LDS is free for write out staging
    if (DTRMNSTC) __syncthreads();

    // Update the complete counter
    flt4 vals[N / NTILE / GrpsShrB] = {};
    // If we're the last k-shard, read back the value and convert...
    if (my_cntr + 1 == k_rnd) {
      cntr[adr_] = 0;  // clear for next round
      if constexpr (DTRMNSTC) {
  #pragma unroll
        for (int ks = 0; ks < k_rnd; ks++) {
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            int g_nindx =
                (nt * NTILE + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) / 4;
            int g_adr = g_mindx * 4 + 0 + M * g_nindx * 4;
            __builtin_amdgcn_global_load_lds(
                (float4*)(&glbl[g_adr + M * N * ks]),
                &(((float4*)s)[(threadIdx.y * THRDS) + ks * THRDS * 4 +
                               nt * THRDS * 4 * k_rnd]),
                16, 0, 0);
          }
        }
        if (BIAS)
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            for (uint32_t j = 0; j < 4; j++) {
              int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                          (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
              biases[nt][j] = BIAS[(mindx % Bx) + (nindx % By) * Bx];
            }
          }
        asm volatile("s_waitcnt 0");
        for (int ks = 0; ks < k_rnd; ks++) {
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            float4 eval = ((float4*)s)[(threadIdx.x + threadIdx.y * THRDS) +
                                       ks * THRDS * 4 + nt * THRDS * 4 * k_rnd];
            vals[nt].f4 += eval;
          }
        }
      } else {
        for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
          int g_nindx =
              (nt * NTILE + (N / GrpsShrB) * (threadIdx.y % GrpsShrB)) / 4;
          int g_adr = g_mindx * 4 + 0 + M * g_nindx * 4;
          vals[nt].f4 = *(float4*)(&glbl[g_adr]);
          *(float4*)(&glbl[g_adr]) = {};  // clear out for next round
        }
        if (BIAS)
          for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
            for (uint32_t j = 0; j < 4; j++) {
              int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                          (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
              biases[nt][j] = BIAS[(mindx % Bx) + (nindx % By) * Bx];
            }
          }
      }
      __builtin_amdgcn_sched_barrier(0);
      for (uint32_t nt = 0; nt < N / NTILE / GrpsShrB; nt++) {
        for (uint32_t j = 0; j < 4; j++) {
          int nindx = (j + (threadIdx.x / 16) * 4) + nt * NTILE +
                      (N / GrpsShrB) * (threadIdx.y % GrpsShrB);
          if (nindx < actlN) {
            int adr = mindx + M * nindx;
            if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
              vals[nt].s8[j] += __bfloat162float(biases[nt][j]);
              C[adr] = __float2bfloat16(vals[nt].s8[j]);
            } else {
              vals[nt].s8[j] += __half2float(biases[nt][j]);
              C[adr] = __float2half(vals[nt].s8[j]);
            }
          }
        }
      }
    }

  #ifndef WVSPLITKRC_1KPASS
    m0 += CuCount * WvPrGrp * YTILE / GrpsShrB;
    m = (m0 + m1) % Mmod;
    k_str = (m0 / Mmod) * kFit * kfitsPerRdc;
    k_end = (m0 / Mmod + 1) * kFit * kfitsPerRdc;
    if (k_str >= K) break;
    kBase = 0;
  #endif
  }
}
#else
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, int GrpsShrB, int CHUNKK, int DTRMNSTC>
__global__ void wvSplitKrc_(const int actlN, const int K, const int Kap,
                            const int M, const int Bx, const int By,
                            const scalar_t* B, const scalar_t* __restrict__ A,
                            const scalar_t* __restrict__ BIAS, float* glbl,
                            int* cntr, scalar_t* C,
                            const int CuCount){UNREACHABLE_CODE}
#endif  // defined(__HIP__GFX9__) TODO: Add NAVI support

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount) {
  int _DTRMNSTC = 1;  // vllm::vllm_is_batch_invariant();

  auto M_in = in_b.size(0);
  auto N_in = in_a.size(0);
  auto K_in = in_b.size(1);
  auto Kap_in = in_a.stride(0);

  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(in_a.dtype() == in_b.dtype());
  TORCH_CHECK(K_in % 8 == 0, "k % 8 == 0");
  TORCH_CHECK(in_a.dtype() == torch::kFloat16 ||
              in_a.dtype() == torch::kBFloat16);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_a.dtype()).device(in_a.device()));

  auto N_p2 = 1U << (32 - __builtin_clz(N_in - 1));

  dim3 grid(CuCount);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // const int max_lds_len = get_lds_size() / 2;

  // With 64 Ms per CU (each of 4 SIMDs working on a 16x16 tile),
  // and each working on a 512-shard of K, how many CUs would we need?
  int rndup_cus = ((M_in + 64 - 1) / 64) * ((K_in + 512 - 1) / 512);

  // How many of 4 waves in a group can work on same 16 Ms at same time? First
  // try to maximize this. This reduces the Ms each group works on, i.e.
  // increasing the number of CUs needed.
  int GrpsShrB = min(N_p2 / 16, 4);

  // Given the above, how many CUs would we need?
  int CuNeeded = rndup_cus * GrpsShrB;

  if (CuNeeded > CuCount) throw std::runtime_error("Invalid wvSplitKrc size");

  // Can we increase SplitK by shrinking the K-shared to 256?
  int chunkk = (CuNeeded * 2 <= CuCount) ? 2 : 1;

  static torch::Tensor axl_glbl =
      torch::zeros(
          128 * 1024 * (_DTRMNSTC ? 12 : 1),
          torch::TensorOptions().dtype(torch::kFloat32).device(in_a.device()))
          .detach();
  static torch::Tensor axl_cntr =
      torch::zeros(
          128 * 1024 * (_DTRMNSTC ? 12 : 1) / 4,
          torch::TensorOptions().dtype(torch::kInt).device(in_a.device()))
          .detach();
  auto glbl = axl_glbl.data_ptr<float>();
  auto cntr = axl_cntr.data_ptr<int>();

#define WVSPLITKrc(_N, _GrpsShrB, _CHUNKK)                                     \
  {                                                                            \
    dim3 block(64, 4);                                                         \
    if (_DTRMNSTC)                                                             \
      wvSplitKrc_<fptype, 64, 16, 4, 8, 1, _N, _GrpsShrB, _CHUNKK, 1>          \
          <<<grid, block, 0, stream>>>(N_in, K_in, Kap_in, M_in, Bx_in, By_in, \
                                       af4, bf4, biasf4, glbl, cntr, c,        \
                                       CuCount);                               \
    else                                                                       \
      wvSplitKrc_<fptype, 64, 16, 4, 8, 1, _N, _GrpsShrB, _CHUNKK, 0>          \
          <<<grid, block, 0, stream>>>(N_in, K_in, Kap_in, M_in, Bx_in, By_in, \
                                       af4, bf4, biasf4, glbl, cntr, c,        \
                                       CuCount);                               \
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(in_a.scalar_type(), "wvSplitKrc", [&] {
    using fptype = typename scalar<scalar_t>::type;
    const fptype* af4 = reinterpret_cast<const fptype*>(in_a.data_ptr());
    const fptype* bf4 = reinterpret_cast<const fptype*>(in_b.data_ptr());
    const fptype* biasf4 =
        (in_bias.has_value() && in_bias->numel() > 0)
            ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
            : nullptr;
    fptype* c = reinterpret_cast<fptype*>(out_c.data_ptr());

    switch (N_p2) {
      case 16:
        WVSPLITKrc(16, 1, 1) break;
      case 32:
        if (chunkk == 2) WVSPLITKrc(32, 2, 2) else WVSPLITKrc(32, 2, 1) break;
      case 64:
        if (chunkk == 2) WVSPLITKrc(64, 4, 2) else WVSPLITKrc(64, 4, 1) break;
      case 128:
        if (chunkk == 2) WVSPLITKrc(128, 4, 2) else WVSPLITKrc(128, 4, 1) break;
      default:
        throw std::runtime_error(
            "Unsupported N value: " + std::to_string(M_in) + "," +
            std::to_string(K_in) + "," + std::to_string(N_in));
    }
  });
  return out_c;
}

#if defined(__HIP__MI3XX__) || defined(__GFX12__)
template <typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp,
          int A_CHUNK, int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitKQ_hf_sml_(const int K, const int Kap, const int Kbp, const int M,
                      const int Bx, const int By, const fp8_t* B,
                      const fp8_t* __restrict__ A,
                      const scalar_t* __restrict__ BIAS, scalar_t* C,
                      const float* __restrict__ s_A,
                      const float* __restrict__ s_B, const int _WvPrGrp,
                      const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE;
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 4) * sizeof(float)))) float;
  using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
  using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
  union bigType {
    char f8[A_CHUNK];
    char2 c2[A_CHUNK / 2];
    scalar_t h[A_CHUNK / 2];
    float f[A_CHUNK / 4];
    int i[A_CHUNK / 4];
    long l[A_CHUNK / 8];
    intx4 l2[A_CHUNK / 16];
    scalar8 h8;
  };

  __shared__ fp8_t s[max_lds_len];

  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
  #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
  #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  #endif
  }
  asm volatile("s_waitcnt vmcnt(0)");
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  float sA = *s_A;
  float sB = *s_B;

  while (m < M) {
  #ifdef __GFX12__
    // gfx12: per-lane scalar accumulation via v_dot4_f32_fp8_fp8
    float sum[N][YTILE] = {};
  #else
    // gfx9: MFMA accumulation
    scalar8 sum[N][YTILE] = {};
  #endif
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];

      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const fp8_t* B_ = &B[min__(k_, K - A_CHUNK)];
  #pragma unroll
        for (uint32_t y = 0; y < YTILE; ++y) {
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
        }
      }

  // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
          bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
        }
      }

  // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        for (uint32_t n = 0; n < N; n++) {
  #ifdef __GFX12__
          // gfx12: 4 x dot4 per A_CHUNK=16 bytes (4 FP8 per dot4)
          for (int y = 0; y < YTILE; ++y) {
    #pragma unroll
            for (int i = 0; i < A_CHUNK / 4; i++) {
              sum[n][y] = __builtin_amdgcn_dot4_f32_fp8_fp8(
                  bigA[n][k2].i[i], bigB[y][k2].i[i], sum[n][y]);
            }
          }
  #else
          // gfx9: MFMA path
          for (int i = 0; i < A_CHUNK; i += 8) {
            for (int y = 0; y < YTILE; ++y) {
              sum[n][y] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                  bigA[n][k2].l[i / 8], bigB[y][k2].l[i / 8], sum[n][y], 0, 0,
                  0);
            }
          }
  #endif
        }
      }
    }

    // Final reduction
  #ifdef __GFX12__
    // gfx12 wave32: DPP row_shr within 16-lane rows + cross-row shuffle
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:1 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        sum[n][y] += __shfl_xor(sum[n][y], 16);
      }
    }
  #else
    // gfx9 MFMA reduction
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        float accm0 = sum[n][y][0];
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][1], 0x101, 0xf, 0xf,
                                          1);  // row_shl1
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][2], 0x102, 0xf, 0xf,
                                          1);  // row_shl2
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][3], 0x103, 0xf, 0xf,
                                          1);  // row_shl3
        accm0 += __shfl_down(accm0, 20);
        accm0 += __shfl_down(accm0, 40);
        sum[n][y][0] = accm0;
      }
    }
  #endif

    const bool writeback_lane =
  #ifdef __GFX12__
        threadIdx.x == (THRDS - 1);
  #else
        threadIdx.x == 0;
  #endif
    if (writeback_lane) {
      scalar_t biases[N][YTILE] = {};
      if (BIAS)
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
          }
        }
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          if (y + m >= M) break;  // To avoid mem access fault.
  #ifdef __GFX12__
          float result = sum[n][y] * sA * sB;
  #else
          float result = sum[n][y][0] * sA * sB;
  #endif
          if constexpr (std::is_same_v<scalar_t, half>) {
            result += __half2float(biases[n][y]);
          } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
            result += __bfloat162float(biases[n][y]);
          }
          C[m + y + n * M] = __float2s<scalar_t>(result);
        }
      }
    }

    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else   // !defined(__HIP__MI3XX__) && !defined(__GFX12__)
template <typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp,
          int A_CHUNK, int UNRL, int N>
__global__ void wvSplitKQ_hf_sml_(const int K, const int Kap, const int Kbp,
                                  const int M, const int Bx, const int By,
                                  const fp8_t* B, const fp8_t* __restrict__ A,
                                  const scalar_t* __restrict__ BIAS,
                                  scalar_t* C, const float* __restrict__ s_A,
                                  const float* __restrict__ s_B,
                                  const int _WvPrGrp, const int CuCount) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__MI3XX__) || defined(__GFX12__)

#if defined(__HIP__MI3XX__) || defined(__GFX12__)
template <typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp,
          int A_CHUNK, int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitKQ_hf_(const int K, const int Kap, const int Kbp, const int M,
                  const int Bx, const int By, const fp8_t* B,
                  const fp8_t* __restrict__ A,
                  const scalar_t* __restrict__ BIAS, scalar_t* C,
                  const float* __restrict__ s_A, const float* __restrict__ s_B,
                  const int _WvPrGrp, const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE;
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 4) * sizeof(float)))) float;
  using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
  using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
  union bigType {
    char f8[A_CHUNK];
    char2 c2[A_CHUNK / 2];
    scalar_t h[A_CHUNK / 2];
    float f[A_CHUNK / 4];
    int i[A_CHUNK / 4];
    long l[A_CHUNK / 8];
    intx4 l2[A_CHUNK / 16];
    scalar8 h8;
  };

  __shared__ fp8_t s[max_lds_len];

  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
  #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
  #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  #endif
  }
  asm volatile("s_waitcnt vmcnt(0)");
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  float sA = *s_A;
  float sB = *s_B;

  while (m < M) {
  #ifdef __GFX12__
    // gfx12: per-lane scalar accumulation via v_dot4_f32_fp8_fp8
    float sum[N][YTILE] = {};
  #else
    // gfx9: MFMA accumulation
    scalar8 sum[N][YTILE] = {};
  #endif
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];

      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const fp8_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; ++y) {
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
        }
      }

  // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
          if (k_ + Kap * n < max_lds_len)
            bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
          else
            bigA[n][k2] = *((const bigType*)(&(A[k_ + Kap * n])));
        }
      }

  // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        for (uint32_t n = 0; n < N; n++) {
  #ifdef __GFX12__
          // gfx12: 4 x dot4 per A_CHUNK=16 bytes (4 FP8 per dot4)
          for (int y = 0; y < YTILE; ++y) {
    #pragma unroll
            for (int i = 0; i < A_CHUNK / 4; i++) {
              sum[n][y] = __builtin_amdgcn_dot4_f32_fp8_fp8(
                  bigA[n][k2].i[i], bigB[y][k2].i[i], sum[n][y]);
            }
          }
  #else
          // gfx9: MFMA path
          for (int i = 0; i < A_CHUNK; i += 8) {
            for (int y = 0; y < YTILE; ++y) {
              sum[n][y] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                  bigA[n][k2].l[i / 8], bigB[y][k2].l[i / 8], sum[n][y], 0, 0,
                  0);
            }
          }
  #endif
        }
      }
    }

    // Final reduction
  #ifdef __GFX12__
    // gfx12 wave32: DPP row_shr within 16-lane rows + cross-row shuffle
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:1 bound_ctrl:0 "
            : "=v"(sum[n][y])
            : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
        sum[n][y] += __shfl_xor(sum[n][y], 16);
      }
    }
  #else
    // gfx9 MFMA reduction
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < YTILE; y++) {
        float accm0 = sum[n][y][0];
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][1], 0x101, 0xf, 0xf,
                                          1);  // row_shl1
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][2], 0x102, 0xf, 0xf,
                                          1);  // row_shl2
        accm0 += __builtin_amdgcn_mov_dpp(sum[n][y][3], 0x103, 0xf, 0xf,
                                          1);  // row_shl3
        accm0 += __shfl_down(accm0, 20);
        accm0 += __shfl_down(accm0, 40);
        sum[n][y][0] = accm0;
      }
    }
  #endif

    const bool writeback_lane =
  #ifdef __GFX12__
        threadIdx.x == (THRDS - 1);
  #else
        threadIdx.x == 0;
  #endif
    if (writeback_lane) {
      scalar_t biases[N][YTILE] = {};
      if (BIAS)
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
          }
        }
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          if (y + m >= M) break;  // To avoid mem access fault.
  #ifdef __GFX12__
          float result = sum[n][y] * sA * sB;
  #else
          float result = sum[n][y][0] * sA * sB;
  #endif
          if constexpr (std::is_same_v<scalar_t, half>) {
            result += __half2float(biases[n][y]);
          } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
            result += __bfloat162float(biases[n][y]);
          }
          C[m + y + n * M] = __float2s<scalar_t>(result);
        }
      }
    }

    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else   // !defined(__HIP__MI3XX__) && !defined(__GFX12__)
template <typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp,
          int A_CHUNK, int UNRL, int N>
__global__ void wvSplitKQ_hf_(const int K, const int Kap, const int Kbp,
                              const int M, const int Bx, const int By,
                              const fp8_t* B, const fp8_t* __restrict__ A,
                              const scalar_t* __restrict__ BIAS, scalar_t* C,
                              const float* __restrict__ s_A,
                              const float* __restrict__ s_B, const int _WvPrGrp,
                              const int CuCount) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__MI3XX__) || defined(__GFX12__)

void wvSplitKQ(const at::Tensor& in_b, const at::Tensor& in_a,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount) {
  static c10::ScalarType kFp8Type = is_fp8_ocp()
                                        ? c10::ScalarType::Float8_e4m3fn
                                        : c10::ScalarType::Float8_e4m3fnuz;
  auto M_in = in_b.size(0);
  auto K_in = in_b.size(1);
  auto N_in = in_a.size(0);
  auto Kap_in = in_a.stride(0);
  auto Kbp_in = in_b.stride(0);
  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(K_in % 16 == 0, "k % 16 == 0");
  TORCH_CHECK(in_a.dtype() == in_b.dtype() && in_a.dtype() == kFp8Type);
  TORCH_CHECK(out_c.dtype() == torch::kFloat16 ||
              out_c.dtype() == torch::kBFloat16);

  dim3 grid(CuCount);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int max_lds_len = get_lds_size();

#define WVSPLITKQ_IMPL(_THRDS, _WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N) \
  {                                                                            \
    dim3 block(_THRDS, _WvPrGrp);                                              \
    if ((Kap_in * N_in <= max_lds_len) && (M_in % _YTILEs == 0)) {             \
      int __wvPrGrp = min(_WvPrGrp, mindiv(M_in, CuCount * _YTILEs, 16));      \
      wvSplitKQ_hf_sml_<fptype, fp8_t, _THRDS, _YTILEs, _WvPrGrp, 16, _UNRLs,  \
                        _N><<<grid, block, 0, stream>>>(                       \
          K_in, Kap_in, Kbp_in, M_in, Bx_in, By_in, b_ptr, a_ptr, bias_ptr,    \
          c_ptr, s_a, s_b, __wvPrGrp, CuCount);                                \
    } else {                                                                   \
      int __wvPrGrp = min(_WvPrGrp, mindiv(M_in, CuCount * _YTILEm, 16));      \
      wvSplitKQ_hf_<fptype, fp8_t, _THRDS, _YTILEm, _WvPrGrp, 16, _UNRLm, _N>  \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,      \
                                       By_in, b_ptr, a_ptr, bias_ptr, c_ptr,   \
                                       s_a, s_b, __wvPrGrp, CuCount);          \
    }                                                                          \
  }

#define WVSPLITKQ(_WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N)      \
  if (on_gfx12())                                                      \
    WVSPLITKQ_IMPL(32, _WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N) \
  else                                                                 \
    WVSPLITKQ_IMPL(64, _WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N)

  AT_DISPATCH_REDUCED_FLOATING_TYPES(out_c.scalar_type(), "wvSplitKQ", [&] {
    using fptype = typename scalar<scalar_t>::type;
    auto c_ptr = reinterpret_cast<fptype*>(out_c.data_ptr());
    auto s_a = scale_a.data_ptr<float>();
    auto s_b = scale_b.data_ptr<float>();
    VLLM_DISPATCH_FP8_TYPES(in_a.scalar_type(), "wvSplitKQ", [&] {
      auto a_ptr = in_a.data_ptr<fp8_t>();
      auto b_ptr = in_b.data_ptr<fp8_t>();
      auto bias_ptr = (in_bias.has_value() && in_bias->numel() > 0)
                          ? reinterpret_cast<fptype*>(in_bias->data_ptr())
                          : nullptr;
      switch (N_in) {
        case 1:
          WVSPLITKQ(16, 2, 2, 2, 2, 1)
          break;
        case 2:
          WVSPLITKQ(16, 2, 2, 2, 2, 2)
          break;
        case 3:
          WVSPLITKQ(16, 2, 2, 1, 1, 3)
          break;
        case 4:
          WVSPLITKQ(16, 2, 2, 1, 1, 4)
          break;
        default:
          throw std::runtime_error(
              "Unsupported N value: " + std::to_string(M_in) + "," +
              std::to_string(K_in) + "," + std::to_string(N_in));
      }
    });
  });
}
