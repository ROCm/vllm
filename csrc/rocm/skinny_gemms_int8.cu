#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <algorithm>

#include "../cuda_compat.h"
#include "dispatch_utils.h"

#include "skinny_gemms_int8/kernel.cuh"  // arch helpers + scalar<> traits
#include "skinny_gemms_int8/launch.h"    // per-N launchers (defined in shards)

torch::Tensor wvSplitK_int8(const at::Tensor& in_a, const at::Tensor& in_b,
                            const at::Tensor& in_scale,
                            const std::optional<at::Tensor>& in_bias,
                            const int64_t CuCount, const int64_t group_size) {
  auto M_in = in_a.size(0);
  auto K_in = in_a.size(1);
  auto N_in = in_b.size(0);
  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(in_a.dtype() == torch::kInt8, "Weight must be int8");
  TORCH_CHECK(
      in_b.dtype() == torch::kFloat16 || in_b.dtype() == torch::kBFloat16,
      "Activation must be float16 or bfloat16");
  TORCH_CHECK(in_scale.dtype() == in_b.dtype(),
              "Scale dtype must match activation dtype");
  TORCH_CHECK(K_in % 16 == 0, "K must be divisible by 16 for int8 kernel");

  // Per-channel: group_size == -1, scale shape [M].
  // Per-group: group_size in {32, 64, 128} (or any multiple of 16 dividing K),
  //            scale shape [M, K/group_size].
  if (group_size == -1) {
    TORCH_CHECK(in_scale.dim() == 1, "Per-channel scale must be 1-D [M]");
    TORCH_CHECK(in_scale.size(0) == M_in,
                "Per-channel scale size must match M");
  } else {
    TORCH_CHECK(group_size >= 16, "group_size must be >= 16 (A_CHUNK)");
    TORCH_CHECK((group_size % 16) == 0,
                "group_size must be a multiple of 16 (A_CHUNK)");
    TORCH_CHECK(K_in % group_size == 0,
                "K must be divisible by group_size=", group_size);
    int64_t num_groups = K_in / group_size;
    TORCH_CHECK(in_scale.dim() == 2,
                "Per-group scale must be 2-D [M, K/group_size]");
    TORCH_CHECK(in_scale.size(0) == M_in && in_scale.size(1) == num_groups,
                "Per-group scale must be [M, K/group_size] = [", M_in, ", ",
                num_groups, "], got [", in_scale.size(0), ", ",
                in_scale.size(1), "]");
    TORCH_CHECK(in_scale.is_contiguous(), "Per-group scale must be contiguous");
  }

  const int max_lds_len = get_lds_size_int8() / 2;
  TORCH_CHECK(K_in * N_in <= max_lds_len,
              "K*N exceeds LDS capacity; only sml variant is supported. "
              "K=",
              K_in, " N=", N_in, " K*N=", K_in * N_in, " max=", max_lds_len);

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_REDUCED_FLOATING_TYPES(in_b.scalar_type(), "wvSplitK_int8", [&] {
    using fptype = typename scalar<scalar_t>::type;
    const int8_t* wptr = in_a.data_ptr<int8_t>();
    const fptype* aptr = reinterpret_cast<const fptype*>(in_b.data_ptr());
    const fptype* sptr = reinterpret_cast<const fptype*>(in_scale.data_ptr());
    const fptype* biasptr =
        (in_bias.has_value() && in_bias->numel() > 0)
            ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
            : nullptr;
    fptype* cptr = reinterpret_cast<fptype*>(out_c.data_ptr());

    int sYT = (M_in + CuCount * 4 - 1) / (CuCount * 4);
    int thrds = is_gfx11_int8() ? 32 : 64;
    int ytile, unrl;
    if (N_in >= 4 && sYT >= 480) {
      ytile = 4;
      unrl = 1;
    } else {
      ytile = 1;
      unrl = 4;
    }

    TORCH_CHECK(M_in % ytile == 0, "M must be divisible by YTILE=", ytile);

    switch (N_in) {
      case 1:
        launch_int8_n1(grid, stream, K_in, M_in, Bx_in, By_in, wptr, aptr, sptr,
                       biasptr, cptr, CuCount, thrds, ytile, unrl, group_size);
        break;
      case 2:
        launch_int8_n2(grid, stream, K_in, M_in, Bx_in, By_in, wptr, aptr, sptr,
                       biasptr, cptr, CuCount, thrds, ytile, unrl, group_size);
        break;
      case 3:
        launch_int8_n3(grid, stream, K_in, M_in, Bx_in, By_in, wptr, aptr, sptr,
                       biasptr, cptr, CuCount, thrds, ytile, unrl, group_size);
        break;
      case 4:
        launch_int8_n4(grid, stream, K_in, M_in, Bx_in, By_in, wptr, aptr, sptr,
                       biasptr, cptr, CuCount, thrds, ytile, unrl, group_size);
        break;
      case 5:
        launch_int8_n5(grid, stream, K_in, M_in, Bx_in, By_in, wptr, aptr, sptr,
                       biasptr, cptr, CuCount, thrds, ytile, unrl, group_size);
        break;
      default:
        throw std::runtime_error(
            "Unsupported N value: " + std::to_string(M_in) + "," +
            std::to_string(K_in) + "," + std::to_string(N_in));
    }
  });

  return out_c;
}

// Sweep function disabled by default to reduce compile time.
// Build with -DVLLM_SKINNY_GEMM_SWEEP to enable.
#ifdef VLLM_SKINNY_GEMM_SWEEP
torch::Tensor wvSplitK_int8_sweep(const at::Tensor& in_a,
                                  const at::Tensor& in_b,
                                  const at::Tensor& in_scale,
                                  const std::optional<at::Tensor>& in_bias,
                                  const int64_t CuCount, const int64_t ytile,
                                  const int64_t unrl, const int64_t achunk,
                                  const int64_t wvprgrp) {
  auto M_in = in_a.size(0);
  auto K_in = in_a.size(1);
  auto N_in = in_b.size(0);

  TORCH_CHECK(in_a.dtype() == torch::kInt8, "Weight must be int8");
  TORCH_CHECK(in_b.dtype() == torch::kFloat16,
              "Sweep only supports float16 activations");
  TORCH_CHECK(in_scale.dtype() == torch::kFloat16,
              "Sweep only supports float16 scale");
  TORCH_CHECK(in_scale.size(0) == M_in, "Scale size must match M");
  TORCH_CHECK(K_in % achunk == 0, "K must be divisible by achunk=", achunk);
  TORCH_CHECK(M_in % ytile == 0, "M must be divisible by ytile=", ytile);

  const int max_lds_len = get_lds_size_int8() / 2;
  TORCH_CHECK(K_in * N_in <= max_lds_len, "K*N exceeds LDS capacity. K=", K_in,
              " N=", N_in);

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  dim3 grid(CuCount);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using fptype = half;
  const int8_t* wptr = in_a.data_ptr<int8_t>();
  const fptype* aptr = reinterpret_cast<const fptype*>(in_b.data_ptr());
  const fptype* sptr = reinterpret_cast<const fptype*>(in_scale.data_ptr());
  const fptype* biasptr = nullptr;
  fptype* cptr = reinterpret_cast<fptype*>(out_c.data_ptr());

  const int THRDS = is_gfx11_int8() ? 32 : 64;

  switch (N_in) {
    case 1:
      launch_int8_n1_sweep(grid, stream, K_in, M_in, 1, 1, wptr, aptr, sptr,
                           biasptr, cptr, CuCount, THRDS, ytile, wvprgrp,
                           achunk, unrl);
      break;
    case 2:
      launch_int8_n2_sweep(grid, stream, K_in, M_in, 1, 1, wptr, aptr, sptr,
                           biasptr, cptr, CuCount, THRDS, ytile, wvprgrp,
                           achunk, unrl);
      break;
    case 3:
      launch_int8_n3_sweep(grid, stream, K_in, M_in, 1, 1, wptr, aptr, sptr,
                           biasptr, cptr, CuCount, THRDS, ytile, wvprgrp,
                           achunk, unrl);
      break;
    case 4:
      launch_int8_n4_sweep(grid, stream, K_in, M_in, 1, 1, wptr, aptr, sptr,
                           biasptr, cptr, CuCount, THRDS, ytile, wvprgrp,
                           achunk, unrl);
      break;
    default:
      TORCH_CHECK(false, "Unsupported N=", N_in);
  }

  return out_c;
}
#endif  // VLLM_SKINNY_GEMM_SWEEP
