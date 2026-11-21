/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Kernel arguments and launchers of the Muse Spark fused all-reduces, one
 * instantiation per world size. The kernels live in muse_fused_allreduce.cuh,
 * instantiated by muse_fused_allreduce_w<NR>.cu; muse_fused_allreduce.cu holds
 * the torch entry points and dispatches on the world size.
 */

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace vllm::muse_far {

// Room for the inbox pointers of the largest world, so that the argument
// layout does not depend on the world size.
constexpr int MAX_NR = 8;

// Communication modes. The one-block-per-row kernels take the first four;
// MULTI_BLOCK selects the kernels that serve each row with three blocks.
enum Mode : int32_t {
  TWO_HOP = 0,
  ONE_HOP = 1,
  NO_COMM = 2,
  TWO_HOP_CHUNKED = 3,
  MULTI_BLOCK = 4,
};

struct Comm {
  // This op's inbox region on every rank (the first NR entries).
  uint8_t* inbox[MAX_NR];
  // One-block-per-row kernels: [max_rows][2] (inbox slot of the row's next
  // call in [0, 3), owner of the previous row-call | OWNER_FP8, -1 before the
  // first). Multi-block kernels: [3][max_rows] (sender, owner and row block of
  // every row): the slot of the row's next call, plus 3 after the first call.
  int32_t* state;
  int32_t rank;
  uint32_t max_rows;
  int32_t rows;
  int32_t mode;
};

struct FinalizeArgs {
  const uint16_t* y;      // [M * 8, 4096] bf16 per-route expert rows
  const uint16_t* gamma;  // [4096] bf16 post-expert norm weight
  const float* tw;        // [M, 8] fp32 route weights
  const int32_t* ids;     // [M, 8] global expert ids
  const int32_t* emap;    // [E] global -> local expert, -1 on other ranks
  uint16_t* out;          // [M, 4096] bf16, or null
  uint8_t* q;             // [M, 4096] fp8 e4m3fn, or null
  float* qs;              // [M] fp32 per-token scales (with q)
  float eps;
  float gain;
  Comm comm;
};

struct BoundaryArgs {
  const uint16_t* partial;  // [M, 8192] bf16 this rank's o_proj output
  const float* stream;      // [M, 8192] fp32
  const float* alpha;       // [8192] fp32
  const float* beta;        // [8192] fp32
  const uint16_t* post_w;   // [8192] bf16 (ones and gain 0 for no weight)
  const uint16_t* pre_w;    // [8192] bf16 (ones and gain 0 for no weight)
  float* new_stream;        // [M, 8192] fp32
  uint16_t* out;            // [M, 8192] bf16, or null
  uint8_t* q;               // [M, 8192] fp8 e4m3fn, or null
  float* qs;                // [M] fp32
  float post_eps;
  float pre_eps;
  float post_gain;
  float pre_gain;
  int32_t post_input_scale;
  int32_t pre_input_scale;
  Comm comm;
};

template <int NR>
void launch_finalize(const FinalizeArgs& a, bool read_all, cudaStream_t stream);
template <int NR>
void launch_boundary(const BoundaryArgs& a, cudaStream_t stream);

// EXT is `extern` to name an instantiation that another unit provides, and
// empty in the unit that provides it.
#define VLLM_MUSE_FAR_DECLARE(EXT, NR)                             \
  EXT template void launch_finalize<NR>(const FinalizeArgs&, bool, \
                                        cudaStream_t);             \
  EXT template void launch_boundary<NR>(const BoundaryArgs&, cudaStream_t)

VLLM_MUSE_FAR_DECLARE(extern, 4);
VLLM_MUSE_FAR_DECLARE(extern, 8);

}  // namespace vllm::muse_far
