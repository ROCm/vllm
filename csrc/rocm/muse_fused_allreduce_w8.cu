// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Instantiates the Muse Spark fused all-reduce kernels for 8 ranks only; see
// muse_fused_allreduce.h.
#include "muse_fused_allreduce.cuh"

namespace vllm::muse_far {
VLLM_MUSE_FAR_DECLARE(, 8);
}  // namespace vllm::muse_far
