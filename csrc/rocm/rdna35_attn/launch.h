// Launch ABI of the RDNA3.5 decode-attention variants.
//
// Every variant (rdna35_decode_attn.cu, specialised by compile-time defines)
// exposes one host function of type LaunchFn.  The kernel translation units
// see only this header, not the list of variants, so a change to the list
// rebuilds only the units whose variants changed.
#pragma once

#include <hip/hip_runtime.h>

namespace rdna35 {

// Pointers are raw so the kernel units need no torch headers: a unit then
// compiles in about the time of its kernels, not of torch/extension.h.
struct LaunchArgs {
  const void* q;
  const void* kv;
  const int* bt;
  float* acc;
  float* m;
  float* l;
  int* cnt;
  void* out;
  const int* seq_lens;
  int bt_width;
  // Per-sequence strides of a batch launch; 0 for one sequence.
  int bt_stride, acc_stride, ml_stride, cnt_stride;
  int nseq;
  float scale;
  hipStream_t stream;
};

using LaunchFn = void (*)(const LaunchArgs&);

}  // namespace rdna35
