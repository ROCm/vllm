// Launch ABI of the RDNA3.5 attention variants.
//
// Every decode variant (rdna35_decode_attn.cu, specialised by compile-time
// defines) exposes one host function of type LaunchFn, every prefill variant
// (rdna35_prefill_attn.cu) one of type PrefillFn.  The kernel translation units
// see only this header, not the list of variants, so a change to the list
// rebuilds only the units whose variants changed.
#pragma once

#include <hip/hip_runtime.h>

namespace rdna35 {

// Pointers are raw so the kernel units need no torch headers: a unit then
// compiles in about the time of its kernels, not of torch/extension.h.
// benchmarks/kernels/gfx1151_decode_attn/tools/jit.py mirrors it field for
// field.
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

// One sequence of num_tokens query tokens, its seq_lens[0] - num_tokens
// earlier tokens cached.  partial_o / partial_ml / counters are the split-KV
// scratch (make_prefill_scratch); max_seq_len, the host's copy of S, only
// sizes the split.
struct PrefillArgs {
  const void* q;
  const void* kv;
  const int* bt;
  void* out;
  const int* seq_lens;
  float* partial_o;
  float* partial_ml;
  int* counters;
  int num_tokens;
  int max_seq_len;
  float scale;
  hipStream_t stream;
};

using PrefillFn = void (*)(const PrefillArgs&);

}  // namespace rdna35
