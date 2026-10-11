// Launch ABI of the RDNA3.5 attention variants: each decode variant exposes a
// LaunchFn, each prefill variant a PrefillFn.
#pragma once

#include <hip/hip_runtime.h>

namespace rdna_singleseq {

// Raw pointers keep torch headers out of the kernels. Mirrored by
// benchmarks/kernels/rocm_rdna_singleseq_hip_attn/tools/jit.py.
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
  float scale;
  hipStream_t stream;
};

using LaunchFn = void (*)(const LaunchArgs&);

// One sequence of num_tokens query tokens. partial_o, partial_ml and counters
// are the split-KV scratch; max_seq_len only sizes the split.
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

}  // namespace rdna_singleseq
