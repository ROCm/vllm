// Launch ABI of the RDNA3.5 attention kernels: each decode build exposes a
// LaunchFn, each prefill build a PrefillFn.
#pragma once

#include <hip/hip_runtime.h>

namespace rdna {

// Raw pointers keep torch headers out of the kernels. Mirrored by
// benchmarks/kernels/rocm_rdna_hip_attn/tools/jit.py.
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
  // Batched builds: sequences, and per-sequence strides.
  int nseq;
  int bt_stride, acc_stride, ml_stride, cnt_stride;
  // Multimodal ranges in device memory, read at run time so CUDA graphs
  // follow them: [count, start0, end0, ...], at most mm_capacity ranges, in
  // positions of the sequence; sequence s reads mm_ranges + s * mm_stride
  // (0: all share them). Null for no ranges.
  const int* mm_ranges;
  int mm_capacity;
  int mm_stride;
};

using LaunchFn = void (*)(const LaunchArgs&);

// The launch function of each entry of kDecodeConfigs, null until its unit
// (cmake/rocm_rdna_attn.cmake) registers it. Defined in
// rdna_registry.cu.
LaunchFn* decode_launchers();
// Likewise for kDecodeBatchConfigs.
LaunchFn* decode_batch_launchers();

// One sequence of num_tokens query tokens. partial_o, partial_ml and counters
// are the split-KV scratch; max_seq_len only sizes the split. mm_ranges holds
// num_mm_ranges (start, end) multimodal ranges per sequence, empty ones with
// start == end. With a plan (plan_prefill's [num_entries, 4] work list), a
// batch at query_start; head_group 0 is the build's.
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
  const int* mm_ranges;
  int num_mm_ranges;
  const int* plan;
  const int* query_start;
  int num_entries;
  int bt_stride;
  int head_group;
  // Non-causal builds: V, and the token strides of Q, K (in `kv`) and V.
  const void* v;
  int q_stride;
  int k_stride;
  int v_stride;
};

using PrefillFn = void (*)(const PrefillArgs&);

// Likewise for kPrefillConfigs.
PrefillFn* prefill_launchers();

}  // namespace rdna
