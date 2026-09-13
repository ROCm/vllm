// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Causal decode attention for MHA (num_q_heads == num_kv_heads) on AMD RDNA3.5
// (gfx1151).
//
// The work is split over the KV axis and merged afterwards, in two kernels:
//
//   rdna35_causal_mha   one wave per (sequence, KV head group, KV segment).
//                       Walks its slice of the context and writes an fp32
//                       partial: the running max, the running sum, and the
//                       unnormalised P.V accumulator.
//   rdna35_mha_reduce   one thread per output element.  Merges the partials of
//                       a row with a log-sum-exp rescale and writes the fp16 or
//                       bf16 output.
//
// PHASES of the split kernel, in order:
//
//   1. Task decomposition.  blockIdx.x unpacks to (sequence, head group,
//      segment); the wave's lanes then split into HEADS_PER_WAVE groups, one
//      per KV head, so a lane owns VEC2_PER_LANE elements of one head.
//   2. Q load.  The wave's query rows are read once into registers and stay
//      there for the whole KV walk.
//   3. KV burst load.  UNROLL tokens of K and V are loaded per iteration.
//      Adjacent KV heads are contiguous in the paged cache, so a head group
//      reads as one run.  Bursts that lie inside one page take a fast address
//      path that hoists the block-table lookup out of the burst.
//   4. Scoring.  Q.K per token as a dot2 chain, reduced across the lanes that
//      share a KV token, then scaled, optionally soft-capped, and causally
//      masked.
//   5. Online softmax.  The burst's maximum is taken first and the
//      accumulators are rescaled once per burst rather than once per token.
//   6. P.V accumulation.  Also a dot2 chain, against the softmax weight packed
//      as a one-hot pair.
//   7. Partial write.  Each wave owns its (head, segment) partial outright, so
//      it writes straight to global memory with no LDS staging and no barrier.
//
// Restrictions the host side MUST enforce before calling (see
// vllm/v1/attention/ops/rdna35_causal_mha_attn.py, which mirrors this list):
// fp16 or bf16, no quantized KV, Hq == Hkv, head_dim in {64,128,256,512},
// num_q_tokens in {1,4}, causal, no sliding window / ALiBi / sinks, and a
// [num_blocks, 2, block_size, num_kv_heads, head_size] KV cache (NHD).

#include <torch/all.h>
#include <c10/hip/HIPStream.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <type_traits>

// The kernels use RDNA3 wave32 DPP row_ror encodings, v_dot2 and v_exp_f32,
// which only exist on gfx11 device passes.  __GFX11__ is predefined by the
// compiler on a gfx11 device pass (the same macro skinny_gemms / attention and
// moe_gemm_w4a16_wmma use).  Compile the real bodies for gfx11 and on the host
// pass, which needs the full definition; emit stubs for any other device arch
// so the TU links in non-gfx11 and multi-arch builds.  The op is never called
// off gfx11 -- Python gates on on_gfx1151().
#if !defined(__HIPCC__) || defined(__GFX11__)
  #define VLLM_RDNA35_MHA_REAL_BODY 1
#endif

#define RDNA35_MHA_INLINE __device__ __forceinline__

namespace rdna35_mha_common {

// v_exp_f32 is natively base-2, so the online softmax works in the log2 domain
// and scale/softcap arrive pre-multiplied by log2(e); __expf would emit a fixup
// multiply on every call.
RDNA35_MHA_INLINE float exp2_fast(float x) { return __builtin_amdgcn_exp2f(x); }

// vec2 is the 32-bit pair the dot2 instruction consumes.  Adding bf16 means
// adding one specialisation here: gfx1151 has v_dot2_f32_bf16 as well.
template <typename T>
struct elem_traits;

template <>
struct elem_traits<half> {
  using vec2 = half2;
  static RDNA35_MHA_INLINE float dot2(const vec2 a, const vec2 b, float acc) {
    return __builtin_amdgcn_fdot2(a, b, acc, false);
  }
  static RDNA35_MHA_INLINE float lo(const vec2 v) { return __half2float(v.x); }
  static RDNA35_MHA_INLINE float hi(const vec2 v) { return __half2float(v.y); }
  static RDNA35_MHA_INLINE vec2 zero2() { return __float2half2_rn(0.f); }
  static RDNA35_MHA_INLINE half from_float(float v) {
    return __float2half_rn(v);
  }

  // The softmax weight as the one-hot pair (weight, 0), so the P.V product can
  // be a dot2.  Same scheme as Triton's `acc += tl.dot(P.to(V.dtype), V)`,
  // which rounds the weight to the KV dtype so the matmul takes both operands
  // in fp16.
  static RDNA35_MHA_INLINE vec2 pack_weight(float weight) {
    return __halves2half2(__float2half_rn(weight), (half)0.f);
  }
  static RDNA35_MHA_INLINE float unpack_weight(const vec2 weight) {
    return __half2float(weight.x);
  }

  // acc.lo += weight*v.lo, acc.hi += weight*v.hi, as two dot2 against the
  // one-hot weight.  pack_weight builds (weight, 0), so dot2 against it selects
  // v's LOW half; the swapped (0, weight) selects the high one.
  static RDNA35_MHA_INLINE void accum_pv(float& lo_acc, float& hi_acc,
                                         const vec2 weight, const vec2 v) {
    lo_acc = __builtin_amdgcn_fdot2(weight, v, lo_acc, false);
    hi_acc =
        __builtin_amdgcn_fdot2(__lowhigh2highlow(weight), v, hi_acc, false);
  }
};

// bf16.  gfx1151 has v_dot2_f32_bf16 next to the fp16 one, so this mirrors the
// specialisation above.
//
// Accuracy is NOT the same as fp16 and the caller should not expect it to be:
// bf16 carries 8 mantissa bits against fp16's 11, so Q.K accumulates about 8x
// the rounding error per element.  The accumulators and the entire softmax stay
// fp32 -- only the Q/K/V operands and the output are narrowed -- so the error
// is input quantisation, not a change of algorithm.  The tests use a separate
// tolerance for it.
template <>
struct elem_traits<__hip_bfloat16> {
  using vec2 = __hip_bfloat162;
  // The builtin takes a native 2-wide short vector.  HIP's short2 is a
  // HIP_vector_type class, which does not convert to it, and __hip_bfloat162 is
  // a struct so __builtin_bit_cast rejects it as not trivially copyable -- so
  // read the two raw halves and build the vector explicitly.
  using short2_native = short __attribute__((ext_vector_type(2)));
  static RDNA35_MHA_INLINE short2_native as_short2(const vec2 v) {
    short2_native out;
    out.x = static_cast<short>(__bfloat16_as_ushort(v.x));
    out.y = static_cast<short>(__bfloat16_as_ushort(v.y));
    return out;
  }
  static RDNA35_MHA_INLINE float dot2(const vec2 a, const vec2 b, float acc) {
    return __builtin_amdgcn_fdot2_f32_bf16(as_short2(a), as_short2(b), acc,
                                           false);
  }
  static RDNA35_MHA_INLINE float lo(const vec2 v) {
    return __bfloat162float(v.x);
  }
  static RDNA35_MHA_INLINE float hi(const vec2 v) {
    return __bfloat162float(v.y);
  }
  static RDNA35_MHA_INLINE vec2 zero2() {
    return __hip_bfloat162(__float2bfloat16(0.f), __float2bfloat16(0.f));
  }
  static RDNA35_MHA_INLINE __hip_bfloat16 from_float(float v) {
    return __float2bfloat16(v);
  }

  // The P.V path, mirroring the fp16 traits: the softmax weight as the one-hot
  // pair (weight, 0) so the product can be a dot2 rather than an fma_mix, which
  // VOPD cannot pair.  dot2 against (w, 0) selects v's LOW half; the swapped
  // (0, w) selects the high one.
  static RDNA35_MHA_INLINE vec2 pack_weight(float weight) {
    return __hip_bfloat162(__float2bfloat16(weight), __float2bfloat16(0.f));
  }
  static RDNA35_MHA_INLINE float unpack_weight(const vec2 weight) {
    return __bfloat162float(weight.x);
  }
  static RDNA35_MHA_INLINE void accum_pv(float& lo_acc, float& hi_acc,
                                         const vec2 weight, const vec2 v) {
    const vec2 swapped(weight.y, weight.x);
    lo_acc = __builtin_amdgcn_fdot2_f32_bf16(as_short2(weight), as_short2(v),
                                             lo_acc, false);
    hi_acc = __builtin_amdgcn_fdot2_f32_bf16(as_short2(swapped), as_short2(v),
                                             hi_acc, false);
  }
};

// Merges the per-segment partials into the final output, one thread per output
// element.  Walking HEAD_DIM from a single wave per (head, m) row instead
// leaves only Hq waves at NUM_Q_TOKENS=1 on a 40-CU part, which measures pure
// latency; a thread per element raises the wave count by
// HEAD_DIM/THREADS_PER_BLOCK for the same bytes.
//
// The per-segment weight is hoisted into a register array computed once rather
// than recomputing exp2(seg_max - global_max) for every (segment, d) pair.  An
// empty segment carries a -inf max, which becomes a zero weight rather than a
// skipped iteration, so the accumulation stays branch-free -- the split kernels
// always write a finite acc, so the zero never multiplies garbage.
//
//   THREADS_PER_BLOCK  must tile HEAD_DIM.  Sets both the d-parallelism and
//                      how many blocks a row of HEAD_DIM splits into.
//
// head and seq get their own grid dimensions so the kernel never divides by
// the runtime Hq; blockIdx.x unpacks with compile-time constants only.
//
// The segment count is a runtime argument, not a template parameter.  The rule
// that picks it emits values that are not powers of two (Hq=40 gives 6, Hq=48
// gives 5), so templating it would force the dispatch to switch over an
// instantiated set and reject the rest.  The per-segment maxes and weights are
// therefore not held in register arrays: pass one takes the maximum, pass two
// accumulates the sum and the output together, re-reading partial_max.  That
// re-read is a few KB against the hundreds of MB of KV the split kernel moves.
//
// Both loops are unrolled by a fixed factor with independent accumulators, so
// the loads issue back to back rather than serialising on one accumulator.
// Only the unroll factor is compile-time; the trip count stays runtime, and a
// scalar tail handles counts that are not a multiple of it.
template <typename T, int NUM_Q_TOKENS, int HEAD_DIM, int THREADS_PER_BLOCK>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void rdna35_mha_reduce(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum, T* __restrict__ out, const int Hq,
    const int num_kv_segments) {
#if !defined(VLLM_RDNA35_MHA_REAL_BODY)
  // Non-gfx11 device pass: a stub so the TU links in multi-arch builds.  The
  // op is never called off gfx11 (Python gates on on_gfx1151()).
  return;
#else
  static_assert(
      HEAD_DIM % THREADS_PER_BLOCK == 0 || THREADS_PER_BLOCK % HEAD_DIM == 0,
      "THREADS_PER_BLOCK must tile HEAD_DIM");
  constexpr int DIM_CHUNKS =
      (HEAD_DIM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  const int chunk = blockIdx.x % DIM_CHUNKS;
  const int m = (blockIdx.x / DIM_CHUNKS) % NUM_Q_TOKENS;
  const int q_head = blockIdx.y;
  const int seq = blockIdx.z;
  const int d = chunk * THREADS_PER_BLOCK + (int)threadIdx.x;

  const size_t stat_base =
      (size_t)(seq * Hq + q_head) * num_kv_segments * NUM_Q_TOKENS + m;

  // Unrolled by a fixed factor, with independent partial maxes so the loads
  // issue back to back instead of serializing on one accumulator.  UNROLL is a
  // compile-time constant while the trip count stays runtime, which is the
  // point: templating the trip count is what forced the dispatch to switch over
  // an instantiated set of segment counts in the first place.
  //
  // This matters more than the traffic share suggests.  The reduce moves
  // 0.03% of the split kernel's bytes at ctx=65536 -- but 13% at ctx=128, where
  // the KV walk is short and this pass is a real fraction of the kernel.  A
  // first version without the unroll cost 3.8% at ctx=128, decaying to 0.3% by
  // ctx=8192, exactly tracking that curve.
  constexpr int UNROLL = 4;
  float gmax[UNROLL];
  #pragma unroll
  for (int u = 0; u < UNROLL; ++u) gmax[u] = -INFINITY;
  int seg = 0;
  for (; seg + UNROLL <= num_kv_segments; seg += UNROLL) {
  #pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      gmax[u] = fmaxf(
          gmax[u], partial_max[stat_base + (size_t)(seg + u) * NUM_Q_TOKENS]);
  }
  for (; seg < num_kv_segments; ++seg)
    gmax[0] =
        fmaxf(gmax[0], partial_max[stat_base + (size_t)seg * NUM_Q_TOKENS]);
  float global_max = gmax[0];
  #pragma unroll
  for (int u = 1; u < UNROLL; ++u) global_max = fmaxf(global_max, gmax[u]);
  if (global_max == -INFINITY) global_max = 0.f;

  const size_t part_base =
      ((size_t)(seq * Hq + q_head) * num_kv_segments * NUM_Q_TOKENS + m) *
          HEAD_DIM +
      d;

  // Sum and accumulate in one pass: the weight is consumed by both, so holding
  // it across the two loops was the only thing the register array bought.
  // Unrolled the same way, and for the same reason -- the fp32 adds here are a
  // dependent chain, so independent accumulators let the loads overlap it.
  float gsum[UNROLL], gacc[UNROLL];
  #pragma unroll
  for (int u = 0; u < UNROLL; ++u) gsum[u] = gacc[u] = 0.f;

  seg = 0;
  for (; seg + UNROLL <= num_kv_segments; seg += UNROLL) {
  #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const size_t s = (size_t)(seg + u) * NUM_Q_TOKENS;
      const float seg_max = partial_max[stat_base + s];
      // An empty segment carries a -inf max, which becomes a zero weight rather
      // than a skipped iteration, so the accumulation stays branch-free.
      const float weight =
          (seg_max == -INFINITY) ? 0.f : exp2_fast(seg_max - global_max);
      gsum[u] += partial_sum[stat_base + s] * weight;
      gacc[u] += partial_out[part_base + s * HEAD_DIM] * weight;
    }
  }
  for (; seg < num_kv_segments; ++seg) {
    const size_t s = (size_t)seg * NUM_Q_TOKENS;
    const float seg_max = partial_max[stat_base + s];
    const float weight =
        (seg_max == -INFINITY) ? 0.f : exp2_fast(seg_max - global_max);
    gsum[0] += partial_sum[stat_base + s] * weight;
    gacc[0] += partial_out[part_base + s * HEAD_DIM] * weight;
  }
  float global_sum = gsum[0], acc = gacc[0];
  #pragma unroll
  for (int u = 1; u < UNROLL; ++u) {
    global_sum += gsum[u];
    acc += gacc[u];
  }

  // Raw v_rcp_f32: global_sum is a sum of exp2 values in [0,1] with at least
  // one term equal to 1, so it lies in [1, num_kv_segments] -- 1 ulp is far
  // inside the fp16 output this feeds.  IEEE division would emit the full
  // Newton-Raphson.
  const float inv_sum =
      global_sum > 0.f ? __builtin_amdgcn_rcpf(global_sum) : 0.f;

  // `seq` indexes the output as well as the partials.  Without it every
  // sequence in a batch wrote the same [NUM_Q_TOKENS, Hq, HEAD_DIM] rows and
  // only the last block scheduled survived -- silently, since the launch is
  // valid and every sequence's partials are correct.  Q is read with the same
  // stride in the split kernel, so both ends now agree on
  // [num_seqs, NUM_Q_TOKENS, Hq, HEAD_DIM].
  // Via the traits rather than a cast: __hip_bfloat16 has no implicit
  // conversion from float, so `(T)x` compiles for half and not for bf16.
  out[(size_t)((seq * NUM_Q_TOKENS + m) * Hq + q_head) * HEAD_DIM + d] =
      elem_traits<T>::from_float(acc * inv_sum);
#endif  // VLLM_RDNA35_MHA_REAL_BODY
}

}  // namespace rdna35_mha_common

#define RDNA35_MHA_INLINE __device__ __forceinline__

namespace rdna35_mha {

// exp2_fast, elem_traits and the reduce kernel are shared with the GQA and MQA
// kernels; see attn_wide_common.hip.
using rdna35_mha_common::elem_traits;
using rdna35_mha_common::exp2_fast;

// Sums across the LANES lanes that share one KV token.  The XOR butterfly
// leaves the total in every participating lane, and only the steps the group
// needs are emitted -- at LANES=8 that is three instructions, against the five
// a whole-wave reduction costs.  Same shape as attn_gqa_wide's
// lane_group_sum; the two kernels agree on the encoding.
template <int LANES>
RDNA35_MHA_INLINE float lane_group_sum(float v) {
  static_assert(LANES == 4 || LANES == 8 || LANES == 16 || LANES == 32,
                "reduction supports 4, 8, 16 or 32 lanes per KV token");
  if constexpr (LANES >= 2)
    v += __builtin_amdgcn_mov_dpp(v, 0x161, 0xf, 0xf, 1);  // row_xmask:1
  if constexpr (LANES >= 4)
    v += __builtin_amdgcn_mov_dpp(v, 0x162, 0xf, 0xf, 1);  // row_xmask:2
  if constexpr (LANES >= 8)
    v += __builtin_amdgcn_mov_dpp(v, 0x164, 0xf, 0xf, 1);  // row_xmask:4
  if constexpr (LANES >= 16)
    v += __builtin_amdgcn_mov_dpp(v, 0x168, 0xf, 0xf, 1);  // row_xmask:8
  if constexpr (LANES == 32) v += __shfl_xor(v, 16);
  return v;
}

RDNA35_MHA_INLINE float softcap_score(float score, float cap, float inv_cap) {
  const float x = score * inv_cap;
  const float e = __expf(-2.f * fabsf(x));
  const float tanh_x = (1.f - e) / (1.f + e);
  return cap * copysignf(tanh_x, x);
}

// The segment count is a runtime argument rather than a template parameter: it
// appears only in index arithmetic, never in an array bound, so it costs no
// registers, while templating it would multiply the instantiation count by
// every value the dispatch offers.  The cost is that a runtime divisor needs a
// magic-number reciprocal sequence where a compile-time one is a shift -- paid
// once per task, against a KV walk of thousands of tokens.
//
// The parameter order below is shared with attn_gqa_wide and attn_mqa_wide:
// element type, query tokens, head size, then whatever decomposition that
// kernel has, then UNROLL, USE_SOFTCAP, and the mode switches.
//
// HEADS_PER_WAVE exists because head_dim 64 needs it.  A wave has 32 lanes and
// one KV token's head is HEAD_DIM halves, so at HEADS_PER_WAVE=1 each lane gets
// HEAD_DIM/32 halves -- at HEAD_DIM=512 that is 32 B (a b128 pair), but at 64
// it is 4 B and the compiler emits global_load_b32, a quarter of the width the
// memory path wants.  Worse, the whole 32-lane wave then reduces one dot2 per
// token, so the 5-step DPP chain is ~83% of the inner loop.
//
// The paged KV layout is [num_blocks, 2, block_size, num_kv_heads, head_size],
// so for a fixed token ADJACENT KV HEADS ARE ADJACENT IN MEMORY.  MHA has one
// KV head per query head and therefore heads to spare: a wave takes
// HEADS_PER_WAVE of them as a single contiguous run, which both widens the load
// and shrinks the reduction to log2(32/HEADS_PER_WAVE) steps.
//
//   HEADS_PER_WAVE   1        2        4
//   lanes per head   32       16       8
//   bytes per lane   D/16     D/8      D/4      (D=64: 4, 8, 16)
//   reduction steps  5        4        3
//
// Only head_dim 64 groups: above it a single head already fills the lane, so
// there is no load width left to win and the tuned rule picks 1.
//
//   T                element type of Q/K/V
//   NUM_Q_TOKENS     query tokens (1 plain decode, 2-4 speculative)
//   HEAD_DIM         head size (64, 128, 256 or 512)
//   HEADS_PER_WAVE   adjacent KV heads one wave loads together
//   UNROLL           KV tokens loaded and scored per burst
//   USE_SOFTCAP      whether to apply the tanh cap
template <typename T, int NUM_Q_TOKENS, int HEAD_DIM, int HEADS_PER_WAVE,
          int UNROLL, bool USE_SOFTCAP>
__global__ __launch_bounds__(32) void rdna35_causal_mha(
    const T* __restrict__ q,        // [num_seqs, NUM_Q_TOKENS, Hq, HEAD_DIM]
    const T* __restrict__ k_cache,  // [nblocks, 2, block_size, Hkv, HEAD_DIM]
    const T* __restrict__ v_cache, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    // [num_seqs, Hq, num_kv_segments, NUM_Q_TOKENS, HEAD_DIM]
    float* __restrict__ partial_out, float* __restrict__ partial_max,
    float* __restrict__ partial_sum, const int Hq, const int Hkv,
    const int block_size, const int max_blocks_per_seq,
    const int num_kv_segments,
    // scale and softcap arrive pre-multiplied by log2(e) so the online softmax
    // can use the natively base-2 v_exp_f32; the reduce kernel must match.
    const float scale, const float softcap, const float inv_softcap,
    const int block_stride, const int num_seqs, const int num_tasks) {
  using Traits = elem_traits<T>;
  using vec2 = typename Traits::vec2;
  // The 32 lanes split into HEADS_PER_WAVE groups, one per KV head, so a group
  // is LANES_PER_HEAD lanes covering HEAD_DIM halves.
  constexpr int LANES_PER_HEAD = 32 / HEADS_PER_WAVE;
  constexpr int VEC2_PER_LANE = HEAD_DIM / (LANES_PER_HEAD * 2);
  static_assert(LANES_PER_HEAD * HEADS_PER_WAVE == 32,
                "HEADS_PER_WAVE must divide 32");
  static_assert(LANES_PER_HEAD >= 4,
                "the reduction supports at most 8 heads per wave");
  static_assert(VEC2_PER_LANE * LANES_PER_HEAD * 2 == HEAD_DIM,
                "HEAD_DIM must divide by LANES_PER_HEAD*2");

  const int lane = threadIdx.x;
  // Position within this lane's head, and which head of the group it serves.
  // At HEADS_PER_WAVE=1 head_slot is 0 and elem is the whole lane index, i.e.
  // exactly the old mapping.
  const int elem = lane % LANES_PER_HEAD;
  const int head_slot = lane / LANES_PER_HEAD;

  {
    if (blockIdx.x >= (unsigned)num_tasks) return;
    int task = blockIdx.x;
    const int seg = task % num_kv_segments;
    task /= num_kv_segments;
    // The head axis counts GROUPS of HEADS_PER_WAVE, not heads.
    const int head_group = task % (Hkv / HEADS_PER_WAVE);
    task /= (Hkv / HEADS_PER_WAVE);
    const int seq = task;

    // Adjacent KV heads are adjacent in memory, so the group is one contiguous
    // run of HEADS_PER_WAVE*HEAD_DIM halves and each lane reads its slice of
    // it.
    const int kv_head = head_group * HEADS_PER_WAVE + head_slot;
    const int q_head = kv_head;  // MHA: one query head per KV head

    const int seq_len = seq_lens[seq];
    const int ctx_len = seq_len - NUM_Q_TOKENS;
    const int tokens_per_seg =
        (seq_len + num_kv_segments - 1) / num_kv_segments;
    const int kv_beg = seg * tokens_per_seg;
    const int kv_end = min(kv_beg + tokens_per_seg, seq_len);
    // An empty segment still walks the epilogue with a (-inf, 0) contribution
    // rather than skipping ahead: the log-sum-exp already handles that, and
    // there is no LDS merge here to leave a peer waiting.
    const bool seg_empty = kv_beg >= kv_end;

    vec2 q_regs[NUM_Q_TOKENS][VEC2_PER_LANE];
#pragma unroll
    for (int m = 0; m < NUM_Q_TOKENS; ++m) {
      // `seq` indexes Q, matching the output write in rdna35_mha_reduce.  Both
      // omitted it before, so a batch read one sequence's Q and wrote one
      // sequence's rows while the partials -- which always carried seq -- were
      // per-sequence and correct.  Only num_seqs == 1 was ever exercised.
      const vec2* q_row = (const vec2*)__builtin_assume_aligned(
          q + (size_t)((seq * NUM_Q_TOKENS + m) * Hq + q_head) * HEAD_DIM, 16);
#pragma unroll
      for (int i = 0; i < VEC2_PER_LANE; ++i)
        q_regs[m][i] = q_row[elem * VEC2_PER_LANE + i];
    }

    float acc[NUM_Q_TOKENS][VEC2_PER_LANE * 2];
    float run_max[NUM_Q_TOKENS], run_sum[NUM_Q_TOKENS];
#pragma unroll
    for (int m = 0; m < NUM_Q_TOKENS; ++m) {
      run_max[m] = -INFINITY;
      run_sum[m] = 0.f;
#pragma unroll
      for (int i = 0; i < VEC2_PER_LANE * 2; ++i) acc[m][i] = 0.f;
    }

    vec2 k_burst[UNROLL][VEC2_PER_LANE], v_burst[UNROLL][VEC2_PER_LANE];

    // Lambdas capturing k_burst/v_burst by reference, not __device__ functions:
    // only capture keeps the burst in registers.
    //
    // The load only issues; every s_waitcnt lives in the compute lambda, so the
    // whole UNROLL-deep burst is in flight before any of it is touched.
    // CheckTag and SameBlockTag are compile-time tags because a runtime test
    // inside the unrolled body costs a per-step exec-mask block.
    auto load_burst = [&](auto CheckTag, auto SameBlockTag, int burst_start) {
      constexpr bool CHECK_BOUNDS = decltype(CheckTag)::value;
      constexpr bool SAME_BLOCK = decltype(SameBlockTag)::value;
      if constexpr (SAME_BLOCK) {
        // Every position in the burst lives in one KV block, so the block-table
        // lookup and the 64-bit base are computed once and the per-step delta
        // is the loop-invariant Hkv*HEAD_DIM.  block_size is a runtime
        // argument, so pos/block_size and pos%block_size would otherwise
        // compile to a magic-number division per step.
        const int block_id =
            block_table[seq * max_blocks_per_seq + burst_start / block_size];
        const size_t base =
            (size_t)block_id * block_stride +
            (size_t)(burst_start % block_size) * Hkv * HEAD_DIM +
            (size_t)kv_head * HEAD_DIM;
        const size_t row_step = (size_t)Hkv * HEAD_DIM;
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
          if constexpr (CHECK_BOUNDS) {
            // Zero rather than break: the two-pass form below walks every slot
            // and multiplies by exp2(-inf - new_max) = 0, and 0 * uninitialised
            // is NaN the moment a stale register holds one.
            if (burst_start + u >= kv_end) {
#pragma unroll
              for (int i = 0; i < VEC2_PER_LANE; ++i) {
                k_burst[u][i] = Traits::zero2();
                v_burst[u][i] = Traits::zero2();
              }
              continue;
            }
          }
          const vec2* k_row = (const vec2*)__builtin_assume_aligned(
              k_cache + base + u * row_step, 16);
          const vec2* v_row = (const vec2*)__builtin_assume_aligned(
              v_cache + base + u * row_step, 16);
#pragma unroll
          for (int i = 0; i < VEC2_PER_LANE; ++i)
            k_burst[u][i] = k_row[elem * VEC2_PER_LANE + i];
#pragma unroll
          for (int i = 0; i < VEC2_PER_LANE; ++i)
            v_burst[u][i] = v_row[elem * VEC2_PER_LANE + i];
        }
      } else {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
          const int pos = burst_start + u;
          if constexpr (CHECK_BOUNDS) {
            if (pos >= kv_end) {  // see the SAME_BLOCK arm above
#pragma unroll
              for (int i = 0; i < VEC2_PER_LANE; ++i) {
                k_burst[u][i] = Traits::zero2();
                v_burst[u][i] = Traits::zero2();
              }
              continue;
            }
          }
          const int block_id =
              block_table[seq * max_blocks_per_seq + pos / block_size];
          const size_t base = (size_t)block_id * block_stride +
                              (size_t)(pos % block_size) * Hkv * HEAD_DIM +
                              (size_t)kv_head * HEAD_DIM;
          const vec2* k_row =
              (const vec2*)__builtin_assume_aligned(k_cache + base, 16);
          const vec2* v_row =
              (const vec2*)__builtin_assume_aligned(v_cache + base, 16);
#pragma unroll
          for (int i = 0; i < VEC2_PER_LANE; ++i)
            k_burst[u][i] = k_row[elem * VEC2_PER_LANE + i];
#pragma unroll
          for (int i = 0; i < VEC2_PER_LANE; ++i)
            v_burst[u][i] = v_row[elem * VEC2_PER_LANE + i];
        }
      }
    };

    // Score the whole burst, then rescale once.  Past the first few tokens the
    // running max rarely moves, so the per-token `acc *= rescale` is almost
    // always a multiply by 1.0 across NUM_Q_TOKENS*VEC2_PER_LANE*2
    // accumulators.  Costs NUM_Q_TOKENS*UNROLL scores held live across the two
    // passes, which is affordable while NUM_Q_TOKENS*VEC2_PER_LANE is small.
    auto compute_burst = [&](auto CheckTag, int burst_start) {
      constexpr bool CHECK_BOUNDS = decltype(CheckTag)::value;
      float scores[NUM_Q_TOKENS][UNROLL];
      float burst_max[NUM_Q_TOKENS];
#pragma unroll
      for (int m = 0; m < NUM_Q_TOKENS; ++m) burst_max[m] = -INFINITY;
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const int pos = burst_start + u;
        const bool in_range = !CHECK_BOUNDS || pos < kv_end;
#pragma unroll
        for (int m = 0; m < NUM_Q_TOKENS; ++m) {
          float score = 0.f;
#pragma unroll
          for (int i = 0; i < VEC2_PER_LANE; ++i)
            score = Traits::dot2(q_regs[m][i], k_burst[u][i], score);
          score = lane_group_sum<LANES_PER_HEAD>(score) * scale;
          if constexpr (USE_SOFTCAP)
            score = softcap_score(score, softcap, inv_softcap);
          if (pos > ctx_len + m) score = -INFINITY;
          if constexpr (CHECK_BOUNDS) {
            if (!in_range) score = -INFINITY;
          }
          scores[m][u] = score;
          burst_max[m] = fmaxf(burst_max[m], score);
        }
      }

#pragma unroll
      for (int m = 0; m < NUM_Q_TOKENS; ++m) {
        const float new_max = fmaxf(run_max[m], burst_max[m]);
        // A non-empty segment can still be entirely masked, leaving both at
        // -inf; -inf - -inf is NaN, so skip rather than rescale by it.
        if (new_max == -INFINITY) continue;
        const float rescale = exp2_fast(run_max[m] - new_max);
        run_max[m] = new_max;
        float sum = run_sum[m] * rescale;
#pragma unroll
        for (int i = 0; i < VEC2_PER_LANE * 2; ++i) acc[m][i] *= rescale;
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
          const float weight = exp2_fast(scores[m][u] - new_max);
          sum += weight;
          // P.V through dot2 against a one-hot weight (w, 0): dot2 selects
          // v's low half, and the swapped (0, w) selects the high one.  The
          // natural `acc += weight * float(v)` form lowers to v_fma_mix_f32,
          // which is VOP3P packed math and so cannot be VOPD-paired, while
          // v_dot2acc_f32_f16 pairs freely.
          const vec2 w2 = Traits::pack_weight(weight);
#pragma unroll
          for (int i = 0; i < VEC2_PER_LANE; ++i)
            Traits::accum_pv(acc[m][2 * i], acc[m][2 * i + 1], w2,
                             v_burst[u][i]);
        }
        run_sum[m] = sum;
      }
    };

    const int full_burst_end =
        seg_empty ? kv_beg : kv_beg + ((kv_end - kv_beg) / UNROLL) * UNROLL;
    for (int pos = kv_beg; pos < full_burst_end; pos += UNROLL) {
      // A burst that lies inside one page hoists the block-table lookup and the
      // 64-bit base out of the burst, leaving a loop-invariant row stride.
      if (pos % block_size + UNROLL <= block_size)
        load_burst(std::false_type{}, std::true_type{}, pos);
      else
        load_burst(std::false_type{}, std::false_type{}, pos);
      compute_burst(std::false_type{}, pos);
    }
    if (full_burst_end < kv_end) {
      if (full_burst_end % block_size + UNROLL <= block_size)
        load_burst(std::true_type{}, std::true_type{}, full_burst_end);
      else
        load_burst(std::true_type{}, std::false_type{}, full_burst_end);
      compute_burst(std::true_type{}, full_burst_end);
    }

    // The wave owns this (head, segment) partial outright, so it writes
    // straight out: no LDS staging, no barrier.
    const size_t out_base =
        (((size_t)(seq * Hq + q_head) * num_kv_segments + seg) * NUM_Q_TOKENS) *
        (size_t)HEAD_DIM;
#pragma unroll
    for (int m = 0; m < NUM_Q_TOKENS; ++m) {
#pragma unroll
      for (int i = 0; i < VEC2_PER_LANE; ++i) {
        partial_out[out_base + (size_t)m * HEAD_DIM +
                    (elem * VEC2_PER_LANE + i) * 2] = acc[m][2 * i];
        partial_out[out_base + (size_t)m * HEAD_DIM +
                    (elem * VEC2_PER_LANE + i) * 2 + 1] = acc[m][2 * i + 1];
      }
      // elem == 0, NOT lane == 0: each head group owns its own stats, and with
      // HEADS_PER_WAVE > 1 a single lane would write only head_slot 0's,
      // leaving the rest of the group's partials uninitialised for the reduce.
      if (elem == 0) {
        const int stat_index =
            ((seq * Hq + q_head) * num_kv_segments + seg) * NUM_Q_TOKENS + m;
        partial_max[stat_index] = run_max[m];
        partial_sum[stat_index] = run_sum[m];
      }
    }
  }
}

// The reduce pass that merges the per-segment partials lives in
// attn_wide_common.hip: all three split kernels write the same
// [num_seqs, Hq, num_kv_segments, NUM_Q_TOKENS, HEAD_DIM] layout, so they
// share one reduce kernel rather than three identical copies.

}  // namespace rdna35_mha

namespace {

// The shipped config.  The op takes num_kv_segments == 0 to mean "use this",
// so callers never choose one themselves.
//
// A rule rather than a lookup table: over the tuned grid it lands within a few
// percent of the best config per cell, where the spread between the best and
// worst config of a cell reaches 37x, and a table keyed on context length would
// buy about 1% for an entry per (head count, head_dim, M, ctx).
//
// The segment count is NOT constrained to a power of two and must not be
// rounded to one.  It targets a constant task count -- about 6.4 tasks per SIMD
// across the 80 SIMDs on gfx1151 -- and rounding down costs 11-43% of the
// parallelism at the head counts where it bites (Hq=40 wants 6, Hq=48 wants 5,
// Hq=71 wants 7).  Both kernels take it at runtime for that reason.
bool rdna35_mha_tuned(int num_q_tokens, int head_dim, int Hkv,
                      int* num_kv_segments, int* unroll, int* heads_per_wave) {
  const int vec2_per_lane = head_dim / 64;
  if (vec2_per_lane < 1 || vec2_per_lane * 64 != head_dim) return false;

  // Head grouping first: it changes how many tasks there are.  2 heads per wave
  // below 16 heads and 4 at or above -- both give a b128 lane at head_dim 64,
  // and below the crossover grouping by 4 would leave too few groups (Hq=8
  // gives two).  Above head_dim 64 a single head already fills the lane.
  int hpw = 1;
  if (vec2_per_lane == 1) {
    const int want = (Hkv <= 16) ? 2 : 4;
    for (int c = want; c >= 2; c >>= 1) {
      if (Hkv % c == 0) {
        hpw = c;
        break;
      }
    }
  }

  // The numerator is 256 at head_dim 64 and 512 above it: a grouped wave covers
  // hpw heads per step, so the same task count is reached with fewer segments.
  // It is conditioned rather than lowered outright because 256 costs geomean
  // 1.070 (worst 1.305) at head_dim >= 128, against 1.012 / 1.086 for 512.
  //
  // The count stays keyed on Hkv, not on the group count.  Holding the task
  // count near its ungrouped value by dividing by the groups instead measures
  // worse everywhere it was tried: a grouped wave does hpw tokens' worth of
  // work per step, so fewer, fatter tasks saturate much like more, thinner
  // ones, and the extra segments only fragment the KV walk.
  const int numerator = (vec2_per_lane == 1) ? 256 : 512;
  int segments = numerator / (Hkv * vec2_per_lane);

  // The cap is 32 at head_dim 64 and 16 above it, for the same reason: at
  // head_dim 64 vec2_per_lane is 1, so the quotient is 8x larger and a cap of
  // 16 would bind on every shape.  Conditioned for the same reason as well --
  // 32 everywhere costs a little at head_dim >= 128.
  const int cap = (vec2_per_lane == 1) ? 32 : 16;
  segments = segments < 1 ? 1 : (segments > cap ? cap : segments);

  const int q_times_vec2 = num_q_tokens * vec2_per_lane;
  *num_kv_segments = segments;
  *heads_per_wave = hpw;
  *unroll = (q_times_vec2 <= 4 || (Hkv * segments <= 128 && q_times_vec2 <= 8))
                ? 4
                : 2;
  return true;
}

// THREADS_PER_BLOCK for the reduce, derived from HEAD_DIM.  The kernel is one
// thread per output element, so this only changes how the same total threads
// are grouped, and the value below is the fastest at each head size.
//
// At HEAD_DIM 64 it is a HARD CAP rather than a preference: with
// THREADS_PER_BLOCK > HEAD_DIM the surplus threads compute d >= HEAD_DIM and
// write past the end of the row.  The kernel's static_assert does not catch it
// (256 % 64 == 0 passes), so this function is the only guard.
constexpr int reduce_tpb(int head_dim) {
  return head_dim >= 256 ? 256 : (head_dim >= 128 ? 128 : 64);
}

}  // namespace

// Dispatch over (element type, num_q_tokens, head_dim, head group, unroll,
// softcap).  Only M in {1,4} is instantiated -- M=1 is plain decode and M=4
// speculative -- and the host pads query lengths 2 and 3 up to 4 rather than
// doubling the build for them.  Padding costs a few percent and keeps every
// launch on a tuned configuration.
#define RDNA35_MHA_LAUNCH(T_, M_, D_, HPW_, UNRL_, SOFTCAP_)                \
  do {                                                                      \
    /* One task per (seq, head GROUP, segment): a wave owns HPW_ heads. */  \
    const int num_tasks = num_seqs * (num_kv_heads / (HPW_)) * nseg;        \
    rdna35_mha::rdna35_causal_mha<T_, M_, D_, HPW_, UNRL_, SOFTCAP_>        \
        <<<dim3(num_tasks), dim3(32), 0, stream>>>(                         \
            (const T_*)query.data_ptr(), (const T_*)key_cache.data_ptr(),   \
            (const T_*)value_cache.data_ptr(),                              \
            block_table.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),  \
            partial_out.data_ptr<float>(), partial_max.data_ptr<float>(),   \
            partial_sum.data_ptr<float>(), num_heads, num_kv_heads,         \
            block_size, max_blocks_per_seq, nseg, scale_log2, softcap_log2, \
            inv_softcap, block_stride, num_seqs, num_tasks);                \
    rdna35_mha_common::rdna35_mha_reduce<T_, M_, D_, reduce_tpb(D_)>        \
        <<<dim3((D_ / reduce_tpb(D_)) * M_, num_heads, num_seqs),           \
           dim3(reduce_tpb(D_)), 0, stream>>>(                              \
            partial_out.data_ptr<float>(), partial_max.data_ptr<float>(),   \
            partial_sum.data_ptr<float>(), (T_*)out.data_ptr(), num_heads,  \
            nseg);                                                          \
    return;                                                                 \
  } while (0)

#define RDNA35_MHA_CASE(T_, M_, D_, HPW_, UNRL_)                          \
  do {                                                                    \
    if (num_q_tokens == (M_) && head_size == (D_) && unroll == (UNRL_) && \
        hpw == (HPW_) && num_kv_heads % (HPW_) == 0) {                    \
      if (softcap > 0.f)                                                  \
        RDNA35_MHA_LAUNCH(T_, M_, D_, HPW_, UNRL_, true);                 \
      else                                                                \
        RDNA35_MHA_LAUNCH(T_, M_, D_, HPW_, UNRL_, false);                \
    }                                                                     \
  } while (0)

// The rule only ever emits unroll 2 or 4; 1 is instantiated because the
// dispatch is keyed on the value the rule returns and a stale rule should fail
// loudly rather than silently pick a neighbour.
#define RDNA35_MHA_GRID(T_, M_, D_, HPW_) \
  do {                                    \
    RDNA35_MHA_CASE(T_, M_, D_, HPW_, 1); \
    RDNA35_MHA_CASE(T_, M_, D_, HPW_, 2); \
    RDNA35_MHA_CASE(T_, M_, D_, HPW_, 4); \
  } while (0)

#define RDNA35_MHA_ALL_SHAPES(T_)                                            \
  do {                                                                       \
    /* head_dim 64 builds the grouped arms too; above it a single head fills \
       the lane, so only the ungrouped one exists. */                        \
    RDNA35_MHA_GRID(T_, 1, 64, 1);                                           \
    RDNA35_MHA_GRID(T_, 1, 64, 2);                                           \
    RDNA35_MHA_GRID(T_, 1, 64, 4);                                           \
    RDNA35_MHA_GRID(T_, 4, 64, 1);                                           \
    RDNA35_MHA_GRID(T_, 4, 64, 2);                                           \
    RDNA35_MHA_GRID(T_, 4, 64, 4);                                           \
    RDNA35_MHA_GRID(T_, 1, 128, 1);                                          \
    RDNA35_MHA_GRID(T_, 4, 128, 1);                                          \
    RDNA35_MHA_GRID(T_, 1, 256, 1);                                          \
    RDNA35_MHA_GRID(T_, 4, 256, 1);                                          \
    RDNA35_MHA_GRID(T_, 1, 512, 1);                                          \
    RDNA35_MHA_GRID(T_, 4, 512, 1);                                          \
  } while (0)

// query/out: [num_seqs, num_q_tokens, num_heads, head_size]
// key_cache/value_cache: [num_blocks, block_size, num_kv_heads, head_size]
//   views into vLLM's [num_blocks, 2, ...] allocation, so block_stride is taken
//   from stride(0) rather than recomputed.
// seq_lens[i] is the TOTAL length including the num_q_tokens new tokens.
//
// Every constraint is a TORCH_CHECK rather than a status code: the Python
// predicate (rdna35_causal_mha_attn.can_run) is expected to have screened the
// call, so reaching here with a bad shape is a bug, and the failure mode this
// replaces -- returning -1 with `out` left untouched -- is silently wrong
// output.
void rdna35_causal_mha_attn(torch::Tensor& out, torch::Tensor& query,
                            torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& block_table, torch::Tensor& seq_lens,
                            torch::Tensor& partial_out,
                            torch::Tensor& partial_max,
                            torch::Tensor& partial_sum, double scale,
                            double softcap) {
  TORCH_CHECK(query.dim() == 4,
              "query must be [num_seqs, M, num_heads, head_size]");
  TORCH_CHECK(out.sizes() == query.sizes(), "out must match query shape");
  TORCH_CHECK(query.scalar_type() == out.scalar_type(),
              "out dtype must match query");
  TORCH_CHECK(key_cache.scalar_type() == query.scalar_type() &&
                  value_cache.scalar_type() == query.scalar_type(),
              "KV cache dtype must match query (no quantized KV)");
  TORCH_CHECK(query.is_contiguous() && out.is_contiguous(),
              "query and out must be contiguous");

  const int num_seqs = static_cast<int>(query.size(0));
  const int num_q_tokens = static_cast<int>(query.size(1));
  const int num_heads = static_cast<int>(query.size(2));
  const int head_size = static_cast<int>(query.size(3));

  TORCH_CHECK(
      key_cache.dim() == 4 && value_cache.dim() == 4,
      "kv cache must be [num_blocks, block_size, num_kv_heads, head_size]");
  const int block_size = static_cast<int>(key_cache.size(1));
  const int num_kv_heads = static_cast<int>(key_cache.size(2));
  TORCH_CHECK(
      key_cache.size(3) == head_size && value_cache.size(3) == head_size,
      "kv cache head_size must match query");
  TORCH_CHECK(
      value_cache.size(1) == block_size && value_cache.size(2) == num_kv_heads,
      "key and value caches must have the same layout");

  // MHA only: the kernel maps one query head per KV head by construction.
  TORCH_CHECK(num_heads == num_kv_heads,
              "rdna35_causal_mha_attn is MHA-only: num_heads (", num_heads,
              ") must equal num_kv_heads (", num_kv_heads, ")");
  TORCH_CHECK(
      num_q_tokens == 1 || num_q_tokens == 4,
      "only M=1 and M=4 are instantiated (the host pads 2 and 3 to 4), got ",
      num_q_tokens);

  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == num_seqs,
              "block_table must be [num_seqs, max_blocks_per_seq]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 &&
                  seq_lens.scalar_type() == torch::kInt32,
              "block_table and seq_lens must be int32");
  TORCH_CHECK(seq_lens.numel() >= num_seqs, "seq_lens shorter than num_seqs");
  const int max_blocks_per_seq = static_cast<int>(block_table.size(1));

  // Both caches are views into one [num_blocks, 2, block_size, H, D] tensor, so
  // the element stride between blocks is 2x the per-cache block extent.  Taking
  // it from the tensor keeps a separate-allocation layout working too.
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0),
              "key and value caches must share a block stride");
  const int64_t block_stride_64 = key_cache.stride(0);
  TORCH_CHECK(block_stride_64 <= std::numeric_limits<int>::max(),
              "block stride overflows int");
  const int block_stride = static_cast<int>(block_stride_64);

  int nseg = 0, unroll = 0, hpw = 1;
  TORCH_CHECK(rdna35_mha_tuned(num_q_tokens, head_size, num_kv_heads, &nseg,
                               &unroll, &hpw),
              "no tuned config for head_size ", head_size);
  TORCH_CHECK(hpw >= 1 && num_kv_heads % hpw == 0, "head group ", hpw,
              " does not divide num_kv_heads ", num_kv_heads);

  // The partials are indexed [num_seqs, num_heads, nseg, M, head_size], so a
  // segment count that disagrees with the launch is a memory fault rather than
  // a compile error.  Check it here instead.
  TORCH_CHECK(partial_out.dim() == 5 && partial_max.dim() == 4 &&
                  partial_sum.dim() == 4,
              "partial buffers have the wrong rank");
  TORCH_CHECK(
      partial_out.size(0) >= num_seqs && partial_out.size(1) == num_heads &&
          partial_out.size(2) >= nseg && partial_out.size(3) == num_q_tokens &&
          partial_out.size(4) == head_size,
      "partial_out must be [>=num_seqs, num_heads, >=", nseg, ", ",
      num_q_tokens, ", ", head_size, "]");
  TORCH_CHECK(partial_out.size(2) == nseg, "partial buffers were sized for ",
              partial_out.size(2), " KV segments but the tuned config wants ",
              nseg);
  TORCH_CHECK(partial_out.scalar_type() == torch::kFloat32 &&
                  partial_max.scalar_type() == torch::kFloat32 &&
                  partial_sum.scalar_type() == torch::kFloat32,
              "partial buffers must be float32");
  // Both kernels index the partials with raw pointer arithmetic, so a strided
  // view -- e.g. slicing the M dimension off a workspace allocated at M=4 --
  // would silently read the wrong elements rather than fault.
  TORCH_CHECK(partial_out.is_contiguous() && partial_max.is_contiguous() &&
                  partial_sum.is_contiguous(),
              "partial buffers must be contiguous");
  // NOT is_contiguous(): the caches are k/v views into one [num_blocks, 2, ...]
  // allocation, so they are strided across blocks by construction -- that gap
  // is exactly what block_stride carries. What the addressing does require is
  // that each block be dense internally.
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1 &&
                  key_cache.stride(2) == head_size &&
                  value_cache.stride(2) == head_size &&
                  key_cache.stride(1) == (int64_t)num_kv_heads * head_size &&
                  value_cache.stride(1) == (int64_t)num_kv_heads * head_size,
              "kv cache blocks must be dense [block_size, num_kv_heads, "
              "head_size] (NHD); got key strides ",
              key_cache.strides());
  TORCH_CHECK(block_table.is_contiguous() && seq_lens.is_contiguous(),
              "block_table and seq_lens must be contiguous");

  // Fold log2(e) into the scale so the softmax can use the natively base-2
  // v_exp_f32.  cap*tanh(s/cap) is homogeneous of degree 1, so scaling the cap
  // by the same factor keeps the softcap exact.
  constexpr float LOG2E = 1.44269504088896340736f;
  const float scale_log2 = static_cast<float>(scale) * LOG2E;
  const float softcap_log2 = static_cast<float>(softcap) * LOG2E;
  const float inv_softcap = softcap_log2 > 0.f ? 1.f / softcap_log2 : 0.f;

  // The hipified stream accessor, as moe_gemm_w4a16_wmma.cu uses: the
  // c10/cuda headers pull in a cuda_cmake_macros.h that is generated at torch
  // build time and absent from the wheel.
  const hipStream_t stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA();

  if (query.scalar_type() == at::ScalarType::Half) {
    RDNA35_MHA_ALL_SHAPES(half);
  } else if (query.scalar_type() == at::ScalarType::BFloat16) {
    RDNA35_MHA_ALL_SHAPES(__hip_bfloat16);
  } else {
    TORCH_CHECK(false,
                "rdna35_causal_mha_attn supports float16 and bfloat16, got ",
                query.scalar_type());
  }

  TORCH_CHECK(false,
              "rdna35_causal_mha_attn: no instantiation for M=", num_q_tokens,
              " head_size=", head_size, " unroll=", unroll);
}

#undef RDNA35_MHA_ALL_SHAPES
#undef RDNA35_MHA_GRID
#undef RDNA35_MHA_CASE
#undef RDNA35_MHA_LAUNCH
