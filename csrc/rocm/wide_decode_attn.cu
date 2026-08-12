// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Wide decode attention for MHA (num_q_heads == num_kv_heads) on AMD RDNA3.5
// (gfx1151).  A split/reduce pair: attn_mha_wide scores one (sequence, KV head,
// KV segment) per wave into fp32 partials, attn_wide_reduce merges them with
// the usual log-sum-exp rescale.
//
// Ported from the standalone attn_decode_hip/ harness, which is where it was
// developed and tuned.  Against Triton's unified attention over 120 cells
// (head counts 8/16/32/64 x head_dim 128/256/512 x M 1/4 x ctx 128..65536) it
// measures 1.0-4.7x on the 2D path and 0.98-6.0x on the 3D one, reaching
// 90-95% of the 238 GiB/s streaming peak at long context.
//
// Restrictions the host side MUST enforce before calling (see
// vllm/v1/attention/ops/rocm_wide_decode_attn.py, which mirrors this list):
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
  #define VLLM_WIDE_ATTN_REAL_BODY 1
#endif



#define ATTN_WIDE_DEVICE_INLINE __device__ __forceinline__

namespace attn_wide {

// v_exp_f32 is natively base-2, so the online softmax works in the log2 domain
// and scale/softcap arrive pre-multiplied by log2(e); __expf would emit a fixup
// multiply on every call.
ATTN_WIDE_DEVICE_INLINE float exp2_fast(float x) {
  return __builtin_amdgcn_exp2f(x);
}

// vec2 is the 32-bit pair the dot2 instruction consumes.  Adding bf16 means
// adding one specialisation here: gfx1151 has v_dot2_f32_bf16 as well.
template <typename T>
struct elem_traits;

template <>
struct elem_traits<half> {
  using vec2 = half2;
  static ATTN_WIDE_DEVICE_INLINE float dot2(const vec2 a, const vec2 b,
                                            float acc) {
    return __builtin_amdgcn_fdot2(a, b, acc, false);
  }
  static ATTN_WIDE_DEVICE_INLINE float lo(const vec2 v) {
    return __half2float(v.x);
  }
  static ATTN_WIDE_DEVICE_INLINE float hi(const vec2 v) {
    return __half2float(v.y);
  }
  static ATTN_WIDE_DEVICE_INLINE vec2 zero2() { return __float2half2_rn(0.f); }
  static ATTN_WIDE_DEVICE_INLINE half from_float(float v) {
    return __float2half_rn(v);
  }

  // The three below serve attn_mqa_wide's MQAW_PV_FP16 ablation only.  They are
  // __forceinline__ statics, so the kernels that never call them emit nothing
  // for them.
  //
  // The softmax weight as the one-hot pair (weight, 0), so the PV product can
  // be a dot2.  This is Triton's scheme: triton_unified_attention.py does
  // `acc += tl.dot(P.to(V.dtype), V)`, rounding the weight to the KV dtype so
  // the matmul takes both operands in fp16.
  static ATTN_WIDE_DEVICE_INLINE vec2 pack_weight(float weight) {
    return __halves2half2(__float2half_rn(weight), (half)0.f);
  }
  static ATTN_WIDE_DEVICE_INLINE float unpack_weight(const vec2 weight) {
    return __half2float(weight.x);
  }

  // acc.lo += weight*v.lo, acc.hi += weight*v.hi, as two dot2 against the
  // one-hot weight.  pack_weight builds (weight, 0), so dot2 against it selects
  // v's LOW half; the swapped (0, weight) selects the high one.
  static ATTN_WIDE_DEVICE_INLINE void accum_pv(float& lo_acc, float& hi_acc,
                                               const vec2 weight,
                                               const vec2 v) {
    lo_acc = __builtin_amdgcn_fdot2(weight, v, lo_acc, false);
    hi_acc = __builtin_amdgcn_fdot2(__lowhigh2highlow(weight), v, hi_acc, false);
  }
};

// bf16, as the comment above anticipated: gfx1151 has v_dot2_f32_bf16 next to
// the fp16 one, so this is the same shape of specialisation.  The builtin takes
// the pair as a short2 bit pattern rather than a bf16 vector type, hence the
// reinterpret; __hip_bfloat162 is layout-compatible with it (two 16-bit lanes,
// low first).
//
// Accuracy is NOT the same as fp16 and the caller should not expect it to be:
// bf16 carries 8 mantissa bits against fp16's 11, so Q.K accumulates about 8x
// the rounding error per element.  The accumulators and the entire softmax stay
// fp32 -- only the Q/K/V operands and the output are narrowed -- so the error is
// input quantisation, not a change of algorithm.  The tests use a separate
// tolerance for it.
template <>
struct elem_traits<__hip_bfloat16> {
  using vec2 = __hip_bfloat162;
  // The builtin takes a native 2-wide short vector.  HIP's short2 is a
  // HIP_vector_type class, which does not convert to it, and __hip_bfloat162 is
  // a struct so __builtin_bit_cast rejects it as not trivially copyable -- so
  // read the two raw halves and build the vector explicitly.
  using short2_native = short __attribute__((ext_vector_type(2)));
  static ATTN_WIDE_DEVICE_INLINE short2_native as_short2(const vec2 v) {
    short2_native out;
    out.x = static_cast<short>(__bfloat16_as_ushort(v.x));
    out.y = static_cast<short>(__bfloat16_as_ushort(v.y));
    return out;
  }
  static ATTN_WIDE_DEVICE_INLINE float dot2(const vec2 a, const vec2 b,
                                            float acc) {
    return __builtin_amdgcn_fdot2_f32_bf16(as_short2(a), as_short2(b), acc,
                                           false);
  }
  static ATTN_WIDE_DEVICE_INLINE float lo(const vec2 v) {
    return __bfloat162float(v.x);
  }
  static ATTN_WIDE_DEVICE_INLINE float hi(const vec2 v) {
    return __bfloat162float(v.y);
  }
  static ATTN_WIDE_DEVICE_INLINE vec2 zero2() {
    return __hip_bfloat162(__float2bfloat16(0.f), __float2bfloat16(0.f));
  }
  static ATTN_WIDE_DEVICE_INLINE __hip_bfloat16 from_float(float v) {
    return __float2bfloat16(v);
  }

  // The P.V path, mirroring the fp16 traits: the softmax weight as the one-hot
  // pair (weight, 0) so the product can be a dot2 rather than an fma_mix, which
  // VOPD cannot pair.  dot2 against (w, 0) selects v's LOW half; the swapped
  // (0, w) selects the high one.
  static ATTN_WIDE_DEVICE_INLINE vec2 pack_weight(float weight) {
    return __hip_bfloat162(__float2bfloat16(weight), __float2bfloat16(0.f));
  }
  static ATTN_WIDE_DEVICE_INLINE float unpack_weight(const vec2 weight) {
    return __bfloat162float(weight.x);
  }
  static ATTN_WIDE_DEVICE_INLINE void accum_pv(float& lo_acc, float& hi_acc,
                                               const vec2 weight,
                                               const vec2 v) {
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
// The segment count is a RUNTIME argument here too, which it was not before.
// It used to be a template parameter because the per-segment maxes and weights
// lived in register arrays, and an array bound must be a constant.  That forced
// the dispatch to switch over an instantiated set -- {1,2,4,8,16} -- while
// attn_mha_wide_tuned computes clamp(512/(Hkv*vec2), 1, 16), which is NOT always
// one of those: Hq=40 gives 6, Hq=48 gives 5, Hq=71 gives 7.  On that mismatch
// the split kernel had already launched and the reduce dispatch returned -1,
// leaving `out` holding whatever it held before -- wrong output, no error.  The
// affected head counts are ordinary MHA models (Llama-2-13B, Qwen-14B, OPT-13B,
// Baichuan-13B at Hq=40), so this was not a corner case.
//
// Rounding the count down to a power of two would have kept the arrays, but the
// count is chosen to hit a target task count (~512/vec2, about 6.4 tasks per
// SIMD across the 80 SIMDs); rounding costs 11-43% of the parallelism.
//
// So the arrays go instead, in exchange for re-reading partial_max: pass one
// takes the max, pass two accumulates the sum and the output together.  That is
// affordable because this kernel is 0.04-0.8% of the traffic the split kernel
// moves -- the re-read is a few KB against hundreds of MB of KV.  One
// instantiation now covers every count, and the dispatch can no longer fail.
template <typename T, int NUM_Q_TOKENS, int HEAD_DIM, int THREADS_PER_BLOCK>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void attn_wide_reduce(
    const float* __restrict__ partial_out, const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum, T* __restrict__ out, const int Hq,
    const int num_kv_segments) {
#if !defined(VLLM_WIDE_ATTN_REAL_BODY)
  // Non-gfx11 device pass: a stub so the TU links in multi-arch builds.  The
  // op is never called off gfx11 (Python gates on on_gfx1151()).
  return;
#else
  static_assert(HEAD_DIM % THREADS_PER_BLOCK == 0
                    || THREADS_PER_BLOCK % HEAD_DIM == 0,
                "THREADS_PER_BLOCK must tile HEAD_DIM");
  constexpr int DIM_CHUNKS =
      (HEAD_DIM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  const int chunk = blockIdx.x % DIM_CHUNKS;
  const int m     = (blockIdx.x / DIM_CHUNKS) % NUM_Q_TOKENS;
  const int q_head = blockIdx.y;
  const int seq    = blockIdx.z;
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
      ((size_t)(seq * Hq + q_head) * num_kv_segments * NUM_Q_TOKENS + m)
          * HEAD_DIM
      + d;

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
#endif  // VLLM_WIDE_ATTN_REAL_BODY
}

}  // namespace attn_wide




#define MHA_DEVICE_INLINE __device__ __forceinline__

namespace mha_wide {

// exp2_fast, elem_traits and the reduce kernel are shared with the GQA and MQA
// kernels; see attn_wide_common.hip.
using attn_wide::exp2_fast;
using attn_wide::elem_traits;

// Sums across the LANES lanes that share one KV token.  The XOR butterfly
// leaves the total in every participating lane, and only the steps the group
// needs are emitted -- at LANES=8 that is three instructions, against the five
// a whole-wave reduction costs.  Same shape as attn_gqa_wide's
// lane_group_sum; the two kernels agree on the encoding.
template <int LANES>
MHA_DEVICE_INLINE float lane_group_sum(float v) {
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

MHA_DEVICE_INLINE float softcap_score(float score, float cap, float inv_cap) {
  const float x = score * inv_cap;
  const float e = __expf(-2.f * fabsf(x));
  const float tanh_x = (1.f - e) / (1.f + e);
  return cap * copysignf(tanh_x, x);
}

// The segment count is a runtime argument, not a template parameter -- the same
// choice attn_gqa_wide and attn_mqa_wide make, and for the same reason.  It
// appears only in index arithmetic, never in an array bound, so it costs no
// registers, while templating it multiplies the instantiation count by however
// many values the dispatch offers.
//
// It is NOT free: a compile-time divisor turns the two divisions below into
// shifts, where a runtime one needs a magic-number reciprocal sequence.
// Measured on gfx1151, that prologue is 108 instructions against 66.  But it is
// a prologue -- once per task, against a KV walk of thousands of tokens -- and
// paying it buys back a 5x cut in build size here and keeps the three kernels
// saying the same thing.  The reduce kernel still takes the count at compile
// time, because it holds that many maxes in registers; see attn_wide_common.
//
// The parameter order below is shared with attn_gqa_wide and attn_mqa_wide:
// element type, query tokens, head size, then whatever decomposition that
// kernel has, then UNROLL, USE_SOFTCAP, and the mode switches.
//
// HEADS_PER_WAVE exists because head_dim 64 needs it.  A wave has 32 lanes and
// one KV token's head is HEAD_DIM halves, so at HEADS_PER_WAVE=1 each lane gets
// HEAD_DIM/32 halves -- at HEAD_DIM=512 that is 32 B (a b128 pair), but at 64 it
// is 4 B and the compiler emits global_load_b32, a quarter of the width the
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
// This is the axis attn_gqa_wide calls KV_HEADS_PER_WAVE.  It was removed from
// this kernel once, correctly, when head_dim 64 was not yet a target: at
// head_dim >= 128 a single head already fills the lane and there is no load
// width left to win, so every HEADS_PER_WAVE tied.  head_dim 64 changed that
// premise, so the tuned rule below picks 4 there and 1 everywhere else.
//
//   T                element type of Q/K/V
//   NUM_Q_TOKENS     query tokens (1 plain decode, 2-4 speculative)
//   HEAD_DIM         head size (64, 128, 256 or 512)
//   HEADS_PER_WAVE   adjacent KV heads one wave loads together
//   UNROLL           KV tokens loaded and scored per burst
//   USE_SOFTCAP      whether to apply the tanh cap
//   BURST_MAX_MODE   0 rescales per token, non-zero rescales once per burst
//   FAST_ADDR_MODE   -1 picks by the rule below, 0 off, non-zero on
//   PV_DOT2_MODE     non-zero routes the P.V product through dot2 against a
//                    one-hot weight instead of fma_mix; on by default, see the
//                    loop below for what it does and does not buy
template <typename T, int NUM_Q_TOKENS, int HEAD_DIM, int HEADS_PER_WAVE,
          int UNROLL, bool USE_SOFTCAP, int BURST_MAX_MODE = 1,
          int FAST_ADDR_MODE = -1, int PV_DOT2_MODE = 1>
__global__ __launch_bounds__(32) void attn_mha_wide(
    const T* __restrict__ q,           // [num_seqs, NUM_Q_TOKENS, Hq, HEAD_DIM]
    const T* __restrict__ k_cache,     // [nblocks, 2, block_size, Hkv, HEAD_DIM]
    const T* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    // [num_seqs, Hq, num_kv_segments, NUM_Q_TOKENS, HEAD_DIM]
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    const int Hq, const int Hkv, const int block_size,
    const int max_blocks_per_seq, const int num_kv_segments,
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

  // UNROLL, BURST_MAX_MODE and FAST_ADDR_MODE carry their swept defaults;
  // sweep_mha_wide.py overrides them to re-derive those choices.
  //
  // Burst-max is unconditionally on: the general kernel gates it on
  // NUM_Q_TOKENS*VEC2_PER_LANE <= 16, which would disable it at HEAD_DIM=512
  // with NUM_Q_TOKENS=4, but the sweep prefers it in all 24 cells.
  // s_setprio around the load burst was also swept and ties exactly 12-12, so
  // it is gone.

  const int lane = threadIdx.x;
  // Position within this lane's head, and which head of the group it serves.
  // At HEADS_PER_WAVE=1 head_slot is 0 and elem is the whole lane index, i.e.
  // exactly the old mapping.
  const int elem      = lane % LANES_PER_HEAD;
  const int head_slot = lane / LANES_PER_HEAD;

  {
    if (blockIdx.x >= (unsigned)num_tasks) return;
    int task = blockIdx.x;
    const int seg = task % num_kv_segments;  task /= num_kv_segments;
    // The head axis counts GROUPS of HEADS_PER_WAVE, not heads.
    const int head_group = task % (Hkv / HEADS_PER_WAVE);
    task /= (Hkv / HEADS_PER_WAVE);
    const int seq = task;

    // Adjacent KV heads are adjacent in memory, so the group is one contiguous
    // run of HEADS_PER_WAVE*HEAD_DIM halves and each lane reads its slice of it.
    const int kv_head = head_group * HEADS_PER_WAVE + head_slot;
    const int q_head  = kv_head;       // MHA: one query head per KV head

    const int seq_len = seq_lens[seq];
    const int ctx_len = seq_len - NUM_Q_TOKENS;
    const int tokens_per_seg =
        (seq_len + num_kv_segments - 1) / num_kv_segments;
    const int kv_beg  = seg * tokens_per_seg;
    const int kv_end  = min(kv_beg + tokens_per_seg, seq_len);
    // An empty segment still walks the epilogue with a (-inf, 0) contribution
    // rather than skipping ahead: the log-sum-exp already handles that, and
    // there is no LDS merge here to leave a peer waiting.
    const bool seg_empty = kv_beg >= kv_end;

    vec2 q_regs[NUM_Q_TOKENS][VEC2_PER_LANE];
    #pragma unroll
    for (int m = 0; m < NUM_Q_TOKENS; ++m) {
      // `seq` indexes Q, matching the output write in attn_wide_reduce.  Both
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
        const size_t base = (size_t)block_id * block_stride
                          + (size_t)(burst_start % block_size) * Hkv * HEAD_DIM
                          + (size_t)kv_head * HEAD_DIM;
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
            if (pos >= kv_end) {   // see the SAME_BLOCK arm above
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
          const size_t base = (size_t)block_id * block_stride
                            + (size_t)(pos % block_size) * Hkv * HEAD_DIM
                            + (size_t)kv_head * HEAD_DIM;
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
    constexpr bool USE_BURST_MAX = (BURST_MAX_MODE != 0);
    constexpr bool PV_DOT2 = (PV_DOT2_MODE != 0);

    auto compute_burst = [&](auto CheckTag, int burst_start) {
      constexpr bool CHECK_BOUNDS = decltype(CheckTag)::value;
      if constexpr (!USE_BURST_MAX) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
          const int pos = burst_start + u;
          if constexpr (CHECK_BOUNDS) {
            if (pos >= kv_end) break;
          }
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
            const float new_max = fmaxf(run_max[m], score);
            // Both -inf means every position seen so far is masked, and
            // -inf - -inf is NaN.  Reachable whenever a segment holds only
            // masked positions, which short contexts at high
            // num_kv_segments produce.
            if (new_max == -INFINITY) continue;
            const float rescale = exp2_fast(run_max[m] - new_max);
            const float weight  = exp2_fast(score - new_max);
            run_sum[m] = run_sum[m] * rescale + weight;
            run_max[m] = new_max;
            #pragma unroll
            for (int i = 0; i < VEC2_PER_LANE; ++i) {
              acc[m][2*i]   = acc[m][2*i]   * rescale
                            + weight * Traits::lo(v_burst[u][i]);
              acc[m][2*i+1] = acc[m][2*i+1] * rescale
                            + weight * Traits::hi(v_burst[u][i]);
            }
          }
        }
        return;
      }

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
          if constexpr (PV_DOT2) {
            // The same product through dot2 against a one-hot weight.  Same
            // instruction COUNT as the fma_mix form; the point is that
            // v_fma_mix_f32 is VOP3P packed math, which the ISA forbids VOPD
            // from pairing, while v_dot2acc_f32_f16 pairs freely -- the Q.K
            // loop already gets 7 of its 9 dot2 issued as v_dual_dot2acc.
            //
            // MEASURED: this removes all 64 v_fma_mix_f32 from the loop and
            // changes the runtime by NOTHING -- 1.000x, 0.993x, 1.002x, 0.994x,
            // 0.997x across ctx 512..16384 at 32x32x64 M=1.  The kernel is at
            // 97% of this machine's measured streaming ceiling (216 GiB/s, not
            // the 238 nominal), so it waits on memory, not on issue slots, and
            // freeing slots only lengthens the wait.  Kept on anyway: it costs
            // one VGPR (54 vs 53) at the same occupancy, and it drops a packed-
            // math dependency that would matter if this ever stopped being
            // memory-bound (short contexts, or a future WMMA path).
            const vec2 w2 = Traits::pack_weight(weight);
            #pragma unroll
            for (int i = 0; i < VEC2_PER_LANE; ++i)
              Traits::accum_pv(acc[m][2*i], acc[m][2*i+1], w2, v_burst[u][i]);
          } else {
            #pragma unroll
            for (int i = 0; i < VEC2_PER_LANE; ++i) {
              acc[m][2*i]   += weight * Traits::lo(v_burst[u][i]);
              acc[m][2*i+1] += weight * Traits::hi(v_burst[u][i]);
            }
          }
        }
        run_sum[m] = sum;
      }
    };

    // The duplicated load_burst holds a second copy of the burst's address
    // chain, so it only pays where the burst is narrow.
    constexpr bool FAST_ADDR =
        (FAST_ADDR_MODE < 0) ? (VEC2_PER_LANE <= 8) : (FAST_ADDR_MODE != 0);

    const int full_burst_end =
        seg_empty ? kv_beg : kv_beg + ((kv_end - kv_beg) / UNROLL) * UNROLL;
    for (int pos = kv_beg; pos < full_burst_end; pos += UNROLL) {
      if constexpr (FAST_ADDR) {
        if (pos % block_size + UNROLL <= block_size)
          load_burst(std::false_type{}, std::true_type{},  pos);
        else
          load_burst(std::false_type{}, std::false_type{}, pos);
      } else {
        load_burst(std::false_type{}, std::false_type{}, pos);
      }
      compute_burst(std::false_type{}, pos);
    }
    if (full_burst_end < kv_end) {
      if constexpr (FAST_ADDR) {
        if (full_burst_end % block_size + UNROLL <= block_size)
          load_burst(std::true_type{}, std::true_type{},  full_burst_end);
        else
          load_burst(std::true_type{}, std::false_type{}, full_burst_end);
      } else {
        load_burst(std::true_type{}, std::false_type{}, full_burst_end);
      }
      compute_burst(std::true_type{}, full_burst_end);
    }

    // The wave owns this (head, segment) partial outright, so it writes
    // straight out: no LDS staging, no barrier.
    const size_t out_base =
        (((size_t)(seq * Hq + q_head) * num_kv_segments + seg) * NUM_Q_TOKENS)
        * (size_t)HEAD_DIM;
    #pragma unroll
    for (int m = 0; m < NUM_Q_TOKENS; ++m) {
      #pragma unroll
      for (int i = 0; i < VEC2_PER_LANE; ++i) {
        partial_out[out_base + (size_t)m * HEAD_DIM
                    + (elem * VEC2_PER_LANE + i) * 2]     = acc[m][2*i];
        partial_out[out_base + (size_t)m * HEAD_DIM
                    + (elem * VEC2_PER_LANE + i) * 2 + 1] = acc[m][2*i+1];
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

}  // namespace mha_wide


namespace {

// THE SHIPPED CONFIG.  The op takes num_kv_segments == 0 to mean "use this", so
// callers never choose a config themselves.
//
// A rule, not a lookup table, and that is a measured choice rather than a
// preference: sweep_mha_wide.py and bench_mha_wide share a grid, so both cover
// the same 120 cells (head counts 8/16/32/64 x head_dim 128/256/512 x M 1/4 x
// ctx 128..65536), and against the best config per cell this rule costs
//
//   mean 1.013x, median 1.010x, worst 1.072x  (110/120 within 3%)
//
// while the spread between the best and worst config of a cell reaches 37x.
// A table with a ctx axis would buy ~1% for an entry per (head count, head_dim,
// M, ctx).  The worst cell is 8x8x256 M=1 ctx=128, a tiny shape where the whole
// kernel is 17us and the segments are shorter than one burst.
//
// Both constants were re-fitted against the sweep rather than assumed: the 512
// numerator and the cap of 16 are the joint optimum of their family (numerator
// 256 -> 1.078x, 1024 -> 1.063x, cap 8 -> 1.041x).  Simplifying the unroll test
// to `M*vec2_per_lane <= 4` alone costs 1.024x with a 1.21x worst case, so the
// two-clause form earns its second clause.
//
// The segment count is NOT constrained to a power of two, and must not be
// rounded to one: it targets a constant task count (~512/vec2, about 6.4 tasks
// per SIMD across the 80 SIMDs on gfx1151), and rounding down costs 11-43% of
// the parallelism at the head counts where it bites (Hq=40 gives 6, Hq=48
// gives 5, Hq=71 gives 7).  Both kernels take it at runtime for that reason.
bool wide_attn_tuned(int num_q_tokens, int head_dim, int Hkv,
                     int* num_kv_segments, int* unroll, int* heads_per_wave) {
  const int vec2_per_lane = head_dim / 64;
  if (vec2_per_lane < 1 || vec2_per_lane * 64 != head_dim) return false;

  // Head grouping first: it changes how many tasks there are, so the segment
  // count below is derived after it.  2 heads per wave below 16 heads, 4 at or
  // above -- both give a b128 lane at head_dim 64, and the crossover is where
  // the sweep put it (grouping by 4 at Hq=8 leaves only two groups).  Above
  // head_dim 64 a single head already fills the lane, so no grouping.
  int hpw = 1;
  if (vec2_per_lane == 1) {
    const int want = (Hkv <= 16) ? 2 : 4;
    for (int c = want; c >= 2; c >>= 1) {
      if (Hkv % c == 0) { hpw = c; break; }
    }
  }

  // The numerator is 256 at head_dim 64 and 512 above it.  512 was fitted where
  // a wave serves one head; a grouped wave covers hpw of them per step, so the
  // same task count is reached with fewer segments.  Measured over 24 cells at
  // head_dim 64 with every axis open, 256 costs geomean 1.021 against the best
  // config per cell where 512 costs 1.047 -- and over the 96 cells at head_dim
  // >= 128, 256 would be a REGRESSION (1.070 / 1.305 against 1.012 / 1.086).
  //
  // The segment count stays keyed on Hkv, NOT on the group count, even though
  // grouping cuts the task count by hpw.  Deriving it from groups to hold the
  // task count near 512 is the obvious move and measures worse in 8 of 8 cells
  // (32x32x64 M=1 ctx=1024: 0.90x -> 0.85x of Triton).  A grouped wave does hpw
  // tokens' worth of work per step, so 128 fat tasks saturate much like 512
  // thin ones, and the extra segments only fragment the KV walk.
  const int numerator = (vec2_per_lane == 1) ? 256 : 512;
  int segments = numerator / (Hkv * vec2_per_lane);

  // The cap is 32 at head_dim 64 and 16 above it, for the same reason the
  // numerator differs: at head_dim 64 vec2 is 1, so the quotient is 8x larger
  // and the clamp binds on every shape.  Measured over 136 cells, against the
  // best config per cell:
  //
  //                        all 136        head_dim 64    head_dim >=128
  //   cap 16 everywhere    1.030 / 1.424  1.077 / 1.424  1.0115 / 1.086
  //   cap 32 everywhere    1.022 / 1.123  1.045 / 1.123  1.0122 / 1.086
  //   cap 32 iff D == 64   1.021 / 1.123  1.045 / 1.123  1.0115 / 1.086
  //                                              (geomean / worst)
  const int cap = (vec2_per_lane == 1) ? 32 : 16;
  segments = segments < 1 ? 1 : (segments > cap ? cap : segments);

  const int q_times_vec2 = num_q_tokens * vec2_per_lane;
  *num_kv_segments = segments;
  *heads_per_wave = hpw;
  *unroll = (q_times_vec2 <= 4
             || (Hkv * segments <= 128 && q_times_vec2 <= 8)) ? 4 : 2;
  return true;
}

// THREADS_PER_BLOCK for the reduce, derived from HEAD_DIM rather than fixed at
// 64.  The kernel is one thread per output element, so this only changes how
// the same total threads are grouped.  Measured on gfx1151, output bit-identical
// to TPB=64: D>=256 -> 256 is fastest (1.06-1.20x), D=128 -> 128 (1.01-1.03x;
// 256 loses, down to 0.83x).  D=64 -> 64 is a HARD CAP, not a preference: with
// TPB > HEAD_DIM the surplus threads compute d >= HEAD_DIM and write past the
// row.  Confirmed against a guard region -- D=64/TPB=256 overwrote 192 halves
// past the output with no launch error, and the static_assert does not catch it
// (256 % 64 == 0 passes).
constexpr int reduce_tpb(int head_dim) {
  return head_dim >= 256 ? 256 : (head_dim >= 128 ? 128 : 64);
}

}  // namespace

// Dispatch over (element type, num_q_tokens, head_dim, unroll, softcap).  Only
// M in {1,4} is instantiated: M=1 is plain decode and M=4 speculative, and the
// host pads 2 and 3 up to 4 rather than doubling the build for them.  Padding
// costs a measured geomean 1.024x (worst 1.081x) and, more to the point, keeps
// every launch on one of the two M values the tuning sweep actually covered.
#define WIDE_ATTN_LAUNCH(T_, M_, D_, HPW_, UNRL_, SOFTCAP_)                    \
  do {                                                                         \
    /* One task per (seq, head GROUP, segment): a wave owns HPW_ heads. */      \
    const int num_tasks = num_seqs * (num_kv_heads / (HPW_)) * nseg;           \
    mha_wide::attn_mha_wide<T_, M_, D_, HPW_, UNRL_, SOFTCAP_>                 \
        <<<dim3(num_tasks), dim3(32), 0, stream>>>(                            \
            (const T_*)query.data_ptr(), (const T_*)key_cache.data_ptr(),      \
            (const T_*)value_cache.data_ptr(),                                 \
            block_table.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),     \
            partial_out.data_ptr<float>(), partial_max.data_ptr<float>(),      \
            partial_sum.data_ptr<float>(),                                     \
            num_heads, num_kv_heads, block_size, max_blocks_per_seq, nseg,     \
            scale_log2, softcap_log2, inv_softcap, block_stride, num_seqs,     \
            num_tasks);                                                        \
    attn_wide::attn_wide_reduce<T_, M_, D_, reduce_tpb(D_)>                    \
        <<<dim3((D_ / reduce_tpb(D_)) * M_, num_heads, num_seqs),              \
           dim3(reduce_tpb(D_)), 0, stream>>>(                                 \
            partial_out.data_ptr<float>(), partial_max.data_ptr<float>(),      \
            partial_sum.data_ptr<float>(), (T_*)out.data_ptr(), num_heads,     \
            nseg);                                                             \
    return;                                                                    \
  } while (0)

#define WIDE_ATTN_CASE(T_, M_, D_, HPW_, UNRL_)                                \
  do {                                                                         \
    if (num_q_tokens == (M_) && head_size == (D_) && unroll == (UNRL_)         \
        && hpw == (HPW_) && num_kv_heads % (HPW_) == 0) {                      \
      if (softcap > 0.f) WIDE_ATTN_LAUNCH(T_, M_, D_, HPW_, UNRL_, true);      \
      else               WIDE_ATTN_LAUNCH(T_, M_, D_, HPW_, UNRL_, false);     \
    }                                                                          \
  } while (0)

// The rule only ever emits unroll 2 or 4; 1 is instantiated because the
// dispatch is keyed on the value the rule returns and a stale rule should fail
// loudly rather than silently pick a neighbour.
#define WIDE_ATTN_GRID(T_, M_, D_, HPW_)                                       \
  do {                                                                         \
    WIDE_ATTN_CASE(T_, M_, D_, HPW_, 1);                                       \
    WIDE_ATTN_CASE(T_, M_, D_, HPW_, 2);                                       \
    WIDE_ATTN_CASE(T_, M_, D_, HPW_, 4);                                       \
  } while (0)

#define WIDE_ATTN_ALL_SHAPES(T_)                                               \
  do {                                                                         \
    /* head_dim 64 builds the grouped arms too; above it a single head fills  \
       the lane, so only the ungrouped one exists. */                          \
    WIDE_ATTN_GRID(T_, 1, 64, 1); WIDE_ATTN_GRID(T_, 1, 64, 2);                \
    WIDE_ATTN_GRID(T_, 1, 64, 4);                                              \
    WIDE_ATTN_GRID(T_, 4, 64, 1); WIDE_ATTN_GRID(T_, 4, 64, 2);                \
    WIDE_ATTN_GRID(T_, 4, 64, 4);                                              \
    WIDE_ATTN_GRID(T_, 1, 128, 1); WIDE_ATTN_GRID(T_, 4, 128, 1);              \
    WIDE_ATTN_GRID(T_, 1, 256, 1); WIDE_ATTN_GRID(T_, 4, 256, 1);              \
    WIDE_ATTN_GRID(T_, 1, 512, 1); WIDE_ATTN_GRID(T_, 4, 512, 1);              \
  } while (0)

// query/out: [num_seqs, num_q_tokens, num_heads, head_size]
// key_cache/value_cache: [num_blocks, block_size, num_kv_heads, head_size]
//   views into vLLM's [num_blocks, 2, ...] allocation, so block_stride is taken
//   from stride(0) rather than recomputed.
// seq_lens[i] is the TOTAL length including the num_q_tokens new tokens.
//
// Every constraint is a TORCH_CHECK rather than a status code: the Python
// predicate (rocm_wide_decode_attn.can_run) is expected to have screened the
// call, so reaching here with a bad shape is a bug, and the failure mode this
// replaces -- returning -1 with `out` left untouched -- is silently wrong
// output.
void wide_decode_attn(torch::Tensor& out, torch::Tensor& query,
                      torch::Tensor& key_cache, torch::Tensor& value_cache,
                      torch::Tensor& block_table, torch::Tensor& seq_lens,
                      torch::Tensor& partial_out, torch::Tensor& partial_max,
                      torch::Tensor& partial_sum, double scale,
                      double softcap) {
  TORCH_CHECK(query.dim() == 4, "query must be [num_seqs, M, num_heads, head_size]");
  TORCH_CHECK(out.sizes() == query.sizes(), "out must match query shape");
  TORCH_CHECK(query.scalar_type() == out.scalar_type(), "out dtype must match query");
  TORCH_CHECK(key_cache.scalar_type() == query.scalar_type() &&
                  value_cache.scalar_type() == query.scalar_type(),
              "KV cache dtype must match query (no quantized KV)");
  TORCH_CHECK(query.is_contiguous() && out.is_contiguous(),
              "query and out must be contiguous");

  const int num_seqs = static_cast<int>(query.size(0));
  const int num_q_tokens = static_cast<int>(query.size(1));
  const int num_heads = static_cast<int>(query.size(2));
  const int head_size = static_cast<int>(query.size(3));

  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "kv cache must be [num_blocks, block_size, num_kv_heads, head_size]");
  const int block_size = static_cast<int>(key_cache.size(1));
  const int num_kv_heads = static_cast<int>(key_cache.size(2));
  TORCH_CHECK(key_cache.size(3) == head_size && value_cache.size(3) == head_size,
              "kv cache head_size must match query");
  TORCH_CHECK(value_cache.size(1) == block_size &&
                  value_cache.size(2) == num_kv_heads,
              "key and value caches must have the same layout");

  // MHA only: the kernel maps one query head per KV head by construction.
  TORCH_CHECK(num_heads == num_kv_heads,
              "wide_decode_attn is MHA-only: num_heads (", num_heads,
              ") must equal num_kv_heads (", num_kv_heads, ")");
  TORCH_CHECK(num_q_tokens == 1 || num_q_tokens == 4,
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
  TORCH_CHECK(wide_attn_tuned(num_q_tokens, head_size, num_kv_heads, &nseg,
                              &unroll, &hpw),
              "no tuned config for head_size ", head_size);
  TORCH_CHECK(hpw >= 1 && num_kv_heads % hpw == 0,
              "head group ", hpw, " does not divide num_kv_heads ",
              num_kv_heads);

  // The partials are indexed [num_seqs, num_heads, nseg, M, head_size], so a
  // segment count that disagrees with the launch is a memory fault rather than
  // a compile error.  Check it here instead.
  TORCH_CHECK(partial_out.dim() == 5 && partial_max.dim() == 4 &&
                  partial_sum.dim() == 4,
              "partial buffers have the wrong rank");
  TORCH_CHECK(partial_out.size(0) >= num_seqs && partial_out.size(1) == num_heads &&
                  partial_out.size(2) >= nseg && partial_out.size(3) == num_q_tokens &&
                  partial_out.size(4) == head_size,
              "partial_out must be [>=num_seqs, num_heads, >=", nseg, ", ",
              num_q_tokens, ", ", head_size, "]");
  TORCH_CHECK(partial_out.size(2) == nseg,
              "partial buffers were sized for ", partial_out.size(2),
              " KV segments but the tuned config wants ", nseg);
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
    WIDE_ATTN_ALL_SHAPES(half);
  } else if (query.scalar_type() == at::ScalarType::BFloat16) {
    WIDE_ATTN_ALL_SHAPES(__hip_bfloat16);
  } else {
    TORCH_CHECK(false, "wide_decode_attn supports float16 and bfloat16, got ",
                query.scalar_type());
  }

  TORCH_CHECK(false, "wide_decode_attn: no instantiation for M=", num_q_tokens,
              " head_size=", head_size, " unroll=", unroll);
}

#undef WIDE_ATTN_ALL_SHAPES
#undef WIDE_ATTN_GRID
#undef WIDE_ATTN_CASE
#undef WIDE_ATTN_LAUNCH
