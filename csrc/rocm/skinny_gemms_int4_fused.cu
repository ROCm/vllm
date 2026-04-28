// Fused GEMM1 + activation + GEMM2 mega-kernel for the W4A16 MoE decode path.
// Strategy C: 8 expert-block groups of `wgs_per_group` workgroups each;
// each group cooperates on one expert end-to-end. The activation is staged
// through a small global-memory scratch (kept resident in L2 via regular
// loads/stores) with an in-kernel atomic barrier between Phase 1 (GEMM1 +
// silu*mul) and Phase 2 (GEMM2). Targets gfx1151 in WGP mode (default; ~6 us
// faster than -mcumode for this shape -- the WGP scheduler hides s_waitcnt
// stalls by interleaving the 2 co-resident WGs' waves on the WGP's 8 SIMDs),
// grid = num_groups * wgs_per_group, e.g. dim3(40) for 8 groups of 5 WGs
// landing as 2 WGs per WGP across 20 WGPs. CU mode is opt-in via the host
// wrapper's WVSPLITK_FUSED_USE_CUMODE=1 env knob (kept as a deadlock-safety
// fallback if a future shape breaks the 2-WGs-per-WGP residency budget).
//
// LDS layout per WG (shared between phases, fits in MOE_LDS_ELEMS=8192 fp16):
//   s[0..K_hidden)              - hidden_states[src_row]  (Phase 1 input)
//   s[K_hidden..K_hidden+K_inter) - per-group activation (Phase 2 input)
//
// Bit-exact fp16 rounding at the activation boundary (matches what the
// existing 3-kernel pipeline does: GEMM1 -> fp16 in HBM -> silu_and_mul ->
// fp16 in HBM -> GEMM2).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA-style includes; hipify translates these to <hip/...> at build
// time.  Same convention as csrc/rocm/skinny_gemms_int4.cu.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>

// All helpers / kernel templates live in an anonymous namespace so they
// don't collide with identically-named template specializations in
// skinny_gemms_int4.cu (`__s2float`, `__float2s`, `loadnt`, etc.) when
// both translation units link into _rocm_C.so.
namespace {

// Match skinny_gemms_int4.cu's macro convention: in vLLM's HIP build
// mode (-x hip via clang++), __HIPCC__ isn't always set, but __GFX*__
// is.  Drop the __HIPCC__ guard so the kernel templates instantiate.
#if defined(__GFX11__) || defined(__GFX12__)
  #define __HIP__GFX1X__
#endif

// Portable unreachable marker for the host-compile stub of the kernel
// template (see the `#else` branch of the gfx1X guard below).  We can't
// use `assert(false)` here -- in vLLM's clang `-x hip` build mode the
// __global__ stub is parsed in both host and device passes, and clang
// can't resolve `__assert_fail` for the device pass when this file is
// fed directly as `.hip` (no hipify translation).  __builtin_trap()
// resolves on both passes.
#define UNREACHABLE_CODE __builtin_trap();

// Per-WG LDS budget for the fused kernel.  Same 16 KB cap as the upstream
// MoE path (skinny_gemms_int4.cu line 804) so the kernel sits at the
// occupancy sweet spot already validated by the existing bench.
[[maybe_unused]] constexpr int FUSED_LDS_ELEMS = 8192;

// ---------------------------------------------------------------------------
// Helpers (mirrors wvsplitk_int4_local.hip's prologue)
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float __s2float(T v);

template <>
__device__ __forceinline__ float __s2float(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float __s2float(__hip_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T __float2s(float v);

template <>
__device__ __forceinline__ half __float2s(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ __hip_bfloat16 __float2s(float v) {
  return __float2bfloat16(v);
}

template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
  return __builtin_nontemporal_load(addr);
}

template <typename T>
struct scalar {};

template <>
struct scalar<c10::Half> {
  using type = half;
};

template <>
struct scalar<c10::BFloat16> {
  using type = __hip_bfloat16;
};

#define DOT2C(V0, V2, V3)                                                   \
  if constexpr (std::is_same_v<scalar_t, half>) {                           \
    V0 = __builtin_amdgcn_fdot2(*((half2*)(&(V2))), *((half2*)(&(V3))), V0, \
                                false);                                     \
  } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {          \
    float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *           \
               __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));            \
    V0 += (s.x + s.y);                                                      \
  }

#if defined(__HIP__GFX1X__)
  #define REDUCE_SUM_WAVE32(val)  \
    do {                          \
      val += __shfl_xor(val, 1);  \
      val += __shfl_xor(val, 2);  \
      val += __shfl_xor(val, 4);  \
      val += __shfl_xor(val, 8);  \
      val += __shfl_xor(val, 16); \
    } while (0)
#endif

// Inline int4 -> fp16/bf16 dequant.  Same bit-magic as the upstream
// compute_sml_ FP16_MAGIC fast path (shifts the 4-bit nibbles into the
// fp16 mantissa of 1024.0f then subtracts 1032.0f = bias 8 + scale 16).
//
// HAS_ZP unused here (symmetric scales only -- the existing bench is
// symmetric quant); kept for symmetry with the upstream macro.
#define DEQUANT_INT4_HALF_FAST(SRC_BIGB, DST_CVTB)                          \
  do {                                                                      \
    constexpr uint32_t FP16_MAGIC = 0x64006400u;                            \
    constexpr uint32_t BIAS_LO = 0x64086408u;                               \
    constexpr uint32_t SCALE16 = 0x2C002C00u;                               \
    constexpr uint32_t BIAS_HI = 0xD480D480u;                               \
    _Pragma("unroll")                                                       \
    for (uint32_t w = 0; w < A_CHUNK / 8; w++) {                            \
      uint32_t qa = (SRC_BIGB).u32[w];                                      \
      uint32_t lo0 = (qa & 0x000F000Fu) | FP16_MAGIC;                       \
      uint32_t hi0 = (qa & 0x00F000F0u) | FP16_MAGIC;                       \
      qa >>= 8;                                                             \
      uint32_t lo1 = (qa & 0x000F000Fu) | FP16_MAGIC;                       \
      uint32_t hi1 = (qa & 0x00F000F0u) | FP16_MAGIC;                       \
      *(half2*)&(DST_CVTB).f[w * 4 + 0] =                                   \
          __hsub2(*(half2*)&lo0, *(const half2*)&BIAS_LO);                  \
      *(half2*)&(DST_CVTB).f[w * 4 + 1] =                                   \
          __hfma2(*(half2*)&hi0, *(const half2*)&SCALE16,                   \
                  *(const half2*)&BIAS_HI);                                 \
      *(half2*)&(DST_CVTB).f[w * 4 + 2] =                                   \
          __hsub2(*(half2*)&lo1, *(const half2*)&BIAS_LO);                  \
      *(half2*)&(DST_CVTB).f[w * 4 + 3] =                                   \
          __hfma2(*(half2*)&hi1, *(const half2*)&SCALE16,                   \
                  *(const half2*)&BIAS_HI);                                 \
    }                                                                       \
  } while (0)

#define DEQUANT_INT4_BF16_SCALAR(SRC_BIGB, DST_CVTB)                        \
  do {                                                                      \
    _Pragma("unroll")                                                       \
    for (uint32_t w = 0; w < A_CHUNK / 8; w++) {                            \
      uint32_t qa = (SRC_BIGB).u32[w];                                      \
      (DST_CVTB).h[w * 8 + 0] = (scalar_t)((int)(qa & 0xF) - 8);            \
      (DST_CVTB).h[w * 8 + 1] = (scalar_t)((int)((qa >> 16) & 0xF) - 8);    \
      (DST_CVTB).h[w * 8 + 2] = (scalar_t)((int)((qa >> 4) & 0xF) - 8);     \
      (DST_CVTB).h[w * 8 + 3] = (scalar_t)((int)((qa >> 20) & 0xF) - 8);    \
      (DST_CVTB).h[w * 8 + 4] = (scalar_t)((int)((qa >> 8) & 0xF) - 8);     \
      (DST_CVTB).h[w * 8 + 5] = (scalar_t)((int)((qa >> 24) & 0xF) - 8);    \
      (DST_CVTB).h[w * 8 + 6] = (scalar_t)((int)((qa >> 12) & 0xF) - 8);    \
      (DST_CVTB).h[w * 8 + 7] = (scalar_t)((int)((qa >> 28) & 0xF) - 8);    \
    }                                                                       \
  } while (0)

#define DEQUANT_INT4(SRC_BIGB, DST_CVTB)             \
  do {                                                \
    if constexpr (std::is_same_v<scalar_t, half>) {  \
      DEQUANT_INT4_HALF_FAST(SRC_BIGB, DST_CVTB);    \
    } else {                                          \
      DEQUANT_INT4_BF16_SCALAR(SRC_BIGB, DST_CVTB);  \
    }                                                 \
  } while (0)

// ---------------------------------------------------------------------------
// Fused mega-kernel
// ---------------------------------------------------------------------------
//
// Grid: dim3(num_groups * wgs_per_group).  Each WG belongs to group
//   g = blockIdx.x / wgs_per_group   (one expert per group)
//   t = blockIdx.x % wgs_per_group   (M-tile within the expert)
//
// Phase 1 (GEMM1 + activation):
//   - Cooperative load of hidden_states[src_row] into LDS s[0..K_hidden).
//   - 5 WGs split K_inter output rows; per row, simultaneously accumulate
//     gate (W1[m]) and up (W1[m+K_inter]) dot products against s[].
//   - Bit-exact rounding: round gate, up to fp16, upcast for silu*mul,
//     round result to fp16, write to d_act_scratch[g, m] (regular store --
//     stays in L2, not NT).
//
// Atomic barrier (per-group, 5-wide): atomicAdd(&d_barrier[g], 1) and spin
// until counter == wgs_per_group.  __threadfence on both sides.
//
// Phase 2 (GEMM2):
//   - Cooperative load of d_act_scratch[g] into LDS s[K_hidden..K_hidden+K_inter)
//     (regular load -- L2 hit).
//   - 5 WGs split K_hidden output rows; per row, dot against the activation
//     in LDS, NT-store to gemm2_out[slot, m].
//
// Final: each WG atomicSub's the barrier counter so it's back to 0 for the
// next launch (no host-side memset needed).
#if defined(__HIP__GFX1X__)
template <typename scalar_t, int THRDS, int YTILE_G1, int YTILE_G2,
          int WvPrGrp_G1, int WvPrGrp_G2,
          int A_CHUNK, int UNRL_G1, int UNRL_G2,
          int GROUP_SIZE>
__global__ void __launch_bounds__(
    ((WvPrGrp_G1 > WvPrGrp_G2) ? WvPrGrp_G1 : WvPrGrp_G2) * THRDS)
moe_wvSplitK_int4_fused_grouped_(
    const int K_hidden, const int K_inter,
    const uint8_t* __restrict__ w1_packed_base,
    const scalar_t* __restrict__ w1_scale_base,
    const uint8_t* __restrict__ w2_packed_base,
    const scalar_t* __restrict__ w2_scale_base,
    const scalar_t* __restrict__ a_base,
    scalar_t* __restrict__ gemm2_out_base,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_token_ids,
    const int top_k,
    const long expert_stride_w1, const long expert_stride_s1,
    const long expert_stride_w2, const long expert_stride_s2,
    const int wgs_per_group, const int num_groups,
    scalar_t* __restrict__ d_act_scratch,
    int* __restrict__ d_barrier,
    int* __restrict__ d_p1_row_ctr,
    int* __restrict__ d_p1_done_ctr) {
  // The launch-time block_y dimension is set to max(W_g1, W_g2) so both
  // phases fit; the smaller phase's "extra" waves enter the phase body
  // but their `wave_active` guard makes them skip the work (they still
  // participate in __syncthreads() so cooperative-load and atomic-counter
  // dispatch barriers don't deadlock).
  constexpr int WvPrGrpMax =
      (WvPrGrp_G1 > WvPrGrp_G2) ? WvPrGrp_G1 : WvPrGrp_G2;
  union bigTypeA {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
  };
  union bigTypeW {
    uint8_t b[A_CHUNK / 2];
    uint32_t u32[A_CHUNK / 8];
    float f[A_CHUNK / 8];
  };

  __shared__ scalar_t s[FUSED_LDS_ELEMS];

  // -------------------------------------------------------------------------
  // Scheduler hints for the three-phase K-loop.
  //
  // SCHED_GROUP_*: __builtin_amdgcn_sched_group_barrier forces the LLVM
  //   AMDGPU scheduler to keep the VMEM/LDS/VALU phases in strict order
  //   (mask 0x020 = VMEM_READ, 0x100 = DS_READ, 0x002 = VALU).  Without
  //   these hints, the register-pressure-driven heuristic can interleave
  //   instructions in ways that defeat the 3-phase pattern's deep VMEM
  //   pipeline.  size=0xff means "as many of that kind as available".
  //
  // WAVE_PRIO_*: bump this wave's priority during VMEM dispatch via
  //   s_setprio so the SIMD scheduler keeps issuing the wave's loads
  //   contiguously instead of preempting it for another wave's VALU.
  //   Doesn't help once the wave is on s_waitcnt (already blocked) but
  //   nudges the dispatch-side scheduling.
  //
  // Both are guarded by FUSED_NO_SCHED_HINTS so we can A/B without
  // editing the kernel body.
#ifndef FUSED_NO_SCHED_HINTS
  #define SCHED_GROUP_VMEM(_n) __builtin_amdgcn_sched_group_barrier(0x020, (_n), 0)
  #define SCHED_GROUP_LDS(_n)  __builtin_amdgcn_sched_group_barrier(0x100, (_n), 0)
  #define SCHED_GROUP_VALU(_n) __builtin_amdgcn_sched_group_barrier(0x002, (_n), 0)
  #define WAVE_PRIO_HIGH()     asm volatile("s_setprio 3" ::: "memory")
  #define WAVE_PRIO_LOW()      asm volatile("s_setprio 0" ::: "memory")
#else
  #define SCHED_GROUP_VMEM(_n) ((void)0)
  #define SCHED_GROUP_LDS(_n)  ((void)0)
  #define SCHED_GROUP_VALU(_n) ((void)0)
  #define WAVE_PRIO_HIGH()     ((void)0)
  #define WAVE_PRIO_LOW()      ((void)0)
#endif

  // -------------------------------------------------------------------------
  // LDS bank-conflict mitigation.
  //
  // Without padding, every `bigA = bigTypeA` LDS load has an 8-way bank
  // conflict: lane L in a wave reads from byte offset `base + L*32`, so
  // the dword stride between lanes is `A_CHUNK / 2 = 8 dwords`, which
  // shares gcd(8, 32) = 8 with the LDS's 32 banks.  ATT confirmed this
  // is dormant (~1.5% of total stall, hidden behind VMEM waitcnt), but we
  // pad anyway as a reference implementation.
  //
  // The padding inserts `LDS_PAD_FP16 = A_CHUNK/2` empty fp16 slots after
  // every A_CHUNK-element stripe, making the inter-lane LDS stride
  // `1.5 * A_CHUNK = 24 fp16 = 12 dwords` for A_CHUNK=16.
  // gcd(12, 32) = 4, so the conflict factor halves (8-way -> 4-way).
  // Going lower than 4-way would require dropping below b128 alignment,
  // which the compiler does not honour for our `bigTypeA` load.
  //
  // Layout in s[]:
  //   stripe i (a contiguous run of A_CHUNK fp16) starts at slot
  //   `i * (A_CHUNK + LDS_PAD_FP16)`.  Logical fp16 element `k` is at
  //   `LDS_PAD(k) = k + (k / A_CHUNK) * LDS_PAD_FP16`.
  // The hidden_states region uses padded indices 0 .. LDS_PAD(K_hidden).
  // The activation region starts at `K_hidden_pad_offset = LDS_PAD(K_hidden)`
  // and uses padded indices 0 .. LDS_PAD(K_inter) within that region.
  // Total LDS used = LDS_PAD(K_hidden) + LDS_PAD(K_inter) <= FUSED_LDS_ELEMS.
  constexpr int LDS_PAD_FP16 = A_CHUNK / 2;
  #define LDS_PAD(_k) ((_k) + ((_k) / A_CHUNK) * LDS_PAD_FP16)

  const int g = blockIdx.x / wgs_per_group;
  const int t = blockIdx.x % wgs_per_group;
  if (g >= num_groups) return;

  // The kernel requires expert_ids[g] in [0, E).  Padding entries (-1)
  // would deadlock the cross-WG atomic barrier: a "fast-path" early
  // return that does atomicAdd + spin + atomicSub races with siblings
  // still in the spin loop (their post-release atomicSub can drop the
  // counter below wgs_per_group while a sibling is still reading the
  // counter, so it never observes the release).  The host wrapper does
  // NOT runtime-validate this -- expert_ids.min().item() forces a
  // device->host sync (+15-20 us) that would dominate the kernel run.
  // The only call site (decode-only, block_size_m=1 in
  // hybrid_w4a16_moe.py) never produces -1, since moe_align_block_size
  // is skipped on that path entirely.
  const int expert_id = expert_ids[g];

  // Resolve src row of A and slot id of the output.  For decode (M=1,
  // top_k>=1) all groups share src_row=0, but we leave the general
  // formula in for forward-compat.
  long src_row, slot_id;
  if (sorted_token_ids) {
    slot_id = (long)sorted_token_ids[g];
    src_row = slot_id / top_k;
  } else {
    slot_id = (long)g;
    src_row = (long)g;
  }

  // W1 / S1 used to live here as outer-scope pointers, but the work-
  // stealing refactor moved their resolution into `do_p1_chunk`
  // (per-`g_cur`).  In steal mode the helper loop has its own re-
  // resolved `W1_h`/`S1_h`; in static mode `do_p1_chunk(g, m_static)`
  // also re-resolves from `g`.  Keep them out of outer scope so the
  // -Werror=unused-variable build doesn't trip.
  const uint8_t* W2 =
      w2_packed_base + (long)expert_id * expert_stride_w2;
  const scalar_t* S2 =
      w2_scale_base + (long)expert_id * expert_stride_s2;
  const scalar_t* A = a_base + src_row * K_hidden;
  scalar_t* C = gemm2_out_base + slot_id * K_hidden;
  scalar_t* act_scratch_g = d_act_scratch + (long)g * K_inter;

  // Padded byte offset where the activation region begins inside s[].
  // Computed once; depends only on the runtime K_hidden parameter.
  const int K_hidden_pad_offset = LDS_PAD(K_hidden);

  // -------------------------------------------------------------------------
  // Cooperative load of hidden_states into s[0..LDS_PAD(K_hidden)).  Each WG
  // of the group does its own copy (LDS is per-WG).  All groups in flight
  // read the same src_row=0 slice for decode -> L2 absorbs the redundant
  // reads.  k_in is always a multiple of A_CHUNK, so LDS_PAD(k_in) lands
  // exactly at a stripe boundary -- the bigTypeA store fits within the
  // stripe's A_CHUNK data slots, leaving the LDS_PAD_FP16 trailing slots
  // unwritten by design.  Uses all WvPrGrpMax waves so even the asymmetric
  // configs (W_g1 != W_g2) saturate the load with the launched WG size.
  // -------------------------------------------------------------------------
  {
    constexpr int total_threads = WvPrGrpMax * THRDS;
    const int tid_lin = threadIdx.y * THRDS + threadIdx.x;
    for (int k = 0; k < K_hidden; k += total_threads * A_CHUNK) {
      int k_in = k + tid_lin * A_CHUNK;
      if (k_in >= K_hidden) break;
      *((bigTypeA*)(&s[LDS_PAD(k_in)])) = *((const bigTypeA*)(&A[k_in]));
    }
  }
  __syncthreads();

  // -------------------------------------------------------------------------
  // Cross-group P1 work-stealing -- gated at compile time.  When
  // FUSED_NO_STEAL is defined the kernel uses the legacy static
  // `m += wgs_per_group * WvPrGrp * YTILE_G1` stride and the per-group
  // `d_barrier` arrival counter.  Default (no -DFUSED_NO_STEAL) uses the
  // atomic row dispatch: a WG that finishes its native group's P1 chunks
  // helps drain other groups' P1 chunks, then the per-group barrier polls
  // `d_p1_done_ctr[g]` until all `K_inter / (W*Y)` chunks are processed.
  //
  // The cross-group help is only safe when the LDS hidden_states is
  // identical across groups, which is the M=1 decode contract enforced by
  // the host wrapper (every group's `sorted_token_ids[g] / top_k` resolves
  // to the same `src_row`).  See the precondition comment at lines 305+.
  // -------------------------------------------------------------------------
  __shared__ uint32_t m_shared;

#ifndef FUSED_SKIP_PHASE1
  // =========================================================================
  // Phase 1: GEMM1 + activation
  //
  // WG `t` walks K_inter/2*WG-stride rows of activation output.  For each
  // row m we compute simultaneously:
  //   gate_acc[y] = sum_k A[k] * dequant(W1[m+y, k])
  //   up_acc[y]   = sum_k A[k] * dequant(W1[m+y+K_inter, k])
  // and write silu(gate_h) * up_h  (bit-exact fp16 rounding) to scratch.
  // =========================================================================
  {
    const int K_packed = K_hidden / 2;
    const int num_groups_g1 =
        (GROUP_SIZE > 0) ? (K_hidden / GROUP_SIZE) : 0;

    bigTypeA bigA_g1[UNRL_G1];
    // Phase 1 split into separate gate-pass and up-pass: only ONE B
    // stream is live at a time, halving bigB register pressure compared
    // to the paired layout (~64 VGPRs saved).  Both passes use the same
    // SWP'd bigB_curr / bigB_next double-buffer.  HBM volume is unchanged
    // (we still read every gate row + every up row exactly once per token
    // per expert), but the K-loop runs twice per output row -- once per
    // pass -- with cleaner register flow and (likely) better scheduler
    // freedom from the lower pressure.  gate_sum is held in registers
    // across the up-pass for the silu*mul.
    bigTypeW bigB_curr[YTILE_G1][UNRL_G1];
    bigTypeW bigB_next[YTILE_G1][UNRL_G1];

    // ---------------------------------------------------------------
    // Per-pass K-loop.  `is_up` selects which W1 half to read (gate
    // half: rows [m, m+YTILE_G1)) vs up half: rows [m+K_inter,
    // m+K_inter+YTILE_G1)) and which accumulator to update.
    //
    // `W1_use` / `S1_use` are passed in (rather than captured) so the
    // same lambda body works for both the native group's expert and any
    // helped group's expert in the work-stealing dispatch.
    // ---------------------------------------------------------------
    auto run_kpass = [&] (uint32_t m_curr, bool is_up,
                          float (&acc)[YTILE_G1],
                          const uint8_t* W1_use,
                          const scalar_t* S1_use) {
      const uint32_t row_off = is_up ? (uint32_t)K_inter : 0;

      // Prologue: prime bigB_curr for k1=0.
      WAVE_PRIO_HIGH();
      {
        constexpr uint32_t k1_init = 0;
        #pragma unroll
        for (uint32_t k2 = 0; k2 < UNRL_G1; k2++) {
          uint32_t k = k1_init + k2 * THRDS * A_CHUNK;
          uint32_t k_ = k + threadIdx.x * A_CHUNK;
          if (k_ >= (uint32_t)K_hidden) break;
          const uint8_t* B = &W1_use[(m_curr + row_off) * K_packed + k_ / 2];
          for (int y = 0; y < YTILE_G1; y++) {
            const float* src = (const float*)(&B[y * K_packed]);
            #pragma unroll
            for (int i = 0; i < A_CHUNK / 8; i++)
              bigB_curr[y][k2].f[i] = loadnt((float*)&src[i]);
          }
        }
      }
      WAVE_PRIO_LOW();
      SCHED_GROUP_VMEM(0xff);

      for (uint32_t k1 = 0; k1 < (uint32_t)K_hidden;
           k1 += THRDS * A_CHUNK * UNRL_G1) {
        const uint32_t k1_next = k1 + THRDS * A_CHUNK * UNRL_G1;
        const bool has_next = k1_next < (uint32_t)K_hidden;

        // Prefetch next iter's bigB
        if (has_next) {
          WAVE_PRIO_HIGH();
          #pragma unroll
          for (uint32_t k2 = 0; k2 < UNRL_G1; k2++) {
            uint32_t k = k1_next + k2 * THRDS * A_CHUNK;
            uint32_t k_ = k + threadIdx.x * A_CHUNK;
            if (k_ >= (uint32_t)K_hidden) break;
            const uint8_t* B = &W1_use[(m_curr + row_off) * K_packed + k_ / 2];
            for (int y = 0; y < YTILE_G1; y++) {
              const float* src = (const float*)(&B[y * K_packed]);
              #pragma unroll
              for (int i = 0; i < A_CHUNK / 8; i++)
                bigB_next[y][k2].f[i] = loadnt((float*)&src[i]);
            }
          }
          WAVE_PRIO_LOW();
        }
        SCHED_GROUP_VMEM(0xff);

        // Load bigA from LDS
        #pragma unroll
        for (uint32_t k2 = 0; k2 < UNRL_G1; k2++) {
          uint32_t k = k1 + k2 * THRDS * A_CHUNK;
          uint32_t k_ = k + threadIdx.x * A_CHUNK;
          if (k_ >= (uint32_t)K_hidden) break;
          bigA_g1[k2] = *((const bigTypeA*)(&s[LDS_PAD(k_)]));
        }
        SCHED_GROUP_LDS(0xff);

        // Dequant + DOT2C (single stream)
        #pragma unroll
        for (uint32_t k2 = 0; k2 < UNRL_G1; k2++) {
          uint32_t k = k1 + k2 * THRDS * A_CHUNK;
          uint32_t k_ = k + threadIdx.x * A_CHUNK;
          if (k_ >= (uint32_t)K_hidden) break;

          #pragma unroll
          for (int y = 0; y < YTILE_G1; y++) {
            bigTypeA cvtB;
            DEQUANT_INT4(bigB_curr[y][k2], cvtB);
            if constexpr (GROUP_SIZE > 0) {
              float partial = 0;
              #pragma unroll
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(partial, bigA_g1[k2].f[b], cvtB.f[b])
              }
              uint32_t group_idx = k_ / GROUP_SIZE;
              acc[y] +=
                  partial *
                  __s2float(
                      S1_use[(m_curr + y + row_off) * num_groups_g1 +
                             group_idx]);
            }
          }
        }
        SCHED_GROUP_VALU(0xff);

        // Rotate
        #pragma unroll
        for (uint32_t k2 = 0; k2 < UNRL_G1; k2++) {
          #pragma unroll
          for (int y = 0; y < YTILE_G1; y++) {
            bigB_curr[y][k2] = bigB_next[y][k2];
          }
        }
      }
    };

    // -----------------------------------------------------------------
    // Per-WG inner: compute one row chunk (W*Y rows starting at
    // `m_chunk_base`) for group `g_cur`.  Re-resolves the per-expert
    // pointers each call so the same chunk runner serves both the native
    // group's drain phase and the cross-group help phase.  The pointer
    // math is ~3 SALU ops + 1 scalar L1 load (`expert_ids[g_cur]`) per
    // chunk; for the bench shape that's ~12 chunks/group × 8 groups ≈
    // 100 extra SALU ops over the kernel run -- well below the noise
    // floor.
    // -----------------------------------------------------------------
    auto do_p1_chunk = [&] (int g_cur, uint32_t m_chunk_base) {
      const int expert_id_h = expert_ids[g_cur];
      const uint8_t* W1_h = w1_packed_base +
          (long)expert_id_h * expert_stride_w1;
      const scalar_t* S1_h = w1_scale_base +
          (long)expert_id_h * expert_stride_s1;
      scalar_t* act_h = d_act_scratch + (long)g_cur * K_inter;

      uint32_t m = m_chunk_base + (uint32_t)threadIdx.y * YTILE_G1;
      // Per-wave bound: when K_inter % (W*Y) != 0 the tail chunk
      // straddles K_inter; waves whose first row is past K_inter must
      // skip the k-pass (otherwise the W1[m, k] addresses go past the
      // expert's row range).  For the bench shape K_inter=768 and
      // W*Y=64 divides evenly so this branch is taken on no chunk;
      // kept for forward-compat.  When W_g1 < W_g2 the launched WG has
      // WvPrGrpMax waves but only the first WvPrGrp_G1 should compute
      // P1 (chunk size = WvPrGrp_G1 * YTILE_G1; waves at threadIdx.y >=
      // WvPrGrp_G1 would land outside the chunk's row range).
      const bool wave_active = (threadIdx.y < WvPrGrp_G1) &&
                               (m < (uint32_t)K_inter);

      float gate_sum[YTILE_G1];
      float up_sum[YTILE_G1];
      #pragma unroll
      for (int y = 0; y < YTILE_G1; y++) {
        gate_sum[y] = 0.0f;
        up_sum[y]   = 0.0f;
      }

      if (wave_active) {
        run_kpass(m, /*is_up=*/false, gate_sum, W1_h, S1_h);
        run_kpass(m, /*is_up=*/true,  up_sum,   W1_h, S1_h);
      }

      #pragma unroll
      for (int y = 0; y < YTILE_G1; y++) {
        REDUCE_SUM_WAVE32(gate_sum[y]);
        REDUCE_SUM_WAVE32(up_sum[y]);
      }

      // Bit-exact silu_and_mul + scratch write.  Only lane 0 of each
      // wave writes (its post-reduction sum is shared across lanes).
      if (wave_active && threadIdx.x == 0) {
        #pragma unroll
        for (int y = 0; y < YTILE_G1; y++) {
          if ((m + y) < (uint32_t)K_inter) {
            half gate_h_v = __float2half_rn(gate_sum[y]);
            half up_h_v   = __float2half_rn(up_sum[y]);
            float gate_f = __half2float(gate_h_v);
            float up_f   = __half2float(up_h_v);
            float silu = gate_f / (1.0f + __expf(-gate_f));
            float act = silu * up_f;
            act_h[m + y] = __float2s<scalar_t>(act);
          }
        }
      }
    };

#ifdef FUSED_NO_STEAL
    // ---- Legacy static-stride dispatch (A/B fallback) ----
    {
      uint32_t m_static = (uint32_t)t * WvPrGrp_G1 * YTILE_G1;
      while (m_static < (uint32_t)K_inter) {
        do_p1_chunk(g, m_static);
        m_static += (uint32_t)wgs_per_group * WvPrGrp_G1 * YTILE_G1;
      }
    }
#else
    // ---- Cross-group P1 work-stealing ----
    constexpr uint32_t CHUNK_ROWS = (uint32_t)WvPrGrp_G1 * YTILE_G1;
    // 1) Drain native group first.
    while (true) {
      if (threadIdx.x == 0 && threadIdx.y == 0)
        m_shared = (uint32_t)atomicAdd(&d_p1_row_ctr[g],
                                       (int)CHUNK_ROWS);
      __syncthreads();
      uint32_t m_chunk = m_shared;
      __syncthreads();  // pin m_shared until all threads have read it
      if (m_chunk >= (uint32_t)K_inter) break;
      do_p1_chunk(g, m_chunk);
      if (threadIdx.x == 0 && threadIdx.y == 0)
        atomicAdd(&d_p1_done_ctr[g], 1);
    }
    // 2) Help every other group in round-robin order.
    for (int off = 1; off < num_groups; ++off) {
      const int g_h = (g + off) % num_groups;
      while (true) {
        if (threadIdx.x == 0 && threadIdx.y == 0)
          m_shared = (uint32_t)atomicAdd(&d_p1_row_ctr[g_h],
                                         (int)CHUNK_ROWS);
        __syncthreads();
        uint32_t m_chunk = m_shared;
        __syncthreads();
        if (m_chunk >= (uint32_t)K_inter) break;
        do_p1_chunk(g_h, m_chunk);
        if (threadIdx.x == 0 && threadIdx.y == 0)
          atomicAdd(&d_p1_done_ctr[g_h], 1);
      }
    }
#endif  // FUSED_NO_STEAL
  }

#endif  // FUSED_SKIP_PHASE1

  // -------------------------------------------------------------------------
  // Cross-WG barrier (per-group).  HIP `atomicAdd` defaults to system-scope
  // sequential consistency, so a release/acquire fence pair around the
  // atomic is redundant -- the atomic itself acts as the device-wide memory
  // ordering point.  We keep only the intra-WG `__syncthreads()` to ensure
  // ALL threads in this WG completed Phase 1 before any of them release the
  // barrier.
  //
  // Steal mode polls `d_p1_done_ctr[g]` directly: the barrier opens for
  // group g exactly when all `ceil(K_inter/(W*Y))` of g's chunks have been
  // processed (by native WGs or helpers).  Static mode falls back to the
  // wgs_per_group-arrival counter on `d_barrier[g]`.
  // -------------------------------------------------------------------------
  __syncthreads();
#ifdef FUSED_NO_STEAL
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd(&d_barrier[g], 1);
    while (atomicAdd(&d_barrier[g], 0) < wgs_per_group) { /* spin */ }
  }
#else
  {
    const int p1_chunks_per_group =
        (K_inter + WvPrGrp_G1 * YTILE_G1 - 1) / (WvPrGrp_G1 * YTILE_G1);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      while (atomicAdd(&d_p1_done_ctr[g], 0) < p1_chunks_per_group) {
        /* spin */
      }
    }
  }
#endif
  __syncthreads();

#ifdef FUSED_SKIP_PHASE2
  // Diagnostic: bypass Phase 2 entirely.  The atomicSub at the bottom
  // still runs to keep the barrier counter clean for the next launch.
  goto phase2_skip;
#endif

  // -------------------------------------------------------------------------
  // Cooperative load of d_act_scratch[g] into s[K_hidden_pad_offset..]
  // (the activation region of the padded LDS layout).  Regular load (not
  // NT) so the L2-resident scratch from Phase 1 is hit.
  // -------------------------------------------------------------------------
  {
    constexpr int total_threads = WvPrGrpMax * THRDS;
    const int tid_lin = threadIdx.y * THRDS + threadIdx.x;
    for (int k = 0; k < K_inter; k += total_threads * A_CHUNK) {
      int k_in = k + tid_lin * A_CHUNK;
      if (k_in >= K_inter) break;
      *((bigTypeA*)(&s[K_hidden_pad_offset + LDS_PAD(k_in)])) =
          *((const bigTypeA*)(&act_scratch_g[k_in]));
    }
  }
  __syncthreads();

  // =========================================================================
  // Phase 2: GEMM2.  Same K-loop structure as compute_sml_, but the LDS
  // base is at s[K_hidden] and the M-stride uses wgs_per_group (this
  // group's WGs cooperate, not all CuCount WGs).
  //
  // Only the first WvPrGrp_G2 waves of the WG do P2 work.  When
  // WvPrGrp_G1 > WvPrGrp_G2 the trailing waves enter this block (they
  // already did the cooperative load above) but their `m` is past
  // K_hidden so the while loop exits immediately for them.
  // =========================================================================
  if (threadIdx.y < WvPrGrp_G2) {
    const int K_packed = K_inter / 2;
    const int num_groups_g2 =
        (GROUP_SIZE > 0) ? (K_inter / GROUP_SIZE) : 0;

    float sum[YTILE_G2];
    bigTypeA bigA_g2[UNRL_G2];
    bigTypeW bigB[YTILE_G2][UNRL_G2];

    uint32_t m = (t * WvPrGrp_G2 + threadIdx.y) * YTILE_G2;
    while (m < (uint32_t)K_hidden) {
      #pragma unroll
      for (int y = 0; y < YTILE_G2; y++) sum[y] = 0.0f;

      for (uint32_t k1 = 0; k1 < (uint32_t)K_inter;
           k1 += THRDS * A_CHUNK * UNRL_G2) {
        // ---- Load bigB (NT) ----
        WAVE_PRIO_HIGH();
        #pragma unroll
        for (uint32_t k2 = 0; k2 < UNRL_G2; k2++) {
          uint32_t k = k1 + k2 * THRDS * A_CHUNK;
          uint32_t k_ = k + threadIdx.x * A_CHUNK;
          if (k_ >= (uint32_t)K_inter) break;

          const uint8_t* B_ = &W2[(m + 0) * K_packed + k_ / 2];
          for (int y = 0; y < YTILE_G2; y++) {
            const float* src = (const float*)(&B_[y * K_packed]);
            #pragma unroll
            for (int i = 0; i < A_CHUNK / 8; i++)
              bigB[y][k2].f[i] = loadnt((float*)&src[i]);
          }
        }
        WAVE_PRIO_LOW();
        SCHED_GROUP_VMEM(0xff);

        // ---- Load bigA from LDS activation region ----
        #pragma unroll
        for (uint32_t k2 = 0; k2 < UNRL_G2; k2++) {
          uint32_t k = k1 + k2 * THRDS * A_CHUNK;
          uint32_t k_ = k + threadIdx.x * A_CHUNK;
          if (k_ >= (uint32_t)K_inter) break;
          bigA_g2[k2] =
              *((const bigTypeA*)(&s[K_hidden_pad_offset + LDS_PAD(k_)]));
        }
        SCHED_GROUP_LDS(0xff);

        // ---- Dequant + DOT2C ----
        #pragma unroll
        for (uint32_t k2 = 0; k2 < UNRL_G2; k2++) {
          uint32_t k = k1 + k2 * THRDS * A_CHUNK;
          uint32_t k_ = k + threadIdx.x * A_CHUNK;
          if (k_ >= (uint32_t)K_inter) break;

          #pragma unroll
          for (int y = 0; y < YTILE_G2; y++) {
            bigTypeA cvtB;
            DEQUANT_INT4(bigB[y][k2], cvtB);

            if constexpr (GROUP_SIZE > 0) {
              float partial = 0;
              #pragma unroll
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(partial, bigA_g2[k2].f[b], cvtB.f[b])
              }
              uint32_t group_idx = k_ / GROUP_SIZE;
              sum[y] +=
                  partial *
                  __s2float(S2[(m + y) * num_groups_g2 + group_idx]);
            }
          }
        }
        SCHED_GROUP_VALU(0xff);
      }

      // ---- Wave reduce ----
      #pragma unroll
      for (int y = 0; y < YTILE_G2; y++) REDUCE_SUM_WAVE32(sum[y]);

      // ---- Write fp16 output to gemm2_out[slot, m] (regular store; no
      // NT here because no caller will reread it on the kernel side --
      // but the output is small enough that L2 traffic is negligible).
      if (threadIdx.x == 0) {
        #pragma unroll
        for (int y = 0; y < YTILE_G2; y++) {
          if ((m + y) < (uint32_t)K_hidden) {
            C[m + y] = __float2s<scalar_t>(sum[y]);
          }
        }
      }

      m += (uint32_t)wgs_per_group * WvPrGrp_G2 * YTILE_G2;
    }
  }

  // -------------------------------------------------------------------------
  // Reset counters for the next launch.
  //
  // FUSED_NO_STEAL:  each WG decrements `d_barrier[g]` once; sum is
  //   wgs_per_group, so d_barrier[g] returns to 0.  No memset needed.
  //
  // Steal mode:  the entry-side counters were incremented an unknown
  //   number of times (helpers + natives), so per-WG decrement does not
  //   restore them.  Instead we use `d_barrier[g]` as a per-group exit
  //   rendezvous: every WG of group g atomicAdd's once, and native t==0
  //   waits until the count == wgs_per_group, then writes 0 to all
  //   three counters.  Subsequent kernel launches on the same stream
  //   are serialized after this reset.
  // -------------------------------------------------------------------------
#ifdef FUSED_SKIP_PHASE2
phase2_skip:
#endif
  __syncthreads();
#ifdef FUSED_NO_STEAL
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicSub(&d_barrier[g], 1);
  }
#else
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    const int prev = atomicAdd(&d_barrier[g], 1);
    // The WG whose atomicAdd brings the count to wgs_per_group is the
    // last one out for group g.  It owns the per-group reset so that
    // subsequent launches see d_barrier[g] = 0 = d_p1_*_ctr[g].
    if (prev + 1 == wgs_per_group) {
      atomicExch(&d_p1_row_ctr[g],  0);
      atomicExch(&d_p1_done_ctr[g], 0);
      atomicExch(&d_barrier[g],     0);
    }
  }
#endif
  #undef LDS_PAD
}
#else   // !defined(__HIP__GFX1X__)
// Host-compile-pass stub.  The host wrapper expands `hipLaunchKernelGGL` with
// the kernel name as a template specialization; that template must be in
// scope on both the host and device passes for clang to emit a launch stub.
// Mirrors the upstream wvsplitk_int4_local.hip stub-on-non-RDNA pattern.
template <typename scalar_t, int THRDS, int YTILE_G1, int YTILE_G2,
          int WvPrGrp_G1, int WvPrGrp_G2,
          int A_CHUNK, int UNRL_G1, int UNRL_G2,
          int GROUP_SIZE>
__global__ void
moe_wvSplitK_int4_fused_grouped_(
    const int K_hidden, const int K_inter,
    const uint8_t* w1_packed_base, const scalar_t* w1_scale_base,
    const uint8_t* w2_packed_base, const scalar_t* w2_scale_base,
    const scalar_t* a_base, scalar_t* gemm2_out_base,
    const int* expert_ids, const int* sorted_token_ids, const int top_k,
    const long expert_stride_w1, const long expert_stride_s1,
    const long expert_stride_w2, const long expert_stride_s2,
    const int wgs_per_group, const int num_groups,
    scalar_t* d_act_scratch, int* d_barrier,
    int* d_p1_row_ctr, int* d_p1_done_ctr) {
  UNREACHABLE_CODE
}
#endif  // __HIP__GFX1X__

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------
//
// Allocates two persistent device buffers the first time it's called:
//   - d_act_scratch  [num_groups * K_inter]  (fp16/bf16)  ~12 KB for the bench
//   - d_barrier      [num_groups]            (int)        32 B for the bench
// Both are sized at "max seen" and reused across calls; the kernel itself
// resets d_barrier via atomicSub at the bottom so we never need a memset.
//
// Tensor shapes (decode-only, M=1, scattered):
//   a:                [M, K_hidden]               fp16/bf16
//   w1_packed:        [E, N_gateup, K_hidden//8]  int32 (ExLlama shuffle)
//   w1_scale:         [E, N_gateup, K_hidden//G]  fp16/bf16
//   w2_packed:        [E, K_hidden, K_inter//8]   int32
//   w2_scale:         [E, K_hidden, K_inter//G]   fp16/bf16
//   gemm2_out:        [P, K_hidden]               fp16/bf16  (slot-major)
//   expert_ids:       [P]                         int32
//   sorted_token_ids: [P]                         int32      (or empty)

namespace {

// Persistent scratch buffers, sized lazily.  Singleton-style (one set
// per process / GPU) -- the bench only ever runs against one device.
//
// Three int counter arrays, all sized `num_groups`:
//   - d_p1_row_ctr   : atomic dispatch counter for cross-group P1 rows.
//                      Each WG `atomicAdd(&d_p1_row_ctr[g], W*Y)` to grab
//                      a row chunk; reset at kernel exit.
//   - d_p1_done_ctr  : counts P1 chunks completed per group (native +
//                      helpers); the post-P1 barrier polls this until it
//                      reaches `ceil(K_inter / (W*Y))`.
//   - d_exit_barrier : one slot per group, counts arriving WGs at the end
//                      of Phase 2.  When full, native t==0 resets the
//                      three counters above for the next launch.
// The `d_barrier` field below is the legacy barrier counter still used by
// the static-stride fallback (FUSED_NO_STEAL build); when work-stealing is
// enabled (default) it's repurposed as `d_exit_barrier`.
struct FusedScratch {
  void* d_act_scratch = nullptr;
  int* d_barrier = nullptr;       // also used as exit-barrier in steal mode
  int* d_p1_row_ctr = nullptr;    // steal-mode only
  int* d_p1_done_ctr = nullptr;   // steal-mode only
  size_t act_bytes = 0;
  size_t barrier_count = 0;
  std::mutex m;
};

FusedScratch& scratch_singleton() {
  static FusedScratch s;
  return s;
}

void ensure_scratch(size_t want_act_bytes, int want_barrier_count) {
  auto& sc = scratch_singleton();
  std::lock_guard<std::mutex> lk(sc.m);
  if (sc.act_bytes < want_act_bytes) {
    // CUDA-style runtime API: hipify translates `cudaMalloc` /
    // `cudaFree` / `cudaMemset` / `cudaSuccess` to the hip equivalents
    // at build time.  Use the CUDA names so hipify recognizes the TU
    // as CUDA source and emits a .hip output (otherwise it sees the
    // file as already-HIP and silently skips, breaking the build).
    if (sc.d_act_scratch) (void)cudaFree(sc.d_act_scratch);
    sc.d_act_scratch = nullptr;
    if (cudaMalloc(&sc.d_act_scratch, want_act_bytes) != cudaSuccess) {
      throw std::runtime_error("fused_megakernel: cudaMalloc(d_act_scratch) failed");
    }
    sc.act_bytes = want_act_bytes;
  }
  if ((int)sc.barrier_count < want_barrier_count) {
    if (sc.d_barrier) (void)cudaFree(sc.d_barrier);
    if (sc.d_p1_row_ctr) (void)cudaFree(sc.d_p1_row_ctr);
    if (sc.d_p1_done_ctr) (void)cudaFree(sc.d_p1_done_ctr);
    sc.d_barrier = nullptr;
    sc.d_p1_row_ctr = nullptr;
    sc.d_p1_done_ctr = nullptr;
    const size_t bytes = (size_t)want_barrier_count * sizeof(int);
    if (cudaMalloc(&sc.d_barrier, bytes) != cudaSuccess ||
        cudaMalloc(&sc.d_p1_row_ctr, bytes) != cudaSuccess ||
        cudaMalloc(&sc.d_p1_done_ctr, bytes) != cudaSuccess) {
      throw std::runtime_error("fused_megakernel: cudaMalloc(counters) failed");
    }
    // Zero on first allocation so the spin-waits / atomicAdd dispatch see
    // a clean state.  The kernel resets all three counters at exit so we
    // never need a memset between subsequent launches.
    (void)cudaMemset(sc.d_barrier, 0, bytes);
    (void)cudaMemset(sc.d_p1_row_ctr, 0, bytes);
    (void)cudaMemset(sc.d_p1_done_ctr, 0, bytes);
    sc.barrier_count = want_barrier_count;
  }
}

}  // namespace

// Force-config env: WVSPLITK_FUSED_FORCE_YU
//
// Forms (parsed in order):
//   1. "Y<g1>U<u1>W<w1>Y<g2>U<u2>W<w2>"   -- per-phase asymmetric W.
//   2. "Y<g1>U<u1>Y<g2>U<u2>W<w>"          -- single shared W (back-compat).
//   3. "Y<g1>U<u1>Y<g2>U<u2>"              -- legacy 4-axis (defaults W=4).
//
// e.g. "Y4U2W16Y4U2W8" picks W_g1=16, W_g2=8.
struct FusedForceCfg {
  // Defaults: winning tuple from the 324-config WGP-mode sweep in
  // STATIC dispatch (Apr 2026 follow-up; the work-stealing path made
  // things worse once Phase 2 W=32 absorbs the imbalance via WGP
  // wave interleaving, so the production default is now static
  // dispatch + this tuple).  Y8U2W8Y4U2W32 ran at 97.68 us /
  // 180 GiB/s effective weight BW, ~5 us (4.8%) faster than the
  // pre-sweep Y4U2W16Y4U2W16 at 102.6 us.  See
  // wvsplitk_int4_local/OPTIMIZATION_LOG.md for the sweep tables.
  int yg1 = 8, ug1 = 2, w_g1 = 8;
  int yg2 = 4, ug2 = 2, w_g2 = 32;
  bool from_env = false;
};

static FusedForceCfg parse_fused_force() {
  FusedForceCfg cfg;
  const char* env = std::getenv("WVSPLITK_FUSED_FORCE_YU");
  if (!env || !env[0]) return cfg;
  int yg1 = 0, ug1 = 0, w_g1 = 0, yg2 = 0, ug2 = 0, w_g2 = 0, w = 0;
  // Form 1: per-phase W.
  if (std::sscanf(env, "Y%dU%dW%dY%dU%dW%d",
                  &yg1, &ug1, &w_g1, &yg2, &ug2, &w_g2) == 6 &&
      yg1 > 0 && ug1 > 0 && w_g1 > 0 &&
      yg2 > 0 && ug2 > 0 && w_g2 > 0) {
    cfg.yg1 = yg1; cfg.ug1 = ug1; cfg.w_g1 = w_g1;
    cfg.yg2 = yg2; cfg.ug2 = ug2; cfg.w_g2 = w_g2;
    cfg.from_env = true;
  }
  // Form 2: single shared W (back-compat).
  else if (std::sscanf(env, "Y%dU%dY%dU%dW%d",
                       &yg1, &ug1, &yg2, &ug2, &w) == 5 &&
           yg1 > 0 && ug1 > 0 && yg2 > 0 && ug2 > 0 && w > 0) {
    cfg.yg1 = yg1; cfg.ug1 = ug1; cfg.w_g1 = w;
    cfg.yg2 = yg2; cfg.ug2 = ug2; cfg.w_g2 = w;
    cfg.from_env = true;
  }
  // Form 3: legacy 4-axis (defaults W=4).
  else if (std::sscanf(env, "Y%dU%dY%dU%d",
                       &yg1, &ug1, &yg2, &ug2) == 4 &&
           yg1 > 0 && ug1 > 0 && yg2 > 0 && ug2 > 0) {
    cfg.yg1 = yg1; cfg.ug1 = ug1; cfg.w_g1 = 4;
    cfg.yg2 = yg2; cfg.ug2 = ug2; cfg.w_g2 = 4;
    cfg.from_env = true;
  }
  return cfg;
}

// FUSED_LAUNCH_TUPLE: now takes per-phase W (W_g1, W_g2).  The launch-time
// block_y is max(W_g1, W_g2) so both phases fit on the same WG; whichever
// phase has the smaller W has its trailing waves no-op via the wave_active
// guard (Phase 1) / outer if-gate (Phase 2).
#define FUSED_LAUNCH_TUPLE(_THRDS, _YG1, _UG1, _YG2, _UG2, _W1, _W2,       \
                           _AC, _GS)                                        \
  do {                                                                      \
    constexpr int _W_MAX = (_W1 > _W2) ? _W1 : _W2;                         \
    dim3 block(_THRDS, _W_MAX);                                             \
    dim3 grid((unsigned)(num_groups * wgs_per_group));                      \
    /* CUDA-style triple-chevron launch: hipify rewrites it into a HIP    \
       launch.  Using <<<>>> instead of hipLaunchKernelGGL keeps the    \
       file CUDA-recognisable so hipify emits the .hip output. */         \
    moe_wvSplitK_int4_fused_grouped_<fptype, _THRDS, _YG1, _YG2,            \
                                     _W1, _W2, _AC, _UG1, _UG2, _GS>       \
        <<<grid, block, 0, stream>>>(                                       \
        K_hidden, K_inter, w1ptr, s1ptr, w2ptr, s2ptr, aptr, cptr,         \
        eidptr, stidptr, top_k_in,                                          \
        expert_stride_w1, expert_stride_s1,                                 \
        expert_stride_w2, expert_stride_s2,                                 \
        wgs_per_group, num_groups,                                          \
        scratch_ptr, barrier_ptr,                                           \
        p1_row_ptr, p1_done_ptr);                                           \
  } while (0)

// Dispatch table: 324 (YG1, UG1, W_g1, YG2, UG2, W_g2) instantiations
// generated by `gen_fused_dispatch_grid.py` and emitted to
// `dispatch_grid.inc`.  THRDS and A_CHUNK fixed at 32 / 16.  Re-run the
// script after editing axis ranges; the .inc is checked into the tree so
// the JIT build doesn't need a Python codegen step.
//
// Implemented as a templated free function so we can `#include` the
// generated if/else cascade at file scope (a `#include` cannot live
// inside an AT_DISPATCH macro arg).  The `if constexpr` guard restricts
// the giant grid to the bench shape's (fp16, GS=128) combination; other
// (fptype, GS) combos fall through to the throw, keeping the kernel
// instantiation count bounded at 324 instead of 1296.
template <typename fptype, int _GS>
__attribute__((noinline))
static void fused_launch_grid(
    const FusedForceCfg& cfg,
    int K_hidden, int K_inter,
    const uint8_t* w1ptr, const fptype* s1ptr,
    const uint8_t* w2ptr, const fptype* s2ptr,
    const fptype* aptr, fptype* cptr,
    const int* eidptr, const int* stidptr, int top_k_in,
    long expert_stride_w1, long expert_stride_s1,
    long expert_stride_w2, long expert_stride_s2,
    int wgs_per_group, int num_groups,
    fptype* scratch_ptr, int* barrier_ptr,
    int* p1_row_ptr, int* p1_done_ptr,
    cudaStream_t stream) {
  if constexpr (std::is_same_v<fptype, half> && _GS == 128) {
    // Trimmed top-N tuples from the Apr 2026 WGP-mode sweep on the
    // decode shape (M=1 K_hidden=2048 K_inter=768 E=128 top_k=8).
    // Production default + a handful of near-best alternates.  See
    // rocm-scripts/wvsplitk_int4_local/OPTIMIZATION_LOG.md for the
    // sweep table.  The full 324/729-tuple sweep grid lives in
    // rocm-scripts (out-of-tree) so a vLLM build doesn't drag in a
    // ~13 MB .so.
    #include "skinny_gemms_int4_fused_dispatch_grid.inc"
  } else {
    throw std::runtime_error(
        "fused_megakernel: this build only supports (fp16, GS=128) -- "
        "the 324-tuple sweep grid is gated on that combo to keep the .so "
        "size in check.  Rebuild without the sweep grid for bf16/GS=32.");
  }
}

}  // anonymous namespace (closes the helpers/kernel-templates wrapper
   // opened near the top of the file so they don't ODR-collide with
   // skinny_gemms_int4.cu's identically-named template specializations).

torch::Tensor fused_moe_wvSplitK_int4_megakernel(
    torch::Tensor a, torch::Tensor w1, torch::Tensor s1,
    torch::Tensor w2, torch::Tensor s2, torch::Tensor gemm2_out,
    torch::Tensor expert_ids, torch::Tensor sorted_token_ids,
    int64_t top_k, int64_t group_size, int64_t cu_count) {
  TORCH_CHECK(a.is_cuda() && a.dim() == 2,
              "a must be 2-D CUDA tensor [M, K_hidden]");
  TORCH_CHECK(w1.is_cuda() && w1.dim() == 3,
              "w1 must be 3-D CUDA tensor [E, N_gateup, K_hidden//8]");
  TORCH_CHECK(s1.is_cuda() && s1.dim() == 3,
              "s1 must be 3-D CUDA tensor [E, N_gateup, K_hidden//G]");
  TORCH_CHECK(w2.is_cuda() && w2.dim() == 3,
              "w2 must be 3-D CUDA tensor [E, K_hidden, K_inter//8]");
  TORCH_CHECK(s2.is_cuda() && s2.dim() == 3,
              "s2 must be 3-D CUDA tensor [E, K_hidden, K_inter//G]");
  TORCH_CHECK(gemm2_out.is_cuda() && gemm2_out.dim() == 2,
              "gemm2_out must be 2-D CUDA tensor [P, K_hidden]");
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32,
              "expert_ids must be int32");
  TORCH_CHECK(sorted_token_ids.dtype() == torch::kInt32,
              "sorted_token_ids must be int32");

  const int K_hidden = static_cast<int>(a.size(1));
  const int N_gateup = static_cast<int>(w1.size(1));
  const int K_inter = static_cast<int>(w2.size(2)) * 8;
  TORCH_CHECK(N_gateup == 2 * K_inter,
              "fused_megakernel expects gated W1 (N_gateup == 2*K_inter); "
              "got N_gateup=",
              N_gateup, " K_inter=", K_inter);

  const int num_groups = static_cast<int>(expert_ids.size(0));
  TORCH_CHECK(num_groups > 0, "num_groups must be positive");

  // PRECONDITION (caller-enforced, not validated here): expert_ids must
  // be in [0, E).  Padding entries (-1) would deadlock the cross-WG
  // atomic barrier inside the kernel: the counter requires every WG to
  // arrive+spin, but a no-work WG returning early or doing a fast-path
  // arrive/depart races siblings still in the spin loop.  We do NOT
  // validate at runtime: `expert_ids.min().item()` would force a
  // device->host sync on every call (+15-20 us), which dominates the
  // kernel runtime.  The only call site (decode-only, block_size_m=1 in
  // hybrid_w4a16_moe.py) never produces -1 -- moe_align_block_size is
  // skipped entirely on that path.
  //
  // PRECONDITION (steal mode only -- default): every group's
  // `sorted_token_ids[g] / top_k` must resolve to the same `src_row`.
  // The cross-group P1 work-stealing path runs a helper WG's K-loop
  // against the **native** group's LDS-staged hidden_states; if a
  // helped group resolved to a different src_row, the helper would
  // produce garbage.  The decode (M=1) call site satisfies this: every
  // (token, expert) pair in `sorted_token_ids` shares the lone token's
  // row, so src_row==0 for all groups.  Build with -DFUSED_NO_STEAL
  // (or set WVSPLITK_FUSED_NO_STEAL=1 on the bench env) to fall back
  // to the static-stride dispatch when this invariant cannot be
  // guaranteed.

  // Strategy C: groups-of-CUs decomposition.  wgs_per_group derived from
  // grid budget so dim3(num_groups * wgs_per_group) <= cu_count.  For the
  // bench (8 groups, 40 CUs) this gives wgs_per_group = 5.  Caller passes
  // cu_count obtained from num_compute_units().
  int wgs_per_group = static_cast<int>(cu_count) / num_groups;
  if (wgs_per_group < 1) wgs_per_group = 1;
  // Cap to a reasonable upper bound -- launch grid otherwise grows
  // unbounded for E=128 calls etc.
  if (wgs_per_group > 16) wgs_per_group = 16;

  const long expert_stride_w1 =
      w1.stride(0) * static_cast<long>(sizeof(int32_t));
  const long expert_stride_s1 = s1.stride(0);
  const long expert_stride_w2 =
      w2.stride(0) * static_cast<long>(sizeof(int32_t));
  const long expert_stride_s2 = s2.stride(0);

  bool scattered = sorted_token_ids.numel() > 0;
  int top_k_in = scattered ? static_cast<int>(top_k) : 1;

  // Allocate persistent scratch the first time we see this size.  The
  // act_scratch byte size depends on dtype + shape; recompute every call.
  size_t act_bytes =
      (size_t)num_groups * (size_t)K_inter * a.element_size();
  ensure_scratch(act_bytes, num_groups);
  auto& sc = scratch_singleton();
  void* scratch_ptr_void = sc.d_act_scratch;
  int* barrier_ptr = sc.d_barrier;
  int* p1_row_ptr = sc.d_p1_row_ctr;
  int* p1_done_ptr = sc.d_p1_done_ctr;

  // PyTorch ROCm in this venv exposes the c10::cuda / at::cuda
  // symbols (the "MasqueradingAsCUDA" naming), not the HIP-suffixed
  // variants -- linking against c10_hip / torch_hip resolves to the
  // CUDA-named symbols.  Match skinny_gemms_int4.cu's API choice.
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  FusedForceCfg cfg = parse_fused_force();

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      a.scalar_type(), "fused_moe_wvSplitK_int4_megakernel", [&] {
        using fptype = typename scalar<scalar_t>::type;

        const uint8_t* w1ptr =
            reinterpret_cast<const uint8_t*>(w1.data_ptr());
        const fptype* s1ptr =
            reinterpret_cast<const fptype*>(s1.data_ptr());
        const uint8_t* w2ptr =
            reinterpret_cast<const uint8_t*>(w2.data_ptr());
        const fptype* s2ptr =
            reinterpret_cast<const fptype*>(s2.data_ptr());
        const fptype* aptr =
            reinterpret_cast<const fptype*>(a.data_ptr());
        fptype* cptr = reinterpret_cast<fptype*>(gemm2_out.data_ptr());
        const int* eidptr = expert_ids.data_ptr<int32_t>();
        const int* stidptr =
            scattered ? sorted_token_ids.data_ptr<int32_t>() : nullptr;
        fptype* scratch_ptr = reinterpret_cast<fptype*>(scratch_ptr_void);

        if (group_size == 128) {
          fused_launch_grid<fptype, 128>(
              cfg, K_hidden, K_inter,
              w1ptr, s1ptr, w2ptr, s2ptr, aptr, cptr,
              eidptr, stidptr, top_k_in,
              expert_stride_w1, expert_stride_s1,
              expert_stride_w2, expert_stride_s2,
              wgs_per_group, num_groups,
              scratch_ptr, barrier_ptr,
              p1_row_ptr, p1_done_ptr, stream);
        } else if (group_size == 32) {
          fused_launch_grid<fptype, 32>(
              cfg, K_hidden, K_inter,
              w1ptr, s1ptr, w2ptr, s2ptr, aptr, cptr,
              eidptr, stidptr, top_k_in,
              expert_stride_w1, expert_stride_s1,
              expert_stride_w2, expert_stride_s2,
              wgs_per_group, num_groups,
              scratch_ptr, barrier_ptr,
              p1_row_ptr, p1_done_ptr, stream);
        } else {
          throw std::runtime_error(
              "fused_megakernel: only group_size in {32, 128} supported");
        }
      });
  // Return the (in-place mutated) output tensor so the schema's
  // `Tensor` return + the `Tensor!` annotation on gemm2_out give
  // PyTorch's alias analysis a single tensor to track.
  return gemm2_out;
}

#undef FUSED_LAUNCH_TUPLE
