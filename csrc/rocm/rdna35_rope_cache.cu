// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The work a decode step does on Q, K and V between the qkv projection and
// RDNA35 decode attention, in one launch: optional per-head RMSNorm of Q and
// K, NeoX RoPE (full or partial), optional weightless RMSNorm of V, and the
// write of K and V into the packed paged cache, (blocks, Hkv, block, 2*D) in
// logical order with K and V the two halves of the last dimension.
//
// Decode only: T = nseq * M tokens with M <= 8, so the kernel is a few
// dependent memory round trips and nothing else.  The layout is chosen to
// overlap them:
//   - a head is served by a group of G lanes (G = D/2 / VEC rounded up to a
//     power of two, at most 32), so a wave serves 32/G heads of one token and
//     one kind (Q, K or V);
//   - a lane owns RoPE pairs (j, j + rot/2) for j < rot/2 and pass-through
//     elements [rot, D), each element exactly once, so the norm is a DPP
//     reduction inside the group;
//   - every load is issued up front and unconditionally, the address clamped
//     into the row and the value discarded where it does not belong: guarded
//     loads leave the waitcnt pass unable to count them, and
//     the cos/sin rows, which wait for the position, then overlap the Q/K
//     loads instead of following the norm.
//
// Rounding follows the native ops the kernel replaces: RMSNorm rounds to the
// weight's dtype before multiplying (vllm/ir/ops/layernorm.py), an fp32
// weight (GemmaRMSNorm's 1 + w) does not; RoPE is computed in fp32 from the
// rounded norm output and rounded once.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

namespace {

constexpr int kWave = 32;
constexpr int kWarps = 4;

// Q/K norm weight: none, the activation dtype, or fp32.
enum WeightMode { kNoWeight = 0, kWeightAct = 1, kWeightF32 = 2 };

constexpr int pow2_ceil(int x) {
  int p = 1;
  while (p < x) p *= 2;
  return p;
}

// Lanes per head and heads per wave for a head size and vector width.
template <int D, int VEC>
struct Geometry {
  static constexpr int kGroup =
      D / 2 / VEC >= kWave ? kWave : pow2_ceil(D / 2 / VEC);
  static constexpr int kHeadsPerWave = kWave / kGroup;
  static constexpr int kStep = kGroup * VEC;
  static constexpr int kPairChunks = (D / 2 + kStep - 1) / kStep;
  static constexpr int kPassChunks = (D + kStep - 1) / kStep;
};

// fp16 / bf16 bits <-> fp32.  The kernel works on 32-bit words, two elements
// at a time: gfx11 has no bf16 conversion instruction, and element-wise
// 16-bit handling costs a b16 move or select per element on top.
template <bool BF>
__device__ __forceinline__ float to_f(uint16_t x) {
  if constexpr (BF) {
    return __uint_as_float(static_cast<uint32_t>(x) << 16);
  } else {
    return __half2float(__ushort_as_half(x));
  }
}

// fp32 rounded to nearest even at bf16 precision, the result in the upper
// half of the word.  A NaN produced by arithmetic is the canonical quiet NaN,
// which the add leaves a NaN.
__device__ __forceinline__ uint32_t bf16_rne(float f) {
  const uint32_t u = __float_as_uint(f);
  return u + 0x7fffu + ((u >> 16) & 1u);
}

template <bool BF>
__device__ __forceinline__ uint16_t to_h(float f) {
  if constexpr (BF) {
    return static_cast<uint16_t>(bf16_rne(f) >> 16);
  } else {
    return __half_as_ushort(__float2half(f));
  }
}

// f rounded to the activation dtype, kept in fp32.
template <bool BF>
__device__ __forceinline__ float round_act(float f) {
  if constexpr (BF) {
    return __uint_as_float(bf16_rne(f) & 0xffff0000u);
  } else {
    return __half2float(__float2half(f));
  }
}

template <int N>
struct alignas(2 * N) Vec {
  uint16_t e[N];
};

template <int N>
struct alignas(4 * N > 16 ? 16 : 4 * N) FVec {
  float e[N];
};

template <bool BF, int N>
__device__ __forceinline__ void unpack(const Vec<N>& v, float (&f)[N]) {
  if constexpr (N == 1) {
    f[0] = to_f<BF>(v.e[0]);
  } else {
    uint32_t w[N / 2];
    __builtin_memcpy(w, &v, sizeof(w));
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      if constexpr (BF) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
      } else {
        f[2 * i] = __half2float(__ushort_as_half(static_cast<uint16_t>(w[i])));
        f[2 * i + 1] =
            __half2float(__ushort_as_half(static_cast<uint16_t>(w[i] >> 16)));
      }
    }
  }
}

template <bool BF, int N>
__device__ __forceinline__ Vec<N> pack(const float (&f)[N]) {
  Vec<N> v;
  if constexpr (N == 1) {
    v.e[0] = to_h<BF>(f[0]);
  } else {
    uint32_t w[N / 2];
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      if constexpr (BF) {
        // The upper halves of both rounded words in one v_perm_b32.
        w[i] = __builtin_amdgcn_perm(bf16_rne(f[2 * i + 1]), bf16_rne(f[2 * i]),
                                     0x07060302u);
      } else {
        w[i] = static_cast<uint32_t>(to_h<false>(f[2 * i])) |
               (static_cast<uint32_t>(to_h<false>(f[2 * i + 1])) << 16);
      }
    }
    __builtin_memcpy(&v, w, sizeof(w));
  }
  return v;
}

// The Q/K norm weight of one chunk, loaded with the activations.
template <bool BF, int WMODE, int N>
struct Weight {
  using Raw = std::conditional_t<WMODE == kWeightF32, FVec<N>, Vec<N>>;
  Raw raw;
  __device__ __forceinline__ void load(const void* w, int i) {
    if constexpr (WMODE == kWeightF32) {
      raw = *reinterpret_cast<const FVec<N>*>(static_cast<const float*>(w) + i);
    } else if constexpr (WMODE == kWeightAct) {
      raw =
          *reinterpret_cast<const Vec<N>*>(static_cast<const uint16_t*>(w) + i);
    }
  }
  __device__ __forceinline__ void floats(float (&f)[N]) const {
    if constexpr (WMODE == kWeightF32) {
#pragma unroll
      for (int e = 0; e < N; ++e) f[e] = raw.e[e];
    } else {
      unpack<BF, N>(raw, f);
    }
  }
};

// Every wave loads all of it (args_to_sgprs), so it is kept small: strides
// in elements fit 32 bits, and a grid of ~2000 waves otherwise spends its
// time on scalar argument traffic.
struct alignas(8) Params {
  const int64_t* pos;
  const int64_t* slots;
  uint16_t* q;
  uint16_t* q_out;
  uint16_t* k;
  uint16_t* k_out;
  uint16_t* v;      // read only
  uint16_t* v_out;  // V as written to the cache, optional
  // [max_pos, 2 * rot_half], cos then sin.  Without RoPE, rot_half is 0 and
  // this points at any valid row: every read lands on its first element.
  const uint16_t* cos_sin;
  const void* q_w;
  const void* k_w;
  uint16_t* cache;
  int32_t q_st, q_sh, qo_st, qo_sh, k_st, k_sh, ko_st, ko_sh, v_st, v_sh;
  int32_t vo_st, vo_sh;
  int32_t c_block, c_head, c_page;
  int32_t pos_offset;
  int32_t rot_half;
  int32_t block_size;
  // slot / block_size as a multiply-high, Hacker's Delight round-up method.
  uint32_t bs_magic;
  int32_t bs_shift;
  int32_t nslots;
  int32_t hq, nk, nv;  // heads of each kind in the grid
  float eps;
  float q_scale_beta;  // llama_4_scaling, 0 = off
  float q_scale_orig_max;
  int32_t v_norm;
};

// Sum over the G lanes of a head group, result in every lane of the group.
template <int G>
__device__ __forceinline__ float group_sum(float x) {
  // row_xmask:N reads lane i ^ N within the row of 16; the move is a free
  // modifier on the add, where __shfl_xor would go through ds_bpermute.
  auto xmask_add = [](float v, auto ctrl) {
    return v +
           __int_as_float(__builtin_amdgcn_update_dpp(
               0, __float_as_int(v), decltype(ctrl)::value, 0xf, 0xf, true));
  };
  if constexpr (G > 1) x = xmask_add(x, std::integral_constant<int, 0x161>{});
  if constexpr (G > 2) x = xmask_add(x, std::integral_constant<int, 0x162>{});
  if constexpr (G > 4) x = xmask_add(x, std::integral_constant<int, 0x164>{});
  if constexpr (G > 8) x = xmask_add(x, std::integral_constant<int, 0x168>{});
  if constexpr (G > 16) {
    // The other row of 16, same lane.
    x += __int_as_float(
        __builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x),
                                     0x76543210u, 0xfedcba98u, false, false));
  }
  return x;
}

// x already multiplied by rsqrt(mean + eps), then the weight with the
// rounding of the module the kernel replaces.
template <bool BF, int WMODE>
__device__ __forceinline__ float weighted(float x, float w) {
  if constexpr (WMODE == kWeightAct) {
    // RMSNorm: x.to(weight.dtype) * weight, rounded again by the cast back.
    return round_act<BF>(round_act<BF>(x) * w);
  } else {
    // GemmaRMSNorm: x * (1 + w) in fp32, rounded once.
    return round_act<BF>(x * w);
  }
}

// A read-only load the compiler must issue as a scalar load.
template <typename T>
__device__ __forceinline__ T load_const(const T* ptr, int i) {
  return ((const __attribute__((address_space(4))) T*)ptr)[i];
}

// Every kernel argument into SGPRs at once.  The compiler otherwise loads
// each field just before its first use, and with the Q/K/V branches in
// between that was five serial scalar round trips before the first
// activation load.
// The position and slot pointers come back out of the asm: read from the
// struct after it, the compiler reloads them from the argument segment.
__device__ __forceinline__ void args_to_sgprs(const Params& p,
                                              const int64_t*& pos,
                                              const int64_t*& slots) {
  constexpr int kWords = sizeof(Params) / sizeof(uint64_t);
  constexpr int kPos = offsetof(Params, pos) / sizeof(uint64_t);
  constexpr int kSlots = offsetof(Params, slots) / sizeof(uint64_t);
  uint64_t w[kWords];
  __builtin_memcpy(w, &p, sizeof(w));
#pragma unroll
  for (int i = 0; i < kWords; ++i) {
    if (i == kPos || i == kSlots) {
      asm volatile("" : "+s"(w[i]));
    } else {
      asm volatile("" ::"s"(w[i]));
    }
  }
  pos = reinterpret_cast<const int64_t*>(w[kPos]);
  slots = reinterpret_cast<const int64_t*>(w[kSlots]);
}

__device__ __forceinline__ void split_slot(const Params& p, int64_t slot,
                                           int64_t& block, int& offset) {
  const uint32_t n = static_cast<uint32_t>(slot);
  const uint32_t t = __umulhi(n, p.bs_magic);
  const uint32_t b = (t + ((n - t) >> 1)) >> (p.bs_shift - 1);
  block = b;
  offset = static_cast<int>(n - b * static_cast<uint32_t>(p.block_size));
}

// FULL: RoPE over the whole head (rot == D, most models), so there is no
// pass-through part and the rotation half is a constant.
template <bool BF, int D, int VEC, int WMODE, bool FULL>
__global__ __launch_bounds__(kWave* kWarps) void rope_cache_kernel(Params p) {
  using V = Vec<VEC>;
  using W = Weight<BF, WMODE, VEC>;
  using Geo = Geometry<D, VEC>;
  constexpr int G = Geo::kGroup;
  constexpr int HPW = Geo::kHeadsPerWave;
  constexpr int NR = Geo::kPairChunks;
  constexpr int NP = Geo::kPassChunks;
  // Pass-through chunks of a Q/K row; arrays keep one element when there are
  // none.
  constexpr int NQ = FULL ? 0 : NP;
  constexpr int NQA = NQ > 0 ? NQ : 1;
  constexpr bool kNorm = WMODE != kNoWeight;

  const int64_t* pos_ptr;
  const int64_t* slots_ptr;
  args_to_sgprs(p, pos_ptr, slots_ptr);
  const int t = blockIdx.x;
  // Through the constant address space the position and the slot stay scalar
  // loads after the asm above (which defeats the no-clobber proof a global
  // pointer needs), and are waited for only where used: the activation loads
  // go out first.
  const int64_t pos = load_const(pos_ptr, t);
  const int64_t slot_raw = load_const(slots_ptr, t < p.nslots ? t : 0);
  const int wave = blockIdx.y * kWarps + threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  const int gl = lane % G;
  const int sub = lane / G;

  // A branch-free prologue: every kernel argument is needed whatever the wave
  // serves, so all of them arrive in one round trip, and the position, the
  // slot and the activations in the next.  Branching on Q/K/V first let the
  // compiler sink half the argument loads into the branches, two round trips
  // before the position was even requested.
  const int waves_q = (p.hq + HPW - 1) / HPW;
  const int waves_k = (p.nk + HPW - 1) / HPW;
  const int waves_v = (p.nv + HPW - 1) / HPW;
  const bool is_q = wave < waves_q;
  const bool is_v = wave >= waves_q + waves_k;
  const int first = is_q   ? wave
                    : is_v ? wave - waves_q - waves_k
                           : wave - waves_q;
  const int nh = is_q ? p.hq : is_v ? p.nv : p.nk;
  const int head_raw = first * HPW + sub;
  // Lanes past the last head of their kind, and the waves that fill the last
  // workgroup, read a clamped row and store nothing.
  const bool live = head_raw < nh && wave < waves_q + waves_k + waves_v;
  const int head = max(0, min(head_raw, nh - 1));
  uint16_t* src = (is_q   ? p.q
                   : is_v ? p.v
                          : p.k) +
                  t * (is_q   ? p.q_st
                       : is_v ? p.v_st
                              : p.k_st) +
                  head * (is_q   ? p.q_sh
                          : is_v ? p.v_sh
                                 : p.k_sh);

  if (is_v) {
    // V: copy into the cache, after an optional weightless RMSNorm, and to
    // v_out if given.
    V x[NP];
#pragma unroll
    for (int i = 0; i < NP; ++i) {
      const int j = (i * G + gl) * VEC;
      x[i] = *(const V*)(src + (j < D ? j : 0));
    }
    // The loads go out before the slot is waited for; otherwise the compiler
    // tests the slot first, since a negative one makes the loads dead.
    __builtin_amdgcn_sched_barrier(0);
    int64_t slot_v = slot_raw;
    asm volatile("" : "+s"(slot_v));
    const int64_t slot = t < p.nslots ? slot_v : int64_t{-1};
    uint16_t* vout =
        p.v_out == nullptr ? nullptr : p.v_out + t * p.vo_st + head * p.vo_sh;
    if (slot < 0 && vout == nullptr) return;
    uint16_t* dst = nullptr;
    if (slot >= 0) {
      int64_t block;
      int offset;
      split_slot(p, slot, block, offset);
      dst =
          p.cache + block * p.c_block + head * p.c_head + offset * p.c_page + D;
    }
    if (p.v_norm) {
      float f[NP][VEC];
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < NP; ++i) {
        unpack<BF, VEC>(x[i], f[i]);
        const bool ok = (i * G + gl) * VEC < D;
#pragma unroll
        for (int e = 0; e < VEC; ++e) ss += ok ? f[i][e] * f[i][e] : 0.f;
      }
      const float r = rsqrtf(group_sum<G>(ss) / D + p.eps);
#pragma unroll
      for (int i = 0; i < NP; ++i) {
#pragma unroll
        for (int e = 0; e < VEC; ++e) f[i][e] *= r;
        x[i] = pack<BF, VEC>(f[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < NP; ++i) {
      const int j = (i * G + gl) * VEC;
      if (live && j < D) {
        if (dst != nullptr) *(V*)(dst + j) = x[i];
        if (vout != nullptr) *(V*)(vout + j) = x[i];
      }
    }
    return;
  }

  // Q or K.  Every load first: activations, weights, then the cos/sin rows,
  // which wait for the position.
  const int rh = FULL ? D / 2 : p.rot_half;
  const int rot = 2 * rh;
  const void* wp = is_q ? p.q_w : p.k_w;

  V xa[NR], xb[NR], xc[NQA], vc[NR], vs[NR];
  W wa[NR], wb[NR], wc[NQA];
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    const int j = (i * G + gl) * VEC;
    const int jc = j < rh ? j : 0;
    xa[i] = *(const V*)(src + jc);
    xb[i] = *(const V*)(src + rh + jc);
    if constexpr (kNorm) {
      wa[i].load(wp, jc);
      wb[i].load(wp, rh + jc);
    }
  }
#pragma unroll
  for (int i = 0; i < NQ; ++i) {
    const int j = rot + (i * G + gl) * VEC;
    const int jc = j < D ? j : 0;
    xc[i] = *(const V*)(src + jc);
    if constexpr (kNorm) wc[i].load(wp, jc);
  }
  // The activations and weights go out before the position is waited for;
  // the compiler otherwise hoists the cos/sin addresses, and the wait, above
  // them.
  __builtin_amdgcn_sched_barrier(0);
  int64_t pos_q = pos;
  asm volatile("" : "+s"(pos_q));
  const uint16_t* cs = p.cos_sin + (pos_q + p.pos_offset) * rot;
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    const int j = (i * G + gl) * VEC;
    const int jc = j < rh ? j : 0;
    vc[i] = *(const V*)(cs + jc);
    vs[i] = *(const V*)(cs + rh + jc);
  }
  const int64_t slot = !is_q && t < p.nslots ? slot_raw : int64_t{-1};

  float a[NR][VEC], b[NR][VEC], c[NQA][VEC];
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    unpack<BF, VEC>(xa[i], a[i]);
    unpack<BF, VEC>(xb[i], b[i]);
  }
#pragma unroll
  for (int i = 0; i < NQ; ++i) unpack<BF, VEC>(xc[i], c[i]);

  if constexpr (kNorm) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      const bool ok = (i * G + gl) * VEC < rh;
#pragma unroll
      for (int e = 0; e < VEC; ++e)
        ss += ok ? a[i][e] * a[i][e] + b[i][e] * b[i][e] : 0.f;
    }
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
      const bool ok = rot + (i * G + gl) * VEC < D;
#pragma unroll
      for (int e = 0; e < VEC; ++e) ss += ok ? c[i][e] * c[i][e] : 0.f;
    }
    const float r = rsqrtf(group_sum<G>(ss) / D + p.eps);
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      float fa[VEC], fb[VEC];
      wa[i].floats(fa);
      wb[i].floats(fb);
#pragma unroll
      for (int e = 0; e < VEC; ++e) {
        a[i][e] = weighted<BF, WMODE>(a[i][e] * r, fa[e]);
        b[i][e] = weighted<BF, WMODE>(b[i][e] * r, fb[e]);
      }
    }
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
      float fc[VEC];
      wc[i].floats(fc);
#pragma unroll
      for (int e = 0; e < VEC; ++e)
        c[i][e] = weighted<BF, WMODE>(c[i][e] * r, fc[e]);
    }
  }

  // Mistral's llama_4_scaling multiplies the rotated Q, rounded, by a
  // position-dependent factor; the store rounds again.
  const bool scale_q = is_q && p.q_scale_beta != 0.f;
  const float qs =
      scale_q
          ? 1.f + p.q_scale_beta * logf(1.f + floorf(static_cast<float>(pos_q) /
                                                     p.q_scale_orig_max))
          : 1.f;
  auto finish = [&](float y) { return scale_q ? round_act<BF>(y) * qs : y; };

  uint16_t* const out_base = is_q ? p.q_out : p.k_out;
  uint16_t* dst = out_base == nullptr
                      ? src
                      : (is_q ? p.q_out + t * p.qo_st + head * p.qo_sh
                              : p.k_out + t * p.ko_st + head * p.ko_sh);
  uint16_t* cdst = nullptr;
  if (slot >= 0) {
    int64_t block;
    int offset;
    split_slot(p, slot, block, offset);
    cdst = p.cache + block * p.c_block + head * p.c_head + offset * p.c_page;
  }

#pragma unroll
  for (int i = 0; i < NR; ++i) {
    const int j = (i * G + gl) * VEC;
    if (live && j < rh) {
      float co[VEC], si[VEC], o1[VEC], o2[VEC];
      unpack<BF, VEC>(vc[i], co);
      unpack<BF, VEC>(vs[i], si);
#pragma unroll
      for (int e = 0; e < VEC; ++e) {
        o1[e] = finish(a[i][e] * co[e] - b[i][e] * si[e]);
        o2[e] = finish(b[i][e] * co[e] + a[i][e] * si[e]);
      }
      const V r1 = pack<BF, VEC>(o1), r2 = pack<BF, VEC>(o2);
      *(V*)(dst + j) = r1;
      *(V*)(dst + rh + j) = r2;
      if (cdst != nullptr) {
        *(V*)(cdst + j) = r1;
        *(V*)(cdst + rh + j) = r2;
      }
    }
  }
  // Pass-through elements change only through the norm or the Q scaling;
  // in place without either there is nothing to write back.
  const bool write_pass = out_base != nullptr || kNorm || scale_q;
#pragma unroll
  for (int i = 0; i < NQ; ++i) {
    const int j = rot + (i * G + gl) * VEC;
    if (live && j < D) {
      float o[VEC];
#pragma unroll
      for (int e = 0; e < VEC; ++e) o[e] = finish(c[i][e]);
      const V r = pack<BF, VEC>(o);
      if (write_pass) *(V*)(dst + j) = r;
      if (cdst != nullptr) *(V*)(cdst + j) = r;
    }
  }
}

template <bool BF, int D, int VEC, int WMODE>
void launch(const Params& p, int num_tokens, cudaStream_t stream) {
  const bool full = 2 * p.rot_half == D;
  constexpr int HPW = Geometry<D, VEC>::kHeadsPerWave;
  const int waves =
      (p.hq + HPW - 1) / HPW + (p.nk + HPW - 1) / HPW + (p.nv + HPW - 1) / HPW;
  const dim3 grid(num_tokens, (waves + kWarps - 1) / kWarps);
  if (full) {
    rope_cache_kernel<BF, D, VEC, WMODE, true>
        <<<grid, kWave * kWarps, 0, stream>>>(p);
  } else {
    rope_cache_kernel<BF, D, VEC, WMODE, false>
        <<<grid, kWave * kWarps, 0, stream>>>(p);
  }
}

template <bool BF, int D, int VEC>
void dispatch_weight(const Params& p, int wmode, int n, cudaStream_t s) {
  switch (wmode) {
    case kNoWeight:
      return launch<BF, D, VEC, kNoWeight>(p, n, s);
    case kWeightAct:
      return launch<BF, D, VEC, kWeightAct>(p, n, s);
    default:
      return launch<BF, D, VEC, kWeightF32>(p, n, s);
  }
}

template <bool BF, int D>
void dispatch_vec(const Params& p, int vec, int wmode, int n, cudaStream_t s) {
  switch (vec) {
    case 8:
      return dispatch_weight<BF, D, 8>(p, wmode, n, s);
    case 4:
      return dispatch_weight<BF, D, 4>(p, wmode, n, s);
    case 2:
      return dispatch_weight<BF, D, 2>(p, wmode, n, s);
    default:
      return dispatch_weight<BF, D, 1>(p, wmode, n, s);
  }
}

template <bool BF>
void dispatch_dim(const Params& p, int64_t d, int vec, int wmode, int n,
                  cudaStream_t s) {
  switch (d) {
    case 64:
      return dispatch_vec<BF, 64>(p, vec, wmode, n, s);
    case 96:
      return dispatch_vec<BF, 96>(p, vec, wmode, n, s);
    case 128:
      return dispatch_vec<BF, 128>(p, vec, wmode, n, s);
    case 256:
      return dispatch_vec<BF, 256>(p, vec, wmode, n, s);
    default:
      return dispatch_vec<BF, 512>(p, vec, wmode, n, s);
  }
}

// Strides and offsets travel as 32 bits in Params.
int32_t narrow(int64_t x) {
  TORCH_CHECK(x >= INT32_MIN && x <= INT32_MAX, "stride ", x,
              " does not fit 32 bits");
  return static_cast<int32_t>(x);
}

// [T, H, D] with a unit last stride; the token and head strides are free.
void check_heads(const torch::Tensor& x, const char* name, int64_t T, int64_t D,
                 at::ScalarType dtype) {
  TORCH_CHECK(x.dim() == 3 && x.size(0) == T && x.size(2) == D, name,
              " must be [", T, ", H, ", D, "], got ", x.sizes());
  TORCH_CHECK(x.stride(2) == 1, name, " must have a unit last stride");
  TORCH_CHECK(x.scalar_type() == dtype, name, " must be ", dtype);
}

}  // namespace

void rdna35_rope_cache(
    torch::Tensor& positions, torch::Tensor& q, std::optional<torch::Tensor> k,
    std::optional<torch::Tensor> v, std::optional<torch::Tensor> cos_sin_cache,
    std::optional<torch::Tensor> q_weight,
    std::optional<torch::Tensor> k_weight, bool v_norm, double eps,
    int64_t pos_offset, double q_scale_beta, int64_t q_scale_orig_max,
    std::optional<torch::Tensor> kv_cache,
    std::optional<torch::Tensor> slot_mapping,
    std::optional<torch::Tensor> q_out, std::optional<torch::Tensor> k_out,
    std::optional<torch::Tensor> v_out) {
  const auto dtype = q.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "q must be fp16 or bf16, got ", dtype);
  const int64_t T = q.size(0);
  const int64_t D = q.size(2);
  TORCH_CHECK(D == 64 || D == 96 || D == 128 || D == 256 || D == 512,
              "head size ", D, " not built");
  check_heads(q, "q", T, D, dtype);
  if (q_out) check_heads(*q_out, "q_out", T, D, dtype);
  TORCH_CHECK(!q_out || q_out->size(1) == q.size(1), "q_out heads != q heads");
  if (k) check_heads(*k, "k", T, D, dtype);
  if (k_out) {
    TORCH_CHECK(k.has_value(), "k_out without k");
    check_heads(*k_out, "k_out", T, D, dtype);
    TORCH_CHECK(k_out->size(1) == k->size(1), "k_out heads != k heads");
  }
  if (v) check_heads(*v, "v", T, D, dtype);
  if (v_out) {
    TORCH_CHECK(v.has_value(), "v_out without v");
    check_heads(*v_out, "v_out", T, D, dtype);
    TORCH_CHECK(v_out->size(1) == v->size(1), "v_out heads != v heads");
  }
  TORCH_CHECK(!v_norm || v.has_value(), "v_norm without v");

  // mrope/imrope: in decode the three position rows are equal, so row 0 is
  // plain RoPE.  Not checked: the caller guarantees it (multimodal prefill,
  // whose rows differ, must not come here).
  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");
  TORCH_CHECK((positions.dim() == 1 ||
               (positions.dim() == 2 && positions.size(0) == 3)) &&
                  positions.size(-1) == T && positions.stride(-1) == 1,
              "positions must be a contiguous [T] or [3, T], got ",
              positions.sizes());

  int rot_half = 0;
  if (cos_sin_cache) {
    const auto& cs = *cos_sin_cache;
    TORCH_CHECK(
        cs.dim() == 2 && cs.is_contiguous() && cs.scalar_type() == dtype,
        "cos_sin_cache must be a contiguous [max_pos, rot] ", dtype);
    TORCH_CHECK(cs.size(1) % 2 == 0 && cs.size(1) <= D,
                "rotary dim must be even and at most ", D);
    rot_half = cs.size(1) / 2;
  }

  int wmode = kNoWeight;
  if (q_weight) {
    const auto wt = q_weight->scalar_type();
    TORCH_CHECK(wt == dtype || wt == at::kFloat, "q_weight must be ", dtype,
                " or fp32");
    wmode = wt == at::kFloat ? kWeightF32 : kWeightAct;
    TORCH_CHECK(q_weight->numel() == D && q_weight->is_contiguous(),
                "q_weight must be a contiguous [", D, "]");
    if (k) {
      TORCH_CHECK(k_weight.has_value(), "q is normalised but k has no weight");
    }
  }
  if (k_weight) {
    TORCH_CHECK(q_weight.has_value() &&
                    k_weight->scalar_type() == q_weight->scalar_type() &&
                    k_weight->numel() == D && k_weight->is_contiguous(),
                "k_weight must be a contiguous [", D, "] of q_weight's dtype");
  }

  const bool write_cache = kv_cache.has_value();
  if (write_cache) {
    TORCH_CHECK(k.has_value() && v.has_value(),
                "the cache write needs k and v");
    TORCH_CHECK(slot_mapping.has_value(), "the cache write needs slot_mapping");
    TORCH_CHECK(v->size(1) == k->size(1), "k and v head counts differ");
    const auto& c = *kv_cache;
    TORCH_CHECK(c.dim() == 4 && c.size(1) == k->size(1) && c.size(3) == 2 * D &&
                    c.stride(3) == 1 && c.scalar_type() == dtype,
                "kv_cache must be (blocks, Hkv, block, 2*D) of ", dtype,
                " with a unit last stride, got ", c.sizes());
    TORCH_CHECK(c.size(2) >= 2 && c.size(0) * c.size(2) < (int64_t{1} << 32),
                "slots must fit in 32 bits and blocks hold at least 2");
    TORCH_CHECK(slot_mapping->scalar_type() == at::kLong &&
                    slot_mapping->dim() == 1 && slot_mapping->size(0) <= T &&
                    slot_mapping->is_contiguous(),
                "slot_mapping must be a contiguous int64 [<= T]");
  }
  TORCH_CHECK(!v.has_value() || write_cache, "v is only used by the write");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const auto* props = at::cuda::getCurrentDeviceProperties();
  // Head groups and the DPP reduction assume wave32.
  TORCH_CHECK(props->warpSize == kWave,
              "rdna35_rope_cache requires wave32; device reports warpSize ",
              props->warpSize);
  const std::string arch(props->gcnArchName);
  TORCH_CHECK(arch.rfind("gfx115", 0) == 0,
              "rdna35_rope_cache is RDNA3.5-only; device is ", arch);
  if (T == 0) return;

  Params p{};
  p.pos = positions.data_ptr<int64_t>();  // row 0 of [3, T]
  p.pos_offset = narrow(pos_offset);
  p.q = reinterpret_cast<uint16_t*>(q.data_ptr());
  p.q_st = narrow(q.stride(0));
  p.q_sh = narrow(q.stride(1));
  if (q_out) {
    p.q_out = reinterpret_cast<uint16_t*>(q_out->data_ptr());
    p.qo_st = narrow(q_out->stride(0));
    p.qo_sh = narrow(q_out->stride(1));
  }
  if (k) {
    p.k = reinterpret_cast<uint16_t*>(k->data_ptr());
    p.k_st = narrow(k->stride(0));
    p.k_sh = narrow(k->stride(1));
  }
  if (k_out) {
    p.k_out = reinterpret_cast<uint16_t*>(k_out->data_ptr());
    p.ko_st = narrow(k_out->stride(0));
    p.ko_sh = narrow(k_out->stride(1));
  }
  if (v_out) {
    p.v_out = reinterpret_cast<uint16_t*>(v_out->data_ptr());
    p.vo_st = narrow(v_out->stride(0));
    p.vo_sh = narrow(v_out->stride(1));
  }
  if (v) {
    p.v = reinterpret_cast<uint16_t*>(v->data_ptr());
    p.v_st = narrow(v->stride(0));
    p.v_sh = narrow(v->stride(1));
  }
  p.cos_sin = cos_sin_cache
                  ? reinterpret_cast<const uint16_t*>(cos_sin_cache->data_ptr())
                  : p.q;
  p.rot_half = rot_half;
  p.q_w = q_weight ? q_weight->data_ptr() : nullptr;
  p.k_w = k_weight ? k_weight->data_ptr() : nullptr;
  if (write_cache) {
    p.cache = reinterpret_cast<uint16_t*>(kv_cache->data_ptr());
    p.c_block = narrow(kv_cache->stride(0));
    p.c_head = narrow(kv_cache->stride(1));
    p.c_page = narrow(kv_cache->stride(2));
    p.block_size = kv_cache->size(2);
    int s = 0;
    while ((int64_t{1} << s) < p.block_size) ++s;
    p.bs_shift = s;
    p.bs_magic = static_cast<uint32_t>(
        ((uint64_t{1} << 32) * ((uint64_t{1} << s) - p.block_size)) /
            p.block_size +
        1);
    p.slots = slot_mapping->data_ptr<int64_t>();
    p.nslots = slot_mapping->size(0);
  }
  // The prologue reads a row and a slot whatever the wave serves: absent
  // tensors point at q and at the positions.
  if (!k) {
    p.k = p.q;
    p.k_st = p.q_st;
    p.k_sh = p.q_sh;
  }
  if (!write_cache) {
    p.v = p.q;
    p.v_st = p.q_st;
    p.v_sh = p.q_sh;
    p.slots = p.pos;
    p.nslots = 0;
  }
  p.hq = q.size(1);
  p.nk = k ? k->size(1) : 0;
  p.nv = write_cache ? v->size(1) : 0;
  p.eps = static_cast<float>(eps);
  p.q_scale_beta = static_cast<float>(q_scale_beta);
  p.q_scale_orig_max = static_cast<float>(q_scale_orig_max);
  TORCH_CHECK(q_scale_beta == 0.0 || q_scale_orig_max > 0,
              "q scaling needs a positive original max position");
  p.v_norm = v_norm;

  // Whether every row start, rotation half and stride allows a vector width.
  auto fits = [&](int vec) {
    auto aligned = [](const void* ptr, int64_t bytes) {
      return ptr == nullptr || reinterpret_cast<uintptr_t>(ptr) % bytes == 0;
    };
    const int64_t bytes = 2 * vec;
    const int64_t w_bytes = wmode == kWeightF32 ? std::min(16, 4 * vec) : bytes;
    const int64_t strides[] = {p.q_st,  p.q_sh,  p.qo_st,   p.qo_sh,  p.k_st,
                               p.k_sh,  p.ko_st, p.ko_sh,   p.v_st,   p.v_sh,
                               p.vo_st, p.vo_sh, p.c_block, p.c_head, p.c_page};
    for (const int64_t s : strides)
      if (s % vec) return false;
    return rot_half % vec == 0 && D % vec == 0 && aligned(p.q, bytes) &&
           aligned(p.q_out, bytes) && aligned(p.k, bytes) &&
           aligned(p.k_out, bytes) && aligned(p.v, bytes) &&
           aligned(p.v_out, bytes) && aligned(p.cos_sin, bytes) &&
           aligned(p.cache, bytes) && aligned(p.q_w, w_bytes) &&
           aligned(p.k_w, w_bytes);
  };
  // As many lanes per head as keeps one chunk of RoPE pairs per lane: the
  // per-lane chain of conversions and roundings is what a decode call waits
  // on.  Wider vectors once the grid passes ~800 waves, where wave count
  // costs more than lane work (measured over T = 1..32, D = 64..512).
  auto waves = [&](int vec) {
    int g = 1;
    while (g < D / 2 / vec && g < kWave) g *= 2;
    const int hpw = kWave / g;
    return T * ((p.hq + hpw - 1) / hpw + (p.nk + hpw - 1) / hpw +
                (p.nv + hpw - 1) / hpw);
  };
  int vec = std::max<int>(2, D / 64);
  while (vec < 8 && waves(vec) > 800) vec *= 2;
  while (vec > 1 && !fits(vec)) vec /= 2;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (dtype == at::kBFloat16) {
    dispatch_dim<true>(p, D, vec, wmode, T, stream);
  } else {
    dispatch_dim<false>(p, D, vec, wmode, T, stream);
  }
}

#ifdef RDNA35_ROPE_TORCH_EXT
  #include <torch/extension.h>
// The kernel tools build this file on its own to try variants.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_cache", &rdna35_rope_cache, "", py::arg("positions"),
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("cos_sin_cache"),
        py::arg("q_weight"), py::arg("k_weight"), py::arg("v_norm"),
        py::arg("eps"), py::arg("pos_offset"), py::arg("q_scale_beta"),
        py::arg("q_scale_orig_max"), py::arg("kv_cache"),
        py::arg("slot_mapping"), py::arg("q_out"), py::arg("k_out"),
        py::arg("v_out") = py::none());
}
#endif
