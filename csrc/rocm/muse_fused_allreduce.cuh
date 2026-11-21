// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Muse Spark TP all-reduces fused with the kernels around them (gfx950, xGMI
// peers), for NR ranks:
//   finalize: MoE post-expert finalize -> all-reduce [-> per-token FP8 rows for
//   the next GEMM];
//   boundary: attention all-reduce -> post-norm -> gated residual -> pre-norm
//   (BF16 and/or FP8);
// and the multi-block variants of both: the same ops with each row served by
// three blocks (small M at large NR, below).
// The finalize and the all-reduce protocol are adapted from SGLang's
// rocm_muse_spark/finalize_allreduce.hip (Apache-2.0).
//
// One block per token row. The block computes this rank's partial row and
// all-reduces it with the same row's blocks on the other ranks:
//   two-hop: the partial goes to the row's owner rank (r * NR) / M, which sums
//   the NR partials in rank order (fp32, one bf16 rounding) and pushes the sum
//   to the other ranks;
//   one-hop: every rank pushes its partial to every peer and sums the NR in
//   rank order itself;
//   chunked two-hop: rank j owns the j-th NR-th of every row; each rank pushes
//   every other chunk of its partial to the chunk's owner, which sums the NR
//   partials of its chunk in rank order and pushes the sum to the other ranks
//   (a reduce-scatter, then an all-gather, within the row). Every block then
//   moves 2 (NR - 1) / NR of a row, while a row owner moves NR - 1 rows.
// Either way every rank ends with the same bf16 row, so the epilogues produce
// the same bits on every rank.
//
// Peers write into each other's inboxes (peer-mapped device memory). The data
// is its own flag: inbox records start as the bf16 -0.0 pattern (0x8000),
// senders map -0.0 to +0.0, readers spin with cache-bypassing loads until no
// lane holds the sentinel. Where only the per-token FP8 rows are wanted, the
// owner quantizes its sum and pushes FP8 rows and scales instead (half the
// bytes of the second hop) into a separate region whose sentinel is the FP8 -0
// byte 0x80 (and 0x80808080, a negative float, for the scale). Each row keeps
// the inbox slot of its next call, in [0, 3), and the owner of its previous
// call (-1 before the first), advanced identically on every rank because every
// rank issues the same calls. A row-call uses its slot s, then resets the
// records the previous row-call left in this rank's inbox (slot (s + 2) % 3),
// which no peer writes again before this rank's next call of that row. The
// state is bounded, so it never wraps.
//
// The device code is built for gfx950 only; on other targets the kernels trap.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "muse_fused_allreduce.h"

namespace vllm::muse_far {

// Row state: the owner of the previous row-call, plus OWNER_FP8 if its second
// hop was FP8; OWNER_CHUNK for a chunked two-hop call.
constexpr int32_t OWNER_MASK = 15;
constexpr int32_t OWNER_FP8 = 16;
constexpr int32_t OWNER_CHUNK = 8;
static_assert(OWNER_CHUNK >= MAX_NR && OWNER_CHUNK <= OWNER_MASK,
              "owner encoding");
constexpr int TOPK = 8;
constexpr uint32_t SENT2 = 0x80008000u;
constexpr uint32_t SENT8 = 0x80808080u;
constexpr float FP8_MAX = 448.0f;
constexpr int FIN_HID = 4096;
constexpr int FIN_NT = 512;
constexpr int BND_HID = 8192;
constexpr int BND_NT = 1024;

#if defined(__gfx950__)

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef __amdgpu_buffer_rsrc_t rsrc_t;
// Cache policy of peer stores and polls: sc0 sc1 (system scope); the top bit
// makes a load volatile.
constexpr int CPOL_SYS = 17;
constexpr int CPOL_SYS_VOL = static_cast<int>(17u | (1u << 31));

__device__ __forceinline__ rsrc_t make_rsrc(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc((void*)base, 0, 0x7fffffff,
                                           0x00020000);
}
__device__ __forceinline__ void bst_x4(rsrc_t r, uint32_t off, uint4 v) {
  u32x4 t = {v.x, v.y, v.z, v.w};
  __builtin_amdgcn_raw_buffer_store_b128(t, r, off, 0, CPOL_SYS);
}
__device__ __forceinline__ uint4 bld_x4(rsrc_t r, uint32_t off) {
  u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, CPOL_SYS_VOL);
  asm volatile("" ::: "memory");
  return make_uint4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ void wg_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

__device__ __forceinline__ float bf2f(uint32_t b16) {
  return __uint_as_float(b16 << 16);
}
__device__ __forceinline__ uint32_t f2bf(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u)
    return (u >> 16) | 0x40u;  // NaN stays NaN
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ uint32_t no_neg_zero(uint32_t b16) {
  return b16 == 0x8000u ? 0u : b16;
}
__device__ __forceinline__ uint4 no_neg_zero4(uint4 v) {
  uint32_t w[4] = {v.x, v.y, v.z, v.w};
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    w[i] = no_neg_zero(w[i] & 0xffffu) | (no_neg_zero(w[i] >> 16) << 16);
  return make_uint4(w[0], w[1], w[2], w[3]);
}
__device__ __forceinline__ bool has_sentinel(uint4 v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  bool hit = false;
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    hit |= ((w[i] & 0xffffu) == 0x8000u) | ((w[i] >> 16) == 0x8000u);
  return hit;
}
__device__ __forceinline__ bool has_sentinel8(uint32_t w) {
  const uint32_t x = w ^ SENT8;  // a zero byte where w holds 0x80
  return ((x - 0x01010101u) & ~x & 0x80808080u) != 0;
}
__device__ __forceinline__ void bst_x2(rsrc_t r, uint32_t off, uint2 v) {
  __builtin_amdgcn_raw_buffer_store_b64(
      (__attribute__((ext_vector_type(2))) unsigned int){v.x, v.y}, r, off, 0,
      CPOL_SYS);
}
__device__ __forceinline__ uint2 poll_x2(rsrc_t r, uint32_t off) {
  for (;;) {
    const auto v =
        __builtin_amdgcn_raw_buffer_load_b64(r, off, 0, CPOL_SYS_VOL);
    asm volatile("" ::: "memory");
    if (!has_sentinel8(v[0]) && !has_sentinel8(v[1]))
      return make_uint2(v[0], v[1]);
  }
}
__device__ __forceinline__ void bst_x1(rsrc_t r, uint32_t off, uint32_t v) {
  __builtin_amdgcn_raw_buffer_store_b32(v, r, off, 0, CPOL_SYS);
}
__device__ __forceinline__ uint32_t poll_x1(rsrc_t r, uint32_t off) {
  for (;;) {
    const uint32_t v =
        __builtin_amdgcn_raw_buffer_load_b32(r, off, 0, CPOL_SYS_VOL);
    asm volatile("" ::: "memory");
    if (v != SENT8) return v;
  }
}
__device__ __forceinline__ uint4 poll_x4(rsrc_t r, uint32_t off) {
  uint4 t = bld_x4(r, off);
  while (has_sentinel(t)) t = bld_x4(r, off);
  return t;
}
__device__ __forceinline__ void unpack8(uint4 v, float* f) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = bf2f(w[i] & 0xffffu);
    f[2 * i + 1] = bf2f(w[i] >> 16);
  }
}
__device__ __forceinline__ void add_pack(uint4 v, float* acc, bool first) {
  float f[8];
  unpack8(v, f);
  #pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = first ? f[i] : acc[i] + f[i];
}
__device__ __forceinline__ uint4 pack_acc(const float* acc) {
  uint32_t w[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    w[i] = no_neg_zero(f2bf(acc[2 * i])) |
           (no_neg_zero(f2bf(acc[2 * i + 1])) << 16);
  return make_uint4(w[0], w[1], w[2], w[3]);
}

template <int CTRL>
__device__ __forceinline__ float dpp(float v) {
  return __builtin_bit_cast(
      float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL,
                                         0xf, 0xf, false));
}
__device__ __forceinline__ float lane(float v, int l) {
  return __builtin_bit_cast(
      float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), l));
}
// Wave-wide sum / max without LDS: quad_perm [1,0,3,2], [2,3,0,1], row_ror 4, 8
// leave every lane of a 16-lane row with the row's result; the four rows are
// combined from lanes 0, 16, 32, 48.
__device__ __forceinline__ float wave_sum(float v) {
  v += dpp<0xb1>(v);
  v += dpp<0x4e>(v);
  v += dpp<0x124>(v);
  v += dpp<0x128>(v);
  return (lane(v, 0) + lane(v, 16)) + (lane(v, 32) + lane(v, 48));
}
__device__ __forceinline__ float wave_max(float v) {
  v = fmaxf(v, dpp<0xb1>(v));
  v = fmaxf(v, dpp<0x4e>(v));
  v = fmaxf(v, dpp<0x124>(v));
  v = fmaxf(v, dpp<0x128>(v));
  return fmaxf(fmaxf(lane(v, 0), lane(v, 16)), fmaxf(lane(v, 32), lane(v, 48)));
}

// Block-wide sums of N values per thread; `red` holds N * NT / 64 floats. LB:
// the barrier waits for LDS only
// (__syncthreads also waits for the block's outstanding global stores).
template <int NT, int N, bool LB = false>
__device__ __forceinline__ void block_sums(float (&v)[N], float* red) {
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    v[j] = wave_sum(v[j]);
    if ((threadIdx.x & 63) == 0) red[j * (NT / 64) + (threadIdx.x >> 6)] = v[j];
  }
  if constexpr (LB)
    wg_barrier();
  else
    __syncthreads();
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    float t = 0.f;
  #pragma unroll
    for (int q = 0; q < NT / 64; ++q) t += red[j * (NT / 64) + q];
    v[j] = t;
  }
}
// Block-wide maxima of N values per thread; `red` holds N * NT / 64 floats.
template <int NT, int N, bool LB = false>
__device__ __forceinline__ void block_maxes(float (&v)[N], float* red) {
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    v[j] = wave_max(v[j]);
    if ((threadIdx.x & 63) == 0) red[j * (NT / 64) + (threadIdx.x >> 6)] = v[j];
  }
  if constexpr (LB)
    wg_barrier();
  else
    __syncthreads();
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    float t = 0.f;
  #pragma unroll
    for (int q = 0; q < NT / 64; ++q) t = fmaxf(t, red[j * (NT / 64) + q]);
    v[j] = t;
  }
}

// Per-token FP8 (e4m3fn) of bf16 rows held as PPT 8-wide packs per thread: q =
// x * (1 / s), s = max(amax / 448, 1e-12).
template <int PPT>
__device__ __forceinline__ float pack_amax(const uint4 (&v)[PPT]) {
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < PPT; ++i) {
    float x[8];
    unpack8(v[i], x);
  #pragma unroll
    for (int k = 0; k < 8; ++k) amax = fmaxf(amax, fabsf(x[k]));
  }
  return amax;
}
// Eight bf16 values to FP8 e4m3fn codes (FP8 -0, the byte 0x80, stored as +0).
__device__ __forceinline__ uint2 quant_pack(uint4 v, float inv) {
  float x[8];
  unpack8(v, x);
  uint32_t w[2];
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    float q[4];
  #pragma unroll
    for (int k = 0; k < 4; ++k)
      q[k] = fminf(fmaxf(x[4 * h + k] * inv, -FP8_MAX), FP8_MAX);
    int p = __builtin_amdgcn_cvt_pk_fp8_f32(q[0], q[1], 0, false);
    p = __builtin_amdgcn_cvt_pk_fp8_f32(q[2], q[3], p, true);
    uint32_t u = (uint32_t)p;
  #pragma unroll
    for (int b = 0; b < 4; ++b)
      if (((u >> (8 * b)) & 0xffu) == 0x80u) u &= ~(0xffu << (8 * b));
    w[h] = u;
  }
  return make_uint2(w[0], w[1]);
}
__device__ __forceinline__ float fp8_scale(float amax) {
  return fmaxf(amax / FP8_MAX, 1e-12f);
}
template <int NT, int PPT>
__device__ __forceinline__ void quant_store_scaled(const uint4 (&v)[PPT],
                                                   float amax, uint2* qrow,
                                                   float* qs, int r) {
  const float s = fp8_scale(amax), inv = 1.0f / s;
  #pragma unroll
  for (int i = 0; i < PPT; ++i)
    qrow[threadIdx.x + i * NT] = quant_pack(v[i], inv);
  if (threadIdx.x == 0) qs[r] = s;
}
template <int NT, int PPT, bool LB = false>
__device__ __forceinline__ void quant_store(const uint4 (&v)[PPT], uint2* qrow,
                                            float* qs, int r, float* red) {
  float amax[1] = {pack_amax<PPT>(v)};
  block_maxes<NT, 1, LB>(amax, red);
  quant_store_scaled<NT, PPT>(v, amax[0], qrow, qs, r);
}

// The all-reduce of row r across the NR ranks; `v` holds this rank's bf16
// partial on entry (PPT packs per thread, pack p = tid + i * NT) and the bf16
// sum on return. Inbox of each rank: stage-1 records [3 slots][NR
// senders][max_rows][row], then stage-2 records [3 slots][max_rows][row], then
// (FP8_HOP) FP8 stage-2 records [3 slots][max_rows][HID codes, 16-byte scale
// record]. `run` returns the row's next state, which `commit` stores once every
// wave of the block has passed a barrier after `run` (every wave reads the
// state on entry).
struct RowState {
  int32_t slot;   // inbox slot of the row's next call, in [0, 3)
  int32_t owner;  // owner of the previous call (| OWNER_FP8), -1: none
};
// CHUNKS: support the chunked two-hop (mode 3); off, its branches stay out of
// the kernel.
template <int NR, int HID, int NT, bool FP8_HOP = false, bool CHUNKS = false>
struct RowAllReduce {
  static_assert(NR >= 2 && NR <= MAX_NR, "unsupported world size");
  static constexpr uint32_t ROW_BYTES = HID * 2;
  static constexpr uint32_t Q_ROW_BYTES = HID + 16;
  static constexpr int PPT = ROW_BYTES / 16 / NT;
  static_assert(PPT * NT * 16 == ROW_BYTES,
                "row must split into 16-byte packs over the block");

  __device__ static uint32_t s1_off(const Comm& c, uint32_t slot, int sender,
                                    int row) {
    return ((slot * NR + sender) * c.max_rows + row) * ROW_BYTES;
  }
  __device__ static uint32_t s2_off(const Comm& c, uint32_t slot, int row) {
    return (3u * NR * c.max_rows + slot * c.max_rows + row) * ROW_BYTES;
  }
  __device__ static uint32_t q2_off(const Comm& c, uint32_t slot, int row) {
    return 3u * (NR + 1) * c.max_rows * ROW_BYTES +
           (slot * c.max_rows + row) * Q_ROW_BYTES;
  }

  // Chunked two-hop: pack p belongs to rank p / CHUNK. Chunks are whole waves,
  // so a wave's peer is uniform.
  static constexpr int CHUNK = NT * PPT / NR;
  static_assert(CHUNK % 64 == 0, "chunks must be whole waves");
  __device__ static int chunk_owner(int p) {
    return __builtin_amdgcn_readfirstlane(p / CHUNK);
  }

  // Rank-order sum of pack `off` of row r over the NR stage-1 records (this
  // rank's own partial is `own`).
  __device__ static uint4 sum_pack(const Comm& c, rsrc_t my, uint32_t slot,
                                   int r, uint32_t off, uint4 own) {
    float sum[8];
  #pragma unroll
    for (int src = 0; src < NR; ++src) {
      const uint4 t =
          src == c.rank ? own : poll_x4(my, s1_off(c, slot, src, r) + off);
      add_pack(t, sum, src == 0);
    }
    return pack_acc(sum);
  }

  __device__ static void commit(const Comm& c, int r, RowState st) {
    if (c.mode != 2 && threadIdx.x == 0) {
      c.state[2 * r] = st.slot;
      c.state[2 * r + 1] = st.owner;
    }
  }

  // Phases of one row, so that a block can carry several rows without waiting
  // on another row's peers: `begin` pushes this rank's partial (no waiting),
  // `middle` sums what the row's peers pushed (one hop, or the owner of a
  // two-hop row, which then pushes the sum), `end` receives the owner's sum
  // (two-hop non-owners) and resets the previous call's records. `fp8`: the
  // owner pushes the FP8 rows of its sum (`push_fp8`) instead of the bf16 sum
  // (two hops only).
  struct Row {
    uint32_t slot;
    int owner, prev_owner;
    bool fp8, prev_fp8;
  };

  __device__ static Row begin(const Comm& c, int r, uint4 (&v)[PPT],
                              bool fp8 = false) {
    Row x{};
    if (c.mode == 2) return x;
  #pragma unroll
    for (int i = 0; i < PPT; ++i) v[i] = no_neg_zero4(v[i]);
    x.slot = (uint32_t)c.state[2 * r] % 3u;
    const int prev = c.state[2 * r + 1];
    x.prev_owner = prev < 0 ? -1 : prev & OWNER_MASK;
    x.prev_fp8 = prev >= 0 && (prev & OWNER_FP8);
    x.fp8 = FP8_HOP && fp8 && c.mode == 0;
    x.owner = c.mode == 1             ? c.rank
              : CHUNKS && c.mode == 3 ? OWNER_CHUNK
                                      : (r * NR) / c.rows;
    const uint32_t base = s1_off(c, x.slot, c.rank, r) + threadIdx.x * 16;
    if (CHUNKS && c.mode == 3) {
  #pragma unroll
      for (int i = 0; i < PPT; ++i) {
        const int o = chunk_owner(threadIdx.x + i * NT);
        if (o != c.rank)
          bst_x4(make_rsrc(c.inbox[o]), base + i * NT * 16, v[i]);
      }
    } else if (c.mode == 1) {
  #pragma unroll
      for (int k = 1; k < NR; ++k) {
        const rsrc_t dst = make_rsrc(c.inbox[(c.rank + k) % NR]);
  #pragma unroll
        for (int i = 0; i < PPT; ++i) bst_x4(dst, base + i * NT * 16, v[i]);
      }
    } else if (x.owner != c.rank) {
      const rsrc_t dst = make_rsrc(c.inbox[x.owner]);
  #pragma unroll
      for (int i = 0; i < PPT; ++i) bst_x4(dst, base + i * NT * 16, v[i]);
    }
    return x;
  }

  __device__ static void middle(const Comm& c, int r, const Row& x,
                                uint4 (&v)[PPT]) {
    if (c.mode == 2) return;
    const rsrc_t my = make_rsrc(c.inbox[c.rank]);
    if (CHUNKS && c.mode == 3) {
  #pragma unroll
      for (int i = 0; i < PPT; ++i) {
        const int p = threadIdx.x + i * NT;
        if (chunk_owner(p) != c.rank) continue;
        v[i] = sum_pack(c, my, x.slot, r, p * 16, v[i]);
  #pragma unroll
        for (int k = 1; k < NR; ++k)
          bst_x4(make_rsrc(c.inbox[(c.rank + k) % NR]),
                 s2_off(c, x.slot, r) + p * 16, v[i]);
      }
      return;
    }
    if (x.owner != c.rank) return;
  #pragma unroll
    for (int i = 0; i < PPT; ++i) {
      const uint32_t off = (threadIdx.x + i * NT) * 16;
      v[i] = sum_pack(c, my, x.slot, r, off, v[i]);
      if (c.mode == 0 && !x.fp8) {
  #pragma unroll
        for (int k = 1; k < NR; ++k)
          bst_x4(make_rsrc(c.inbox[(c.rank + k) % NR]),
                 s2_off(c, x.slot, r) + off, v[i]);
      }
    }
  }

  __device__ static RowState end(const Comm& c, int r, const Row& x,
                                 uint4 (&v)[PPT]) {
    if (c.mode == 2) return {0, 0};
    const rsrc_t my = make_rsrc(c.inbox[c.rank]);
    const uint32_t tbase = threadIdx.x * 16;
    if (CHUNKS && c.mode == 3) {
  #pragma unroll
      for (int i = 0; i < PPT; ++i) {
        const int p = threadIdx.x + i * NT;
        if (chunk_owner(p) != c.rank)
          v[i] = poll_x4(my, s2_off(c, x.slot, r) + p * 16);
      }
    } else if (x.owner != c.rank && !x.fp8) {
  #pragma unroll
      for (int i = 0; i < PPT; ++i)
        v[i] = poll_x4(my, s2_off(c, x.slot, r) + tbase + i * NT * 16);
    }
    // reset what the previous call of this row left in this rank's inbox
    const uint32_t prev = (x.slot + 2u) % 3u;
    const uint4 sent = make_uint4(SENT2, SENT2, SENT2, SENT2);
    if (CHUNKS && x.prev_owner == OWNER_CHUNK) {
  #pragma unroll
      for (int i = 0; i < PPT; ++i) {
        const int p = threadIdx.x + i * NT;
        if (chunk_owner(p) == c.rank) {
  #pragma unroll
          for (int k = 1; k < NR; ++k)
            bst_x4(my, s1_off(c, prev, (c.rank + k) % NR, r) + p * 16, sent);
        } else {
          bst_x4(my, s2_off(c, prev, r) + p * 16, sent);
        }
      }
    } else if (x.prev_owner == c.rank) {
  #pragma unroll
      for (int k = 1; k < NR; ++k) {
        const uint32_t cbase = s1_off(c, prev, (c.rank + k) % NR, r) + tbase;
  #pragma unroll
        for (int i = 0; i < PPT; ++i) bst_x4(my, cbase + i * NT * 16, sent);
      }
    } else if (x.prev_owner >= 0 && !x.prev_fp8) {
  #pragma unroll
      for (int i = 0; i < PPT; ++i)
        bst_x4(my, s2_off(c, prev, r) + tbase + i * NT * 16, sent);
    } else if (x.prev_owner >= 0) {
      const uint32_t qbase = q2_off(c, prev, r);
  #pragma unroll
      for (int i = 0; i < PPT; ++i)
        bst_x2(my, qbase + (threadIdx.x + i * NT) * 8,
               make_uint2(SENT8, SENT8));
      if (threadIdx.x == 0) bst_x1(my, qbase + HID, SENT8);
    }
    return {(int32_t)((x.slot + 1u) % 3u), x.owner | (x.fp8 ? OWNER_FP8 : 0)};
  }

  // FP8 second hop: the owner pushes its quantized sum (`q`, scale `s`) to the
  // other ranks, which wait for it.
  __device__ static void push_fp8(const Comm& c, int r, const Row& x,
                                  const uint2 (&q)[PPT], float s) {
    const uint32_t qbase = q2_off(c, x.slot, r);
  #pragma unroll
    for (int k = 1; k < NR; ++k) {
      const rsrc_t dst = make_rsrc(c.inbox[(c.rank + k) % NR]);
  #pragma unroll
      for (int i = 0; i < PPT; ++i)
        bst_x2(dst, qbase + (threadIdx.x + i * NT) * 8, q[i]);
      if (threadIdx.x == 0) bst_x1(dst, qbase + HID, __float_as_uint(s));
    }
  }
  __device__ static void poll_fp8(const Comm& c, int r, const Row& x,
                                  uint2 (&q)[PPT], float& s) {
    const rsrc_t my = make_rsrc(c.inbox[c.rank]);
    const uint32_t qbase = q2_off(c, x.slot, r);
  #pragma unroll
    for (int i = 0; i < PPT; ++i)
      q[i] = poll_x2(my, qbase + (threadIdx.x + i * NT) * 8);
    if (threadIdx.x == 0) s = __uint_as_float(poll_x1(my, qbase + HID));
  }

  __device__ static RowState run(const Comm& c, int r, uint4 (&v)[PPT]) {
    const Row x = begin(c, r, v);
    middle(c, r, x, v);
    return end(c, r, x, v);
  }
};

// ---------------------------------------------------------------------------------------------------------------
// MoE finalize + all-reduce. Numerics of vLLM's Triton
// _post_expert_finalize_kernel (weight first):
//   out[r] = sum_{local k} tw[r,k] * rsqrt(mean((y[r,k] * w)^2) + eps) *
//   (y[r,k] * w),  w = gamma + gain,
// in fp32 with one bf16 rounding per partial. Routes whose expert is on another
// rank (emap[id] < 0) are never read: their rows were not written.

// READ_ALL: load all 8 route rows (no ids -> load dependency); else only the
// local ones.
template <int NR, bool READ_ALL>
__device__ __forceinline__ void finalize_row(const FinalizeArgs& a) {
  using AR = RowAllReduce<NR, FIN_HID, FIN_NT, true, true>;
  constexpr int PPT = AR::PPT;
  constexpr int PACKS = FIN_HID / 8;
  constexpr int NW = FIN_NT / 64;
  const int r = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
  __shared__ float red[TOPK][NW];
  __shared__ float qred[NW];

  // Route metadata is block-uniform. read_all: load all 8 route rows and skip
  // the other ranks' ones; else load only the local routes, compacted (slot j =
  // j-th local route; slots past the last re-read it, so every load is
  // unconditional and issued back to back).
  const int4 idv[2] = {((const int4*)(a.ids + r * TOPK))[0],
                       ((const int4*)(a.ids + r * TOPK))[1]};
  const float4 twv[2] = {((const float4*)(a.tw + r * TOPK))[0],
                         ((const float4*)(a.tw + r * TOPK))[1]};
  const int ids_[TOPK] = {idv[0].x, idv[0].y, idv[0].z, idv[0].w,
                          idv[1].x, idv[1].y, idv[1].z, idv[1].w};
  const float tws[TOPK] = {twv[0].x, twv[0].y, twv[0].z, twv[0].w,
                           twv[1].x, twv[1].y, twv[1].z, twv[1].w};
  const uint4* ybase = (const uint4*)(a.y + (size_t)r * TOPK * FIN_HID);
  uint4 raw[TOPK][PPT];
  if constexpr (READ_ALL) {
  #pragma unroll
    for (int s = 0; s < TOPK; ++s)
  #pragma unroll
      for (int i = 0; i < PPT; ++i)
        raw[s][i] = ybase[s * PACKS + tid + i * FIN_NT];
  }
  uint32_t lm = 0;
  #pragma unroll
  for (int s = 0; s < TOPK; ++s)
    lm |= (uint32_t)(ids_[s] >= 0 && a.emap[ids_[s]] >= 0) << s;
  const int nl = __builtin_popcount(lm);
  int sidx[TOPK];
  bool valid[TOPK];
  {
    uint32_t rem = lm;
    int last = 0;
  #pragma unroll
    for (int j = 0; j < TOPK; ++j) {
      if constexpr (READ_ALL) {
        sidx[j] = j;
        valid[j] = (lm >> j) & 1u;
      } else {
        const int s = rem ? __builtin_ctz(rem) : last;
        last = s;
        rem &= rem - 1u;
        sidx[j] = s;
        valid[j] = j < nl;
      }
    }
  }
  if constexpr (!READ_ALL) {
  #pragma unroll
    for (int j = 0; j < TOPK; ++j)
  #pragma unroll
      for (int i = 0; i < PPT; ++i)
        raw[j][i] = ybase[sidx[j] * PACKS + tid + i * FIN_NT];
  }
  float w[PPT][8];
  #pragma unroll
  for (int i = 0; i < PPT; ++i) {
    unpack8(((const uint4*)a.gamma)[tid + i * FIN_NT], w[i]);
  #pragma unroll
    for (int j = 0; j < 8; ++j) w[i][j] = w[i][j] + a.gain;
  }
  // Per-thread sums of squares of the 8 slots, then one transposing butterfly
  // over the wave: 3 exchange steps halve the vector (xor 32/16/8), 3 more
  // reduce the scalar (xor 4/2/1); lane l ends with slot (l >> 3) & 7.
  float ss[TOPK];
  #pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    ss[j] = 0.f;
    if (valid[j]) {
  #pragma unroll
      for (int i = 0; i < PPT; ++i) {
        float x[8];
        unpack8(raw[j][i], x);
  #pragma unroll
        for (int k = 0; k < 8; ++k) {
          const float v = x[k] * w[i][k];
          ss[j] = ss[j] + v * v;
        }
      }
    }
  }
  {
    const bool hi5 = lane & 32;
    float s4[4];
  #pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float send = hi5 ? ss[q] : ss[q + 4];
      const float keep = hi5 ? ss[q + 4] : ss[q];
      s4[q] = keep + __shfl_xor(send, 32, 64);
    }
    const bool hi4 = lane & 16;
    float s2[2];
  #pragma unroll
    for (int q = 0; q < 2; ++q) {
      const float send = hi4 ? s4[q] : s4[q + 2];
      const float keep = hi4 ? s4[q + 2] : s4[q];
      s2[q] = keep + __shfl_xor(send, 16, 64);
    }
    const bool hi3 = lane & 8;
    float s1 = (hi3 ? s2[1] : s2[0]) + __shfl_xor(hi3 ? s2[0] : s2[1], 8, 64);
    s1 = s1 + __shfl_xor(s1, 4, 64);
    s1 = s1 + __shfl_xor(s1, 2, 64);
    s1 = s1 + __shfl_xor(s1, 1, 64);
    const int slot_of_lane =
        ((lane >> 5) & 1) * 4 + ((lane >> 4) & 1) * 2 + ((lane >> 3) & 1);
    if ((lane & 7) == 0) red[slot_of_lane][wave] = s1;
  }
  __syncthreads();
  float acc[PPT][8];
  #pragma unroll
  for (int i = 0; i < PPT; ++i)
  #pragma unroll
    for (int k = 0; k < 8; ++k) acc[i][k] = 0.f;
  #pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    if (valid[j]) {
      float twj = 0.f;
  #pragma unroll
      for (int s = 0; s < TOPK; ++s) twj = s == sidx[j] ? tws[s] : twj;
      float total = 0.f;
  #pragma unroll
      for (int q = 0; q < NW; ++q) total = total + red[j][q];
      const float scale = rsqrtf(total * (1.0f / FIN_HID) + a.eps) * twj;
  #pragma unroll
      for (int i = 0; i < PPT; ++i) {
        float x[8];
        unpack8(raw[j][i], x);
  #pragma unroll
        for (int k = 0; k < 8; ++k)
          acc[i][k] = acc[i][k] + (x[k] * w[i][k]) * scale;
      }
    }
  }
  uint4 v[PPT];
  #pragma unroll
  for (int i = 0; i < PPT; ++i) v[i] = pack_acc(acc[i]);

  const typename AR::Row x = AR::begin(a.comm, r, v, a.q != nullptr);
  AR::middle(a.comm, r, x, v);
  const RowState next = AR::end(a.comm, r, x, v);
  if (x.fp8) {
    uint2 q[PPT];
    float s = 0.f;
    if (x.owner == a.comm.rank) {
      float amax[1] = {pack_amax<PPT>(v)};
      block_maxes<FIN_NT, 1>(amax, qred);
      s = fp8_scale(amax[0]);
  #pragma unroll
      for (int i = 0; i < PPT; ++i) q[i] = quant_pack(v[i], 1.0f / s);
      AR::push_fp8(a.comm, r, x, q, s);
    } else {
      AR::poll_fp8(a.comm, r, x, q, s);
    }
    uint2* qrow = (uint2*)(a.q + (size_t)r * FIN_HID);
  #pragma unroll
    for (int i = 0; i < PPT; ++i) qrow[tid + i * FIN_NT] = q[i];
    if (tid == 0) a.qs[r] = s;
  }
  wg_barrier();
  AR::commit(a.comm, r, next);
  if (x.fp8) return;

  if (a.out) {
    uint4* outrow = (uint4*)(a.out + (size_t)r * FIN_HID);
  #pragma unroll
    for (int i = 0; i < PPT; ++i) outrow[tid + i * FIN_NT] = v[i];
  }
  if (a.q)
    quant_store<FIN_NT, PPT>(v, (uint2*)(a.q + (size_t)r * FIN_HID), a.qs, r,
                             qred);
}

// ---------------------------------------------------------------------------------------------------------------
// Attention boundary: all-reduce of the o_proj partials, then in fp32
//   y = post_norm(branch);  new_stream = alpha * stream + beta * y  (stored
//   fp32);  z = pre_norm(new_stream)
// stored as bf16 (`out`) and/or per-token FP8 of z rounded to bf16 (`q`, `qs`),
// as _norm_residual_norm_fwd_kernel. A norm with `input_scale` multiplies by
// its weight (plus gain) before the reduction, else after.

// The epilogue of row r, from its all-reduced bf16 packs `v` and fp32 stream
// `st`.
template <int NT, int PPT, class Args, bool LB = false>
__device__ __forceinline__ void boundary_epilogue(
    const Args& a, int r, uint4 (&v)[PPT], float (&st)[PPT][8],
    const float (&al)[PPT][8], const float (&be)[PPT][8],
    const float (&pw)[PPT][8], const float (&qw)[PPT][8]) {
  __shared__ float red[4 * (NT / 64)];
  __shared__ float qred[NT / 64];
  // One reduction for both norms: with b the (input-scaled) branch, y = b * rs1
  // (* post weight), A = alpha * s and B = beta * b (both times the pre weight
  // under pre input scale), the pre-norm input is A + rs1 * B and
  //   sum((A + rs1 * B)^2) = sum(A^2) + 2 * rs1 * sum(A * B) + rs1^2 *
  //   sum(B^2).
  const bool post_in = a.post_input_scale, pre_in = a.pre_input_scale;
  float x[PPT][8];
  float sums[4] = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll
  for (int i = 0; i < PPT; ++i) {
    unpack8(v[i], x[i]);
  #pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (post_in) x[i][k] = x[i][k] * pw[i][k];
      const float g = pre_in ? qw[i][k] : 1.f;
      const float A = al[i][k] * st[i][k] * g,
                  B = be[i][k] * (post_in ? x[i][k] : x[i][k] * pw[i][k]) * g;
      sums[0] = sums[0] + x[i][k] * x[i][k];
      sums[1] = sums[1] + A * A;
      sums[2] = sums[2] + A * B;
      sums[3] = sums[3] + B * B;
    }
  }
  block_sums<NT, 4, LB>(sums, red);
  const float rs1 = rsqrtf(sums[0] * (1.0f / BND_HID) + a.post_eps);
  const float ss2 = sums[1] + 2.f * rs1 * sums[2] + rs1 * rs1 * sums[3];
  const float rs2 = rsqrtf(fmaxf(ss2, 0.f) * (1.0f / BND_HID) + a.pre_eps);
  // gated residual (stored fp32) and pre-norm
  float4* nrow = (float4*)(a.new_stream + (size_t)r * BND_HID);
  #pragma unroll
  for (int i = 0; i < PPT; ++i) {
  #pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float y = post_in ? x[i][k] * rs1 : x[i][k] * rs1 * pw[i][k];
      const float ns = al[i][k] * st[i][k] + be[i][k] * y;
      st[i][k] = ns;
      x[i][k] = pre_in ? ns * qw[i][k] * rs2 : ns * rs2 * qw[i][k];
    }
    const int p = threadIdx.x + i * NT;
    nrow[2 * p] = make_float4(st[i][0], st[i][1], st[i][2], st[i][3]);
    nrow[2 * p + 1] = make_float4(st[i][4], st[i][5], st[i][6], st[i][7]);
    v[i] = pack_acc(x[i]);
  }
  if (a.out) {
    uint4* orow = (uint4*)(a.out + (size_t)r * BND_HID);
  #pragma unroll
    for (int i = 0; i < PPT; ++i) orow[threadIdx.x + i * NT] = v[i];
  }
  if (a.q)
    quant_store<NT, PPT, LB>(v, (uint2*)(a.q + (size_t)r * BND_HID), a.qs, r,
                             qred);
}

template <int PPT>
__device__ __forceinline__ void load_stream(const BoundaryArgs& a, int r,
                                            int nt, float (&st)[PPT][8]) {
  const float4* srow = (const float4*)(a.stream + (size_t)r * BND_HID);
  #pragma unroll
  for (int i = 0; i < PPT; ++i) {
    const int p = threadIdx.x + i * nt;
    const float4 s0 = srow[2 * p], s1 = srow[2 * p + 1];
    const float s8[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
  #pragma unroll
    for (int k = 0; k < 8; ++k) st[i][k] = s8[k];
  }
}

// Row blockIdx.x. A block waits only on the same row's blocks on the peers,
// and blocks are dispatched in order, so a grid of any size completes. The
// caller keeps it within one block per CU so that every row's exchange is in
// flight at once, instead of later rows waiting for earlier ones to leave a CU.
template <int NR>
__device__ __forceinline__ void boundary_row(const BoundaryArgs& a) {
  constexpr int NT = BND_NT;
  using AR = RowAllReduce<NR, BND_HID, NT>;
  constexpr int PPT = AR::PPT;
  const int r = blockIdx.x;
  uint4 v[PPT];
  #pragma unroll
  for (int i = 0; i < PPT; ++i)
    v[i] =
        ((const uint4*)(a.partial + (size_t)r * BND_HID))[threadIdx.x + i * NT];
  // The partial is pushed first, so its loads are issued before the epilogue's
  // (a push then waits for them alone); the machine scheduler would otherwise
  // be free to issue them last.
  __builtin_amdgcn_sched_barrier(0);
  // Everything the epilogue reads is loaded before the all-reduce waits.
  float st[PPT][8], al[PPT][8], be[PPT][8], pw[PPT][8], qw[PPT][8];
  load_stream<PPT>(a, r, NT, st);
  #pragma unroll
  for (int i = 0; i < PPT; ++i) {
    const int p = threadIdx.x + i * NT;
    const float4 a0 = ((const float4*)a.alpha)[2 * p],
                 a1 = ((const float4*)a.alpha)[2 * p + 1];
    const float4 b0 = ((const float4*)a.beta)[2 * p],
                 b1 = ((const float4*)a.beta)[2 * p + 1];
    const float a8[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float b8[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
    unpack8(((const uint4*)a.post_w)[p], pw[i]);
    unpack8(((const uint4*)a.pre_w)[p], qw[i]);
  #pragma unroll
    for (int k = 0; k < 8; ++k) {
      al[i][k] = a8[k];
      be[i][k] = b8[k];
      pw[i][k] += a.post_gain;
      qw[i][k] += a.pre_gain;
    }
  }
  const RowState next = AR::run(a.comm, r, v);
  boundary_epilogue<NT, PPT>(a, r, v, st, al, be, pw, qw);
  AR::commit(a.comm, r, next);  // after the epilogue's barriers: every wave has
                                // read the row state
}
// ---------------------------------------------------------------------------------------------------------------
// Multi-block rows, for small M at large NR, where a lone block per row would
// carry all of its row's traffic to NR - 1 peers. Each row gets three blocks,
// launched role-major (block b: role b / M, row b % M):
//   sender: pushes chunk j of this rank's partial row (the j-th NR-th of its
//   packs) to rank j's stage-1 record
//           [slot][rank][r], for every j but this rank;
//   owner:  sums chunk `rank` over the NR partials in rank order (fp32, one
//   bf16 rounding) and pushes the bf16 sum to
//           every rank's stage-2 record [slot][r], its own included;
//   row:    reads the whole row from its stage-2 record and runs the epilogue.
// Every rank ends with the same bits, which are those of the one-block-per-row
// kernels. Senders and owners never wait for a block of their own launch, and
// blocks are dispatched in order, so they are on their way before any row block
// spins. Row blocks issue no store before their last block reduction (a barrier
// waits for the wave's outstanding stores). Each block keeps its own row state
// (the three blocks of a row advance together): the slot s of the row's next
// call, in [0, 3), plus MB_HAS_PREV after the first call. The row block resets,
// at its end, the records the row's previous call left in this rank's inbox
// (slot (s + 2) % 3).

enum MbRole { MB_SENDER = 0, MB_OWNER = 1, MB_ROW = 2 };
constexpr int32_t MB_HAS_PREV = 3;  // added to the slot after the first call

template <int NR, int HID, int NT>
struct MbRow {
  static constexpr uint32_t ROW_BYTES = HID * 2;
  static_assert(ROW_BYTES == NT * 16, "one 16-byte pack per thread");
  static constexpr int CHUNK = NT / NR;  // packs per chunk
  static_assert(CHUNK % 64 == 0, "chunks must be whole waves");
  __device__ static uint32_t s1_off(const Comm& c, uint32_t slot, int sender,
                                    int row) {
    return ((slot * NR + sender) * c.max_rows + row) * ROW_BYTES;
  }
  __device__ static uint32_t s2_off(const Comm& c, uint32_t slot, int row) {
    return (3u * NR * c.max_rows + slot * c.max_rows + row) * ROW_BYTES;
  }
  __device__ static int owner(int p) {
    return __builtin_amdgcn_readfirstlane(p / CHUNK);
  }

  // This block's role and row.
  struct Who {
    MbRole role;
    int r;
  };
  __device__ static Who who(const Comm& c) {
    return {(MbRole)(blockIdx.x / c.rows), (int)(blockIdx.x % c.rows)};
  }
  // The block's row state: the slot of this call and whether the row had a
  // previous call. Every wave has its copy before the barrier, so `commit` at
  // the end needs none.
  struct State {
    uint32_t slot;
    bool has_prev;
  };
  __device__ static State state(const Comm& c, const Who& w) {
    const int32_t n = c.state[w.role * (int)c.max_rows + w.r];
    // The load has landed in every wave before any wave can pass the barrier
    // and commit (s_barrier alone does not wait for it).
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
    return {(uint32_t)n % 3u, n >= MB_HAS_PREV};
  }
  __device__ static void commit(const Comm& c, const Who& w, const State& s) {
    if (threadIdx.x == 0)
      c.state[w.role * (int)c.max_rows + w.r] =
          (int32_t)((s.slot + 1u) % 3u) + MB_HAS_PREV;
  }
  // Sender: push this thread's pack `v` (sanitized partial) to its chunk's
  // owner.
  __device__ static void send(const Comm& c, const Who& w, uint32_t slot,
                              uint4 v) {
    const int p = threadIdx.x, o = owner(p);
    if (o != c.rank)
      bst_x4(make_rsrc(c.inbox[o]), s1_off(c, slot, c.rank, w.r) + p * 16, v);
  }
  // Owner: rank-order sum of this rank's chunk and its push to every rank.
  // Every peer's load is in flight at once (cache-bypassing but not volatile: a
  // volatile load is followed by a wait for it), reloaded until all arrived.
  __device__ static void own(const Comm& c, const Who& w, uint32_t slot,
                             uint4 v) {
    const int p = threadIdx.x;
    if (owner(p) != c.rank) return;
    const rsrc_t my = make_rsrc(c.inbox[c.rank]);
    uint4 t[NR];
    for (;;) {
      asm volatile("" ::: "memory");
  #pragma unroll
      for (int src = 0; src < NR; ++src) {
        if (src == c.rank) {
          t[src] = v;
        } else {
          const u32x4 x = __builtin_amdgcn_raw_buffer_load_b128(
              my, s1_off(c, slot, src, w.r) + p * 16, 0, CPOL_SYS);
          t[src] = make_uint4(x[0], x[1], x[2], x[3]);
        }
      }
      bool miss = false;
  #pragma unroll
      for (int src = 0; src < NR; ++src)
        miss |= src != c.rank && has_sentinel(t[src]);
      if (!miss) break;
    }
    float sum[8];
  #pragma unroll
    for (int src = 0; src < NR; ++src) add_pack(t[src], sum, src == 0);
    const uint4 s = pack_acc(sum);
    // Start at a different peer per wave and row, so that the pushes in flight
    // use every link.
    const int rot = __builtin_amdgcn_readfirstlane((p >> 6) + w.r);
  #pragma unroll
    for (int k = 0; k < NR; ++k)
      bst_x4(make_rsrc(c.inbox[(c.rank + k + rot) % NR]),
             s2_off(c, slot, w.r) + p * 16, s);
  }
  // Row block: this thread's pack of the all-reduced row r.
  __device__ static uint4 receive(const Comm& c, uint32_t slot, int r) {
    return poll_x4(make_rsrc(c.inbox[c.rank]),
                   s2_off(c, slot, r) + threadIdx.x * 16);
  }
  // Row block, at its end: reset the records of the previous call of row r
  // (`prev` slot) in this rank's inbox.
  __device__ static void reset(const Comm& c, uint32_t prev, int r) {
    const int p = threadIdx.x;
    const rsrc_t my = make_rsrc(c.inbox[c.rank]);
    const uint4 sent = make_uint4(SENT2, SENT2, SENT2, SENT2);
    if (owner(p) == c.rank) {
  #pragma unroll
      for (int k = 1; k < NR; ++k)
        bst_x4(my, s1_off(c, prev, (c.rank + k) % NR, r) + p * 16, sent);
    }
    bst_x4(my, s2_off(c, prev, r) + p * 16, sent);
  }
};

// Boundary over multi-block rows (BoundaryArgs fields, Comm).

// The boundary epilogue of a multi-block row (one pack per thread):
// boundary_epilogue's math and rounding points, with the input-scale choices as
// template arguments, packed fp32 math (no MFMA in this kernel), hardware bf16
// rounding, and every store after the last block reduction.
typedef __bf16 bf16x2_t __attribute__((ext_vector_type(2)));
typedef float f32x2_t __attribute__((ext_vector_type(2)));
__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) {
  const f32x2_t f = {lo, hi};
  return __builtin_bit_cast(uint32_t, __builtin_convertvector(f, bf16x2_t));
}
// Eight bf16 values to FP8 e4m3fn codes (scaled by `inv`, saturated).
__device__ __forceinline__ uint2 quant_pack_out(uint4 v, float inv) {
  float x[8];
  unpack8(v, x);
  uint32_t w[2];
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    float q[4];
  #pragma unroll
    for (int k = 0; k < 4; ++k)
      q[k] = __builtin_amdgcn_fmed3f(x[4 * h + k] * inv, -FP8_MAX, FP8_MAX);
    int p = __builtin_amdgcn_cvt_pk_fp8_f32(q[0], q[1], 0, false);
    p = __builtin_amdgcn_cvt_pk_fp8_f32(q[2], q[3], p, true);
    w[h] = (uint32_t)p;
  }
  return make_uint2(w[0], w[1]);
}
// The four norm sums of the packs held by this thread (x = the reduced branch):
// {sum x^2, sum A^2, sum A B, sum B^2}.
template <bool POST_IN, bool PRE_IN>
__device__ __forceinline__ void bnd_norm_sums(
    const uint32_t (&w)[4], const f32x2_t (&st)[4], const f32x2_t (&al)[4],
    const f32x2_t (&be)[4], const f32x2_t (&pw)[4], const f32x2_t (&qw)[4],
    f32x2_t (&x)[4], float (&sums)[4]) {
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    x[j] = (f32x2_t){bf2f(w[j] & 0xffffu), bf2f(w[j] >> 16)};
  f32x2_t s01 = {0.f, 0.f},
          s23 = {0.f, 0.f};  // {sum x^2, sum A^2}, {sum A B, sum B^2}
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    if (POST_IN) x[j] = x[j] * pw[j];
    f32x2_t A = al[j] * st[j], B = be[j] * (POST_IN ? x[j] : x[j] * pw[j]);
    if (PRE_IN) {
      A = A * qw[j];
      B = B * qw[j];
    }
  #pragma unroll
    for (int e = 0; e < 2; ++e) {
      const f32x2_t xa = {x[j][e], A[e]}, ab = {A[e], B[e]}, bb = {B[e], B[e]};
      s01 = __builtin_elementwise_fma(xa, xa, s01);
      s23 = __builtin_elementwise_fma(ab, bb, s23);
    }
  }
  sums[0] = s01[0];
  sums[1] = s01[1];
  sums[2] = s23[0];
  sums[3] = s23[1];
}
template <int NT, bool POST_IN, bool PRE_IN>
__device__ __forceinline__ void mb_bnd_epilogue(const BoundaryArgs& a, int r,
                                                uint4 v, const f32x2_t (&st)[4],
                                                const f32x2_t (&al)[4],
                                                const f32x2_t (&be)[4],
                                                const f32x2_t (&pw)[4],
                                                const f32x2_t (&qw)[4]) {
  __shared__ float red[4 * (NT / 64)];
  __shared__ float qred[NT / 64];
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  f32x2_t x[4];
  float sums[4];
  bnd_norm_sums<POST_IN, PRE_IN>(w, st, al, be, pw, qw, x, sums);
  block_sums<NT, 4, true>(sums, red);
  const float rs1 = rsqrtf(sums[0] * (1.0f / BND_HID) + a.post_eps);
  const float ss2 = sums[1] + 2.f * rs1 * sums[2] + rs1 * rs1 * sums[3];
  const float rs2 = rsqrtf(fmaxf(ss2, 0.f) * (1.0f / BND_HID) + a.pre_eps);
  const f32x2_t rs1v = {rs1, rs1}, rs2v = {rs2, rs2};
  f32x2_t ns[4], z[4];
  float zmax = 0.f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const f32x2_t y = POST_IN ? x[j] * rs1v : x[j] * rs1v * pw[j];
    ns[j] = al[j] * st[j] + be[j] * y;
    z[j] = PRE_IN ? ns[j] * qw[j] * rs2v : ns[j] * rs2v * qw[j];
    zmax = fmaxf(zmax, fmaxf(fabsf(z[j][0]), fabsf(z[j][1])));
  }
  const uint4 o =
      make_uint4(bf16x2(z[0][0], z[0][1]), bf16x2(z[1][0], z[1][1]),
                 bf16x2(z[2][0], z[2][1]), bf16x2(z[3][0], z[3][1]));
  float sc = 0.f;
  if (a.q) {
    // max |bf16(z)| = bf16(max |z|): rounding is monotonic.
    float amax[1] = {zmax};
    block_maxes<NT, 1, true>(amax, qred);
    sc = fp8_scale(bf2f(bf16x2(amax[0], 0.f) & 0xffffu));
  }
  const int p = threadIdx.x;
  float4* nrow = (float4*)(a.new_stream + (size_t)r * BND_HID);
  nrow[2 * p] = make_float4(ns[0][0], ns[0][1], ns[1][0], ns[1][1]);
  nrow[2 * p + 1] = make_float4(ns[2][0], ns[2][1], ns[3][0], ns[3][1]);
  if (a.out) ((uint4*)(a.out + (size_t)r * BND_HID))[p] = o;
  if (a.q) {
    ((uint2*)(a.q + (size_t)r * BND_HID))[p] = quant_pack_out(o, 1.0f / sc);
    if (p == 0) a.qs[r] = sc;
  }
}

// Epilogue inputs of pack p of row r, as element pairs (norm weights plus their
// gains).
__device__ __forceinline__ void mb_bnd_inputs(
    const BoundaryArgs& a, int r, int p, f32x2_t (&s2)[4], f32x2_t (&al2)[4],
    f32x2_t (&be2)[4], f32x2_t (&pw2)[4], f32x2_t (&qw2)[4]) {
  {
    const float4* srow = (const float4*)(a.stream + (size_t)r * BND_HID);
    const float4 s0 = srow[2 * p], s1 = srow[2 * p + 1];
    const float4 a0 = ((const float4*)a.alpha)[2 * p],
                 a1 = ((const float4*)a.alpha)[2 * p + 1];
    const float4 b0 = ((const float4*)a.beta)[2 * p],
                 b1 = ((const float4*)a.beta)[2 * p + 1];
    float pw[8], qw[8];
    unpack8(((const uint4*)a.post_w)[p], pw);
    unpack8(((const uint4*)a.pre_w)[p], qw);
    const float s8[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
    const float a8[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float b8[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      s2[j] = (f32x2_t){s8[2 * j], s8[2 * j + 1]};
      al2[j] = (f32x2_t){a8[2 * j], a8[2 * j + 1]};
      be2[j] = (f32x2_t){b8[2 * j], b8[2 * j + 1]};
      pw2[j] = (f32x2_t){pw[2 * j] + a.post_gain, pw[2 * j + 1] + a.post_gain};
      qw2[j] = (f32x2_t){qw[2 * j] + a.pre_gain, qw[2 * j + 1] + a.pre_gain};
    }
  }
}

template <int NR>
__device__ __forceinline__ void mb_boundary(const BoundaryArgs& a) {
  constexpr int NT = BND_NT;
  using R = MbRow<NR, BND_HID, NT>;
  const Comm& c = a.comm;
  const typename R::Who w = R::who(c);
  const typename R::State st = R::state(c, w);
  const uint32_t slot = st.slot;
  const int r = w.r, p = threadIdx.x;
  if (w.role != MB_ROW) {
    const uint4 v =
        no_neg_zero4(((const uint4*)(a.partial + (size_t)r * BND_HID))[p]);
    if (w.role == MB_SENDER)
      R::send(c, w, slot, v);
    else
      R::own(c, w, slot, v);
    R::commit(c, w, st);
    return;
  }
  f32x2_t s2[4], al2[4], be2[4], pw2[4], qw2[4];
  mb_bnd_inputs(a, r, p, s2, al2, be2, pw2, qw2);
  // vmcnt also counts stores: with every load done here, no later wait waits
  // for a store's acknowledgement.
  __builtin_amdgcn_s_waitcnt(
      0x0f70);  // vmcnt(0), as an instruction the compiler's wait tracking sees
  const uint4 sum = R::receive(c, slot, r);
  if (a.post_input_scale) {
    if (a.pre_input_scale)
      mb_bnd_epilogue<NT, true, true>(a, r, sum, s2, al2, be2, pw2, qw2);
    else
      mb_bnd_epilogue<NT, true, false>(a, r, sum, s2, al2, be2, pw2, qw2);
  } else if (a.pre_input_scale) {
    mb_bnd_epilogue<NT, false, true>(a, r, sum, s2, al2, be2, pw2, qw2);
  } else {
    mb_bnd_epilogue<NT, false, false>(a, r, sum, s2, al2, be2, pw2, qw2);
  }
  if (st.has_prev) R::reset(c, (slot + 2u) % 3u, r);
  R::commit(c, w, st);
}

// Finalize over multi-block rows (FinalizeArgs fields, Comm). The sender and
// the owner recompute the row's partial (the route norms need whole route rows;
// only this rank's routes are read).

// This rank's finalize partial of row r (one pack per thread; numerics of
// finalize_row).
template <bool READ_ALL>
__device__ __forceinline__ uint4 fin_partial(const FinalizeArgs& a, int r) {
  constexpr int PACKS = FIN_HID / 8;
  constexpr int NW = FIN_NT / 64;
  static_assert(PACKS == FIN_NT, "one pack per thread");
  const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
  __shared__ float red[TOPK][NW];
  const int4 idv[2] = {((const int4*)(a.ids + r * TOPK))[0],
                       ((const int4*)(a.ids + r * TOPK))[1]};
  const float4 twv[2] = {((const float4*)(a.tw + r * TOPK))[0],
                         ((const float4*)(a.tw + r * TOPK))[1]};
  const int ids_[TOPK] = {idv[0].x, idv[0].y, idv[0].z, idv[0].w,
                          idv[1].x, idv[1].y, idv[1].z, idv[1].w};
  const float tws[TOPK] = {twv[0].x, twv[0].y, twv[0].z, twv[0].w,
                           twv[1].x, twv[1].y, twv[1].z, twv[1].w};
  const uint4* ybase = (const uint4*)(a.y + (size_t)r * TOPK * FIN_HID);
  uint4 raw[TOPK];
  if constexpr (READ_ALL) {
  #pragma unroll
    for (int s = 0; s < TOPK; ++s) raw[s] = ybase[s * PACKS + tid];
  }
  uint32_t lm = 0;
  #pragma unroll
  for (int s = 0; s < TOPK; ++s)
    lm |= (uint32_t)(ids_[s] >= 0 && a.emap[ids_[s]] >= 0) << s;
  const int nl = __builtin_popcount(lm);
  int sidx[TOPK];
  bool valid[TOPK];
  {
    uint32_t rem = lm;
    int last = 0;
  #pragma unroll
    for (int j = 0; j < TOPK; ++j) {
      if constexpr (READ_ALL) {
        sidx[j] = j;
        valid[j] = (lm >> j) & 1u;
      } else {
        const int s = rem ? __builtin_ctz(rem) : last;
        last = s;
        rem &= rem - 1u;
        sidx[j] = s;
        valid[j] = j < nl;
      }
    }
  }
  if constexpr (!READ_ALL) {
  #pragma unroll
    for (int j = 0; j < TOPK; ++j) raw[j] = ybase[sidx[j] * PACKS + tid];
  }
  float w[8];
  unpack8(((const uint4*)a.gamma)[tid], w);
  #pragma unroll
  for (int j = 0; j < 8; ++j) w[j] = w[j] + a.gain;
  float ss[TOPK];
  #pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    ss[j] = 0.f;
    if (valid[j]) {
      float x[8];
      unpack8(raw[j], x);
  #pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float v = x[k] * w[k];
        ss[j] = ss[j] + v * v;
      }
    }
  }
  {
    const bool hi5 = lane & 32;
    float s4[4];
  #pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float send = hi5 ? ss[q] : ss[q + 4];
      const float keep = hi5 ? ss[q + 4] : ss[q];
      s4[q] = keep + __shfl_xor(send, 32, 64);
    }
    const bool hi4 = lane & 16;
    float s2[2];
  #pragma unroll
    for (int q = 0; q < 2; ++q) {
      const float send = hi4 ? s4[q] : s4[q + 2];
      const float keep = hi4 ? s4[q + 2] : s4[q];
      s2[q] = keep + __shfl_xor(send, 16, 64);
    }
    const bool hi3 = lane & 8;
    float s1 = (hi3 ? s2[1] : s2[0]) + __shfl_xor(hi3 ? s2[0] : s2[1], 8, 64);
    s1 = s1 + __shfl_xor(s1, 4, 64);
    s1 = s1 + __shfl_xor(s1, 2, 64);
    s1 = s1 + __shfl_xor(s1, 1, 64);
    const int slot_of_lane =
        ((lane >> 5) & 1) * 4 + ((lane >> 4) & 1) * 2 + ((lane >> 3) & 1);
    if ((lane & 7) == 0) red[slot_of_lane][wave] = s1;
  }
  __syncthreads();
  float acc[8];
  #pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = 0.f;
  #pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    if (valid[j]) {
      float twj = 0.f;
  #pragma unroll
      for (int s = 0; s < TOPK; ++s) twj = s == sidx[j] ? tws[s] : twj;
      float total = 0.f;
  #pragma unroll
      for (int q = 0; q < NW; ++q) total = total + red[j][q];
      const float scale = rsqrtf(total * (1.0f / FIN_HID) + a.eps) * twj;
      float x[8];
      unpack8(raw[j], x);
  #pragma unroll
      for (int k = 0; k < 8; ++k) acc[k] = acc[k] + (x[k] * w[k]) * scale;
    }
  }
  return no_neg_zero4(pack_acc(acc));
}

template <int NR, bool READ_ALL>
__device__ __forceinline__ void mb_finalize(const FinalizeArgs& a) {
  using R = MbRow<NR, FIN_HID, FIN_NT>;
  const Comm& c = a.comm;
  const typename R::Who w = R::who(c);
  const typename R::State st = R::state(c, w);
  const uint32_t slot = st.slot;
  const int r = w.r;
  if (w.role != MB_ROW) {
    const uint4 v = fin_partial<READ_ALL>(a, r);
    if (w.role == MB_SENDER)
      R::send(c, w, slot, v);
    else
      R::own(c, w, slot, v);
    R::commit(c, w, st);
    return;
  }
  const uint4 sum = R::receive(c, slot, r);
  // Every store after the block reduction.
  __shared__ float qred[FIN_NT / 64];
  float sc = 0.f;
  if (a.q) {
    float amax[1] = {pack_amax<1>({sum})};
    block_maxes<FIN_NT, 1, true>(amax, qred);
    sc = fp8_scale(amax[0]);
  }
  if (a.out) ((uint4*)(a.out + (size_t)r * FIN_HID))[threadIdx.x] = sum;
  if (a.q) {
    ((uint2*)(a.q + (size_t)r * FIN_HID))[threadIdx.x] =
        quant_pack_out(sum, 1.0f / sc);
    if (threadIdx.x == 0) a.qs[r] = sc;
  }
  if (st.has_prev) R::reset(c, (slot + 2u) % 3u, r);
  R::commit(c, w, st);
}

#endif  // defined(__gfx950__)

template <int NR, bool READ_ALL>
__global__ void __launch_bounds__(FIN_NT) finalize_kernel(FinalizeArgs a) {
#if defined(__gfx950__)
  finalize_row<NR, READ_ALL>(a);
#else
  __builtin_trap();
#endif
}

template <int NR>
__global__ void __launch_bounds__(BND_NT) boundary_kernel(BoundaryArgs a) {
#if defined(__gfx950__)
  boundary_row<NR>(a);
#else
  __builtin_trap();
#endif
}

template <int NR>
__global__ void __launch_bounds__(BND_NT) mb_boundary_kernel(BoundaryArgs a) {
#if defined(__gfx950__)
  mb_boundary<NR>(a);
#else
  __builtin_trap();
#endif
}

template <int NR, bool READ_ALL>
__global__ void __launch_bounds__(FIN_NT) mb_finalize_kernel(FinalizeArgs a) {
#if defined(__gfx950__)
  mb_finalize<NR, READ_ALL>(a);
#else
  __builtin_trap();
#endif
}

template <int NR>
void launch_finalize(const FinalizeArgs& a, bool read_all,
                     cudaStream_t stream) {
  const int m = a.comm.rows;
  if (a.comm.mode == MULTI_BLOCK) {
    if (read_all)
      mb_finalize_kernel<NR, true><<<3 * m, FIN_NT, 0, stream>>>(a);
    else
      mb_finalize_kernel<NR, false><<<3 * m, FIN_NT, 0, stream>>>(a);
  } else if (read_all) {
    finalize_kernel<NR, true><<<m, FIN_NT, 0, stream>>>(a);
  } else {
    finalize_kernel<NR, false><<<m, FIN_NT, 0, stream>>>(a);
  }
}

template <int NR>
void launch_boundary(const BoundaryArgs& a, cudaStream_t stream) {
  const int m = a.comm.rows;
  if (a.comm.mode == MULTI_BLOCK)
    mb_boundary_kernel<NR><<<3 * m, BND_NT, 0, stream>>>(a);
  else
    boundary_kernel<NR><<<m, BND_NT, 0, stream>>>(a);
}

}  // namespace vllm::muse_far
