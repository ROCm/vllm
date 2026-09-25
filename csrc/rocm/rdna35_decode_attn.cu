// RDNA3.5 (gfx1151) decode attention over vLLM's paged KV cache, GQA-packed.
//
// Reached from Rdna35HipAttentionBackend.  The cache is vLLM's single paged
// tensor with K and V packed in the content dim, exactly as
// TritonAttentionBackend produces it, so no repacking is needed:
//
//   logical  (NB, HKV, BS, 2*D)
//   NHD      (NB, BS, HKV, 2*D)
//   HND      (NB, HKV, BS, 2*D)
//
// Every shape is a compile-time define, so one build serves one shape tuple;
// vllm/v1/attention/ops/rdna35_hip_decode.py drives the compilation.
//
// One workgroup per (kv head, row group, KV segment).  Every query row that
// reads a kv head -- GQA heads times MAXM tokens -- is served by the same
// workgroup, so each KV byte is read from memory once, not once per q head.
//
// A block is TILES tiles of 16 keys.  Each tile belongs to DSPL waves, each
// owning 1/DSPL of the head dim: they split Q@K over d and sum their partial
// scores through LDS, then each runs P@V for its own d.  Every wave keeps its
// own online softmax, so the waves merge only once, at the end.  Both
// products are WMMA 16x16x16 f16 -> f32:
//
//   S^T[key][row] = K[key][:] . Q[row][:]        A = K tile, B = Q^T
//   O^T[d][row]  += V[key][d] . P[row][key]      A = V^T,    B = P^T
#include <hip/hip_runtime.h>

#include <cmath>

#ifndef HEAD_DIM
  #define HEAD_DIM 128
#endif
#ifndef NUM_Q_HEADS
  #define NUM_Q_HEADS 32
#endif
#ifndef NUM_KV_HEADS
  #define NUM_KV_HEADS 8
#endif
#ifndef MAXM
  #define MAXM 1
#endif
#ifndef BS
  #define BS 16
#endif
#ifndef LAYOUT
  #define LAYOUT 1  // 0 = NHD, 1 = HND
#endif
// Most KV segments a (kv head, row group) is split over.  The number active
// is clamp(nblocks / MINB, 1, NSEG), decided at run time from S: the grid is
// fixed when a CUDA graph is captured, S is not.
#ifndef NSEG
  #define NSEG 8
#endif
// Row groups per kv head.  Each workgroup carries ROWS/RG of the kv head's
// rows and reads all of its KV, the RG workgroups sharing it through L2.
#ifndef RG
  #define RG 1
#endif
// Least KV blocks an active segment is given.
#ifndef MINB
  #define MINB 1
#endif
#ifndef MUTATE
  #define MUTATE 0
#endif
// Waves per workgroup.
#ifndef NW
  #define NW 8
#endif
// Waves sharing one key tile, each owning 1/DSPL of the head dim.  Keeps a
// wave's accumulator, K and V slice within budget at large D.
#ifndef DSPL
  #define DSPL (HEAD_DIM >= 256 ? HEAD_DIM / 128 : 1)
#endif

// Measurement only, wrong answers: 1 skips the V loads, 2 thins P@V to one
// WMMA per tile, 4 skips the K loads, 16 thins Q@K to one WMMA per tile, 32
// returns at once, 64 loads each tile and does nothing else, 128 skips the
// DSPL waves' score exchange.
#ifndef ABLATE
  #define ABLATE 0
#endif
// Measurement only: records the 100 MHz realtime counter at phase boundaries
// of thread 0 of workgroups 0..7 into g_ts, read back by the timings() op.
// 1 marks the phases, 3 also waits for the first tile's data to separate
// memory from compute.
#ifndef TIMING
  #define TIMING 0
#endif

#define WAVE 32
#define BLOCK (NW * WAVE)
#define GQA (NUM_Q_HEADS / NUM_KV_HEADS)
#define ROWS (GQA * MAXM)
#define ROWS_W (ROWS / RG)
#define RTILES ((ROWS_W + 15) / 16)
#define ROWPAD (RTILES * 16)
#define TILES (NW / DSPL)
#define KBLK (16 * TILES)
#define DPART (HEAD_DIM / DSPL)
// d per lane of a K or V row: lane l16 owns DPART/16 consecutive d.
#define VD (DPART / 16)
#define NCHUNK (HEAD_DIM / 16)
#define NCP (NCHUNK / DSPL)
#define KV_ROW (2 * HEAD_DIM)
#define PAGE_ELEMS (BS * NUM_KV_HEADS * KV_ROW)
#if LAYOUT == 0
  #define KEY_STRIDE (NUM_KV_HEADS * KV_ROW)
#else
  #define KEY_STRIDE KV_ROW
#endif
#define LOG2E 1.44269504088896340736f
#define QPAD 8
// P reaches P@V as fp16 high plus fp16 low half, ~22 bits rather than 11:
// with one fp16 P the output misses the 1e-3 relative bound wherever it is
// near zero.  When a row tile has at most 8 real rows the low half rides in
// the padding columns of the same WMMA; otherwise it takes a second one.
#define PPACK (ROWS_W <= 8)

// Per-wave K tile: 16 keys of DPART halves.  K is read row-wise, like V, and
// transposed to the lane-per-key WMMA operand through it; the 16-byte pad
// puts the sixteen keys one operand read touches on distinct banks.
#define KT_ROW (DPART * 2 + 16)
#define KT_BYTES (NW * 16 * KT_ROW)
#define QS_BYTES (ROWPAD * (HEAD_DIM + QPAD) * 2)
// Merge buffer: one slot per (live tile, d part) holds the real rows of a row
// tile, each DPART floats plus a pad.
#define PADM 4
#define MROW (DPART + PADM)
#define RV (ROWS_W < 16 ? ROWS_W : 16)
#define MSLOT (RV * MROW)
#define MBUDGET (36 * 1024)
#define SLOTS_FIT(t) ((t) * DSPL * MSLOT * 4 <= MBUDGET)
// Live tiles after the merge's tree rounds: the most that fit the budget.
#define TFIN                          \
  (SLOTS_FIT(TILES)       ? TILES     \
   : SLOTS_FIT(TILES / 2) ? TILES / 2 \
   : SLOTS_FIT(TILES / 4) ? TILES / 4 \
   : SLOTS_FIT(TILES / 8) ? TILES / 8 \
                          : 1)
#define MRG_SLOTS \
  (TFIN == TILES ? TILES : (TFIN > TILES / 2 ? TFIN : TILES / 2))
#define MRG_BYTES (TILES == 1 ? 0 : MRG_SLOTS * DSPL * MSLOT * 4)
#define LOOP_BYTES (QS_BYTES + KT_BYTES)
#define LDS_RAW (LOOP_BYTES > MRG_BYTES ? LOOP_BYTES : MRG_BYTES)

static_assert(NUM_Q_HEADS % NUM_KV_HEADS == 0, "GQA must be integral");
static_assert(GQA % RG == 0, "row groups split whole q heads");
static_assert(BS % 16 == 0, "a 16-key tile must sit inside one page");
static_assert(NW % DSPL == 0, "whole tiles per workgroup");
static_assert(NCHUNK % DSPL == 0, "whole chunks per d part");
static_assert(VD == 4 || VD == 8, "a lane's K and V slice is one b64 or b128");
static_assert(TILES <= WAVE, "one lane per tile loads its page index");
static_assert(DSPL == 1 || RTILES * 1024 <= 16 * KT_ROW,
              "the score exchange must fit a wave's K tile");

#if TIMING
__device__ unsigned long long g_ts[8 * 16];
  #define TS(i)                                                              \
    do {                                                                     \
      if (tid == 0 && blockIdx.x < 8)                                        \
        g_ts[blockIdx.x * 16 + (i)] = __builtin_amdgcn_s_sendmsg_rtnl(0x83); \
    } while (0)
#else
  #define TS(i) \
    do {        \
    } while (0)
#endif

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef unsigned u8v __attribute__((ext_vector_type(8)));
typedef unsigned u4v __attribute__((ext_vector_type(4)));
typedef unsigned u2v __attribute__((ext_vector_type(2)));
#if VD == 8
typedef u4v vrow_t;
#else
typedef u2v vrow_t;
#endif

// One wave's share of a tile: half hi of the wave holds keys 2e + hi, lane
// l16 VD contiguous d of each, for K and for V.  WMMA wants the two halves of
// a wave to hold the same operand; loading different keys into them and
// exchanging halves with one permlanex16 per dword halves both the registers
// and the loads.
struct Tile {
  vrow_t k[8];
  vrow_t v[8];
};

// Element offset of (page, slot) for this kv head.
__device__ __forceinline__ size_t tile_off(int page, unsigned slot, int kvh) {
#if LAYOUT == 0
  return (size_t)page * PAGE_ELEMS +
         ((size_t)slot * NUM_KV_HEADS + kvh) * KV_ROW;
#else
  return (size_t)page * PAGE_ELEMS + ((size_t)kvh * BS + slot) * KV_ROW;
#endif
}

// The other 16-lane half's value for this lane: lane l <-> l ^ 16.
__device__ __forceinline__ unsigned xhalf_u(unsigned v) {
  return (unsigned)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210u,
                                                0xFEDCBA98u, false, false);
}

__device__ __forceinline__ float xhalf(float v) {
  return __builtin_bit_cast(float, xhalf_u(__builtin_bit_cast(unsigned, v)));
}

// Within each 16-lane row: lane i takes lane i & 7's value, or lane i | 8's.
__device__ __forceinline__ unsigned lower8(unsigned v) {
  return (unsigned)__builtin_amdgcn_permlane16((int)v, (int)v, 0x76543210u,
                                               0x76543210u, false, false);
}

__device__ __forceinline__ float upper8f(float v) {
  return __builtin_bit_cast(
      float, __builtin_amdgcn_permlane16(
                 __builtin_bit_cast(int, v), __builtin_bit_cast(int, v),
                 0xFEDCBA98u, 0xFEDCBA98u, false, false));
}

__device__ __forceinline__ float weight_of(float m, float gmax) {
  return (gmax == -INFINITY) ? 0.f : __builtin_amdgcn_exp2f(m - gmax);
}

__device__ __forceinline__ unsigned pack2(float a, float b) {
  return __builtin_bit_cast(unsigned, h2{(_Float16)a, (_Float16)b});
}

// Page index of every 16-key tile of block b: lane t holds tile t's.  Tiles
// past the sequence clamp to its last page, so every address stays inside a
// page vLLM allocated.
__device__ __forceinline__ int block_pages(const int* __restrict__ bt, int b,
                                           int lane, int S) {
  const int key = b * KBLK + (lane % TILES) * 16;
  return bt[(unsigned)min(key, S - 1) / BS];
}

// WMMA's k index is free as long as A and B agree on it -- and each half of
// the wave may order it differently: measured on gfx1151, the lower half
// produces the even output rows from its own copy of A (only its even rows
// are read) and of B, the upper half the odd ones from its own.  So both
// products order k as "this half's keys, then the other half's": the lower
// half puts keys 2i first, the upper half keys 2i+1, which is the order the
// S^T accumulator already hands each half its keys in.  Every operand is
// then (own, other half's) in every lane, with no select on the half.
//
// B = P^T: lane l (row l16) has P for keys 2e + hi in p[e].
__device__ __forceinline__ h16 p_frag(const float* p) {
  u8v r;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    r[i] = pack2(p[2 * i], p[2 * i + 1]);
    r[4 + i] = xhalf_u(r[i]);
  }
  return __builtin_bit_cast(h16, r);
}

// A = V^T for output element j of this lane's slice: row d = base + VD*l16 +
// j, keys in the order above.  Half hi holds keys 2e + hi in v[e]; it packs
// its four dwords and receives the other half's four.
__device__ __forceinline__ h16 v_frag(const vrow_t* v, int j) {
  const int w = j >> 1;
  // v_perm_b32(hi_src, lo_src, sel): half j&1 of each source dword, lo_src
  // into the low half of the result.
  const unsigned sel = (j & 1) ? 0x07060302u : 0x05040100u;
  u8v r;
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    r[a] = __builtin_amdgcn_perm(v[2 * a + 1][w], v[2 * a][w], sel);
    r[4 + a] = xhalf_u(r[a]);
  }
  return __builtin_bit_cast(h16, r);
}

// A wave's real rows of one row tile, rows `stride` floats apart.  Lane l16
// is the row; for element e it owns VD consecutive d at VD*(2e + hi).
__device__ __forceinline__ void store_rows(float* dst, const f8* acc, int l16,
                                           int hi, int stride = MROW) {
  float* row = dst + l16 * stride;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float v[VD];
#pragma unroll
    for (int j = 0; j < VD; ++j) v[j] = acc[j][e];
#pragma unroll
    for (int k = 0; k < VD; k += 4)
      *(f4*)(row + VD * (2 * e + hi) + k) = *(const f4*)(v + k);
  }
}

__device__ __forceinline__ void add_rows(f8* acc, const float* src, int l16,
                                         int hi) {
  const float* row = src + l16 * MROW;
#pragma unroll
  for (int e = 0; e < 8; ++e)
#pragma unroll
    for (int k = 0; k < VD; k += 4) {
      const f4 v = *(const f4*)(row + VD * (2 * e + hi) + k);
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[k + j][e] += v[j];
    }
}

template <typename OutT>
__global__ __launch_bounds__(BLOCK) void decode_attn(
    const _Float16* __restrict__ q, const _Float16* __restrict__ kv,
    const int* __restrict__ bt, float* __restrict__ p_acc,
    float* __restrict__ p_m, float* __restrict__ p_l, int* __restrict__ p_cnt,
    OutT* __restrict__ out, int S, float scale) {
  const int tid = threadIdx.x;
  const int lane = tid & (WAVE - 1);
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE);
  const int l16 = lane & 15;
  const int hi = lane >> 4;
  const int tw = wave / DSPL;  // this wave's tile within the block
  const int dp = wave % DSPL;  // and its part of the head dim

  const int bid = blockIdx.x;
  const int rg = bid % RG;
  const int kvh = (bid / RG) % NUM_KV_HEADS;
  const int seg = bid / (RG * NUM_KV_HEADS);

  __builtin_assume(S > 0);
  const int nblk = (S + KBLK - 1) / KBLK;
  const int nseg = max(1, min(NSEG, nblk / MINB));
  if (seg >= nseg) return;
  TS(0);
  if (ABLATE & 32) {
    if (S == -1) out[0] = (OutT)0;
    return;
  }

  // Every KV address depends on the page table.
  int pages = block_pages(bt, seg, lane, S);
  // Q goes out before the KV.  Load completion is counted in order, so a Q
  // load issued behind the KV tiles would hold its LDS store, and the barrier
  // behind it, until the last KV byte landed.
  constexpr int QITER = (ROWPAD * HEAD_DIM + BLOCK * 8 - 1) / (BLOCK * 8);
  h8 qx[QITER];
#pragma unroll
  for (int it = 0; it < QITER; ++it) {
    const int i = (it * BLOCK + tid) * 8;
    const int r = i / HEAD_DIM, d = i % HEAD_DIM;
    qx[it] = h8{};
    if (i < ROWPAD * HEAD_DIM && r < ROWS_W) {
      const int h = kvh * GQA + rg * (GQA / RG) + r / MAXM;
      const int m = r % MAXM;
      qx[it] = *(const h8*)(q + ((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + d);
    }
  }

  // q_s and the K tiles are dead once the main loop ends; the merge reuses
  // their storage.
  __shared__ __attribute__((aligned(16))) char lds_raw[LDS_RAW];
  auto q_s = reinterpret_cast<_Float16 (*)[HEAD_DIM + QPAD]>(lds_raw);
  float* mrg_s = reinterpret_cast<float*>(lds_raw);
  char* kt_s = lds_raw + QS_BYTES + wave * 16 * KT_ROW;
  // Wave w's partial scores go in wave w's own K tile.  It writes them only
  // after its own Q@K has read that tile, its partners read them between the
  // two barriers of the exchange, and it refills the tile with the next K
  // only after the second barrier.
  float* sx_s = reinterpret_cast<float*>(lds_raw + QS_BYTES);
#define SX_SLOT(w) ((w) * (16 * KT_ROW / 4))
  __shared__ float m_s[NW][ROWPAD];
  __shared__ float l_s[NW][ROWPAD];
  __shared__ float gm_s[ROWPAD];
  __shared__ float gl_s[ROWPAD];

  const _Float16* kvh_base = kv + tile_off(0, 0, kvh);
  // This wave's slice of a K or V row, and its chunks of the head dim.
  const int col = dp * DPART + VD * l16;
  const int c0 = dp * NCP;

  // Issue block b's tile for this wave, and the page lookup of the block
  // after it.  Blocks are issued in order, so `pages` always holds b's.
  auto issue = [&](Tile& t, int b) {
    const int page = __builtin_amdgcn_readlane(pages, tw);
    if (b + nseg < nblk) pages = block_pages(bt, b + nseg, lane, S);
    const _Float16* tp = kvh_base + (size_t)page * PAGE_ELEMS +
                         (size_t)((b * KBLK + tw * 16) % BS) * KEY_STRIDE +
                         (size_t)hi * KEY_STRIDE + col;
    // All of V, then all of K.  Interleaving them row by row measured 1.7 to
    // 3.3 us slower to land at S=128 (32/32/128), same bytes, same addresses.
#pragma unroll
    for (int e = 0; e < 8; ++e)
      t.v[e] = (ABLATE & 1)
                   ? vrow_t{}
                   : *(const vrow_t*)(tp + 2 * e * KEY_STRIDE + HEAD_DIM);
#pragma unroll
    for (int e = 0; e < 8; ++e)
      t.k[e] =
          (ABLATE & 4) ? vrow_t{} : *(const vrow_t*)(tp + 2 * e * KEY_STRIDE);
    // The scheduler must neither sink these to their first use nor hoist
    // work above them: left alone under VGPR pressure it issued a tile's K
    // two loads at a time with a wait after each.
    asm volatile("" ::: "memory");
  };

  Tile ta;
  issue(ta, seg);

#pragma unroll
  for (int it = 0; it < QITER; ++it) {
    const int i = (it * BLOCK + tid) * 8;
    if (i < ROWPAD * HEAD_DIM) *(h8*)&q_s[i / HEAD_DIM][i % HEAD_DIM] = qx[it];
  }

  const float scale2 = scale * LOG2E;
  const int ctx = S - MAXM;

  float m_run[RTILES], l_run[RTILES];
  f8 acc[RTILES][VD];
#pragma unroll
  for (int rt = 0; rt < RTILES; ++rt) {
    m_run[rt] = -INFINITY;
    l_run[rt] = 0.f;
#pragma unroll
    for (int j = 0; j < VD; ++j) acc[rt][j] = f8{};
  }
  // Token index of this lane's row in each row tile, for the causal mask.
  int mrow[RTILES];
#pragma unroll
  for (int rt = 0; rt < RTILES; ++rt) {
    const int r = rt * 16 + l16;
    mrow[rt] = (r < ROWS_W) ? (r % MAXM) : (MAXM - 1);
  }

  __syncthreads();
  TS(1);

  // Stage a tile's K for Q@K: rows in, keys out.  LDS ops of one wave
  // complete in order, so no barrier.
  // Stored as integers and read back as halves, which type-based alias
  // analysis would let pass each other without the asm.
  auto stage_k = [&](const Tile& t) {
#pragma unroll
    for (int e = 0; e < 8; ++e)
      *(vrow_t*)(kt_s + (2 * e + hi) * KT_ROW + VD * l16 * 2) = t.k[e];
    asm volatile("" ::: "memory");
  };

  // Q@K, the softmax update and P@V for one tile whose K is already staged.
  auto process = [&](Tile& t, int b) {
    const int kt = b * KBLK + tw * 16;
#if ABLATE & 64
    unsigned x = 0;
  #pragma unroll
    for (int e = 0; e < 8; ++e)
  #pragma unroll
      for (int w = 0; w < VD / 2; ++w) x ^= t.v[e][w] ^ t.k[e][w];
    acc[0][0][0] += __builtin_bit_cast(float, x & 0x3f800000u);
    return;
#endif
#if TIMING == 3
    if (b == seg) {
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      TS(10);
    }
#endif
    f8 s[RTILES];
#pragma unroll
    for (int rt = 0; rt < RTILES; ++rt) s[rt] = f8{};
#pragma unroll
    for (int c = 0; c < NCP; ++c) {
      const h16 a = *(const h16*)(kt_s + l16 * KT_ROW + c * 32);
#pragma unroll
      for (int rt = 0; rt < RTILES; ++rt) {
        const h16 bq = *(const h16*)&q_s[rt * 16 + l16][(c0 + c) * 16];
        if (!(ABLATE & 16) || c == 0)
          s[rt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bq, s[rt]);
      }
    }
#if DSPL > 1 && !(ABLATE & 128)
    // The scores overwrite this wave's K tile, which the Q@K above read as
    // halves; keep the float stores below those reads.  Then sum the tile's
    // DSPL partials in the same order in every wave, so they all see
    // bit-identical S and agree on the softmax.
    asm volatile("" ::: "memory");
  #pragma unroll
    for (int rt = 0; rt < RTILES; ++rt)
      *(f8*)&sx_s[SX_SLOT(wave) + rt * 256 + lane * 8] = s[rt];
    __syncthreads();
  #pragma unroll
    for (int rt = 0; rt < RTILES; ++rt) {
      f8 sum = *(const f8*)&sx_s[SX_SLOT(tw * DSPL) + rt * 256 + lane * 8];
  #pragma unroll
      for (int q2 = 1; q2 < DSPL; ++q2)
        sum += *(const f8*)&sx_s[SX_SLOT(tw * DSPL + q2) + rt * 256 + lane * 8];
      s[rt] = sum;
    }
    // Partners are done reading before anyone refills its K tile.
    __syncthreads();
#endif
    if (b == seg) TS(2);

    // A tile that reaches past S: its keys are masked below, and their V is
    // zeroed so that a NaN in an unused slot cannot reach P@V as 0 * NaN.
    if (__builtin_expect(kt + 16 > S, 0)) {
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (kt + 2 * e + hi >= S) t.v[e] = vrow_t{};
    }

    // Lane holds row l16, keys kt + 2e + hi.
#pragma unroll
    for (int rt = 0; rt < RTILES; ++rt) {
      float mx = -INFINITY;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int key = kt + 2 * e + hi;
#if MUTATE == 1
        const bool valid = key <= ctx + mrow[rt] + 1 && key < S;
#else
        const bool valid = key <= ctx + mrow[rt];
#endif
        s[rt][e] = valid ? s[rt][e] * scale2 : -INFINITY;
        mx = fmaxf(mx, s[rt][e]);
      }
      mx = fmaxf(mx, xhalf(mx));
      const float mnew = fmaxf(m_run[rt], mx);
      const float alpha =
          (mnew == -INFINITY) ? 1.f : __builtin_amdgcn_exp2f(m_run[rt] - mnew);
      m_run[rt] = mnew;
      float p[8], lo[8], sum = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        p[e] =
            (mnew == -INFINITY) ? 0.f : __builtin_amdgcn_exp2f(s[rt][e] - mnew);
        lo[e] = p[e] - (float)(_Float16)p[e];
        sum += p[e];
      }
      l_run[rt] = l_run[rt] * alpha + sum + xhalf(sum);

#if PPACK
      // Columns 0..7 carry P's high half for rows 0..7 and columns 8..15 its
      // low half for the same rows, so one WMMA does the work of two.  A lane
      // of columns 8..15 therefore accumulates for row l16-8, and scales by
      // that row's alpha.  A mask, not a ?: -- the compiler turned the
      // conditional into a branch that computed pl only in lanes 8..15, and
      // those read it from 0..7.
      const u8v ph = __builtin_bit_cast(u8v, p_frag(p));
      const u8v pl = __builtin_bit_cast(u8v, p_frag(lo));
      const unsigned upper = 0u - (unsigned)(l16 >> 3);
      u8v pk;
  #pragma unroll
      for (int i = 0; i < 8; ++i)
        pk[i] = (ph[i] & ~upper) | (lower8(pl[i]) & upper);
      const h16 bp = __builtin_bit_cast(h16, pk);
      const float ascale = __builtin_bit_cast(
          float, (__builtin_bit_cast(unsigned, alpha) & ~upper) |
                     (lower8(__builtin_bit_cast(unsigned, alpha)) & upper));
#else
      const h16 bp = p_frag(p);
      const h16 bl = p_frag(lo);
      const float ascale = alpha;
#endif
#pragma unroll
      for (int j = 0; j < VD; ++j) {
        acc[rt][j] *= ascale;
        if ((ABLATE & 2) && j) continue;
        const h16 a = v_frag(t.v, j);
        acc[rt][j] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bp, acc[rt][j]);
#if !PPACK
        acc[rt][j] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bl, acc[rt][j]);
#endif
      }
    }
    asm volatile("" ::: "memory");
  };

  // One tile per wave in flight.  Keeping the next tile's loads in flight
  // as well (two register sets, or the next K and V held until the current
  // P@V is done) needs 60-100 VGPRs more than a wave has without spilling,
  // and spilled it measured 59 % against 77 % of roof on 32/4/128.
  for (int b = seg; b < nblk; b += nseg) {
    stage_k(ta);
    process(ta, b);
    if (b + nseg < nblk) issue(ta, b + nseg);
  }
  TS(3);
#if PPACK
  // Fold the low half's columns onto the rows they belong to.
  #pragma unroll
  for (int j = 0; j < VD; ++j)
  #pragma unroll
    for (int e = 0; e < 8; ++e) acc[0][j][e] += upper8f(acc[0][j][e]);
#endif

  // Merge the waves.  Every wave rescales to the workgroup's max; then, if
  // the partials of all tiles do not fit the budget, tree rounds halve the
  // live tiles (upper half stores, lower half adds in registers) until they
  // do; then the live tiles store and the whole workgroup sums them in one
  // pass.  Only real rows are stored -- at M=1 and GQA=4 that is 4 of the 16
  // a row tile carries.  Lane holds row l16 and, for element e of slice j,
  // output d = dp*DPART + VD*(2e + hi) + j.
#pragma unroll
  for (int rt = 0; rt < RTILES; ++rt)
    if (hi == 0) {
      m_s[wave][rt * 16 + l16] = m_run[rt];
      l_s[wave][rt * 16 + l16] = l_run[rt];
    }
  __syncthreads();
  float gsum[RTILES];
#pragma unroll
  for (int rt = 0; rt < RTILES; ++rt) {
    const int r = rt * 16 + l16;
    float mx = -INFINITY;
#pragma unroll
    for (int w = dp; w < NW; w += DSPL) mx = fmaxf(mx, m_s[w][r]);
    float sm = 0.f;
#pragma unroll
    for (int w = dp; w < NW; w += DSPL)
      sm += weight_of(m_s[w][r], mx) * l_s[w][r];
    gsum[rt] = sm;
    if (wave == 0 && hi == 0) {
      gm_s[r] = mx;
      gl_s[r] = sm;
    }
    const float f = weight_of(m_run[rt], mx);
#pragma unroll
    for (int j = 0; j < VD; ++j) acc[rt][j] *= f;
  }

  TS(4);
  const int grp = kvh * RG + rg;
  const size_t pb = ((size_t)grp * NSEG + seg) * ROWPAD;
  const bool publish = nseg > 1;

#pragma unroll
  for (int rt = 0; rt < RTILES; ++rt) {
    // Real rows in this row tile.
    const int rv = min(16, ROWS_W - rt * 16);
    // The previous row tile's readers must be done with the region.
    if (rt) __syncthreads();
#pragma unroll
    for (int st = TILES / 2; st >= TFIN; st >>= 1) {
      if (tw >= st && tw < 2 * st && l16 < rv)
        store_rows(mrg_s + ((tw - st) * DSPL + dp) * MSLOT, acc[rt], l16, hi);
      __syncthreads();
      if (tw < st && l16 < rv)
        add_rows(acc[rt], mrg_s + (tw * DSPL + dp) * MSLOT, l16, hi);
      __syncthreads();
    }
#if TFIN == 1
    // One live tile: its waves write straight from registers.
    if (tw == 0 && l16 < rv) {
      const int r = rt * 16 + l16;
      if (!publish) {
        const int h = kvh * GQA + rg * (GQA / RG) + r / MAXM;
        const int m = r % MAXM;
        const float inv = __builtin_amdgcn_rcpf(gsum[rt]);
        OutT* o = out + ((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + dp * DPART;
  #pragma unroll
        for (int e = 0; e < 8; ++e) {
          _Float16 x[VD];
  #pragma unroll
          for (int j = 0; j < VD; ++j) x[j] = (_Float16)(acc[rt][j][e] * inv);
          *(vrow_t*)(o + VD * (2 * e + hi)) = *(const vrow_t*)x;
        }
      } else {
        store_rows(p_acc + (pb + rt * 16) * HEAD_DIM + dp * DPART, acc[rt], l16,
                   hi, HEAD_DIM);
      }
    }
#else
    if (tw < TFIN && l16 < rv)
      store_rows(mrg_s + (tw * DSPL + dp) * MSLOT, acc[rt], l16, hi);
    __syncthreads();
    // Eight consecutive d of one real row per thread.
    for (int i = tid * 8; i < rv * HEAD_DIM; i += BLOCK * 8) {
      const int rl = i / HEAD_DIM, d = i % HEAD_DIM;
      const int r = rt * 16 + rl;
      const float* src = mrg_s + (d / DPART) * MSLOT + rl * MROW + d % DPART;
      f4 lo4 = *(const f4*)src, hi4 = *(const f4*)(src + 4);
  #pragma unroll
      for (int t = 1; t < TFIN; ++t) {
        lo4 += *(const f4*)(src + t * DSPL * MSLOT);
        hi4 += *(const f4*)(src + t * DSPL * MSLOT + 4);
      }
      if (!publish) {
        const float inv = __builtin_amdgcn_rcpf(gl_s[r]);
        h8 o;
  #pragma unroll
        for (int k = 0; k < 4; ++k) {
          o[k] = (_Float16)(lo4[k] * inv);
          o[4 + k] = (_Float16)(hi4[k] * inv);
        }
        const int h = kvh * GQA + rg * (GQA / RG) + r / MAXM;
        const int m = r % MAXM;
        *(h8*)(out + ((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + d) = o;
      } else {
        float* dst = p_acc + (pb + r) * HEAD_DIM + d;
        *(f4*)dst = lo4;
        *(f4*)(dst + 4) = hi4;
      }
    }
#endif
  }
  TS(5);
  if (!publish) return;

  // Split KV: publish (acc, m, l), and the last segment to arrive merges.
  // Nobody waits: the late arriver is already resident when it does the
  // atomic, and it resets the counter, so a CUDA-graph replay starts clean.
  // Counting arrivals first and letting only the non-last segments fence
  // their partials out measured 0.5-1.6 % worse: their fence then sits on
  // the last one's path instead of ahead of its atomic.
  if (tid < ROWS_W) {
    p_m[pb + tid] = gm_s[tid];
    p_l[pb + tid] = gl_s[tid];
  }
  __threadfence();
  __shared__ int lds_last;
  __syncthreads();
  if (tid == 0) lds_last = (atomicAdd(&p_cnt[grp], 1) == nseg - 1);
  __syncthreads();
  TS(6);
  if (!lds_last) return;
  if (tid == 0) p_cnt[grp] = 0;
  __threadfence();

  // Every load of the merge goes out at once -- each thread's rows' m and l
  // and its slice of every segment's partial, NSEG unrolled -- so the last
  // arriver pays one L2 round trip, not one for the running max, one for the
  // weights and one for the partials.
  for (int i = tid * 4; i < ROWS_W * HEAD_DIM; i += BLOCK * 4) {
    const int r = i / HEAD_DIM, d = i % HEAD_DIM;
    float pm[NSEG], pl[NSEG];
    f4 pa[NSEG];
#pragma unroll
    for (int sg = 0; sg < NSEG; ++sg) {
      const size_t row = ((size_t)grp * NSEG + sg) * ROWPAD + r;
      pm[sg] = sg < nseg ? p_m[row] : -INFINITY;
      pl[sg] = sg < nseg ? p_l[row] : 0.f;
      pa[sg] = sg < nseg ? *(const f4*)(p_acc + row * HEAD_DIM + d) : f4{};
    }
    float gm = -INFINITY;
#pragma unroll
    for (int sg = 0; sg < NSEG; ++sg) gm = fmaxf(gm, pm[sg]);
    f4 num = {};
    float den = 0.f;
#pragma unroll
    for (int sg = 0; sg < NSEG; ++sg) {
      const float a = weight_of(pm[sg], gm);
      den = fmaf(a, pl[sg], den);
      num += a * pa[sg];
    }
    const float inv = __builtin_amdgcn_rcpf(den);
    const int h = kvh * GQA + rg * (GQA / RG) + r / MAXM;
    const int m = r % MAXM;
    OutT* o = out + ((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + d;
#pragma unroll
    for (int k = 0; k < 4; ++k) o[k] = (OutT)(num[k] * inv);
  }
  TS(7);
}

#ifndef RDNA35_TORCH_EXT
// Instantiated for ISA inspection when built without the torch op.
template __global__ void decode_attn<_Float16>(const _Float16*, const _Float16*,
                                               const int*, float*, float*,
                                               float*, int*, _Float16*, int,
                                               float);
#else
  #include <ATen/cuda/CUDAContext.h>
  #include <c10/cuda/CUDAGuard.h>
  #include <torch/extension.h>

// The scratch buffers (acc/m/l/cnt) are caller-allocated on purpose: this runs
// inside a CUDA-graph capture, and an allocation there would break it.
void decode_attn_op(torch::Tensor& q, torch::Tensor& kv_cache,
                    torch::Tensor& block_table, torch::Tensor& out,
                    torch::Tensor& acc, torch::Tensor& m, torch::Tensor& l,
                    torch::Tensor& cnt, int64_t seq_len, double scale) {
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous(),
              "q and out must be contiguous");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32,
              "block table must be int32");
  TORCH_CHECK(q.size(0) == MAXM, "q has ", q.size(0), " tokens, kernel built ",
              "for MAXM=", MAXM);
  TORCH_CHECK(q.size(1) == NUM_Q_HEADS && q.size(2) == HEAD_DIM,
              "q shape does not match the compiled variant");
  // fp16 only: the products are fp16 WMMAs and the loads reinterpret the
  // cache as _Float16.  bf16 would read the same bits as fp16 and return
  // finite nonsense, so refuse it here rather than downstream.
  TORCH_CHECK(q.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
              "kernel is fp16 only, got q=", q.scalar_type(),
              " out=", out.scalar_type());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // Row group fastest, then kv head, then segment: the row groups of one kv
  // head read the same KV and are dispatched side by side to share it in L2.
  dim3 grid(NSEG * NUM_KV_HEADS * RG), block(BLOCK);
  hipLaunchKernelGGL(
      decode_attn<_Float16>, grid, block, 0, stream,
      reinterpret_cast<const _Float16*>(q.data_ptr()),
      reinterpret_cast<const _Float16*>(kv_cache.data_ptr()),
      block_table.data_ptr<int>(), acc.data_ptr<float>(), m.data_ptr<float>(),
      l.data_ptr<float>(), cnt.data_ptr<int>(),
      reinterpret_cast<_Float16*>(out.data_ptr()), (int)seq_len, (float)scale);
}

  #if TIMING
torch::Tensor timings() {
  auto t = torch::zeros({8 * 16}, torch::kInt64);
  hipMemcpyFromSymbol(t.data_ptr(), HIP_SYMBOL(g_ts), sizeof(g_ts));
  return t;
}
  #endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("decode_attn", &decode_attn_op, "RDNA3.5 GQA paged decode attention");
  #if TIMING
  mod.def("timings", &timings);
  #endif
}
#endif
