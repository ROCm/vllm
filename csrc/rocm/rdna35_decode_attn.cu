// RDNA3.5 (gfx1151) decode attention over vLLM's paged KV cache.
//
// Reached from Rdna35HipAttentionBackend.  The cache is vLLM's single paged
// tensor with K and V packed in the content dim, exactly as
// TritonAttentionBackend produces it, so no repacking is needed:
//
//   logical  get_kv_cache_shape() -> (NB, HKV, BS, 2*D)
//   NHD      stride_order (0,2,1,3) -> (NB, BS, HKV, 2*D)   stride_head = 2*D*2
//   B HND      stride_order (0,1,2,3) -> (NB, HKV, BS, 2*D)   stride_head =
//   BS*2*D*2 B
//
// Every shape is a compile-time define, so one build serves one shape tuple;
// vllm/v1/attention/ops/rdna35_hip_decode.py drives the compilation.
//
// Knobs that move the strides:
//   KV_PAD    extra fp16 per (token, head) content row  -> stride_head +=
//   2B*KV_PAD PAGE_PAD  extra fp16 per page (vLLM's page_size_padded /
//   `alignment`) BS        block_size LAYOUT    0 = NHD, 1 = HND
#include <hip/hip_runtime.h>

#include <cmath>

#ifndef HEAD_DIM
  #define HEAD_DIM 256
#endif
#ifndef NUM_Q_HEADS
  #define NUM_Q_HEADS 32
#endif
#ifndef NUM_KV_HEADS
  #define NUM_KV_HEADS 16
#endif
#ifndef MAXM
  #define MAXM 4
#endif
#ifndef NSEG
  #define NSEG 16
#endif
#ifndef KV_PAD
  #define KV_PAD 0
#endif
#ifndef PAGE_PAD
  #define PAGE_PAD 0
#endif
#ifndef BS
  #define BS 16
#endif
#ifndef LAYOUT
  #define LAYOUT 0  // 0 = NHD, 1 = HND
#endif
#ifndef KPW
  #define KPW 4
#endif
#ifndef BLOCK
  #define BLOCK 256
#endif
#ifndef MUTATE
  #define MUTATE 0
#endif
#ifndef ILV
  #define ILV 1
#endif
// How the score butterfly is shared between the two cross-lane pipes, in
// quarters of its elements: 0 puts every element on the LDS pipe
// (ds_bpermute), 4 puts every element on the VALU (v_permlane16/x16), 3 splits
// them three to one.  Neither extreme is best -- see the butterfly itself, and
// OPTIMIZATIONS.md entry 001.
#ifndef BFLY
  #define BFLY 0
#endif
// 1 transposes the grid to (NUM_Q_HEADS, NSEG), making the head the fastest
// axis.  See the comment where seg and h are read.
#ifndef GRIDT
  #define GRIDT 0
#endif
// Measurement only -- ABLATE != 0 COMPUTES THE WRONG ANSWER on purpose.
//
// The KV bytes per tile do not depend on MAXM but the arithmetic over them
// does, so at MAXM > 1 what bounds this kernel is per-query-row VALU work.
// Each bit thins one block of it to a single iteration and leaves the loads,
// the loop and the epilogue intact, so the delta is that block alone and the
// result is an upper bound on what restructuring it could ever buy.
//
//   1  P@V accumulation      2  score butterfly      4  Q@K dot product
//
// check.py MUST fail on any non-zero value; if it passes, the block was
// already dead and the measurement means nothing.
#ifndef ABLATE
  #define ABLATE 0
#endif
// Merge the NWAVE per-wave partials inside the workgroup, in LDS, instead of
// routing them through global memory.  With NSEG==1 that finishes the job and
// no second kernel runs.  With NSEG>1 a head's partials do span workgroups so
// a global reduction is still needed, but over NSEG values instead of
// NSEG*NWAVE: an eighth of the traffic and an eighth of the work.
#ifndef FUSED
  #define FUSED 1
#endif
// How many waves share the query-token dimension.  This is the one knob that
// separates the two decompositions that used to be two kernels:
//
//   MSPLIT == 1      every wave carries all MAXM tokens over its own KV slice
//   MSPLIT == MAXM   every wave carries one token and shares its slice
//
// and any divisor in between.  MPW tokens per wave, NSLICE KV slices per
// workgroup, and the product MPW*NWAVE == MAXM*NSLICE is the partial count
// either way -- which is why the LDS footprint is 32 KiB at MSPLIT=1 and
// 8 KiB at MSPLIT=MAXM for MAXM=4.
//
// Raising it trades KV parallelism for serial depth: the workgroup covers
// NSLICE slices instead of NWAVE, and each wave's epilogue merges NSLICE
// partials instead of NWAVE.  It pays when the fixed cost is large next to the
// KV budget, which is the head-starved regime -- measured 1.14x at
// Hq=8/Hkv=4 and 6% worse at Hq=32/Hkv=16, both at S=128.  Default 1, the
// decomposition every measurement before this change was taken with.
#define WAVE 32
#define NWAVE (BLOCK / WAVE)
#define MPW (MAXM / MSPLIT)      // query tokens carried by one wave
#define NSLICE (NWAVE / MSPLIT)  // KV slices a workgroup covers
// Eight fp16 per lane is one b128, the width this memory system sustains, so
// it is held fixed and the wave is divided instead: LPR lanes cover one row
// and the WAVE/LPR groups each take a different token.  D=512 is the one that
// cannot -- 512/8 would need 64 lanes -- so it keeps two b128 per lane.
// Only D=128 takes it.  Measured across the shape table: 128 improves on all
// 28 of its shapes, 0.87-0.97x.  64 is mixed -- 0.95x at Hq=32/Hkv=8 but
// 1.10x at Hq=16/Hkv=2, where the KV budget is so small the kernel is
// latency-bound and the registers the split costs outweigh the wider load.
// 256 already loads b128 and 512 cannot: 512/8 would want 64 lanes per row.
#ifndef DPL
  #define DPL (HEAD_DIM == 128 ? 8 : HEAD_DIM / WAVE)
#endif
#define LPR (HEAD_DIM / DPL)  // lanes covering one row
#define SUB (WAVE / LPR)      // tokens a wave carries side by side
// Widening the load must not widen the tile.  A lane held KPW rows of DPL
// fp16; now it holds KPWE rows of 8, and KPWE*SUB == KPW keeps both the tile
// at the same tokens and the bytes in flight per lane unchanged -- the same
// bytes arrive in fewer, wider transactions, which is the whole point.
// Without this the tile grew SUB-fold and at S=128 the extra slots simply went
// unused: Hq=16/Hkv=2/D=64 measured 5.79 -> 6.44 us, half the waves idle.
#define KPWE ((KPW / SUB) > 0 ? (KPW / SUB) : 1)
#define KV_ROW (2 * HEAD_DIM + KV_PAD)  // K and V packed, then pad
#define PAGE_ELEMS (BS * NUM_KV_HEADS * KV_ROW + PAGE_PAD)
#define GQA (NUM_Q_HEADS / NUM_KV_HEADS)

// Defined after NWAVE on purpose.  An identifier the preprocessor has not
// seen evaluates to 0 inside #if, silently, so when this sat above the WAVE
// block LDS_FOR() read NWAVE as zero, every branch compared 0 <= 65536, and
// the rule always picked MSPLIT=1.  It was masked because the loader passes
// -DMSPLIT explicitly; D=512 is the first shape where the fallback matters.
//
// Each partial row costs HEAD_DIM floats of acc plus its m and l scalars; the
// two scalar arrays are what took MAXM=8 at MSPLIT=1 to 66048 B, over the
// 64 KiB ceiling, when only lds_acc was counted.
// One float of padding per lane slice.  A lane owns DPL consecutive floats of
// a row, so unpadded its slice starts at lrow*DPL and the bank index
// (lrow*DPL) % 32 takes gcd(DPL,32) distinct values -- two of them at D=512,
// i.e. every lane in the wave piles onto one of two banks and the write
// serialises 16 ways.  Profiled at 43% LDSBankConflict.  DPL is always even,
// so DPL+1 is coprime with 32 and the slice starts spread over all 32 banks.
// The reader pays for it: consecutive d now straddle the pad, which costs a
// 2-way conflict there.  Trading 16-way on the write for 2-way on the read.
#define LDS_SLICE (DPL + 1)
// Passes the epilogue cuts the head dimension into before reducing in LDS.
//
// The partial count is MAXM*NSLICE*SUB and each partial costs HEAD_DIM floats,
// so the accumulator alone can spend the entire 64 KiB: at D=512/MAXM=4 that
// happens at NSLICE=8, and again at DPL=32 where SUB doubles the row count.
// No pad or swizzle recovers those cases -- the budget is gone before the pad
// is counted.
//
// Cutting the reduction into LDSPLIT passes over d divides the *simultaneous*
// footprint instead. Each pass stores only its slice of every partial, so the
// cost is one extra barrier per pass and the softmax arithmetic is untouched.
// 1 is the single-pass layout, byte for byte.
//
// It is an enabler, not an optimisation: on its own it measures neutral. What
// it buys is configurations the ceiling would otherwise forbid.
#ifndef LDSPLIT
  #define LDSPLIT 1
#endif
#define CHUNK_LPR (LPR / LDSPLIT)     // lane-rows whose slice is resident
#define CHUNK_D (HEAD_DIM / LDSPLIT)  // output elements resident per pass
#define LDS_STRIDE (CHUNK_LPR * LDS_SLICE)
#define LDS_OFF(d) ((((d) % CHUNK_D) / DPL) * LDS_SLICE + ((d) % DPL))
#define LDS_ROW_BYTES (LDS_STRIDE * 4 + 8)
#define LDS_FOR(MS) ((MAXM / (MS)) * NWAVE * SUB * LDS_ROW_BYTES)

// Default to the decomposition every measurement before this change was taken
// with, raising it only when the partials would not otherwise fit.
// Finish the cross-workgroup reduction inside this kernel instead of launching
// reduce_segments for it.  Each workgroup publishes its partials, fences, and
// bumps an arrival counter for its head; the one that sees NSEG-1 is the last
// and does the merge, then resets the counter for the next launch.
//
// Nobody waits: the late arriver is already resident when it does the atomic,
// so there is no spin and no way for an unscheduled workgroup to wedge a
// replay.  The reset is what makes a CUDA-graph replay deterministic -- each
// launch increments exactly NSEG times and the last one zeroes it, and kernels
// on one stream are ordered, so launch N+1 cannot race launch N's reset.
//
// Worth more than the launch it saves: L2 is invalidated on kernel launch
// (measured 87% -> 2% hit rate across dispatches), so the second kernel's
// read-back of the partials cannot hit L2 by construction, while this one can.
#ifndef FUSEDRED
  #define FUSEDRED 0
#endif

#ifndef MSPLIT
  #if LDS_FOR(1) <= 65536
    #define MSPLIT 1
  #elif (MAXM % 2 == 0) && (NWAVE % 2 == 0) && (LDS_FOR(2) <= 65536)
    #define MSPLIT 2
  #elif (MAXM % 4 == 0) && (NWAVE % 4 == 0) && (LDS_FOR(4) <= 65536)
    #define MSPLIT 4
  #else
    #define MSPLIT 8
  #endif
#endif
// Partials the global reduction merges, when it runs at all.
#define NPART (FUSED ? NSEG : NSEG * NWAVE)
#define LOG2E 1.44269504088896340736f
// Partials one workgroup produces per query token: one per KV slice per group.
#define NPARTW (NSLICE * SUB)

// DPL fp16 per lane covers HEAD_DIM across the wave: 2, 4, 8 and 16 elements
// for D = 64, 128, 256 and 512, which the compiler issues as b32, b64, b128
// and a pair of b128.  Only 256 loads at the full b128 width the memory system
// likes; the narrow ends are correct first and fast later.
static_assert(LPR >= 1 && LPR <= WAVE && WAVE % LPR == 0,
              "a row must be covered by a whole divisor of the wave");
static_assert(BS % (SUB * KPWE) == 0,
              "a tile of SUB*KPWE tokens must not straddle two blocks");
#if FUSED
static_assert(HEAD_DIM % BLOCK == 0 || BLOCK % HEAD_DIM == 0,
              "the fused epilogue strides the output by BLOCK");
#endif
static_assert(KV_PAD % 8 == 0, "KV_PAD must keep rows 16B aligned");
static_assert(PAGE_PAD % 8 == 0, "PAGE_PAD must keep pages 16B aligned");
// Lets the block table be read once per tile instead of once per token: jb is
// always a multiple of KPW, so jb..jb+KPW-1 cannot straddle two blocks.
static_assert(BS % KPW == 0, "a KPW tile must not straddle two blocks");
static_assert(NWAVE % MSPLIT == 0 && MAXM % MSPLIT == 0,
              "MSPLIT must divide both the wave count and the token count");
// A lane owns DPL consecutive d, so a chunk boundary must fall between lanes
// rather than inside one: only then does a lane's whole slice belong to one
// pass, and the chunk predicate stay a comparison on lrow.
static_assert(LPR % LDSPLIT == 0,
              "LDSPLIT must divide the lanes covering one row");
#if LDSPLIT > 1
static_assert(FUSED, "LDSPLIT only applies to the fused epilogue");
#endif
static_assert(LDS_FOR(MSPLIT) <= 65536,
              "partials must fit the 64 KiB per-workgroup LDS; raise MSPLIT");
#if MSPLIT > 1
static_assert(FUSED,
              "the token-per-wave split only implements the fused epilogue");
#endif

typedef _Float16 h2v __attribute__((ext_vector_type(2)));
typedef float f4v __attribute__((ext_vector_type(4)));
// One lane's slice of a K, V or Q row.  Sized in floats because that is how
// fdot2 consumes it: each float carries the two fp16 of one dot step.
#define FPL (DPL / 2)
typedef float fvec __attribute__((ext_vector_type(FPL)));

__device__ __forceinline__ h2v as_h2(float x) {
  return __builtin_bit_cast(h2v, x);
}

// The hardware reciprocal, not a division.  `num / den` makes the compiler
// emit the full IEEE sequence -- eleven dependent instructions with Newton
// refinement and edge-case fixup -- for a result that is immediately rounded
// to fp16.  v_rcp_f32 is good to ~1 ULP in fp32, far beyond what 11 bits of
// mantissa can hold, and matches the __builtin_amdgcn_exp2f already used for
// the softmax.
__device__ __forceinline__ float fast_div(float num, float den) {
  return num * __builtin_amdgcn_rcpf(den);
}

// byte offset (in fp16 elements) of the K row for (token j, kv head kvh),
// given the physical block already resolved for j.
// j is unsigned so that `% BS` is a mask rather than the five-instruction
// signed sequence (ashr/lshr/add/and/sub) the compiler must emit when the sign
// is unknown.  The token index is non-negative by construction -- it starts at
// gw*KPW and only increases -- but nothing in the types says so.
//
// Cast at the call, do not make the loop induction variable itself unsigned:
// that also removes the sequence but measured 13% WORSE at S=128 (7.59 -> 8.56
// at Hq=8/Hkv=4), by disturbing the scheduling of the load clause.
__device__ __forceinline__ size_t kv_off(int blk, unsigned j, int kvh) {
  const unsigned slot = j % BS;
#if LAYOUT == 0  // NHD: (NB, BS, HKV, 2D)
  return (size_t)blk * PAGE_ELEMS +
         ((size_t)slot * NUM_KV_HEADS + kvh) * KV_ROW;
#else  // HND: (NB, HKV, BS, 2D)
  return (size_t)blk * PAGE_ELEMS + ((size_t)kvh * BS + slot) * KV_ROW;
#endif
}

// 128 VGPR with no spills is the right operating point: asking the compiler for
// more waves per EU drops it to 96 VGPR but spills 120 B, which costs 2.1x at
// S=128 and 2.5x elsewhere.  Occupancy is not what limits this kernel.
// LDS row holding partial p of the token whose wave base is wb.  p splits into
// a KV slice and a wave group; at SUB == 1 it collapses to wb + p*MSPLIT,
// which is what the single-token-per-wave layout used.
// A partial's softmax weight.  An empty segment carries m = -INFINITY, so
// exp2 of -inf is zero and it drops out; when every partial is empty gmax is
// -inf too and (-inf) - (-inf) would be NaN, so that case is selected away.
__device__ __forceinline__ float weight_of(float m, float gmax) {
  return (gmax == -INFINITY) ? 0.f : __builtin_amdgcn_exp2f(m - gmax);
}

__device__ __forceinline__ int partial_row(int wb, int p) {
  return (wb + (p / SUB) * MSPLIT) * SUB + (p % SUB);
}

template <typename OutT>
__global__ __launch_bounds__(BLOCK) void decode_attn(
    const _Float16* __restrict__ q, const _Float16* __restrict__ kv,
    const int* __restrict__ bt, float* __restrict__ p_acc,
    float* __restrict__ p_m, float* __restrict__ p_l, int* __restrict__ p_cnt,
    OutT* __restrict__ out, int S, float scale) {
  // Which grid axis is dispatched fastest.  Workgroups adjacent in launch
  // order are the ones most likely to be co-resident and to hit each other's
  // lines, and every q head of a kv head reads byte-identical addresses, so
  // ordering by head rather than by segment is what makes that sharing
  // available.  0 is the original (NSEG, NUM_Q_HEADS).
#if GRIDT
  const int seg = blockIdx.y;
#else
  const int seg = blockIdx.x;
#endif
  const int tid = threadIdx.x;
  const int lane = tid & (WAVE - 1);
  const int wave = tid / WAVE;
  const int lrow = lane % LPR;  // this lane's slice of the row
  const int grp = lane / LPR;   // which of the SUB tokens it carries
  const int dl = lrow * DPL;
#if GRIDT
  const int h = blockIdx.x;
#else
  const int h = blockIdx.y;
#endif
  const int kvh = h / GQA;
  // Wave w owns tokens mbase, mbase+MSPLIT, ... and KV slice w / MSPLIT.  At
  // MSPLIT == 1 that is mbase = 0 and slice = wave, i.e. every wave takes
  // every token over its own slice.
  const int mbase = wave % MSPLIT;
  const int slice = wave / MSPLIT;

  // 32-bit offsets on purpose: the whole Q tile is MAXM*NUM_Q_HEADS*HEAD_DIM
  // elements, so the index cannot exceed 16 bits here, and size_t arithmetic
  // makes the compiler build a full 64-bit address in VGPRs (v_lshlrev_b64
  // plus add_co pairs) for each of the MAXM loads instead of using the
  // SGPR-base + 32-bit-VGPR-offset form.
  fvec qr[MPW];
#pragma unroll
  for (int t = 0; t < MPW; ++t) {
    const unsigned qoff =
        ((unsigned)(mbase + t * MSPLIT) * NUM_Q_HEADS + (unsigned)h) *
            HEAD_DIM +
        (unsigned)dl;
    qr[t] = *(const fvec*)(q + qoff);
  }

  float acc[MPW][DPL], mx[MPW], ls[MPW];
#pragma unroll
  for (int t = 0; t < MPW; ++t) {
#pragma unroll
    for (int i = 0; i < DPL; ++i) acc[t][i] = 0.f;
    mx[t] = -INFINITY;
    ls[t] = 0.f;
  }

  // S is the sequence length the backend was handed, and it rejects an empty
  // one before it ever reaches here, so the loop below runs at least once and
  // ctx cannot underflow past -MAXM.  Stating it drops the zero-trip guard and
  // a compare; the codegen is smaller either way, and at this shape 81% of the
  // call is fixed cost, so do not expect it to show up as time.
  __builtin_assume(S > 0);
  const float scale2 = scale * LOG2E;
  const int ctx = S - MAXM;
#if ILV
  #if NSEG == 1
  const int gw = seg * NSLICE + slice;
  #else
  // Correctness, not speed: without readfirstlane the compiler cannot see that
  // jstart is wave-uniform, so it guards the loop with s_and_saveexec and only
  // restores exec inside the epilogue.  A wave with no tiles then reaches the
  // LDS stores with exec = 0 and skips them silently, while s_barrier (scalar)
  // still fires, so its neighbours read uninitialised LDS.  Only NSEG > 1 can
  // leave a whole wave empty -- and forcing the SGPR costs ~20% at NSEG == 1,
  // so it stays out of that path.
  const int gw = __builtin_amdgcn_readfirstlane(seg * NSLICE + slice);
  #endif
  const int j1 = S;
  const int jstart = gw * KPWE * SUB;
  const int jstep = NSEG * NSLICE * KPWE * SUB;
#else
  const int sl0 = (S + NSEG - 1) / NSEG;
  const int seg_len = (sl0 + KPWE * SUB - 1) / (KPWE * SUB) * (KPWE * SUB);
  const int j1 = min(S, seg * seg_len + seg_len);
  const int jstart = seg * seg_len + slice * KPWE * SUB;
  const int jstep = NSLICE * KPWE * SUB;
#endif

  for (int jb = jstart; jb < j1; jb += jstep) {
    // Issued together on purpose.  The compiler emits k0,v0,k1,k2,k3,v1,v2,v3
    // and resumes at s_waitcnt vmcnt(7), so V never blocks Q@K; its latency
    // hides behind the dot products, the lane reduction and the softmax.
    // Measured: deferring V to just before P@V costs 5.5% at S=32768, where
    // keeping eight loads in flight is what sustains the bandwidth.
    fvec kr[KPWE], vr[KPWE];
    // One block-table read per tile, forced into a scalar register.  The four
    // tokens share a block, and the value is wave-uniform, so the alternative
    // is four vector loads of the same 4 bytes broadcast to 32 lanes -- a third
    // of the loop's VMEM slots spent re-reading one integer.
    const int blk = __builtin_amdgcn_readfirstlane(bt[(unsigned)jb / BS]);
    // Block-table entries are page indices into the KV cache, never negative.
    // Without this the signed int forces the sign-extension path when blk
    // feeds the 64-bit address below.
    __builtin_assume(blk >= 0);
#if LAYOUT == 1
    // Under HND the tile's KPW tokens are consecutive slots of one block (that
    // is what the BS % KPW assert buys), so they sit at a fixed KV_ROW stride
    // and the whole tile addresses off a single base plus compile-time
    // offsets.  The largest is (KPW-1)*KV_ROW + HEAD_DIM = 3584 B, inside the
    // i13 INST_OFFSET, so one address feeds all eight loads instead of KPW
    // independent ones -- and the scalar block-table load feeds one address
    // chain rather than four.  NHD cannot do this: its token stride is
    // NUM_KV_HEADS*KV_ROW, which overflows the immediate.
    const size_t base =
        kv_off(blk, (unsigned)jb, kvh) + (size_t)grp * KV_ROW + dl;
  #pragma unroll
    for (int c = 0; c < KPWE; ++c) {
      const size_t off = base + (size_t)c * SUB * KV_ROW;
      kr[c] = *(const fvec*)(kv + off);
      vr[c] = *(const fvec*)(kv + off + HEAD_DIM);
    }
    // Tokens past S read whatever the page holds beyond the sequence.  The
    // address is in bounds by construction, not by luck: jb is a multiple of
    // KPW and BS % KPW == 0, so slot = jb % BS is too and slot + KPW-1 <= BS-1
    // keeps the tile inside the block; the furthest element of the furthest
    // lane is HEAD_DIM + 31*DPL + 7 = 511 < KV_ROW, so it stays inside the
    // row, hence the page, which vLLM allocates whole.
    //
    // The data is another matter: 0 * NaN is NaN and would poison the
    // accumulator even though the causal mask already zeroed this token's
    // weight.  K needs no guard -- its garbage dies in the mask's select.
    //
    // Only the last tile of a sequence whose length is not a multiple of KPW
    // can overrun, so the test is a wave-uniform scalar branch; testing per
    // element instead costs KPW*4 v_cndmask on every tile.
    if (__builtin_expect(jb + KPWE * SUB > S, 0)) {
  #pragma unroll
      for (int c = 0; c < KPWE; ++c)
        if (jb + c * SUB + grp >= S) vr[c] = fvec{};
    }
#else
  #pragma unroll
    for (int c = 0; c < KPWE; ++c) {
      int jj = jb + c * SUB + grp;
      jj = (jj < S) ? jj : (S - 1);
      const size_t off = kv_off(blk, (unsigned)jj, kvh) + dl;
      kr[c] = *(const fvec*)(kv + off);
      vr[c] = *(const fvec*)(kv + off + HEAD_DIM);
    }
#endif

    float s[KPWE][MPW];
#pragma unroll
    for (int c = 0; c < KPWE; ++c)
#pragma unroll
      for (int t = 0; t < MPW; ++t) {
        float d = 0.f;
#pragma unroll
        for (int e = 0; e < ((ABLATE & 4) ? 1 : FPL); ++e)
          d = __builtin_amdgcn_fdot2(as_h2(qr[t][e]), as_h2(kr[c][e]), d,
                                     false);
        s[c][t] = d;
      }

    // Measured: lowering these to DPP row_xmask instead is 18% SLOWER across
    // the whole context sweep.  The inner loop uses no LDS, so the LDS pipe is
    // idle and ds_bpermute runs there in parallel; DPP moves the work onto the
    // busy VALU pipe and costs a third of the dual-issue pairing as well.
    //
    // ds_bpermute is called directly rather than through __shfl_xor, which
    // cannot see that the partner index is in range and clamps it: a
    // v_cmp_gt_u32 against 32 and a v_cndmask per stride, guarding a condition
    // that `lane ^ st` with lane < 32 and st <= 16 can never violate, plus five
    // VGPRs held live for the whole kernel to carry the clamped indices.
    //
    // BFLY routes each butterfly element to one of the two pipes: ds_bpermute
    // on the LDS pipe, or v_permlane16/x16 on the VALU.  Unlike DPP, permlane
    // keeps the adds plain and VOPD-pairable, which is why it wins where DPP
    // lost.
    //
    // The split is over elements, not strides.  The strides are a dependence
    // chain
    // -- stride 2 consumes stride 1 -- so moving whole strides to the other
    // pipe buys nothing, while the KPWE*MPW elements inside one stride are
    // independent and can occupy both pipes at once.  BFLY is that ratio in
    // quarters: 0 sends every element to the LDS pipe, 4 sends every element to
    // the VALU.
#define BFLY_ON_VALU(idx) (((idx) & 3) < BFLY)
// Nibble i of the (lo, hi) pair is the source lane for destination lane i
// inside each 16-lane row, so the pair encodes an XOR-by-st swizzle.
#define BFLY_LO(st)          \
  ((st) == 1   ? 0x67452301u \
   : (st) == 2 ? 0x54761032u \
   : (st) == 4 ? 0x32107654u \
               : 0xFEDCBA98u)
#define BFLY_HI(st)          \
  ((st) == 1   ? 0xEFCDAB89u \
   : (st) == 2 ? 0xDCFE98BAu \
   : (st) == 4 ? 0xBA98FEDCu \
               : 0x76543210u)
// st == 16 is the only stride that leaves the 16-lane row.
#define BFLY_VALU(st, v)                                                    \
  ((st) >= 16 ? __builtin_amdgcn_permlanex16(                               \
                    __builtin_bit_cast(int, v), __builtin_bit_cast(int, v), \
                    0x76543210u, 0xFEDCBA98u, false, false)                 \
              : __builtin_amdgcn_permlane16(                                \
                    __builtin_bit_cast(int, v), __builtin_bit_cast(int, v), \
                    BFLY_LO(st), BFLY_HI(st), false, false))
#define BFLY_XOR(st, addr, idx, v) \
  (BFLY_ON_VALU(idx)               \
       ? BFLY_VALU(st, v)          \
       : __builtin_amdgcn_ds_bpermute(addr, __builtin_bit_cast(int, v)))
#pragma unroll
    for (int st = 1; st < ((ABLATE & 2) ? 2 : LPR); st <<= 1) {
      const int addr = (lane ^ st) << 2;  // ds_bpermute indexes lanes by byte
#pragma unroll
      for (int c = 0; c < KPWE; ++c)
#pragma unroll
        for (int t = 0; t < MPW; ++t)
          s[c][t] = __builtin_bit_cast(
                        float, BFLY_XOR(st, addr, c * MPW + t, s[c][t])) +
                    s[c][t];
    }

    // (jj <= ctx + m) already implies (jj < S): ctx + m <= S - 1 for every m,
    // so the bound test this mask also carried was dead weight.
    //
    // Measured and rejected: skipping the mask entirely on tiles below ctx via
    // a wave-uniform branch costs 1.6-7.7%.  The branch stops the scheduler
    // software-pipelining across it, which is worth more than the selects.
#pragma unroll
    for (int c = 0; c < KPWE; ++c) {
      const int jj = jb + c * SUB + grp;
#pragma unroll
      for (int t = 0; t < MPW; ++t) {
        const int m = mbase + t * MSPLIT;
#if MUTATE == 1
        const bool valid = (jj <= ctx + m + 1);
#else
        const bool valid = (jj <= ctx + m);
#endif
        s[c][t] = valid ? s[c][t] * scale2 : -INFINITY;
      }
    }

#pragma unroll
    for (int t = 0; t < MPW; ++t) {
      float mnew = mx[t];
#pragma unroll
      for (int c = 0; c < KPWE; ++c) mnew = fmaxf(mnew, s[c][t]);
      const float alpha =
          (mnew == -INFINITY) ? 0.f : __builtin_amdgcn_exp2f(mx[t] - mnew);
      mx[t] = mnew;
      ls[t] *= alpha;
#pragma unroll
      for (int i = 0; i < DPL; ++i) acc[t][i] *= alpha;
      float lsum = 0.f;
#pragma unroll
      for (int c = 0; c < KPWE; ++c) {
        const float p =
            (mnew == -INFINITY) ? 0.f : __builtin_amdgcn_exp2f(s[c][t] - mnew);
        s[c][t] = p;
        lsum += p;
      }
      ls[t] += lsum;
    }

#pragma unroll
    for (int c = 0; c < KPWE; ++c) {
      const _Float16* vv = (const _Float16*)&vr[c];
#pragma unroll
      for (int t = 0; t < MPW; ++t)
#pragma unroll
        // Not worth pairing into VOPD: thinning this loop to 1/8 of its work
        // buys 2.2% at S=128, and perfect dual-issue is only worth half of it.
        // Occupancy already hides the VALU behind other waves' loads.
        for (int i = 0; i < ((ABLATE & 1) ? 1 : DPL); ++i)
          acc[t][i] += s[c][t] * (float)vv[i];
    }
  }

#if FUSED
  // With NSEG==1 the grid is (1, NUM_Q_HEADS), so all NWAVE partials of a head
  // are produced by this one workgroup.  Merging them here costs a barrier and
  // HEAD_DIM floats of LDS per wave; routing them through global memory for a
  // second kernel costs a 2 MiB round trip and a launch, which at S=128 is as
  // many bytes as the KV stream itself.
  //
  // Every wave writes its MPW partials once, so all MAXM*NSLICE of them are
  // live at the same time and a single barrier serves the whole reduction.
  // The row for (token m, slice s) is (m/MSPLIT)*NWAVE + m%MSPLIT + s*MSPLIT,
  // which at MSPLIT == 1 is m*NWAVE + s and at MSPLIT == MAXM is m + s*MAXM.
  // SUB rows per wave now, not one: the groups of a wave hold the same output
  // elements over different tokens, so their partials are merged here with the
  // ones from the other KV slices rather than in a separate in-wave butterfly.
  __shared__ float lds_acc[MPW * NWAVE * SUB * LDS_STRIDE];
  __shared__ float lds_m[MPW * NWAVE * SUB];
  __shared__ float lds_l[MPW * NWAVE * SUB];

  // m and l are per-partial scalars, not per-element, so they are written once
  // and outlive every chunk; only the accumulator is re-staged per pass.
  #pragma unroll
  for (int t = 0; t < MPW; ++t)
    if (lrow == 0) {
      lds_m[(t * NWAVE + wave) * SUB + grp] = mx[t];
      lds_l[(t * NWAVE + wave) * SUB + grp] = ls[t];
    }

  #pragma unroll
  for (int cc = 0; cc < LDSPLIT; ++cc) {
  #if LDSPLIT > 1
    // Readers of the previous chunk must be done before its storage is reused.
    if (cc) __syncthreads();
    if (lrow / CHUNK_LPR == cc)
  #endif
    {
  #pragma unroll
      for (int t = 0; t < MPW; ++t)
  #pragma unroll
        for (int i = 0; i < DPL; ++i)
          lds_acc[((size_t)(t * NWAVE + wave) * SUB + grp) * LDS_STRIDE +
                  (lrow % CHUNK_LPR) * LDS_SLICE + i] = acc[t][i];
    }
    __syncthreads();

  #pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      const int wb = (m / MSPLIT) * NWAVE + (m % MSPLIT);

      float gmax = -INFINITY;
  #pragma unroll
      for (int p = 0; p < NPARTW; ++p)
        gmax = fmaxf(gmax, lds_m[partial_row(wb, p)]);

      // A wave with no tiles carries mx = -INFINITY and ls = 0, so its weight
      // is exp2(-inf) = 0 and it drops out on its own.  With NSEG > 1 a whole
      // workgroup can be empty though, and then gmax is -inf as well:
      // (-inf) - (-inf) is NaN, which would poison the partial.
      // Kept live rather than recomputed in the d loop: recomputing trades 32
      // VGPR at D=64 for a 1.5x loss at D=128 (122.62 -> 184.23 us at S=4096),
      // which is the wrong side of that trade.
      float a[NPARTW], den = 0.f;
  #pragma unroll
      for (int p = 0; p < NPARTW; ++p) {
        const int w = partial_row(wb, p);
        a[p] = weight_of(lds_m[w], gmax);
        den = fmaf(a[p], lds_l[w], den);
      }

  #if NSEG > 1
      // (num, gmax, den) is itself a valid partial softmax state, so hand the
      // global reduction one per (head, segment) rather than NWAVE of them.
      const size_t pb = ((size_t)h * NSEG + seg) * MAXM + m;
      // Every chunk recomputes the same value; only the first publishes it.
      if (tid == 0 && cc == 0) {
        p_m[pb] = gmax;
        p_l[pb] = den;
      }
  #endif
      // BLOCK need not equal HEAD_DIM once D is free: D=512 has more output
      // elements than threads and D=64 has fewer.  The equal case is spelled
      // out because the compiler cannot prove tid < HEAD_DIM and otherwise
      // wraps the body in an exec mask and a branch on a condition that is
      // always true.  A chunked epilogue covers less than HEAD_DIM per pass, so
      // it always takes the general loop.
  #if BLOCK == HEAD_DIM && LDSPLIT == 1
      const int d = tid;
      {
  #else
      for (int d = cc * CHUNK_D + tid; d < (cc + 1) * CHUNK_D; d += BLOCK) {
  #endif
        float num = 0.f;
  #pragma unroll
        for (int p = 0; p < NPARTW; ++p)
          num = fmaf(
              a[p],
              lds_acc[(size_t)partial_row(wb, p) * LDS_STRIDE + LDS_OFF(d)],
              num);
  #if NSEG == 1
        out[((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + d] =
            (OutT)fast_div(num, den);
  #else
        p_acc[pb * HEAD_DIM + d] = num;
  #endif
      }
    }
  }

  #if NSEG > 1 && FUSEDRED
  // Publish before announcing: the fence orders this workgroup's partial
  // stores ahead of the atomic, so whoever reads them after seeing the count
  // is guaranteed to see them.
  __threadfence();
  __shared__ int lds_last;
  if (tid == 0) lds_last = (atomicAdd(&p_cnt[h], 1) == NSEG - 1);
  __syncthreads();
  if (!lds_last) return;

  // Last arriver for this head. The counter goes back to zero here so the next
  // launch starts clean without the host touching it.
  if (tid == 0) p_cnt[h] = 0;
  __threadfence();

    #pragma unroll
  for (int m = 0; m < MAXM; ++m) {
    const size_t rb = (size_t)h * NSEG * MAXM + m;
    float gmax = -INFINITY;
    for (int s = 0; s < NSEG; ++s)
      gmax = fmaxf(gmax, p_m[rb + (size_t)s * MAXM]);

    for (int d = tid; d < HEAD_DIM; d += BLOCK) {
      float num = 0.f, den = 0.f;
      for (int s = 0; s < NSEG; ++s) {
        const size_t b = rb + (size_t)s * MAXM;
        // An empty segment carries p_m = -INFINITY, so exp2 of -inf is 0 and
        // it drops out without a branch.
        const float a = __builtin_amdgcn_exp2f(p_m[b] - gmax);
        den = fmaf(a, p_l[b], den);
        num = fmaf(a, p_acc[b * HEAD_DIM + d], num);
      }
      out[((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + d] =
          (OutT)fast_div(num, den);
    }
  }
  #endif
#else
  const size_t base = ((size_t)h * (NSEG * NWAVE) + seg * NWAVE + wave) * MAXM;
  #pragma unroll
  for (int m = 0; m < MAXM; ++m) {
  #pragma unroll
    for (int i = 0; i < DPL; ++i)
      p_acc[(base + m) * HEAD_DIM + dl + i] = acc[m][i];
    if (lane == 0) {
      p_m[base + m] = mx[m];
      p_l[base + m] = ls[m];
    }
  }
#endif
}

// Templated on the output type so the epilogue writes straight into vLLM's
// output tensor instead of through a staging buffer.
// One float4 per lane, not one float: the partials are read with the same
// global_load_b128 the main loop relies on, so the pass moves 16 B per lane per
// segment instead of 4.  Hence RED_THREADS = HEAD_DIM/4 threads per block.
#define RED_THREADS (HEAD_DIM / 4)

template <typename OutT>
__global__ __launch_bounds__(RED_THREADS) void reduce_segments(
    const float* __restrict__ p_acc, const float* __restrict__ p_m,
    const float* __restrict__ p_l, OutT* __restrict__ out) {
  const int m = blockIdx.x;
  const int h = blockIdx.y;
  const int d4 = threadIdx.x * 4;
  const size_t base = (size_t)h * NPART * MAXM + m;  // NPART partials for h

  // p_m and p_l are block-uniform, so these are scalar loads; only p_acc is
  // per-lane.  Deliberately not cached in registers: NPART reaches 128 at
  // NSEG=16 and the arrays would cost more VGPRs than the reload costs.
  float gmax = -INFINITY;
#pragma unroll
  for (int s = 0; s < NPART; ++s)
    gmax = fmaxf(gmax, p_m[base + (size_t)s * MAXM]);

  f4v num = {0.f, 0.f, 0.f, 0.f};
  float den = 0.f;
#pragma unroll
  for (int s = 0; s < NPART; ++s) {
    const size_t b = base + (size_t)s * MAXM;
    // A wave with no tiles carries p_m = -INFINITY, so its weight is exp2 of
    // -inf = 0 and it drops out without a branch.
    const float a = __builtin_amdgcn_exp2f(p_m[b] - gmax);
    den = fmaf(a, p_l[b], den);
    const f4v v = *(const f4v*)(p_acc + b * HEAD_DIM + d4);
#pragma unroll
    for (int i = 0; i < 4; ++i) num[i] = fmaf(a, v[i], num[i]);
  }

  const float inv = __builtin_amdgcn_rcpf(den);
  OutT* o = out + ((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + d4;
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = (OutT)(num[i] * inv);
}

// ------------------------------------------------------------- torch op ----
#ifdef RDNA35_TORCH_EXT

  #include <ATen/cuda/CUDAContext.h>
  #include <c10/cuda/CUDAGuard.h>
  // extension.h rather than all.h: it is the one that pulls in pybind11.
  #include <torch/extension.h>

// The scratch buffers (acc/m/l) are caller-allocated on purpose: this runs
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
  // fp16 only: the inner product is __builtin_amdgcn_fdot2, and the loads
  // reinterpret the cache as _Float16.  bf16 would read the same bits as fp16
  // and return finite nonsense, so refuse it here rather than downstream.
  TORCH_CHECK(q.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
              "kernel is fp16 only, got q=", q.scalar_type(),
              " out=", out.scalar_type());

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  #if GRIDT
  dim3 grid(NUM_Q_HEADS, NSEG), block(BLOCK);
  #else
  dim3 grid(NSEG, NUM_Q_HEADS), block(BLOCK);
  #endif
  dim3 rgrid(MAXM, NUM_Q_HEADS);

  const auto* qp = reinterpret_cast<const _Float16*>(q.data_ptr());
  const auto* kvp = reinterpret_cast<const _Float16*>(kv_cache.data_ptr());
  const int* btp = block_table.data_ptr<int>();
  float* accp = acc.data_ptr<float>();
  float* mp = m.data_ptr<float>();
  float* lp = l.data_ptr<float>();
  int* cntp = cnt.data_ptr<int>();

  auto* outp = reinterpret_cast<_Float16*>(out.data_ptr());
  hipLaunchKernelGGL(decode_attn<_Float16>, grid, block, 0, stream, qp, kvp,
                     btp, accp, mp, lp, cntp, outp, (int)seq_len, (float)scale);
  #if (NSEG > 1 && !FUSEDRED) || !FUSED
  hipLaunchKernelGGL(reduce_segments<_Float16>, rgrid, dim3(RED_THREADS), 0,
                     stream, accp, mp, lp, outp);
  #endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("decode_attn", &decode_attn_op, "RDNA3.5 paged decode attention");
}

#endif  // RDNA35_TORCH_EXT
