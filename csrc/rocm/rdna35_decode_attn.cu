// RDNA3.5 decode attention over vLLM's paged KV cache, GQA-packed.
//
// The cache is HND with K and V packed in the content dim:
//
//   (num_pages, NUM_KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM)
//
// Every build is one row of vllm/v1/attention/ops/rdna35_variants.csv, whose
// header names the defines (csrc/rocm/generate_rdna35_attn.py).
//
// One workgroup per (kv head, row group, KV segment) serves every query row
// that reads its kv head, so each KV byte is read from memory once.  Both
// products are WMMA 16x16x16 -> f32:
//
//   S^T[key][row] = K[key][:] . Q[row][:]        A = K tile, B = Q^T
//   O^T[d][row]  += V[key][d] . P[row][key]      A = V^T,    B = P^T
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

#include "rdna35_decode_attn.h"

#if !defined(HEAD_DIM) || !defined(NUM_Q_HEADS) || !defined(NUM_KV_HEADS)
  #error "HEAD_DIM, NUM_Q_HEADS and NUM_KV_HEADS come from the variant's row"
#endif
#if !defined(WINDOW) || !defined(MAX_QUERY_LEN) || !defined(BF16) || \
    !defined(BATCHED) || !defined(PAGE_SIZE)
  #error \
      "WINDOW, MAX_QUERY_LEN, BF16, BATCHED and PAGE_SIZE come from the variant's row"
#endif
#if !defined(MAX_SEGMENTS) || !defined(ROW_GROUPS) ||                        \
    !defined(MIN_SEGMENT_BLOCKS) || !defined(WAVES) || !defined(PREFETCH) || \
    !defined(V_IN_LDS) || !defined(DOT_PRODUCT)
  #error \
      "MAX_SEGMENTS, ROW_GROUPS, MIN_SEGMENT_BLOCKS, WAVES, PREFETCH, V_IN_LDS and DOT_PRODUCT come from the variant's row"
#endif

// WINDOW: keys p - WINDOW + 1 .. p for a query at p; 0 is full causal.
// BATCHED: one sequence per grid.y.
// MAX_SEGMENTS: most KV segments; clamp(blocks / MIN_SEGMENT_BLOCKS, 1,
//   MAX_SEGMENTS) are active, from the S read on the device, so a captured
//   CUDA graph follows S.
// ROW_GROUPS: workgroups sharing a kv head's GQA * MAX_QUERY_LEN rows.
// HEAD_DIM_SPLIT: waves splitting one key tile's head dim; 0 is the default.
// PREFETCH: a second tile's loads in flight per wave.
// V_IN_LDS: V staged in LDS beside K, so the next tile's loads go out early.
// DOT_PRODUCT: the per-q-head dot-product body instead of WMMA.
constexpr int kHeadDim = HEAD_DIM;
constexpr int kNumQHeads = NUM_Q_HEADS;
constexpr int kNumKvHeads = NUM_KV_HEADS;
constexpr int kWindow = WINDOW;
constexpr int kMaxQueryLen = MAX_QUERY_LEN;
constexpr bool kBf16 = BF16;
constexpr bool kBatched = BATCHED;
constexpr int kPageSize = PAGE_SIZE;
constexpr int kMaxSegments = MAX_SEGMENTS;
constexpr int kRowGroups = ROW_GROUPS;
constexpr int kMinSegmentBlocks = MIN_SEGMENT_BLOCKS;
constexpr int kWaves = WAVES;
constexpr bool kPrefetch = PREFETCH;
constexpr bool kVInLds = V_IN_LDS;
#if defined(HEAD_DIM_SPLIT) && HEAD_DIM_SPLIT
constexpr int kHeadDimSplit = HEAD_DIM_SPLIT;
#else
constexpr int kHeadDimSplit = kHeadDim >= 256 ? kHeadDim / 128 : 1;
#endif
// Measurement only, wrong answers: 1 skips the V loads, 2 thins P@V to one
// WMMA per tile, 4 skips the K loads, 16 thins Q@K to one WMMA per tile, 32
// returns at once, 64 loads each tile and does nothing else, 128 skips the
// score exchange.
#ifdef ABLATE
constexpr int kAblate = ABLATE;
#else
constexpr int kAblate = 0;
#endif

// Polls of the merge generation, ~0.1 us each, before a shared-merge waiter
// leaves its slice to the last arriver.
constexpr int kMaxSpins = 128;
// Workgroups a batch aims for when the host caps its segments.
constexpr int kBatchTargetWorkgroups = 128;
constexpr int kWaveSize = 32;
constexpr int kThreads = kWaves * kWaveSize;
constexpr int kQHeadsPerKvHead = kNumQHeads / kNumKvHeads;
constexpr int kKvRowElems = 2 * kHeadDim;
constexpr int kPageElems = kPageSize * kNumKvHeads * kKvRowElems;
constexpr float kLog2e = 1.44269504088896340736f;

static_assert(kNumQHeads % kNumKvHeads == 0, "GQA must be integral");
static_assert(kPageSize % 16 == 0, "a 16-key tile must sit inside one page");

#include "rdna35_attn_common.cuh"

// P's high half as an element, and as the float it stands for.  bf16
// truncates (gfx1151 has no f32 -> bf16 conversion); the low half carries
// what truncation drops.
__device__ __forceinline__ unsigned pack2(float a, float b) {
  if constexpr (kBf16)
    return __builtin_amdgcn_perm(__builtin_bit_cast(unsigned, b),
                                 __builtin_bit_cast(unsigned, a), 0x07060302u);
  else
    return __builtin_bit_cast(unsigned, e2{(elem_t)a, (elem_t)b});
}

__device__ __forceinline__ float elem_part(float p) {
  if constexpr (kBf16)
    return __builtin_bit_cast(float,
                              __builtin_bit_cast(unsigned, p) & 0xFFFF0000u);
  else
    return (float)(elem_t)p;
}

// WMMA's k order is free as long as A and B agree, and each half of the wave
// computes its output rows from its own copy of A and B.  So both products
// order k as "this half's keys, then the other half's", the order the S^T
// accumulator hands each half its keys in: no select on the half.
//
// B = P^T: lane l (row l & 15) holds P for keys 2e + (l >> 4) in probs[e].
__device__ __forceinline__ e16 p_frag(const float* probs) {
  u8v packed;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    packed[i] = pack2(probs[2 * i], probs[2 * i + 1]);
    packed[4 + i] = xhalf_u(packed[i]);
  }
  return __builtin_bit_cast(e16, packed);
}

#if DOT_PRODUCT
// One workgroup per (q head, segment), each wave kTileKeys keys at a time
// with the head dim across its lanes: no WMMA padding and no score exchange,
// which wins at short context.  A kv head's KV is read once per q head,
// through L2.
constexpr int kDimsPerLane = kHeadDim == 128 ? 8 : kHeadDim / kWaveSize;
constexpr int kLanesPerKey = kHeadDim / kDimsPerLane;
constexpr int kKeysPerStep = kWaveSize / kLanesPerKey;
constexpr int kKeysPerLane = std::max(4 / kKeysPerStep, 1);
constexpr int kTileKeys = kKeysPerLane * kKeysPerStep;
constexpr int kPairsPerLane = kDimsPerLane / 2;
constexpr int kSliceStride = kDimsPerLane + 1;
constexpr int kPartialStride = kLanesPerKey * kSliceStride;
constexpr int kPartials = kWaves * kKeysPerStep;
constexpr int kAccBytes = kMaxQueryLen * kPartials * kPartialStride * 4;
constexpr int kStatBytes = kMaxQueryLen * kPartials * 4;

static_assert(kLanesPerKey >= 1 && kLanesPerKey <= kWaveSize &&
                  kWaveSize % kLanesPerKey == 0,
              "a key must be covered by a whole divisor of the wave");
static_assert(kPageSize % kTileKeys == 0,
              "a tile must not straddle two blocks");
static_assert(kAccBytes + 2 * kStatBytes + 16 <= 65536,
              "the partials must fit LDS");

typedef float fvec __attribute__((ext_vector_type(kPairsPerLane)));

constexpr int kLds = kAccBytes + 2 * kStatBytes + 16;
constexpr int kGrid = kMaxSegments * kNumQHeads;
constexpr bool kSharedMerge = false;

// Head fastest: the q heads of a kv head, which read the same KV, are
// dispatched side by side.
__device__ __forceinline__ void dot_coords(int& segment, int& qHead) {
  const int workgroup = blockIdx.x;
  qHead = workgroup % kNumQHeads;
  segment = workgroup / kNumQHeads;
}

// The block index of the wave's first tile, read before S is known.
__device__ __forceinline__ int first_pages(const int* __restrict__ blockTable,
                                           int blockTableWidth) {
  int segment, qHead;
  dot_coords(segment, qHead);
  const int globalWave = __builtin_amdgcn_readfirstlane(
      segment * kWaves + threadIdx.x / kWaveSize);
  return blockTable[min(globalWave * kKeysPerLane * kKeysPerStep / kPageSize,
                        blockTableWidth - 1)];
}

__device__ __forceinline__ float dot2(float a, float b, float c) {
  if constexpr (kBf16)
    return __builtin_amdgcn_fdot2_f32_bf16(
        __builtin_bit_cast(s2v, a), __builtin_bit_cast(s2v, b), c, false);
  else
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a),
                                  __builtin_bit_cast(h2, b), c, false);
}

__device__ __forceinline__ int partial_row(int waveBase, int partial) {
  return waveBase * kKeysPerStep + partial;
}

// Lane i takes lane i ^ step's value, on the VALU rather than the LDS pipe.
__device__ __forceinline__ float butterfly(float value, int step) {
  const int bits = __builtin_bit_cast(int, value);
  if (step >= 16)
    return __builtin_bit_cast(
        float, __builtin_amdgcn_permlanex16(bits, bits, 0x76543210u,
                                            0xFEDCBA98u, false, false));
  const unsigned selectLow = step == 1   ? 0x67452301u
                             : step == 2 ? 0x54761032u
                             : step == 4 ? 0x32107654u
                                         : 0xFEDCBA98u;
  const unsigned selectHigh = step == 1   ? 0xEFCDAB89u
                              : step == 2 ? 0xDCFE98BAu
                              : step == 4 ? 0xBA98FEDCu
                                          : 0x76543210u;
  return __builtin_bit_cast(
      float, __builtin_amdgcn_permlane16(bits, bits, selectLow, selectHigh,
                                         false, false));
}

template <typename OutT>
__device__ __forceinline__ void body(
    const elem_t* __restrict__ q, const elem_t* __restrict__ kv,
    const int* __restrict__ blockTable, float* __restrict__ partialAcc,
    float* __restrict__ partialMax, float* __restrict__ partialSum,
    int* __restrict__ counters, OutT* __restrict__ out, const int seqLen,
    int pageIndices, int blockTableWidth, float scale, int cooperative,
    char* __restrict__ lds) {
  (void)cooperative;
  if (blockIdx.x >= kGrid) return;
  const int thread = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(thread / kWaveSize);
  int segment, qHead;
  dot_coords(segment, qHead);
  const int lane = thread & (kWaveSize - 1);
  const int keyLane = lane % kLanesPerKey;
  const int keyInStep = lane / kLanesPerKey;
  const int laneDim = keyLane * kDimsPerLane;
  const int kvHead = qHead / kQHeadsPerKvHead;

  fvec qRegs[kMaxQueryLen];
  #pragma unroll
  for (int t = 0; t < kMaxQueryLen; ++t) {
    const unsigned qOffset =
        ((unsigned)t * kNumQHeads + (unsigned)qHead) * kHeadDim +
        (unsigned)laneDim;
    qRegs[t] = *(const fvec*)(q + qOffset);
  }

  float accum[kMaxQueryLen][kDimsPerLane], runMax[kMaxQueryLen],
      runSum[kMaxQueryLen];
  #pragma unroll
  for (int t = 0; t < kMaxQueryLen; ++t) {
  #pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i) accum[t][i] = 0.f;
    runMax[t] = -INFINITY;
    runSum[t] = 0.f;
  }

  __builtin_assume(seqLen > 0);
  const float scaleLog2 = scale * kLog2e;
  const int context = seqLen - kMaxQueryLen;
  // readfirstlane: otherwise the loop below gets an exec mask, and a wave
  // with no tiles reached the LDS stores with exec = 0.
  const int globalWave =
      __builtin_amdgcn_readfirstlane(segment * kWaves + wave);
  // With a window, tiles start at the first visible key, on a tile boundary.
  const int firstKey = kWindow ? max(0, seqLen - kMaxQueryLen - (kWindow - 1)) /
                                     kTileKeys * kTileKeys
                               : 0;
  const int keyStart = firstKey + globalWave * kKeysPerLane * kKeysPerStep;
  const int keyStep = kMaxSegments * kWaves * kTileKeys;
  // Each tile's page is read a tile ahead.  The first came in with S, unless
  // a window makes it depend on S.
  int page;
  if constexpr (kWindow)
    page = __builtin_amdgcn_readfirstlane(
        blockTable[min(keyStart / kPageSize, blockTableWidth - 1)]);
  else
    page = __builtin_amdgcn_readfirstlane(pageIndices);

  for (int tileKey = keyStart; tileKey < seqLen; tileKey += keyStep) {
    fvec kRegs[kKeysPerLane], vRegs[kKeysPerLane];
    __builtin_assume(page >= 0);
    const size_t base =
        (size_t)page * kPageElems +
        ((size_t)kvHead * kPageSize + (unsigned)tileKey % kPageSize) *
            kKvRowElems +
        (size_t)keyInStep * kKvRowElems + laneDim;
  #pragma unroll
    for (int c = 0; c < kKeysPerLane; ++c) {
      const size_t offset = base + (size_t)c * kKeysPerStep * kKvRowElems;
      kRegs[c] = *(const fvec*)(kv + offset);
      vRegs[c] = *(const fvec*)(kv + offset + kHeadDim);
    }
    page = __builtin_amdgcn_readfirstlane(
        blockTable[min((tileKey + keyStep) / kPageSize, blockTableWidth - 1)]);
    // Keys past S read whatever their page holds: K's is masked, V's must
    // not reach the accumulator as 0 * NaN.
    if (__builtin_expect(tileKey + kTileKeys > seqLen, 0)) {
  #pragma unroll
      for (int c = 0; c < kKeysPerLane; ++c)
        if (tileKey + c * kKeysPerStep + keyInStep >= seqLen) vRegs[c] = fvec{};
    }

    float scores[kKeysPerLane][kMaxQueryLen];
  #pragma unroll
    for (int c = 0; c < kKeysPerLane; ++c)
  #pragma unroll
      for (int t = 0; t < kMaxQueryLen; ++t) {
        float dot = 0.f;
  #pragma unroll
        for (int e = 0; e < kPairsPerLane; ++e)
          dot = dot2(qRegs[t][e], kRegs[c][e], dot);
        scores[c][t] = dot;
      }
  #pragma unroll
    for (int step = 1; step < kLanesPerKey; step <<= 1) {
  #pragma unroll
      for (int c = 0; c < kKeysPerLane; ++c)
  #pragma unroll
        for (int t = 0; t < kMaxQueryLen; ++t)
          scores[c][t] = butterfly(scores[c][t], step) + scores[c][t];
    }
  #pragma unroll
    for (int c = 0; c < kKeysPerLane; ++c) {
      const int key = tileKey + c * kKeysPerStep + keyInStep;
  #pragma unroll
      for (int t = 0; t < kMaxQueryLen; ++t) {
        const bool causal = key <= context + t;
        if constexpr (kWindow)
          scores[c][t] =
              (key < context + t - (kWindow - 1)) ? -INFINITY : scores[c][t];
        scores[c][t] = causal ? scores[c][t] * scaleLog2 : -INFINITY;
      }
    }
  #pragma unroll
    for (int t = 0; t < kMaxQueryLen; ++t) {
      float newMax = runMax[t];
  #pragma unroll
      for (int c = 0; c < kKeysPerLane; ++c)
        newMax = fmaxf(newMax, scores[c][t]);
      const float rescale = (newMax == -INFINITY)
                                ? 0.f
                                : __builtin_amdgcn_exp2f(runMax[t] - newMax);
      runMax[t] = newMax;
      runSum[t] *= rescale;
  #pragma unroll
      for (int i = 0; i < kDimsPerLane; ++i) accum[t][i] *= rescale;
      float tileSum = 0.f;
  #pragma unroll
      for (int c = 0; c < kKeysPerLane; ++c) {
        const float prob = (newMax == -INFINITY)
                               ? 0.f
                               : __builtin_amdgcn_exp2f(scores[c][t] - newMax);
        scores[c][t] = prob;
        tileSum += prob;
      }
      runSum[t] += tileSum;
    }
  #pragma unroll
    for (int c = 0; c < kKeysPerLane; ++c) {
      const elem_t* v = (const elem_t*)&vRegs[c];
  #pragma unroll
      for (int t = 0; t < kMaxQueryLen; ++t)
  #pragma unroll
        for (int i = 0; i < kDimsPerLane; ++i)
          accum[t][i] += scores[c][t] * (float)v[i];
    }
  }

  // The waves' partials meet in LDS, then (with segments) in global memory.
  float* const ldsAccum = reinterpret_cast<float*>(lds);
  float* const ldsMax = reinterpret_cast<float*>(lds + kAccBytes);
  float* const ldsSum = ldsMax + kMaxQueryLen * kPartials;
  // Volatile: otherwise LLVM hoists the other threads' load above the
  // barrier, and they skip or half-do the merge.
  volatile int& ldsIsLast =
      *reinterpret_cast<volatile int*>(ldsSum + kMaxQueryLen * kPartials);
  #pragma unroll
  for (int t = 0; t < kMaxQueryLen; ++t) {
    if (keyLane == 0) {
      ldsMax[(t * kWaves + wave) * kKeysPerStep + keyInStep] = runMax[t];
      ldsSum[(t * kWaves + wave) * kKeysPerStep + keyInStep] = runSum[t];
    }
  #pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i)
      ldsAccum[((t * kWaves + wave) * kKeysPerStep + keyInStep) *
                   kPartialStride +
               keyLane * kSliceStride + i] = accum[t][i];
  }
  lds_barrier();

  #pragma unroll
  for (int t = 0; t < kMaxQueryLen; ++t) {
    const int waveBase = t * kWaves;
    float maxAll = -INFINITY;
  #pragma unroll
    for (int p = 0; p < kPartials; ++p)
      maxAll = fmaxf(maxAll, ldsMax[partial_row(waveBase, p)]);
    float weights[kPartials], denominator = 0.f;
  #pragma unroll
    for (int p = 0; p < kPartials; ++p) {
      const int row = partial_row(waveBase, p);
      weights[p] = weight_of(ldsMax[row], maxAll);
      denominator = fmaf(weights[p], ldsSum[row], denominator);
    }
    const size_t partialRow =
        ((size_t)qHead * kMaxSegments + segment) * kMaxQueryLen + t;
    if constexpr (kMaxSegments > 1) {
      if (thread == 0) {
        partialMax[partialRow] = maxAll;
        partialSum[partialRow] = denominator;
      }
    }
    for (int d = thread; d < kHeadDim; d += kThreads) {
      float numerator = 0.f;
  #pragma unroll
      for (int p = 0; p < kPartials; ++p)
        numerator =
            fmaf(weights[p],
                 ldsAccum[partial_row(waveBase, p) * kPartialStride +
                          (d / kDimsPerLane) * kSliceStride + d % kDimsPerLane],
                 numerator);
      if constexpr (kMaxSegments == 1)
        out[((size_t)t * kNumQHeads + qHead) * kHeadDim + d] =
            to_elem(numerator * __builtin_amdgcn_rcpf(denominator));
      else
        partialAcc[partialRow * kHeadDim + d] = numerator;
    }
  }

  if constexpr (kMaxSegments > 1) {
    // The last segment to arrive merges, and resets the counter for the next
    // launch.
    __threadfence();
    if (thread == 0)
      ldsIsLast = (atomicAdd(&counters[qHead], 1) == kMaxSegments - 1);
    lds_barrier();
    if (!ldsIsLast) return;
    if (thread == 0) counters[qHead] = 0;
    __threadfence();
  #pragma unroll
    for (int t = 0; t < kMaxQueryLen; ++t) {
      const size_t rowBase = (size_t)qHead * kMaxSegments * kMaxQueryLen + t;
      float maxAll = -INFINITY;
  #pragma unroll
      for (int s = 0; s < kMaxSegments; ++s)
        maxAll = fmaxf(maxAll, partialMax[rowBase + (size_t)s * kMaxQueryLen]);
      for (int d = thread; d < kHeadDim; d += kThreads) {
        float numerator = 0.f, denominator = 0.f;
  #pragma unroll
        for (int s = 0; s < kMaxSegments; ++s) {
          const size_t row = rowBase + (size_t)s * kMaxQueryLen;
          const float weight = weight_of(partialMax[row], maxAll);
          denominator = fmaf(weight, partialSum[row], denominator);
          numerator = fmaf(weight, partialAcc[row * kHeadDim + d], numerator);
        }
        out[((size_t)t * kNumQHeads + qHead) * kHeadDim + d] =
            to_elem(numerator * __builtin_amdgcn_rcpf(denominator));
      }
    }
  }
}

#else  // DOT_PRODUCT
constexpr int kRowsPerKvHead = kQHeadsPerKvHead * kMaxQueryLen;
// Row groups take consecutive (q head, token) rows, not whole q heads.
constexpr int kGroupRows = kRowsPerKvHead / kRowGroups;
constexpr int kRowTiles = (kGroupRows + 15) / 16;
constexpr int kPaddedRows = kRowTiles * 16;
constexpr int kKeyTiles = kWaves / kHeadDimSplit;
constexpr int kBlockKeys = 16 * kKeyTiles;
constexpr int kPartDims = kHeadDim / kHeadDimSplit;
constexpr int kPartChunks = kPartDims / 16;
// Dims of a K or V row per lane: lane l owns kLaneDims consecutive ones.
constexpr int kLaneDims = kPartDims / 16;
// P reaches P@V as a high plus a low half, ~22 bits in fp16 and ~16 in bf16;
// one fp16 P misses the 1e-3 relative bound near zero.  With at most 8 real
// rows per tile the low half rides in the padding columns of the same WMMA.
constexpr bool kPackLowHalf = kGroupRows <= 8;
// A K (or V) tile in LDS: 16 rows padded by 16 bytes, so the 16 keys one
// operand read touches sit on distinct banks.
constexpr int kTileRowBytes = kPartDims * 2 + 16;
constexpr int kTilesPerWave = kVInLds ? 2 : 1;
constexpr int kWaveTileBytes = kTilesPerWave * 16 * kTileRowBytes;
constexpr int kQRowPad = 8;
constexpr int kQBytes = kPaddedRows * (kHeadDim + kQRowPad) * 2;
// The merge buffer: one slot per (live tile, part) holds a row tile's real
// rows, kPartDims floats and a pad each.
constexpr int kMergeRowStride = kPartDims + 4;
constexpr int kMergeSlotFloats = std::min(kGroupRows, 16) * kMergeRowStride;
constexpr int kMergeBudgetBytes = 36 * 1024;
constexpr bool slots_fit(int tiles) {
  return tiles * kHeadDimSplit * kMergeSlotFloats * 4 <= kMergeBudgetBytes;
}
// Live tiles after the merge's tree rounds: the most that fit the budget.
constexpr int kFinalTiles = slots_fit(kKeyTiles)       ? kKeyTiles
                            : slots_fit(kKeyTiles / 2) ? kKeyTiles / 2
                            : slots_fit(kKeyTiles / 4) ? kKeyTiles / 4
                            : slots_fit(kKeyTiles / 8) ? kKeyTiles / 8
                                                       : 1;
constexpr int kMergeSlots =
    kFinalTiles == kKeyTiles ? kKeyTiles : std::max(kFinalTiles, kKeyTiles / 2);
constexpr int kMergeBytes =
    kKeyTiles == 1 ? 0 : kMergeSlots * kHeadDimSplit * kMergeSlotFloats * 4;
constexpr int kRawBytes =
    std::max(kQBytes + kWaves * kWaveTileBytes, kMergeBytes);
// Split KV: one workgroup merging every segment's partials is bound by its
// CU's bandwidth from 64 KiB on; past that every segment merges a slice.
constexpr bool kSharedMerge =
    kGroupRows * kHeadDim * kMaxSegments * 4 >= 64 * 1024;
// Per-wave max and sum, then the group's, then the merge's go flag.
constexpr int kStatsOffset = (kRawBytes + 15) / 16 * 16;
constexpr int kLds =
    kStatsOffset + (2 * kWaves * kPaddedRows + 2 * kPaddedRows) * 4 + 16;
// Counters: arrivals, then merge generations, one per (kv head, row group);
// then two committed-slice masks per group.
constexpr int kGrid = kMaxSegments * kNumKvHeads * kRowGroups;

static_assert(kRowsPerKvHead % kRowGroups == 0,
              "row groups split the rows evenly");
static_assert(kWaves % kHeadDimSplit == 0, "whole tiles per workgroup");
static_assert(kHeadDim / 16 % kHeadDimSplit == 0, "whole chunks per part");
static_assert(kLaneDims == 4 || kLaneDims == 8,
              "a lane's K and V slice is one b64 or b128");
static_assert(kKeyTiles <= kWaveSize, "one lane per tile loads its page");
static_assert(kHeadDimSplit == 1 || kRowTiles * 1024 <= 16 * kTileRowBytes,
              "the score exchange must fit a wave's K tile");
static_assert(!kVInLds || !kPrefetch,
              "V_IN_LDS replaces PREFETCH's second tile");

using vrow_t = std::conditional_t<kLaneDims == 8, u4v, u2v>;

// One wave's share of a 16-key tile: half h of the wave holds keys 2e + h,
// lane l kLaneDims dims of each.  The halves load different keys and swap
// with one permlanex16 per dword, which halves registers and loads.
struct Tile {
  vrow_t k[8];
  vrow_t v[8];
};

// Page index of every 16-key tile of a block: lane t holds tile t's.  Tiles
// past the sequence clamp to its last page.
__device__ __forceinline__ int block_pages(const int* __restrict__ blockTable,
                                           int block, int lane, int seqLen) {
  const int key = block * kBlockKeys + (lane % kKeyTiles) * 16;
  // One tile: a uniform index, read through the scalar cache.
  if constexpr (kKeyTiles == 1)
    return blockTable[__builtin_amdgcn_readfirstlane(min(key, seqLen - 1)) /
                      kPageSize];
  return blockTable[(unsigned)min(key, seqLen - 1) / kPageSize];
}

// The same before S is known: clamped to the table's width instead, whose
// every entry is an allocated page.
__device__ __forceinline__ int block_pages_w(const int* __restrict__ blockTable,
                                             int block, int lane,
                                             int blockTableWidth) {
  const int tableIndex =
      (block * kBlockKeys + (lane % kKeyTiles) * 16) / kPageSize;
  if constexpr (kKeyTiles == 1)
    return blockTable[__builtin_amdgcn_readfirstlane(
        min(tableIndex, blockTableWidth - 1))];
  return blockTable[min(tableIndex, blockTableWidth - 1)];
}

// A = V^T for element `dim` of this lane's slice, keys in p_frag's order.
__device__ __forceinline__ e16 v_frag(const vrow_t* v, int dim) {
  const int word = dim >> 1;
  // v_perm_b32(hi, lo, selector): half dim & 1 of each source dword.
  const unsigned selector = (dim & 1) ? 0x07060302u : 0x05040100u;
  u8v packed;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    packed[i] =
        __builtin_amdgcn_perm(v[2 * i + 1][word], v[2 * i][word], selector);
    packed[4 + i] = xhalf_u(packed[i]);
  }
  return __builtin_bit_cast(e16, packed);
}

// A wave's real rows of one row tile, `stride` floats apart: for element e
// lane l owns kLaneDims consecutive dims at kLaneDims * (2e + laneHalf).
__device__ __forceinline__ void store_rows(float* dst, const f8* accum,
                                           int laneRow, int laneHalf,
                                           int stride = kMergeRowStride) {
  float* row = dst + laneRow * stride;
  #pragma unroll
  for (int e = 0; e < 8; ++e) {
    float values[kLaneDims];
  #pragma unroll
    for (int j = 0; j < kLaneDims; ++j) values[j] = accum[j][e];
  #pragma unroll
    for (int k = 0; k < kLaneDims; k += 4)
      *(f4*)(row + kLaneDims * (2 * e + laneHalf) + k) =
          *(const f4*)(values + k);
  }
}

__device__ __forceinline__ void add_rows(f8* accum, const float* src,
                                         int laneRow, int laneHalf) {
  const float* row = src + laneRow * kMergeRowStride;
  #pragma unroll
  for (int e = 0; e < 8; ++e)
  #pragma unroll
    for (int k = 0; k < kLaneDims; k += 4) {
      const f4 values = *(const f4*)(row + kLaneDims * (2 * e + laneHalf) + k);
  #pragma unroll
      for (int j = 0; j < 4; ++j) accum[k + j][e] += values[j];
    }
}

// The first block's page indices, read before S is known.
__device__ __forceinline__ int first_pages(const int* __restrict__ blockTable,
                                           int blockTableWidth) {
  const int segment = blockIdx.x / (kRowGroups * kNumKvHeads);
  return block_pages_w(blockTable, segment, threadIdx.x & (kWaveSize - 1),
                       blockTableWidth);
}

template <typename OutT>
__device__ __forceinline__ void body(
    const elem_t* __restrict__ q, const elem_t* __restrict__ kv,
    const int* __restrict__ blockTable, float* __restrict__ partialAcc,
    float* __restrict__ partialMax, float* __restrict__ partialSum,
    int* __restrict__ counters, OutT* __restrict__ out, const int seqLen,
    int pageIndices, int blockTableWidth, float scale, int cooperative,
    char* __restrict__ lds) {
  const int thread = threadIdx.x;
  const int lane = thread & (kWaveSize - 1);
  const int wave = __builtin_amdgcn_readfirstlane(thread / kWaveSize);
  const int laneRow = lane & 15;
  const int laneHalf = lane >> 4;
  const int keyTile = wave / kHeadDimSplit;
  const int dimPart = wave % kHeadDimSplit;

  const int workgroup = blockIdx.x;
  const int rowGroup = workgroup % kRowGroups;
  const int kvHead = (workgroup / kRowGroups) % kNumKvHeads;
  const int segment = workgroup / (kRowGroups * kNumKvHeads);

  __builtin_assume(seqLen > 0);
  const int numBlocks = (seqLen + kBlockKeys - 1) / kBlockKeys;
  // With a window, the first block holding a key some query can see.
  const int firstBlock =
      kWindow ? max(0, seqLen - kMaxQueryLen - (kWindow - 1)) / kBlockKeys : 0;
  int numSegments;
  if constexpr (kBatched) {
    // A batch brings its own parallelism: the host caps its segments.
    const int segmentCap = cooperative >> 1;
    cooperative &= 1;
    numSegments = max(1, min(min(kMaxSegments, segmentCap),
                             (numBlocks - firstBlock) / kMinSegmentBlocks));
  } else {
    numSegments =
        max(1, min(kMaxSegments, (numBlocks - firstBlock) / kMinSegmentBlocks));
  }
  if (segment >= numSegments) return;
  // The page indices that came in with S were for block segment.
  if constexpr (kWindow)
    pageIndices = block_pages(blockTable, firstBlock + segment, lane, seqLen);
  if constexpr (kAblate & 32) {
    if (seqLen == -1) out[0] = (OutT)0;
    return;
  }

  // Q goes out before the KV: loads complete in order, and a Q load behind
  // the KV would hold the barrier until the last KV byte landed.
  constexpr int kQLoads =
      (kPaddedRows * kHeadDim + kThreads * 8 - 1) / (kThreads * 8);
  e8 qLoads[kQLoads];
  #pragma unroll
  for (int it = 0; it < kQLoads; ++it) {
    const int elem = (it * kThreads + thread) * 8;
    // From a clamped row, not under a branch, which made the compiler wait
    // for it before the first KV load.
    const int row = min(elem / kHeadDim, kGroupRows - 1), dim = elem % kHeadDim;
    const int qHead = kvHead * kQHeadsPerKvHead +
                      (rowGroup * kGroupRows + row) / kMaxQueryLen;
    const int token = (rowGroup * kGroupRows + row) % kMaxQueryLen;
    qLoads[it] =
        *(const e8*)(q + ((size_t)token * kNumQHeads + qHead) * kHeadDim + dim);
    if (elem / kHeadDim >= kGroupRows) qLoads[it] = e8{};
  }
  // The group's merge generation cannot move before this segment arrives.
  int* const generation =
      counters + kNumKvHeads * kRowGroups + kvHead * kRowGroups + rowGroup;
  // Bit s set: segment s merges its own slice.  One mask for this launch's
  // generation and one for the next's.
  int* const committedMask = counters + 2 * kNumKvHeads * kRowGroups +
                             2 * (kvHead * kRowGroups + rowGroup);
  const int startGeneration =
      kSharedMerge ? __hip_atomic_load(generation, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT)
                   : 0;

  // Q and the K tiles are dead after the main loop; the merge reuses them.
  auto qShared = reinterpret_cast<elem_t(*)[kHeadDim + kQRowPad]>(lds);
  float* mergeShared = reinterpret_cast<float*>(lds);
  char* waveTile = lds + kQBytes + wave * kTilesPerWave * 16 * kTileRowBytes;
  // A wave's partial scores go in its own K tile, after its Q@K read it.
  float* scoreShared = reinterpret_cast<float*>(lds + kQBytes);
  auto waveMax = reinterpret_cast<float (*)[kPaddedRows]>(lds + kStatsOffset);
  auto waveSum = reinterpret_cast<float (*)[kPaddedRows]>(
      lds + kStatsOffset + kWaves * kPaddedRows * 4);
  float* const groupMax = reinterpret_cast<float*>(
      lds + kStatsOffset + 2 * kWaves * kPaddedRows * 4);
  float* const groupSum = groupMax + kPaddedRows;

  const elem_t* kvHeadBase = kv + (size_t)kvHead * (kPageSize * kKvRowElems);
  const int laneDim = dimPart * kPartDims + kLaneDims * laneRow;
  const int firstChunk = dimPart * kPartChunks;

  // Issue this wave's tile of `block`, and the page lookup of the block after
  // it: blocks go out in order, so pageIndices always holds this block's.
  auto issue = [&](Tile& tile, int block) {
    const int page = __builtin_amdgcn_readlane(pageIndices, keyTile);
    if (block + numSegments < numBlocks)
      pageIndices = block_pages(blockTable, block + numSegments, lane, seqLen);
    const elem_t* src =
        kvHeadBase + (size_t)page * kPageElems +
        (size_t)((block * kBlockKeys + keyTile * 16) % kPageSize) *
            kKvRowElems +
        (size_t)laneHalf * kKvRowElems + laneDim;
    // All of V, then all of K: interleaved, they landed later.
  #pragma unroll
    for (int e = 0; e < 8; ++e)
      tile.v[e] = (kAblate & 1)
                      ? vrow_t{}
                      : *(const vrow_t*)(src + 2 * e * kKvRowElems + kHeadDim);
  #pragma unroll
    for (int e = 0; e < 8; ++e)
      tile.k[e] = (kAblate & 4) ? vrow_t{}
                                : *(const vrow_t*)(src + 2 * e * kKvRowElems);
    // Keeps the scheduler from sinking the loads to their first use.
    asm volatile("" ::: "memory");
  };

  Tile tileA;
  issue(tileA, firstBlock + segment);

  #pragma unroll
  for (int it = 0; it < kQLoads; ++it) {
    const int elem = (it * kThreads + thread) * 8;
    if (elem < kPaddedRows * kHeadDim)
      *(e8*)&qShared[elem / kHeadDim][elem % kHeadDim] = qLoads[it];
  }

  const float scaleLog2 = scale * kLog2e;
  const int context = seqLen - kMaxQueryLen;

  float runMax[kRowTiles], runSum[kRowTiles];
  f8 accum[kRowTiles][kLaneDims];
  #pragma unroll
  for (int rt = 0; rt < kRowTiles; ++rt) {
    runMax[rt] = -INFINITY;
    runSum[rt] = 0.f;
  #pragma unroll
    for (int j = 0; j < kLaneDims; ++j) accum[rt][j] = f8{};
  }
  // Token of this lane's row in each row tile, for the causal mask.
  int rowToken[kRowTiles];
  #pragma unroll
  for (int rt = 0; rt < kRowTiles; ++rt) {
    const int row = rt * 16 + laneRow;
    rowToken[rt] = (row < kGroupRows)
                       ? ((rowGroup * kGroupRows + row) % kMaxQueryLen)
                       : (kMaxQueryLen - 1);
  }

  lds_barrier();

  // Stage a tile in this wave's LDS tile, rows in, keys out; one wave's LDS
  // ops complete in order.  The asm keeps these integer stores from passing
  // the half reads of the same bytes.
  auto stageKv = [&](const Tile& tile) {
  #pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int offset =
          (2 * e + laneHalf) * kTileRowBytes + kLaneDims * laneRow * 2;
      *(vrow_t*)(waveTile + offset) = tile.k[e];
      *(vrow_t*)(waveTile + 16 * kTileRowBytes + offset) = tile.v[e];
    }
    asm volatile("" ::: "memory");
  };
  auto stageK = [&](const Tile& tile) {
  #pragma unroll
    for (int e = 0; e < 8; ++e)
      *(vrow_t*)(waveTile + (2 * e + laneHalf) * kTileRowBytes +
                 kLaneDims * laneRow * 2) = tile.k[e];
    asm volatile("" ::: "memory");
  };

  // Q@K, the softmax update and P@V for a tile staged at kTile; with
  // V_IN_LDS, V is read from there too.
  auto process = [&](Tile& tile, vrow_t* v, const char* kTile, int block) {
    const int tileKey = block * kBlockKeys + keyTile * 16;
    if constexpr (kAblate & 64) {
      unsigned x = 0;
  #pragma unroll
      for (int e = 0; e < 8; ++e)
  #pragma unroll
        for (int w = 0; w < kLaneDims / 2; ++w)
          x ^= tile.v[e][w] ^ tile.k[e][w];
      accum[0][0][0] += __builtin_bit_cast(float, x & 0x3f800000u);
      return;
    }
    f8 scores[kRowTiles];
  #pragma unroll
    for (int rt = 0; rt < kRowTiles; ++rt) scores[rt] = f8{};
  #pragma unroll
    for (int c = 0; c < kPartChunks; ++c) {
      const e16 kOperand =
          *(const e16*)(kTile + laneRow * kTileRowBytes + c * 32);
  #pragma unroll
      for (int rt = 0; rt < kRowTiles; ++rt) {
        const e16 qOperand =
            *(const e16*)&qShared[rt * 16 + laneRow][(firstChunk + c) * 16];
        if (!(kAblate & 16) || c == 0)
          scores[rt] = wmma(kOperand, qOperand, scores[rt]);
      }
    }
    if constexpr (kHeadDimSplit > 1 && !(kAblate & 128)) {
      // Sum the parts' scores in the same order in every wave, so they all
      // agree bit for bit on the softmax.  The asm keeps the float stores
      // below Q@K's half reads of the same tile.
      asm volatile("" ::: "memory");
  #pragma unroll
      for (int rt = 0; rt < kRowTiles; ++rt)
        *(f8*)&scoreShared[wave * kTilesPerWave * (16 * kTileRowBytes / 4) +
                           rt * 256 + lane * 8] = scores[rt];
      lds_barrier();
  #pragma unroll
      for (int rt = 0; rt < kRowTiles; ++rt) {
        f8 sum = *(const f8*)&scoreShared[(wave - dimPart) * kTilesPerWave *
                                              (16 * kTileRowBytes / 4) +
                                          rt * 256 + lane * 8];
  #pragma unroll
        for (int part = 1; part < kHeadDimSplit; ++part)
          sum +=
              *(const f8*)&scoreShared[(wave - dimPart + part) * kTilesPerWave *
                                           (16 * kTileRowBytes / 4) +
                                       rt * 256 + lane * 8];
        scores[rt] = sum;
      }
      // Partners are done reading before anyone refills its K tile.
      lds_barrier();
    }
    vrow_t ldsV[8];
    if constexpr (kVInLds) {
  #pragma unroll
      for (int e = 0; e < 8; ++e)
        ldsV[e] = *(const vrow_t*)(kTile + 16 * kTileRowBytes +
                                   (2 * e + laneHalf) * kTileRowBytes +
                                   kLaneDims * laneRow * 2);
      v = ldsV;
    }

    // Zero the V of keys past S, and of keys before every query's window,
    // whose pages vLLM may have freed: no NaN reaches P@V as 0 * NaN.
    if (__builtin_expect(tileKey + 16 > seqLen, 0)) {
  #pragma unroll
      for (int e = 0; e < 8; ++e)
        if (tileKey + 2 * e + laneHalf >= seqLen) v[e] = vrow_t{};
    }
    if constexpr (kWindow) {
      if (__builtin_expect(tileKey < context - (kWindow - 1), 0)) {
  #pragma unroll
        for (int e = 0; e < 8; ++e)
          if (tileKey + 2 * e + laneHalf < context - (kWindow - 1))
            v[e] = vrow_t{};
      }
    }

    // Lane holds row laneRow, keys tileKey + 2e + laneHalf.  Only a tile past
    // the first query's keys needs the causal mask, and only one below the last
    // query's window its lower edge.  The scale is applied in the exponent.
    const bool pastContext = tileKey + 16 > context + 1;
    const bool belowWindow =
        kWindow && tileKey < context + kMaxQueryLen - kWindow;
  #pragma unroll
    for (int rt = 0; rt < kRowTiles; ++rt) {
      if (__builtin_expect(pastContext || belowWindow, 0)) {
  #pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int key = tileKey + 2 * e + laneHalf;
          const bool causal = key <= context + rowToken[rt];
          if constexpr (kWindow)
            if (key < context + rowToken[rt] - (kWindow - 1))
              scores[rt][e] = -INFINITY;
          if (!causal) scores[rt][e] = -INFINITY;
        }
      }
      float tileMax = scores[rt][0];
  #pragma unroll
      for (int e = 1; e < 8; ++e) tileMax = fmaxf(tileMax, scores[rt][e]);
      tileMax = fmaxf(tileMax, xhalf(tileMax)) * scaleLog2;
      const float newMax = fmaxf(runMax[rt], tileMax);
      const float rescale = (newMax == -INFINITY)
                                ? 1.f
                                : __builtin_amdgcn_exp2f(runMax[rt] - newMax);
      runMax[rt] = newMax;
      float probs[8], probsLow[8], tileSum = 0.f;
  #pragma unroll
      for (int e = 0; e < 8; ++e) {
        probs[e] = (newMax == -INFINITY)
                       ? 0.f
                       : __builtin_amdgcn_exp2f(
                             __builtin_fmaf(scores[rt][e], scaleLog2, -newMax));
        probsLow[e] = probs[e] - elem_part(probs[e]);
        tileSum += probs[e];
      }
      runSum[rt] = runSum[rt] * rescale + tileSum + xhalf(tileSum);

      e16 pOperand, pLowOperand;
      float accumScale;
      if constexpr (kPackLowHalf) {
        // Columns 0..7 carry P's high half for rows 0..7, columns 8..15 its
        // low half for the same rows: lanes 8..15 accumulate for row
        // laneRow - 8 and take its rescale.  A mask, not a ?:, which the
        // compiler made a branch computing the low half in lanes 8..15 only.
        const u8v high = __builtin_bit_cast(u8v, p_frag(probs));
        const u8v low = __builtin_bit_cast(u8v, p_frag(probsLow));
        const unsigned upperMask = 0u - (unsigned)(laneRow >> 3);
        u8v packed;
  #pragma unroll
        for (int i = 0; i < 8; ++i)
          packed[i] = (high[i] & ~upperMask) | (lower8(low[i]) & upperMask);
        pOperand = __builtin_bit_cast(e16, packed);
        const unsigned rescaleBits = __builtin_bit_cast(unsigned, rescale);
        accumScale =
            __builtin_bit_cast(float, (rescaleBits & ~upperMask) |
                                          (lower8(rescaleBits) & upperMask));
      } else {
        pOperand = p_frag(probs);
        pLowOperand = p_frag(probsLow);
        accumScale = rescale;
      }
  #pragma unroll
      for (int j = 0; j < kLaneDims; ++j) {
        accum[rt][j] *= accumScale;
        if ((kAblate & 2) && j) continue;
        const e16 vOperand = v_frag(v, j);
        accum[rt][j] = wmma(vOperand, pOperand, accum[rt][j]);
        if constexpr (!kPackLowHalf)
          accum[rt][j] = wmma(vOperand, pLowOperand, accum[rt][j]);
      }
    }
    asm volatile("" ::: "memory");
  };

  if constexpr (kPrefetch) {
    Tile tileB;
    for (int block = firstBlock + segment; block < numBlocks;
         block += 2 * numSegments) {
      stageK(tileA);
      if (block + numSegments < numBlocks) issue(tileB, block + numSegments);
      process(tileA, tileA.v, waveTile, block);
      if (block + numSegments >= numBlocks) break;
      stageK(tileB);
      if (block + 2 * numSegments < numBlocks)
        issue(tileA, block + 2 * numSegments);
      process(tileB, tileB.v, waveTile, block + numSegments);
    }
  } else if constexpr (kVInLds) {
    for (int block = firstBlock + segment; block < numBlocks;
         block += numSegments) {
      stageKv(tileA);
      if (block + numSegments < numBlocks) issue(tileA, block + numSegments);
      process(tileA, nullptr, waveTile, block);
    }
  } else {
    for (int block = firstBlock + segment; block < numBlocks;
         block += numSegments) {
      stageK(tileA);
      process(tileA, tileA.v, waveTile, block);
      if (block + numSegments < numBlocks) issue(tileA, block + numSegments);
    }
  }
  if constexpr (kPackLowHalf) {
    // Fold the low half's columns onto their rows.
  #pragma unroll
    for (int j = 0; j < kLaneDims; ++j)
  #pragma unroll
      for (int e = 0; e < 8; ++e) accum[0][j][e] += upper8f(accum[0][j][e]);
  }

  // Merge the waves: each rescales to the workgroup's max, tree rounds halve
  // the live tiles until their partials fit the budget, then the live tiles
  // store their real rows and the workgroup sums them.
  #pragma unroll
  for (int rt = 0; rt < kRowTiles; ++rt)
    if (laneHalf == 0) {
      waveMax[wave][rt * 16 + laneRow] = runMax[rt];
      waveSum[wave][rt * 16 + laneRow] = runSum[rt];
    }
  lds_barrier();
  float groupSums[kRowTiles];
  #pragma unroll
  for (int rt = 0; rt < kRowTiles; ++rt) {
    const int row = rt * 16 + laneRow;
    float maxAll = -INFINITY;
  #pragma unroll
    for (int w = dimPart; w < kWaves; w += kHeadDimSplit)
      maxAll = fmaxf(maxAll, waveMax[w][row]);
    float sum = 0.f;
  #pragma unroll
    for (int w = dimPart; w < kWaves; w += kHeadDimSplit)
      sum += weight_of(waveMax[w][row], maxAll) * waveSum[w][row];
    groupSums[rt] = sum;
    if (keyTile == 0 && dimPart == 0 && laneHalf == 0) {
      groupMax[row] = maxAll;
      groupSum[row] = sum;
    }
    const float weight = weight_of(runMax[rt], maxAll);
  #pragma unroll
    for (int j = 0; j < kLaneDims; ++j) accum[rt][j] *= weight;
  }

  const int group = kvHead * kRowGroups + rowGroup;
  const size_t partialBase =
      ((size_t)group * kMaxSegments + segment) * kPaddedRows;
  const bool publish = numSegments > 1;

  #pragma unroll
  for (int rt = 0; rt < kRowTiles; ++rt) {
    const int realRows = min(16, kGroupRows - rt * 16);
    // The previous row tile's readers must be done with the buffer.
    if (rt) lds_barrier();
  #pragma unroll
    for (int stride = kKeyTiles / 2; stride >= kFinalTiles; stride >>= 1) {
      if (keyTile >= stride && keyTile < 2 * stride && laneRow < realRows)
        store_rows(
            mergeShared + ((keyTile - stride) * kHeadDimSplit + dimPart) *
                              kMergeSlotFloats,
            accum[rt], laneRow, laneHalf);
      lds_barrier();
      if (keyTile < stride && laneRow < realRows)
        add_rows(accum[rt],
                 mergeShared +
                     (keyTile * kHeadDimSplit + dimPart) * kMergeSlotFloats,
                 laneRow, laneHalf);
      lds_barrier();
    }
    if constexpr (kFinalTiles == 1) {
      // One live tile: its waves write straight from registers.
      if (keyTile == 0 && laneRow < realRows) {
        const int row = rt * 16 + laneRow;
        if (!publish) {
          const int qHead = kvHead * kQHeadsPerKvHead +
                            (rowGroup * kGroupRows + row) / kMaxQueryLen;
          const int token = (rowGroup * kGroupRows + row) % kMaxQueryLen;
          const float inv = __builtin_amdgcn_rcpf(groupSums[rt]);
          OutT* dst = out + ((size_t)token * kNumQHeads + qHead) * kHeadDim +
                      dimPart * kPartDims;
  #pragma unroll
          for (int e = 0; e < 8; ++e) {
            elem_t x[kLaneDims];
  #pragma unroll
            for (int j = 0; j < kLaneDims; ++j)
              x[j] = to_elem(accum[rt][j][e] * inv);
            *(vrow_t*)(dst + kLaneDims * (2 * e + laneHalf)) =
                *(const vrow_t*)x;
          }
        } else {
          store_rows(partialAcc + (partialBase + rt * 16) * kHeadDim +
                         dimPart * kPartDims,
                     accum[rt], laneRow, laneHalf, kHeadDim);
        }
      }
    } else {
      if (keyTile < kFinalTiles && laneRow < realRows)
        store_rows(mergeShared +
                       (keyTile * kHeadDimSplit + dimPart) * kMergeSlotFloats,
                   accum[rt], laneRow, laneHalf);
      lds_barrier();
      // Eight consecutive dims of one real row per thread.
      const int firstRow = rt * 16;
      const int numRows = min(16, kGroupRows - firstRow);
      for (int elem = thread * 8; elem < numRows * kHeadDim;
           elem += kThreads * 8) {
        const int tileRow = elem / kHeadDim, dim = elem % kHeadDim;
        const int row = firstRow + tileRow;
        const float* src = mergeShared +
                           ((tileRow / 16) * kHeadDimSplit + dim / kPartDims) *
                               kMergeSlotFloats +
                           (tileRow % 16) * kMergeRowStride + dim % kPartDims;
        f4 low = *(const f4*)src, high = *(const f4*)(src + 4);
  #pragma unroll
        for (int t = 1; t < kFinalTiles; ++t) {
          low += *(const f4*)(src + t * kHeadDimSplit * kMergeSlotFloats);
          high += *(const f4*)(src + t * kHeadDimSplit * kMergeSlotFloats + 4);
        }
        if (!publish) {
          const float inv = __builtin_amdgcn_rcpf(groupSum[row]);
          e8 values;
  #pragma unroll
          for (int k = 0; k < 4; ++k) {
            values[k] = to_elem(low[k] * inv);
            values[4 + k] = to_elem(high[k] * inv);
          }
          const int qHead = kvHead * kQHeadsPerKvHead +
                            (rowGroup * kGroupRows + row) / kMaxQueryLen;
          const int token = (rowGroup * kGroupRows + row) % kMaxQueryLen;
          *(e8*)(out + ((size_t)token * kNumQHeads + qHead) * kHeadDim + dim) =
              values;
        } else {
          float* dst = partialAcc + (partialBase + row) * kHeadDim + dim;
          *(f4*)dst = low;
          *(f4*)(dst + 4) = high;
        }
      }
    }
  }
  if (!publish) return;

  // Split KV: publish (accum, max, sum); the last segment to arrive merges
  // and resets the counter, so a CUDA-graph replay starts clean.  With a
  // shared merge (the host checked the whole grid is resident, so waiting
  // cannot deadlock) the others wait for it to bump the generation, and
  // every segment merges its slice.  A loop: with two waves and M >= 5 the
  // rows outnumber the threads.
  for (int row = thread; row < kGroupRows; row += kThreads) {
    partialMax[partialBase + row] = groupMax[row];
    partialSum[partialBase + row] = groupSum[row];
  }
  __threadfence();
  cooperative = kSharedMerge && cooperative;
  // Volatile, like ldsIsLast in the dot body.
  volatile int& ldsGo =
      *reinterpret_cast<volatile int*>(groupSum + kPaddedRows);
  lds_barrier();
  if (thread == 0) {
    const bool last = atomicAdd(&counters[group], 1) == numSegments - 1;
    int go = last ? 2 : 0;
    if (last) {
      if (cooperative) {
        committedMask[(startGeneration + 1) & 1] = 0;
        // The generation first: the counter only has to be clean by the
        // next launch.
        __hip_atomic_store(generation, startGeneration + 1, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_AGENT);
      }
      counters[group] = 0;
    } else if (cooperative) {
      // Bounded: residency is the host's estimate, and a waiter spinning on a
      // CU the last segment needs would never see it arrive.
      for (int spin = 0; spin < kMaxSpins; ++spin) {
        if (__hip_atomic_load(generation, __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_AGENT) != startGeneration) {
          // Committed: the last arriver skips this slice.  A late commit
          // only makes both merge it, writing the same values.
          atomicOr(&committedMask[startGeneration & 1], (int)(1u << segment));
          go = 1;
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
    ldsGo = go;
  }
  lds_barrier();
  if (!ldsGo) return;
  __threadfence();
  const bool merger = ldsGo == 2;
  constexpr int kGroupElems = kGroupRows * kHeadDim;
  const int sliceElems =
      cooperative ? (kGroupElems / 4 + numSegments - 1) / numSegments * 4
                  : kGroupElems;
  // Own slice first; then the last arriver takes every uncommitted slice.
  unsigned pendingSlices = 0;
  for (int slice = cooperative ? segment : 0;;) {
    const int sliceBegin = slice * sliceElems;
    const int sliceEnd = min(kGroupElems, sliceBegin + sliceElems);
    // Every load goes out at once: one L2 round trip.
    for (int elem = sliceBegin + thread * 4; elem < sliceEnd;
         elem += kThreads * 4) {
      const int row = elem / kHeadDim, dim = elem % kHeadDim;
      float segMax[kMaxSegments], segSum[kMaxSegments];
      f4 segAcc[kMaxSegments];
  #pragma unroll
      for (int s = 0; s < kMaxSegments; ++s) {
        const size_t partialRow =
            ((size_t)group * kMaxSegments + s) * kPaddedRows + row;
        segMax[s] = s < numSegments ? partialMax[partialRow] : -INFINITY;
        segSum[s] = s < numSegments ? partialSum[partialRow] : 0.f;
        segAcc[s] = s < numSegments
                        ? *(const f4*)(partialAcc + partialRow * kHeadDim + dim)
                        : f4{};
      }
      float maxAll = -INFINITY;
  #pragma unroll
      for (int s = 0; s < kMaxSegments; ++s) maxAll = fmaxf(maxAll, segMax[s]);
      f4 numerator = {};
      float denominator = 0.f;
  #pragma unroll
      for (int s = 0; s < kMaxSegments; ++s) {
        const float weight = weight_of(segMax[s], maxAll);
        denominator = fmaf(weight, segSum[s], denominator);
        numerator += weight * segAcc[s];
      }
      const float inv = __builtin_amdgcn_rcpf(denominator);
      const int qHead = kvHead * kQHeadsPerKvHead +
                        (rowGroup * kGroupRows + row) / kMaxQueryLen;
      const int token = (rowGroup * kGroupRows + row) % kMaxQueryLen;
      OutT* dst = out + ((size_t)token * kNumQHeads + qHead) * kHeadDim + dim;
  #pragma unroll
      for (int k = 0; k < 4; ++k) dst[k] = to_elem(numerator[k] * inv);
    }
    if (!cooperative || !merger) break;
    if (slice == segment) {
      // Each thread reads the mask itself: any mix of views covers every
      // slice, since a committed owner merges its slice whole.
      const unsigned committed =
          __hip_atomic_load(&committedMask[startGeneration & 1],
                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      pendingSlices = ~committed &
                      ((numSegments < 32 ? 1u << numSegments : 0u) - 1) &
                      ~(1u << segment);
    }
    if (!pendingSlices) break;
    slice = __builtin_ctz(pendingSlices);
    pendingSlices &= pendingSlices - 1;
  }
}
#endif  // DOT_PRODUCT

// Per-sequence strides of a batch: grid.y is the sequence, and the pointers
// move to its rows before the body runs.
struct Batch {
  int blockTableStride, accStride, statStride, counterStride;
};

template <typename OutT>
__global__ __launch_bounds__(kThreads) void decode_attn(
    const elem_t* __restrict__ q, const elem_t* __restrict__ kv,
    const int* __restrict__ blockTable, float* __restrict__ partialAcc,
    float* __restrict__ partialMax, float* __restrict__ partialSum,
    int* __restrict__ counters, OutT* __restrict__ out,
    const int* __restrict__ seqLens, int blockTableWidth, float scale,
    int cooperative
#if BATCHED
    ,
    Batch batch
#endif
) {
  __shared__ __attribute__((aligned(16))) char lds[kLds];
#if BATCHED
  const int seq = blockIdx.y;
  constexpr int kSeqElems = kMaxQueryLen * kNumQHeads * kHeadDim;
  q += (size_t)seq * kSeqElems;
  out += (size_t)seq * kSeqElems;
  blockTable += (size_t)seq * batch.blockTableStride;
  partialAcc += (size_t)seq * batch.accStride;
  partialMax += (size_t)seq * batch.statStride;
  partialSum += (size_t)seq * batch.statStride;
  counters += (size_t)seq * batch.counterStride;
  seqLens += seq;
#endif
  // S and the first page indices go out together; every KV address depends
  // on the page table.
  const int seqLen = __builtin_amdgcn_readfirstlane(*seqLens);
  const int pageIndices = first_pages(blockTable, blockTableWidth);
  if constexpr (kBatched) {
    // A CUDA-graph batch padded past its real sequences.  Single-sequence
    // builds skip the check: waiting for S first costs 1-3 %.
    if (seqLen <= 0) return;
  }
  body<OutT>(q, kv, blockTable, partialAcc, partialMax, partialSum, counters,
             out, seqLen, pageIndices, blockTableWidth, scale, cooperative,
             lds);
}

// Arguments already checked by the registry (rdna35_decode_registry.cu).
static void launch(const rdna35::LaunchArgs& a) {
  // Row group fastest, then kv head, then segment: the row groups of a kv
  // head read the same KV side by side, sharing it in L2.
  dim3 grid(kGrid, a.nseq), threads(kThreads);
  static const int residentWorkgroups = [&] {
    if (!kSharedMerge) return 0;
    int device, numWgps, perWgp = 0;
    (void)hipGetDevice(&device);
    (void)hipDeviceGetAttribute(&numWgps, hipDeviceAttributeMultiprocessorCount,
                                device);
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &perWgp, decode_attn<elem_t>, kThreads, 0);
    // The runtime assumes 64 KiB of LDS where a gfx1151 WGP has 128, and
    // reports one workgroup per WGP above 32 KiB.  Count what fits: 1536
    // VGPRs per SIMD in blocks of 24, no workgroup spanning both CUs.
    hipFuncAttributes attrs;
    hipDeviceProp_t prop;
    if (hipFuncGetAttributes(&attrs, reinterpret_cast<const void*>(
                                         decode_attn<elem_t>)) == hipSuccess &&
        hipGetDeviceProperties(&prop, device) == hipSuccess &&
        strstr(prop.gcnArchName, "gfx1151")) {
      const int wavesPerSimd =
          std::min(16, 1536 / ((attrs.numRegs + 23) / 24 * 24));
      const int byWaves = 2 * (2 * wavesPerSimd / kWaves);
      const int byLds = attrs.sharedSizeBytes
                            ? 128 * 1024 / (int)attrs.sharedSizeBytes
                            : byWaves;
      perWgp = std::max(perWgp, std::min(byWaves, byLds));
    }
    return perWgp * numWgps;
  }();
  // The shared merge waits across workgroups: only when all of them fit.
  int cooperative = (int)(residentWorkgroups >= (int)(grid.x * grid.y));
#if BATCHED
  // Segments per sequence, capped so the batch lands near
  // kBatchTargetWorkgroups; the body reads the cap from the upper bits.
  const int groups = kGrid / kMaxSegments;
  cooperative |= std::max(1, (kBatchTargetWorkgroups + a.nseq * groups - 1) /
                                 (a.nseq * groups))
                 << 1;
  const Batch batch{a.bt_stride, a.acc_stride, a.ml_stride, a.cnt_stride};
#endif
  hipLaunchKernelGGL(decode_attn<elem_t>, grid, threads, 0, a.stream,
                     static_cast<const elem_t*>(a.q),
                     static_cast<const elem_t*>(a.kv), a.bt, a.acc, a.m, a.l,
                     a.cnt, static_cast<elem_t*>(a.out), a.seq_lens, a.bt_width,
                     a.scale, cooperative
#if BATCHED
                     ,
                     batch
#endif
  );
}
