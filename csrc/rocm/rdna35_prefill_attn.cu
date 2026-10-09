// RDNA3.5 prefill attention over vLLM's paged KV cache: one sequence, M query
// tokens after a cached prefix, causal (optionally windowed).
//
// The cache is the decode kernel's, HND with K and V packed in the content dim:
//
//   (num_pages, NUM_KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM)
//
// Every build is one row of vllm/v1/attention/ops/rdna35_prefill_variants.csv.
//
// FlashAttention-2 on WMMA 16x16x16.  One workgroup per (kv head, query tile
// of kBlockRows rows, KV segment); rows are token-major inside a kv head (row
// r: token r / G, q head kvHead * G + r % G).  Each wave owns 16 rows over
// 1/HEAD_DIM_SPLIT of the head dim and every KEY_WAVES-th 16-key sub-tile of a
// step; the step's KEY_TILE keys are staged in LDS for all of them.
//
//   S^T[key][row] = K[key][:] . Q[row][:]        A = K tile, B = Q^T
//   O^T[d][row]  += V[key][d] . P[row][key]      A = V^T,    B = P^T
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "rdna35_decode_attn.h"

#if !defined(HEAD_DIM) || !defined(NUM_Q_HEADS) || !defined(NUM_KV_HEADS) || \
    !defined(WINDOW) || !defined(BF16) || !defined(PAGE_SIZE)
  #error \
      "HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, WINDOW, BF16 and PAGE_SIZE come from the variant's row"
#endif
#if !defined(WAVES) || !defined(KEY_TILE) || !defined(HEAD_DIM_SPLIT) || \
    !defined(V_T) || !defined(DOUBLE_BUFFER) || !defined(KEY_WAVES) ||   \
    !defined(MAX_SEGMENTS) || !defined(HEAD_GROUP)
  #error \
      "WAVES, KEY_TILE, HEAD_DIM_SPLIT, V_T, DOUBLE_BUFFER, KEY_WAVES, MAX_SEGMENTS and HEAD_GROUP come from the variant's row"
#endif

// KEY_TILE: keys staged in LDS per step, a multiple of 16.
// HEAD_DIM_SPLIT: waves splitting a row tile's head dim; their Q@K partials
//   are summed through LDS.
// V_T: V staged transposed, so P@V reads its operand with no VALU.
// DOUBLE_BUFFER: two LDS buffers, one barrier per step.
// KEY_WAVES: waves sharing a row tile, merged at the end.
// MAX_SEGMENTS: most KV segments per tile when the tiles alone leave the GPU
//   idle; the host picks how many.
// HEAD_GROUP: kv heads whose tiles interleave in the grid.
constexpr int kHeadDim = HEAD_DIM;
constexpr int kNumQHeads = NUM_Q_HEADS;
constexpr int kNumKvHeads = NUM_KV_HEADS;
constexpr int kWindow = WINDOW;
constexpr bool kBf16 = BF16;
constexpr int kPageSize = PAGE_SIZE;
constexpr int kWaves = WAVES;
constexpr int kKeyTile = KEY_TILE;
constexpr int kDimParts = HEAD_DIM_SPLIT;
constexpr bool kVT = V_T;
constexpr bool kDoubleBuffer = DOUBLE_BUFFER;
constexpr int kKeyWaves = KEY_WAVES;
constexpr int kMaxSegments = MAX_SEGMENTS;

// Workgroups below which the host splits the KV, and the fewest steps of the
// longest tile a segment gets.  Split, groups * segments < 2 * the target:
// the scratch (make_prefill_scratch) holds that many row tiles.
constexpr int kTargetWorkgroups = 80;
constexpr int kMinSegmentSteps = 4;

constexpr int kWaveSize = 32;
constexpr int kThreads = kWaves * kWaveSize;
constexpr int kG = kNumQHeads / kNumKvHeads;
constexpr int kRowWaves = kWaves / (kDimParts * kKeyWaves);
constexpr int kBlockRows = kRowWaves * 16;
constexpr int kSubTiles = kKeyTile / 16;
constexpr int kWaveSubTiles = kSubTiles / kKeyWaves;
constexpr int kPartDims = kHeadDim / kDimParts;
constexpr int kPartChunks = kPartDims / 16;
// Dims of a V row per lane: lane l owns kLaneDims consecutive ones.
constexpr int kLaneDims = kPartDims / 16;
constexpr int kLaneWords = kLaneDims / 2;
constexpr int kKvRowElems = 2 * kHeadDim;
constexpr int kPageElems = kPageSize * kNumKvHeads * kKvRowElems;
// LDS rows padded by 16 bytes: the 16 rows one operand read touches sit on
// distinct banks.
constexpr int kRowPitch = kHeadDim + 8;
constexpr int kVtPitch = kKeyTile + 16;
constexpr int kBuffers = kDoubleBuffer ? 2 : 1;
constexpr int kRowChunks = kHeadDim / 8;
constexpr int kKUnits = kKeyTile * kRowChunks;
// With V_T a V unit is keys kk and kk + 2 of a sub-tile, whose dims pair into
// the dwords of adjacent slots.
constexpr int kVUnits = (kVT ? kKeyTile / 2 : kKeyTile) * kRowChunks;
constexpr int kKLoads = (kKUnits + kThreads - 1) / kThreads;
constexpr int kVLoads = (kVUnits + kThreads - 1) / kThreads;
constexpr int kVPerUnit = kVT ? 2 : 1;
// Pages a step spans.
constexpr int kStepPages = kKeyTile > kPageSize ? kKeyTile / kPageSize : 1;
constexpr float kLog2e = 1.44269504088896340736f;

// The largest divisor of NUM_KV_HEADS up to HEAD_GROUP.
constexpr int head_group() {
  int g = HEAD_GROUP < NUM_KV_HEADS ? HEAD_GROUP : NUM_KV_HEADS;
  while (NUM_KV_HEADS % g) --g;
  return g;
}
constexpr int kHeadGroup = head_group();

static_assert(kNumQHeads % kNumKvHeads == 0, "GQA must be integral");
static_assert(kWaves % (kDimParts * kKeyWaves) == 0, "whole row tiles");
static_assert(kHeadDim % (16 * kDimParts) == 0, "whole chunks per part");
static_assert(kSubTiles % kKeyWaves == 0, "whole sub-tiles per key wave");
static_assert(kKeyWaves == 1 || kDimParts == 1,
              "key waves without a head-dim split");
static_assert(kLaneDims % 2 == 0, "a lane's V slice is whole dwords");
static_assert(kKeyTile % 16 == 0 && kPageSize % 16 == 0,
              "a 16-key sub-tile sits inside one page");

#include "rdna35_attn_common.cuh"

// P as an element, rounded to nearest even.
__device__ __forceinline__ unsigned pack2(float a, float b) {
  return __builtin_bit_cast(unsigned, e2{to_elem(a), to_elem(b)});
}

// B = P^T: lane l (row l & 15) holds P for keys 2e + (l >> 4) in probs[e];
// k runs over this half's keys, then the other half's.
__device__ __forceinline__ e16 p_frag(const float* probs) {
  u8v packed;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    packed[i] = pack2(probs[2 * i], probs[2 * i + 1]);
    packed[4 + i] = xhalf_u(packed[i]);
  }
  return __builtin_bit_cast(e16, packed);
}

// kLaneDims consecutive dims of one V row.
struct VRow {
  unsigned w[kLaneWords];
};

// A = V^T for element `dim` of this lane's slice, keys in p_frag's order.
__device__ __forceinline__ e16 v_frag(const VRow* v, int dim) {
  const int word = dim >> 1;
  const unsigned selector = (dim & 1) ? 0x07060302u : 0x05040100u;
  u8v packed;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    packed[i] =
        __builtin_amdgcn_perm(v[2 * i + 1].w[word], v[2 * i].w[word], selector);
    packed[4 + i] = xhalf_u(packed[i]);
  }
  return __builtin_bit_cast(e16, packed);
}

// V_T: row (part, element j, lane row r) holds dim part * kPartDims +
// kLaneDims * r + j of every key; a sub-tile's keys sit even ones first, then
// odd ones, so each half-wave reads its p_frag order as two 16-byte pieces.
__device__ __forceinline__ int vt_row(int dim) {
  const int part = dim / kPartDims, local = dim % kPartDims;
  return part * kPartDims + (local % kLaneDims) * 16 + local / kLaneDims;
}

__device__ __forceinline__ int vt_slot(int key) {
  return (key & 1) * 8 + (key >> 1);
}

__global__ void __launch_bounds__(kThreads)
    prefill_attn(const elem_t* __restrict__ q, const elem_t* __restrict__ kv,
                 const int* __restrict__ blockTable, elem_t* __restrict__ out,
                 const int* __restrict__ seqLens, int numTokens, float scale,
                 float* __restrict__ partialO, float* __restrict__ partialML,
                 int* __restrict__ counters, int numSegments) {
  __shared__
      __attribute__((aligned(16))) elem_t kTile[kBuffers][kKeyTile][kRowPitch];
  __shared__ __attribute__((aligned(16))) elem_t
      vTile[kBuffers][kVT ? kHeadDim : kKeyTile][kVT ? kVtPitch : kRowPitch];
  __shared__ __attribute__((aligned(
      16))) float scoreX[kDimParts > 1 ? kWaves * kWaveSubTiles * 256 : 1];

  const int thread = threadIdx.x;
  const int lane = thread & (kWaveSize - 1);
  const int wave = __builtin_amdgcn_readfirstlane(thread / kWaveSize);
  const int laneRow = lane & 15;
  const int laneHalf = lane >> 4;
  const int dimPart = wave % kDimParts;
  const int keyWave = wave / kDimParts % kKeyWaves;
  const int rowWave = wave / (kDimParts * kKeyWaves);
  const int firstChunk = dimPart * kPartChunks;

  // A tile's segments are consecutive workgroups.  The last tokens' tiles see
  // the most keys and go out first, one tile of each of kHeadGroup kv heads at
  // a time: the resident workgroups balance and share a few heads' KV in L2.
  const int numRows = numTokens * kG;
  const int numTiles = (numRows + kBlockRows - 1) / kBlockRows;
  const int segment = (int)blockIdx.x % numSegments;
  const int block = (int)blockIdx.x / numSegments;
  const int groupBlocks = kHeadGroup * numTiles;
  const int within = block % groupBlocks;
  const int tile = numTiles - 1 - within / kHeadGroup;
  const int kvHead = block / groupBlocks * kHeadGroup + within % kHeadGroup;

  const int seqLen = __builtin_amdgcn_readfirstlane(*seqLens);
  const int context = seqLen - numTokens;
  const int tileRow = tile * kBlockRows;
  const int firstToken = tileRow / kG;
  const int lastToken = min(numTokens - 1, (tileRow + kBlockRows - 1) / kG);
  // Keys from the first token's window to the last token: the causal skip.
  const int keyEnd = context + lastToken + 1;
  const int windowFirst =
      kWindow ? max(0, context + firstToken - (kWindow - 1)) : 0;
  const int tileKeyBegin = windowFirst / kKeyTile * kKeyTile;
  // This segment's share of the tile's steps, possibly none.
  const int numSteps = (keyEnd - tileKeyBegin + kKeyTile - 1) / kKeyTile;
  const int keyBegin =
      tileKeyBegin + segment * numSteps / numSegments * kKeyTile;
  const int stepsEnd = min(
      keyEnd, tileKeyBegin + (segment + 1) * numSteps / numSegments * kKeyTile);
  const int lastPage = (seqLen - 1) / kPageSize;

  // This lane's row, clamped into the sequence.
  const int waveRow = tileRow + rowWave * 16;
  const int row = min(waveRow + laneRow, numRows - 1);
  const int rowToken = row / kG;
  const int rowHead = kvHead * kG + row % kG;
  const int waveFirstToken = waveRow / kG;
  const int waveLastToken = min(numTokens - 1, (waveRow + 15) / kG);

  // A step's page indices: lane l holds its page l % kStepPages, fetched a
  // step ahead so a KV load never waits for its table load.
  auto step_pages = [&](int keyStart) {
    return blockTable[min(keyStart / kPageSize + lane % kStepPages, lastPage)];
  };
  int pagesNext = step_pages(keyBegin);

  e16 qOperand[kPartChunks];
#pragma unroll
  for (int c = 0; c < kPartChunks; ++c)
    qOperand[c] =
        *(const e16*)(q + ((size_t)rowToken * kNumQHeads + rowHead) * kHeadDim +
                      (firstChunk + c) * 16);

  const elem_t* kvHeadBase = kv + (size_t)kvHead * (kPageSize * kKvRowElems);
  e8 kStaged[kKLoads];
  e8 vStaged[kVLoads][kVPerUnit];
  auto v_unit_key = [&](int unit) {
    const int pair = unit / kRowChunks;
    if constexpr (kVT)
      return pair / 8 * 16 + (pair % 8 >> 1) * 4 + (pair & 1);
    else
      return pair;
  };
  auto load_row = [&](int pages, int keyStart, int key, int col) {
    const int position = min(keyStart + key, seqLen - 1);
    const int page = __shfl(pages, position / kPageSize - keyStart / kPageSize);
    return *(const e8*)(kvHeadBase + (size_t)page * kPageElems +
                        (position % kPageSize) * kKvRowElems + col);
  };
  // Issue the loads of the step at keyStart into registers.
  auto issue = [&](int keyStart) {
    const int pages = pagesNext;
    pagesNext = step_pages(keyStart + kKeyTile);
#pragma unroll
    for (int i = 0; i < kKLoads; ++i) {
      const int unit = i * kThreads + thread;
      if (kKUnits % kThreads == 0 || unit < kKUnits)
        kStaged[i] =
            load_row(pages, keyStart, unit / kRowChunks, unit % kRowChunks * 8);
    }
#pragma unroll
    for (int i = 0; i < kVLoads; ++i) {
      const int unit = i * kThreads + thread;
      if (kVUnits % kThreads == 0 || unit < kVUnits) {
        const int key = v_unit_key(unit), col = unit % kRowChunks * 8;
#pragma unroll
        for (int x = 0; x < kVPerUnit; ++x)
          vStaged[i][x] =
              load_row(pages, keyStart, key + 2 * x, kHeadDim + col);
      }
    }
  };
  // Keys past S, or before every row's window, may sit on freed pages: zeros,
  // so no NaN reaches P@V as 0 * NaN.
  auto valid = [&](int position) {
    return position < seqLen && position >= windowFirst;
  };
  // Store the issued step to LDS buffer `buffer`.
  auto stage = [&](int buffer, int keyStart) {
#pragma unroll
    for (int i = 0; i < kKLoads; ++i) {
      const int unit = i * kThreads + thread;
      if (kKUnits % kThreads == 0 || unit < kKUnits) {
        const int key = unit / kRowChunks, col = unit % kRowChunks * 8;
        *(e8*)&kTile[buffer][key][col] =
            valid(keyStart + key) ? kStaged[i] : e8{};
      }
    }
#pragma unroll
    for (int i = 0; i < kVLoads; ++i) {
      const int unit = i * kThreads + thread;
      if (kVUnits % kThreads == 0 || unit < kVUnits) {
        const int key = v_unit_key(unit), col = unit % kRowChunks * 8;
        if constexpr (kVT) {
          const e8 a = valid(keyStart + key) ? vStaged[i][0] : e8{};
          const e8 b = valid(keyStart + key + 2) ? vStaged[i][1] : e8{};
          const int slot = key / 16 * 16 + vt_slot(key % 16);
#pragma unroll
          for (int d = 0; d < 8; ++d)
            *(e2*)&vTile[buffer][vt_row(col + d)][slot] = e2{a[d], b[d]};
        } else {
          *(e8*)&vTile[buffer][key][col] =
              valid(keyStart + key) ? vStaged[i][0] : e8{};
        }
      }
    }
  };

  const float scaleLog2 = scale * kLog2e;
  float runMax = -INFINITY, runSum = 0.f;
  f8 accum[kLaneDims];
#pragma unroll
  for (int j = 0; j < kLaneDims; ++j) accum[j] = f8{};

  // Q@K, the softmax update and P@V for the step at keyStart, staged in
  // `buffer`.  Lane holds row laneRow, keys keyStart + 16 st + 2e + laneHalf.
  auto compute = [&](int buffer, int keyStart) {
    // A sub-tile no row of the wave sees (past its last token, or below its
    // first token's window) skips both products.
    bool live[kWaveSubTiles];
#pragma unroll
    for (int ls = 0, st = keyWave; ls < kWaveSubTiles; ++ls, st += kKeyWaves) {
      const int first = keyStart + st * 16;
      live[ls] =
          first <= context + waveLastToken &&
          (!kWindow || first + 15 >= context + waveFirstToken - (kWindow - 1));
    }
    f8 scores[kWaveSubTiles];
#pragma unroll
    for (int ls = 0; ls < kWaveSubTiles; ++ls) scores[ls] = f8{};
    // Chunks outer: consecutive WMMAs feed independent accumulators.
#pragma unroll
    for (int c = 0; c < kPartChunks; ++c)
#pragma unroll
      for (int ls = 0, st = keyWave; ls < kWaveSubTiles;
           ++ls, st += kKeyWaves) {
        const e16 kOperand = *(
            const e16*)&kTile[buffer][st * 16 + laneRow][(firstChunk + c) * 16];
        if (live[ls]) scores[ls] = wmma(kOperand, qOperand[c], scores[ls]);
      }
    if constexpr (kDimParts > 1) {
      // Sum the parts' scores in the same order in every wave, so they all
      // agree bit for bit on the softmax; with two parts a + b == b + a.
#pragma unroll
      for (int ls = 0; ls < kWaveSubTiles; ++ls)
        *(f8*)&scoreX[(wave * kWaveSubTiles + ls) * 256 + lane * 8] =
            scores[ls];
      lds_barrier();
#pragma unroll
      for (int ls = 0; ls < kWaveSubTiles; ++ls) {
        if constexpr (kDimParts == 2) {
          scores[ls] +=
              *(const f8*)&scoreX[((wave ^ 1) * kWaveSubTiles + ls) * 256 +
                                  lane * 8];
        } else {
          f8 sum = {};
#pragma unroll
          for (int part = 0; part < kDimParts; ++part)
            sum += *(const f8*)&scoreX
                       [((wave - dimPart + part) * kWaveSubTiles + ls) * 256 +
                        lane * 8];
          scores[ls] = sum;
        }
      }
    }

    // Masks only on a tile's diagonal steps and below its window.
    const bool causalEdge = keyStart + kKeyTile - 1 > context + firstToken;
    const bool windowEdge =
        kWindow && keyStart < context + lastToken - (kWindow - 1);
    if (__builtin_expect(causalEdge || windowEdge, 0)) {
#pragma unroll
      for (int ls = 0, st = keyWave; ls < kWaveSubTiles; ++ls, st += kKeyWaves)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int key = keyStart + st * 16 + 2 * e + laneHalf;
          if (key > context + rowToken) scores[ls][e] = -INFINITY;
          if constexpr (kWindow)
            if (key < context + rowToken - (kWindow - 1))
              scores[ls][e] = -INFINITY;
        }
    }
    float tileMax = -INFINITY;
#pragma unroll
    for (int ls = 0; ls < kWaveSubTiles; ++ls)
#pragma unroll
      for (int e = 0; e < 8; ++e) tileMax = fmaxf(tileMax, scores[ls][e]);
    tileMax = fmaxf(tileMax, xhalf(tileMax)) * scaleLog2;
    // A row with no visible key yet keeps max -inf: exponents against 0 give
    // P = 0, and a rescale of 0 of an all-zero O.
    const float newMax = fmaxf(runMax, tileMax);
    const float safeMax = newMax == -INFINITY ? 0.f : newMax;
    const float rescale = __builtin_amdgcn_exp2f(runMax - safeMax);
    runMax = newMax;
    // Most steps leave every row's max where it was.
    if (__builtin_amdgcn_ballot_w32(rescale != 1.f))
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) accum[j] *= rescale;
    e16 pOperand[kWaveSubTiles];
    float tileSum = 0.f;
#pragma unroll
    for (int ls = 0; ls < kWaveSubTiles; ++ls) {
      float probs[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        probs[e] = __builtin_amdgcn_exp2f(
            __builtin_fmaf(scores[ls][e], scaleLog2, -safeMax));
        tileSum += probs[e];
      }
      pOperand[ls] = p_frag(probs);
    }
    runSum = runSum * rescale + tileSum + xhalf(tileSum);

#pragma unroll
    for (int ls = 0, st = keyWave; ls < kWaveSubTiles; ++ls, st += kKeyWaves) {
      VRow v[8];
      if constexpr (!kVT) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const unsigned* src =
              (const unsigned*)&vTile[buffer][st * 16 + 2 * e + laneHalf]
                                     [dimPart * kPartDims +
                                      kLaneDims * laneRow];
#pragma unroll
          for (int w = 0; w < kLaneWords; ++w) v[e].w[w] = src[w];
        }
      }
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) {
        e16 vOperand;
        if constexpr (kVT) {
          const elem_t* vRow =
              &vTile[buffer][dimPart * kPartDims + j * 16 + laneRow][st * 16];
          const e8 own = *(const e8*)(vRow + laneHalf * 8);
          const e8 other = *(const e8*)(vRow + (1 - laneHalf) * 8);
          vOperand = __builtin_shufflevector(own, other, 0, 1, 2, 3, 4, 5, 6, 7,
                                             8, 9, 10, 11, 12, 13, 14, 15);
        } else {
          vOperand = v_frag(v, j);
        }
        if (live[ls]) accum[j] = wmma(vOperand, pOperand[ls], accum[j]);
      }
    }
  };

  if (keyBegin < stepsEnd) {
    if constexpr (kDoubleBuffer) {
      issue(keyBegin);
      stage(0, keyBegin);
      if (keyBegin + kKeyTile < stepsEnd) issue(keyBegin + kKeyTile);
      lds_barrier();
      int buffer = 0;
      for (int keyStart = keyBegin; keyStart < stepsEnd; keyStart += kKeyTile) {
        compute(buffer, keyStart);
        if (keyStart + kKeyTile < stepsEnd) {
          stage(buffer ^ 1, keyStart + kKeyTile);
          if (keyStart + 2 * kKeyTile < stepsEnd)
            issue(keyStart + 2 * kKeyTile);
        }
        lds_barrier();
        buffer ^= 1;
      }
    } else {
      issue(keyBegin);
      for (int keyStart = keyBegin; keyStart < stepsEnd; keyStart += kKeyTile) {
        lds_barrier();
        stage(0, keyStart);
        lds_barrier();
        if (keyStart + kKeyTile < stepsEnd) issue(keyStart + kKeyTile);
        compute(0, keyStart);
      }
    }
  }

  if constexpr (kKeyWaves > 1) {
    // Merge the key waves: each rescales to the row's max over them; key wave
    // 0 sums their O and carries on alone.
    __shared__ float keyMax[kKeyWaves][kBlockRows],
        keySum[kKeyWaves][kBlockRows];
    __shared__ __attribute__((aligned(16))) float
        keyO[kKeyWaves > 1 ? kKeyWaves - 1 : 1][kBlockRows][kHeadDim];
    const int localRow = rowWave * 16 + laneRow;
    __syncthreads();
    if (laneHalf == 0) {
      keyMax[keyWave][localRow] = runMax;
      keySum[keyWave][localRow] = runSum;
    }
    __syncthreads();
    float maxAll = -INFINITY;
#pragma unroll
    for (int k = 0; k < kKeyWaves; ++k)
      maxAll = fmaxf(maxAll, keyMax[k][localRow]);
    float total = 0.f;
#pragma unroll
    for (int k = 0; k < kKeyWaves; ++k)
      total += weight_of(keyMax[k][localRow], maxAll) * keySum[k][localRow];
    const float weight = weight_of(runMax, maxAll);
#pragma unroll
    for (int j = 0; j < kLaneDims; ++j) accum[j] *= weight;
    runMax = maxAll;
    runSum = total;
    if (keyWave > 0)
#pragma unroll
      for (int e = 0; e < 8; ++e)
#pragma unroll
        for (int j = 0; j < kLaneDims; ++j)
          keyO[keyWave - 1][localRow][kLaneDims * (2 * e + laneHalf) + j] =
              accum[j][e];
    __syncthreads();
    if (keyWave == 0)
#pragma unroll
      for (int k = 1; k < kKeyWaves; ++k)
#pragma unroll
        for (int e = 0; e < 8; ++e)
#pragma unroll
          for (int j = 0; j < kLaneDims; ++j)
            accum[j][e] +=
                keyO[k - 1][localRow][kLaneDims * (2 * e + laneHalf) + j];
  }
  const bool owner = kKeyWaves == 1 || keyWave == 0;

  if (numSegments == 1) {
    if (!owner || waveRow + laneRow >= numRows) return;
    const float inv = __builtin_amdgcn_rcpf(runSum);
    elem_t* dst = out + ((size_t)rowToken * kNumQHeads + rowHead) * kHeadDim +
                  dimPart * kPartDims;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      elem_t x[kLaneDims];
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) x[j] = to_elem(accum[j][e] * inv);
      unsigned* d = (unsigned*)(dst + kLaneDims * (2 * e + laneHalf));
#pragma unroll
      for (int w = 0; w < kLaneWords; ++w)
        d[w] = __builtin_bit_cast(unsigned, e2{x[2 * w], x[2 * w + 1]});
    }
    return;
  }

  // Split KV: publish (O, max, sum) per row; the last segment to arrive
  // merges and resets the counter.
  const size_t partialBase =
      ((size_t)block * numSegments + segment) * kBlockRows;
  if (owner) {
    const size_t partialRow = partialBase + rowWave * 16 + laneRow;
    float* dst = partialO + partialRow * kHeadDim + dimPart * kPartDims;
#pragma unroll
    for (int e = 0; e < 8; ++e)
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j)
        dst[kLaneDims * (2 * e + laneHalf) + j] = accum[j][e];
    if (laneHalf == 0 && dimPart == 0) {
      partialML[partialRow * 2] = runMax;
      partialML[partialRow * 2 + 1] = runSum;
    }
  }
  __threadfence();
  __shared__ int isLast;
  __syncthreads();
  if (thread == 0) {
    isLast = atomicAdd(&counters[block], 1) == numSegments - 1;
    if (isLast) counters[block] = 0;
  }
  __syncthreads();
  if (!isLast) return;
  __threadfence();
  for (int elem = thread * 4; elem < kBlockRows * kHeadDim;
       elem += kThreads * 4) {
    const int r = elem / kHeadDim, d = elem % kHeadDim;
    const int outRow = tileRow + r;
    if (outRow >= numRows) continue;
    const size_t rowBase = (size_t)block * numSegments * kBlockRows + r;
    float maxAll = -INFINITY;
    for (int sg = 0; sg < numSegments; ++sg)
      maxAll =
          fmaxf(maxAll, partialML[(rowBase + (size_t)sg * kBlockRows) * 2]);
    f4 numerator = {};
    float denominator = 0.f;
    for (int sg = 0; sg < numSegments; ++sg) {
      const size_t pr = rowBase + (size_t)sg * kBlockRows;
      const float weight = weight_of(partialML[pr * 2], maxAll);
      denominator = fmaf(weight, partialML[pr * 2 + 1], denominator);
      numerator += weight * *(const f4*)(partialO + pr * kHeadDim + d);
    }
    const float inv = __builtin_amdgcn_rcpf(denominator);
    elem_t* dst =
        out +
        ((size_t)(outRow / kG) * kNumQHeads + kvHead * kG + outRow % kG) *
            kHeadDim +
        d;
#pragma unroll
    for (int k = 0; k < 4; ++k) dst[k] = to_elem(numerator[k] * inv);
  }
}

// Arguments already checked by the registry (rdna35_decode_registry.cu).
// Segments per tile: enough workgroups to fill the GPU, each at least
// kMinSegmentSteps steps of the longest tile.
static void launch(const rdna35::PrefillArgs& a) {
  const int numTiles = (a.num_tokens * kG + kBlockRows - 1) / kBlockRows;
  const int groups = numTiles * kNumKvHeads;
  int segments = 1;
  if (kMaxSegments > 1 && groups < kTargetWorkgroups) {
    const int steps = (a.max_seq_len + kKeyTile - 1) / kKeyTile;
    segments = std::min({(kTargetWorkgroups + groups - 1) / groups,
                         kMaxSegments, std::max(1, steps / kMinSegmentSteps)});
  }
  hipLaunchKernelGGL(prefill_attn, dim3(groups * segments), dim3(kThreads), 0,
                     a.stream, static_cast<const elem_t*>(a.q),
                     static_cast<const elem_t*>(a.kv), a.bt,
                     static_cast<elem_t*>(a.out), a.seq_lens, a.num_tokens,
                     a.scale, a.partial_o, a.partial_ml, a.counters, segments);
}
