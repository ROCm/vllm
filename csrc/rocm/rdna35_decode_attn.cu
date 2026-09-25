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
// products are WMMA 16x16x16 fp16 (or bf16, with KV_BF16=1) -> f32:
//
//   S^T[key][row] = K[key][:] . Q[row][:]        A = K tile, B = Q^T
//   O^T[d][row]  += V[key][d] . P[row][key]      A = V^T,    B = P^T
#include <hip/hip_runtime.h>

#include <cmath>

#ifndef RDNA35_BODY
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
  // Element type of Q, the KV cache and the output: 0 = fp16, 1 = bf16.
  #ifndef KV_BF16
    #define KV_BF16 0
  #endif
  // Most KV segments a (kv head, row group) is split over.  The number active
  // is clamp(nblocks / MINB, 1, NSEG), decided at run time from S: the grid is
  // fixed when a CUDA graph is captured, S is not -- the kernel reads it from
  // the device, since a host argument would be frozen into the graph too.
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
  // Waves sharing one key tile, each carrying 1/RSPL of the workgroup's row
  // tiles over its whole d part.  The in-workgroup form of RG: the waves read
  // the same tile at nearly the same time, so its KV comes from DRAM once
  // without depending on how separate workgroups drift.
  #ifndef RSPL
    #define RSPL 1
  #endif

  // A second decomposition for long sequences: at S >= SW the kernel runs
  // mode B with the knobs below instead of mode A's.  The grid and the block
  // are fixed at graph capture, S is not, so the choice is made on the device;
  // short and long sequences want different splits of the same configuration
  // (row groups and no merge, against rows split inside the workgroup and many
  // segments).  SW 0 builds mode A alone.
  #ifndef SW
    #define SW 0
  #endif
  #ifndef NSEG2
    #define NSEG2 NSEG
  #endif
  #ifndef RG2
    #define RG2 RG
  #endif
  #ifndef MINB2
    #define MINB2 MINB
  #endif
  #ifndef DSPL2
    #define DSPL2 DSPL
  #endif
  #ifndef RSPL2
    #define RSPL2 RSPL
  #endif
  #ifndef NW2
    #define NW2 NW
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
  // Launched with the larger mode's waves.
  #define BLOCK ((SW && NW2 > NW ? NW2 : NW) * WAVE)
  #define GQA (NUM_Q_HEADS / NUM_KV_HEADS)
  #define ROWS (GQA * MAXM)
  #define NCHUNK (HEAD_DIM / 16)
  #define KV_ROW (2 * HEAD_DIM)
  #define PAGE_ELEMS (BS * NUM_KV_HEADS * KV_ROW)
  #if LAYOUT == 0
    #define KEY_STRIDE (NUM_KV_HEADS * KV_ROW)
  #else
    #define KEY_STRIDE KV_ROW
  #endif
  #define LOG2E 1.44269504088896340736f
  #define QPAD 8
  // Merge buffer: one slot per (live tile, d part) holds the real rows of a row
  // tile, each DPART floats plus a pad.
  #define PADM 4
  #define MBUDGET (36 * 1024)

static_assert(NUM_Q_HEADS % NUM_KV_HEADS == 0, "GQA must be integral");
static_assert(BS % 16 == 0, "a 16-key tile must sit inside one page");

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

  #if KV_BF16
typedef __bf16 elem_t;
  #else
typedef _Float16 elem_t;
  #endif
typedef elem_t e16 __attribute__((ext_vector_type(16)));
typedef elem_t e8 __attribute__((ext_vector_type(8)));
typedef elem_t e2 __attribute__((ext_vector_type(2)));
typedef short s16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef unsigned u8v __attribute__((ext_vector_type(8)));
typedef unsigned u4v __attribute__((ext_vector_type(4)));
typedef unsigned u2v __attribute__((ext_vector_type(2)));

// Element offset of (page, slot) for this kv head.
__device__ __forceinline__ size_t tile_off(int page, unsigned slot, int kvh) {
  #if LAYOUT == 0
  return (size_t)page * PAGE_ELEMS +
         ((size_t)slot * NUM_KV_HEADS + kvh) * KV_ROW;
  #else
  return (size_t)page * PAGE_ELEMS + ((size_t)kvh * BS + slot) * KV_ROW;
  #endif
}

// Every barrier in this kernel orders LDS only; global visibility, where the
// split-KV merge needs it, is __threadfence's job.  __syncthreads() also
// waits for every outstanding global load (vmcnt(0)) and invalidates L0, so
// the barrier that publishes Q waited for the whole first KV tile to land.
__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// The cross-lane moves below all read lanes that are active, so fetch-
// inactive changes nothing -- except that with it set the compiler no longer
// ties the destination to a copy of the source: 66 of 207 v_mov gone.
//
// The other 16-lane half's value for this lane: lane l <-> l ^ 16.
__device__ __forceinline__ unsigned xhalf_u(unsigned v) {
  return (unsigned)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210u,
                                                0xFEDCBA98u, true, false);
}

__device__ __forceinline__ float xhalf(float v) {
  return __builtin_bit_cast(float, xhalf_u(__builtin_bit_cast(unsigned, v)));
}

// Within each 16-lane row: lane i takes lane i & 7's value, or lane i | 8's.
__device__ __forceinline__ unsigned lower8(unsigned v) {
  return (unsigned)__builtin_amdgcn_permlane16((int)v, (int)v, 0x76543210u,
                                               0x76543210u, true, false);
}

__device__ __forceinline__ float upper8f(float v) {
  return __builtin_bit_cast(
      float, __builtin_amdgcn_permlane16(
                 __builtin_bit_cast(int, v), __builtin_bit_cast(int, v),
                 0xFEDCBA98u, 0xFEDCBA98u, true, false));
}

__device__ __forceinline__ float weight_of(float m, float gmax) {
  return (gmax == -INFINITY) ? 0.f : __builtin_amdgcn_exp2f(m - gmax);
}

// P's high half as an element, and as the float it stands for.  bf16
// truncates: gfx1151 has no f32 -> bf16 conversion, so rounding would be
// several VALU ops per value where truncation packs two in one v_perm, and
// the low half carries whatever truncation drops.
__device__ __forceinline__ unsigned pack2(float a, float b) {
  #if KV_BF16
  return __builtin_amdgcn_perm(__builtin_bit_cast(unsigned, b),
                               __builtin_bit_cast(unsigned, a), 0x07060302u);
  #else
  return __builtin_bit_cast(unsigned, e2{(elem_t)a, (elem_t)b});
  #endif
}

__device__ __forceinline__ float elem_part(float p) {
  #if KV_BF16
  return __builtin_bit_cast(float,
                            __builtin_bit_cast(unsigned, p) & 0xFFFF0000u);
  #else
  return (float)(elem_t)p;
  #endif
}

// An output value, rounded to nearest even.  The bf16 cast also keeps NaN a
// NaN, which costs a compare and a select per value; here a NaN may come out
// as Inf, still not finite.
__device__ __forceinline__ elem_t to_elem(float x) {
  #if KV_BF16
  const unsigned u = __builtin_bit_cast(unsigned, x);
  return __builtin_bit_cast(
      elem_t, (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16));
  #else
  return (elem_t)x;
  #endif
}

__device__ __forceinline__ f8 wmma(e16 a, e16 b, f8 c) {
  #if KV_BF16
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
      __builtin_bit_cast(s16, a), __builtin_bit_cast(s16, b), c);
  #else
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  #endif
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
__device__ __forceinline__ e16 p_frag(const float* p) {
  u8v r;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    r[i] = pack2(p[2 * i], p[2 * i + 1]);
    r[4 + i] = xhalf_u(r[i]);
  }
  return __builtin_bit_cast(e16, r);
}

  #define RDNA35_BODY 1
  #define M_NSEG NSEG
  #define M_RG RG
  #define M_MINB MINB
  #define M_DSPL DSPL
  #define M_RSPL RSPL
  #define M_NW NW
namespace mode_a {
  #include __FILE_NAME__
}  // namespace mode_a
  #undef M_NSEG
  #undef M_RG
  #undef M_MINB
  #undef M_DSPL
  #undef M_RSPL
  #undef M_NW
  #if SW
    #define M_NSEG NSEG2
    #define M_RG RG2
    #define M_MINB MINB2
    #define M_DSPL DSPL2
    #define M_RSPL RSPL2
    #define M_NW NW2
namespace mode_b {
    #include __FILE_NAME__
}  // namespace mode_b
    #undef M_NSEG
    #undef M_RG
    #undef M_MINB
    #undef M_DSPL
    #undef M_RSPL
    #undef M_NW
  #else
namespace mode_b = mode_a;
  #endif

constexpr int kLdsMax =
    mode_a::kLds > mode_b::kLds ? mode_a::kLds : mode_b::kLds;
constexpr int kGridMax =
    mode_a::kGrid > mode_b::kGrid ? mode_a::kGrid : mode_b::kGrid;

template <typename OutT>
__global__ __launch_bounds__(BLOCK) void decode_attn(
    const elem_t* __restrict__ q, const elem_t* __restrict__ kv,
    const int* __restrict__ bt, float* __restrict__ p_acc,
    float* __restrict__ p_m, float* __restrict__ p_l, int* __restrict__ p_cnt,
    OutT* __restrict__ out, const int* __restrict__ seq_lens, int bt_width,
    float scale, int coop) {
  __shared__ __attribute__((aligned(16))) char lds[kLdsMax];
  // S and the first page indices (both modes', with two) go out together:
  // neither waits for the other.  Every KV address depends on the page table.
  const int S = __builtin_amdgcn_readfirstlane(*seq_lens);
  const int pa = mode_a::first_pages(bt, bt_width);
  #if SW
  const int pb = mode_b::first_pages(bt, bt_width);
  // Mode B, the long-sequence one, falls through: whichever body sits inside
  // the branch is compiled worse (measured 2-4 % on mode A there, 15 % on
  // mode B).
  if (S < SW) {
    mode_a::body<OutT>(q, kv, bt, p_acc, p_m, p_l, p_cnt, out, S, pa, bt_width,
                       scale, coop, lds);
    return;
  }
  mode_b::body<OutT>(q, kv, bt, p_acc, p_m, p_l, p_cnt + mode_a::kCnt, out, S,
                     pb, bt_width, scale, coop, lds);
  #else
  mode_a::body<OutT>(q, kv, bt, p_acc, p_m, p_l, p_cnt, out, S, pa, bt_width,
                     scale, coop, lds);
  #endif
}

  #ifndef RDNA35_TORCH_EXT
// Instantiated for ISA inspection when built without the torch op.
template __global__ void decode_attn<elem_t>(const elem_t*, const elem_t*,
                                             const int*, float*, float*, float*,
                                             int*, elem_t*, const int*, int,
                                             float, int);
  #else
    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDAGuard.h>
    #include <torch/extension.h>

// The scratch buffers (acc/m/l/cnt) are caller-allocated on purpose: this runs
// inside a CUDA-graph capture, and an allocation there would break it.
void decode_attn_op(torch::Tensor& q, torch::Tensor& kv_cache,
                    torch::Tensor& block_table, torch::Tensor& out,
                    torch::Tensor& acc, torch::Tensor& m, torch::Tensor& l,
                    torch::Tensor& cnt, torch::Tensor& seq_lens, double scale) {
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous(),
              "q and out must be contiguous");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 &&
                  block_table.dim() == 1 && block_table.is_contiguous(),
              "block table must be one contiguous int32 row");
  // A tensor, not an int: under CUDA-graph capture a host argument is frozen
  // at its capture-time value, and every replay would attend over that many
  // keys whatever the sequence has grown to.
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 && seq_lens.numel() >= 1,
              "seq_lens must be an int32 tensor holding the sequence length");
  TORCH_CHECK(q.size(0) == MAXM, "q has ", q.size(0), " tokens, kernel built ",
              "for MAXM=", MAXM);
  TORCH_CHECK(q.size(1) == NUM_Q_HEADS && q.size(2) == HEAD_DIM,
              "q shape does not match the compiled variant");
  // The loads reinterpret every tensor as the element type the variant was
  // built for.  The other 16-bit type would read the same bits and return
  // finite nonsense, so refuse it here rather than downstream.
  constexpr auto dtype = KV_BF16 ? at::kBFloat16 : at::kHalf;
  TORCH_CHECK(q.scalar_type() == dtype && kv_cache.scalar_type() == dtype &&
                  out.scalar_type() == dtype,
              "kernel built for ", dtype, ", got q=", q.scalar_type(),
              " kv_cache=", kv_cache.scalar_type(), " out=", out.scalar_type());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // Row group fastest, then kv head, then segment: the row groups of one kv
  // head read the same KV and are dispatched side by side to share it in L2.
  dim3 grid(kGridMax), block(BLOCK);
  static const int coop = [&] {
    if (!mode_a::kSharedMerge && !mode_b::kSharedMerge) return 0;
    int dev, wgps, per = 0;
    (void)hipGetDevice(&dev);
    (void)hipDeviceGetAttribute(&wgps, hipDeviceAttributeMultiprocessorCount,
                                dev);
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &per, decode_attn<elem_t>, BLOCK, 0);
    // The runtime assumes 64 KiB of LDS where a gfx1151 WGP has 128, and
    // reports one workgroup per WGP for anything over 32 KiB.  Count what
    // fits, rounding every way down: 1536 VGPRs per SIMD in blocks of 24,
    // and no workgroup spanning the WGP's two CUs.
    hipFuncAttributes fa;
    hipDeviceProp_t prop;
    if (hipFuncGetAttributes(&fa, reinterpret_cast<const void*>(
                                      decode_attn<elem_t>)) == hipSuccess &&
        hipGetDeviceProperties(&prop, dev) == hipSuccess &&
        strstr(prop.gcnArchName, "gfx1151")) {
      const int wps = std::min(16, 1536 / ((fa.numRegs + 23) / 24 * 24));
      const int by_waves = 2 * (2 * wps / (BLOCK / WAVE));
      const int by_lds =
          fa.sharedSizeBytes ? 128 * 1024 / (int)fa.sharedSizeBytes : by_waves;
      per = std::max(per, std::min(by_waves, by_lds));
    }
    return (int)(per * wgps >= (int)grid.x);
  }();
  hipLaunchKernelGGL(
      decode_attn<elem_t>, grid, block, 0, stream,
      reinterpret_cast<const elem_t*>(q.data_ptr()),
      reinterpret_cast<const elem_t*>(kv_cache.data_ptr()),
      block_table.data_ptr<int>(), acc.data_ptr<float>(), m.data_ptr<float>(),
      l.data_ptr<float>(), cnt.data_ptr<int>(),
      reinterpret_cast<elem_t*>(out.data_ptr()), seq_lens.data_ptr<int>(),
      (int)block_table.size(0), (float)scale, coop);
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

#else
  // ---------------------------------------------------------------- body ----
  // Everything below depends on the decomposition knobs, read through their
  // M_ names; the outer pass instantiates it once per mode.
  #define ROWS_W (ROWS / M_RG)
  #define RTILES ((ROWS_W + 15) / 16)
  #define ROWPAD (RTILES * 16)
  #define TILES (M_NW / (M_DSPL * M_RSPL))
  // Row tiles per wave; wave rs of a tile carries row tiles rs, rs + M_RSPL,
  // ...
  #define RTW (RTILES / M_RSPL)
  // With M_RSPL > 1 the tile is shared: each of its M_RSPL waves loads EPW of
  // the eight key pairs, stages them in LDS, and all read the whole tile back.
  #define SHT (M_RSPL > 1)
  #define EPW (8 / M_RSPL)
  #define KBLK (16 * TILES)
  #define DPART (HEAD_DIM / M_DSPL)
  // d per lane of a K or V row: lane l16 owns DPART/16 consecutive d.
  #define VD (DPART / 16)
  #define NCP (NCHUNK / M_DSPL)
  // P reaches P@V as a high plus a low half, ~22 bits rather than 11 in fp16
  // and ~16 rather than 8 in bf16: with one fp16 P the output misses the 1e-3
  // relative bound wherever it is near zero.  When a row tile has at most 8
  // real rows the low half rides in the padding columns of the same WMMA;
  // otherwise it takes a second one.
  #define PPACK (ROWS_W <= 8)

  // Per-wave K tile: 16 keys of DPART halves.  K is read row-wise, like V, and
  // transposed to the lane-per-key WMMA operand through it; the 16-byte pad
  // puts the sixteen keys one operand read touches on distinct banks.
  #define KT_ROW (DPART * 2 + 16)
  #define KT_BYTES (M_NW * 16 * KT_ROW)
  #define QS_BYTES (ROWPAD * (HEAD_DIM + QPAD) * 2)
  #define MROW (DPART + PADM)
  #define RV (ROWS_W < 16 ? ROWS_W : 16)
  #define MSLOT (RV * MROW)
  #define SLOTS_FIT(t) ((t) * M_DSPL * M_RSPL * MSLOT * 4 <= MBUDGET)
  // Live tiles after the merge's tree rounds: the most that fit the budget.
  #define TFIN                          \
    (SLOTS_FIT(TILES)       ? TILES     \
     : SLOTS_FIT(TILES / 2) ? TILES / 2 \
     : SLOTS_FIT(TILES / 4) ? TILES / 4 \
     : SLOTS_FIT(TILES / 8) ? TILES / 8 \
                            : 1)
  #define MRG_SLOTS \
    (TFIN == TILES ? TILES : (TFIN > TILES / 2 ? TFIN : TILES / 2))
  #define MRG_BYTES \
    (TILES == 1 && M_RSPL == 1 ? 0 : MRG_SLOTS * M_DSPL * M_RSPL * MSLOT * 4)
  #if SHT
    // Per (tile, d part) a K and a V tile, double-buffered, and the score
    // exchange apart from them: partners may still read the shared K.
    #define TB_BYTES (2 * TILES * M_DSPL * 2 * 16 * KT_ROW)
    #define SX_BYTES (M_DSPL > 1 ? M_NW * RTW * 1024 : 0)
    #define LOOP_BYTES (QS_BYTES + TB_BYTES + SX_BYTES)
  #else
    #define LOOP_BYTES (QS_BYTES + KT_BYTES)
  #endif
  #define LDS_RAW (LOOP_BYTES > MRG_BYTES ? LOOP_BYTES : MRG_BYTES)
  // Split KV: a single workgroup merging every segment's partials is bound by
  // its own CU's bandwidth once they reach 64 KiB -- 2 us at 256 KiB.  Past
  // that the segments wait for each other and each merges a slice instead.
  #define SHARED_MERGE (ROWS_W * HEAD_DIM * M_NSEG * 4 >= 64 * 1024)
  #define M_BLOCK (M_NW * WAVE)
  // m_s, l_s, gm_s, gl_s and the merge's go flag, after the raw buffer.
  #define LDS_MS ((LDS_RAW + 15) / 16 * 16)
  #define LDS_ALL (LDS_MS + (2 * M_NW * ROWPAD + 2 * ROWPAD) * 4 + 16)

static_assert(GQA % M_RG == 0, "row groups split whole q heads");
static_assert(M_NW % (M_DSPL * M_RSPL) == 0, "whole tiles per workgroup");
static_assert(RTILES % M_RSPL == 0, "whole row tiles per wave");
static_assert(8 % M_RSPL == 0, "whole key pairs per wave of a shared tile");
static_assert(NCHUNK % M_DSPL == 0, "whole chunks per d part");
static_assert(VD == 4 || VD == 8, "a lane's K and V slice is one b64 or b128");
static_assert(TILES <= WAVE, "one lane per tile loads its page index");
static_assert(M_DSPL == 1 || SHT || RTW * 1024 <= 16 * KT_ROW,
              "the score exchange must fit a wave's K tile");

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
  vrow_t k[8 / M_RSPL];
  vrow_t v[8 / M_RSPL];
};

// Page index of every 16-key tile of block b: lane t holds tile t's.  Tiles
// past the sequence clamp to its last page, so every address stays inside a
// page vLLM allocated.
__device__ __forceinline__ int block_pages(const int* __restrict__ bt, int b,
                                           int lane, int S) {
  const int key = b * KBLK + (lane % TILES) * 16;
  // One tile: a uniform index, which then comes from the scalar cache.
  if (TILES == 1)
    return bt[__builtin_amdgcn_readfirstlane(min(key, S - 1)) / BS];
  return bt[(unsigned)min(key, S - 1) / BS];
}

// The same before S is known: clamped to the block table's width instead.
// Every entry of the table names an allocated page, so the address stays in
// the cache, and the keys past S are masked like any others.
__device__ __forceinline__ int block_pages_w(const int* __restrict__ bt, int b,
                                             int lane, int width) {
  const int blk = (b * KBLK + (lane % TILES) * 16) / BS;
  if (TILES == 1)
    return bt[__builtin_amdgcn_readfirstlane(min(blk, width - 1))];
  return bt[min(blk, width - 1)];
}

// A = V^T for output element j of this lane's slice: row d = base + VD*l16 +
// j, keys in the order above.  Half hi holds keys 2e + hi in v[e]; it packs
// its four dwords and receives the other half's four.
__device__ __forceinline__ e16 v_frag(const vrow_t* v, int j) {
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
  return __builtin_bit_cast(e16, r);
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

constexpr int kLds = LDS_ALL;
constexpr int kGrid = M_NSEG * NUM_KV_HEADS * M_RG;
// Arrival counters, then merge generations, one each per (kv head, row group).
constexpr int kCnt = 2 * NUM_KV_HEADS * M_RG;
constexpr bool kSharedMerge = SHARED_MERGE;

// The first block's page indices, which go out before S is known.
__device__ __forceinline__ int first_pages(const int* __restrict__ bt,
                                           int bt_width) {
  const int seg = blockIdx.x / (M_RG * NUM_KV_HEADS);
  return block_pages_w(bt, seg, threadIdx.x & (WAVE - 1), bt_width);
}

template <typename OutT>
__device__ __forceinline__ void body(
    const elem_t* __restrict__ q, const elem_t* __restrict__ kv,
    const int* __restrict__ bt, float* __restrict__ p_acc,
    float* __restrict__ p_m, float* __restrict__ p_l, int* __restrict__ p_cnt,
    OutT* __restrict__ out, const int S, int pages, int bt_width, float scale,
    int coop, char* __restrict__ lds) {
  const int tid = threadIdx.x;
  const int lane = tid & (WAVE - 1);
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE);
  const int l16 = lane & 15;
  const int hi = lane >> 4;
  const int tw = wave / (M_DSPL * M_RSPL);  // this wave's tile within the block
  const int rs = (wave / M_DSPL) % M_RSPL;  // its share of the row tiles
  const int dp = wave % M_DSPL;             // and its part of the head dim
  // Workgroup row tile of this wave's row tile rt.
  #define RT(rt) ((rt) * M_RSPL + rs)
  // A mode may use fewer waves than the block was launched with; the rest
  // leave now, and a finished wave no longer counts at a barrier.
  if (M_NW < BLOCK / WAVE && wave >= M_NW) return;

  const int bid = blockIdx.x;
  const int rg = bid % M_RG;
  const int kvh = (bid / M_RG) % NUM_KV_HEADS;
  const int seg = bid / (M_RG * NUM_KV_HEADS);

  __builtin_assume(S > 0);
  const int nblk = (S + KBLK - 1) / KBLK;
  const int nseg = max(1, min(M_NSEG, nblk / M_MINB));
  if (seg >= nseg) return;
  TS(0);
  if (ABLATE & 32) {
    if (S == -1) out[0] = (OutT)0;
    return;
  }

  // Q goes out before the KV.  Load completion is counted in order, so a Q
  // load issued behind the KV tiles would hold its LDS store, and the barrier
  // behind it, until the last KV byte landed.
  constexpr int QITER = (ROWPAD * HEAD_DIM + M_BLOCK * 8 - 1) / (M_BLOCK * 8);
  e8 qx[QITER];
  #pragma unroll
  for (int it = 0; it < QITER; ++it) {
    const int i = (it * M_BLOCK + tid) * 8;
    // Unconditional, from a clamped row: a load under a branch made the
    // compiler wait vmcnt(0) -- for Q as well as the page table -- before it
    // could issue the first KV load.
    const int r = min(i / HEAD_DIM, ROWS_W - 1), d = i % HEAD_DIM;
    const int h = kvh * GQA + rg * (GQA / M_RG) + r / MAXM;
    const int m = r % MAXM;
    qx[it] = *(const e8*)(q + ((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + d);
    if (i / HEAD_DIM >= ROWS_W) qx[it] = e8{};
  }
  // The group's merge generation cannot move before this segment arrives, so
  // it is read now rather than on the way to the arrival.
  int* const gen = p_cnt + NUM_KV_HEADS * M_RG + kvh * M_RG + rg;
  const int g0 = SHARED_MERGE ? __hip_atomic_load(gen, __ATOMIC_RELAXED,
                                                  __HIP_MEMORY_SCOPE_AGENT)
                              : 0;

  // q_s and the K tiles are dead once the main loop ends; the merge reuses
  // their storage.
  char* const lds_raw = lds;
  auto q_s = reinterpret_cast<elem_t(*)[HEAD_DIM + QPAD]>(lds_raw);
  float* mrg_s = reinterpret_cast<float*>(lds_raw);
  #if SHT
  // This wave's tile group's K tile in buffer `buf`; its V tile follows.
  char* const tb_s = lds_raw + QS_BYTES + (tw * M_DSPL + dp) * 2 * 16 * KT_ROW;
    #define KT_BUF(buf) (tb_s + (buf) * (TB_BYTES / 2))
  float* sx_s = reinterpret_cast<float*>(lds_raw + QS_BYTES + TB_BYTES);
    #define SX_SLOT(w) ((w) * RTW * 256)
  #else
  char* kt_s = lds_raw + QS_BYTES + wave * 16 * KT_ROW;
  // Wave w's partial scores go in wave w's own K tile.  It writes them only
  // after its own Q@K has read that tile, its partners read them between the
  // two barriers of the exchange, and it refills the tile with the next K
  // only after the second barrier.
  float* sx_s = reinterpret_cast<float*>(lds_raw + QS_BYTES);
    #define SX_SLOT(w) ((w) * (16 * KT_ROW / 4))
  #endif
  auto m_s = reinterpret_cast<float (*)[ROWPAD]>(lds + LDS_MS);
  auto l_s =
      reinterpret_cast<float (*)[ROWPAD]>(lds + LDS_MS + M_NW * ROWPAD * 4);
  float* const gm_s =
      reinterpret_cast<float*>(lds + LDS_MS + 2 * M_NW * ROWPAD * 4);
  float* const gl_s = gm_s + ROWPAD;

  const elem_t* kvh_base = kv + tile_off(0, 0, kvh);
  // This wave's slice of a K or V row, and its chunks of the head dim.
  const int col = dp * DPART + VD * l16;
  const int c0 = dp * NCP;

  // Issue block b's tile for this wave, and the page lookup of the block
  // after it.  Blocks are issued in order, so `pages` always holds b's.
  auto issue = [&](Tile& t, int b) {
    const int page = __builtin_amdgcn_readlane(pages, tw);
  #if !SHT
    if (b + nseg < nblk) pages = block_pages(bt, b + nseg, lane, S);
  #endif
    const elem_t* tp = kvh_base + (size_t)page * PAGE_ELEMS +
                       (size_t)((b * KBLK + tw * 16) % BS) * KEY_STRIDE +
                       (size_t)hi * KEY_STRIDE + col;
    // All of V, then all of K.  Interleaving them row by row measured 1.7 to
    // 3.3 us slower to land at S=128 (32/32/128), same bytes, same addresses.
    // A shared tile's wave loads only its EPW key pairs.
    const elem_t* tw_p = tp + (size_t)(SHT ? 2 * rs * EPW : 0) * KEY_STRIDE;
  #pragma unroll
    for (int e = 0; e < 8 / M_RSPL; ++e)
      t.v[e] = (ABLATE & 1)
                   ? vrow_t{}
                   : *(const vrow_t*)(tw_p + 2 * e * KEY_STRIDE + HEAD_DIM);
  #pragma unroll
    for (int e = 0; e < 8 / M_RSPL; ++e)
      t.k[e] =
          (ABLATE & 4) ? vrow_t{} : *(const vrow_t*)(tw_p + 2 * e * KEY_STRIDE);
    // The scheduler must neither sink these to their first use nor hoist
    // work above them: left alone under VGPR pressure it issued a tile's K
    // two loads at a time with a wait after each.
    asm volatile("" ::: "memory");
  #if SHT
    // A shared tile's next page indices go out after its loads: ahead of
    // them, a vector page-table load held the KV loads behind a vmcnt(0).
    // Unconditional (block_pages clamps to S), so the waits can count it.
    pages = block_pages(bt, b + nseg, lane, S);
  #endif
  };

  Tile ta;
  issue(ta, seg);

  #pragma unroll
  for (int it = 0; it < QITER; ++it) {
    const int i = (it * M_BLOCK + tid) * 8;
    if (i < ROWPAD * HEAD_DIM) *(e8*)&q_s[i / HEAD_DIM][i % HEAD_DIM] = qx[it];
  }

  const float scale2 = scale * LOG2E;
  const int ctx = S - MAXM;

  float m_run[RTW], l_run[RTW];
  f8 acc[RTW][VD];
  #pragma unroll
  for (int rt = 0; rt < RTW; ++rt) {
    m_run[rt] = -INFINITY;
    l_run[rt] = 0.f;
  #pragma unroll
    for (int j = 0; j < VD; ++j) acc[rt][j] = f8{};
  }
  // Token index of this lane's row in each row tile, for the causal mask.
  int mrow[RTW];
  #pragma unroll
  for (int rt = 0; rt < RTW; ++rt) {
    const int r = RT(rt) * 16 + l16;
    mrow[rt] = (r < ROWS_W) ? (r % MAXM) : (MAXM - 1);
  }

  lds_barrier();
  TS(1);

  // Stage a tile's K for Q@K: rows in, keys out.  LDS ops of one wave
  // complete in order, so no barrier.
  // Stored as integers and read back as halves, which type-based alias
  // analysis would let pass each other without the asm.
  #if SHT
  // A shared tile: this wave's key pairs of K and V into buffer kb.
  auto stage = [&](const Tile& t, char* kb) {
    #pragma unroll
    for (int i = 0; i < EPW; ++i) {
      const int row = (2 * (rs * EPW + i) + hi) * KT_ROW + VD * l16 * 2;
      *(vrow_t*)(kb + row) = t.k[i];
      *(vrow_t*)(kb + 16 * KT_ROW + row) = t.v[i];
    }
    asm volatile("" ::: "memory");
  };
  #else
  auto stage_k = [&](const Tile& t) {
    #pragma unroll
    for (int e = 0; e < 8; ++e)
      *(vrow_t*)(kt_s + (2 * e + hi) * KT_ROW + VD * l16 * 2) = t.k[e];
    asm volatile("" ::: "memory");
  };
  #endif

  // Q@K, the softmax update and P@V for one tile whose K is staged at kb.
  // v is the tile's V in registers; a shared tile reads it from after K.
  auto process = [&](Tile& t, vrow_t* v, const char* kb, int b) {
    const int kt = b * KBLK + tw * 16;
  #if ABLATE & 64
    unsigned x = 0;
    #pragma unroll
    for (int e = 0; e < 8 / M_RSPL; ++e)
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
    f8 s[RTW];
  #pragma unroll
    for (int rt = 0; rt < RTW; ++rt) s[rt] = f8{};
  #pragma unroll
    for (int c = 0; c < NCP; ++c) {
      const e16 a = *(const e16*)(kb + l16 * KT_ROW + c * 32);
  #pragma unroll
      for (int rt = 0; rt < RTW; ++rt) {
        const e16 bq = *(const e16*)&q_s[RT(rt) * 16 + l16][(c0 + c) * 16];
        if (!(ABLATE & 16) || c == 0) s[rt] = wmma(a, bq, s[rt]);
      }
    }
  #if M_DSPL > 1 && !(ABLATE & 128)
    // The scores overwrite this wave's K tile, which the Q@K above read as
    // halves; keep the float stores below those reads.  Then sum the tile's
    // M_DSPL partials in the same order in every wave, so they all see
    // bit-identical S and agree on the softmax.
    asm volatile("" ::: "memory");
    #pragma unroll
    for (int rt = 0; rt < RTW; ++rt)
      *(f8*)&sx_s[SX_SLOT(wave) + rt * 256 + lane * 8] = s[rt];
    lds_barrier();
    #pragma unroll
    for (int rt = 0; rt < RTW; ++rt) {
      f8 sum = *(const f8*)&sx_s[SX_SLOT(wave - dp) + rt * 256 + lane * 8];
    #pragma unroll
      for (int q2 = 1; q2 < M_DSPL; ++q2)
        sum += *(const f8*)&sx_s[SX_SLOT(wave - dp + q2) + rt * 256 + lane * 8];
      s[rt] = sum;
    }
    #if !SHT
    // Partners are done reading before anyone refills its K tile.  A shared
    // tile's barrier before the next Q@K already orders it.
    lds_barrier();
    #endif
  #endif
    if (b == seg) TS(2);
  #if SHT
    vrow_t vs[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e)
      vs[e] = *(const vrow_t*)(kb + 16 * KT_ROW + (2 * e + hi) * KT_ROW +
                               VD * l16 * 2);
    v = vs;
  #endif

    // A tile that reaches past S: its keys are masked below, and their V is
    // zeroed so that a NaN in an unused slot cannot reach P@V as 0 * NaN.
    if (__builtin_expect(kt + 16 > S, 0)) {
  #pragma unroll
      for (int e = 0; e < 8; ++e)
        if (kt + 2 * e + hi >= S) v[e] = vrow_t{};
    }

    // Lane holds row l16, keys kt + 2e + hi.  Only a tile that reaches past
    // the first query token's keys needs the causal mask.  The scale is
    // applied inside the exponent: it is positive, so the max commutes.
    const bool tail = kt + 16 > ctx + 1;
  #pragma unroll
    for (int rt = 0; rt < RTW; ++rt) {
      if (__builtin_expect(tail, 0)) {
  #pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int key = kt + 2 * e + hi;
  #if MUTATE == 1
          const bool valid = key <= ctx + mrow[rt] + 1 && key < S;
  #else
          const bool valid = key <= ctx + mrow[rt];
  #endif
          if (!valid) s[rt][e] = -INFINITY;
        }
      }
      float mx = s[rt][0];
  #pragma unroll
      for (int e = 1; e < 8; ++e) mx = fmaxf(mx, s[rt][e]);
      mx = fmaxf(mx, xhalf(mx)) * scale2;
      const float mnew = fmaxf(m_run[rt], mx);
      const float alpha =
          (mnew == -INFINITY) ? 1.f : __builtin_amdgcn_exp2f(m_run[rt] - mnew);
      m_run[rt] = mnew;
      float p[8], lo[8], sum = 0.f;
  #pragma unroll
      for (int e = 0; e < 8; ++e) {
        p[e] = (mnew == -INFINITY) ? 0.f
                                   : __builtin_amdgcn_exp2f(__builtin_fmaf(
                                         s[rt][e], scale2, -mnew));
        lo[e] = p[e] - elem_part(p[e]);
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
      const e16 bp = __builtin_bit_cast(e16, pk);
      const float ascale = __builtin_bit_cast(
          float, (__builtin_bit_cast(unsigned, alpha) & ~upper) |
                     (lower8(__builtin_bit_cast(unsigned, alpha)) & upper));
  #else
      const e16 bp = p_frag(p);
      const e16 bl = p_frag(lo);
      const float ascale = alpha;
  #endif
  #pragma unroll
      for (int j = 0; j < VD; ++j) {
        acc[rt][j] *= ascale;
        if ((ABLATE & 2) && j) continue;
        const e16 a = v_frag(v, j);
        acc[rt][j] = wmma(a, bp, acc[rt][j]);
  #if !PPACK
        acc[rt][j] = wmma(a, bl, acc[rt][j]);
  #endif
      }
    }
    asm volatile("" ::: "memory");
  };

  // One tile per wave in flight.  Keeping the next tile's loads in flight
  // as well (two register sets, or the next K and V held until the current
  // P@V is done) needs 60-100 VGPRs more than a wave has without spilling,
  // and spilled it measured 59 % against 77 % of roof on 32/4/128.
  #if SHT
  // Stage, put the next tile's loads in flight, then compute this one: the
  // registers a shared tile frees pay for the prefetch.  The LDS wait goes
  // before the loads, so the barrier does not wait for a page-table s_load
  // counted with LDS.  Two buffers: a wave refills one only after the
  // barrier that every wave reaches after reading it.
  int buf = 0;
  for (int b = seg; b < nblk; b += nseg) {
    char* const kb = KT_BUF(buf);
    stage(ta, kb);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    if (b + nseg < nblk) issue(ta, b + nseg);
    asm volatile("s_barrier" ::: "memory");
    process(ta, nullptr, kb, b);
    buf ^= 1;
  }
  #else
  for (int b = seg; b < nblk; b += nseg) {
    stage_k(ta);
    process(ta, ta.v, kt_s, b);
    if (b + nseg < nblk) issue(ta, b + nseg);
  }
  #endif
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
  for (int rt = 0; rt < RTW; ++rt)
    if (hi == 0) {
      m_s[wave][RT(rt) * 16 + l16] = m_run[rt];
      l_s[wave][RT(rt) * 16 + l16] = l_run[rt];
    }
  lds_barrier();
  float gsum[RTW];
  #pragma unroll
  for (int rt = 0; rt < RTW; ++rt) {
    const int r = RT(rt) * 16 + l16;
    float mx = -INFINITY;
  #pragma unroll
    for (int w = rs * M_DSPL + dp; w < M_NW; w += M_DSPL * M_RSPL)
      mx = fmaxf(mx, m_s[w][r]);
    float sm = 0.f;
  #pragma unroll
    for (int w = rs * M_DSPL + dp; w < M_NW; w += M_DSPL * M_RSPL)
      sm += weight_of(m_s[w][r], mx) * l_s[w][r];
    gsum[rt] = sm;
    if (tw == 0 && dp == 0 && hi == 0) {
      gm_s[r] = mx;
      gl_s[r] = sm;
    }
    const float f = weight_of(m_run[rt], mx);
  #pragma unroll
    for (int j = 0; j < VD; ++j) acc[rt][j] *= f;
  }

  TS(4);
  const int grp = kvh * M_RG + rg;
  const size_t pb = ((size_t)grp * M_NSEG + seg) * ROWPAD;
  const bool publish = nseg > 1;

  #pragma unroll
  for (int rt = 0; rt < RTW; ++rt) {
    // Real rows in this row tile.
    const int rv = min(16, ROWS_W - RT(rt) * 16);
    // The previous row tile's readers must be done with the region.
    if (rt) lds_barrier();
  #pragma unroll
    for (int st = TILES / 2; st >= TFIN; st >>= 1) {
      if (tw >= st && tw < 2 * st && l16 < rv)
        store_rows(mrg_s + (((tw - st) * M_RSPL + rs) * M_DSPL + dp) * MSLOT,
                   acc[rt], l16, hi);
      lds_barrier();
      if (tw < st && l16 < rv)
        add_rows(acc[rt], mrg_s + ((tw * M_RSPL + rs) * M_DSPL + dp) * MSLOT,
                 l16, hi);
      lds_barrier();
    }
  #if TFIN == 1
    // One live tile: its waves write straight from registers.
    if (tw == 0 && l16 < rv) {
      const int r = RT(rt) * 16 + l16;
      if (!publish) {
        const int h = kvh * GQA + rg * (GQA / M_RG) + r / MAXM;
        const int m = r % MAXM;
        const float inv = __builtin_amdgcn_rcpf(gsum[rt]);
        OutT* o = out + ((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + dp * DPART;
    #pragma unroll
        for (int e = 0; e < 8; ++e) {
          elem_t x[VD];
    #pragma unroll
          for (int j = 0; j < VD; ++j) x[j] = to_elem(acc[rt][j][e] * inv);
          *(vrow_t*)(o + VD * (2 * e + hi)) = *(const vrow_t*)x;
        }
      } else {
        store_rows(p_acc + (pb + RT(rt) * 16) * HEAD_DIM + dp * DPART, acc[rt],
                   l16, hi, HEAD_DIM);
      }
    }
  #else
    if (tw < TFIN && l16 < rv)
      store_rows(mrg_s + ((tw * M_RSPL + rs) * M_DSPL + dp) * MSLOT, acc[rt],
                 l16, hi);
    lds_barrier();
    // Eight consecutive d of one real row per thread, over the M_RSPL row
    // tiles this round's waves stored.
    const int r0 = rt * M_RSPL * 16;
    const int nr = min(M_RSPL * 16, ROWS_W - r0);
    for (int i = tid * 8; i < nr * HEAD_DIM; i += M_BLOCK * 8) {
      const int rl = i / HEAD_DIM, d = i % HEAD_DIM;
      const int r = r0 + rl;
      const float* src = mrg_s + ((rl / 16) * M_DSPL + d / DPART) * MSLOT +
                         (rl % 16) * MROW + d % DPART;
      f4 lo4 = *(const f4*)src, hi4 = *(const f4*)(src + 4);
    #pragma unroll
      for (int t = 1; t < TFIN; ++t) {
        lo4 += *(const f4*)(src + t * M_RSPL * M_DSPL * MSLOT);
        hi4 += *(const f4*)(src + t * M_RSPL * M_DSPL * MSLOT + 4);
      }
      if (!publish) {
        const float inv = __builtin_amdgcn_rcpf(gl_s[r]);
        e8 o;
    #pragma unroll
        for (int k = 0; k < 4; ++k) {
          o[k] = to_elem(lo4[k] * inv);
          o[4 + k] = to_elem(hi4[k] * inv);
        }
        const int h = kvh * GQA + rg * (GQA / M_RG) + r / MAXM;
        const int m = r % MAXM;
        *(e8*)(out + ((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + d) = o;
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
  // With a shared merge (`coop`: the host has checked the whole grid can be
  // resident at once, so waiting cannot deadlock) the others wait for it to
  // bump the group's generation, and every segment merges its slice.
  // Counting arrivals first and letting only the non-last segments fence
  // their partials out measured 0.5-1.6 % worse: their fence then sits on
  // the last one's path instead of ahead of its atomic.
  #if M_RSPL > 1
  // gm_s holds every wave's rows; with one tile no merge barrier ordered it.
  lds_barrier();
  #endif
  if (tid < ROWS_W) {
    p_m[pb + tid] = gm_s[tid];
    p_l[pb + tid] = gl_s[tid];
  }
  __threadfence();
  coop = SHARED_MERGE && coop;
  int& lds_go = *reinterpret_cast<int*>(gl_s + ROWPAD);
  lds_barrier();
  if (tid == 0) {
    const bool last = atomicAdd(&p_cnt[grp], 1) == nseg - 1;
    if (last) {
      // The generation first: the counter only has to be clean by the next
      // launch, and a release store behind it would wait for its ack.
      if (coop)
        __hip_atomic_store(gen, g0 + 1, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_AGENT);
      p_cnt[grp] = 0;
    } else if (coop) {
      while (__hip_atomic_load(gen, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT) == g0)
        __builtin_amdgcn_s_sleep(1);
    }
    lds_go = last || coop;
  }
  lds_barrier();
  TS(6);
  if (!lds_go) return;
  __threadfence();
  constexpr int NEL = ROWS_W * HEAD_DIM;
  const int chunk = coop ? (NEL / 4 + nseg - 1) / nseg * 4 : NEL;
  const int i0 = coop ? seg * chunk : 0;
  const int i1 = min(NEL, i0 + chunk);

  // Every load of the merge goes out at once -- each thread's rows' m and l
  // and its slice of every segment's partial, M_NSEG unrolled -- so the last
  // arriver pays one L2 round trip, not one for the running max, one for the
  // weights and one for the partials.
  for (int i = i0 + tid * 4; i < i1; i += M_BLOCK * 4) {
    const int r = i / HEAD_DIM, d = i % HEAD_DIM;
    float pm[M_NSEG], pl[M_NSEG];
    f4 pa[M_NSEG];
  #pragma unroll
    for (int sg = 0; sg < M_NSEG; ++sg) {
      const size_t row = ((size_t)grp * M_NSEG + sg) * ROWPAD + r;
      pm[sg] = sg < nseg ? p_m[row] : -INFINITY;
      pl[sg] = sg < nseg ? p_l[row] : 0.f;
      pa[sg] = sg < nseg ? *(const f4*)(p_acc + row * HEAD_DIM + d) : f4{};
    }
    float gm = -INFINITY;
  #pragma unroll
    for (int sg = 0; sg < M_NSEG; ++sg) gm = fmaxf(gm, pm[sg]);
    f4 num = {};
    float den = 0.f;
  #pragma unroll
    for (int sg = 0; sg < M_NSEG; ++sg) {
      const float a = weight_of(pm[sg], gm);
      den = fmaf(a, pl[sg], den);
      num += a * pa[sg];
    }
    const float inv = __builtin_amdgcn_rcpf(den);
    const int h = kvh * GQA + rg * (GQA / M_RG) + r / MAXM;
    const int m = r % MAXM;
    OutT* o = out + ((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + d;
  #pragma unroll
    for (int k = 0; k < 4; ++k) o[k] = to_elem(num[k] * inv);
  }
  TS(7);
}

  #undef RT
  #undef KT_BUF
  #undef SX_SLOT
  #undef ROWS_W
  #undef RTILES
  #undef ROWPAD
  #undef TILES
  #undef RTW
  #undef SHT
  #undef EPW
  #undef KBLK
  #undef DPART
  #undef VD
  #undef NCP
  #undef PPACK
  #undef KT_ROW
  #undef KT_BYTES
  #undef QS_BYTES
  #undef MROW
  #undef RV
  #undef MSLOT
  #undef SLOTS_FIT
  #undef TFIN
  #undef MRG_SLOTS
  #undef MRG_BYTES
  #undef TB_BYTES
  #undef SX_BYTES
  #undef LOOP_BYTES
  #undef LOOP_BYTES
  #undef LDS_RAW
  #undef SHARED_MERGE
  #undef LDS_MS
  #undef M_BLOCK
  #undef LDS_ALL

#endif  // RDNA35_BODY
