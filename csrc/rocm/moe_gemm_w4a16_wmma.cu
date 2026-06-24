// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 MoE producer/consumer WMMA prefill GEMM for AMD RDNA3 (gfx11).
// Always compiled into _rocm_C; the WMMA body below is gfx11-only (empty stub
// on other device arches, so multi-arch/CDNA builds still link) and registered
// as torch.ops._rocm_C.moe_gemm_w4a16 in torch_bindings.cpp. Python gates calls
// on on_gfx11().
//
//   moe_gemm_w4a16(A, w_packed, w_scale, sorted_token_ids, expert_ids, C,
//                  n_valid_tokens, top_k, block_m, num_blocks) -> ()
// computes C[slot] = A[token//top_k] @ dequant(W[expert]) in place with
// identical semantics/layout to invoke_fused_moe_kernel_hybrid_triton. Callers
// gate the shape (Python prefill_uses_rdna_moe_gemm), so an unsupported shape
// raises via TORCH_CHECK (skinny_gemms-style) rather than returning a fall-back
// status. Reads the weight N-row stride at runtime, so any padding (incl. the
// +128B gfx11x cache-cliff layout) is handled with no separate code path.
//
// The dequant helpers + moe_gemm1_kernel template are tuned for the
// Qwen3.6-35B-A3B prefill MoE shapes.

#include <torch/all.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// ----------------------------- device code -----------------------------
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
// 4 uints = 8 bf16 = 128 bits: one global_load_b128 per lane for the A producer
// load (replaces the per-element 16-bit gather global_load_d16_b16/_hi_b16).
typedef unsigned int v4u __attribute__((ext_vector_type(4)));

// ---- int4 -> bf16 dequant ("magic" trick) ----
__device__ __forceinline__ unsigned int two_nibbles_to_bf16(
    unsigned int packed) {
  unsigned int out;
  asm volatile("v_and_or_b32 %0, %1, %2, %3"
               : "=v"(out)
               : "v"(packed), "v"(0x000F000Fu), "v"(0x43004300u));
  return out;
}
__device__ __forceinline__ void unpack_8_int4(unsigned int* out4,
                                              unsigned int packed) {
  out4[0] = two_nibbles_to_bf16(packed);
  packed >>= 4;
  out4[1] = two_nibbles_to_bf16(packed);
  packed >>= 4;
  out4[2] = two_nibbles_to_bf16(packed);
  packed >>= 4;
  out4[3] = two_nibbles_to_bf16(packed);
}

// The kernel body uses RDNA3 __builtin_amdgcn_wmma_* intrinsics, which only
// exist on gfx11 device passes. __GFX11__ is predefined by the compiler on a
// gfx11 device pass (the same macro skinny_gemms / attention use). Compile the
// real body for gfx11 (and on the host pass, which needs the full definition);
// emit an empty stub for any other device arch so the TU links in non-gfx11 /
// multi-arch builds. The op is never called off gfx11 (gated by on_gfx11()).
#if defined(__HIPCC__) && defined(__GFX11__)
  #define MOE_W4A16_GFX11 1
#endif

// Template parameters. K, N and the weight N-row stride are RUNTIME args, so a
// single instantiation serves any shape meeting the tile divisibility
// (K % StepK == 0, N % TileN == 0) and any weight N-row padding; only the tile
// shape below is compile-time:
//   GroupSize           quant group size (scales are [E, N, K / GroupSize]).
//   TopK                MoE top_k; a power of two, so A's source row is
//                       token >> log2(TopK).
//   TileN               N columns one workgroup computes per N-tile
//                       (requires N % TileN == 0).
//   BlockM              rows per moe_align block (the alignment block_m).
//   StepK               K elements consumed per ring stage; a multiple of 16
//                       (the WMMA K dim) that divides GroupSize so a stage
//                       stays within a single quant group.
//   LdsPad              per-row padding on the LDS tiles that breaks shared-
//                       memory bank conflicts on the WMMA loads (tuned).
//   ProducerWaves       waves that copy A + packed weights global -> LDS ring.
//   ConsumerWaves       waves that dequant int4 -> bf16 and WMMA-accumulate.
//   RingBufferDepth     number of LDS ring stages (software-pipeline depth).
//   MinWorkgroupsPerCU  __launch_bounds__ occupancy hint (min blocks per CU).
template <int GroupSize, int TopK, int TileN, int BlockM, int StepK, int LdsPad,
          int ProducerWaves, int ConsumerWaves, int RingBufferDepth,
          int MinWorkgroupsPerCU>
__global__ __launch_bounds__(
    (ProducerWaves + ConsumerWaves) * 32,
    MinWorkgroupsPerCU) void moe_gemm1_kernel(const __hip_bfloat16* __restrict__ A,  // [tokens, K] acts
                                              const int* __restrict__ w_packed,  // [E, N, w_row_stride]
                                              const __hip_bfloat16* __restrict__ w_scale,  // [E, N, K/G]
                                              const int* __restrict__ sorted_token_ids,  // slot -> flat token
                                              const int* __restrict__ expert_ids,  // block -> expert
                                              __hip_bfloat16* __restrict__ C,  // [slots, N] output
                                              int K, int N, int w_row_stride,
                                              int n_valid_tokens, int n_slots,
                                              int n_blocks) {
#if defined(MOE_W4A16_GFX11) || !defined(__HIP_DEVICE_COMPILE__)
  const int quant_groups = K / GroupSize;
  const int num_k_steps = K / StepK;
  constexpr int M_TILES = BlockM / 16;
  constexpr int N_PER_CWAVE = TileN / ConsumerWaves;
  constexpr int N_TILES = N_PER_CWAVE / 16;
  constexpr int lds_A_stride = StepK + LdsPad;
  constexpr int lds_W_stride = TileN + LdsPad;
  constexpr int producer_threads = ProducerWaves * 32;
  constexpr int A_per_thread = (BlockM * StepK) / producer_threads;
  constexpr int W_per_thread = ((StepK / 8) * TileN) / producer_threads;
  constexpr int A_VEC = 8;  // bf16 per A vector load (v4u -> global_load_b128)

  static_assert(StepK % 16 == 0,
                "StepK must be a multiple of 16 (WMMA K-step)");
  static_assert(GroupSize % StepK == 0,
                "StepK must divide the quant GroupSize");
  static_assert(StepK <= GroupSize, "assumes >=1 ring stage per quant group");
  static_assert(N_PER_CWAVE % 16 == 0,
                "TileN/ConsumerWaves must be a mult of 16");
  static_assert(BlockM % 16 == 0,
                "BlockM must be a multiple of 16 (WMMA M-tile)");
  static_assert((TopK & (TopK - 1)) == 0,
                "TopK should be a power of two (divide -> shift)");
  static_assert((BlockM * StepK) % producer_threads == 0,
                "A tile must distribute evenly over producer threads");
  static_assert(((StepK / 8) * TileN) % producer_threads == 0,
                "weight tile must distribute evenly over producer threads");
  static_assert(StepK % A_VEC == 0, "StepK must be divisible by A_VEC");
  static_assert((BlockM * StepK) % (producer_threads * A_VEC) == 0,
                "A tile must distribute evenly over producer_threads * A_VEC");

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const bool producer = (wave < ProducerWaves);
  const int consumer_wave = wave - ProducerWaves;
  const int wave_n_base = consumer_wave * N_PER_CWAVE;

  __shared__ alignas(16) __bf16 lds_A[RingBufferDepth][BlockM * lds_A_stride];
  __shared__ int lds_W[RingBufferDepth][(StepK / 8) * lds_W_stride];
  __shared__ int s_token_row[BlockM];
  __shared__ int s_token[BlockM];
  __shared__ unsigned char s_valid[BlockM];

  bf16x16 ones;
  {
    unsigned int* words = (unsigned int*)&ones;
  #pragma unroll
    for (int i = 0; i < 8; i++) words[i] = 0x3F803F80u;
  }

  #define PRODUCE(stage, k_base)                                            \
    do {                                                                    \
      v4u a_buf[A_per_thread / A_VEC];                                      \
      int w_buf[W_per_thread];                                              \
      _Pragma("unroll") for (int v = 0; v < A_per_thread / A_VEC; v++) {    \
        int c = v * producer_threads + tid;                                 \
        int row = (c * A_VEC) / StepK, col = (c * A_VEC) % StepK;           \
        a_buf[v] =                                                          \
            *(const v4u*)&A[(size_t)s_token_row[row] * K + (k_base) + col]; \
      }                                                                     \
      _Pragma("unroll") for (int i = 0; i < W_per_thread; i++) {            \
        int flat = i * producer_threads + tid;                              \
        int ncol = flat % TileN, pk = flat / TileN;                         \
        w_buf[i] =                                                          \
            w_packed[w_expert_base + (long)(n_base + ncol) * w_row_stride + \
                     ((k_base) >> 3) + pk];                                 \
      }                                                                     \
      _Pragma("unroll") for (int v = 0; v < A_per_thread / A_VEC; v++) {    \
        int c = v * producer_threads + tid;                                 \
        int row = (c * A_VEC) / StepK, col = (c * A_VEC) % StepK;           \
        if constexpr (lds_A_stride % 8 == 0) {                              \
          *(v4u*)&lds_A[stage][row * lds_A_stride + col] = a_buf[v];        \
        } else {                                                            \
          int* d = (int*)&lds_A[stage][row * lds_A_stride + col];           \
          int* s = (int*)&a_buf[v];                                         \
          _Pragma("unroll") for (int j = 0; j < 4; j++) d[j] = s[j];        \
        }                                                                   \
      }                                                                     \
      _Pragma("unroll") for (int i = 0; i < W_per_thread; i++) {            \
        int flat = i * producer_threads + tid;                              \
        int ncol = flat % TileN, pk = flat / TileN;                         \
        lds_W[stage][pk * lds_W_stride + ncol] = w_buf[i];                  \
      }                                                                     \
    } while (0)

  #define CONSUME(stage, group)                                              \
    do {                                                                     \
      float scale[N_TILES];                                                  \
      _Pragma("unroll") for (int nt = 0; nt < N_TILES; nt++) scale[nt] =     \
          (float)                                                            \
              w_scale[scale_expert_base +                                    \
                      (long)(n_base + wave_n_base + nt * 16 + (lane & 15)) * \
                          quant_groups +                                     \
                      (group)];                                              \
      f32x8 raw_acc[M_TILES * N_TILES];                                      \
      f32x8 sum_a[M_TILES];                                                  \
      _Pragma("unroll") for (int i = 0; i < M_TILES * N_TILES; i++)          \
          raw_acc[i] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};                      \
      _Pragma("unroll") for (int i = 0; i < M_TILES; i++) sum_a[i] =         \
          (f32x8){0, 0, 0, 0, 0, 0, 0, 0};                                   \
      _Pragma("unroll") for (int kk = 0; kk < StepK / 16; kk++) {            \
        bf16x16 a_frag[M_TILES], b_frag[N_TILES];                            \
        _Pragma("unroll") for (int mt = 0; mt < M_TILES; mt++) {             \
          int row = mt * 16 + (lane & 15);                                   \
          _Pragma("unroll") for (int el = 0; el < 16; el++) a_frag[mt][el] = \
              lds_A[stage][row * lds_A_stride + kk * 16 + el];               \
        }                                                                    \
        _Pragma("unroll") for (int nt = 0; nt < N_TILES; nt++) {             \
          int col = wave_n_base + nt * 16 + (lane & 15);                     \
          int w_lo = lds_W[stage][(kk * 2) * lds_W_stride + col];            \
          int w_hi = lds_W[stage][(kk * 2 + 1) * lds_W_stride + col];        \
          unsigned int* frag = (unsigned int*)&b_frag[nt];                   \
          unpack_8_int4(frag, (unsigned int)w_lo);                           \
          unpack_8_int4(frag + 4, (unsigned int)w_hi);                       \
        }                                                                    \
        _Pragma("unroll") for (int mt = 0; mt < M_TILES; mt++) {             \
          sum_a[mt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(           \
              a_frag[mt], ones, sum_a[mt]);                                  \
          _Pragma("unroll") for (int nt = 0; nt < N_TILES; nt++)             \
              raw_acc[mt * N_TILES + nt] =                                   \
                  __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(               \
                      a_frag[mt], b_frag[nt], raw_acc[mt * N_TILES + nt]);   \
        }                                                                    \
      }                                                                      \
      _Pragma("unroll") for (int mt = 0; mt < M_TILES; mt++)                 \
          _Pragma("unroll") for (int nt = 0; nt < N_TILES; nt++) {           \
        float sc = scale[nt];                                                \
        int tile = mt * N_TILES + nt;                                        \
        _Pragma("unroll") for (int el = 0; el < 8; el++) acc[tile][el] +=    \
            sc * (raw_acc[tile][el] - 136.0f * sum_a[mt][el]);               \
      }                                                                      \
    } while (0)

  const int block = blockIdx.x % n_blocks;
  const int n_tile = blockIdx.x / n_blocks;
  const int expert = expert_ids[block];
  // Padding blocks (over-launch up to the EM cap) carry expert_id == -1.
  // Skip them exactly like Triton's `if off_experts == -1: return`, so the
  // host can launch a sync-free upper-bound grid.
  if (expert < 0) return;
  const int n_base = n_tile * TileN;
  const long w_expert_base = (long)expert * N * w_row_stride;
  const long scale_expert_base = (long)expert * N * quant_groups;

  if (tid < BlockM) {
    int slot = block * BlockM + tid;
    int token =
        (slot < n_slots) ? sorted_token_ids[slot] : (n_valid_tokens + 1);
    s_valid[tid] = (token < n_valid_tokens) ? 1 : 0;
    s_token_row[tid] = (token < n_valid_tokens) ? (token / TopK) : 0;
    s_token[tid] = token;
  }
  __syncthreads();

  f32x8 acc[M_TILES * N_TILES];
  #pragma unroll
  for (int i = 0; i < M_TILES * N_TILES; i++)
    acc[i] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  if (producer) {
  #pragma unroll
    for (int s = 0; s < RingBufferDepth - 1; s++)
      if (s < num_k_steps) PRODUCE(s, s * StepK);
  }
  __syncthreads();
  for (int step = 0; step < num_k_steps; step++) {
    int prefetch = step + (RingBufferDepth - 1);
    if (producer && prefetch < num_k_steps)
      PRODUCE(prefetch % RingBufferDepth, prefetch * StepK);
    if (!producer) CONSUME(step % RingBufferDepth, (step * StepK) / GroupSize);
    __syncthreads();
  }

  if (!producer) {
    int col_lane = lane & 15, row_parity = lane >> 4;
  #pragma unroll
    for (int mt = 0; mt < M_TILES; mt++)
  #pragma unroll
      for (int nt = 0; nt < N_TILES; nt++) {
        int tile = mt * N_TILES + nt;
        int out_col = n_base + wave_n_base + nt * 16 + col_lane;
  #pragma unroll
        for (int el = 0; el < 8; el++) {
          int out_row = mt * 16 + 2 * el + row_parity;
          if (out_row < BlockM && s_valid[out_row])
            C[(size_t)s_token[out_row] * N + out_col] =
                (__hip_bfloat16)acc[tile][el];
        }
      }
  }
  #undef PRODUCE
  #undef CONSUME
#endif  // MOE_W4A16_GFX11 || !__HIP_DEVICE_COMPILE__
}
// --------------------------- end device code ---------------------------

// ---- host launch dispatch ----
// valid_blocks = ceil(ntpp / BlockM); but the op receives n_valid_tokens (=
// M*top_k) and the routing tensors. The grid uses the number of populated
// blocks = expert_ids.numel() (one expert id per BlockM block, the alignment
// block count), matching the standalone harness's valid_blocks.
template <int GroupSize, int TopK, int TileN, int BlockM, int StepK, int LdsPad,
          int ProducerWaves, int ConsumerWaves, int RingBufferDepth,
          int MinWorkgroupsPerCU>
static void launch_moe(const at::Tensor& A, const at::Tensor& w_packed,
                       const at::Tensor& w_scale, const at::Tensor& sti,
                       const at::Tensor& eid, at::Tensor& C, int K, int N,
                       int n_valid_tokens, int n_blocks) {
  const int n_slots = (int)sti.size(0);
  // Weight N-row stride is read from the tensor, so any padding (incl. the
  // gfx11x cache-cliff +128B layout) is handled automatically.
  const int w_row_stride = (int)w_packed.stride(1);
  // n_blocks = sync-free launch upper bound (EM cap, like Triton). Padding
  // blocks read expert_id == -1 and early-return via the in-kernel guard.
  constexpr int THREADS = (ProducerWaves + ConsumerWaves) * 32;
  dim3 block(THREADS);
  dim3 grid((N / TileN) * n_blocks);
  hipStream_t stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA();
  moe_gemm1_kernel<GroupSize, TopK, TileN, BlockM, StepK, LdsPad, ProducerWaves,
                   ConsumerWaves, RingBufferDepth, MinWorkgroupsPerCU>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
          w_packed.data_ptr<int>(),
          reinterpret_cast<const __hip_bfloat16*>(w_scale.data_ptr()),
          sti.data_ptr<int>(), eid.data_ptr<int>(),
          reinterpret_cast<__hip_bfloat16*>(C.data_ptr()), K, N, w_row_stride,
          n_valid_tokens, n_slots, n_blocks);
}

// Runs a tuned instantiation for the given (top_k, K, N) shape; the in-place C
// is the only output. Callers (the vLLM op + its Python predicate
// prefill_uses_rdna_moe_gemm) gate the shape first, so the TORCH_CHECK guards
// below are "can't happen" invariants -- mirroring skinny_gemms' switch/throw
// dispatch rather than returning a fall-back boolean.
void moe_gemm_w4a16(at::Tensor A, at::Tensor w_packed, at::Tensor w_scale,
                    at::Tensor sorted_token_ids, at::Tensor expert_ids,
                    at::Tensor C, int64_t n_valid_tokens, int64_t top_k,
                    int64_t block_m, int64_t num_blocks) {
  TORCH_CHECK(A.scalar_type() == at::kBFloat16, "A must be bf16");
  TORCH_CHECK(C.scalar_type() == at::kBFloat16, "C must be bf16");
  TORCH_CHECK(w_packed.scalar_type() == at::kInt, "w_packed must be int32");
  TORCH_CHECK(w_scale.scalar_type() == at::kBFloat16, "w_scale must be bf16");
  // A, C, w_scale must be contiguous; w_packed only needs a contiguous
  // innermost (K//8) dim -- its N-row stride is read at runtime, so any padding
  // (incl. the gfx11x cache-cliff +128B layout) is handled automatically.
  TORCH_CHECK(A.is_contiguous() && w_scale.is_contiguous() && C.is_contiguous(),
              "A, C and w_scale must be contiguous");
  TORCH_CHECK(w_packed.stride(2) == 1, "w_packed K//8 dim must be contiguous");

  const int K = (int)A.size(1);
  const int N = (int)w_packed.size(1);
  TORCH_CHECK((int)w_packed.size(2) == K / 8, "w_packed last dim must be K//8");
  // The kernel assumes the expert stride is N * row_stride.
  const long w_row_stride = w_packed.stride(1);
  TORCH_CHECK(w_packed.stride(0) == (long)N * w_row_stride,
              "w_packed expert stride must equal N * row_stride");
  const int G = K / (int)w_scale.size(2);
  const int nvt = (int)n_valid_tokens;
  int nb = (int)num_blocks;
  const int nb_cap = (int)expert_ids.size(0);
  if (nb > nb_cap) nb = nb_cap;  // never index past the expert_ids buffer

  // Constraint-based tile selection. Two tile families keyed by top_k (the MoE
  // structure: gemm1 = routed up/gate proj, top_k>1; gemm2 = down proj,
  // top_k==1). Each accepts ANY (K, N) meeting the tile's divisibility --
  // K % G == 0 (implies K % StepK == 0 since StepK | G) and N % TileN == 0 --
  // so the kernel is not pinned to the exact Qwen3.6 shapes. Only G==128 and
  // block_m==32 are instantiated.
  // launch_moe<GroupSize, TopK, TileN, BlockM, StepK, LdsPad, ProducerWaves,
  //            ConsumerWaves, RingBufferDepth, MinWorkgroupsPerCU>
  TORCH_CHECK(G == 128 && (K % G) == 0 && block_m == 32,
              "moe_gemm_w4a16 requires G==128, K%G==0, block_m==32 "
              "(got G=",
              G, " K=", K, " block_m=", block_m, ")");

  switch (top_k) {
    case 8:  // gemm1 family: TileN=256, StepK=128.
      TORCH_CHECK((N % 256) == 0, "moe_gemm_w4a16 gemm1 requires N % 256 == 0");
      launch_moe<128, 8, 256, 32, 128, 4, 2, 4, 2, 3>(
          A, w_packed, w_scale, sorted_token_ids, expert_ids, C, K, N, nvt, nb);
      break;
    case 1:  // gemm2 family: TileN=512, StepK=64.
      TORCH_CHECK((N % 512) == 0, "moe_gemm_w4a16 gemm2 requires N % 512 == 0");
      launch_moe<128, 1, 512, 32, 64, 2, 4, 8, 3, 2>(
          A, w_packed, w_scale, sorted_token_ids, expert_ids, C, K, N, nvt, nb);
      break;
    default:
      TORCH_CHECK(false, "moe_gemm_w4a16 unsupported top_k=", top_k,
                  " (expected 1 or 8)");
  }
}
