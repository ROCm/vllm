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
// Merge the per-wave partials inside the workgroup instead of through a second
// kernel.  Only valid when NSEG==1, where a head's partials never leave one
// workgroup.  Exposed as a knob so both paths can be measured against each
// other; NSEG>1 must keep the global reduction.
#ifndef FUSED
  #define FUSED (NSEG == 1)
#endif
#if FUSED && NSEG != 1
  #error "FUSED requires NSEG==1: partials span workgroups otherwise"
#endif

#define WAVE 32
#define NWAVE (BLOCK / WAVE)
#define DPL (HEAD_DIM / WAVE)
#define KV_ROW (2 * HEAD_DIM + KV_PAD)  // K and V packed, then pad
#define PAGE_ELEMS (BS * NUM_KV_HEADS * KV_ROW + PAGE_PAD)
#define GQA (NUM_Q_HEADS / NUM_KV_HEADS)
#define LOG2E 1.44269504088896340736f

static_assert(DPL == 8, "8 fp16 per lane (b128)");
#if FUSED
static_assert(BLOCK == HEAD_DIM, "fused epilogue gives each thread one d");
#endif
static_assert(KV_PAD % 8 == 0, "KV_PAD must keep rows 16B aligned");
static_assert(PAGE_PAD % 8 == 0, "PAGE_PAD must keep pages 16B aligned");

typedef _Float16 h2v __attribute__((ext_vector_type(2)));
typedef float f4v __attribute__((ext_vector_type(4)));

__device__ __forceinline__ h2v as_h2(float x) {
  return __builtin_bit_cast(h2v, x);
}

// byte offset (in fp16 elements) of the K row for (token j, kv head kvh)
__device__ __forceinline__ size_t kv_off(const int* __restrict__ bt, int j,
                                         int kvh) {
  const int blk = bt[j / BS];
  const int slot = j % BS;
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
template <typename OutT>
__global__ __launch_bounds__(BLOCK) void decode_attn(
    const _Float16* __restrict__ q, const _Float16* __restrict__ kv,
    const int* __restrict__ bt, float* __restrict__ p_acc,
    float* __restrict__ p_m, float* __restrict__ p_l, OutT* __restrict__ out,
    int S, float scale) {
  const int seg = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (WAVE - 1);
  const int wave = tid / WAVE;
  const int dl = lane * DPL;
  const int h = blockIdx.y;
  const int kvh = h / GQA;

  f4v qr[MAXM];
#pragma unroll
  for (int m = 0; m < MAXM; ++m)
    qr[m] = *(const f4v*)(q + ((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + dl);

  float acc[MAXM][DPL], mx[MAXM], ls[MAXM];
#pragma unroll
  for (int m = 0; m < MAXM; ++m) {
#pragma unroll
    for (int i = 0; i < DPL; ++i) acc[m][i] = 0.f;
    mx[m] = -INFINITY;
    ls[m] = 0.f;
  }

  const float scale2 = scale * LOG2E;
  const int ctx = S - MAXM;
#if ILV
  const int gw = seg * NWAVE + wave;
  const int j1 = S;
  const int jstart = gw * KPW;
  const int jstep = NSEG * NWAVE * KPW;
#else
  const int sl0 = (S + NSEG - 1) / NSEG;
  const int seg_len = (sl0 + KPW - 1) / KPW * KPW;
  const int j1 = min(S, seg * seg_len + seg_len);
  const int jstart = seg * seg_len + wave * KPW;
  const int jstep = NWAVE * KPW;
#endif

  for (int jb = jstart; jb < j1; jb += jstep) {
    // Issued together on purpose.  The compiler emits k0,v0,k1,k2,k3,v1,v2,v3
    // and resumes at s_waitcnt vmcnt(7), so V never blocks Q@K; its latency
    // hides behind the dot products, the lane reduction and the softmax.
    // Measured: deferring V to just before P@V costs 5.5% at S=32768, where
    // keeping eight loads in flight is what sustains the bandwidth.
    f4v kr[KPW], vr[KPW];
#pragma unroll
    for (int c = 0; c < KPW; ++c) {
      int jj = jb + c;
      jj = (jj < S) ? jj : (S - 1);
      const size_t off = kv_off(bt, jj, kvh) + dl;
      kr[c] = *(const f4v*)(kv + off);
      vr[c] = *(const f4v*)(kv + off + HEAD_DIM);
    }

    float s[KPW][MAXM];
#pragma unroll
    for (int c = 0; c < KPW; ++c)
#pragma unroll
      for (int m = 0; m < MAXM; ++m) {
        float t = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e)
          t = __builtin_amdgcn_fdot2(as_h2(qr[m][e]), as_h2(kr[c][e]), t,
                                     false);
        s[c][m] = t;
      }

// Measured: lowering these to DPP row_xmask instead is 18% SLOWER across the
// whole context sweep.  The inner loop uses no LDS, so the LDS pipe is idle
// and ds_bpermute runs there in parallel; DPP moves the work onto the busy
// VALU pipe and costs a third of the dual-issue pairing as well.
#pragma unroll
    for (int st = 1; st < WAVE; st <<= 1)
#pragma unroll
      for (int c = 0; c < KPW; ++c)
#pragma unroll
        for (int m = 0; m < MAXM; ++m) s[c][m] += __shfl_xor(s[c][m], st, WAVE);

#pragma unroll
    for (int c = 0; c < KPW; ++c) {
      const int jj = jb + c;
#pragma unroll
      for (int m = 0; m < MAXM; ++m) {
#if MUTATE == 1
        const bool valid = (jj < j1) && (jj <= ctx + m + 1);
#else
        const bool valid = (jj < j1) && (jj <= ctx + m);
#endif
        s[c][m] = valid ? s[c][m] * scale2 : -INFINITY;
      }
    }

#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      float mnew = mx[m];
#pragma unroll
      for (int c = 0; c < KPW; ++c) mnew = fmaxf(mnew, s[c][m]);
      const float alpha =
          (mnew == -INFINITY) ? 0.f : __builtin_amdgcn_exp2f(mx[m] - mnew);
      mx[m] = mnew;
      ls[m] *= alpha;
#pragma unroll
      for (int i = 0; i < DPL; ++i) acc[m][i] *= alpha;
      float lsum = 0.f;
#pragma unroll
      for (int c = 0; c < KPW; ++c) {
        const float p =
            (mnew == -INFINITY) ? 0.f : __builtin_amdgcn_exp2f(s[c][m] - mnew);
        s[c][m] = p;
        lsum += p;
      }
      ls[m] += lsum;
    }

#pragma unroll
    for (int c = 0; c < KPW; ++c) {
      const _Float16* vv = (const _Float16*)&vr[c];
#pragma unroll
      for (int m = 0; m < MAXM; ++m)
#pragma unroll
        // Not worth pairing into VOPD: thinning this loop to 1/8 of its work
        // buys 2.2% at S=128, and perfect dual-issue is only worth half of it.
        // Occupancy already hides the VALU behind other waves' loads.
        for (int i = 0; i < DPL; ++i) acc[m][i] += s[c][m] * (float)vv[i];
    }
  }

#if FUSED
  // With NSEG==1 the grid is (1, NUM_Q_HEADS), so all NWAVE partials of a head
  // are produced by this one workgroup.  Merging them here costs a barrier and
  // HEAD_DIM floats of LDS per wave; routing them through global memory for a
  // second kernel costs a 2 MiB round trip and a launch, which at S=128 is as
  // many bytes as the KV stream itself.
  //
  // Staged one m at a time to keep LDS at NWAVE*HEAD_DIM floats (8 KiB) rather
  // than MAXM times that.
  __shared__ float lds_acc[NWAVE * HEAD_DIM];
  __shared__ float lds_m[NWAVE];
  __shared__ float lds_l[NWAVE];

  for (int m = 0; m < MAXM; ++m) {
    __syncthreads();  // the previous m's readers are done with lds_acc
  #pragma unroll
    for (int i = 0; i < DPL; ++i) lds_acc[wave * HEAD_DIM + dl + i] = acc[m][i];
    if (lane == 0) {
      lds_m[wave] = mx[m];
      lds_l[wave] = ls[m];
    }
    __syncthreads();

    float gmax = -INFINITY;
  #pragma unroll
    for (int w = 0; w < NWAVE; ++w) gmax = fmaxf(gmax, lds_m[w]);

    // A wave that got no tiles carries mx=-INFINITY and ls=0, so its weight is
    // exp2(-inf) = 0 and it drops out without special-casing.
    float a[NWAVE], den = 0.f;
  #pragma unroll
    for (int w = 0; w < NWAVE; ++w) {
      a[w] = __builtin_amdgcn_exp2f(lds_m[w] - gmax);
      den = fmaf(a[w], lds_l[w], den);
    }

    float num = 0.f;
  #pragma unroll
    for (int w = 0; w < NWAVE; ++w)
      num = fmaf(a[w], lds_acc[w * HEAD_DIM + tid], num);
    out[((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + tid] = (OutT)(num / den);
  }
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
  constexpr int NPART = NSEG * NWAVE;
  const int m = blockIdx.x;
  const int h = blockIdx.y;
  const int d4 = threadIdx.x * 4;
  const size_t base = (size_t)h * NPART * MAXM + m;

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

  // One reciprocal instead of four divides.
  const float inv = 1.f / den;
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
                    int64_t seq_len, double scale) {
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

  dim3 grid(NSEG, NUM_Q_HEADS), block(BLOCK);
  dim3 rgrid(MAXM, NUM_Q_HEADS);

  const auto* qp = reinterpret_cast<const _Float16*>(q.data_ptr());
  const auto* kvp = reinterpret_cast<const _Float16*>(kv_cache.data_ptr());
  const int* btp = block_table.data_ptr<int>();
  float* accp = acc.data_ptr<float>();
  float* mp = m.data_ptr<float>();
  float* lp = l.data_ptr<float>();

  auto* outp = reinterpret_cast<_Float16*>(out.data_ptr());
  hipLaunchKernelGGL(decode_attn<_Float16>, grid, block, 0, stream, qp, kvp,
                     btp, accp, mp, lp, outp, (int)seq_len, (float)scale);
  #if !FUSED
  hipLaunchKernelGGL(reduce_segments<_Float16>, rgrid, dim3(RED_THREADS), 0,
                     stream, accp, mp, lp, outp);
  #endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("decode_attn", &decode_attn_op, "RDNA3.5 paged decode attention");
}

#endif  // RDNA35_TORCH_EXT
