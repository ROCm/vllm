// f5-layout — v5 decode kernel re-hosted on vLLM's PAGED KV layout.
//
// Difference vs decode_attn_v5.hip: the KV cache is no longer a dense
// [S, HKV, D] pair of arrays.  It is vLLM's single paged tensor with K and V
// packed in the content dim, exactly as TritonAttentionBackend produces it:
//
//   logical  get_kv_cache_shape() -> (NB, HKV, BS, 2*D)
//   NHD      stride_order (0,2,1,3) -> (NB, BS, HKV, 2*D)   stride_head = 2*D*2 B
//   HND      stride_order (0,1,2,3) -> (NB, HKV, BS, 2*D)   stride_head = BS*2*D*2 B
//
// Knobs that move the strides:
//   KV_PAD    extra fp16 per (token, head) content row  -> stride_head += 2B*KV_PAD
//   PAGE_PAD  extra fp16 per page (vLLM's page_size_padded / `alignment`)
//   BS        block_size
//   LAYOUT    0 = NHD, 1 = HND
//   SHUF      1 = block table is a random permutation (realistic allocator)
//   POOLX     pool is POOLX x larger than the sequence needs
//
// Maths, masking, softmax, epilogue: byte-identical to v5.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

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

#define WAVE 32
#define NWAVE (BLOCK / WAVE)
#define DPL (HEAD_DIM / WAVE)
#define KV_ROW (2 * HEAD_DIM + KV_PAD)  // K and V packed, then pad
#define PAGE_ELEMS (BS * NUM_KV_HEADS * KV_ROW + PAGE_PAD)
#define GQA (NUM_Q_HEADS / NUM_KV_HEADS)
#define LOG2E 1.44269504088896340736f

static_assert(DPL == 8, "8 fp16 per lane (b128)");
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
#if MUTATE == 2
  return (size_t)blk * PAGE_ELEMS + ((size_t)kvh * BS + slot) * KV_ROW;
#else
  return (size_t)blk * PAGE_ELEMS + ((size_t)slot * NUM_KV_HEADS + kvh) * KV_ROW;
#endif
#else  // HND: (NB, HKV, BS, 2D)
  return (size_t)blk * PAGE_ELEMS + ((size_t)kvh * BS + slot) * KV_ROW;
#endif
}

__global__ __launch_bounds__(BLOCK) void decode_attn(
    const _Float16* __restrict__ q, const _Float16* __restrict__ kv,
    const int* __restrict__ bt, float* __restrict__ p_acc,
    float* __restrict__ p_m, float* __restrict__ p_l, int S, float scale) {
  const int seg = blockIdx.x;
  const int h = blockIdx.y;
  const int kvh = h / GQA;
  const int tid = threadIdx.x;
  const int lane = tid & (WAVE - 1);
  const int wave = tid / WAVE;
  const int dl = lane * DPL;

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
    f4v kr[KPW], vr[KPW];
#pragma unroll
    for (int c = 0; c < KPW; ++c) {
      int jj = jb + c;
      jj = (jj < S) ? jj : (S - 1);
      const size_t off = kv_off(bt, jj, kvh) + dl;
      kr[c] = *(const f4v*)(kv + off);
      #if MUTATE == 3
      vr[c] = *(const f4v*)(kv + off);
#else
      vr[c] = *(const f4v*)(kv + off + HEAD_DIM);
#endif
    }

    float s[KPW][MAXM];
#pragma unroll
    for (int c = 0; c < KPW; ++c)
#pragma unroll
      for (int m = 0; m < MAXM; ++m) {
        float t = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e)
          t = __builtin_amdgcn_fdot2(as_h2(qr[m][e]), as_h2(kr[c][e]), t, false);
        s[c][m] = t;
      }

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
        for (int i = 0; i < DPL; ++i) acc[m][i] += s[c][m] * (float)vv[i];
    }
  }

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
}

__global__ __launch_bounds__(HEAD_DIM) void reduce_segments(
    const float* __restrict__ p_acc, const float* __restrict__ p_m,
    const float* __restrict__ p_l, float* __restrict__ out) {
  const int m = blockIdx.x;
  const int h = blockIdx.y;
  const int d = threadIdx.x;

  float gmax = -INFINITY;
  for (int s = 0; s < NSEG * NWAVE; ++s)
    gmax = fmaxf(gmax, p_m[((size_t)h * (NSEG * NWAVE) + s) * MAXM + m]);

  float num = 0.f, den = 0.f;
  for (int s = 0; s < NSEG * NWAVE; ++s) {
    const size_t b = ((size_t)h * (NSEG * NWAVE) + s) * MAXM + m;
    const float a =
        (gmax == -INFINITY) ? 0.f : __builtin_amdgcn_exp2f(p_m[b] - gmax);
    num = fmaf(a, p_acc[b * HEAD_DIM + d], num);
    den = fmaf(a, p_l[b], den);
  }
  out[((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + d] = num / den;
}

// ---------------------------------------------------------------- host ----
#define CHK(x)                                                             \
  do {                                                                     \
    hipError_t e_ = (x);                                                   \
    if (e_ != hipSuccess) {                                                \
      printf("HIP error %s at line %d\n", hipGetErrorString(e_), __LINE__); \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

static std::vector<char> slurp(const char* path, size_t want) {
  FILE* f = fopen(path, "rb");
  if (!f) { printf("cannot open %s\n", path); exit(1); }
  std::vector<char> b(want);
  if (fread(b.data(), 1, want, f) != want) {
    printf("short read on %s (wanted %zu)\n", path, want);
    exit(1);
  }
  fclose(f);
  return b;
}

int main(int argc, char** argv) {
  if (argc < 7) {
    printf("usage: %s S iters q.bin k.bin v.bin out.bin [ncopy] [shuf] [poolx]\n",
           argv[0]);
    return 1;
  }
  const int S = atoi(argv[1]);
  const int iters = atoi(argv[2]);
  const int ncopy = (argc > 7) ? atoi(argv[7]) : 4;
  const int shuf = (argc > 8) ? atoi(argv[8]) : 0;
  const int poolx = (argc > 9) ? atoi(argv[9]) : 1;
  const float scale = 1.f / sqrtf((float)HEAD_DIM);

  const size_t qn = (size_t)MAXM * NUM_Q_HEADS * HEAD_DIM;
  const size_t kvn_dense = (size_t)S * NUM_KV_HEADS * HEAD_DIM;

  auto hq = slurp(argv[3], qn * 2);
  auto hk = slurp(argv[4], kvn_dense * 2);
  auto hv = slurp(argv[5], kvn_dense * 2);

  const int npages_used = (S + BS - 1) / BS;
  const int npages = npages_used * poolx;
  const size_t pool_elems = (size_t)npages * PAGE_ELEMS;

  // block table: which physical page holds logical block i
  std::vector<int> bt(npages_used);
  {
    std::vector<int> all(npages);
    std::iota(all.begin(), all.end(), 0);
    if (shuf) {
      std::mt19937 g(12345);
      std::shuffle(all.begin(), all.end(), g);
    }
    for (int i = 0; i < npages_used; ++i) bt[i] = all[i];
  }

  std::vector<_Float16> pool(pool_elems, (_Float16)0.f);
  {
    const _Float16* sk = (const _Float16*)hk.data();
    const _Float16* sv = (const _Float16*)hv.data();
    for (int t = 0; t < S; ++t) {
      const int blk = bt[t / BS], slot = t % BS;
      for (int hh = 0; hh < NUM_KV_HEADS; ++hh) {
#if LAYOUT == 0
        const size_t o = (size_t)blk * PAGE_ELEMS +
                         ((size_t)slot * NUM_KV_HEADS + hh) * KV_ROW;
#else
        const size_t o =
            (size_t)blk * PAGE_ELEMS + ((size_t)hh * BS + slot) * KV_ROW;
#endif
        memcpy(&pool[o], &sk[((size_t)t * NUM_KV_HEADS + hh) * HEAD_DIM],
               HEAD_DIM * sizeof(_Float16));
        memcpy(&pool[o + HEAD_DIM],
               &sv[((size_t)t * NUM_KV_HEADS + hh) * HEAD_DIM],
               HEAD_DIM * sizeof(_Float16));
      }
    }
  }

  _Float16 *dq, *dkv[16];
  int* dbt;
  float *dacc, *dm, *dl, *dout;
  CHK(hipMalloc(&dq, qn * 2));
  CHK(hipMemcpy(dq, hq.data(), qn * 2, hipMemcpyHostToDevice));
  CHK(hipMalloc(&dbt, npages_used * sizeof(int)));
  CHK(hipMemcpy(dbt, bt.data(), npages_used * sizeof(int),
                hipMemcpyHostToDevice));
  for (int c = 0; c < ncopy; ++c) {
    CHK(hipMalloc(&dkv[c], pool_elems * 2));
    CHK(hipMemcpy(dkv[c], pool.data(), pool_elems * 2, hipMemcpyHostToDevice));
  }
  const size_t accn = (size_t)NUM_Q_HEADS * NSEG * NWAVE * MAXM * HEAD_DIM;
  const size_t mln = (size_t)NUM_Q_HEADS * NSEG * NWAVE * MAXM;
  CHK(hipMalloc(&dacc, accn * 4));
  CHK(hipMalloc(&dm, mln * 4));
  CHK(hipMalloc(&dl, mln * 4));
  CHK(hipMalloc(&dout, qn * 4));

  dim3 grid(NSEG, NUM_Q_HEADS), block(BLOCK);
  dim3 rgrid(MAXM, NUM_Q_HEADS);

  auto launch = [&](int c) {
    hipLaunchKernelGGL(decode_attn, grid, block, 0, 0, dq, dkv[c], dbt, dacc, dm,
                       dl, S, scale);
    hipLaunchKernelGGL(reduce_segments, rgrid, dim3(HEAD_DIM), 0, 0, dacc, dm,
                       dl, dout);
  };

  launch(0);
  CHK(hipDeviceSynchronize());
  CHK(hipGetLastError());

  std::vector<float> ho(qn);
  CHK(hipMemcpy(ho.data(), dout, qn * 4, hipMemcpyDeviceToHost));
  FILE* f = fopen(argv[6], "wb");
  fwrite(ho.data(), 4, qn, f);
  fclose(f);

  if (iters > 0) {
    hipEvent_t a, b;
    CHK(hipEventCreate(&a));
    CHK(hipEventCreate(&b));
    auto timeit = [&](auto&& fn) {
      for (int i = 0; i < 5; ++i) fn(i % ncopy);
      CHK(hipDeviceSynchronize());
      CHK(hipEventRecord(a));
      for (int i = 0; i < iters; ++i) fn(i % ncopy);
      CHK(hipEventRecord(b));
      CHK(hipEventSynchronize(b));
      float ms = 0.f;
      CHK(hipEventElapsedTime(&ms, a, b));
      return 1000.0 * ms / iters;
    };
    const double us = timeit([&](int c) {
      hipLaunchKernelGGL(decode_attn, grid, block, 0, 0, dq, dkv[c], dbt, dacc,
                         dm, dl, S, scale);
    });
    const double bytes = 2.0 * (double)S * NUM_KV_HEADS * HEAD_DIM * 2.0;
#if LAYOUT == 0
    const long sh = (long)KV_ROW * 2;              // NHD: head stride
    const long st = (long)NUM_KV_HEADS * KV_ROW * 2;  // token stride
#else
    const long sh = (long)BS * KV_ROW * 2;
    const long st = (long)KV_ROW * 2;
#endif
    printf(
        "TIME paged LAYOUT=%d BS=%d KV_PAD=%d PAGE_PAD=%d shuf=%d poolx=%d  "
        "stride_head=%ld stride_tok=%ld page=%ld  %.2f us  (%.1f GiB/s)\n",
        LAYOUT, BS, KV_PAD, PAGE_PAD, shuf, poolx, sh, st,
        (long)PAGE_ELEMS * 2, us, bytes / (us * 1e-6) / (1024.0 * 1024 * 1024));
  }
  printf("OK    M=%d NSEG=%d LAYOUT=%d BS=%d KV_PAD=%d PAGE_PAD=%d S=%d\n", MAXM,
         NSEG, LAYOUT, BS, KV_PAD, PAGE_PAD, S);
  return 0;
}
