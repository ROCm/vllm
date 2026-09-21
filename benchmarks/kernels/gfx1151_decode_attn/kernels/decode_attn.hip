// f3a — decode attention kernel in HIP for gfx1151. CORRECTNESS reference.
//
// Online softmax with split-KV. Deliberately unoptimised: this kernel is the
// fleet's correctness control, not a performance candidate.
//
//   Q : [M, HQ, D]                 fp16
//   K : [S, HKV, D + KV_PAD]       fp16   (KV_PAD = stride padding, in elems)
//   V : [S, HKV, D + KV_PAD]       fp16
//   O : [M, HQ, D]                 fp32
//
// Causal mask: query m (0-indexed within the M tokens) sees keys
// 0 .. (S - M) + m.   GQA: query head h reads kv head h / (HQ / HKV).
//
// Two kernels: `decode_attn` produces per-segment (m, l, acc); `reduce_segments`
// merges them with the standard flash rescaling.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#ifndef MAXM  // number of query tokens (compile-time so acc[] stays in VGPRs)
#define MAXM 4
#endif
#ifndef NSEG  // split-KV segments   <-- parametrizable, not tuned here
#define NSEG 16
#endif
#ifndef KV_PAD  // extra fp16 elements per (token, kv head) row  <-- ditto
#define KV_PAD 0
#endif
#ifndef TILE  // keys processed per inner iteration
#define TILE 32
#endif
#ifndef LDS_PAD  // LDS row padding, kills the 8-way bank conflict on Ks
#define LDS_PAD 8
#endif

#define KV_ROW (HEAD_DIM + KV_PAD)
#define KV_TOK (NUM_KV_HEADS * KV_ROW)
#define GQA (NUM_Q_HEADS / NUM_KV_HEADS)

__global__ __launch_bounds__(HEAD_DIM) void decode_attn(
    const _Float16* __restrict__ q, const _Float16* __restrict__ k,
    const _Float16* __restrict__ v, float* __restrict__ p_acc,
    float* __restrict__ p_m, float* __restrict__ p_l, int S, float scale) {
  const int seg = blockIdx.x;
  const int h = blockIdx.y;
  const int kvh = h / GQA;
  const int d = threadIdx.x;

  __shared__ _Float16 Ks[TILE][HEAD_DIM + LDS_PAD];
  __shared__ _Float16 Qs[MAXM][HEAD_DIM + LDS_PAD];
  __shared__ float Ps[TILE][MAXM];
  __shared__ float Mn[MAXM];
  __shared__ float Ln[MAXM];

#pragma unroll
  for (int m = 0; m < MAXM; ++m)
    Qs[m][d] = q[((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + d];

  float acc[MAXM], mx[MAXM], ls[MAXM];
#pragma unroll
  for (int m = 0; m < MAXM; ++m) {
    acc[m] = 0.f;
    mx[m] = -INFINITY;
    ls[m] = 0.f;
  }

  const int ctx = S - MAXM;  // query m sees keys 0 .. ctx + m
  const int seg_len = (S + NSEG - 1) / NSEG;
  const int j0 = seg * seg_len;
  const int j1 = min(S, j0 + seg_len);
  __syncthreads();

  for (int jb = j0; jb < j1; jb += TILE) {
    const int nk = min(TILE, j1 - jb);

#pragma unroll
    for (int r = 0; r < TILE; ++r)
      Ks[r][d] = (r < nk) ? k[(size_t)(jb + r) * KV_TOK + kvh * KV_ROW + d]
                          : (_Float16)0.f;
    __syncthreads();

    // ---- scores: one thread per (key, query token) ----
    if (d < TILE * MAXM) {
      const int r = d / MAXM;
      const int m = d - r * MAXM;
      float s = 0.f;
      for (int e = 0; e < HEAD_DIM; ++e) s += (float)Qs[m][e] * (float)Ks[r][e];
      const bool valid = (r < nk) && (jb + r <= ctx + m);
      Ps[r][m] = valid ? s * scale : -INFINITY;
    }
    __syncthreads();

    if (d < MAXM) {
      float t = -INFINITY;
      for (int r = 0; r < TILE; ++r) t = fmaxf(t, Ps[r][d]);
      Mn[d] = t;
    }
    __syncthreads();

    // ---- online softmax rescale (uniform across the block) ----
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      const float mnew = fmaxf(mx[m], Mn[m]);
      const float alpha = (mnew == -INFINITY) ? 0.f : __expf(mx[m] - mnew);
      acc[m] *= alpha;
      ls[m] *= alpha;
      mx[m] = mnew;
    }

    if (d < TILE * MAXM) {
      const int r = d / MAXM;
      const int m = d - r * MAXM;
      Ps[r][m] = (mx[m] == -INFINITY) ? 0.f : __expf(Ps[r][m] - mx[m]);
    }
    __syncthreads();

    if (d < MAXM) {
      float t = 0.f;
      for (int r = 0; r < TILE; ++r) t += Ps[r][d];
      Ln[d] = t;
    }
    __syncthreads();

#pragma unroll
    for (int m = 0; m < MAXM; ++m) ls[m] += Ln[m];

    // ---- P @ V : one thread per value-dim, coalesced over d ----
    for (int r = 0; r < nk; ++r) {
      const float vv = (float)v[(size_t)(jb + r) * KV_TOK + kvh * KV_ROW + d];
#pragma unroll
      for (int m = 0; m < MAXM; ++m) acc[m] += Ps[r][m] * vv;
    }
    __syncthreads();
  }

  const size_t base = ((size_t)h * NSEG + seg) * MAXM;
#pragma unroll
  for (int m = 0; m < MAXM; ++m) {
    p_acc[(base + m) * HEAD_DIM + d] = acc[m];
    if (d == 0) {
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
  for (int s = 0; s < NSEG; ++s)
    gmax = fmaxf(gmax, p_m[((size_t)h * NSEG + s) * MAXM + m]);

  float num = 0.f, den = 0.f;
  for (int s = 0; s < NSEG; ++s) {
    const size_t b = ((size_t)h * NSEG + s) * MAXM + m;
    const float a = (gmax == -INFINITY) ? 0.f : __expf(p_m[b] - gmax);
    num = fmaf(a, p_acc[b * HEAD_DIM + d], num);
    den = fmaf(a, p_l[b], den);
  }
  out[((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + d] = num / den;
}

// ---------------------------------------------------------------- host ----
#define CHK(x)                                                        \
  do {                                                                \
    hipError_t e_ = (x);                                              \
    if (e_ != hipSuccess) {                                           \
      printf("HIP error %s at line %d\n", hipGetErrorString(e_), __LINE__); \
      exit(1);                                                        \
    }                                                                 \
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

// Copies a dense [S, HKV, D] host buffer into the padded [S, HKV, KV_ROW] one.
static void pad_layout(const std::vector<char>& dense, std::vector<_Float16>& out,
                       int S) {
  const _Float16* src = (const _Float16*)dense.data();
  for (int t = 0; t < S; ++t)
    for (int hh = 0; hh < NUM_KV_HEADS; ++hh)
      memcpy(&out[(size_t)t * KV_TOK + hh * KV_ROW],
             &src[((size_t)t * NUM_KV_HEADS + hh) * HEAD_DIM],
             HEAD_DIM * sizeof(_Float16));
}

int main(int argc, char** argv) {
  if (argc < 7) {
    printf("usage: %s S iters q.bin k.bin v.bin out.bin [ncopy]\n", argv[0]);
    return 1;
  }
  const int S = atoi(argv[1]);
  const int iters = atoi(argv[2]);
  const int ncopy = (argc > 7) ? atoi(argv[7]) : 4;
  const float scale = 1.f / sqrtf((float)HEAD_DIM);

  const size_t qn = (size_t)MAXM * NUM_Q_HEADS * HEAD_DIM;
  const size_t kvn_dense = (size_t)S * NUM_KV_HEADS * HEAD_DIM;
  const size_t kvn = (size_t)S * KV_TOK;

  auto hq = slurp(argv[3], qn * 2);
  auto hk = slurp(argv[4], kvn_dense * 2);
  auto hv = slurp(argv[5], kvn_dense * 2);

  std::vector<_Float16> pk(kvn, (_Float16)0.f), pv(kvn, (_Float16)0.f);
  pad_layout(hk, pk, S);
  pad_layout(hv, pv, S);

  _Float16 *dq, *dk[16], *dv[16];
  float *dacc, *dm, *dl, *dout;
  CHK(hipMalloc(&dq, qn * 2));
  CHK(hipMemcpy(dq, hq.data(), qn * 2, hipMemcpyHostToDevice));
  for (int c = 0; c < ncopy; ++c) {
    CHK(hipMalloc(&dk[c], kvn * 2));
    CHK(hipMalloc(&dv[c], kvn * 2));
    CHK(hipMemcpy(dk[c], pk.data(), kvn * 2, hipMemcpyHostToDevice));
    CHK(hipMemcpy(dv[c], pv.data(), kvn * 2, hipMemcpyHostToDevice));
  }
  const size_t accn = (size_t)NUM_Q_HEADS * NSEG * MAXM * HEAD_DIM;
  const size_t mln = (size_t)NUM_Q_HEADS * NSEG * MAXM;
  CHK(hipMalloc(&dacc, accn * 4));
  CHK(hipMalloc(&dm, mln * 4));
  CHK(hipMalloc(&dl, mln * 4));
  CHK(hipMalloc(&dout, qn * 4));

  dim3 grid(NSEG, NUM_Q_HEADS), block(HEAD_DIM);
  dim3 rgrid(MAXM, NUM_Q_HEADS);

  auto launch = [&](int c) {
    hipLaunchKernelGGL(decode_attn, grid, block, 0, 0, dq, dk[c], dv[c], dacc,
                       dm, dl, S, scale);
    hipLaunchKernelGGL(reduce_segments, rgrid, block, 0, 0, dacc, dm, dl, dout);
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
    for (int i = 0; i < 5; ++i) launch(i % ncopy);
    CHK(hipDeviceSynchronize());
    hipEvent_t a, b;
    CHK(hipEventCreate(&a));
    CHK(hipEventCreate(&b));
    CHK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) launch(i % ncopy);
    CHK(hipEventRecord(b));
    CHK(hipEventSynchronize(b));
    float ms = 0.f;
    CHK(hipEventElapsedTime(&ms, a, b));
    const double us = 1000.0 * ms / iters;
    const double bytes = 2.0 * (double)S * NUM_KV_HEADS * HEAD_DIM * 2.0;
    printf("TIME  M=%d NSEG=%d KV_PAD=%d TILE=%d  %.2f us  (%.1f GiB/s of KV)\n",
           MAXM, NSEG, KV_PAD, TILE, us, bytes / (us * 1e-6) / (1024.0 * 1024 * 1024));
  }
  printf("OK    M=%d NSEG=%d KV_PAD=%d S=%d\n", MAXM, NSEG, KV_PAD, S);
  return 0;
}
