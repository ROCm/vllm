// f3b — decode attention for gfx1151, memory-path redesign of f3a (v4).
//
// v4 kills the block epilogue.  v2 merged its 8 per-wave softmax streams with
// NWAVE serialised __syncthreads() rounds, a fixed per-block cost that made the
// kernel scale badly with NSEG (515 us at NSEG=16 vs 174 at NSEG=2).  Here each
// wave is simply its own output segment: it writes (acc, m, l) straight to
// global and reduce_segments merges NSEG*NWAVE of them.  Zero barriers, zero
// LDS in the whole kernel.
//
// v6 generalises v5 off D=256 without narrowing the load.  v5 fixed
// DPL = HEAD_DIM/WAVE = 8 so one row was one global_load_b128 per lane; at
// D=128 that formula gives 4 fp16 = b64 and throws away half of every load.
//
// Instead DPL stays at 8 and the *wave* is split: LPR = HEAD_DIM/8 lanes cover
// one row, and the WAVE/LPR = SUB lane-groups of a wave each walk their own
// token (brief option B, "two rows per lane").  D=256 -> LPR=32, SUB=1, byte
// for byte the v5 schedule.  D=128 -> LPR=16, SUB=2: still b128 per lane, two
// tokens in flight per wave instead of one.
//
// The SUB streams never have to meet: each lane-group is already an
// independent online-softmax stream, so it becomes its own output segment and
// reduce_segments merges NSEG*NWAVE*SUB of them.  Still zero barriers, zero
// LDS, and the causal mask stays per-lane because each lane knows its token.
//
// Each of the 8 waves in the block is an independent online-softmax stream over
// an interleaved subset of the segment's keys, so the inner loop has *no*
// barriers.  The 8 streams are merged through LDS once, at the end.
//
//   Q : [M, HQ, D]                 fp16
//   K : [S, HKV, D + KV_PAD]       fp16
//   V : [S, HKV, D + KV_PAD]       fp16
//   O : [M, HQ, D]                 fp32
//
// Causal mask: query m sees keys 0 .. (S - M) + m.  GQA: q head h -> kv head
// h / (HQ / HKV).  Softmax runs in the log2 domain (v_exp_f32).
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
#ifndef MAXM
#define MAXM 4
#endif
#ifndef NSEG
#define NSEG 16
#endif
#ifndef KV_PAD
#define KV_PAD 0
#endif
#ifndef KPW  // keys per wave per inner iteration (= loads in flight / 2)
#define KPW 4
#endif
#ifndef BLOCK
#define BLOCK 256
#endif
#ifndef NT  // non-temporal K/V loads
#define NT 0
#endif
#ifndef SETPRIO  // raise priority around the load burst
#define SETPRIO 0
#endif
#ifndef MUTATE  // negative control: 0 = off
#define MUTATE 0
#endif
#ifndef ILV  // 1 = segments interleave over the whole grid (keeps L2 locality)
#define ILV 1
#endif
#ifndef EPI  // 0 = serialised LDS rounds, 1 = LDS float atomics, 2 = wave->global
#define EPI 2
#endif

#define WAVE 32
#define NWAVE (BLOCK / WAVE)
#define DPL 8                  // fp16 per lane = one global_load_b128, always
#define LPR (HEAD_DIM / DPL)   // lanes that together cover one K/V row
#define SUB (WAVE / LPR)       // independent token streams per wave
#define NSEGT (NSEG * NWAVE * SUB)  // total output segments
#define KV_ROW (HEAD_DIM + KV_PAD)
#define KV_TOK (NUM_KV_HEADS * KV_ROW)
#define GQA (NUM_Q_HEADS / NUM_KV_HEADS)
#define LOG2E 1.44269504088896340736f

static_assert(HEAD_DIM % DPL == 0, "head dim must be a multiple of 8 fp16");
static_assert(LPR <= WAVE, "head dim > 8*WAVE needs more than one wave per row");
static_assert(WAVE % LPR == 0, "lanes per row must divide the wave evenly");
static_assert(KV_PAD % 8 == 0, "KV_PAD must keep rows 16B aligned");

typedef _Float16 h2v __attribute__((ext_vector_type(2)));
typedef float f4v __attribute__((ext_vector_type(4)));

__device__ __forceinline__ h2v as_h2(float x) {
  return __builtin_bit_cast(h2v, x);
}

__device__ __forceinline__ f4v load128(const _Float16* p) {
#if NT
  return __builtin_nontemporal_load((const f4v*)p);
#else
  return *(const f4v*)p;
#endif
}

__global__ __launch_bounds__(BLOCK) void decode_attn(
    const _Float16* __restrict__ q, const _Float16* __restrict__ k,
    const _Float16* __restrict__ v, float* __restrict__ p_acc,
    float* __restrict__ p_m, float* __restrict__ p_l, int S, float scale) {
  const int seg = blockIdx.x;
  const int h = blockIdx.y;
  const int kvh = h / GQA;
  const int tid = threadIdx.x;
  const int lane = tid & (WAVE - 1);
  const int wave = tid / WAVE;
  const int sub = lane / LPR;          // which token stream this lane serves
  const int dl = (lane % LPR) * DPL;   // first head dim owned by this lane

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
  // Every wave in the grid walks the KV in lockstep, KPW keys at a time, so all
  // blocks touch the same tokens at the same time and the 32 MiB of KV streams
  // past the cache exactly once.  Contiguous segments instead give each block
  // its own cursor and multiply the live working set by NSEG.
  const int gw = seg * NWAVE + wave;
  const int j1 = S;
  const int jstart = gw * (KPW * SUB);
  const int jstep = NSEG * NWAVE * KPW * SUB;
#else
  const int kpb = KPW * SUB;  // tokens a wave consumes per inner iteration
  const int sl0 = (S + NSEG - 1) / NSEG;
  const int seg_len = (sl0 + kpb - 1) / kpb * kpb;
  const int j1 = min(S, seg * seg_len + seg_len);
  const int jstart = seg * seg_len + wave * kpb;
  const int jstep = NWAVE * kpb;
#endif

  for (int jb = jstart; jb < j1; jb += jstep) {
#if SETPRIO
    __builtin_amdgcn_s_setprio(3);
#endif
    f4v kr[KPW], vr[KPW];
#pragma unroll
    for (int c = 0; c < KPW; ++c) {
      // Interleave the SUB streams so the lane-groups of a wave hit adjacent
      // tokens in the same cycle rather than SUB*KPW apart.
      int jj = jb + c * SUB + sub;
      jj = (jj < S) ? jj : (S - 1);  // clamp: OOB lanes are masked below
      const size_t off = (size_t)jj * KV_TOK + (size_t)kvh * KV_ROW + dl;
      kr[c] = load128(k + off);
      vr[c] = load128(v + off);
    }
#if SETPRIO
    __builtin_amdgcn_s_setprio(0);
#endif

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

    // reduce the partial dots over the LPR lanes that split one row; lanes of
    // different streams must not mix, and LPR is a power of two so an xor
    // butterfly below LPR never leaves the group
#pragma unroll
    for (int st = 1; st < LPR; st <<= 1)
#pragma unroll
      for (int c = 0; c < KPW; ++c)
#pragma unroll
        for (int m = 0; m < MAXM; ++m) s[c][m] += __shfl_xor(s[c][m], st, WAVE);

#pragma unroll
    for (int c = 0; c < KPW; ++c) {
      const int jj = jb + c * SUB + sub;
#pragma unroll
      for (int m = 0; m < MAXM; ++m) {
#if MUTATE == 1  // off-by-one causal mask
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

  // ---- each wave is its own output segment: no barriers, no LDS ----
  const size_t base =
      ((size_t)h * NSEGT + (seg * NWAVE + wave) * SUB + sub) * MAXM;
#pragma unroll
  for (int m = 0; m < MAXM; ++m) {
#pragma unroll
    for (int i = 0; i < DPL; ++i)
      p_acc[(base + m) * HEAD_DIM + dl + i] = acc[m][i];
    if (dl == 0) {
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
  for (int s = 0; s < NSEGT; ++s)
    gmax = fmaxf(gmax, p_m[((size_t)h * NSEGT + s) * MAXM + m]);

  float num = 0.f, den = 0.f;
  for (int s = 0; s < NSEGT; ++s) {
    const size_t b = ((size_t)h * NSEGT + s) * MAXM + m;
    const float a =
        (gmax == -INFINITY) ? 0.f : __builtin_amdgcn_exp2f(p_m[b] - gmax);
    num = fmaf(a, p_acc[b * HEAD_DIM + d], num);
    den = fmaf(a, p_l[b], den);
  }
  out[((size_t)m * NUM_Q_HEADS + h) * HEAD_DIM + d] = num / den;
}

// ---------------------------------------------------------------- host ----
#define CHK(x)                                                               \
  do {                                                                       \
    hipError_t e_ = (x);                                                     \
    if (e_ != hipSuccess) {                                                  \
      printf("HIP error %s at line %d\n", hipGetErrorString(e_), __LINE__);   \
      exit(1);                                                               \
    }                                                                        \
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
  const size_t accn = (size_t)NUM_Q_HEADS * NSEGT * MAXM * HEAD_DIM;
  const size_t mln = (size_t)NUM_Q_HEADS * NSEGT * MAXM;
  CHK(hipMalloc(&dacc, accn * 4));
  CHK(hipMalloc(&dm, mln * 4));
  CHK(hipMalloc(&dl, mln * 4));
  CHK(hipMalloc(&dout, qn * 4));

  dim3 grid(NSEG, NUM_Q_HEADS), block(BLOCK);
  dim3 rgrid(MAXM, NUM_Q_HEADS);

  auto launch = [&](int c) {
    hipLaunchKernelGGL(decode_attn, grid, block, 0, 0, dq, dk[c], dv[c], dacc,
                       dm, dl, S, scale);
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
    const double us_attn = timeit([&](int c) {
      hipLaunchKernelGGL(decode_attn, grid, block, 0, 0, dq, dk[c], dv[c], dacc,
                         dm, dl, S, scale);
    });
    const double us_red = timeit([&](int) {
      hipLaunchKernelGGL(reduce_segments, rgrid, dim3(HEAD_DIM), 0, 0, dacc, dm,
                         dl, dout);
    });
    const double us = timeit(launch);
    printf("SPLIT attn=%.2f us  reduce=%.2f us  total=%.2f us\n", us_attn,
           us_red, us);
    const double bytes = 2.0 * (double)S * NUM_KV_HEADS * HEAD_DIM * 2.0;
    printf("TIME v6 D=%d M=%d NSEG=%d KV_PAD=%d KPW=%d NT=%d SP=%d ILV=%d  %.2f us  (%.1f GiB/s of KV)\n",
           HEAD_DIM, MAXM, NSEG, KV_PAD, KPW, NT, SETPRIO, ILV, us,
           bytes / (us * 1e-6) / (1024.0 * 1024 * 1024));
  }
  printf("OK    D=%d M=%d NSEG=%d KV_PAD=%d S=%d SUB=%d\n", HEAD_DIM, MAXM,
         NSEG, KV_PAD, S, SUB);
  return 0;
}
