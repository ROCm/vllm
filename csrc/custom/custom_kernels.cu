
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <algorithm>

#if defined(__HIPCC__) && (defined(__gfx90a__) || defined(__gfx940__) || \
                           defined(__gfx941__) || defined(__gfx942__))
  #define __HIP__MI300_MI250__
#endif

#if defined(NDEBUG)
  #undef NDEBUG
  #include <assert.h>
  #define UNREACHABLE_CODE assert(false);
  #define NDEBUG
#else
  #define UNREACHABLE_CODE assert(false);
#endif

constexpr int WARP_SIZE = 64;

template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
  return __builtin_nontemporal_load(addr);
}

__device__ __forceinline__ float4 load_ntmprl(const float4* addr) {
  auto addr_alias = reinterpret_cast<const float*>(addr);
  auto dat0 = loadnt(addr_alias);
  auto dat1 = loadnt(addr_alias + 1);
  auto dat2 = loadnt(addr_alias + 2);
  auto dat3 = loadnt(addr_alias + 3);
  // auto dat0 = *(addr_alias);
  // auto dat1 = *(addr_alias+1);
  // auto dat2 = *(addr_alias+2);
  // auto dat3 = *(addr_alias+3);
  return make_float4(dat0, dat1, dat2, dat3);
}

// TBlock fetches entire rows of A, and entire col of B (K dimension); assume
// N=1 for time being grid is M/A_NUM_ROWS blocks
template <int NUM_A_ROWS_PER_BLOCK>
__global__ void LLGemm1_kernel(float4* af4, __half2* bf4, __half2* c,
                               const int K) {
  __shared__ float red_smem[NUM_A_ROWS_PER_BLOCK][WARP_SIZE];
  const int row_addr = blockIdx.x * NUM_A_ROWS_PER_BLOCK * K / 8;
  const int threadid = threadIdx.x;
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;
  const int qwarpid = threadid / 16;
  const int qthreadid = threadid % 16;
  float4 rowA_elem4[NUM_A_ROWS_PER_BLOCK];
  __half2 colB_elem4x, colB_elem4y, colB_elem4z, colB_elem4w;
  float4 sum4;  //[NUM_A_ROWS_PER_BLOCK];
  float acc[NUM_A_ROWS_PER_BLOCK] = {0.0};
  __half2 acch2;
  __half2 oval;

  // As we later use warp shuffle operations, we may have more threads in the
  // block than the actual available data, hence the if guard here.
  if (threadid * 8 < K) {
#pragma unroll
    for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
      // rowA_elem4[i] holds 8 * half numbers seen as a single float4.
      rowA_elem4[i] = load_ntmprl(&af4[row_addr + threadid + K / 8 * i]);
    }
  }

  colB_elem4x = bf4[threadid * 4 + 0];
  colB_elem4y = bf4[threadid * 4 + 1];
  colB_elem4z = bf4[threadid * 4 + 2];
  colB_elem4w = bf4[threadid * 4 + 3];

  __half2 Af2;
  __half2 Bf2;
  float2 S;

  auto Ah2ptr = reinterpret_cast<__half2*>(&rowA_elem4);
  __half2* ah2lptr;

#pragma unroll
  for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
    // Multiply-add on 8 half.
    ah2lptr = Ah2ptr + i * 4;
    Af2 = *(ah2lptr);
    acch2 = __hmul2(Af2, colB_elem4x);
    Af2 = *(ah2lptr + 1);
    acch2 = __hfma2(Af2, colB_elem4y, acch2);
    Af2 = *(ah2lptr + 2);
    acch2 = __hfma2(Af2, colB_elem4z, acch2);
    Af2 = *(ah2lptr + 3);
    acch2 = __hfma2(Af2, colB_elem4w, acch2);
    S = __half22float2(acch2);

    // See comment above concerning the if guard.
    if (threadid * 8 < K) {
      acc[i] = S.x + S.y;  // accumulation on float
    }
  }

// all reduce across warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
#pragma unroll
    for (int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++) {
      acc[i] += __shfl_xor(acc[i], mask);
    }
  }

  // Warp leaders store the data to shared memory.
  if (lane < NUM_A_ROWS_PER_BLOCK) {
    red_smem[lane][warp] = acc[lane];
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  if (qwarpid < NUM_A_ROWS_PER_BLOCK) {
    acc[qwarpid] = qthreadid < num_warps ? red_smem[qwarpid][qthreadid] : 0.f;
#pragma unroll
    for (int mask = 16 / 2; mask >= 1; mask /= 2) {
      acc[qwarpid] += __shfl_xor(acc[qwarpid], mask);
    }
    float oval2 = __shfl_xor(acc[qwarpid], 16);

    if (threadid % WARP_SIZE == 0 or threadid % WARP_SIZE == 32) {
      oval = __float22half2_rn(make_float2(acc[qwarpid], oval2));
      c[blockIdx.x * NUM_A_ROWS_PER_BLOCK / 2 + qwarpid / 2] = oval;
    }
  }
}

// define the kernel calling code:
// template <typename T>
void LLGemm1(void* in_a, void* in_b, void* out_c, const int M, const int K,
             cudaStream_t stream, const int rows_per_block = 4) {
  float4* af4 = reinterpret_cast<float4*>(in_a);
  auto* bf4 = reinterpret_cast<__half2*>(in_b);
  auto* c = reinterpret_cast<__half2*>(out_c);

  // NUM_TREADS need to be a multiple of WARP_SIZE, as we are using warp shuffle
  // operations.
  const int NUM_THREADS =
      K * 2 / 16 % WARP_SIZE == 0
          ? K * 2 / 16
          : K * 2 / 16 + (WARP_SIZE - K * 2 / 16 % WARP_SIZE);

  int NUM_BLOCKS = M / rows_per_block;

  if (rows_per_block == 2) {
    LLGemm1_kernel<2><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
  } else if (rows_per_block == 4) {
    LLGemm1_kernel<4><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
  } else if (rows_per_block == 8) {
    LLGemm1_kernel<8><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
  } else if (rows_per_block == 16) {
    LLGemm1_kernel<16><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
  } else {
    NUM_BLOCKS = M / 4;
    LLGemm1_kernel<4><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
  }

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

// instantiate the kernel template for T=float:
// template void AddGPUKernel<float>(float *in_a, float *in_b, float *out_c,
// const int M, const int K, cudaStream_t stream);

const unsigned int TILE_WIDTH = 32;

// Compute C = A * B
__global__ void matrixMultiplyShared(float* A, float* B, float* C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
  __shared__ float sA[TILE_WIDTH][TILE_WIDTH];  // Tile size of 32x32
  __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

  int Row = blockDim.y * blockIdx.y + threadIdx.y;
  int Col = blockDim.x * blockIdx.x + threadIdx.x;
  float Cvalue = 0.0;
  sA[threadIdx.y][threadIdx.x] = 0.0;
  sB[threadIdx.y][threadIdx.x] = 0.0;

  for (int ph = 0; ph < (((numAColumns - 1) / TILE_WIDTH) + 1); ph++) {
    if ((Row < numARows) && (threadIdx.x + (ph * TILE_WIDTH)) < numAColumns) {
      sA[threadIdx.y][threadIdx.x] =
          A[(Row * numAColumns) + threadIdx.x + (ph * TILE_WIDTH)];
    } else {
      sA[threadIdx.y][threadIdx.x] = 0.0;
    }
    if (Col < numBColumns && (threadIdx.y + ph * TILE_WIDTH) < numBRows) {
      sB[threadIdx.y][threadIdx.x] =
          B[(threadIdx.y + ph * TILE_WIDTH) * numBColumns + Col];
    } else {
      sB[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    for (int j = 0; j < TILE_WIDTH; ++j) {
      Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
    }
  }
  if (Row < numCRows && Col < numCColumns) {
    C[Row * numCColumns + Col] = Cvalue;
  }
}

void MMGPUKernel(float* in_a, float* in_b, float* out_c, int numARows,
                 int numAColumns, int numBRows, int numBColumns, int numCRows,
                 int numCColumns, cudaStream_t stream) {
  // Initialize the grid and block dimensions
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((numCColumns / TILE_WIDTH) + 1, (numCRows / TILE_WIDTH) + 1, 1);
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(
      in_a, in_b, out_c, numARows, numAColumns, numBRows, numBColumns, numCRows,
      numCColumns);

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

template <int nThreads_per_row, int CTA, int MT0, int MT1>
__global__ __launch_bounds__(512) void HGEMV_WFPerRow(
    int m, int n, const _Float16* A, int lda, const _Float16* x, _Float16* y) {
  int num_row_per_block = CTA / nThreads_per_row;
  int row_id = (blockIdx.x * num_row_per_block + threadIdx.y) * MT0;
  int inc = (gridDim.x * num_row_per_block) * MT0;

  while (row_id < m) {
    float2 sum2[MT0];

#pragma unroll
    for (int i = 0; i < MT0; ++i) {
      sum2[i] = {0.0, 0.0};
    }

    for (int j = threadIdx.x; j < n; j += (nThreads_per_row * MT1)) {
      bool is_active = j < n;
      if (is_active) {
        float2 x2[MT1 >> 1];
#pragma unroll
        for (int offset = 0; offset < MT1; offset += 2) {
          x2[offset >> 1] = {x[j + nThreads_per_row * offset],
                             x[j + nThreads_per_row * (offset + 1)]};
        }
        float2 a2[MT0][MT1 >> 1];
#pragma unroll
        for (int i = 0; i < MT0; i++) {
#pragma unroll
          for (int offset = 0; offset < MT1; offset += 2) {
            a2[i][offset >> 1] = {
                A[(row_id + i) * n + j + nThreads_per_row * offset],
                A[(row_id + i) * n + j + nThreads_per_row * (offset + 1)]};
          }
        }

#pragma unroll
        for (int i = 0; i < MT0; i++) {
#pragma unroll
          for (int offset = 0; offset < (MT1 >> 1); offset++) {
            sum2[i] += a2[i][offset] * x2[offset];
          }
        }
      }
    }
    float sum[MT0];
#pragma unroll
    for (int i = 0; i < MT0; i++) {
      sum[i] = sum2[i].x + sum2[i].y;
    }

#pragma unroll
    for (int i = 0; i < MT0; i++) {
#pragma unroll
      for (int offset = nThreads_per_row >> 1; offset >= 1;
           offset = offset >> 1) {
        sum[i] += __shfl_down(sum[i], offset, nThreads_per_row);
      }
    }
    if (threadIdx.x == 0) {
#pragma unroll
      for (int i = 0; i < MT0; i++) {
        y[row_id + i] = sum[i];
      }
    }
    row_id += inc;
  }
}

void LLGemmZZ(void* in_a, void* in_b, void* out_c, const int M, const int K,
              cudaStream_t stream, const int solidx = 0) {
  // m -> M, n-> K
  dim3 grid(1024);
  dim3 block(64, 8);
  if (solidx == 0) {
    HGEMV_WFPerRow<64, 512, 4, 8><<<grid, block, 0, stream>>>(
        M, K, reinterpret_cast<const _Float16*>(in_a), K,
        reinterpret_cast<const _Float16*>(in_b),
        reinterpret_cast<_Float16*>(out_c));
  } else if (solidx == 1) {
    HGEMV_WFPerRow<64, 512, 2, 8><<<grid, block, 0, stream>>>(
        M, K, reinterpret_cast<const _Float16*>(in_a), K,
        reinterpret_cast<const _Float16*>(in_b),
        reinterpret_cast<_Float16*>(out_c));
  } else if (solidx == 2) {
    HGEMV_WFPerRow<64, 512, 1, 8><<<grid, block, 0, stream>>>(
        M, K, reinterpret_cast<const _Float16*>(in_a), K,
        reinterpret_cast<const _Float16*>(in_b),
        reinterpret_cast<_Float16*>(out_c));
  } else {
    HGEMV_WFPerRow<64, 512, 4, 8><<<grid, block, 0, stream>>>(
        M, K, reinterpret_cast<const _Float16*>(in_a), K,
        reinterpret_cast<const _Float16*>(in_b),
        reinterpret_cast<_Float16*>(out_c));
  }
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

/////////////////////////////////////////////

using half8 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

/*template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
          return __builtin_nontemporal_load(addr);
          //return *((T*)addr);
}*/

#define THRDS 64
#define YTILE 2
#define WvPrGrp 16
#define A_CHUNK 8
#define UNRL 2
#define M 1
#define DTYPE half

#if defined(__HIP__MI300_MI250__)  // TODO: Add NAVI support

__global__ void wvSpltK_hf_m1_sml_(const int K, const int N, const DTYPE* B,
                                   const DTYPE* __restrict__ A, DTYPE* C,
                                   const int CuCount) {
  union bigType {
    DTYPE h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    __int128_t b128;
    half8 h8;
  };

  __shared__ half s[1024 * 32];

  uint64_t n = (blockIdx.x * WvPrGrp + threadIdx.y) * YTILE;

  for (uint32_t k = 0; k < min(K * M, 32 * 1024);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);
    if (k_in >= min(K * M, 32 * 1024)) break;
    ((bigType*)(&s[k_in]))->b128 = ((bigType*)(&A[k_in]))->b128;
  }
  __syncthreads();

  float sum[M][YTILE];

  while (n < N) {
    for (int i = 0; i < YTILE; i++)
      for (int m = 0; m < M; m++) sum[m][i] = 0;

    bigType bigA[M][UNRL];
    bigType bigB0[UNRL];
  #if (YTILE >= 2)
    bigType bigB1[UNRL];
  #endif
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        const half* B_ = &B[(n + 0) * K + k_];
        bigB0[k2].h8 = (loadnt((half8*)(&B_[0 * K])));
  #if (YTILE >= 2)
        bigB1[k2].h8 = (loadnt((half8*)(&B_[1 * K])));
  #endif
      }
      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // Fetch A activation matrix in interleaved fashion from LDS or memory
        for (int m = 0; m < M; m++) {
          bigA[m][k2] = *((const bigType*)(&(s[k_ + K * m])));
        }
      }

      // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
  #pragma unroll
        for (uint32_t m = 0; m < M; m++) {
          // Do the matrix multiplication of activation and weight matrix
          // - Remember the accumulation is happening for K-split of 64!
  #pragma unroll
          for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][0])
                : "0"(sum[m][0]), "v"(bigA[m][k2].f[b]), "v"(bigB0[k2].f[b]));

            //----------------------------------------------------
            // The following code with YTILE > 1
            //----------------------------------------------------
  #if (YTILE >= 2)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][1])
                : "0"(sum[m][1]), "v"(bigA[m][k2].f[b]), "v"(bigB1[k2].f[b]));
  #endif
          }
        }
      }
    }

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    for (int m = 0; m < M; m++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
      }
    }

    if (threadIdx.x == 63) {
      for (int m = 0; m < M; m++) {
        for (int i = 0; i < YTILE; i++) {
          C[n + i + m * N] = __float2half(sum[m][i]);
        }
      }
    }

    n += CuCount * WvPrGrp * YTILE;
  }
}

#else  // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support

__global__ void wvSpltK_hf_m1_sml_(const int K, const int N, const DTYPE* B,
                                   const DTYPE* __restrict__ A, DTYPE* C,
                                   const int CuCount){UNREACHABLE_CODE}

#endif  // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

#if defined(__HIP__MI300_MI250__)  // TODO: Add NAVI support

__global__ void wvSpltK_hf_m1_(const int K, const int N, const DTYPE* B,
                               const DTYPE* __restrict__ A, DTYPE* C,
                               const int CuCount) {
  union bigType {
    DTYPE h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    __int128_t b128;
    half8 h8;
  };

  //----------------------------------------------------
  // Reserving 64 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //	     then this is not goint to work!
  //----------------------------------------------------
  __shared__ half s[1024 * 32];

  //----------------------------------------------------
  // Computation of columns that need to be committed to memory!
  //----------------------------------------------------
  uint32_t commitColumn[YTILE];
  for (uint32_t i = 0; i < YTILE; i++) {
    commitColumn[i] = 1;
  }

  //----------------------------------------------------
  // Indexing function into the column of weight matrix B
  // Algorithm does 64 lane k-splitting / wave and uses
  // WG ID and Thread ID to find the index.
  //----------------------------------------------------
  uint64_t n = (blockIdx.x * WvPrGrp + threadIdx.y) * YTILE;

  // Check whether there will be fragmenation!
  // This will happen only for the last wave!
  if (n < N && (n + YTILE) >= N) {
    uint32_t startColumn = N - YTILE;
    for (uint32_t i = 0; i < (n - startColumn); i++) {
      commitColumn[i] = 0;
    }
    n = startColumn;
  }

  //----------------------------------------------------
  // Fetch the activation matrix to LDS
  // Loop iteration:
  // - Each thread (lane) is fetching 8 elements (A_Chunk)
  // - Each wave will fetch 64*8=> 512 elements
  // - Each WG will fetch 512 * 16 => 8K elements
  // - Then the WG will move to another 8 K elements
  // TODO: Logic below will only work when K is multiple of 8
  //----------------------------------------------------
  for (uint32_t k = 0; k < min(K * M, 32 * 1024);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    // Transpose of A implementation
    // uint32_t k_ot = (k_in / M) + (k_in % M) * K; // transopse for
    // bank-conflict-free readback

    if (k_in >= min(K * M, 32 * 1024)) break;

    ((bigType*)(&s[k_in]))->b128 = ((bigType*)(&A[k_in]))->b128;
    //((bigType*)(&s[k_ot]))->b128 = ((bigType*)(&A[k_in]))->b128;
  }
  __syncthreads();

  float sum[M][YTILE];

  //----------------------------------------------------
  // Each wave works on a single column of weight matrix.
  // There are 16 waves per WG, and hence, each WG is
  // working on 16 columns of weight matrix. Moreover,
  // we tile in column direction by YTILE, so when YTILE=1
  // the above math is right, however, when YTILE=2 then
  // each wave  will be working on 2 columns and WG will
  // be working on 32 columns.
  //
  // Top level loop that makes WGs persistent!
  // - WGs iterates across columns of weight matrix
  // - Each wave within WG works on a given column(s)
  // - After completing first set of columns, WGs start
  //   working on the next set of available columns
  //----------------------------------------------------
  while (n < N) {
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    for (int i = 0; i < YTILE; i++)
      for (int m = 0; m < M; m++) sum[m][i] = 0;

    bigType bigA[M][UNRL];
    bigType bigB0[UNRL];
  #if (YTILE >= 2)
    bigType bigB1[UNRL];
  #endif
  #if (YTILE >= 3)
    bigType bigB2[UNRL];
  #endif
  #if (YTILE >= 4)
    bigType bigB3[UNRL];
  #endif
  #if (YTILE >= 5)
    bigType bigB4[UNRL];
  #endif
  #if (YTILE >= 6)
    bigType bigB5[UNRL];
  #endif
  #if (YTILE >= 7)
    bigType bigB6[UNRL];
  #endif
  #if (YTILE >= 8)
    bigType bigB7[UNRL];
  #endif
  #if (YTILE >= 9)
    bigType bigB8[UNRL];
  #endif
  #if (YTILE >= 10)
    bigType bigB9[UNRL];
  #endif
  #if (YTILE >= 11)
    bigType bigB10[UNRL];
  #endif
    //----------------------------------------------------
    // Fetch weight matrix B in interleaved K-split!
    // - Each thread (lane) is fetching 8 elements (A_Chunk)
    // - Each wave will fetch 64*8=> 512 elements (1024B)
    // - YTILE represents the number of column being serviced
    //   by wave
    // - Loop for fetching weight matrix (B) are unrolled
    //
    // Fetch activation matrix A from LDS
    // - Loop for fetching activation matrix (A) are unrolled
    //
    // Finally, do the matrix multiplication in an unrolled
    // fashion. This provides lot of food for compiler
    // scheduling.
    //
    // TODO: Logic below will only work when K is multiple of 8
    //----------------------------------------------------
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // if (k_ >= K) break;
        // bool skip = (k_ >= K);
        // bool dummy = (k_ >= K);

        const half* B_ = &B[(n + 0) * K + k_];
        bigB0[k2].h8 = (loadnt((half8*)(&B_[0 * K])));
        //----------------------------------------------------
        // The following code with YTILE > 1 has to be deleted
        //----------------------------------------------------
  #if (YTILE >= 2)
        // if (n+1>=N) continue;
        bigB1[k2].h8 = (loadnt((half8*)(&B_[1 * K])));
  #endif
  #if (YTILE >= 3)
        // if (n+2>=N) continue;
        bigB2[k2].h8 = (loadnt((half8*)(&B_[2 * K])));
  #endif
  #if (YTILE >= 4)
        // if (n+3>=N) continue;
        bigB3[k2].h8 = (loadnt((half8*)(&B_[3 * K])));
  #endif
  #if (YTILE >= 5)
        // if (n+4>=N) continue;
        bigB4[k2].h8 = (loadnt((half8*)(&B_[4 * K])));
  #endif
  #if (YTILE >= 6)
        // if (n+5>=N) continue;
        bigB5[k2].h8 = (loadnt((half8*)(&B_[5 * K])));
  #endif
  #if (YTILE >= 7)
        // if (n+6>=N) continue;
        bigB6[k2].h8 = (loadnt((half8*)(&B_[6 * K])));
  #endif
  #if (YTILE >= 8)
        // if (n+7>=N) continue;
        bigB7[k2].h8 = (loadnt((half8*)(&B_[7 * K])));
  #endif
        /*
        #if (YTILE >= 9)
                        if (n+8>=N) continue; bigB8[k2].h8 =
        (loadnt((half8*)(&B_[8 * K]))); #endif #if (YTILE >= 10) if (n+9>=N)
        continue; bigB9[k2].h8 = (loadnt((half8*)(&B_[9 * K]))); #endif #if
        (YTILE >= 11) if (n+10>=N) continue; bigB10[k2].h8 =
        (loadnt((half8*)(&B_[10 * K]))); #endif
        */
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // Fetch A activation matrix in interleaved fashion from LDS or memory

        for (int m = 0; m < M; m++) {
          if (k_ + K * m < 32 * 1024)
            bigA[m][k2] = *((const bigType*)(&(s[k_ + K * m])));
          else
            bigA[m][k2] = *((const bigType*)(&(A[k_ + K * m])));
        }
      }

      // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
  #pragma unroll
        for (uint32_t m = 0; m < M; m++) {
          // Do the matrix multiplication of activation and weight matrix
          // - Remember the accumulation is happening for K-split of 64!
  #pragma unroll
          for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][0])
                : "0"(sum[m][0]), "v"(bigA[m][k2].f[b]), "v"(bigB0[k2].f[b]));

            //----------------------------------------------------
            // The following code with YTILE > 1
            //----------------------------------------------------
  #if (YTILE >= 2)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][1])
                : "0"(sum[m][1]), "v"(bigA[m][k2].f[b]), "v"(bigB1[k2].f[b]));
  #endif
  #if (YTILE >= 3)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][2])
                : "0"(sum[m][2]), "v"(bigA[m][k2].f[b]), "v"(bigB2[k2].f[b]));
  #endif
  #if (YTILE >= 4)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][3])
                : "0"(sum[m][3]), "v"(bigA[m][k2].f[b]), "v"(bigB3[k2].f[b]));
  #endif
  #if (YTILE >= 5)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][4])
                : "0"(sum[m][4]), "v"(bigA[m][k2].f[b]), "v"(bigB4[k2].f[b]));
  #endif
  #if (YTILE >= 6)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][5])
                : "0"(sum[m][5]), "v"(bigA[m][k2].f[b]), "v"(bigB5[k2].f[b]));
  #endif
  #if (YTILE >= 7)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][6])
                : "0"(sum[m][6]), "v"(bigA[m][k2].f[b]), "v"(bigB6[k2].f[b]));
  #endif
  #if (YTILE >= 8)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][7])
                : "0"(sum[m][7]), "v"(bigA[m][k2].f[b]), "v"(bigB7[k2].f[b]));
  #endif
  #if (YTILE >= 9)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][8])
                : "0"(sum[m][8]), "v"(bigA[m][k2].f[b]), "v"(bigB8[k2].f[b]));
  #endif
  #if (YTILE >= 10)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][9])
                : "0"(sum[m][9]), "v"(bigA[m][k2].f[b]), "v"(bigB9[k2].f[b]));
  #endif
  #if (YTILE >= 11)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][10])
                : "0"(sum[m][10]), "v"(bigA[m][k2].f[b]), "v"(bigB10[k2].f[b]));
  #endif
          }
        }
      }
    }

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    for (int m = 0; m < M; m++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
      }
    }

    if (threadIdx.x == 63) {
      for (int m = 0; m < M; m++) {
        for (int i = 0; i < YTILE; i++) {
          if (commitColumn[i]) C[n + i + m * N] = __float2half(sum[m][i]);
        }
      }
    }

    n += CuCount * WvPrGrp * YTILE;

    // if (threadIdx.x == 0)
    // n = atomicAdd(((unsigned int*)(C)), YTILE);
    // n = __shfl(n, 0, 64);

    // Check whether there will be fragmenation!
    // This will happen only for the last wave!
    if (n < N && (n + YTILE) >= N) {
      uint32_t startColumn = N - YTILE;
      for (uint32_t i = 0; i < (n - startColumn); i++) {
        commitColumn[i] = 0;
      }
      n = startColumn;
    }
  }
}

#else  // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support

__global__ void wvSpltK_hf_m1_(const int K, const int N, const DTYPE* B,
                               const DTYPE* __restrict__ A, DTYPE* C,
                               const int CuCount){UNREACHABLE_CODE}

#endif  // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

#undef YTILE
#undef UNRL
#undef M

#define YTILE 2
#define UNRL 2
#define M 2

#if defined(__HIP__MI300_MI250__)  // TODO: Add NAVI support

__global__ void wvSpltK_hf_m2_(const int K, const int N, const DTYPE* B,
                               const DTYPE* __restrict__ A, DTYPE* C,
                               const int CuCount) {
  union bigType {
    DTYPE h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    __int128_t b128;
    half8 h8;
  };

  //----------------------------------------------------
  // Reserving 64 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //	     then this is not goint to work!
  //----------------------------------------------------
  __shared__ half s[1024 * 32];

  //----------------------------------------------------
  // Computation of columns that need to be committed to memory!
  //----------------------------------------------------
  uint32_t commitColumn[YTILE];
  for (uint32_t i = 0; i < YTILE; i++) {
    commitColumn[i] = 1;
  }

  //----------------------------------------------------
  // Indexing function into the column of weight matrix B
  // Algorithm does 64 lane k-splitting / wave and uses
  // WG ID and Thread ID to find the index.
  //----------------------------------------------------
  uint64_t n = (blockIdx.x * WvPrGrp + threadIdx.y) * YTILE;

  // Check whether there will be fragmenation!
  // This will happen only for the last wave!
  if (n < N && (n + YTILE) >= N) {
    uint32_t startColumn = N - YTILE;
    for (uint32_t i = 0; i < (n - startColumn); i++) {
      commitColumn[i] = 0;
    }
    n = startColumn;
  }

  //----------------------------------------------------
  // Fetch the activation matrix to LDS
  // Loop iteration:
  // - Each thread (lane) is fetching 8 elements (A_Chunk)
  // - Each wave will fetch 64*8=> 512 elements
  // - Each WG will fetch 512 * 16 => 8K elements
  // - Then the WG will move to another 8 K elements
  // TODO: Logic below will only work when K is multiple of 8
  //----------------------------------------------------
  for (uint32_t k = 0; k < min(K * M, 32 * 1024);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    // Transpose of A implementation
    // uint32_t k_ot = (k_in / M) + (k_in % M) * K; // transopse for
    // bank-conflict-free readback

    if (k_in >= min(K * M, 32 * 1024)) break;

    ((bigType*)(&s[k_in]))->b128 = ((bigType*)(&A[k_in]))->b128;
    //((bigType*)(&s[k_ot]))->b128 = ((bigType*)(&A[k_in]))->b128;
  }
  __syncthreads();

  float sum[M][YTILE];

  //----------------------------------------------------
  // Each wave works on a single column of weight matrix.
  // There are 16 waves per WG, and hence, each WG is
  // working on 16 columns of weight matrix. Moreover,
  // we tile in column direction by YTILE, so when YTILE=1
  // the above math is right, however, when YTILE=2 then
  // each wave  will be working on 2 columns and WG will
  // be working on 32 columns.
  //
  // Top level loop that makes WGs persistent!
  // - WGs iterates across columns of weight matrix
  // - Each wave within WG works on a given column(s)
  // - After completing first set of columns, WGs start
  //   working on the next set of available columns
  //----------------------------------------------------
  while (n < N) {
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    for (int i = 0; i < YTILE; i++)
      for (int m = 0; m < M; m++) sum[m][i] = 0;

    bigType bigA[M][UNRL];
    bigType bigB0[UNRL];
  #if (YTILE >= 2)
    bigType bigB1[UNRL];
  #endif
  #if (YTILE >= 3)
    bigType bigB2[UNRL];
  #endif
  #if (YTILE >= 4)
    bigType bigB3[UNRL];
  #endif
  #if (YTILE >= 5)
    bigType bigB4[UNRL];
  #endif
  #if (YTILE >= 6)
    bigType bigB5[UNRL];
  #endif
  #if (YTILE >= 7)
    bigType bigB6[UNRL];
  #endif
  #if (YTILE >= 8)
    bigType bigB7[UNRL];
  #endif
  #if (YTILE >= 9)
    bigType bigB8[UNRL];
  #endif
  #if (YTILE >= 10)
    bigType bigB9[UNRL];
  #endif
  #if (YTILE >= 11)
    bigType bigB10[UNRL];
  #endif
    //----------------------------------------------------
    // Fetch weight matrix B in interleaved K-split!
    // - Each thread (lane) is fetching 8 elements (A_Chunk)
    // - Each wave will fetch 64*8=> 512 elements (1024B)
    // - YTILE represents the number of column being serviced
    //   by wave
    // - Loop for fetching weight matrix (B) are unrolled
    //
    // Fetch activation matrix A from LDS
    // - Loop for fetching activation matrix (A) are unrolled
    //
    // Finally, do the matrix multiplication in an unrolled
    // fashion. This provides lot of food for compiler
    // scheduling.
    //
    // TODO: Logic below will only work when K is multiple of 8
    //----------------------------------------------------
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // if (k_ >= K) break;
        // bool skip = (k_ >= K);
        // bool dummy = (k_ >= K);

        const half* B_ = &B[(n + 0) * K + k_];
        bigB0[k2].h8 = (loadnt((half8*)(&B_[0 * K])));
        //----------------------------------------------------
        // The following code with YTILE > 1 has to be deleted
        //----------------------------------------------------
  #if (YTILE >= 2)
        // if (n+1>=N) continue;
        bigB1[k2].h8 = (loadnt((half8*)(&B_[1 * K])));
  #endif
  #if (YTILE >= 3)
        // if (n+2>=N) continue;
        bigB2[k2].h8 = (loadnt((half8*)(&B_[2 * K])));
  #endif
  #if (YTILE >= 4)
        // if (n+3>=N) continue;
        bigB3[k2].h8 = (loadnt((half8*)(&B_[3 * K])));
  #endif
  #if (YTILE >= 5)
        // if (n+4>=N) continue;
        bigB4[k2].h8 = (loadnt((half8*)(&B_[4 * K])));
  #endif
  #if (YTILE >= 6)
        // if (n+5>=N) continue;
        bigB5[k2].h8 = (loadnt((half8*)(&B_[5 * K])));
  #endif
  #if (YTILE >= 7)
        // if (n+6>=N) continue;
        bigB6[k2].h8 = (loadnt((half8*)(&B_[6 * K])));
  #endif
  #if (YTILE >= 8)
        // if (n+7>=N) continue;
        bigB7[k2].h8 = (loadnt((half8*)(&B_[7 * K])));
  #endif
        /*
        #if (YTILE >= 9)
                        if (n+8>=N) continue; bigB8[k2].h8 =
        (loadnt((half8*)(&B_[8 * K]))); #endif #if (YTILE >= 10) if (n+9>=N)
        continue; bigB9[k2].h8 = (loadnt((half8*)(&B_[9 * K]))); #endif #if
        (YTILE >= 11) if (n+10>=N) continue; bigB10[k2].h8 =
        (loadnt((half8*)(&B_[10 * K]))); #endif
        */
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // Fetch A activation matrix in interleaved fashion from LDS or memory

        for (int m = 0; m < M; m++) {
          if (k_ + K * m < 32 * 1024)
            bigA[m][k2] = *((const bigType*)(&(s[k_ + K * m])));
          else
            bigA[m][k2] = *((const bigType*)(&(A[k_ + K * m])));
        }
      }

      // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
  #pragma unroll
        for (uint32_t m = 0; m < M; m++) {
          // Do the matrix multiplication of activation and weight matrix
          // - Remember the accumulation is happening for K-split of 64!
  #pragma unroll
          for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][0])
                : "0"(sum[m][0]), "v"(bigA[m][k2].f[b]), "v"(bigB0[k2].f[b]));

            //----------------------------------------------------
            // The following code with YTILE > 1
            //----------------------------------------------------
  #if (YTILE >= 2)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][1])
                : "0"(sum[m][1]), "v"(bigA[m][k2].f[b]), "v"(bigB1[k2].f[b]));
  #endif
  #if (YTILE >= 3)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][2])
                : "0"(sum[m][2]), "v"(bigA[m][k2].f[b]), "v"(bigB2[k2].f[b]));
  #endif
  #if (YTILE >= 4)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][3])
                : "0"(sum[m][3]), "v"(bigA[m][k2].f[b]), "v"(bigB3[k2].f[b]));
  #endif
  #if (YTILE >= 5)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][4])
                : "0"(sum[m][4]), "v"(bigA[m][k2].f[b]), "v"(bigB4[k2].f[b]));
  #endif
  #if (YTILE >= 6)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][5])
                : "0"(sum[m][5]), "v"(bigA[m][k2].f[b]), "v"(bigB5[k2].f[b]));
  #endif
  #if (YTILE >= 7)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][6])
                : "0"(sum[m][6]), "v"(bigA[m][k2].f[b]), "v"(bigB6[k2].f[b]));
  #endif
  #if (YTILE >= 8)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][7])
                : "0"(sum[m][7]), "v"(bigA[m][k2].f[b]), "v"(bigB7[k2].f[b]));
  #endif
  #if (YTILE >= 9)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][8])
                : "0"(sum[m][8]), "v"(bigA[m][k2].f[b]), "v"(bigB8[k2].f[b]));
  #endif
  #if (YTILE >= 10)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][9])
                : "0"(sum[m][9]), "v"(bigA[m][k2].f[b]), "v"(bigB9[k2].f[b]));
  #endif
  #if (YTILE >= 11)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][10])
                : "0"(sum[m][10]), "v"(bigA[m][k2].f[b]), "v"(bigB10[k2].f[b]));
  #endif
          }
        }
      }
    }

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    for (int m = 0; m < M; m++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
      }
    }

    if (threadIdx.x == 63) {
      for (int m = 0; m < M; m++) {
        for (int i = 0; i < YTILE; i++) {
          if (commitColumn[i]) C[n + i + m * N] = __float2half(sum[m][i]);
        }
      }
    }

    n += CuCount * WvPrGrp * YTILE;

    // if (threadIdx.x == 0)
    // n = atomicAdd(((unsigned int*)(C)), YTILE);
    // n = __shfl(n, 0, 64);

    // Check whether there will be fragmenation!
    // This will happen only for the last wave!
    if (n < N && (n + YTILE) >= N) {
      uint32_t startColumn = N - YTILE;
      for (uint32_t i = 0; i < (n - startColumn); i++) {
        commitColumn[i] = 0;
      }
      n = startColumn;
    }
  }
}

#else  // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support

__global__ void wvSpltK_hf_m2_(const int K, const int N, const DTYPE* B,
                               const DTYPE* __restrict__ A, DTYPE* C,
                               const int CuCount){UNREACHABLE_CODE}

#endif  // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

#undef YTILE
#undef UNRL
#undef M

#define YTILE 5
#define UNRL 2
#define M 3

#if defined(__HIP__MI300_MI250__)  // TODO: Add NAVI support

__global__ void wvSpltK_hf_m3_(const int K, const int N, const DTYPE* B,
                               const DTYPE* __restrict__ A, DTYPE* C,
                               const int CuCount) {
  union bigType {
    DTYPE h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    __int128_t b128;
    half8 h8;
  };

  //----------------------------------------------------
  // Reserving 64 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //	     then this is not goint to work!
  //----------------------------------------------------
  __shared__ half s[1024 * 32];

  //----------------------------------------------------
  // Computation of columns that need to be committed to memory!
  //----------------------------------------------------
  uint32_t commitColumn[YTILE];
  for (uint32_t i = 0; i < YTILE; i++) {
    commitColumn[i] = 1;
  }

  //----------------------------------------------------
  // Indexing function into the column of weight matrix B
  // Algorithm does 64 lane k-splitting / wave and uses
  // WG ID and Thread ID to find the index.
  //----------------------------------------------------
  uint64_t n = (blockIdx.x * WvPrGrp + threadIdx.y) * YTILE;

  // Check whether there will be fragmenation!
  // This will happen only for the last wave!
  if (n < N && (n + YTILE) >= N) {
    uint32_t startColumn = N - YTILE;
    for (uint32_t i = 0; i < (n - startColumn); i++) {
      commitColumn[i] = 0;
    }
    n = startColumn;
  }

  //----------------------------------------------------
  // Fetch the activation matrix to LDS
  // Loop iteration:
  // - Each thread (lane) is fetching 8 elements (A_Chunk)
  // - Each wave will fetch 64*8=> 512 elements
  // - Each WG will fetch 512 * 16 => 8K elements
  // - Then the WG will move to another 8 K elements
  // TODO: Logic below will only work when K is multiple of 8
  //----------------------------------------------------
  for (uint32_t k = 0; k < min(K * M, 32 * 1024);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    // Transpose of A implementation
    // uint32_t k_ot = (k_in / M) + (k_in % M) * K; // transopse for
    // bank-conflict-free readback

    if (k_in >= min(K * M, 32 * 1024)) break;

    ((bigType*)(&s[k_in]))->b128 = ((bigType*)(&A[k_in]))->b128;
    //((bigType*)(&s[k_ot]))->b128 = ((bigType*)(&A[k_in]))->b128;
  }
  __syncthreads();

  float sum[M][YTILE];

  //----------------------------------------------------
  // Each wave works on a single column of weight matrix.
  // There are 16 waves per WG, and hence, each WG is
  // working on 16 columns of weight matrix. Moreover,
  // we tile in column direction by YTILE, so when YTILE=1
  // the above math is right, however, when YTILE=2 then
  // each wave  will be working on 2 columns and WG will
  // be working on 32 columns.
  //
  // Top level loop that makes WGs persistent!
  // - WGs iterates across columns of weight matrix
  // - Each wave within WG works on a given column(s)
  // - After completing first set of columns, WGs start
  //   working on the next set of available columns
  //----------------------------------------------------
  while (n < N) {
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    for (int i = 0; i < YTILE; i++)
      for (int m = 0; m < M; m++) sum[m][i] = 0;

    bigType bigA[M][UNRL];
    bigType bigB0[UNRL];
  #if (YTILE >= 2)
    bigType bigB1[UNRL];
  #endif
  #if (YTILE >= 3)
    bigType bigB2[UNRL];
  #endif
  #if (YTILE >= 4)
    bigType bigB3[UNRL];
  #endif
  #if (YTILE >= 5)
    bigType bigB4[UNRL];
  #endif
  #if (YTILE >= 6)
    bigType bigB5[UNRL];
  #endif
  #if (YTILE >= 7)
    bigType bigB6[UNRL];
  #endif
  #if (YTILE >= 8)
    bigType bigB7[UNRL];
  #endif
  #if (YTILE >= 9)
    bigType bigB8[UNRL];
  #endif
  #if (YTILE >= 10)
    bigType bigB9[UNRL];
  #endif
  #if (YTILE >= 11)
    bigType bigB10[UNRL];
  #endif
    //----------------------------------------------------
    // Fetch weight matrix B in interleaved K-split!
    // - Each thread (lane) is fetching 8 elements (A_Chunk)
    // - Each wave will fetch 64*8=> 512 elements (1024B)
    // - YTILE represents the number of column being serviced
    //   by wave
    // - Loop for fetching weight matrix (B) are unrolled
    //
    // Fetch activation matrix A from LDS
    // - Loop for fetching activation matrix (A) are unrolled
    //
    // Finally, do the matrix multiplication in an unrolled
    // fashion. This provides lot of food for compiler
    // scheduling.
    //
    // TODO: Logic below will only work when K is multiple of 8
    //----------------------------------------------------
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // if (k_ >= K) break;
        // bool skip = (k_ >= K);
        // bool dummy = (k_ >= K);

        const half* B_ = &B[(n + 0) * K + k_];
        bigB0[k2].h8 = (loadnt((half8*)(&B_[0 * K])));
        //----------------------------------------------------
        // The following code with YTILE > 1 has to be deleted
        //----------------------------------------------------
  #if (YTILE >= 2)
        // if (n+1>=N) continue;
        bigB1[k2].h8 = (loadnt((half8*)(&B_[1 * K])));
  #endif
  #if (YTILE >= 3)
        // if (n+2>=N) continue;
        bigB2[k2].h8 = (loadnt((half8*)(&B_[2 * K])));
  #endif
  #if (YTILE >= 4)
        // if (n+3>=N) continue;
        bigB3[k2].h8 = (loadnt((half8*)(&B_[3 * K])));
  #endif
  #if (YTILE >= 5)
        // if (n+4>=N) continue;
        bigB4[k2].h8 = (loadnt((half8*)(&B_[4 * K])));
  #endif
  #if (YTILE >= 6)
        // if (n+5>=N) continue;
        bigB5[k2].h8 = (loadnt((half8*)(&B_[5 * K])));
  #endif
  #if (YTILE >= 7)
        // if (n+6>=N) continue;
        bigB6[k2].h8 = (loadnt((half8*)(&B_[6 * K])));
  #endif
  #if (YTILE >= 8)
        // if (n+7>=N) continue;
        bigB7[k2].h8 = (loadnt((half8*)(&B_[7 * K])));
  #endif
        /*
        #if (YTILE >= 9)
                        if (n+8>=N) continue; bigB8[k2].h8 =
        (loadnt((half8*)(&B_[8 * K]))); #endif #if (YTILE >= 10) if (n+9>=N)
        continue; bigB9[k2].h8 = (loadnt((half8*)(&B_[9 * K]))); #endif #if
        (YTILE >= 11) if (n+10>=N) continue; bigB10[k2].h8 =
        (loadnt((half8*)(&B_[10 * K]))); #endif
        */
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // Fetch A activation matrix in interleaved fashion from LDS or memory

        for (int m = 0; m < M; m++) {
          if (k_ + K * m < 32 * 1024)
            bigA[m][k2] = *((const bigType*)(&(s[k_ + K * m])));
          else
            bigA[m][k2] = *((const bigType*)(&(A[k_ + K * m])));
        }
      }

      // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
  #pragma unroll
        for (uint32_t m = 0; m < M; m++) {
          // Do the matrix multiplication of activation and weight matrix
          // - Remember the accumulation is happening for K-split of 64!
  #pragma unroll
          for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][0])
                : "0"(sum[m][0]), "v"(bigA[m][k2].f[b]), "v"(bigB0[k2].f[b]));

            //----------------------------------------------------
            // The following code with YTILE > 1
            //----------------------------------------------------
  #if (YTILE >= 2)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][1])
                : "0"(sum[m][1]), "v"(bigA[m][k2].f[b]), "v"(bigB1[k2].f[b]));
  #endif
  #if (YTILE >= 3)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][2])
                : "0"(sum[m][2]), "v"(bigA[m][k2].f[b]), "v"(bigB2[k2].f[b]));
  #endif
  #if (YTILE >= 4)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][3])
                : "0"(sum[m][3]), "v"(bigA[m][k2].f[b]), "v"(bigB3[k2].f[b]));
  #endif
  #if (YTILE >= 5)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][4])
                : "0"(sum[m][4]), "v"(bigA[m][k2].f[b]), "v"(bigB4[k2].f[b]));
  #endif
  #if (YTILE >= 6)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][5])
                : "0"(sum[m][5]), "v"(bigA[m][k2].f[b]), "v"(bigB5[k2].f[b]));
  #endif
  #if (YTILE >= 7)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][6])
                : "0"(sum[m][6]), "v"(bigA[m][k2].f[b]), "v"(bigB6[k2].f[b]));
  #endif
  #if (YTILE >= 8)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][7])
                : "0"(sum[m][7]), "v"(bigA[m][k2].f[b]), "v"(bigB7[k2].f[b]));
  #endif
  #if (YTILE >= 9)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][8])
                : "0"(sum[m][8]), "v"(bigA[m][k2].f[b]), "v"(bigB8[k2].f[b]));
  #endif
  #if (YTILE >= 10)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][9])
                : "0"(sum[m][9]), "v"(bigA[m][k2].f[b]), "v"(bigB9[k2].f[b]));
  #endif
  #if (YTILE >= 11)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][10])
                : "0"(sum[m][10]), "v"(bigA[m][k2].f[b]), "v"(bigB10[k2].f[b]));
  #endif
          }
        }
      }
    }

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    for (int m = 0; m < M; m++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
      }
    }

    if (threadIdx.x == 63) {
      for (int m = 0; m < M; m++) {
        for (int i = 0; i < YTILE; i++) {
          if (commitColumn[i]) C[n + i + m * N] = __float2half(sum[m][i]);
        }
      }
    }

    n += CuCount * WvPrGrp * YTILE;

    // if (threadIdx.x == 0)
    // n = atomicAdd(((unsigned int*)(C)), YTILE);
    // n = __shfl(n, 0, 64);

    // Check whether there will be fragmenation!
    // This will happen only for the last wave!
    if (n < N && (n + YTILE) >= N) {
      uint32_t startColumn = N - YTILE;
      for (uint32_t i = 0; i < (n - startColumn); i++) {
        commitColumn[i] = 0;
      }
      n = startColumn;
    }
  }
}

#else  // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support

__global__ void wvSpltK_hf_m3_(const int K, const int N, const DTYPE* B,
                               const DTYPE* __restrict__ A, DTYPE* C,
                               const int CuCount){UNREACHABLE_CODE}

#endif  // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

#undef YTILE
#undef UNRL
#undef M

#define YTILE 7
#define UNRL 1
#define M 4

#if defined(__HIP__MI300_MI250__)  // TODO: Add NAVI support

__global__ void wvSpltK_hf_m4_(const int K, const int N, const DTYPE* B,
                               const DTYPE* __restrict__ A, DTYPE* C,
                               const int CuCount) {
  union bigType {
    DTYPE h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    __int128_t b128;
    half8 h8;
  };

  //----------------------------------------------------
  // Reserving 64 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //	     then this is not goint to work!
  //----------------------------------------------------
  __shared__ half s[1024 * 32];

  //----------------------------------------------------
  // Computation of columns that need to be committed to memory!
  //----------------------------------------------------
  uint32_t commitColumn[YTILE];
  for (uint32_t i = 0; i < YTILE; i++) {
    commitColumn[i] = 1;
  }

  //----------------------------------------------------
  // Indexing function into the column of weight matrix B
  // Algorithm does 64 lane k-splitting / wave and uses
  // WG ID and Thread ID to find the index.
  //----------------------------------------------------
  uint64_t n = (blockIdx.x * WvPrGrp + threadIdx.y) * YTILE;

  // Check whether there will be fragmenation!
  // This will happen only for the last wave!
  if (n < N && (n + YTILE) >= N) {
    uint32_t startColumn = N - YTILE;
    for (uint32_t i = 0; i < (n - startColumn); i++) {
      commitColumn[i] = 0;
    }
    n = startColumn;
  }

  //----------------------------------------------------
  // Fetch the activation matrix to LDS
  // Loop iteration:
  // - Each thread (lane) is fetching 8 elements (A_Chunk)
  // - Each wave will fetch 64*8=> 512 elements
  // - Each WG will fetch 512 * 16 => 8K elements
  // - Then the WG will move to another 8 K elements
  // TODO: Logic below will only work when K is multiple of 8
  //----------------------------------------------------
  for (uint32_t k = 0; k < min(K * M, 32 * 1024);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    // Transpose of A implementation
    // uint32_t k_ot = (k_in / M) + (k_in % M) * K; // transopse for
    // bank-conflict-free readback

    if (k_in >= min(K * M, 32 * 1024)) break;

    ((bigType*)(&s[k_in]))->b128 = ((bigType*)(&A[k_in]))->b128;
    //((bigType*)(&s[k_ot]))->b128 = ((bigType*)(&A[k_in]))->b128;
  }
  __syncthreads();

  float sum[M][YTILE];

  //----------------------------------------------------
  // Each wave works on a single column of weight matrix.
  // There are 16 waves per WG, and hence, each WG is
  // working on 16 columns of weight matrix. Moreover,
  // we tile in column direction by YTILE, so when YTILE=1
  // the above math is right, however, when YTILE=2 then
  // each wave  will be working on 2 columns and WG will
  // be working on 32 columns.
  //
  // Top level loop that makes WGs persistent!
  // - WGs iterates across columns of weight matrix
  // - Each wave within WG works on a given column(s)
  // - After completing first set of columns, WGs start
  //   working on the next set of available columns
  //----------------------------------------------------
  while (n < N) {
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    for (int i = 0; i < YTILE; i++)
      for (int m = 0; m < M; m++) sum[m][i] = 0;

    bigType bigA[M][UNRL];
    bigType bigB0[UNRL];
  #if (YTILE >= 2)
    bigType bigB1[UNRL];
  #endif
  #if (YTILE >= 3)
    bigType bigB2[UNRL];
  #endif
  #if (YTILE >= 4)
    bigType bigB3[UNRL];
  #endif
  #if (YTILE >= 5)
    bigType bigB4[UNRL];
  #endif
  #if (YTILE >= 6)
    bigType bigB5[UNRL];
  #endif
  #if (YTILE >= 7)
    bigType bigB6[UNRL];
  #endif
  #if (YTILE >= 8)
    bigType bigB7[UNRL];
  #endif
  #if (YTILE >= 9)
    bigType bigB8[UNRL];
  #endif
  #if (YTILE >= 10)
    bigType bigB9[UNRL];
  #endif
  #if (YTILE >= 11)
    bigType bigB10[UNRL];
  #endif
    //----------------------------------------------------
    // Fetch weight matrix B in interleaved K-split!
    // - Each thread (lane) is fetching 8 elements (A_Chunk)
    // - Each wave will fetch 64*8=> 512 elements (1024B)
    // - YTILE represents the number of column being serviced
    //   by wave
    // - Loop for fetching weight matrix (B) are unrolled
    //
    // Fetch activation matrix A from LDS
    // - Loop for fetching activation matrix (A) are unrolled
    //
    // Finally, do the matrix multiplication in an unrolled
    // fashion. This provides lot of food for compiler
    // scheduling.
    //
    // TODO: Logic below will only work when K is multiple of 8
    //----------------------------------------------------
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // if (k_ >= K) break;
        // bool skip = (k_ >= K);
        // bool dummy = (k_ >= K);

        const half* B_ = &B[(n + 0) * K + k_];
        bigB0[k2].h8 = (loadnt((half8*)(&B_[0 * K])));
        //----------------------------------------------------
        // The following code with YTILE > 1 has to be deleted
        //----------------------------------------------------
  #if (YTILE >= 2)
        // if (n+1>=N) continue;
        bigB1[k2].h8 = (loadnt((half8*)(&B_[1 * K])));
  #endif
  #if (YTILE >= 3)
        // if (n+2>=N) continue;
        bigB2[k2].h8 = (loadnt((half8*)(&B_[2 * K])));
  #endif
  #if (YTILE >= 4)
        // if (n+3>=N) continue;
        bigB3[k2].h8 = (loadnt((half8*)(&B_[3 * K])));
  #endif
  #if (YTILE >= 5)
        // if (n+4>=N) continue;
        bigB4[k2].h8 = (loadnt((half8*)(&B_[4 * K])));
  #endif
  #if (YTILE >= 6)
        // if (n+5>=N) continue;
        bigB5[k2].h8 = (loadnt((half8*)(&B_[5 * K])));
  #endif
  #if (YTILE >= 7)
        // if (n+6>=N) continue;
        bigB6[k2].h8 = (loadnt((half8*)(&B_[6 * K])));
  #endif
  #if (YTILE >= 8)
        // if (n+7>=N) continue;
        bigB7[k2].h8 = (loadnt((half8*)(&B_[7 * K])));
  #endif
        /*
        #if (YTILE >= 9)
                        if (n+8>=N) continue; bigB8[k2].h8 =
        (loadnt((half8*)(&B_[8 * K]))); #endif #if (YTILE >= 10) if (n+9>=N)
        continue; bigB9[k2].h8 = (loadnt((half8*)(&B_[9 * K]))); #endif #if
        (YTILE >= 11) if (n+10>=N) continue; bigB10[k2].h8 =
        (loadnt((half8*)(&B_[10 * K]))); #endif
        */
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // Fetch A activation matrix in interleaved fashion from LDS or memory

        for (int m = 0; m < M; m++) {
          if (k_ + K * m < 32 * 1024)
            bigA[m][k2] = *((const bigType*)(&(s[k_ + K * m])));
          else
            bigA[m][k2] = *((const bigType*)(&(A[k_ + K * m])));
        }
      }

      // Do the matrix multiplication in interleaved manner
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
  #pragma unroll
        for (uint32_t m = 0; m < M; m++) {
          // Do the matrix multiplication of activation and weight matrix
          // - Remember the accumulation is happening for K-split of 64!
  #pragma unroll
          for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][0])
                : "0"(sum[m][0]), "v"(bigA[m][k2].f[b]), "v"(bigB0[k2].f[b]));

            //----------------------------------------------------
            // The following code with YTILE > 1
            //----------------------------------------------------
  #if (YTILE >= 2)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][1])
                : "0"(sum[m][1]), "v"(bigA[m][k2].f[b]), "v"(bigB1[k2].f[b]));
  #endif
  #if (YTILE >= 3)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][2])
                : "0"(sum[m][2]), "v"(bigA[m][k2].f[b]), "v"(bigB2[k2].f[b]));
  #endif
  #if (YTILE >= 4)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][3])
                : "0"(sum[m][3]), "v"(bigA[m][k2].f[b]), "v"(bigB3[k2].f[b]));
  #endif
  #if (YTILE >= 5)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][4])
                : "0"(sum[m][4]), "v"(bigA[m][k2].f[b]), "v"(bigB4[k2].f[b]));
  #endif
  #if (YTILE >= 6)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][5])
                : "0"(sum[m][5]), "v"(bigA[m][k2].f[b]), "v"(bigB5[k2].f[b]));
  #endif
  #if (YTILE >= 7)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][6])
                : "0"(sum[m][6]), "v"(bigA[m][k2].f[b]), "v"(bigB6[k2].f[b]));
  #endif
  #if (YTILE >= 8)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][7])
                : "0"(sum[m][7]), "v"(bigA[m][k2].f[b]), "v"(bigB7[k2].f[b]));
  #endif
  #if (YTILE >= 9)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][8])
                : "0"(sum[m][8]), "v"(bigA[m][k2].f[b]), "v"(bigB8[k2].f[b]));
  #endif
  #if (YTILE >= 10)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][9])
                : "0"(sum[m][9]), "v"(bigA[m][k2].f[b]), "v"(bigB9[k2].f[b]));
  #endif
  #if (YTILE >= 11)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][10])
                : "0"(sum[m][10]), "v"(bigA[m][k2].f[b]), "v"(bigB10[k2].f[b]));
  #endif
          }
        }
      }
    }

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    for (int m = 0; m < M; m++) {
      for (int y = 0; y < YTILE; y++) {
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
        asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
            : "=v"(sum[m][y])
            : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
      }
    }

    if (threadIdx.x == 63) {
      for (int m = 0; m < M; m++) {
        for (int i = 0; i < YTILE; i++) {
          if (commitColumn[i]) C[n + i + m * N] = __float2half(sum[m][i]);
        }
      }
    }

    n += CuCount * WvPrGrp * YTILE;

    // if (threadIdx.x == 0)
    // n = atomicAdd(((unsigned int*)(C)), YTILE);
    // n = __shfl(n, 0, 64);

    // Check whether there will be fragmenation!
    // This will happen only for the last wave!
    if (n < N && (n + YTILE) >= N) {
      uint32_t startColumn = N - YTILE;
      for (uint32_t i = 0; i < (n - startColumn); i++) {
        commitColumn[i] = 0;
      }
      n = startColumn;
    }
  }
}

#else  // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support

__global__ void wvSpltK_hf_m4_(const int K, const int N, const DTYPE* B,
                               const DTYPE* __restrict__ A, DTYPE* C,
                               const int CuCount) {
  UNREACHABLE_CODE
}

#endif  // defined(__HIP__MI300_MI250__) TODO: Add NAVI support



#undef M
#undef YTILE
#undef UNRL
#define UNRL 1
//#define M_BLOCK 4

template <int M_BLOCK, int YTILE>
__global__ void 
__launch_bounds__(WvPrGrp * THRDS)
wvSpltK_fsdMoe_hf_(
                               const DTYPE* __restrict__ A, 
                               const DTYPE* __restrict__ B, 
			       DTYPE* C,
                               const float* __restrict__ topk_weights, 
                               const int* __restrict__ topk_ids, 
                               const int* __restrict__ sorted_token_ids, 
                               const int* __restrict__ expert_ids, 
                               const int* __restrict__ num_tokens_post_padded, 
		               const int M_in, const int N, const int K, const int E, 
                               const int num_valid_tokens, 
		               const int stride_am,
                               const int stride_ak,
                               const int stride_be,
                               const int stride_bk,
                               const int stride_bn,
                               const int stride_cm,
                               const int stride_cn,
		               const bool mul_routed_weight, 
		               const int top_k,
		               const int CuCount 
			       ) { 
  bool PCML = (K * M_in > 32*1024); 
  union bigType {
    DTYPE h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half8 h8;
  };

  __shared__ half s[1024 * 32];

  uint32_t commitColumn[YTILE];
  for (uint32_t i = 0; i < YTILE; i++) {
    commitColumn[i] = 1;
  }

  uint32_t n = (blockIdx.x * WvPrGrp + threadIdx.y) * YTILE;

  if (n < N && (n + YTILE) >= N) {
    uint32_t startColumn = N - YTILE;
    for (uint32_t i = 0; i < (n - startColumn); i++) {
      commitColumn[i] = 0;
    }
    n = startColumn;
  }

  if (!PCML) {
  for (uint32_t k = 0; k < min(K * M_in, 32 * 1024);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    if (k_in >= min(K * M_in, 32 * 1024)) break;

    *((bigType*)(&s[k_in])) = *((bigType*)(&A[k_in]));
  }
  __syncthreads();
  }

  int YW = (YTILE * WvPrGrp);
  int TWC = (THRDS * WvPrGrp * A_CHUNK);
  int TUC = (THRDS * UNRL * A_CHUNK);
  uint32_t kBase = 0;
  //find biggest k size that fits in LDS
  uint32_t kFit = (32*1024)/M_BLOCK;
  //kFit = (kFit%TWC==0) ? kFit : (kFit-kFit%TWC+TWC); //round up to multiple of TUC
  kFit = (kFit%TUC==0) ? kFit : (kFit-kFit%TUC); //round down to multiple of TUC
  //if (kFit == 0) kFit = TUC;
  kFit = min(kFit, K);

  //if (kFit < TUC) PCML = false; 

  float sum[M_BLOCK][YTILE];

  //TRITON
  //offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  //offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
  //token_mask = offs_token < num_valid_tokens
  int offs_token[M_BLOCK];
  bool token_mask[M_BLOCK];  // add to A[] /top_k*k
  int off_experts;  // add to B[] *K*N loads

  uint32_t Nrndp = (N%YW==0) ? N : (N-N%YW+YW); // Note: All waves in the group need to stay alive to the bitter end, just in case they're needed for cooperative loading of next chunk of A[] into LDS. Such Zomby waves are prevented from doing any real work with continues in the loop below.   
  if (!PCML) Nrndp = N; //unless its not peicmeal
  while (n < Nrndp) {
    kBase = 0;
    for (uint32_t e = 0; e < num_tokens_post_padded[0]; e+=M_BLOCK) { 
    kBase = 0;
    
    for (int m=0; m<M_BLOCK; m++) {
	// get the list of Ms corresponding to this M_BLOCK
        offs_token[m] = sorted_token_ids[e+m];
        token_mask[m] = offs_token[m] < num_valid_tokens;
    }
    
    //set the expert for this M_BLOCK
    off_experts = expert_ids[e/M_BLOCK];
    
    for (int i = 0; i < YTILE; i++)
      for (int m = 0; m < M_BLOCK; m++) 
	      sum[m][i] = 0;

    bigType bigA[M_BLOCK][UNRL];
    bigType bigB[YTILE][UNRL];
    //----------------------------------------------------
    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
    if (PCML) {
        if ((k1 == 0) || (k1 == kBase + kFit)) { // load next chunk of A[] to LDS
                if (k1 != 0) kBase += kFit;
		__syncthreads();
                for (uint32_t k = 0; k < kFit; k += TWC) {
	            uint32_t kOff = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);    
                    if (kBase + kOff >= K) break;
                    if (kOff >= kFit) break;
                    for (uint32_t m = 0; m < M_BLOCK; m++) {
		      if (!token_mask[m]) continue;
                      uint32_t k_in = kBase + (offs_token[m]/top_k) * K    + kOff;
                      uint32_t k_ot =         m                     * kFit + kOff;
 		      *((bigType*)(&s[k_ot])) = *((bigType*)(&A[k_in]));
		    }
		}
                __syncthreads();
	}
    } 
    
    // kept alive just to participate in A[] loads
    if (n >= N) continue;

#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // load only 1 column of weights, despite the moe-gate, made possible by expert list.
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=0; y<YTILE; y++)
	    bigB[y][k2].h8 = (loadnt((half8*)(&B_[y * K])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

        // Fetch A activation matrix in interleaved fashion from LDS or memory

        for (int m = 0; m < M_BLOCK; m++) 
	{
	  if (!token_mask[m]) continue;
        if (PCML) {
          //bigA[m][k2] = *((const bigType*)(&(s[k_-kBase + kFit*m])));
	  // skip A[] fetches for Ms that are disabled
          bigA[m][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
        } else {
	  int aidx = k_ + (offs_token[m]/top_k) * K;
          if (aidx + A_CHUNK <= 32 * 1024)
            bigA[m][k2] = *((const bigType*)(&(s[aidx])));
          else
            bigA[m][k2] = *((const bigType*)(&(A[aidx])));
        }
        }
      }

      // Do the matrix multiplication in interleaved manner
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
#pragma unroll
        for (uint32_t m = 0; m < M_BLOCK; m++) {
	  // skip compute for Ms that are disabled
	  if (!token_mask[m]) continue;
          // Do the matrix multiplication of activation and weight matrix
          // - Remember the accumulation is happening for K-split of 64!
#pragma unroll
          for (uint32_t b = 0; b < A_CHUNK / 2; b++)
           for (int y=0; y<YTILE; y++)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][y])
                : "0"(sum[m][y]), "v"(bigA[m][k2].f[b]), "v"(bigB[y][k2].f[b]));
        }
      }
    }

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    for (int m = 0; m < M_BLOCK; m++) {
      // skip reduce for Ms that are disabled
      if (!token_mask[m]) continue;

      for (int y = 0; y < YTILE; y++) {
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
              : "=v"(sum[m][y])
              : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
              : "=v"(sum[m][y])
              : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
              : "=v"(sum[m][y])
              : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
              : "=v"(sum[m][y])
              : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
              : "=v"(sum[m][y])
              : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
              : "=v"(sum[m][y])
              : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
       }
    }

    if (threadIdx.x == 63) {
      for (int m = 0; m < M_BLOCK; m++) {
        for (int i = 0; i < YTILE; i++) 
	{
	  // skip write out for Ms that are disabled
	  if (!token_mask[m]) continue;
  	  if (mul_routed_weight)
              sum[m][i] *= token_mask[m] ? topk_weights[offs_token[m]/top_k] : 0;
          int oidx = n + i + offs_token[m] * N;
          if (commitColumn[i]) 
		  C[oidx] = __float2half(sum[m][i]);
        }
      }
    }
    }

    n += CuCount * WvPrGrp * YTILE;

    // Check whether there will be fragmenation!
    // This will happen only for the last wave!
    if (n < N && (n + YTILE) >= N) {
      uint32_t startColumn = N - YTILE;
      for (uint32_t i = 0; i < (n - startColumn); i++) {
        commitColumn[i] = 0;
      }
      n = startColumn;
    }
  }
}

#define mfmaTILEn 16 
#define mfmaTILEk 4
//#undef WvPrGrp
//#define WvPrGrp 8
#define USEMFMA
//#define PIPELINED_33334x
//#define PIPELINED_556x
#define PIPELINED4x

template <int M_BLOCK, int YTILE>
__global__ void 
__launch_bounds__(WvPrGrp * THRDS)
wvSpltK_fsdMoe_hf_mfma16_(
                               const DTYPE* __restrict__ A, 
                               const DTYPE* __restrict__ B, 
			       DTYPE* C,
                               const float* __restrict__ topk_weights, 
                               const int* __restrict__ topk_ids, 
                               const int* __restrict__ sorted_token_ids, 
                               const int* __restrict__ expert_ids, 
                               const int* __restrict__ num_tokens_post_padded, 
		               const int M_in, const int N, const int K, const int E, 
                               const int num_valid_tokens, 
		               const int stride_am,
                               const int stride_ak,
                               const int stride_be,
                               const int stride_bk,
                               const int stride_bn,
                               const int stride_cm,
                               const int stride_cn,
		               const bool mul_routed_weight, 
		               const int top_k,
		               const int CuCount 
			       ) { 

using halfCxT = __attribute__((__vector_size__(mfmaTILEn * A_CHUNK / 2 * sizeof(float)))) float;
using halfC   = __attribute__((__vector_size__(A_CHUNK / 2 * sizeof(float)))) float;
using halfT   = __attribute__((__vector_size__(mfmaTILEk / 2 * sizeof(float)))) float;
				     
bool PCML = true;//(K * M_in > 32*1024); 
  union bigType {
    DTYPE h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half8 h8;
    int i[A_CHUNK / 2];
    long int l[A_CHUNK / 4];
    halfT hT[A_CHUNK / mfmaTILEk];
    halfC hC;
  };
  union bigTypeXt{
	bigType B[mfmaTILEn];    
	halfCxT  hCT;
  };


  __shared__ half s[1024 * 32];

  uint32_t commitColumn[YTILE];
  for (uint32_t i = 0; i < YTILE; i++) {
    commitColumn[i] = 1;
  }

  int ETILE = (CuCount * WvPrGrp ) / (N/YTILE); // bump up etile to fill machine
  if (ETILE < 1) ETILE = 1;                //TODO: what is best default min ETILE? 
  if (M_in >= 128) ETILE = min(M_in/64, 15); // Heuristic: Add an ETILE for every 64 Ms

  const int num_tblk = num_tokens_post_padded[0] / M_BLOCK;
  
  // its worth spending time trying to load balance for this num_tokens...
  if ((CuCount/(ETILE*2) > 0) && (num_tblk>0))// TODO: make sure all overflow/inf conditions are avoided 
  {
    int nPrRnd0 = ((CuCount/(ETILE))*WvPrGrp)*YTILE; 
    int nRnds0 = (N + nPrRnd0 - 1 ) / nPrRnd0; 
    int tRnds0 = (num_tblk + (ETILE) - 1) / (ETILE);
    int rnds0 = nRnds0 * tRnds0;

    int nPrRnd1n = ((CuCount/(ETILE/2))*WvPrGrp)*YTILE; 
    int nRnds1n = (N + nPrRnd1n - 1 ) / nPrRnd1n; 
    int tRnds1n = (num_tblk + (ETILE/2) - 1) / (ETILE/2);
    int rnds1n = nRnds1n * tRnds1n;

    int nPrRnd1p = ((CuCount/(ETILE*2))*WvPrGrp)*YTILE; 
    int nRnds1p = (N + nPrRnd1p - 1 ) / nPrRnd1p; 
    int tRnds1p = (num_tblk + (ETILE*2) - 1) / (ETILE*2);
    int rnds1p = nRnds1p * tRnds1p;
    
    int etl = ETILE;
    if (rnds0 > rnds1n)  { etl = ETILE/2; rnds0 = rnds1n; }
    if (rnds0 > rnds1p)  { etl = ETILE*2; rnds0 = rnds1p; }
    ETILE = etl;
  }

  uint32_t n = ((blockIdx.x/ETILE) * WvPrGrp + threadIdx.y) * YTILE;

/*  if (n < N && (n + YTILE) >= N) {
    uint32_t startColumn = N - YTILE;
    for (uint32_t i = 0; i < (n - startColumn); i++) {
      commitColumn[i] = 0;
    }
    n = startColumn;
  }*/

  if (!PCML) {
  for (uint32_t k = 0; k < min(K * M_in, 32 * 1024);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

    if (k_in >= min(K * M_in, 32 * 1024)) break;

    *((bigType*)(&s[k_in])) = *((bigType*)(&A[k_in]));
  }
  __syncthreads();
  }

  int YW = (YTILE * WvPrGrp);
  int TWC = (THRDS * WvPrGrp * A_CHUNK);
  int TUC = (THRDS * UNRL * A_CHUNK);
  uint32_t kBase = 0;
  //find biggest k size that fits in LDS
  uint32_t kFit = (32*1024)/M_BLOCK;
  //kFit = (kFit%TWC==0) ? kFit : (kFit-kFit%TWC+TWC); //round up to multiple of TUC
  kFit = (kFit%TUC==0) ? kFit : (kFit-kFit%TUC); //round down to multiple of TUC
  //if (kFit == 0) kFit = TUC;
  kFit = min(kFit, K);

#ifdef USEMFMA
    using float4_ = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
    float4_ sum4;
#else
  float sum[M_BLOCK][YTILE];
#endif

  //TRITON
  //offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  //offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
  //token_mask = offs_token < num_valid_tokens
  uint32_t offs_token[M_BLOCK];
  bool token_mask[M_BLOCK];  // add to A[] /top_k*k
  uint32_t off_experts;  // add to B[] *K*N loads

  int kShfl = A_CHUNK * THRDS * ( threadIdx.y + (threadIdx.x/16));
  int kSprd = A_CHUNK * ( threadIdx.x );

  uint32_t Nrndp = (N%YW==0) ? N : (N-N%YW+YW); // Note: All waves in the group need to stay alive to the bitter end, just in case they're needed for cooperative loading of next chunk of A[] into LDS. Such Zomby waves are prevented from doing any real work with continues in the loop below.   
  if (!PCML) Nrndp = N; //unless its not peicmeal
  while (n < Nrndp) {
    kBase = 0;
    for (uint32_t e = (blockIdx.x % ETILE) * M_BLOCK; e < num_tokens_post_padded[0]; e+=M_BLOCK*ETILE) { 
    kBase = 0;
    
#pragma unroll M_BLOCK
    for (uint32_t m=0; m<M_BLOCK; m++) {
	// get the list of Ms corresponding to this M_BLOCK
        offs_token[m] = sorted_token_ids[e+m];
        token_mask[m] = offs_token[m] < num_valid_tokens;
    }
    
    //set the expert for this M_BLOCK
    off_experts = expert_ids[e/M_BLOCK];

#ifdef USEMFMA 
    //asm("v_accvgpr_write %0, 0x0" : "=a"(sum4[0]) : ); // this triggers hip use of acc registers
    sum4 = {0};
#else
    for (int i = 0; i < YTILE; i++)
      for (int m = 0; m < M_BLOCK; m++) 
	      sum[m][i] = 0;
#endif

    bigType bigA[M_BLOCK][UNRL];
#ifdef USEMFMA
    bigTypeXt bigB[YTILE/mfmaTILEn][UNRL];
#else
    bigType bigB[YTILE][UNRL];
#endif
    //----------------------------------------------------
    bool shflk = true;
    for (uint32_t k1_ = 0; k1_ < K; k1_ += THRDS * A_CHUNK * UNRL) {
    if (PCML) {
        if ((k1_ == 0) || (k1_ == kBase + kFit)) { // load next chunk of A[] to LDS
                if (k1_ != 0) kBase += kFit;
		shflk = (kBase + kFit <= K); //don't shfl k (for hotspot avoidance) if this block doesn't cover full range
		__syncthreads();
                 int m = threadIdx.y % M_BLOCK;
                 if (token_mask[m])
                 for (uint32_t k = 0; k < kFit; k += TWC/M_BLOCK) {
                    uint32_t kOff = k + ((((threadIdx.y/M_BLOCK) * THRDS + threadIdx.x) ) * A_CHUNK);    
                    if (kBase + kOff >= K) break;
                    if (kOff >= kFit) break;
#ifdef USEMFMA
                    uint32_t k_in = kBase + (offs_token[m]/top_k) * K + kOff;
                    uint32_t k_ot =         m * K + kOff; // yes, K should be kFit here. but we'lltranspose this below anyway
                    // Transpose A for MFMAs
	            uint32_t k_in_x = (k_ot / A_CHUNK) % (K / A_CHUNK);
	            uint32_t k_in_y = (k_ot / A_CHUNK) / (K / A_CHUNK);
                    uint32_t k_ot_x = (k_in_x / mfmaTILEn) * mfmaTILEn + (k_in_y % mfmaTILEn);
                    uint32_t k_ot_y = (k_in_y / mfmaTILEn) * mfmaTILEn + (k_in_x % mfmaTILEn);
                    
	            k_ot = (k_ot_y * (kFit / A_CHUNK) + k_ot_x) * A_CHUNK;

                    *((bigType*)(&s[k_ot])) = *((bigType*)(&A[k_in]));
		    //}
#else
                    //int m = threadIdx.x % M_BLOCK;
                    //for (uint32_t m = 0; m < M_BLOCK; m++) {
 	            //if (!token_mask[m]) continue;
                    uint32_t k_in = kBase + (offs_token[m]/top_k) * K    + kOff;
                    uint32_t k_ot =         m * kFit + kOff;
		    *((bigType*)(&s[k_ot])) = *((bigType*)(&A[k_in]));
		    //}
#endif
		}
                __syncthreads();
	}
    } 
        
    // kept alive just to participate in A[] loads
    if (n >= N) continue;
    
    int k1 = k1_;
    if (shflk) k1 = kBase + (((k1_-kBase) + kShfl) % kFit ); // shfl loads within this lane, to reduce temporal hotspotting

        #define StgMfma4(_LN) { \
          for (uint32_t _t = 0; _t < A_CHUNK/mfmaTILEk; _t++) { \
            sum4 = __builtin_amdgcn_mfma_f32_16x16x16f16( \
            bigB[0][k2].B[_LN].hT[_t], \
	    bigA[_LN][k2].hT[_t], \
	    sum4, 0, 0, 0); \
          } \
        }


#ifdef PIPELINED1x
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
	uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=0; y<YTILE; y++) // should this be M_BLOCK?
	    bigB[0][k2].B[y].hC = (loadnt((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 0; m < M_BLOCK; m++) 
	{
          bigA[m][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	for (int l=0; l<mfmaTILEn; l++)
          StgMfma4(l);
      }


#elif defined(PIPELINED2x) // 2x

///////////////////////////ROUND 1//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
	uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=0; y<YTILE/2; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 0; m < M_BLOCK/2; m++) 
	{
          bigA[m][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
      for (int l=0; l<mfmaTILEn/2; l++)
          StgMfma4(l);
      }

///////////////////////////ROUND 2//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=YTILE/2; y<YTILE; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y-YTILE/2].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-YTILE/2].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = M_BLOCK/2; m < M_BLOCK; m++) 
	{
          bigA[m-M_BLOCK/2][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
      for (int l=0; l<mfmaTILEn/2; l++)
          StgMfma4(l);
      }

#elif defined(PIPELINED4x) //4x

///////////////////////////ROUND 1//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
	uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=0; y<YTILE/4; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 0; m < M_BLOCK/4; m++) 
	{
          bigA[m][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<mfmaTILEn/4; l++)
          StgMfma4(l);
      }

///////////////////////////ROUND 2//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=YTILE/4; y<YTILE/2; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y-YTILE/4].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-YTILE/4].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = M_BLOCK/4; m < M_BLOCK/2; m++) 
	{
          bigA[m-M_BLOCK/4][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<mfmaTILEn/4; l++)
          StgMfma4(l);
      }
///////////////////////////ROUND 3//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=YTILE/2; y<3*YTILE/4; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y-YTILE/2].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-YTILE/2].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = M_BLOCK/2; m < 3*M_BLOCK/4; m++) 
	{
          bigA[m-M_BLOCK/2][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<mfmaTILEn/4; l++)
          StgMfma4(l);
      }

///////////////////////////ROUND 4//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=3*YTILE/4; y<YTILE; y++) // should this be M_BLOCK?
            //bigB[0][k2].B[y-3*YTILE/4].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-3*YTILE/4].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 3*M_BLOCK/4; m < M_BLOCK; m++) 
	{
          bigA[m-3*M_BLOCK/4][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<mfmaTILEn/4; l++)
          StgMfma4(l);
      }

#elif defined(PIPELINED_33334x) //3334x

///////////////////////////ROUND 1//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
	uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=0; y<3; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 0; m < 3; m++) 
	{
          bigA[m][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<3; l++)
          StgMfma4(l);
      }

///////////////////////////ROUND 2//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=3; y<6; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y-YTILE/4].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-3].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 3; m < 6; m++) 
	{
          bigA[m-3][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<3; l++)
          StgMfma4(l);
      }
///////////////////////////ROUND 3//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=6; y<9; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y-YTILE/2].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-6].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 6; m < 9; m++) 
	{
          bigA[m-6][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<3; l++)
          StgMfma4(l);
      }

///////////////////////////ROUND 4//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=9; y<12; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y-YTILE/2].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-9].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 9; m < 12; m++) 
	{
          bigA[m-9][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<3; l++)
          StgMfma4(l);
      }

///////////////////////////ROUND 5//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=12; y<16; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y-YTILE/2].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-12].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 12; m < 16; m++) 
	{
          bigA[m-12][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<4; l++)
          StgMfma4(l);
      }




#elif defined(PIPELINED_556x) //556x

///////////////////////////ROUND 1//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
	uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=0; y<5; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 0; m < 5; m++) 
	{
          bigA[m][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<5; l++)
          StgMfma4(l);
      //}

///////////////////////////ROUND 2//////////////////////////
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + kSprd;
      //  if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=5; y<10; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y-YTILE/4].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-5].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 5; m < 10; m++) 
	{
          bigA[m-5][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<5; l++)
          StgMfma4(l);
      //}
///////////////////////////ROUND 3//////////////////////////
      //#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + kSprd;
      //  if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=10; y<16; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y-YTILE/2].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-10].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 10; m < 16; m++) 
	{
          bigA[m-10][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<6; l++)
          StgMfma4(l);
      }

#elif defined(PIPELINED8x) //8x

///////////////////////////ROUND 1//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
	uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=0; y<YTILE/8; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 0; m < M_BLOCK/8; m++) 
	{
          bigA[m][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	for (int l=0; l<mfmaTILEn/8; l++)
          StgMfma4(l);
      }

///////////////////////////ROUND 2//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=YTILE/8; y<2*YTILE/8; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y-YTILE/8].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-YTILE/8].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = M_BLOCK/8; m < 2*M_BLOCK/8; m++) 
	{
          bigA[m-M_BLOCK/8][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	for (int l=0; l<mfmaTILEn/8; l++)
          StgMfma4(l);
      }
///////////////////////////ROUND 3//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=2*YTILE/8; y<3*YTILE/8; y++) // should this be M_BLOCK?
	    //bigB[0][k2].B[y-2*YTILE/8].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-2*YTILE/8].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 2*M_BLOCK/8; m < 3*M_BLOCK/8; m++) 
	{
          bigA[m-2*M_BLOCK/8][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	for (int l=0; l<mfmaTILEn/8; l++)
          StgMfma4(l);
      }

///////////////////////////ROUND 4//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=3*YTILE/8; y<4*YTILE/8; y++) // should this be M_BLOCK?
            //bigB[0][k2].B[y-3*YTILE/8].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-3*YTILE/8].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 3*M_BLOCK/8; m < 4*M_BLOCK/8; m++) 
	{
          bigA[m-3*M_BLOCK/8][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	for (int l=0; l<mfmaTILEn/8; l++)
          StgMfma4(l);
      }

///////////////////////////ROUND 5//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=4*YTILE/8; y<5*YTILE/8; y++) // should this be M_BLOCK?
            //bigB[0][k2].B[y-4*YTILE/8].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-4*YTILE/8].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 4*M_BLOCK/8; m < 5*M_BLOCK/8; m++) 
	{
          bigA[m-4*M_BLOCK/8][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	for (int l=0; l<mfmaTILEn/8; l++)
          StgMfma4(l);
      }

///////////////////////////ROUND 6//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=5*YTILE/8; y<6*YTILE/8; y++) // should this be M_BLOCK?
            //bigB[0][k2].B[y-5*YTILE/8].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-5*YTILE/8].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 5*M_BLOCK/8; m < 6*M_BLOCK/8; m++) 
	{
          bigA[m-5*M_BLOCK/8][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<mfmaTILEn/8; l++)
          StgMfma4(l);
      }

///////////////////////////ROUND 6//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=6*YTILE/8; y<7*YTILE/8; y++) // should this be M_BLOCK?
            //bigB[0][k2].B[y-6*YTILE/8].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-6*YTILE/8].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 6*M_BLOCK/8; m < 7*M_BLOCK/8; m++) 
	{
          bigA[m-6*M_BLOCK/8][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<mfmaTILEn/8; l++)
          StgMfma4(l);
      }
///////////////////////////ROUND 7//////////////////////////
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
        for (int y=7*YTILE/8; y<8*YTILE/8; y++) // should this be M_BLOCK?
            //bigB[0][k2].B[y-7*YTILE/8].hC = (loadnt((halfC*)(&B_[y * K])));
	    bigB[0][k2].B[y-7*YTILE/8].hC = *(((halfC*)(&B_[y * K])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;
        for (int m = 7*M_BLOCK/8; m < 8*M_BLOCK/8; m++) 
	{
          bigA[m-7*M_BLOCK/8][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
	}
      //}
//#pragma unroll
      //for (uint32_t k2 = 0; k2 < UNRL; k2++) {
      //  uint32_t k = k1 + k2 * THRDS * A_CHUNK;
      //  uint32_t k_ = k + threadIdx.x * A_CHUNK;
      //  if (k_ >= K) break;
	for (int l=0; l<mfmaTILEn/8; l++)
          StgMfma4(l);
      }

#else  // !PIPELINED

#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
	uint32_t k_ = k + kSprd;
        if (k_ >= K) break;

	const half* B_ = &B[(n + 0) * K + k_ + off_experts*K*N];
#ifdef USEMFMA
        for (int y=0; y<YTILE; y++) // should this be M_BLOCK?
	    bigB[0][k2].B[y].hC = (loadnt((halfC*)(&B_[y * K])));
#else
        for (int y=0; y<YTILE; y++)
	    bigB[y][k2].h8 = (loadnt((half8*)(&B_[y * K])));
#endif
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + kSprd;
        if (k_ >= K) break;

        // Fetch A activation matrix in interleaved fashion from LDS or memory

        for (int m = 0; m < M_BLOCK; m++) 
	{
#ifdef USEMFMA
#else
	  if (!token_mask[m]) continue;
#endif
        if (PCML) {
          //bigA[m][k2] = *((const bigType*)(&(s[k_-kBase + kFit*m])));
	  // skip A[] fetches for Ms that are disabled
          bigA[m][k2] = *((const bigType*)(&(s[k_-kBase + m*kFit ])));
        } else {
	  int aidx = k_ + (offs_token[m]/top_k) * K;
          if (aidx + A_CHUNK <= 32 * 1024)
            bigA[m][k2] = *((const bigType*)(&(s[aidx])));
          else
            bigA[m][k2] = *((const bigType*)(&(A[aidx])));
        }
        }
      }

      // Do the matrix multiplication in interleaved manner
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;

#ifdef USEMFMA
	bigType stgB;
	for (int l=0; l<mfmaTILEn; l++)
          StgMfma4(l);
#else

#pragma unroll
        for (uint32_t m = 0; m < M_BLOCK; m++) {
	  // skip compute for Ms that are disabled
	  if (!token_mask[m]) continue;
          // Do the matrix multiplication of activation and weight matrix
          // - Remember the accumulation is happening for K-split of 64!
#pragma unroll
          for (uint32_t b = 0; b < A_CHUNK / 2; b++)
           for (int y=0; y<YTILE; y++)
            asm("v_dot2c_f32_f16 %0, %2, %3"
                : "=v"(sum[m][y])
                : "0"(sum[m][y]), "v"(bigA[m][k2].f[b]), "v"(bigB[y][k2].f[b]));
        }
#endif
      }


#endif // !PIPELINED

    } // K-loop

    //That's enough walking K. Let's write this puppy out...
#ifdef USEMFMA
    //for (int m = 0; m < M_BLOCK; m++)
    //    if (threadIdx.x % mfmaTILEn == m % mfmaTILEn)
	    int m = threadIdx.x % mfmaTILEn;
	    if (token_mask[m]) {
	      const int thisTokOff = offs_token[m];
              #pragma unroll 4
              for (int y = 0; y < 4; y++) {
  	        if (mul_routed_weight)
                    sum4[y] *= topk_weights[thisTokOff];//token_mask[m] ? topk_weights[offs_token[m]] : 0;
                int oidx = n + (threadIdx.x/mfmaTILEn)*mfmaTILEk + y + thisTokOff * N;
                //if (commitColumn[i]) 
                    C[oidx] = __float2half(sum4[y]);
	      }
           }
#else

    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    for (int m = 0; m < M_BLOCK; m++) {
      // skip reduce for Ms that are disabled
      if (!token_mask[m]) continue;

      for (int y = 0; y < YTILE; y++) {
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
              : "=v"(sum[m][y])
              : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
              : "=v"(sum[m][y])
              : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
              : "=v"(sum[m][y])
              : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
              : "=v"(sum[m][y])
              : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
              : "=v"(sum[m][y])
              : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
          asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
              : "=v"(sum[m][y])
              : "0"(sum[m][y]), "v"(sum[m][y]), "v"(sum[m][y]));
       }
    }

    if (threadIdx.x == 63) {
      for (int m = 0; m < M_BLOCK; m++) {
        for (int i = 0; i < YTILE; i++) 
	{
	  // skip write out for Ms that are disabled
	  if (!token_mask[m]) continue;
  	  if (mul_routed_weight)
              sum[m][i] *= token_mask[m] ? topk_weights[offs_token[m]/top_k] : 0;
          int oidx = n + i + offs_token[m] * N;
          if (commitColumn[i]) 
		  C[oidx] = __float2half(sum[m][i]);
        }
      }
    }
#endif

    }

    n += (CuCount / ETILE) * WvPrGrp * YTILE;

    // Check whether there will be fragmenation!
    // This will happen only for the last wave!
    if (n < N && (n + YTILE) >= N) {
      uint32_t startColumn = N - YTILE;
      for (uint32_t i = 0; i < (n - startColumn); i++) {
        commitColumn[i] = 0;
      }
      n = startColumn;
    }
  }
}



// a = torch.randn((m, k)
// b1 = torch.randn((e, 2 * n, k)
// b2 = torch.randn((e, k, n)
// topk_weights = torch.randn((m, e), device='cuda', dtype=dtype)

void wvSpltK_fsdMoe_(void* in_a, void* in_b, void* out_c, 
		     void* topk_weights, 
		     void* topk_ids, 
		     void* sorted_token_ids, 
		     void* expert_ids,
		     void* num_tokens_post_padded, 
		     const int M_in, const int N_in, const int K_in, const int E, 
		     const int num_valid_tokens, 
		     const int stride_am,
                     const int stride_ak,
                     const int stride_be,
                     const int stride_bk,
                     const int stride_bn,
                     const int stride_cm,
                     const int stride_cn,
		     const int m_blck_sz, 
		     const bool mul_routed_weight, 
		     const int top_k,
		     cudaStream_t stream, const int CuCount) {
  dim3 grid(CuCount);
  dim3 block(THRDS, WvPrGrp);
  auto* a = reinterpret_cast<const half*>(in_a);
  auto* b = reinterpret_cast<const half*>(in_b);
  auto* c = reinterpret_cast<half*>(out_c);
  auto* topk_weights_ = reinterpret_cast<const float*>(topk_weights);
  auto* topk_ids_ = reinterpret_cast<const int*>(topk_ids);
  auto* sorted_token_ids_ = reinterpret_cast<const int*>(sorted_token_ids);
  auto* expert_ids_ = reinterpret_cast<const int*>(expert_ids);
  auto* num_tokens_post_padded_ = reinterpret_cast<const int*>(num_tokens_post_padded); 
  switch (m_blck_sz) {
    case 1:
      wvSpltK_fsdMoe_hf_<1,4><<<grid, block, 0, stream>>>(a, b, c, topk_weights_, topk_ids_, sorted_token_ids_, expert_ids_, num_tokens_post_padded_, M_in, N_in, K_in, E, num_valid_tokens, stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn, mul_routed_weight, top_k, CuCount);
      break;
    case 2:
      wvSpltK_fsdMoe_hf_<2,4><<<grid, block, 0, stream>>>(a, b, c, topk_weights_, topk_ids_, sorted_token_ids_, expert_ids_, num_tokens_post_padded_, M_in, N_in, K_in, E, num_valid_tokens, stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn, mul_routed_weight, top_k, CuCount);
      break;
    case 3:
      wvSpltK_fsdMoe_hf_<3,4><<<grid, block, 0, stream>>>(a, b, c, topk_weights_, topk_ids_, sorted_token_ids_, expert_ids_, num_tokens_post_padded_, M_in, N_in, K_in, E, num_valid_tokens, stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn, mul_routed_weight, top_k, CuCount);
      break;
    case 4:
      wvSpltK_fsdMoe_hf_<4,4><<<grid, block, 0, stream>>>(a, b, c, topk_weights_, topk_ids_, sorted_token_ids_, expert_ids_, num_tokens_post_padded_, M_in, N_in, K_in, E, num_valid_tokens, stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn, mul_routed_weight, top_k, CuCount);
      break;
    case 5:
      wvSpltK_fsdMoe_hf_<5,4><<<grid, block, 0, stream>>>(a, b, c, topk_weights_, topk_ids_, sorted_token_ids_, expert_ids_, num_tokens_post_padded_, M_in, N_in, K_in, E, num_valid_tokens, stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn, mul_routed_weight, top_k, CuCount);
      break;
    case 6:
      wvSpltK_fsdMoe_hf_<6,4><<<grid, block, 0, stream>>>(a, b, c, topk_weights_, topk_ids_, sorted_token_ids_, expert_ids_, num_tokens_post_padded_, M_in, N_in, K_in, E, num_valid_tokens, stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn, mul_routed_weight, top_k, CuCount);
      break;
    case 16:
      wvSpltK_fsdMoe_hf_mfma16_<16,16><<<grid, block, 0, stream>>>(a, b, c, topk_weights_, topk_ids_, sorted_token_ids_, expert_ids_, num_tokens_post_padded_, M_in, N_in, K_in, E, num_valid_tokens, stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn, mul_routed_weight, top_k, CuCount);
      break;

  }
}

void wvSpltK_(void* in_a, void* in_b, void* out_c, const int M_in,
              const int K_in, const int N_in, cudaStream_t stream,
              const int CuCount = 0) {
  dim3 grid(CuCount);
  dim3 block(THRDS, WvPrGrp);
  half* af4 = reinterpret_cast<half*>(in_a);
  const half* bf4 = reinterpret_cast<const half*>(in_b);
  auto* c = reinterpret_cast<half*>(out_c);
  switch (N_in) {
    case 1:
      if ((K_in <= 32 * 1024) && (M_in % 2 == 0)) {
        wvSpltK_hf_m1_sml_<<<grid, block, 0, stream>>>(K_in, M_in, af4, bf4, c,
                                                       CuCount);
      } else {
        wvSpltK_hf_m1_<<<grid, block, 0, stream>>>(K_in, M_in, af4, bf4, c,
                                                   CuCount);
      }
      break;
    case 2:
      wvSpltK_hf_m2_<<<grid, block, 0, stream>>>(K_in, M_in, af4, bf4, c,
                                                 CuCount);
      break;
    case 3:
      wvSpltK_hf_m3_<<<grid, block, 0, stream>>>(K_in, M_in, af4, bf4, c,
                                                 CuCount);
      break;
    case 4:
      wvSpltK_hf_m4_<<<grid, block, 0, stream>>>(K_in, M_in, af4, bf4, c,
                                                 CuCount);
      break;
    default:
      throw std::runtime_error("Unsupported N value: " + std::to_string(M_in) +
                               "," + std::to_string(K_in) + "," +
                               std::to_string(N_in));
  }

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
  }
}
