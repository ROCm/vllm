// clang-format off
// adapted from https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_kernel.cuh
#pragma once

#include "../../torch_utils.h"
#include <torch/csrc/stable/macros.h>
#include "../selective_scan.h"

#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_scan.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif

#include "../selective_scan.h"
#include "../static_switch.h"

template<int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_,
         bool kIsVariableB_, bool kIsVariableC_,
         bool kHasZ_, bool kVarlen_, typename input_t_, typename weight_t_, typename state_t_>
struct Selective_Scan_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    using state_t = state_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNRows = kNRows_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kVarlen_ ? false : kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kHasZ = kHasZ_;
    static constexpr bool kVarlen = kVarlen_;

    static constexpr bool kDirectIO = kVarlen_ ? false : kIsEvenLen && kNLoads == 1;
    static constexpr int kNLoadsIndex = kNItems / 4;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems , cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads ,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params) {
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr bool kVarlen = Ktraits::kVarlen;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    // cast to lvalue reference of expected type
    // char *smem_loadstorescan = smem_ + 2 * MAX_DSTATE * sizeof(weight_t);
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_ + 2 * MAX_DSTATE * sizeof(weight_t));
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_loadstorescan);
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    // weight_t *smem_a = reinterpret_cast<weight_t *>(smem_ + smem_loadstorescan_size);
    // weight_t *smem_bc = reinterpret_cast<weight_t *>(smem_a + MAX_DSTATE);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    int seqlen = params.seqlen;
    int sequence_start_index = batch_id;
    if constexpr (kVarlen){
        int *query_start_loc = reinterpret_cast<int *>(params.query_start_loc_ptr);
        sequence_start_index = query_start_loc[batch_id];
        seqlen = query_start_loc[batch_id + 1] - sequence_start_index;
    }
    const bool has_initial_state = params.has_initial_state_ptr == nullptr ? false
        : reinterpret_cast<bool *>(params.has_initial_state_ptr)[batch_id];

    const int* cache_indices = params.cache_indices_ptr == nullptr ? nullptr
        : reinterpret_cast<int *>(params.cache_indices_ptr);
    int cache_index;
    if (cache_indices == nullptr) {
        cache_index = batch_id;
    } else if (params.cache_enabled) {
        const int* initial_state_idx = reinterpret_cast<const int*>(params.initial_state_idx_ptr);
        cache_index = cache_indices[batch_id * params.cache_indices_stride + initial_state_idx[batch_id]];
    } else {
        cache_index = cache_indices[batch_id];
    }
    // Skip batch entries whose cache index maps to the null block (padding).
    if (cache_indices != nullptr && cache_index == params.null_block_id){
        return;
    }
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + sequence_start_index * params.u_batch_stride
        + dim_id * kNRows * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + sequence_start_index * params.delta_batch_stride
        + dim_id * kNRows * params.delta_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + sequence_start_index * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + sequence_start_index * params.C_batch_stride + group_id * params.C_group_stride;

    typename Ktraits::state_t *ssm_states;
    if (params.cache_enabled) {
        // APC mode: ssm_states points to the base, we'll use absolute cache slots later
        ssm_states = reinterpret_cast<typename Ktraits::state_t *>(params.ssm_states_ptr) +
            dim_id * kNRows * params.ssm_states_dim_stride;
    } else {
        // Non-APC mode: offset by cache_index as before
        ssm_states = reinterpret_cast<typename Ktraits::state_t *>(params.ssm_states_ptr) +
            cache_index * params.ssm_states_batch_stride +
            dim_id * kNRows * params.ssm_states_dim_stride;
    }
    
    float D_val[kNRows] = {0};
    if (params.D_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            D_val[r] = reinterpret_cast<float *>(params.D_ptr)[dim_id * kNRows + r];
        }
    }
    float delta_bias[kNRows] = {0};
    if (params.delta_bias_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            delta_bias[r] = reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id * kNRows + r];
        }
    }

    // Use block_size for chunking when APC is enabled, otherwise use 2048 for backwards compatibility
    const int block_size = params.cache_enabled ? params.block_size : 2048;

    const int* batch_cache_indices = cache_indices != nullptr ?
                                     cache_indices + batch_id * params.cache_indices_stride : nullptr;
    const int* block_idx_first_scheduled = params.block_idx_first_scheduled_token_ptr != nullptr ?
                                           reinterpret_cast<const int*>(params.block_idx_first_scheduled_token_ptr) : nullptr;
    const int* block_idx_last_scheduled = params.block_idx_last_scheduled_token_ptr != nullptr ?
                                          reinterpret_cast<const int*>(params.block_idx_last_scheduled_token_ptr) : nullptr;
    const int* initial_state_idx = params.initial_state_idx_ptr != nullptr ?
                                   reinterpret_cast<const int*>(params.initial_state_idx_ptr) : nullptr;
    const int* cu_chunk_seqlen = params.cu_chunk_seqlen_ptr != nullptr ?
                                 reinterpret_cast<const int*>(params.cu_chunk_seqlen_ptr) : nullptr;
    const int* last_chunk_indices = params.last_chunk_indices_ptr != nullptr ?
                                    reinterpret_cast<const int*>(params.last_chunk_indices_ptr) : nullptr;

    const size_t load_cache_slot = params.cache_enabled && batch_cache_indices != nullptr ? batch_cache_indices[initial_state_idx[batch_id]] : cache_index;

    const int block_idx_first = (params.cache_enabled && block_idx_first_scheduled != nullptr) ?
                                 block_idx_first_scheduled[batch_id] : 0;

    // Determine chunk boundaries from pre-computed metadata (APC mode)
    // or fall back to simple block_size chunking.
    int first_chunk_idx, n_chunks;
    int current_position;

    if (cu_chunk_seqlen != nullptr && last_chunk_indices != nullptr) {
        const int last_chunk_idx = last_chunk_indices[batch_id];
        first_chunk_idx = (batch_id == 0) ? 0 : last_chunk_indices[batch_id - 1] + 1;
        n_chunks = last_chunk_idx - first_chunk_idx + 1;
        // Derive current_position: if the first chunk is partial (fills remainder
        // of a started block), offset into the block accordingly.
        const int first_chunk_tokens = cu_chunk_seqlen[first_chunk_idx + 1] - cu_chunk_seqlen[first_chunk_idx];
        const int chunk_start_offset = (n_chunks > 1 && first_chunk_tokens < block_size)
                                        ? (block_size - first_chunk_tokens) : 0;
        current_position = block_idx_first * block_size + chunk_start_offset;
    } else {
        first_chunk_idx = 0;
        n_chunks = (seqlen + block_size - 1) / block_size;
        current_position = 0;
    }

    int tokens_processed = 0;

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int chunk_tokens = (cu_chunk_seqlen != nullptr)
            ? cu_chunk_seqlen[first_chunk_idx + chunk + 1] - cu_chunk_seqlen[first_chunk_idx + chunk]
            : min(block_size, seqlen - tokens_processed);
        if (chunk_tokens <= 0) break;
        input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];

        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, chunk_tokens);
            if constexpr (!kDirectIO) { __syncthreads(); }
            load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, chunk_tokens);
        }
        u += chunk_tokens;
        delta += chunk_tokens;
    
        float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float u_val = float(u_vals[r][i]);
                delta_vals[r][i] = float(delta_vals_load[r][i]) + delta_bias[r];
                if (params.delta_softplus) {
                    delta_vals[r][i] = delta_vals[r][i] <= 20.f ? log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
                }
                delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                out_vals[r][i] = D_val[r] * u_val;
            }
        }

        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            weight_t A_val[kNRows];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                A_val[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
                // Multiply the real part of A with LOG2E so we can use exp2f instead of expf.
                constexpr float kLog2e = M_LOG2E;
                A_val[r] *= kLog2e;
            }
            // This variable holds B * C if both B and C are constant across seqlen. If only B varies
            // across seqlen, this holds C. If only C varies across seqlen, this holds B.
            // If both B and C vary, this is unused.
            weight_t BC_val[kNRows];
            weight_t B_vals[kNItems], C_vals[kNItems];
            if constexpr (kIsVariableB) {
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, chunk_tokens);
                if constexpr (!kIsVariableC) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                    }
                }
            }
            if constexpr (kIsVariableC) {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, chunk_tokens);
                if constexpr (!kIsVariableB) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
                    }
                }
            }
            if constexpr (!kIsVariableB && !kIsVariableC) {
                #pragma unroll
                for (int r = 0; r < kNRows; ++r) {
                    BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride] * C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                }
            }

            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                if (r > 0) { __syncthreads(); }  // Scan could be using the same smem
                scan_t thread_data[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    thread_data[i] = make_float2(exp2f(delta_vals[r][i] * A_val[r]),
                                                 !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i]);
                    if (threadIdx.x * kNItems + i >= chunk_tokens) {
                        thread_data[i] = make_float2(1.f, 0.f);
                    }
                }
                // Initialize running total
                scan_t running_prefix;
                if (chunk > 0) {
                    running_prefix = smem_running_prefix[state_idx + r * MAX_DSTATE];
                } else {
                    // Load initial state
                    if (params.cache_enabled && has_initial_state && batch_cache_indices != nullptr) {
                        size_t state_offset = load_cache_slot * params.ssm_states_batch_stride +
                                             r * params.ssm_states_dim_stride +
                                             state_idx * params.ssm_states_dstate_stride;
                        running_prefix = make_float2(1.0, float(ssm_states[state_offset]));
                    } else if (has_initial_state) {
                        // Non-APC mode: load from current batch position
                        running_prefix = make_float2(1.0, float(ssm_states[state_idx * params.ssm_states_dstate_stride]));
                    } else {
                        // No initial state
                        running_prefix = make_float2(1.0, 0.0);
                    }
                }

                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                // There's a syncthreads in the scan op, so we don't need to sync here.
                // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
                if (threadIdx.x == 0) {
                    smem_running_prefix[state_idx + r * MAX_DSTATE] = prefix_op.running_prefix;

                    // Store state at the end of each aligned chunk when cache is enabled
                    if (params.cache_enabled && batch_cache_indices != nullptr) {
                        size_t cache_slot;
                        if (chunk == n_chunks - 1) {
                            cache_slot = batch_cache_indices[block_idx_last_scheduled[batch_id]];
                        } else {
                            const int block_idx_completed = (current_position + chunk_tokens - 1) / block_size;
                            cache_slot = batch_cache_indices[block_idx_completed];
                        }

                        size_t state_offset = cache_slot * params.ssm_states_batch_stride +
                                             r * params.ssm_states_dim_stride +
                                             state_idx * params.ssm_states_dstate_stride;

                        ssm_states[state_offset] = typename Ktraits::state_t(prefix_op.running_prefix.y);
                    } else if (!params.cache_enabled && chunk == n_chunks - 1) {
                        // Non-APC mode: store only final state at current batch position
                        ssm_states[state_idx * params.ssm_states_dstate_stride] = typename Ktraits::state_t(prefix_op.running_prefix.y);
                    }
                }
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const weight_t C_val = !kIsVariableC
                        ? BC_val[r]
                        : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
                    out_vals[r][i] += thread_data[i].y * C_val;
                }
            }
        }
        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + sequence_start_index * params.out_batch_stride
            + dim_id * kNRows * params.out_d_stride + tokens_processed;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            store_output<Ktraits>(out + r * params.out_d_stride, out_vals[r], smem_store, chunk_tokens);
        }

        if constexpr (kHasZ) {
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + sequence_start_index * params.z_batch_stride
                + dim_id * kNRows * params.z_d_stride + tokens_processed;
            input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + sequence_start_index * params.out_z_batch_stride
                + dim_id * kNRows * params.out_z_d_stride + tokens_processed;
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                input_t z_vals[kNItems];
                __syncthreads();
                load_input<Ktraits>(z + r * params.z_d_stride, z_vals, smem_load, chunk_tokens);
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float z_val = z_vals[i];
                    out_vals[r][i] *= z_val / (1 + expf(-z_val));
                }
                __syncthreads();
                store_output<Ktraits>(out_z + r * params.out_z_d_stride, out_vals[r], smem_store, chunk_tokens);
            }
        }

        Bvar += chunk_tokens;
        Cvar += chunk_tokens;

        tokens_processed += chunk_tokens;
        current_position += chunk_tokens;
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t, typename state_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    // Only kNRows == 1 is tested for now, which ofc doesn't differ from previously when we had each block
    // processing 1 row.
    constexpr int kNRows = 1;
    // kIsVariableB, kIsVariableC and kHasZ are all set to True to reduce binary size
    constexpr bool kIsVariableB = true;
    constexpr bool kIsVariableC = true;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.z_ptr != nullptr , kHasZ, [&] {
            BOOL_SWITCH(params.query_start_loc_ptr != nullptr , kVarlen, [&] {
                using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kIsVariableB, kIsVariableC, kHasZ,  kVarlen, input_t, weight_t, state_t>;
                constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
                dim3 grid(params.batch, params.dim / kNRows);
                auto kernel = &selective_scan_fwd_kernel<Ktraits>;
                if (kSmemSize >= 48 * 1024) {
#ifdef USE_ROCM
                    STD_CUDA_CHECK(hipFuncSetAttribute(
                        reinterpret_cast<const void*>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#else
                    STD_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#endif
                }
                kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                STD_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

template<typename input_t, typename weight_t, typename state_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {

    #ifndef USE_ROCM
        if (params.cache_enabled && params.block_size == 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 128) {
            selective_scan_fwd_launch<32, 4, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 256) {
            selective_scan_fwd_launch<32, 8, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_fwd_launch<32, 16, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t, state_t>(params, stream);
        } else {
            selective_scan_fwd_launch<128, 16, input_t, weight_t, state_t>(params, stream);
        }
    #else
        if (params.cache_enabled && params.block_size == 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 256) {
            selective_scan_fwd_launch<64, 4, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_fwd_launch<64, 8, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t, state_t>(params, stream);
        } else {
            selective_scan_fwd_launch<128, 16, input_t, weight_t, state_t>(params, stream);
        }
    #endif
}
