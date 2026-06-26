// clang-format off
// adapted from https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_kernel.cuh
#include "../torch_utils.h"
#include <torch/csrc/stable/macros.h>
#include "selective_scan.h"

#include <cstring>

#define CHECK_SHAPE(x, ...) STD_TORCH_CHECK(x.sizes().equals(torch::headeronly::IntHeaderOnlyArrayRef({__VA_ARGS__})), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, STYPE, NAME, ...)       \
    if (ITYPE == torch::headeronly::ScalarType::Half) {                             \
        using input_t = torch::headeronly::Half;                                                  \
        using weight_t = float;                                                     \
        if (STYPE == torch::headeronly::ScalarType::Half) {                         \
            using state_t = torch::headeronly::Half;                                              \
            __VA_ARGS__();                                                          \
        } else if (STYPE == torch::headeronly::ScalarType::Float) {                 \
            using state_t = float;                                                  \
            __VA_ARGS__();                                                          \
        } else {                                                                    \
            STD_TORCH_CHECK(false, #NAME " not implemented for state type '", STYPE, "'"); \
        }                                                                           \
    } else if (ITYPE == torch::headeronly::ScalarType::BFloat16) {                  \
        using input_t = torch::headeronly::BFloat16;                                              \
        using weight_t = float;                                                     \
        if (STYPE == torch::headeronly::ScalarType::BFloat16) {                     \
            using state_t = torch::headeronly::BFloat16;                                          \
            __VA_ARGS__();                                                          \
        } else if (STYPE == torch::headeronly::ScalarType::Float) {                 \
            using state_t = float;                                                  \
            __VA_ARGS__();                                                          \
        } else {                                                                    \
            STD_TORCH_CHECK(false, #NAME " not implemented for state type '", STYPE, "'"); \
        }                                                                           \
    } else if (ITYPE == torch::headeronly::ScalarType::Float)  {                    \
        using input_t = float;                                                      \
        using weight_t = float;                                                     \
        using state_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        STD_TORCH_CHECK(false, #NAME " not implemented for input type '", ITYPE, "'"); \
    }


template<typename input_t, typename weight_t, typename state_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream);

void set_ssm_params_fwd(SSMParamsBase &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const torch::stable::Tensor u,
                        const torch::stable::Tensor delta,
                        const torch::stable::Tensor A,
                        const torch::stable::Tensor B,
                        const torch::stable::Tensor C,
                        const torch::stable::Tensor out,
                        const torch::stable::Tensor z,
                        const torch::stable::Tensor out_z,
                        const std::optional<torch::stable::Tensor>& D,
                        const std::optional<torch::stable::Tensor>& delta_bias,
                        const torch::stable::Tensor ssm_states,
                        bool has_z,
                        bool delta_softplus,
                        const std::optional<torch::stable::Tensor>& query_start_loc,
                        const std::optional<torch::stable::Tensor>& cache_indices,
                        const std::optional<torch::stable::Tensor>& has_initial_state,
                        bool varlen,
                        int64_t null_block_id,
                        int64_t block_size,
                        const std::optional<torch::stable::Tensor> &block_idx_first_scheduled_token,
                        const std::optional<torch::stable::Tensor> &block_idx_last_scheduled_token,
                        const std::optional<torch::stable::Tensor> &initial_state_idx,
                        const std::optional<torch::stable::Tensor> &cu_chunk_seqlen,
                        const std::optional<torch::stable::Tensor> &last_chunk_indices) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.dstate = dstate;
    params.n_groups = n_groups;
    params.dim_ngroups_ratio = dim / n_groups;
    params.null_block_id = null_block_id;

    params.delta_softplus = delta_softplus;

    params.is_variable_B = is_variable_B;
    params.is_variable_C = is_variable_C;

    // Set the pointers and strides.
    params.u_ptr = u.data_ptr();
    params.delta_ptr = delta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.B_ptr = B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.D_ptr = D.has_value() ? D.value().data_ptr() : nullptr;
    params.delta_bias_ptr = delta_bias.has_value() ? delta_bias.value().data_ptr() : nullptr;
    params.out_ptr = out.data_ptr();
    params.ssm_states_ptr = ssm_states.data_ptr();
    params.z_ptr = has_z ? z.data_ptr() : nullptr;
    params.out_z_ptr = has_z ? out_z.data_ptr() : nullptr;
    params.query_start_loc_ptr = query_start_loc.has_value() ? query_start_loc.value().data_ptr() : nullptr;
    params.cache_indices_ptr = cache_indices.has_value() ? cache_indices.value().data_ptr() : nullptr;
    params.has_initial_state_ptr = has_initial_state.has_value() ? has_initial_state.value().data_ptr() : nullptr;

    // Set cache parameters - cache is enabled if we have direct cache writing params
    params.cache_enabled = block_idx_first_scheduled_token.has_value();
    params.block_size = static_cast<int>(block_size);

    // Set direct cache writing pointers
    params.block_idx_first_scheduled_token_ptr = block_idx_first_scheduled_token.has_value() ? block_idx_first_scheduled_token.value().data_ptr() : nullptr;
    params.block_idx_last_scheduled_token_ptr = block_idx_last_scheduled_token.has_value() ? block_idx_last_scheduled_token.value().data_ptr() : nullptr;
    params.initial_state_idx_ptr = initial_state_idx.has_value() ? initial_state_idx.value().data_ptr() : nullptr;
    params.cu_chunk_seqlen_ptr = cu_chunk_seqlen.has_value() ? cu_chunk_seqlen.value().data_ptr() : nullptr;
    params.last_chunk_indices_ptr = last_chunk_indices.has_value() ? last_chunk_indices.value().data_ptr() : nullptr;

    // All stride are in elements, not bytes.
    params.A_d_stride = A.stride(0);
    params.A_dstate_stride = A.stride(1);

    if (varlen){
        params.B_batch_stride = B.stride(2);
        params.B_group_stride = B.stride(0);
        params.B_dstate_stride = B.stride(1);
        params.C_batch_stride = C.stride(2);
        params.C_group_stride = C.stride(0);
        params.C_dstate_stride = C.stride(1);

        params.u_batch_stride = u.stride(1);
        params.u_d_stride = u.stride(0);
        params.delta_batch_stride = delta.stride(1);
        params.delta_d_stride = delta.stride(0);
        if (has_z) {
            params.z_batch_stride = z.stride(1);
            params.z_d_stride = z.stride(0);
            params.out_z_batch_stride = out_z.stride(1);
            params.out_z_d_stride = out_z.stride(0);
        }
        params.out_batch_stride = out.stride(1);
        params.out_d_stride = out.stride(0);

        params.ssm_states_batch_stride = ssm_states.stride(0);
        params.ssm_states_dim_stride = ssm_states.stride(1);
        params.ssm_states_dstate_stride = ssm_states.stride(2);

        params.cache_indices_stride = cache_indices.has_value() ? cache_indices.value().stride(0) : 0;

    }
    else{
        if (!is_variable_B) {
            params.B_d_stride = B.stride(0);
        } else {
            params.B_batch_stride = B.stride(0);
            params.B_group_stride = B.stride(1);
        }
        params.B_dstate_stride = !is_variable_B ? B.stride(1) : B.stride(2);
        if (!is_variable_C) {
            params.C_d_stride = C.stride(0);
        } else {
            params.C_batch_stride = C.stride(0);
            params.C_group_stride = C.stride(1);
        }
        params.C_dstate_stride = !is_variable_C ? C.stride(1) : C.stride(2);
        params.u_batch_stride = u.stride(0);
        params.u_d_stride = u.stride(1);
        params.delta_batch_stride = delta.stride(0);
        params.delta_d_stride = delta.stride(1);
        if (has_z) {
            params.z_batch_stride = z.stride(0);
            params.z_d_stride = z.stride(1);
            params.out_z_batch_stride = out_z.stride(0);
            params.out_z_d_stride = out_z.stride(1);
        }
        params.out_batch_stride = out.stride(0);
        params.out_d_stride = out.stride(1);

        params.ssm_states_batch_stride = ssm_states.stride(0);
        params.ssm_states_dim_stride = ssm_states.stride(1);
        params.ssm_states_dstate_stride = ssm_states.stride(2);

        params.cache_indices_stride = cache_indices.has_value() ? cache_indices.value().stride(0) : 0;
    }
}

void selective_scan_fwd(const torch::stable::Tensor &u, const torch::stable::Tensor &delta,
                  const torch::stable::Tensor &A, const torch::stable::Tensor &B, const torch::stable::Tensor &C,
                  const std::optional<torch::stable::Tensor> &D_,
                  const std::optional<torch::stable::Tensor> &z_,
                  const std::optional<torch::stable::Tensor> &delta_bias_,
                  bool delta_softplus,
                  const std::optional<torch::stable::Tensor> &query_start_loc,
                  const std::optional<torch::stable::Tensor> &cache_indices,
                  const std::optional<torch::stable::Tensor> &has_initial_state,
                  const torch::stable::Tensor &ssm_states,
                  // used to identify padding entries if cache_indices provided
                  // in case of padding, the kernel will return early
                  int64_t null_block_id,
                  int64_t block_size,
                  const std::optional<torch::stable::Tensor> &block_idx_first_scheduled_token,
                  const std::optional<torch::stable::Tensor> &block_idx_last_scheduled_token,
                  const std::optional<torch::stable::Tensor> &initial_state_idx,
                  const std::optional<torch::stable::Tensor> &cu_chunk_seqlen,
                  const std::optional<torch::stable::Tensor> &last_chunk_indices) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    STD_TORCH_CHECK(input_type == torch::headeronly::ScalarType::Float || input_type == torch::headeronly::ScalarType::Half || input_type == torch::headeronly::ScalarType::BFloat16);
    STD_TORCH_CHECK(weight_type == torch::headeronly::ScalarType::Float);

    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;

    STD_TORCH_CHECK(delta.scalar_type() == input_type);
    STD_TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
    STD_TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));

    STD_TORCH_CHECK(u.is_cuda());
    STD_TORCH_CHECK(delta.is_cuda());
    STD_TORCH_CHECK(A.is_cuda());
    STD_TORCH_CHECK(B.is_cuda());
    STD_TORCH_CHECK(C.is_cuda());

    STD_TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    STD_TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

    const auto sizes = u.sizes();
    const bool varlen = query_start_loc.has_value();
    const int batch_size = varlen ? query_start_loc.value().sizes()[0] - 1 : sizes[0];
    const int dim = varlen ? sizes[0] : sizes[1];
    const int seqlen = varlen ? sizes[1] : sizes[2];
    const int dstate = A.size(1);
    const int n_groups = varlen ? B.size(0) : B.size(1);

    STD_TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    if (varlen) {
        CHECK_SHAPE(u, dim, seqlen);
        CHECK_SHAPE(delta, dim, seqlen);
    } else {
        CHECK_SHAPE(u, batch_size, dim, seqlen);
        CHECK_SHAPE(delta, batch_size, dim, seqlen);
    }
    CHECK_SHAPE(A, dim, dstate);
    STD_TORCH_CHECK(is_variable_B, "is_variable_B = False is disabled in favor of reduced binary size");
    if (varlen) {
        CHECK_SHAPE(B, n_groups, dstate, seqlen);
    } else {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, seqlen);
    }
    STD_TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);

    STD_TORCH_CHECK(is_variable_C, "is_variable_C = False is disabled in favor of reduced binary size");
    if (varlen) {
        CHECK_SHAPE(C, n_groups, dstate, seqlen);
    } else {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, seqlen);
    }
    STD_TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);

    if (D_.has_value()) {
        auto D = D_.value();
        STD_TORCH_CHECK(D.scalar_type() == torch::headeronly::ScalarType::Float);
        STD_TORCH_CHECK(D.is_cuda());
        STD_TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        STD_TORCH_CHECK(delta_bias.scalar_type() == torch::headeronly::ScalarType::Float);
        STD_TORCH_CHECK(delta_bias.is_cuda());
        STD_TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim);
    }


    if (has_initial_state.has_value()) {
        auto has_initial_state_ = has_initial_state.value();
        STD_TORCH_CHECK(has_initial_state_.scalar_type() == torch::headeronly::ScalarType::Bool);
        STD_TORCH_CHECK(has_initial_state_.is_cuda());
        CHECK_SHAPE(has_initial_state_, batch_size);
    }


    if (query_start_loc.has_value()) {
        auto query_start_loc_ = query_start_loc.value();
        STD_TORCH_CHECK(query_start_loc_.scalar_type() == torch::headeronly::ScalarType::Int);
        STD_TORCH_CHECK(query_start_loc_.is_cuda());
    }


    if (cache_indices.has_value()) {
        auto cache_indices_ = cache_indices.value();
        STD_TORCH_CHECK(cache_indices_.scalar_type() == torch::headeronly::ScalarType::Int);
        STD_TORCH_CHECK(cache_indices_.is_cuda());

        // cache_indices can be either 1D (batch_size,) for non-APC mode
        // or 2D (batch_size, max_positions) for APC mode
        const bool is_apc_mode = block_idx_first_scheduled_token.has_value();
        if (is_apc_mode) {
            STD_TORCH_CHECK(cache_indices_.dim() == 2, "cache_indices must be 2D for APC mode");
            STD_TORCH_CHECK(cache_indices_.size(0) == batch_size, "cache_indices first dimension must match batch_size");
        } else {
            CHECK_SHAPE(cache_indices_, batch_size);
        }
    }


    torch::stable::Tensor z, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        STD_TORCH_CHECK(z.scalar_type() == input_type);
        STD_TORCH_CHECK(z.is_cuda());
        STD_TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        if (varlen){
            CHECK_SHAPE(z, dim, seqlen);
        } else {
            CHECK_SHAPE(z, batch_size, dim, seqlen);
        }

        out_z = z;
    }

    // Right now u has BHL layout and delta has HBL layout, and we want out to have HBL layout
    torch::stable::Tensor out = delta;
    // ssm_states can now be either the same as input_type or float32
    auto state_type = ssm_states.scalar_type();
    STD_TORCH_CHECK(state_type == input_type || state_type == torch::headeronly::ScalarType::Float);
    STD_TORCH_CHECK(ssm_states.is_cuda());
    STD_TORCH_CHECK(ssm_states.stride(-1) == 1);

    SSMParamsBase params;
    set_ssm_params_fwd(params, batch_size, dim, seqlen, dstate, n_groups, is_variable_B, is_variable_C,
                       u, delta, A, B, C, out, z, out_z,
                       D_,
                       delta_bias_,
                       ssm_states,
                       has_z,
                       delta_softplus,
                       query_start_loc,
                       cache_indices,
                       has_initial_state,
                       varlen,
                       null_block_id,
                       block_size,
                       block_idx_first_scheduled_token,
                       block_idx_last_scheduled_token,
                       initial_state_idx,
                       cu_chunk_seqlen,
                       last_chunk_indices
                       );


    const torch::stable::accelerator::DeviceGuard device_guard(u.get_device_index());
    auto stream = get_current_cuda_stream();
    DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), ssm_states.scalar_type(), "selective_scan_fwd", [&] {
        selective_scan_fwd_cuda<input_t, weight_t, state_t>(params, stream);
    });
}
