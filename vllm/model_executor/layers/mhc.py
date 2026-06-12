# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

# this import will also register the custom ops
# import vllm.model_executor.kernels.mhc  # noqa: F401
import vllm.model_executor.kernels.mhc as mhc_kernels
from vllm.model_executor.custom_op import CustomOp
from vllm.utils.import_utils import has_tilelang

HAS_TILELANG = has_tilelang()

# Route the mHC pre/post/fused-post-pre blocks through the AITER HIP kernels
# (mhc_pre / mhc_post) instead of the tilelang kernels. These are ~2.2x lighter
# (see ATOM) and were previously disabled in vLLM over an aiter accuracy bug at
# large token counts; that bug is fixed upstream (ROCm/aiter b639cb63 "Fix
# sqrsum store race", plus #3417 mhc_pre_big_fuse accuracy and #3396 fused
# rmsnorm). Off by default; enable for DeepSeek-V4 on gfx950 with a recent aiter.
_USE_AITER_MHC = os.environ.get("VLLM_DSV4_AITER_MHC", "0") == "1"


# --8<-- [start:mhc_pre]
@CustomOp.register("mhc_pre")
class MHCPreOp(CustomOp):
    """MHC pre block.

    Computes mix logits from RMS-normalized HC residual streams, then
    returns post_mix, comb_mix, and
    layer_input = sum_i pre_mix_i * residual_i.
    """

    # --8<-- [end:mhc_pre]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.mhc_pre_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        )

    def forward_hip(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # AITER mHC kernels are ~2.2x lighter than tilelang (see ATOM). They were
        # disabled here over an aiter accuracy bug at large token counts; that bug
        # is fixed upstream (b639cb63 sqrsum race, #3417, #3396), so re-enable
        # behind VLLM_DSV4_AITER_MHC. The aiter pre op has no fused-rmsnorm-into-pre
        # path wired, so only take it when norm_weight is None (DSv4 norms separately).
        if (
            _USE_AITER_MHC
            and norm_weight is None
            and residual.shape[-1] % 256 == 0
        ):
            return torch.ops.vllm.mhc_pre_aiter(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
            )
        if HAS_TILELANG:
            return torch.ops.vllm.mhc_pre_tilelang(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                norm_weight,
                norm_eps,
            )
        else:
            return self.forward_native(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                norm_weight,
                norm_eps,
            )

    def forward_native(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc_kernels.mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )


# --8<-- [start:mhc_post]
@CustomOp.register("mhc_post")
class MHCPostOp(CustomOp):
    """MHC post block.

    Combines the layer output with the HC residual streams:
    out_j = post_layer_mix_j * x + sum_i comb_res_mix_ij * residual_i.
    """

    # --8<-- [end:mhc_post]

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.mhc_post_tilelang(
            x, residual, post_layer_mix, comb_res_mix
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        # See MHCPreOp.forward_hip: AITER mHC re-enabled behind VLLM_DSV4_AITER_MHC
        # now that the upstream accuracy fixes are present.
        if _USE_AITER_MHC and residual.shape[-1] % 256 == 0:
            return torch.ops.vllm.mhc_post_aiter(
                x, residual, post_layer_mix, comb_res_mix
            )
        if HAS_TILELANG:
            return torch.ops.vllm.mhc_post_tilelang(
                x, residual, post_layer_mix, comb_res_mix
            )
        else:
            return self.forward_native(x, residual, post_layer_mix, comb_res_mix)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return mhc_kernels.mhc_post_torch(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )


# --8<-- [start:hc_head]
@CustomOp.register("hc_head")
class HCHeadOp(CustomOp):
    """HC head reduction for DeepSeek V4.

    Computes gates from the RMS-normalized flattened HC residual and
    returns out = sum_i gate_i * residual_i, collapsing hc_mult streams
    to one.
    """

    # --8<-- [end:hc_head]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
            hs_flat,
            hc_fn,
            hc_scale,
            hc_base,
            rms_norm_eps,
            hc_eps,
        )
        return out.view(*outer_shape, hidden_size)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)

        if HAS_TILELANG:
            out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                rms_norm_eps,
                hc_eps,
            )
        else:
            num_tokens = hs_flat.shape[0]
            out = torch.empty(
                num_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            torch.ops.vllm.hc_head_triton(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                out,
                hidden_size,
                rms_norm_eps,
                hc_eps,
                hc_mult,
            )

        return out.view(*outer_shape, hidden_size)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError("Native implementation of hc_head is not available")


# --8<-- [start:mhc_fused_post_pre]
@CustomOp.register("mhc_fused_post_pre")
class MHCFusedPostPreOp(CustomOp):
    """Fused MHC post block followed by the next MHC pre block.

    Equivalent to applying MHCPostOp and then MHCPreOp to the updated
    residual streams, returning residual_cur, post_mix_cur, comb_mix_cur,
    and layer_input_cur.
    """

    # --8<-- [end:mhc_fused_post_pre]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # AITER has no single fused post+pre kernel; mhc_fused_post_pre_aiter
        # composes the two lighter aiter kernels (post then pre) and is
        # numerically equivalent. Gated on VLLM_DSV4_AITER_MHC; only when
        # norm_weight is None (no fused-rmsnorm-into-pre path wired).
        if (
            _USE_AITER_MHC
            and norm_weight is None
            and residual.shape[-1] % 256 == 0
        ):
            return torch.ops.vllm.mhc_fused_post_pre_aiter(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                tile_n,
            )
        return torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError(
            "Native implementation of mhc_fused_post_pre is not available"
        )
