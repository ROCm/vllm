# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._ops import OpOverload

import vllm.model_executor.layers.quantization.utils.fp8_utils  # noqa: F401
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.activation_quant_fusion import ActivationQuantPattern
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
)
from vllm.platforms import current_platform

from .fusion import (
    FusedRMSQuantKey,
)

from .inductor_pass import enable_fake_mode
from .matcher_utils import (
    MatcherFusedAddRMSNorm,
    MatcherQuantFP8,
    MatcherRMSNorm,
    MatcherSiluAndMul,
)
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()


class AiterRMSNormQuantPattern:
    def __init__(
        self, epsilon: float, key: FusedRMSQuantKey, match_aiter_quant: bool = True
    ):
        self.epsilon = epsilon
        self.quant_dtype = key.quant.dtype

        self.rmsnorm_matcher = (
            MatcherRMSNorm(epsilon, match_rocm_aiter=True)
            if not key.fused_add
            else MatcherFusedAddRMSNorm(epsilon, match_rocm_aiter=True)
        )
        self.quant_matcher = MatcherQuantFP8(
            key.quant,
            match_rocm_aiter=match_aiter_quant,
        )


class AiterRMSNormDynamicQuantPattern(AiterRMSNormQuantPattern):
    """AITER RMSNorm + Dynamic Quantization pattern."""

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_fused_dynamic_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        match_aiter_quant: bool = True,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = self.rmsnorm_matcher(input, weight)
            result, scale = self.quant_matcher(result_rms)
            return result, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result = self.FUSED_OP(
                x=input,
                weight=weight,
                epsilon=self.epsilon,
                quant_dtype=self.quant_dtype,
            )

            return result[0], result[1]

        pm.register_replacement(
            pattern,
            replacement,
            self.rmsnorm_matcher.inputs(),
            pm.fwd_only,
            pm_pass,
        )


class AiterFusedAddRMSNormDynamicQuantPattern(AiterRMSNormQuantPattern):
    """AITER RMSNorm Fused Add + Dynamic Quantization pattern."""

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_fused_add_dynamic_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        match_aiter_quant: bool = True,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual_out = self.rmsnorm_matcher(input, weight, residual)
            result, scale = self.quant_matcher(result_rms)

            return result, residual_out, scale

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result = self.FUSED_OP(
                x=input,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon,
                quant_dtype=self.quant_dtype,
            )

            return result[0], result[1], result[2]

        pm.register_replacement(
            pattern,
            replacement,
            self.rmsnorm_matcher.inputs(),
            pm.fwd_only,
            pm_pass,
        )


class AiterRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    This pattern fuses aiter rms_norm & group fp8 quant custom
    ops into an aiter rms_norm_group_fp8_quant op.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = self.rmsnorm_matcher(input, weight)
            result, scale = self.quant_matcher(result_rms)
            return result, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at = self.FUSED_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )

            return at[0], at[1]

        pm.register_replacement(
            pattern, replacement, self.rmsnorm_matcher.inputs(), pm.fwd_only, pm_pass
        )


class AiterFusedAddRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    This pattern fuses aiter rms_norm_with_add & group fp8 quant custom ops
    into a aiter rms_norm_with_add_group_fp8_quant op.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_group_add_fused_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual_out = self.rmsnorm_matcher(input, weight, residual)
            result, scale = self.quant_matcher(result_rms)

            return result, residual_out, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            at = self.FUSED_OP(
                x=input,
                residual=residual,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )

            # result, scale, residual
            return at[0], at[1], at[2]

        pm.register_replacement(
            pattern, replacement, self.rmsnorm_matcher.inputs(), pm.fwd_only, pm_pass
        )


class RocmAiterRMSNormFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses aiter rms_norm & vllm/aiter quant custom ops
    into a fused rms_norm_quant op.
    It also supports fused_add_rms_norm.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_rms_norm_quant_fusion_pass"
        )

        # Make sure fused add patterns are before simple rms norm,
        # as the latter is a subset of the former in torch ops
        for epsilon in [1e-5, 1e-6]:
            #  Fuse aiter rms_norm + aiter dynamic group fp8 quant
            AiterRMSFp8GroupQuantPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128)
            ).register(self.patterns)

            # Fuse aiter fused_add_rms_norm + aiter dynamic group fp8 quant
            AiterFusedAddRMSFp8GroupQuantPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128)
            ).register(self.patterns)

            for match_aiter_quant in [True, False]:
                # Fuse aiter rms_norm + (aiter / vllm built-in)
                # dynamic per-token fp8 quant
                AiterRMSNormDynamicQuantPattern(
                    epsilon, FP8_DTYPE, match_aiter_quant=match_aiter_quant
                ).register(self.patterns)

                # Fuse aiter fused_add_rms_norm + (aiter / vllm built-in)
                # dynamic per-token fp8 quant
                AiterFusedAddRMSNormDynamicQuantPattern(
                    epsilon, FP8_DTYPE, match_aiter_quant=match_aiter_quant
                ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        fusion_patterns = [
            AiterRMSNormDynamicQuantPattern,
            AiterFusedAddRMSNormDynamicQuantPattern,
            AiterRMSFp8GroupQuantPattern,
            AiterFusedAddRMSFp8GroupQuantPattern,
        ]
        return self.hash_source(self, *fusion_patterns)


class AiterRMSNormMXFP4QuantPattern:
    """
    Fuses aiter rms_norm + aiter mxfp4_quant into
    aiter rmsnorm_mxfp4_quant.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_mxfp4_quant_op()
    QUANT_OP = rocm_aiter_ops.get_mxfp4_quant_op()

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon
        self.rmsnorm_matcher = MatcherRMSNorm(epsilon, match_rocm_aiter=True)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = self.rmsnorm_matcher(input, weight)
            x_q, x_s = self.QUANT_OP(x=result_rms)
            return x_q, x_s

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result = self.FUSED_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
            )
            return result[0], result[1]

        pm.register_replacement(
            pattern,
            replacement,
            self.rmsnorm_matcher.inputs(),
            pm.fwd_only,
            pm_pass,
        )


class AiterFusedAddRMSNormMXFP4QuantPattern:
    """
    Fuses aiter fused_add_rms_norm + aiter mxfp4_quant into
    aiter rmsnorm_with_add_mxfp4_quant.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_with_add_mxfp4_quant_op()
    QUANT_OP = rocm_aiter_ops.get_mxfp4_quant_op()

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(
            epsilon, match_rocm_aiter=True
        )

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual_out = self.rmsnorm_matcher(
                input, weight, residual
            )
            x_q, x_s = self.QUANT_OP(x=result_rms)
            return x_q, residual_out, x_s

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result = self.FUSED_OP(
                x=input,
                residual=residual,
                weight=weight,
                variance_epsilon=self.epsilon,
            )
            return result[0], result[1], result[2]

        pm.register_replacement(
            pattern,
            replacement,
            self.rmsnorm_matcher.inputs(),
            pm.fwd_only,
            pm_pass,
        )


class RocmAiterRMSNormMXFP4FusionPass(VllmPatternMatcherPass):
    """
    Fuses aiter rms_norm (or fused_add_rms_norm) + mxfp4_quant
    into a single fused kernel.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_rms_norm_mxfp4_quant_fusion_pass"
        )

        for epsilon in [1e-5, 1e-6]:
            AiterFusedAddRMSNormMXFP4QuantPattern(epsilon).register(
                self.patterns
            )
            AiterRMSNormMXFP4QuantPattern(epsilon).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s MXFP4 patterns", self.matched_count)

    def uuid(self) -> str:
        fusion_patterns = [
            AiterRMSNormMXFP4QuantPattern,
            AiterFusedAddRMSNormMXFP4QuantPattern,
        ]
        return self.hash_source(self, *fusion_patterns)


class AiterSiluMulFp8GroupQuantPattern(ActivationQuantPattern):
    """
    This pattern fuses aiter silu_and_mul & group fp8 quant custom
    ops into an aiter silu_and_mul_group_fp8_quant op.
    """

    FUSED_SILU_MUL_QUANT_OP = rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_op()

    def __init__(self, quant_op: OpOverload) -> None:
        self.silu_and_mul_matcher = MatcherSiluAndMul()
        self.quant_op = quant_op

    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.silu_and_mul_matcher.inputs()[0],
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at1 = self.silu_and_mul_matcher(input)
            at2 = self.quant_op(at1, 128)
            return at2[0], at2[1]

        def replacement(
            input: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at = self.FUSED_SILU_MUL_QUANT_OP(x=input, group_size=128)
            return at[0], at[1]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class RocmAiterSiluMulFp8GroupQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """

    AITER_GROUP_FP8_QUANT_OP = rocm_aiter_ops.get_group_quant_op()
    TRITON_GROUP_FP8_QUANT_OP = torch.ops.vllm.triton_per_token_group_quant_fp8.default

    QUANT_OPS = [AITER_GROUP_FP8_QUANT_OP, TRITON_GROUP_FP8_QUANT_OP]

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_silu_mul_fp8_group_quant_fusion_pass"
        )

        for quant_op in self.QUANT_OPS:
            AiterSiluMulFp8GroupQuantPattern(quant_op).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        fusion_patterns = [
            ActivationQuantPattern,
            AiterSiluMulFp8GroupQuantPattern,
        ]
        return VllmInductorPass.hash_source(self, *fusion_patterns)


class AiterGemmReduceRMSNormMXFP4Pattern:
    """
    Matches the MLA pattern:
      gemm_with_dynamic_quant -> split -> rmsnorm_mxfp4_quant(q_c)
                                       -> split -> rms_norm(kv_c)
                                                 -> k_pe

    Replaces with a single rocm_aiter_qkv_proj_layernorm op that calls
    gemm_afp4wfp4(skip_reduce=True) followed by fused_reduce_rms_mxfp4_quant,
    saving the separate reduce kernel launch.
    """

    FUSED_OP = torch.ops.vllm.rocm_aiter_qkv_proj_layernorm.default
    GEMM_OP = torch.ops.vllm.gemm_with_dynamic_quant.default
    RMSNORM_MXFP4_QUANT_OP = (
        torch.ops.vllm.rocm_aiter_rmsnorm_mxfp4_quant.default
    )
    RMSNORM_OP = torch.ops.vllm.rocm_aiter_rms_norm.default

    def __init__(
        self,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        epsilon: float,
    ) -> None:
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.epsilon = epsilon

    def register(self, pm_pass: PatternMatcherPass) -> None:
        q_lora_rank = self.q_lora_rank
        kv_lora_rank = self.kv_lora_rank
        qk_rope_head_dim = self.qk_rope_head_dim
        epsilon = self.epsilon
        N = q_lora_rank + kv_lora_rank + qk_rope_head_dim

        GEMM_OP = self.GEMM_OP
        RMSNORM_MXFP4_QUANT_OP = self.RMSNORM_MXFP4_QUANT_OP
        RMSNORM_OP = self.RMSNORM_OP
        FUSED_OP = self.FUSED_OP

        def pattern(
            x_q: torch.Tensor,
            x_s: torch.Tensor,
            w: torch.Tensor,
            ws: torch.Tensor,
            q_weight: torch.Tensor,
            kv_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            qkv = GEMM_OP(x_q, w, ws, False, torch.bfloat16, x_s)
            split1 = torch.ops.aten.split_with_sizes.default(
                qkv, [q_lora_rank, kv_lora_rank + qk_rope_head_dim], -1
            )
            q_c = operator.getitem(split1, 0)
            kv_lora = operator.getitem(split1, 1)
            rms_result = RMSNORM_MXFP4_QUANT_OP(
                q_c, q_weight, epsilon
            )
            q_cq = operator.getitem(rms_result, 0)
            q_cs = operator.getitem(rms_result, 1)
            split2 = torch.ops.aten.split_with_sizes.default(
                kv_lora, [kv_lora_rank, qk_rope_head_dim], -1
            )
            kv_c = operator.getitem(split2, 0)
            k_pe = operator.getitem(split2, 1)
            kv_c_normed = RMSNORM_OP(kv_c, kv_weight, epsilon)
            return q_cq, q_cs, kv_c_normed, k_pe

        def replacement(
            x_q: torch.Tensor,
            x_s: torch.Tensor,
            w: torch.Tensor,
            ws: torch.Tensor,
            q_weight: torch.Tensor,
            kv_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            result = FUSED_OP(
                x_q, x_s, w, ws,
                q_weight, epsilon,
                kv_weight, epsilon,
                q_lora_rank, kv_lora_rank, qk_rope_head_dim,
            )
            return (
                operator.getitem(result, 0),
                operator.getitem(result, 1),
                operator.getitem(result, 2),
                operator.getitem(result, 3),
            )

        M = 5
        K = 128
        x_q = torch.empty((M, K // 2), dtype=torch.uint8, device="cuda")
        x_s = torch.empty(
            (M, K // 32), dtype=torch.uint8, device="cuda"
        )
        w = torch.empty((N, K // 2), dtype=torch.uint8, device="cuda")
        ws = torch.empty(
            (N, K // 32), dtype=torch.uint8, device="cuda"
        )
        q_weight = torch.empty(
            q_lora_rank, dtype=torch.bfloat16, device="cuda"
        )
        kv_weight = torch.empty(
            kv_lora_rank, dtype=torch.bfloat16, device="cuda"
        )

        inputs = [x_q, x_s, w, ws, q_weight, kv_weight]
        pm.register_replacement(
            pattern, replacement, inputs, pm.fwd_only, pm_pass
        )


class RocmAiterGemmReduceRMSNormMXFP4FusionPass(VllmPatternMatcherPass):
    """
    Graph rewrite pass that fuses a split-k GEMM's internal reduce with
    downstream RMSNorm and MXFP4 quantization.

    Matches the MLA pattern:
      gemm_with_dynamic_quant -> split -> rmsnorm_mxfp4_quant(q_c)
                                       -> split -> rms_norm(kv_c)
                                                 -> k_pe

    Replaces with a single rocm_aiter_qkv_proj_layernorm op that calls
    gemm_afp4wfp4(skip_reduce=True) followed by fused_reduce_rms_mxfp4_quant,
    saving the separate reduce kernel launch.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_gemm_reduce_rmsnorm_mxfp4_fusion_pass"
        )

        hf_config = getattr(config.model_config, "hf_config", None)
        if hf_config is None:
            return

        q_lora_rank = getattr(hf_config, "q_lora_rank", None)
        kv_lora_rank = getattr(hf_config, "kv_lora_rank", None)
        qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", None)
        rms_norm_eps = getattr(hf_config, "rms_norm_eps", None)

        if not all(v is not None for v in [
            q_lora_rank, kv_lora_rank, qk_rope_head_dim, rms_norm_eps
        ]):
            return

        AiterGemmReduceRMSNormMXFP4Pattern(
            q_lora_rank, kv_lora_rank, qk_rope_head_dim, rms_norm_eps
        ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug(
            "Fused %d gemm+reduce+rmsnorm+mxfp4_quant patterns",
            self.matched_count,
        )

    def uuid(self) -> str:
        return self.hash_source(self, AiterGemmReduceRMSNormMXFP4Pattern)
