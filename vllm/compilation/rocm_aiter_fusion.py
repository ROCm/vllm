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
from .fx_utils import find_getitem, find_getitem_maybe, is_func
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


class RocmAiterGemmReduceRMSNormMXFP4FusionPass(VllmInductorPass):
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

    FUSED_OP = torch.ops.vllm.rocm_aiter_qkv_proj_layernorm.default
    GEMM_OP = torch.ops.vllm.gemm_with_dynamic_quant.default
    RMSNORM_MXFP4_QUANT_OP = (
        torch.ops.vllm.rocm_aiter_rmsnorm_mxfp4_quant.default
    )
    RMSNORM_OP = torch.ops.vllm.rocm_aiter_rms_norm.default

    def __init__(self, config: VllmConfig):
        super().__init__(config)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        count = 0
        for node in list(graph.nodes):
            if not is_func(node, self.GEMM_OP):
                continue

            match = self._match_pattern(node)
            if match is None:
                continue

            self._replace_pattern(graph, match)
            count += 1

        logger.debug(
            "Fused %d gemm+reduce+rmsnorm+mxfp4_quant patterns", count
        )

    def _match_pattern(
        self, gemm_node: fx.Node
    ) -> dict[str, fx.Node | int] | None:
        """
        Match the MLA subgraph pattern starting from a gemm_with_dynamic_quant
        node. Returns a dict of matched nodes, or None if no match.
        """
        split1_node = None
        for user in gemm_node.users:
            if is_func(user, torch.ops.aten.split_with_sizes.default):
                split1_node = user
                break

        if split1_node is None:
            return None

        q_c_item = find_getitem_maybe(split1_node, 0)
        kv_lora_item = find_getitem_maybe(split1_node, 1)

        if q_c_item is None or kv_lora_item is None:
            return None

        rmsnorm_mxfp4_quant_node = None
        for user in q_c_item.users:
            if is_func(user, self.RMSNORM_MXFP4_QUANT_OP):
                rmsnorm_mxfp4_quant_node = user
                break

        if rmsnorm_mxfp4_quant_node is None:
            return None

        split2_node = None
        for user in kv_lora_item.users:
            if is_func(user, torch.ops.aten.split_with_sizes.default):
                split2_node = user
                break

        if split2_node is None:
            return None

        kv_c_item = find_getitem_maybe(split2_node, 0)
        k_pe_item = find_getitem_maybe(split2_node, 1)

        if kv_c_item is None or k_pe_item is None:
            return None

        rmsnorm_node = None
        for user in kv_c_item.users:
            if is_func(user, self.RMSNORM_OP):
                rmsnorm_node = user
                break

        if rmsnorm_node is None:
            return None

        return {
            "gemm": gemm_node,
            "split1": split1_node,
            "q_c_item": q_c_item,
            "kv_lora_item": kv_lora_item,
            "rmsnorm_mxfp4_quant": rmsnorm_mxfp4_quant_node,
            "split2": split2_node,
            "kv_c_item": kv_c_item,
            "k_pe_item": k_pe_item,
            "rmsnorm": rmsnorm_node,
        }

    def _replace_pattern(
        self, graph: fx.Graph, match: dict[str, fx.Node | int]
    ) -> None:
        gemm_node = match["gemm"]
        split1_node = match["split1"]
        rmsnorm_mxfp4_quant_node = match["rmsnorm_mxfp4_quant"]
        rmsnorm_node = match["rmsnorm"]
        k_pe_item = match["k_pe_item"]

        # gemm_with_dynamic_quant(x_q, w, ws, asm_flag, dtype, x_s)
        x_q = gemm_node.args[0]
        w = gemm_node.args[1]
        ws = gemm_node.args[2]
        x_s = gemm_node.args[5]

        # rmsnorm_mxfp4_quant(q_c, weight, eps)
        q_a_weight = rmsnorm_mxfp4_quant_node.args[1]
        q_a_eps = rmsnorm_mxfp4_quant_node.args[2]

        # rms_norm(kv_c, weight, eps)
        kv_a_weight = rmsnorm_node.args[1]
        kv_a_eps = rmsnorm_node.args[2]

        # split dimensions
        split1_sizes = split1_node.args[1]
        split2_sizes = match["split2"].args[1]

        q_lora_rank = split1_sizes[0]
        kv_lora_rank = split2_sizes[0]
        qk_rope_head_dim = split2_sizes[1]

        # Insert fused op before the split
        with graph.inserting_before(split1_node):
            fused_node = graph.call_function(
                self.FUSED_OP,
                args=(
                    x_q,
                    x_s,
                    w,
                    ws,
                    q_a_weight,
                    q_a_eps,
                    kv_a_weight,
                    kv_a_eps,
                    q_lora_rank,
                    kv_lora_rank,
                    qk_rope_head_dim,
                ),
            )
            q_cq_node = graph.call_function(
                operator.getitem, args=(fused_node, 0)
            )
            q_cs_node = graph.call_function(
                operator.getitem, args=(fused_node, 1)
            )
            kv_c_normed_node = graph.call_function(
                operator.getitem, args=(fused_node, 2)
            )
            k_pe_reduced_node = graph.call_function(
                operator.getitem, args=(fused_node, 3)
            )

        # Compute FakeTensor metadata for the new nodes
        self._set_meta(
            gemm_node,
            fused_node,
            q_cq_node,
            q_cs_node,
            kv_c_normed_node,
            k_pe_reduced_node,
            x_q,
            x_s,
            w,
            ws,
            q_a_weight,
            q_a_eps,
            kv_a_weight,
            kv_a_eps,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
        )

        # Rewire downstream users
        rms_quant_q = find_getitem(rmsnorm_mxfp4_quant_node, 0)
        rms_quant_s = find_getitem(rmsnorm_mxfp4_quant_node, 1)

        rms_quant_q.replace_all_uses_with(q_cq_node)
        rms_quant_s.replace_all_uses_with(q_cs_node)
        rmsnorm_node.replace_all_uses_with(kv_c_normed_node)
        k_pe_item.replace_all_uses_with(k_pe_reduced_node)

        # Erase dead nodes in reverse dependency order
        self._erase_dead_nodes(
            graph,
            [
                rms_quant_q,
                rms_quant_s,
                rmsnorm_mxfp4_quant_node,
                match["q_c_item"],
                rmsnorm_node,
                match["kv_c_item"],
                k_pe_item,
                match["split2"],
                match["kv_lora_item"],
                split1_node,
                gemm_node,
            ],
        )

    def _set_meta(
        self,
        gemm_node,
        fused_node,
        q_cq_node,
        q_cs_node,
        kv_c_normed_node,
        k_pe_reduced_node,
        x_q,
        x_s,
        w,
        ws,
        q_a_weight,
        q_a_eps,
        kv_a_weight,
        kv_a_eps,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
    ):
        """Compute and set FakeTensor metadata for the new nodes."""
        gemm_fake = gemm_node.meta["val"]
        fake_mode = gemm_fake.fake_mode

        def _get_val(arg):
            return arg.meta["val"] if isinstance(arg, fx.Node) else arg

        with fake_mode:
            fake_out = self.FUSED_OP(
                _get_val(x_q),
                _get_val(x_s),
                _get_val(w),
                _get_val(ws),
                _get_val(q_a_weight),
                q_a_eps,
                _get_val(kv_a_weight),
                kv_a_eps,
                q_lora_rank,
                kv_lora_rank,
                qk_rope_head_dim,
            )

        fused_node.meta["val"] = fake_out
        q_cq_node.meta["val"] = fake_out[0]
        q_cs_node.meta["val"] = fake_out[1]
        kv_c_normed_node.meta["val"] = fake_out[2]
        k_pe_reduced_node.meta["val"] = fake_out[3]

    @staticmethod
    def _erase_dead_nodes(
        graph: fx.Graph, nodes: list[fx.Node]
    ) -> None:
        for node in nodes:
            if len(node.users) == 0:
                graph.erase_node(node)

    def uuid(self) -> str:
        return self.hash_source(self)
