# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable

import torch
from torch._ops import OpOverload
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import (PatternMatcherPass, fwd_only,
                                             register_replacement, Match)

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .inductor_pass import enable_fake_mode
from .fx_utils import find_getitem_maybe
from .multi_output_match import MultiOutputMatch
from .vllm_inductor_pass import VllmInductorPass


logger = init_logger(__name__)


def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16, device="cuda")


def empty_fp8(*args, **kwargs):
    fp8 = current_platform.fp8_dtype()
    return torch.empty(*args, **kwargs, dtype=fp8, device="cuda")


def empty_fp32(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.float32, device="cuda")


def empty_fp4(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.uint8, device="cuda")


class QuantMultiOutputMatch(MultiOutputMatch):

    def __init__(self, match: Match, quant_op, fused_op):
        super().__init__(match)
        assert isinstance(quant_op, OpOverload)
        assert isinstance(fused_op, OpOverload)
        self.QUANT_OP = quant_op  # in-place quant op
        self.FUSED_OP = fused_op  # in-place fused quant op

    def insert_fused_node(self, fused_return_mapping: dict[int, tuple[torch.fx.Node, int]], **kwargs):
        """
        This utility function inserts an auto-functionalized node for FUSED_OP.
        It also correctly sets its meta value and rebinds the users of the
        unfused nodes to use the fused node instead.

        :param fused_return_mapping: A dictionary, mapping from getitem indices
        of the fused node result to a tuple of the old node and a getitem index.
        :param kwargs: kwargs that get directly forwarded to the auto_fn node

        Example:
        If we want to replace this graph:
        _, x1, x2 = auto_fn(op1)
        _, y1, y2 = auto_fn(op2)

        with
        _, x1, y2, x2 = auto_fn(FUSED_OP)

        we would call:
        insert_fused_node({1: (op1_node, 1), 2: (op2_node, 2), 3: (op1_node, 2)}

        Note that the 0th element is None for auto-functionalized in-place ops.
        Hence, others appear 1-indexed.
        """
        fused_node = self.insert_auto_fn(self.FUSED_OP, kwargs)
        indices = fused_return_mapping.keys()
        getitem_nodes = self.insert_getitems(fused_node, indices)

        # Prepare the meta value, use a list so it's mutable
        meta_val = [None] * (max(indices) + 1)

        # Iterate through elements of the tuple produced by fused_node
        for idx, getitem_node in zip(indices, getitem_nodes):
            old_node, old_idx = fused_return_mapping[idx]

            # If the old value was never used, the old_getitem might not exist
            old_getitem = find_getitem_maybe(old_node, old_idx)
            if old_getitem is not None:
                # Rebind the users of match getitem nodes to use the new nodes.
                # The old nodes will be removed by DCE at the end of the pass.
                old_getitem.replace_all_uses_with(getitem_node)
                getitem_node.meta["val"] = old_getitem.meta["val"]

            # Extract the appropriate meta value
            # It is present even if the getitem node does not exist
            meta_val[idx] = old_node.meta["val"][old_idx]

        # Fix the meta value on the new fused node
        fused_node.meta["val"] = tuple(meta_val)


ADD_RMS_OP = torch.ops._C.fused_add_rms_norm.default


class AddRMSNormMXFP4GemmPattern:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.FUSED_OP = torch.ops.vllm.add_rmsnorm_mxfp4_gemm.default
        self.QUANT_F4GEMM_OP = torch.ops.vllm.gemm_with_dynamic_quant.default
        
    def register(self, pm_pass: PatternMatcherPass, record_match: Callable[[MultiOutputMatch], bool]):

        def pattern(
            result: torch.Tensor, result_rms: torch.Tensor,
            input: torch.Tensor, residual_out: torch.Tensor,
            residual: torch.Tensor, weight_rms: torch.Tensor,
            weight_gemm: torch.Tensor, scale: torch.Tensor):
            at1 = auto_functionalized(ADD_RMS_OP,
                                    result=result_rms,
                                    input=input,
                                    residual_out=residual_out,
                                    residual=residual,
                                    weight=weight_rms,
                                    epsilon=self.epsilon)
            at2 = auto_functionalized(self.QUANT_F4GEMM_OP,
                                    result=result,
                                    x=at1[1],
                                    weight=weight_gemm,
                                    weight_scale=scale,
                                    x_scales=None)
            return at2[1], at1[2]
        
        def replacement(
            result: torch.Tensor, result_rms: torch.Tensor,
            input: torch.Tensor, residual_out: torch.Tensor,
            residual: torch.Tensor, weight_rms: torch.Tensor,
            weight_gemm: torch.Tensor, scale: torch.Tensor):
            at = auto_functionalized(self.FUSED_OP,
                                    result=result,
                                    input=input,
                                    residual_out=residual_out,
                                    residual=residual,
                                    weight_rms=weight_rms,
                                    weight_gemm=weight_gemm,
                                    scale=scale,
                                    epsilon=self.epsilon)
            return at[1], at[2]

        inputs = [
            empty_bf16(4, 4),  # result
            empty_bf16(4, 4),  # result_rms
            empty_bf16(4, 4),  # input
            empty_bf16(4, 4),  # residual_out
            empty_bf16(4, 4),  # residual
            empty_bf16(1, 4),   # weight_rms
            empty_fp4(4, 4),   # weight_gemm
            empty_fp4(1, 1),    # scale
        ]

        register_replacement(
            pattern,
            replacement,
            inputs,
            fwd_only,
            pm_pass,
            extra_check=lambda m: record_match(
                self.Match(m, self.QUANT_F4GEMM_OP, self.FUSED_OP)))

    class Match(QuantMultiOutputMatch):

        def process(self):
            # Find the nodes in the match that we need to rebind
            add_rms_node = self.find_auto_fn(ADD_RMS_OP)
            quant_f4gemm_node = self.find_auto_fn(self.QUANT_OP)

            assert len(add_rms_node.users) == 2
            assert len(quant_f4gemm_node.users) == 1

            # First, insert a new auto_functionalized node for the fused op,
            # as well as getitem nodes to extract the result and residual.
            # The auto_fn node returns a tuple of (None, result, residual).
            #
            # The resulting graph looks like this:
            # at = auto_functionalized(torch.ops._C.fused_add_rms_norm_static_fp8_quant.default, ...)  # noqa
            # result_node_new = at[1]
            # residual_node_new = at[2]
            with self.inserting_after_match():
                # Missing epsilon, scalars cannot be inputs to the pattern
                kwargs = self.match.kwargs.copy()
                del kwargs["result_rms"]  # not used in the fused op
                # 0 is always None
                fused_return_mapping = {1: (quant_f4gemm_node, 1), 2: (add_rms_node, 2)}
                self.insert_fused_node(fused_return_mapping,
                                       **kwargs,
                                       epsilon=add_rms_node.kwargs["epsilon"])


class ROCmFusionPass(VllmInductorPass):
    """
    This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.matches: list[MultiOutputMatch] = []
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_fusion_pass")
        
        for epsilon in [1e-5, 1e-6]:
            AddRMSNormMXFP4GemmPattern(epsilon).register(
                self.patterns, self.record_match)
            
            # WARNING: This is a hack to clear the pattern matcher cache
            # and allow multiple values of epsilon.
            torch._inductor.pattern_matcher._seen_patterns.clear()
    
    def record_match(self, match: MultiOutputMatch) -> bool:
        # Hijack the extra_check to record the match and
        # save it for post-processing.
        self.matches.append(match)

        # Return False to prevent automatic replacement.
        return False
    
    def process_matches(self, graph: torch.fx.Graph):
        """
        Manually process multi-output matches and replace them with fused nodes.
        See MultiOutputMatch for more details.
        """
        for match in self.matches:
            match.process()

        # Finally, remove matched nodes
        graph.eliminate_dead_code()
        assert all(node not in graph.nodes for match in self.matches
                   for node in match.match.nodes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_rocm_fusion")

        count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", count)
        self.dump_graph(graph, "after_pattern_match")

        # Manually process multi-output matches (and run DCE)
        self.process_matches(graph)
        logger.debug("Post-processed %s matches", len(self.matches))
        self.dump_graph(graph, "after_rocm_fusion")
        self.matches.clear()
        self.end_and_log()
