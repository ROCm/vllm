# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import operator

import torch
from torch import fx
from torch._guards import detect_fake_mode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
    MAX_SKINNY_BATCH_SIZE,
    padded_row_stride_bytes,
)

from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


class W4A16RowPadPass(VllmInductorPass):
    """Row-pad the activations and outputs of the gfx115x hybrid W4A16 GEMMs.

    gfx115x W4A16 GEMMs want rows that are an odd multiple of 128 B apart (see
    ``padded_row_stride_bytes``). A padded tensor is a view whose rows are
    further apart than its width, which kernels that index row t at t * width
    would misread, so padding is applied only where every reader is known to
    honour strides:

    - The output of a W4A16 op is padded when all its readers, through views,
      are ATen/prims ops (lowered by Inductor, or stride-aware ATen kernels)
      or the activation of another W4A16 op.
    - The activation x of a W4A16 op is padded when its producer is an ATen
      compute op whose readers pass the same test. The op's
      needs_exact_strides tag then makes Inductor lay x out padded, so the
      producer writes padded rows directly.

    Everything else (opaque custom ops, collectives, tensors leaving the
    graph, uncompiled code) keeps dense layouts. Only graphs compiled for
    batches above MAX_SKINNY_BATCH_SIZE are touched: decode uses the skinny
    kernel, which keeps dense layouts.
    """

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.start > MAX_SKINNY_BATCH_SIZE

    @staticmethod
    def _is_aten(node: fx.Node) -> bool:
        target = node.target
        return isinstance(target, torch._ops.OpOverload) and target.namespace in (
            "aten",
            "prims",
        )

    def _is_view(self, node: fx.Node) -> bool:
        return node.target is operator.getitem or (
            self._is_aten(node) and node.target.is_view
        )

    def _readers_honour_strides(self, node: fx.Node, op) -> bool:
        for user in node.users:
            if user.op != "call_function":
                return False  # graph output: the reader is unknown
            if user.target is op and user.args[0] is node:
                continue
            if self._is_view(user):
                if not self._readers_honour_strides(user, op):
                    return False
            elif not self._is_aten(user):
                return False
        return True

    @staticmethod
    def _row_stride(n_cols: int, elem_size: int) -> int:
        return padded_row_stride_bytes(n_cols * elem_size) // elem_size

    def _with_out_row_stride(self, op, args: tuple, kwargs: dict, ld: int) -> tuple:
        schema_args = op._schema.arguments
        full = list(args)
        for arg in schema_args[len(full) :]:
            full.append(kwargs.get(arg.name, arg.default_value))
        full[[a.name for a in schema_args].index("out_row_stride")] = ld
        return tuple(full)

    def _pad_output(self, node: fx.Node, op) -> bool:
        val = node.meta["val"]
        ld = self._row_stride(val.shape[-1], val.element_size())
        if ld == val.shape[-1] or not self._readers_honour_strides(node, op):
            return False
        node.args = self._with_out_row_stride(op, node.args, node.kwargs, ld)
        node.kwargs = {}
        # Inductor pairs these with node.args when applying needs_exact_strides.
        if "eager_input_vals" in node.meta:
            fake_args, fake_kwargs = node.meta["eager_input_vals"]
            fake_args = self._with_out_row_stride(op, fake_args, fake_kwargs, ld)
            node.meta["eager_input_vals"] = (fake_args, {})
        return True

    def _pad_input(self, node: fx.Node, op) -> bool:
        x = node.args[0]
        if (
            "eager_input_vals" not in node.meta
            or x.op != "call_function"
            or not self._is_aten(x)
            or self._is_view(x)
            or not self._readers_honour_strides(x, op)
        ):
            return False
        fake_args, fake_kwargs = node.meta["eager_input_vals"]
        fake_x = fake_args[0]
        ld = self._row_stride(fake_x.shape[-1], fake_x.element_size())
        if ld == fake_x.shape[-1]:
            return False
        with fake_x.fake_mode:
            padded = torch.empty_strided(
                fake_x.shape, (ld, 1), dtype=fake_x.dtype, device=fake_x.device
            )
        node.meta["eager_input_vals"] = ((padded, *fake_args[1:]), fake_kwargs)
        return True

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        op = getattr(torch.ops.vllm, "rdna_hybrid_w4a16_apply", None)
        if op is None:
            return
        op = op.default
        nodes = list(graph.find_nodes(op="call_function", target=op))
        n_out = sum(self._pad_output(n, op) for n in nodes)
        n_in = sum(self._pad_input(n, op) for n in nodes)
        if n_out:
            # Re-derive strides downstream of the padded outputs.
            inputs = [p.meta["val"] for p in graph.find_nodes(op="placeholder")]
            FakeTensorProp(
                graph.owning_module, mode=detect_fake_mode(inputs)
            ).propagate_dont_convert_inputs(*inputs)
        logger.debug(
            "W4A16RowPadPass padded %d/%d outputs, %d/%d inputs",
            n_out,
            len(nodes),
            n_in,
            len(nodes),
        )

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, padded_row_stride_bytes)
