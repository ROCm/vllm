# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16RowPadPass pads a W4A16 output or activation only when every reader
is known to honour strides, and never changes results."""

import pytest
import torch

import vllm
from tests.compile.backend import TestBackend
from vllm.config import CompilationConfig, CompilationMode, PassConfig, VllmConfig
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx115x  # noqa: E402

if not on_gfx115x():
    pytest.skip("gfx115x only", allow_module_level=True)

from vllm.compilation.passes.utility.w4a16_row_pad import (  # noqa: E402
    W4A16RowPadPass,
)
from vllm.config.scheduler import SchedulerConfig  # noqa: E402
from vllm.config.utils import Range  # noqa: E402
from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (  # noqa: E402
    MAX_SKINNY_BATCH_SIZE,
    pack_int4_exllama_shuffle,
    padded_row_stride_bytes,
)
from vllm.utils.platform_utils import num_compute_units  # noqa: E402

M, K, N, G = 32, 512, 256, 64
DTYPE = torch.bfloat16


def _ld(cols: int) -> int:
    return padded_row_stride_bytes(cols * 2) // 2


class Model(torch.nn.Module):
    def __init__(self, reader: str) -> None:
        super().__init__()
        w = torch.randint(0, 16, (N, K), dtype=torch.int32)
        self.w_q = pack_int4_exllama_shuffle(w).view(torch.int8)
        self.w_s = (0.05 * torch.rand(N, K // G)).to(DTYPE)
        self.cu_count = num_compute_units()
        self.reader = reader

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # An ATen producer: Inductor can lay x out padded.
        x = inp * 2.0
        y = torch.ops.vllm.rdna_hybrid_w4a16_apply(
            x, self.w_q, self.w_s, None, None, self.cu_count, G
        )
        if self.reader == "aten":
            d = N // 2
            return torch.nn.functional.silu(y[:, :d]) * y[:, d:]
        if self.reader == "opaque":
            out = torch.empty(M, N // 2, dtype=DTYPE)
            torch.ops._C.silu_and_mul(out, y)
            return out
        return y  # graph output


def _run(reader: str) -> tuple[TestBackend, torch.Tensor, torch.Tensor]:
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(pad_w4a16_rows=True),
        )
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        backend = TestBackend(W4A16RowPadPass(vllm_config))
        model = Model(reader)
        inp = (0.25 * torch.randn(M, K)).to(DTYPE)
        torch._dynamo.mark_dynamic(inp, 0)
        expected = model(inp)
        actual = torch.compile(model, backend=backend)(inp)
    return backend, expected, actual


def _apply_node(backend: TestBackend) -> torch.fx.Node:
    (node,) = backend.graph_post_pass.find_nodes(
        op="call_function", target=torch.ops.vllm.rdna_hybrid_w4a16_apply.default
    )
    return node


def test_pads_output_and_input_read_by_aten_ops():
    backend, expected, actual = _run("aten")

    node = _apply_node(backend)
    assert node.args[-1] == _ld(N)
    assert node.meta["val"].stride() == (_ld(N), 1)
    assert node.meta["eager_input_vals"][0][0].stride() == (_ld(K), 1)
    torch.testing.assert_close(actual, expected)


def test_keeps_output_read_by_opaque_op_dense():
    backend, expected, actual = _run("opaque")

    node = _apply_node(backend)
    assert node.meta["val"].stride() == (N, 1)
    torch.testing.assert_close(actual, expected)


def test_keeps_graph_output_dense():
    backend, expected, actual = _run("output")

    node = _apply_node(backend)
    assert node.meta["val"].stride() == (N, 1)
    assert actual.stride() == (N, 1)
    torch.testing.assert_close(actual, expected)


def test_decode_graphs_are_left_alone():
    """Enabling the pass splits off the skinny (decode) batch sizes into their
    own compile range, which the pass does not touch."""
    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        pass_config=PassConfig(pad_w4a16_rows=True),
    )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=2048, max_model_len=2048, is_encoder_decoder=False
        ),
        compilation_config=compilation_config,
    )
    decode, prefill = compilation_config.get_compile_ranges()

    assert decode == Range(start=1, end=MAX_SKINNY_BATCH_SIZE)
    assert prefill == Range(start=MAX_SKINNY_BATCH_SIZE + 1, end=2048)
    row_pad = W4A16RowPadPass(vllm_config)
    assert not row_pad.is_applicable_for_range(decode)
    assert row_pad.is_applicable_for_range(prefill)
