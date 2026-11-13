# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 MoE experts on hipBLASLt (ROCm).

Each projection is one hipBLASLt grouped GEMM over the active experts, or one
dense W4A16 GEMM per active expert where hipBLASLt has no grouped solution.
Routing is resolved on the host (one sync per layer), so this backend cannot be
captured in a CUDA graph; run it with ``cudagraph_mode=NONE``.
"""

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mixed_precision.hipblaslt_w4a16 import (
    ZERO_POINT_ALIGNMENT,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonWNA16Experts,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static,
    kInt4Static32,
    kInt4Static32Asym,
    kInt4StaticAsym,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _align(n: int) -> int:
    return -(-n // ZERO_POINT_ALIGNMENT) * ZERO_POINT_ALIGNMENT


def pack_moe_scale_buffers(
    scale: torch.Tensor, qzeros: torch.Tensor | None
) -> torch.Tensor:
    """Per-expert hipBLASLt W4A16 scale allocations, as `[E, bytes]` uint8.

    Args:
        scale: `[E, N, K // G]` fp16/bf16 (must match the activation dtype).
        qzeros: `[E, N // 2, K // G]` uint8, two zero-points per byte along N
            (even row in the low nibble), or None for symmetric weights.

    Each row holds the `[N][K/G]` scales and, for asymmetric weights, at the
    next 256 B boundary the `[N/8][K/G]` 32-bit zero-point words with row `n`
    in nibble `n % 8`, as `hipblaslt_w4a16_gemm` expects.
    """
    E, N, G = scale.shape
    scale_bytes = scale.contiguous().view(torch.uint8).view(E, -1)
    if qzeros is None:
        return scale_bytes
    assert N % 8 == 0 and qzeros.shape == (E, N // 2, G), qzeros.shape
    zp_off = _align(scale_bytes.size(1))
    zp_bytes = N // 8 * G * 4
    buf = torch.zeros(
        E, _align(zp_off + zp_bytes), dtype=torch.uint8, device=scale.device
    )
    buf[:, : scale_bytes.size(1)] = scale_bytes
    # Byte j of word (n // 8, g) holds rows 8 * (n // 8) + 2j and + 2j + 1.
    zp = qzeros.view(torch.uint8).view(E, N // 8, 4, G).permute(0, 1, 3, 2)
    buf[:, zp_off : zp_off + zp_bytes] = zp.reshape(E, -1)
    return buf


class HipblasltWNA16Experts(mk.FusedMoEExpertsModular):
    """W4A16 experts on hipBLASLt.

    `w1` / `w2` are `[E, N, K // 2]` uint8 with nibbles in K order; the scales
    come from `pack_moe_scale_buffers`, and a non-None `w1_zp` marks the layer
    as asymmetric (the zero-points themselves live in the scale buffers).
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        assert quant_config.use_int4_w4a16, "Supports only int4_w4a16"
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_rocm():
            return False
        from vllm.model_executor.kernels.linear.mixed_precision.hipblaslt_w4a16 import (  # noqa: E501
            is_built,
        )

        return is_built()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return activation_key is None and weight_key in (
            kInt4Static,
            kInt4Static32,
            kInt4StaticAsym,
            kInt4Static32Asym,
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return TritonWNA16Experts._supports_activation(activation)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return TritonWNA16Experts._supports_parallel_config(moe_parallel_config)

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        assert a1.dim() == 2
        N = w1.size(1) // self.moe_config.w13_num_shards
        return w1.size(0), a1.size(0), N, a1.size(-1), topk_ids.size(1)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        gate_up = 2 * N if activation.is_gated else N
        return ((M, K), (M * topk * (gate_up + N + 2 * K),), (M, K))

    def _gemm(self, out, x, w, scale, counts) -> None:
        assert self.quant_config.block_shape is not None
        grouped = torch.ops._rocm_C.hipblaslt_w4a16_moe_gemm(
            out,
            x,
            w.view(torch.int8),
            scale,
            counts,
            self.quant_config.block_shape[1],
            self.quant_config.w1_zp is not None,
        )
        if not grouped:
            logger.info_once(
                "hipBLASLt has no grouped W4A16 solution for N=%d K=%d; "
                "running one dense GEMM per expert.",
                out.size(1),
                x.size(1),
            )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        assert self.w1_scale is not None and self.w2_scale is not None
        assert not apply_router_weight_on_input
        assert expert_map is None, "expert parallelism is not supported"

        E, M, N, K, top_k = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        rows = M * top_k
        widths = (w1.size(1), N, K, K)
        sizes = [rows * n for n in widths]
        gate_up, act, down_sorted, down = (
            t.view(rows, n)
            for t, n in zip(workspace2.view(-1)[: sum(sizes)].split(sizes), widths)
        )

        # Slots with id < 0 (no expert) sort last as pseudo-expert E.
        flat = topk_ids.flatten().to(torch.int64)
        flat = torch.where(flat < 0, E, flat)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=E + 1).cpu()
        valid = rows - int(counts[E])
        counts = counts[:E]

        x_sorted = hidden_states[order[:valid] // top_k]
        self._gemm(gate_up[:valid], x_sorted, w1, self.w1_scale, counts)
        self.activation(activation, act[:valid], gate_up[:valid])
        self._gemm(down_sorted[:valid], act[:valid], w2, self.w2_scale, counts)
        down_sorted[valid:].zero_()

        down[order] = down_sorted
        down = down.view(M, top_k, K)
        down.mul_(topk_weights.view(M, top_k, 1).to(down.dtype))
        ops.moe_sum(down, output)
