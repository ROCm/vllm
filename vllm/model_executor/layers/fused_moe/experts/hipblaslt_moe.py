# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 MoE experts on hipBLASLt (ROCm).

Each projection is one hipBLASLt grouped GEMM over the active experts, or one
dense W4A16 GEMM per active expert where hipBLASLt has no grouped solution.
Group sizes and pointers are written on the device, so the call is CUDA-graph
capturable. Only the per-expert fallback (no grouped solution) syncs.
"""

import functools
import os
from dataclasses import dataclass

import torch

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
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_unpermute,
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
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


def _align(n: int) -> int:
    return -(-n // ZERO_POINT_ALIGNMENT) * ZERO_POINT_ALIGNMENT


# gfx115x picks the DRAM channel from address bits 11:8 (16 x 256 B). Weight
# rows whose stride is a multiple of 256 B put rows read at the same K offset
# on few channels, so such strides get 128 B of padding.
_CHANNEL_BYTES = 256
_ROW_PAD_BYTES = 128


def pad_moe_weight_rows(w: torch.Tensor) -> torch.Tensor:
    """Return `w` ([E, N, K // 2] uint8) as a view whose row stride is not a
    multiple of 256 B; hipBLASLt reads it through lda."""
    E, N, row = w.shape
    if row % _CHANNEL_BYTES:
        return w
    padded = torch.zeros(E, N, row + _ROW_PAD_BYTES, dtype=w.dtype, device=w.device)
    padded[..., :row] = w
    return padded[..., :row]


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


# Up to this many routed rows, each row is its own group (n <= 1); above it,
# each expert is a group. Small batches thus get a grid sized for their rows.
_MAX_ROW_GROUPS = 64

# Large batches: "device" (default) writes the per-expert counts on the GPU into
# a plan sized for the total rows (graph-capturable, used under capture in every
# mode); "sbound" reads the counts once per layer and picks a plan sized for
# exactly the tiles they need (same E2E speed here); "host" sets up an
# exact-size problem per call.
_PREFILL_MODE = os.environ.get("VLLM_HIPBLASLT_MOE_PREFILL", "device")
_ROWS_BUCKET = 256
# "sbound": outside graph capture, bound = sum of per-group rows rounded up to
# the prefill N tile (MT64x32), host bounds {S, 0, ..., 0}; cached per S bucket.
_N_TILE = 32
_S_BUCKET = 128


@dataclass
class _Groups:
    """Device-side grouping of expert-sorted rows for one MoE layer call."""

    expert: torch.Tensor  # int32 [G]; E = empty group
    count: torch.Tensor  # int32 [G]
    offset: torch.Tensor  # int32 [G], first row of the group
    max_n: int  # bound on the total rows (1 = one row per group)
    # Per-expert counts on the host, read once per layer (gate_up and down
    # share the routing) by the exact-size path.
    host_counts: torch.Tensor | None = None

    @staticmethod
    def make(
        sorted_ids: torch.Tensor, flat: torch.Tensor, E: int, M: int, rows: int
    ) -> "_Groups":
        dev = sorted_ids.device
        if rows <= _MAX_ROW_GROUPS:
            return _Groups(
                expert=sorted_ids.to(torch.int32),
                count=torch.ones(rows, dtype=torch.int32, device=dev),
                offset=torch.arange(rows, dtype=torch.int32, device=dev),
                max_n=1,
            )
        counts = torch.zeros(E + 1, dtype=torch.int32, device=dev)
        counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
        counts = counts[:E]
        # The grid is sized for max_n rows in total, split over the groups on
        # the device; bucketed to bound the number of cached plans.
        return _Groups(
            expert=torch.arange(E, dtype=torch.int32, device=dev),
            count=counts,
            offset=(torch.cumsum(counts, 0, dtype=torch.int32) - counts),
            max_n=-(-rows // _ROWS_BUCKET) * _ROWS_BUCKET,
        )


@functools.cache
def _slot_consts(rows: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-slot count (1) and offset (slot index), built once per batch size."""
    return (
        torch.ones(rows, dtype=torch.int32, device=device),
        torch.arange(rows, dtype=torch.int32, device=device),
    )


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

    def _gemm(self, out, x, w, scale, groups: "_Groups", x_row_div: int = 1) -> None:
        assert self.quant_config.block_shape is not None
        args = (self.quant_config.block_shape[1], self.quant_config.w1_zp is not None)
        w = w.view(torch.int8)
        eager_batch = groups.max_n > 1 and not torch.cuda.is_current_stream_capturing()
        exact = eager_batch and _PREFILL_MODE == "host"
        max_n, zero_tail = groups.max_n, False
        if eager_batch and _PREFILL_MODE == "sbound":
            # Bound = rows rounded up per group to the N tile: no surplus tile
            # columns. Needs the counts on the host (one sync per layer).
            counts = self._host_counts(groups, w.size(0))
            tiles = (counts + _N_TILE - 1) // _N_TILE * _N_TILE
            max_n = -(-int(tiles.sum()) // _S_BUCKET) * _S_BUCKET
            zero_tail = True
            if max_n == 0:
                return  # nothing routed; the caller masks the empty rows
        label = (
            f"hipblaslt_moe_w4a16 {x.size(0)}x{out.size(1)}x{x.size(1)} "
            f"E={w.size(0)} G={groups.expert.numel()} "
            f"max_n={'exact' if exact else max_n}{'/S' if zero_tail else ''} "
            f"g={args[0]} {'asym' if args[1] else 'sym'}"
        )
        with record_function_or_nullcontext(label):
            if exact:
                self._gemm_host(out, x, w, scale, groups, args)
                return
            grouped = torch.ops._rocm_C.hipblaslt_w4a16_moe_grouped_gemm(
                out,
                x,
                w,
                scale,
                groups.expert,
                groups.count,
                groups.offset,
                max_n,
                *args,
                zero_tail,
                x_row_div,
            )
        if grouped:
            return
        assert groups.max_n > 1, (
            "hipBLASLt has no grouped W4A16 solution for this layer; the "
            "per-expert fallback needs expert-sorted rows, decode keeps slot order"
        )
        assert not torch.cuda.is_current_stream_capturing(), (
            "hipBLASLt has no grouped W4A16 solution for this layer; the "
            "per-expert fallback cannot be captured in a CUDA graph"
        )
        logger.warning_once(
            "hipBLASLt has no grouped W4A16 solution for N=%d K=%d; running "
            "one dense GEMM per expert.",
            out.size(1),
            x.size(1),
        )
        self._gemm_host(out, x, w, scale, groups, args)

    @staticmethod
    def _host_counts(groups: "_Groups", E: int) -> torch.Tensor:
        """Per-expert row counts on the host; one sync per layer (gate_up and
        down share the routing)."""
        if groups.host_counts is None:
            c = torch.zeros(E + 1, dtype=torch.int32, device=groups.count.device)
            e = groups.expert.long()
            c.index_add_(0, torch.where((e < 0) | (e > E), E, e), groups.count)
            groups.host_counts = c[:E].to("cpu", torch.int64)
        return groups.host_counts

    @staticmethod
    def _gemm_host(out, x, w, scale, groups: "_Groups", args) -> None:
        """Exact per-expert row counts on the host (one sync per layer). The op
        tries one grouped GEMM sized to those counts, else one dense GEMM per
        expert."""
        counts = HipblasltWNA16Experts._host_counts(groups, w.size(0))
        rows = int(counts.sum())
        torch.ops._rocm_C.hipblaslt_w4a16_moe_gemm(
            out[:rows], x[:rows], w, scale, counts, *args
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

        flat = topk_ids.flatten()
        weights = topk_weights.float()  # moe_unpermute scales in fp32
        if rows <= _MAX_ROW_GROUPS:
            # One group per routed slot, kept in slot order: no sort, gather or
            # scatter. gate_up reads token row slot // top_k directly; slots
            # with id < 0 get zero scales, so their rows are exact zeros.
            ones, offsets = _slot_consts(rows, flat.device)
            groups = _Groups(flat.to(torch.int32), ones, offsets, max_n=1)
            self._gemm(gate_up, hidden_states, w1, self.w1_scale, groups, top_k)
            self.activation(activation, act, gate_up)
            self._gemm(down, act, w2, self.w2_scale, groups)
            moe_unpermute(output, down, weights, offsets)
            return
        else:
            # Slots with id < 0 (no expert) sort last as pseudo-expert E.
            flat = flat.to(torch.int64)
            flat = torch.where(flat < 0, E, flat)
            order = torch.argsort(flat, stable=True)
            sorted_ids = flat[order]
            groups = _Groups.make(sorted_ids, flat, E, M, rows)
            x_sorted = hidden_states[order // top_k]
            self._gemm(gate_up, x_sorted, w1, self.w1_scale, groups)
            self.activation(activation, act, gate_up)
            self._gemm(down_sorted, act, w2, self.w2_scale, groups)
            down_sorted.masked_fill_((sorted_ids == E).unsqueeze(1), 0)
            # Weighted reduction straight from the expert-sorted rows.
            inv = torch.empty_like(order, dtype=torch.int32)
            inv[order] = torch.arange(rows, dtype=torch.int32, device=inv.device)
            moe_unpermute(output, down_sorted, weights, inv)
