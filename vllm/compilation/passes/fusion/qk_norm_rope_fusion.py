# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import ParamSpec

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_fp32, empty_i64


def _apply_interleaved_rope_functional(
    x: torch.Tensor, mrope_section: list[int]
) -> torch.Tensor:
    """Functional version of apply_interleaved_rope mirroring the form
    produced by functionalization of the in-place version in the model:

        slice_2 = slice(clone, 1, 1, 60, 3)
        slice_1 = slice(select(x, 0, k), 1, 1, 60, 3)
        copy   = copy.default(slice_2, slice_1)
        clone  = slice_scatter(clone, copy, 1, 1, 60, 3)
    """
    x_t = x[0].clone()
    h_end = mrope_section[1] * 3
    w_end = mrope_section[2] * 3
    # NOTE: dim is the *positive* last-axis index (1 for the [T, 64] slice
    # produced by select(x, 0, k) on a [3, T, 64] tensor).  Functionalization
    # in the actual model graph uses the positive form, so we must too.
    last_dim = 1
    # H slice writeback.
    src_h = torch.ops.aten.slice.Tensor(
        torch.ops.aten.select.int(x, 0, 1), last_dim, 1, h_end, 3
    )
    dst_h = torch.ops.aten.slice.Tensor(x_t, last_dim, 1, h_end, 3)
    cp_h = torch.ops.aten.copy.default(dst_h, src_h)
    x_t = torch.ops.aten.slice_scatter.default(x_t, cp_h, last_dim, 1, h_end, 3)
    # W slice writeback.
    src_w = torch.ops.aten.slice.Tensor(
        torch.ops.aten.select.int(x, 0, 2), last_dim, 2, w_end, 3
    )
    dst_w = torch.ops.aten.slice.Tensor(x_t, last_dim, 2, w_end, 3)
    cp_w = torch.ops.aten.copy.default(dst_w, src_w)
    x_t = torch.ops.aten.slice_scatter.default(x_t, cp_w, last_dim, 2, w_end, 3)
    return x_t


logger = init_logger(__name__)

FUSED_QK_ROPE_OP = torch.ops._C.fused_qk_norm_rope.default
FUSED_QK_MROPE_OP = torch.ops._C.fused_qk_norm_mrope.default

P = ParamSpec("P")


class QkNormRopePattern:
    """
    Match the unfused sequence in attention blocks and replace with the fused op.

    Unfused (conceptually):
      q, k, v = split(qkv, [qsz, kvsz, kvsz], -1)
      qh = reshape(q, [-1, num_heads, head_dim])
      kh = reshape(k, [-1, num_kv_heads, head_dim])
      qn = rms_norm(qh, q_weight, eps)
      kn = rms_norm(kh, k_weight, eps)
      qf = reshape(qn, [-1, num_heads * head_dim])
      kf = reshape(kn, [-1, num_kv_heads * head_dim])
      qf, kf = rotary_embedding(positions, qf, kf, head_dim, cos_sin_cache, is_neox)
      return qf, kf, v

    Fused replacement:
      fused_qk_norm_rope(qkv, num_heads, num_kv_heads, num_kv_heads, head_dim,
                         eps, q_weight, k_weight, cos_sin_cache, is_neox,
                         positions.view(-1))
      return split(qkv, [qsz, kvsz, kvsz], -1)
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
        is_neox: bool,
        rope_flashinfer: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        self.is_neox = is_neox
        self.rope_flashinfer = rope_flashinfer
        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_flashinfer=self.rope_flashinfer,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        # Sample inputs to help pattern tracing
        T = 5
        qkv = empty_bf16(T, self.q_size + 2 * self.kv_size)
        positions = empty_i64(T)
        q_weight = empty_bf16(1, self.head_dim)
        k_weight = empty_bf16(1, self.head_dim)
        if self.rope_flashinfer:
            cos_sin_cache = empty_fp32(4096, self.head_dim)
        else:
            cos_sin_cache = empty_bf16(4096, self.head_dim)
        return [
            qkv,
            positions,
            q_weight,
            k_weight,
            cos_sin_cache,
        ]

    @staticmethod
    def wrap_trace_fn(
        trace_fn: Callable[P, fx.GraphModule],
        *process_fx_fns: Callable[[fx.GraphModule], None],
    ) -> Callable[P, fx.GraphModule]:
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> fx.GraphModule:
            gm = trace_fn(*args, **kwargs)
            for process_fx in process_fx_fns:
                process_fx(gm)

            return gm

        return wrapped

    @staticmethod
    def fx_view_to_reshape(gm: torch.fx.GraphModule) -> None:
        from torch._inductor.fx_passes.post_grad import view_to_reshape

        view_to_reshape(gm)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # split qkv -> q,k,v
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # Q path: view -> RMS -> view back to q.shape
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_normed_by_head = vllm.ir.ops.rms_norm(q_by_head, q_weight, self.eps)
            q_flat = q_normed_by_head.view(q.shape)

            # K path: view -> RMS -> view back to k.shape
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_normed_by_head = vllm.ir.ops.rms_norm(k_by_head, k_weight, self.eps)
            k_flat = k_normed_by_head.view(k.shape)

            # RoPE: apply to flattened q/k
            q_rope, k_rope = self.rope_matcher(positions, q_flat, k_flat, cos_sin_cache)
            return q_rope, k_rope, v

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Run fused qk_norm_rope op
            result = auto_functionalized(
                FUSED_QK_ROPE_OP,
                qkv=qkv,
                num_heads_q=self.num_heads,
                num_heads_k=self.num_kv_heads,
                num_heads_v=self.num_kv_heads,
                head_dim=self.head_dim,
                eps=self.eps,
                q_weight=q_weight,
                k_weight=k_weight,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                position_ids=positions.view(-1),
                forced_token_heads_per_warp=-1,
            )
            result_qkv = result[1]

            # Split back to q,k,v and return
            return result_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # type: ignore[no-any-return]

        # NOTE: use fx_view_to_reshape to unify view/reshape to simplify
        # pattern and increase matching opportunities
        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            QkNormRopePattern.wrap_trace_fn(
                pm.fwd_only,
                QkNormRopePattern.fx_view_to_reshape,
            ),
            pm_pass,
        )


class QkNormMRopePattern:
    """
    Match Q-norm + K-norm + MRotaryEmbedding (multimodal RoPE) pattern.

    Unlike QkNormRopePattern, this matches the M-rope decomposition where
    positions has shape [3, num_tokens] and cos/sin are picked per-dim from
    one of three axes (T/H/W) according to mrope_section [t, h, w].

    Replaces with a single fused_qk_norm_mrope HIP kernel call.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
        is_neox: bool,
        mrope_section: tuple[int, int, int],
        mrope_interleaved: bool,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        self.is_neox = is_neox
        self.mrope_section = tuple(mrope_section)
        self.mrope_interleaved = mrope_interleaved

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        qkv = empty_bf16(T, self.q_size + 2 * self.kv_size)
        # Positions for mrope are [3, T] (T/H/W positions per token).
        positions = empty_i64(3, T)
        q_weight = empty_bf16(1, self.head_dim)
        k_weight = empty_bf16(1, self.head_dim)
        cos_sin_cache = empty_bf16(4096, self.rotary_dim)
        return [qkv, positions, q_weight, k_weight, cos_sin_cache]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # Inline the mrope decomposition directly so the FX graph the matcher
        # produces matches the model's actual graph (no spurious flatten/view
        # round-trips between RMSNorm and rotary).
        head_dim = self.head_dim
        rotary_dim = self.rotary_dim
        num_q_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        q_size = self.q_size
        kv_size = self.kv_size
        eps = self.eps
        mrope_section = self.mrope_section
        mrope_interleaved = self.mrope_interleaved

        def _rotate_one_neox(x_by_head, cos2d, sin2d):
            # Mirror ApplyRotaryEmb.forward_static for is_neox=True.
            # Each call unsqueezes cos/sin internally so that the FX graph
            # has a *separate* unsqueeze per (Q, K) path — matching how
            # MRotaryEmbedding.forward_native invokes apply_rotary_emb
            # twice (once for query, once for key), which is the form
            # Inductor produces in the actual model graph.
            cos = cos2d.unsqueeze(-2)
            sin = sin2d.unsqueeze(-2)
            x_rot = x_by_head[..., :rotary_dim]
            x_pass = x_by_head[..., rotary_dim:]
            x1, x2 = torch.chunk(x_rot, 2, dim=-1)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            x_rot_out = torch.cat((o1, o2), dim=-1)
            x_out = torch.cat((x_rot_out, x_pass), dim=-1)
            return x_out

        def _rotate_neox(q_by_head, k_by_head, cos2d, sin2d):
            q_out = _rotate_one_neox(q_by_head, cos2d, sin2d)
            k_out = _rotate_one_neox(k_by_head, cos2d, sin2d)
            return q_out, k_out

        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

            # Q-norm: view→rms_norm→ (no flatten back).
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
            q_normed = vllm.ir.ops.rms_norm(q_by_head, q_weight, eps)

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
            k_normed = vllm.ir.ops.rms_norm(k_by_head, k_weight, eps)

            # MRoPE cos/sin computation.
            cos_sin = cos_sin_cache[positions]
            cos, sin = cos_sin.chunk(2, dim=-1)
            if mrope_interleaved:
                cos = _apply_interleaved_rope_functional(cos, list(mrope_section))
                sin = _apply_interleaved_rope_functional(sin, list(mrope_section))
            else:
                cos = torch.cat(
                    [
                        m[i]
                        for i, m in enumerate(cos.split(list(mrope_section), dim=-1))
                    ],
                    dim=-1,
                )
                sin = torch.cat(
                    [
                        m[i]
                        for i, m in enumerate(sin.split(list(mrope_section), dim=-1))
                    ],
                    dim=-1,
                )

            q_out, k_out = _rotate_neox(q_normed, k_normed, cos, sin)
            return q_out, k_out, v

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            q = q.contiguous()
            k = k.contiguous()
            mt, mh, mw = mrope_section
            # The fused kernel requires contiguous tensors.  In the actual
            # model, `positions` may be a non-contiguous slice/view of a
            # larger position buffer, and `cos_sin_cache` is a (large) cache
            # that should already be contiguous but defensively materialise.
            result = auto_functionalized(
                FUSED_QK_MROPE_OP,
                q=q,
                k=k,
                num_q_heads=num_q_heads,
                num_k_heads=num_kv_heads,
                head_dim=head_dim,
                eps=eps,
                q_weight=q_weight,
                k_weight=k_weight,
                cos_sin_cache=cos_sin_cache.contiguous(),
                positions=positions.contiguous(),
                mrope_section_t=mt,
                mrope_section_h=mh,
                mrope_section_w=mw,
                mrope_interleaved=mrope_interleaved,
            )
            # Reshape back to per-head shape (matches the pattern output).
            q_out = result[1].view(*q.shape[:-1], num_q_heads, head_dim)
            k_out = result[2].view(*k.shape[:-1], num_kv_heads, head_dim)
            return q_out, k_out, v

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            QkNormRopePattern.wrap_trace_fn(
                pm.fwd_only,
                QkNormRopePattern.fx_view_to_reshape,
            ),
            pm_pass,
        )


class QKNormRoPEFusionPass(VllmPatternMatcherPass):
    """Fuse Q/K RMSNorm + RoPE into fused_qk_norm_rope when the custom op exists."""

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="qk_norm_rope_fusion_pass"
        )

        dtype = config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once(
                "QK Norm+RoPE fusion not enabled: unsupported dtype %s", dtype
            )
            return

        # use one attn layer to get meta (such as head_dim) for QkNormRopePattern
        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(
            config, Attention
        )
        if len(attn_layers) == 0:
            logger.warning_once(
                "QK Norm+RoPE fusion enabled, but no Attention layers were discovered."
            )
            return
        layer = next(iter(attn_layers.values()))

        # Discover whether the model uses MRotaryEmbedding (multimodal RoPE).
        # If so, register a separate fused-mrope pattern.  When +rotary_embedding
        # is in custom_ops MRotaryEmbedding's forward_cuda dispatches to the
        # opaque triton_mrope kernel which the matcher can't see, so we only
        # try to match the mrope pattern when MRotaryEmbedding is using its
        # native (decomposed) forward.
        mrope_section: tuple[int, int, int] | None = None
        mrope_interleaved: bool = False
        mrope_head_size: int | None = None
        mrope_rotary_dim: int | None = None
        if not RotaryEmbedding.enabled():
            text_cfg = getattr(config.model_config.hf_config, "text_config", None)
            if text_cfg is None:
                text_cfg = config.model_config.hf_config
            # Qwen3-Omni nests text under thinker_config.text_config.
            thinker_cfg = getattr(config.model_config.hf_config, "thinker_config", None)
            if thinker_cfg is not None:
                inner = getattr(thinker_cfg, "text_config", None)
                if inner is not None:
                    text_cfg = inner
            rope_scaling = getattr(text_cfg, "rope_scaling", None)
            if rope_scaling is not None:
                section = rope_scaling.get("mrope_section")
                if section is not None and len(section) == 3:
                    mt, mh, mw = section
                    mrope_section = (int(mt), int(mh), int(mw))
                    mrope_interleaved = bool(
                        rope_scaling.get("mrope_interleaved", False)
                        or rope_scaling.get("interleaved", False)
                    )
                    mrope_head_size = getattr(text_cfg, "head_dim", None)
                    if mrope_head_size is None:
                        mrope_head_size = getattr(text_cfg, "hidden_size", 0) // max(
                            getattr(text_cfg, "num_attention_heads", 1), 1
                        )
                    mrope_rotary_dim = mrope_head_size  # full rotary by default

        for epsilon in [1e-5, 1e-6]:
            for neox in [True, False]:
                if RotaryEmbedding.enabled():
                    for rope_flashinfer in [False, True]:
                        QkNormRopePattern(
                            head_dim=layer.head_size,
                            num_heads=layer.num_heads,
                            num_kv_heads=layer.num_kv_heads,
                            eps=epsilon,
                            is_neox=neox,
                            rope_flashinfer=rope_flashinfer,
                        ).register(self.patterns)
                else:
                    QkNormRopePattern(
                        head_dim=layer.head_size,
                        num_heads=layer.num_heads,
                        num_kv_heads=layer.num_kv_heads,
                        eps=epsilon,
                        is_neox=neox,
                    ).register(self.patterns)

        if (
            mrope_section is not None
            and mrope_head_size in (64, 128, 256)
            and mrope_rotary_dim == mrope_head_size
        ):
            mt, mh, mw = mrope_section
            for epsilon in [1e-5, 1e-6]:
                # Qwen3-Omni / Qwen3-VL use is_neox=True; only register that
                # to keep pattern count small.
                QkNormMRopePattern(
                    head_dim=mrope_head_size,
                    rotary_dim=mrope_rotary_dim,
                    num_heads=layer.num_heads,
                    num_kv_heads=layer.num_kv_heads,
                    eps=epsilon,
                    is_neox=True,
                    mrope_section=(mt, mh, mw),
                    mrope_interleaved=mrope_interleaved,
                ).register(self.patterns)
            logger.info(
                "QK Norm + MRoPE fusion enabled (mrope_section=%s, interleaved=%s)",
                mrope_section,
                mrope_interleaved,
            )

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Fused QK Norm+RoPE on %s sites", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, QkNormRopePattern, QkNormMRopePattern)
