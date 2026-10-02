# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from collections.abc import Callable
from typing import ParamSpec

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_fp32, empty_i64

logger = init_logger(__name__)

P = ParamSpec("P")

# Head sizes AITER's fused_qk_norm_rope_cache_pts_quant_shuffle() supports;
# other sizes hard-abort, so skip those layers.  An impl with another kernel
# declares its own (fused_qk_norm_rope_kvcache_head_sizes).
SUPPORTED_FUSED_QK_NORM_ROPE_KVCACHE_HEAD_DIMS: tuple[int, ...] = (64, 128, 256)


# ---------------------------------------------------------------------------
# Custom op: fused QK-norm + RoPE + KV cache update
# ---------------------------------------------------------------------------


def fused_qk_norm_rope_and_unified_kv_cache_update_impl(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: LayerNameType,
    v_norm: bool = False,
    v_out: torch.Tensor | None = None,
) -> torch.Tensor:
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        # Only impls that declare fused_qk_norm_rope_kvcache_v_norm get the
        # weightless V norm (Gemma 4), and v_out to write the normalised V to.
        extra = {"v_norm": True, "v_out": v_out} if v_norm else {}
        attn_layer.impl.do_qk_norm_rope_kvcache_update(
            attn_layer,
            qkv,
            q_out,
            k_out,
            positions,
            q_weight,
            k_weight,
            rms_norm_eps,
            cos_sin_cache,
            is_neox,
            kv_cache,
            layer_slot_mapping,
            **extra,
        )
    else:
        # Profiling/dummy run: define q_out/k_out (consumed by attention).
        q_out.zero_()
        k_out.zero_()
        if v_out is not None:
            v_out.zero_()

    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


def fused_qk_norm_rope_and_unified_kv_cache_update_fake(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: LayerNameType,
    v_norm: bool = False,
    v_out: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


direct_register_custom_op(
    op_name="fused_qk_norm_rope_and_unified_kv_cache_update",
    op_func=fused_qk_norm_rope_and_unified_kv_cache_update_impl,
    mutates_args=["q_out", "k_out", "v_out"],
    fake_impl=fused_qk_norm_rope_and_unified_kv_cache_update_fake,
)


# ---------------------------------------------------------------------------
# Pattern: QK-norm + RoPE + unified_kv_cache_update
# ---------------------------------------------------------------------------


class QkNormRopeKvCachePattern:
    """
    Match the unfused sequence:
      q, k, v = split(qkv, ...)
      q = rms_norm(q.view(heads), q_weight).view(flat)
      k = rms_norm(k.view(heads), k_weight).view(flat)
      q, k = rotary_embedding(positions, q, k, cos_sin_cache, is_neox)
      q = q.view(num_heads, head_dim)
      k = k.view(num_kv_heads, head_dim)
      v = v.view(num_kv_heads, head_dim)
      dummy = unified_kv_cache_update(k, v, layer_name)

    Replace with:
      q_out = empty(...)
      k_out = empty(...)
      dummy = fused_qk_norm_rope_and_unified_kv_cache_update(
          q_out, k_out, qkv, positions, q_weight, k_weight,
          eps, cos_sin_cache, is_neox, layer_name)
      v = split(qkv, ...)[2].view(num_kv_heads, head_dim)
    """

    FUSED_OP = torch.ops.vllm.fused_qk_norm_rope_and_unified_kv_cache_update.default

    def __init__(
        self,
        layer: Attention,
        eps: float,
        is_neox: bool,
        quant_query: bool,
        v_norm: bool = False,
        shapes: dict[tuple[int, ...], tuple[int, ...]] | None = None,
    ) -> None:
        self.layer_name = layer.layer_name
        self.eps = eps
        self.is_neox = is_neox
        self.quant_query = quant_query
        # Gemma 4 normalises V (weightless RMSNorm) before the cache write.
        self.v_norm = v_norm
        self._set_shape(
            layer.num_heads, layer.num_kv_heads, layer.head_size, layer.head_size_v
        )
        # The pattern ignores sizes, and with LayerName the layer too, so one
        # pattern matches layers of every shape: (q, k, v) split sizes ->
        # (heads, kv heads, head size, v head size) of the layers it may
        # replace, the matched one's set before its replacement is traced.
        # None: this layer's shape only.
        self.shapes = shapes

        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )

    def _set_shape(self, num_heads, num_kv_heads, head_size, head_size_v) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size_v
        self.q_size = num_heads * head_size
        self.k_size = num_kv_heads * head_size
        self.v_size = num_kv_heads * head_size_v

    def get_inputs(self) -> list:
        T = 5
        L = 4096
        qkv = empty_bf16(T, self.q_size + self.k_size + self.v_size)
        positions = empty_i64(T)
        q_weight = empty_bf16(1, self.head_size)
        k_weight = empty_bf16(1, self.head_size)
        cos_sin_cache = empty_bf16(L, self.head_size)
        inputs = [qkv, positions, q_weight, k_weight, cos_sin_cache]
        if self.quant_query:
            q_scale = empty_fp32(1)
            inputs += [q_scale]
        if _USE_LAYERNAME:
            inputs.append(_encode_layer_name(self.layer_name))
        return inputs

    def pattern_non_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_name: LayerNameType,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q_by_head = q.view(-1, self.q_size // self.head_size, self.head_size)
        q_normed = vllm.ir.ops.rms_norm(q_by_head, q_weight, self.eps)
        q_flat = q_normed.view(-1, self.q_size)

        k_by_head = k.view(-1, self.k_size // self.head_size, self.head_size)
        k_normed = vllm.ir.ops.rms_norm(k_by_head, k_weight, self.eps)
        k_flat = k_normed.view(-1, self.k_size)

        q_rope, k_rope = self.rope_matcher(positions, q_flat, k_flat, cos_sin_cache)

        q_rope = q_rope.view(-1, self.num_heads, self.head_size)
        k_rope = k_rope.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size_v)
        if self.v_norm:
            # The model flattens the normalised V and attention views it back
            # by head: a reshape pair NoOpEliminationPass removes first.
            v = vllm.ir.ops.rms_norm(v, None, self.eps)
        dummy = torch.ops.vllm.unified_kv_cache_update(k_rope, v, layer_name)
        return dummy, q_rope, k_rope, v

    def replacement_non_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_name: LayerNameType,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_out = torch.empty(
            qkv.shape[0],
            self.num_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        k_out = torch.empty(
            qkv.shape[0],
            self.num_kv_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        _, _, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        v = v.view(qkv.shape[0], self.num_kv_heads, self.head_size_v)
        if not self.v_norm:
            results = auto_functionalized(
                self.FUSED_OP,
                q_out=q_out,
                k_out=k_out,
                qkv=qkv,
                positions=positions,
                q_weight=q_weight,
                k_weight=k_weight,
                rms_norm_eps=self.eps,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                layer_name=layer_name,
                v_out=None,
            )
            return results[0], results[1], results[2], v
        # The normalised V, as written to the cache, is what attention takes.
        v_out = torch.empty_like(v, memory_format=torch.contiguous_format)
        results = auto_functionalized(
            self.FUSED_OP,
            q_out=q_out,
            k_out=k_out,
            qkv=qkv,
            positions=positions,
            q_weight=q_weight,
            k_weight=k_weight,
            rms_norm_eps=self.eps,
            cos_sin_cache=cos_sin_cache,
            is_neox=self.is_neox,
            layer_name=layer_name,
            v_norm=True,
            v_out=v_out,
        )
        return results[0], results[1], results[2], results[3]

    def pattern_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_scale: torch.Tensor,
        layer_name: LayerNameType,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q_by_head = q.view(-1, self.q_size // self.head_size, self.head_size)
        q_normed = vllm.ir.ops.rms_norm(q_by_head, q_weight, self.eps)
        q_flat = q_normed.view(-1, self.q_size)

        k_by_head = k.view(-1, self.k_size // self.head_size, self.head_size)
        k_normed = vllm.ir.ops.rms_norm(k_by_head, k_weight, self.eps)
        k_flat = k_normed.view(-1, self.k_size)

        q_rope, k_rope = self.rope_matcher(positions, q_flat, k_flat, cos_sin_cache)
        # Match the quant-query op Attention.forward inserts (fp8 KV + UNIFIED).
        # Explicit auto_functionalized (out=[1]) keeps the quant node in the pattern.
        q_out = torch.empty_like(q_rope, dtype=current_platform.fp8_dtype())
        q_quant = auto_functionalized(
            torch.ops.vllm.rocm_aiter_per_tensor_quant.default,
            out=q_out,
            x=q_rope,
            scale=q_scale,
            is_dynamic=False,
        )
        # `scale` is mutable: its copy_ write-back to _q_scale bumps the mutation
        # region, so keep q flat (a reshape lands past the barrier and won't match).
        q_rope_fp8 = q_quant[1]
        q_scale_out = q_quant[2]

        k_rope = k_rope.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size_v)
        dummy = torch.ops.vllm.unified_kv_cache_update(k_rope, v, layer_name)
        return dummy, q_rope_fp8, k_rope, v, q_scale_out

    def replacement_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_scale: torch.Tensor,
        layer_name: LayerNameType,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_out = torch.empty(
            qkv.shape[0],
            self.num_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        k_out = torch.empty(
            qkv.shape[0],
            self.num_kv_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        _, _, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        v = v.view(qkv.shape[0], self.num_kv_heads, self.head_size_v)
        results = auto_functionalized(
            self.FUSED_OP,
            q_out=q_out,
            k_out=k_out,
            qkv=qkv,
            positions=positions,
            q_weight=q_weight,
            k_weight=k_weight,
            rms_norm_eps=self.eps,
            cos_sin_cache=cos_sin_cache,
            is_neox=self.is_neox,
            layer_name=layer_name,
            v_out=None,
        )
        # Re-apply the quant on the kernel's bf16 q_out; fused op does not quant q.
        # Same explicit auto_functionalized form as the pattern: [1] = quantized
        # q, [2] = scale (returned so the buffer-writeback use is preserved).
        q_fp8_flat = results[1].view(-1, self.q_size)
        q_fp8_out = torch.empty_like(q_fp8_flat, dtype=current_platform.fp8_dtype())
        q_requant = auto_functionalized(
            torch.ops.vllm.rocm_aiter_per_tensor_quant.default,
            out=q_fp8_out,
            x=q_fp8_flat,
            scale=q_scale,
            is_dynamic=False,
        )
        q_fp8 = q_requant[1]  # flat to mirror the pattern (see note above)
        q_scale_out = q_requant[2]
        return results[0], q_fp8, results[2], v, q_scale_out

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

    def _register(self, pattern, replacement, pm_pass) -> None:
        trace_fn = QkNormRopeKvCachePattern.wrap_trace_fn(
            pm.fwd_only,
            QkNormRopeKvCachePattern.fx_view_to_reshape,
        )

        # Pre-build the search pattern with `ignore_types=(int, torch.SymInt)`
        # and pass it via `search_fn_pattern=` so torch skips both of its
        # internal `fx_to_pattern` calls and treats dynamic-shape SymInts as
        # wildcards.
        inputs = self.get_inputs()
        argnames = [*inspect.signature(pattern).parameters.keys()]
        search_gm = trace_fn(pattern, inputs)
        search_fn_pattern = pm.fx_to_pattern(
            search_gm,
            ignore_types=(int, torch.SymInt),
            argnames=argnames,
        )

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            trace_fn,
            pm_pass,
            extra_check=self._match_shape,
            search_fn_pattern=search_fn_pattern,
        )

    def _match_shape(self, match: pm.Match) -> bool:
        """Accept a match whose qkv split is a shape this pattern may replace,
        and size the replacement for it.  Sizes are wildcards in the pattern:
        without this a 256-dim pattern would replace a 512-dim layer (Gemma 4)
        and split its qkv wrongly."""
        split = next(
            (
                n
                for n in match.nodes
                if n.target == torch.ops.aten.split_with_sizes.default
            ),
            None,
        )
        if split is None:
            return False
        sizes = tuple(split.args[1])
        if self.shapes is None:
            return sizes == (self.q_size, self.k_size, self.v_size)
        if sizes not in self.shapes:
            return False
        self._set_shape(*self.shapes[sizes])
        return True

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # make_fx counts `self` in bound-method code params; wrap as plain fns.
        # Distinct names per branch so mypy doesn't see one name, two
        # signatures.  With LayerName the layer is a pattern input (one pattern
        # serves every layer of a shape); otherwise a closure constant.
        ln = _encode_layer_name(self.layer_name)
        pat_q, rep_q = self.pattern_fp8_quant_query, self.replacement_fp8_quant_query
        pat, rep = (
            self.pattern_non_fp8_quant_query,
            self.replacement_non_fp8_quant_query,
        )
        if self.quant_query and _USE_LAYERNAME:

            def pattern_q_ln(qkv, positions, qw, kw, cos_sin_cache, q_scale, ln_in):
                return pat_q(qkv, positions, qw, kw, cos_sin_cache, q_scale, ln_in)

            def replacement_q_ln(qkv, positions, qw, kw, cos_sin_cache, q_scale, ln_in):
                return rep_q(qkv, positions, qw, kw, cos_sin_cache, q_scale, ln_in)

            self._register(pattern_q_ln, replacement_q_ln, pm_pass)
        elif self.quant_query:

            def pattern_q(qkv, positions, qw, kw, cos_sin_cache, q_scale):
                return pat_q(qkv, positions, qw, kw, cos_sin_cache, q_scale, ln)

            def replacement_q(qkv, positions, qw, kw, cos_sin_cache, q_scale):
                return rep_q(qkv, positions, qw, kw, cos_sin_cache, q_scale, ln)

            self._register(pattern_q, replacement_q, pm_pass)
        elif _USE_LAYERNAME:

            def pattern_noq_ln(qkv, positions, qw, kw, cos_sin_cache, ln_in):
                return pat(qkv, positions, qw, kw, cos_sin_cache, ln_in)

            def replacement_noq_ln(qkv, positions, qw, kw, cos_sin_cache, ln_in):
                return rep(qkv, positions, qw, kw, cos_sin_cache, ln_in)

            self._register(pattern_noq_ln, replacement_noq_ln, pm_pass)
        else:

            def pattern_noq(qkv, positions, qw, kw, cos_sin_cache):
                return pat(qkv, positions, qw, kw, cos_sin_cache, ln)

            def replacement_noq(qkv, positions, qw, kw, cos_sin_cache):
                return rep(qkv, positions, qw, kw, cos_sin_cache, ln)

            self._register(pattern_noq, replacement_noq, pm_pass)


# ---------------------------------------------------------------------------
# Pass class
# ---------------------------------------------------------------------------


class QkNormRopeKvCacheFusionPass(VllmPatternMatcherPass):
    """
    Fuse QK-norm + RoPE + KV cache update into a single kernel, the attention
    impl's (AITER's on the AITER backends, rdna35_rope_cache on ROCM_ATTN on
    gfx1151).

    Supersedes both QKNormRoPEFusionPass and RopeKVCacheFusionPass for
    attention layers that support the combined operation, eliminating two
    separate kernel launches and the intermediate memory traffic.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="qk_norm_rope_kvcache_fusion_pass"
        )

        cc = config.compilation_config
        self.max_token_num = cc.pass_config.rope_kvcache_fusion_max_token_num

        dtype = config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once(
                "QK Norm+RoPE+KVCache fusion not enabled: unsupported dtype %s", dtype
            )
            return

        attn_layers = get_layers_from_vllm_config(config, Attention)

        # The fp8 query variant matches AITER's quant op, registered only when
        # AITER is enabled.
        quant_options = [False]
        if hasattr(torch.ops.vllm, "rocm_aiter_per_tensor_quant"):
            quant_options.append(True)
        supported: list[Attention] = []
        for _, layer in attn_layers.items():
            if not layer.impl.fused_qk_norm_rope_kvcache_supported():
                continue
            head_sizes = getattr(
                layer.impl,
                "fused_qk_norm_rope_kvcache_head_sizes",
                SUPPORTED_FUSED_QK_NORM_ROPE_KVCACHE_HEAD_DIMS,
            )
            if layer.head_size not in head_sizes:
                logger.warning_once(
                    "QK Norm+RoPE+KVCache fusion not enabled for a layer: "
                    "head_size=%d is not supported by the fused kernel "
                    "(supported: %s). Falling back to the unfused path.",
                    layer.head_size,
                    head_sizes,
                )
                continue
            if layer.head_size_v != layer.head_size:
                # The fused kernel uses a single head_dim for q/k/v.
                logger.warning_once(
                    "QK Norm+RoPE+KVCache fusion not enabled for a layer: "
                    "head_size_v=%d differs from head_size=%d, which the fused "
                    "kernel does not support. Falling back to the unfused path.",
                    layer.head_size_v,
                    layer.head_size,
                )
                continue
            supported.append(layer)

        # With LayerName the layer name is a wildcard and the pattern ignores
        # sizes: one pattern serves every supported layer, sized per match.
        # Otherwise each layer's name is part of its own pattern.
        shapes: dict[tuple[int, ...], tuple[int, ...]] | None = None
        if _USE_LAYERNAME and supported:
            shapes = {}
            for layer in supported:
                hq, hkv = layer.num_heads, layer.num_kv_heads
                d, dv = layer.head_size, layer.head_size_v
                shapes[(hq * d, hkv * d, hkv * dv)] = (hq, hkv, d, dv)
            supported = supported[:1]
        for layer in supported:
            v_norms = [False]
            if getattr(layer.impl, "fused_qk_norm_rope_kvcache_v_norm", False):
                v_norms.append(True)
            for epsilon in [1e-5, 1e-6]:
                for neox in [True, False]:
                    for quant_q in quant_options:
                        for v_norm in v_norms:
                            if quant_q and v_norm:
                                continue
                            QkNormRopeKvCachePattern(
                                layer=layer,
                                eps=epsilon,
                                is_neox=neox,
                                quant_query=quant_q,
                                v_norm=v_norm,
                                shapes=shapes,
                            ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.info(
            "QK-Norm+RoPE+KVCache fusion: replaced %s pattern(s)",
            self.matched_count,
        )

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.end <= self.max_token_num

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, QkNormRopeKvCachePattern)
