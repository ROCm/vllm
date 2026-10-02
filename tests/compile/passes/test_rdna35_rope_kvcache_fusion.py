# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The RoPE (+ q/k RMSNorm) + KV cache fusion passes on ROCM_ATTN (gfx1151).

Its hooks run rdna35_rope_cache.  The cases are what an end-to-end run needs
and a single-layer AITER test does not reach: the layer name as a pattern
input (LayerName), two layers of different head sizes in one graph (Gemma 4),
whose patterns differ only in sizes the qk-norm pattern ignores, Gemma 4's
weightless V norm, and the RoPE pass writing a contiguous query and key for
the decode kernel instead of rotating the qkv view in place.
"""

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.compilation.passes.fusion.qk_norm_rope_kvcache_fusion import (
    QkNormRopeKvCacheFusionPass,
)
from vllm.compilation.passes.fusion.rope_kvcache_fusion import RopeKVCacheFusionPass
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.compilation.passes.utility.scatter_split_replace import (
    ScatterSplitReplacementPass,
)
from vllm.compilation.passes.utility.split_coalescing import SplitCoalescingPass
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")

EPS = 1e-6
BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def _skip_unless_built():
    from vllm.platforms.rocm import on_gfx1151
    from vllm.v1.attention.ops.rdna35_rope_cache import rdna35_rope_cache_available

    if not on_gfx1151() or not rdna35_rope_cache_available():
        pytest.skip("rdna35_rope_cache is built for gfx1151")


class _AttnLayer(torch.nn.Module):
    """qkv -> [q/k RMSNorm] -> RoPE -> [V norm] -> KV cache update, as the
    models write it, for one attention layer."""

    def __init__(self, vllm_config, idx, hq, hkv, d, qk_norm, v_norm):
        super().__init__()
        self.hq, self.hkv, self.d = hq, hkv, d
        self.qk_norm, self.v_norm = qk_norm, v_norm
        self.layer_name = f"model.layers.{idx}.self_attn.attn"
        self.rotary_emb = RotaryEmbedding(d, d, 4096, 1e6, True, torch.bfloat16)
        if qk_norm:
            self.q_norm = RMSNorm(d, EPS, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(d, EPS, dtype=torch.bfloat16)
            for n in (self.q_norm, self.k_norm):
                n.weight.data.uniform_(0.5, 1.5)
        if v_norm:
            self.v_norm_mod = RMSNorm(d, EPS, has_weight=False)
        self.attn = Attention(
            num_heads=hq,
            head_size=d,
            scale=d**-0.5,
            num_kv_heads=hkv,
            cache_config=vllm_config.cache_config,
            prefix=self.layer_name,
            attn_backend=AttentionBackendEnum.ROCM_ATTN.get_class(),
        )
        backend = self.attn.get_attn_backend()
        self.builder = backend.get_builder_cls()(
            kv_cache_spec=self.attn.get_kv_cache_spec(vllm_config),
            layer_names=[self.layer_name],
            vllm_config=vllm_config,
            device=torch.device("cuda"),
        )
        self.backend = backend

    def make_cache(self, t):
        common = create_common_attn_metadata(
            BatchSpec(seq_lens=[1] * t, query_lens=[1] * t),
            BLOCK_SIZE,
            torch.device("cuda"),
            arange_block_indices=True,
        )
        shape = self.backend.get_kv_cache_shape(t, BLOCK_SIZE, self.hkv, self.d)
        order = self.backend.get_kv_cache_stride_order()
        raw = torch.zeros(tuple(shape[i] for i in order), dtype=torch.bfloat16)
        self.attn.kv_cache = raw.permute(*[order.index(i) for i in range(len(order))])
        return self.builder.build(common_prefix_len=0, common_attn_metadata=common)

    def forward(self, qkv, positions):
        qkv = qkv.clone()
        q_size, kv_size = self.hq * self.d, self.hkv * self.d
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        if self.qk_norm:
            q = self.q_norm(q.view(-1, self.hq, self.d)).view(-1, q_size)
            k = self.k_norm(k.view(-1, self.hkv, self.d)).view(-1, kv_size)
        q, k = self.rotary_emb(positions, q, k)
        if self.v_norm:
            # Gemma 4 writes unflatten/flatten; the same reshapes, which a
            # plain torch.compile (unlike vLLM's) breaks the graph on.
            v = self.v_norm_mod(v.view(-1, self.hkv, self.d)).view(-1, kv_size)
        q = q.view(-1, self.hq, self.d)
        k = k.view(-1, self.hkv, self.d)
        v = v.view(-1, self.hkv, self.d)
        dep = torch.ops.vllm.unified_kv_cache_update(
            k, v, _encode_layer_name(self.layer_name)
        )
        # q, k and v are returned as attention would consume them: the
        # patterns expect k and v to have a user besides the cache update.
        return q, k, v, dep


class _Model(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, positions, *qkvs):
        out = []
        for layer, qkv in zip(self.layers, qkvs):
            out += layer(qkv, positions)
        return out


def _run(vllm_config, layers, fusion_pass):
    t = 5
    model = _Model(layers)
    qkvs = [
        torch.randn(t, (lay.hq + 2 * lay.hkv) * lay.d, dtype=torch.bfloat16)
        for lay in layers
    ]
    pos = torch.arange(100, 100 + t, dtype=torch.long)

    def call(fn):
        with set_forward_context(None, vllm_config):
            ctx = get_forward_context()
            ctx.slot_mapping = {
                lay.layer_name: lay.make_cache(t).slot_mapping for lay in layers
            }
            outs = fn(pos, *qkvs)
            caches = [lay.attn.kv_cache.clone() for lay in layers]
        # Every fourth output is the cache update's empty dependency tensor.
        return [o.clone() for i, o in enumerate(outs) if i % 4 != 3], caches

    ref, ref_caches = call(model)
    backend = TestBackend(
        NoOpEliminationPass(vllm_config),
        SplitCoalescingPass(vllm_config),
        ScatterSplitReplacementPass(vllm_config),
        fusion_pass,
        PostCleanupPass(vllm_config),
    )
    for x in [pos, *qkvs]:
        torch._dynamo.mark_dynamic(x, 0)
    got, got_caches = call(torch.compile(model, backend=backend))
    assert fusion_pass.matched_count == len(layers)
    for a, b in zip(ref + ref_caches, got + got_caches):
        torch.testing.assert_close(a, b, atol=2e-2, rtol=2e-2)
    return backend


def _config(**passes):
    return VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        cache_config=CacheConfig(block_size=BLOCK_SIZE),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rotary_embedding"],
            pass_config=PassConfig(eliminate_noops=True, **passes),
        ),
    )


def test_rope_kvcache_writes_contiguous_q_and_k():
    """Llama-style layers: the RoPE pass, with q and k to buffers of their own."""
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    vllm_config = _config(fuse_rope_kvcache=True)
    with vllm.config.set_current_vllm_config(vllm_config):
        layers = [_AttnLayer(vllm_config, 0, 32, 8, 128, False, False)]
        backend = _run(vllm_config, layers, RopeKVCacheFusionPass(vllm_config))
    backend.check_after_ops(
        [torch.ops.vllm.fused_rope_out_and_unified_kv_cache_update.default]
    )


@pytest.mark.parametrize("v_norm", [False, True])
def test_qk_norm_rope_kvcache_two_head_sizes(v_norm):
    """Qwen3-style (v_norm=False) and Gemma 4-style layers, a 256- and a
    512-dim layer in one graph: each fused with its own sizes."""
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    vllm_config = _config(fuse_qk_norm_rope_kvcache=True)
    with vllm.config.set_current_vllm_config(vllm_config):
        layers = [
            _AttnLayer(vllm_config, 0, 8, 1, 256, True, v_norm),
            _AttnLayer(vllm_config, 1, 8, 1, 512, True, v_norm),
        ]
        backend = _run(vllm_config, layers, QkNormRopeKvCacheFusionPass(vllm_config))
    backend.check_after_ops(
        [torch.ops.vllm.fused_qk_norm_rope_and_unified_kv_cache_update.default]
    )


@pytest.mark.parametrize(
    "dtype,backend,passes,custom_ops,fused",
    [
        (torch.bfloat16, None, {}, [], (True, True)),
        (torch.bfloat16, "ROCM_ATTN", {}, [], (True, True)),
        (torch.bfloat16, None, {"fuse_rope_kvcache": False}, [], (False, True)),
        (
            torch.bfloat16,
            None,
            {"fuse_rope_kvcache": False, "fuse_qk_norm_rope_kvcache": False},
            [],
            (False, False),
        ),
        (torch.bfloat16, None, {}, ["-rotary_embedding"], (False, False)),
        (torch.bfloat16, "TRITON_ATTN", {}, [], (False, False)),
        # The kernel takes fp16 and bf16 only.
        (torch.float32, None, {}, [], (False, False)),
    ],
)
def test_on_by_default_only_for_rocm_attn(dtype, backend, passes, custom_ops, fused):
    """The fusion and what it needs (the rotary custom op, the cache update
    inside the graph) are defaults for ROCM_ATTN on gfx1151: what the user set is
    kept, and a model another backend serves keeps its graph."""
    cc = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        attention_config=AttentionConfig(backend=backend),
        compilation_config=CompilationConfig(
            custom_ops=list(custom_ops), pass_config=PassConfig(**passes)
        ),
    ).compilation_config
    pc = cc.pass_config
    assert (pc.fuse_rope_kvcache, pc.fuse_qk_norm_rope_kvcache) == fused
    assert ("+rotary_embedding" in cc.custom_ops) == any(fused)
    assert ("vllm::unified_kv_cache_update" in cc.splitting_ops) != any(fused)
