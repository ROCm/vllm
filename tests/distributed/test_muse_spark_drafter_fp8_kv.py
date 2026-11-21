# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's DSpark draft kernels (`--monokernel`, TP8) on an FP8 (e4m3) KV
cache with non-unit `k_scale` / `v_scale`, against the stock draft layers on
the same cache.

The first draft layer's K / V insert must be vLLM's fp8 insert (`bf16(x) /
scale`, e4m3) up to one e4m3 code. The second layer's K / V, which depend on
the whole first layer (its attention reads the scaled cache), must be within
the FP8 rounding of the stock values. Deeper layers are not compared: with
random weights each layer amplifies rounding differences about 2x, so the
stock layers themselves drift that far from a 1-ulp change of their input."""

import json
import os

import pytest
import torch

from vllm.platforms import current_platform

from ..utils import multi_gpu_test
from .eplb_utils import distributed_run, set_env_vars_and_device

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Muse Spark ROCm draft kernels"
)

WORLD = 8
K_SCALE, V_SCALE = 0.5, 2.0
# The second layer's K / V (dequantized): e4m3 rounding of inputs that differ
# in bf16 rounding (2.2% measured; the stock layers move 3% for a 1-ulp input).
SECOND_LAYER_RTOL = 5e-2


def _kv_scales(kv: torch.Tensor) -> torch.Tensor:
    """`[..., K | V]` (64 + 64): k_scale over the K half, v_scale over the V half."""
    scale = torch.full(kv.shape[-1:], K_SCALE, device=kv.device)
    scale[kv.shape[-1] // 2 :] = V_SCALE
    return scale


def _e4m3_codes(x: torch.Tensor) -> torch.Tensor:
    """e4m3 bytes as integers ordered like their values."""
    bits = x.view(torch.int8).long()
    return torch.where(bits < 0, -128 - bits, bits)


def drafter_fp8_kv_matches_the_stock_layers(
    env: dict[str, str], world_size: int, root: str
) -> None:
    set_env_vars_and_device(env)
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    os.environ["VLLM_KV_CACHE_LAYOUT"] = "LBHNC"
    from tests.models.test_muse_spark_rocm_drafter import (
        DSPARK_CHECKPOINT,
        QUERY_ROWS,
        DraftStep,
        TinyDraft,
        load_draft,
    )
    from tests.models.test_muse_spark_rocm_mono import (
        TINY_TEXT,
        _checkpoint_config,
        _vllm_config,
    )
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import set_current_vllm_config
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized
    from vllm.models.muse_spark.amd.mono.drafter import DSparkDrafter, draft_dims
    from vllm.models.muse_spark.amd.mono.kernels import attn_frag, draft
    from vllm.v1.kv_cache_interface import AttentionSpec
    from vllm.v1.worker.workspace import init_workspace_manager

    rocm_aiter_ops.refresh_env_variables()
    rank = int(env["RANK"])
    device = torch.device("cuda", torch.accelerator.current_device_index())
    configs = os.path.join(root, f"rank{rank}")
    for name, config in (
        ("tiny", _checkpoint_config(TINY_TEXT)),
        ("dspark", DSPARK_CHECKPOINT),
    ):
        os.makedirs(os.path.join(configs, name), exist_ok=True)
        with open(os.path.join(configs, name, "config.json"), "w") as f:
            json.dump(config, f)
    vllm_config = _vllm_config(
        os.path.join(configs, "tiny"),
        os.path.join(configs, "dspark"),
        tp=world_size,
        cache_dtype="fp8",
    )
    with set_current_vllm_config(vllm_config), torch.inference_mode():
        ensure_model_parallel_initialized(world_size, 1)
        init_workspace_manager(device)
        # replicated weights must be the same on every rank (one seed)
        lm = load_draft(vllm_config, device)
        for layer in lm.model.layers:
            attn = layer.self_attn.attn
            attn._k_scale.fill_(K_SCALE)
            attn._v_scale.fill_(V_SCALE)
            attn._k_scale_float, attn._v_scale_float = K_SCALE, V_SCALE
        heads = DSPARK_CHECKPOINT["num_key_value_heads"] // world_size
        spec = AttentionSpec(
            block_size=16, num_kv_heads=heads, head_size=64, dtype=torch.uint8
        )
        tiny = TinyDraft(lm, vllm_config, spec, kv_fp8=True)
        drafter = DSparkDrafter(
            lm.model,
            {"draft": draft, "attn_frag": attn_frag},
            rank=get_tp_group().rank_in_group,
            group=get_tp_group().cpu_group,
            device=device,
            dims=draft_dims(lm.config, world_size),
            kv_fp8=True,
        )
        try:
            for num_reqs in (1, 2):
                step = DraftStep(tiny, [QUERY_ROWS] * num_reqs, seed=num_reqs)
                lm.mono = None
                hidden_s, kv_s = step.run(tiny)
                lm.mono = drafter
                hidden_m, kv_m = step.run(tiny)
                torch.accelerator.synchronize()
                rows = QUERY_ROWS * num_reqs
                assert rows in drafter._engaged
                assert torch.isfinite(hidden_m).all()
                for i, name in enumerate(kv_s):
                    # `[block][kv head][16 tokens][K | V]`: the query rows' slots
                    slots = step.slots[name][:rows]
                    blk, tok = slots // 16, slots % 16
                    old = step.kv_init[name]
                    stock = kv_s[name][blk, :, tok]
                    mono = kv_m[name][blk, :, tok]
                    assert not torch.equal(stock, old[blk, :, tok])
                    if i == 0:
                        diff = _e4m3_codes(mono) - _e4m3_codes(stock)
                        assert diff.abs().max() <= 1, name
                    elif i == 1:
                        s = stock.view(torch.float8_e4m3fn).float() * _kv_scales(stock)
                        m = mono.view(torch.float8_e4m3fn).float() * _kv_scales(mono)
                        err = (m - s).norm() / s.norm()
                        assert err <= SECOND_LAYER_RTOL, (name, float(err))
                    # and nothing else
                    kv_m[name][blk, :, tok] = old[blk, :, tok]
                    assert torch.equal(kv_m[name], old), name
        finally:
            lm.mono = None
            drafter.close()
        torch.distributed.barrier()


@multi_gpu_test(num_gpus=WORLD)
def test_muse_spark_drafter_fp8_kv_matches_the_stock_layers(tmp_path, monkeypatch):
    """1 and 2 requests (the S = 8 / 16 builds) on the checkpoint's draft shape."""
    pytest.importorskip("flydsl")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("the draft kernels run on gfx950")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "cache"))
    distributed_run(drafter_fp8_kv_matches_the_stock_layers, WORLD, str(tmp_path))
