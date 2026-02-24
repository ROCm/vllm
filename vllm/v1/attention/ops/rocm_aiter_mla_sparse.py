# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib

import torch

from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops

@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,  # [num_tokens, head_dim]
    kv_cache_ptr,  # [n_blks, blk_size//tile_block, head_dim // 16B, tile_block, 16B]
    # [n_blocks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_tokens,
    head_dim: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, head_dim)
    if LAYOUT == "SHUFFLE":
        tile_offset = (
            offset // HEAD_TILE_SIZE * BLOCK_TILE_SIZE * HEAD_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tile_offset = offset
    tile_store_offset = tile_offset
    # for idx in tl.range(tid, num_tokens, n_program):
    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    tile_block_id = block_offset // BLOCK_TILE_SIZE
    tile_block_offset = block_offset % BLOCK_TILE_SIZE
    val = tl.load(src_ptr + offset)
    amax = tl.max(val.abs(), axis=-1).to(tl.float32)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = (val.to(tl.float32) / scale).to(kv_cache_ptr.type.element_ty)
    if LAYOUT == "SHUFFLE":
        dst_ptr = (
            kv_cache_ptr
            + block_id * kv_cache_value_stride
            + tile_block_id * BLOCK_TILE_SIZE * head_dim
            + tile_block_offset * HEAD_TILE_SIZE
        )
    else:
        dst_ptr = (
            kv_cache_ptr + block_id * kv_cache_value_stride + block_offset * head_dim
        )
    tl.store(dst_ptr + tile_store_offset, fp8_val)
    dst_scale_ptr = kv_cache_scale_ptr + block_id * kv_cache_scale_stride + block_offset
    tl.store(dst_scale_ptr, scale)

def indexer_k_quant_and_cache_triton(
    k: torch.Tensor,
    kv_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
    block_tile_size=16,
    head_tile_size=16,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    # In real layout, we store the first portion as kv cache value
    # and second portion as kv cache scale
    kv_cache = kv_cache.view(num_blocks, -1)
    kv_cache_value = kv_cache[:, : block_size * head_dim]
    kv_cache_scale = kv_cache[:, block_size * head_dim :].view(torch.float32)

    head_tile_size = head_tile_size // kv_cache.element_size()
    grid = (num_tokens,)
    _indexer_k_quant_and_cache_kernel[grid](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_mapping,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_tokens,
        head_dim,
        "NHD",
        block_tile_size,
        head_tile_size,
        IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz,
        USE_UE8M0=scale_fmt == "ue8m0",
    )

@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,  # [n_blks,blk_size//tile_blk,head_dim//16B,tile_blk,16B]
    # [n_blks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    k_fp8_ptr,  # [num_tokens, head_dim]
    k_scale_ptr,  # [num_tokens]
    block_table_ptr,  # [batch_size, block_table_stride]
    cu_seqlen_ptr,  # [batch_size + 1]
    token_to_seq_ptr,  # [num_tokens]
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    LAYOUT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, HEAD_DIM)
    batch_id = tl.load(token_to_seq_ptr + tid)
    batch_start = tl.load(cu_seqlen_ptr + batch_id)
    batch_end = tl.load(cu_seqlen_ptr + batch_id + 1)
    batch_offset = tid - batch_start
    if tid >= batch_end:
        return
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    block_table_offset = batch_id * block_table_stride + block_table_id
    block_id = tl.load(block_table_ptr + block_table_offset)
    tiled_block_id = block_offset // BLOCK_TILE_SIZE
    tiled_block_offset = block_offset % BLOCK_TILE_SIZE
    if LAYOUT == "SHUFFLE":
        src_cache_offset = (
            block_id * kv_cache_stride
            + tiled_block_id * HEAD_DIM * BLOCK_TILE_SIZE
            + tiled_block_offset * HEAD_TILE_SIZE
        )
    else:
        src_cache_offset = block_id * kv_cache_stride + block_offset * HEAD_DIM
    src_scale_offset = block_id * kv_cache_scale_stride + block_offset
    dst_offset = tid * HEAD_DIM
    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset
    scale_val = tl.load(src_scale_ptr)
    tl.store(k_scale_ptr + tid, scale_val)
    if LAYOUT == "SHUFFLE":
        tiled_src_offset = (
            offset // HEAD_TILE_SIZE * HEAD_TILE_SIZE * BLOCK_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tiled_src_offset = offset
    val = tl.load(src_cache_ptr + tiled_src_offset)
    tl.store(dst_k_ptr + offset, val)

def cp_gather_indexer_k_quant_cache_triton(
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
    token_to_seq: torch.Tensor,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    # we assume the kv cache already been split to 2 portion
    k_cache = k_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_value = k_cache[:, : block_size * head_dim].view(fp8_dtype)
    k_cache_scale = k_cache[:, block_size * head_dim :].view(torch.float32)

    grid = (num_tokens,)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    _cp_gather_indexer_quant_cache_kernel[grid](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        token_to_seq,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        "NHD",
        head_dim,
        block_tile_size,
        head_tile_size,
    )

# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    from vllm.utils.math_utils import cdiv

    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    block_size = kv_cache.shape[1]
    N = kv_cache.shape[0]
    # = [N, 132*16] 
    kv_cache = kv_cache.reshape([N, (dim+4)*block_size])
    kv_cache, scale = kv_cache[:, :dim*block_size], kv_cache[:, dim*block_size:]
    kv_cache = kv_cache.reshape([N, block_size, 1, dim])
    scale = scale.reshape([N, block_size, 1, 4])
    scale = scale.contiguous().view(torch.float)
    q = q.float()
    indexed_scale = scale[block_tables[0]]

    #  print("Scale", block_tables[0,:10], indexed_scale.flatten()[:10])

    kv_cache = kv_cache.view(fp8_dtype).float() * scale
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device="cuda"
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                0.0, #float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)

            #    print("Overflow", i, block_rk, context_len, s, torch.max(qx.to(torch.float32)), torch.max(kx.to(torch.float32)), torch.max(weight_slice.to(torch.float32)))
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits

def rocm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    use_torch: bool
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    @functools.lru_cache
    def paged_mqa_logits_module():
        paged_mqa_logits_module_path = None
        if importlib.util.find_spec("aiter.ops.triton.pa_mqa_logits") is not None:
            paged_mqa_logits_module_path = "aiter.ops.triton.pa_mqa_logits"
        elif (
            importlib.util.find_spec("aiter.ops.triton.attention.pa_mqa_logits")
            is not None
        ):
            paged_mqa_logits_module_path = "aiter.ops.triton.attention.pa_mqa_logits"

        if paged_mqa_logits_module_path is not None:
            try:
                module = importlib.import_module(paged_mqa_logits_module_path)
                return module
            except ImportError:
                return None
        return None

    aiter_paged_mqa_logits_module = None
    if rocm_aiter_ops.is_enabled():
        aiter_paged_mqa_logits_module = paged_mqa_logits_module()
    # FIXME(ganyi): Temporarily disable the aiter path until nightly docker
    # update aiter to the fix PR.
    if use_torch:
       aiter_paged_mqa_logits_module = None

    if aiter_paged_mqa_logits_module is not None:
        #ref = fp8_paged_mqa_logits_torch(q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len)
        dim = q_fp8.shape[3]
        N = kv_cache_fp8.shape[0]
        block_size = kv_cache_fp8.shape[1]
        # = [N, 132*16] 
        if block_size > 1:
           kv_cache_fp8_1 = kv_cache_fp8.reshape([N, (dim+4)*block_size])
           kv_cache_fp8_1, scale = kv_cache_fp8_1[:, :dim*block_size], kv_cache_fp8[:, dim*block_size:]
           kv_cache_fp8_1 = kv_cache_fp8_1.reshape([N, block_size, 1, dim])
           scale = scale.reshape([N, block_size, 1, 4])
           kv_cache_fp8_1 = torch.concatenate([kv_cache_fp8_1, scale], axis=3)
        else:
           kv_cache_fp8_1 = kv_cache_fp8

        deepgemm_fp8_paged_mqa_logits_stage1 = (
            aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits_stage1
        )
        batch_size, next_n, heads, _ = q_fp8.shape
        out_qk = torch.full(
            (heads, batch_size * next_n, max_model_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )
        ChunkQ = 64
        while (heads % ChunkQ):
           ChunkQ = ChunkQ // 2
        deepgemm_fp8_paged_mqa_logits_stage1(
            q_fp8,
            kv_cache_fp8_1,
            weights,
            out_qk,
            context_lens,
            block_tables,
            max_model_len,
            ChunkQ
        )
        testval = out_qk.sum(dim=0)
        return testval
        finite_mask1 = torch.isfinite(ref)
        finite_mask2 = torch.isfinite(testval)
        mismatch1 = (finite_mask1 != finite_mask2)
        mismatch2 = torch.logical_and(torch.logical_and(finite_mask1, finite_mask2), torch.abs(ref-testval)>0.01*torch.maximum(torch.abs(ref), torch.abs(testval)))
        s1 = torch.sum(mismatch1.to(torch.uint32))
        s2 = torch.sum(mismatch2.to(torch.uint32))
        print(ref.shape, testval.shape, s1, s2, ref.flatten()[:10], testval.flatten()[:10])
        if s1>0:
            w = torch.where(mismatch1)
            pos = tuple(w[x][0] for x in range(len(ref.shape)))
            print("First infinite mismatch", pos, ref[pos], testval[pos])
        if s2>0:
            w = torch.where(mismatch2)
            pos = tuple(w[x][0] for x in range(len(ref.shape)))
            print("First value mismatch", pos, ref[pos], testval[pos])
        return ref
    else:
        return fp8_paged_mqa_logits_torch(
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len
        )

# Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    kv, scale = kv
    seq_len_kv = kv.shape[0]
    k = (kv.to(torch.float32) * scale)
    k = k.to(torch.bfloat16)
    q = q.to(torch.bfloat16)
    M = q.shape[0]

    min_seqlen = torch.min(cu_seqlen_ks)
    max_seqlen = torch.max(cu_seqlen_ke)

    dev = kv.device.index
    silent = True #(dev != 0)

    if not silent:
        infs = torch.sum(torch.logical_not(torch.isfinite(k[min_seqlen:max_seqlen])).to(torch.int32)).item()
        if infs>0:
            print("WARNING: fp8_mqa_logits_torch: ", infs, "/", k[min_seqlen:max_seqlen].flatten().shape[0], "nonfinite in k")
            pos_m, pos_n = torch.where(torch.logical_not(torch.isfinite(k[min_seqlen:max_seqlen])))
            print("First inf:", pos_m[0].item()+min_seqlen.item(), pos_n[0].item(), k[pos_m[0]+min_seqlen, pos_n[0]].item()) 
            if infs>1:
                print("Second inf:", pos_m[1].item()+min_seqlen.item(), pos_n[1].item(), k[pos_m[1]+min_seqlen, pos_n[1]].item()) 
        infs = torch.sum(torch.logical_not(torch.isfinite(q)).to(torch.int32)).item()
        if infs>0:
            print("WARNING: fp8_mqa_logits_torch: ", infs, "/", q.flatten().shape[0], "nonfinite in q")
        infs = torch.sum(torch.logical_not(torch.isfinite(weights)).to(torch.int32)).item()
        if infs>0:
            print("WARNING: fp8_mqa_logits_torch: ", infs, "/", weights.flatten().shape[0], "nonfinite in weights")
    do_padding = False
    pad_left = 0
    if do_padding:
        print("Padding:", min_seqlen, max_seqlen, kv.shape[0])
        max_seqlen = min(max_seqlen, kv.shape[0])
        pad_left = min_seqlen
        pad_right = kv.shape[0] - max_seqlen
        k = k[pad_left:]
        cu_seqlen_ks -= pad_left
        cu_seqlen_ke -= pad_right

        if pad_right>0:
            k = k[:-pad_right]

    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] + pad_left >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] + pad_left < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float()
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))
    if do_padding:
        if pad_left > 0:
            logits = torch.concatenate([torch.full([M,pad_left], float("-inf"), device=logits.device, dtype=torch.float32), logits], axis=1)
        if pad_right > 0:
            logits = torch.concatenate([logits, torch.full([M,pad_right], float("-inf"), device=logits.device, dtype=torch.float32)], axis=1)

    if not silent:
        legal_logits = torch.where(mask, logits, float(0.0))
        infs = torch.sum(torch.logical_not(torch.isfinite(legal_logits)).to(torch.int32)).item()
        if infs>0:
            print("WARNING: fp8_mqa_logits_torch:", infs, "/", torch.sum(mask.to(torch.int32)).item(), "nonfinite in output")
            pos_m, pos_n = torch.where(torch.logical_not(torch.isfinite(legal_logits)))
            x = pos_m[0]
            y = pos_n[0]
            score2 = torch.einsum("hd,d->h", q[x], k[y]).float()
            print(x, y, score2, (score2.relu() * weights[x]).sum(dim=0))

    return logits

def rocm_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """

    # TODO(ganyi): Temporarily workaround, will remove the module check and reference
    # path after aiter merge this kernel into main
    from vllm._aiter_ops import rocm_aiter_ops

    #kv = (kv[0].reshape([kv[0].shape[0]*kv[0].shape[1], kv[0].shape[2]]), kv[1].flatten())

    @functools.lru_cache
    def mqa_logits_module():
        mqa_logits_module_path = None
        if importlib.util.find_spec("aiter.ops.triton.fp8_mqa_logits") is not None:
            mqa_logits_module_path = "aiter.ops.triton.fp8_mqa_logits"
        elif (
            importlib.util.find_spec("aiter.ops.triton.attention.fp8_mqa_logits")
            is not None
        ):
            mqa_logits_module_path = "aiter.ops.triton.attention.fp8_mqa_logits"

        if mqa_logits_module_path is not None:
            try:
                module = importlib.import_module(mqa_logits_module_path)
                return module
            except ImportError:
                return None
        return None

    aiter_mqa_logits_module = None
    if rocm_aiter_ops.is_enabled():
        aiter_mqa_logits_module = mqa_logits_module()

    if aiter_mqa_logits_module is not None:
        fp8_mqa_logits = aiter_mqa_logits_module.fp8_mqa_logits
        kv, scale = kv
        return fp8_mqa_logits(q, kv, scale, weights, cu_seqlen_ks, cu_seqlen_ke)
    else:
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)

def rocm_aiter_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    # profile run
    # NOTE(Chen): create the max possible flattened_kv. So that
    # profile_run can get correct memory usage.
    _flattened_kv = torch.empty(
        [total_seq_lens, head_dim + 4], device=k.device, dtype=torch.uint8
    )
    fp8_dtype = current_platform.fp8_dtype()
    _k_fp8 = _flattened_kv[..., :head_dim].view(fp8_dtype).contiguous()
    _k_scale = _flattened_kv[..., head_dim:].view(torch.float32).contiguous()
    return topk_indices_buffer

def rocm_aiter_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None
) -> torch.Tensor:

    #W, B, X, H = kv_cache.shape
    #kv_cache = kv_cache.reshape([W, 
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    dev = q_fp8.device.index
    silent = True #(dev != 0)
    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        return rocm_aiter_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    #  print("slot_mapping", slot_mapping[:10])

    if not silent:
        print("rocm_aiter_sparse_attn_indexer", k.shape, kv_cache.shape, slot_mapping[:5], num_decode_tokens)

    # Store 'k' into long-term kv cache. slot_mapping maps from current tokens to locations in cache.
    ops.indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )
    #scales = kv_cache[..., -4:].view(float)

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    #topk_indices_buffer[:] = -1
    if has_prefill:
        prefill_metadata = attn_metadata.prefill
        if not silent:
            print("Prefilling", len(prefill_metadata.chunks), "chunks", sum([x.token_end-x.token_start for x in prefill_metadata.chunks]), "tokens")
        for chunk in prefill_metadata.chunks:

            uniform_rows = torch.all(chunk.cu_seqlen_ks == chunk.cu_seqlen_ks[0]) and torch.all(chunk.cu_seqlen_ke == chunk.cu_seqlen_ke[0])
            if not silent:
                print("prefill", chunk.total_seq_lens, chunk.token_start, chunk.token_end, chunk.cu_seqlen_ks[:3], chunk.cu_seqlen_ke[:3], uniform_rows)
            block_size = kv_cache.shape[1]
            k_fp8 = torch.empty(
                [chunk.total_seq_lens, head_dim],
                device=k.device,
                dtype=fp8_dtype,
            )
            k_scale = torch.empty(
                [chunk.total_seq_lens, 4],
                device=k.device,
                dtype=torch.uint8,
            )
            ops.cp_gather_indexer_k_quant_cache(
                kv_cache,
                k_fp8,
                k_scale,
                chunk.block_table,
                chunk.cu_seq_lens,
            )
            if not silent:
                print("kv_cache", kv_cache.shape, k_fp8.shape, chunk.block_table[0,:5])
                col = chunk.block_table[0,0]
                print("cp_gather_indexer_k_quant_cache", kv_cache[col,0], kv_cache[col,0,-4:].view(torch.float32), k_fp8[0].view(torch.uint8), k_scale[0].view(torch.float32))
                col = chunk.block_table[0,1]
                print("cp_gather_indexer_k_quant_cache", kv_cache[col,0], kv_cache[col,0,-4:].view(torch.float32), k_fp8[1].view(torch.uint8), k_scale[1].view(torch.float32))

            logits = rocm_fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            logits = torch.where(torch.isnan(logits), float("-inf"), logits)

            num_rows = logits.shape[0]
            assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            if True:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
                topk_indices = torch.where(topk_indices>=0, topk_indices+chunk.cu_seqlen_ks[:,None], -1)
            else:
                #torch_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
                #uniform_rows = torch.all(chunk.cu_seqlen_ks == chunk.cu_seqlen_ks[0]) and torch.all(chunk.cu_seqlen_ke == chunk.cu_seqlen_ke[0])

                if uniform_rows:
                    row_start = int(chunk.cu_seqlen_ks[0])
                    row_end = int(chunk.cu_seqlen_ke[0])
                    k_i = min(topk_tokens, row_end-row_start)
                    idx = logits[:, row_start:row_end].topk(k_i, dim=-1)[1]
                    topk_indices[:, :k_i] = row_start + idx
                    topk_indices[:, k_i:topk_tokens] = -1
                else:
                    for i in range(num_rows):
                        row_start = int(chunk.cu_seqlen_ks[i])
                        row_end = int(chunk.cu_seqlen_ke[i])
                        k_i = min(topk_tokens, row_end-row_start)
                        idx = logits[i, row_start:row_end].topk(k_i, dim=-1)[1]
                        topk_indices[i, :k_i] = row_start + idx
                        topk_indices[i, k_i:topk_tokens] = -1

                topk_indices_2 = torch.zeros_like(topk_indices, dtype=topk_indices.dtype)
                if not silent:
                    print("top_k_per_row_prefill", logits.shape, chunk.cu_seqlen_ks.shape, 
                                            chunk.cu_seqlen_ke.shape,
                    topk_indices_2.shape,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens)
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices_2,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
                topk_indices_2 = torch.where(topk_indices_2>=0, topk_indices_2+chunk.cu_seqlen_ks[:,None], -1)

                if not silent:
                    check = compare_top_k_results(logits, topk_indices, topk_indices_2, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, topk_tokens)
                """
                topk_indices_1, _ = torch.sort(topk_indices)
                topk_indices_2, _ = torch.sort(topk_indices_2)
                mismatches = (topk_indices_1 != topk_indices_2).to(torch.int32)
                print("top_k_per_row_prefill:", torch.sum(mismatches), "mismatches")
                print("Shapes:", logits.shape, 
                if torch.sum(mismatches)>0:
                    mismatches = mismatches.flatten()
                    pos = torch.where(mismatches)
                    print(topk_indices.shape, pos[0], topk_indices_1.flatten()[pos[0]], topk_indices_2.flatten()[pos[0]])
                """

    if has_decode:
        decode_metadata = attn_metadata.decode
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n

        logits = rocm_fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
            use_torch=False
        )

        num_rows = logits.shape[0]
        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
        topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.seq_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer
