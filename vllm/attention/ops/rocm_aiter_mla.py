# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op
import vllm.envs as envs
from aiter.mla import mla_decode_fwd_dispatch
from aiter import per_tensor_quant

def get_aiter_mla_metadata(max_batch_size: int, block_size: int,
                           max_block_per_batch: int,
                           device: torch.device) -> tuple[torch.Tensor, ...]:
    paged_kv_indices = torch.zeros(max_batch_size * max_block_per_batch,
                                   dtype=torch.int32,
                                   device=device)
    paged_kv_indptr = torch.zeros(max_batch_size + 1,
                                  dtype=torch.int32,
                                  device=device)
    paged_kv_last_page_lens = torch.full((max_batch_size, ),
                                         block_size,
                                         dtype=torch.int32)
    qo_indptr = torch.zeros(max_batch_size + 1, dtype=torch.int, device=device)
    return paged_kv_indices, paged_kv_indptr, paged_kv_last_page_lens, qo_indptr


def aiter_mla_decode_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    sm_scale=None,  # 1.0 / (qk_head_dim**0.5)
    varlen=False,
    logit_cap=0.0,
    num_kv_splits=None,
    num_kv_splits_indptr=None,
    work_metadata=None,
    work_indptr=None,
    work_info_set=None,
    reduce_indptr=None,
    reduce_final_map=None,
    reduce_partial_map=None,

    # batch_split_table=None,
    # split_table=None,
    # splits=None,
    # q_rope=None,
    # k_rope=None, 
):
    if envs.VLLM_MLA_FP8_PADDING:
        assert q.shape.__len__() == 3, f"q shape: {q.shape}"
        assert o.shape.__len__() == 3, f"o shape: {o.shape}"
        
        q_fp8, q_scale = per_tensor_quant(q, quant_dtype=torch.float8_e4m3fnuz)
        kv_buffer = kv_buffer.to(torch.float8_e4m3fnuz)
        kv_scale = torch.ones([1], dtype=torch.float, device=kv_buffer.device) 
        batch_size = q_fp8.shape[0]
        q_fp8_padded = torch.ones(batch_size * 2, q_fp8.shape[1], q_fp8.shape[2], dtype=q_fp8.dtype, device=q_fp8.device)
        q_fp8_padded[::2] = q_fp8
        qo_indptr_padded = torch.arange(0, (batch_size + 1) * 2, 2, dtype=qo_indptr.dtype, device=qo_indptr.device)
        o_padded = torch.empty((o.shape[0] * 2, o.shape[1], o.shape[2]), dtype=o.dtype, device=o.device).fill_(-1)
        max_seqlen_q_new = 2
        mla_decode_fwd_dispatch(q_fp8_padded,
                                kv_buffer.view(-1, 1, 1, q.shape[-1]),
                                o_padded,
                                qo_indptr_padded,
                                kv_indptr,
                                kv_indices,
                                kv_last_page_lens,
                                max_seqlen_q_new,
                                sm_scale=sm_scale,
                                logit_cap=logit_cap,
                                num_kv_splits=num_kv_splits,
                                num_kv_splits_indptr=num_kv_splits_indptr,
                                work_meta_data=work_metadata,
                                work_indptr=work_indptr,
                                work_info_set=work_info_set,
                                reduce_indptr=reduce_indptr,
                                reduce_final_map=reduce_final_map,
                                reduce_partial_map=reduce_partial_map,
                                q_scale=q_scale,
                                kv_scale=kv_scale,
                                )
        o[:] = o_padded[::2]  # Extract every second element
    else:
        mla_decode_fwd_dispatch(q,
                                kv_buffer.view(-1, 1, 1, q.shape[-1]),
                                o,
                                qo_indptr,
                                kv_indptr,
                                kv_indices,
                                kv_last_page_lens,
                                1,
                                sm_scale=sm_scale,
                                logit_cap=logit_cap,
                                num_kv_splits=num_kv_splits,
                                num_kv_splits_indptr=num_kv_splits_indptr,
                                work_meta_data=work_metadata,
                                work_indptr=work_indptr,
                                work_info_set=work_info_set,
                                reduce_indptr=reduce_indptr,
                                reduce_final_map=reduce_final_map,
                                reduce_partial_map=reduce_partial_map
                                )


def mla_decode_fwd_impl(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    max_seqlen_q: int,
    sm_scale: Optional[float] = None,
    varlen: Optional[bool] = False,
    logit_cap: Optional[float] = 0.0,
    num_kv_splits: Optional[int] = 1,
    num_kv_splits_indptr: Optional[torch.Tensor] = None,
    work_metadata: Optional[torch.Tensor] = None,
    work_indptr: Optional[torch.Tensor] = None,
    work_info_set: Optional[torch.Tensor] = None,
    reduce_indptr: Optional[torch.Tensor] = None,
    reduce_final_map: Optional[torch.Tensor] = None,
    reduce_partial_map: Optional[torch.Tensor] = None,

    # batch_split_table: Optional[torch.Tensor] = None,
    # split_table: Optional[torch.Tensor] = None,
    # splits: Optional[torch.Tensor] = None,
    # q_rope: Optional[torch.Tensor] = None,
    # k_rope: Optional[torch.Tensor] = None,
) -> None:
    from aiter.mla import mla_decode_fwd_dispatch

    mla_decode_fwd_dispatch(q,
                            kv_buffer.view(-1, 1, 1, q.shape[-1]),
                            o,
                            qo_indptr,
                            kv_indptr,
                            kv_indices,
                            kv_last_page_lens,
                            max_seqlen_q,
                            sm_scale=sm_scale,
                            logit_cap=logit_cap,
                            num_kv_splits=num_kv_splits,
                            num_kv_splits_indptr=num_kv_splits_indptr,
                            work_meta_data= work_metadata,
                            work_indptr=work_indptr,
                            work_info_set=work_info_set,
                            reduce_indptr=reduce_indptr,
                            reduce_final_map=reduce_final_map,
                            reduce_partial_map=reduce_partial_map,
                            # batch_split_table=batch_split_table,
                            # split_table=split_table,
                            # cu_num=splits,
                            )


def mla_decode_fwd_fake(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    max_seqlen_q: int,
    sm_scale: Optional[float] = None,
    varlen: Optional[bool] = False,
    logit_cap: Optional[float] = 0.0,
    num_kv_splits: Optional[int] = 1,
    num_kv_splits_indptr: Optional[torch.Tensor] = None,
    work_metadata: Optional[torch.Tensor] = None,
    work_indptr: Optional[torch.Tensor] = None,
    work_info_set: Optional[torch.Tensor] = None,
    reduce_indptr: Optional[torch.Tensor] = None,
    reduce_final_map: Optional[torch.Tensor] = None,
    reduce_partial_map: Optional[torch.Tensor] = None,

    # batch_split_table: Optional[torch.Tensor] = None,
    # split_table: Optional[torch.Tensor] = None,
    # splits: Optional[torch.Tensor] = None,
    # q_rope: Optional[torch.Tensor] = None,
    # k_rope: Optional[torch.Tensor] = None,
) -> None:
    pass


if current_platform.is_rocm():
    direct_register_custom_op(op_name="rocm_aiter_mla_decode_fwd",
                              op_func=mla_decode_fwd_impl,
                              mutates_args=["o"],
                              fake_impl=mla_decode_fwd_fake,
                              tags=[torch.Tag.needs_fixed_stride_order])
