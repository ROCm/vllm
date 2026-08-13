# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import ParallelConfig
from vllm.v1.attention.backend import CommonAttentionMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class UBatchSlice:
    request_slice: slice
    token_slice: slice

    def is_empty(self) -> bool:
        return (
            self.request_slice.start == self.request_slice.stop
            or self.token_slice.start == self.token_slice.stop
        )

    @property
    def num_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start


UBatchSlices: TypeAlias = list[UBatchSlice]


def is_last_ubatch_empty(
    orig_num_tokens: int, padded_num_tokens: int, num_ubatches: int
) -> bool:
    return (padded_num_tokens // num_ubatches) * (num_ubatches - 1) >= orig_num_tokens


def check_ubatch_thresholds(
    config: ParallelConfig, num_tokens: int, uniform_decode: bool
) -> bool:
    if not config.use_ubatching:
        return False
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold
    else:
        return num_tokens >= config.dbo_prefill_token_threshold


def align_ubatch_splits_to_requests(vllm_config: "VllmConfig") -> bool:
    """Whether microbatch cuts must land on request boundaries.

    Scoped to DeepEP high throughput. `GPUUBatchWrapper.__call__` fabricates each
    microbatch's cross-rank token vector from its own slice size, which is only
    correct while every rank cuts at `num_tokens_padded // num_ubatches`. A snapped
    cut is a function of the local request boundaries, so the ranks no longer agree.
    High throughput never reads that vector -- its per-rank counts come from
    `Buffer.get_dispatch_layout` -- but the batched formats low latency selects do
    (`estimate_expected_m`), so widening this needs the real per-rank sizes carried
    in the collective first.

    This also implies `cudagraph_mode` is NONE (`CompilationConfig` forces it for
    high throughput at `data_parallel_size > 1`), which matters because
    `GPUUBatchWrapper` keys captured microbatched graphs on the total token count
    alone.
    """
    parallel_config = vllm_config.parallel_config
    return (
        envs.VLLM_BATCH_INVARIANT
        and parallel_config.use_ubatching
        and parallel_config.data_parallel_size > 1
        and parallel_config.all2all_backend == "deepep_high_throughput"
    )


def can_align_ubatch_split(num_scheduled_tokens: np.ndarray, num_ubatches: int) -> bool:
    """Whether this batch has enough requests to split on request boundaries.

    One boundary is needed per split point, and only the interior ones count, so
    `num_ubatches` non-empty requests are the minimum. A batch prefilling a single
    request has none, and cannot be microbatched without cutting that request --
    which is what `request_aligned_split_points` refuses to do. All DP ranks have
    to agree before any of them splits, so this is answered before the collective
    rather than at slice-creation time.
    """
    return int((np.asarray(num_scheduled_tokens) > 0).sum()) >= num_ubatches


def request_aligned_split_points(
    num_scheduled_tokens: np.ndarray,
    num_tokens_padded: int,
    num_ubatches: int,
) -> list[int] | None:
    """Split points on request boundaries, as near an even division as possible.

    A microbatch boundary that lands inside a request makes that request's
    continuation carry `seq_len` unchanged against a shortened `query_len` (see
    `_make_metadata_with_slice`), so MLA reads a non-zero context length for it and
    decomposes its prefill into a context pass plus a new-token pass merged by
    log-sum-exp, instead of one causal pass. That is mathematically equivalent and
    numerically different, and where the cut lands is a function of the DP peers'
    load, so the same request gets a different answer run to run.

    So under batch invariance the cut is snapped to a request boundary and the
    straddling request is deferred to one side whole, matching the scheduler's rule
    of deferring rather than shrinking.

    Returns None when there are not enough interior boundaries, which the caller
    must treat as "do not microbatch" -- falling back to an even division would
    reintroduce exactly the cut this avoids.
    """
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int64)
    np.cumsum(num_scheduled_tokens, out=cu_num_tokens[1:])
    total = int(cu_num_tokens[-1])

    # Interior boundaries only: 0 or `total` would leave a microbatch empty, and
    # duplicates come from requests scheduled zero tokens.
    boundaries = np.unique(cu_num_tokens[1:-1])
    boundaries = boundaries[(boundaries > 0) & (boundaries < total)]
    if boundaries.size < num_ubatches - 1:
        return None

    split_points: list[int] = []
    for i in range(1, num_ubatches):
        # Target the even division of the *padded* count, because that is what the
        # unaligned path splits on and what the DP ranks have agreed to run.
        target = int(num_tokens_padded) * i // num_ubatches
        lower = split_points[-1] if split_points else 0
        available = boundaries[boundaries > lower]
        # Leave one boundary behind for each split point still to be placed.
        still_to_place = num_ubatches - 1 - len(split_points)
        if available.size < still_to_place:
            return None
        if still_to_place > 1:
            available = available[: available.size - (still_to_place - 1)]
        split_points.append(int(available[np.argmin(np.abs(available - target))]))

    return split_points


# This pads the last ubatch slice out to the total number of tokens
# (num_tokens + padding) since we do `create_ubatch_slices` before applying DP padding.
def _pad_out_ubatch_slices(
    ubatch_slices: UBatchSlices, num_total_tokens: int, num_reqs_padded: int
) -> UBatchSlices:
    last_slice = ubatch_slices[-1]
    padded_last_request_slice = slice(last_slice.request_slice.start, num_reqs_padded)
    padded_last_token_slice = slice(last_slice.token_slice.start, num_total_tokens)

    return ubatch_slices[:-1] + [
        UBatchSlice(padded_last_request_slice, padded_last_token_slice)
    ]


def maybe_create_ubatch_slices(
    should_ubatch: bool,
    num_scheduled_tokens: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_ubatches: int,
    split_point: list[int] | int | None = None,
    align_to_request_boundaries: bool = False,
) -> tuple[UBatchSlices | None, UBatchSlices | None]:
    if not should_ubatch:
        return None, None

    if split_point is None and align_to_request_boundaries:
        # None here means the batch has no interior request boundary. The DP ranks
        # agree on that before they agree to microbatch (`can_align_ubatch_split`),
        # so reaching this with alignment on means the two disagreed; splitting
        # anyway would cut a request, which is the thing being avoided.
        split_point = request_aligned_split_points(
            num_scheduled_tokens, num_tokens_padded, num_ubatches
        )
        assert split_point is not None, (
            "microbatching was agreed for a batch with no interior request "
            "boundary; the DP coordination should have declined it"
        )

    if split_point is None:
        split_point = int(num_tokens_padded) // num_ubatches

    # A sequence is the cuts themselves; a scalar is a stride giving evenly spaced
    # ones. Tested as a sequence because callers reach here with numpy integers.
    if isinstance(split_point, (list, tuple)):
        token_split_points = [int(point) for point in split_point]
    else:
        token_split_points = [int(split_point) * i for i in range(1, num_ubatches)]

    # TODO(lucas): Refactor the gpu_model_runner.py so we can pass
    # in cu_num_tokens directly (i.e. query_start_loc)
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    ubatch_slices = []
    start_token = 0

    # Add the end point to the split points to make iteration easier
    all_points = token_split_points + [cu_num_tokens[-1]]

    for end_token in all_points:
        token_slice = slice(start_token, end_token)

        # Determine request slices using exclusive stop semantics
        # Ubatch includes requests whose tokens overlap [start_token, end_token)

        # Start at the request that contains the start_token
        # or the request starting exactly at start_token (if on boundary)
        req_start = int(np.searchsorted(cu_num_tokens, start_token, side="right") - 1)

        # Stop at the request that starts at or after end_token
        req_stop = int(np.searchsorted(cu_num_tokens, end_token, side="left"))

        req_slice = slice(req_start, req_stop)
        ubatch_slices.append(UBatchSlice(req_slice, token_slice))

        start_token = end_token

    ubatch_slices_padded = _pad_out_ubatch_slices(
        ubatch_slices, num_tokens_padded, num_reqs_padded
    )

    assert sum(s.num_tokens for s in ubatch_slices_padded) == num_tokens_padded

    return ubatch_slices, ubatch_slices_padded


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    """
    Creates a new query_start_loc that corresponds to the requests in
    request_slice.

    Note: This function creates a new tensor to hold the new query_start_locs.
    This will break cudagraph compatibility.
    """
    return (
        query_start_loc[request_slice.start : request_slice.stop + 1]
        - query_start_loc[request_slice.start]
    )


def _make_metadata_with_slice(
    ubatch_slice: UBatchSlice, attn_metadata: CommonAttentionMetadata
) -> CommonAttentionMetadata:
    """
    This function creates a new CommonAttentionMetadata that corresponds to
    the requests included in ubatch_slice
    """

    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    start_locs = attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], (
        "Token slice start outside of first request"
    )
    # NOTE: last token can be outside of the last request if we have CG padding.

    # If the request is split across ubatches, we have to adjust the metadata.
    # splits_first_request: The first request in this slice is the continuation of
    #                       a request that started in a previous slice.
    # splits_last_request:  The last request in this slice continues into the
    #                       next slice.
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(
        attn_metadata.query_start_loc, request_slice
    )

    assert len(query_start_loc) >= 2, (
        f"query_start_loc must have at least 2 elements, got {len(query_start_loc)}"
    )

    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped
    seq_lens = attn_metadata.seq_lens[request_slice]
    # Read raw fields to avoid triggering the deprecated D2H-syncing properties.
    seq_lens_cpu = (
        attn_metadata._seq_lens_cpu[request_slice]
        if attn_metadata._seq_lens_cpu is not None
        else None
    )
    seq_lens_cpu_upper_bound = (
        attn_metadata.seq_lens_cpu_upper_bound[request_slice]
        if attn_metadata.seq_lens_cpu_upper_bound is not None
        else None
    )
    num_computed_tokens_cpu = (
        attn_metadata._num_computed_tokens_cpu[request_slice]
        if attn_metadata._num_computed_tokens_cpu is not None
        else None
    )

    if splits_last_request:
        # NOTE: We use start_locs (the original query_start_loc_cpu) to calculate
        # the tokens skipped because query_start_loc_cpu might have been modified
        # if splits_first_request is True.
        tokens_skipped = start_locs[last_req + 1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped

        # Make sure we don't modify the seq_lens tensors
        #  (not cudagraph compatible)
        seq_lens = seq_lens.clone()
        seq_lens[-1] -= tokens_skipped
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu.clone()
            seq_lens_cpu[-1] -= tokens_skipped
        if seq_lens_cpu_upper_bound is not None:
            seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound.clone()
            seq_lens_cpu_upper_bound[-1] -= tokens_skipped

    assert seq_lens_cpu_upper_bound is not None
    # Preserve the max_seq_len override set during CUDA-graph capture so
    # the attention backend selects the correct kernel for SWA layers.
    max_seq_len = max(int(seq_lens_cpu_upper_bound.max()), attn_metadata.max_seq_len)

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item()
    )

    # This is to account for the case where we are in a dummy
    # run and query_start_loc_cpu is full of 0s
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
    )


def split_attn_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: CommonAttentionMetadata,
) -> list[CommonAttentionMetadata]:
    """
    Creates a new CommonAttentionMetadata instance that corresponds to the
    requests for each UBatchSlice in ubatch_slices.

    Note: This function does not modify common_attn_metadata
    """
    results = []
    for ubatch_slice in ubatch_slices:
        results.append(_make_metadata_with_slice(ubatch_slice, common_attn_metadata))

    return results
