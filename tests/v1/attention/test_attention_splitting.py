# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.attention.test_attention_backends import BATCH_SPECS
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
)
from vllm.v1.worker.dp_utils import _post_process_ubatch
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    _make_metadata_with_slice,
    align_ubatch_splits_to_requests,
    can_align_ubatch_split,
    maybe_create_ubatch_slices,
    request_aligned_split_points,
    slice_query_start_locs,
    split_attn_metadata,
)


@pytest.fixture
def sample_query_start_loc():
    """Sample query_start_loc tensor for testing"""
    return torch.tensor([0, 5, 12, 20, 35, 50])


def test_basic_slice_middle(sample_query_start_loc):
    """Test slicing from middle of tensor"""
    req_slice = slice(1, 3)  # slice from index 1 to 3
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 7, 15])
    assert torch.equal(result, expected)


def test_slice_from_beginning(sample_query_start_loc):
    """Test slicing from the beginning of tensor"""
    req_slice = slice(0, 2)  # slice from index 0 to 2
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 5, 12])
    assert torch.equal(result, expected)


def test_slice_to_end(sample_query_start_loc):
    """Test slicing to the end of tensor"""
    req_slice = slice(3, 5)  # slice from index 3 to 5 (last index)
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 15, 30])
    assert torch.equal(result, expected)


def test_single_element_slice(sample_query_start_loc):
    """Test slice that results in single element"""
    req_slice = slice(2, 3)  # slice from index 2 to 3
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 8])
    assert torch.equal(result, expected)


def test_full_tensor_slice(sample_query_start_loc):
    """Test slicing the entire tensor"""
    req_slice = slice(0, 5)  # slice entire tensor
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 5, 12, 20, 35, 50])
    assert torch.equal(result, expected)


def test_slice_bounds_edge_cases(sample_query_start_loc):
    # Test slice that goes exactly to the last element
    req_slice = slice(4, 5)  # Last index
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 15])
    assert torch.equal(result, expected)


@pytest.fixture
def small_decode_metadata():
    """Create metadata for small decode batch"""
    batch_spec = BATCH_SPECS["small_decode"]
    device = torch.device("cpu")
    return create_common_attn_metadata(batch_spec, block_size=16, device=device)


@pytest.fixture
def large_decode_metadata():
    """Create metadata for small decode batch"""
    batch_spec = BATCH_SPECS["large_decode"]
    device = torch.device("cpu")
    return create_common_attn_metadata(batch_spec, block_size=16, device=device)


@pytest.fixture
def mixed_small_metadata():
    """Create metadata for mixed small batch"""
    batch_spec = BATCH_SPECS["mixed_small"]
    device = torch.device("cpu")
    return create_common_attn_metadata(batch_spec, block_size=16, device=device)


# Tests for _make_metadata_with_slice
def test_make_metadata_with_slice_decode_batch(small_decode_metadata):
    """Test slicing decode batch metadata"""
    # Split first request only
    ubatch_slice = UBatchSlice(slice(0, 1), slice(0, 1))

    result = _make_metadata_with_slice(ubatch_slice, small_decode_metadata)

    # Check sliced results
    assert result.num_reqs == 1  # slice(0, 1) gives 1 requests
    assert result.num_actual_tokens == 1  # slice(0, 1) gives 1 token
    assert result.max_query_len == 1
    assert torch.equal(result.query_start_loc, torch.tensor([0, 1]))
    assert torch.equal(result.seq_lens, torch.tensor([32]))


def test_make_metadata_with_slice_mixed_batch(mixed_small_metadata):
    """Test slicing mixed batch metadata"""
    ubatch_slice = UBatchSlice(slice(1, 3), slice(1, 7))  # Requests 1-3, tokens 1-7

    result = _make_metadata_with_slice(ubatch_slice, mixed_small_metadata)

    assert result.num_reqs == 2  # slice(1, 3) gives 2 requests
    assert result.num_actual_tokens == 6  # slice(1, 7) gives 6 tokens
    assert result.max_query_len == 5
    assert torch.equal(result.query_start_loc, torch.tensor([0, 1, 6]))
    assert torch.equal(result.seq_lens, torch.tensor([40, 48]))


def test_split_attn_metadata_decode_batch(large_decode_metadata):
    """Test splitting decode batch into two equal parts"""
    num_tokens = large_decode_metadata.num_reqs
    mid_point = num_tokens // 2
    ubatch_slices = [
        UBatchSlice(slice(0, mid_point), slice(0, mid_point)),
        UBatchSlice(slice(mid_point, num_tokens), slice(mid_point, num_tokens)),
    ]

    results = split_attn_metadata(ubatch_slices, large_decode_metadata)

    assert len(results) == 2

    # Check first split
    assert results[0].num_reqs == mid_point
    assert results[0].num_actual_tokens == mid_point
    assert torch.equal(results[0].seq_lens, torch.tensor([2048] * mid_point))

    # Check second split
    assert results[1].num_reqs == mid_point
    assert results[1].num_actual_tokens == mid_point
    assert torch.equal(results[1].seq_lens, torch.tensor([2048] * mid_point))


def apply_split_decodes_and_prefills(
    query_lens: list[int],
    decode_threshold: int,
    require_uniform: bool,
    padded_num_tokens: int | None = None,
    is_prefilling: list[bool] | None = None,
    treat_short_extends_as_decodes: bool = True,
):
    """Helper function to apply split_decodes_and_prefills and return
    the results."""
    device = torch.device("cpu")
    seq_lens = [10 * (i + 1) for i in range(len(query_lens))]
    common_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=16,
        device=device,
    )

    if padded_num_tokens is not None:
        common_metadata.num_actual_tokens = padded_num_tokens
    if is_prefilling is not None:
        common_metadata.is_prefilling = torch.tensor(is_prefilling)

    return split_decodes_and_prefills(
        common_metadata,
        decode_threshold=decode_threshold,
        require_uniform=require_uniform,
        treat_short_extends_as_decodes=treat_short_extends_as_decodes,
    )


def test_split_decodes_and_prefills_nonuniform_all_ones():
    query_lens = [1, 1, 1]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 1, False)
    )
    assert num_decodes == 3
    assert num_prefills == 0
    assert num_decode_tokens == 3
    assert num_prefill_tokens == 0


def test_split_decodes_and_prefills_nonuniform_all_short_decodes():
    query_lens = [1, 2, 1, 3, 2, 1, 2]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, False)
    )
    assert num_decodes == 7
    assert num_prefills == 0
    assert num_decode_tokens == sum(query_lens)
    assert num_prefill_tokens == 0


def test_split_decodes_and_prefills_nonuniform_all_prefills():
    query_lens = [4, 5, 6, 7]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, False)
    )
    assert num_decodes == 0
    assert num_prefills == 4
    assert num_decode_tokens == 0
    assert num_prefill_tokens == sum(query_lens)


def test_split_decodes_and_prefills_nonuniform_mixed_batch():
    query_lens = [2, 1, 3, 4, 5, 6, 7, 8]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 4, False)
    )
    assert num_decodes == 4  # 2, 1, 3, 4 are all <= 4
    assert num_prefills == 4  # 5, 6, 7, 8 are all > 4
    assert num_decode_tokens == 10  # 2 + 1 + 3 + 4
    assert num_prefill_tokens == 26  # 5 + 6 + 7 + 8


def test_split_decodes_and_prefills_uniform_all_ones():
    query_lens = [1, 1, 1]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 1, True)
    )
    assert num_decodes == 3
    assert num_prefills == 0
    assert num_decode_tokens == 3
    assert num_prefill_tokens == 0


def test_split_decodes_and_prefills_uniform_short_extend():
    result = apply_split_decodes_and_prefills(
        [1, 1],
        decode_threshold=1,
        require_uniform=True,
        is_prefilling=[False, True],
        treat_short_extends_as_decodes=False,
    )
    assert result == (1, 1, 1, 1)


def test_split_decodes_and_prefills_uniform_all_short_decodes():
    query_lens = [2, 2, 1, 3, 2, 1, 2]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, True)
    )
    assert num_decodes == 2
    assert num_prefills == 5
    assert num_decode_tokens == 4
    assert num_prefill_tokens == (1 + 3 + 2 + 1 + 2)


def test_split_decodes_and_prefills_uniform_all_prefills():
    query_lens = [4, 5, 6, 7]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, True)
    )
    assert num_decodes == 0
    assert num_prefills == 4
    assert num_decode_tokens == 0
    assert num_prefill_tokens == sum(query_lens)


def test_split_decodes_and_prefills_uniform_mixed_batch_all_uniform_decodes():
    query_lens = [2, 2, 2, 4, 5, 6, 7, 8]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 4, True)
    )
    assert num_decodes == 3  # 2, 2, 2 are all <= 4 and uniform
    assert num_prefills == 5  # 4, 5, 6, 7, 8 are all > 4
    assert num_decode_tokens == 6  # 2 + 2 + 2
    assert num_prefill_tokens == 30  # 4 + 5 + 6 + 7 + 8


def test_split_decodes_and_prefills_uniform_mixed_batch_non_uniform_decodes():
    query_lens = [2, 1, 2, 4, 5, 6, 7, 8]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 4, True)
    )
    assert num_decodes == 1  # only the first 2 is taken as decode
    assert num_prefills == 7  # 1, 2, 4, 5, 6, 7, 8 are all > 4 or non-uniform
    assert num_decode_tokens == 2  # only the first 2
    assert num_prefill_tokens == (sum(query_lens) - 2)  # rest of the tokens


def test_split_decodes_and_prefills_uniform_padded_batch_all_same():
    """uniform batch where all query lengths are identical with 0 length padded reqs."""
    # All query lengths are 2, with decode_threshold=3 (so 2 <= 3)
    # This triggers the padded uniform path at line 891
    query_lens = [2, 2, 2, 0]
    padded_num_tokens = 8
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, True, padded_num_tokens)
    )
    # With uniform batch, all requests are treated as decodes
    assert num_decodes == 4
    assert num_prefills == 0
    assert num_decode_tokens == padded_num_tokens
    assert num_prefill_tokens == 0


@pytest.mark.parametrize(
    "seq_lens,query_lens,split_point,expected_first_reqs,expected_second_reqs",
    [
        # Split in the middle of request 1
        ([32, 40], [8, 8], 12, 2, 1),
        # Split inside the first request
        ([32, 40], [8, 8], 4, 1, 2),
    ],
)
def test_prefill_split_across_ubatches(
    seq_lens, query_lens, split_point, expected_first_reqs, expected_second_reqs
):
    """Test splitting a prefill across ubatches"""
    import numpy as np

    device = torch.device("cpu")
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(batch_spec, block_size=16, device=device)

    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    qsl_np = common.query_start_loc_cpu.numpy()
    num_tokens = common.num_actual_tokens

    ubatch_slices, _ = maybe_create_ubatch_slices(
        True,
        num_scheduled_tokens,
        num_tokens,
        batch_spec.batch_size,
        split_point=split_point,
        num_ubatches=2,
    )
    assert ubatch_slices is not None and len(ubatch_slices) == 2

    first_meta = _make_metadata_with_slice(ubatch_slices[0], common)
    second_meta = _make_metadata_with_slice(ubatch_slices[1], common)

    # Token counts match the split
    assert first_meta.num_actual_tokens == split_point
    assert second_meta.num_actual_tokens == num_tokens - split_point

    # Number of requests per ubatch
    assert first_meta.num_reqs == expected_first_reqs
    assert second_meta.num_reqs == expected_second_reqs

    # Identify which request is split and how many tokens are in the first chunk
    split_req_idx = int(np.searchsorted(qsl_np, split_point, side="right") - 1)
    tokens_in_first_chunk = split_point - int(qsl_np[split_req_idx])
    orig_q_lens = common.query_start_loc_cpu[1:] - common.query_start_loc_cpu[:-1]

    # Check query length continuity: first-chunk + second-chunk == original qlen
    # First ubatch last request query length
    qlen_first_last = int(
        first_meta.query_start_loc_cpu[-1] - first_meta.query_start_loc_cpu[-2]
    )
    # Second ubatch first request query length
    qlen_second_first = int(
        second_meta.query_start_loc_cpu[1] - second_meta.query_start_loc_cpu[0]
    )
    assert qlen_first_last == tokens_in_first_chunk
    assert qlen_first_last + qlen_second_first == int(orig_q_lens[split_req_idx])

    # Check seq_lens adjustments
    # Context lengths per original request
    context_lens = [s - q for s, q in zip(seq_lens, query_lens)]

    # First ubatch: last request's seq_len should be
    #  context + tokens_in_first_chunk
    expected_seqlen = context_lens[split_req_idx] + tokens_in_first_chunk
    assert int(first_meta.seq_lens[-1]) == expected_seqlen

    # For full preceding requests in first ubatch, seq_lens should match
    #  originals
    for i in range(first_meta.num_reqs - 1):
        assert int(first_meta.seq_lens[i]) == seq_lens[i]

    # Second ubatch: first request (continuation) seq_len should be full
    #  original
    assert int(second_meta.seq_lens[0]) == seq_lens[split_req_idx]
    # Any following full requests in second ubatch should match originals
    for j in range(1, second_meta.num_reqs):
        # Map to original request index
        orig_idx = split_req_idx + j
        assert int(second_meta.seq_lens[j]) == seq_lens[orig_idx]


def test_build_attention_metadata_zeros_stale_is_prefilling():
    """_build_attention_metadata zeroes is_prefilling for padded rows."""
    from unittest.mock import MagicMock, patch

    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    num_reqs = 3
    num_reqs_padded = 5

    # Real rows [0-2] have known computed/prompt values; padded rows [3-4]
    # carry stale data from a prior prefill (num_computed < num_prompt → True).
    num_computed = torch.tensor([50, 100, 200, 10, 20], dtype=torch.int32)
    num_prompt = torch.tensor([50, 200, 200, 100, 200], dtype=torch.int32)

    runner = MagicMock()
    runner.kv_cache_config.kv_cache_groups = [
        MagicMock()
    ]  # non-empty: skip early return
    runner.attn_groups = [[]]  # empty inner list: inner loop never runs
    runner.input_batch.num_computed_tokens_cpu_tensor = num_computed
    runner.input_batch.num_prompt_tokens_cpu_tensor = num_prompt
    runner.optimistic_seq_lens_cpu = torch.tensor([100, 200, 300, 0, 0])
    runner.query_start_loc.gpu = torch.zeros(num_reqs_padded + 1, dtype=torch.int32)
    runner.query_start_loc.cpu = torch.zeros(num_reqs_padded + 1, dtype=torch.int32)
    runner.seq_lens = torch.zeros(num_reqs_padded, dtype=torch.int32)
    runner.positions = torch.zeros(num_reqs_padded, dtype=torch.int64)
    runner.routed_experts_initialized = False
    runner.use_async_spec_decode = False
    runner.dcp_world_size = 1
    runner.speculative_config = None
    runner.is_mm_prefix_lm = False
    runner._get_encoder_seq_lens.return_value = (None, None)

    # Intercept CommonAttentionMetadata construction to capture is_prefilling.
    # With speculative_config=None the constructor is called exactly once (for
    # cm_base), so captured reflects what the fix produced before storage.
    captured_is_prefilling = None
    original_init = CommonAttentionMetadata.__init__

    def capturing_init(self, *args, **kwargs):
        nonlocal captured_is_prefilling
        if "is_prefilling" in kwargs:
            captured_is_prefilling = kwargs["is_prefilling"]
        original_init(self, *args, **kwargs)

    with patch.object(CommonAttentionMetadata, "__init__", capturing_init):
        GPUModelRunner._build_attention_metadata(
            runner,
            num_tokens=num_reqs,
            num_reqs=num_reqs,
            max_query_len=1,
            num_tokens_padded=num_reqs_padded,
            num_reqs_padded=num_reqs_padded,
            slot_mappings={0: torch.zeros(num_reqs_padded, dtype=torch.int64)},
        )

    assert captured_is_prefilling is not None
    assert not captured_is_prefilling[0]  # decode  (50 >= 50)
    assert captured_is_prefilling[1]  # prefill (100 < 200)
    assert not captured_is_prefilling[2]  # decode  (200 >= 200)
    assert not captured_is_prefilling[3]  # stale data (10 < 100) zeroed
    assert not captured_is_prefilling[4]  # stale data (20 < 200) zeroed


@pytest.mark.parametrize(
    "query_lens,num_tokens_padded,expected",
    [
        # The even division already lands on a boundary: nothing to move.
        ([100, 100, 100, 100], 400, [200]),
        # 225 falls inside request 1; both neighbours are 75 away, take the lower.
        ([150, 150, 150], 450, [150]),
        # DP padding pushes the target past every real boundary; the last one wins.
        ([10, 10, 10], 600, [20]),
        # Requests scheduled zero tokens do not contribute a distinct boundary.
        ([100, 0, 100], 200, [100]),
        # More than two microbatches needs one distinct boundary per cut.
        ([100] * 8, 800, [200, 400, 600]),
    ],
)
def test_request_aligned_split_points(query_lens, num_tokens_padded, expected):
    """Split points must coincide with request boundaries."""
    import numpy as np

    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    num_ubatches = len(expected) + 1
    points = request_aligned_split_points(
        num_scheduled_tokens, num_tokens_padded, num_ubatches
    )
    assert points == expected

    cu = np.concatenate(([0], np.cumsum(num_scheduled_tokens)))
    for point in points:
        assert point in cu


@pytest.mark.parametrize(
    "query_lens,num_ubatches",
    [
        # A lone prefill has no interior boundary, so it cannot be cut at all.
        ([900], 2),
        ([900, 0], 2),
        # Two requests give one boundary, one short of what three ubatches need.
        ([100, 100], 3),
    ],
)
def test_request_aligned_split_points_declines(query_lens, num_ubatches):
    """With too few requests there is nowhere to cut, and None says so.

    The caller must decline to microbatch rather than fall back to an even
    division: falling back would reintroduce the mid-request cut this avoids.
    `can_align_ubatch_split` is the pre-collective form of the same question and
    has to agree, since the DP ranks decide before the split point is computed.
    """
    import numpy as np

    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    assert (
        request_aligned_split_points(
            num_scheduled_tokens, int(num_scheduled_tokens.sum()), num_ubatches
        )
        is None
    )
    assert not can_align_ubatch_split(num_scheduled_tokens, num_ubatches)


@pytest.mark.parametrize(
    "seq_lens,query_lens",
    [
        # Uneven, so the even split at 8 would land inside the second request
        # and the aligned one has somewhere else to go.
        ([32, 40], [5, 11]),
        ([500, 900, 700], [500, 900, 700]),
        ([64, 64, 64, 64], [7, 21, 3, 33]),
    ],
)
def test_aligned_ubatch_slices_never_split_a_request(seq_lens, query_lens):
    """The property the alignment exists for.

    A slice whose first or last request is a continuation makes MLA read a
    non-zero context length for it and decompose its prefill differently, which
    is batch variant because the cut moves with the DP peers' load. Aligned
    slices must leave every request whole, which shows up as an unchanged
    seq_lens and a query_start_loc starting at zero.
    """
    import numpy as np

    device = torch.device("cpu")
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(batch_spec, block_size=16, device=device)
    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    num_tokens = common.num_actual_tokens

    ubatch_slices, _ = maybe_create_ubatch_slices(
        True,
        num_scheduled_tokens,
        num_tokens,
        len(query_lens),
        num_ubatches=2,
        align_to_request_boundaries=True,
    )
    assert ubatch_slices is not None and len(ubatch_slices) == 2

    start_locs = common.query_start_loc_cpu
    for ubatch_slice in ubatch_slices:
        req_slice = ubatch_slice.request_slice
        token_slice = ubatch_slice.token_slice
        # Neither end of the slice may fall inside a request.
        assert token_slice.start == start_locs[req_slice.start]
        assert token_slice.stop == start_locs[req_slice.stop]

        meta = _make_metadata_with_slice(ubatch_slice, common)
        # A whole request keeps its sequence length and starts its own numbering.
        assert meta.query_start_loc_cpu[0] == 0
        torch.testing.assert_close(
            meta.seq_lens, common.seq_lens[req_slice], rtol=0, atol=0
        )

    # Every token still gets computed exactly once.
    assert sum(s.num_tokens for s in ubatch_slices) == num_tokens


def test_unaligned_split_is_unchanged_by_default():
    """Alignment is opt-in; the default path keeps cutting wherever it likes."""
    import numpy as np

    query_lens = [150, 150, 150]
    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)

    default, _ = maybe_create_ubatch_slices(
        True, num_scheduled_tokens, 450, len(query_lens), num_ubatches=2
    )
    assert default is not None
    # 225 is inside the second request, which is exactly what alignment avoids.
    assert default[0].token_slice.stop == 225


def _collective(should_ubatch, can_align, tokens=100, padded=None, dp_size=2):
    """The reduced tensor `_post_process_ubatch` reads, without a DP group.

    Rows are (orig tokens, padded tokens, should-ubatch, cudagraph mode,
    can-align), summed across ranks by the caller's all-reduce; each column is
    one rank.
    """
    tensor = torch.zeros(5, dp_size, dtype=torch.int32)
    tensor[0, :] = tokens
    tensor[1, :] = tokens if padded is None else padded
    tensor[2, :] = torch.tensor(should_ubatch, dtype=torch.int32)
    tensor[4, :] = torch.tensor(can_align, dtype=torch.int32)
    return tensor


def test_one_rank_without_a_boundary_vetoes_ubatching_for_everyone():
    """The alignment decision has to be unanimous, not per rank.

    A rank that cannot cut on a request boundary must stop its peers from
    microbatching too. If it declined alone it would sit out collectives the
    others had already entered, so the veto travels in the same all-reduce that
    carries the should-ubatch flag.
    """
    assert _post_process_ubatch(_collective([1, 1], [1, 1]), num_ubatches=2)
    assert not _post_process_ubatch(_collective([1, 1], [1, 0]), num_ubatches=2)
    assert not _post_process_ubatch(_collective([1, 1], [0, 0]), num_ubatches=2)


def test_ubatching_still_declines_for_its_own_reasons():
    """The new row only ever removes microbatching, it cannot force it on.

    Worth pinning because the veto is read before the empty-last-ubatch check:
    a rank that says "I could align" must not thereby skip the older reasons to
    decline.
    """
    assert not _post_process_ubatch(_collective([1, 0], [1, 1]), num_ubatches=2)
    # Padding so far past the real token count that the second microbatch would
    # be all padding -- declined before the new row is consulted.
    assert not _post_process_ubatch(
        _collective([1, 1], [1, 1], tokens=10, padded=100), num_ubatches=2
    )


@pytest.mark.parametrize(
    "batch_invariant,use_ubatching,dp_size,backend,expected",
    [
        (True, True, 2, "deepep_high_throughput", True),
        # Off in the mode this exists to serve, and off for every backend whose
        # microbatch DP metadata is fabricated from the local slice size.
        (False, True, 2, "deepep_high_throughput", False),
        (True, False, 2, "deepep_high_throughput", False),
        (True, True, 1, "deepep_high_throughput", False),
        (True, True, 2, "deepep_low_latency", False),
    ],
)
def test_alignment_gate(
    monkeypatch, batch_invariant, use_ubatching, dp_size, backend, expected
):
    """All four predicates are load-bearing, so each is pinned separately."""
    from types import SimpleNamespace

    import vllm.envs as envs
    from vllm.v1.worker import ubatch_utils

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(ubatch_utils.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            use_ubatching=use_ubatching,
            data_parallel_size=dp_size,
            all2all_backend=backend,
        )
    )
    assert align_ubatch_splits_to_requests(config) is expected
