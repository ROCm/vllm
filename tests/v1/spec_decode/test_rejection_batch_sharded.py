# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-sharded verification must reproduce the replicated rejection sampler
bit for bit: each TP rank verifies only the requests it owns (slot % TP), with
its own rows of the batch-sharded draft-logits cache, and must accept and
resample the same tokens the replicated sampler does for those requests.

Every TP rank is simulated in-process. Block verification with full draft
logits, top-p 0.95 and a greedy/sampled mix, at MiniMax-M3.1's vocab size.
"""

import numpy as np
import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

DEVICE = "cuda"
TP_SIZE = 4
MAX_NUM_REQS = 64
NUM_STEPS = 7
VOCAB = 200064
TOP_P = 0.95

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _make_batch(rng: np.random.Generator, slots: np.ndarray):
    num_reqs = len(slots)
    rows_per_req = NUM_STEPS + 1
    num_logits = num_reqs * rows_per_req
    g = torch.Generator(device=DEVICE).manual_seed(int(rng.integers(1 << 31)))
    # Peaked rows, with drafts close to the target so that most requests
    # accept a few tokens and then reject or take the bonus token.
    target = 4 * torch.randn(num_logits, VOCAB, generator=g, device=DEVICE)
    draft_logits = torch.zeros(MAX_NUM_REQS, NUM_STEPS, VOCAB, device=DEVICE)
    noise = torch.randn(num_reqs, NUM_STEPS, VOCAB, generator=g, device=DEVICE)
    req_target = target.view(num_reqs, rows_per_req, VOCAB)[:, :NUM_STEPS]
    draft_logits[torch.from_numpy(slots).to(DEVICE)] = req_target + noise
    draft_tokens = draft_logits[torch.from_numpy(slots).to(DEVICE)].argmax(-1)

    draft_sampled = torch.zeros(num_reqs, rows_per_req, dtype=torch.int64)
    draft_sampled[:, 0] = torch.from_numpy(rng.integers(0, VOCAB, size=num_reqs))
    draft_sampled[:, 1:] = draft_tokens.cpu()
    # A placeholder draft that must be rejected.
    draft_sampled[num_reqs // 2, 3] = -1
    last_pos = rng.integers(100, 100000, size=num_reqs)
    pos = last_pos[:, None] + np.arange(rows_per_req)

    temperature = torch.zeros(MAX_NUM_REQS, device=DEVICE)
    temps = rng.choice([0.0, 0.6, 1.0, 1.0, 1.0], size=num_reqs)
    temperature[torch.from_numpy(slots).to(DEVICE)] = torch.from_numpy(temps).to(
        device=DEVICE, dtype=torch.float32
    )
    seeds = torch.from_numpy(
        rng.integers(0, 2**62, size=MAX_NUM_REQS, dtype=np.int64)
    ).to(DEVICE)
    return {
        "target": target.to(torch.bfloat16),
        "draft_logits": draft_logits.to(torch.bfloat16),
        "draft_sampled": draft_sampled.view(-1).to(DEVICE),
        "pos": torch.from_numpy(pos.reshape(-1)).to(DEVICE),
        "temperature": temperature,
        "seeds": seeds,
    }


def _verify(batch, slots, rows, draft_logits, num_shards, shared_processed=None):
    """Sampler.apply_sampling_params (temperature, top-p) then rejection
    sampling, over the requests `slots` whose logits rows are `rows`."""
    num_reqs = len(slots)
    rows_t = torch.from_numpy(rows).to(DEVICE)
    idx_mapping = torch.from_numpy(slots).to(device=DEVICE, dtype=torch.int32)
    expanded_idx_mapping = idx_mapping.repeat_interleave(NUM_STEPS + 1)
    expanded_local_pos = torch.arange(
        NUM_STEPS + 1, device=DEVICE, dtype=torch.int32
    ).repeat(num_reqs)
    cu_num_logits = torch.arange(
        0, (num_reqs + 1) * (NUM_STEPS + 1), NUM_STEPS + 1, device=DEVICE
    ).to(torch.int32)
    temperature = batch["temperature"]
    if shared_processed is None:
        processed = batch["target"][rows_t].float()
        temp = temperature[expanded_idx_mapping.long()]
        processed = torch.where(
            (temp > 0)[:, None], processed / temp.clamp_min(1e-6)[:, None], processed
        )
        top_p = torch.full((processed.shape[0],), TOP_P, device=DEVICE)
        processed = apply_top_k_top_p(processed, None, top_p)
    else:
        processed = shared_processed[rows_t]
    sampled, num_sampled = rejection_sample(
        processed,
        draft_logits,
        batch["draft_sampled"][rows_t],
        cu_num_logits,
        batch["pos"][rows_t],
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        batch["seeds"],
        NUM_STEPS,
        use_block_verification=True,
        draft_logits_num_shards=num_shards,
    )
    return processed, sampled, num_sampled


def _shard_rows(slots: np.ndarray, rank: int):
    local = np.flatnonzero(slots % TP_SIZE == rank)
    rows = (local[:, None] * (NUM_STEPS + 1) + np.arange(NUM_STEPS + 1)).reshape(-1)
    return local, rows


@pytest.mark.parametrize("num_reqs", [5, 12, 37, 64])
@pytest.mark.parametrize("shared_top_p", [True, False])
def test_batch_sharded_verify_matches_replicated(num_reqs: int, shared_top_p: bool):
    """shared_top_p feeds both paths the same processed rows, isolating the
    rejection sampler; otherwise each path runs top-p on its own batch, which
    may pick a different top-p kernel for the smaller per-rank batch."""
    rng = np.random.default_rng(num_reqs)
    slots = rng.choice(MAX_NUM_REQS, size=num_reqs, replace=False).astype(np.int64)
    batch = _make_batch(rng, slots)
    all_rows = np.arange(num_reqs * (NUM_STEPS + 1))
    ref_processed, ref_sampled, ref_num_sampled = _verify(
        batch, slots, all_rows, batch["draft_logits"], 1
    )
    assert 0 < ref_num_sampled.float().mean() < NUM_STEPS + 1

    for rank in range(TP_SIZE):
        local, rows = _shard_rows(slots, rank)
        if len(local) == 0:
            continue
        local_slots = slots[local]
        sharded_cache = batch["draft_logits"].new_zeros(
            cdiv(MAX_NUM_REQS, TP_SIZE), NUM_STEPS, VOCAB
        )
        owned = torch.from_numpy(local_slots).to(DEVICE)
        sharded_cache[owned // TP_SIZE] = batch["draft_logits"][owned]
        _, sampled, num_sampled = _verify(
            batch,
            local_slots,
            rows,
            sharded_cache,
            TP_SIZE,
            ref_processed if shared_top_p else None,
        )
        local_t = torch.from_numpy(local).to(DEVICE)
        assert torch.equal(num_sampled, ref_num_sampled[local_t])
        for i, n in enumerate(num_sampled.tolist()):
            assert torch.equal(sampled[i, :n], ref_sampled[local[i], :n])
