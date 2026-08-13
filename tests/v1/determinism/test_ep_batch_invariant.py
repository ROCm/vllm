# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert parallelism must not make a token's output depend on the batch.

Under EP the MoE layer stops being a local computation. `AgRsAll2AllManager`
(`vllm/distributed/device_communicators/all2all.py`) all-gathers every rank's
tokens, each rank runs its own slice of the experts over the *whole* gathered
batch, and the per-rank partial results are summed back down by a single
`reduce_scatterv`. So a token's output is a sum over EP ranks of GEMMs whose row
counts, whose activation-quantization amax, and whose collective message sizes
are all functions of what every other rank happened to be doing. Each of those
pieces has been screened on its own elsewhere in this suite; this test is the
only one that checks they compose.

The pieces, and where they are covered individually:

- the combine's reduction order -- `test_tp_reduce_scatter_batch_invariant.py`
- the expert GEMM's row-permutation invariance --
  `test_moe_row_permutation_batch_invariant.py`
- dynamic per-tensor MoE activation scales -- `test_moe_act_quant_batch_invariant.py`
- cross-replica DP padding -- `test_dp_batch_invariant.py`

**A 4-wide combine is the minimum that means anything, and `ep_size` is not
how you get one.** The combine reduces over EP ranks, and a 2-rank reduction is
order independent, so a 2-rank run passes against a fully batch-variant
implementation. Upstream issue #30321 is exactly this trap: a user reported
DP+EP inconsistent under the mode at DP=8 and noted that "when dp=2, the results
regained consistency".

Two things conspire to make this easy to get wrong. First, EP size is not an
axis you allocate ranks to -- `FusedMoEParallelConfig.make` derives
`ep_size = dp_size * pcp_size * tp_size` -- so `--data-parallel-size 4
--enable-expert-parallel` on four GPUs is DP=4 *and* EP=4. Second, and less
obvious, **`ep_size == 4` does not imply the combine reduces over four ranks**.
`AgRsAll2AllManager._get_comm_group` picks the *DP* group unless the MoE is
sequence parallel, so DP=2 x TP=2 also reports `ep_size=4` while its combine is
2-wide and proves nothing about reduction order -- measured, not inferred. Hence
DP=4 with TP=1, and hence the `comm_world_size` guard below rather than a guard
on `ep_size`.

Sequence-parallel MoE would put the combine on the EP group and make DP=2 x TP=2
a genuine 4-wide test, but `use_sequence_parallel_moe` is consulted per model
and OLMoE does not pass it through, so that path is untested here.

The `solo_again` and `other_rank_solo` conditions are what make the rest mean
something: with the mode off the repeated solo run is clean while merely which
DP rank served an otherwise byte-identical request moves its logprobs, so the
comparison tracks batch composition rather than run-to-run noise. Several
conditions rather than one, because a variant combine can leave any single
condition clean by luck.

`--quantization fp8` is covered as a second configuration because it is the only
one that reaches the dynamic per-tensor activation scale, whose amax spans
whatever the all2all delivered rather than just the local batch. It runs under
the mode only: with the mode off the scale stays per-tensor, so `prepare` hands
a 1-row scale to `all_gatherv` alongside a many-row activation and
`_all_gather_single` asserts as soon as the DP ranks are unevenly loaded. The
promotion to per-token is what makes the shapes agree.

Scope, and what is deliberately not here:

- **Only `allgather_reducescatter`, the default.** It is also the only backend
  reachable in this configuration: DeepEP and MoRI need their kernels installed.
  Their combines look fixed-order by source audit, but an audit is not a
  measurement; `test_ep_all2all_batch_invariant.py` is where they are measured.
- **The combine only runs when DP > 1.** `use_all2all_kernels` is
  `use_ep and (dp_size > 1 or pcp_size > 1 or is_sequence_parallel)`, so TP=4
  plus `--enable-expert-parallel` with DP=1 shards the experts but never builds
  an all2all manager -- the cross-rank sum is the ordinary TP all-reduce, which
  `test_tp_all_reduce_batch_invariant.py` owns. A "TP+EP" version of this test
  would not touch `combine` at all.
- Untested: EP > 4, EPLB (`--enable-eplb` moves experts between ranks at run
  time, which is a batch dependency of a different kind), microbatching
  (`--enable-dbo`, rejected on this backend), PCP, and sequence-parallel MoE.

A server that silently fell back to non-EP, or that imported a different vLLM
checkout, would sail through the comparison below while proving nothing. Hence
`assert_server_ran_this_tree`, and hence the failure if the AgRs combine was
never observed at world size 4.
"""

import os

import pytest
from utils import (
    INSTRUMENTATION_IMPORT_HOOK,
    BackgroundLoad,
    assert_server_ran_this_tree,
    dp_completion,
    instrumented_server_env,
    read_records,
    skip_if_not_cuda_alike,
)

from tests.utils import RemoteOpenAIServer, large_gpu_mark, multi_gpu_marks

pytestmark = [
    skip_if_not_cuda_alike,
    large_gpu_mark(min_gb=40),
    *multi_gpu_marks(num_gpus=4),
]

# Must be an MoE checkpoint, and one whose expert count divides four.
MODEL = os.getenv("VLLM_EP_TEST_MODEL", "allenai/OLMoE-1B-7B-0924")
DP_SIZE = 4
# Not rank 0. Index 0 is privileged in more places than is comfortable, and
# keying a determinism verdict on it has produced false negatives here before.
NEEDLE_RANK = 2
NEEDLE_MAX_TOKENS = int(os.getenv("VLLM_EP_NEEDLE_MAX_TOKENS", "64"))
RAMP_SECONDS = float(os.getenv("VLLM_EP_RAMP_SECONDS", "10"))
DRAIN_SECONDS = float(os.getenv("VLLM_EP_DRAIN_SECONDS", "6"))

NEEDLE_PROMPT = (
    "Explain, step by step, how a four-stroke internal combustion engine "
    "converts chemical energy in fuel into rotational mechanical work, and "
    "where the main thermodynamic losses occur."
)

_PATCHERS = '''
"""Record what the MoE layer was actually configured as, to $EP_LOG.<pid>."""
import json
import os
import sys
import threading

_LOG = os.environ.get("EP_LOG")
_PATH = f"{_LOG}.{os.getpid()}" if _LOG else None
_LOCK = threading.Lock()
_SEEN = set()


def _emit(kind, **fields):
    try:
        import vllm
        import vllm.envs as envs

        record = {
            "kind": kind,
            "vllm_file": vllm.__file__,
            "batch_invariant": bool(envs.VLLM_BATCH_INVARIANT),
        }
        record.update(fields)
        with _LOCK, open(_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\\n")
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[ep-instr] {e}\\n")


def _once(signature):
    with _LOCK:
        if signature in _SEEN:
            return False
        _SEEN.add(signature)
        return True


def _patch_moe_config(module):
    original = module.FusedMoEParallelConfig.make

    def make(*args, **kwargs):
        config = original(*args, **kwargs)
        if _once(("cfg", config.ep_size, config.use_ep)):
            _emit(
                "moe_parallel_config",
                ep_size=config.ep_size,
                dp_size=config.dp_size,
                tp_size=config.tp_size,
                use_ep=config.use_ep,
                all2all_backend=config.all2all_backend,
                use_ag_rs_all2all_kernels=config.use_ag_rs_all2all_kernels,
            )
        return config

    module.FusedMoEParallelConfig.make = staticmethod(make)

    promote = module.maybe_promote_act_quant_for_batch_invariance

    def wrapped(quant_config):
        out = promote(quant_config)
        if _once("act_quant"):
            _emit(
                "act_quant",
                quant_dtype=str(getattr(quant_config, "quant_dtype", None)),
                was_dynamic_per_tensor=bool(
                    getattr(quant_config, "is_dynamic_per_tensor_act", False)
                ),
                promoted=out is not quant_config,
            )
        return out

    module.maybe_promote_act_quant_for_batch_invariance = wrapped


def _patch_all2all(module):
    original = module.AgRsAll2AllManager.combine

    def combine(self, hidden_states, is_sequence_parallel=False):
        group = self._get_comm_group(is_sequence_parallel)
        if _once(("combine", group.world_size, group.unique_name)):
            _emit(
                "combine",
                comm_world_size=group.world_size,
                comm_group=group.unique_name,
            )
        return original(self, hidden_states, is_sequence_parallel)

    module.AgRsAll2AllManager.combine = combine


_TARGETS = {
    "vllm.model_executor.layers.fused_moe.config": _patch_moe_config,
    "vllm.distributed.device_communicators.all2all": _patch_all2all,
}
'''

_INSTRUMENTATION = _PATCHERS + INSTRUMENTATION_IMPORT_HOOK


def _load(url, ranks, concurrency, seed=0) -> BackgroundLoad:
    """Companion requests spread over `ranks`, one rank per worker thread."""

    def send(rng, index):
        prompt = " ".join(
            str(rng.randint(0, 99999)) for _ in range(rng.randint(32, 96))
        )
        dp_completion(url, MODEL, prompt, 384, ranks[index % len(ranks)])

    return BackgroundLoad(
        send,
        concurrency=concurrency,
        ramp_seconds=RAMP_SECONDS,
        drain_seconds=DRAIN_SECONDS,
        seed=seed,
    )


def _needle(url, rank=NEEDLE_RANK) -> dict:
    response = dp_completion(
        url, MODEL, NEEDLE_PROMPT, NEEDLE_MAX_TOKENS, rank, logprobs=1
    )
    choice = response["choices"][0]
    return {
        "tokens": choice["logprobs"]["tokens"],
        "logprobs": choice["logprobs"]["token_logprobs"],
    }


def _make_server(tmp_path, extra_args):
    log_prefix = str(tmp_path / "ep_instr")

    args = [
        "--data-parallel-size",
        str(DP_SIZE),
        "--data-parallel-size-local",
        str(DP_SIZE),
        "--enable-expert-parallel",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "128",
        # The needle's prefill should be recomputed every time, so the
        # comparison covers it and not just the decodes.
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        os.getenv("VLLM_EP_TEST_GPU_MEMORY_UTILIZATION", "0.20"),
        *extra_args,
    ]
    env = instrumented_server_env(tmp_path, _INSTRUMENTATION, EP_LOG=log_prefix)
    # `seed` is the constructor's business: RemoteOpenAIServer appends `--seed`
    # itself and rejects an explicit one in `vllm_serve_args`.
    server = RemoteOpenAIServer(
        MODEL, args, env_dict=env, seed=20240919, max_wait_seconds=1800
    )
    return server, log_prefix


@pytest.fixture(params=["bf16", "fp8"])
def ep_server(request, tmp_path, enable_batch_invariant_mode):
    """A DP=4 + EP=4 server, unquantized and with online fp8.

    Depends on the autouse `enable_batch_invariant_mode` fixture rather than
    setting `VLLM_BATCH_INVARIANT` itself, so the server inherits whatever the
    fixture put in the environment.

    fp8 is a separate parameter rather than a separate test because it is the
    only configuration that reaches a dynamic per-tensor MoE activation scale --
    an amax over every row the kernel was handed, which under EP means every row
    the all2all delivered from every rank.
    """
    extra = ["--quantization", "fp8"] if request.param == "fp8" else []
    server, log_prefix = _make_server(tmp_path, extra)
    with server:
        yield request.param, server, log_prefix


def test_ep_needle_is_invariant_to_batch_composition(ep_server):
    """One request's logprobs must not move when the EP batch around it does."""
    quantization, server, log_prefix = ep_server
    url = server.url_for("v1/completions")
    peers = [r for r in range(DP_SIZE) if r != NEEDLE_RANK]

    # Discarded: keeps first-request state out of the comparison.
    _needle(url)

    conditions: dict[str, dict] = {}
    # Companions on the needle's own rank change its local batch directly;
    # companions on peer ranks only change what the all-gather delivers into
    # *this* rank's expert GEMMs and what the combine has to reduce. Both are
    # batch dependencies and neither subsumes the other.
    plan = [
        ("solo", 0, [NEEDLE_RANK], 0),
        ("own_rank_32", 32, [NEEDLE_RANK], 1),
        ("peer_ranks_24", 24, peers, 2),
        ("all_ranks_48", 48, list(range(DP_SIZE)), 3),
        ("solo_again", 0, [NEEDLE_RANK], 0),
    ]
    for label, concurrency, ranks, seed in plan:
        with _load(url, ranks, concurrency, seed) as load:
            conditions[label] = _needle(url)
        # Checked per condition: a peer throwing HTTP errors would otherwise
        # burn the remaining conditions before saying so.
        load.assert_ran_cleanly(f"{label} companions")

    # Informational in spirit but asserted anyway: which replica served an
    # otherwise byte-identical request is as much "the batch" as anything else,
    # and with the mode off this one moves.
    conditions["other_rank_solo"] = _needle(url, (NEEDLE_RANK + 1) % DP_SIZE)

    records = read_records(log_prefix)
    assert_server_ran_this_tree(
        {r["vllm_file"] for r in records}, {r["batch_invariant"] for r in records}
    )

    # Vacuity guards. A server that fell back to non-EP, or that never ran the
    # combine, would pass every comparison below without testing anything.
    configs = [r for r in records if r["kind"] == "moe_parallel_config"]
    assert configs and all(c["use_ep"] and c["ep_size"] == DP_SIZE for c in configs), (
        f"expert parallelism was not active as configured: {configs}. The MoE "
        f"layer must report use_ep with ep_size == {DP_SIZE}, or the experts "
        f"were never sharded and nothing here is an EP result."
    )
    assert all(
        c["all2all_backend"] == "allgather_reducescatter"
        and c["use_ag_rs_all2all_kernels"]
        for c in configs
    ), f"a different all2all backend served: {configs}"
    combines = [r for r in records if r["kind"] == "combine"]
    assert combines and all(c["comm_world_size"] == DP_SIZE for c in combines), (
        f"the all2all combine was never observed reducing over {DP_SIZE} ranks "
        f"(saw {combines}). A 2-rank reduction is order independent, so a "
        f"verdict from one would hold against a fully batch-variant combine."
    )
    if quantization == "fp8":
        promotions = [r for r in records if r["kind"] == "act_quant"]
        assert promotions and all(p["promoted"] for p in promotions), (
            f"--quantization fp8 was expected to reach a dynamic per-tensor "
            f"activation scale and have it promoted to per-token under the "
            f"mode, but the promotion did not fire: {promotions}. Without it "
            f"this configuration is not testing the path it exists to test."
        )

    base = conditions["solo"]
    failures = []
    for label, out in conditions.items():
        if label == "solo":
            continue
        if out["tokens"] != base["tokens"]:
            failures.append(f"{label}: sampled different tokens")
            continue
        moved = [
            index
            for index, (a, b) in enumerate(zip(base["logprobs"], out["logprobs"]))
            if a != b
        ]
        if moved:
            delta = max(abs(base["logprobs"][i] - out["logprobs"][i]) for i in moved)
            failures.append(
                f"{label}: {len(moved)}/{len(base['logprobs'])} logprobs moved "
                f"(first at {moved[0]}), max |delta| {delta:.3e}"
            )
    assert not failures, (
        "the needle's logprobs depend on the EP batch around it, with "
        f"quantization={quantization}:\n  " + "\n  ".join(failures)
    )
