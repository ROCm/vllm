# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whether a forward pass is split in two must not change what it computes.

`--enable-dbo` lets the model runner cut a batch into two microbatches and
overlap their compute with their all2all. The split is not a local decision:
`check_ubatch_thresholds` only *proposes* one from this rank's token count, and
`_synchronize_dp_ranks` then all-reduces the proposals so that either every DP
rank microbatches or none does. So whether *your* forward pass is cut in two is
decided by what the other replicas are doing, and the cut changes the reduction
decomposition directly -- each half runs its own dispatch, its own expert GEMMs
at half the rows, and its own combine.

That is a stronger dependency than the DP token padding covered by
`test_dp_batch_invariant.py`, which changes only the shape of a forward pass and
not how many there are.

This is the configuration `test_ep_batch_invariant.py` records as unreachable.
It still cannot run on the default `allgather_reducescatter`
(`VllmConfig.__post_init__` asserts `use_ubatching` implies
`deepep_low_latency`, `deepep_high_throughput` or `nixl_ep`), but DeepEP high
throughput is admissible under the mode -- see
`test_ep_all2all_batch_invariant.py` -- so it is reachable there.

The DBO thresholds are lowered below so the split is reachable at test scale.
`own_32` and `own_32_peers_32` put the same 32 companions on the needle's own
rank, so it proposes microbatching on every coordination step in both. In the
first the peers are idle, propose `False`, and cancel the split; in the second
they are loaded and it goes through. The needle's forward pass was cut in two,
or not, according to what three other replicas were doing.

`peers_32` is the sharper version, and needs the asymmetry spelled out: a rank
holding a single decode is below `dbo_decode_token_threshold` and can never
*propose* a split. It can only have one forced on it during a step it does
propose -- here its prefill -- so it ubatches where `solo_again` does not. Same
local batch, opposite outcome, decided entirely off-rank.

The assertions below are on the observed decisions rather than on the load
having been applied, since a run in which no forward pass was ever split would
compare a shape against itself.

Deliberately not a condition: sending the needle to a *different* DP rank and
expecting the same logprobs. That asserts the wrong axis -- it varies nothing
about any batch, and what it measures is whether two replicas agree, which they
need not. Inductor settles some kernel configs by timing candidates on the
device, once per process, so two ranks can freeze different winners. Each rank
stays bitwise repeatable against itself, so this mode's contract is intact.

`nixl_ep`, the third backend microbatching accepts, is not covered: it still
needs its kernels. `deepep_low_latency` requires a DeepEP whose low-latency
`dispatch`/`combine` restore release/acquire ordering around the signalling
slots; against one without it that arm hangs at startup rather than failing
fast. `--ubatch-size` > 2 is untested; only DBO's two-way split runs here.
"""

import os
import time

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
from vllm.utils.import_utils import has_deep_ep

pytestmark = [
    skip_if_not_cuda_alike,
    large_gpu_mark(min_gb=40),
    *multi_gpu_marks(num_gpus=4),
]

MODEL = os.getenv("VLLM_EP_TEST_MODEL", "allenai/OLMoE-1B-7B-0924")
DP_SIZE = 4
# Not rank 0: it is the DP coordinator and the first rank of every group, and
# keying a determinism verdict on index 0 has produced false negatives here.
NEEDLE_RANK = 2
NEEDLE_MAX_TOKENS = int(os.getenv("VLLM_EP_NEEDLE_MAX_TOKENS", "64"))
# Defaults are 32 and 512, which a four-rank test at this scale never reaches
# on the decode side. Lowering them is what makes the split observable; it does
# not change how the decision is taken.
DBO_DECODE_THRESHOLD = 8
DBO_PREFILL_THRESHOLD = 32
RAMP_SECONDS = float(os.getenv("VLLM_EP_RAMP_SECONDS", "12"))
DRAIN_SECONDS = float(os.getenv("VLLM_EP_DRAIN_SECONDS", "8"))

NEEDLE_PROMPT = (
    "Explain, step by step, how a four-stroke internal combustion engine "
    "converts chemical energy in fuel into rotational mechanical work, and "
    "where the main thermodynamic losses occur."
)

_PATCHERS = '''
"""Record the mode, the all2all manager, and every ubatch decision."""
import json
import os
import sys
import threading

_LOG = os.environ.get("UBATCH_LOG")
_PATH = f"{_LOG}.{os.getpid()}" if _LOG else None
_LOCK = threading.Lock()
_SEEN = set()


def _emit(kind, **fields):
    if not _PATH:
        return
    try:
        import time

        record = {"kind": kind, "t": time.time()}
        record.update(fields)
        with _LOCK, open(_PATH, "a") as f:
            f.write(json.dumps(record, default=repr) + "\\n")
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[ubatch-instr] {e}\\n")


def _once(signature):
    with _LOCK:
        if signature in _SEEN:
            return False
        _SEEN.add(signature)
        return True


def _patch_dp_utils(module):
    import vllm
    import vllm.envs as envs

    _emit(
        "env",
        vllm_file=vllm.__file__,
        batch_invariant=bool(envs.VLLM_BATCH_INVARIANT),
    )
    original = module._synchronize_dp_ranks

    # Forwarded blind rather than by name: this instrument only reads the first
    # five arguments, and a wrapper that restates the whole signature turns any
    # later addition to it into a TypeError inside the server, which surfaces as
    # an unexplained "Server exited unexpectedly" at fixture setup.
    def wrapper(
        num_tokens_unpadded,
        num_tokens_padded,
        should_attempt_ubatching,
        cudagraph_mode,
        parallel_config,
        *args,
        **kwargs,
    ):
        out = original(
            num_tokens_unpadded,
            num_tokens_padded,
            should_attempt_ubatching,
            cudagraph_mode,
            parallel_config,
            *args,
            **kwargs,
        )
        should_ubatch, after_padding, _ = out
        try:
            rank = parallel_config.data_parallel_rank
            _emit(
                "dp",
                dp_rank=rank,
                unpadded=int(num_tokens_unpadded),
                padded=None if after_padding is None else int(after_padding[rank]),
                proposed=bool(should_attempt_ubatching),
                ubatched=bool(should_ubatch),
            )
        except Exception as e:  # pragma: no cover
            _emit("dp_err", err=repr(e))
        return out

    module._synchronize_dp_ranks = wrapper


def _patch_ubatch_utils(module):
    """The slices themselves: proof a forward pass really was cut in two."""
    original = module.maybe_create_ubatch_slices

    def wrapper(should_ubatch, num_scheduled_tokens, *args, **kwargs):
        out = original(should_ubatch, num_scheduled_tokens, *args, **kwargs)
        slices, _padded = out
        if slices is not None:
            _emit("slices", n=len(slices), tokens=[s.num_tokens for s in slices])
        return out

    module.maybe_create_ubatch_slices = wrapper


def _patch_all2all(module):
    """A silent fallback to AllGather+ReduceScatter would prove nothing."""
    for name in (
        "AgRsAll2AllManager",
        "DeepEPHTAll2AllManager",
        "DeepEPLLAll2AllManager",
        "MoriAll2AllManager",
    ):
        cls = getattr(module, name, None)
        if cls is None:
            continue
        original_init = cls.__init__

        def make(name, original_init):
            def wrapper(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                if _once(("manager", name)):
                    _emit("manager", cls=name)

            return wrapper

        cls.__init__ = make(name, original_init)


_TARGETS = {
    "vllm.v1.worker.dp_utils": _patch_dp_utils,
    "vllm.v1.worker.ubatch_utils": _patch_ubatch_utils,
    "vllm.distributed.device_communicators.all2all": _patch_all2all,
}
'''

_INSTRUMENTATION = _PATCHERS + INSTRUMENTATION_IMPORT_HOOK


def _load(url, ranks, concurrency, seed=0) -> BackgroundLoad:
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
    started = time.time()
    response = dp_completion(
        url, MODEL, NEEDLE_PROMPT, NEEDLE_MAX_TOKENS, rank, logprobs=1
    )
    choice = response["choices"][0]
    return {
        "started": started,
        "finished": time.time(),
        "tokens": choice["logprobs"]["tokens"],
        "logprobs": choice["logprobs"]["token_logprobs"],
    }


def _decisions(records: list[dict], window: dict) -> list[dict]:
    """The needle rank's coordination steps while the needle was generating."""
    return [
        r
        for r in records
        if r.get("kind") == "dp"
        and r.get("dp_rank") == NEEDLE_RANK
        and window["started"] <= r["t"] <= window["finished"]
    ]


# The all2all manager each backend must build. Asserted rather than assumed:
# a silent fallback to AllGather+ReduceScatter would not accept microbatching at
# all, so the arm would pass while testing nothing.
_EXPECTED_MANAGER = {
    "deepep_high_throughput": "DeepEPHTAll2AllManager",
    "deepep_low_latency": "DeepEPLLAll2AllManager",
}


@pytest.fixture(params=sorted(_EXPECTED_MANAGER))
def dbo_server(request, tmp_path, enable_batch_invariant_mode):
    """A DP=4 + EP server with DBO enabled, once per all2all backend.

    Skipped without the `deep_ep` package: the engine never comes up and the
    server sits out its whole `max_wait_seconds`, so an unequipped machine
    burns half an hour per arm to produce an error where a skip is honest. A
    DeepEP that lacks the low-latency ordering fix (see the module docstring)
    times out the same way, and cannot be detected by import.

    Function scoped and explicitly dependent on the autouse
    `enable_batch_invariant_mode` fixture: a module-scoped server is built
    before that fixture runs, so it would launch with VLLM_BATCH_INVARIANT
    unset while this process believed it set; the `modes` assertion catches
    that.
    """
    if not has_deep_ep():
        pytest.skip("requires the deep_ep package")

    log_prefix = str(tmp_path / "ubatch")

    args = [
        "--data-parallel-size",
        str(DP_SIZE),
        "--data-parallel-size-local",
        str(DP_SIZE),
        "--enable-expert-parallel",
        # Microbatching is rejected on allgather_reducescatter.
        "--all2all-backend",
        request.param,
        "--enable-dbo",
        "--dbo-decode-token-threshold",
        str(DBO_DECODE_THRESHOLD),
        "--dbo-prefill-token-threshold",
        str(DBO_PREFILL_THRESHOLD),
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "128",
        # The needle's prefill should be recomputed every time, so the
        # comparison covers it and not just the decodes.
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        os.getenv("VLLM_EP_TEST_GPU_MEMORY_UTILIZATION", "0.30"),
    ]
    env = instrumented_server_env(tmp_path, _INSTRUMENTATION, UBATCH_LOG=log_prefix)
    server = RemoteOpenAIServer(
        MODEL, args, env_dict=env, seed=20240919, max_wait_seconds=1800
    )
    with server:
        yield server, log_prefix, _EXPECTED_MANAGER[request.param]


def test_microbatched_needle_is_invariant_to_batch_composition(dbo_server):
    """The needle must not move when DBO cuts its forward pass in two."""
    server, log_prefix, expected_manager = dbo_server
    url = server.url_for("v1/completions")
    peers = [r for r in range(DP_SIZE) if r != NEEDLE_RANK]

    # Discarded: keeps first-request state out of the comparison.
    _needle(url)

    conditions: dict[str, dict] = {}
    # `own_32` and `own_32_peers_32` put the same load on the needle's own
    # rank and differ only in whether the peers are busy, which is what
    # decides whether the proposed split actually happens.
    plan = [
        ("solo", 0, [NEEDLE_RANK], 0),
        ("own_32", 32, [NEEDLE_RANK], 1),
        ("own_32_peers_32", 32 + 3 * 32, list(range(DP_SIZE)), 2),
        ("peers_32", 3 * 32, peers, 3),
        ("all_48", 48, list(range(DP_SIZE)), 4),
        ("solo_again", 0, [NEEDLE_RANK], 0),
    ]
    for label, concurrency, ranks, seed in plan:
        with _load(url, ranks, concurrency, seed) as load:
            conditions[label] = _needle(url)
        # Checked per condition: a peer throwing HTTP errors would otherwise
        # burn the remaining conditions before saying so.
        load.assert_ran_cleanly(f"{label} companions")

    records = read_records(log_prefix)

    env_records = [r for r in records if r.get("kind") == "env"]
    assert_server_ran_this_tree(
        {r["vllm_file"] for r in env_records},
        {r["batch_invariant"] for r in env_records},
    )
    managers = {r["cls"] for r in records if r.get("kind") == "manager"}
    assert managers == {expected_manager}, (
        f"the workers built {managers or 'no'} all2all manager(s), not "
        f"{expected_manager}; a fallback to AllGather+ReduceScatter would not "
        "accept microbatching at all."
    )

    # Vacuity. A run in which no forward pass was ever split, or in which the
    # split never varied, compares a batch shape against itself.
    slices = [r for r in records if r.get("kind") == "slices"]
    assert slices, (
        "maybe_create_ubatch_slices never returned slices, so no forward pass "
        "was microbatched anywhere in this run and the comparison below says "
        "nothing about microbatching."
    )
    ubatched = {
        label: sum(1 for r in _decisions(records, out) if r["ubatched"])
        for label, out in conditions.items()
    }
    proposed = {
        label: sum(1 for r in _decisions(records, out) if r["proposed"])
        for label, out in conditions.items()
    }
    assert any(ubatched.values()) and not all(ubatched.values()), (
        f"the needle rank was microbatched on {ubatched} steps per condition. "
        "The verdict needs conditions on both sides of the split -- all or "
        "nothing means its forward pass had the same shape throughout. "
        f"Proposals were {proposed}."
    )
    # The asymmetry that makes this test about the collective decision rather
    # than about the local batch: the same local proposal, cancelled or not by
    # the peers.
    assert proposed["own_32"] > 0 and ubatched["own_32"] == 0, (
        f"expected the needle rank to propose microbatching with 32 companions "
        f"and be overruled by its idle peers, but it proposed "
        f"{proposed['own_32']} and was microbatched {ubatched['own_32']} times."
    )
    assert ubatched["own_32_peers_32"] > 0, (
        f"expected the same local load to be microbatched once the peers were "
        f"busy too, but the needle rank was microbatched "
        f"{ubatched['own_32_peers_32']} times."
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
                f"(first at {moved[0]}), max |delta| {delta:.3e}, with the "
                f"needle rank microbatched on {ubatched[label]} steps"
            )
    assert not failures, (
        "the needle's logprobs depend on whether its forward pass was split:\n  "
        + "\n  ".join(failures)
    )
