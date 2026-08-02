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

Measured on 4x gfx950, OLMoE-1B-7B bf16, DP=4/TP=1 (EP=4), TRITON_ATTN,
cudagraphs on, needle pinned to DP rank 2, 64 generated logprobs compared
bitwise against the same prompt run solo, with the DBO thresholds lowered to 8
(decode) and 32 (prefill) so the split is reachable at test scale:

                        needle rank 2      mode on   mode off (max |delta|)
  condition             proposed/ubatched
  own_32                    65 / 0            0/64     64/64  (5.5e-2)
  own_32_peers_32           65 / 64           0/64     64/64  (6.4e-2)
  peers_32                   1 / 1            0/64     64/64  (1.1e-1)
  all_48                    65 / 64           0/64     64/64  (6.1e-2)
  solo_again                 1 / 0            0/64      0/64
  other_rank_solo            0 / 0            0/64      0/64

256 moved positions in the mode-off arm, none in the mode-on arm, with the two
arms taking *identical* microbatching decisions -- the middle column is the same
in both.

The middle column is the point of the test. `own_32` and `own_32_peers_32` put
the same 32 companions on the needle's own rank; the needle rank proposed
microbatching on all 65 of its coordination steps in both. In the first, the
peers were idle, they proposed `False`, and the split was cancelled on every
step. In the second the peers were loaded and it went through on 64 of 65. The
needle's forward pass was cut in two, or not, according to what three other
replicas were doing, and its logprobs did not move either way.

`peers_32` is the sharper version of the same thing and needs the asymmetry
spelled out: a rank holding a single decode has fewer tokens than
`dbo_decode_token_threshold` and so can never *propose* a split. It can only
have one forced on it during a step it does propose -- here its 33-token prefill
-- which is why that row proposes once and ubatches once, while `solo` proposes
once and ubatches zero times. Same local batch, opposite outcome, decided
entirely off-rank.

A run in which no forward pass was ever actually split would prove nothing, so
the assertions below are on the observed decisions rather than on the load
having been applied: the needle rank must have been microbatched in some
conditions and not in others, and `maybe_create_ubatch_slices` must have
returned real slices.

Not covered: `deepep_low_latency` and `nixl_ep`, the other two backends
microbatching accepts. The first is no longer *refused* under the mode --
`BatchedTritonExperts` now declares batch invariance for its unquantized path
and `test_ep_all2all_batch_invariant.py` asserts DeepEP LL end to end -- so it
is now reachable here and simply untested, which is a weaker gap than it was
and worth closing. The second still needs its kernels. `--ubatch-size` > 2 is
also untested; only DBO's two-way split is exercised here.
"""

import json
import os
import random
import threading
import time
from pathlib import Path

import pytest
import requests
from utils import skip_if_not_cuda_alike

import vllm
import vllm.envs as envs
from tests.utils import RemoteOpenAIServer, large_gpu_mark, multi_gpu_marks

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

# Written to the server's PYTHONPATH as sitecustomize.py so it loads in the API
# server, every engine core and every worker, including spawned ones.
_INSTRUMENTATION = '''
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

    def wrapper(
        num_tokens_unpadded,
        num_tokens_padded,
        should_attempt_ubatching,
        cudagraph_mode,
        parallel_config,
    ):
        out = original(
            num_tokens_unpadded,
            num_tokens_padded,
            should_attempt_ubatching,
            cudagraph_mode,
            parallel_config,
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

if _LOG:
    import importlib.abc
    import importlib.util

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            patcher = _TARGETS.get(name)
            if patcher is None:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            original_exec = spec.loader.exec_module

            def exec_module(module, _exec=original_exec, _patch=patcher):
                _exec(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
'''


def _completion(url, prompt, max_tokens, rank, logprobs=None, timeout=900):
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 20240919,
    }
    if logprobs is not None:
        body["logprobs"] = logprobs
    response = requests.post(
        url, json=body, headers={"X-data-parallel-rank": str(rank)}, timeout=timeout
    )
    response.raise_for_status()
    return response.json()


class _Load:
    """Keeps `concurrency` unrelated requests in flight, spread over `ranks`."""

    def __init__(self, url, ranks, concurrency, seed=0):
        self.url, self.ranks, self.concurrency = url, ranks, concurrency
        self.seed = seed
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.errors: list[str] = []

    def _run(self, index: int) -> None:
        rng = random.Random(self.seed * 1000 + index)
        rank = self.ranks[index % len(self.ranks)]
        while not self._stop.is_set():
            prompt = " ".join(
                str(rng.randint(0, 99999)) for _ in range(rng.randint(32, 96))
            )
            try:
                _completion(self.url, prompt, 384, rank)
            except Exception as e:
                self.errors.append(repr(e))
                time.sleep(0.5)

    def __enter__(self) -> "_Load":
        for i in range(self.concurrency):
            thread = threading.Thread(target=self._run, args=(i,), daemon=True)
            thread.start()
            self._threads.append(thread)
        if self.concurrency:
            time.sleep(RAMP_SECONDS)
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=300)
        # Let the queues drain so the next condition starts from idle.
        time.sleep(DRAIN_SECONDS)


def _needle(url, rank=NEEDLE_RANK) -> dict:
    started = time.time()
    response = _completion(url, NEEDLE_PROMPT, NEEDLE_MAX_TOKENS, rank, logprobs=1)
    choice = response["choices"][0]
    return {
        "started": started,
        "finished": time.time(),
        "tokens": choice["logprobs"]["tokens"],
        "logprobs": choice["logprobs"]["token_logprobs"],
    }


def _records(log_prefix: str) -> list[dict]:
    directory, prefix = os.path.split(log_prefix)
    out: list[dict] = []
    for name in os.listdir(directory):
        if name.startswith(prefix + "."):
            with open(os.path.join(directory, name)) as f:
                out.extend(json.loads(line) for line in f if line.strip())
    return out


def _decisions(records: list[dict], window: dict) -> list[dict]:
    """The needle rank's coordination steps while the needle was generating."""
    return [
        r
        for r in records
        if r.get("kind") == "dp"
        and r.get("dp_rank") == NEEDLE_RANK
        and window["started"] <= r["t"] <= window["finished"]
    ]


@pytest.fixture
def dbo_server(tmp_path, enable_batch_invariant_mode):
    """A DP=4 + EP server on DeepEP high throughput with DBO enabled.

    Function scoped and explicitly dependent on the autouse
    `enable_batch_invariant_mode` fixture: a module-scoped server is built
    before that fixture runs, so it would launch with VLLM_BATCH_INVARIANT
    unset while this process believed it set. The `modes` assertion catches
    that, and has.
    """
    (tmp_path / "sitecustomize.py").write_text(_INSTRUMENTATION)
    log_prefix = str(tmp_path / "ubatch")

    args = [
        "--data-parallel-size",
        str(DP_SIZE),
        "--data-parallel-size-local",
        str(DP_SIZE),
        "--enable-expert-parallel",
        # Microbatching is rejected on allgather_reducescatter.
        "--all2all-backend",
        "deepep_high_throughput",
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
    # RemoteOpenAIServer launches the `vllm` console script off PATH rather than
    # `sys.executable -m`, so the server does not inherit this process's
    # interpreter. Where an unrelated vLLM is wired in through a user-site
    # `.pth` that resolves to a different tree. The path entry makes the common
    # case work; the `vllm_file` assertion is what keeps it honest.
    repo_root = str(Path(vllm.__file__).resolve().parent.parent)
    env = {
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), repo_root, os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
        "UBATCH_LOG": log_prefix,
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
    }
    server = RemoteOpenAIServer(
        MODEL, args, env_dict=env, seed=20240919, max_wait_seconds=1800
    )
    with server:
        yield server, log_prefix


def test_microbatched_needle_is_invariant_to_batch_composition(dbo_server):
    """The needle must not move when DBO cuts its forward pass in two."""
    server, log_prefix = dbo_server
    url = server.url_for("v1/completions")
    peers = [r for r in range(DP_SIZE) if r != NEEDLE_RANK]

    # Discarded: keeps first-request state out of the comparison.
    _needle(url)

    conditions: dict[str, dict] = {}
    loads: dict[str, _Load] = {}
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
        with _Load(url, ranks, concurrency, seed) as load:
            conditions[label] = _needle(url)
        loads[label] = load

    conditions["other_rank_solo"] = _needle(url, (NEEDLE_RANK + 1) % DP_SIZE)

    for label, load in loads.items():
        assert not load.errors, f"{label} companions failed: {load.errors[:3]}"

    records = _records(log_prefix)

    # The server is a separate process launched off PATH, so it can silently be
    # a different vLLM tree or a different mode than the one under test.
    served = {r["vllm_file"] for r in records if r.get("kind") == "env"}
    assert served == {vllm.__file__}, (
        f"the server imported vLLM from {served}, but this test process is "
        f"{vllm.__file__}; nothing it reports is evidence about this tree."
    )
    modes = {r["batch_invariant"] for r in records if r.get("kind") == "env"}
    assert modes == {envs.VLLM_BATCH_INVARIANT}, (
        f"the server's effective VLLM_BATCH_INVARIANT is {modes}, but this "
        f"process has {envs.VLLM_BATCH_INVARIANT}; the two arms of this test "
        "are not running the mode they claim to."
    )
    managers = {r["cls"] for r in records if r.get("kind") == "manager"}
    assert managers == {"DeepEPHTAll2AllManager"}, (
        f"the workers built {managers or 'no'} all2all manager(s); a fallback "
        "to AllGather+ReduceScatter would not accept microbatching at all."
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
