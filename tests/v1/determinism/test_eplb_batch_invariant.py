# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A token's logprobs must not move because EPLB's *other* traffic changed.

EPLB relocates physical experts between EP ranks to balance observed load.
Two different things follow from that, and this module asserts only one of
them.

**Within-batch invariance -- asserted here.** With a placement held fixed,
adding unrelated companions to the needle's own rank must not change its
output. This is the mode's contract, and it is the property that
`num_redundant_experts > 0` breaks: replicas of one logical expert are chosen
by hashing the token's row index in the current forward, so companions move
the needle to a different rank's GEMM. That configuration is refused at config
construction (`VllmConfig.__post_init__`) and the refusal is covered by
`test_eplb_config_batch_invariant.py`; here `num_redundant_experts` is 0,
where routing is a pure function of the token.

**Cross-request history dependence -- documented, deliberately not asserted.**
Rearrangement follows the load EPLB has *already* seen, so two byte-identical
solo requests either side of a rearrangement legitimately differ. Measured on
this configuration under the mode: 64 of 64 logprobs moved across a
rearrangement, max |delta| 1.67, with different sampled tokens; the same run
with `--enable-eplb` removed moved 0 of 64 on all seven comparisons. That is a
dependence on preceding traffic rather than on the batch, so it is outside
this mode's contract -- `VllmConfig` warns about it rather than refusing -- and
an assertion that two solo requests agree would fail for a reason this suite
does not gate. The comparison below is therefore made *inside a window with no
committed rearrangement*, which is what leaves batch composition as the only
thing that differs between its two arms.

**The vacuity problem, and both halves of the guard.** A converged EPLB server
that never rearranges again is indistinguishable from an ordinary EP server,
and would report invariance while measuring nothing about EPLB. So the test
asserts (1) that rearrangement really fires in this deployment -- by the end of
the warm-up at least one `EplbState.rearrange` call has actually changed
`physical_to_logical_map` -- and (2) that none fires inside the comparison
window, retaking the window if one does. Both are needed: (1) alone would let a
rearrangement contaminate the verdict, (2) alone is satisfied by a server where
EPLB is inert.

`step_interval` is pinned to 40 rather than left at its default of 3000.
Rearrangement is counted in forward steps, and nothing this test can afford to
send comes near 3000 of them, so at the default guard (1) fires and correctly
refuses the verdict.

What makes a still window exist at all is `rearrange`'s short-circuit when the
proposed mapping improves rank imbalance by less than 5%, so both remaining
knobs are about keeping that short-circuit reachable. `window_size` is left at
its 1000-step default: at the 20 used for the original measurements the load
estimate is 20 steps of noise, a 5% "improvement" is nearly always available,
and placement churns indefinitely instead of settling -- measured, with
committed rearrangements still arriving every few seconds after 80 seconds of
load, and the window guard firing because of it. Note also that the
short-circuit is ROCm-only; on CUDA every proposed rearrangement is applied
whether or not it improves anything, so placement may never settle there and
the guard will say so rather than pass.

**`use_async` is pinned to False, and that is not the default.** With
`EPLBConfig.use_async` left at its default of True the map is unchanged when
`rearrange` returns -- the transfer is in flight, and `step` commits one layer
at a time as it lands -- and, more decisively, the short-circuit sits inside
`if not self.is_async or is_profile`, so async EPLB applies every proposal it
computes. Measured here: 1327 committed layer moves per rank over 145 seconds,
arriving every 0.1 s with a maximum gap of 8 s and no quiet interval anywhere,
against 34 whole-map rearrangements and 89 skipped proposals in the sync run.
Placement is never still under async, all three measurement windows churned
(29k slots each), and the test refuses the verdict rather than reporting a
pass. So this module measures the non-default configuration, because it is the
only one in which the property is observable: async is not shown to be batch
variant, it is shown to be unmeasurable this way. It is also nondeterministic
in a way sync EPLB is not, since when a swap lands depends on transfer
progress rather than on a step count. Reproduce with
`VLLM_EPLB_TEST_USE_ASYNC=1`.

Measured on 4x gfx950, DeepSeek-V2-Lite, DP=4 x EP=4 (`allgather_reducescatter`,
`TritonExperts`), needle pinned to DP rank 2, 64 logprobs, mode on: 0 of 64
positions moved with 32 companions on the needle's own rank, its padded batch
going 32 -> 7762 tokens, across a window in which EPLB committed nothing and
the run as a whole committed 10 rearrangements. The same comparison with the
mode off moved 64 of 64, max |delta| 0.137, at 32 -> 64 tokens.

DeepSeek-V2-Lite rather than the OLMoE used elsewhere in this suite because
EPLB needs a model whose MoE layers vLLM builds with an expert map, and a
30 GB checkpoint is the cheapest one to hand. Per rank it measured 14.5 GiB of
weights, 0.5 GiB of peak activation and 2.1 GiB of CUDA graphs, plus a KV
cache that has to hold `max_num_seqs * max_model_len` = 512K tokens at 29.9
KiB each, or 15.3 GiB -- about 32.4 GiB, which at the 0.55 memory fraction
used here needs a 64 GB device.

Why a server rather than the offline `LLM` API, and how the per-rank pin
works: see `test_ep_all2all_batch_invariant.py`, whose conventions this
follows.
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
    large_gpu_mark(min_gb=64),
    *multi_gpu_marks(num_gpus=4),
]

MODEL = os.getenv("VLLM_EPLB_TEST_MODEL", "deepseek-ai/DeepSeek-V2-Lite")
DP = 4
# Not rank 0: it is the DP coordinator and the first rank of every group, so a
# verdict keyed on it alone would not generalise.
NEEDLE_RANK = 2
NEEDLE_MAX_TOKENS = 64

# Forward steps between rearrangement attempts. See the module docstring for
# why the 3000 default cannot be used.
STEP_INTERVAL = int(os.getenv("VLLM_EPLB_TEST_STEP_INTERVAL", "40"))
# Steps of expert load each attempt averages over, left at its default. A short
# window makes the load estimate noisy enough that a 5% "improvement" is nearly
# always available, and placement then churns indefinitely instead of settling
# -- measured at window_size=20, where committed rearrangements were still
# arriving every few seconds after 80 seconds of load.
WINDOW_SIZE = int(os.getenv("VLLM_EPLB_TEST_WINDOW_SIZE", "1000"))
# Off, against the `EPLBConfig.use_async` default of True. See the module
# docstring: the default is not a configuration in which this property can be
# measured, and this knob exists so that claim can be reproduced.
USE_ASYNC = os.getenv("VLLM_EPLB_TEST_USE_ASYNC", "0") == "1"

# Companions for the measured condition, all on the needle's own rank: that is
# what changes the needle's row index within its forward, and therefore the
# condition a redundant-expert replica hash was measured to fail.
LOAD_CONCURRENCY = int(os.getenv("VLLM_EPLB_LOAD_CONCURRENCY", "32"))
# Spent before the measurement to let placement converge, on every rank.
WARMUP_CONCURRENCY = int(os.getenv("VLLM_EPLB_WARMUP_CONCURRENCY", "48"))
WARMUP_SECONDS = float(os.getenv("VLLM_EPLB_WARMUP_SECONDS", "60"))
# Measurement windows to try before giving up on finding one EPLB held still.
WINDOW_ATTEMPTS = int(os.getenv("VLLM_EPLB_WINDOW_ATTEMPTS", "3"))
LOAD_RAMP_SECONDS = float(os.getenv("VLLM_EPLB_LOAD_RAMP_SECONDS", "12"))
DRAIN_SECONDS = 8.0
# The load must drag the needle rank's padded token count at least this far
# above what it ran alone, otherwise the needle saw identical shapes twice.
MIN_LOADED_PAD = 16

NEEDLE_PROMPT = (
    "Explain, step by step, how a four-stroke internal combustion engine "
    "converts chemical energy in fuel into rotational mechanical work, and "
    "where the main thermodynamic losses occur."
)

# Written to the server's PYTHONPATH as sitecustomize.py so it loads in the API
# server, every engine core and every worker, including spawned ones.
_INSTRUMENTATION = '''
"""Log the mode as each process sees it, the DP shapes, and every rearrange."""
import os
import sys
import threading

_LOG = os.environ.get("EPLB_LOG")
_LOCK = threading.Lock()
_SEEN = set()


def _emit(**record):
    if not _LOG:
        return
    try:
        import json
        import time

        record.setdefault("t", time.time())
        with _LOCK, open(f"{_LOG}.{os.getpid()}", "a") as f:
            f.write(json.dumps(record, default=repr) + "\\n")
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[eplb] emit failed: {e}\\n")


def _patch_dp_utils(module):
    import vllm
    import vllm.envs as envs

    _emit(
        event="env",
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
        _, after_padding, _ = out
        try:
            rank = parallel_config.data_parallel_rank
            _emit(
                event="dp",
                dp_rank=rank,
                padded=None if after_padding is None else int(after_padding[rank]),
            )
        except Exception as e:  # pragma: no cover
            _emit(event="dp_err", err=repr(e))
        return out

    module._synchronize_dp_ranks = wrapper


def _patch_v2_dp_utils(module):
    """The same shape record, for the V2 model runner.

    The V1 runner coordinates through `dp_utils._synchronize_dp_ranks`; the V2
    runner -- which is what this configuration selects, `gpu_worker.py`'s
    "Using V2 Model Runner" -- coordinates through this function instead and
    never calls the other one. Patching only one of them silently yields no
    shape records at all, which is what the guard below would then report.
    """
    original = module.sync_cudagraph_and_dp_padding

    def wrapper(*args, **kwargs):
        out = original(*args, **kwargs)
        try:
            dp_rank = kwargs.get("dp_rank", args[6] if len(args) > 6 else None)
            _, across = out
            _emit(
                event="dp",
                dp_rank=dp_rank,
                padded=None if across is None else int(across[dp_rank]),
            )
        except Exception as e:  # pragma: no cover
            _emit(event="dp_err", err=repr(e))
        return out

    module.sync_cudagraph_and_dp_padding = wrapper


def _patch_moe_config(module):
    """Record the EP width actually built.

    `ep_size` alone would not prove the experts are spread over four ranks,
    but together with `use_ep` and DP=4 it distinguishes a real EP deployment
    from a replicated-expert one, where EPLB would have nothing to move.
    """
    original = module.FusedMoEParallelConfig.make

    def make(*args, **kwargs):
        config = original(*args, **kwargs)
        key = ("cfg", config.ep_size, config.use_ep)
        with _LOCK:
            new = key not in _SEEN
            _SEEN.add(key)
        if new:
            _emit(
                event="moe",
                ep_size=config.ep_size,
                dp_size=config.dp_size,
                tp_size=config.tp_size,
                use_ep=config.use_ep,
                all2all_backend=config.all2all_backend,
            )
        return config

    module.FusedMoEParallelConfig.make = staticmethod(make)


def _patch_eplb(module):
    """Record every rearrange, and whether it actually moved an expert.

    `rearrange` short-circuits when the proposed mapping improves imbalance by
    less than a threshold, so "rearrange was called" is not evidence that
    placement changed. Comparing `physical_to_logical_map` either side of the
    call is.
    """
    cls = getattr(module, "EplbState", None)
    if cls is None:  # pragma: no cover
        _emit(event="eplb_err", err="EplbState missing")
        return

    original = cls.rearrange

    def rearrange(self, *args, **kwargs):
        before = {}
        try:
            for key, state in self.model_states.items():
                before[key] = state.physical_to_logical_map.clone()
        except Exception as e:  # pragma: no cover
            _emit(event="eplb_err", err=f"snapshot: {e!r}")
        out = original(self, *args, **kwargs)
        try:
            moved = 0
            for key, state in self.model_states.items():
                previous = before.get(key)
                if previous is not None:
                    moved += int((state.physical_to_logical_map != previous).sum())
            _emit(
                event="rearrange",
                is_profile=bool(kwargs.get("is_profile", args[0] if args else False)),
                slots_moved=moved,
            )
        except Exception as e:  # pragma: no cover
            _emit(event="eplb_err", err=f"compare: {e!r}")
        return out

    cls.rearrange = rearrange

    original_move = getattr(module, "_move_to_workspace", None)

    def move_to_workspace(*args, **kwargs):
        """Async EPLB's commit point.

        With `use_async=True` the map is untouched when `rearrange` returns:
        the transfer is still in flight, and `step` commits one layer at a
        time as each rank's copy lands. Snapshotting `rearrange` alone would
        report zero placement changes for a server that is rearranging
        constantly, so the liveness guard would fail and the window guard
        would pass -- both wrong.
        """
        model_state = kwargs.get("model_state", args[0] if args else None)
        before = None
        if model_state is not None:
            before = model_state.physical_to_logical_map.clone()
        out = original_move(*args, **kwargs)
        try:
            if before is not None:
                _emit(
                    event="rearrange",
                    is_profile=False,
                    async_commit=True,
                    slots_moved=int(
                        (model_state.physical_to_logical_map != before).sum()
                    ),
                )
        except Exception as e:  # pragma: no cover
            _emit(event="eplb_err", err=f"workspace: {e!r}")
        return out

    if original_move is not None:
        module._move_to_workspace = move_to_workspace

    original_add_model = cls.add_model

    def add_model(self, *args, **kwargs):
        out = original_add_model(self, *args, **kwargs)
        try:
            for state in self.model_states.values():
                _emit(
                    event="eplb_config",
                    physical_experts=list(state.physical_to_logical_map.shape),
                    max_replica_count=int(state.logical_replica_count.max()),
                    step_interval=int(self.expert_rearrangement_step_interval),
                )
        except Exception as e:  # pragma: no cover
            _emit(event="eplb_err", err=f"add_model: {e!r}")
        return out

    cls.add_model = add_model


_PATCHES = {
    "vllm.v1.worker.dp_utils": _patch_dp_utils,
    "vllm.v1.worker.gpu.dp_utils": _patch_v2_dp_utils,
    "vllm.model_executor.layers.fused_moe.config": _patch_moe_config,
    "vllm.distributed.eplb.eplb_state": _patch_eplb,
}


if _LOG:
    import importlib.abc
    import importlib.util

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            fn = _PATCHES.get(name)
            if fn is None:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            original_exec = spec.loader.exec_module

            def exec_module(module, _exec=original_exec, _fn=fn):
                _exec(module)
                _fn(module)

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
    """Keeps `concurrency` unrelated requests in flight across `ranks`."""

    def __init__(self, url: str, concurrency: int, ranks: list[int], seed: int = 0):
        self.url, self.concurrency, self.ranks, self.seed = (
            url,
            concurrency,
            ranks,
            seed,
        )
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
            time.sleep(LOAD_RAMP_SECONDS)
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=300)
        # Let the queues drain so the next condition starts from idle.
        time.sleep(DRAIN_SECONDS)


def _needle(url) -> dict:
    started = time.time()
    response = _completion(
        url, NEEDLE_PROMPT, NEEDLE_MAX_TOKENS, NEEDLE_RANK, logprobs=1
    )
    choice = response["choices"][0]
    return {
        "started": started,
        "finished": time.time(),
        "tokens": choice["logprobs"]["tokens"],
        "logprobs": choice["logprobs"]["token_logprobs"],
    }


def _records(log_prefix: str) -> list[dict]:
    directory, prefix = os.path.split(log_prefix)
    out = []
    for name in os.listdir(directory):
        if not name.startswith(prefix + "."):
            continue
        with open(os.path.join(directory, name)) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def _committed_rearranges(records: list[dict], start: float, end: float) -> list[dict]:
    return [
        r
        for r in records
        if r.get("event") == "rearrange"
        and not r.get("is_profile")
        and r.get("slots_moved")
        and start <= r["t"] <= end
    ]


def _needle_rank_pads(records: list[dict], needle: dict) -> list[int]:
    return [
        r["padded"]
        for r in records
        if r.get("event") == "dp"
        and r.get("dp_rank") == NEEDLE_RANK
        and r.get("padded") is not None
        and needle["started"] <= r["t"] <= needle["finished"]
    ]


@pytest.fixture
def eplb_server(tmp_path, enable_batch_invariant_mode):
    """A DP=4 + EP server running EPLB with no redundant experts.

    Function-scoped and explicitly dependent on the autouse
    `enable_batch_invariant_mode` fixture. A module-scoped server is built
    before that function-scoped fixture runs, so it would launch with
    VLLM_BATCH_INVARIANT unset while this process believes it is set. The
    `modes` assertion below catches that.
    """
    (tmp_path / "sitecustomize.py").write_text(_INSTRUMENTATION)
    log_prefix = str(tmp_path / "eplb")

    eplb_config = {
        "window_size": WINDOW_SIZE,
        "step_interval": STEP_INTERVAL,
        "num_redundant_experts": 0,
        "use_async": USE_ASYNC,
    }
    args = [
        "--data-parallel-size",
        str(DP),
        "--data-parallel-size-local",
        str(DP),
        "--enable-expert-parallel",
        "--enable-eplb",
        "--eplb-config",
        json.dumps(eplb_config),
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "128",
        # The needle's prefill should be recomputed every time, so the
        # comparison covers it and not just the decodes.
        "--no-enable-prefix-caching",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        os.getenv("VLLM_EPLB_TEST_GPU_MEMORY_UTILIZATION", "0.55"),
    ]
    # RemoteOpenAIServer launches the `vllm` console script off PATH rather
    # than `sys.executable -m`, so it does not inherit this process's
    # interpreter. The path entry makes the common case work; the `vllm_file`
    # assertion below is what keeps it honest.
    repo_root = str(Path(vllm.__file__).resolve().parent.parent)
    env = {
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), repo_root, os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
        "EPLB_LOG": log_prefix,
    }
    with RemoteOpenAIServer(MODEL, args, env_dict=env, max_wait_seconds=1800) as server:
        yield server, log_prefix


def test_eplb_needle_is_invariant_to_batch_composition(eplb_server):
    """With placement held still, companions must not move the needle."""
    server, log_prefix = eplb_server
    url = server.url_for("v1/completions")

    # Discarded: keeps first-request state out of the comparison.
    _needle(url)
    # Spend EPLB's convergence before measuring: a rearrangement inside the
    # window below would be history dependence, which this test does not
    # assert on, and the guard would refuse the verdict.
    with _Load(url, WARMUP_CONCURRENCY, list(range(DP)), seed=1) as warmup:
        time.sleep(WARMUP_SECONDS)
    assert not warmup.errors, f"the warm-up load failed: {warmup.errors[:3]}"

    records = _records(log_prefix)

    served = {r["vllm_file"] for r in records if r.get("event") == "env"}
    assert served == {vllm.__file__}, (
        f"the server imported vLLM from {served}, but this test process is "
        f"{vllm.__file__}; nothing it reports is evidence about this tree."
    )

    modes = {r["batch_invariant"] for r in records if r.get("event") == "env"}
    assert modes == {envs.VLLM_BATCH_INVARIANT}, (
        f"the server's effective VLLM_BATCH_INVARIANT is {modes}, but this "
        f"process has {envs.VLLM_BATCH_INVARIANT}; the two arms of this test "
        "are not running the mode they claim to"
    )

    moe = {(r["ep_size"], r["use_ep"]) for r in records if r.get("event") == "moe"}
    assert moe == {(DP, True)}, (
        f"the workers built MoE layers with (ep_size, use_ep) {moe}, not "
        f"{{({DP}, True)}}. Without expert parallelism EPLB has nothing to "
        "move between ranks."
    )

    eplb_configs = [r for r in records if r.get("event") == "eplb_config"]
    assert eplb_configs, "no EplbState was built, so EPLB is not running at all"
    replica_counts = {r["max_replica_count"] for r in eplb_configs}
    assert replica_counts == {1}, (
        f"physical experts have replica counts {replica_counts}; this test "
        "covers num_redundant_experts=0, and anything else is refused by "
        "VllmConfig under the mode."
    )
    intervals = {r["step_interval"] for r in eplb_configs}
    assert intervals == {STEP_INTERVAL}, (
        f"EPLB is rearranging every {intervals} steps, not {STEP_INTERVAL}"
    )

    # Vacuity, half one: a server whose placement never changes is an ordinary
    # EP server, and a green run against it says nothing about EPLB. Checked
    # after the warm-up, which is where rearrangement is supposed to happen.
    committed = _committed_rearranges(records, 0.0, time.time())
    assert committed, (
        f"EPLB never moved an expert during the warm-up "
        f"({len([r for r in records if r.get('event') == 'rearrange'])} "
        "rearrange calls, none of which changed physical_to_logical_map), so "
        "this test would measure a static placement and prove nothing about "
        "EPLB. Raise VLLM_EPLB_WARMUP_SECONDS or lower "
        "VLLM_EPLB_TEST_STEP_INTERVAL."
    )

    # Vacuity, half two: the two arms must differ only in batch composition. A
    # rearrangement between them is history dependence -- real, expected and
    # not what this test asserts -- so that window's verdict is not readable
    # and another window is taken instead. Placement settles as EPLB's load
    # estimate fills, so the retries are not a coin flip being reflipped.
    alone = loaded = None
    churn = []
    for attempt in range(WINDOW_ATTEMPTS):
        candidate_alone = _needle(url)
        with _Load(url, LOAD_CONCURRENCY, [NEEDLE_RANK], seed=2 + attempt) as load:
            candidate_loaded = _needle(url)
        assert not load.errors, (
            f"the background load did not run cleanly: {load.errors[:3]}"
        )
        records = _records(log_prefix)
        during = _committed_rearranges(
            records, candidate_alone["started"], candidate_loaded["finished"]
        )
        churn.append(sum(r["slots_moved"] for r in during))
        if not during:
            alone, loaded = candidate_alone, candidate_loaded
            break
    assert alone is not None and loaded is not None, (
        f"EPLB moved experts in all {WINDOW_ATTEMPTS} measurement windows "
        f"(slots moved per window: {churn}), so no two needles ran against the "
        "same placement and any difference between them could not be "
        "attributed to batch composition. Lengthen the warm-up, or raise "
        "VLLM_EPLB_TEST_WINDOW_SIZE so the load estimate driving rearrangement "
        "stops churning."
    )

    # Vacuity, half three: if the load never changed the needle rank's shapes
    # then it ran the same forward twice and the comparison is empty.
    alone_pads = _needle_rank_pads(records, alone)
    loaded_pads = _needle_rank_pads(records, loaded)
    assert alone_pads and loaded_pads, (
        f"no DP coordination was recorded on rank {NEEDLE_RANK} while the "
        f"needle ran (alone: {len(alone_pads)} steps, loaded: "
        f"{len(loaded_pads)}). Either the instrumentation did not load or the "
        "ranks are not coordinating."
    )
    assert max(loaded_pads) >= MIN_LOADED_PAD and max(loaded_pads) > max(alone_pads), (
        f"rank {NEEDLE_RANK}'s padded token count did not move (alone: max "
        f"{max(alone_pads)}, loaded: max {max(loaded_pads)}), so it dispatched "
        "the same shapes in both conditions and this verdict would be vacuous. "
        "Raise VLLM_EPLB_LOAD_CONCURRENCY."
    )

    assert loaded["tokens"] == alone["tokens"], (
        f"the needle sampled different tokens once the server was busy: "
        f"{alone['tokens']} vs {loaded['tokens']}"
    )
    moved = [
        i
        for i, (a, b) in enumerate(zip(alone["logprobs"], loaded["logprobs"]))
        if a != b
    ]
    assert not moved, (
        f"the needle's logprobs changed at positions {moved} because unrelated "
        f"traffic shared its rank: its own rank's padded token count went from "
        f"{max(alone_pads)} to {max(loaded_pads)} while its request was "
        f"byte-identical and EPLB moved no expert in between. max |delta| = "
        f"{max(abs(alone['logprobs'][i] - loaded['logprobs'][i]) for i in moved)}"
    )
