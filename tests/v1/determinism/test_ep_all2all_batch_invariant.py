# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A token's logprobs must not depend on the batch its all2all backend carried.

`--all2all-backend` selects the dispatch/combine that moves tokens between EP
ranks. The combine sums each token's expert contributions across ranks, so its
reduction order is a batch-composition dependency in exactly the way this suite
cares about, and it is a different surface per backend: DeepEP's NVSHMEM
kernels, MoRI's IntraNode kernels and the default AllGather+ReduceScatter are
three separate reductions. This module covers the ROCm-native ones.

**EP=4, not EP=2.** A 2-rank combine is a sum of two terms and is order
independent whatever the implementation does, so an EP=2 test passes against a
batch-variant combine. Four ranks is the smallest size that measures anything,
which is why this module asks for four GPUs.

Measured on 8x gfx950, OLMoE-1B-7B bf16, DP=4/TP=1 (EP=4), TRITON_ATTN,
cudagraphs on, needle pinned to DP rank 2:

- **deepep_high_throughput**: brought up under the mode; the needle's 32
  logprobs were bitwise identical between an idle server and one saturated on
  all four ranks, while its own rank's padded token count went 40 -> 367. With
  the mode off the same comparison moved 31 of 32 positions (max |delta|
  3.1e-2), so the metric is not blind.
- **deepep_low_latency**: *cannot* be brought up under the mode. It forces
  `FusedMoEActivationFormat.BatchedExperts`
  (`FusedMoEParallelConfig.use_batched_activation_format`), whose only experts
  class is `BatchedTritonExperts`, which does not override
  `_supports_batch_invariance()` and so inherits `False` from
  `FusedMoEExperts`. The oracle then finds no candidate and raises
  `NotImplementedError: No Unquantized MoE backend supports the deployment
  configuration`. With the mode off it runs and is demonstrably variant (31 of
  32 positions, max |delta| 3.5e-2), so this is a real gap and not an absent
  feature.
- **mori_high_throughput / mori_low_latency**: *cannot* be brought up under the
  mode either, for a different reason. `FusedMoEConfig.__post_init__` asserts
  `rocm_aiter_fmoe_enabled` for MoRI, and setting `VLLM_ROCM_USE_AITER` at all
  makes both MoE oracles commit to the AITER backend with no fallback
  (`oracle/unquantized.py`, `oracle/fp8.py`: `return _return_or_raise(AITER,
  ...)`). `AiterExperts` also inherits `_supports_batch_invariance() == False`,
  so startup dies with "Unquantized MoE backend ROCm AITER does not support the
  deployment configuration since kernel does not support batch invariance".
  Passing `--moe-backend triton` gets the engine up -- and MoRI's IntraNode
  combine was in fact bitwise invariant there (0 of 32 positions moved, with a
  BI=0 control that moved 32 of 32 and the needle rank's padding going 40 ->
  367) -- but that configuration is not usable: its output is degenerate. That
  is not MoRI's doing. `RoutedExperts.expert_map` hands the experts
  `self.expert_mask` instead of `self._expert_map` whenever
  `rocm_aiter_fmoe_enabled`, which is the *environment variable* and not the
  MoE backend the oracle picked, so `--moe-backend triton` with
  `VLLM_ROCM_USE_AITER_MOE=1` feeds AITER's 0/1 mask to Triton's local-index
  expert map. It reproduces on DeepEP with the same override (degenerate and
  batch variant both with the mode on and off) and disappears at DP=1, where
  there is no expert map. Since MoRI *requires* the AITER env, there is no
  configuration in which it is both correct and invariant on this stack.

  The two single-node MoRI variants are also not two independent tests: both
  were observed selecting `EpDispatchCombineKernelType.IntraNode`
  (`MoriAll2AllManager._make_all2all_kwargs` branches on `self.internode`, not
  on the backend literal).

Only `deepep_high_throughput` is asserted here, because it is the only one that
can currently run under the mode. The others are recorded above rather than
skipped silently so that a change which makes them admissible is noticed.

Why a server rather than the offline `LLM` API: all2all kernels need
`dp_size > 1` (`FusedMoEParallelConfig.use_all2all_kernels`), and `LLM()`
rejects `data_parallel_size > 1` as unsupported for single-process usage. The
per-rank pin is the `X-data-parallel-rank` header, honoured by
`DPLBAsyncMPClient.get_core_engine_for_request`.

The vacuity guard reads the padded token count out of
`dp_utils._synchronize_dp_ranks` through a `sitecustomize` on the server's
PYTHONPATH. It cannot be read from the all2all entry points instead: once a
decode shape is captured those stop executing from Python on replay, so their
call counts go quiet during exactly the steps being measured.
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
DP = 4
# Not rank 0: it is the DP coordinator and the first rank of every group, so a
# verdict keyed on it alone would not generalise.
NEEDLE_RANK = 2
NEEDLE_MAX_TOKENS = 32
LOAD_CONCURRENCY = int(os.getenv("VLLM_EP_LOAD_CONCURRENCY", "24"))
LOAD_RAMP_SECONDS = float(os.getenv("VLLM_EP_LOAD_RAMP_SECONDS", "12"))
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
"""Log the mode as each process sees it, and every DP coordination decision."""
import os
import sys

_LOG = os.environ.get("EP_A2A_LOG")


def _emit(**record):
    if not _LOG:
        return
    try:
        import json
        import time

        record.setdefault("t", time.time())
        with open(f"{_LOG}.{os.getpid()}", "a") as f:
            f.write(json.dumps(record, default=repr) + "\\n")
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[ep-a2a] emit failed: {e}\\n")


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


def _patch_all2all(module):
    """Record which All2AllManager was constructed.

    A silent fallback to AllGather+ReduceScatter would look perfectly clean and
    would prove nothing about DeepEP, so the test asserts on this.
    """
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
                _emit(event="manager", cls=name)

            return wrapper

        cls.__init__ = make(name, original_init)


_PATCHES = {
    "vllm.v1.worker.dp_utils": _patch_dp_utils,
    "vllm.distributed.device_communicators.all2all": _patch_all2all,
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
    """Keeps `concurrency` unrelated requests in flight across every DP rank."""

    def __init__(self, url: str, concurrency: int):
        self.url, self.concurrency = url, concurrency
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.errors: list[str] = []

    def _run(self, seed: int) -> None:
        rng = random.Random(seed)
        while not self._stop.is_set():
            prompt = " ".join(
                str(rng.randint(0, 99999)) for _ in range(rng.choice([16, 48, 96, 160]))
            )
            try:
                _completion(self.url, prompt, rng.choice([64, 128, 256]), seed % DP)
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
            thread.join(timeout=240)
        # Let the queues drain so the next condition starts from idle.
        time.sleep(6.0)


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
def deepep_ht_server(tmp_path, enable_batch_invariant_mode):
    """A DP=4 + EP server on the DeepEP high-throughput all2all.

    Function-scoped and explicitly dependent on the autouse
    `enable_batch_invariant_mode` fixture. A module-scoped server is built
    before that function-scoped fixture runs, so it would launch with
    VLLM_BATCH_INVARIANT unset while this process believes it is set -- and
    a copy of this file that overrides the fixture would not actually flip
    the arm. The `modes` assertion below catches that, and did.

    `VLLM_ROCM_USE_AITER` is deliberately left *unset* rather than set to 0:
    the MoE oracles treat the variable being set at all as a request to commit
    to the AITER backend, so exporting it either way changes kernel selection.
    """
    (tmp_path / "sitecustomize.py").write_text(_INSTRUMENTATION)
    log_prefix = str(tmp_path / "ep_a2a")

    args = [
        "--data-parallel-size",
        str(DP),
        "--data-parallel-size-local",
        str(DP),
        "--enable-expert-parallel",
        "--all2all-backend",
        "deepep_high_throughput",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "64",
        # The needle's prefill should be recomputed every time, so the
        # comparison covers it and not just the decodes.
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        os.getenv("VLLM_EP_TEST_GPU_MEMORY_UTILIZATION", "0.55"),
    ]
    # RemoteOpenAIServer launches the `vllm` console script off PATH rather
    # than `sys.executable -m`, so it does not inherit this process's
    # interpreter. On a machine with an unrelated vLLM checkout wired in
    # through a user-site `.pth`, that is a *different tree* and the test would
    # measure a build nobody is asking about while passing. The path entry
    # makes the common case work; the `vllm_file` assertion below is what keeps
    # it honest.
    repo_root = str(Path(vllm.__file__).resolve().parent.parent)
    env = {
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), repo_root, os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
        "EP_A2A_LOG": log_prefix,
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
    }
    with RemoteOpenAIServer(MODEL, args, env_dict=env, max_wait_seconds=1800) as server:
        yield server, log_prefix


def test_deepep_high_throughput_combine_does_not_see_the_batch(deepep_ht_server):
    """The needle's logprobs must not move when the rest of the server does."""
    server, log_prefix = deepep_ht_server
    url = server.url_for("v1/completions")

    # Discarded: keeps first-request state out of the comparison.
    _needle(url)

    with _Load(url, 0):
        alone = _needle(url)
    with _Load(url, LOAD_CONCURRENCY) as load:
        loaded = _needle(url)
    assert not load.errors, (
        f"the background load did not run cleanly: {load.errors[:3]}"
    )

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

    managers = {r["cls"] for r in records if r.get("event") == "manager"}
    assert managers == {"DeepEPHTAll2AllManager"}, (
        f"the workers built {managers or 'no'} all2all manager(s). A fallback "
        "to AllGather+ReduceScatter would pass this test while proving nothing "
        "about DeepEP."
    )

    # Vacuity: if the load never changed the needle rank's shapes then it ran
    # the same forward twice and the comparison below is empty.
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
        "Raise VLLM_EP_LOAD_CONCURRENCY."
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
        f"traffic was in flight: its own rank's padded token count went from "
        f"{max(alone_pads)} to {max(loaded_pads)} while its request was "
        f"byte-identical. max |delta| = "
        f"{max(abs(alone['logprobs'][i] - loaded['logprobs'][i]) for i in moved)}"
    )
