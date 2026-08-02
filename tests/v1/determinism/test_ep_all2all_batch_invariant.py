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
- **mori_high_throughput**: brought up under the mode and asserted here, but
  only after two things were fixed. `RoutedExperts.expert_map` used to hand the
  experts AITER's 0/1 mask whenever `VLLM_ROCM_USE_AITER_MOE` was set rather
  than when AITER was actually *selected*, which made the only previously
  reachable MoRI measurement (via `--moe-backend triton`) degenerate; and
  `FusedMoEConfig.__post_init__` asserted `rocm_aiter_fmoe_enabled` for MoRI,
  which is now a warning. See that comment for why the coupling is a
  performance contract and not a data-format one. With both in place MoRI's
  IntraNode combine is bitwise invariant: 0 of 32 positions moved, against a
  mode-off control at the *same* load exposure (needle rank padding 40 -> 48 in
  both arms) that moved 32 of 32, max |delta| 3.6e-2. Its output is also
  bitwise identical to the allgather_reducescatter path on the same prompts
  (4 prompts x 24 logprobs plus the 32-token needle, max |delta| exactly 0).

  `--max-num-batched-tokens` is pinned low for this arm on purpose. MoRI hands
  the experts a fixed-size receive buffer of `ep_size *
  max_num_batched_tokens` rows and only `AiterExperts` truncates it, so
  `TritonExperts` runs every step at that full M -- measured at exactly 8192
  rows on every one of 2000 calls, where the allgather_reducescatter arm on the
  same model ran M between 4 and 1751. Leaving the default would make this test
  slow and memory-hungry for no extra coverage.

- **mori_low_latency**: not a second test. Both single-node MoRI variants were
  observed selecting `EpDispatchCombineKernelType.IntraNode`
  (`MoriAll2AllManager._make_all2all_kwargs` branches on `self.internode`, not
  on the backend literal), so the low-latency literal exercises the same
  kernel.

`deepep_low_latency` is recorded above rather than skipped silently so that a
change which makes it admissible is noticed.

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
# The MoRI arm needs more. What the guard below wants is for the needle's
# prefill step to have company, and that arm runs the whole padded receive
# buffer through Triton every step -- it served roughly a third as many
# requests as the others in the same wall clock. At the shared default the
# needle's prefill got a step to itself in both conditions and the guard
# (correctly) refused the verdict; at this exposure it does not.
MORI_LOAD_CONCURRENCY = int(os.getenv("VLLM_EP_MORI_LOAD_CONCURRENCY", "96"))
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


def _patch_modular_kernel(module):
    """Record the (prepare/finalize, experts) pair the MoE was built from.

    The manager record above only proves which all2all was constructed. It
    says nothing about which experts class consumed it, and the pairing is
    exactly what is under test here: MoRI with `AiterExperts` is a different
    claim from MoRI with `TritonExperts`, and only the latter runs under the
    mode.
    """
    cls = getattr(module, "FusedMoEKernelModularImpl", None)
    if cls is None:
        return
    original_init = cls.__init__

    def wrapper(self, prepare_finalize, fused_experts, *args, **kwargs):
        original_init(self, prepare_finalize, fused_experts, *args, **kwargs)
        _emit(
            event="mk",
            pf=type(prepare_finalize).__name__,
            experts=type(fused_experts).__name__,
        )

    cls.__init__ = wrapper


_PATCHES = {
    "vllm.v1.worker.dp_utils": _patch_dp_utils,
    "vllm.distributed.device_communicators.all2all": _patch_all2all,
    "vllm.model_executor.layers.fused_moe.modular_kernel": _patch_modular_kernel,
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


def _ep_server(tmp_path, all2all_backend: str, extra_args: list[str] | None = None):
    """A DP=4 + EP server on `all2all_backend`.

    `VLLM_ROCM_USE_AITER` is deliberately left *unset* rather than set to 0:
    the MoE oracles treat the variable being set at all as a request to commit
    to the AITER backend, so exporting it either way changes kernel selection.
    Leaving it unset is also what makes the MoRI arm select `TritonExperts`
    without a `--moe-backend` override.
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
        all2all_backend,
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "64",
        # The needle's prefill should be recomputed every time, so the
        # comparison covers it and not just the decodes.
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        os.getenv("VLLM_EP_TEST_GPU_MEMORY_UTILIZATION", "0.55"),
        *(extra_args or []),
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


@pytest.fixture
def deepep_ht_server(tmp_path, enable_batch_invariant_mode):
    """A DP=4 + EP server on the DeepEP high-throughput all2all.

    Function-scoped and explicitly dependent on the autouse
    `enable_batch_invariant_mode` fixture. A module-scoped server is built
    before that function-scoped fixture runs, so it would launch with
    VLLM_BATCH_INVARIANT unset while this process believes it is set -- and
    a copy of this file that overrides the fixture would not actually flip
    the arm. The `modes` assertion below catches that, and did.
    """
    yield from _ep_server(tmp_path, "deepep_high_throughput")


@pytest.fixture
def mori_ht_server(tmp_path, enable_batch_invariant_mode):
    """A DP=4 + EP server on MoRI.

    Function-scoped for the same reason as `deepep_ht_server`.

    `--max-num-batched-tokens` is pinned because MoRI's receive buffer is
    `ep_size * max_num_batched_tokens` rows and `TritonExperts` runs the whole
    thing every step; see the module docstring.
    """
    yield from _ep_server(
        tmp_path,
        "mori_high_throughput",
        ["--max-num-batched-tokens", "2048"],
    )


def _assert_needle_does_not_see_the_batch(
    server,
    log_prefix: str,
    *,
    manager_cls: str,
    prepare_finalize_cls: str,
    experts_cls: str,
    load_concurrency: int = LOAD_CONCURRENCY,
) -> None:
    """The needle's logprobs must not move when the rest of the server does."""
    url = server.url_for("v1/completions")

    # Discarded: keeps first-request state out of the comparison.
    _needle(url)

    with _Load(url, 0):
        alone = _needle(url)
    with _Load(url, load_concurrency) as load:
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
    assert managers == {manager_cls}, (
        f"the workers built {managers or 'no'} all2all manager(s), not "
        f"{manager_cls}. A fallback to AllGather+ReduceScatter would pass this "
        "test while proving nothing about the backend under test."
    )

    # The manager alone does not pin the measurement: it is the (prepare,
    # experts) pair that decides what the combine feeds and what consumes it.
    pairs = {(r["pf"], r["experts"]) for r in records if r.get("event") == "mk"}
    assert pairs == {(prepare_finalize_cls, experts_cls)}, (
        f"the workers built MoE kernels {pairs}, not "
        f"{{('{prepare_finalize_cls}', '{experts_cls}')}}."
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
        "Raise VLLM_EP_LOAD_CONCURRENCY (or VLLM_EP_MORI_LOAD_CONCURRENCY)."
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


def test_deepep_high_throughput_combine_does_not_see_the_batch(deepep_ht_server):
    server, log_prefix = deepep_ht_server
    _assert_needle_does_not_see_the_batch(
        server,
        log_prefix,
        manager_cls="DeepEPHTAll2AllManager",
        prepare_finalize_cls="DeepEPHTPrepareAndFinalize",
        experts_cls="TritonExperts",
    )


def test_mori_high_throughput_combine_does_not_see_the_batch(mori_ht_server):
    server, log_prefix = mori_ht_server
    _assert_needle_does_not_see_the_batch(
        server,
        log_prefix,
        manager_cls="MoriAll2AllManager",
        prepare_finalize_cls="MoriPrepareAndFinalize",
        experts_cls="TritonExperts",
        load_concurrency=MORI_LOAD_CONCURRENCY,
    )
