# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A token's output must not depend on what a *different* DP replica is doing.

`coordinate_batch_across_dp` (`vllm/v1/worker/dp_utils.py`) all-reduces each
rank's token count every step and, when the synced cudagraph mode is not NONE,
pads every rank up to the maximum. So a rank decoding a single token runs a
forward pass whose token count -- and therefore its cudagraph bucket, its GEMM
tile and split selection, and its collective message sizes -- is chosen by
another replica's batch. That is a cross-process batch dependency: not "your
output depends on what else is in your batch" but "your output depends on what
is in someone else's". This test holds rank 0's request fixed and varies rank
1's load.

Measured on 2x gfx950, DP=2, OLMoE-1B-7B bf16, TRITON_ATTN, cudagraphs on: with
rank 0 decoding one token throughout, its padded token count tracked the peer --
1 with rank 1 idle, 2 with two concurrent peer requests, 48 with 48, and 200
with 200. The needle's logprobs were bitwise identical across all of those and
across two separate server processes; with the mode off the same comparison
split into distinct classes (idle/small vs large, 64/64 logprobs differing, max
|delta| 6.5e-2), so the metric can see the effect it is being asked to rule out.

Scope, and why the pieces that are missing are missing:

- **Dense models are a non-topic.** `vllm/v1/engine/core.py` only builds a
  `DPEngineCoreProc` when `model_config.is_moe`; otherwise it forces
  `data_parallel_size = 1` per engine ("Non-MoE DP ranks are completely
  independent"), and `coordinate_batch_across_dp` early-exits at DP=1.
  `EngineArgs` additionally rejects external-LB DP for non-MoE. A dense DP=2
  server therefore never coordinates anything, and a dense version of this test
  would be green while asserting nothing. Hence the MoE checkpoint.
- **This screens padding, not the combine.** A 2-rank reduction is
  order-independent, so nothing here can catch an order-sensitive DP/EP combine.
  DP+EP is a separate surface (upstream issue #30321). Note that even with EP
  off, DP>1 MoE falls back to AllGather+ReduceScatter dispatch/combine, so the
  peer's tokens do flow through this rank's MoE GEMMs here.
- **Microbatching is untested.** The other half of the negotiation is
  `ubatch_slices`: whether *your* forward pass is split in two is agreed
  collectively, and splitting changes the reduction decomposition directly. It
  cannot be reached in this configuration -- `--enable-dbo` is rejected unless
  the all2all backend is deepep_low_latency, deepep_high_throughput or nixl_ep.
  It also has an asymmetry worth knowing when someone does test it:
  `check_ubatch_thresholds` reads the *local* batch, so a rank holding one
  decode can never propose ubatching; the observable direction is a busy rank
  whose split is cancelled by an idle peer.
- Untested: DP>2, DP+TP, DP+PP.

Per-rank load control needs a live server: the request-level pin is the
`X-data-parallel-rank` header (`vllm/entrypoints/generate/base/serving.py`),
honoured by `DPLBAsyncMPClient.get_core_engine_for_request`. The offline `LLM`
API load-balances by queue depth with no way to address a rank, so it cannot
express this experiment.

The padded token counts are read back through a `sitecustomize` written into a
tmp dir and put on the server's `PYTHONPATH`, which wraps
`dp_utils._synchronize_dp_ranks` in every process the server spawns. It is not
decoration: `test_dp_padding_from_a_peer_replica_does_not_change_logprobs`
fails if the pad did not move, because a run where the peer load never reached
the coordination path would otherwise report invariance without having tested
anything.
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
    *multi_gpu_marks(num_gpus=2),
]

# Needs to be an MoE checkpoint or the DP ranks are decoupled: see the module
# docstring. OLMoE is the smallest MoE the suite has a use for at ~14GB.
MODEL = os.getenv("VLLM_DP_TEST_MODEL", "allenai/OLMoE-1B-7B-0924")
NEEDLE_MAX_TOKENS = int(os.getenv("VLLM_DP_NEEDLE_MAX_TOKENS", "32"))
# Concurrent requests pinned to rank 1 while the needle runs on rank 0.
PEER_CONCURRENCY = int(os.getenv("VLLM_DP_PEER_CONCURRENCY", "32"))
PEER_RAMP_SECONDS = float(os.getenv("VLLM_DP_PEER_RAMP_SECONDS", "12"))
# The peer must drag rank 0's padded count at least this far above its own
# single decode token, otherwise the needle comparison is vacuous.
MIN_PEER_PAD = int(os.getenv("VLLM_DP_MIN_PEER_PAD", "16"))

NEEDLE_PROMPT = (
    "Explain, step by step, how a four-stroke internal combustion engine "
    "converts chemical energy in fuel into rotational mechanical work, and "
    "where the main thermodynamic losses occur."
)

# Written to the server's PYTHONPATH as sitecustomize.py so that it loads in the
# API server, every engine core and every worker, including spawned ones. It
# shadows any other sitecustomize on the path; on the platforms this suite runs
# on that file is empty, and if it were not, the server would fail to start
# rather than quietly mismeasure.
_INSTRUMENTATION = '''
"""Log every DP coordination decision to $DP_COORD_LOG.<pid>."""
import os
import sys

_LOG = os.environ.get("DP_COORD_LOG")


def _patch(module):
    import json
    import time

    original = module._synchronize_dp_ranks
    path = f"{_LOG}.{os.getpid()}"

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
        should_ubatch, after_padding, synced_cudagraph_mode = out
        try:
            import vllm.envs as envs

            import vllm

            record = {
                "t": time.time(),
                "vllm_file": vllm.__file__,
                "dp_rank": parallel_config.data_parallel_rank,
                "unpadded": int(num_tokens_unpadded),
                "padded": (
                    None
                    if after_padding is None
                    else int(after_padding[parallel_config.data_parallel_rank])
                ),
                "should_ubatch": bool(should_ubatch),
                "synced_cudagraph_mode": int(synced_cudagraph_mode),
                "batch_invariant": bool(envs.VLLM_BATCH_INVARIANT),
            }
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\\n")
        except Exception as e:  # pragma: no cover
            sys.stderr.write(f"[dp-coord] failed to log: {e}\\n")
        return out

    module._synchronize_dp_ranks = wrapper


if _LOG:
    import importlib.abc
    import importlib.util

    _TARGET = "vllm.v1.worker.dp_utils"

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            original_exec = spec.loader.exec_module

            def exec_module(module, _exec=original_exec):
                _exec(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
'''


class _PeerLoad:
    """Keeps `concurrency` requests in flight against one DP rank."""

    def __init__(self, url: str, rank: int, concurrency: int):
        self.url, self.rank, self.concurrency = url, rank, concurrency
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.errors: list[str] = []

    def _run(self, seed: int) -> None:
        rng = random.Random(seed)
        while not self._stop.is_set():
            prompt = " ".join(str(rng.randint(0, 99999)) for _ in range(64))
            try:
                _completion(self.url, prompt, 512, self.rank)
            except Exception as e:
                self.errors.append(repr(e))
                time.sleep(0.5)

    def __enter__(self) -> "_PeerLoad":
        for i in range(self.concurrency):
            thread = threading.Thread(target=self._run, args=(i,), daemon=True)
            thread.start()
            self._threads.append(thread)
        if self.concurrency:
            time.sleep(PEER_RAMP_SECONDS)
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=180)
        # Let the peer's queue drain so the next condition starts from idle.
        time.sleep(5.0)


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


def _needle(url) -> dict:
    """One fixed request on rank 0, timestamped so its steps can be found."""
    started = time.time()
    response = _completion(url, NEEDLE_PROMPT, NEEDLE_MAX_TOKENS, rank=0, logprobs=1)
    choice = response["choices"][0]
    return {
        "started": started,
        "finished": time.time(),
        "tokens": choice["logprobs"]["tokens"],
        "logprobs": choice["logprobs"]["token_logprobs"],
    }


def _rank0_decode_pads(log_prefix: str, needle: dict) -> list[int]:
    """Rank 0's padded token counts on its single-token decode steps.

    Restricted to steps where rank 0 had exactly one token of its own, i.e. the
    needle's decodes. Its prefill step is excluded on purpose: that one is ~40
    tokens wide by itself, so a max over every step in the window would be 40
    in both conditions and would hide whether the peer moved anything.
    """
    directory, prefix = os.path.split(log_prefix)
    pads = []
    for name in os.listdir(directory):
        if not name.startswith(prefix + "."):
            continue
        with open(os.path.join(directory, name)) as f:
            for line in f:
                record = json.loads(line)
                if (
                    record["dp_rank"] == 0
                    and record["padded"] is not None
                    and record["unpadded"] == 1
                    and needle["started"] <= record["t"] <= needle["finished"]
                ):
                    pads.append(record["padded"])
    return pads


def _worker_field(log_prefix: str, field: str) -> set:
    directory, prefix = os.path.split(log_prefix)
    values = set()
    for name in os.listdir(directory):
        if name.startswith(prefix + "."):
            with open(os.path.join(directory, name)) as f:
                for line in f:
                    values.add(json.loads(line)[field])
    return values


@pytest.fixture
def dp_server(tmp_path, enable_batch_invariant_mode):
    """A DP=2 server with the coordination path instrumented.

    Depends on the autouse `enable_batch_invariant_mode` fixture rather than
    setting `VLLM_BATCH_INVARIANT` itself: the server inherits whatever the
    fixture put in the environment, so a copy of this file that overrides the
    fixture actually runs with the mode off instead of silently running both
    arms the same way.
    """
    instrumentation = tmp_path / "sitecustomize.py"
    instrumentation.write_text(_INSTRUMENTATION)
    log_prefix = str(tmp_path / "dp_coord")

    args = [
        "--data-parallel-size",
        "2",
        "--data-parallel-size-local",
        "2",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "64",
        # The needle's prefill should be recomputed every time, so that the
        # comparison covers it and not just the decodes.
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        os.getenv("VLLM_DP_TEST_GPU_MEMORY_UTILIZATION", "0.45"),
    ]
    # The repo root goes on PYTHONPATH ahead of everything else because
    # RemoteOpenAIServer launches the `vllm` console script off PATH rather than
    # `sys.executable -m`, so the server does not inherit this process's
    # interpreter or its venv. Without this it resolves `vllm` from whatever the
    # script's shebang interpreter happens to have on its path -- on a machine
    # with an unrelated vLLM checkout wired in through a user-site
    # `easy-install.pth`, that is a *different tree*, and the test would measure
    # a build nobody is asking about while passing. `_assert_server_runs_this_tree`
    # below is the check that keeps this honest; the path entry only makes the
    # common case work.
    repo_root = str(Path(vllm.__file__).resolve().parent.parent)
    env = {
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), repo_root, os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
        "DP_COORD_LOG": log_prefix,
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
    }
    with RemoteOpenAIServer(MODEL, args, env_dict=env, max_wait_seconds=1200) as server:
        yield server, log_prefix


def test_dp_padding_from_a_peer_replica_does_not_change_logprobs(dp_server):
    """Rank 0's logprobs must not move when only rank 1's batch changes."""
    server, log_prefix = dp_server
    url = server.url_for("v1/completions")

    # Discarded: keeps first-request state out of the comparison, so that a
    # difference below is attributable to the peer load.
    _needle(url)

    with _PeerLoad(url, rank=1, concurrency=0):
        alone = _needle(url)
    with _PeerLoad(url, rank=1, concurrency=PEER_CONCURRENCY) as peer:
        with_peer = _needle(url)
    assert not peer.errors, f"the peer load did not run cleanly: {peer.errors[:3]}"

    # The server is a separate process launched off PATH, so it can silently be
    # a different vLLM than the one under test. Then everything below would
    # measure someone else's build and pass.
    served = _worker_field(log_prefix, "vllm_file")
    assert served == {vllm.__file__}, (
        f"the server imported vLLM from {served}, but this test process is "
        f"{vllm.__file__}. The server subprocess is running a different tree, "
        f"so nothing it reports is evidence about this one."
    )

    modes = _worker_field(log_prefix, "batch_invariant")
    assert modes == {envs.VLLM_BATCH_INVARIANT}, (
        f"the server's effective VLLM_BATCH_INVARIANT is {modes}, but this "
        f"process has {envs.VLLM_BATCH_INVARIANT}; the two arms of this test "
        f"are not running the mode they claim to"
    )

    # The vacuity check, and the reason this test cannot pass while blind: if
    # the peer never moved rank 0's padded token count then rank 0 ran the same
    # shapes in both conditions and the comparison below proves nothing.
    alone_pads = _rank0_decode_pads(log_prefix, alone)
    peer_pads = _rank0_decode_pads(log_prefix, with_peer)
    assert alone_pads and peer_pads, (
        "no DP coordination was recorded on rank 0's decode steps while the "
        f"needle ran (solo: {len(alone_pads)} steps, with peer: "
        f"{len(peer_pads)} steps). Either the instrumentation did not load or "
        "the ranks are not coordinating -- check that the model is an MoE "
        "checkpoint, since dense DP ranks are forced to data_parallel_size=1."
    )
    assert max(peer_pads) >= MIN_PEER_PAD and max(peer_pads) > max(alone_pads), (
        "rank 0 decoded one token per step throughout, but its padded token "
        "count did not follow the peer's batch (solo: max "
        f"{max(alone_pads)} over {len(alone_pads)} decode steps, with "
        f"{PEER_CONCURRENCY} concurrent peer requests: max {max(peer_pads)} "
        f"over {len(peer_pads)} decode steps). The DP padding mechanism this "
        "test exists to screen never fired, so its verdict would be vacuous. "
        "Raise the peer load, or check that cudagraphs are enabled -- padding "
        "is skipped when the synced cudagraph mode is NONE."
    )

    assert with_peer["tokens"] == alone["tokens"], (
        f"rank 0 sampled different tokens once rank 1 was busy: "
        f"{alone['tokens']} vs {with_peer['tokens']}"
    )
    moved = [
        index
        for index, (a, b) in enumerate(zip(alone["logprobs"], with_peer["logprobs"]))
        if a != b
    ]
    assert not moved, (
        f"rank 0's logprobs changed at positions {moved} because rank 1 was "
        f"busy: its padded token count went from {max(alone_pads)} to "
        f"{max(peer_pads)} tokens while its own request was byte-identical. "
        f"max |delta| = "
        f"{max(abs(alone['logprobs'][i] - with_peer['logprobs'][i]) for i in moved)}"
    )
