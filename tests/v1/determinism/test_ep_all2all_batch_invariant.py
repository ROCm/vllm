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

  (`FusedMoEParallelConfig.use_batched_activation_format`), whose only experts

- **mori_low_latency**: not a second test. Both single-node MoRI variants were
  observed selecting `EpDispatchCombineKernelType.IntraNode`
  (`MoriAll2AllManager._make_all2all_kwargs` branches on `self.internode`, not
  on the backend literal), so the low-latency literal exercises the same
  kernel.

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
from vllm.utils.import_utils import has_deep_ep, has_mori

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
# The default is tuned for the bf16 high-throughput arm. What the guard below
# wants is for the needle's prefill step to have company; MoRI runs the whole
# padded receive buffer through Triton every step and serves far fewer requests
# in the same wall clock, so at the shared default the needle's prefill gets a
# step to itself and the guard correctly refuses a verdict.
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
_PATCHERS = '''
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


_TARGETS = {
    "vllm.v1.worker.dp_utils": _patch_dp_utils,
    "vllm.distributed.device_communicators.all2all": _patch_all2all,
    "vllm.model_executor.layers.fused_moe.modular_kernel": _patch_modular_kernel,
}

'''

_INSTRUMENTATION = _PATCHERS + INSTRUMENTATION_IMPORT_HOOK


def _load(url: str, concurrency: int, model: str = MODEL) -> BackgroundLoad:
    """Unrelated requests in flight across every DP rank."""

    def send(rng, index):
        prompt = " ".join(
            str(rng.randint(0, 99999)) for _ in range(rng.choice([16, 48, 96, 160]))
        )
        dp_completion(url, model, prompt, rng.choice([64, 128, 256]), index % DP)

    return BackgroundLoad(
        send,
        concurrency=concurrency,
        ramp_seconds=LOAD_RAMP_SECONDS,
        drain_seconds=6.0,
        join_timeout=240,
    )


def _needle(url) -> dict:
    started = time.time()
    response = dp_completion(
        url, MODEL, NEEDLE_PROMPT, NEEDLE_MAX_TOKENS, NEEDLE_RANK, logprobs=1
    )
    choice = response["choices"][0]
    return {
        "started": started,
        "finished": time.time(),
        "tokens": choice["logprobs"]["tokens"],
        "logprobs": choice["logprobs"]["token_logprobs"],
    }


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

    Gated on the backend's optional package being importable. Without this the
    engine fails to come up and `RemoteOpenAIServer` sits out its whole
    `max_wait_seconds=1800`, so a machine with neither package spends three
    hours producing six errors where six skips are the honest result.

    `VLLM_ROCM_USE_AITER` is deliberately left *unset* rather than set to 0:
    the MoE oracles treat the variable being set at all as a request to commit
    to the AITER backend, so exporting it either way changes kernel selection.
    Leaving it unset is also what makes the MoRI arm select `TritonExperts`
    without a `--moe-backend` override.
    """
    if all2all_backend.startswith("deepep") and not has_deep_ep():
        pytest.skip("requires the deep_ep package")
    if all2all_backend.startswith("mori") and not has_mori():
        pytest.skip("requires the mori package")

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
    env = instrumented_server_env(tmp_path, _INSTRUMENTATION, EP_A2A_LOG=log_prefix)
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

    with _load(url, 0):
        alone = _needle(url)
    with _load(url, load_concurrency) as load:
        loaded = _needle(url)
    load.assert_ran_cleanly()

    records = read_records(log_prefix)

    # `quant_err` is a key inside an `mk` record, not an event of its own.
    errors = [
        r
        for r in records
        if str(r.get("event", "")).endswith("_err") or "quant_err" in r
    ]
    assert not errors, f"the instrumentation raised: {errors[:3]}"

    assert_server_ran_this_tree(
        {r["vllm_file"] for r in records if r.get("event") == "env"},
        {r["batch_invariant"] for r in records if r.get("event") == "env"},
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
