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

Measured on 4x gfx950, OLMoE-1B-7B bf16, TRITON_ATTN, cudagraphs on, needle
pinned to DP rank 2 and compared bitwise over 64 generated logprobs against the
same prompt run solo:

                                  mode on   mode off (max |delta|)
  32 companions, needle's rank      0/64      64/64  (1.4e-1)
  96 companions, needle's rank      0/64      64/64  (5.4e-2)
  24 companions, peer ranks only    0/64      63/64  (1.2e0, different token)
  48 companions, all four ranks     0/64      64/64  (6.5e-2)
  same prompt solo, other rank      0/64      64/64  (6.3e-2)
  solo, repeated                    0/64       0/64

447 moved positions in the mode-off arm, none in the mode-on arm. The last two
rows are the ones that make the rest mean something. `solo` repeated is clean
with the mode *off* as well, so the mode-off differences track batch composition
rather than run-to-run noise: the metric is specific, not just sensitive. And
with the mode off, merely which DP rank served an otherwise byte-identical
request changes its logprobs.

A second, sharper control was run outside this file, because the mode-off arm
moves many things at once. With the mode left fully on and *only*
`CudaCommunicator.reduce_scatterv` monkeypatched back onto the library path --
which is exactly what upstream has, its `reduce_scatterv` having no
`VLLM_BATCH_INVARIANT` branch -- the same protocol gives 320 moved positions
across five of eight conditions. So the batch variance upstream reports in
issue #30321 is attributable to that one function, and the fixed-order
reduce-scatter is what closes it. Note also that three of those eight conditions
came back *clean* under a combine that is genuinely variant: any single
condition can pass by luck, which is why this test runs several and why a
one-condition version of it would be worth very little.

`--quantization fp8` is covered as a second configuration because it is the only
one that reaches the dynamic per-tensor activation scale, whose amax spans
whatever the all2all delivered rather than just the local batch. Its mode-off
arm cannot be collected, and the reason is worth knowing before someone tries:
with the mode off the scale stays per-tensor, so `prepare` hands a 1-row scale
to `all_gatherv` alongside a many-row activation
(`naive_dp_ep._quantize_and_setup_dispatch` only skips the gather for `ndim == 0`
scales), and `_all_gather_single` asserts `1 != <this rank's token count>` as
soon as the DP ranks are unevenly loaded. Reproduced three times out of three,
always on the first condition with concurrent load, and it kills the server. The
promotion to per-token is what makes the shapes agree, so under the mode the
configuration runs. The mode-on arm is therefore validated against a variant
*reduce-scatter* instead, which is a control the configuration can survive: with
the promotion left in place and only `reduce_scatterv` restored to the library
path, the fp8 needle moves 319/512 positions across four conditions with max
|delta| 2.7 and a different token sampled from position 2 onward. So the metric
demonstrably detects batch variance in *this* configuration, not merely in the
bf16 one.

Scope, and what is deliberately not here:

- **Only `allgather_reducescatter`, the default.** It is also the only backend
  reachable in this configuration: DeepEP and MoRI need their kernels installed.
  Their combines are fixed-order by source audit (DeepEP HT walks source ranks
  ascending after a full wait, DeepEP LL sums over top-k slot order after a grid
  barrier, MoRI reduces via `WarpAccum` in ascending expert-slot order, all with
  fp32 accumulators and no float atomics), but an audit is not a measurement and
  none of them is exercised here.
- **The combine only runs when DP > 1.** `use_all2all_kernels` is
  `use_ep and (dp_size > 1 or pcp_size > 1 or is_sequence_parallel)`, so TP=4
  plus `--enable-expert-parallel` with DP=1 shards the experts but never builds
  an all2all manager -- the cross-rank sum is the ordinary TP all-reduce, which
  `test_tp_all_reduce_batch_invariant.py` owns. A "TP+EP" version of this test
  would not touch `combine` at all.
- Untested: EP > 4, EPLB (`--enable-eplb` moves experts between ranks at run
  time, which is a batch dependency of a different kind), microbatching
  (`--enable-dbo`, rejected on this backend), PCP, and sequence-parallel MoE.

Two neighbouring configurations were measured while writing this and are green,
but are not asserted here because neither exercises the combine: TP=4 + EP=4
with DP=1 (0/64 with the mode on, 384 moved with it off) and DP=2 x TP=2 with
EP=4 (0/64 on, 380 moved off, but only a 2-wide combine).

The instrumentation is not decoration. A server that silently fell back to
non-EP, or that imported a different vLLM checkout, would sail through the
comparison below while proving nothing, and both have happened in this project.
`_worker_facts` asserts the workers' `vllm.__file__` and effective
`VLLM_BATCH_INVARIANT` against this process, and
`test_ep_needle_is_invariant_to_batch_composition` fails if the AgRs combine was
never observed at world size 4.
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

# Written to the server's PYTHONPATH as sitecustomize.py so it loads in the API
# server, every engine core and every worker, including spawned ones.
_INSTRUMENTATION = '''
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


class _Load:
    """Keeps `concurrency` requests in flight, spread over `ranks`."""

    def __init__(self, url, ranks, concurrency, seed=0):
        self.url, self.ranks, self.concurrency = url, ranks, concurrency
        self.seed = seed
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.errors: list[str] = []
        self.completed = 0

    def _run(self, index: int) -> None:
        rng = random.Random(self.seed * 1000 + index)
        rank = self.ranks[index % len(self.ranks)]
        while not self._stop.is_set():
            prompt = " ".join(
                str(rng.randint(0, 99999)) for _ in range(rng.randint(32, 96))
            )
            try:
                _completion(self.url, prompt, 384, rank)
                self.completed += 1
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


def _needle(url, rank=NEEDLE_RANK) -> dict:
    response = _completion(url, NEEDLE_PROMPT, NEEDLE_MAX_TOKENS, rank, logprobs=1)
    choice = response["choices"][0]
    return {
        "tokens": choice["logprobs"]["tokens"],
        "logprobs": choice["logprobs"]["token_logprobs"],
    }


def _records(log_prefix: str) -> list[dict]:
    directory, prefix = os.path.split(log_prefix)
    out: list[dict] = []
    for name in os.listdir(directory):
        if name.startswith(prefix + "."):
            with open(os.path.join(directory, name)) as f:
                out.extend(json.loads(line) for line in f)
    return out


def _worker_facts(records: list[dict]) -> None:
    """The server is a separate process launched off PATH, so it can silently
    be a different vLLM or a different mode than the one under test."""
    served = {r["vllm_file"] for r in records}
    assert served == {vllm.__file__}, (
        f"the server imported vLLM from {served}, but this test process is "
        f"{vllm.__file__}. The server subprocess is running a different tree, "
        f"so nothing it reports is evidence about this one."
    )
    modes = {r["batch_invariant"] for r in records}
    assert modes == {envs.VLLM_BATCH_INVARIANT}, (
        f"the server's effective VLLM_BATCH_INVARIANT is {modes}, but this "
        f"process has {envs.VLLM_BATCH_INVARIANT}; a copy of this file that "
        f"overrides the mode fixture would otherwise run both arms the same way"
    )


def _make_server(tmp_path, extra_args):
    instrumentation = tmp_path / "sitecustomize.py"
    instrumentation.write_text(_INSTRUMENTATION)
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
    # The repo root goes on PYTHONPATH ahead of everything else because
    # RemoteOpenAIServer launches the `vllm` console script off PATH rather than
    # `sys.executable -m`, so the server does not inherit this process's
    # interpreter. On a machine with an unrelated vLLM wired in through a
    # user-site `easy-install.pth` that resolves to a *different tree*.
    # `_worker_facts` is the check that keeps this honest; the path entry only
    # makes the common case work.
    repo_root = str(Path(vllm.__file__).resolve().parent.parent)
    env = {
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), repo_root, os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
        "EP_LOG": log_prefix,
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
    }
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
    loads: dict[str, _Load] = {}
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
        with _Load(url, ranks, concurrency, seed) as load:
            conditions[label] = _needle(url)
        loads[label] = load

    # Informational in spirit but asserted anyway: which replica served an
    # otherwise byte-identical request is as much "the batch" as anything else,
    # and with the mode off this one moves.
    conditions["other_rank_solo"] = _needle(url, (NEEDLE_RANK + 1) % DP_SIZE)

    for label, load in loads.items():
        assert not load.errors, f"{label} companions failed: {load.errors[:3]}"

    records = _records(log_prefix)
    _worker_facts(records)

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
