# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix caching must not make a token's output depend on the cache's history.

Every other end-to-end test in this suite passes `--no-enable-prefix-caching`.
The reason is sound -- with caching on, a needle's prefill collapses to a single
token after its first run, so a batch-composition sweep would compare decodes
only -- but prefix caching is **on by default**, so TP, PP, DP, DCP and EP have
all been certified in a configuration nobody deploys. This test is the one that
leaves the default alone.

Caching is also the surface this project has already found a real bug on.
`VllmConfig.__post_init__` disables prefix caching for **MLA** models under the
mode, because an MLA prefill splits its attention at the context/new-token
boundary and a cache hit puts that boundary wherever the cache happened to
reach -- a function of every request that ran before. Standard attention was
measured unaffected, but only at DP=1/TP=1, long before any expert-parallel or
data-parallel machinery existed. Here it is measured at DP=4 x EP=4.

Two dependencies are screened, and the second is the novel one:

1. **The needle's own cache state.** The same prompt is run cold, and at hits of
   400, 800, 1600 and 2384 tokens. Under the mode the scheduler caps a prefill
   chunk at `batch_invariant_prefill_chunk` so the chunk *length* is a function
   of the request alone -- but the chunk *origin* is not, because chunks start
   at the hit length. At the settings below the cap is 1984, so cold prefills as
   [1984, 416] and a 400-token hit prefills as [1984, 16]: the same number of
   chunks, with the boundary moved from 1984 to 2384. That is precisely the
   degree of freedom the cap cannot close, and it is what made MLA unsafe.

2. **A peer replica's cache state.** This one is a genuinely new dependency
   chain rather than a new configuration of an old one. `num_tokens_unpadded` is
   `scheduler_output.total_num_scheduled_tokens`, i.e. a sum of
   `request.num_tokens - num_computed_tokens`; a cache hit lowers it. That
   number is row 1 of the DP all-reduce in `dp_utils.coordinate_batch_across_dp`
   and `_post_process_dp_padding` takes the `max()` across ranks. So the padding
   a peer induces on this rank now depends on that peer's cache state -- on
   requests that finished earlier, possibly for a different user. The
   `peercold`/`peerwarm` conditions below hold concurrency, prompt length and
   rank set fixed and vary only whether the peers' prompts are in cache.

   Measured, and the direction is not the obvious one: with peers running cold
   2000-token prefills the needle's padded width stayed at 16, because a
   prefill-heavy step is not cudagraph-eligible and `should_dp_pad` is False;
   with the peers' prompts served from cache they drop into uniform decode,
   cudagraphs engage, and the needle was padded out to 72. The peers' cache
   state controls both *whether* DP padding fires and how wide it goes.

Measured on 4x gfx950, OLMoE-1B-7B bf16, TRITON_ATTN, cudagraphs on, DP=4 x
EP=4 on `allgather_reducescatter`, prefix caching left at its default, needle
pinned to DP rank 2, 64 generated logprobs compared bitwise:

                              mode on   mode off (max |delta|)
  needle cold vs 400-hit        0/64      60/64  (6.6e-4)
  needle cold vs full hit       0/64      57/64  (5.7e-4)
  400-hit vs full hit           0/64      59/64  (6.9e-4)
  800-hit vs 1600-hit           0/64      60/64  (2.4e-3)
  peer cache state, needle cold 0/64      54/64  (7.7e-4)
  peer cache state, needle warm 0/64      53/64  (1.1e-3)
  baseline repeated             0/64       0/64

Over the full 18-condition matrix that produced these, **all 153 pairwise
comparisons were bitwise equal under the mode, and 150 of 153 differed with it
off** -- the three that did not are exactly the repeats of the baseline. That
last row is what makes the rest mean something: the mode-off differences track
the cache and the batch rather than run-to-run noise, so the metric is specific
and not merely sensitive.

Each condition uses a **unique `cache_salt`**. The salt enters only the block
hash (`kv_cache_utils._gen_extra_hash_keys`, first block only, and the chain
carries it forward), never the model input, so the needle's token ids are
byte-identical in every condition while its hit length is under the test's
control. Without that, conditions would contaminate each other -- the first run
of the needle would warm the prefix for all the later ones -- and there would be
no way to obtain a cold arm on a server that has already seen the prompt.

The instrumentation is not decoration. With caching on it is very easy to build
a comparison in which the needle takes an identical code path on both sides,
which passes while asserting nothing. `test_prefix_cache_and_ep` therefore fails
unless it observed at least four distinct cache-hit lengths, at least three
distinct scheduled-token sequences, and a peer-induced change in this rank's
padded token count -- as well as the usual checks that EP was really active at
width four, that prefix caching was really on, and that the server is this tree
running this mode.

Untested: MLA (caching is disabled for it under the mode, by design), TP>1 with
caching, PP with caching, hits that are not block-aligned, and cache eviction
under memory pressure -- the KV cache here is ~650k tokens per rank, so nothing
the peers did could evict the needle's prefix, and a run that had to evict is a
different experiment.
"""

import json
import os
import random
import threading
import time
import uuid
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

MODEL = os.getenv("VLLM_PC_TEST_MODEL", "allenai/OLMoE-1B-7B-0924")
DP_SIZE = 4
# Not rank 0. Index 0 is privileged in more places than is comfortable and
# keying a determinism verdict on it has produced false negatives here before.
NEEDLE_RANK = 2
# 2400 > the 1984 chunk cap, so a cold needle prefills in two chunks and the
# hit ladder can move the boundary between them.
NEEDLE_TOKENS = 2400
PEER_TOKENS = 2000
NEEDLE_MAX_TOKENS = int(os.getenv("VLLM_PC_NEEDLE_MAX_TOKENS", "64"))
PEER_MAX_TOKENS = 256
MAX_NUM_BATCHED_TOKENS = 2048
MAX_NUM_SEQS = 64
RAMP_SECONDS = float(os.getenv("VLLM_PC_RAMP_SECONDS", "12"))
DRAIN_SECONDS = float(os.getenv("VLLM_PC_DRAIN_SECONDS", "8"))

_INSTRUMENTATION = '''
"""Record the cache hits, scheduled tokens and DP padding to $PC_LOG.<pid>."""
import json
import os
import sys
import threading

_LOG = os.environ.get("PC_LOG")
_PATH = f"{_LOG}.{os.getpid()}" if _LOG else None
_LOCK = threading.Lock()
_SEEN = set()
_HANDLE = None
_WRITES = 0
_DP_RANK = int(os.environ.get("PC_DP_RANK", "2"))


def _emit(kind, **fields):
    global _HANDLE, _WRITES
    try:
        import vllm
        import vllm.envs as envs

        record = {
            "kind": kind,
            "vllm_file": vllm.__file__,
            "batch_invariant": bool(envs.VLLM_BATCH_INVARIANT),
        }
        record.update(fields)
        with _LOCK:
            if _HANDLE is None:
                _HANDLE = open(_PATH, "a", buffering=1 << 16)
            _HANDLE.write(json.dumps(record, default=str) + "\\n")
            _WRITES += 1
            if kind != "dp" or _WRITES % 200 == 0:
                _HANDLE.flush()
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[pc-instr] {e}\\n")


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


def _patch_all2all(module):
    original = module.AgRsAll2AllManager.combine

    def combine(self, hidden_states, is_sequence_parallel=False):
        group = self._get_comm_group(is_sequence_parallel)
        if _once(("combine", group.world_size, group.unique_name)):
            _emit("combine", comm_world_size=group.world_size)
        return original(self, hidden_states, is_sequence_parallel)

    module.AgRsAll2AllManager.combine = combine


def _patch_kv_cache_manager(module):
    original = module.KVCacheManager.get_computed_blocks

    def get_computed_blocks(self, request):
        out = original(self, request)
        if "needle" in request.request_id:
            _emit(
                "cache_hit",
                req=request.request_id,
                hit=int(out[1]),
                num_tokens=int(request.num_tokens),
            )
        return out

    module.KVCacheManager.get_computed_blocks = get_computed_blocks


def _patch_scheduler(module):
    original = module.Scheduler.schedule

    def schedule(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        try:
            if _once("chunk_cap"):
                _emit(
                    "chunk_cap",
                    chunk_cap=int(getattr(self, "batch_invariant_prefill_chunk", -1)),
                    prefix_caching=bool(
                        self.vllm_config.cache_config.enable_prefix_caching
                    ),
                    block_size=int(self.cache_config.block_size),
                )
            for req_id in out.num_scheduled_tokens:
                if "needle-" in req_id:
                    _emit(
                        "sched",
                        req=req_id,
                        scheduled=int(out.num_scheduled_tokens[req_id]),
                    )
        except Exception as e:  # pragma: no cover
            sys.stderr.write(f"[pc-instr sched] {e}\\n")
        return out

    module.Scheduler.schedule = schedule


def _patch_dp_utils(module):
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
            import time

            rank = parallel_config.data_parallel_rank
            if rank == _DP_RANK and after_padding is not None:
                _emit(
                    "dp",
                    t=time.time(),
                    unpadded=int(num_tokens_unpadded),
                    padded=int(after_padding[rank]),
                )
        except Exception as e:  # pragma: no cover
            sys.stderr.write(f"[pc-instr dp] {e}\\n")
        return out

    module._synchronize_dp_ranks = wrapper


_TARGETS = {
    "vllm.model_executor.layers.fused_moe.config": _patch_moe_config,
    "vllm.distributed.device_communicators.all2all": _patch_all2all,
    "vllm.v1.core.kv_cache_manager": _patch_kv_cache_manager,
    "vllm.v1.core.sched.scheduler": _patch_scheduler,
    "vllm.v1.worker.dp_utils": _patch_dp_utils,
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


def _completion(url, prompt, max_tokens, rank, salt, request_id, logprobs=None):
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 20240919,
        "cache_salt": salt,
    }
    if logprobs is not None:
        body["logprobs"] = logprobs
        # The needle must emit exactly max_tokens in every condition, or an
        # early stop would silently truncate the comparison instead of failing.
        body["ignore_eos"] = True
    response = requests.post(
        url,
        json=body,
        headers={"X-data-parallel-rank": str(rank), "X-Request-Id": request_id},
        timeout=1200,
    )
    response.raise_for_status()
    return response.json()


class _Load:
    """Keeps `concurrency` requests in flight, spread over `ranks`.

    `mode` selects the *peers' cache state* while holding everything else
    fixed: "cold" sends a fresh random prompt every time so every peer prefill
    is a full miss, "warm" sends one pre-warmed prompt so every peer prefill is
    a near-total hit. Same concurrency, same prompt length, same ranks.
    """

    def __init__(self, url, ranks, concurrency, mode="cold", seed=0, warm_prompt=None):
        self.url, self.ranks, self.concurrency = url, ranks, concurrency
        self.mode, self.seed, self.warm_prompt = mode, seed, warm_prompt
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.errors: list[str] = []

    def _run(self, index: int) -> None:
        rng = random.Random(self.seed * 1000 + index)
        rank = self.ranks[index % len(self.ranks)]
        while not self._stop.is_set():
            if self.mode == "warm":
                prompt, salt = self.warm_prompt, "peer-warm-shared"
            else:
                prompt = [rng.randint(1000, 30000) for _ in range(PEER_TOKENS)]
                salt = f"peer-cold-{uuid.uuid4().hex}"
            try:
                _completion(
                    self.url, prompt, PEER_MAX_TOKENS, rank, salt, uuid.uuid4().hex
                )
            except Exception as e:
                self.errors.append(repr(e))
                time.sleep(0.5)

    def __enter__(self) -> "_Load":
        if self.concurrency and self.mode == "warm":
            # Every rank the load touches must already hold the shared prefix,
            # or the first requests race each other and only partly hit.
            for rank in self.ranks:
                _completion(
                    self.url,
                    self.warm_prompt,
                    1,
                    rank,
                    "peer-warm-shared",
                    uuid.uuid4().hex,
                )
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
            thread.join(timeout=600)
        time.sleep(DRAIN_SECONDS)


def _records(log_prefix: str) -> list[dict]:
    directory, prefix = os.path.split(log_prefix)
    out: list[dict] = []
    for name in os.listdir(directory):
        if name.startswith(prefix + "."):
            with open(os.path.join(directory, name)) as f:
                out.extend(json.loads(line) for line in f if line.strip())
    return out


@pytest.fixture
def pc_server(tmp_path, enable_batch_invariant_mode):
    """A DP=4 + EP=4 server with prefix caching left at its default.

    Function scoped and depending on `enable_batch_invariant_mode` on purpose:
    a module-scoped server would be built before that autouse fixture runs and
    would launch with the mode off, so both arms would silently be the same arm.

    Note what is *not* passed: no `--enable-prefix-caching` and no
    `--no-enable-prefix-caching`. The point of this test is the default, and
    the `prefix_caching` assertion below reads back what the scheduler actually
    got rather than trusting the flag.
    """
    instrumentation = tmp_path / "sitecustomize.py"
    instrumentation.write_text(_INSTRUMENTATION)
    log_prefix = str(tmp_path / "pc_instr")

    args = [
        "--data-parallel-size",
        str(DP_SIZE),
        "--data-parallel-size-local",
        str(DP_SIZE),
        "--enable-expert-parallel",
        # OLMoE's max_position_embeddings is 4096; needle 2400+64, peers 2000+256.
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--enable-prompt-tokens-details",
        "--gpu-memory-utilization",
        os.getenv("VLLM_PC_TEST_GPU_MEMORY_UTILIZATION", "0.30"),
    ]
    # RemoteOpenAIServer launches the `vllm` console script off PATH rather than
    # `sys.executable -m`, so the server does not inherit this process's
    # interpreter. On a box with an unrelated vLLM wired in through a user-site
    # `easy-install.pth` that resolves to a *different tree*. The path entry
    # makes the common case work; `_worker_facts` is what keeps it honest.
    repo_root = str(Path(vllm.__file__).resolve().parent.parent)
    env = {
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), repo_root, os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
        "PC_LOG": log_prefix,
        "PC_DP_RANK": str(NEEDLE_RANK),
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
    }
    with RemoteOpenAIServer(
        MODEL, args, env_dict=env, seed=20240919, max_wait_seconds=1800
    ) as server:
        yield server, log_prefix


def test_prefix_cache_and_ep(pc_server):
    """A needle must not move with its own cache state or with its peers'."""
    server, log_prefix = pc_server
    url = server.url_for("v1/completions")
    peers = [r for r in range(DP_SIZE) if r != NEEDLE_RANK]

    tokenized = requests.post(
        server.url_for("tokenize"),
        json={
            "model": MODEL,
            "prompt": (
                "Explain, step by step, how a four-stroke internal combustion "
                "engine converts chemical energy in fuel into rotational "
                "mechanical work, and where the main thermodynamic losses occur. "
            )
            * 100,
        },
        timeout=120,
    )
    tokenized.raise_for_status()
    ids = tokenized.json()["tokens"]
    assert len(ids) >= NEEDLE_TOKENS, f"seed text tokenized to only {len(ids)}"
    # An explicit token-id prompt, so the hit ladder lands exactly where asked
    # and every condition's prompt is byte-identical.
    needle = ids[:NEEDLE_TOKENS]
    peer_warm_prompt = [
        random.Random(777).randint(1000, 30000) for _ in range(PEER_TOKENS)
    ]

    # Discarded: keeps first-request state out of the comparison.
    _completion(url, needle, 8, NEEDLE_RANK, uuid.uuid4().hex, "discard")

    # label, needle hit target (None = cold), peer concurrency, ranks, peer mode
    plan = [
        ("cold_idle", None, 0, [NEEDLE_RANK], "cold"),
        ("hit400_idle", 400, 0, [NEEDLE_RANK], "cold"),
        ("hit1600_idle", 1600, 0, [NEEDLE_RANK], "cold"),
        ("hitfull_idle", NEEDLE_TOKENS, 0, [NEEDLE_RANK], "cold"),
        ("cold_idle_again", None, 0, [NEEDLE_RANK], "cold"),
        ("hitfull_peers24", NEEDLE_TOKENS, 24, peers, "cold"),
        ("cold_peercold32", None, 32, peers, "cold"),
        ("cold_peerwarm32", None, 32, peers, "warm"),
        ("hitfull_peercold32", NEEDLE_TOKENS, 32, peers, "cold"),
        ("hitfull_peerwarm32", NEEDLE_TOKENS, 32, peers, "warm"),
    ]

    conditions: dict[str, dict] = {}
    windows: dict[str, tuple[float, float]] = {}
    loads: dict[str, _Load] = {}
    for label, hit_target, concurrency, ranks, mode in plan:
        # A unique salt per condition: same tokens, different block hashes, so
        # conditions cannot warm each other and a cold arm stays available.
        salt = f"{label}-{uuid.uuid4().hex}"
        with _Load(
            url, ranks, concurrency, mode, hash(label) % 97, peer_warm_prompt
        ) as load:
            if hit_target is not None:
                # Warm under the same load, so the hit is LRU-recent even while
                # the peers are churning the cache.
                _completion(
                    url,
                    needle[:hit_target],
                    1,
                    NEEDLE_RANK,
                    salt,
                    f"needlewarm-{label}",
                )
                time.sleep(1.0)
            started = time.time()
            response = _completion(
                url,
                needle,
                NEEDLE_MAX_TOKENS,
                NEEDLE_RANK,
                salt,
                f"needle-{label}",
                logprobs=1,
            )
            windows[label] = (started, time.time())
        loads[label] = load
        choice = response["choices"][0]
        conditions[label] = {
            "tokens": choice["logprobs"]["tokens"],
            "logprobs": choice["logprobs"]["token_logprobs"],
            "cached": (response["usage"].get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
        }

    for label, load in loads.items():
        assert not load.errors, f"{label} companions failed: {load.errors[:3]}"

    records = _records(log_prefix)

    # The server is a separate process launched off PATH, so it can silently be
    # a different vLLM or a different mode than the one under test.
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

    # --- the configuration really is the one this test is about -------------
    configs = [r for r in records if r["kind"] == "moe_parallel_config"]
    assert configs and all(c["use_ep"] and c["ep_size"] == DP_SIZE for c in configs), (
        f"expert parallelism was not active as configured: {configs}"
    )
    assert all(
        c["all2all_backend"] == "allgather_reducescatter"
        and c["use_ag_rs_all2all_kernels"]
        for c in configs
    ), f"a different all2all backend served: {configs}"
    combines = [r for r in records if r["kind"] == "combine"]
    assert combines and all(c["comm_world_size"] == DP_SIZE for c in combines), (
        f"the all2all combine was never observed reducing over {DP_SIZE} ranks "
        f"(saw {combines}); a 2-rank reduction is order independent and would "
        f"hold against a fully batch-variant combine."
    )
    caps = [r for r in records if r["kind"] == "chunk_cap"]
    assert caps and all(c["prefix_caching"] for c in caps), (
        f"prefix caching was NOT enabled in the scheduler: {caps}. This whole "
        f"test is about the default-on configuration; with caching off every "
        f"condition below runs the identical code path and proves nothing."
    )

    # --- vacuity: the needle's code path really did differ ------------------
    hits = {
        r["req"].split("cmpl-needle-")[1].rsplit("-0-", 1)[0]: r["hit"]
        for r in records
        if r["kind"] == "cache_hit" and "cmpl-needle-" in r["req"]
    }
    observed = {label: hits.get(label) for label in conditions}
    assert len(set(observed.values()) - {None}) >= 4, (
        f"expected at least four distinct prefix-cache hit lengths across the "
        f"conditions but saw {observed}. If the needle hit the same amount of "
        f"cache every time it took one code path throughout and this test "
        f"asserted nothing."
    )
    for label, cond in conditions.items():
        assert cond["cached"] == observed[label], (
            f"{label}: the server reported cached_tokens={cond['cached']} but "
            f"the scheduler recorded a hit of {observed[label]}"
        )

    sched: dict[str, list[int]] = {}
    for r in records:
        if r["kind"] == "sched" and "cmpl-needle-" in r["req"]:
            label = r["req"].split("cmpl-needle-")[1].rsplit("-0-", 1)[0]
            sched.setdefault(label, []).append(r["scheduled"])
    prefills = {label: tuple(n for n in seq if n > 1) for label, seq in sched.items()}
    assert len(set(prefills.values())) >= 3, (
        f"expected at least three distinct scheduled-token sequences across "
        f"the conditions but saw {prefills}. The chunk boundaries the cache hit "
        f"was supposed to move did not move."
    )

    # The novel dependency: a peer's cache state must have changed the width of
    # the forward pass this rank ran, or that pair of conditions is vacuous.
    def pads(label):
        start, end = windows[label]
        return {
            r["padded"]
            for r in records
            if r["kind"] == "dp" and start <= r["t"] <= end and r["unpadded"] == 1
        }

    moved_padding = [
        (a, b, pads(a), pads(b))
        for a, b in (
            ("cold_peercold32", "cold_peerwarm32"),
            ("hitfull_peercold32", "hitfull_peerwarm32"),
        )
    ]
    assert any(pa and pb and pa != pb for _, _, pa, pb in moved_padding), (
        "the peers' cache state did not change this rank's padded token count "
        "on any single-token decode step: "
        f"{[(a, b, sorted(pa), sorted(pb)) for a, b, pa, pb in moved_padding]}. "
        "The SP round-up or the cudagraph bucket absorbed it, so the peer "
        "cache-state conditions ran identical shapes and prove nothing."
    )

    # --- the verdict --------------------------------------------------------
    base = conditions["cold_idle"]
    failures = []
    for label, out in conditions.items():
        if label == "cold_idle":
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
                f"{label} (hit {observed[label]}, chunks {prefills.get(label)}): "
                f"{len(moved)}/{len(base['logprobs'])} logprobs moved "
                f"(first at {moved[0]}), max |delta| {delta:.3e}"
            )
    assert not failures, (
        "the needle's logprobs depend on prefix-cache state under DP=4 x EP=4:\n  "
        + "\n  ".join(failures)
    )
