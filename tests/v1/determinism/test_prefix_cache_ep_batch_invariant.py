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

1. **The needle's own cache state.** The same prompt is run cold, and at hits
   of 400, 1600 and 2384 tokens (a full-prompt hit block-aligns to 2384). Under
   the mode the scheduler caps a prefill chunk at
   `batch_invariant_prefill_chunk` so the chunk *length* is a function of the
   request alone -- but the chunk *origin* is not, because chunks start at the
   hit length. At the settings below the cap is 1984, so cold prefills as
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

   The direction is not the obvious one: peers running cold prefills induce no
   padding at all, because a prefill-heavy step is not cudagraph-eligible and
   `should_dp_pad` is False. Served from cache the peers drop into uniform
   decode, cudagraphs engage, and the needle is padded out. The peers' cache
   state controls both *whether* DP padding fires and how wide it goes.

The repeated baseline condition is what makes the rest mean something: it is
clean with the mode off as well, so the mode-off differences track the cache and
the batch rather than run-to-run noise.

Each condition uses a **unique `cache_salt`**. The salt enters only the block
hash (`kv_cache_utils._gen_extra_hash_keys`, first block only, and the chain
carries it forward), never the model input, so the needle's token ids are
byte-identical in every condition while its hit length is under the test's
control. Without that, conditions would contaminate each other -- the first run
of the needle would warm the prefix for all the later ones -- and there would be
no way to obtain a cold arm on a server that has already seen the prompt.

With caching on it is easy to build a comparison in which the needle takes an
identical code path on both sides, which passes while asserting nothing.
`test_prefix_cache_and_ep` therefore fails
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

import os
import random
import time
import uuid
import zlib

import pytest
import requests
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

_PATCHERS = '''
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
'''

_INSTRUMENTATION = _PATCHERS + INSTRUMENTATION_IMPORT_HOOK


def _completion(url, prompt, max_tokens, rank, salt, request_id, logprobs=None):
    body = {"cache_salt": salt}
    if logprobs is not None:
        # The needle must emit exactly max_tokens in every condition, or an
        # early stop would silently truncate the comparison instead of failing.
        body["ignore_eos"] = True
    return dp_completion(
        url,
        MODEL,
        prompt,
        max_tokens,
        rank,
        logprobs=logprobs,
        timeout=1200,
        extra_body=body,
        extra_headers={"X-Request-Id": request_id},
    )


def _peer_load(url, ranks, concurrency, mode="cold", seed=0, warm_prompt=None):
    """Companion load whose *cache state* is the only thing `mode` varies.

    "cold" sends a fresh random prompt every time so every peer prefill is a
    full miss, "warm" sends one pre-warmed prompt so every peer prefill is a
    near-total hit. Same concurrency, same prompt length, same ranks.
    """

    def send(rng, index):
        if mode == "warm":
            prompt, salt = warm_prompt, "peer-warm-shared"
        else:
            prompt = [rng.randint(1000, 30000) for _ in range(PEER_TOKENS)]
            salt = f"peer-cold-{uuid.uuid4().hex}"
        _completion(
            url,
            prompt,
            PEER_MAX_TOKENS,
            ranks[index % len(ranks)],
            salt,
            uuid.uuid4().hex,
        )

    def prewarm():
        # Every rank the load touches must already hold the shared prefix, or
        # the first requests race each other and only partly hit.
        if mode == "warm":
            for rank in ranks:
                _completion(
                    url, warm_prompt, 1, rank, "peer-warm-shared", uuid.uuid4().hex
                )

    return BackgroundLoad(
        send,
        concurrency=concurrency,
        ramp_seconds=RAMP_SECONDS,
        drain_seconds=DRAIN_SECONDS,
        join_timeout=600,
        seed=seed,
        prepare=prewarm,
    )


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
    env = instrumented_server_env(
        tmp_path, _INSTRUMENTATION, PC_LOG=log_prefix, PC_DP_RANK=str(NEEDLE_RANK)
    )
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
    for label, hit_target, concurrency, ranks, mode in plan:
        # A unique salt per condition: same tokens, different block hashes, so
        # conditions cannot warm each other and a cold arm stays available.
        salt = f"{label}-{uuid.uuid4().hex}"
        with _peer_load(
            url,
            ranks,
            concurrency,
            mode,
            zlib.crc32(label.encode()) % 97,
            peer_warm_prompt,
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
        load.assert_ran_cleanly(f"{label} companions")
        choice = response["choices"][0]
        conditions[label] = {
            "tokens": choice["logprobs"]["tokens"],
            "logprobs": choice["logprobs"]["token_logprobs"],
            "cached": (response["usage"].get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
        }

    records = read_records(log_prefix)
    assert_server_ran_this_tree(
        {r["vllm_file"] for r in records}, {r["batch_invariant"] for r in records}
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
