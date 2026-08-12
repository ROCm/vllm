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
- **deepep_low_latency**: brought up under the mode and asserted here. It
  forces `FusedMoEActivationFormat.BatchedExperts`
  (`FusedMoEParallelConfig.use_batched_activation_format`), whose only experts
  class on ROCm is `BatchedTritonExperts` -- so that one class decided whether
  this backend was reachable at all, and until it was measured it inherited
  `_supports_batch_invariance() -> False` and the oracle raised
  `NotImplementedError: No Unquantized MoE backend supports the deployment
  configuration`. It now declares `True` for the unquantized path on the
  evidence recorded in that method, and withholds its fp8 schemes as unmeasured
  under the mode, for the reason recorded in `_supports_quant_scheme`. This arm
  is therefore bf16 only, which is what OLMoE is.

  Measured: 0 of 32 logprobs moved while the needle rank's padded token count
  went 40 -> 220, reproduced over two runs. With the mode off the same
  comparison moved 31 of 32 (max |delta| 3.5e-2) at padding 40 -> 256, so the
  metric is not blind. This is the only arm whose experts class is
  `BatchedTritonExperts` rather than `TritonExperts`, and the
  (prepare_finalize, experts) pair is asserted so a silent fallback cannot
  pass it.
- **mori_high_throughput**: brought up under the mode and asserted here, but
  only after two things were fixed. `RoutedExperts.expert_map` used to hand the
  experts AITER's 0/1 mask whenever `VLLM_ROCM_USE_AITER_MOE` was set rather
  than when AITER was actually *selected*, which made the only previously
  reachable MoRI measurement (via `--moe-backend triton`) degenerate; and
  `FusedMoEConfig.__post_init__` asserted `rocm_aiter_fmoe_enabled` for MoRI,
  which is now neither an assert nor a warning. See that comment for why the
  coupling is a performance contract and not a data-format one. With both in
  place MoRI's
  IntraNode combine is bitwise invariant: 0 of 32 positions moved, against a
  mode-off control at the *same* load exposure (needle rank padding 40 -> 48 in
  both arms) that moved 32 of 32, max |delta| 3.6e-2. Its output is also
  bitwise identical to the allgather_reducescatter path on the same prompts
  (4 prompts x 24 logprobs plus the 32-token needle, max |delta| exactly 0).

  `--max-num-batched-tokens` is pinned low for this arm on purpose. MoRI
  allocates a receive buffer of `ep_size * max_num_batched_tokens` rows, and
  `TritonExperts` used to see M = 8192 on every step where the
  allgather_reducescatter arm on the same model ran M between 4 and 8192 over
  526 distinct values. It no longer does: `MoriPrepareAndFinalize.prepare`
  marks the undelivered rows invalid, so the expert GEMMs, the pad-aware
  activation and the pad-aware reduction skip them (mean
  `num_tokens_post_padded` 67662 -> 2866, against 2960 for
  allgather_reducescatter), and it hands the experts only the first
  `ep_size *` (this step's DP token count) rows, so M tracks the batch. What is
  still sized by `max_num_batched_tokens` is MoRI's own symmetric allocation,
  which is what the pin is for -- leaving the default would make this test
  memory-hungry for no extra coverage.

- **mori_low_latency**: not a second test. Both single-node MoRI variants were
  observed selecting `EpDispatchCombineKernelType.IntraNode`
  (`MoriAll2AllManager._make_all2all_kwargs` branches on `self.internode`, not
  on the backend literal), so the low-latency literal exercises the same
  kernel.

`deepep_low_latency` was recorded here as inadmissible for as long as it was,
rather than skipped silently, which is how the change that made it admissible
got noticed. Its fp8 schemes are still inadmissible and are still recorded
rather than skipped, for the same reason.

The third arm covers MoE **LoRA** on top of EP, which is a different failure
than a variant combine and is the regression this file's LoRA test exists for.
`PunicaWrapperGPU`'s LoRA metadata -- `no_lora_flag`, `active_lora_ids`,
`num_active_loras` -- describes the batch *this rank scheduled*, but under EP
the MoE LoRA runs on the all-gathered batch. A rank whose own requests carried
no adapter early-returned on `no_lora_flag` while it still owned experts
serving everyone else's tokens, and the LoRA delta for those tokens was
silently dropped. Which tokens lost it depended on what the rest of the cluster
was running, so a lightly loaded server produced the *wrong* answer and a busy
one the right one. Measured pre-fix at 32 of 32 logprobs moved *and the
generated text changing*, which is what reverting the two fixed files makes
this test report; post-fix 0 of 32. A mode-off control also moves 32 of 32 but
keeps its tokens (max |delta| 2.9e-1) -- that is the reduction-order drift the
rest of this file is about, not a delta that was never applied.

Only `allgather_reducescatter` is testable here, and that is not a choice:
`FusedMoEWithLoRA._ep_check` asserts that backend outright, so LoRA + DeepEP
and LoRA + MoRI cannot be brought up at all.

The LoRA arm runs `--enforce-eager`, also not a preference. Its vacuity guard
has to show that a rank with no LoRA tokens of its own was serving experts for
tokens that had them, which means reading `add_lora_fused_moe`'s view of the
metadata from Python -- and under cudagraphs that call stops executing on
replay, so the needle's steps go unrecorded. Measured: with cudagraphs the
needle window contained zero LoRA records, eager 1056 per rank. Both capture
the bug (pre-fix moved 32 of 32 either way), only eager can prove it did.

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
import time
from pathlib import Path

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
# The MoRI arm needs more. What the guard below wants is for the needle's
# prefill step to have company, and that arm runs the whole padded receive
# buffer through Triton every step -- it served roughly a third as many
# requests as the others in the same wall clock. At the shared default the
# needle's prefill got a step to itself in both conditions and the guard
# (correctly) refused the verdict; at this exposure it does not.
MORI_LOAD_CONCURRENCY = int(os.getenv("VLLM_EP_MORI_LOAD_CONCURRENCY", "96"))
# The LoRA arm is eager, so it serves fewer requests per second and needs more
# in flight to reach the same exposure.
LORA_LOAD_CONCURRENCY = int(os.getenv("VLLM_EP_LORA_LOAD_CONCURRENCY", "48"))
# The fp8 arms need more for the opposite reason to MoRI: fp8 weights and GEMMs
# make each step cheaper, so at the shared default the server drains the queue
# between the needle's steps and its prefill again gets a step to itself. The
# block arm was observed refusing its verdict on exactly that -- padded count 40
# alone and 40 loaded -- and passing at this exposure. Marginal either way is
# not good enough for a suite that wants gating status, so both fp8 arms are
# raised rather than only the one seen to fail.
FP8_LOAD_CONCURRENCY = int(os.getenv("VLLM_EP_FP8_LOAD_CONCURRENCY", "96"))
# The bf16 low-latency arm needs it for the same reason its fp8 siblings above
# do, and it was left on the shared default only because it had not been seen to
# fail there. It has now: running the file in order it refused its verdict on the
# same signature, padded count 40 alone and 40 loaded, twice out of two, while
# passing twice out of two when selected on its own -- the low-latency path is
# cheap enough per step that the server drains between the needle's steps, and it
# is the second server in the process, so its compile caches are warm and the
# needle returns sooner. At this exposure it passes in file order.
LL_LOAD_CONCURRENCY = int(os.getenv("VLLM_EP_LL_LOAD_CONCURRENCY", "96"))
LOAD_RAMP_SECONDS = float(os.getenv("VLLM_EP_LOAD_RAMP_SECONDS", "12"))
# The load must drag the needle rank's padded token count at least this far
# above what it ran alone, otherwise the needle saw identical shapes twice.
MIN_LOADED_PAD = 16

LORA_NAME = "synthetic-moe"
LORA_RANK = 16
# The alone window must contain at least this many ranks that ran the MoE LoRA
# on the gathered batch while their own metadata said they had none of it --
# the state the dropped-delta bug needed. All three non-needle ranks were
# measured in it; asking for two leaves headroom without ever accepting a
# verdict that rests on a single idle rank.
MIN_IDLE_LORA_RANKS = 2
# ...and the loaded window must contain none, or nearly none: it is the arm
# where every rank has its own LoRA tokens and therefore the arm that got the
# right answer pre-fix. Measured 0 of ~1080 calls on all four ranks.
MAX_LOADED_IDLE_FRACTION = 0.5

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
        # The quantization scheme is read off the experts *after* their
        # __init__, which is where `maybe_promote_act_quant_for_batch_invariance`
        # runs. So this records what will execute, not what the checkpoint
        # asked for -- the two differ under the mode and the fp8 arms assert
        # on the difference.
        quant = {}
        try:
            quant = dict(
                quant_dtype=str(fused_experts.quant_dtype),
                per_act_token_quant=bool(fused_experts.per_act_token_quant),
                block_shape=fused_experts.block_shape,
            )
        except Exception as e:  # pragma: no cover
            quant = {"quant_err": repr(e)}
        _emit(
            event="mk",
            pf=type(prepare_finalize).__name__,
            experts=type(fused_experts).__name__,
            **quant,
        )

    cls.__init__ = wrapper


def _patch_deepep_ll(module):
    """Record whether DeepEP LL dispatched fp8 or bf16.

    `use_fp8_dispatch` decides whether the low-latency kernels quantize inside
    `_do_quant` and carry scales through the buffers, or move bf16 and leave
    the experts to quantize. It is derived from the quant config in
    `maybe_make_prepare_finalize`, not from the checkpoint's dtype, so an fp8
    model can perfectly well dispatch bf16 -- and an arm that means to cover
    the fp8 dispatch has to check rather than assume.
    """
    cls = getattr(module, "DeepEPLLPrepareAndFinalize", None)
    if cls is None:
        return
    original_init = cls.__init__

    def wrapper(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _emit(event="deepep_ll", use_fp8_dispatch=bool(self.use_fp8_dispatch))

    cls.__init__ = wrapper


def _patch_punica(module):
    """Record what each rank believed about its own LoRA work, per MoE call.

    This is the mechanism guard for the LoRA arm. `x` is the all-gathered
    activation -- every rank's tokens -- while `token_mapping_meta` was built
    from the batch this rank scheduled, so `no_lora=True` is a rank that is
    about to run (pre-fix: skip) the MoE LoRA for other ranks' tokens. If no
    such rank existed while the needle ran, the arm never reached the state the
    fix is about and its verdict means nothing.

    Both reads are of CPU-side tensors that the metadata already maintains, so
    this costs no device sync and does not perturb the schedule.
    """
    cls = getattr(module, "PunicaWrapperGPU", None)
    if cls is None:
        return
    original = cls.add_lora_fused_moe

    def wrapper(self, y, x, *args, **kwargs):
        try:
            meta = self.token_mapping_meta
            _emit(
                event="lora_moe",
                rows=int(x.size(0)),
                no_lora=bool(meta.no_lora_flag_cpu[0].item()),
                n_active=int(meta.num_active_loras_cpu[0].item()),
            )
        except Exception as e:  # pragma: no cover
            _emit(event="lora_moe_err", err=repr(e))
        return original(self, y, x, *args, **kwargs)

    cls.add_lora_fused_moe = wrapper


_TARGETS = {
    "vllm.v1.worker.dp_utils": _patch_dp_utils,
    "vllm.distributed.device_communicators.all2all": _patch_all2all,
    "vllm.model_executor.layers.fused_moe.modular_kernel": _patch_modular_kernel,
    "vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll": (
        _patch_deepep_ll
    ),
    "vllm.lora.punica_wrapper.punica_gpu": _patch_punica,
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


def _needle(url, model: str = MODEL) -> dict:
    started = time.time()
    response = dp_completion(
        url, model, NEEDLE_PROMPT, NEEDLE_MAX_TOKENS, NEEDLE_RANK, logprobs=1
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


def _lora_calls_by_rank(records: list[dict], needle: dict) -> dict[int, dict]:
    """Per-DP-rank MoE-LoRA activity during one needle request.

    `no_lora` counts the calls where this rank's own scheduled batch carried no
    adapter -- the state that used to skip the kernel for the gathered batch.
    """
    pid_rank = {
        r["pid"]: r["dp_rank"]
        for r in records
        if r.get("event") == "dp" and r.get("dp_rank") is not None
    }
    tally: dict[int, dict] = {}
    for r in records:
        if r.get("event") != "lora_moe":
            continue
        if not needle["started"] <= r["t"] <= needle["finished"]:
            continue
        rank = pid_rank.get(r["pid"])
        if rank is None:
            continue
        entry = tally.setdefault(rank, {"calls": 0, "no_lora": 0, "rows": set()})
        entry["calls"] += 1
        entry["no_lora"] += int(r["no_lora"])
        entry["rows"].add(r["rows"])
    return tally


def _write_synthetic_moe_adapter(directory: Path) -> Path:
    """A random rank-16 LoRA on every expert projection of `MODEL`.

    Synthetic on purpose. The property under test is that a token's LoRA delta
    is applied identically whatever else the cluster is doing -- bitwise
    invariance, not output quality -- and these weights are noise, so the
    generations they produce mean nothing and are never inspected.

    `lora_B` is *not* zero initialised as PEFT leaves it: a zero B makes the
    delta identically zero, the base model answers, and the test would pass
    without the MoE LoRA kernels ever having mattered.
    """
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL)
    hidden, inter = config.hidden_size, config.intermediate_size
    torch.manual_seed(1234)

    tensors: dict[str, torch.Tensor] = {}

    def add(name: str, fan_in: int, fan_out: int) -> None:
        tensors[f"{name}.lora_A.weight"] = (
            torch.randn(LORA_RANK, fan_in, dtype=torch.float32) * fan_in**-0.5
        ).to(torch.bfloat16)
        tensors[f"{name}.lora_B.weight"] = (
            torch.randn(fan_out, LORA_RANK, dtype=torch.float32) * 0.02
        ).to(torch.bfloat16)

    for layer in range(config.num_hidden_layers):
        for expert in range(config.num_experts):
            prefix = f"base_model.model.model.layers.{layer}.mlp.experts.{expert}"
            add(f"{prefix}.gate_proj", hidden, inter)
            add(f"{prefix}.up_proj", hidden, inter)
            add(f"{prefix}.down_proj", inter, hidden)

    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": MODEL,
                "r": LORA_RANK,
                "lora_alpha": 2 * LORA_RANK,
                "lora_dropout": 0.0,
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": True,
                "target_modules": ["gate_proj", "up_proj", "down_proj"],
                "modules_to_save": None,
            }
        )
    )
    return directory


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
def deepep_ll_server(tmp_path, enable_batch_invariant_mode):
    """A DP=4 + EP server on the DeepEP low-latency all2all.

    Function-scoped for the same reason as `deepep_ht_server`.

    This is the only arm whose experts class is `BatchedTritonExperts` rather
    than `TritonExperts`: DeepEP LL forces the batched activation format, and
    the assertion below is what keeps that from silently falling back.
    """
    yield from _ep_server(tmp_path, "deepep_low_latency")


@pytest.fixture
def deepep_ll_fp8_server(tmp_path, enable_batch_invariant_mode):
    """DeepEP LL serving OLMoE under online per-tensor fp8.

    `--quantization fp8` gives `_Fp8OnlineMoEBase` a per-tensor weight key with
    `kFp8DynamicTensorSym` activations, which under the mode
    `maybe_promote_act_quant_for_batch_invariance` turns into per-token. That
    is the one row in `BatchedTritonExperts._supports_quant_scheme` no
    checkpoint produces, and it is reachable only this way.
    """
    yield from _ep_server(tmp_path, "deepep_low_latency", ["--quantization", "fp8"])


@pytest.fixture
def deepep_ll_fp8_block_server(tmp_path, enable_batch_invariant_mode):
    """DeepEP LL serving OLMoE under online block fp8.

    Not a duplicate of the arm above: this is the only way to reach DeepEP
    LL's *fp8 dispatch*. `use_fp8_dispatch` is
    `quant_dtype == fp8 and block_shape == DEEPEP_QUANT_BLOCK_SHAPE`
    ([128, 128]), so the per-tensor arm dispatches bf16 and leaves the experts
    to quantize, while this one quantizes inside `_do_quant` and carries the
    scales through the low-latency buffers.
    """
    yield from _ep_server(
        tmp_path, "deepep_low_latency", ["--quantization", "fp8_per_block"]
    )


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


@pytest.fixture
def lora_ep_server(tmp_path, enable_batch_invariant_mode):
    """A DP=4 + EP server serving a synthetic MoE LoRA over AllGather+ReduceScatter.

    Function-scoped for the same reason as `deepep_ht_server`.

    The backend is not a choice: `FusedMoEWithLoRA._ep_check` asserts
    `allgather_reducescatter`, so this is the only all2all LoRA can run on.
    `--enforce-eager` is what makes the mechanism guard readable; see the
    module docstring.
    """
    adapter = _write_synthetic_moe_adapter(tmp_path / "moe_lora")
    yield from _ep_server(
        tmp_path,
        "allgather_reducescatter",
        [
            "--enable-lora",
            "--lora-modules",
            f"{LORA_NAME}={adapter}",
            "--max-lora-rank",
            str(LORA_RANK),
            "--max-num-batched-tokens",
            "2048",
            "--enforce-eager",
        ],
    )


def _assert_a_rank_served_lora_it_did_not_schedule(
    records: list[dict], alone: dict, loaded: dict
) -> None:
    """The LoRA arm's vacuity guard: it must reach the state the bug needed.

    Load alone is not enough here. The dropped delta needed a rank holding *no
    LoRA tokens of its own* while it owned experts for a gathered batch that
    had them -- so the alone window has to contain such a rank, and the loaded
    window has to not, or the two conditions differ in exposure but not in the
    thing that broke and the comparison proves nothing.
    """
    alone_lora = _lora_calls_by_rank(records, alone)
    loaded_lora = _lora_calls_by_rank(records, loaded)

    assert set(alone_lora) == set(range(DP)), (
        f"MoE-LoRA calls were recorded on ranks {sorted(alone_lora)}, not all "
        f"{DP}. Either the adapter never reached the forward path or the "
        "instrumentation did not load -- note it goes quiet under cudagraphs, "
        "which is why this arm is eager."
    )

    needle_rows = alone_lora[NEEDLE_RANK]["rows"]
    same_rows = {rank for rank, d in alone_lora.items() if d["rows"] == needle_rows}
    assert same_rows == set(range(DP)), (
        "the ranks did not run the MoE LoRA over the same activation sizes "
        f"as rank {NEEDLE_RANK} "
        f"({ {r: sorted(d['rows']) for r, d in alone_lora.items()} }), so they "
        "were not all working on one all-gathered batch and an idle rank would "
        "not have been holding anybody else's tokens."
    )

    idle = sorted(
        rank
        for rank, d in alone_lora.items()
        if rank != NEEDLE_RANK and d["calls"] and d["no_lora"] == d["calls"]
    )
    assert len(idle) >= MIN_IDLE_LORA_RANKS, (
        f"only ranks {idle} ran every MoE-LoRA call with no LoRA token of "
        f"their own while the needle ran alone "
        f"({ {r: (d['no_lora'], d['calls']) for r, d in alone_lora.items()} } "
        "as no_lora/calls). That is the state the dropped-delta bug needed, so "
        "without it this arm never tested the fix."
    )
    assert alone_lora[NEEDLE_RANK]["no_lora"] < alone_lora[NEEDLE_RANK]["calls"], (
        f"rank {NEEDLE_RANK} scheduled no LoRA token either, so the needle "
        f"request did not use the '{LORA_NAME}' adapter at all."
    )

    busiest = max(
        (d["no_lora"] / d["calls"], rank)
        for rank, d in loaded_lora.items()
        if d["calls"]
    )
    assert busiest[0] <= MAX_LOADED_IDLE_FRACTION, (
        f"rank {busiest[1]} still had no LoRA work of its own for "
        f"{busiest[0]:.0%} of its MoE-LoRA calls under load "
        f"({ {r: (d['no_lora'], d['calls']) for r, d in loaded_lora.items()} } "
        "as no_lora/calls). Both windows are then in the same state and the "
        "comparison cannot see a delta that is dropped in one of them. Raise "
        "VLLM_EP_LORA_LOAD_CONCURRENCY."
    )


def _assert_needle_does_not_see_the_batch(
    server,
    log_prefix: str,
    *,
    manager_cls: str,
    prepare_finalize_cls: str,
    experts_cls: str,
    load_concurrency: int = LOAD_CONCURRENCY,
    model: str = MODEL,
    require_idle_lora_rank: bool = False,
    expect_fp8_dispatch: bool | None = None,
    expect_quant: dict | None = None,
) -> None:
    """The needle's logprobs must not move when the rest of the server does."""
    url = server.url_for("v1/completions")

    # Discarded: keeps first-request state out of the comparison.
    _needle(url, model)

    with _load(url, 0, model):
        alone = _needle(url, model)
    with _load(url, load_concurrency, model) as load:
        loaded = _needle(url, model)
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

    if expect_quant is not None:
        # What the experts will actually run, read after their __init__ and so
        # after any batch-invariance promotion. Asserted rather than assumed:
        # the whole reason an fp8 arm exists is that it runs a different
        # scheme from the bf16 arms, and a checkpoint that silently loaded
        # unquantized would otherwise pass by re-measuring those.
        observed = {
            tuple(sorted((k, repr(r.get(k))) for k in expect_quant))
            for r in records
            if r.get("event") == "mk"
        }
        want = {tuple(sorted((k, repr(v)) for k, v in expect_quant.items()))}
        assert observed == want, (
            f"the experts were built with {observed}, not {want}. Under the "
            "mode the scheme here is the *promoted* one, not the "
            "checkpoint's."
        )

    if expect_fp8_dispatch is not None:
        dispatch = {
            r["use_fp8_dispatch"] for r in records if r.get("event") == "deepep_ll"
        }
        assert dispatch == {expect_fp8_dispatch}, (
            f"DeepEP LL reported use_fp8_dispatch={dispatch}, expected "
            f"{{{expect_fp8_dispatch}}}. It is derived from the quant config "
            "(fp8 dtype *and* a [128, 128] block shape), so a per-tensor fp8 "
            "checkpoint legitimately dispatches bf16 -- but which one ran "
            "decides which code this arm covered."
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

    if require_idle_lora_rank:
        _assert_a_rank_served_lora_it_did_not_schedule(records, alone, loaded)

    assert loaded["tokens"] == alone["tokens"], (
        f"the needle sampled different tokens once the server was busy: "
        f"{alone['tokens']} vs {loaded['tokens']}"
    )
    moved = [
        i
        for i, (a, b) in enumerate(zip(alone["logprobs"], loaded["logprobs"]))
        if a != b
    ]
    # Printed on success too: the docstrings above quote these numbers, and a
    # passing run is the only place the exposure actually achieved is visible.
    print(
        f"\n{prepare_finalize_cls}/{experts_cls}: {len(moved)}/"
        f"{len(alone['logprobs'])} logprobs moved, needle-rank padding "
        f"{max(alone_pads)} -> {max(loaded_pads)}"
    )
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


def test_deepep_low_latency_combine_does_not_see_the_batch(deepep_ll_server):
    """The needle keeps its logprobs on the batched-format experts path.

    This is the end-to-end half of the `BatchedTritonExperts` certification.
    The kernel-level evidence is in `_supports_batch_invariance`; what only a
    server can show is that the pieces around it -- DeepEP LL's atomic slot
    assignment, its weighted combine, and the token count each expert is handed
    varying with the cluster's load -- do not reintroduce the dependence.

    `experts_cls` is asserted because the whole point of this arm is the
    batched class: a fallback to `TritonExperts` would pass while measuring the
    backend that `test_deepep_high_throughput_combine_does_not_see_the_batch`
    already covers.
    """
    server, log_prefix = deepep_ll_server
    _assert_needle_does_not_see_the_batch(
        server,
        log_prefix,
        manager_cls="DeepEPLLAll2AllManager",
        prepare_finalize_cls="DeepEPLLPrepareAndFinalize",
        experts_cls="BatchedTritonExperts",
        load_concurrency=LL_LOAD_CONCURRENCY,
    )


def test_deepep_low_latency_fp8_promotion_engages_end_to_end(deepep_ll_fp8_server):
    """The fp8 batched-experts path, end to end, with the promotion asserted.

    Named for what it establishes, which is that the path engages -- not for
    batch invariance, which it cannot show. **These two fp8 arms are one-sided
    guards**: they have no admissible mode-off control (see below), so passing
    is not evidence that the mode achieved anything, while *failing* would be
    real evidence of a batch dependence. Keep them for the second half. The
    sibling bf16 arms in this file are the two-sided ones, and the difference
    is deliberate rather than an oversight to be tidied away by renaming these
    back to `..._does_not_see_the_batch`.

    `BatchedTritonExperts`' fp8 schemes are admitted under the mode on
    kernel-level and whole-class evidence recorded in
    `_supports_batch_invariance`; what only a server shows is the pieces
    around them. Here that is the promotion in particular: the checkpoint asks
    for dynamic per-tensor activations, whose scale is an amax over whatever
    the all2all delivered, and the mode replaces it with a per-token scale
    before anything reads the scheme. `expect_quant` asserts the promoted
    form rather than trusting it, because a per-tensor scale that survived to
    the experts is precisely the batch dependence this arm would otherwise be
    reporting as absent.

    `use_fp8_dispatch` is False here and that is correct, not a fallback: it
    requires a [128, 128] block shape. The `..._fp8_block_dispatch_engages...`
    arm below is the one that covers the fp8 dispatch.

    **This arm has no mode-off control, and cannot have one in the usual
    shape.** Measured: 0 of 32 logprobs moved with the needle rank's padding
    going 40 -> 112. The control was attempted three times and the same
    configuration with `VLLM_BATCH_INVARIANT=0` does not produce comparable
    numbers -- it produces `nan`, which the API rejects with
    `BadRequestError: Out of range float values are not JSON compliant: nan`.
    Seen in 2 of 2 mode-off runs that got that far, once on the loaded needle
    and once on the idle one, and never in any mode-on run.

    An earlier version of this docstring attributed the NaN to the activation
    scale, since with the mode off it stays a dynamic per-tensor amax taken
    over the whole `E x max_num_tokens` dispatch buffer -- including rows
    DeepEP LL never delivered -- while under the mode it is per-token and an
    undelivered row can only poison itself. **That attribution has since been
    measured and excluded, and both halves are worth recording.**

    The scale defect is real: device-side, in all 16 of OLMoE's MoE layers,
    the buffer amax is achieved on an *undelivered* row, so the scale is never
    set by the data being quantized, with inflation up to 22.75x on serving
    traffic. The undelivered rows are not uninitialised -- one DeepEP LL
    buffer is shared by every layer, so they hold whichever layer last
    dispatched the largest activation (40.75, identical across all 16 layers).

    But it does not cause this. Replacing the scale with an amax over
    delivered rows only, as a static scale, changing nothing else, leaves the
    NaN exactly as it was: 6 of 6 requests after the first. No NaN or Inf ever
    enters the quantizer either.

    What the NaN was: it was in the forward pass and not in logprob
    serialization (with `logprobs` omitted the same request returned 200 and
    garbage text), it was in prefill, and it was **one-way persistent** -- the
    first request after startup always clean and every request after it NaN,
    in 4 of 4 server runs, with no load required. That signature points at
    persistent state rather than at arithmetic on one batch.

    **It no longer reproduces (2026-08-09).** The `--enforce-eager` control
    this docstring called for was finally run, together with the arm itself,
    two servers each: 8 of 8 logprob requests clean on all four, with and
    without eager. `first_request_ok` true everywhere, so none of them is the
    dead-server case that looks the same.

    The probe was positive-controlled rather than trusted: injecting a NaN into
    `compute_logits` produced `Out of range float values are not JSON
    compliant: nan` on 8 of 8 requests, the same string this arm used to fail
    with. So the clean result is a real negative and not a blind instrument.

    The likely cause is `8b5db7cb6d`, which fixed a **capture-only** defect
    with exactly this signature: `batched_moe_kernel_quantize_input` ignored
    `expert_num_tokens` under capture and amaxed the whole
    `[E, max_num_tokens, N]` buffer, measured at 9.95e29 on 100% of calls,
    while eager bounded it per expert. That is stated as the likely cause, not
    a demonstrated one -- nobody bisected it, and "does not reproduce" is
    weaker evidence than "reproduced, then fixed".

    The mode-on arm above still has no mode-off control in the usual shape,
    which remains the thing a reader needs to know.
    """
    server, log_prefix = deepep_ll_fp8_server
    _assert_needle_does_not_see_the_batch(
        server,
        log_prefix,
        manager_cls="DeepEPLLAll2AllManager",
        prepare_finalize_cls="DeepEPLLPrepareAndFinalize",
        experts_cls="BatchedTritonExperts",
        expect_fp8_dispatch=False,
        expect_quant={
            "quant_dtype": "torch.float8_e4m3fn",
            "per_act_token_quant": True,
            "block_shape": None,
        },
        load_concurrency=FP8_LOAD_CONCURRENCY,
    )


def test_deepep_low_latency_fp8_block_dispatch_engages_end_to_end(
    deepep_ll_fp8_block_server,
):
    """The same, on the path where DeepEP LL itself quantizes.

    With a [128, 128] block shape the low-latency kernels dispatch fp8: the
    scales are produced in `_do_quant` and carried through the buffers, so the
    experts receive quantized activations they did not quantize. That is code
    no other arm in this suite runs, and it is the half of the fp8 admission
    that the whole-class sweep -- which goes through the no-comms
    `BatchedPrepareAndFinalize` -- could not reach.

    No promotion here: a block-quantized activation scale is per (token,
    k-tile) already, so `maybe_promote_act_quant_for_batch_invariance` leaves
    it alone. `per_act_token_quant` is therefore False and the block shape is
    what carries the granularity.

    **This arm's mode-off control runs, and says the metric is blind.**
    Measured: 0 of 32 moved under the mode at padding 40 -> 198; with
    `VLLM_BATCH_INVARIANT=0` at a *higher* exposure (40 -> 224) the same
    comparison moved 1 of 32, max |delta| 6.0e-8. Against the bf16 arms, where
    mode-off moves 31 of 32 at 3.5e-2, this configuration is already very
    nearly batch invariant without the mode, so a passing verdict here is not
    evidence that the mode is doing anything. Whether that is because block
    quantization coarsens the arithmetic below the logprob quantum or because
    the path genuinely has no batch dependence left is **not established** --
    the two look identical from here, and only the second would license the
    arm.

    Kept rather than skipped, because what it does establish is mechanical and
    is asserted above: DeepEP LL really did dispatch fp8, the batched experts
    really did consume it, and the pair did not fall back. The invariance
    verdict is the part that is uncontrolled.
    """
    server, log_prefix = deepep_ll_fp8_block_server
    _assert_needle_does_not_see_the_batch(
        server,
        log_prefix,
        manager_cls="DeepEPLLAll2AllManager",
        prepare_finalize_cls="DeepEPLLPrepareAndFinalize",
        experts_cls="BatchedTritonExperts",
        expect_fp8_dispatch=True,
        expect_quant={
            "quant_dtype": "torch.float8_e4m3fn",
            "per_act_token_quant": False,
            "block_shape": [128, 128],
        },
        load_concurrency=FP8_LOAD_CONCURRENCY,
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


def test_moe_lora_delta_does_not_see_the_batch(lora_ep_server):
    """A LoRA'd token keeps its delta whatever the other EP ranks are serving.

    Regression test for the rank-local LoRA metadata being used to decide
    whether to run the MoE LoRA on the all-gathered batch: an idle rank skipped
    it and dropped the delta for every gathered token routed to the experts it
    owned. Pre-fix this arm moved 32 of 32 logprobs and changed the generated
    text; post-fix 0 of 32.

    The adapter is synthetic (random weights over every expert projection), so
    what it generates is meaningless -- the claim is bitwise invariance of the
    delta, not quality. `_assert_a_rank_served_lora_it_did_not_schedule` is
    what keeps that claim from being vacuous.
    """
    server, log_prefix = lora_ep_server
    _assert_needle_does_not_see_the_batch(
        server,
        log_prefix,
        manager_cls="AgRsAll2AllManager",
        prepare_finalize_cls="MoEPrepareAndFinalizeNaiveDPEPModular",
        experts_cls="TritonExperts",
        load_concurrency=LORA_LOAD_CONCURRENCY,
        model=LORA_NAME,
        require_idle_lora_rank=True,
    )
