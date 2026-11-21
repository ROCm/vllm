# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hot-expert replicas with a per-step assignment for decode-size MoE calls.

With expert parallelism every rank waits at the MoE reduction for the rank
whose experts cost the most in the step. In a decode-size call an expert's cost
is reading its weights, once per block of its routes, so a rank's cost follows
the number of distinct experts its routes select, which varies from step to
step. Splitting an expert's routes between two holders (as a per-route replica
choice does) makes both read its weights; a replica helps only when the expert
moves whole.

Each rank here also holds copies of `R` other ranks' experts per MoE layer,
chosen once from an expert load profile (`plan_layer`), in separate replica
weights. In every compact-expert call (`ops/compact_moe.py`) the routing kernel
computes from the step's routes, identically in every program of every rank,
which replicated experts move to their copy, and each rank computes all routes
of the experts assigned to it. Every route is computed exactly once, by one
rank, with the expert's own weights; only the rank whose partial sum carries
it changes. Calls that do not take the compact experts run on the owners.

Configuration (`--additional-config`):

* `decode_expert_replicas`: copies per rank and MoE layer (default 0: off).
  Each copy is one expert of every MoE layer, allocated after the weights and
  so taken from the KV cache.
* `decode_expert_load_profile`: a `.npy` float array `[num_hidden_layers,
  num_experts]` of non-negative expert selection loads by model layer (rows of
  layers without routed experts are ignored). Only the order and relative size
  within a layer matter.

A profile can be built from the server's own routing: serve without replicas
and with `--enable-return-routed-experts`, send representative requests, save
the JSON responses one per line, and sum their `routed_experts` with
`tools/expert_load_profile.py`.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger

from .ops.compact_moe import (
    REPLICA_MAX_RANKS,
    ReplicaTable,
    replica_table,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator

    from .model import MuseSparkForCausalLM

logger = init_logger(__name__)

REPLICAS_KEY = "decode_expert_replicas"
PROFILE_KEY = "decode_expert_load_profile"


@dataclass(frozen=True)
class DecodeReplicaConfig:
    replicas: int
    # [num_hidden_layers, num_experts] float64 expert loads.
    profile: np.ndarray
    profile_path: str

    @classmethod
    def from_vllm_config(
        cls, vllm_config: "VllmConfig"
    ) -> "DecodeReplicaConfig | None":
        """The replica options of `additional_config`, validated against the
        model and parallel configuration; None when replicas are off."""
        extra = vllm_config.additional_config
        if not isinstance(extra, dict):
            return None
        replicas = extra.get(REPLICAS_KEY, 0)
        path = extra.get(PROFILE_KEY)
        if not isinstance(replicas, int) or isinstance(replicas, bool) or replicas < 0:
            raise ValueError(f"{REPLICAS_KEY} must be a non-negative integer.")
        if replicas == 0:
            if path is not None:
                raise ValueError(f"{PROFILE_KEY} is set but {REPLICAS_KEY} is 0.")
            return None
        if not isinstance(path, str) or not path:
            raise ValueError(f"{REPLICAS_KEY} needs {PROFILE_KEY} (a .npy path).")

        parallel = vllm_config.parallel_config
        world = parallel.tensor_parallel_size * parallel.data_parallel_size
        if not parallel.enable_expert_parallel or world < 2:
            raise ValueError(f"{REPLICAS_KEY} needs expert parallelism over 2+ ranks.")
        if world > REPLICA_MAX_RANKS:
            raise ValueError(
                f"{REPLICAS_KEY} supports up to {REPLICA_MAX_RANKS} expert-parallel "
                f"ranks, got {world}."
            )
        if parallel.enable_eplb:
            raise ValueError(f"{REPLICAS_KEY} cannot be combined with EPLB.")
        if vllm_config.kernel_config.monokernel:
            raise ValueError(
                f"{REPLICAS_KEY} applies to the layer-wise decode MoE and cannot "
                "be combined with the monokernel."
            )
        config = vllm_config.model_config.hf_text_config
        num_layers, num_experts = config.num_hidden_layers, config.num_local_experts
        if num_experts % world or replicas > num_experts // world:
            raise ValueError(
                f"{REPLICAS_KEY}={replicas} exceeds the {num_experts // world} "
                "experts each rank owns."
            )
        try:
            profile = np.load(path)
        except (OSError, ValueError) as err:
            raise ValueError(f"cannot read {PROFILE_KEY} {path!r}: {err}") from err
        if profile.shape != (num_layers, num_experts):
            raise ValueError(
                f"{PROFILE_KEY} has shape {profile.shape}; expected "
                f"({num_layers}, {num_experts}) (layers, experts)."
            )
        profile = profile.astype(np.float64)
        if not np.isfinite(profile).all() or (profile < 0).any():
            raise ValueError(f"{PROFILE_KEY} must be finite and non-negative.")
        return cls(replicas, profile, path)


@dataclass
class LayerDecodeReplicas:
    table: ReplicaTable
    # The step's `[E]` local index (see `compact_moe_experts`): the rank's own
    # experts at rest; each call rewrites the replicated experts' entries.
    step_index: torch.Tensor
    # The replica slots' expert tensors (w13, w2, w13 scales, w2 scales), in
    # slot order, in the local experts' layouts.
    weights: tuple[torch.Tensor, ...]


def plan_layer(
    load: np.ndarray, world: int, replicas: int
) -> list[tuple[int, int, int]]:
    """`(expert, copy rank, copy slot)` for the `world * replicas` most loaded
    experts, hottest first: each is copied to the non-owner rank with a free slot
    and the least expected load (owned load plus half of each copy placed
    there). Linear placement (rank r owns experts `[r E/P, (r + 1) E/P)`)."""
    per = load.shape[0] // world
    rank_load = load.reshape(world, per).sum(1).astype(np.float64)
    free = [replicas] * world
    plan = []
    for e in np.argsort(-load, kind="stable")[: world * replicas]:
        if load[e] <= 0:
            break
        owner = int(e) // per
        cand = [r for r in range(world) if r != owner and free[r] > 0]
        if not cand:
            continue
        r = min(cand, key=lambda x: rank_load[x])
        plan.append((int(e), r, replicas - free[r]))
        free[r] -= 1
        rank_load[r] += load[e] / 2
        rank_load[owner] -= load[e] / 2
    return plan


def layer_table(
    plan: list[tuple[int, int, int]],
    rank: int,
    world: int,
    num_experts: int,
    replicas: int,
    device: torch.device,
) -> ReplicaTable:
    local = num_experts // world
    owner = [e // local for e in range(num_experts)]
    copies = sorted((s, e) for e, c, s in plan if c == rank)
    assert [s for s, _ in copies] == list(range(len(copies)))
    slot_expert = list(range(rank * local, (rank + 1) * local)) + [e for _, e in copies]
    slot_expert += [-1] * (replicas - len(copies))
    entries = [(e, c, local + s) for e, c, s in plan]
    return replica_table(owner, entries, rank, world, slot_expert, device)


def _replicate(
    tensors: list[torch.Tensor],
    local: int,
    plan: list[tuple[int, int, int]],
    rank: int,
    world: int,
    replicas: int,
    group: "GroupCoordinator",
) -> list[torch.Tensor]:
    """For each `[local * rows, ...]` expert tensor (one expert's rows
    contiguous), a `[replicas * rows, ...]` tensor of this rank's copies in slot
    order (every rank calls it: one all_to_all per tensor).

    The exchange uses the group's existing device communicator when it has
    one, so that no further communicator (and its device buffers) is created
    for a one-time copy."""
    import torch.distributed as dist

    pynccl = getattr(group.device_communicator, "pynccl_comm", None)
    if pynccl is not None and pynccl.disabled:
        pynccl = None
    sends = sorted((c, s, e) for e, c, s in plan if e // local == rank)
    send_counts = [sum(1 for c, _, _ in sends if c == dst) for dst in range(world)]
    recv_slots: list[int] = []
    recv_counts = []
    for src in range(world):
        slots = sorted(s for e, c, s in plan if c == rank and e // local == src)
        recv_slots += slots
        recv_counts.append(len(slots))
    out = []
    for t in tensors:
        # Byte views: FP4 / E8M0 storages have no gather or scatter kernels.
        flat = t.view(torch.uint8).reshape(local, -1)
        per = flat.shape[1]
        idx = torch.tensor([e - rank * local for _, _, e in sends], device=t.device)
        send = flat[idx].contiguous() if sends else flat[:0].contiguous()
        recv = torch.empty(sum(recv_counts), per, dtype=torch.uint8, device=t.device)
        out_sizes = [n * per for n in recv_counts]
        in_sizes = [n * per for n in send_counts]
        if pynccl is not None:
            pynccl.all_to_all_single(recv.view(-1), send.view(-1), out_sizes, in_sizes)
        else:
            dist.all_to_all_single(
                recv.view(-1),
                send.view(-1),
                output_split_sizes=out_sizes,
                input_split_sizes=in_sizes,
                group=group.device_group,
            )
        copies = torch.zeros((replicas, per), dtype=torch.uint8, device=t.device)
        if recv_slots:
            copies[torch.tensor(recv_slots, device=t.device)] = recv
        shape = list(t.shape)
        shape[0] = shape[0] // local * replicas
        out.append(copies.view(t.dtype).view(shape))
    return out


def attach_decode_replicas(model: "MuseSparkForCausalLM") -> None:
    """Give every MoE layer of `model` that runs the native-routes decode path
    its replica copies (`moe.decode_replicas`), when replicas are configured."""
    config = model.decode_replica_config
    if config is None:
        return
    from vllm.distributed import get_ep_group

    from .model import MuseSparkMoE

    ep = get_ep_group()
    rank, world = ep.rank_in_group, ep.world_size
    nbytes = 0
    layers = 0
    for layer in model.model.layers[model.model.start_layer : model.model.end_layer]:
        moe = getattr(layer, "feed_forward", None)
        if not isinstance(moe, MuseSparkMoE) or moe.decode_moe is None:
            continue
        routed = moe.experts.routed_experts
        num_experts = moe.n_logical_experts
        local = num_experts // world
        linear = torch.full((num_experts,), -1, dtype=torch.int32)
        linear[rank * local : (rank + 1) * local] = torch.arange(
            local, dtype=torch.int32
        )
        index = routed._expert_map
        if (
            routed.moe_config.moe_parallel_config.ep_size != world
            or index is None
            or not torch.equal(index.cpu().to(torch.int32), linear)
        ):
            raise ValueError(
                "decode_expert_replicas needs the linear expert placement."
            )
        plan = plan_layer(config.profile[moe.layer_idx], world, config.replicas)
        if not plan:
            continue
        moe_kernel = routed.quant_method.moe_kernel
        assert moe_kernel is not None
        quant_config = moe_kernel.fused_experts.quant_config
        params = [
            routed.w13_weight,
            routed.w2_weight,
            quant_config.w1_scale,
            quant_config.w2_scale,
        ]
        copies = _replicate(
            [p.data for p in params],
            local,
            plan,
            rank,
            world,
            config.replicas,
            ep,
        )
        nbytes += sum(t.numel() * t.element_size() for t in copies)
        device = routed.w13_weight.device
        moe.decode_replicas = LayerDecodeReplicas(
            table=layer_table(plan, rank, world, num_experts, config.replicas, device),
            step_index=linear.to(device),
            weights=tuple(copies),
        )
        layers += 1
    if layers == 0:
        raise ValueError(
            "decode_expert_replicas is set, but no MoE layer runs the Muse Spark "
            "native-routes decode MoE (ROCm gfx950, expert-parallel MXFP4 "
            "experts)."
        )
    logger.info(
        "Muse Spark decode expert replicas: %d per rank on %d layers "
        "(%.2f GiB) from %s.",
        config.replicas,
        layers,
        nbytes / 2**30,
        config.profile_path,
    )
