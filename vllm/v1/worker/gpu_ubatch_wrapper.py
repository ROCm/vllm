# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.compilation.cuda_graph import (
    CUDAGraphWrapper,
    OwnedCUDAGraph,
    begin_cudagraph_owner_teardown,
    create_cudagraph,
    cudagraph_capture_attempt,
    cudagraph_owner_activity,
    register_cudagraph_owner,
)
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_ep_group
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import (
    DPMetadata,
    create_forward_context,
    get_forward_context,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.deep_gemm import set_num_sms as deep_gemm_set_num_sms
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.worker.ubatching import UBatchContext, make_ubatch_contexts

logger = init_logger(__name__)


def _cat_ubatch_outputs(
    sorted_results: list,
) -> "torch.Tensor | tuple[torch.Tensor, ...]":
    """Concatenate per-ubatch model outputs along the batch dim.

    Most models return a single hidden-states tensor per ubatch. Target
    models running with auxiliary output (e.g. EAGLE3 speculative decoding,
    which collects aux hidden states for the drafter) return a tuple of
    tensors instead. Fan out over tuple components so `torch.cat` sees
    matching shapes and the caller receives the same structure the model
    produced for a single ubatch (#40769).
    """
    if sorted_results and isinstance(sorted_results[0], tuple):
        return tuple(torch.cat(parts, dim=0) for parts in zip(*sorted_results))
    return torch.cat(sorted_results, dim=0)


@dataclass
class UbatchMetadata:
    context: UBatchContext
    input_ids: torch.Tensor
    positions: torch.Tensor
    inputs_embeds: torch.Tensor | None
    intermediate_tensors: IntermediateTensors | None
    num_tokens: int


@dataclass
class CUDAGraphMetaData:
    cudagraph: torch.cuda.CUDAGraph
    ubatch_metadata: UbatchMetadata
    outputs: Any | None = None


class SMControlContextManager:
    def __init__(
        self,
        comm_sms: int,
        set_comm_sms: Callable[[int], None],
        set_compute_sms: Callable[[int], None],
    ):
        """
        Context manager for controlling SM (Streaming Multiprocessor)
        allocation. Upon entering the context, it sets the number of SMs
        allocated for communication and computation to comm_sms and
        total_sms - comm_sms respectively. Upon exiting, it restores the
        allocation to use all available SMs (i.e. total_sms).

        Args:
            comm_sms (int): The number of SMs to allocate for communication.
                (The remainder will be used for computation.)
            set_comm_sms (Callable[[int], None]):
                A function that sets the number of SMs for communication.
            set_compute_sms (Callable[[int], None]):
                A function that sets the number of SMs for computation.
        """

        assert current_platform.is_cuda() or current_platform.is_rocm(), (
            "SM/CU control is supported on CUDA and ROCm platforms"
        )
        device = torch.accelerator.current_device_index()
        total_sms = num_compute_units(device)

        assert comm_sms < total_sms
        self.total_sms = total_sms
        self.compute_sms = total_sms - comm_sms
        self.comm_sms = comm_sms
        self.set_comm_sms = set_comm_sms
        self.set_compute_sms = set_compute_sms

    def __enter__(self):
        self.set_comm_sms(self.comm_sms)
        self.set_compute_sms(self.compute_sms)

    def __exit__(self, exc_type, exc_value, traceback):
        self.set_comm_sms(self.total_sms)
        self.set_compute_sms(self.total_sms)


class UBatchWrapper:
    _THREAD_QUIESCE_TIMEOUT_S = envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        device: torch.device,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.cuda.Stream(device=device)
        # Ubatch threads plus the main thread
        self.ready_barrier = threading.Barrier(
            self.vllm_config.parallel_config.num_ubatches + 1
        )

        self.cudagraphs: dict[int, CUDAGraphMetaData] = {}

        self.sm_control = self._create_sm_control_context(vllm_config)
        self.device = device
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None
        self._outstanding_ubatch_threads: list[threading.Thread] = []
        self._ubatch_threads_lock = threading.Lock()
        self.cudagraph_wrapper = None
        self._cudagraph_teardown_started = False
        self._cudagraph_activity_enabled = register_cudagraph_owner(self, vllm_config)

        # Register the outer owner before constructing its nested wrapper so
        # registry quiescence cannot close the child beneath an admitted call.
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = CUDAGraphWrapper(
                runnable, vllm_config, runtime_mode=runtime_mode
            )

    @property
    def graph_pool(self):
        if self.cudagraph_wrapper is not None:
            return self.cudagraph_wrapper.graph_pool
        return None

    def clear_graphs(self) -> None:
        self.clear_cudagraph_state()

    def begin_cudagraph_teardown(self) -> None:
        begin_cudagraph_owner_teardown(self)
        outstanding = tuple(getattr(self, "_outstanding_ubatch_threads", ()))

        alive = self._join_ubatch_threads(
            outstanding,
            timeout_s=self._THREAD_QUIESCE_TIMEOUT_S,
        )
        self._raise_for_active_threads(alive, "CUDA graph teardown")
        if self.cudagraph_wrapper is not None:
            self.cudagraph_wrapper.begin_cudagraph_teardown()

    def iter_cudagraphs(self) -> Iterable[OwnedCUDAGraph]:
        device = torch.device(self.device)
        for metadata in self.cudagraphs.values():
            yield OwnedCUDAGraph(metadata.cudagraph, device)
        if self.cudagraph_wrapper is not None:
            yield from self.cudagraph_wrapper.iter_cudagraphs()

    def clear_cudagraph_state(self) -> None:
        self.cudagraphs.clear()
        getattr(self, "_outstanding_ubatch_threads", []).clear()
        if self.cudagraph_wrapper is not None:
            self.cudagraph_wrapper.clear_cudagraph_state()

    def _record_outstanding_threads(
        self, threads: Iterable[threading.Thread]
    ) -> list[threading.Thread]:
        lock = getattr(self, "_ubatch_threads_lock", None)
        if lock is None:
            lock = self._ubatch_threads_lock = threading.Lock()
        with lock:
            outstanding = getattr(self, "_outstanding_ubatch_threads", [])
            candidates = [*outstanding, *threads]
            seen: set[int] = set()
            alive: list[threading.Thread] = []
            for thread in candidates:
                if id(thread) in seen:
                    continue
                seen.add(id(thread))
                if thread.is_alive():
                    alive.append(thread)
            self._outstanding_ubatch_threads = alive
            return alive

    def _join_ubatch_threads(
        self,
        threads: Iterable[threading.Thread],
        *,
        timeout_s: float | None = None,
    ) -> list[threading.Thread]:
        threads = tuple(threads)
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        for thread in threads:
            timeout = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            thread.join(timeout=timeout)
        return self._record_outstanding_threads(threads)

    @staticmethod
    def _raise_for_active_threads(
        threads: list[threading.Thread], operation: str
    ) -> None:
        if threads:
            raise RuntimeError(
                f"{len(threads)} ubatch thread(s) remained active during {operation}"
            )

    def _barrier_timeout(self) -> float | None:
        return self._THREAD_QUIESCE_TIMEOUT_S if current_platform.is_rocm() else None

    @contextmanager
    def _manage_ubatch_threads(
        self,
        threads: Iterable[threading.Thread],
        ubatch_metadata: Iterable[UbatchMetadata],
    ) -> Iterator[list[threading.Thread]]:
        """Start ubatch threads and guarantee that none are stranded.

        The only special startup case is a later thread failing to start while
        an earlier sibling is waiting at the barrier. Aborting the barrier
        releases that sibling; waking the CPU events handles failures after all
        threads have reached the barrier.
        """
        started: list[threading.Thread] = []
        startup_complete = False
        failed = True
        try:
            for thread in threads:
                thread.start()
                started.append(thread)
            startup_complete = True
            yield started
            failed = False
        finally:
            if not startup_complete:
                self.ready_barrier.abort()
            if failed:
                for metadata in ubatch_metadata:
                    metadata.context.cpu_wait_event.set()

            alive = self._join_ubatch_threads(
                started,
                timeout_s=self._barrier_timeout(),
            )
            if getattr(self.ready_barrier, "broken", False) and not alive:
                self.ready_barrier.reset()
            if alive:
                if failed:
                    logger.error(
                        "%d ubatch thread(s) remained active during cleanup",
                        len(alive),
                    )
                else:
                    self._raise_for_active_threads(alive, "cleanup")

    @staticmethod
    def _create_sm_control_context(vllm_config: VllmConfig):
        comm_sms: int = envs.VLLM_DBO_COMM_SMS
        rocm_deepep_ht_dbo = (
            current_platform.is_rocm()
            and vllm_config.parallel_config.enable_dbo
            and vllm_config.parallel_config.all2all_backend == "deepep_high_throughput"
        )
        if rocm_deepep_ht_dbo:
            # On ROCm, reserving CUs for DeepEP HT communication under DBO
            # corrupts DP+EP generation accuracy. Keep the backend active, but
            # leave all CUs visible to the compute and communication kernels.
            comm_sms = 0

        set_comm_sms = lambda sms: None
        if vllm_config.parallel_config.enable_expert_parallel:
            # Currently only DeepEP highthroughput supports SM control so this
            # only affects that case.
            ep_group = get_ep_group()
            device_communicator = ep_group.device_communicator
            all2all_manager = None
            if device_communicator is not None:
                all2all_manager = device_communicator.all2all_manager

            if all2all_manager is not None:
                max_sms_used = all2all_manager.max_sms_used()
                if max_sms_used is not None:
                    comm_sms = min(comm_sms, max_sms_used)

            if comm_sms > 0 and all2all_manager is not None:
                set_comm_sms = lambda sms: all2all_manager.set_num_sms(sms)

        # TODO(lucas): support other kernels besides DeepGEMM
        set_compute_sms = lambda sms: None
        if has_deep_gemm() and comm_sms > 0:
            set_compute_sms = lambda sms: deep_gemm_set_num_sms(sms)

        return SMControlContextManager(
            comm_sms=comm_sms,
            set_comm_sms=set_comm_sms,
            set_compute_sms=set_compute_sms,
        )

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        runnable = self.__dict__.get("runnable")
        if runnable is not None and hasattr(runnable, key):
            return getattr(runnable, key)
        if self.__dict__.get("is_debugging_mode", False):
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of "
                f"cudagraph wrapper: {self.__dict__.get('_runnable_str')}"
            )
        raise AttributeError

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def _capture_ubatches(
        self, ubatch_metadata, model
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Capture a cudagraph for a microbatched run.

        The logic here is somewhat complicated because we need to make sure that
        each of the ubatch threads initialize the cuda context before we start
        the graph capture.

        The flow is as follows:
        1. The main thread starts up each ubatch thread. Each thread will
        initialize its cuda context (torch.cuda.current_blas_handle())
        before going to sleep upon entering the ubatch_context.

        2. The main thread starts the graph capture and wakes up the first
        ubatch thread.

        3. Each ubatch thread runs the model to completion and returns the
        completed output tensors back to the main thread.

        4. The main thread stores the captured cudagraph along with its metadata
        and returns
        """

        @torch.inference_mode()
        def _capture_ubatch_thread(results, ubatch_metadata):
            torch.accelerator.set_device_index(self.device)
            ubatch_context = ubatch_metadata.context
            with torch.cuda.stream(ubatch_context.compute_stream):
                _ = torch.cuda.current_blas_handle()
            with torch.cuda.stream(ubatch_context.comm_stream):
                _ = torch.cuda.current_blas_handle()
            with ubatch_context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )

            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []
        compute_stream = ubatch_metadata[0].context.compute_stream
        num_tokens = sum(m.num_tokens for m in ubatch_metadata)

        # Ubatches will manually manage the forward context, so we override
        # it to None here so we can have it restored correctly later
        with override_forward_context(None):
            threads = (
                threading.Thread(
                    target=_capture_ubatch_thread,
                    args=(
                        results,
                        metadata,
                    ),
                )
                for metadata in ubatch_metadata
            )
            with self._manage_ubatch_threads(
                threads, ubatch_metadata
            ) as ubatch_threads:
                self.ready_barrier.wait(timeout=self._barrier_timeout())

                # Capture the cudagraph. Store it before any capture setup so
                # partial failures remain visible to owner teardown.
                def install_graph(graph: torch.cuda.CUDAGraph) -> None:
                    self.cudagraphs[num_tokens] = CUDAGraphMetaData(
                        cudagraph=graph,
                        ubatch_metadata=ubatch_metadata,
                    )

                create_cudagraph(
                    self,
                    self.device,
                    install_graph,
                )
                cudagraph_metadata = self.cudagraphs[num_tokens]
                if self.graph_pool is not None:
                    set_graph_pool_id(self.graph_pool)
                else:
                    set_graph_pool_id(current_platform.graph_pool_handle())

                # Sync offloader's copy stream before capture.
                # Ensure any pre-capture prefetches from offloader are complete.
                get_offloader().sync_prev_onload()

                with (
                    cudagraph_capture_attempt(self),
                    torch.cuda.graph(
                        cudagraph_metadata.cudagraph,
                        stream=compute_stream,
                        pool=self.graph_pool,
                    ),
                ):
                    ubatch_metadata[0].context.cpu_wait_event.set()
                    self._join_ubatch_threads(ubatch_threads)
                    sorted_results = [value for position, value in sorted(results)]
                    result = _cat_ubatch_outputs(sorted_results)
                    cudagraph_metadata.outputs = result
                    # Join offloader's copy stream after forward to avoid
                    # unjoined stream error. The last layer's start_prefetch
                    # forks copy_stream, but wait_prefetch only happens in
                    # the next forward pass.
                    get_offloader().join_after_forward()
        return cudagraph_metadata.outputs

    def _run_ubatches(
        self, ubatch_metadata, model
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        @torch.inference_mode()
        def _ubatch_thread(results, model, ubatch_metadata):
            with ubatch_metadata.context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )
            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []

        # Ubatch threads will manually manage the forward context, so we
        # override it to None here so we can have it restored correctly
        # after both threads have finished
        with override_forward_context(None):
            threads = (
                threading.Thread(
                    target=_ubatch_thread,
                    args=(
                        results,
                        model,
                        metadata,
                    ),
                )
                for metadata in ubatch_metadata
            )
            with self._manage_ubatch_threads(
                threads, ubatch_metadata
            ) as ubatch_threads:
                self.ready_barrier.wait(timeout=self._barrier_timeout())
                ubatch_metadata[0].context.cpu_wait_event.set()
                self._join_ubatch_threads(ubatch_threads)
        sorted_results = [value for position, value in sorted(results)]
        result = _cat_ubatch_outputs(sorted_results)
        return result

    def _make_ubatch_metadata(
        self,
        ubatch_slices,
        attn_metadata,
        slot_mapping,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
        compute_stream,
        dp_metadata,
        batch_descriptor,
        cudagraph_runtime_mode,
    ) -> list[UbatchMetadata]:
        # Create one forward context per ubatch
        forward_contexts = []
        # slot_mapping can be None, an empty dict (from create_forward_context
        # converting None to {}), or a list of dicts (one per ubatch)
        has_slot_mapping = slot_mapping and isinstance(slot_mapping, list)
        for i, ubatch_slice in enumerate(ubatch_slices):
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i] if attn_metadata is not None else None,
                    self.vllm_config,
                    dp_metadata=dp_metadata[i],
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping=slot_mapping[i] if has_slot_mapping else None,
                )
            )

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        ubatch_metadata: list[UbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
                sliced_inputs_embeds,
                sliced_intermediate_tensors,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.token_slice.stop
                    - ubatch_slice.token_slice.start,
                )
            )

        return ubatch_metadata

    def _slice_model_inputs(
        self,
        tokens_slice: slice,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
    ):
        sliced_input_ids = input_ids[tokens_slice] if input_ids is not None else None
        # if we are using mrope. Mrope adds an additional dimension to the
        # positions tensor
        if positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]
        sliced_inputs_embeds = (
            inputs_embeds[tokens_slice] if inputs_embeds is not None else None
        )
        sliced_intermediate_tensors = (
            intermediate_tensors[tokens_slice]
            if intermediate_tensors is not None
            else None
        )

        return (
            sliced_input_ids,
            sliced_positions,
            sliced_inputs_embeds,
            sliced_intermediate_tensors,
        )

    @cudagraph_owner_activity
    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        # If there's no ubatching, just run the runnable object
        if ubatch_slices is None:
            # This is to account for the case where ubatching was aborted.
            # When we capture full graphs we only capture one graph per shape,
            # meaning that if we have a ubatched  cudagraph for the current
            # num_tokens, we don't have a non-ubatched one. Without this
            # check, the cudagraph wrapper will try to capture a cudagraph
            # for this shape during a normal run.
            if cudagraph_runtime_mode is CUDAGraphMode.FULL:
                assert batch_descriptor is not None
                if batch_descriptor.num_tokens in self.cudagraphs:
                    cudagraph_runtime_mode = CUDAGraphMode.NONE

            if cudagraph_runtime_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
                return self.runnable(*args, **kwargs)
            else:
                assert self.cudagraph_wrapper is not None
                return self.cudagraph_wrapper(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        slot_mapping = forward_context.slot_mapping
        num_tokens = sum(ubatch_slice.num_tokens for ubatch_slice in ubatch_slices)
        input_ids = kwargs["input_ids"]
        positions = kwargs["positions"]
        intermediate_tensors = kwargs["intermediate_tensors"]
        inputs_embeds = kwargs["inputs_embeds"]
        compute_stream = torch.cuda.current_stream()

        dp_metadata = forward_context.dp_metadata

        # We shouldn't be here unless we are running with multiple DP ranks
        assert dp_metadata is not None
        ubatch_dp_metadata = []
        for ubatch_slice in ubatch_slices:
            dp_size = self.vllm_config.parallel_config.data_parallel_size
            ubatch_num_tokens_across_dp = torch.tensor(
                [ubatch_slice.num_tokens] * dp_size, device="cpu", dtype=torch.int32
            )
            ubatch_dp_metadata.append(
                DPMetadata.make(
                    self.vllm_config.parallel_config,
                    ubatch_slice.num_tokens,
                    ubatch_num_tokens_across_dp,
                )
            )

        if (
            num_tokens not in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with self.sm_control:
                return self._capture_ubatches(ubatch_metadata, self.runnable)
        elif (
            num_tokens in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            cudagraph_metadata = self.cudagraphs[num_tokens]
            # Sync offloader before replay - ensures any external dependencies
            # from pre-capture prefetches are satisfied.
            get_offloader().sync_prev_onload()
            cudagraph_metadata.cudagraph.replay()
            return cudagraph_metadata.outputs
        else:
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with self.sm_control:
                return self._run_ubatches(ubatch_metadata, self.runnable)
