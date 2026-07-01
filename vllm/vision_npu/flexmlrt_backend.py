# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
FlexMLRT-based vision NPU backend with CPU preprocessing.

VitisAI-compiled models partition operations between CPU and NPU. This backend
implements the CPU preprocessing operations before calling FlexMLRT for NPU
execution, matching the behavior of VitisAI ExecutionProvider.
"""

import asyncio
import contextlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import vllm.envs as envs

from .backend import NPUVisionBackend
from .cpu_preprocess import get_preprocessor, read_cache_io_spec

logger = logging.getLogger(__name__)

# Cache environment variables for performance (avoids repeated lookups)
VLLM_NPU_TIMING = envs.VLLM_NPU_TIMING
VLLM_NPU_ASYNC_PIPELINE = envs.VLLM_NPU_ASYNC_PIPELINE


@contextlib.contextmanager
def npu_timing(operation: str, logger_obj=None):
    """Zero-overhead timing for NPU operations when VLLM_NPU_TIMING=1.

    Args:
        operation: Name of the operation being timed
        logger_obj: Optional logger to use (defaults to module logger)
    """
    if not VLLM_NPU_TIMING:
        yield
        return

    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000
        log_func = logger_obj.info if logger_obj else logger.info
        log_func("[NPU Timing] %s: %.2fms", operation, elapsed_ms)


class FlexMLRTVisionBackend(NPUVisionBackend):
    """FlexMLRT implementation of NPU vision backend with CPU preprocessing.

    Uses AMD FlexMLRT library to run vision models on Ryzen AI NPU.
    Implements CPU preprocessing operations that VitisAI EP normally handles.
    """

    def __init__(self, model_cache_path: str, device_name: str = "stx",
                 model_type: str | None = None):
        """Initialize the generic FlexMLRT vision backend.

        Args:
            model_cache_path: Path to VAIP model cache (vaiml_par_0 directory)
            device_name: XRT device name ("stx" for Strix, "phx" for Phoenix)
            model_type: model's config.model_type; selects the preprocessor.
        """
        from vllm.vision_npu._vision_flexmlrt_cpu import VisionFlexMLRTModel

        self.model = VisionFlexMLRTModel(model_cache_path, device_name)
        # Per-model preprocessor (registry) + IO names/shape/batch read from the
        # cache's own spec — the backend itself is model-agnostic.
        self.preprocessor = get_preprocessor(model_cache_path, model_type)
        self.io = read_cache_io_spec(model_cache_path)
        logger.info(
            "[FlexMLRT Backend] model_type=%s  in=%s%s  out=%s%s",
            model_type, self.io["input_name"], self.io["in_shape"],
            self.io["output_name"], self.io["out_shape"],
        )

    def forward(self, pixel_values, geometry=None) -> np.ndarray:
        """Generic vision encode: preprocess -> NPU (per group) -> postprocess.

        Args:
            pixel_values: model's raw vision input (the preprocessor adapts it).
            geometry: model's per-item geometry (grid_thw / tgt_sizes); the
                preprocessor uses or ignores it.
        Returns:
            [tokens, hidden] (or [n, tokens, hidden]) float32 embeddings.
        """
        total_start = time.monotonic() if VLLM_NPU_TIMING else None

        groups = self.preprocessor.preprocess(pixel_values, geometry)
        outputs = []
        for g in groups:
            with npu_timing("NPU inference", logger):
                outputs.append(self.model.forward(
                    g, self.io["input_name"], self.io["output_name"],
                    self.io["out_shape"],
                ))
        final_output = self.preprocessor.postprocess(outputs, geometry)

        if VLLM_NPU_TIMING and total_start is not None:
            total_ms = (time.monotonic() - total_start) * 1000
            logger.info(
                "[NPU Timing] Total vision pipeline: %.2fms (%d NPU call(s))",
                total_ms, len(groups),
            )
        return final_output

    @property
    def output_dim(self) -> int:
        """Output embedding dim (from the cache's own IO spec)."""
        return int(self.io["out_shape"][-1])


class AsyncFlexMLRTVisionBackend:
    """Async wrapper for FlexMLRT backend enabling NPU+GPU pipelining.

    Allows NPU vision processing for request N+1 to overlap with GPU LLM
    processing for request N, improving throughput for multi-request workloads.

    Example throughput improvement:
    - Sequential: Request1(NPU 13.5s + GPU 20s) → Request2(NPU 13.5s + GPU
      20s) = 67s for 2 requests
    - Pipelined: Request1(NPU 13.5s) → overlap(NPU 13.5s for Req2 || GPU 20s
      for Req1) → GPU 20s for Req2 = 47s for 2 requests
    - Speedup: 1.43x for 2 requests, approaches 1.5x+ for longer sequences
    """

    def __init__(self, model_cache_path: str, device_name: str = "stx",
                 model_type: str | None = None):
        """Initialize async wrapper with underlying synchronous backend.

        Args:
            model_cache_path: Path to VAIP model cache (vaiml_par_0 directory)
            device_name: XRT device name ("stx" for Strix, "phx" for Phoenix)
            model_type: model's config.model_type (forwarded to the sync backend).
        """
        # Underlying synchronous backend (generic; selects preprocessor by model_type)
        self.sync_backend = FlexMLRTVisionBackend(
            model_cache_path, device_name, model_type
        )

        # Thread pool for NPU inference (separate from GPU thread)
        # Single worker ensures NPU executes one request at a time
        self.npu_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="npu_vision"
        )

        # Stats for monitoring
        self.npu_queue_size = 0
        self.total_requests = 0

        if VLLM_NPU_ASYNC_PIPELINE:
            logger.info(
                "[Async FlexMLRT Backend] Initialized with async pipelining enabled"
            )
        else:
            logger.info(
                "[Async FlexMLRT Backend] Initialized "
                "(async disabled, use VLLM_NPU_ASYNC_PIPELINE=1)"
            )

    async def forward_async(
        self, pixel_values: np.ndarray, grid_thw: np.ndarray
    ) -> np.ndarray:
        """Async version that enables NPU-GPU pipelining.

        Submits NPU work to a dedicated executor, allowing it to run concurrently
        with GPU work from other requests.

        Args:
            pixel_values: [seq_len, feature_dim] float32 array from HF processor
            grid_thw: [num_images, 3] int64 array

        Returns:
            embeddings: [merged_seq_len, hidden_dim] float32 array
        """
        loop = asyncio.get_event_loop()

        self.npu_queue_size += 1
        self.total_requests += 1
        request_id = self.total_requests

        if VLLM_NPU_TIMING:
            logger.info(
                "[Async NPU] Request %s submitted to NPU queue (queue size: %s)",
                request_id,
                self.npu_queue_size,
            )

        try:
            # Submit to NPU executor (non-blocking from caller's perspective)
            # This allows GPU to continue processing previous requests while NPU works
            result = await loop.run_in_executor(
                self.npu_executor, self.sync_backend.forward, pixel_values, grid_thw
            )

            if VLLM_NPU_TIMING:
                logger.info(
                    "[Async NPU] Request %s completed NPU processing", request_id
                )

            return result
        finally:
            self.npu_queue_size -= 1

    def forward(self, pixel_values: np.ndarray, grid_thw: np.ndarray) -> np.ndarray:
        """Synchronous interface with async execution underneath.

        Submits work to NPU executor thread, allowing multiple requests to pipeline.
        This blocks the caller until NPU processing completes, but allows other
        threads (e.g., GPU LLM processing) to run concurrently.
        """
        import threading
        from datetime import datetime

        self.npu_queue_size += 1
        self.total_requests += 1
        request_id = self.total_requests

        submit_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        caller_thread = threading.get_ident()

        if VLLM_NPU_TIMING:
            logger.info(
                "[Async NPU Pipeline] Request %s SUBMITTED at %s by Thread-%s "
                "(queue size: %s)",
                request_id,
                submit_time,
                caller_thread,
                self.npu_queue_size,
            )

        try:
            # Submit to executor - allows pipelining with GPU work from
            # other requests
            if VLLM_NPU_TIMING:
                logger.info(
                    "[Async NPU Pipeline] Request %s submitting to ThreadPoolExecutor "
                    "(queue size before: %s)",
                    request_id,
                    self.npu_queue_size,
                )

            future = self.npu_executor.submit(
                self._forward_with_timing, pixel_values, grid_thw, request_id
            )

            if VLLM_NPU_TIMING:
                logger.info(
                    "[Async NPU Pipeline] Request %s future created, "
                    "now waiting for result...",
                    request_id,
                )

            # Block until NPU processing completes
            result = future.result()

            complete_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if VLLM_NPU_TIMING:
                logger.info(
                    "[Async NPU Pipeline] Request %s COMPLETED at %s on Thread-%s",
                    request_id,
                    complete_time,
                    caller_thread,
                )

            return result
        finally:
            self.npu_queue_size -= 1

    def _forward_with_timing(
        self, pixel_values: np.ndarray, grid_thw: np.ndarray, request_id: int
    ) -> np.ndarray:
        """Internal forward with NPU start/end timing."""
        import threading
        from datetime import datetime

        worker_thread = threading.get_ident()
        npu_start_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if VLLM_NPU_TIMING:
            logger.info(
                "[Async NPU Pipeline] Request %s NPU STARTED at %s on "
                "NPU-Worker-Thread-%s",
                request_id,
                npu_start_time,
                worker_thread,
            )

        result = self.sync_backend.forward(pixel_values, grid_thw)

        npu_end_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if VLLM_NPU_TIMING:
            logger.info(
                "[Async NPU Pipeline] Request %s NPU FINISHED at %s on "
                "NPU-Worker-Thread-%s",
                request_id,
                npu_end_time,
                worker_thread,
            )

        return result

    @property
    def output_dim(self) -> int:
        """Get output embedding dimension from FlexMLRT model."""
        return self.sync_backend.output_dim

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, "npu_executor"):
            self.npu_executor.shutdown(wait=True)
