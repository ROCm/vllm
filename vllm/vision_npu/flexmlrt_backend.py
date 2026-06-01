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
from .cpu_preprocess import get_cpu_preprocessor

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

    def __init__(self, model_cache_path: str, device_name: str = "stx"):
        """Initialize FlexMLRT vision model with CPU preprocessing.

        Args:
            model_cache_path: Path to VAIP model cache (vaiml_par_0 directory)
            device_name: XRT device name ("stx" for Strix, "phx" for Phoenix)
        """
        from vllm.vision_npu._vision_flexmlrt_cpu import VisionFlexMLRTModel

        self.model = VisionFlexMLRTModel(model_cache_path, device_name)

        # Initialize CPU preprocessor
        self.preprocessor = get_cpu_preprocessor(model_cache_path, optimized=True)
        logger.info("[FlexMLRT Backend] Initialized with CPU preprocessing")

    def forward(self, pixel_values: np.ndarray, grid_thw: np.ndarray) -> np.ndarray:
        """Run vision encoding with CPU preprocessing + NPU execution.

        Pipeline:
        1. CPU preprocessing: [4292, 1176] → [1073, 4, 1280]
        2. NPU execution: [1073, 4, 1280] → [1073, 3584]
        3. CPU postprocessing: Apply reverse_index reordering

        Args:
            pixel_values: [seq_len, feature_dim] float32 array from HF processor
            grid_thw: [num_images, 3] int64 array (unused for now)

        Returns:
            embeddings: [merged_seq_len, hidden_dim] float32 array
        """
        total_start = time.monotonic() if VLLM_NPU_TIMING else None

        # Convert numpy to torch for preprocessing
        with npu_timing("NumPy→Torch conversion", logger):
            if isinstance(pixel_values, np.ndarray):
                pixel_values_torch = torch.from_numpy(pixel_values).float()
            else:
                pixel_values_torch = pixel_values.float()

        # Step 1: CPU preprocessing
        logger.debug(
            "[FlexMLRT Backend] Preprocessing input shape: %s", pixel_values.shape
        )
        with npu_timing("CPU preprocessing (total)", logger):
            preprocessed = self.preprocessor.preprocess(pixel_values_torch)

        # Step 2: NPU execution
        logger.debug(
            "[FlexMLRT Backend] Running NPU inference on shape: %s",
            preprocessed.shape,
        )
        with npu_timing("NPU inference", logger):
            npu_output = self.model.forward(preprocessed)

        # Step 3: CPU postprocessing
        logger.debug(
            "[FlexMLRT Backend] Postprocessing NPU output shape: %s", npu_output.shape
        )
        with npu_timing("CPU postprocessing", logger):
            final_output = self.preprocessor.postprocess(npu_output)

        logger.debug("[FlexMLRT Backend] Final output shape: %s", final_output.shape)

        # Log total time and memory stats
        if VLLM_NPU_TIMING and total_start is not None:
            total_ms = (time.monotonic() - total_start) * 1000
            logger.info("[NPU Timing] Total vision pipeline: %.2fms", total_ms)
            logger.info("[NPU Memory] Input: %.2f MB", pixel_values.nbytes / 1024**2)
            logger.info(
                "[NPU Memory] Preprocessed: %.2f MB", preprocessed.nbytes / 1024**2
            )
            logger.info("[NPU Memory] Output: %.2f MB", final_output.nbytes / 1024**2)
            logger.info(
                "[ViT Output] Shape: %s \u2192 %d patches \u00d7 %d embedding_dim",
                final_output.shape,
                final_output.shape[0],
                final_output.shape[1],
            )

        return final_output

    @property
    def output_dim(self) -> int:
        """Get output embedding dimension from FlexMLRT model."""
        return self.model.output_dim()

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

    def __init__(self, model_cache_path: str, device_name: str = "stx"):
        """Initialize async wrapper with underlying synchronous backend.

        Args:
            model_cache_path: Path to VAIP model cache (vaiml_par_0 directory)
            device_name: XRT device name ("stx" for Strix, "phx" for Phoenix)
        """
        # Underlying synchronous backend
        self.sync_backend = FlexMLRTVisionBackend(model_cache_path, device_name)

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

