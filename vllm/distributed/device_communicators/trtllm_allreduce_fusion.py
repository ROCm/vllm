# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM's local implementation of TRTLLM all-reduce fusion distributed environment.

This module provides VLLMAiterDistEnv, a modified version of aiter's AiterDistEnv
that follows vLLM's CustomAllreduce pattern for CUDA graph capture handling.

The key difference from aiter's original AiterDistEnv:
- Uses _IS_CAPTURING flag (set by capture() context) to distinguish:
  1. Warmup within capture context: return None (signal fallback)
  2. Actual graph recording: call the real kernel
  3. Normal inference: call the real kernel
- consume_capture() is called AFTER capture context exits (like register_graph_buffers)
"""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from contextlib import contextmanager
from aiter.ops.trtllm_all_reduce_fusion import (
    AiterDistEnv,
    fp8,
    fp8_max_val,
    fp8_policy_id,
    trtllm_allreduce_rms,
)

from vllm.logger import init_logger


logger = init_logger(__name__)


class VLLMAiterDistEnv(AiterDistEnv):
    # Piggy back on AiterDistEnv for logging

    def __init__(
        self,
        group: ProcessGroup = None,
        device_id: int = None,
        max_size_in_bytes=16384 * 16384,
        comm_ptrs_buf_len=1024 * 256,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(
            group=group,
            device_id=device_id,
            max_size_in_bytes=max_size_in_bytes,
            comm_ptrs_buf_len=comm_ptrs_buf_len,
            dtype=dtype,
        )
        self._IS_CAPTURING = False

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            # logger.info("VLLMAiterDistEnv.capture() ENTER - _IS_CAPTURING=True")
            yield
        finally:
            self._IS_CAPTURING = False
            # logger.info("VLLMAiterDistEnv.capture() EXIT - _IS_CAPTURING=False, disabled=%s", self.disabled)
            if not self.disabled:
                # logger.info("VLLMAiterDistEnv.capture() calling consume_capture()")
                super().consume_capture()
                # logger.info("VLLMAiterDistEnv.capture() consume_capture() done")
                self._IS_CAPTURED = False

    def allreduce_add_rms_fused(
        self,
        allreduce_in: torch.Tensor,
        residual_in: torch.Tensor,
        rms_weight: torch.Tensor,
        eps: float,
        fp8_out: bool = False
    ):
        """
        Fused all-reduce + add residual + RMSNorm + optional FP8 quantization.

        Following vLLM's CustomAllreduce.custom_all_reduce() pattern:

        1. If _IS_CAPTURING (inside capture context):
           - If actually recording (is_current_stream_capturing): call real kernel
           - If warmup: return None to signal fallback to NCCL
        2. If not _IS_CAPTURING (normal inference): call real kernel
        """
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                # logger.info("allreduce_add_rms_fused: _IS_CAPTURING=True, stream capturing -> calling fused kernel")
                return self._call_fused_kernel(
                    allreduce_in, residual_in, rms_weight, eps, fp8_out
                )
            else:
                # Warmup phase - return None to signal NCCL fallback
                # logger.info("allreduce_add_rms_fused: _IS_CAPTURING=True, not stream capturing (warmup) -> returning None")
                return None
        else:
            # Normal inference - call the real kernel
            # logger.info("allreduce_add_rms_fused: _IS_CAPTURING=False (normal inference) -> calling fused kernel")
            return self._call_fused_kernel(
                allreduce_in, residual_in, rms_weight, eps, fp8_out
            )

    def _call_fused_kernel(
        self,
        allreduce_in: torch.Tensor,
        residual_in: torch.Tensor,
        rms_weight: torch.Tensor,
        eps: float,
        fp8_out: bool = False
    ):
        """
        Internal method to call the TRTLLM fused kernel.

        Note: We call the parent's capture_() method to ensure proper internal
        tracking of tensors for IPC handle exchange during CUDA graph capture.
        """
        # logger.info("_call_fused_kernel: ENTER, shape=%s, fp8_out=%s", allreduce_in.shape, fp8_out)

        # Call parent's capture_() for proper internal state tracking
        # Tracks tensor addresses for IPC handle exchange during CUDA graph capture
        # logger.info("_call_fused_kernel: calling super().capture_()")
        super().capture_(allreduce_in)
        # logger.info("_call_fused_kernel: super().capture_() done")

        residual_out = torch.empty_like(residual_in)
        if fp8_out:
            norm_out = torch.empty_like(allreduce_in, dtype=fp8)
            scale_out = torch.empty(
                allreduce_in.shape[0],
                1,
                dtype=torch.float32,
                device=allreduce_in.device,
            )
        else:
            norm_out = torch.empty_like(allreduce_in)
            scale_out = torch.empty(
                1, dtype=torch.float32, device=allreduce_in.device
            )

        # logger.info("_call_fused_kernel: calling trtllm_allreduce_rms")
        trtllm_allreduce_rms(
            self.fptr,
            allreduce_in,
            residual_in,
            rms_weight,
            residual_out,
            norm_out,
            scale_out,
            eps,
            fp8_policy_id if fp8_out else 0,
        )

        # Synchronize only when not capturing to check for errors
        # if not torch.cuda.is_current_stream_capturing():
        #     logger.info("_call_fused_kernel: synchronizing (not capturing)")
        #     torch.cuda.synchronize()
        #     logger.info("_call_fused_kernel: synchronize done")
        # else:
        #     logger.info("_call_fused_kernel: skipping synchronize (capturing)")

        # logger.info("_call_fused_kernel: EXIT")
        return residual_out, norm_out, scale_out


class AiterCommManager:
    """
    Manager for AITER TRTLLM fused all-reduce + RMSNorm distributed environment.

    Following the sglang pattern for lazy initialization:
    - Constructor takes no arguments
    - AiterDistEnv is only created during initialize()
    - Re-initialization is allowed if group/device_id/dtype changes
    """

    def __init__(self):
        self.group = None
        self.device_id = None
        self.dtype = None
        self.initialized = False
        self.dist_env = None

    def initialize(
        self,
        group,
        device_id,
        dtype: torch.dtype,
    ):
        if self.initialized and group == self.group and device_id == self.device_id:
            return

        logger.info(
            "Initializing AiterCommManager: group=%s, device_id=%d, dtype=%s",
            group, device_id, dtype
        )

        self.cleanup()

        self.group = group
        self.device_id = device_id
        self.dtype = dtype
        self.dist_env = VLLMAiterDistEnv(
            group=self.group,
            device_id=self.device_id,
            dtype=self.dtype
        )

        self.initialized = True

    def cleanup(self):
        self.dist_env = None
        self.initialized = False
