# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU DFlare speculator.

DFlare uses the same parallel query scheduling, slot preparation, rejection
sampling, and CUDA-graph boundaries as DFlash.  Its draft model differs in
how it consumes the concatenated target hidden states, so only the model and
method dispatch need to be specialized here.
"""

from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


class DFlareSpeculator(DFlashSpeculator):
    _speculator_name = "DFlare"
