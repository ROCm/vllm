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
    # V2's shared DFlash input kernel marks rejected context rows with
    # PAD_SLOT_ID before precompute_and_store_context_kv. The cache update
    # therefore ignores them without the explicit tensor compaction used by
    # the legacy V1 DFlareProposer.
    _speculator_name = "DFlare"
    _reuse_full_graph_metadata = True
