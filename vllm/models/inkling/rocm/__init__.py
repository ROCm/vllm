# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm (gfx9xx) kernels for Inkling.

Only ops that have no ROCm build in the ``nvidia`` package are reimplemented
here; the Triton ops and the torch-level module wiring are shared and run on
ROCm unchanged.
"""
