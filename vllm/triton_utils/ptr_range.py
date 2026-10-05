# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


def _view_ptr_range(self: torch.Tensor) -> int:
    """Upper bound on the extent of the view in bytes.

    Triton's AMD backend uses more efficient 32-bit buffer ops when
    ``ptr_range()`` returns < 2 GiB.  When no ``ptr_range()`` is
    defined, it judges from the full size of the underlying
    allocation, which can exceed 2 GiB for things like the KV cache.
    """
    nbytes = self.untyped_storage().nbytes()
    if nbytes < 2**31:  # no practical reason to narrow the range
        return nbytes
    if self.numel() == 0:
        return 0
    if self.is_nested or min(self.stride(), default=0) < 0:
        # the algorithm below isn't suitable
        return nbytes
    # Defensive: Include the stride gap following the last element
    # in the physically outermost dimension as part of the viewed
    # range, consistent with gaps after earlier elements.
    last = 0
    outer = 0
    for s, st in zip(self.shape, self.stride()):
        last += (s - 1) * st
        outer = max(outer, s * st)
    return max(last + 1, outer) * self.element_size()


def install_view_ptr_range() -> None:
    """Make every tensor report its view's byte range to Triton."""
    torch.Tensor.ptr_range = _view_ptr_range  # type: ignore[attr-defined]
