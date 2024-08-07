import asyncio
import sys
from typing import (TYPE_CHECKING, Any, AsyncIterator, Awaitable, Protocol,
                    Tuple, TypeVar)

import pytest

from vllm.utils import deprecate_kwargs, merge_async_iterators

from .utils import error_on_warning

if sys.version_info < (3, 10):
    if TYPE_CHECKING:
        _AwaitableT = TypeVar("_AwaitableT", bound=Awaitable[Any])
        _AwaitableT_co = TypeVar("_AwaitableT_co",
                                 bound=Awaitable[Any],
                                 covariant=True)

        class _SupportsSynchronousAnext(Protocol[_AwaitableT_co]):

            def __anext__(self) -> _AwaitableT_co:
                ...

    def anext(i: "_SupportsSynchronousAnext[_AwaitableT]", /) -> "_AwaitableT":
        return i.__anext__()


@pytest.mark.asyncio
async def test_merge_async_iterators():

    async def mock_async_iterator(idx: int) -> AsyncIterator[str]:
        try:
            while True:
                yield f"item from iterator {idx}"
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    iterators = [mock_async_iterator(i) for i in range(3)]
    merged_iterator: AsyncIterator[Tuple[int, str]] = merge_async_iterators(
        *iterators)

    async def stream_output(generator: AsyncIterator[Tuple[int, str]]):
        async for idx, output in generator:
            print(f"idx: {idx}, output: {output}")

    task = asyncio.create_task(stream_output(merged_iterator))
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for iterator in iterators:
        try:
            await asyncio.wait_for(anext(iterator), 1)
        except StopAsyncIteration:
            # All iterators should be cancelled and print this message.
            print("Iterator was cancelled normally")
        except (Exception, asyncio.CancelledError) as e:
            raise AssertionError() from e


def test_deprecate_kwargs_always():

    @deprecate_kwargs("old_arg", is_deprecated=True)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="'old_arg'"):
        dummy(old_arg=1)

    with error_on_warning():
        dummy(new_arg=1)


def test_deprecate_kwargs_never():

    @deprecate_kwargs("old_arg", is_deprecated=False)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with error_on_warning():
        dummy(old_arg=1)

    with error_on_warning():
        dummy(new_arg=1)


def test_deprecate_kwargs_dynamic():
    is_deprecated = True

    @deprecate_kwargs("old_arg", is_deprecated=lambda: is_deprecated)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="'old_arg'"):
        dummy(old_arg=1)

    with error_on_warning():
        dummy(new_arg=1)

    is_deprecated = False

    with error_on_warning():
        dummy(old_arg=1)

    with error_on_warning():
        dummy(new_arg=1)


def test_deprecate_kwargs_additional_message():

    @deprecate_kwargs("old_arg", is_deprecated=True, additional_message="abcd")
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="abcd"):
        dummy(old_arg=1)


def is_rocm62():
    import torch
    return isinstance(torch.version.hip,
                      str) and torch.version.hip.startswith("6.2")


def xfail_if_rocm62(function=None,
                    reason: str = "Tests are not yet ready for ROCm 6.2",
                    strict: bool = False):
    if function:
        return pytest.mark.xfail(is_rocm62(), reason=reason,
                                 strict=strict)(function)
    else:
        assert callable(function)
        return pytest.mark.xfail(is_rocm62(), reason=reason, strict=strict)
