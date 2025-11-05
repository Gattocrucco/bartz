# bartz/src/bartz/_profiler.py
#
# Copyright (c) 2025, The Bartz Contributors
#
# This file is part of bartz.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Module with utilities related to profiling bartz."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from cProfile import Profile
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from jax import block_until_ready, jit
from jax.lax import scan

PROFILE_MODE: bool = False

T = TypeVar('T')
Carry = TypeVar('Carry')


def get_profile_mode() -> bool:
    """Return the current profile mode status.

    Returns
    -------
    True if profile mode is enabled, False otherwise.
    """
    return PROFILE_MODE


def set_profile_mode(value: bool, /) -> None:
    """Set the profile mode status.

    Parameters
    ----------
    value
        If True, enable profile mode. If False, disable it.
    """
    global PROFILE_MODE  # noqa: PLW0603
    PROFILE_MODE = value


@contextmanager
def profile_mode(value: bool, /) -> Iterator[None]:
    """Context manager to temporarily set profile mode.

    Parameters
    ----------
    value
        Profile mode value to set within the context.

    Examples
    --------
    >>> with profile_mode(True):
    ...     # Code runs with profile mode enabled
    ...     pass
    """
    old_value = get_profile_mode()
    set_profile_mode(value)
    try:
        yield
    finally:
        set_profile_mode(old_value)


@contextmanager
def trace(outfile: Path | str) -> Iterator[None]:
    """Enable profiling and save results to a file.

    This activates both `profile_mode` and `cProfile`, and saves the profiling
    statistics to the specified file.

    Parameters
    ----------
    outfile
        Path to the output file where profiling statistics will be saved.

    Examples
    --------
    >>> with trace('profile_output.prof'):
    ...     # Code runs with profiling enabled
    ...     pass
    >>> # Analyze with: python -m pstats profile_output.prof
    """
    outfile = Path(outfile)
    profiler = Profile()

    with profile_mode(True):
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            profiler.dump_stats(outfile)


def jit_and_block_if_profiling(func: Callable[..., T]) -> Callable[..., T]:
    """Apply JIT compilation and block if profiling is enabled.

    When profile mode is off, the function runs without JIT.
    When profile mode is on, the function is JIT compiled and blocks inputs and
    outputs to ensure proper profiling.

    Parameters
    ----------
    func
        Function to wrap.

    Returns
    -------
    Wrapped function.
    """
    jitted_func = jit(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if get_profile_mode():
            args, kwargs = block_until_ready((args, kwargs))
            result = jitted_func(*args, **kwargs)
            return block_until_ready(result)
        else:
            return func(*args, **kwargs)

    return wrapper


def jit_if_not_profiling(func: Callable[..., T]) -> Callable[..., T]:
    """Apply JIT compilation only when not profiling.

    When profile mode is off, the function is JIT compiled for performance.
    When profile mode is on, the function runs without JIT.

    Parameters
    ----------
    func
        Function to wrap.

    Returns
    -------
    Wrapped function.
    """
    jitted_func = jit(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if get_profile_mode():
            return func(*args, **kwargs)
        else:
            return jitted_func(*args, **kwargs)

    return wrapper


def scan_if_not_profiling(
    f: Callable[[Carry, None], tuple[Carry, None]],
    init: Carry,
    xs: None,
    length: int,
    /,
) -> tuple[Carry, None]:
    """Restricted replacement for `jax.lax.scan` that uses a Python loop when profiling.

    When profile mode is off, uses `jax.lax.scan` for efficiency.
    When profile mode is on, uses a Python for loop for better profiling visibility.

    Parameters
    ----------
    f
        Scan body function with signature (carry, None) -> (carry, None).
    init
        Initial carry value.
    xs
        Input values to scan over (not supported).
    length
        Integer specifying the number of loop iterations.

    Returns
    -------
    Tuple of (final_carry, None) (stacked outputs not supported).
    """
    assert xs is None
    if get_profile_mode():
        carry = init
        for _i in range(length):
            carry, _ = f(carry, None)
        return carry, None

    else:
        return scan(f, init, None, length)
