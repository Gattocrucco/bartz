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
from jax.profiler import TraceAnnotation
from jax.stages import Compiled, Wrapped

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
def trace(outfile: Path | str | None = None) -> Iterator[None]:
    """Enable profiling and optionally save results to a file.

    This context manager puts bartz into a special "profiling mode". Performance
    may be lower.

    Parameters
    ----------
    outfile
        Path to the output file where Python profiling statistics will be saved.
        The file convenionally has extension `.prof` and can be read by standard
        Python tooling. If `None`, profiling is left to the user, and the
        context manager only puts `bartz` in profiling mode.

    Examples
    --------
    >>> with trace('profile_output.prof'):
    ...     # Code runs with profiling enabled
    ...     pass
    >>> # Analyze with: python -m pstats profile_output.prof

    Notes
    -----
    In profiling mode, the MCMC loop is not compiled into a single function, but
    instead compiled in smaller pieces that are instrumented to show up in the
    jax tracer and Python profiling statistics.

    Python profiling is activated within the context, and the results are saved
    to a specified output file. Jax tracing is not enabled by this context
    manager and if used must be handled separately by the user; this context
    manager only makes sure that the execution flow will be more interpretable
    in the traces.
    """
    with profile_mode(True):
        if outfile is None:
            yield
        else:
            outfile = Path(outfile)
            profiler = Profile()
            profiler.enable()
            try:
                yield
            finally:
                profiler.disable()
                profiler.dump_stats(outfile)


def jit_and_block_if_profiling(
    func: Callable[..., T], *args, **kwargs
) -> Callable[..., T]:
    """Apply JIT compilation and block if profiling is enabled.

    When profile mode is off, the function runs without JIT.
    When profile mode is on, the function is JIT compiled and blocks inputs and
    outputs to ensure proper profiling.

    Parameters
    ----------
    func
        Function to wrap.
    *args
    **kwargs
        Additional arguments to pass to `jax.jit`.

    Returns
    -------
    Wrapped function.

    Notes
    -----
    Under profiling mode, the function invocation is handled such that custom
    jax trace events and pstats dummy functions with names `jab_compile[func_name]`
    and `jab_run[func_name]` are created.
    """
    jitted_func = jit(func, *args, **kwargs)

    compile_event_name = f'jab_compile[{func.__name__}]'

    def compile_wrapper(func: Wrapped, *args, **kwargs) -> Compiled:
        with TraceAnnotation(compile_event_name):
            return func.lower(*args, **kwargs).compile()

    compile_wrapper.__code__.replace(
        co_name=compile_event_name,
        co_filename=func.__code__.co_filename,
        co_firstlineno=func.__code__.co_firstlineno,
    )

    run_event_name = f'jab_run[{func.__name__}]'

    def run_wrapper(func: Compiled, *args, **kwargs) -> T:
        with TraceAnnotation(run_event_name):
            result = func(*args, **kwargs)
            return block_until_ready(result)

    run_wrapper.__code__.replace(
        co_name=run_event_name,
        co_filename=func.__code__.co_filename,
        co_firstlineno=func.__code__.co_firstlineno,
    )

    @wraps(func)
    def jab_wrapper(*args: Any, **kwargs: Any) -> T:
        if get_profile_mode():
            args, kwargs = block_until_ready((args, kwargs))
            compiled_func = compile_wrapper(jitted_func, *args, **kwargs)
            return run_wrapper(compiled_func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    return jab_wrapper


def jit_if_not_profiling(func: Callable[..., T], *args, **kwargs) -> Callable[..., T]:
    """Apply JIT compilation only when not profiling.

    When profile mode is off, the function is JIT compiled for performance.
    When profile mode is on, the function runs without JIT.

    Parameters
    ----------
    func
        Function to wrap.
    *args
    **kwargs
        Additional arguments to pass to `jax.jit`.

    Returns
    -------
    Wrapped function.
    """
    jitted_func = jit(func, *args, **kwargs)

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
