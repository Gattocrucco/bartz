# bartz/src/bartz/mcmcloop.py
#
# Copyright (c) 2024-2025, Giacomo Petrillo
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

"""Functions that implement the full BART posterior MCMC loop.

The main entry point is `run_mcmc`.
"""

from collections.abc import Callable
from dataclasses import fields
from functools import partial, wraps
from typing import Any, Protocol

import jax
import numpy
from equinox import Module
from jax import debug, lax, tree
from jax import numpy as jnp
from jaxtyping import Array, Float32, Int32, Key, PyTree, Real, Shaped, UInt

from bartz import grove, jaxext, mcmcstep
from bartz.mcmcstep import State


class BurninTrace(Module):
    """MCMC trace with only diagnostic values."""

    sigma2: Float32[Array, '*trace_length'] | None
    grow_prop_count: Int32[Array, '*trace_length']
    grow_acc_count: Int32[Array, '*trace_length']
    prune_prop_count: Int32[Array, '*trace_length']
    prune_acc_count: Int32[Array, '*trace_length']
    log_likelihood: Float32[Array, '*trace_length'] | None
    log_trans_prior: Float32[Array, '*trace_length'] | None

    @classmethod
    def from_state(cls, state: State) -> 'BurninTrace':
        """Create a single-item burn-in trace from a MCMC state."""
        return cls(
            sigma2=state.sigma2,
            grow_prop_count=state.forest.grow_prop_count,
            grow_acc_count=state.forest.grow_acc_count,
            prune_prop_count=state.forest.prune_prop_count,
            prune_acc_count=state.forest.prune_acc_count,
            log_likelihood=state.forest.log_likelihood,
            log_trans_prior=state.forest.log_trans_prior,
        )


class MainTrace(BurninTrace):
    """MCMC trace with trees and diagnostic values."""

    leaf_tree: Real[Array, '*trace_length 2**d']
    var_tree: Real[Array, '*trace_length 2**(d-1)']
    split_tree: Real[Array, '*trace_length 2**(d-1)']
    offset: Float32[Array, '*trace_length']

    @classmethod
    def from_state(cls, state: State) -> 'MainTrace':
        """Create a single-item main trace from a MCMC state."""
        return cls(
            leaf_tree=state.forest.leaf_tree,
            var_tree=state.forest.var_tree,
            split_tree=state.forest.split_tree,
            offset=state.offset,
            **vars(BurninTrace.from_state(state)),
        )


CallbackState = PyTree[Any, 'T']


class Callback(Protocol):
    """Callback type for `run_mcmc`."""

    def __call__(
        self,
        *,
        bart: State,
        burnin: bool,
        overflow: bool,
        i_total: int,
        i_skip: int,
        callback_state: CallbackState,
        n_burn: int,
        n_save: int,
        n_skip: int,
        i_outer: int,
        inner_loop_length: int,
    ) -> tuple[State, CallbackState] | None:
        """Do an arbitrary action after an iteration of the MCMC.

        Parameters
        ----------
        bart
            The MCMC state just after updating it.
        burnin
            Whether the last iteration was in the burn-in phase.
        overflow
            Whether the last iteration was in the overflow phase (iterations
            not saved due to `inner_loop_length` not being a divisor of the
            total number of iterations).
        i_total
            The index of the last MCMC iteration (0-based).
        i_skip
            The number of MCMC updates from the last saved state. The initial
            state counts as saved, even if it's not copied into the trace.
        callback_state
            The callback state, initially set to the argument passed to
            `run_mcmc`, afterwards to the value returned by the last invocation
            of `inner_callback` or `outer_callback`.
        n_burn
        n_save
        n_skip
            The corresponding `run_mcmc` arguments as-is.
        i_outer
            The index of the last outer loop iteration (0-based).
        inner_loop_length
            The number of MCMC iterations in the inner loop.

        Returns
        -------
        bart : State
            A possibly modified MCMC state. To avoid modifying the state,
            return the `bart` argument passed to the callback as-is.
        callback_state : CallbackState
            The new state to be passed on the next callback invocation.

        Notes
        -----
        For convenience, the callback may return `None`, and the states won't
        be updated.
        """
        ...


def run_mcmc(
    key: Key[Array, ''],
    bart: State,
    n_save: int,
    *,
    n_burn: int = 0,
    n_skip: int = 1,
    inner_loop_length: int | None = None,
    allow_overflow: bool = False,
    inner_callback: Callback | None = None,
    outer_callback: Callback | None = None,
    callback_state: CallbackState = None,
    burnin_extractor: Callable[[State], PyTree] = BurninTrace.from_state,
    main_extractor: Callable[[State], PyTree] = MainTrace.from_state,
) -> tuple[State, PyTree[Shaped[Array, 'n_burn *']], PyTree[Shaped[Array, 'n_save *']]]:
    """
    Run the MCMC for the BART posterior.

    Parameters
    ----------
    key
        A key for random number generation.
    bart
        The initial MCMC state, as created and updated by the functions in
        `bartz.mcmcstep`. The MCMC loop uses buffer donation to avoid copies,
        so this variable is invalidated after running `run_mcmc`. Make a copy
        beforehand to use it again.
    n_save
        The number of iterations to save.
    n_burn
        The number of initial iterations which are not saved.
    n_skip
        The number of iterations to skip between each saved iteration, plus 1.
        The effective burn-in is ``n_burn + n_skip - 1``.
    inner_loop_length
        The MCMC loop is split into an outer and an inner loop. The outer loop
        is in Python, while the inner loop is in JAX. `inner_loop_length` is the
        number of iterations of the inner loop to run for each iteration of the
        outer loop. If not specified, the outer loop will iterate just once,
        with all iterations done in a single inner loop run. The inner stride is
        unrelated to the stride used for saving the trace.
    allow_overflow
        If `False`, `inner_loop_length` must be a divisor of the total number of
        iterations ``n_burn + n_skip * n_save``. If `True` and
        `inner_loop_length` is not a divisor, some of the MCMC iterations in the
        last outer loop iteration will not be saved to the trace.
    inner_callback
    outer_callback
        Arbitrary functions run during the loop after updating the state.
        `inner_callback` is called after each update, while `outer_callback` is
        called after completing an inner loop. For the signature, see
        `Callback`. `inner_callback` is called under the jax jit, so the
        argument values are not available at the time the Python code is
        executed. Use the utilities in `jax.debug` to access the values at
        actual runtime. The callbacks may return new values for the MCMC state
        and the callback state.
    callback_state
        The initial state for the callbacks.
    burnin_extractor
    main_extractor
        Functions that extract the variables to be saved respectively only in
        the main trace and in both traces, given the MCMC state as argument.
        Must return a pytree, and must be vmappable.

    Returns
    -------
    bart : State
        The final MCMC state.
    burnin_trace : PyTree[Shaped[Array, 'n_burn *']]
        The trace of the burn-in phase. For the default layout, see `BurninTrace`.
    main_trace : PyTree[Shaped[Array, 'n_save *']]
        The trace of the main phase. For the default layout, see `MainTrace`.

    Raises
    ------
    ValueError
        If `inner_loop_length` is not a divisor of the total number of
        iterations and `allow_overflow` is `False`.

    Notes
    -----
    The number of MCMC updates is ``n_burn + n_skip * n_save``. The traces do
    not include the initial state, and include the final state.
    """

    def empty_trace(length, bart, extractor):
        return jax.vmap(extractor, in_axes=None, out_axes=0, axis_size=length)(bart)

    burnin_trace = empty_trace(n_burn, bart, burnin_extractor)
    main_trace = empty_trace(n_save, bart, main_extractor)

    # determine number of iterations for inner and outer loops
    n_iters = n_burn + n_skip * n_save
    if inner_loop_length is None:
        inner_loop_length = n_iters
    n_outer = n_iters // inner_loop_length
    if n_iters % inner_loop_length:
        if allow_overflow:
            n_outer += 1
        else:
            msg = f'{n_iters=} is not divisible by {inner_loop_length=}'
            raise ValueError(msg)

    carry = (bart, 0, key, burnin_trace, main_trace, callback_state)
    for i_outer in range(n_outer):
        carry = _run_mcmc_inner_loop(
            carry,
            inner_loop_length,
            inner_callback,
            burnin_extractor,
            main_extractor,
            n_burn,
            n_save,
            n_skip,
            i_outer,
        )
        if outer_callback is not None:
            bart, i_total, key, burnin_trace, main_trace, callback_state = carry
            i_total -= 1  # because i_total is updated at the end of the inner loop
            i_skip = _compute_i_skip(i_total, n_burn, n_skip)
            rt = outer_callback(
                bart=bart,
                burnin=i_total < n_burn,
                overflow=i_total >= n_iters,
                i_total=i_total,
                i_skip=i_skip,
                callback_state=callback_state,
                n_burn=n_burn,
                n_save=n_save,
                n_skip=n_skip,
                i_outer=i_outer,
                inner_loop_length=inner_loop_length,
            )
            if rt is not None:
                bart, callback_state = rt
                i_total += 1
                carry = (bart, i_total, key, burnin_trace, main_trace, callback_state)

    bart, _, _, burnin_trace, main_trace, _ = carry
    return bart, burnin_trace, main_trace


def _compute_i_skip(i_total, n_burn, n_skip):
    burnin = i_total < n_burn
    return jnp.where(
        burnin,
        i_total + 1,
        (i_total + 1) % n_skip + jnp.where(i_total + 1 < n_skip, n_burn, 0),
    )


@partial(jax.jit, donate_argnums=(0,), static_argnums=(1, 2, 3, 4))
def _run_mcmc_inner_loop(
    carry,
    inner_loop_length,
    inner_callback,
    burnin_extractor,
    main_extractor,
    n_burn,
    n_save,
    n_skip,
    i_outer,
):
    def loop(carry, _):
        bart, i_total, key, burnin_trace, main_trace, callback_state = carry

        keys = jaxext.split(key)
        key = keys.pop()
        bart = mcmcstep.step(keys.pop(), bart)

        burnin = i_total < n_burn
        if inner_callback is not None:
            i_skip = _compute_i_skip(i_total, n_burn, n_skip)
            rt = inner_callback(
                bart=bart,
                burnin=burnin,
                overflow=i_total >= n_burn + n_save * n_skip,
                i_total=i_total,
                i_skip=i_skip,
                callback_state=callback_state,
                n_burn=n_burn,
                n_save=n_save,
                n_skip=n_skip,
                i_outer=i_outer,
                inner_loop_length=inner_loop_length,
            )
            if rt is not None:
                bart, callback_state = rt

        i_burnin = i_total  # out of bounds after burnin
        i_main = jnp.where(burnin, 0, (i_total - n_burn) // n_skip)

        def update_trace(index, trace, state):
            def assign_at_index(trace_array, state_array):
                if trace_array.size:
                    return trace_array.at[index, ...].set(state_array)
                else:
                    # this handles the case where a trace is empty (e.g.,
                    # no burn-in) because jax refuses to index into an array
                    # of length 0
                    return trace_array

            return tree.map(assign_at_index, trace, state)

        burnin_trace = update_trace(i_burnin, burnin_trace, burnin_extractor(bart))
        main_trace = update_trace(i_main, main_trace, main_extractor(bart))

        i_total += 1
        carry = (bart, i_total, key, burnin_trace, main_trace, callback_state)
        return carry, None

    carry, _ = lax.scan(loop, carry, None, inner_loop_length)
    return carry


def make_print_callbacks(dot_every_inner=1, report_every_outer=1):
    """
    Prepare logging callbacks for `run_mcmc`.

    Prepare callbacks which print a dot on every iteration, and a longer
    report outer loop iteration.

    Parameters
    ----------
    dot_every_inner : int, default 1
        A dot is printed every `dot_every_inner` MCMC iterations.
    report_every_outer : int, default 1
        A report is printed every `report_every_outer` outer loop
        iterations.

    Returns
    -------
    kwargs : dict
        A dictionary with the arguments to pass to `run_mcmc` as keyword
        arguments to set up the callbacks.

    Examples
    --------
    >>> run_mcmc(..., **make_print_callbacks())
    """
    return dict(
        inner_callback=_print_callback_inner,
        outer_callback=_print_callback_outer,
        callback_state=dict(
            dot_every_inner=dot_every_inner, report_every_outer=report_every_outer
        ),
    )


def _print_callback_inner(*, i_total, callback_state, **_):
    dot_every_inner = callback_state['dot_every_inner']
    if dot_every_inner is not None:
        cond = (i_total + 1) % dot_every_inner == 0
        debug.callback(_print_dot, cond)


def _print_dot(cond):
    if cond:
        print('.', end='', flush=True)  # noqa: T201
        # logging can't do in-line printing so I'll stick to print


def _print_callback_outer(
    *,
    bart: State,
    burnin,
    overflow,
    i_total,
    n_burn,
    n_save,
    n_skip,
    callback_state,
    i_outer,
    inner_loop_length,
    **_,
):
    report_every_outer = callback_state['report_every_outer']
    if report_every_outer is not None:
        dot_every_inner = callback_state['dot_every_inner']
        if dot_every_inner is None:
            newline = False
        else:
            newline = dot_every_inner < inner_loop_length
        debug.callback(
            _print_report,
            cond=(i_outer + 1) % report_every_outer == 0,
            newline=newline,
            burnin=burnin,
            overflow=overflow,
            i_total=i_total,
            n_iters=n_burn + n_save * n_skip,
            grow_prop_count=bart.forest.grow_prop_count,
            grow_acc_count=bart.forest.grow_acc_count,
            prune_prop_count=bart.forest.prune_prop_count,
            prune_acc_count=bart.forest.prune_acc_count,
            prop_total=len(bart.forest.leaf_tree),
            fill=grove.forest_fill(bart.forest.split_tree),
        )


def _convert_jax_arrays_in_args(func):
    """Remove jax arrays from a function arguments.

    Converts all jax.Array instances in the arguments to either Python scalars
    or numpy arrays.
    """

    def convert_jax_arrays(pytree):
        def convert_jax_arrays(val):
            if not isinstance(val, jax.Array):
                return val
            elif val.shape:
                return numpy.array(val)
            else:
                return val.item()

        return tree.map(convert_jax_arrays, pytree)

    @wraps(func)
    def new_func(*args, **kw):
        args = convert_jax_arrays(args)
        kw = convert_jax_arrays(kw)
        return func(*args, **kw)

    return new_func


@_convert_jax_arrays_in_args
# convert all jax arrays in arguments because operations on them could lead to
# deadlock with the main thread
def _print_report(
    *,
    cond,
    newline,
    burnin,
    overflow,
    i_total,
    n_iters,
    grow_prop_count,
    grow_acc_count,
    prune_prop_count,
    prune_acc_count,
    prop_total,
    fill,
):
    if cond:
        newline = '\n' if newline else ''

        def acc_string(acc_count, prop_count):
            if prop_count:
                return f'{acc_count / prop_count:.0%}'
            else:
                return 'n/d'

        grow_prop = grow_prop_count / prop_total
        prune_prop = prune_prop_count / prop_total
        grow_acc = acc_string(grow_acc_count, grow_prop_count)
        prune_acc = acc_string(prune_acc_count, prune_prop_count)

        if burnin:
            flag = ' (burnin)'
        elif overflow:
            flag = ' (overflow)'
        else:
            flag = ''

        print(  # noqa: T201, see _print_dot for why not logging
            f'{newline}It {i_total + 1}/{n_iters} '
            f'grow P={grow_prop:.0%} A={grow_acc}, '
            f'prune P={prune_prop:.0%} A={prune_acc}, '
            f'fill={fill:.0%}{flag}'
        )


class Trace(grove.TreeHeaps, Protocol):
    """Protocol for a MCMC trace."""

    offset: Float32[Array, ' trace_length']


class TreesTrace(Module):
    """Implementation of `bartz.grove.TreeHeaps` for an MCMC trace."""

    leaf_tree: Float32[Array, 'trace_length num_trees 2**d']
    var_tree: UInt[Array, 'trace_length num_trees 2**(d-1)']
    split_tree: UInt[Array, 'trace_length num_trees 2**(d-1)']

    @classmethod
    def from_dataclass(cls, obj: grove.TreeHeaps):
        """Create a `TreesTrace` from any `bartz.grove.TreeHeaps`."""
        return cls(**{f.name: getattr(obj, f.name) for f in fields(cls)})


@jax.jit
def evaluate_trace(
    trace: Trace, X: UInt[Array, 'p n']
) -> Float32[Array, 'trace_length n']:
    """
    Compute predictions for all iterations of the BART MCMC.

    Parameters
    ----------
    trace
        A trace of the BART MCMC, as returned by `run_mcmc`.
    X
        The predictors matrix, with `p` predictors and `n` observations.

    Returns
    -------
    The predictions for each iteration of the MCMC.
    """
    evaluate_trees = partial(grove.evaluate_forest, sum_trees=False)
    evaluate_trees = jaxext.autobatch(evaluate_trees, 2**29, (None, 0))
    trees = TreesTrace.from_dataclass(trace)

    def loop(_, item):
        offset, trees = item
        values = evaluate_trees(X, trees)
        return None, offset + jnp.sum(values, axis=0, dtype=jnp.float32)

    _, y = lax.scan(loop, None, (trace.offset, trees))
    return y


@partial(jax.jit, static_argnums=(0,))
def compute_varcount(
    p: int, trace: grove.TreeHeaps
) -> Int32[Array, 'trace_length {p}']:
    """
    Count how many times each predictor is used in each MCMC state.

    Parameters
    ----------
    p
        The number of predictors.
    trace
        A trace of the BART MCMC, as returned by `run_mcmc`.

    Returns
    -------
    Histogram of predictor usage in each MCMC state.
    """
    vmapped_var_histogram = jax.vmap(grove.var_histogram, in_axes=(None, 0, 0))
    return vmapped_var_histogram(p, trace.var_tree, trace.split_tree)
