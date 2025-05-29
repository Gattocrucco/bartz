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

"""Functions that implement the full BART posterior MCMC loop."""

import functools

import jax
import numpy
from jax import debug, lax, tree
from jax import numpy as jnp
from jaxtyping import Array, Real

from . import grove, jaxext, mcmcstep
from .mcmcstep import State


def default_onlymain_extractor(state: State) -> dict[str, Real[Array, 'samples *']]:
    """Extract variables for the main trace, to be used in `run_mcmc`."""
    return dict(
        leaf_trees=state.forest.leaf_trees,
        var_trees=state.forest.var_trees,
        split_trees=state.forest.split_trees,
        offset=state.offset,
    )


def default_both_extractor(state: State) -> dict[str, Real[Array, 'samples *'] | None]:
    """Extract variables for main & burn-in traces, to be used in `run_mcmc`."""
    return dict(
        sigma2=state.sigma2,
        grow_prop_count=state.forest.grow_prop_count,
        grow_acc_count=state.forest.grow_acc_count,
        prune_prop_count=state.forest.prune_prop_count,
        prune_acc_count=state.forest.prune_acc_count,
        log_likelihood=state.forest.log_likelihood,
        log_trans_prior=state.forest.log_trans_prior,
    )


def run_mcmc(
    key,
    bart,
    n_save,
    *,
    n_burn=0,
    n_skip=1,
    inner_loop_length=None,
    allow_overflow=False,
    inner_callback=None,
    outer_callback=None,
    callback_state=None,
    onlymain_extractor=default_onlymain_extractor,
    both_extractor=default_both_extractor,
):
    """
    Run the MCMC for the BART posterior.

    Parameters
    ----------
    key : jax.dtypes.prng_key array
        A key for random number generation.
    bart : dict
        The initial MCMC state, as created and updated by the functions in
        `bartz.mcmcstep`. The MCMC loop uses buffer donation to avoid copies,
        so this variable is invalidated after running `run_mcmc`. Make a copy
        beforehand to use it again.
    n_save : int
        The number of iterations to save.
    n_burn : int, default 0
        The number of initial iterations which are not saved.
    n_skip : int, default 1
        The number of iterations to skip between each saved iteration, plus 1.
        The effective burn-in is ``n_burn + n_skip - 1``.
    inner_loop_length : int, optional
        The MCMC loop is split into an outer and an inner loop. The outer loop
        is in Python, while the inner loop is in JAX. `inner_loop_length` is the
        number of iterations of the inner loop to run for each iteration of the
        outer loop. If not specified, the outer loop will iterate just once,
        with all iterations done in a single inner loop run. The inner stride is
        unrelated to the stride used for saving the trace.
    allow_overflow : bool, default False
        If `False`, `inner_loop_length` must be a divisor of the total number of
        iterations ``n_burn + n_skip * n_save``. If `True` and
        `inner_loop_length` is not a divisor, some of the MCMC iterations in the
        last outer loop iteration will not be saved to the trace.
    inner_callback : callable, optional
    outer_callback : callable, optional
        Arbitrary functions run during the loop after updating the state.
        `inner_callback` is called after each update, while `outer_callback` is
        called after completing an inner loop. The callbacks are invoked with
        the following arguments, passed by keyword:

        bart : dict
            The MCMC state just after updating it.
        burnin : bool
            Whether the last iteration was in the burn-in phase.
        overflow : bool
            Whether the last iteration was in the overflow phase (iterations
            not saved due to `inner_loop_length` not being a divisor of the
            total number of iterations).
        i_total : int
            The index of the last MCMC iteration (0-based).
        i_skip : int
            The number of MCMC updates from the last saved state. The initial
            state counts as saved, even if it's not copied into the trace.
        callback_state : jax pytree
            The callback state, initially set to the argument passed to
            `run_mcmc`, afterwards to the value returned by the last invocation
            of `inner_callback` or `outer_callback`.
        n_burn, n_save, n_skip : int
            The corresponding arguments as-is.
        i_outer : int
            The index of the last outer loop iteration (0-based).
        inner_loop_length : int
            The number of MCMC iterations in the inner loop.

        `inner_callback` is called under the jax jit, so the argument values are
        not available at the time the Python code is executed. Use the utilities
        in `jax.debug` to access the values at actual runtime.

        The callbacks must return two values:

        bart : dict
            A possibly modified MCMC state. To avoid modifying the state,
            return the `bart` argument passed to the callback as-is.
        callback_state : jax pytree
            The new state to be passed on the next callback invocation.

        For convenience, if a callback returns `None`, the states are not
        updated.
    callback_state : jax pytree, optional
        The initial state for the callbacks.
    onlymain_extractor : callable, optional
    both_extractor : callable, optional
        Functions that extract the variables to be saved respectively only in
        the main trace and in both traces, given the MCMC state as argument.
        Must return a pytree, and must be vmappable.

    Returns
    -------
    bart : dict
        The final MCMC state.
    burnin_trace : dict of (n_burn, ...) arrays
        The trace of the burn-in phase, containing the following subset of
        fields from the `bart` dictionary, with an additional head index that
        runs over MCMC iterations: 'sigma2', 'grow_prop_count',
        'grow_acc_count', 'prune_prop_count', 'prune_acc_count' (or if specified
        the fields in `tracevars_both`).
    main_trace : dict of (n_save, ...) arrays
        The trace of the main phase, containing the following subset of fields
        from the `bart` dictionary, with an additional head index that runs over
        MCMC iterations: 'leaf_trees', 'var_trees', 'split_trees' (or if
        specified the fields in `tracevars_onlymain`), plus the fields in
        `burnin_trace`.

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

    trace_both = empty_trace(n_burn + n_save, bart, both_extractor)
    trace_onlymain = empty_trace(n_save, bart, onlymain_extractor)

    # determine number of iterations for inner and outer loops
    n_iters = n_burn + n_skip * n_save
    if inner_loop_length is None:
        inner_loop_length = n_iters
    n_outer = n_iters // inner_loop_length
    if n_iters % inner_loop_length:
        if allow_overflow:
            n_outer += 1
        else:
            raise ValueError(f'{n_iters=} is not divisible by {inner_loop_length=}')

    carry = (bart, 0, key, trace_both, trace_onlymain, callback_state)
    for i_outer in range(n_outer):
        carry = _run_mcmc_inner_loop(
            carry,
            inner_loop_length,
            inner_callback,
            onlymain_extractor,
            both_extractor,
            n_burn,
            n_save,
            n_skip,
            i_outer,
        )
        if outer_callback is not None:
            bart, i_total, key, trace_both, trace_onlymain, callback_state = carry
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
                carry = (bart, i_total, key, trace_both, trace_onlymain, callback_state)

    bart, _, _, trace_both, trace_onlymain, _ = carry

    burnin_trace = tree.map(lambda x: x[:n_burn, ...], trace_both)
    main_trace = tree.map(lambda x: x[n_burn:, ...], trace_both)
    main_trace.update(trace_onlymain)

    return bart, burnin_trace, main_trace


def _compute_i_skip(i_total, n_burn, n_skip):
    burnin = i_total < n_burn
    return jnp.where(
        burnin,
        i_total + 1,
        (i_total + 1) % n_skip + jnp.where(i_total + 1 < n_skip, n_burn, 0),
    )


@functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1, 2, 3, 4))
def _run_mcmc_inner_loop(
    carry,
    inner_loop_length,
    inner_callback,
    onlymain_extractor,
    both_extractor,
    n_burn,
    n_save,
    n_skip,
    i_outer,
):
    def loop(carry, _):
        bart, i_total, key, trace_both, trace_onlymain, callback_state = carry

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

        i_onlymain = jnp.where(burnin, 0, (i_total - n_burn) // n_skip)
        i_both = jnp.where(burnin, i_total, n_burn + i_onlymain)

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

        trace_onlymain = update_trace(
            i_onlymain, trace_onlymain, onlymain_extractor(bart)
        )
        trace_both = update_trace(i_both, trace_both, both_extractor(bart))

        i_total += 1
        carry = (bart, i_total, key, trace_both, trace_onlymain, callback_state)
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
        print('.', end='', flush=True)


def _print_callback_outer(
    *,
    bart,
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
            prop_total=len(bart.forest.leaf_trees),
            fill=grove.forest_fill(bart.forest.split_trees),
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

    @functools.wraps(func)
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
                return ' n/d'

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

        print(
            f'{newline}It {i_total + 1}/{n_iters} '
            f'grow P={grow_prop:.0%} A={grow_acc}, '
            f'prune P={prune_prop:.0%} A={prune_acc}, '
            f'fill={fill:.0%}{flag}'
        )


@jax.jit
def evaluate_trace(trace, X):
    """
    Compute predictions for all iterations of the BART MCMC.

    Parameters
    ----------
    trace : dict
        A trace of the BART MCMC, as returned by `run_mcmc`.
    X : array (p, n)
        The predictors matrix, with `p` predictors and `n` observations.

    Returns
    -------
    y : array (n_trace, n)
        The predictions for each iteration of the MCMC.
    """
    evaluate_trees = functools.partial(grove.evaluate_forest, sum_trees=False)
    evaluate_trees = jaxext.autobatch(evaluate_trees, 2**29, (None, 0, 0, 0))

    def loop(_, row):
        values = evaluate_trees(
            X, row['leaf_trees'], row['var_trees'], row['split_trees']
        )
        return None, row['offset'] + jnp.sum(values, axis=0, dtype=jnp.float32)

    _, y = lax.scan(loop, None, trace)
    return y
