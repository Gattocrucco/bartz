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

"""
Functions that implement the full BART posterior MCMC loop.
"""

import functools

import jax
from jax import debug, lax, random, tree
from jax import numpy as jnp

from . import grove, jaxext, mcmcstep

TRACEVARS_BOTH_DEFAULT = (
    'sigma2',
    'grow_prop_count',
    'grow_acc_count',
    'prune_prop_count',
    'prune_acc_count',
    'ratios',
)
TRACEVARS_ONLYMAIN_DEFAULT = ('leaf_trees', 'var_trees', 'split_trees')


@functools.partial(
    jax.jit,
    static_argnums=(2,),
    static_argnames=('n_save', 'n_burn', 'n_skip', 'callback'),
)
def run_mcmc(
    key,
    bart,
    n_save,
    *,
    n_burn=0,
    n_skip=1,
    callback=None,
    callback_state=None,
    tracevars_onlymain=TRACEVARS_ONLYMAIN_DEFAULT,
    tracevars_both=TRACEVARS_BOTH_DEFAULT,
):
    """
    Run the MCMC for the BART posterior.

    Parameters
    ----------
    key : jax.dtypes.prng_key array
        A key for random number generation.
    bart : dict
        The initial MCMC state, as created and updated by the functions in
        `bartz.mcmcstep`.
    n_save : int
        The number of iterations to save.
    n_burn : int, default 0
        The number of initial iterations which are not saved.
    n_skip : int, default 1
        The number of iterations to skip between each saved iteration, plus 1.
        The effective burn-in is ``n_burn + n_skip - 1``.
    callback : callable, optional
        An arbitrary function run at each iteration, called with the following
        arguments, passed by keyword:

        bart : dict
            The MCMC state just after updating it.
        burnin : bool
            Whether the last iteration was in the burn-in phase.
        i_total : int
            The index of the last iteration (0-based).
        i_skip : int
            The number of MCMC updates from the last saved state. The initial
            state counts as saved, even if it's not copied into the trace.
        n_burn, n_save, n_skip : int
            The corresponding arguments as-is.
        callback_state : jax pytree
            The callback state, initially set to the argument passed to
            `run_mcmc`, afterwards to value returned by the last invocation of
            `callback`.

        Since this function is called under the jax jit, the values are not
        available at the time the Python code is executed. Use the utilities in
        `jax.debug` to access the values at actual runtime.

        The function must return two values:

        bart : dict
            A possibly modified MCMC state. To avoid modifying the state,
            return the `bart` argument passed to `callback` as-is.
        callback_state : jax pytree
            The new state for the callback.

        For convenience, if `callback` returns `None`, the states are not
        updated.
    callback_state : jax pytree, optional
        The initial state for the callback.
    tracevars_onlymain : tuple of str
        The names of the fields in `bart` to include in the main trace but not
        in the burnin trace. For the burn-in/main distinction to make sense,
        these fields should contain scalars or small arrays.
    tracevars_both : tuple of str
        The names of the fields in `bart` to include in both traces.

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

    Notes
    -----
    The number of MCMC updates is ``n_burn + n_skip * n_save``. The traces do
    not include the initial state, and include the final state.
    """

    def empty_trace(length, bart, tracelist):
        bart = {k: v for k, v in bart.items() if k in tracelist}
        return jax.vmap(lambda x: x, in_axes=None, out_axes=0, axis_size=length)(bart)

    trace_both = empty_trace(n_burn + n_save, bart, tracevars_both)
    trace_onlymain = empty_trace(n_save, bart, tracevars_onlymain)

    callback_kw = dict(n_burn=n_burn, n_save=n_save, n_skip=n_skip)

    carry = (bart, 0, key, trace_both, trace_onlymain, callback_state)

    def loop(carry, _):
        bart, i_total, key, trace_both, trace_onlymain, callback_state = carry

        key, subkey = random.split(key)
        bart = mcmcstep.step(subkey, bart)

        burnin = i_total < n_burn
        if callback is not None:
            i_skip = jnp.where(
                burnin,
                i_total + 1,
                (i_total + 1) % n_skip + jnp.where(i_total + 1 < n_skip, n_burn, 0),
            )
            rt = callback(
                bart=bart,
                burnin=burnin,
                i_total=i_total,
                i_skip=i_skip,
                callback_state=callback_state,
                **callback_kw,
            )
            if rt is not None:
                bart, callback_state = rt

        i_onlymain = jnp.where(burnin, 0, (i_total - n_burn) // n_skip)
        i_both = jnp.where(burnin, i_total, n_burn + i_onlymain)

        def update_trace(index, trace, bart):
            bart = {k: v for k, v in bart.items() if k in trace}

            def assign_at_index(trace_array, state_array):
                if trace_array.size:
                    return trace_array.at[index, ...].set(state_array)
                else:
                    # this handles the case where a trace is empty (e.g.,
                    # no burn-in) because jax refuses to index into an array
                    # of length 0
                    return trace_array

            return tree.map(assign_at_index, trace, bart)

        trace_onlymain = update_trace(i_onlymain, trace_onlymain, bart)
        trace_both = update_trace(i_both, trace_both, bart)

        i_total += 1
        carry = (bart, i_total, key, trace_both, trace_onlymain, callback_state)
        return carry, None

    carry, _ = lax.scan(loop, carry, None, n_burn + n_skip * n_save)

    bart, _, _, trace_both, trace_onlymain, _ = carry

    burnin_trace = tree.map(lambda x: x[:n_burn, ...], trace_both)
    main_trace = tree.map(lambda x: x[n_burn:, ...], trace_both)
    main_trace.update(trace_onlymain)

    return bart, burnin_trace, main_trace


def simple_print_callback(
    *, bart, burnin, i_total, i_skip, n_burn, n_save, n_skip, callback_state
):
    """
    A logging callback function for MCMC iterations.

    Use this function with `run_mcmc` as:

    >>> run_mcmc(..., callback=simple_print_callback, callback_state=<printevery>)

    Where <printevery> is an integer specifying how frequently to print.
    """
    printevery = callback_state
    prop_total = len(bart['leaf_trees'])
    # all the following arithmetic could be moved in the python callback under
    # the conditional
    grow_prop = bart['grow_prop_count'] / prop_total
    prune_prop = bart['prune_prop_count'] / prop_total
    grow_acc = bart['grow_acc_count'] / bart['grow_prop_count']
    prune_acc = bart['prune_acc_count'] / bart['prune_prop_count']
    n_total = n_burn + n_save * n_skip
    printcond = (i_total + 1) % printevery == 0
    debug.callback(
        _simple_print_callback,
        burnin,
        i_total,
        n_total,
        grow_prop,
        grow_acc,
        prune_prop,
        prune_acc,
        printcond,
    )


def _simple_print_callback(
    burnin, i_total, n_total, grow_prop, grow_acc, prune_prop, prune_acc, printcond
):
    if printcond:
        burnin_flag = ' (burnin)' if burnin else ''
        total_str = str(n_total)
        ndigits = len(total_str)
        i_str = str(i_total.item() + 1).rjust(ndigits)
        # I do i_total.item() + 1 instead of just i_total + 1 to solve a bug
        # originating when jax is combined with some outdated dependencies. (I
        # did not track down which dependencies exactly.) Doing .item() makes
        # the + 1 operation be done by Python instead of by jax. The bug is that
        # jax hangs completely, with a secondary thread blocked at this line.
        print(
            f'Iteration {i_str}/{total_str} '
            f'P_grow={grow_prop:.2f} P_prune={prune_prop:.2f} '
            f'A_grow={grow_acc:.2f} A_prune={prune_acc:.2f}{burnin_flag}'
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

    def loop(_, state):
        values = evaluate_trees(
            X, state['leaf_trees'], state['var_trees'], state['split_trees']
        )
        return None, jnp.sum(values, axis=0, dtype=jnp.float32)

    _, y = lax.scan(loop, None, trace)
    return y
