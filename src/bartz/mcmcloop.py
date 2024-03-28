# bartz/src/bartz/mcmcloop.py
#
# Copyright (c) 2024, Giacomo Petrillo
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
from jax import random
from jax import debug
from jax import numpy as jnp
from jax import lax

from . import mcmcstep
from . import grove

@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def run_mcmc(bart, n_burn, n_save, n_skip, callback, key):
    """
    Run the MCMC for the BART posterior.

    Parameters
    ----------
    bart : dict
        The initial MCMC state, as created and updated by the functions in
        `bartz.mcmcstep`.
    n_burn : int
        The number of initial iterations which are not saved.
    n_save : int
        The number of iterations to save.
    n_skip : int
        The number of iterations to skip between each saved iteration, plus 1.
    callback : callable
        An arbitrary function run at each iteration, called with the following
        arguments, passed by keyword:

        bart : dict
            The current MCMC state.
        burnin : bool
            Whether the last iteration was in the burn-in phase.
        i_total : int
            The index of the last iteration (0-based).
        i_skip : int
            The index of the last iteration, starting from the last saved
            iteration.
        n_burn, n_save, n_skip : int
            The corresponding arguments as-is.

        Since this function is called under the jax jit, the values are not
        available at the time the Python code is executed. Use the utilities in
        `jax.debug` to access the values at actual runtime.
    key : jax.dtypes.prng_key array
        The key for random number generation.

    Returns
    -------
    bart : dict
        The final MCMC state.
    burnin_trace : dict
        The trace of the burn-in phase, containing the following subset of
        fields from the `bart` dictionary, with an additional head index that
        runs over MCMC iterations: 'sigma2', 'grow_prop_count',
        'grow_acc_count', 'prune_prop_count', 'prune_acc_count'.
    main_trace : dict
        The trace of the main phase, containing the following subset of fields
        from the `bart` dictionary, with an additional head index that runs
        over MCMC iterations: 'leaf_trees', 'var_trees', 'split_trees', plus
        the fields in `burnin_trace`.
    """

    tracelist_burnin = 'sigma2', 'grow_prop_count', 'grow_acc_count', 'prune_prop_count', 'prune_acc_count'

    tracelist_main = tracelist_burnin + ('leaf_trees', 'var_trees', 'split_trees')

    callback_kw = dict(n_burn=n_burn, n_save=n_save, n_skip=n_skip)

    def inner_loop(carry, _, tracelist, burnin):
        bart, i_total, i_skip, key = carry
        key, subkey = random.split(key)
        bart = mcmcstep.step(bart, subkey)
        callback(bart=bart, burnin=burnin, i_total=i_total, i_skip=i_skip, **callback_kw)
        output = {key: bart[key] for key in tracelist}
        return (bart, i_total + 1, i_skip + 1, key), output

    def empty_trace(bart, tracelist):
        return {
            key: jnp.empty((0,) + bart[key].shape, bart[key].dtype)
            for key in tracelist
        }

    if n_burn > 0:
        carry = bart, 0, 0, key
        burnin_loop = functools.partial(inner_loop, tracelist=tracelist_burnin, burnin=True)
        (bart, i_total, _, key), burnin_trace = lax.scan(burnin_loop, carry, None, n_burn)
    else:
        i_total = 0
        burnin_trace = empty_trace(bart, tracelist_burnin)

    def outer_loop(carry, _):
        bart, i_total, key = carry
        main_loop = functools.partial(inner_loop, tracelist=[], burnin=False)
        inner_carry = bart, i_total, 0, key
        (bart, i_total, _, key), _ = lax.scan(main_loop, inner_carry, None, n_skip)
        output = {key: bart[key] for key in tracelist_main}
        return (bart, i_total, key), output

    if n_save > 0:
        carry = bart, i_total, key
        (bart, _, _), main_trace = lax.scan(outer_loop, carry, None, n_save)
    else:
        main_trace = empty_trace(bart, tracelist_main)

    return bart, burnin_trace, main_trace

    # TODO I could add an argument callback_state to carry over state. This would allow e.g. accumulating counts. If I made the callback return the mcmc state, I could modify the mcmc from the callback.

@functools.lru_cache
    # cache to make the callback function object unique, such that the jit
    # of run_mcmc recognizes it => with the callback state, I can make
    # printevery a runtime quantity
def make_simple_print_callback(printevery):
    """
    Create a logging callback function for MCMC iterations.

    Parameters
    ----------
    printevery : int
        The number of iterations between each log.

    Returns
    -------
    callback : callable
        A function in the format required by `run_mcmc`.
    """
    def callback(*, bart, burnin, i_total, i_skip, n_burn, n_save, n_skip):
        prop_total = len(bart['leaf_trees'])
        grow_prop = bart['grow_prop_count'] / prop_total
        prune_prop = bart['prune_prop_count'] / prop_total
        grow_acc = bart['grow_acc_count'] / bart['grow_prop_count']
        prune_acc = bart['prune_acc_count'] / bart['prune_prop_count']
        n_total = n_burn + n_save * n_skip
        printcond = (i_total + 1) % printevery == 0
        debug.callback(_simple_print_callback, burnin, i_total, n_total, grow_prop, grow_acc, prune_prop, prune_acc, printcond)
    return callback

def _simple_print_callback(burnin, i_total, n_total, grow_prop, grow_acc, prune_prop, prune_acc, printcond):
    if printcond:
        burnin_flag = ' (burnin)' if burnin else ''
        total_str = str(n_total)
        ndigits = len(total_str)
        i_str = str(i_total + 1).rjust(ndigits)
        print(f'Iteration {i_str}/{total_str} '
            f'P_grow={grow_prop:.2f} P_prune={prune_prop:.2f} '
            f'A_grow={grow_acc:.2f} A_prune={prune_acc:.2f}{burnin_flag}')

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
    def loop(_, state):
        return None, grove.evaluate_forest(X, state['leaf_trees'], state['var_trees'], state['split_trees'], jnp.float32)
    _, y = lax.scan(loop, None, trace)
    return y
