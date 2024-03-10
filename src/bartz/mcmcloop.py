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

from . import mcmcstep

def run_mcmc(bart, n_burn, n_save, n_skip, callback, key):

    tracelist_burnin = 'sigma2', 'grow_prop_count', 'grow_acc_count', 'prune_prop_count', 'prune_acc_count'

    tracelist_main = tracelist_burnin + ('leaf_trees', 'var_trees', 'split_trees')

    callback_kw = dict(n_burn=n_burn, n_save=n_save, n_skip=n_skip)

    def inner_loop(carry, _, tracelist, burnin):
        bart, i_total, i_skip, key = carry
        key, subkey = random.split(key)
        bart = mcmcstep.mcmc_step(bart, subkey)
        callback(bart=bart, burnin=burnin, i_total=i_total, i_skip=i_skip, **callback_kw)
        output = {key: bart[key] for key in tracelist}
        return (bart, i_total + 1, i_skip + 1, key), output

    key, subkey = random.split(key)
    carry = bart, 0, 0, subkey
    burnin_loop = functools.partial(inner_loop, tracelist=tracelist_burnin, burnin=True)
    (bart, i_total, _, _), burnin_trace = lax.scan(burnin_loop, carry, None, n_burn)

    def outer_loop(carry, _):
        bart, i_total, key = carry
        main_inner_loop = functools.partial(inner_loop, tracelist=[], burnin=False)
        key, subkey = random.split(key)
        inner_carry = bart, i_total, 0, subkey
        (bart, i_total, _, _), _ = lax.scan(main_inner_loop, inner_carry, None, n_skip)
        output = {key: bart[key] for key in tracelist_main}
        return (bart, i_total, key), output

    carry = bart, i_total, key
    (bart, _, _), main_trace = lax.scan(outer_loop, carry, None, n_save)

    return bart, burnin_trace, main_trace

def simple_print_callback(*, bart, burnin, i_total, i_skip, n_burn, n_save, n_skip, printevery):
    prop_total = len(bart['leaf_trees'])
    grow_prop = bart['grow_prop_count'] / prop_total
    prune_prop = bart['prune_prop_count'] / prop_total
    grow_acc = bart['grow_acc_count'] / bart['grow_prop_count']
    prune_acc = bart['prune_acc_count'] / bart['prune_prop_count']
    n_total = n_burn + n_save
    debug.callback(simple_print_callback_impl, burnin, i_total, n_total, grow_prop, grow_acc, prune_prop, prune_acc, printevery)

def simple_print_callback_impl(burnin, i_total, n_total, grow_prop, grow_acc, prune_prop, prune_acc, printevery):
    if i_total % printevery == 0:
        burnin_flag = ' (burnin)' if burnin else ''
        print(f'Iteration {i_total + 1:4d}/{n_total:d}{burnin_flag} '
            f'P_grow={grow_prop:.2f} P_prune={prune_prop:.2f} '
            f'A_grow={grow_acc:.2f} A_prune={prune_acc:.2f}')
