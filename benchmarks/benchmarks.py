# bartz/benchmarks/benchmarks.py
#
# Copyright (c) 2025, Giacomo Petrillo
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

import functools
import inspect

import jax
from jax import numpy as jnp
from jax import random

import bartz


class TimeCompilation:
    def setup(self):
        key = random.key(202504251557)
        keys = list(random.split(key, 3))

        # generate simulated data
        p = 2
        n = 30
        sigma = 0.1
        T = 2
        X = random.uniform(keys.pop(), (p, n), float, -2, 2)
        f = lambda X: jnp.sum(jnp.cos(2 * jnp.pi / T * X), axis=0)
        y = f(X) + sigma * random.normal(keys.pop(), (n,))

        # dry run the interface to generate an initial bart state conveniently
        if inspect.ismodule(bartz.BART):
            gbart = bartz.BART.gbart
        else:
            gbart = bartz.BART  # pre v0.2.0
        bart = gbart(X, y, ndpost=0, nskip=0)
        self.kw = dict(
            key=keys.pop(),
            bart=bart._mcmc_state,
            n_save=1,
            n_burn=1,
            n_skip=1,
            callback=lambda **_: None,
        )

    def time_run_mcmc_compile(self):
        @functools.partial(
            jax.jit, static_argnames=('n_save', 'n_burn', 'n_skip', 'callback')
        )
        def f(**kw):
            return bartz.mcmcloop.run_mcmc(**kw)

        f.lower(**self.kw).compile()
