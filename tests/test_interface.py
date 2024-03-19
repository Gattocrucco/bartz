# bartz/tests/test_interface.py
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

from jax import numpy as jnp
from jax import random

import bartz

def test_basic(key):
    
    # DGP config
    n = 30 # number of datapoints
    p = 10 # number of covariates
    sigma = 0.1 # noise standard deviation
    def f(x): # conditional mean
        T = 2
        return jnp.sum(jnp.cos(2 * jnp.pi / T * x), axis=0)
    def gen_X(key, p, n): # predictors
        return random.uniform(key, (p, n), float, -2, 2)

    # random keys
    key1, key2, key3 = random.split(key, 3)

    # generate data
    X = gen_X(key1, p, 2 * n)
    y = f(X) + sigma * random.normal(key2, (2 * n,))

    # split in train/test
    X_train, X_test = X[:, :n], X[:, n:]
    y_train, y_test = y[:n], y[n:]

    # fit with bartz
    bart = bartz.BART(X_train, y_train, x_test=X_test, seed=key3)

    bad = bart._check_trees()
    bad_count = jnp.count_nonzero(bad)
    assert bad_count == 0

# I should share n, p, and various integers between as many tests as possible to reduce compilation times.
