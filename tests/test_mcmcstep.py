# bartz/tests/test_mcmcstep.py
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

"""Test `bartz.mcmcstep`."""

from jax import numpy as jnp
from jax import vmap
from jax.random import bernoulli, clone, permutation, randint
from jaxtyping import Array, Bool, Int32, Key
from scipy import stats

from bartz.jaxext import split
from bartz.mcmcstep import randint_masked


def vmap_randint_masked(
    key: Key[Array, ''], mask: Bool[Array, ' n'], size: int
) -> Int32[Array, '* n']:
    """Vectorized version of `randint_masked`."""
    vrm = vmap(randint_masked, in_axes=(0, None))
    keys = split(key, 1)
    return vrm(keys.pop(size), mask)


class TestRandintMasked:
    """Test `mcmcstep.randint_masked`."""

    def test_all_false(self, keys):
        """Check what happens when no value is allowed."""
        for size in range(1, 10):
            u = randint_masked(keys.pop(), jnp.zeros(size, bool))
            assert u == size

    def test_all_true(self, keys):
        """Check it's equivalent to `randint` when all values are allowed."""
        key = keys.pop()
        size = 10_000
        u1 = randint_masked(key, jnp.ones(size, bool))
        u2 = randint(clone(key), (), 0, size)
        assert u1 == u2

    def test_no_disallowed_values(self, keys):
        """Check disallowed values are never selected."""
        key = keys.pop()
        for _ in range(100):
            keys = split(key, 3)
            mask = bernoulli(keys.pop(), 0.5, (10,))
            if not jnp.any(mask):
                continue
            u = randint_masked(keys.pop(), mask)
            assert 0 <= u < mask.size
            assert mask[u]
            key = keys.pop()

    def test_correct_distribution(self, keys):
        """Check the distribution of values is uniform."""
        # create mask
        num_allowed = 10
        mask = jnp.zeros(2 * num_allowed, bool)
        mask = mask.at[:num_allowed].set(True)
        indices = jnp.arange(mask.size)
        indices = permutation(keys.pop(), indices)
        mask = mask[indices]

        # sample values
        n = 10_000
        u: Int32[Array, '{n}'] = vmap_randint_masked(keys.pop(), mask, n)
        u = indices[u]
        assert jnp.all(u < num_allowed)

        # check that the distribution is uniform
        # likelihood ratio test for multinomial with free p vs. constant p
        k = jnp.bincount(u, length=num_allowed)
        llr = jnp.sum(jnp.where(k, k * jnp.log(k / n * num_allowed), 0))
        lamda = 2 * llr
        pvalue = stats.chi2.sf(lamda, num_allowed - 1)
        assert pvalue > 0.1
