# bartz/tests/test_mcmcloop.py
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

"""Test `bartz.mcmcloop`."""

from functools import partial

from equinox import filter_jit
from jax import numpy as jnp
from jax import vmap
from jax.tree import map_with_path
from jax.tree_util import tree_map
from jaxtyping import Array, Float32, UInt8
from numpy.testing import assert_array_equal

from bartz import mcmcloop, mcmcstep


def gen_data(
    p: int, n: int
) -> tuple[UInt8[Array, 'p n'], Float32[Array, ' n'], UInt8[Array, ' p']]:
    """Generate pretty nonsensical data."""
    X = jnp.arange(p * n, dtype=jnp.uint8).reshape(p, n)
    X = vmap(jnp.roll)(X, jnp.arange(p))
    max_split = jnp.full(p, 255, jnp.uint8)
    y = jnp.cos(jnp.linspace(0, 2 * jnp.pi / 32 * n, n))
    return X, y, max_split


def make_p_nonterminal(maxdepth: int) -> Float32[Array, ' {maxdepth}-1']:
    """Prepare the p_nonterminal argument to `mcmcstep.init`."""
    depth = jnp.arange(maxdepth - 1)
    base = 0.95
    power = 2
    return base / (1 + depth).astype(float) ** power


@filter_jit
def init(p: int, n: int, ntree: int, **kwargs):
    """Simplified version of `bartz.mcmcstep.init` with data pre-filled."""
    X, y, max_split = gen_data(p, n)
    return mcmcstep.init(
        X=X,
        y=y,
        max_split=max_split,
        num_trees=ntree,
        p_nonterminal=make_p_nonterminal(6),
        sigma_mu2=1.0,
        sigma2_alpha=1,
        sigma2_beta=1,
        min_points_per_decision_node=10,
        filter_splitless_vars=False,
        **kwargs,
    )


class TestRunMcmc:
    """Test `mcmcloop.run_mcmc`."""

    def test_final_state_overflow(self, keys):
        """Check that the final state is the one in the trace even if there's overflow."""
        initial_state = init(10, 100, 20)
        final_state, _, main_trace = mcmcloop.run_mcmc(
            keys.pop(), initial_state, 10, inner_loop_length=9
        )

        assert_array_equal(final_state.forest.leaf_tree, main_trace.leaf_tree[-1])
        assert_array_equal(final_state.forest.var_tree, main_trace.var_tree[-1])
        assert_array_equal(final_state.forest.split_tree, main_trace.split_tree[-1])
        assert_array_equal(final_state.sigma2, main_trace.sigma2[-1])

    def test_zero_iterations(self, keys):
        """Check there's no error if the loop does not run."""
        initial_state = init(10, 100, 20)
        final_state, burnin_trace, main_trace = mcmcloop.run_mcmc(
            keys.pop(), initial_state, 0, n_burn=0
        )

        tree_map(partial(assert_array_equal, strict=True), initial_state, final_state)

        def assert_empty_trace(path, x):  # noqa: ARG001, for debugging
            assert x.shape[0] == 0

        map_with_path(assert_empty_trace, burnin_trace)
        map_with_path(assert_empty_trace, main_trace)
