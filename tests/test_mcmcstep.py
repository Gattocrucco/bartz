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

import pytest
from jax import numpy as jnp
from jax import vmap
from jax.random import bernoulli, clone, permutation, randint
from jaxtyping import Array, Bool, Int32, Key
from numpy.testing import assert_array_equal
from scipy import stats

from bartz.jaxext import minimal_unsigned_dtype, split
from bartz.mcmcstep._moves import ancestor_variables, randint_masked
from tests.util import manual_tree


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
            if not jnp.any(mask):  # pragma: no cover, rarely happens
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


class TestAncestorVariables:
    """Test `mcmcstep._moves.ancestor_variables`."""

    @pytest.fixture
    def depth2_tree(self):
        R"""
        Tree with var_tree of size 4 (tree_depth=2, max_num_ancestors=1).

        Structure (heap indices):
              1 (root, var=2)
             / \
            2   3 (vars=0, 1)
           /\   /\
          4 5  6  7 (leaves, in leaf_tree only)

        Note: var_tree indices are 1-3, leaf indices 4-7 are beyond var_tree.
        """
        tree = manual_tree(
            [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[2], [0, 1]], [[5], [3, 4]]
        )
        var_tree = tree.var_tree.astype(jnp.uint8)
        max_split = jnp.full(5, 10, jnp.uint8)
        return var_tree, max_split

    @pytest.fixture
    def depth3_tree(self):
        """
        Tree with var_tree of size 8 (tree_depth=3, max_num_ancestors=2).

        Heap indices 1-7 in var_tree, 8-15 leaves.
        """
        tree = manual_tree(
            [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0] * 8],
            [[3], [2, 1], [0, 4, 5, 6]],
            [[1], [2, 3], [4, 5, 6, 7]],
        )
        var_tree = tree.var_tree.astype(jnp.uint8)
        max_split = jnp.full(10, 10, jnp.uint8)
        return var_tree, max_split

    def test_root_node(self, depth2_tree):
        """Check that root node has no ancestors (all slots filled with p)."""
        var_tree, max_split = depth2_tree

        # Root node (index 1) has no ancestors
        result = ancestor_variables(var_tree, max_split, jnp.int32(1))
        # var_tree size=4 -> tree_depth=2 -> max_num_ancestors=1
        # All slots should be p (sentinel) since root has no ancestors
        assert_array_equal(result, [max_split.size])

    def test_child_of_root(self, depth2_tree):
        """Check that children of root have one ancestor (the root's variable)."""
        var_tree, max_split = depth2_tree

        # Left child of root (index 2): ancestor is root (var=2)
        result = ancestor_variables(var_tree, max_split, jnp.int32(2))
        assert result.shape == (1,)
        assert_array_equal(result, [2])

        # Right child of root (index 3): ancestor is root (var=2)
        result = ancestor_variables(var_tree, max_split, jnp.int32(3))
        assert_array_equal(result, [2])

    def test_deep_node(self, depth3_tree):
        """Check ancestors for nodes at depth 3."""
        var_tree, max_split = depth3_tree

        # Node 4: parent is 2 (var=2), grandparent is 1 (var=3)
        result = ancestor_variables(var_tree, max_split, jnp.int32(4))
        assert result.shape == (2,)
        assert_array_equal(result, [3, 2])

        # Node 5: parent is 2 (var=2), grandparent is 1 (var=3)
        result = ancestor_variables(var_tree, max_split, jnp.int32(5))
        assert_array_equal(result, [3, 2])

        # Node 6: parent is 3 (var=1), grandparent is 1 (var=3)
        result = ancestor_variables(var_tree, max_split, jnp.int32(6))
        assert_array_equal(result, [3, 1])

        # Node 7: parent is 3 (var=1), grandparent is 1 (var=3)
        result = ancestor_variables(var_tree, max_split, jnp.int32(7))
        assert_array_equal(result, [3, 1])

    def test_intermediate_node(self, depth3_tree):
        """Check ancestors for an intermediate (non-leaf) node."""
        var_tree, max_split = depth3_tree

        # Node 2: parent is 1 (var=3), one slot filled, one sentinel
        result = ancestor_variables(var_tree, max_split, jnp.int32(2))
        assert_array_equal(result, [max_split.size, 3])

        # Node 3: parent is 1 (var=3), one slot filled, one sentinel
        result = ancestor_variables(var_tree, max_split, jnp.int32(3))
        assert_array_equal(result, [max_split.size, 3])

    def test_single_variable(self):
        """Check with only one variable (p=1)."""
        tree = manual_tree(
            [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0], [0, 0]], [[5], [3, 4]]
        )
        var_tree = tree.var_tree.astype(jnp.uint8)
        max_split = jnp.ones(1, minimal_unsigned_dtype(10))

        # Node 2: ancestor is root (var=0)
        result = ancestor_variables(var_tree, max_split, jnp.int32(2))
        assert_array_equal(result, [0])

        # Root has no ancestors
        result = ancestor_variables(var_tree, max_split, jnp.int32(1))
        assert_array_equal(result, [max_split.size])

    def test_type_edge(self, depth3_tree):
        """Check that types are handled correctly when using uint8 and uint16 together."""
        var_tree, max_split = depth3_tree
        var_tree = var_tree.astype(jnp.uint8)
        max_split = jnp.full(256, 10, jnp.uint8)
        assert minimal_unsigned_dtype(max_split.size) == jnp.uint16

        # Node 2: parent is 1 (var=3), one slot filled, one sentinel
        result = ancestor_variables(var_tree, max_split, jnp.int32(2))
        assert_array_equal(result, [max_split.size, 3])

        # Node 3: parent is 1 (var=3), one slot filled, one sentinel
        result = ancestor_variables(var_tree, max_split, jnp.int32(3))
        assert_array_equal(result, [max_split.size, 3])
