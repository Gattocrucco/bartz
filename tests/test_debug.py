# bartz/tests/test_debug.py
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

"""Test `bartz.debug`."""

from jax import numpy as jnp

from bartz.debug import format_tree
from bartz.mcmcloop import TreesTrace


def manual_tree(
    leaf: list[list[float]], var: list[list[int]], split: list[list[int]]
) -> TreesTrace:
    """Facilitate the hardcoded definition of tree heaps."""
    assert len(leaf) == len(var) + 1 == len(split) + 1

    def check_powers_of_2(seq: list[list]):
        """Check if the lengths of the lists in `seq` are powers of 2."""
        return all(len(x) == 2**i for i, x in enumerate(seq))

    tree = TreesTrace(
        jnp.concatenate([jnp.zeros(1), *map(jnp.array, leaf)]),
        jnp.concatenate([jnp.zeros(1, int), *map(jnp.array, var)]),
        jnp.concatenate([jnp.zeros(1, int), *map(jnp.array, split)]),
    )
    assert tree.leaf_tree.dtype == jnp.float32
    assert tree.var_tree.dtype == jnp.int32
    assert tree.split_tree.dtype == jnp.int32
    return tree


def test_format_tree():
    """Check the output of `format_tree` on a single example."""
    tree = manual_tree(
        [
            [1.0],
            [2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
        [
            [4],
            [1, 2],
        ],
        [
            [15],
            [0, 3],
        ],
    )
    s = format_tree(tree)
    print(s)
    ref_s = """\
 1 ┐(4: 15)
 2 ├── 2.0
 3 └──┐(2: 3)
 6    ├──╢6.0
 7    └──╢7.0"""
    assert s == ref_s
