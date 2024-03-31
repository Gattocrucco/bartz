# bartz/src/bartz/grove.py
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

Functions to create and manipulate binary trees.

A tree is represented with arrays as a heap. The root node is at index 1. The children nodes of a node at index :math:`i` are at indices :math:`2i` (left child) and :math:`2i + 1` (right child). The array element at index 0 is unused.

A decision tree is represented by tree arrays: 'leaf', 'var', and 'split'.

The 'leaf' array contains the values in the leaves.

The 'var' array contains the axes along which the decision nodes operate.

The 'split' array contains the decision boundaries. The boundaries are open on the right, i.e., a point belongs to the left child iff x < split. Whether a node is a leaf is indicated by the corresponding 'split' element being 0.

Since the nodes at the bottom can only be leaves and not decision nodes, the 'var' and 'split' arrays have half the length of the 'leaf' array.

"""

import functools
import math

import jax
from jax import numpy as jnp
from jax import lax

from . import jaxext

def make_tree(depth, dtype):
    """
    Make an array to represent a binary tree.

    Parameters
    ----------
    depth : int
        The maximum depth of the tree. Depth 1 means that there is only a root
        node.
    dtype : dtype
        The dtype of the array.

    Returns
    -------
    tree : array
        An array of zeroes with shape (2 ** depth,).
    """
    return jnp.zeros(2 ** depth, dtype)

def tree_depth(tree):
    """
    Return the maximum depth of a tree.

    Parameters
    ----------
    tree : array
        A tree created by `make_tree`. If the array is ND, the tree structure is
        assumed to be along the last axis.

    Returns
    -------
    depth : int
        The maximum depth of the tree.
    """
    return int(round(math.log2(tree.shape[-1])))

def traverse_tree(x, var_tree, split_tree):
    """
    Find the leaf where a point falls into.

    Parameters
    ----------
    x : array (p,)
        The coordinates to evaluate the tree at.
    var_tree : array (2 ** (d - 1),)
        The decision axes of the tree.
    split_tree : array (2 ** (d - 1),)
        The decision boundaries of the tree.

    Returns
    -------
    index : int
        The index of the leaf.
    """

    carry = (
        jnp.zeros((), bool),
        jnp.ones((), jaxext.minimal_unsigned_dtype(2 * var_tree.size - 1)),
    )

    def loop(carry, _):
        leaf_found, index = carry

        split = split_tree[index]
        var = var_tree[index]
        
        leaf_found |= split == 0
        child_index = (index << 1) + (x[var] >= split)
        index = jnp.where(leaf_found, index, child_index)

        return (leaf_found, index), None

    depth = tree_depth(var_tree)
    (_, index), _ = lax.scan(loop, carry, None, depth, unroll=16)
    return index

@functools.partial(jaxext.vmap_nodoc, in_axes=(None, 0, 0))
@functools.partial(jaxext.vmap_nodoc, in_axes=(1, None, None))
def traverse_forest(X, var_trees, split_trees):
    """
    Find the leaves where points fall into.

    Parameters
    ----------
    X : array (p, n)
        The coordinates to evaluate the trees at.
    var_trees : array (m, 2 ** (d - 1))
        The decision axes of the trees.
    split_trees : array (m, 2 ** (d - 1))
        The decision boundaries of the trees.

    Returns
    -------
    indices : array (m, n)
        The indices of the leaves.
    """
    return traverse_tree(X, var_trees, split_trees)

def evaluate_forest(X, leaf_trees, var_trees, split_trees, dtype):
    """
    Evaluate a ensemble of trees at an array of points.

    Parameters
    ----------
    X : array (p, n)
        The coordinates to evaluate the trees at.
    leaf_trees : array (m, 2 ** d)
        The leaf values of the tree or forest. If the input is a forest, the
        first axis is the tree index, and the values are summed.
    var_trees : array (m, 2 ** (d - 1))
        The decision axes of the trees.
    split_trees : array (m, 2 ** (d - 1))
        The decision boundaries of the trees.
    dtype : dtype
        The dtype of the output.

    Returns
    -------
    out : array (n,)
        The sum of the values of the trees at the points in `X`.
    """
    indices = traverse_forest(X, var_trees, split_trees)
    ntree, _ = leaf_trees.shape
    tree_index = jnp.arange(ntree, dtype=jaxext.minimal_unsigned_dtype(ntree - 1))[:, None]
    leaves = leaf_trees[tree_index, indices]
    return jnp.sum(leaves, axis=0, dtype=dtype)
        # this sum suggests to swap the vmaps, but I think it's better for X
        # copying to keep it that way

def is_actual_leaf(split_tree, *, add_bottom_level=False):
    """
    Return a mask indicating the leaf nodes in a tree.

    Parameters
    ----------
    split_tree : int array (2 ** (d - 1),)
        The splitting points of the tree.
    add_bottom_level : bool, default False
        If True, the bottom level of the tree is also considered.

    Returns
    -------
    is_actual_leaf : bool array (2 ** (d - 1) or 2 ** d,)
        The mask indicating the leaf nodes. The length is doubled if
        `add_bottom_level` is True.
    """
    size = split_tree.size
    is_leaf = split_tree == 0
    if add_bottom_level:
        size *= 2
        is_leaf = jnp.concatenate([is_leaf, jnp.ones_like(is_leaf)])
    index = jnp.arange(size, dtype=jaxext.minimal_unsigned_dtype(size - 1))
    parent_index = index >> 1
    parent_nonleaf = split_tree[parent_index].astype(bool)
    parent_nonleaf = parent_nonleaf.at[1].set(True)
    return is_leaf & parent_nonleaf

def is_leaves_parent(split_tree):
    """
    Return a mask indicating the nodes with leaf (and only leaf) children.

    Parameters
    ----------
    split_tree : int array (2 ** (d - 1),)
        The decision boundaries of the tree.

    Returns
    -------
    is_leaves_parent : bool array (2 ** (d - 1),)
        The mask indicating which nodes have leaf children.
    """
    index = jnp.arange(split_tree.size, dtype=jaxext.minimal_unsigned_dtype(2 * split_tree.size - 1))
    left_index = index << 1 # left child
    right_index = left_index + 1 # right child
    left_leaf = split_tree.at[left_index].get(mode='fill', fill_value=0) == 0
    right_leaf = split_tree.at[right_index].get(mode='fill', fill_value=0) == 0
    is_not_leaf = split_tree.astype(bool)
    return is_not_leaf & left_leaf & right_leaf
        # the 0-th item has split == 0, so it's not counted

def tree_depths(tree_length):
    """
    Return the depth of each node in a binary tree.

    Parameters
    ----------
    tree_length : int
        The length of the tree array, i.e., 2 ** d.

    Returns    
    -------
    depth : array (tree_length,)
        The depth of each node. The root node (index 1) has depth 0. The depth
        is the position of the most significant non-zero bit in the index. The
        first element (the unused node) is marked as depth 0.
    """
    depths = []
    depth = 0
    for i in range(tree_length):
        if i == 2 ** depth:
            depth += 1
        depths.append(depth - 1)
    depths[0] = 0
    return jnp.array(depths, jaxext.minimal_unsigned_dtype(max(depths)))
