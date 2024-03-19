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

A decision tree is represented by tree arrays: 'leaf', 'var', and 'split'. The 'leaf' array contains the values in the leaves. The 'var' array contains the axes along which the decision nodes operate. The 'split' array contains the decision boundaries.

Whether a node is a leaf is indicated by the corresponding 'split' element being 0.

Since the nodes at the bottom can only be leaves and not decision nodes, the 'var' and 'split' arrays have half the length of the 'leaf' array.

The unused array element at index 0 is always fixed to 0 by convention.

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

    Notes
    -----
    The tree is represented as a heap, with the root node at index 1, and the
    children of the node at index i at indices 2 * i and 2 * i + 1. The element
    at index 0 is unused.
    """
    return jnp.zeros(2 ** depth, dtype)

def tree_depth(tree):
    """
    Return the maximum depth of a binary tree created by `make_tree`.

    Parameters
    ----------
    tree : array
        A binary tree created by `make_tree`. If the array is ND, the tree
        structure is assumed to be along the last axis.

    Returns
    -------
    depth : int
        The maximum depth of the tree.
    """
    return int(round(math.log2(tree.shape[-1])))

def evaluate_tree(X, leaf_trees, var_trees, split_trees, out_dtype):
    """
    Evaluate a decision tree or forest.

    Parameters
    ----------
    X : array (p,)
        The coordinates to evaluate the tree at.
    leaf_trees : array (n,) or (m, n)
        The leaf values of the tree or forest. If the input is a forest, the
        first axis is the tree index, and the values are summed.
    var_trees : array (n,) or (m, n)
        The variable indices of the tree or forest. Each index is in [0, p) and
        indicates which value of `X` to consider.
    split_trees : array (n,) or (m, n)
        The split values of the tree or forest. Leaf nodes are indicated by the
        condition `split == 0`. If non-zero, the node has children, and its left
        children is assigned points which satisfy `x < split`.
    out_dtype : dtype
        The dtype of the output.

    Returns
    -------
    out : scalar
        The value of the tree or forest at the given point.
    """

    is_forest = leaf_trees.ndim == 2
    if is_forest:
        m, _ = leaf_trees.shape
        forest_shape = m,
        tree_index = jnp.arange(m, dtype=minimal_unsigned_dtype(m - 1)),
    else:
        forest_shape = ()
        tree_index = ()

    carry = (
        jnp.zeros(forest_shape, bool),
        jnp.zeros((), out_dtype),
        jnp.ones(forest_shape, minimal_unsigned_dtype(leaf_trees.shape[-1] - 1))
    )

    def loop(carry, _):
        leaf_found, out, node_index = carry

        is_leaf = split_trees.at[tree_index + (node_index,)].get(mode='fill', fill_value=0) == 0
        leaf_value = leaf_trees[tree_index + (node_index,)]
        if is_forest:
            leaf_sum = jnp.sum(leaf_value, where=is_leaf) # TODO set dtype to large float
                # alternative: dot(is_leaf, leaf_value):
                # - maybe faster
                # - maybe less accurate
                # - fucked by nans
        else:
            leaf_sum = jnp.where(is_leaf, leaf_value, 0)
        out += leaf_sum
        leaf_found |= is_leaf
        
        split = split_trees.at[tree_index + (node_index,)].get(mode='fill', fill_value=0)
        var = var_trees.at[tree_index + (node_index,)].get(mode='fill', fill_value=0)
        x = X[var]
        
        node_index <<= 1
        node_index += x >= split
        node_index = jnp.where(leaf_found, 0, node_index)

        carry = leaf_found, out, node_index
        return carry, _

    depth = tree_depth(leaf_trees)
    (_, out, _), _ = lax.scan(loop, carry, None, depth)
    return out

def minimal_unsigned_dtype(max_value):
    """
    Return the smallest unsigned integer dtype that can represent a given
    maximum value.
    """
    if max_value < 2 ** 8:
        return jnp.uint8
    if max_value < 2 ** 16:
        return jnp.uint16
    if max_value < 2 ** 32:
        return jnp.uint32
    return jnp.uint64

@functools.partial(jaxext.vmap_nodoc, in_axes=(1, None, None, None, None), out_axes=0)
def evaluate_tree_vmap_x(X, leaf_trees, var_trees, split_trees, out_dtype):
    """
    Evaluate a decision tree or forest over multiple points.

    Parameters
    ----------
    X : array (p, n)
        The points to evaluate the tree at.
    leaf_trees : array (n,) or (m, n)
        The leaf values of the tree or forest. If the input is a forest, the
        first axis is the tree index, and the values are summed.
    var_trees : array (n,) or (m, n)
        The variable indices of the tree or forest. Each index is in [0, p) and
        indicates which value of `X` to consider.
    split_trees : array (n,) or (m, n)
        The split values of the tree or forest. Leaf nodes are indicated by the
        condition `split == 0`. If non-zero, the node has children, and its left
        children is assigned points which satisfy `x < split`.
    out_dtype : dtype
        The dtype of the output.

    Returns
    -------
    out : (n,)
        The value of the tree or forest at each point.
    """
    return evaluate_tree(X, leaf_trees, var_trees, split_trees, out_dtype)

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
    index = jnp.arange(size, dtype=minimal_unsigned_dtype(size - 1))
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
        The splitting points of the tree.

    Returns
    -------
    is_leaves_parent : bool array (2 ** (d - 1),)
        The mask indicating which nodes have leaf children.
    """
    index = jnp.arange(split_tree.size, dtype=minimal_unsigned_dtype(2 * split_tree.size - 1))
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
    return jnp.array(depths, minimal_unsigned_dtype(max(depths)))

def index_depth(index, tree_length):
    """
    Return the depth of a node in a binary tree.

    Parameters
    ----------
    index : int
        The index of the node.
    tree_length : int
        The length of the tree array, i.e., 2 ** d.

    Returns
    -------
    depth : int
        The depth of the node. The root node (index 1) has depth 0. The depth is
        the position of the most significant non-zero bit in the index. If
        ``index == 0``, return -1.
    """
    depths = tree_depths(tree_length)
    return depths[index]
