# bartz/src/bartz/debug.py
#
# Copyright (c) 2024-2025, Giacomo Petrillo
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
import re
from dataclasses import dataclass
from inspect import signature
from math import ceil, log2

import jax
import numpy
from equinox import Module
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Float32, UInt

from bartz.mcmcloop import Trace, TreesTrace

from . import grove, jaxext


def format_tree(tree: grove.TreeHeaps, *, print_all=False) -> str:
    tee = '├──'
    corner = '└──'
    join = '│  '
    space = '   '
    down = '┐'
    bottom = '╢'  # '┨' #

    def traverse_tree(lines, index, depth, indent, first_indent, next_indent, unused):
        if index >= len(tree.leaf_tree):
            return

        var = tree.var_tree.at[index].get(mode='fill', fill_value=0)
        split = tree.split_tree.at[index].get(mode='fill', fill_value=0)

        is_leaf = split == 0
        left_child = 2 * index
        right_child = 2 * index + 1

        if print_all:
            if unused:
                category = 'unused'
            elif is_leaf:
                category = 'leaf'
            else:
                category = 'decision'
            node_str = f'{category}({var}, {split}, {tree.leaf_tree[index]})'
        else:
            assert not unused
            if is_leaf:
                node_str = f'{tree.leaf_tree[index]:#.2g}'
            else:
                node_str = f'({var}: {split})'

        if not is_leaf or (print_all and left_child < len(tree.leaf_tree)):
            link = down
        elif not print_all and left_child >= len(tree.leaf_tree):
            link = bottom
        else:
            link = ' '

        max_number = len(tree.leaf_tree) - 1
        ndigits = len(str(max_number))
        number = str(index).rjust(ndigits)

        lines.append(f' {number} {indent}{first_indent}{link}{node_str}')

        indent += next_indent
        unused = unused or is_leaf

        if unused and not print_all:
            return

        traverse_tree(lines, left_child, depth + 1, indent, tee, join, unused)
        traverse_tree(lines, right_child, depth + 1, indent, corner, space, unused)

    lines = []
    traverse_tree(lines, 1, 0, '', '', '', False)
    return '\n'.join(lines)


def tree_actual_depth(split_tree):
    is_leaf = grove.is_actual_leaf(split_tree, add_bottom_level=True)
    depth = grove.tree_depths(is_leaf.size)
    depth = jnp.where(is_leaf, depth, 0)
    return jnp.max(depth)


def forest_depth_distr(split_trees):
    depth = grove.tree_depth(split_trees) + 1
    depths = jax.vmap(tree_actual_depth)(split_trees)
    return jnp.bincount(depths, length=depth)


def trace_depth_distr(split_trees_trace):
    return jax.vmap(forest_depth_distr)(split_trees_trace)


def points_per_leaf_distr(var_tree, split_tree, X):
    traverse_tree = jax.vmap(grove.traverse_tree, in_axes=(1, None, None))
    indices = traverse_tree(X, var_tree, split_tree)
    count_tree = jnp.zeros(
        2 * split_tree.size, dtype=jaxext.minimal_unsigned_dtype(indices.size)
    )
    count_tree = count_tree.at[indices].add(1)
    is_leaf = grove.is_actual_leaf(split_tree, add_bottom_level=True).view(jnp.uint8)
    return jnp.bincount(count_tree, is_leaf, length=X.shape[1] + 1)


def forest_points_per_leaf_distr(trees: grove.TreeHeaps, X):
    distr = jnp.zeros(X.shape[1] + 1, int)

    def loop(distr, heaps: tuple[Array, Array]):
        return distr + points_per_leaf_distr(*heaps, X), None

    distr, _ = lax.scan(loop, distr, (trees.var_tree, trees.split_tree))
    return distr


def trace_points_per_leaf_distr(trace: grove.TreeHeaps, X):
    def loop(_, trace):
        return None, forest_points_per_leaf_distr(trace, X)

    _, distr = lax.scan(loop, None, trace)
    return distr


check_functions = []


BoolLike = bool | Bool[Array, '']


def check(func):
    """Check the signature and add the function to the list `check_functions`."""
    sig = signature(func)
    assert str(sig) == f'(tree: bartz.grove.TreeHeaps, max_split) -> {BoolLike}', str(
        sig
    )
    check_functions.append(func)
    return func


@check
def check_types(tree: grove.TreeHeaps, max_split) -> BoolLike:
    expected_var_dtype = jaxext.minimal_unsigned_dtype(max_split.size - 1)
    expected_split_dtype = max_split.dtype
    return (
        tree.var_tree.dtype == expected_var_dtype
        and tree.split_tree.dtype == expected_split_dtype
    )


@check
def check_sizes(tree: grove.TreeHeaps, max_split) -> BoolLike:  # noqa: ARG001
    return tree.leaf_tree.size == 2 * tree.var_tree.size == 2 * tree.split_tree.size


@check
def check_unused_node(tree: grove.TreeHeaps, max_split) -> BoolLike:  # noqa: ARG001
    return (tree.var_tree[0] == 0) & (tree.split_tree[0] == 0)


@check
def check_leaf_values(tree: grove.TreeHeaps, max_split) -> BoolLike:  # noqa: ARG001
    return jnp.all(jnp.isfinite(tree.leaf_tree))


@check
def check_stray_nodes(tree: grove.TreeHeaps, max_split) -> BoolLike:  # noqa: ARG001
    """Check if there is any node marked-non-leaf with a marked-leaf parent."""
    index = jnp.arange(
        2 * tree.split_tree.size,
        dtype=jaxext.minimal_unsigned_dtype(2 * tree.split_tree.size - 1),
    )
    parent_index = index >> 1
    is_not_leaf = tree.split_tree.at[index].get(mode='fill', fill_value=0) != 0
    parent_is_leaf = tree.split_tree[parent_index] == 0
    stray = is_not_leaf & parent_is_leaf
    stray = stray.at[1].set(False)
    return ~jnp.any(stray)


@check
def check_rule_consistency(tree: grove.TreeHeaps, max_split) -> BoolLike:
    """Check that decision rules define proper subsets of ancestor rules."""
    if tree.var_tree.size < 4:
        return True
    lower = jnp.full(max_split.size, jnp.iinfo(jnp.int32).min)
    upper = jnp.full(max_split.size, jnp.iinfo(jnp.int32).max)

    def _check_recursive(node, lower, upper):
        var = tree.var_tree[node]
        split = tree.split_tree[node]
        bad = jnp.where(split, (split <= lower[var]) | (split >= upper[var]), False)
        if node < tree.var_tree.size // 2:
            bad |= _check_recursive(
                2 * node,
                lower,
                upper.at[jnp.where(split, var, max_split.size)].set(split),
            )
            bad |= _check_recursive(
                2 * node + 1,
                lower.at[jnp.where(split, var, max_split.size)].set(split),
                upper,
            )
        return bad

    return ~_check_recursive(1, lower, upper)


def check_tree(tree: grove.TreeHeaps, max_split) -> Bool[Array, '']:
    error_type = jaxext.minimal_unsigned_dtype(2 ** len(check_functions) - 1)
    error = error_type(0)
    for i, func in enumerate(check_functions):
        ok = func(tree, max_split)
        ok = jnp.bool_(ok)
        bit = (~ok) << i
        error |= bit
    return error


def describe_error(error):
    return [func.__name__ for i, func in enumerate(check_functions) if error & (1 << i)]


check_forest = jax.vmap(check_tree, in_axes=(0, None))


@functools.partial(jax.vmap, in_axes=(0, None))
def check_trace(trace: Trace, max_split: UInt[Array, ' p']):
    trees = TreesTrace.from_dataclass(trace)
    return check_forest(trees, max_split)


def get_next_line(s: str, i: int) -> tuple[str, int]:
    """Get the next line from a string and the new index."""
    i_new = s.find('\n', i)
    if i_new == -1:
        return s[i:], len(s)
    return s[i:i_new], i_new + 1


@dataclass
class BARTTraceMeta:
    ndpost: int
    ntree: int
    p: int
    max_split: int
    max_heap_index: int
    var_dtype: DTypeLike
    split_dtype: DTypeLike


def scan_BART_trees(trees: str) -> BARTTraceMeta:
    meta = BARTTraceMeta(
        ndpost=0,
        ntree=0,
        p=0,
        max_split=0,
        max_heap_index=0,
        var_dtype=numpy.uint32,
        split_dtype=numpy.uint32,
    )

    # parse first line
    line, i_char = get_next_line(trees, 0)
    match = re.fullmatch(r'(\d+) (\d+) (\d+)', line)
    if match is None:
        msg = 'Malformed header at i_line=0'
        raise ValueError(msg)
    meta.ndpost, meta.ntree, meta.p = map(int, match.groups())

    # cycle over iterations and trees
    i_line = 1
    for i_iter in range(meta.ndpost):
        for i_tree in range(meta.ntree):
            # parse first line of tree definition
            line, i_char = get_next_line(trees, i_char)
            i_line += 1
            match = re.fullmatch(r'(\d+)', line)
            if match is None:
                msg = f'Malformed tree header at {i_iter=} {i_tree=} {i_line=}'
                raise ValueError(msg)
            num_nodes = int(line)

            # cycle over nodes
            for i_node in range(num_nodes):
                # parse node definition
                line, i_char = get_next_line(trees, i_char)
                i_line += 1
                match = re.fullmatch(
                    r'(\d+) (\d+) (\d+) (-?\d+(\.\d+)?(e(\+|-|)\d+)?)', line
                )
                if match is None:
                    msg = f'Malformed node definition at {i_iter=} {i_tree=} {i_node=} {i_line=}'
                    raise ValueError(msg)
                i_heap = int(match.group(1))
                split = int(match.group(3))

                # update maxima
                meta.max_split = max(meta.max_split, split)
                meta.max_heap_index = max(meta.max_heap_index, i_heap)

    assert i_char <= len(trees)
    if i_char < len(trees):
        msg = f'Leftover {len(trees) - i_char} characters in string'
        raise ValueError(msg)

    # determine minimal integer types
    meta.var_dtype = jaxext.minimal_unsigned_dtype(meta.p - 1)
    # + 1 because BART is 0-based while bartz 1-based
    meta.split_dtype = jaxext.minimal_unsigned_dtype(meta.max_split + 1)

    return meta


class BARTTrace(Module):
    leaf_tree: Float32[Array, 'ndpost ntree 2**d']
    var_tree: UInt[Array, 'ndpost ntree 2**(d-1)']
    split_tree: UInt[Array, 'ndpost ntree 2**(d-1)']
    offset: Float32[Array, ' ndpost']


def trees_BART_to_bartz(
    trees: str,
    *,
    min_maxdepth: int = 0,
    offset: float | Float[Array, ''] | None = None,
) -> tuple[BARTTrace, BARTTraceMeta]:
    """Convert trees from the R BART format to bartz format.

    Parameters
    ----------
    trees
        The string representation of a trace of trees of the R BART package.
        Can be accessed from ``mc_gbart(...).treedraws['trees']``.
    min_maxdepth
        The maximum tree depth of the output will be set to the maximum
        observed depth in the input trees. Use this parameter to require at
        least this maximum depth in the output format.
    offset
        The trace returned by `run_mcmc` contains an offset to be summed to the
        sum of trees. To match that behavior, this function returns an offset
        as well, zero by default. Set with this parameter otherwise.

    Returns
    -------
    trace : BARTTrace
        A representation of the trees compatible with the trace returned by
        `run_mcmc`.
    meta : BARTTraceMeta
        The metadata of the trace, containing the number of iterations,
        trees, and the maximum split value.
    """
    # scan all the string checking for errors and determining sizes
    meta = scan_BART_trees(trees)

    # skip first line
    line, i_char = get_next_line(trees, 0)

    heap_size = 2 ** ceil(log2(meta.max_heap_index + 1))
    heap_size = max(heap_size, 2**min_maxdepth)
    leaf_trees = numpy.zeros((meta.ndpost, meta.ntree, heap_size), dtype=numpy.float32)
    var_trees = numpy.zeros(
        (meta.ndpost, meta.ntree, heap_size // 2), dtype=meta.var_dtype
    )
    split_trees = numpy.zeros(
        (meta.ndpost, meta.ntree, heap_size // 2), dtype=meta.split_dtype
    )

    # cycle over iterations and trees
    i_line = 1
    for i_iter in range(meta.ndpost):
        for i_tree in range(meta.ntree):
            # parse first line of tree definition
            line, i_char = get_next_line(trees, i_char)
            i_line += 1
            num_nodes = int(line)

            is_internal = numpy.zeros(heap_size // 2, dtype=bool)

            # cycle over nodes
            for _ in range(num_nodes):
                # parse node definition
                line, i_char = get_next_line(trees, i_char)
                i_line += 1
                values = line.split()
                i_heap = int(values[0])
                var = int(values[1])
                split = int(values[2])
                leaf = float(values[3])

                # update values
                leaf_trees[i_iter, i_tree, i_heap] = leaf
                is_internal[i_heap // 2] = True
                if i_heap < heap_size // 2:
                    var_trees[i_iter, i_tree, i_heap] = var
                    split_trees[i_iter, i_tree, i_heap] = split + 1

            is_internal[0] = False
            split_trees[i_iter, i_tree, ~is_internal] = 0

    return BARTTrace(
        leaf_tree=jnp.array(leaf_trees),
        var_tree=jnp.array(var_trees),
        split_tree=jnp.array(split_trees),
        offset=jnp.zeros(meta.ndpost)
        if offset is None
        else jnp.repeat(offset, meta.ndpost),
    ), meta
