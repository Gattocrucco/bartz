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
from jaxtyping import Array, DTypeLike, Float32, UInt

from . import grove, jaxext


def print_tree(leaf_tree, var_tree, split_tree, print_all=False):
    tee = '├──'
    corner = '└──'
    join = '│  '
    space = '   '
    down = '┐'
    bottom = '╢'  # '┨' #

    def traverse_tree(index, depth, indent, first_indent, next_indent, unused):
        if index >= len(leaf_tree):
            return

        var = var_tree.at[index].get(mode='fill', fill_value=0)
        split = split_tree.at[index].get(mode='fill', fill_value=0)

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
            node_str = f'{category}({var}, {split}, {leaf_tree[index]})'
        else:
            assert not unused
            if is_leaf:
                node_str = f'{leaf_tree[index]:#.2g}'
            else:
                node_str = f'({var}: {split})'

        if not is_leaf or (print_all and left_child < len(leaf_tree)):
            link = down
        elif not print_all and left_child >= len(leaf_tree):
            link = bottom
        else:
            link = ' '

        max_number = len(leaf_tree) - 1
        ndigits = len(str(max_number))
        number = str(index).rjust(ndigits)

        print(f' {number} {indent}{first_indent}{link}{node_str}')

        indent += next_indent
        unused = unused or is_leaf

        if unused and not print_all:
            return

        traverse_tree(left_child, depth + 1, indent, tee, join, unused)
        traverse_tree(right_child, depth + 1, indent, corner, space, unused)

    traverse_tree(1, 0, '', '', '', False)


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


def forest_points_per_leaf_distr(bart, X):
    distr = jnp.zeros(X.shape[1] + 1, int)
    trees = bart['var_trees'], bart['split_trees']

    def loop(distr, tree):
        return distr + points_per_leaf_distr(*tree, X), None

    distr, _ = lax.scan(loop, distr, trees)
    return distr


def trace_points_per_leaf_distr(bart, X):
    def loop(_, bart):
        return None, forest_points_per_leaf_distr(bart, X)

    _, distr = lax.scan(loop, None, bart)
    return distr


check_functions = []


def check(func):
    """Check the signature and add the function to the list `check_functions`."""
    sig = signature(func)
    assert str(sig) == '(leaf_tree, var_tree, split_tree, max_split)'
    check_functions.append(func)
    return func


@check
def check_types(leaf_tree, var_tree, split_tree, max_split):
    expected_var_dtype = jaxext.minimal_unsigned_dtype(max_split.size - 1)
    expected_split_dtype = max_split.dtype
    return (
        var_tree.dtype == expected_var_dtype
        and split_tree.dtype == expected_split_dtype
    )


@check
def check_sizes(leaf_tree, var_tree, split_tree, max_split):
    return leaf_tree.size == 2 * var_tree.size == 2 * split_tree.size


@check
def check_unused_node(leaf_tree, var_tree, split_tree, max_split):
    return (var_tree[0] == 0) & (split_tree[0] == 0)


@check
def check_leaf_values(leaf_tree, var_tree, split_tree, max_split):
    return jnp.all(jnp.isfinite(leaf_tree))


@check
def check_stray_nodes(leaf_tree, var_tree, split_tree, max_split):
    """Check if there is any node marked-non-leaf with a marked-leaf parent."""
    index = jnp.arange(
        2 * split_tree.size,
        dtype=jaxext.minimal_unsigned_dtype(2 * split_tree.size - 1),
    )
    parent_index = index >> 1
    is_not_leaf = split_tree.at[index].get(mode='fill', fill_value=0) != 0
    parent_is_leaf = split_tree[parent_index] == 0
    stray = is_not_leaf & parent_is_leaf
    stray = stray.at[1].set(False)
    return ~jnp.any(stray)


@check
def check_rule_consistency(leaf_tree, var_tree, split_tree, max_split):
    """Check that decision rules define proper subsets of ancestor rules."""
    if var_tree.size < 4:
        return True
    lower = jnp.full(max_split.size, jnp.iinfo(jnp.int32).min)
    upper = jnp.full(max_split.size, jnp.iinfo(jnp.int32).max)

    def _check_recursive(node, lower, upper):
        var = var_tree[node]
        split = split_tree[node]
        bad = jnp.where(split, (split <= lower[var]) | (split >= upper[var]), False)
        if node < var_tree.size // 2:
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


def check_tree(leaf_tree, var_tree, split_tree, max_split):
    error_type = jaxext.minimal_unsigned_dtype(2 ** len(check_functions) - 1)
    error = error_type(0)
    for i, func in enumerate(check_functions):
        ok = func(leaf_tree, var_tree, split_tree, max_split)
        ok = jnp.bool_(ok)
        bit = (~ok) << i
        error |= bit
    return error


def describe_error(error):
    return [func.__name__ for i, func in enumerate(check_functions) if error & (1 << i)]


check_forest = jax.vmap(check_tree, in_axes=(0, 0, 0, None))


@functools.partial(jax.vmap, in_axes=(0, None))
def check_trace(trace, max_split: UInt[Array, ' p']):
    return check_forest(
        trace['leaf_trees'],
        trace['var_trees'],
        trace['split_trees'],
        max_split,
    )


class Trees(Module):
    leaf_trees: Float32[Array, 'ndpost ntree 2**d']
    var_trees: UInt[Array, 'ndpost ntree 2**(d-1)']
    split_trees: UInt[Array, 'ndpost ntree 2**(d-1)']


@dataclass
class TreesMeta:
    ndpost: int
    ntree: int
    p: int
    max_split: int
    max_heap_index: int
    var_dtype: DTypeLike
    split_dtype: DTypeLike


def get_next_line(s: str, i: int) -> tuple[str, int]:
    """Get the next line from a string and the new index."""
    i_new = s.find('\n', i)
    if i_new == -1:
        return s[i:], len(s)
    return s[i:i_new], i_new + 1


def trees_BART_to_bartz(
    trees: str, min_maxdepth=0, only_meta: bool = False
) -> tuple[Trees | None, TreesMeta]:
    if only_meta:
        meta = TreesMeta(
            ndpost=0,
            ntree=0,
            p=0,
            max_split=0,
            max_heap_index=0,
            var_dtype=numpy.uint32,
            split_dtype=numpy.uint32,
        )
    else:
        _, meta = trees_BART_to_bartz(trees, min_maxdepth, True)

    # parse first line
    line, i_char = get_next_line(trees, 0)
    if only_meta:
        match = re.fullmatch(r'(\d+) (\d+) (\d+)', line)
        if match is None:
            raise ValueError('Malformed header at i_line=0')
        meta.ndpost, meta.ntree, meta.p = map(int, match.groups())

    # output arrays
    if not only_meta:
        heap_size = 2 ** ceil(log2(meta.max_heap_index + 1))
        heap_size = max(heap_size, 2**min_maxdepth)
        leaf_trees = numpy.zeros(
            (meta.ndpost, meta.ntree, heap_size), dtype=numpy.float32
        )
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
            if only_meta:
                match = re.fullmatch(r'(\d+)', line)
                if match is None:
                    raise ValueError(
                        f'Malformed tree header at {i_iter=} {i_tree=} {i_line=}'
                    )
            num_nodes = int(line)

            if not only_meta:
                is_internal = numpy.zeros(heap_size // 2, dtype=bool)

            # cycle over nodes
            for i_node in range(num_nodes):
                # parse node definition
                line, i_char = get_next_line(trees, i_char)
                i_line += 1
                match = re.fullmatch(r'(\d+) (\d+) (\d+) ([-+e\d\.]+)', line)
                if match is None:
                    raise ValueError(
                        f'Malformed node definition at {i_iter=} {i_tree=} {i_node=} {i_line=}'
                    )
                i_heap = int(match.group(1))
                var = int(match.group(2))
                split = int(match.group(3))
                leaf = float(match.group(4))

                if only_meta:
                    # update maximum
                    meta.max_split = max(meta.max_split, split)
                    meta.max_heap_index = max(meta.max_heap_index, i_heap)
                else:
                    # update values
                    leaf_trees[i_iter, i_tree, i_heap] = leaf
                    is_internal[i_heap // 2] = True
                    if i_heap < heap_size // 2:
                        var_trees[i_iter, i_tree, i_heap] = var
                        split_trees[i_iter, i_tree, i_heap] = split + 1

            if not only_meta:
                is_internal[0] = False
                split_trees[i_iter, i_tree, ~is_internal] = 0

    assert i_char <= len(trees)
    if i_char < len(trees):
        raise ValueError(f'Leftover {len(trees) - i_char} characters in string')

    if only_meta:
        # determine minimal integer type for predictor indices
        if meta.p < (1 << 8):
            meta.var_dtype = jnp.uint8
        elif meta.p < (1 << 16):
            meta.var_dtype = jnp.uint16
        else:
            meta.var_dtype = jnp.uint32

        # determine minimal integer type for split indices
        # + 1 because BART is 0-based while bartz 1-based
        if meta.max_split + 1 < (1 << 8):
            meta.split_dtype = jnp.uint8
        elif meta.max_split + 1 < (1 << 16):
            meta.split_dtype = jnp.uint16
        else:
            meta.split_dtype = jnp.uint32

    # convert to jax arrays
    if only_meta:
        out = None
    else:
        out = Trees(
            leaf_trees=jnp.array(leaf_trees),
            var_trees=jnp.array(var_trees),
            split_trees=jnp.array(split_trees),
        )

    return out, meta
