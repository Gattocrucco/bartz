import functools

import jax
from jax import numpy as jnp
from jax import lax

from . import grove
from . import mcmcstep
from . import jaxext

def print_tree(leaf_tree, var_tree, split_tree, print_all=False):

    tee = '├──'
    corner = '└──'
    join = '│  '
    space = '   '
    down = '┐'
    bottom = '╢' # '┨' # 

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
    count_tree = jnp.zeros(2 * split_tree.size, dtype=jaxext.minimal_unsigned_dtype(indices.size))
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

def check_types(leaf_tree, var_tree, split_tree, max_split):
    expected_var_dtype = jaxext.minimal_unsigned_dtype(max_split.size - 1)
    expected_split_dtype = max_split.dtype
    return var_tree.dtype == expected_var_dtype and split_tree.dtype == expected_split_dtype

def check_sizes(leaf_tree, var_tree, split_tree, max_split):
    return leaf_tree.size == 2 * var_tree.size == 2 * split_tree.size

def check_unused_node(leaf_tree, var_tree, split_tree, max_split):
    return (var_tree[0] == 0) & (split_tree[0] == 0)

def check_leaf_values(leaf_tree, var_tree, split_tree, max_split):
    return jnp.all(jnp.isfinite(leaf_tree))

def check_stray_nodes(leaf_tree, var_tree, split_tree, max_split):
    index = jnp.arange(2 * split_tree.size, dtype=jaxext.minimal_unsigned_dtype(2 * split_tree.size - 1))
    parent_index = index >> 1
    is_not_leaf = split_tree.at[index].get(mode='fill', fill_value=0) != 0
    parent_is_leaf = split_tree[parent_index] == 0
    stray = is_not_leaf & parent_is_leaf
    stray = stray.at[1].set(False)
    return ~jnp.any(stray)

check_functions = [
    check_types,
    check_sizes,
    check_unused_node,
    check_leaf_values,
    check_stray_nodes,
]

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
    return [
        func.__name__
        for i, func in enumerate(check_functions)
        if error & (1 << i)
    ]

check_forest = jax.vmap(check_tree, in_axes=(0, 0, 0, None))

@functools.partial(jax.vmap, in_axes=(0, None))
def check_trace(trace, state):
    return check_forest(trace['leaf_trees'], trace['var_trees'], trace['split_trees'], state['max_split'])
