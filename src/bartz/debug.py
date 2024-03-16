import functools

import jax
from jax import numpy as jnp

from . import mcmcstep

@functools.partial(jax.vmap, in_axes=(0, None)) # vectorize over trace
def trace_evaluate_trees(bart, X):
    """
    Evaluate all trees, for all samples, at all x. Out axes:
        0: mcmc sample
        1: tree
        2: X
    """
    return evaluate_all_trees_impl(X, bart['leaf_trees'], bart['var_trees'], bart['split_trees'])

@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0)) # vectorize over forest
def evaluate_all_trees_impl(X, leaf_trees, var_trees, split_trees):
    return mcmcstep.evaluate_tree_vmap_x(X, leaf_trees, var_trees, split_trees, jnp.float32)

def print_tree(leaf_tree, var_tree, split_tree, print_all=False):

    tee = '├──'
    corner = '└──'
    join = '│  '
    space = '   '
    down = '┐'

    def traverse_tree(index, depth, indent, first_indent, next_indent, unused):
        if index >= len(leaf_tree):
            return

        is_leaf = split_tree[index] == 0

        if print_all:
            if unused:
                category = 'unused'
            elif is_leaf:
                category = 'leaf'
            else:
                category = 'decision'
            node_str = f'{category}({var_tree[index]}, {split_tree[index]}, {leaf_tree[index]})'
        else:
            assert not unused
            if is_leaf:
                node_str = f'{leaf_tree[index]:#.2g}'
            else:
                node_str = f'{var_tree[index]}: {split_tree[index]}'

        if is_leaf:
            link = ' '
        else:
            link = down
        print(f'{indent}{first_indent}{link}{node_str}')

        left_child = 2 * index
        right_child = 2 * index + 1

        indent += next_indent
        unused = unused or is_leaf
        
        if unused and not print_all:
            return

        traverse_tree(left_child, depth + 1, indent, tee, join, unused)
        traverse_tree(right_child, depth + 1, indent, corner, space, unused)

    traverse_tree(1, 0, '', '', '', False)
