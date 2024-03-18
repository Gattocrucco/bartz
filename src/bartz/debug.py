import functools

import jax
from jax import numpy as jnp
from jax import lax

from . import grove
from . import mcmcstep

def trace_evaluate_trees(bart, X):
    """
    Evaluate all trees, for all samples, at all x. Out axes:
        0: mcmc sample
        1: tree
        2: X
    """
    def loop(_, bart):
        return None, evaluate_all_trees(X, bart['leaf_trees'], bart['var_trees'], bart['split_trees'])
    _, y = lax.scan(loop, None, bart)
    return y

@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0)) # vectorize over forest
def evaluate_all_trees(X, leaf_trees, var_trees, split_trees):
    return grove.evaluate_tree_vmap_x(X, leaf_trees, var_trees, split_trees, jnp.float32)

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

        is_leaf = split_tree[index] == 0
        left_child = 2 * index
        right_child = 2 * index + 1

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
                node_str = f'({var_tree[index]}: {split_tree[index]})'

        if not is_leaf or (print_all and left_child < len(leaf_tree)):
            link = down
        elif not print_all and left_child >= len(leaf_tree):
            link = bottom
        else:
            link = ' '
        print(f'{indent}{first_indent}{link}{node_str}')

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
    dummy = jnp.ones(X.shape[1], jnp.uint8)
    _, count_tree = mcmcstep.agg_values(X, var_tree, split_tree, dummy, dummy.dtype)
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
