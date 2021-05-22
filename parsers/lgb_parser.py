import os
import json
import shutil

import numpy as np

from tree import Tree
from tree import TreeEnsemble


def parse_lgb_ensemble(model):
    """
    Parse CatBoost model based on its json representation.
    """
    json_data = _get_json_data_from_lgb_model(model)
    trees = np.array([_parse_lgb_tree(tree_dict) for tree_dict in json_data], dtype=np.dtype(object))

    # multiclass
    if hasattr(model, 'classes_'):
        n_class = model.classes_.shape[0]
        if n_class > 2:
            n_trees = int(trees.shape[0] / n_class)
            trees = trees.reshape((n_trees, n_class))

    scale, bias = 1.0, 0.0

    return trees, scale, bias


# private
def _parse_lgb_tree(tree_dict):
    """
    Data has format:
    {
        ...
        'tree_structure': {
            'split_feature': int,
            'threshold': float,
            'left child': dict
            'right_child': dict,
            ...
        }
    }

    IF 'left_child' or 'right_child' is a leaf, the dict is:
    {
        'leaf_index': int,
        'leaf_value': float,
        'leaf_weight': int,
        'leaf_count': int
    }

    Notes:
        - The structure is given as recursive dicts.

    Traversal:
        - Breadth-first.

    Desired format:
        https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py

    Returns one or a list of Trees (one for each class).
    """

    children_left = []
    children_right = []
    feature = []
    threshold = []
    leaf_vals = []

    node_dict = tree_dict['tree_structure']

    # add root node
    leaf_vals.append(-1)
    feature.append(node_dict['split_feature'])
    threshold.append(node_dict['threshold'])

    node_id = 1
    stack = [(node_dict['left_child'], 1), (node_dict['right_child'], 0)]

    while len(stack) > 0:
        node_dict, is_left = stack.pop(0)

        if node_dict is None:
            if is_left:
                children_left.append(-1)
            else:
                children_right.append(-1)

        else:

            if is_left:
                children_left.append(node_id)
            else:
                children_right.append(node_id)

            if 'split_index' in node_dict:  # split node
                feature.append(node_dict['split_feature'])
                threshold.append(node_dict['threshold'])
                leaf_vals.append(-1)
                stack.append((node_dict['left_child'], 1))
                stack.append((node_dict['right_child'], 0))

            else:  # leaf node
                feature.append(-1)
                threshold.append(-1)
                leaf_vals.append(node_dict['leaf_value'])
                stack.append((None, 1))
                stack.append((None, 0))

            node_id += 1

    result = Tree(children_left, children_right, feature, threshold, leaf_vals)

    return result


def _get_json_data_from_lgb_model(model):
    """
    Parse CatBoost model based on its json representation.
    """
    assert 'LGBM' in str(model)
    json_data = model.booster_.dump_model()['tree_info']  # 1d list of tree dicts
    return json_data
