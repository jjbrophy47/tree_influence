import os
import json
import shutil

import numpy as np

from .tree import Tree


def parse_cb_ensemble(model):
    """
    Parse CatBoost model based on its json representation.
    """
    json_data = _get_json_data_from_cb_model(model)
    trees = np.array([_parse_cb_tree(tree_dict) for tree_dict in json_data], dtype=np.dtype(object))
    _, bias = model.get_scale_and_bias()
    return trees, bias


# private
def _parse_cb_tree(tree_dict):
    """
    Data has format:
    {
        'splits': [{'feature_idx': int}, ...]
        'leaf_values': [float, float, ...]
    }

    IF multiclass, then 'leaf_values': [[float, float, ...], [float, ...]]

    Notes:
        - No. leaves = 2 ^ no. splits.
        - There is only one split condition PER LEVEL in CB trees.
        - 'split' list is given bottom up (need to reverse splits list).

    Returns one tree if binary class., otherwise returns a list of trees,
    one for each class.
    """
    _validate_data(tree_dict)

    children_left = []
    children_right = []
    feature = []
    threshold = []
    leaf_vals = []

    n_class = 2
    if isinstance(tree_dict['leaf_values'][0], list):
        n_class = len(tree_dict['leaf_values'][0])
        leaf_vals = [[] for j in range(n_class)]

    node_id = 0
    for depth, split_dict in enumerate(reversed(tree_dict['splits'])):

        for i in range(2 ** depth):
            feature.append(split_dict['feature_idx'])
            threshold.append(split_dict['border'])

            if n_class > 2:
                for j in range(n_class):
                    leaf_vals[j].append(-1)
            else:
                leaf_vals.append(-1)  # arbitrary

            if depth > 0:
                if i % 2 == 0:
                    children_left.append(node_id)
                else:
                    children_right.append(node_id)

            node_id += 1

    # leaf nodes
    for i in range(2 ** (depth + 1)):
        feature.append(-1)  # arbitrary
        threshold.append(-1)  # arbitrary

        if n_class > 2:
            for j in range(n_class):
                leaf_vals[j].append(tree_dict['leaf_values'][i][j])
        else:
            leaf_vals.append(tree_dict['leaf_values'][i])

        if i % 2 == 0:
            children_left.append(node_id)  # leaf
        else:
            children_right.append(node_id)  # leaf

        node_id += 1

    # fill in rest of nodes
    for i in range(2 ** (depth + 1)):
        children_left.append(-1)
        children_right.append(-1)

    # leaf_vals may be a list of lists, go through each one and make a tree for each one
    if n_class > 2:
        result = [Tree(children_left, children_right, feature, threshold, leaf_vals[j]) for j in range(n_class)]
    else:
        result = Tree(children_left, children_right, feature, threshold, leaf_vals)

    return result


def _get_json_data_from_cb_model(model):
    """
    Parse CatBoost model based on its json representation.
    """
    assert 'CatBoost' in str(model)
    here = os.path.abspath(os.path.dirname(__file__))

    temp_dir = os.path.join(here, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    temp_model_bin_fp = os.path.join(temp_dir, 'model.bin')
    temp_model_json_fp = os.path.join(temp_dir, 'model.json')

    model.save_model(temp_model_bin_fp)
    command = f'{here}/export_catboost {temp_model_bin_fp} > {temp_model_json_fp}'
    os.system(command)

    with open(temp_model_json_fp) as f:
        json_data = json.load(f)

    shutil.rmtree(temp_dir)

    return json_data


def _validate_data(data_json):
    """
    Checks to make sure JSON data is valid.
    """
    for split in data_json['splits']:
        assert isinstance(split['feature_idx'], int)
        assert isinstance(split['border'], (int, float))

    for value in data_json['leaf_values']:
        assert isinstance(value, (int, float, list, tuple))

    num_splits = len(data_json['splits'])
    num_values = len(data_json['leaf_values'])

    assert num_values == 2 ** num_splits
