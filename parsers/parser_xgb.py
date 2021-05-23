import numpy as np

from tree import Tree


def parse_xgb_ensemble(model):
    """
    Parse XGBoost model based on its string representation.
    """
    string_data = _get_string_data_from_xgb_model(model)
    trees = np.array([_parse_xgb_tree(tree_str) for tree_str in string_data], dtype=np.dtype(object))

    if hasattr(model, 'n_classes_'):
        if model.n_classes_ == 2:
            bias = 0.0

        elif model.n_classes_ > 2:
            n_trees = int(trees.shape[0] / model.n_classes_)
            trees = trees.reshape((n_trees, model.n_classes_))
            bias = [0.0] * model.n_classes_

    else:
        bias = model.get_params()['base_score']

    return trees, bias


# private
def _parse_xgb_tree(tree_str):
    """
    Data has format:
    '
        <newlines and tabs><node_id>:[f<feature><<threshold>] yes=<int>,no=<int>,missing=<int>  (decision)
        <newlines and tabs><node_id>:leaf=<leaf_value>  (leaf)
    '

    Notes:
        - The structure is given as a newline and tab-indented string.
        - The node IDs are ordered in a breadth-first manner.

    Desired format:
        https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    """

    children_left = []
    children_right = []
    feature = []
    threshold = []
    leaf_vals = []

    # each line is a node, the no. preceeding tabs indicates its depth
    lines = tree_str.split('\n')

    # depth-first construction of recursive node dict from tree string representation
    line = lines.pop(0)
    node_dict = _parse_line(line)
    node_dict['left_child'] = _add_node(lines, depth=1)
    node_dict['right_child'] = _add_node(lines, depth=1)

    # add root node
    if node_dict['is_leaf']:
        leaf_vals.append(node_dict['leaf_val'])
        feature.append(-1)
        threshold.append(-1)

    else:  # decision node
        leaf_vals.append(-1)
        feature.append(node_dict['feature'])
        threshold.append(node_dict['threshold'])

    node_id = 1
    stack = [(node_dict['left_child'], 1), (node_dict['right_child'], 0)]

    # breadth-first traversal using the recursive node dict
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

            if node_dict['is_leaf']:  # leaf node
                feature.append(-1)
                threshold.append(-1)
                leaf_vals.append(node_dict['leaf_val'])
                stack.append((None, 1))
                stack.append((None, 0))

            else:  # decision node
                feature.append(node_dict['feature'])
                threshold.append(node_dict['threshold'])
                leaf_vals.append(-1)
                stack.append((node_dict['left_child'], 1))
                stack.append((node_dict['right_child'], 0))

            node_id += 1

    result = Tree(children_left, children_right, feature, threshold, leaf_vals)

    return result


def _add_node(lines, depth):
    """
    Search the remaining lines and parses the first one
    with the specified depth (depth = no. tabs).
    """
    node_dict = None

    for i in range(len(lines)):
        cur_depth = lines[i].count('\t')

        # no more nodes in this direction
        if cur_depth < depth:
            break

        # found more nodes in this direction
        elif cur_depth == depth:
            line = lines.pop(i)
            node_dict = _parse_line(line)
            node_dict['left_child'] = _add_node(lines, depth=depth + 1)
            node_dict['right_child'] = _add_node(lines, depth=depth + 1)
            break

    return node_dict


def _get_string_data_from_xgb_model(model):
    """
    Parse CatBoost model based on its json representation.
    """
    assert 'XGB' in str(model)
    string_data = model.get_booster().get_dump()  # 1d list of tree strings
    return string_data


def _parse_line(line):
    """
    Parse node string representation and return a dict with appropriate node values.
    """
    res = {}

    if 'leaf' in line:
        res['is_leaf'] = 1
        res['leaf_val'] = _parse_leaf_node_line(line)

    else:
        res['is_leaf'] = 0
        res['feature'], res['threshold'] = _parse_decision_node_line(line)

    return res


def _parse_decision_node_line(line, tol=1e-8):
    """
    Return feature index and threshold given the string representation of a decision node.

    Note:
        - Border is substracted by `tol` to convert XGB's < operator to a <=.
    """
    substr = line[line.find('[') + 1: line.find(']')]
    feature_str, border_str = substr.split('<')
    feature_ndx = int(feature_str[1:])
    border = float(border_str) - tol
    return feature_ndx, border


def _parse_leaf_node_line(line):
    """
    Return the leaf value given the string representation of a leaf node.
    """
    return float(line.split('=')[1])
