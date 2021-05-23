import numpy as np

import util
from tree import Tree


def parse_skgbm_ensemble(model):
    """
    Parse SKLearn GBM model using its underlying tree representations.

    Note:
        - RandomForestRegressor and RandomForestClassifier, `estimators_`.shape=(no. estimators, no .classes).
        - Each tree's `value`.shape=(no. nodes, 1, 1).
    """
    estimators = model.estimators_
    trees = np.zeros(estimators.shape, dtype=np.dtype(object))

    scale = model.learning_rate

    for i in range(estimators.shape[0]):  # per tree
        for j in range(estimators.shape[1]):  # per class
            t = estimators[i][j].tree_
            children_left = list(t.children_left)
            children_right = list(t.children_right)
            feature = list(t.feature)
            threshold = list(t.threshold)
            leaf_vals = list(t.value.flatten() * scale)
            trees[i][j] = Tree(children_left, children_right, feature, threshold, leaf_vals)

    if trees.shape[1] == 1:
        trees = trees.flatten()

    # set bias
    bias = 0.0

    # regression
    if hasattr(model.init_, 'constant_'):
        assert model.loss == 'ls'  # least squares
        bias = model.init_.constant_.flatten()[0]

    # classification
    else:
        class_prior = model.init_.class_prior_

        # binary
        if class_prior.shape[0] == 2:
            bias = util.logit(class_prior[1])

        # multiclass
        else:
            bias = list(np.log(class_prior))

    return trees, bias


def parse_skrf_ensemble(model):
    """
    Parse SKLearn RF model using its underlying tree representations.

    Note:
        - RandomForestRegressor and RandomForestClassifier, `estimators_`.shape=(no. estimators).
        - For multiclass classification, each tree's `value`.shape=(no. nodes, 1, no. classes).
    """
    estimators = model.estimators_
    n_class = model.n_classes_ if hasattr(model, 'n_classes_') else 0

    if n_class <= 2:  # regression, binary classification
        trees = np.zeros(len(estimators), dtype=np.dtype(object))

    else:  # multiclass
        trees = np.zeros((len(estimators), n_class), dtype=np.dtype(object))

    for i in range(len(estimators)):  # per tree
        t = estimators[i].tree_
        children_left = list(t.children_left)
        children_right = list(t.children_right)
        feature = list(t.feature)
        threshold = list(t.threshold)

        # regression
        if n_class == 0:
            leaf_vals = list(t.value.flatten())
            trees[i] = Tree(children_left, children_right, feature, threshold, leaf_vals)

        # binary classification
        elif n_class == 2:
            value = t.value.squeeze()  # value.shape=(no. nodes, 2)
            leaf_vals = (value / value.sum(axis=1).reshape(-1, 1))[:, 1].tolist()
            trees[i] = Tree(children_left, children_right, feature, threshold, leaf_vals)

        # multiclass classification
        else:
            value = t.value.squeeze()  # value.shape=(no. nodes, no. classes)
            value /= value.sum(axis=1).reshape(-1, 1)  # normalize
            for j in range(value.shape[1]):  # per class
                leaf_vals = list(value[:, j])
                trees[i][j] = Tree(children_left, children_right, feature, threshold, leaf_vals)

    # set bias
    bias = 0.0

    if n_class >= 3:
        bias = [0.0] * n_class

    return trees, bias
