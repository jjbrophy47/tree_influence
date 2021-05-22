import os
import json
import shutil

import numpy as np

import util
from tree import Tree
from tree import TreeEnsemble


def parse_skgbm_ensemble(model):
    """
    Parse SKLearn GBM model using its underlying tree representations.
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
