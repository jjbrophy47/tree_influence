import json
import numpy as np

from abc import abstractmethod
from scipy.special import softmax

import util


class Node(object):

    def __init__(self, node_id, left_child=None, right_child=None,
                 feature=-1, threshold=-1, leaf_val=-1, is_leaf=0):
        self.node_id = node_id
        self.left_child = left_child
        self.right_child = right_child
        self.feature = feature
        self.threshold = threshold
        self.leaf_val = leaf_val
        self.is_leaf = is_leaf


class AbstractTree(object):

    @abstractmethod
    def predict(X):
        pass

class Tree(AbstractTree):
    """
    The Tree object is a binary tree structure.
    The tree structure is used for predictions.

    Reference:
    https://github.com/scikit-learn/scikit-learn/blob/
        15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/tree/_tree.pyx
    """

    def __init__(self, children_left, children_right, feature, threshold, leaf_vals):
        self.children_left = children_left
        self.children_right = children_right
        self.feature = feature
        self.threshold = threshold
        self.leaf_vals = leaf_vals

        # print('children_left', self.children_left)
        # print('children_right', self.children_right)
        # print('feature', self.feature)
        # print('threshold', self.threshold)
        # print('leaf_vals', self.leaf_vals)

        # print(len(self.children_left), len(self.children_right), len(self.feature),
        #       len(self.threshold), len(self.leaf_vals))

        self.root_ = self._add_node(node_id=0)

    def __str__(self):
        return self._get_str(node=self.root_, depth=0)

    def predict(self, X):
        assert X.ndim == 2

        out = np.zeros(X.shape[0], dtype=np.float32)

        for i in range(X.shape[0]):
            node = self.root_
            # print(f'\ntest {i}: {X[i]}')

            while not node.is_leaf:
                # print(node.node_id, node.feature, node.threshold)
                if X[i, node.feature] <= node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child

            # print(node.node_id, node.leaf_val)
            out[i] = node.leaf_val

        return out

    # private
    def _get_str(self, node=None, depth=0):

        if node is None:
            return ''

        space = depth * "\t"

        if node.is_leaf:
            s = f'\n{space}[Node {node.node_id}], leaf_val: {node.leaf_val:.3f}'
        else:
            s = f'\n{space}[Node {node.node_id}], feature: {node.feature}, threshold: {node.threshold:.3f}'

        s += self._get_str(node=node.left_child, depth=depth + 1)
        s += self._get_str(node=node.right_child, depth=depth + 1)

        return s

    def _add_node(self, node_id):
        """
        Recursively create a node and return it.
        """
        node = Node(node_id=node_id)

        if self.children_left[node_id] != self.children_right[node_id]:  # split node
            node.feature = self.feature[node_id]
            node.threshold = self.threshold[node_id]

            if self.children_left[node_id] != -1:
                node.left_child = self._add_node(self.children_left[node_id])

            if self.children_right[node_id] != -1:
                node.right_child = self._add_node(self.children_right[node_id])

        else:  # leaf node
            node.leaf_val = self.leaf_vals[node_id]
            node.is_leaf = 1

        return node


class TreeEnsemble(object):
    """
    Abstract class for the TreeEnsemble class.
    """
    def __init__(self, trees, bias):
        """
        Input
            trees: A 1d (or 2d for multiclass) array of Tree objects.
            bias: A single or 1d list (for multiclass) of floats.

        """
        assert trees.dtype == np.dtype(object)
        self.trees = trees
        self.bias = bias

    def predict(self, X):
        """
        Sums leaf values from all trees for each x in X.

        Returns 1d array of predictions

        NOTE: Only works for binary classification and regression.
              Multiclass classification must override this method.
        """
        X = util.check_input_data(X)

        # sum predictions over all trees
        pred = np.zeros(X.shape[0])
        for i, tree in enumerate(self.trees):
            pred += tree.predict(X)

        return pred


class TreeEnsembleRegressor(TreeEnsemble):
    """
    Model that parses each model.
    """
    def __init__(self, trees, bias):
        super().__init__(trees, bias)
        assert self.trees.ndim == 1
        assert isinstance(bias, float)

    def predict(self, X):
        """
        Classify samples one by one and return the list of probabilities
        # TODO: take average if RF

        Returns 1d array of shape(X.shape[0])
        """
        pred = super().predict(X) + self.bias
        return pred


class TreeEnsembleBinaryClassifier(TreeEnsemble):
    """
    Model that parses each model.
    """
    def __init__(self, trees, bias):
        super().__init__(trees, bias)
        assert self.trees.ndim == 1
        assert isinstance(bias, float)

    def predict(self, X):
        """
        Classify samples one by one and return the list of probabilities
        # TODO: take average if RF
        """
        pred = super().predict(X) + self.bias
        proba = util.sigmoid(pred).reshape(-1, 1)
        proba = np.hstack([1 - proba, proba])
        return proba


class TreeEnsembleMulticlassClassifier(TreeEnsemble):
    """
    Model that parses each model.
    """

    def __init__(self, trees, bias):
        """
        Input should be an array of Tree objects of shape=(no. trees, no. classes)
        """
    def __init__(self, trees, bias):
        super().__init__(trees, bias)
        assert self.trees.ndim == 2
        assert self.trees.shape[1] >= 3
        assert len(bias) >= 3

    def predict(self, X):
        """
        Classify samples one by one and return the list of probabilities
        """
        X = util.check_input_data(X)

        # sum all predictions instead of storing them
        pred = np.zeros((X.shape[0], self.trees.shape[1]), dtype=np.float32)  # shape=(no. instances, no. classes)
        for i in range(self.trees.shape[1]):  # per class
            class_pred = np.zeros(X.shape[0])

            for j in range(self.trees.shape[0]):  # per tree
                class_pred += self.trees[j, i].predict(X)

            pred[:, i] = class_pred + self.bias[i]

        # TODO: take average if RF
        proba = util.softmax(pred)

        return proba


if __name__ == '__main__':
    model = CatBoostClassifier().fit(X, y)
    model = parse_model(model)
    explainer = InTEnt(model)
