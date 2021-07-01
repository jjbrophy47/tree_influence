import numpy as np

from . import util
from ._tree import _Tree


class Tree(object):
    """
    Wrapper for the standardized tree structure object.

    Note:
        - The Tree object is a binary tree structure.
        - The tree structure is used for predictions and extracting
          structure information.

    Reference:
    https://github.com/scikit-learn/scikit-learn/blob/
        15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/tree/_tree.pyx
    """

    def __init__(self, children_left, children_right, feature, threshold, leaf_vals):
        children_left = np.array(children_left, dtype=np.intp)
        children_right = np.array(children_right, dtype=np.intp)
        feature = np.array(feature, dtype=np.intp)
        threshold = np.array(threshold, dtype=np.float32)
        leaf_vals = np.array(leaf_vals, dtype=np.float32)
        self.tree_ = _Tree(children_left, children_right, feature, threshold, leaf_vals)

    def predict(self, X):
        """
        Return 1d array of leaf values, shape=(X.shape[0]).
        """
        assert X.ndim == 2
        return self.tree_.predict(X)

    def apply(self, X):
        """
        Return 1d array of leaf indices, shape=(X.shape[0],).
        """
        assert X.ndim == 2
        return self.tree_.apply(X)

    def get_leaf_values(self):
        """
        Return 1d array of leaf values, shape=(no. leaves,).
        """
        return self.tree_.get_leaf_values()

    def update_node_count(self, X):
        """
        Update node counts based on the paths taken by x in X.
        """
        assert X.ndim == 2
        self.tree_.update_node_count(X)

    def leaf_path(self, X, output=False, weighted=False):
        """
        Return 2d vector of leaf one-hot encodings, shape=(X.shape[0], no. leaves).
        """
        return self.tree_.leaf_path(X, output=output, weighted=weighted)

    def feature_path(self, X, output=False, weighted=False):
        """
        Return 2d vector of feature one-hot encodings, shape=(X.shape[0], no. nodes).
        """
        return self.tree_.feature_path(X, output=output, weighted=weighted)

    @property
    def node_count_(self):
        return self.tree_.node_count_

    @property
    def leaf_count_(self):
        return self.tree_.leaf_count_


class TreeEnsemble(object):
    """
    Abstract class for TreeEnsemble classes.
    """
    def __init__(self, trees, objective, tree_type, bias,
                 learning_rate, l2_leaf_reg, factor):
        """
        Input
            trees: A 1d (or 2d for multiclass) array of Tree objects.
            objective: str, task ("regression", "binary", or "multiclass").
            bias: A single or 1d list (for multiclass) of floats.
                If classification, numbers are in log space.
            tree_type: str, "gbdt" or "rf".
            learning_rate: float, learning rate (GBDT models only).
            l2_leaf_reg: float, leaf regularizer (GBDT models only).
        """
        assert trees.dtype == np.dtype(object)
        self.trees = trees
        self.objective = objective
        self.tree_type = tree_type
        self.bias = bias
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.factor = factor

    def __str__(self):
        """
        Return string representation of model.
        """
        params = self.get_params()
        return str(params)

    def get_params(self):
        """
        Return dict. of object parameters.
        """
        params = {}
        params['objective'] = self.objective
        params['tree_type'] = self.tree_type
        params['bias'] = self.bias
        params['learning_rate'] = self.learning_rate
        params['l2_leaf_reg'] = self.l2_leaf_reg
        params['factor'] = self.factor

    def predict(self, X):
        """
        Sums leaf values from all trees for each x in X.

        Returns 1d array of leaf values.

        NOTE: Only works for binary classification and regression.
              Multiclass classification must override this method.
        """
        X = util.check_input_data(X)

        # sum predictions over all trees
        pred = np.zeros(X.shape[0])
        for i, tree in enumerate(self.trees):
            pred += tree.predict(X)

        pred += self.bias

        if self.tree_type == 'rf':
            pred /= self.trees.shape[0]

        return pred

    def apply(self, X):
        """
        Returns 2d array of leaf indices of shape=(X.shape[0], no. trees).

        Note:
            - Only works for regression and binary, multiclass must override.
        """
        return np.hstack([tree.apply(X).reshape(-1, 1) for tree in self.trees]).astype(np.int32)

    def get_leaf_values(self):
        """
        Returns 1d array of leaf values of shape=(no. leaves across all trees,).

        Note:
            - Only works for regression and binary, multiclass must override.
        """
        return np.concatenate([tree.get_leaf_values() for tree in self.trees]).astype(np.float32)

    def get_leaf_counts(self):
        """
        Returns 1d array of leaf counts, one per tree; shape=(no. trees,).

        Note:
            - Only works for regression and binary, multiclass must override.
        """
        return np.array([tree.leaf_count_ for tree in self.trees]).astype(np.int32)

    def update_node_count(self, X):
        """
        Increment each node's count for each x in X that passes through each node
            for all trees in the ensemble.

        Note:
            - Works for regression, binary, and multiclass.
        """
        for tree in self.trees.flatten():
            tree.update_node_count(X)

    @property
    def n_class_(self):
        result = 0

        if self.objective == 'regression':
            result = 0

        elif self.objective == 'binary':
            result = 2

        else:
            assert self.objective == 'multiclass'
            result = self.trees.shape[1]

        return result


class TreeEnsembleRegressor(TreeEnsemble):
    """
    Extension of the TreeEnsemble class for regression.
    """
    def __init__(self, trees, params):
        super().__init__(trees, **params)
        assert self.trees.ndim == 1
        assert isinstance(self.bias, float)


class TreeEnsembleBinaryClassifier(TreeEnsemble):
    """
    Extension of the TreeEnsemble class for binary classfication.
    """
    def __init__(self, trees, params):
        super().__init__(trees, **params)
        assert self.trees.ndim == 1
        assert isinstance(self.bias, float)

    def predict(self, X):
        """
        Classify samples one by one and return the list of probabilities
        """
        pred = super().predict(X)
        proba = pred.reshape(-1, 1) if self.tree_type == 'rf' else util.sigmoid(pred).reshape(-1, 1)
        proba = np.hstack([1 - proba, proba])

        return proba


class TreeEnsembleMulticlassClassifier(TreeEnsemble):
    """
    Extension of the TreeEnsemble class for multiclass classfication.
    """

    def __init__(self, trees, params):
        """
        Input should be an array of Tree objects of shape=(no. trees, no. classes)
        """
        super().__init__(trees, **params)
        assert self.trees.ndim == 2
        assert self.trees.shape[1] >= 3
        assert len(self.bias) >= 3

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

        # if RF, average
        if self.tree_type == 'rf':
            proba = pred / self.trees.shape[0]

        else:
            proba = util.softmax(pred)

        return proba

    def apply(self, X):
        """
        Returns a 2d array of leaf indices of shape=(no. trees, X.shape[0], no. class).
        """
        leaves = np.zeros((self.trees.shape[0], X.shape[0], self.trees.shape[1]), dtype=np.int32)

        for i in range(self.trees.shape[0]):  # per tree
            for j in range(self.trees.shape[1]):  # per class
                leaves[i, :, j] = self.trees[i][j].apply(X)

        return leaves
