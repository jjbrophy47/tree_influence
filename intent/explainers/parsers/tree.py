import numpy as np

from . import util
from ._tree import _Tree


class Tree(object):
    """
    Standardized tree structure object class.

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


class TreeEnsemble(object):
    """
    Abstract class for TreeEnsemble classes.
    """
    def __init__(self, trees, bias, tree_type='gbdt'):
        """
        Input
            trees: A 1d (or 2d for multiclass) array of Tree objects.
            bias: A single or 1d list (for multiclass) of floats.

        """
        assert trees.dtype == np.dtype(object)
        self.trees = trees
        self.bias = bias
        self.tree_type = tree_type

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

        NOTE: Only works for binary classification and regression.
              Multiclass classification must override this method.
        """
        return np.hstack([tree.apply(X).reshape(-1, 1) for tree in self.trees]).astype(np.int32)


class TreeEnsembleRegressor(TreeEnsemble):
    """
    Extension of the TreeEnsemble class for regression.
    """
    def __init__(self, trees, bias, tree_type='gbdt'):
        super().__init__(trees, bias, tree_type)
        assert self.trees.ndim == 1
        assert isinstance(bias, float)
        self.task_ = 'regression'


class TreeEnsembleBinaryClassifier(TreeEnsemble):
    """
    Extension of the TreeEnsemble class for binary classfication.
    """
    def __init__(self, trees, bias, tree_type='gbdt'):
        super().__init__(trees, bias, tree_type)
        assert self.trees.ndim == 1
        assert isinstance(bias, float)
        self.task_ = 'binary'

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

    def __init__(self, trees, bias, tree_type='gbdt'):
        """
        Input should be an array of Tree objects of shape=(no. trees, no. classes)
        """
        super().__init__(trees, bias, tree_type)
        assert self.trees.ndim == 2
        assert self.trees.shape[1] >= 3
        assert len(bias) >= 3
        self.task_ = 'multiclass'

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
