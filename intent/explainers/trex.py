import numpy as np
from sklearn.preprocessing import LabelBinarizer

from .base import Explainer
from .parsers import util


class TREX(Explainer):
    """
    Tree-Ensemble Representer Examples: Explainer that adapts the
    Representer point method to tree ensembles.

    Notes
    """
    def __init__(self, kernel='wlp', target='actual', lmbd=0.03):
        """
        Input
            kernel: Transformation of the input using the tree-ensemble structure.
                'to_': Tree output; output of each tree in the ensemble.
                'lp_': Leaf path; one-hot encoding of leaf indices across all trees.
                'lo_': Leaf output; like 'lp' but replaces 1s with leaf values.
                'lpw': Weighted leaf path; like 'lp' but replaces 1s with 1 / leaf count.
                'low': Weighted leaf otput; like 'lo' but replace leaf value with 1 / leaf value.
                'fp': Feature path; one-hot encoding of node indices across all trees.
                'fo': Feature output; like 'fp' but replaces leaf 1s with leaf values.
                'fpw': Weighted feature path; like 'fp' but replaces 1s with 1 / node count.
                'fow': Weighted feature path; like 'fo' but replaces leaf 1s with 1 / leaf values.
            target: Targets for the linear model to train on.
                'actual': Ground-truth targets.
                'predicted': Predicted targets from the tree-ensemble.
            lmbd: Regularizer for the linear model; necessary for the Representer decomposition.
        """
        assert kernel in ['to_', 'lp_', 'lo_', 'lpw', 'low', 'fp', 'fo', 'fpw', 'fow']
        assert target in ['actual', 'predicted']
        assert isinstance(lmbd, float)
        self.kernel = kernel
        self.target = target
        self.lmbd = lmbd

    def fit(self, model, X, y):
        """
        - Convert model to internal standardized tree structure.
        - Transform the input data using the specified tree kernel.
        - Fit linear model and compute train instance weights (alpha).

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, task=self.model_.task_)

        self.model_.update_node_counts(X)

        self.X_train_ = self._kernel_transform(X)
        self.y_train_ = LabelBinarizer().fit_transform(y) if self.model_.task_ == 'multiclass' else y

        self.alpha_ = self._compute_train_weights(self.X_train_, self.y_train_)

        return self

    def get_self_influence(self):
        """
        - Compute influence of each training instance on itself.
        - Provides a global importance to all training intances.

        Return
            - 1d array of shape=(no. train) (regression and binary classification).
            - 2d array of shape=(no. train, no. classes) (multiclass).
            - Arrays are returned in the same order as the traing data.
        """
        return self.alpha_

    def explain(self, X, y):
        """
        - Compute attribution of each training instance on the test instance prediction.
            Transform the test instance using the specified tree kernel.
            Compute dot prod. between transformed train and test, weighted by alpha.
        - Provides a local explanation of the test instance prediction.

        Return
            - Regression and binary: 1d array of shape=(no. train).
            - Multiclass: 2d array of shape=(no. train, no. classes).
            - Array is returned in the same order as the traing data.
        """
        X, y = util.check_data(X, y, task=self.model_.task_)
        assert X.shape[0] == 1 and y.shape[0] == 1

        X_test_ = self._kernel_transform(X).flatten()
        sim = np.dot(self.X_train_, X_test_)

        if self.model_.task_ in ['regression', 'binary']:
            influence = sim * self.alpha_  # representer values

        elif self.model_.task_ == 'multiclass':
            influence = np.zeros(self.alpha_.shape, dtype=np.float32)

            for j in range(influence.shape[1]):  # per class
                influence[:, j] = sim * self.alpha_[j]  # representer values

        return influence

    # private
    def _kernel_transform(self, X):
        """
        Transforms each x in X using the specified tree kernel.
        """
        structure_dict = {'t': 'tree', 'l': 'leaf', 'f': 'feature'}
        output_dict = {'p': 'path', 'o': 'output'}
        weight_dict = {'_': 'unweighted', 'w': 'weighted'}
        
        s1, s2, s3 = list(self.kernel)
        structure = structure_dict[s1]
        output = output_dict[s2]
        weight = weight_dict[s3]

        if structure == 'tree':
            X_ = self._tree_kernel_transform(X)

        elif structure == 'leaf':
            X_ = self._leaf_kernel_transform(X, output=output, weight=weight)

        elif structure == 'feature':
            X_ = self._feature_kernel_transform(X, output=output, weight=weight)

        return X_

    def _tree_kernel_transform(self, X):
        """
        Transform each x in X to be a vector of tree outputs.

        Return
            - Regression and binary: 2d array of shape=(no. train, no. trees).
            - Multiclass: 2d array of shape=(no. train, no. trees * no. class).
        """
        trees = self.model_.trees.flatten()
        X_ = np.zeros((X.shape[0], trees.shape[0]))

        for i, tree in enumerate(trees):
            X_[:, i] = tree.predict(X)

        return X_

    def _leaf_kernel_transform(self, X, output='path', weight='unweighted'):
        """
        - Transform each x in X to be a vector of one-hot encoded leaf paths.
        - The `output` and `weight` parameters control the value of the 1s.

        Return
            - Regression and binary: 2d array of shape=(no. train, total no. leaves).
            - Multiclass: 2d array of shape=(no. train, ~total no. leaves * no. class).
        """
        trees = self.model_.trees.flatten()
        total_n_leaves = np.sum([tree.leaf_count_ for tree in trees])

        X_ = np.zeros((X.shape[0], total_n_leaves))

        output = True if output == 'output' else False
        weighted = True if weight == 'weighted' else False

        n_prev_leaves = 0
        for tree in trees:
            start = n_prev_leaves
            stop = n_prev_leaves + tree.leaf_count_
            X_[:, start: stop] = tree.leaf_path(X, output=output, weighted=weighted)
            n_prev_leaves += tree.leaf_count_

        return X_

    def _feature_kernel_transform(self, X, output='path', weight='unweighted'):
        """
        - Transform each x in X to be a vector of one-hot encoded feature paths.
        - The `output` parameter controls the value of the leaf 1s.
        - The `weight` parameter controls the value of all 1s.

        Return
            - Regression and binary: 2d array of shape=(no. train, total no. nodes).
            - Multiclass: 2d array of shape=(no. train, ~total no. nodes * no. class).
        """
        trees = self.model_.trees.flatten()
        total_n_nodes = np.sum([tree.node_count_ for tree in trees])

        X_ = np.zeros((X.shape[0], total_n_nodes))

        output = True if output == 'output' else False
        weighted = True if weight == 'weighted' else False

        n_prev_nodes = 0
        for tree in trees:
            start = n_prev_nodes
            stop = n_prev_nodes + tree.node_count_
            X_[:, start: stop] = tree.feature_path(X, output=output, weighted=weighted)
            n_prev_nodes += tree.node_count_

        return X_

    def _compute_train_weights(self, X, y):
        """
        Fit a linear model to X and y, then extract weights for all
        train instances.
        """
        pass
