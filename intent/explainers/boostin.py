import numpy as np

from .base import Explainer
from .parsers import util


class BoostIn(Explainer):
    """
    Explainer that adapts the TracIn method to tree ensembles.

    Global-Influence Semantics
        - Influence of x_i on itself
        - Inf.(x_i, x_i) = sum of -grad(x_i) * -grad(x_i) * learning_rate over all boosts.
        - Pos. value means a decrease in loss (a.k.a. proponent, helpful).
        - Neg. value means an increase in loss (a.k.a. opponent, harmful).

    Local-Influence Semantics
        - Inf.(x_i, x_t) = sum of -grad(x_i) * -grad(x_t) * learning_rate over all boosts.
        - Pos. value means a decrease in test loss (a.k.a. proponent, helpful).
        - Neg. value means an increase in test loss (a.k.a. opponent, harmful).

    Reference
        - https://github.com/frederick0329/TracIn

    Paper
        - https://arxiv.org/abs/2002.08484

    Note
        - It does not matter if we use pos. or neg. gradients when computing influence.
            For global, the value will always be positive; for local, we only care if
            the signs of the gradients differ.
        - Only support GBDTs.
    """
    def __init__(self, use_leaf=0, local_op='normal', logger=0):
        """
        Input
            use_leaf: bool, If True, only add attribution to examples
                ONLY if those examples share the same leaf as the test example.
            local_op: str, Configures how the local influence is computed. 
            logger: object, If not None, output to logger.
        """
        assert use_leaf in [0, 1]
        assert local_op in ['normal', 'sign', 'sim']
        self.use_leaf = use_leaf
        self.local_op = local_op
        self.logger = logger

    def fit(self, model, X, y):
        """
        - Convert model to internal standardized tree structure.
        - Precompute gradients and leaf indices for each x in X.

        Input
            model: tree ensemble.
            X: 2d array of train examples.
            y: 1d array of train targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        assert self.model_.tree_type != 'rf', 'RF not supported for BoostIn'

        self.n_train_ = X.shape[0]

        self.n_boost_ = self.model_.n_boost_
        self.n_class_ = self.model_.n_class_
        self.learning_rate_ = self.model_.learning_rate

        self.loss_fn_ = util.get_loss_fn(self.model_.objective, self.model_.n_class_, self.model_.factor)

        self.train_gradients_ = self._compute_gradients(X, y)
        self.train_leaves_ = self.model_.apply(X)

        return self

    def get_global_influence(self, X=None, y=None):
        """
        - Provides a global importance to all training examples.

        Input
            X: 2d array of test data.
            y: 2d array of test targets.

        Return
            - 1d array of shape=(no. train,).
                * Arrays are returned in the same order as the traing data.
        """
        # compute self influence, shape=(no. train, no. boost, no. class)
        influence = self.train_gradients_ * self.train_gradients_ * self.learning_rate_
        influence = influence.sum(axis=(1, 2))  # sum over boosts and classes
        return influence

    def get_local_influence(self, X, y):
        """
        - Computes effect of each train example on the loss of the test example.

        Input
            X: 2d array of test data.
            y: 2d array of test targets.

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Array is returned in the same order as the training data.

        Note
            - If `use_leaf` is True, then attribute train attribution to the test loss
                ONLY if the train example is in the same leaf(s) as the test example.
        """
        X, y = util.check_data(X, y, objective=self.model_.objective)

        train_gradients = self.train_gradients_  # shape=(no. train, no. boost, no. class)
        test_gradients = self._compute_gradients(X, y)  # shape=(X.shape[0], no. boost, no. class)
        weight = 1.0 / self.n_train_
        lr = self.learning_rate_

        # get leaf indices each example arrives in
        if self.use_leaf:
            train_leaves = self.train_leaves_  # shape=(no. train, no. boost, no. class)
            test_leaves = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)
            leaf_weights = 1.0 / self.model_.get_leaf_counts()  # shape=(no. boost, no. class)

        # use sign of gradients instead of their sign + magnitude.
        if self.local_op in ['sign', 'sim']:
            train_gradients = np.sign(train_gradients)
            test_gradients = np.sign(test_gradients)

        # result container, shape=(X.shape[0], no. train, no. class)
        influence = np.zeros((self.n_train_, X.shape[0]), dtype=np.float32)

        # compute attributions for each test example
        for i in range(X.shape[0]):

            if self.use_leaf:
                mask = np.where(train_leaves == test_leaves[i], 1, 0)  # shape=(no. train, no. boost, no. class)
                weighted_mask = mask * leaf_weights  # shape=(no. train, no. boost, no. class)
                prod = train_gradients * weighted_mask * test_gradients[i] * lr
                influence[:, i] = np.sum(prod, axis=(1, 2))  # shape=(no. train,)

                if self.local_op == 'sim':
                    sim = np.square(train_gradients * weighted_mask * test_gradients[i])  # (n_train, n_boost, n_class)
                    influence[:, i] = np.sum(sim, axis=(1, 2))  # shape=(no. train,)

            else:
                prod = train_gradients * weight * test_gradients[i] * lr
                influence[:, i] = np.sum(prod, axis=(1, 2))  # shape=(no. train,)

        return influence

    # private
    def _compute_gradients(self, X, y):
        """
        Compute gradients for all train instances across all boosting iterations.

        Input
            X: 2d array of train examples.
            y: 1d array of train targets.

        Return
            - 3d array of shape=(X.shape[0], no. boost, no. class).
        """
        n_train = X.shape[0]

        trees = self.model_.trees
        n_boost = self.model_.n_boost_
        n_class = self.model_.n_class_
        bias = self.model_.bias

        current_approx = np.tile(bias, (n_train, 1)).astype(np.float32)  # shape=(X.shape[0], no. class)
        gradients = np.zeros((n_train, n_boost, n_class))  # shape=(X.shape[0], no. boost, no. class)

        # compute gradients for each boosting iteration
        for boost_idx in range(n_boost):
            gradients[:, boost_idx, :] = self.loss_fn_.gradient(y, current_approx)  # shape=(no. train, no. class)

            # update approximation
            for class_idx in range(n_class):
                current_approx[:, class_idx] += trees[boost_idx, class_idx].predict(X)

        return gradients
