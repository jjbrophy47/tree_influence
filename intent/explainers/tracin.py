import numpy as np

from .base import Explainer
from .parsers import util


class TracIn(Explainer):
    """
    Explainer that adapts the TracIn method to tree ensembles.

    Global-Influence Semantics
        - Global influence of x_i = gradient sum over all boosting iterations.
            * Regression: Pos. no. means increase in the pre-act. pred.; neg. is the opposite.
            * Binary: Pos. no. means increase in the pre-act. pred. for the POS. class; neg. is the opposite.
            * Multiclass: Pos. no. means increase in pre-act. pred. for the kth class; neg. decrease for kth class.
        - Negative gradients are used, since the model needs to move in the direction of the neg. gradient.
        - Could have done self-influence (e.g. measure loss of x_i on itself); however, this would
            amount to multiplying the gradient by itself. Thus, all numbers would be positive indicating
            each example reduces the loss on itself.

    Local-Influence Semantics
        - z and z' are train and test examples, respectively.
        - Original TracIn: TracIn(z, z') = sum_t learning_rate * dot_prod(grad(w_t, z), grad(w_t, z')).
            * Trying to approximate sum_{i=0}^n TracInIdeal(zi, z') = L(W0, z') - L(WT, z').
            * W0 and WT are the initial and final parameters before and after training.
        - Tree-ensemble Tracin: TracIn(z, z') = sum_t grad(z) * grad(z') * learning_rate
            * No dot product since GBDTs do gradient boosting in FUNCTIONAL space not parameter space.
            * Pos. number means reduction in test loss (a.k.a. proponent, helpful).
            * Neg. number means increase in test loss (a.k.a. opponent, harmful).
            * For multiclass, this applies to each class.

    Reference
         - https://github.com/frederick0329/TracIn

    Paper
        - https://arxiv.org/abs/2002.08484

    Note
        - We use negative gradients for both global and local influences.
            * For global, negative gradient is needed since we want the sign of the
                influence for each train example to represent the direction
                the model is needs to go towards.
            * For local, it does not matter if we use the gradient or
                negative gradient. Ultimately, the train gradient is MULTIPLIED by
                the test gradient; the value will only be negative if the
                gradient values have opposite signs, otherwise it will be positive.

    TODO
        - Add RF support?
    """
    def __init__(self, use_leaf=0, verbose=0):
        """
        Input
            use_leaf: bool, If True, only add attribution to examples
                ONLY if those examples share the same leaf as the test example.
            verbose: int, controls the amount of output.
        """
        self.use_leaf = use_leaf
        self.verbose = verbose

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

        self.n_train_ = X.shape[0]

        self.n_boost_ = self.model_.n_boost_
        self.n_class_ = self.model_.n_class_
        self.loss_fn_ = self._get_loss_function()

        self.train_gradients_ = self._compute_gradients(X, y)
        self.train_leaves_ = self.model_.apply(X)

        return self

    def get_global_influence(self):
        """
        - Compute influence of each training instance on itself.
            Sum of gradients across all boosting iterations.

        - Provides a global perspective of which training intances
          are most important.

        Return
            - Regression and binary: 1d array of shape=(no. train,).
            - Multiclass: 2d array of shape=(no. train, no. class).
            - Arrays are returned in the same order as the traing data.
        """
        self_influence = self.train_gradients_.sum(axis=1)  # sum over all boosting iterations

        if self.model_.objective in ['regression', 'binary']:
            self_influence = self_influence.squeeze()  # remove class axis

        return self_influence

    def get_local_influence(self, X, y):
        """
        - Compute influence of each training instance on the test loss
            by computing the dot prod. between the train gradient and the test gradient.

        - Provides a local explanation of the test instance loss.

        - If `use_leaf` is True, then attribute train attribution to the test loss
            ONLY if the train example is in the same leaf(s) as the test example.

        Return
            - Regression and binary: 2d array of shape=(no. train, X.shape[0]).
            - Multiclass: 3d array of shape=(X.shape[0], no. train, no. class).
            - Array is returned in the same order as the training data.
        """
        X, y = util.check_data(X, y, objective=self.model_.objective)

        train_gradients = self.train_gradients_  # shape=(no. train, no. boost, no. class)
        test_gradients = self._compute_gradients(X, y)  # shape=(X.shape[0], no. boost, no. class)
        weight = 1.0 / self.n_train_

        # get leaf indices each example arrives in
        if self.use_leaf:
            train_leaves = self.train_leaves_  # shape=(no. train, no. boost, no. class)
            test_leaves = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)
            leaf_weights = 1.0 / self.model_.get_leaf_counts()  # shape=(no. boost, no. class)

        # result container, shape=(X.shape[0], no. train, no. class)
        influence = np.zeros((X.shape[0], self.n_train_, self.n_class_), dtype=np.float32)

        # compute attributions for each test example
        for i in range(X.shape[0]):

            if self.use_leaf:
                mask = np.where(train_leaves == test_leaves[i], 1, 0)  # shape=(no. train, no. boost, no. class)
                weighted_mask = mask * leaf_weights  # shape=(no. train, no. boost, no. class)
                influence[i, :, :] = np.sum(train_gradients * weighted_mask * test_gradients[i], axis=1)

            else:
                influence[i, :, :] = np.sum(train_gradients * test_gradients[i] * weight, axis=1)

        # reshape result based on the objective
        if self.model_.objective in ['regression', 'binary']:
            influence = influence.transpose(1, 0, 2)  # shape=(no. train, X.shape[0], no. class)
            influence = influence.squeeze()  # remove class axis

        return influence

    # private
    def _compute_gradients(self, X, y):
        """
        Compute negative gradients for all train instances across all boosting iterations.

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
        learning_rate = self.model_.learning_rate

        current_approx = np.tile(bias, (n_train, 1)).astype(np.float32)  # shape=(X.shape[0], no. class)
        gradients = np.zeros((n_train, n_boost, n_class))  # shape=(X.shape[0], no. boost, no. class)

        # compute gradients for each boosting iteration
        for boost_idx in range(n_boost):
            gradients[:, boost_idx, :] = self.loss_fn_.gradient(y, current_approx)  # shape=(no. train, no. class)

            # update approximation
            for class_idx in range(n_class):
                current_approx[:, class_idx] += trees[boost_idx, class_idx].predict(X)

        return -gradients * learning_rate

    def _get_loss_function(self):
        """
        Return the appropriate loss function for the given objective.
        """
        if self.model_.objective == 'regression':
            loss_fn = util.SquaredLoss()

        elif self.model_.objective == 'binary':
            loss_fn = util.LogisticLoss()

        else:
            assert self.model_.objective == 'multiclass'
            n_class = self.model_.n_class_
            loss_fn = util.SoftmaxLoss(factor=self.model_.factor, n_class=n_class)

        return loss_fn
