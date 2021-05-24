import numpy as np
from sklearn.preprocessing import LabelBinarizer

from .base import Explainer
from .parsers import util


class TracIn(Explainer):
    """
    Abstract TracIn explainer that adapts the TracIn method to tree ensembles.

    TODO
        - Should we consider residuals from iteration 0 (initial guess)?
          How would this work for RF?
        - Currently, we are NOT skipping first residuals for self-influence,
          but we ARE skipping first residuals for explain.
        - Also RF is using first residuals for self-influence, should we
          get rid of this since there is no initial guess for RF? Probably yes.

        - RF: remove initial guess (iteraion 0) residuals.
        - GBDT: Should we skip initial residuals for both self-influence and explain?
    """
    def fit(self, model, X, y):
        """
        - Convert model to internal standardized tree structure.
        - Precompute gradients and leaf indices for each x in X.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, task=self.model_.task_)

        if self.model_.task_ == 'multiclass':
            self.label_binarizer_ = LabelBinarizer().fit(y)  # shape=(no. train, no. class)

        self.gradients_ = self._compute_gradients(X, y)
        self.leaves_ = self.model_.apply(X)

        return self

    def get_self_influence(self):
        """
        - Compute influence of each training instance on itself.
            - Sum of gradients across all trees (boosting iterations).
        - Provides a global perspective of which training intances
          are most important.

        Return
            - 1d array of shape=(no. train) (regression and binary classification).
            - 2d array of shape=(no. train, no. classes) (multiclass).
            - Arrays are returned in the same order as the traing data.
        """
        if self.model_.task_ in ['regression', 'binary']:
            self_influence = self.gradients_.sum(axis=1)

        elif self.model_.task_ == 'multiclass':
            self_influence = self.gradients_.sum(axis=0)

        return self_influence

    def explain(self, X, y):
        """
        - Compute influence of each training instance on the test instance loss (or prediction?)
          by computing the dot prod. between each train gradient and the test gradient.
        - Provides a local explanation of the test instance loss (or prediction?).

        Return
            - Regression and binary: 1d array of shape=(no. train).
            - Multiclass: 2d array of shape=(no. train, no. classes).
            - Array is returned in the same order as the traing data.
        """
        X, y = util.check_data(X, y, task=self.model_.task_)
        assert X.shape[0] == 1 and y.shape[0] == 1

        test_grad = self._compute_gradients(X, y)
        test_leaves = self.model_.apply(X)

        # only add influence if train and test end in the same leaf per tree
        mask = np.where(self.leaves_ == test_leaves, 1, 0)

        # train_gradients=(no. train, no. trees), test_gradient=(1, no. trees,)
        # train_leaf_indices=(no. train, no. trees), test_leaf_indices=(1, no. trees,)
        if self.model_.task_ in ['regression', 'binary']:
            train_grad = mask * self.gradients_[:, 1:]  # skip first residuals
            influence = np.dot(train_grad, test_grad[:, 1:].flatten())  # skip first residuals

        # train_gradients=(no. trees, no. train, no. class), test_gradient=(no. trees, 1, no. class)
        # train_leaf_indices=(no. trees, no. train, no. class), test_leaf_indices=(no. trees, 1, no. class)
        elif self.model_.task_ == 'multiclass':
            train_grad = self.gradients_[1:].copy()  # skip first residuals
            train_grad = mask * train_grad
            test_grad = test_grad[1:].copy()  # skip first residuals

            train_grad = train_grad.transpose(2, 1, 0)  # shape=(no. class, no. train, no. trees)
            test_grad = test_grad.transpose(2, 1, 0)  # shape=(no. class, 1, no. trees)

            influence = np.zeros((train_grad.shape[1], train_grad.shape[0]), dtype=np.float32)
            for j in range(influence.shape[1]):  # per class
                influence[:, j] = np.dot(train_grad[j], test_grad[j].flatten())

        return influence

    # private
    def _compute_gradients(self, X, y):
        """
        Compute gradients for all train instances across all trees (boosting iterations).

        Return
            - Regression and binary: 2d array of shape=(X.shape[0], no. trees).
            - Multiclass: 2d array of shape=(no. trees, X.shape[0], no. classes).
        """
        n_train = y.shape[0]

        # get initial raw value biased guess
        if self.model_.task_ == 'regression':
            raw_val = np.tile(self.model_.bias, y.shape[0])  # shape=(no. train,)

        elif self.model_.task_ == 'binary':
            raw_val = np.tile(self.model_.bias, y.shape[0])  # shape=(no. train,)

        elif self.model_.task_ == 'multiclass':
            y = self.label_binarizer_.transform(y)  # shape=(no. train, no. class)
            raw_val = np.tile(self.model_.bias, (y.shape[0], 1))  # shape=(no. train, no. class)

        # compute residuals for initial guess
        yhat = self._raw_val_to_prediction(raw_val, iteration=0)
        e = self._negative_gradient(y, yhat) / n_train  # marginal contribution

        # compute residuals for all subsequent trees
        residuals = [e]
        for i in range(self.model_.trees.shape[0]):
            raw_val += self._predict_iteration(X, iteration=i)
            yhat = self._raw_val_to_prediction(raw_val, iteration=i)
            e = self._negative_gradient(y, yhat) / n_train
            residuals.append(e)

        # shape output
        if self.model_.task_ in ['regression', 'binary']:
            gradients = np.hstack([e.reshape(-1, 1) for e in residuals])  # shape=(X.shape[0], no. trees)

        elif self.model_.task_ == 'multiclass':
            gradients = np.dstack(residuals).transpose(2, 0, 1)  # shape=(no. trees, X.shape[0], no. classes)

        return gradients

    def _negative_gradient(self, y, yhat):
        """
        Compute half of the negative gradient for least squares.

        Output
            regression and binary: shape=(y.shape[0])
            multiclass: shape=(y.shape[0], no. class)
        """
        return y - yhat

    def _predict_iteration(self, X, iteration):
        """
        Get raw leaf values for the specified iteration.

        Output
            regression and binary: shape=(X.shape[0])
            multiclass: shape=(X.shape[0], no. class)
        """
        if self.model_.task_ in ['regression', 'binary']:
            pred = self.model_.trees[iteration].predict(X)

        elif self.model_.task_ == 'multiclass':
            pred = np.zeros((X.shape[0], self.model_.trees.shape[1]), dtype=np.float32)

            for j in range(self.model_.trees.shape[1]):  # per class
                pred[:, j] = self.model_.trees[iteration, j].predict(X)

        return pred

    def _raw_val_to_prediction(self, raw_val, iteration):
        """
        Convert raw leaf values to predictions of the appropriate type.
        """
        if self.model_.tree_type == 'rf':
            yhat = raw_val / (iteration + 1)

        else:
            assert self.model_.tree_type == 'gbdt'

            if self.model_.task_ == 'regression':
                yhat = raw_val  # shape=(raw_val.shape[0],)

            elif self.model_.task_ == 'binary':
                yhat = util.sigmoid(raw_val)  # shape=(raw_val.shape[0],)

            elif self.model_.task_ == 'multiclass':
                yhat = util.softmax(raw_val)  # softmax along axis=1, shape=(raw_val.shape[0], no. class)

        return yhat
