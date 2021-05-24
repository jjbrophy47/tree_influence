import numpy as np
from sklearn.preprocessing import LabelBinarizer

from .base import Explainer
from .parsers import parse_model
from .parsers import util


class TracIn(Explainer):
    """
    Abstract TracIn explainer that adapts the TracIn method to tree ensembles.
    """
    def fit(self, model, X, y):
        """
        Convert model to internal standardized tree structures.
        Perform any initialization necessary for the chosen method.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        self.model_ = parse_model(model)
        X = X.astype(np.float32)

        assert self.model_.task_ in ['regression', 'binary', 'multiclass']

        # convert y to the appropriate dtype
        if self.model_.task_ == 'regression':
            y = y.astype(np.float32)
            self.init_raw_val_ = np.tile(self.model_.bias, y.shape[0])  # shape=(no. train,)

        elif self.model_.task_ == 'binary':
            y = y.astype(np.int32)
            self.init_raw_val_ = np.tile(self.model_.bias, y.shape[0])  # shape=(no. train,)

        elif self.model_.task_ == 'multiclass':
            y = LabelBinarizer().fit_transform(y.astype(np.int32))  # shape=(no. train, no. class)
            self.init_raw_val_ = np.tile(self.model_.bias, (y.shape[0], 1))  # shape=(no. train, no. class)

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

    def explain(self, X, y=None):
        """
        - Compute influence of each training instance on the test instance loss (or prediction?)
          by computing the dot prod. between each train gradient and the test gradient.
        - Provides a local explanation of the test instance loss (or prediction?).

        Return
            - Regression and binary: 1d array of shape=(no. train).
            - Multiclass: 2d array of shape=(no. train, no. classes).
            - Array is returned in the same order as the traing data.
        """
        assert X.ndim == 2 and X.shape[0] == 1
        test_gradient = self._compute_gradients(X, y)
        test_leaves = self.model_.apply(X)

        # only add influence if train and test end in the same leaf per tree
        mask = np.where(self.leaves_ == test_leaves, 1, 0)
        train_gradients = mask * self.gradients_

        # train_gradients=(no. train, no. trees), test_gradient=(no. trees,)
        # train_leaf_indices=(no. train, no. trees), test_leaf_indices=(no. trees,)
        if self.model_.task_ in ['regression', 'binary']:
            influence = np.dot(train_gradients, test_gradient)

        # train_gradients=(no. trees, no. train, no. class), test_gradient=(no. trees, 1, no. class)
        # train_leaf_indices=(no. trees, no. train, no. class), test_leaf_indices=(no. trees, 1, no. class)
        elif self.model_.task == 'multiclass':
            g1 = train_gradients.transpose(1, 0, 2)  # shape=(no. train, no. trees, no. class)
            g1 = g1.reshape(g1.shape[0], g1.shape[1] * g1.shape[2])  # shape=(no. train, no. trees * no. class)
            g2 = test_gradient.transpose(1, 0, 2).flatten()  # shape=(no. trees * no. class,)
            influence = np.dot(g1, g2)

        return influence

    # private
    def _compute_gradients(self, X, y):
        """
        - Compute gradients for all train instances across all trees (boosting iterations).

        Return
            - Regression and binary: 2d array of shape=(no. train, no. trees).
            - Multiclass: 2d array of shape=(no. trees, no. train, no. classes).
        """
        n_train = y.shape[0]
        raw_val = self.init_raw_val_

        yhat = self._raw_val_to_prediction(raw_val)
        e = self._negative_gradient(y, yhat) / n_train  # marginal contribution

        residuals = [e]
        for i in range(self.model_.trees.shape[0]):
            raw_val += self._predict_iteration(X, i)
            yhat = self._raw_val_to_prediction(raw_val)
            e = self._negative_gradient(y, yhat) / n_train
            residuals.append(e)

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

    def _predict_iteration(self, X, i):
        """
        Override to handle multiclass predictions.

        Output
            regression and binary: shape=(X.shape[0])
            multiclass: shape=(X.shape[0], no. class)
        """
        if self.model_.task_ in ['regression', 'binary']:
            pred = self.model_.trees[i].predict(X)

        elif self.model_.task_ == 'multiclass':
            pred = np.zeros((X.shape[0], self.model_.trees.shape[1]), dtype=np.float32)

            for j in range(self.model_.trees.shape[1]):  # per class
                pred[:, j] = self.model_.trees[i, j].predict(X)

        return pred

    def _raw_val_to_prediction(self, raw_val):
        """
        Convert raw leaf values to predictions of the appropriate type.
        """
        if self.model_.task_ == 'regression':
            yhat = raw_val  # shape=(raw_val.shape[0])

        elif self.model_.task_ == 'binary':
            yhat = util.sigmoid(raw_val)  # shape=(raw_val.shape[0])

        elif self.model_.task_ == 'multiclass':
            yhat = util.softmax(raw_val)  # softmax along axis=1, shape=(raw_val.shape[0], no. class)

        return yhat
