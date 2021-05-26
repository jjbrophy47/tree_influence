import numpy as np
from sklearn.preprocessing import LabelBinarizer

from .base import Explainer
from .parsers import util


class TracIn(Explainer):
    """
    Abstract TracIn explainer that adapts the TracIn method to tree ensembles.

    Notes
        - For RFs, there is no initial guess, so each gradients layer is associated
            with a tree (or boosting iteration).
        - For GBDTs, there is an iniitial guess, so there is an extra gradient layer
            at the beginning; this layer can be included or exlcuded in both the global
            and local explanations by using the `initial_grad` argument.
        - Currently, we use error residuals to compute marginal contributions; however,
            we could also try using the tree output instead (approx. of the error residuals).
            There would be no initial guess gradient though, so this would require
            `initial_grad`='keep' for GBDTs.
        - Local explanations for GBDT grad=approx with initial_grad=skip is essentially
            the same as doing the dot product using the LeafOutput tree kernel.
    """
    def __init__(self, grad='residual', initial_grad='keep'):
        """
        Input
            grad
                'residual': Use error residuals when computing marginals.
                'approx': Use tree output (approx. of residuals) when computing marginals.
            initial_grad
                'keep': For GBDTs, include gradient from initial guess
                    when computing local explanation; does not affect self-influence.
                'skip': For GBDTs, exclude gradient from initial guess
                    when computing local explanation; does not affect self-influence.
        """
        assert grad in ['residual', 'approx']
        assert initial_grad in ['keep', 'skip']
        self.grad = grad
        self.initial_grad = initial_grad

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
        self._validate_settings()

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
            if self.initial_grad == 'skip':
                self_influence = self.gradients_[:, 1:].sum(axis=1)
            else:
                self_influence = self.gradients_.sum(axis=1)

        elif self.model_.task_ == 'multiclass':
            if self.initial_grad == 'skip':
                self_influence = self.gradients_[1:].sum(axis=0)
            else:
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

        test_gradients = self._compute_gradients(X, y)
        test_leaves = self.model_.apply(X)

        # only add influence if train and test end in the same leaf per tree
        mask = np.where(self.leaves_ == test_leaves, 1, 0)

        # train_gradients=(no. train, no. trees), test_gradient=(1, no. trees,)
        # train_leaf_indices=(no. train, no. trees), test_leaf_indices=(1, no. trees,)
        if self.model_.task_ in ['regression', 'binary']:
            train_grad = self.gradients_[:, 1:] * mask

            if self.initial_grad == 'skip':
                test_grad = test_gradients[:, 1:].flatten()

            else:  # keep first layer, mutiply mask by subsequent layers
                initial_grad = self.gradients_[:, 0].reshape(-1, 1)
                train_grad = np.hstack([initial_grad, train_grad])
                test_grad = test_gradients.flatten()

            influence = np.dot(train_grad, test_grad)

        # train_gradients=(no. trees, no. train, no. class), test_gradient=(no. trees, 1, no. class)
        # train_leaf_indices=(no. trees, no. train, no. class), test_leaf_indices=(no. trees, 1, no. class)
        elif self.model_.task_ == 'multiclass':
            train_gradients = self.gradients_.transpose(2, 1, 0)  # shape=(no. class, no. train, no. trees)
            test_gradients = test_gradients.transpose(2, 1, 0)  # shape=(no. class, 1, no. trees)
            mask = mask.transpose(2, 1, 0)  # shape=(no. class, no. train, no. trees)

            influence = np.zeros((train_gradients.shape[1], train_gradients.shape[0]), dtype=np.float32)

            for j in range(influence.shape[1]):  # per class
                train_grad = train_gradients[j][:, 1:] * mask[j]

                if self.initial_grad == 'skip':
                    test_grad = test_gradients[j][:, 1:].flatten()

                else:  # keep first layer, mutiply mask by subsequent layers
                    initial_grad = train_gradients[j][:, 0].reshape(-1, 1)
                    train_grad = np.hstack([initial_grad, train_grad])
                    test_grad = test_gradients[j].flatten()

                influence[:, j] = np.dot(train_grad, test_grad)

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

        # compute gradient for initial guess
        yhat = self._raw_val_to_prediction(raw_val, iteration=0)
        residual = self._negative_gradient(y, yhat) / n_train  # marginal contribution

        # compute gradient for all subsequent trees
        gradient = [residual]

        for i in range(self.model_.trees.shape[0]):
            residual_approx = self._predict_iteration(X, iteration=i)

            if self.grad == 'residual':
                raw_val += residual_approx
                yhat = self._raw_val_to_prediction(raw_val, iteration=i)
                residual = self._negative_gradient(y, yhat) / n_train
                gradient.append(residual)

            elif self.grad == 'approx':  # `self.initial_grad` must be 'skip'
                gradient.append(residual_approx)

        # shape output
        if self.model_.task_ in ['regression', 'binary']:
            gradient = np.hstack([e.reshape(-1, 1) for e in gradient])  # shape=(X.shape[0], no. trees)

        elif self.model_.task_ == 'multiclass':
            gradient = np.dstack(gradient).transpose(2, 0, 1)  # shape=(no. trees, X.shape[0], no. classes)

        return gradient

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

    def _validate_settings(self):
        """
        Make sure the model matches the approriate settings of the explainer.
        """
        if self.model_.tree_type == 'rf':
            assert self.grad == 'residual', 'RF only supports grad=residual'

        elif self.model_.tree_type == 'gbdt' and self.grad == 'approx':
            assert self.initial_grad == 'skip', 'GBDT grad=approx only supports initial_grad=skip'
