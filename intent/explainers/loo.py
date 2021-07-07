import numpy as np
from sklearn.base import clone

from .base import Explainer
from .parsers import util


class LOO(Explainer):
    """
    Leave-one-out influence explainer. Retrains the model
    for each train example to get change in loss.

    Global-Influence Semantics
        - Influence of x_i on itself.
        - Inf.(x_i, x_i) := L(y_t, f_{w/o x_i}(x_i)) - L(y_t, f(x_i))
        - Pos. value means removing x_i increases loss (adding x_i decreases loss, helpful).
        - Neg. value means removing x_i decreases loss (adding x_i increases loss, harmful).
        - TODO: should it be the effect of x_i on the entire model's training score?

    Local-Influence Semantics
        - Inf.(x_i, x_t) := L(y_t, f_{w/o x_i}(x_t)) - L(y_t, f(x_t))
        - Pos. value means removing x_i increases loss (adding x_i decreases loss, helpful).
        - Neg. value means removing x_i decreases loss (adding x_i increases loss, harmful).

    Note
        - Supports both GBDTs and RFs.
    """
    def __init__(self, verbose=0):
        """
        Input
            verbose: int, Output verbosity.
        """
        self.verbose = verbose

    def fit(self, model, X, y):
        """
        - Fit one model with for each training example,
            with that training example removed.

        Note
            - Very memory intensive to save all models,
                may have to switch to a streaming approach.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        self.n_class_ = self.model_.n_class_
        self.loss_fn_ = self._get_loss_function()
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # result container
        models = np.zeros(X.shape[0], dtype=np.object)  # shape=(X.shape[0])

        # save fitted model for each training example
        for train_idx in range(X.shape[0]):
            new_X = np.delete(X, train_idx, axis=0)
            new_y = np.delete(y, train_idx)

            models[train_idx] = clone(model).fit(new_X, new_y)

        self.original_model_ = model
        self.models_ = models

        return self

    def get_global_influence(self):
        """
        - Compute influence of each training example on itself.
        - Provides a global perspective of which training intances
          are most important.

        Return
            - 1d array of shape=(no. train,).
            - Arrays are returned in the same order as the traing data.
        """
        X, y = self.X_train_, self.y_train_

        influence = np.zeros((X.shape[0]), dtype=np.float32)  # shape=(X.shape[0],)

        original_losses = self._get_losses(self.original_model_, X, y)  # shape=(X.shape[0],)

        for remove_idx in range(self.models_.shape[0]):
            losses = self._get_losses(self.models_[remove_idx], X[[remove_idx]], y[[remove_idx]])  # shape=(1,)
            influence[remove_idx] = losses[0] - original_losses[remove_idx]  # shape=(X.shape[0],)

        return influence

    def get_local_influence(self, X, y):
        """
        - Compute influence of each training instance on test prediction(s) or loss(es).

        Input
            - X: 2d array of test examples.
            - y: 1d array of test targets

        Return
            - 2d array of shape=(no. train, X.shape[0]).
            - Arrays are returned in the same order as the training data.
        """
        X, y = util.check_data(X, y, objective=self.model_.objective)

        influence = np.zeros((self.X_train_.shape[0], X.shape[0]), dtype=np.float32)

        original_losses = self._get_losses(self.original_model_, X, y)  # shape=(X.shape[0],)

        for remove_idx in range(self.models_.shape[0]):
            losses = self._get_losses(self.models_[remove_idx], X, y)
            influence[remove_idx, :] = losses - original_losses  # shape=(X.shape[0],)

        return influence

    def _get_losses(self, model, X, y):
        """
        Returns 1d array of individual losses of shape=(X.shape[0]).
        """
        if self.model_.objective == 'regression':
            y_pred = model.predict(X)  # shape=(X.shape[0])

        elif self.model_.objective == 'binary':
            y_pred = model.predict_proba(X)[:, 1]  # 1d arry of pos. probabilities

        else:
            assert self.model_.objective == 'multiclass'
            y_pred = model.predict_proba(X)  # shape=(X.shape[0], no. class)

        losses = self.loss_fn_(y, y_pred)
        return losses

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
