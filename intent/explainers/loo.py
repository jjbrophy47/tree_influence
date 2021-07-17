import time
import joblib

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

    Local-Influence Semantics
        - Inf.(x_i, x_t) := L(y_t, f_{w/o x_i}(x_t)) - L(y_t, f(x_t))
        - Pos. value means removing x_i increases loss (adding x_i decreases loss, helpful).
        - Neg. value means removing x_i decreases loss (adding x_i increases loss, harmful).

    Note
        - Supports both GBDTs and RFs.
        - Supports parallelization.
    """
    def __init__(self, global_op='self', n_jobs=1, logger=None):
        """
        Input
            global_op: str, Type of global influence to provide.
                'alpha': Use the learned train weights as the global importance measure.
                'global': Compute effect of each train example on the test set loss.
                'self': Compute effect of each train example on itself.
            n_jobs: int, No. processes to run in parallel.
                -1 means use the no. of available CPU cores.
            logger: object, If not None, output to logger.
        """
        self.global_op = global_op
        self.n_jobs = n_jobs
        self.logger = logger

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
        self.loss_fn_ = util.get_loss_fn(self.model_.objective, self.model_.n_class_, self.model_.factor)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # select no. processes to run in parallel
        if self.n_jobs == -1:
            n_jobs = joblib.cpu_count()

        else:
            assert self.n_jobs >= 1
            n_jobs = min(self.n_jobs, joblib.cpu_count())

        # result container
        models = np.zeros(0, dtype=np.object)

        # trackers
        fits_completed = 0
        fits_remaining = X.shape[0]

        start = time.time()
        if self.logger:
            self.logger.info('\n[INFO] computing LOO values...')
            self.logger.info(f'[INFO] no. cpus: {n_jobs:,}...')

        # fit each model in parallel
        with joblib.Parallel(n_jobs=n_jobs) as parallel:

            # get number of fits to perform for this iteration
            while fits_remaining > 0:
                n = min(100, fits_remaining)

                results = parallel(joblib.delayed(_fit_LOO_model)
                                                 (model, X, y, train_idx) for train_idx in range(fits_completed,
                                                                                                 fits_completed + n))
                fits_completed += n
                fits_remaining -= n

                # synchronization barrier
                models = np.concatenate([models, np.array(results, dtype=np.object)])

                if self.logger:
                    cum_time = time.time() - start
                    self.logger.info(f'[INFO] fits: {fits_completed:,} / {X.shape[0]:,}, cum. time: {cum_time:.3f}s')

        self.models_ = models
        self.original_model_ = model

        return self

    def get_global_influence(self, X=None, y=None):
        """
        - Provides a global importance to all training examples.

        Return
            - 1d array of shape=(no. train,).
                * Arrays are returned in the same order as the traing data.
        """
        influence = np.zeros(self.X_train_.shape[0], dtype=np.float32)  # shape=(no. train,)

        # compute influence of each train example on the test set loss
        if self.global_op == 'global':
            assert X is not None and y is not None
            X, y = util.check_data(X, y, objective=self.model_.objective)

            original_loss = self._get_losses(self.original_model_, X, y, batch=True)  # single float

            for train_idx in range(self.models_.shape[0]):
                loss = self._get_losses(self.models_[train_idx], X, y, batch=True)  # single float
                influence[train_idx] = loss - original_loss  # single float

        # compute influence of each train example on itself
        else:
            assert self.global_op == 'self'
            X, y = self.X_train_, self.y_train_

            original_losses = self._get_losses(self.original_model_, X, y)  # shape=(X.shape[0],)

            for train_idx in range(self.models_.shape[0]):
                losses = self._get_losses(self.models_[train_idx], X[[train_idx]], y[[train_idx]])  # shape=(1,)
                influence[train_idx] = losses[0] - original_losses[train_idx]  # shape=(X.shape[0],)

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

    def _get_losses(self, model, X, y, batch=False):
        """
        Return
            - 1d array of individual losses of shape=(X.shape[0],),
                unless batch=True, then return a single float.
        """
        if self.model_.objective == 'regression':
            y_pred = model.predict(X)  # shape=(X.shape[0])

        elif self.model_.objective == 'binary':
            y_pred = model.predict_proba(X)[:, 1]  # 1d arry of pos. probabilities

        else:
            assert self.model_.objective == 'multiclass'
            y_pred = model.predict_proba(X)  # shape=(X.shape[0], no. class)

        result = self.loss_fn_(y, y_pred, raw=False, batch=batch)

        return result


def _fit_LOO_model(model, X, y, train_idx):
    """
    Fit model after leaving out the specified `train_idx` train example.

    Note
        - Parallelizable method.
    """
    new_X = np.delete(X, train_idx, axis=0)
    new_y = np.delete(y, train_idx)
    return clone(model).fit(new_X, new_y)
