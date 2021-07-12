"""
This is a modified and simplified version of DShap
that only works for binary classification.
"""
import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


class DShap(object):

    def __init__(self,
                 model,
                 X_train,
                 X_val,
                 y_train,
                 y_val,
                 metric='acc',
                 truncation_frac=0.25,
                 tolerance=0.01,
                 algorithm='dshap',
                 random_state=1,
                 logger=None):
        """
        Args:
            model: Trained model.
            X_train: Train covariates.
            y_train: Train labels.
            X_val: Validation covariates.
            y_val: Validation labels.
            metric: Evaluation metric.
            tolerance: Tolerance used to truncate TMC-Shapley.
            random_state: Random seed.
        """
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.metric = metric
        self.truncation_frac = truncation_frac
        self.tolerance = tolerance
        self.logger = logger

        # derived attributes
        if algorithm == 'dshap':
            self.iteration_ = self._standard_iterationB
        elif algorithm == 'dshap_with_unlearning':
            self.iteration_ = self._unlearning_iteration
        else:
            raise ValueError('unknown algorithm {}'.format(algorithm))

        self.rng_ = np.random.default_rng(random_state)
        self.sources_ = np.arange(X_train.shape[0])
        self.random_score_ = self._init_score()
        self.mean_score_ = self._compute_mean_score()
        self.mem_tmc_ = np.zeros((0, self.sources_.shape[0]))

        if self.logger:
            self.logger.info('\nrandom score: {:3f}'.format(self.random_score_))
            self.logger.info('mean score: {:3f}'.format(self.mean_score_))

    def compute_marginals(self, check_every=10, err_tol=0.1):
        """
        Runs TMC-Shapley algorithm.

        Args:
            check_every: No. iterations to run before checking convergence
            err_tol: Convergence tolerance.

        Returns:
            Average marginal contributions of each training instance.
            A positive number means the training instance increased the score,
            and a negative number means the training instance decreased the score.
        """
        start = time.time()

        # result container
        marginals_sum = np.zeros(self.X_train.shape[0])

        # run TMC-Shapley until convergence
        iteration = 0
        while True:

            # run one iteration of the TMC algorithm
            marginals = self.iteration_()
            marginals_sum += marginals

            # history of marignals, shape=(no. iterations, no. samples)
            self.mem_tmc_ = np.vstack([self.mem_tmc_, marginals.reshape(1, -1)])

            # check if TMC-Shapley should finish early, only do this every so often
            if (iteration + 1) % check_every == 0:
                error = self._compute_error(self.mem_tmc_)
                elapsed = time.time() - start

                if self.logger:
                    self.logger.info('[Iter. {:,}], max. error: {:.3f}...{:.3f}s'.format(iteration + 1, error, elapsed))

                # check convergence
                if (iteration + 1) > 100 and error < err_tol:
                    break

            else:
                if self.logger:
                    elapsed = time.time() - start
                    self.logger.info('[Iter. {:,}]...{:.3f}s'.format(iteration + 1, elapsed))

            iteration += 1

        # compute average marginals
        marginals = marginals_sum / iteration

        return marginals

    # private
    def _init_score(self):
        """
        Gives the value of an initial untrained model.
        """
        result = None

        if self.metric == 'acc':
            n = len(self.y_val)
            n_pos = np.sum(self.y_val)
            n_neg = n - n_pos
            acc = n_pos / n if n_pos > n_neg else n_neg / n
            result = acc

        elif self.metric == 'auc':
            result = 0.5

        elif self.metric == 'ap':
            result = np.sum(self.y_val) / len(self.y_val)

        elif self.metric == 'proba':
            result = 0.5

        else:
            raise ValueError('unknown metric {}'.format(self.metric))

        return result

    def _compute_score(self, model, X, y):
        """
        Computes the values of the given model.

        Args:
            model: The model to be evaluated.
        """
        if self.metric == 'acc':
            model_pred = model.predict(X)
            result = accuracy_score(y, model_pred)

        elif self.metric == 'auc':
            model_proba = model.predict_proba(X)[:, 1]
            result = roc_auc_score(y, model_proba)

        elif self.metric == 'ap':
            model_proba = model.predict_proba(X)[:, 1]
            result = average_precision_score(y, model_proba)

        elif self.metric == 'proba':
            model_proba = model.predict_proba(X)
            if model_proba.shape[1] == 1:
                model_proba = model_proba[0]
            result = np.mean(model_proba)

        else:
            raise ValueError('Invalid metric!')

        return result

    def _compute_mean_score(self, num_iter=100):
        """
        Computes the average performance and its error using bagging.
        """
        scores = []
        for _ in range(num_iter):

            # select a subset of bootstrapped samples
            bag_idxs = np.random.choice(self.y_val.shape[0], size=self.y_val.shape[0], replace=True)

            # score this subset
            score = self._compute_score(self.model, X=self.X_val[bag_idxs], y=self.y_val[bag_idxs])
            scores.append(score)

        return np.mean(scores)

    def _compute_error(self, marginals, n_run=100):
        """
        Checks to see if the the marginals are converging.
        """

        # has not run long enough
        # if marginals.shape[0] < n_run:
        #     return 1.0

        # add up all marginals using axis=0, then divide by their iteration
        all_vals = (np.cumsum(marginals, axis=0) / np.arange(1, len(marginals) + 1).reshape(-1, 1))[-n_run:]

        # diff. between last `n_run` runs and last run, divide by last run, average over all points
        errors = np.mean(np.abs(all_vals[-n_run:] - all_vals[-1:]) / (np.abs(all_vals[-1:]) + 1e-12), axis=-1)

        # return max error from one of the points
        return np.max(errors)

    def _standard_iterationA(self):
        """
        Runs one iteration of TMC-Shapley algorithm.
        """
        start = time.time()

        # shuffle the indices of the data
        idxs = np.random.permutation(self.sources_)
        idxs = idxs[:int(len(idxs) * self.truncation_frac)].copy()

        # result container
        marginal_contribs = np.zeros(self.X_train.shape[0])

        # empty containers
        X_batch = np.zeros((0,) + tuple(self.X_train.shape[1:]))
        y_batch = np.zeros(0, int)

        # trackers
        new_score = self.random_score_

        # perform for each data point
        for i, idx in enumerate(idxs):
            old_score = new_score

            # add sample to sample batch
            X_batch = np.vstack([X_batch, self.X_train[idx].reshape(1, -1)])
            y_batch = np.concatenate([y_batch, self.y_train[idx].reshape(1)])

            # train and re-evaluate
            model = clone(self.model)
            model = model.fit(X_batch, y_batch)
            new_score = self._compute_score(model, X=self.X_val, y=self.y_val)

            # add normalized contributions
            marginal_contribs[idx] = (new_score - old_score)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                print('no. samples: {:,}, score: {:.3f}...{:.3f}s'.format(i, new_score, elapsed))

        return marginal_contribs

    def _standard_iterationB(self):
        """
        Runs one iteration of TMC-Shapley algorithm.
        """
        num_classes = len(set(self.y_val))

        # shuffle the indices of the data
        idxs = self.rng_.permutation(self.sources_)

        # result container
        marginal_contribs = np.zeros(self.X_train.shape[0])

        # empty containers
        X_batch = np.zeros((0,) + tuple(self.X_train.shape[1:]))
        y_batch = np.zeros(0, int)

        # trackers
        truncation_counter = 0
        new_score = self.random_score_

        # perform for each data point
        for n, idx in enumerate(idxs):
            old_score = new_score

            # add sample to sample batch
            X_batch = np.vstack([X_batch, self.X_train[idx].reshape(1, -1)])
            y_batch = np.concatenate([y_batch, self.y_train[idx].reshape(1)])

            if len(set(y_batch)) != num_classes:
                continue

            # train and re-evaluate
            model = clone(self.model)
            model = model.fit(X_batch, y_batch)
            new_score = self._compute_score(model, X=self.X_val, y=self.y_val)

            # add normalized contributions
            marginal_contribs[idx] = (new_score - old_score)

            # compute approximation quality
            distance_to_full_score = np.abs(new_score - self.mean_score_)
            # print(n, idx, X_batch.shape, new_score, old_score, marginal_contribs[idx], distance_to_full_score)
            if distance_to_full_score <= self.tolerance * self.mean_score_:
                truncation_counter += 1
                if truncation_counter > 5:
                    break

            # approximation is not converging, keep going
            else:
                truncation_counter = 0

        return marginal_contribs

    def _unlearning_iteration(self):
        """
        Runs one iteration of TMC-Shapley algorithm.
        """
        start = time.time()
        num_classes = len(set(self.y_val))

        # shuffle the indices of the data
        idxs = self.rng_.permutation(self.sources_)
        idxs = idxs[:int(len(idxs) * self.truncation_frac)].copy()

        # train model on truncated portion of the permutation
        X_batch = self.X_train[idxs].copy()
        y_batch = self.y_train[idxs].copy()
        model = clone(self.model).fit(X_batch, y_batch)

        new_score = self._compute_score(model, X=self.X_val, y=self.y_val)
        elapsed = time.time() - start

        # print('no. samples: {:,}, score: {:.3f}...{:.3f}s'.format(len(idxs), new_score, elapsed))

        # result container
        marginal_contribs = np.zeros(self.X_train.shape[0])

        # perform for each data point
        for i, idx in enumerate(idxs):
            old_score = new_score

            # does not contain samples of all classes
            if len(set(y_batch[:-i])) != num_classes:
                continue

            # remove effect of train instance and re-evaluate
            model.delete(i)
            new_score = self._compute_score(model, X=self.X_val, y=self.y_val)

            # add normalized contributions
            marginal_contribs[idx] = (new_score - old_score)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                if self.logger:
                    self.logger.info('no. samples: {:,}, score: {:.3f}...{:.3f}s'.format(len(idxs) - i, new_score, elapsed))

        return marginal_contribs
