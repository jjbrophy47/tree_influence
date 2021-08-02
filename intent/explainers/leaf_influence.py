import time

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from .base import Explainer
from .parsers import util


class LeafInfluence(Explainer):
    """
    LeafInfluence: Explainer that adapts the influence functions method to tree ensembles.

    Global-Influence Semantics
        - Influence of x_i on itself.
        - Original Inf.(x_i, x_i) := L(y, F(x_i)) - L(y, F_{w/o x_i}(x_i))
        - Updated Inf.(x_i, x_i) := L(y, F_{w/o x_i}(x_i)) - L(y, F(x_i))
            * Pos. value means removing x_i increases the loss (i.e. adding x_i decreases loss) (helpful).
            * Neg. value means removing x_i decreases the loss (i.e. adding x_i increases loss) (harmful).

    Local-Influence Semantics
        - Original Inf.(x_i, x_t) := L(y, F(x_t)) - L(y, F_{w/o x_i}(x_t))
        - Updated Inf.(x_i, x_t) := L(y, F_{w/o x_i}(x_t)) - L(y, F(x_t))
            * Pos. value means removing x_i increases the loss (i.e. adding x_i decreases loss) (helpful).
            * Neg. value means removing x_i decreases the loss (i.e. adding x_i increases loss) (harmful).

    Note
        - Wrapper for LeafInfluenceGBDT and LeafInfluenceRF.
        - For GBDT, influence values are multipled by -1 to get the "updated" influence semantics
            above; this is to make the semantics of LeafInfluence values more consistent
            with other influence methods that approximate changes in loss.
        - Does NOT take class or instance weight into account.

    Reference
        - https://github.com/bsharchilev/influence_boosting/blob/master/influence_boosting/influence/leaf_influence.py

    Paper
        - https://arxiv.org/abs/1802.06640

    Note
        - Supports both GBDTs and RFs.
    """
    def __init__(self, update_set=-1, atol=1e-5, logger=None):
        """
        Input
            update_set (GBDT only): int, No. neighboring leaf values to use for approximating leaf influence.
                0: Use no other leaves, influence is computed independent of other trees.
                -1: Use all other trees, most accurate but also most computationally expensive.
                1+: Trade-off between accuracy and computational resources.
            atol: float, Tolerance between actual and predicted leaf values.
            logger: object, If not None, output to logger.
        """
        self.update_set = update_set
        self.atol = atol
        self.logger = logger

    def fit(self, model, X, y):
        """
        Call the appropriate fit method.
        """

        if 'RandomForest' in str(model):
            explainer = LeafInfluenceRF(logger=self.logger)

        else:
            explainer = LeafInfluenceGBDT(update_set=self.update_set, atol=self.atol, logger=self.logger)

        explainer.fit(model, X, y)

        self.explainer_ = explainer

        return self

    def get_global_influence(self, X=None, y=None):
        """
        Input
            X: 2d array of test data.
            y: 2d array of test targets.

        Return
            - 1d array of global influence values of shape=(no. train,).
        """
        return self.explainer_.get_global_influence()

    def get_local_influence(self, X, y):
        """
        Input
            X: 2d array of test data.
            y: 2d array of test targets.

        Return
            - 2d array of local influence values of shape=(no. train, X.shape[0]).
        """
        return self.explainer_.get_local_influence(X, y)

    @property
    def model_(self):
        return self.explainer_.model_

    @property
    def parse_time_(self):
        return self.explainer_.parse_time_


class LeafInfluenceGBDT(Explainer):
    """
    LeafInfluence method designed specifically for GBDTs.
    """
    def __init__(self, update_set=-1, atol=1e-5, logger=None):
        """
        Input
            update_set: int, No. neighboring leaf values to use for approximating leaf influence.
                0: Use no other leaves, influence is computed independent of other trees.
                -1: Use all other trees, most accurate but also most computationally expensive.
                1+: Trade-off between accuracy and computational resources.
            atol: float, Tolerance between actual and predicted leaf values.
            logger: object, If not None, output to logger.
        """
        assert update_set >= -1
        self.update_set = update_set
        self.atol = atol
        self.logger = logger

    def fit(self, model, X, y):
        """
        - Compute leaf values using Newton leaf estimation method;
            make sure these match existing leaf values. Put into a 1d array.

        - Copy leaf values and compute new 1d array of leaf values across all trees,
            one new array resulting from removing each training example x in X.

        - Should end up with a 2d array of shape=(no. train, no. leaves across all trees).
            A bit memory intensive depending on the no. leaves, but should speed up the
            explanation for ANY set of test examples. This array can also be saved to
            disk to avoid recomputing these influence values.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.loss_fn_ = util.get_loss_fn(self.model_.objective, self.model_.n_class_, self.model_.factor)

        # extract tree-ensemble metadata
        trees = self.model_.trees
        n_boost = self.model_.n_boost_
        n_class = self.model_.n_class_
        learning_rate = self.model_.learning_rate
        l2_leaf_reg = self.model_.l2_leaf_reg
        bias = self.model_.bias

        # get no. leaves for each tree
        leaf_counts = self.model_.get_leaf_counts()  # shape=(no. boost, no. class)

        # intermediate containers
        current_approx = np.tile(bias, (X.shape[0], 1)).astype(util.dtype_t)  # shape=(X.shape[0], no. class)
        leaf2docs = []  # list of leaf_idx -> doc_ids dicts
        n_prev_leaves = 0

        # result containers
        naive_gradient_addendum = np.zeros((X.shape[0], n_boost, n_class), dtype=util.dtype_t)
        da_vector_multiplier = np.zeros((X.shape[0], n_boost, n_class), dtype=util.dtype_t)
        denominator = np.zeros(np.sum(leaf_counts), dtype=util.dtype_t)  # shape=(total no. leaves,)
        leaf_values = np.zeros(np.sum(leaf_counts), dtype=util.dtype_t)  # shape=(total no. leaves,)
        n_not_close = 0
        max_diff = 0

        # save gradient information of leaf values for each tree
        for boost_idx in range(n_boost):
            doc_preds = np.zeros((X.shape[0], n_class), dtype=util.dtype_t)

            # precompute gradient statistics
            gradient = self.loss_fn_.gradient(y, current_approx)  # shape=(X.shape[0], no. class)
            hessian = self.loss_fn_.hessian(y, current_approx)  # shape=(X.shape[0], no. class)
            third = self.loss_fn_.third(y, current_approx)  # shape=(X.shape[0], no. class)

            naive_gradient_addendum[:, boost_idx, :] = hessian * doc_preds / learning_rate + gradient
            da_vector_multiplier[:, boost_idx, :] = doc_preds / learning_rate * third + hessian

            for class_idx in range(n_class):

                # get leaf values
                leaf_count = leaf_counts[boost_idx, class_idx]
                leaf_vals = trees[boost_idx, class_idx].get_leaf_values()
                doc2leaf = trees[boost_idx, class_idx].apply(X)
                leaf2doc = {}

                # update predictions for this class
                doc_preds[:, class_idx] = leaf_vals[doc2leaf]

                # sanity check to make sure leaf values are correctly computed
                # also need to save some statistics to update leaf values later
                for leaf_idx in range(leaf_count):
                    doc_ids = np.where(doc2leaf == leaf_idx)[0]
                    leaf2doc[leaf_idx] = set(doc_ids)

                    # compute leaf values using gradients and hessians
                    leaf_enumerator = np.sum(gradient[doc_ids, class_idx])
                    leaf_denominator = np.sum(hessian[doc_ids, class_idx]) + l2_leaf_reg
                    leaf_prediction = -leaf_enumerator / leaf_denominator * learning_rate

                    # compare leaf values to actual leaf values
                    if not np.isclose(leaf_prediction, leaf_vals[leaf_idx], atol=1e-5):
                        n_not_close += 1
                        max_diff = max(max_diff, abs(leaf_prediction - leaf_vals[leaf_idx]))

                    # store statistics
                    denominator[n_prev_leaves + leaf_idx] = leaf_denominator
                    leaf_values[n_prev_leaves + leaf_idx] = leaf_prediction

                n_prev_leaves += leaf_count  # move to next set of tree leaves
                leaf2docs.append(leaf2doc)  # list of dicts, one per tree

            # n_prev_trees += n_class
            current_approx += doc_preds  # update approximation

        # result container
        leaf_derivatives = np.zeros((X.shape[0], np.sum(leaf_counts)), dtype=util.dtype_t)

        # copy and compute new leaf values resulting from the removal of each x in X.
        start = time.time()
        if self.logger:
            self.logger.info(f'\n[INFO] no. leaf vals not within {self.atol} tol.: {n_not_close:,}, '
                             f'max. diff.: {max_diff:.5f}')
            self.logger.info(f'\n[INFO] computing alternate leaf values...')

        # check predicted leaf values do not differ too much from actual model
        if max_diff > self.atol:
            raise ValueError(f'{max_diff:.5f} (max. diff.) > {self.atol} (tolerance)')

        for remove_idx in range(X.shape[0]):

            # display progress
            if self.logger and (remove_idx + 1) % 100 == 0:
                cum_time = time.time() - start
                self.logger.info(f'[INFO] {remove_idx + 1:,} / {X.shape[0]:,}: cum. time: {cum_time:.3f}s')

            # intermediate containers
            da = np.zeros((X.shape[0], n_class), dtype=util.dtype_t)
            tree_idx = 0
            n_prev_leaves = 0

            for boost_idx in range(n_boost):

                for class_idx in range(n_class):

                    leaf_count = leaf_counts[boost_idx, class_idx]
                    update_docs = self._get_docs_to_update(leaf_count, leaf2docs[tree_idx], remove_idx, da)

                    for leaf_idx in range(leaf_count):

                        # get intersection of leaf documents and update documents
                        leaf_docs = leaf2docs[tree_idx][leaf_idx]
                        update_leaf_docs = sorted(update_docs.intersection(leaf_docs))

                        # compute and save leaf derivative
                        grad_enumerator = np.dot(da[update_leaf_docs, class_idx],
                                                 da_vector_multiplier[update_leaf_docs, boost_idx, class_idx])

                        if remove_idx in update_leaf_docs:
                            grad_enumerator += naive_gradient_addendum[remove_idx, boost_idx, class_idx]

                        leaf_derivative = -grad_enumerator / denominator[n_prev_leaves + leaf_idx] * learning_rate

                        # update da
                        da[update_leaf_docs, class_idx] += leaf_derivative

                        # save
                        leaf_derivatives[remove_idx, n_prev_leaves + leaf_idx] = leaf_derivative

                    n_prev_leaves += leaf_count
                    tree_idx += 1

        # save results of this method
        self.leaf_values_ = leaf_values  # shape=(total no. leaves,)
        self.leaf_derivatives_ = leaf_derivatives  # shape=(no. train, total no. leaves)
        self.leaf_counts_ = leaf_counts  # shape=(no. boost, no. class)
        self.bias_ = bias
        self.n_boost_ = n_boost
        self.n_class_ = n_class

        return self

    def get_global_influence(self):
        """
        - Compute change in loss of each training instance on itself.
        - Provides a global importance to all training examples.

        Return
            - 1d array of shape=(no. train,).
                * Array is returned in the same order as the traing data.
        """
        influence = np.zeros((self.X_train_.shape[0], 1, self.n_class_), dtype=util.dtype_t)

        # compute influence of each training example on itself
        for remove_idx in range(self.X_train_.shape[0]):
            X = self.X_train_[[remove_idx]]
            y = self.y_train_[[remove_idx]]
            influence[remove_idx] = self._loss_derivative(X, y, remove_idx)

        # reshape result
        influence = influence.squeeze(axis=1).sum(axis=1)  # remove axis 1, then sum over classes

        return influence

    def get_local_influence(self, X, y):
        """
        - Compute influence of each training example on each test example loss.

        Return
            - 2d array of shape=(no. train, X.shape[0])
                * Train influences are in the same order as the original training order.
        """
        X, y = util.check_data(X, y, objective=self.model_.objective)

        influence = np.zeros((self.X_train_.shape[0], X.shape[0], self.n_class_), dtype=util.dtype_t)

        if self.logger:
            self.logger.info('\n[INFO] computing influence for each test example...')

        # compute influence of each training example on the test example
        for remove_idx in range(self.X_train_.shape[0]):
            influence[remove_idx] = self._loss_derivative(X, y, remove_idx)

        # reshape result
        influence = influence.sum(axis=2)  # sum over class, shape=(no. train, X.shape[0])

        return influence

    # private
    def _loss_derivative(self, X, y, remove_idx):
        """
        Compute the influence on the set of examples (X, y) using the updated
            set of leaf values from removing `remove_idx`.

        Input
            X: 2d array of test examples
            y: 1d array of test targets.
            remove_idx: index of removed train instance

        Return
            - Array of test influences of shape=(X.shape[0], no. class).

        Note
            - We multiply the result by -1 to have consistent semantics
                with other influence methods that approx. loss.
        """
        doc2leaf = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)

        og_pred = np.tile(self.bias_, (X.shape[0], 1)).astype(util.dtype_t)  # shape=(X.shape[0], no. class)
        new_pred = np.zeros((X.shape[0], self.n_class_), dtype=util.dtype_t)  # shape=(X.shape[0], no. class)

        # get prediction of each test example using the original and new leaf values
        tree_idx = 0
        n_prev_leaves = 0

        for boost_idx in range(self.n_boost_):  # per boosting iteration
            for class_idx in range(self.n_class_):  # per class

                for test_idx in range(X.shape[0]):  # per test example
                    leaf_idx = doc2leaf[test_idx][boost_idx][class_idx]
                    og_pred[test_idx, class_idx] += self.leaf_values_[n_prev_leaves + leaf_idx]
                    new_pred[test_idx, class_idx] += self.leaf_derivatives_[remove_idx][n_prev_leaves + leaf_idx]

                n_prev_leaves += self.leaf_counts_[boost_idx, class_idx]
            tree_idx += 1

        return -self.loss_fn_.gradient(y, og_pred) * new_pred

    def _get_docs_to_update(self, leaf_count, leaf_docs, remove_idx, da):
        """
        Return a set of document indices to be udpated for this tree.
        """

        # update only the remove example
        if self.update_set == 0:
            result = set({remove_idx})

        # update all train
        elif self.update_set == -1:
            result = set(np.arange(da.shape[0], dtype=np.int32))  # shape=(no. train,)

        # update examples for the top leaves
        else:

            # sort leaf indices based on largest abs. da sum
            leaf_das = [np.sum(np.abs(da[list(leaf_docs[leaf_idx])])) for leaf_idx in range(leaf_count)]
            top_leaf_ids = np.argsort(leaf_das)[-self.update_set:]
            
            # return remove_idx + document indices for the top `k` leaves
            result = {remove_idx}
            for leaf_idx in top_leaf_ids:
                result |= leaf_docs[leaf_idx]

        return result


class LeafInfluenceRF(Explainer):
    """
    LeafInfluence method designed specifically for RFs.
    """
    def __init__(self, logger=None):
        """
        Input
            logger: object, If not None, output to logger.
        """
        self.logger = logger

    def fit(self, model, X, y):
        """
        - Copy leaf values, and compute new array of leaf values, one
            array per removed training example.

        Input
            model: tree-ensemble model.
            X: 2d array of train data.
            y: 1d array of train targets.

        Return
            - 2d array of leaf values of shape=(no. train, no. leaves across all trees).
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.loss_fn_ = util.get_loss_fn(self.model_.objective, self.model_.n_class_, self.model_.factor)

        # extract tree-ensemble metadata
        trees = self.model_.trees
        n_boost = self.model_.n_boost_
        n_class = self.model_.n_class_
        leaf_counts = self.model_.get_leaf_counts()  # shape=(no. boost, no. class)
        leaf_vals = self.model_.get_leaf_values()  # shape=(total no. leaves,)

        # reshape targets
        if self.model_.objective in ['regression', 'binary']:
            y = y.reshape(-1, 1)  # shape=(no. train, 1)

        else:
            assert self.model_.objective == 'multiclass'
            y = LabelBinarizer().fit_transform(y)  # shape=(no. train, no. class)

        # result container
        new_leaf_values = np.tile(leaf_vals, (X.shape[0], 1))  # shape=(X.shape[0], total no. leaves)

        # update leaf values for leaves affected by each removed train example
        start = time.time()
        if self.logger:
            self.logger.info(f'\n[INFO] computing alternate leaf values...')

        for remove_idx in range(X.shape[0]):

            # display progress
            if self.logger and (remove_idx + 1) % 100 == 0:
                cum_time = time.time() - start
                self.logger.info(f'[INFO] {remove_idx + 1:,} / {X.shape[0]:,}: cum. time: {cum_time:.3f}s')

            n_prev_leaves = 0

            for boost_idx in range(n_boost):

                og_leaf_val = np.zeros(n_class, dtype=util.dtype_t)  # shape=(no. class,)
                new_leaf_val = np.zeros(n_class, dtype=util.dtype_t)  # shape=(no. class,)
                leaf_ids = np.zeros(n_class, dtype=np.int32)  # shape=(no. class,)

                og_boost_count = 0
                new_boost_count = 0

                # update leaf metadata
                for class_idx in range(n_class):

                    # get leaf affected by the removed train example
                    doc2leaf = trees[boost_idx, class_idx].apply(X)  # shape=(X.shape[0],)
                    leaf_idx = doc2leaf[remove_idx]
                    leaf_docs = np.where(doc2leaf == leaf_idx)[0]
                    leaf_ids[class_idx] = leaf_idx

                    og_leaf_val[class_idx] = y[leaf_docs, class_idx].sum()
                    new_leaf_val[class_idx] = y[leaf_docs, class_idx].sum() - y[remove_idx, class_idx]

                    # compute normalization constant
                    if self.model_.objective in ['regression', 'binary']:
                        og_boost_count += len(leaf_docs)
                        new_boost_count += len(leaf_docs) - 1

                    else:
                        assert self.model_.objective == 'multiclass'
                        og_boost_count += og_leaf_val[class_idx]
                        new_boost_count += new_leaf_val[class_idx]

                # leaf is now empty
                if new_boost_count == 0:

                    if self.model_.objective == 'regression':
                        new_leaf_val[0] = 0

                    elif self.model_.objective == 'binary':  # uniform
                        new_leaf_val[0] = 0.5

                    else:
                        assert self.model_.objective == 'multiclass'
                        new_leaf_val = np.full(n_class, 1.0 / n_class, dtype=util.dtype_t)

                # recompute the new leaf value
                else:
                    og_leaf_val = og_leaf_val / og_boost_count
                    new_leaf_val = new_leaf_val / new_boost_count

                # update leaf value and perform sanity check
                for class_idx in range(n_class):
                    leaf_idx = leaf_ids[class_idx]

                    assert np.isclose(leaf_vals[n_prev_leaves + leaf_idx], og_leaf_val[class_idx], atol=1e-5)
                    new_leaf_values[remove_idx, n_prev_leaves + leaf_idx] = new_leaf_val[class_idx]

                    n_prev_leaves += leaf_counts[boost_idx, class_idx]

        self.leaf_values_ = leaf_vals
        self.new_leaf_values_ = new_leaf_values
        self.leaf_counts_ = leaf_counts
        self.bias_ = self.model_.bias
        self.n_boost_ = n_boost
        self.n_class_ = n_class

        return self

    def get_global_influence(self):
        """
        - Compute change in loss of each training instance on itself.
        - Provides a global importance to all training examples.

        Return
            - 1d array of shape=(no. train,).
                * Array is returned in the same order as the traing data.
        """
        influence = np.zeros(self.X_train_.shape[0], dtype=util.dtype_t)

        # compute influence of each training example on itself
        for remove_idx in range(self.X_train_.shape[0]):
            X = self.X_train_[[remove_idx]]
            y = self.y_train_[[remove_idx]]
            influence[remove_idx] = self._loss_difference(X, y, remove_idx)

        return influence

    def get_local_influence(self, X, y):
        """
        - Compute influence of each training example on each test example loss.

        Return
            - 2d array of shape=(no. train, X.shape[0])
                * Train influences are in the same order as the original training order.
        """
        X, y = util.check_data(X, y, objective=self.model_.objective)

        influence = np.zeros((self.X_train_.shape[0], X.shape[0]), dtype=util.dtype_t)

        # compute influence of each training example on the test example
        for remove_idx in range(self.X_train_.shape[0]):
            influence[remove_idx] = self._loss_difference(X, y, remove_idx)

        return influence

    # private
    def _loss_difference(self, X, y, remove_idx):
        """
        Compute the influence on the set of examples (X, y) using the updated
            set of leaf values from removing `remove_idx`.

        Input
            X: 2d array of examples.
            y: 1d array of targets.
            remove_idx: index of removed train instance

        Return
            - Array of influences of shape=(X.shape[0], no. class).
        """
        doc2leaf = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)

        og_pred = np.tile(self.bias_, (X.shape[0], 1)).astype(util.dtype_t)  # shape=(X.shape[0], no. class)
        new_pred = np.tile(self.bias_, (X.shape[0], 1)).astype(util.dtype_t)  # shape=(X.shape[0], no. class)

        # get prediction of each test example using the original and new leaf values
        n_prev_leaves = 0

        for boost_idx in range(self.n_boost_):  # per boosting iteration
            for class_idx in range(self.n_class_):  # per class

                for test_idx in range(X.shape[0]):  # per test example
                    leaf_idx = doc2leaf[test_idx, boost_idx, class_idx]
                    og_pred[test_idx, class_idx] += self.leaf_values_[n_prev_leaves + leaf_idx]
                    new_pred[test_idx, class_idx] += self.new_leaf_values_[remove_idx, n_prev_leaves + leaf_idx]

                n_prev_leaves += self.leaf_counts_[boost_idx, class_idx]

        og_pred /= self.n_boost_
        new_pred /= self.n_boost_

        # compute influence, shape=(X.shape[0],)
        influence = self.loss_fn_(y, new_pred, raw=False) - self.loss_fn_(y, og_pred, raw=False)

        return influence
