import numpy as np

from .base import Explainer
from .parsers import util


class LeafInfluence(Explainer):
    """
    LeafInfluence: Explainer that adapts the
    influence functions method to tree ensembles.

    Semantics
        - Inf.(x_i) := L(y, F(x_t)) - L(y, F_{w/o x_i}(x_t))
        - A neg. number means removing x_i increases the loss (i.e. adding x_i decreases loss) (helpful).
        - A pos. number means removing x_i decreases the loss (i.e. adding x_i increases loss) (harmful).

    Note
        - Does NOT take class or instance weight into account.

    Reference
         https://github.com/bsharchilev/influence_boosting/blob/master/influence_boosting/influence/leaf_influence.py

    Paper
        TODO

    TODO: add RF support?
    """
    def __init__(self, update_set=0, random_state=1, verbose=0):
        """
        Input
            update_set: int, No. neighboring leaf values to use for approximating leaf influence.
                0: Use no other leaves, influence is computed independent of other trees.
                -1: Use all other trees, most accurate but also most computationally expensive.
                1+: Trade-off between accuracy and computational resources.
            l2_leaf_reg: float, Regularization coefficient to prevent leaf values from overfitting.
                First used in XGBoost paper. If set to 0, then learning reverts to traditiional
                gradient tree boosting.
            random_state: int, Random state seed to generate reproducible results.
            verbose: int, Output verbosity.
        """
        assert update_set >= -1
        self.update_set = update_set
        self.random_state = random_state
        self.verbose = verbose

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
        self.loss_fn_ = self._get_loss_function()

        # extract tree-ensemble metadata
        n_tree = self.model_.n_tree_
        n_class = self.model_.n_class_
        learning_rate = self.model_.learning_rate
        l2_leaf_reg = self.model_.l2_leaf_reg
        bias = self.model_.bias

        # get no. leaves for each tree
        leaf_counts = self.model_.get_leaf_counts()  # shape=(no. tree * no. class)

        # intermediate containers
        current_approx = np.tile(bias, (X.shape[0], 1)).astype(np.float32)  # shape=(X.shape[0], no. class)
        leaf2docs = []  # list of leaf_idx -> doc_ids dicts
        n_prev_leaves = 0

        # result containers
        naive_gradient_addendum = np.zeros((X.shape[0], n_tree, n_class), dtype=np.float32)
        da_vector_multiplier = np.zeros((X.shape[0], n_tree, n_class), dtype=np.float32)
        denominator = np.zeros(np.sum(leaf_counts), dtype=np.float32)  # shape=(total no. leaves,)
        leaf_values = np.zeros(np.sum(leaf_counts), dtype=np.float32)  # shape=(total no. leaves,)

        # save gradient information of leaf values for each tree
        for tree_idx, tree in enumerate(n_tree):
            doc_preds = np.zeros((X.shape[0], n_class), dtype=np.float32)

            # precompute gradient statistics
            gradient = self.loss_fn_.gradient(y, current_approx)  # shape=(X.shape[0], no. class)
            hessian = self.loss_fn_.hessian(y, current_approx)  # shape=(X.shape[0], no. class)
            third = self.loss_fn_.third(y, current_approx)  # shape=(X.shape[0], no. class)

            naive_gradient_addendum[:, tree_idx, :] = hessian * doc_preds / learning_rate + gradient
            da_vector_multiplier[:, tree_idx, :] = doc_preds / learning_rate * third + hessian

            for class_idx in range(n_class):

                # get leaf values
                leaf_count = leaf_counts[tree_idx + class_idx]
                leaf_vals = tree.get_leaf_values()
                doc2leaf = tree.apply(X)
                leaf2doc = {}

                # update predictions for this class
                doc_preds[:, class_idx] = leaf_vals[doc2leaf]

                # sanity check to make sure leaf values are correctly computed
                # also need to save some statistics to update leaf values later
                for leaf_idx in range(leaf_count):
                    doc_ids = np.where(doc2leaf == leaf_idx)[0]
                    leaf2doc[leaf_idx] = set(doc_ids)

                    # compute leaf values using gradients and hessians
                    leaf_enumerator = np.sum(gradient[doc_ids][class_idx])
                    leaf_denominator = np.sum(hessian[doc_ids][class_idx]) + l2_leaf_reg
                    leaf_prediction = -leaf_enumerator / leaf_denominator * learning_rate

                    # compare leaf values to actual leaf values
                    assert np.isclose(leaf_prediction, leaf_vals[leaf_idx], atol=1e-5)

                    # store statistics
                    denominator[n_prev_leaves + leaf_idx] = leaf_denominator
                    leaf_values[n_prev_leaves + leaf_idx] = leaf_prediction

                n_prev_leaves += leaf_count  # move to next set of tree leaves
                leaf2docs.append(leaf2doc)  # list of dicts, one per tree

            current_approx += doc_preds  # update approximation

        # result container
        leaf_derivatives = np.zeros((X.shape[0], np.sum(leaf_counts)), dtype=np.float32)

        # copy and compute new leaf values resulting from the removal of each x in X.
        for remove_idx in range(X.shape[0]):

            # intermediate containers
            da = np.zeros((X.shape[0], n_class), dtype=np.float32)
            n_prev_leaves = 0

            for tree_idx in range(n_tree):

                for class_idx in range(n_class):
                    update_docs = self._get_docs_to_update(tree_idx, class_idx, leaf_counts, leaf2docs, remove_idx, da)

                    for leaf_idx in range(leaf_counts[tree_idx + class_idx]):

                        # get intersection of leaf documents and update documents
                        leaf_docs = leaf2docs[tree_idx + class_idx][leaf_idx]
                        update_leaf_docs = sorted(update_docs.intersection(leaf_docs))

                        # compute and save leaf derivative
                        grad_enumerator = np.dot(da[update_leaf_docs][class_idx],
                                                 da_vector_multiplier[:, tree_idx, class_idx][update_leaf_docs])

                        if remove_idx in update_leaf_docs:
                            grad_enumerator += naive_gradient_addendum[:, tree_idx, class_idx][remove_idx]

                        leaf_derivative = -grad_enumerator / denominator[n_prev_leaves + leaf_idx] * learning_rate

                        # update da
                        da[update_leaf_docs][class_idx] += leaf_derivative

                        # save
                        leaf_derivatives[remove_idx][n_prev_leaves + leaf_idx] = leaf_derivative

                    n_prev_leaves += leaf_counts[tree_idx + class_idx]

        # save results of this method
        self.leaf_values_ = leaf_values  # shape=(total no. leaves across ALL trees)
        self.leaf_derivatives_ = leaf_derivatives  # shape=(no. train, total no. leaves)
        self.leaf_counts_ = leaf_counts  # shape=(no. trees * no. class,)
        self.bias_ = bias
        self.n_tree_ = n_tree
        self.n_class_ = n_class

        return self

    def get_self_influence(self):
        """
        - Compute influence of each training instance on itself.
        - Provides a global importance to all training examples.

        Return
            - 2d array of shape=(no. train, no. class).
            - Array is returned in the same order as the traing data.
        """
        self_influence = np.zeros((self.X_train_.shape[0], 1, self.n_class_), dtype=np.float32)

        # compute influence of each training example on itself
        for remove_idx in range(self.X_train_.shape[0]):
            X = self.X_train_[[remove_idx]]
            y = self.y_train_[[remove_idx]]
            self_influence[remove_idx] = self._loss_derivative(X, y, remove_idx)

        return self_influence.squeeze()  # remove axis=1

    def explain(self, X, y):
        """
        - Compute influence of each training example on the loss of the test examples.

        Return
            - 3d array of shape=(X.shape[0], no. train, no. class).
            - Multiclass: 3d array of shape=(no. train, X.shape[0], no. classes).
            - Each 2d array for each test example is in the same order as the traing data.
        """
        X, y = util.check_data(X, y, objective=self.model_.objective)

        influence = np.zeros((self.X_train_.shape[0], X.shape[0], self.n_class_), dtype=np.float32)

        # compute influence of each training example on the test example
        for remove_idx in range(self.X_train_.shape[0]):
            influence[remove_idx] = self._loss_derivative(X, y, remove_idx)

        return influence.transpose(1, 0, 2)  # shape=(no. test, no. train, no. class)

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
        """
        doc2leaf = self.model_.apply(X)  # shape=(X.shape[0], no. tree, no. class)

        og_pred = np.tile(self.bias_, (X.shape[0], 1)).astype(np.float32)  # shape=(X.shape[0], no. class)
        new_pred = np.zeros((X.shape[0], self.n_class_), dtype=np.float32)  # shape=(X.shape[0], no. class)

        # get prediction of each test example using the original and new leaf values
        n_prev_leaves = 0
        for tree_idx in range(self.n_tree_):  # per boosting
            for class_idx in range(self.n_class_):  # per class
                for test_idx in range(X.shape[0]):  # per test example
                    leaf_idx = doc2leaf[test_idx][tree_idx][class_idx]
                    og_pred[test_idx][class_idx] += self.leaf_values_[n_prev_leaves + leaf_idx]
                    new_pred[test_idx][class_idx] += self.leaf_derivatives_[remove_idx][n_prev_leaves + leaf_idx]

                n_prev_leaves += self.leaf_counts_[tree_idx + class_idx]

        return self.loss_fn_.gradient(y, og_pred) * new_pred

    def _get_docs_to_update(self, tree_idx, class_idx, leaf_counts, leaf2docs, remove_idx, da):
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
            leaf_count = leaf_counts[tree_idx + class_idx]
            leaf_docs = leaf2docs[tree_idx + class_idx]

            # sort leaf indices based on largest abs. da sum
            leaf_das = [np.sum(np.abs(da[list(leaf_docs[leaf_idx])])) for leaf_idx in range(leaf_count)]
            top_leaf_ids = np.argsort(leaf_das)[-self.update_set:]
            
            # return remove_idx + document indices for the top `k` leaves
            result = {remove_idx}
            for leaf_idx in top_leaf_ids:
                result |= leaf_docs[leaf_idx]

        return result

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
