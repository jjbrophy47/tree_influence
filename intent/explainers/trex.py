import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from .base import Explainer
from .parsers import util


class Trex(Explainer):
    """
    Tree-Ensemble Representer Examples: Explainer that adapts the
    Representer point method to tree ensembles.

    Notes
        - 'kernel', 'lmbd', and 'target' have a significant effect on approximation accuracy.

    Reference
         - https://github.com/chihkuanyeh/Representer_Point_Selection/blob/master/compute_representer_vals.py
    """
    def __init__(self, kernel='lpw', target='actual', lmbd=0.00003, n_epoch=3000,
                 random_state=1, verbose=0):
        """
        Input
            kernel: Transformation of the input using the tree-ensemble structure.
                'to_': Tree output; output of each tree in the ensemble.
                'lp_': Leaf path; one-hot encoding of leaf indices across all trees.
                'lpw': Weighted leaf path; like 'lp' but replaces 1s with 1 / leaf count.
                'lo_': Leaf output; like 'lp' but replaces 1s with leaf values.
                'low': Weighted leaf otput; like 'lo' but replace leaf value with 1 / leaf value.
                'fp_': Feature path; one-hot encoding of node indices across all trees.
                'fpw': Weighted feature path; like 'fp' but replaces 1s with 1 / node count.
                'fo_': Feature output; like 'fp' but replaces leaf 1s with leaf values.
                'fow': Weighted feature output; like 'fo' but replaces leaf 1s with 1 / leaf values.
            target: Targets for the linear model to train on.
                'actual': Ground-truth targets.
                'predicted': Predicted targets from the tree-ensemble.
            lmbd: Regularizer for the linear model; necessary for the Representer decomposition.
            n_epoch: Max. no. epochs to train the linear model.
            random_state: Random state seed to generate reproducible results.
            verbose: Output verbosity.
        """
        assert kernel in ['to_', 'lp_', 'lpw', 'lo_', 'low', 'fp_', 'fpw', 'fo_', 'fow']
        assert target in ['actual', 'predicted']
        assert isinstance(lmbd, float)
        self.kernel = kernel
        self.target = target
        self.lmbd = lmbd
        self.n_epoch = n_epoch
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, model, X, y):
        """
        - Convert model to internal standardized tree structure.
        - Transform the input data using the specified tree kernel.
        - Fit linear model and compute train instance weights (alpha).

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, task=self.model_.task_)

        self.model_.update_node_count(X)

        self.X_train_ = self._kernel_transform(X)

        # select target
        if self.target == 'actual':

            if self.model_.task_ == 'regression':  # shape=(no. train,)
                self.y_train_ = y

            elif self.model_.task_ == 'multiclass':  # shape=(no. train, no. class)
                self.y_train_ = LabelBinarizer().fit_transform(y)

            elif self.model_.task_ == 'binary':  # shape=(no. train, 2)
                self.y_train_ = OneHotEncoder().fit_transform(y.reshape(-1, 1)).todense()

        elif self.target == 'predicted':

            if self.model_.task_ == 'regression':  # shape=(no. train,)
                self.y_train_ = model.predict(X)

            elif self.model_.task_ == 'binary':  # shape=(no. train, 2)
                self.y_train_ = model.predict_proba(X)

            elif self.model_.task_ == 'multiclass':  # shape=(no. train, no. class)
                self.y_train_ = model.predict_proba(X)

        self.alpha_ = self._compute_train_weights(self.X_train_, self.y_train_)

        return self

    def get_self_influence(self):
        """
        - Compute influence of each training instance on itself.
        - Provides a global importance to all training intances.

        Return
            - 1d array of shape=(no. train) (regression and binary classification).
            - 2d array of shape=(no. train, no. classes) (multiclass).
            - Arrays are returned in the same order as the traing data.
        """
        return self.alpha_

    def explain(self, X, y):
        """
        - Compute attribution of each training instance on the test instance prediction.
            Transform the test instance using the specified tree kernel.
            Compute dot prod. between transformed train and test, weighted by alpha.
        - Provides a local explanation of the test instance prediction.

        Return
            - Regression and binary: 1d array of shape=(no. train).
            - Multiclass: 2d array of shape=(no. train, no. classes).
            - Array is returned in the same order as the traing data.
        """
        X, y = util.check_data(X, y, task=self.model_.task_)
        assert X.shape[0] == 1 and y.shape[0] == 1

        X_test_ = self._kernel_transform(X).flatten()
        sim = np.dot(self.X_train_, X_test_)

        if self.model_.task_ in ['regression', 'binary']:
            influence = sim * self.alpha_  # representer values

        elif self.model_.task_ == 'multiclass':
            influence = np.zeros(self.alpha_.shape, dtype=np.float32)

            for j in range(influence.shape[1]):  # per class
                influence[:, j] = sim * self.alpha_[j]  # representer values

        return influence

    # private
    def _kernel_transform(self, X):
        """
        Transforms each x in X using the specified tree kernel.
        """
        structure_dict = {'t': 'tree', 'l': 'leaf', 'f': 'feature'}
        output_dict = {'p': 'path', 'o': 'output'}
        weight_dict = {'_': 'unweighted', 'w': 'weighted'}
        
        s1, s2, s3 = list(self.kernel)
        structure = structure_dict[s1]
        output = output_dict[s2]
        weight = weight_dict[s3]

        if structure == 'tree':
            X_ = self._tree_kernel_transform(X)

        elif structure == 'leaf':
            X_ = self._leaf_kernel_transform(X, output=output, weight=weight)

        elif structure == 'feature':
            X_ = self._feature_kernel_transform(X, output=output, weight=weight)

        return X_

    def _tree_kernel_transform(self, X):
        """
        Transform each x in X to be a vector of tree outputs.

        Return
            - Regression and binary: 2d array of shape=(no. train, no. trees).
            - Multiclass: 2d array of shape=(no. train, no. trees * no. class).
        """
        trees = self.model_.trees.flatten()
        X_ = np.zeros((X.shape[0], trees.shape[0]))

        for i, tree in enumerate(trees):
            X_[:, i] = tree.predict(X)

        return X_

    def _leaf_kernel_transform(self, X, output='path', weight='unweighted'):
        """
        - Transform each x in X to be a vector of one-hot encoded leaf paths.
        - The `output` and `weight` parameters control the value of the 1s.

        Return
            - Regression and binary: 2d array of shape=(no. train, total no. leaves).
            - Multiclass: 2d array of shape=(no. train, ~total no. leaves * no. class).
        """
        trees = self.model_.trees.flatten()
        total_n_leaves = np.sum([tree.leaf_count_ for tree in trees])

        X_ = np.zeros((X.shape[0], total_n_leaves))

        output = True if output == 'output' else False
        weighted = True if weight == 'weighted' else False

        n_prev_leaves = 0
        for tree in trees:
            start = n_prev_leaves
            stop = n_prev_leaves + tree.leaf_count_
            X_[:, start: stop] = tree.leaf_path(X, output=output, weighted=weighted)
            n_prev_leaves += tree.leaf_count_

        return X_

    def _feature_kernel_transform(self, X, output='path', weight='unweighted'):
        """
        - Transform each x in X to be a vector of one-hot encoded feature paths.
        - The `output` parameter controls the value of the leaf 1s.
        - The `weight` parameter controls the value of all 1s.

        Return
            - Regression and binary: 2d array of shape=(no. train, total no. nodes).
            - Multiclass: 2d array of shape=(no. train, ~total no. nodes * no. class).
        """
        trees = self.model_.trees.flatten()
        total_n_nodes = np.sum([tree.node_count_ for tree in trees])

        X_ = np.zeros((X.shape[0], total_n_nodes))

        output = True if output == 'output' else False
        weighted = True if weight == 'weighted' else False

        n_prev_nodes = 0
        for tree in trees:
            start = n_prev_nodes
            stop = n_prev_nodes + tree.node_count_
            X_[:, start: stop] = tree.feature_path(X, output=output, weighted=weighted)
            n_prev_nodes += tree.node_count_

        return X_

    def _compute_train_weights(self, X, y):
        """
        Fit a linear model to X and y, then extract weights for all
        train instances.
        """
        X = Variable(torch.FloatTensor(X))
        y = Variable(torch.FloatTensor(y))
        N = len(y)

        # randomly initialize weights
        rng = np.random.default_rng(self.random_state)

        # MSE model for regression, Softmax model for classification
        if self.model_.task_ == 'regression':
            assert y.ndim == 1
            W = rng.uniform(-1, 1, size=X.shape[1])
            model = MSEModel(W)

        elif self.model_.task_ in ['binary', 'multiclass']:
            assert y.ndim == 2
            W = rng.uniform(-1, 1, size=(X.shape[1], y.shape[1]))
            model = SoftmaxModel(W)

        # optimization settings
        min_loss = 10000.0
        optimizer = optim.SGD([model.W], lr=1.0)

        # train
        for epoch in range(self.n_epoch):
            phi_loss = 0

            optimizer.zero_grad()
            (Phi, L2) = model(X, y)
            loss = Phi / N + L2 * self.lmbd

            phi_loss += util.to_np(Phi / N)
            loss.backward()

            temp_W = model.W.data
            grad_loss = util.to_np(torch.mean(torch.abs(model.W.grad)))

            # save the W with lowest loss
            if grad_loss < min_loss:

                if epoch == 0:
                    init_grad = grad_loss

                min_loss = grad_loss
                best_W = temp_W

                if min_loss < init_grad / 200:
                    print(f'stopping criteria reached in epoch: {epoch}')
                    break

            self._backtracking_line_search(model, model.W.grad, X, y, loss)

            if epoch % 100 == 0 and self.verbose > 0:
                print(f'Epoch:{epoch:4d}, loss:{util.to_np(loss):.7f}'
                      f', phi_loss:{phi_loss:.7f}, grad:{grad_loss:.7f}')

        # compute alpha based on the representer theorem's decomposition
        output = torch.matmul(X, Variable(best_W))  # shape=(no. train, no. class)
        if self.model_.task_ in ['binary', 'multiclass']:
            output = torch.softmax(output, axis=1)

        alpha = output - y  # half the derivative of mse; derivative of softmax cross-entropy
        alpha = torch.div(alpha, (-2.0 * self.lmbd * N))

        # compute W based on the Representer Theorem decomposition
        W = torch.matmul(torch.t(X), alpha)  # shape=(no. features,)

        output_approx = torch.matmul(X, W)
        if self.model_.task_ in ['binary', 'multiclass']:
            output_approx = torch.softmax(output_approx, axis=1)

        # compute closeness
        y = util.to_np(y).flatten()
        yp = util.to_np(output_approx).flatten()

        l1_diff = np.mean(np.abs(y - yp))
        p_corr, _ = pearsonr(y, yp)
        s_corr, _ = spearmanr(y, yp)

        if self.verbose > 0:
            print(f'L1 diff.: {l1_diff:.5f}, pearsonr: {p_corr:.5f}, spearmanr: {s_corr:.5f}')
            plt.scatter(yp, y)
            plt.show()

        self.alpha_ = util.to_np(alpha)
        self.l1_diff_ = l1_diff
        self.p_corr_ = p_corr
        self.s_corr_ = s_corr

    def _backtracking_line_search(self, model, grad, X, y, val, beta=0.5):
        """
        Search for and then take the biggest possible step for gradient descent.
        """
        N = X.shape[0]

        t = 10.0
        W_O = util.to_np(model.W)
        grad_np = util.to_np(grad)

        while(True):
            model.W = Variable(torch.from_numpy(W_O - t * grad_np).type(torch.float32), requires_grad=True)

            val_n = 0.0
            (Phi, L2) = model(X, y)
            val_n = Phi / N + L2 * self.lmbd

            if t < 0.0000000001:
                print("t too small")
                break

            if util.to_np(val_n - val + t * torch.norm(grad) ** 2 / 2) >= 0:
                t = beta * t
            else:
                break


class SoftmaxModel(nn.Module):
    """
    Model that computes the binary or multiclass cross-entropy loss.
    """

    def __init__(self, W):
        super(SoftmaxModel, self).__init__()
        self.W = Variable(torch.from_numpy(W).type(torch.float32), requires_grad=True)

    def forward(self, X, y):
        """
        Calculate loss for the loss function and L2 regularizer.

        Note
            - This loss function represents "closeness" if y is predicted probabilities.
        """
        D = torch.matmul(X, self.W)  # raw output, shape=(X.shape[0], no. class)
        D = D - torch.logsumexp(D, axis=1).reshape(-1, 1)  # normalize log probs.
        Phi = torch.sum(-torch.sum(D * y, axis=1))  # cross-entropy loss

        # L2 norm.
        W1 = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(W1, W1))

        return (Phi, L2)


class MSEModel(nn.Module):
    """
    Model that computes the mean sqaured error loss.

        Note
            - This loss function represents "closeness" if y is predicted probabilities.
    """

    def __init__(self, W):
        super(MSEModel, self).__init__()
        self.W = Variable(torch.from_numpy(W).type(torch.float32), requires_grad=True)

    def forward(self, X, y):
        """
        Calculate loss for the loss function and L2 regularizer.

        Note
            - This loss function represents "closeness" if y is predicted values.
        """
        D = torch.matmul(X, self.W)  # raw output, shape=(X.shape[0],)
        Phi = torch.sum(torch.square(D - y))  # MSE loss

        # L2 norm.
        W1 = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(W1, W1))

        return (Phi, L2)
