import time
from abc import abstractmethod

from .parsers import parse_model


class Explainer(object):
    """
    Abstract class that all explainers must implement.
    """
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, model, X, y):
        """
        - Convert model to internal standardized tree structures.
        - Perform any initialization necessary for the chosen method.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        start = time.time()
        self.model_ = parse_model(model, X, y)
        assert self.model_.tree_type in ['rf', 'gbdt']
        assert self.model_.objective in ['regression', 'binary', 'multiclass']
        self.parse_time_ = time.time() - start

    @abstractmethod
    def get_global_influence(self, X=None, y=None):
        """
        - Provides a global importance to all training examples.

        Input
            X: 2d array of test examples.
            y: 1d array of test targets.

        Return
            - 1d array of shape=(no. train,).

        Note
            - If X and y are not None, then some explainers may provide
                the effect of each train example on the test set loss.
        """
        pass

    @abstractmethod
    def get_local_influence(self, X, y):
        """
        - Compute influence of each training instance on the test loss.

        Input
            X: 2d array of test examples.
            y: 1d array of test targets.
                * Could be the actual label or the predicted label depending on the explainer.

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Arrays are returned in the same order as the training data.
        """
        pass
