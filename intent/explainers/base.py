"""
TODO
    - How do we pick a comparable way to choose train instances for explaining a single test instance.
        * TracIn measures test LOSS with respect to bossting iterations.
        * Trex decomposes the test PREDICTION into training example attributions.
        * LeafInfluence measures test LOSS with respect to changing leaf values.
        * HYDRA measures TODO...
"""
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
        self.model_ = parse_model(model, X, y)
        assert self.model_.tree_type in ['rf', 'gbdt']
        assert self.model_.objective in ['regression', 'binary', 'multiclass']

    @abstractmethod
    def get_self_influence(self):
        """
        - Compute influence of each training instance on itself.
        - Provides a global perspective of which training intances
          are most important.

        Return
            - Regression and binary: 1d array of shape=(no. train).
            - Multiclass: 2d array of shape=(no. train, no. classes).
            - Arrays are returned in the same order as the traing data.
        """
        pass

    @abstractmethod
    def explain(self, X, y):
        """
        - Compute influence of each training instance on the test instance loss.
        - Provides local explanations for the given test example losses or predictions.

        Input
            - X: 2d array of shape=(1, no. feature).
            - y: 1d array of shape=(target,). True or predicted label (see notes).

        Return
            - Regression and binary: 2d array of shape=(no. train, X.shape[0]).
            - Multiclass: 3d array of shape=(X.shape[0], no. train, no. class).
            - Arrays are returned in the same order as the training data.
        """
        pass
