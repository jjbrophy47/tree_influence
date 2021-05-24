from abc import abstractmethod


class Explainer(object):
    """
    Abstract class that all explainers must implement.
    """
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, model, X, y):
        """
        Convert model to internal standardized tree structures.
        Perform any initialization necessary for the chosen method.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        pass

    @abstractmethod
    def get_self_influence(self):
        """
        - Compute influence of each training instance on itself.
        - Provides a global perspective of which training intances
          are most important.

        Return
            - Regression and binary: 1d array of shape=(no. train).
            - Multiclass: 2d array of shape=(no. train, no. classes).
            - Array is returned in the same order as the traing data.
        """
        pass

    @abstractmethod
    def explain(self, X):
        """
        - Compute influence of each training instance on the test instance loss (or prediction?)
        - Provides a local explanation of the test instance loss (or prediction?).

        Return
            - Regression and binary: 1d array of shape=(no. train).
            - Multiclass: 2d array of shape=(no. train, no. classes).
            - Array is returned in the same order as the traing data.
        """
        pass
