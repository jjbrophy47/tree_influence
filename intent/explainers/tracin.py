import numpy as np

from .base import Explainer
from .parsers import parse_model


class ExplainerTracIn(Explainer):
    """
    Explainer that adapts the TracIn method to tree ensembles.
    """
    def __init__(self):
        pass

    def fit(self, model, X, y):
        """
        Convert model to internal standardized tree structures.
        Perform any initialization necessary for the chosen method.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        self.model_ = parse_model(model)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        return self

    def self_influence(self):
        """
        - Compute influence of each training instance on itself.
        - Provides a global perspective of which training intances
          are most important.

        Return
            - 1d array of shape=(no. train) (regression and binary classification).
            - 2d array of shape=(no. train, no. classes) (multiclass).
            - Arrays are returned in the same order as the traing data.
        """
        return self.model_.predict(self.X_train_)
