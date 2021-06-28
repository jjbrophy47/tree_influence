from .explainers import TracIn
from .explainers import Trex
from .explainers import LeafInfluence


class TreeExplainer(object):
    """
    Wrapper object for the chosen explainer.

    Currently supported models:
        - CatBoostRegressor, CatBoostClassifier
        - LGBMRegressor, LGBMClassifier
        - GradientBoostingRegressor, GradientBoostingClassifier
        - RandomForestREgressor, RandomForestClassifier
        - XGBRegressor, XGBClassifier

    Currently supported explainers:
        - TracIn
        - Representer-point (Trex)
        - Influence Function (LeafInfluence)
        - HYDRA
    """
    def __init__(self, method='tracin', **kwargs):

        if method == 'tracin':
            self.explainer = TracIn(**kwargs)

        elif method == 'trex':
            self.explainer = Trex(**kwargs)

        elif method == 'leaf_influence':
            self.explainer = LeafInfluence(**kwargs)

        elif method == 'hydra':
            self.explainer = Hydra(**kwargs)

        else:
            raise ValueError(f'Unknown method {method}')

    def fit(self, model, X, y):
        """
        Convert model to internal standardized tree structures.
        Perform any initialization necessary for the chosen method.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
        """
        return self.explainer.fit(model, X, y)

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
        return self.explainer.get_self_influence()

    def explain(self, X, y):
        """
        - Compute most influential training instances on the prediction of the
          given test instance.
        """
        return self.explainer.explain(X, y)
