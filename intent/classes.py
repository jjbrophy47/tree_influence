from .explainers import BoostIn
from .explainers import Trex
from .explainers import LeafInfluence
from .explainers import LOO
from .explainers import DShap
from .explainers import Random
from .explainers import Minority
from .explainers import Loss
from .explainers import Similarity
from .explainers import Target


class TreeExplainer(object):
    """
    Wrapper object for the chosen explainer.

    Currently supported models:
        - CatBoostRegressor, CatBoostClassifier
        - LGBMRegressor, LGBMClassifier
        - GradientBoostingRegressor, GradientBoostingClassifier
        - RandomForestRegressor, RandomForestClassifier
        - XGBRegressor, XGBClassifier

    Currently supported explainers:
        - TracIn (BoostIn)
        - Representer-point (Trex)
        - Influence Function (LeafInfluence)
        - LOO
        - Appox. data Shapley
        - Random
        - Random Minority
        - Loss
        - Similarity
        - Target
    """
    def __init__(self, method='boostin', params={}, logger=None):

        if method == 'boostin':
            self.explainer = BoostIn(**params, logger=logger)

        elif method == 'trex':
            self.explainer = Trex(**params, logger=logger)

        elif method == 'leaf_influence':
            self.explainer = LeafInfluence(**params, logger=logger)

        elif method == 'loo':
            self.explainer = LOO(**params, logger=logger)

        elif method == 'dshap':
            self.explainer = DShap(**params, logger=logger)

        elif method == 'random':
            self.explainer = Random(**params, logger=logger)

        elif method == 'minority':
            self.explainer = Minority(**params, logger=logger)

        elif method == 'loss':
            self.explainer = Loss(**params, logger=logger)

        elif method == 'similarity':
            self.explainer = Similarity(**params, logger=logger)

        elif method == 'target':
            self.explainer = Target(**params, logger=logger)

        else:
            raise ValueError(f'Unknown method {method}')

    def fit(self, model, X, y):
        """
        - Convert model to internal standardized tree structures.
        - Perform any initialization necessary for the chosen explainer.

        Input
            model: tree ensemble.
            X: 2d array of train data.
            y: 1d array of train targets.
        """
        return self.explainer.fit(model, X, y)

    def get_global_influence(self):
        """
        - Compute influence of each training instance on itself.
        - Provides a global perspective of which training intances
          are most important.

        Return
            - Regression and binary: 1d array of shape=(no. train,).
            - Multiclass: 2d array of shape=(no. train, no. classes).
            - Arrays are returned in the same order as the traing data.
        """
        return self.explainer.get_global_influence()

    def get_local_influence(self, X, y):
        """
        - Compute most influential training instances on the prediction of the
          given test instance.

        Input
            - X: 2d array of test examples.
            - y: 1d array of test targets
                * Could be the actual label or the predicted label depending on the explainer.

        Return
            - Regression and binary: 2d array of shape=(no. train, X.shape[0]).
            - Multiclass: 3d array of shape=(X.shape[0], no. train, no. class).
            - Arrays are returned in the same order as the training data.
        """
        return self.explainer.get_local_influence(X, y)
