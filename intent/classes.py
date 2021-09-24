from .explainers import BoostIn
from .explainers import Trex
from .explainers import LeafInfluence
from .explainers import LeafInfluenceSP
from .explainers import LeafRefit
from .explainers import LOO
from .explainers import DShap
from .explainers import Random
from .explainers import Minority
from .explainers import Loss
from .explainers import TreeSim
from .explainers import LeafSim
from .explainers import InputSim
from .explainers import Target
from .explainers import SubSample


class TreeExplainer(object):
    """
    Wrapper object for the chosen explainer.

    Currently supported models:
        - LGBMRegressor, LGBMClassifier
        - XGBRegressor, XGBClassifier
        - CatBoostRegressor, CatBoostClassifier
        - HistGradientBoostingRegressor, HistGradientBoostingClassifier
        - GradientBoostingRegressor, GradientBoostingClassifier

    Semi-supported models:
        - RandomForestRegressor, RandomForestClassifier

    Currently supported explainers:
        - BoostIn (adapted TracIn)
        - TREX (adapted representer-point)
        - LeafInfluenceSP (efficient version of LeafInfluence: single point)
        - LeafInfluence (adapted influence functions)
        - LeafRefit (LOO w/ fixed structure)
        - LeafSim  (similarity based on the weighted-leaf-path tree kernel)
        - TreeSim  (similarity based on an arbitrary tree kernel)
        - InputSim (similarity based on input features)
        - SubSample (Approx. Data Shapley)
        - TMC-Shap (Appox. Data Shapley)
        - LOO (leave-one-out retraining)
        - Target (random from same class as test example)
        - Random

    Global-only explainers:
        - Loss (loss of train examples)
        - Minority (random from the minority class)
    """
    def __init__(self, method='boostin', params={}, logger=None):

        if method == 'boostin':
            self.explainer = BoostIn(**params, logger=logger)

        elif method == 'trex':
            self.explainer = Trex(**params, logger=logger)

        elif method == 'leaf_inf':
            self.explainer = LeafInfluence(**params, logger=logger)

        elif method == 'leaf_infSP':
            self.explainer = LeafInfluenceSP(**params, logger=logger)

        elif method == 'leaf_refit':
            self.explainer = LeafRefit(**params, logger=logger)

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

        elif method == 'tree_sim':
            self.explainer = TreeSim(**params, logger=logger)

        elif method == 'leaf_sim':
            self.explainer = LeafSim(**params, logger=logger)

        elif method == 'input_sim':
            self.explainer = InputSim(**params, logger=logger)

        elif method == 'target':
            self.explainer = Target(**params, logger=logger)

        elif method == 'subsample':
            self.explainer = SubSample(**params, logger=logger)

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

    def get_self_influence(self, X, y):
        """
        - Compute influence of each training instance on itself.
        - Provides a global perspective of which training intances
          are most important.

        Input
            - X: 2d array of train examples.
            - y: 1d array of train targets

        Return
            - 1d array of shape=(no. train,).
                * Arrays are returned in the same order as the traing data.
        """
        return self.explainer.get_self_influence(X, y)

    def get_local_influence(self, X, y):
        """
        - Compute most influential training instances on the prediction of the
          given test instance.

        Input
            - X: 2d array of test examples.
            - y: 1d array of test targets
                * Could be the actual label or the predicted label depending on the explainer.

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Arrays are returned in the same order as the training data.
        """
        return self.explainer.get_local_influence(X, y)
