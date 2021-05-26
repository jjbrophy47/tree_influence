from .explainers import TracIn


class TreeExplainer(object):
    """
    Wrapper object for the chosen explainer.

    Currently supported models:
        - CatBoostRegressor, CatBoostClassifier
        - LGBMRegressor, LGBMClassifier
        - GradientBoostingRegressor, GradientBoostingClassifier
        - RandomForestREgressor, RandomForestClassifier
        - XGBRegressor, XGBClassifier
    """
    def __init__(self, method='tracin', **kwargs):

        if method == 'tracin':
            self.explainer = TracIn(**kwargs)

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
