"""
TODO: Add HistGradientBoostingClassifier.
TODO: Add HistGradientBoostingRegressor.
"""
import numpy as np

from .parser_cb import parse_cb_ensemble
from .parser_lgb import parse_lgb_ensemble
from .parser_sk import parse_skgbm_ensemble
from .parser_sk import parse_skrf_ensemble
from .parser_xgb import parse_xgb_ensemble
from .tree import TreeEnsemble


def parse_model(model, X, y):
    """
    Parse underlying structure based on the model type.

    Input
        model: Tree-ensemble object.
        X: 2d array of training data.
        y: 1d array of targets.

    Returns a standardized tree-ensemble representation.
    """

    # extract underlying tree-ensemble model representation
    if 'CatBoost' in str(model):
        trees, params = parse_cb_ensemble(model)

    elif 'LGBM' in str(model):
        trees, params = parse_lgb_ensemble(model, X, y)

    elif 'GradientBoosting' in str(model):
        trees, params = parse_skgbm_ensemble(model)

    elif 'RandomForest' in str(model):
        trees, params = parse_skrf_ensemble(model)

    elif 'XGB' in str(model):
        trees, params = parse_xgb_ensemble(model)

    else:
        raise ValueError(f'Could not parse {str(model)}')

    # create a standardized ensemble of the original model type
    ensemble = TreeEnsemble(trees, **params)

    # sanity check: make sure predictions match
    _check_predictions(model, ensemble, X, y)

    return ensemble


def _check_predictions(original_model, model, X, y):
    """
    Check to make sure both models produce the same predictions.

    WARNING
        - XGB: The representation given by `model.get_booster().get_dump()` is
            sometimes slightly different than the actual internal `model`.
            * This results in train examples sometimes going into DIFFERENT leaves
                between the standardized and original model, resulting in slightly
                different predictions for some examples.
    """
    if model.objective == 'regression':
        p1 = original_model.predict(X)
        p2 = model.predict(X)[:, 0]

    elif model.objective == 'binary':
        p1 = original_model.predict_proba(X)[:, 1]
        p2 = model.predict(X)[:, 0]

    else:
        assert model.objective == 'multiclass'
        p1 = original_model.predict_proba(X)
        p2 = model.predict(X)

    # DEBUG
    # import xgboost

    # print(dir(original_model))
    # leaves = original_model.get_booster().predict(xgboost.DMatrix(X), pred_leaf=True)
    # print(leaves, leaves.shape)

    # print(leaves[9687][15])

    # i = np.where(~np.isclose(p1, p2, atol=1e-5))[0]
    # print(i, i.shape)
    # print(p1[i[:1]])
    # print(p2[i[:1]])

    # print(model.trees[2][0])

    assert np.allclose(p1, p2)
