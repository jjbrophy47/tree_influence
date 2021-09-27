"""
Parse the incoming model into a standardized representation.
"""
import numpy as np

from .parser_cb import parse_cb_ensemble
from .parser_lgb import parse_lgb_ensemble
from .parser_sk import parse_skhgbm_ensemble
from .parser_sk import parse_skgbm_ensemble
from .parser_sk import parse_skrf_ensemble
from .parser_xgb import parse_xgb_ensemble
from .tree import TreeEnsemble
from .util import check_data


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
    if 'LGBM' in str(model):
        trees, params = parse_lgb_ensemble(model, X, y)

    elif 'XGB' in str(model):
        trees, params = parse_xgb_ensemble(model)

    elif 'HistGradientBoosting' in str(model):
        trees, params = parse_skhgbm_ensemble(model)

    elif 'GradientBoosting' in str(model):
        trees, params = parse_skgbm_ensemble(model)

    elif 'CatBoost' in str(model):
        trees, params = parse_cb_ensemble(model)

    elif 'RandomForest' in str(model):
        trees, params = parse_skrf_ensemble(model)

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
    """
    X = check_data(X, y, objective=model.objective)

    if model.objective == 'regression':
        p1 = original_model.predict(X)
        # if not X.flags.writeable:  # const memoryviews not supported in cython 0.29.23
        #     X.flags.writeable = True
        p2 = model.predict(X)[:, 0]

    elif model.objective == 'binary':
        p1 = original_model.predict_proba(X)[:, 1]
        # if not X.flags.writeable:  # const memoryviews not supported in cython 0.29.23
        #     X.flags.writeable = True
        p2 = model.predict(X)[:, 0]

    else:
        assert model.objective == 'multiclass'
        p1 = original_model.predict_proba(X)
        # if not X.flags.writeable:  # const memoryviews not supported in cython 0.29.23
        #     X.flags.writeable = True
        p2 = model.predict(X)

    assert np.allclose(p1, p2)
