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

    # print(model.bias)
    # print(model.trees[21][0])

    # l1 = model.apply(X)
    # t21 = l1.squeeze()[:,21]
    # i1 = np.where(t21 == 41)[0]
    # print(i1, i1.shape)

    # l2 = original_model.predict(X, pred_leaf=True)
    # i2 = np.where(l2[:, 21] == 40)[0]
    # print(i2, i2.shape)

    # print(np.all(i1 == i2))

    # json_data = original_model.booster_.dump_model()['tree_info']
    # print(json_data[21]['tree_structure'])

    # exit(0)

    # print(p1)
    # print(p2)
    # nc = np.where(~np.isclose(p1, p2))

    # print(nc)

    # print(p1[nc])
    # print(p2[nc])

    # leaves = original_model.predict(X, pred_leaf=True)
    # print(leaves.shape)
    # print(leaves[1246][:7], leaves[1246][7:14])

    assert np.allclose(p1, p2)
