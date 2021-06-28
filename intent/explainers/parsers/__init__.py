"""
TODO: Add HistGradientBoostingClassifier.
TODO: Add HistGradientBoostingRegressor.
"""
from .parser_cb import parse_cb_ensemble
from .parser_lgb import parse_lgb_ensemble
from .parser_sk import parse_skgbm_ensemble
from .parser_sk import parse_skrf_ensemble
from .parser_xgb import parse_xgb_ensemble
from .tree import TreeEnsembleRegressor
from .tree import TreeEnsembleBinaryClassifier
from .tree import TreeEnsembleMulticlassClassifier


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
    if params['objective'] == 'regression':
        ensemble = TreeEnsembleRegressor(trees, params)

    elif params['objective'] == 'binary':
        ensemble = TreeEnsembleBinaryClassifier(trees, params)

    elif params['objective'] == 'multiclass':
        ensemble = TreeEnsembleMulticlassClassifier(trees, params)

    else:
        raise ValueError(f'Unknown objective {params["objective"]}')

    return ensemble
