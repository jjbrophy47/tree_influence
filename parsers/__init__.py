from parser_cb import parse_cb_ensemble
from parser_lgb import parse_lgb_ensemble
from parser_sk import parse_skgbm_ensemble
from parser_sk import parse_skrf_ensemble
from tree import TreeEnsembleRegressor
from tree import TreeEnsembleBinaryClassifier
from tree import TreeEnsembleMulticlassClassifier


def parse_model(model):
    """
    Parse underlying structure based on the model type.
    """
    tree_type = 'gbdt'

    # extract underlying tree-ensemble model representation
    if 'CatBoost' in str(model):
        trees, bias = parse_cb_ensemble(model)

    elif 'LGBM' in str(model):
        trees, bias = parse_lgb_ensemble(model)

    elif 'GradientBoosting' in str(model):
        trees, bias = parse_skgbm_ensemble(model)

    elif 'RandomForest' in str(model):
        trees, bias = parse_skrf_ensemble(model)
        tree_type = 'rf'

    else:
        raise ValueError(f'Could not parse {str(model)}')

    # create a standardized ensemble of the original model type
    if 'Regressor' in str(model):
        ensemble = TreeEnsembleRegressor(trees, bias, tree_type)

    else:

        if trees.ndim == 1:
            ensemble = TreeEnsembleBinaryClassifier(trees, bias, tree_type)

        elif trees.ndim == 2 and trees.shape[1] >= 3:
            ensemble = TreeEnsembleMulticlassClassifier(trees, bias, tree_type)

        else:
            raise ValueError(f'Uknown ensemble type')

    return ensemble
