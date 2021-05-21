from cb_parser import parse_cb_ensemble
from tree import TreeEnsembleRegressor
from tree import TreeEnsembleBinaryClassifier
from tree import TreeEnsembleMulticlassClassifier

def parse_model(model):
    """
    Parse underlying structure based on the model type.
    """
    ensemble_type = None

    if 'CatBoost' in str(model):
        trees, scale, bias = parse_cb_ensemble(model)

        if 'Regressor' in str(model):
            ensemble_type = 'regressor'

    else:
        raise ValueError(f'Could not parse {str(model)}')

    # figure out the ensemble type of the original model
    if ensemble_type != 'regressor':

        if trees.ndim == 1:
            ensemble_type = 'binary_classifier'

        elif trees.ndim == 2 and trees.shape[1] >= 3:
            ensemble_type = 'multiclass_classifier'

    # create ensemble out of array of tree objects
    if ensemble_type == 'binary_classifier':
        ensemble = TreeEnsembleBinaryClassifier(trees, scale, bias)

    elif ensemble_type == 'multiclass_classifier':
        ensemble = TreeEnsembleMulticlassClassifier(trees, scale, bias)

    elif ensemble_type == 'regressor':
        ensemble = TreeEnsembleRegressor(trees, scale, bias)

    else:
        raise ValueError(f'Uknown ensemble_type: {ensemble_type}')

    return ensemble
