import argparse

import numpy as np
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier

from __init__ import parse_model


def get_test_data(args, n_class=2):
    rng = np.random.default_rng(args.rs)
    X_train = rng.standard_normal((args.n_train, args.n_feat))
    X_test = rng.standard_normal((args.n_test, args.n_feat))

    if n_class >= 2:  # classification
        y_train = rng.integers(n_class, size=args.n_train)
        y_test = rng.integers(n_class, size=args.n_test)

    elif n_class == -1:  # reegression
        y_train = rng.uniform(-100, 100, size=args.n_train)
        y_test = rng.uniform(-100, 100, size=args.n_test)

    else:
        raise ValueError(f'invalid n_class: {n_class}')

    return X_train, X_test, y_train, y_test


def compare_predictions(a1, a2):
    """
    Compare predictions between original model and extracted model.
    """
    status = 'failed'

    if np.all(a1.flatten() == a2.flatten()):
        status = 'passed'

    else:
        for atol in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            if np.all(np.isclose(a1.flatten(), a2.flatten(), atol=atol)):
                status = f'passed (atol={atol})'
                break

    return status


"""
CatBoost
"""


def test_cb_regressor(args):
    print(f'\n***** test_cb_regressor *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=-1)

    tree = CatBoostRegressor(n_estimators=args.n_tree, max_depth=args.max_depth,
                             random_state=args.rs, logging_level='Silent')
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


def test_cb_binary_classifier(args):
    print(f'\n***** test_cb_binary_classifier *****')
    X_train, X_test, y_train, y_test = get_test_data(args)

    tree = CatBoostClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                              random_state=args.rs, logging_level='Silent')
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict_proba(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


def test_cb_multiclass_classifier(args):
    print(f'\n***** test_cb_multiclass_classifier *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=3)

    tree = CatBoostClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                              random_state=args.rs, logging_level='Silent')
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict_proba(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


"""
LightGBM
"""


def test_lgb_regressor(args):
    print(f'\n***** test_lgb_regressor *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=-1)

    tree = LGBMRegressor(n_estimators=args.n_tree, max_depth=args.max_depth,
                         random_state=args.rs)
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


def test_lgb_binary_classifier(args):
    print(f'\n***** test_lgb_binary_classifier *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=2)

    tree = LGBMClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                          random_state=args.rs)
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict_proba(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


def test_lgb_multiclass_classifier(args):
    print(f'\n***** test_lgb_binary_classifier *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=3)

    tree = LGBMClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                          random_state=args.rs)
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict_proba(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


"""
SKLearn GBM
"""


def test_skgbm_regressor(args):
    print(f'\n***** test_skgbm_regressor *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=-1)

    tree = GradientBoostingRegressor(n_estimators=args.n_tree, max_depth=args.max_depth,
                                     random_state=args.rs)
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


def test_skgbm_binary_classifier(args):
    print(f'\n***** test_skgbm_binary_classifier *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=2)

    tree = GradientBoostingClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                                      random_state=args.rs)
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict_proba(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


def test_skgbm_multiclass_classifier(args):
    print(f'\n***** test_skgbm_multiclass_classifier *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=3)

    tree = GradientBoostingClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                                      random_state=args.rs)
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict_proba(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


"""
SKLearn RF
"""


def test_skrf_regressor(args):
    print(f'\n***** test_skrf_regressor *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=-1)

    tree = RandomForestRegressor(n_estimators=args.n_tree, max_depth=args.max_depth,
                                 random_state=args.rs)
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


def test_skrf_binary_classifier(args):
    print(f'\n***** test_skrf_binary_classifier *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=2)

    tree = RandomForestClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                                  random_state=args.rs)
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict_proba(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


def test_skrf_multiclass_classifier(args):
    print(f'\n***** test_skrf_multiclass_classifier *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=3)

    tree = RandomForestClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                                  random_state=args.rs)
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict_proba(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


"""
XGBoost
"""


def test_xgb_regressor(args):
    print(f'\n***** test_xgb_regressor *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=-1)

    tree = XGBRegressor(n_estimators=args.n_tree, max_depth=args.max_depth,
                        random_state=args.rs)
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


def test_xgb_binary_classifier(args):
    print(f'\n***** test_xgb_binary_classifier *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=2)

    tree = XGBClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                         random_state=args.rs, use_label_encoder=False,
                         eval_metric='logloss')
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict_proba(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


def test_xgb_multiclass_classifier(args):
    print(f'\n***** test_xgb_multiclass_classifier *****')
    X_train, X_test, y_train, y_test = get_test_data(args, n_class=3)

    tree = XGBClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                         random_state=args.rs, use_label_encoder=False,
                         eval_metric='mlogloss')
    tree = tree.fit(X_train, y_train)

    model = parse_model(tree)

    tree_pred = tree.predict_proba(X_test)
    model_pred = model.predict(X_test)

    status = compare_predictions(tree_pred, model_pred)
    print(status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--n_test', type=int, default=2)
    parser.add_argument('--n_feat', type=int, default=5)
    parser.add_argument('--n_tree', type=int, default=2)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--rs', type=int, default=1)
    args = parser.parse_args()

    # tests
    test_cb_regressor(args)
    test_cb_binary_classifier(args)
    test_cb_multiclass_classifier(args)

    test_lgb_regressor(args)
    test_lgb_binary_classifier(args)
    test_lgb_multiclass_classifier(args)

    test_skgbm_regressor(args)
    test_skgbm_binary_classifier(args)
    test_skgbm_multiclass_classifier(args)

    test_skrf_regressor(args)
    test_skrf_binary_classifier(args)
    test_skrf_multiclass_classifier(args)

    test_xgb_regressor(args)
    test_xgb_binary_classifier(args)
    test_xgb_multiclass_classifier(args)
