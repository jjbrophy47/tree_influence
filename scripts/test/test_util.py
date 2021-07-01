import os
import sys
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

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
from intent.explainers import LeafInfluence


def test_self_influence_regression(args, explainer_cls, str_explainer, kwargs):
    print(f'\n***** test_{str_explainer}_self_influence_regression *****')
    args.model_type = 'regressor'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=-1)

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train)
    self_inf = explainer.get_self_influence()

    print('\ny_mean:', y_train.mean())
    print('y_pred         (head):', tree.predict(X_train)[:5])
    print('y_train        (head):', y_train[:5])
    print('self influence (head):', self_inf[:5])

    status = 'passed' if self_inf.shape[0] == y_train.shape[0] else 'failed'
    print(f'\n{status}')


def test_self_influence_binary(args, explainer_cls, explainer_str, kwargs):
    print(f'\n***** test_{explainer_str}_self_influence_binary *****')
    args.model_type = 'binary'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=2)

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train)
    self_inf = explainer.get_self_influence()

    print('\ny_train        (head):', y_train[:5])
    print('self influence (head):\n', self_inf[:5])

    status = 'passed' if self_inf.shape[0] == y_train.shape[0] else 'failed'
    print(f'\n{status}')


def test_self_influence_multiclass(args, explainer_cls, explainer_str, kwargs):
    print(f'\n***** test_{explainer_str}_self_influence_multiclass *****')
    args.model_type = 'multiclass'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=args.n_class)
    n_class = len(np.unique(y_train))

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train)
    self_inf = explainer.get_self_influence()

    print('y_train        (head):', y_train[:5])
    print('self influence (head):\n', self_inf[:5])

    status = 'passed' if self_inf.shape == (y_train.shape[0], n_class) else 'failed'
    print(f'\n{status}')


def test_explain_regression(args, explainer_cls, explainer_str, kwargs):
    print(f'\n***** test_{explainer_str}_explain_regression *****')
    args.model_type = 'regressor'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=-1)
    test_ids = np.array([0, 1])

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train)
    influences = explainer.explain(X_train[test_ids], y_train[test_ids])  # shape=(no. train, no. test)

    for i, test_idx in enumerate(test_ids):

        influence = influences[:, i]
        s_ids = np.argsort(np.abs(influence))[::-1]

        test_pred = tree.predict(X_train[[test_idx]])[0]
        test_label = y_train[test_idx]

        print(f'\nexplain y_train, index: {test_idx}, pred: {test_pred}, target: {test_label}')

        print('sorted indices    (head):', s_ids[:5])
        print('y_train   (head, sorted):', y_train[s_ids][:5])
        print('influence (head, sorted):', influence[s_ids][:5])

    status = 'passed' if influences.shape == (X_train.shape[0], test_ids.shape[0]) else 'failed'
    print(f'\n{status}')


def test_explain_binary(args, explainer_cls, explainer_str, kwargs):
    print(f'\n***** test_{explainer_str}_explain_binary *****')
    args.model_type = 'binary'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=2)
    test_ids = np.array([0, 1])

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train)
    influences = explainer.explain(X_train[test_ids], y_train[test_ids])   # shape=(no. train, no. test)

    for i, test_idx in enumerate(test_ids):

        influence = influences[:, i]
        s_ids = np.argsort(np.abs(influence))[::-1]

        test_pred = tree.predict_proba(X_train[[test_idx]])[0]
        test_label = y_train[test_idx]

        print(f'\nexplain y_train {test_idx}, pred: {test_pred}, target: {test_label}\n')

        print('sorted indices    (head):', s_ids[:5])
        print('y_train   (head, sorted):', y_train[s_ids][:5])
        print('influence (head, sorted):', influence[s_ids][:5])

    status = 'passed' if influences.shape == (X_train.shape[0], test_ids.shape[0]) else 'failed'
    print(f'\n{status}')


def test_explain_multiclass(args, explainer_cls, explainer_str, kwargs):
    print(f'\n***** test_{explainer_str}_explain_multiclass *****')
    args.model_type = 'multiclass'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=args.n_class)
    test_ids = np.array([0, 1])

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = explainer_cls(**kwargs).fit(tree, X_train, y_train)
    influences = explainer.explain(X_train[test_ids], y_train[test_ids])   # shape=(no. test, no. train, no. class)

    for i, test_idx in enumerate(test_ids):

        influence = influences[i]  # shape=(no. train, no. class)
        influence_agg = np.abs(influence).sum(axis=1)
        s_ids = np.argsort(np.abs(influence_agg))[::-1]

        test_pred = tree.predict_proba(X_train[[test_idx]])[0]
        test_label = y_train[test_idx]

        print(f'\nexplain y_train {test_idx}, pred: {test_pred}, target: {test_label}\n')

        print('sorted indices    (head):\n', s_ids[:5])
        print('y_train   (head, sorted):', y_train[s_ids][:5])
        print('influence (head, sorted):\n', influence[s_ids][:5])

    status = 'passed' if influences.shape == (test_ids.shape[0], X_train.shape[0], args.n_class) else 'failed'
    print(f'\n{status}')


# private
def _get_test_data(args, n_class=2):
    """
    Return train and test data for the given objective.
    """
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


def _get_model(args):
    """
    Return tree-ensemble.
    """

    if args.tree_type == 'cb':
        class_fn = CatBoostRegressor if args.model_type == 'regressor' else CatBoostClassifier
        tree = class_fn(n_estimators=args.n_tree, max_depth=args.max_depth,
                        random_state=args.rs, logging_level='Silent')

    elif args.tree_type == 'lgb':
        class_fn = LGBMRegressor if args.model_type == 'regressor' else LGBMClassifier
        tree = class_fn(n_estimators=args.n_tree, max_depth=args.max_depth, random_state=args.rs)

    elif args.tree_type == 'skgbm':
        class_fn = GradientBoostingRegressor if args.model_type == 'regressor' else GradientBoostingClassifier
        tree = class_fn(n_estimators=args.n_tree, max_depth=args.max_depth, random_state=args.rs)

    elif args.tree_type == 'skrf':
        class_fn = RandomForestRegressor if args.model_type == 'regressor' else RandomForestClassifier
        tree = class_fn(n_estimators=args.n_tree, max_depth=args.max_depth, random_state=args.rs)

    elif args.tree_type == 'xgb':

        if args.model_type == 'regressor':
            tree = XGBRegressor(n_estimators=args.n_tree, max_depth=args.max_depth, random_state=args.rs)

        elif args.model_type == 'binary':
            tree = XGBClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                                 random_state=args.rs, use_label_encoder=False,
                                 eval_metric='logloss')

        elif args.model_type == 'multiclass':
            tree = XGBClassifier(n_estimators=args.n_tree, max_depth=args.max_depth,
                                 random_state=args.rs, use_label_encoder=False,
                                 eval_metric='mlogloss')
        else:
            raise ValueError(f'Unknown model_type {args.model_type}')

    else:
        raise ValueError(f'Unknown tree_type {args.tree_type}')

    return tree
