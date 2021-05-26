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
sys.path.insert(0, here + '/../')
from explainers.parsers import parse_model
from explainers import TracIn


def get_test_data(args):
    if args.model_type == 'regressor':
        n_class = -1
    elif args.model_type == 'binary':
        n_class = 2
    elif args.model_type == 'multiclass':
        n_class = 3
    else:
        raise ValueError(f'Unknown model_type {args.model_type}')

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


def get_model(args):
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


def test_tracin_self_influence_regression(args):
    print(f'\n***** test_tracin_self_influence_regression *****')
    args.model_type = 'regressor'
    X_train, X_test, y_train, y_test = get_test_data(args)

    tree = get_model(args)
    tree = tree.fit(X_train, y_train)

    kwargs = {'grad': args.grad, 'initial_grad': args.initial_grad}
    explainer = TracIn(**kwargs).fit(tree, X_train, y_train)
    self_inf = explainer = explainer.get_self_influence()

    print('y_train        (head):', y_train[:5])
    print('self influence (head):', self_inf[:5])

    status = 'passed' if self_inf.shape[0] == y_train.shape[0] else 'failed'
    print(status)


def test_tracin_self_influence_binary(args):
    print(f'\n***** test_tracin_self_influence_binary *****')
    args.model_type = 'binary'
    X_train, X_test, y_train, y_test = get_test_data(args)

    tree = get_model(args)
    tree = tree.fit(X_train, y_train)

    kwargs = {'grad': args.grad, 'initial_grad': args.initial_grad}
    explainer = TracIn(**kwargs).fit(tree, X_train, y_train)
    self_inf = explainer = explainer.get_self_influence()

    print('y_train        (head):', y_train[:5])
    print('self influence (head):', self_inf[:5])

    status = 'passed' if self_inf.shape[0] == y_train.shape[0] else 'failed'
    print(status)


def test_tracin_self_influence_multiclass(args):
    print(f'\n***** test_tracin_self_influence_multiclass *****')
    args.model_type = 'multiclass'
    X_train, X_test, y_train, y_test = get_test_data(args)
    n_class = len(np.unique(y_train))

    tree = get_model(args)
    tree = tree.fit(X_train, y_train)

    kwargs = {'grad': args.grad, 'initial_grad': args.initial_grad}
    explainer = TracIn(**kwargs).fit(tree, X_train, y_train)
    self_inf = explainer = explainer.get_self_influence()

    print('y_train        (head):', y_train[:5])
    print('self influence (head):\n', self_inf[:5])

    status = 'passed' if self_inf.shape == (y_train.shape[0], n_class) else 'failed'
    print(status)


def test_tracin_explain_regression(args):
    print(f'\n***** test_tracin_explain_regression *****')
    args.model_type = 'regressor'
    X_train, X_test, y_train, y_test = get_test_data(args)

    test_ndx = 0

    tree = get_model(args)
    tree = tree.fit(X_train, y_train)

    kwargs = {'grad': args.grad, 'initial_grad': args.initial_grad}
    explainer = TracIn(**kwargs).fit(tree, X_train, y_train)
    influence = explainer = explainer.explain(X_train[[test_ndx]], y_train[[test_ndx]])

    test_pred = tree.predict(X_train[[test_ndx]])
    test_label = y_train[test_ndx]

    s_ids = np.argsort(np.abs(influence))[::-1]

    print(f'explain y_train, index: 0, pred: {test_pred}, target: {test_label}\n')

    print('sorted indices    (head):', s_ids[:5])
    print('y_train   (head, sorted):', y_train[s_ids][:5])
    print('influence (head, sorted):', influence[s_ids][:5])

    status = 'passed' if influence.shape[0] == y_train.shape[0] else 'failed'
    print(status)


def test_tracin_explain_binary(args):
    print(f'\n***** test_tracin_explain_binary *****')
    args.model_type = 'binary'
    X_train, X_test, y_train, y_test = get_test_data(args)
    test_ndx = 0

    tree = get_model(args)
    tree = tree.fit(X_train, y_train)

    kwargs = {'grad': args.grad, 'initial_grad': args.initial_grad}
    explainer = TracIn(**kwargs).fit(tree, X_train, y_train)
    influence = explainer = explainer.explain(X_train[[test_ndx]], y_train[[test_ndx]])

    test_pred = tree.predict_proba(X_train[[test_ndx]])
    test_label = y_train[test_ndx]

    s_ids = np.argsort(np.abs(influence))[::-1]

    print(f'explain y_train 0, pred: {test_pred}, target: {test_label}\n')

    print('sorted indices    (head):', s_ids[:5])
    print('y_train   (head, sorted):', y_train[s_ids][:5])
    print('influence (head, sorted):', influence[s_ids][:5])

    status = 'passed' if influence.shape[0] == y_train.shape[0] else 'failed'
    print(status)


def test_tracin_explain_multiclass(args):
    print(f'\n***** test_tracin_explain_multiclass *****')
    args.model_type = 'multiclass'
    X_train, X_test, y_train, y_test = get_test_data(args)
    test_ndx = 0

    tree = get_model(args)
    tree = tree.fit(X_train, y_train)

    kwargs = {'grad': args.grad, 'initial_grad': args.initial_grad}
    explainer = TracIn(**kwargs).fit(tree, X_train, y_train)
    influence = explainer = explainer.explain(X_train[[test_ndx]], y_train[[test_ndx]])

    influence_agg = np.abs(influence).sum(axis=1)
    s_ids = np.argsort(np.abs(influence_agg))[::-1]

    test_pred = tree.predict_proba(X_train[[test_ndx]])
    test_label = y_train[test_ndx]

    print(f'explain y_train 0, pred: {test_pred}, target: {test_label}\n')

    print('sorted indices    (head):\n', s_ids[:5])
    print('y_train   (head, sorted):', y_train[s_ids][:5])
    print('influence (head, sorted):\n', influence[s_ids][:5])

    status = 'passed' if influence.shape[0] == y_train.shape[0] else 'failed'
    print(status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--n_test', type=int, default=1)
    parser.add_argument('--n_feat', type=int, default=10)
    parser.add_argument('--n_tree', type=int, default=5)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--model_type', type=str, default='dummy')
    parser.add_argument('--rs', type=int, default=1)
    parser.add_argument('--grad', type=str, default='residual', help='residual or approx')
    parser.add_argument('--initial_grad', type=str, default='keep', help='keep or skip')
    args = parser.parse_args()

    # tests
    test_tracin_self_influence_regression(args)
    test_tracin_self_influence_binary(args)
    test_tracin_self_influence_multiclass(args)

    test_tracin_explain_regression(args)
    test_tracin_explain_binary(args)
    test_tracin_explain_multiclass(args)
