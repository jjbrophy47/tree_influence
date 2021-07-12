"""
Utility methods.
"""
import os
import sys
import shutil
import logging
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
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


def get_logger(filename=''):
    """
    Return a logger object to easily save textual output.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    log_handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')

    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    return logger


def clear_dir(in_dir):
    """
    Clear contents of directory.
    """
    if not os.path.exists(in_dir):
        return -1

    # remove contents of the directory
    for fn in os.listdir(in_dir):
        fp = os.path.join(in_dir, fn)

        # directory
        if os.path.isdir(fp):
            shutil.rmtree(fp)

        # file
        else:
            os.remove(fp)

    return 0


def get_data(data_dir, dataset):
    """
    Return train and test data for the specified dataset.
    """
    data = np.load(os.path.join(data_dir, dataset, 'data.npy'), allow_pickle=True)[()]

    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    # get objective for the given dataset
    d = {}
    d['regression'] = ['synthetic_regression']
    d['binary'] = ['synthetic_binary']
    d['multiclass'] = ['synthetic_multiclass']

    objective = ''
    for k in d.keys():
        if dataset in d[k]:
            objective = k
            break

    return X_train, X_test, y_train, y_test, objective


def get_toy_data(dataset, objective, random_state, test_size=0.2):

    if dataset == 'boston':
        assert objective == 'regression'
        data = load_boston()

    elif dataset == 'iris':
        assert objective == 'multiclass'
        data = load_iris()

    elif dataset == 'diabetes':
        assert objective == 'regression'
        data = load_diabetes()

    elif dataset == 'digits':
        assert objective == 'multiclass'
        data = load_digits()

    elif dataset == 'wine':
        assert objective == 'multiclass'
        data = load_wine()

    elif dataset == 'breast_cancer':
        assert objective == 'binary'
        data = load_breast_cancer()

    X = data['data']
    y = data['target']

    stratify = y if task in ['binary', 'multiclass'] else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=stratify)

    return X_train, X_test, y_train, y_test


def get_model(tree_type='lgb', task='regression', n_tree=100, max_depth=5, random_state=1):
    """
    Return the ensemble object from the specified framework and task.
    """

    if tree_type == 'cb':
        class_fn = CatBoostRegressor if task == 'regression' else CatBoostClassifier
        tree = class_fn(n_estimators=n_tree, max_depth=max_depth,
                        leaf_estimation_iterations=1, random_state=random_state,
                        logging_level='Silent')

    elif tree_type == 'lgb':
        class_fn = LGBMRegressor if task == 'regression' else LGBMClassifier
        tree = class_fn(n_estimators=n_tree, max_depth=max_depth, random_state=random_state)

    elif tree_type == 'skgbm':
        class_fn = GradientBoostingRegressor if task == 'regression' else GradientBoostingClassifier
        tree = class_fn(n_estimators=n_tree, max_depth=max_depth, random_state=random_state)

    elif tree_type == 'skrf':
        class_fn = RandomForestRegressor if task == 'regression' else RandomForestClassifier
        tree = class_fn(n_estimators=n_tree, max_depth=max_depth, random_state=random_state)

    elif tree_type == 'xgb':

        if task == 'regression':
            tree = XGBRegressor(n_estimators=n_tree, max_depth=max_depth, random_state=random_state)

        elif task == 'binary':
            tree = XGBClassifier(n_estimators=n_tree, max_depth=max_depth,
                                 random_state=random_state, use_label_encoder=False,
                                 eval_metric='logloss')

        elif task == 'multiclass':
            tree = XGBClassifier(n_estimators=n_tree, max_depth=max_depth,
                                 random_state=random_state, use_label_encoder=False,
                                 eval_metric='mlogloss')
        else:
            raise ValueError(f'Unknown task {task}')

    else:
        raise ValueError(f'Unknown tree_type {tree_type}')

    return tree


def eval_pred(task, tree, X, y, logger, prefix=''):
    """
    Evaluate the predictive performance of the tree on X and y.
    """
    result = {'mse': -1, 'acc': -1, 'auc': -1}

    if task == 'regression':
        pred = tree.predict(X)
        result['mse'] = mean_squared_error(y, pred)

    elif task == 'binary':
        pred = tree.predict(X)
        proba = tree.predict_proba(X)[:, 1]
        result['acc'] = accuracy_score(y, pred)
        result['auc'] = roc_auc_score(y, proba)

    elif task == 'multiclass':
        pred = tree.predict(X)
        result['acc'] = accuracy_score(y, pred)

    logger.info(f"[{prefix}] mse: {result['mse']:>10.3f}, "
                f"acc.: {result['acc']:>10.3f}, "
                f"AUC: {result['auc']:>10.3f}")

    return result
