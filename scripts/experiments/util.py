"""
Utility methods.
"""
import os
import sys
import shutil
import logging
import hashlib
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
from sklearn.metrics import log_loss


# public
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


def stdout_stderr_to_log(filename):
    """
    Log everything printed to stdout or
    stderr to this specified `filename`.
    """
    logfile = open(filename, 'w')

    stderr = sys.stderr
    stdout = sys.stdout

    sys.stdout = Tee(sys.stdout, logfile)
    sys.stderr = sys.stdout

    return logfile, stdout, stderr


def reset_stdout_stderr(logfile, stdout, stderr):
    """
    Restore original stdout and stderr
    """
    sys.stdout = stdout
    sys.stderr = stderr
    logfile.close()


def get_data(data_dir, dataset):
    """
    Return train and test data for the specified dataset.
    """
    data = np.load(os.path.join(data_dir, dataset, 'data.npy'), allow_pickle=True)[()]

    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    # get objective for the given dataset
    d = {}
    d['regression'] = ['synth_regression', 'casp', 'obesity', 'life']
    d['binary'] = ['synth_binary', 'bank_marketing', 'adult', 'surgical', 'vaccine',
                   'diabetes', 'flight_delays']
    d['multiclass'] = ['synth_multiclass', 'poker']

    objective = ''
    for k in d.keys():
        if dataset in d[k]:
            objective = k
            break

    if objective == '':
        raise ValueError(f'No objetive for dataset: {dataset}')

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


def get_model(tree_type='lgb', objective='regression', n_tree=100, max_depth=5, random_state=1):
    """
    Return the ensemble object from the specified framework and objective.
    """

    if tree_type == 'cb':
        class_fn = CatBoostRegressor if objective == 'regression' else CatBoostClassifier
        tree = class_fn(n_estimators=n_tree, max_depth=max_depth,
                        leaf_estimation_iterations=1, random_state=random_state,
                        logging_level='Silent')

    elif tree_type == 'lgb':
        class_fn = LGBMRegressor if objective == 'regression' else LGBMClassifier
        tree = class_fn(n_estimators=n_tree, max_depth=max_depth, num_leaves=2**max_depth,
                        random_state=random_state)

    elif tree_type == 'skgbm':
        class_fn = GradientBoostingRegressor if objective == 'regression' else GradientBoostingClassifier
        tree = class_fn(n_estimators=n_tree, max_depth=max_depth, random_state=random_state)

    elif tree_type == 'skrf':
        class_fn = RandomForestRegressor if objective == 'regression' else RandomForestClassifier
        tree = class_fn(n_estimators=n_tree, max_depth=max_depth, random_state=random_state, bootstrap=False)

    elif tree_type == 'xgb':

        if objective == 'regression':
            tree = XGBRegressor(n_estimators=n_tree, max_depth=max_depth, random_state=random_state)

        elif objective == 'binary':
            tree = XGBClassifier(n_estimators=n_tree, max_depth=max_depth,
                                 random_state=random_state, use_label_encoder=False,
                                 eval_metric='logloss')

        elif objective == 'multiclass':
            tree = XGBClassifier(n_estimators=n_tree, max_depth=max_depth,
                                 random_state=random_state, use_label_encoder=False,
                                 eval_metric='mlogloss')
        else:
            raise ValueError(f'Unknown objective {objective}')

    else:
        raise ValueError(f'Unknown tree_type {tree_type}')

    return tree


def eval_pred(objective, model, X, y, logger, prefix='', loss_fn=None):
    """
    Evaluate the predictive performance of the model on X and y.
    """
    result = {'mse': -1, 'acc': -1, 'auc': -1, 'loss': -1}

    if objective == 'regression':
        pred = model.predict(X)
        result['mse'] = mean_squared_error(y, pred)

    elif objective == 'binary':
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        result['acc'] = accuracy_score(y, pred)
        result['auc'] = roc_auc_score(y, proba)
        result['loss'] = log_loss(y, proba)

    elif objective == 'multiclass':
        pred = model.predict(X)
        proba = model.predict_proba(X)
        result['acc'] = accuracy_score(y, pred)
        result['loss'] = log_loss(y, proba)

    logger.info(f"[{prefix}] mse: {result['mse']:>10.3f}, "
                f"acc.: {result['acc']:>10.3f}, "
                f"AUC: {result['auc']:>10.3f}, "
                f"loss: {result['loss']:>10.3f}")

    return result


def eval_loss(objective, model, X, y, logger, prefix='', eps=1e-5):
    """
    Return individual losses.
    """
    assert X.shape[0] == y.shape[0] == 1

    result = {}

    if objective == 'regression':
        y_hat = model.predict(X)  # shape=(X.shape[0])
        losses = 0.5 * (y - y_hat) ** 2
        result['pred'] = y_hat[0]
        result['loss'] = losses[0]
        loss_type = 'squared_loss'

    elif objective == 'binary':
        y_hat = model.predict_proba(X)[:, 1]  # shape=(X.shape[0])
        y_hat = np.clip(y_hat, eps, 1 - eps)  # prevent log(0)
        losses = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        result['pred'] = y_hat[0]
        result['loss'] = losses[0]
        loss_type = 'logloss'

    else:
        assert objective == 'multiclass'
        target = y[0]
        y_hat = model.predict_proba(X)[0]  # shape=(no. class,)
        y_hat = np.clip(y_hat, eps, 1 - eps)
        result['pred'] = y_hat[target]
        result['loss'] = -np.log(y_hat)[target]
        loss_type = 'cross_entropy_loss'

    logger.info(f"[{prefix}] prediction: {result['pred']:>10.3f}, "
                f"{loss_type}: {result['loss']:>10.3f}, ")

    return result


def dict_to_hash(my_dict):
    """
    Convert to string and concatenate the desired values
    in `my_dict` and return the hashed string.
    """
    d = my_dict.copy()

    # remove keys not desired in the hash string
    for key in ['n_jobs', 'random_state']:
        if key in d:
            del d[key]

    s = ''.join(str(v) for k, v in sorted(d.items()))  # alphabetical key sort

    result = hashlib.md5(s.encode('utf-8')).hexdigest() if s != '' else ''

    return result


def explainer_params_to_dict(explainer, exp_params):
    """
    Return dict of explainer hyperparameters.
    """
    params = {}

    if explainer == 'boostin':
        params['use_leaf'] = exp_params['use_leaf']

    elif explainer == 'leaf_influence':
        params['update_set'] = exp_params['update_set']

    elif explainer == 'trex':
        params['kernel'] = exp_params['kernel']
        params['target'] = exp_params['target']
        params['lmbd'] = exp_params['lmbd']
        params['n_epoch'] = exp_params['n_epoch']
        params['global_op'] = exp_params['global_op']
        params['random_state'] = exp_params['random_state']

    elif explainer == 'loo':
        params['global_op'] = exp_params['global_op']
        params['n_jobs'] = exp_params['n_jobs']

    elif explainer == 'dshap':
        params['trunc_frac'] = exp_params['trunc_frac']
        params['global_op'] = exp_params['global_op']
        params['n_jobs'] = exp_params['n_jobs']
        params['check_every'] = exp_params['check_every']
        params['random_state'] = exp_params['random_state']

    elif explainer == 'random':
        params['random_state'] = exp_params['random_state']

    # create hash string based on the chosen hyperparameters
    hash_str = dict_to_hash(params)

    return params, hash_str


def get_hyperparams(tree_type, dataset):
    """
    Input
        tree_type: str, Tree-ensemble type.
        dataset: str, dataset.

    Return
        - Dict of selected hyperparameters for the inputs.
    """
    lgb = {}
    lgb['surgical'] = {'n_estimators': 200, 'num_leaves': 15, 'max_depth': -1}
    lgb['vaccine'] = {'n_estimators': 100, 'num_leaves': 15, 'max_depth': -1}
    lgb['adult'] = {'n_estimators': 100, 'num_leaves': 31, 'max_depth': -1}
    lgb['bank_marketing'] = {'n_estimators': 50, 'num_leaves': 15, 'max_depth': -1}
    lgb['diabetes'] = {'n_estimators': 200, 'num_leaves': 31, 'max_depth': -1}
    lgb['flight_delays'] = {'n_estimators': 100, 'num_leaves': 91, 'max_depth': -1}
    lgb['casp'] = {'n_estimators': 200, 'num_leaves': 91, 'max_depth': -1}
    lgb['obesity'] = {'n_estimators': 200, 'num_leaves': 91, 'max_depth': -1}
    lgb['life'] = {'n_estimators': 200, 'num_leaves': 61, 'max_depth': -1}

    hp = {}
    hp['lgb'] = lgb

    return hp[tree_type][dataset]


# private
class Tee(object):
    """
    Class to control where output is printed to.
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()   # output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()
