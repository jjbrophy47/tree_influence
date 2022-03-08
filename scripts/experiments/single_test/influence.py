"""
Compute local influence.
"""
import os
import sys
import time
import joblib
import argparse
import resource
from datetime import datetime

import numpy as np
from sklearn.base import clone

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')  # tree_influence
sys.path.insert(0, here + '/../../')  # config
sys.path.insert(0, here + '/../')  # util
import tree_influence
import util
from config import exp_args


def select_n_jobs(n_jobs):
    """
    Select value of n_jobs based on user input and no. available CPUs.

    Input
        n_jobs: int, desired no. jobs; -1 means use all available CPUs.

    Return
        - No. jobs to actually run in parallel.
    """
    if n_jobs == -1:
        result = joblib.cpu_count()

    else:
        assert n_jobs >= 1
        result = min(n_jobs, joblib.cpu_count())

    return result


def get_special_case_tol(dataset, tree_type, method, default_tol=1e-5):
    """
    Special cases for `leaf_inf` and `leaf_refit`.

    Input
        dataset: str, dataset.
        tree_type: str, tree-ensemble model.
        method: str, explainer.
        default_tol: float, original tolerance.

    Return
        - Tolerance (float).
    """
    tol = default_tol

    if method in ['leaf_inf', 'leaf_refit', 'leaf_infLE', 'leaf_refitLE']:

        if tree_type == 'lgb' and dataset == 'flight_delays':
            tol = 1e-1

        elif tree_type == 'cb':

            if dataset == 'bean':
                tol = 0.5

            elif dataset == 'naval':
                tol = 1e-4

    return tol


def select_elements(arr, rng, n):
    """
    - Randomly select `n` elements from `arr`.

    Input
        arr: 1d array of elements.
        rng: numpy pseudo-random number generator.
        n: int, no. elements to sample.

    Return
        - 1d array of shape=(n,).

    Note
        - Any sub-sequence should be exactly the same given
            the same `rng`, regardless of `n`.
    """
    assert arr.ndim == 1

    result = np.zeros(n, dtype=arr.dtype)

    for i in range(n):
        idx = rng.choice(len(arr), size=1, replace=False)[0]
        result[i] = arr[idx]
        arr = np.delete(arr, idx)

    return result


def experiment(args, logger, params, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(args.random_state)
    result = {}

    # data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # train tree-ensemble
    hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=args.random_state)
    tree.set_params(**hp)
    tree = tree.fit(X_train, y_train)
    util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test')

    # randomly select test instances to compute influence values for
    avail_idxs = np.arange(X_test.shape[0])
    n_test = min(args.n_test, len(avail_idxs))
    test_idxs = select_elements(avail_idxs, rng, n=n_test)

    # fit explainer
    start = time.time()
    explainer = tree_influence.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train)
    fit_time = time.time() - start - explainer.parse_time_

    logger.info(f'\n[INFO] explainer fit time: {fit_time:.5f}s')

    # compute influence
    start2 = time.time()
    influence = explainer.get_local_influence(X_test[test_idxs], y_test[test_idxs])  # shape=(no. train, no. test)
    inf_time = time.time() - start2

    logger.info(f'[INFO] explainer influence time: {inf_time:.5f}s')
    logger.info(f'[INFO] total time: {time.time() - begin:.5f}s')

    # generate ALL test predictions, save if necessary
    if objective == 'regression':
        y_train_pred = tree.predict(X_train).reshape(-1, 1)
        y_test_pred = tree.predict(X_test).reshape(-1, 1)

    elif objective == 'binary':
        y_train_pred = tree.predict_proba(X_train)[:, 1].reshape(-1, 1)
        y_test_pred = tree.predict_proba(X_test)[:, 1].reshape(-1, 1)

    else:
        assert objective == 'multiclass'
        y_train_pred = tree.predict_proba(X_train)
        y_test_pred = tree.predict_proba(X_test)

    # save results
    result['influence'] = influence
    result['test_idxs'] = test_idxs
    result['y_test_pred'] = y_test_pred
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['fit_time'] = fit_time
    result['inf_time'] = inf_time
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()
    result['n_jobs'] = args.n_jobs

    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))

    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # get unique hash for this experiment setting
    exp_dict = {'n_test': args.n_test}
    exp_hash = util.dict_to_hash(exp_dict)

    # get unique hash for the explainer
    args.leaf_inf_atol = get_special_case_tol(args.dataset, args.tree_type, args.method, args.leaf_inf_atol)
    params, method_hash = util.explainer_params_to_dict(args.method, vars(args))

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.tree_type,
                           f'exp_{exp_hash}',
                           f'{args.method}_{method_hash}')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, params, out_dir)

    # clean up
    util.remove_logger(logger)


if __name__ == '__main__':
    main(exp_args.get_influence_args().parse_args())
